//! Async-compatible I/O layer for page-level disk operations.
//!
//! This module provides an `AsyncDiskIO` trait that defines an async-compatible
//! interface for reading and writing pages. The current implementation,
//! `SyncFallbackIO`, uses blocking `std::fs` operations as a fallback since
//! `io_uring` is not available in this environment.
//!
//! # Future work
//!
//! When `io_uring` support is available (e.g., via the `io-uring` crate), a
//! `IoUringDiskIO` implementation can be added behind a feature flag:
//!
//! ```toml
//! [features]
//! io_uring = ["io-uring"]
//! ```
//!
//! The `io_uring` implementation would submit read/write requests to a
//! submission queue and poll the completion queue, enabling true async I/O
//! without blocking OS threads.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::storage::page::{Page, PageId, PAGE_SIZE};

// ---------------------------------------------------------------------------
// Trait definition
// ---------------------------------------------------------------------------

/// Async-compatible interface for page-level disk I/O.
///
/// Methods return `Result<T>` directly (rather than `Future<Output = Result<T>>`)
/// because the current implementation is synchronous. When a real async runtime
/// and `io_uring` become available, these can be changed to return `impl Future`
/// or be made `async fn` (with the `async_trait` crate or Rust's native async
/// trait support).
pub trait AsyncDiskIO {
    /// Read the page with the given `page_id` from disk.
    ///
    /// Returns `Err` if the page does not exist or the underlying I/O fails.
    fn async_read_page(&mut self, page_id: PageId) -> Result<Page>;

    /// Write `page` to disk at the location determined by `page.id`.
    ///
    /// The caller is responsible for ensuring the page ID is valid (i.e. the
    /// file has been extended to accommodate it, or the page already exists).
    fn async_write_page(&mut self, page: &Page) -> Result<()>;

    /// Ensure all previously written data is durable on disk.
    fn async_sync(&mut self) -> Result<()>;
}

// ---------------------------------------------------------------------------
// Synchronous fallback implementation
// ---------------------------------------------------------------------------

/// A synchronous implementation of [`AsyncDiskIO`] backed by a standard
/// [`File`]. All operations block the calling thread.
///
/// # cfg note
///
/// This implementation is always compiled. When the `io_uring` feature is
/// enabled a separate `IoUringDiskIO` struct would be preferred at runtime.
// #[cfg(not(feature = "io_uring"))]   // uncomment when io_uring impl exists
pub struct SyncFallbackIO {
    /// Path to the underlying database file.
    path: PathBuf,
    /// Open file handle (read+write).
    file: File,
    /// Total number of pages currently in the file (tracked to detect out-of-range reads).
    num_pages: u32,
}

impl SyncFallbackIO {
    /// Open (or create) a database file at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;

        let file_len = file.metadata()?.len();
        let num_pages = (file_len / PAGE_SIZE as u64) as u32;

        Ok(Self {
            path,
            file,
            num_pages,
        })
    }

    /// Return the file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Return the number of pages currently in the file.
    pub fn num_pages(&self) -> u32 {
        self.num_pages
    }

    /// Extend the file so that it can hold at least `page_id + 1` pages.
    /// This is useful when writing a page beyond the current file length.
    pub fn ensure_capacity(&mut self, page_id: PageId) -> Result<()> {
        let required = page_id + 1;
        if required > self.num_pages {
            let new_len = required as u64 * PAGE_SIZE as u64;
            self.file.set_len(new_len)?;
            self.num_pages = required;
        }
        Ok(())
    }
}

impl AsyncDiskIO for SyncFallbackIO {
    fn async_read_page(&mut self, page_id: PageId) -> Result<Page> {
        if page_id >= self.num_pages {
            return Err(Error::Storage(format!(
                "async_read_page: page {} out of range (num_pages={})",
                page_id, self.num_pages
            )));
        }

        let offset = page_id as u64 * PAGE_SIZE as u64;
        self.file.seek(SeekFrom::Start(offset))?;

        let mut buf = [0u8; PAGE_SIZE];
        self.file.read_exact(&mut buf)?;
        Page::deserialize(&buf)
    }

    fn async_write_page(&mut self, page: &Page) -> Result<()> {
        // Automatically extend the file if necessary.
        self.ensure_capacity(page.id)?;

        let offset = page.id as u64 * PAGE_SIZE as u64;
        self.file.seek(SeekFrom::Start(offset))?;

        let buf = page.serialize();
        self.file.write_all(&buf)?;
        self.file.flush()?;
        Ok(())
    }

    fn async_sync(&mut self) -> Result<()> {
        self.file.sync_all()?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::page::{PageType, PAGE_PAYLOAD_SIZE};
    use std::fs;

    /// Helper: temporary file path for tests.
    fn temp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("toydb_io_test");
        fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn cleanup(path: &Path) {
        let _ = fs::remove_file(path);
    }

    // -----------------------------------------------------------------------
    // Basic read/write roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_write_then_read_roundtrip() {
        let path = temp_path("test_io_roundtrip.db");
        cleanup(&path);

        {
            let mut io = SyncFallbackIO::open(&path).unwrap();
            assert_eq!(io.num_pages(), 0);

            // Create and write a page.
            let mut page = Page::new(0, PageType::Leaf);
            page.data[0] = 0xAB;
            page.data[1] = 0xCD;
            page.data[PAGE_PAYLOAD_SIZE - 1] = 0xFF;
            io.async_write_page(&page).unwrap();

            // Read it back.
            let read_back = io.async_read_page(0).unwrap();
            assert_eq!(read_back.id, 0);
            assert_eq!(read_back.page_type, PageType::Leaf);
            assert_eq!(read_back.data[0], 0xAB);
            assert_eq!(read_back.data[1], 0xCD);
            assert_eq!(read_back.data[PAGE_PAYLOAD_SIZE - 1], 0xFF);
        }

        cleanup(&path);
    }

    // -----------------------------------------------------------------------
    // Multiple pages
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_pages() {
        let path = temp_path("test_io_multi.db");
        cleanup(&path);

        {
            let mut io = SyncFallbackIO::open(&path).unwrap();

            // Write pages 0, 1, 2 with distinct data.
            for i in 0u32..3 {
                let mut page = Page::new(i, PageType::Leaf);
                page.data[0] = i as u8;
                page.data[1] = (i * 10) as u8;
                io.async_write_page(&page).unwrap();
            }

            assert_eq!(io.num_pages(), 3);

            // Read them all back and verify.
            for i in 0u32..3 {
                let page = io.async_read_page(i).unwrap();
                assert_eq!(page.id, i);
                assert_eq!(page.data[0], i as u8);
                assert_eq!(page.data[1], (i * 10) as u8);
            }
        }

        cleanup(&path);
    }

    // -----------------------------------------------------------------------
    // Data integrity / overwrite
    // -----------------------------------------------------------------------

    #[test]
    fn test_overwrite_page() {
        let path = temp_path("test_io_overwrite.db");
        cleanup(&path);

        {
            let mut io = SyncFallbackIO::open(&path).unwrap();

            // Write initial page.
            let mut page = Page::new(0, PageType::Leaf);
            page.data[0] = 0x11;
            io.async_write_page(&page).unwrap();

            // Overwrite with different data.
            page.data[0] = 0x22;
            page.page_type = PageType::Internal;
            io.async_write_page(&page).unwrap();

            // Verify overwritten values.
            let read = io.async_read_page(0).unwrap();
            assert_eq!(read.data[0], 0x22);
            assert_eq!(read.page_type, PageType::Internal);
        }

        cleanup(&path);
    }

    // -----------------------------------------------------------------------
    // Persistence across reopen
    // -----------------------------------------------------------------------

    #[test]
    fn test_persist_across_reopen() {
        let path = temp_path("test_io_persist.db");
        cleanup(&path);

        // Write.
        {
            let mut io = SyncFallbackIO::open(&path).unwrap();
            let mut page = Page::new(0, PageType::Meta);
            page.data[42] = 0xEF;
            io.async_write_page(&page).unwrap();
            io.async_sync().unwrap();
        }

        // Reopen and read.
        {
            let mut io = SyncFallbackIO::open(&path).unwrap();
            assert_eq!(io.num_pages(), 1);
            let page = io.async_read_page(0).unwrap();
            assert_eq!(page.page_type, PageType::Meta);
            assert_eq!(page.data[42], 0xEF);
        }

        cleanup(&path);
    }

    // -----------------------------------------------------------------------
    // Error: read out of range
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_out_of_range() {
        let path = temp_path("test_io_oor.db");
        cleanup(&path);

        {
            let mut io = SyncFallbackIO::open(&path).unwrap();
            // Empty file — any read should fail.
            assert!(io.async_read_page(0).is_err());
            assert!(io.async_read_page(42).is_err());
        }

        cleanup(&path);
    }

    // -----------------------------------------------------------------------
    // Sparse writes (non-sequential page IDs)
    // -----------------------------------------------------------------------

    #[test]
    fn test_sparse_write() {
        let path = temp_path("test_io_sparse.db");
        cleanup(&path);

        {
            let mut io = SyncFallbackIO::open(&path).unwrap();

            // Write page 5 directly (skipping 0-4).
            let mut page = Page::new(5, PageType::Leaf);
            page.data[0] = 0x55;
            io.async_write_page(&page).unwrap();

            assert_eq!(io.num_pages(), 6); // pages 0-5 exist now

            // Read page 5 back.
            let read = io.async_read_page(5).unwrap();
            assert_eq!(read.data[0], 0x55);

            // Pages 0-4 should be readable (zeroed out).
            for i in 0..5 {
                let p = io.async_read_page(i).unwrap();
                // Their content is all zeroes (including the page type byte),
                // so deserialization should still work (PageType::Meta = 0).
                assert_eq!(p.page_type, PageType::Meta);
            }
        }

        cleanup(&path);
    }

    // -----------------------------------------------------------------------
    // Sync does not cause errors
    // -----------------------------------------------------------------------

    #[test]
    fn test_sync_on_empty_file() {
        let path = temp_path("test_io_sync_empty.db");
        cleanup(&path);

        {
            let mut io = SyncFallbackIO::open(&path).unwrap();
            // sync on an empty file should succeed.
            io.async_sync().unwrap();
        }

        cleanup(&path);
    }

    // -----------------------------------------------------------------------
    // Trait object usage (ensure the trait is object-safe)
    // -----------------------------------------------------------------------

    #[test]
    fn test_trait_object() {
        let path = temp_path("test_io_trait_obj.db");
        cleanup(&path);

        {
            let mut io: Box<dyn AsyncDiskIO> = Box::new(SyncFallbackIO::open(&path).unwrap());

            let mut page = Page::new(0, PageType::Leaf);
            page.data[0] = 0x99;
            io.async_write_page(&page).unwrap();

            let read = io.async_read_page(0).unwrap();
            assert_eq!(read.data[0], 0x99);

            io.async_sync().unwrap();
        }

        cleanup(&path);
    }

    // -----------------------------------------------------------------------
    // Large payload fill
    // -----------------------------------------------------------------------

    #[test]
    fn test_full_payload() {
        let path = temp_path("test_io_full_payload.db");
        cleanup(&path);

        {
            let mut io = SyncFallbackIO::open(&path).unwrap();

            let mut page = Page::new(0, PageType::Leaf);
            // Fill entire payload with a pattern.
            for (i, byte) in page.data.iter_mut().enumerate() {
                *byte = (i % 256) as u8;
            }
            io.async_write_page(&page).unwrap();

            let read = io.async_read_page(0).unwrap();
            for (i, byte) in read.data.iter().enumerate() {
                assert_eq!(*byte, (i % 256) as u8, "mismatch at byte {}", i);
            }
        }

        cleanup(&path);
    }
}
