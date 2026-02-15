//! Disk manager — manages a single database file as a flat array of fixed-size pages.
//!
//! The file layout is simply: `[Page0][Page1][Page2]...`
//! where each page occupies exactly `PAGE_SIZE` bytes.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::storage::page::{Page, PageId, PageType, PAGE_SIZE};

/// Manages reading and writing pages to/from a single database file.
pub struct DiskManager {
    /// Path to the database file.
    path: PathBuf,
    /// Open file handle.
    file: File,
    /// Total number of pages currently in the file.
    num_pages: u32,
    /// List of free page IDs available for reuse.
    free_pages: Vec<PageId>,
}

impl DiskManager {
    /// Open (or create) a database file at the given path.
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

        let mut mgr = Self {
            path,
            file,
            num_pages,
            free_pages: Vec::new(),
        };

        // Scan for free pages on open.
        mgr.scan_free_pages()?;

        Ok(mgr)
    }

    /// Scan the file for pages marked as Free and record them.
    fn scan_free_pages(&mut self) -> Result<()> {
        self.free_pages.clear();
        for page_id in 0..self.num_pages {
            // Read just the page type byte (offset 4 within the page).
            let offset = page_id as u64 * PAGE_SIZE as u64 + 4;
            self.file.seek(SeekFrom::Start(offset))?;
            let mut type_byte = [0u8; 1];
            self.file.read_exact(&mut type_byte)?;
            if type_byte[0] == PageType::Free.to_byte() {
                self.free_pages.push(page_id);
            }
        }
        Ok(())
    }

    /// Allocate a new page. Reuses a free page if available, otherwise appends
    /// a new page to the file. Returns the allocated page (zeroed payload, with
    /// the given page type).
    pub fn allocate_page(&mut self, page_type: PageType) -> Result<Page> {
        let page_id = if let Some(free_id) = self.free_pages.pop() {
            free_id
        } else {
            let id = self.num_pages;
            self.num_pages += 1;
            id
        };

        let page = Page::new(page_id, page_type);
        self.write_page(&page)?;
        Ok(page)
    }

    /// Free a page by marking it with PageType::Free.
    pub fn free_page(&mut self, page_id: PageId) -> Result<()> {
        if page_id >= self.num_pages {
            return Err(Error::Storage(format!(
                "Cannot free page {}: out of range (num_pages={})",
                page_id, self.num_pages
            )));
        }
        let mut page = Page::new(page_id, PageType::Free);
        // Zero out the payload.
        page.data.fill(0);
        self.write_page(&page)?;
        self.free_pages.push(page_id);
        Ok(())
    }

    /// Read a page from disk by its PageId.
    pub fn read_page(&mut self, page_id: PageId) -> Result<Page> {
        if page_id >= self.num_pages {
            return Err(Error::Storage(format!(
                "Cannot read page {}: out of range (num_pages={})",
                page_id, self.num_pages
            )));
        }
        let offset = page_id as u64 * PAGE_SIZE as u64;
        self.file.seek(SeekFrom::Start(offset))?;
        let mut buf = [0u8; PAGE_SIZE];
        self.file.read_exact(&mut buf)?;
        Page::deserialize(&buf)
    }

    /// Write a page to disk at the position determined by its PageId.
    pub fn write_page(&mut self, page: &Page) -> Result<()> {
        let offset = page.id as u64 * PAGE_SIZE as u64;
        self.file.seek(SeekFrom::Start(offset))?;
        let buf = page.serialize();
        self.file.write_all(&buf)?;
        self.file.flush()?;
        Ok(())
    }

    /// Return the total number of pages in the file.
    pub fn num_pages(&self) -> u32 {
        self.num_pages
    }

    /// Return the file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Flush all buffered writes to disk.
    pub fn sync(&mut self) -> Result<()> {
        self.file.sync_all()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Helper to create a temporary database file path.
    fn temp_db_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("toydb_test");
        fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    /// Clean up a test database file.
    fn cleanup(path: &Path) {
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_disk_manager_create_empty() {
        let path = temp_db_path("test_dm_create_empty.db");
        cleanup(&path);

        let dm = DiskManager::open(&path).unwrap();
        assert_eq!(dm.num_pages(), 0);

        cleanup(&path);
    }

    #[test]
    fn test_disk_manager_allocate_and_read() {
        let path = temp_db_path("test_dm_alloc_read.db");
        cleanup(&path);

        {
            let mut dm = DiskManager::open(&path).unwrap();
            let mut page = dm.allocate_page(PageType::Leaf).unwrap();
            assert_eq!(page.id, 0);
            assert_eq!(page.page_type, PageType::Leaf);

            // Write some data.
            page.data[0] = 0xDE;
            page.data[1] = 0xAD;
            dm.write_page(&page).unwrap();
            assert_eq!(dm.num_pages(), 1);

            // Read it back.
            let read_back = dm.read_page(0).unwrap();
            assert_eq!(read_back.id, 0);
            assert_eq!(read_back.page_type, PageType::Leaf);
            assert_eq!(read_back.data[0], 0xDE);
            assert_eq!(read_back.data[1], 0xAD);
        }

        cleanup(&path);
    }

    #[test]
    fn test_disk_manager_multiple_pages() {
        let path = temp_db_path("test_dm_multi.db");
        cleanup(&path);

        {
            let mut dm = DiskManager::open(&path).unwrap();
            let p0 = dm.allocate_page(PageType::Meta).unwrap();
            let p1 = dm.allocate_page(PageType::Internal).unwrap();
            let p2 = dm.allocate_page(PageType::Leaf).unwrap();

            assert_eq!(p0.id, 0);
            assert_eq!(p1.id, 1);
            assert_eq!(p2.id, 2);
            assert_eq!(dm.num_pages(), 3);

            let r0 = dm.read_page(0).unwrap();
            let r1 = dm.read_page(1).unwrap();
            let r2 = dm.read_page(2).unwrap();
            assert_eq!(r0.page_type, PageType::Meta);
            assert_eq!(r1.page_type, PageType::Internal);
            assert_eq!(r2.page_type, PageType::Leaf);
        }

        cleanup(&path);
    }

    #[test]
    fn test_disk_manager_read_out_of_range() {
        let path = temp_db_path("test_dm_oor.db");
        cleanup(&path);

        let mut dm = DiskManager::open(&path).unwrap();
        assert!(dm.read_page(0).is_err());
        assert!(dm.read_page(100).is_err());

        cleanup(&path);
    }

    #[test]
    fn test_disk_manager_free_and_reuse() {
        let path = temp_db_path("test_dm_free_reuse.db");
        cleanup(&path);

        {
            let mut dm = DiskManager::open(&path).unwrap();
            let p0 = dm.allocate_page(PageType::Leaf).unwrap();
            let _p1 = dm.allocate_page(PageType::Leaf).unwrap();
            assert_eq!(dm.num_pages(), 2);

            // Free page 0.
            dm.free_page(p0.id).unwrap();

            // Next allocation should reuse page 0.
            let p2 = dm.allocate_page(PageType::Internal).unwrap();
            assert_eq!(p2.id, 0);
            assert_eq!(p2.page_type, PageType::Internal);

            // File still has 2 pages (no growth).
            assert_eq!(dm.num_pages(), 2);
        }

        cleanup(&path);
    }

    #[test]
    fn test_disk_manager_persist_across_reopen() {
        let path = temp_db_path("test_dm_reopen.db");
        cleanup(&path);

        // Write data.
        {
            let mut dm = DiskManager::open(&path).unwrap();
            let mut page = dm.allocate_page(PageType::Leaf).unwrap();
            page.data[0] = 0xCA;
            page.data[1] = 0xFE;
            dm.write_page(&page).unwrap();
            dm.sync().unwrap();
        }

        // Reopen and read.
        {
            let mut dm = DiskManager::open(&path).unwrap();
            assert_eq!(dm.num_pages(), 1);
            let page = dm.read_page(0).unwrap();
            assert_eq!(page.page_type, PageType::Leaf);
            assert_eq!(page.data[0], 0xCA);
            assert_eq!(page.data[1], 0xFE);
        }

        cleanup(&path);
    }

    #[test]
    fn test_disk_manager_free_pages_persist_across_reopen() {
        let path = temp_db_path("test_dm_free_persist.db");
        cleanup(&path);

        // Allocate 3 pages, free the middle one.
        {
            let mut dm = DiskManager::open(&path).unwrap();
            dm.allocate_page(PageType::Leaf).unwrap();
            dm.allocate_page(PageType::Leaf).unwrap();
            dm.allocate_page(PageType::Leaf).unwrap();
            dm.free_page(1).unwrap();
            dm.sync().unwrap();
        }

        // Reopen — should detect page 1 as free.
        {
            let mut dm = DiskManager::open(&path).unwrap();
            assert_eq!(dm.num_pages(), 3);

            // Next allocation should reuse page 1.
            let p = dm.allocate_page(PageType::Internal).unwrap();
            assert_eq!(p.id, 1);
            assert_eq!(p.page_type, PageType::Internal);
        }

        cleanup(&path);
    }

    #[test]
    fn test_disk_manager_free_out_of_range() {
        let path = temp_db_path("test_dm_free_oor.db");
        cleanup(&path);

        let mut dm = DiskManager::open(&path).unwrap();
        assert!(dm.free_page(0).is_err());

        cleanup(&path);
    }
}
