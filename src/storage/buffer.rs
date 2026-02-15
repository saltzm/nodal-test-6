//! Buffer pool — caches pages in memory with LRU eviction.
//!
//! The buffer pool sits between the executor/B-tree and the disk manager,
//! reducing I/O by keeping frequently-accessed pages in memory.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::storage::disk::DiskManager;
use crate::storage::page::{Page, PageId, PageType};

/// A frame in the buffer pool holding a cached page.
struct Frame {
    page: Page,
    dirty: bool,
}

/// An LRU-evicting buffer pool that caches pages from a `DiskManager`.
pub struct BufferPool {
    /// Underlying disk manager.
    disk: DiskManager,
    /// Maximum number of pages to keep in memory.
    capacity: usize,
    /// Cached frames, keyed by page ID.
    frames: HashMap<PageId, Frame>,
    /// LRU order: most-recently-used at the back, least-recently-used at the front.
    /// Contains page IDs currently in the buffer pool.
    lru_order: Vec<PageId>,
}

impl BufferPool {
    /// Create a new buffer pool wrapping the given disk manager.
    pub fn new(disk: DiskManager, capacity: usize) -> Self {
        assert!(capacity > 0, "Buffer pool capacity must be > 0");
        Self {
            disk,
            capacity,
            frames: HashMap::new(),
            lru_order: Vec::new(),
        }
    }

    /// Fetch a page by ID. Returns a clone of the cached page.
    /// If the page is not in the pool, it is read from disk and cached.
    pub fn fetch_page(&mut self, page_id: PageId) -> Result<Page> {
        // If already cached, touch LRU and return.
        if self.frames.contains_key(&page_id) {
            self.touch(page_id);
            return Ok(self.frames[&page_id].page.clone());
        }

        // Need to load from disk. Evict if necessary.
        self.ensure_space()?;

        let page = self.disk.read_page(page_id)?;
        self.frames.insert(
            page_id,
            Frame {
                page: page.clone(),
                dirty: false,
            },
        );
        self.lru_order.push(page_id);

        Ok(page)
    }

    /// Create a new page via the disk manager and cache it in the pool.
    pub fn new_page(&mut self, page_type: PageType) -> Result<Page> {
        self.ensure_space()?;

        let page = self.disk.allocate_page(page_type)?;
        let page_id = page.id;
        self.frames.insert(
            page_id,
            Frame {
                page: page.clone(),
                dirty: false,
            },
        );
        self.lru_order.push(page_id);

        Ok(page)
    }

    /// Write an updated page back to the buffer pool (marks it dirty).
    /// The page must already be in the pool (fetched or newly created).
    pub fn write_page(&mut self, page: Page) -> Result<()> {
        let page_id = page.id;
        if let Some(frame) = self.frames.get_mut(&page_id) {
            frame.page = page;
            frame.dirty = true;
            self.touch(page_id);
            Ok(())
        } else {
            Err(Error::Storage(format!(
                "Cannot write page {}: not in buffer pool. Fetch it first.",
                page_id
            )))
        }
    }

    /// Flush a specific page to disk if it is dirty.
    pub fn flush_page(&mut self, page_id: PageId) -> Result<()> {
        if let Some(frame) = self.frames.get_mut(&page_id) {
            if frame.dirty {
                self.disk.write_page(&frame.page)?;
                frame.dirty = false;
            }
            Ok(())
        } else {
            // Not in pool — nothing to flush.
            Ok(())
        }
    }

    /// Flush all dirty pages to disk.
    pub fn flush_all(&mut self) -> Result<()> {
        // Collect dirty page IDs first to avoid borrow issues.
        let dirty_ids: Vec<PageId> = self
            .frames
            .iter()
            .filter(|(_, f)| f.dirty)
            .map(|(&id, _)| id)
            .collect();
        for id in dirty_ids {
            let frame = self.frames.get(&id).unwrap();
            self.disk.write_page(&frame.page)?;
            self.frames.get_mut(&id).unwrap().dirty = false;
        }
        self.disk.sync()?;
        Ok(())
    }

    /// Evict a specific page from the pool. Flushes if dirty.
    pub fn evict_page(&mut self, page_id: PageId) -> Result<()> {
        if let Some(frame) = self.frames.get(&page_id) {
            if frame.dirty {
                self.disk.write_page(&frame.page)?;
            }
            self.frames.remove(&page_id);
            self.lru_order.retain(|&id| id != page_id);
        }
        Ok(())
    }

    /// Return the number of pages currently in the pool.
    pub fn size(&self) -> usize {
        self.frames.len()
    }

    /// Return the pool capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get a reference to the underlying disk manager.
    pub fn disk(&self) -> &DiskManager {
        &self.disk
    }

    /// Get a mutable reference to the underlying disk manager.
    pub fn disk_mut(&mut self) -> &mut DiskManager {
        &mut self.disk
    }

    // ─── Internal helpers ───

    /// Move the given page_id to the most-recently-used position.
    fn touch(&mut self, page_id: PageId) {
        if let Some(pos) = self.lru_order.iter().position(|&id| id == page_id) {
            self.lru_order.remove(pos);
        }
        self.lru_order.push(page_id);
    }

    /// Ensure there is room for at least one more page. Evicts the LRU page
    /// if the pool is full.
    fn ensure_space(&mut self) -> Result<()> {
        while self.frames.len() >= self.capacity {
            if let Some(victim_id) = self.lru_order.first().copied() {
                self.evict_page(victim_id)?;
            } else {
                return Err(Error::Storage(
                    "Buffer pool is full but LRU list is empty".into(),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn temp_db_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("toydb_test");
        fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn cleanup(path: &std::path::Path) {
        let _ = fs::remove_file(path);
    }

    fn make_pool(name: &str, capacity: usize) -> (BufferPool, PathBuf) {
        let path = temp_db_path(name);
        cleanup(&path);
        let dm = DiskManager::open(&path).unwrap();
        (BufferPool::new(dm, capacity), path)
    }

    #[test]
    fn test_buffer_pool_new_and_fetch() {
        let (mut pool, path) = make_pool("bp_new_fetch.db", 4);

        // Create a page.
        let mut page = pool.new_page(PageType::Leaf).unwrap();
        assert_eq!(page.id, 0);
        assert_eq!(pool.size(), 1);

        // Write data to it.
        page.data[0] = 42;
        pool.write_page(page).unwrap();

        // Fetch it back.
        let fetched = pool.fetch_page(0).unwrap();
        assert_eq!(fetched.data[0], 42);
        assert_eq!(fetched.page_type, PageType::Leaf);

        cleanup(&path);
    }

    #[test]
    fn test_buffer_pool_lru_eviction() {
        // Pool capacity of 3.
        let (mut pool, path) = make_pool("bp_lru.db", 3);

        // Create 3 pages (fills pool).
        let p0 = pool.new_page(PageType::Leaf).unwrap();
        let p1 = pool.new_page(PageType::Leaf).unwrap();
        let _p2 = pool.new_page(PageType::Leaf).unwrap();
        assert_eq!(pool.size(), 3);

        // Access p0 to make it most recently used (order: p1, p2, p0).
        pool.fetch_page(p0.id).unwrap();

        // Create a 4th page — should evict p1 (least recently used).
        let _p3 = pool.new_page(PageType::Leaf).unwrap();
        assert_eq!(pool.size(), 3);

        // p1 should have been evicted — fetching it loads from disk.
        // (We can verify it's no longer in the frame cache by checking size stays 3
        // after a second eviction round.)
        let fetched_p1 = pool.fetch_page(p1.id).unwrap();
        assert_eq!(fetched_p1.page_type, PageType::Leaf);

        cleanup(&path);
    }

    #[test]
    fn test_buffer_pool_dirty_page_flushed_on_eviction() {
        let (mut pool, path) = make_pool("bp_dirty_evict.db", 2);

        // Create and modify a page.
        let mut page = pool.new_page(PageType::Leaf).unwrap();
        page.data[0] = 0xAB;
        pool.write_page(page).unwrap();

        // Create another page.
        pool.new_page(PageType::Leaf).unwrap();

        // Create a third — evicts page 0 (dirty, should be flushed).
        pool.new_page(PageType::Leaf).unwrap();
        assert_eq!(pool.size(), 2);

        // Read page 0 from disk to verify it was flushed.
        let fetched = pool.fetch_page(0).unwrap();
        assert_eq!(fetched.data[0], 0xAB);

        cleanup(&path);
    }

    #[test]
    fn test_buffer_pool_flush_all() {
        let (mut pool, path) = make_pool("bp_flush_all.db", 4);

        // Create and modify two pages.
        let mut p0 = pool.new_page(PageType::Leaf).unwrap();
        p0.data[0] = 0x11;
        pool.write_page(p0).unwrap();

        let mut p1 = pool.new_page(PageType::Leaf).unwrap();
        p1.data[0] = 0x22;
        pool.write_page(p1).unwrap();

        pool.flush_all().unwrap();

        // Drop pool and reopen from disk.
        drop(pool);
        let dm = DiskManager::open(&path).unwrap();
        let mut pool2 = BufferPool::new(dm, 4);

        let fetched0 = pool2.fetch_page(0).unwrap();
        assert_eq!(fetched0.data[0], 0x11);

        let fetched1 = pool2.fetch_page(1).unwrap();
        assert_eq!(fetched1.data[0], 0x22);

        cleanup(&path);
    }

    #[test]
    fn test_buffer_pool_eviction_under_pressure() {
        // Small pool, many pages.
        let (mut pool, path) = make_pool("bp_pressure.db", 3);

        // Create 10 pages with distinct data.
        for i in 0u8..10 {
            let mut page = pool.new_page(PageType::Leaf).unwrap();
            page.data[0] = i;
            pool.write_page(page).unwrap();
        }

        // Pool should only hold 3 pages.
        assert_eq!(pool.size(), 3);

        // All 10 pages should be readable (from cache or disk).
        for i in 0u8..10 {
            let page = pool.fetch_page(i as PageId).unwrap();
            assert_eq!(page.data[0], i, "Page {} has wrong data", i);
        }

        cleanup(&path);
    }

    #[test]
    fn test_buffer_pool_write_unfetched_page_fails() {
        let (mut pool, path) = make_pool("bp_write_unfetched.db", 4);

        // Try to write a page that was never fetched.
        let fake_page = Page::new(99, PageType::Leaf);
        assert!(pool.write_page(fake_page).is_err());

        cleanup(&path);
    }

    #[test]
    fn test_buffer_pool_persist_across_reopen() {
        let path = temp_db_path("bp_reopen.db");
        cleanup(&path);

        // Write data.
        {
            let dm = DiskManager::open(&path).unwrap();
            let mut pool = BufferPool::new(dm, 4);
            let mut page = pool.new_page(PageType::Leaf).unwrap();
            page.data[0] = 0xBE;
            page.data[1] = 0xEF;
            pool.write_page(page).unwrap();
            pool.flush_all().unwrap();
        }

        // Reopen and verify.
        {
            let dm = DiskManager::open(&path).unwrap();
            let mut pool = BufferPool::new(dm, 4);
            let page = pool.fetch_page(0).unwrap();
            assert_eq!(page.data[0], 0xBE);
            assert_eq!(page.data[1], 0xEF);
            assert_eq!(page.page_type, PageType::Leaf);
        }

        cleanup(&path);
    }

    #[test]
    fn test_buffer_pool_flush_specific_page() {
        let (mut pool, path) = make_pool("bp_flush_specific.db", 4);

        let mut page = pool.new_page(PageType::Leaf).unwrap();
        page.data[0] = 0x99;
        pool.write_page(page).unwrap();

        // Flush only page 0.
        pool.flush_page(0).unwrap();

        // Flushing a non-existent page is a no-op.
        pool.flush_page(999).unwrap();

        // Evict page 0 and re-read from disk to confirm flush.
        pool.evict_page(0).unwrap();
        assert_eq!(pool.size(), 0);

        let page = pool.fetch_page(0).unwrap();
        assert_eq!(page.data[0], 0x99);

        cleanup(&path);
    }

    #[test]
    fn test_buffer_pool_capacity() {
        let (pool, path) = make_pool("bp_capacity.db", 8);
        assert_eq!(pool.capacity(), 8);
        assert_eq!(pool.size(), 0);
        cleanup(&path);
    }

    #[test]
    fn test_buffer_pool_lru_ordering_complex() {
        // Verify that accessing pages in specific order affects eviction correctly.
        let (mut pool, path) = make_pool("bp_lru_complex.db", 3);

        // Create pages 0, 1, 2.
        for _ in 0..3 {
            pool.new_page(PageType::Leaf).unwrap();
        }
        // LRU order: [0, 1, 2] (0 is LRU)

        // Access page 0, making it MRU.
        pool.fetch_page(0).unwrap();
        // LRU order: [1, 2, 0]

        // Access page 1, making it MRU.
        pool.fetch_page(1).unwrap();
        // LRU order: [2, 0, 1]

        // Create page 3 — should evict page 2 (LRU).
        pool.new_page(PageType::Leaf).unwrap();
        assert_eq!(pool.size(), 3);

        // page 2 should not be in pool (evicted).
        // pages 0, 1, 3 should be.
        assert!(pool.frames.contains_key(&0));
        assert!(pool.frames.contains_key(&1));
        assert!(pool.frames.contains_key(&3));
        assert!(!pool.frames.contains_key(&2));

        cleanup(&path);
    }
}