//! Table storage — maps table names to B-tree instances and persists the catalog
//! metadata on a dedicated page (page 0).
//!
//! ## Metadata page layout (PageType::Meta, page 0)
//!
//! ```text
//! [num_tables: u32][table0][table1]...
//! ```
//!
//! Each table entry:
//! ```text
//! [name_len: u16][name_bytes: UTF-8][root_page_id: u32]
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::storage::btree::BTree;
use crate::storage::buffer::BufferPool;
use crate::storage::disk::DiskManager;
use crate::storage::page::{PageType, PAGE_PAYLOAD_SIZE};
use crate::storage::wal::{self, Wal};
use crate::types::{Row, Value};

/// Metadata page is always page 0.
const META_PAGE_ID: u32 = 0;

/// Derive the WAL file path from a database file path.
///
/// Convention: `<db_path>.wal` (e.g., `mydb.db` → `mydb.db.wal`).
pub fn wal_path_for(db_path: &Path) -> PathBuf {
    let mut wal = db_path.as_os_str().to_os_string();
    wal.push(".wal");
    PathBuf::from(wal)
}

/// Entry point for the executor to interact with table storage.
///
/// Manages multiple named tables, each backed by a B-tree, in a single
/// database file. A metadata page (page 0) persists the table-name-to-root
/// mapping so it survives close/reopen.
pub struct TableStorage {
    pool: BufferPool,
    /// In-memory mapping from lowercase table name to B-tree root page id.
    tables: HashMap<String, u32>,
    /// Path to the database file (kept for diagnostics / reopen).
    path: PathBuf,
    /// Whether WAL is enabled.
    wal_enabled: bool,
}

impl TableStorage {
    /// Open (or create) a table storage backed by the given database file
    /// (no WAL).
    ///
    /// If the file is empty, a metadata page is allocated. Otherwise, the
    /// existing metadata page is loaded.
    pub fn open(path: impl AsRef<Path>, pool_capacity: usize) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let dm = DiskManager::open(&path)?;
        let mut pool = BufferPool::new(dm, pool_capacity);

        let tables = if pool.disk().num_pages() == 0 {
            // Fresh database — allocate meta page.
            let page = pool.new_page(PageType::Meta)?;
            assert_eq!(page.id, META_PAGE_ID);
            // Write empty catalog: num_tables = 0
            let mut meta = page;
            meta.data[0..4].copy_from_slice(&0u32.to_le_bytes());
            pool.write_page(meta)?;
            HashMap::new()
        } else {
            // Existing database — read meta page.
            let page = pool.fetch_page(META_PAGE_ID)?;
            if page.page_type != PageType::Meta {
                return Err(Error::Storage(format!(
                    "Page 0 has type {:?}, expected Meta",
                    page.page_type
                )));
            }
            Self::deserialize_catalog(&page.data)?
        };

        Ok(Self {
            pool,
            tables,
            path,
            wal_enabled: false,
        })
    }

    /// Open (or create) a table storage with WAL enabled.
    ///
    /// The WAL file path is derived from the database file path by appending
    /// `.wal` (e.g., `mydb.db` → `mydb.db.wal`).
    ///
    /// On open, WAL recovery is performed first (if a WAL file exists),
    /// then the WAL is opened for new writes.
    pub fn open_with_wal(path: impl AsRef<Path>, pool_capacity: usize) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let wal_path = wal_path_for(&path);

        // Perform WAL recovery before opening the database.
        if wal_path.exists() {
            let (_redo, _undo) = wal::recover(&path, &wal_path)?;
        }

        let dm = DiskManager::open(&path)?;
        let wal = Wal::open(&wal_path)?;
        let mut pool = BufferPool::new_with_wal(dm, pool_capacity, wal);

        let tables = if pool.disk().num_pages() == 0 {
            // Fresh database — allocate meta page.
            let page = pool.new_page(PageType::Meta)?;
            assert_eq!(page.id, META_PAGE_ID);
            // Write empty catalog: num_tables = 0
            let mut meta = page;
            meta.data[0..4].copy_from_slice(&0u32.to_le_bytes());
            pool.write_page(meta)?;
            HashMap::new()
        } else {
            // Existing database — read meta page.
            let page = pool.fetch_page(META_PAGE_ID)?;
            if page.page_type != PageType::Meta {
                return Err(Error::Storage(format!(
                    "Page 0 has type {:?}, expected Meta",
                    page.page_type
                )));
            }
            Self::deserialize_catalog(&page.data)?
        };

        Ok(Self {
            pool,
            tables,
            path,
            wal_enabled: true,
        })
    }

    /// Create a new table with the given name.
    ///
    /// Returns an error if a table with the same name (case-insensitive)
    /// already exists.
    pub fn create_table(&mut self, name: &str) -> Result<()> {
        let key = name.to_lowercase();
        if self.tables.contains_key(&key) {
            return Err(Error::Catalog(format!("Table '{}' already exists", name)));
        }

        let tree = BTree::create(&mut self.pool)?;
        let root = tree.root_page_id();
        self.tables.insert(key, root);
        self.flush_catalog()?;
        Ok(())
    }

    /// Insert a row into a table, keyed by `key`.
    pub fn insert_row(&mut self, table: &str, key: Value, row: Row) -> Result<()> {
        let root = self.root_for(table)?;
        let mut tree = BTree::open(root);
        tree.insert(&mut self.pool, key, row)?;
        // The root may have changed after a split.
        self.update_root(table, tree.root_page_id())?;
        Ok(())
    }

    /// Get a row from a table by key.
    pub fn get_row(&mut self, table: &str, key: &Value) -> Result<Option<Row>> {
        let root = self.root_for(table)?;
        let tree = BTree::open(root);
        tree.get(&mut self.pool, key)
    }

    /// Delete a row from a table by key. Returns true if the key existed.
    pub fn delete_row(&mut self, table: &str, key: &Value) -> Result<bool> {
        let root = self.root_for(table)?;
        let mut tree = BTree::open(root);
        tree.delete(&mut self.pool, key)
    }

    /// Full scan of all rows in a table, in key order.
    pub fn scan_table(&mut self, table: &str) -> Result<Vec<(Value, Row)>> {
        let root = self.root_for(table)?;
        let tree = BTree::open(root);
        tree.scan(&mut self.pool)
    }

    /// Range scan: returns entries where `start <= key <= end`, in key order.
    pub fn range_scan_table(
        &mut self,
        table: &str,
        start: &Value,
        end: &Value,
    ) -> Result<Vec<(Value, Row)>> {
        let root = self.root_for(table)?;
        let tree = BTree::open(root);
        tree.range_scan(&mut self.pool, start, end)
    }

    /// List all table names (lowercase).
    pub fn table_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tables.keys().cloned().collect();
        names.sort();
        names
    }

    /// Check whether a table exists.
    pub fn has_table(&self, name: &str) -> bool {
        self.tables.contains_key(&name.to_lowercase())
    }

    /// Flush all dirty pages (including the catalog) to disk.
    pub fn flush(&mut self) -> Result<()> {
        self.flush_catalog()?;
        self.pool.flush_all()
    }

    /// Return the database file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Whether this storage instance has a WAL attached.
    pub fn wal_enabled(&self) -> bool {
        self.wal_enabled
    }

    /// Perform a WAL checkpoint: flush all dirty pages, log a checkpoint
    /// record, then truncate the WAL file. No-op if WAL is not enabled.
    pub fn checkpoint(&mut self) -> Result<()> {
        if !self.wal_enabled {
            return Ok(());
        }
        // Flush all dirty pages (this writes WAL records then data pages).
        self.flush()?;
        // Log a checkpoint record and truncate the WAL.
        if let Some(ref mut wal) = self.pool.wal_mut() {
            wal.log_checkpoint()?;
            wal.truncate()?;
        }
        Ok(())
    }

    /// Return a reference to the underlying buffer pool.
    pub fn pool(&self) -> &BufferPool {
        &self.pool
    }

    /// Return a mutable reference to the underlying buffer pool.
    pub fn pool_mut(&mut self) -> &mut BufferPool {
        &mut self.pool
    }

    // ─── Internal helpers ───

    /// Look up the B-tree root page id for a table (case-insensitive).
    fn root_for(&self, table: &str) -> Result<u32> {
        let key = table.to_lowercase();
        self.tables
            .get(&key)
            .copied()
            .ok_or_else(|| Error::Catalog(format!("Table '{}' not found", table)))
    }

    /// Update the in-memory root and persist the catalog if changed.
    fn update_root(&mut self, table: &str, new_root: u32) -> Result<()> {
        let key = table.to_lowercase();
        let entry = self
            .tables
            .get_mut(&key)
            .ok_or_else(|| Error::Catalog(format!("Table '{}' not found", table)))?;
        if *entry != new_root {
            *entry = new_root;
            self.flush_catalog()?;
        }
        Ok(())
    }

    /// Serialize the catalog into the meta page and mark it dirty.
    fn flush_catalog(&mut self) -> Result<()> {
        let mut page = self.pool.fetch_page(META_PAGE_ID)?;
        let data = Self::serialize_catalog(&self.tables)?;
        page.data[..data.len()].copy_from_slice(&data);
        // Zero remainder to avoid stale data.
        for b in page.data[data.len()..].iter_mut() {
            *b = 0;
        }
        self.pool.write_page(page)?;
        Ok(())
    }

    /// Serialize table name → root_page_id map into bytes.
    fn serialize_catalog(tables: &HashMap<String, u32>) -> Result<Vec<u8>> {
        let num = tables.len() as u32;
        let mut buf = Vec::new();
        buf.extend_from_slice(&num.to_le_bytes());

        // Sort for determinism.
        let mut entries: Vec<_> = tables.iter().collect();
        entries.sort_by_key(|(name, _)| name.as_str());

        for (name, &root) in &entries {
            let name_bytes = name.as_bytes();
            if name_bytes.len() > u16::MAX as usize {
                return Err(Error::Catalog(format!(
                    "Table name '{}' too long (max {} bytes)",
                    name,
                    u16::MAX
                )));
            }
            let name_len = name_bytes.len() as u16;
            buf.extend_from_slice(&name_len.to_le_bytes());
            buf.extend_from_slice(name_bytes);
            buf.extend_from_slice(&root.to_le_bytes());
        }

        if buf.len() > PAGE_PAYLOAD_SIZE {
            return Err(Error::Storage(
                "Catalog metadata exceeds page payload size".into(),
            ));
        }

        Ok(buf)
    }

    /// Deserialize the catalog from a meta page payload.
    fn deserialize_catalog(data: &[u8]) -> Result<HashMap<String, u32>> {
        if data.len() < 4 {
            return Err(Error::Storage(
                "Meta page too small for catalog header".into(),
            ));
        }

        let num = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let mut offset = 4;
        let mut tables = HashMap::new();

        for _ in 0..num {
            // Read name_len
            if offset + 2 > data.len() {
                return Err(Error::Storage(
                    "Meta page truncated: missing name length".into(),
                ));
            }
            let name_len =
                u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2;

            // Read name bytes
            if offset + name_len > data.len() {
                return Err(Error::Storage(
                    "Meta page truncated: missing name data".into(),
                ));
            }
            let name = std::str::from_utf8(&data[offset..offset + name_len])
                .map_err(|e| Error::Storage(format!("Invalid UTF-8 in table name: {}", e)))?
                .to_string();
            offset += name_len;

            // Read root_page_id
            if offset + 4 > data.len() {
                return Err(Error::Storage(
                    "Meta page truncated: missing root page id".into(),
                ));
            }
            let root = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            offset += 4;

            tables.insert(name, root);
        }

        Ok(tables)
    }
}

// ───────────────────────── Tests ─────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn temp_db_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("toydb_test_table_storage");
        fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn cleanup(path: &std::path::Path) {
        let _ = fs::remove_file(path);
    }

    fn make_row(vals: Vec<Value>) -> Row {
        Row::new(vals)
    }

    // ── Basic table creation ──

    #[test]
    fn test_create_table() {
        let path = temp_db_path("ts_create.db");
        cleanup(&path);
        {
            let mut ts = TableStorage::open(&path, 100).unwrap();
            ts.create_table("users").unwrap();
            assert!(ts.has_table("users"));
            assert!(ts.has_table("USERS")); // case-insensitive
            assert!(!ts.has_table("orders"));
        }
        cleanup(&path);
    }

    #[test]
    fn test_create_duplicate_table_fails() {
        let path = temp_db_path("ts_dup.db");
        cleanup(&path);
        {
            let mut ts = TableStorage::open(&path, 100).unwrap();
            ts.create_table("users").unwrap();
            let result = ts.create_table("Users");
            assert!(result.is_err());
        }
        cleanup(&path);
    }

    #[test]
    fn test_create_multiple_tables() {
        let path = temp_db_path("ts_multi.db");
        cleanup(&path);
        {
            let mut ts = TableStorage::open(&path, 100).unwrap();
            ts.create_table("users").unwrap();
            ts.create_table("orders").unwrap();
            ts.create_table("products").unwrap();

            assert!(ts.has_table("users"));
            assert!(ts.has_table("orders"));
            assert!(ts.has_table("products"));
            assert!(!ts.has_table("customers"));

            let names = ts.table_names();
            assert_eq!(names, vec!["orders", "products", "users"]);
        }
        cleanup(&path);
    }

    // ── Insert / Get / Delete ──

    #[test]
    fn test_insert_and_get() {
        let path = temp_db_path("ts_insert_get.db");
        cleanup(&path);
        {
            let mut ts = TableStorage::open(&path, 100).unwrap();
            ts.create_table("users").unwrap();

            let key = Value::Integer(1);
            let row = make_row(vec![Value::Integer(1), Value::Text("Alice".into())]);
            ts.insert_row("users", key.clone(), row.clone()).unwrap();

            let result = ts.get_row("users", &key).unwrap();
            assert_eq!(result, Some(row));

            // Non-existent key
            assert_eq!(ts.get_row("users", &Value::Integer(999)).unwrap(), None);
        }
        cleanup(&path);
    }

    #[test]
    fn test_insert_and_delete() {
        let path = temp_db_path("ts_del.db");
        cleanup(&path);
        {
            let mut ts = TableStorage::open(&path, 100).unwrap();
            ts.create_table("items").unwrap();

            ts.insert_row("items", Value::Integer(1), make_row(vec![Value::Integer(1)])).unwrap();
            ts.insert_row("items", Value::Integer(2), make_row(vec![Value::Integer(2)])).unwrap();

            assert!(ts.delete_row("items", &Value::Integer(1)).unwrap());
            assert!(!ts.delete_row("items", &Value::Integer(1)).unwrap()); // already gone
            assert_eq!(ts.get_row("items", &Value::Integer(1)).unwrap(), None);
            assert!(ts.get_row("items", &Value::Integer(2)).unwrap().is_some());
        }
        cleanup(&path);
    }

    #[test]
    fn test_scan_table() {
        let path = temp_db_path("ts_scan.db");
        cleanup(&path);
        {
            let mut ts = TableStorage::open(&path, 100).unwrap();
            ts.create_table("nums").unwrap();

            for i in 0..10i64 {
                ts.insert_row("nums", Value::Integer(i), make_row(vec![Value::Integer(i)])).unwrap();
            }

            let results = ts.scan_table("nums").unwrap();
            assert_eq!(results.len(), 10);
            for (idx, (k, _)) in results.iter().enumerate() {
                assert_eq!(k, &Value::Integer(idx as i64));
            }
        }
        cleanup(&path);
    }

    #[test]
    fn test_range_scan_table() {
        let path = temp_db_path("ts_range.db");
        cleanup(&path);
        {
            let mut ts = TableStorage::open(&path, 100).unwrap();
            ts.create_table("nums").unwrap();

            for i in 0..20i64 {
                ts.insert_row("nums", Value::Integer(i), make_row(vec![Value::Integer(i)])).unwrap();
            }

            let results = ts.range_scan_table("nums", &Value::Integer(5), &Value::Integer(14)).unwrap();
            let keys: Vec<i64> = results.iter().map(|(k, _)| k.as_integer().unwrap()).collect();
            assert_eq!(keys, (5..=14).collect::<Vec<_>>());
        }
        cleanup(&path);
    }

    // ── Error cases ──

    #[test]
    fn test_operations_on_nonexistent_table() {
        let path = temp_db_path("ts_noexist.db");
        cleanup(&path);
        {
            let mut ts = TableStorage::open(&path, 100).unwrap();

            assert!(ts.insert_row("phantom", Value::Integer(1), make_row(vec![])).is_err());
            assert!(ts.get_row("phantom", &Value::Integer(1)).is_err());
            assert!(ts.delete_row("phantom", &Value::Integer(1)).is_err());
            assert!(ts.scan_table("phantom").is_err());
            assert!(ts.range_scan_table("phantom", &Value::Integer(0), &Value::Integer(10)).is_err());
        }
        cleanup(&path);
    }

    // ── Multiple tables coexist ──

    #[test]
    fn test_multiple_tables_data_isolation() {
        let path = temp_db_path("ts_isolation.db");
        cleanup(&path);
        {
            let mut ts = TableStorage::open(&path, 100).unwrap();
            ts.create_table("users").unwrap();
            ts.create_table("orders").unwrap();

            // Insert into users
            for i in 0..10i64 {
                ts.insert_row("users", Value::Integer(i),
                    make_row(vec![Value::Integer(i), Value::Text(format!("user_{}", i))])).unwrap();
            }

            // Insert into orders (different keys, overlapping range)
            for i in 5..15i64 {
                ts.insert_row("orders", Value::Integer(i),
                    make_row(vec![Value::Integer(i), Value::Text(format!("order_{}", i))])).unwrap();
            }

            // Verify users
            let users = ts.scan_table("users").unwrap();
            assert_eq!(users.len(), 10);
            assert_eq!(users[0].1.values[1], Value::Text("user_0".into()));

            // Verify orders
            let orders = ts.scan_table("orders").unwrap();
            assert_eq!(orders.len(), 10);
            assert_eq!(orders[0].0, Value::Integer(5));
            assert_eq!(orders[0].1.values[1], Value::Text("order_5".into()));

            // Delete from one table doesn't affect the other
            ts.delete_row("users", &Value::Integer(5)).unwrap();
            assert_eq!(ts.get_row("users", &Value::Integer(5)).unwrap(), None);
            assert!(ts.get_row("orders", &Value::Integer(5)).unwrap().is_some());
        }
        cleanup(&path);
    }

    // ── Persistence across close/reopen ──

    #[test]
    fn test_metadata_persists_across_reopen() {
        let path = temp_db_path("ts_persist_meta.db");
        cleanup(&path);

        // Create tables and insert data.
        {
            let mut ts = TableStorage::open(&path, 100).unwrap();
            ts.create_table("users").unwrap();
            ts.create_table("orders").unwrap();

            ts.insert_row("users", Value::Integer(1),
                make_row(vec![Value::Integer(1), Value::Text("Alice".into())])).unwrap();
            ts.insert_row("users", Value::Integer(2),
                make_row(vec![Value::Integer(2), Value::Text("Bob".into())])).unwrap();

            ts.insert_row("orders", Value::Integer(100),
                make_row(vec![Value::Integer(100), Value::Integer(1)])).unwrap();

            ts.flush().unwrap();
        }

        // Reopen and verify.
        {
            let mut ts = TableStorage::open(&path, 100).unwrap();

            assert!(ts.has_table("users"));
            assert!(ts.has_table("orders"));
            assert_eq!(ts.table_names(), vec!["orders", "users"]);

            // Verify users data
            let alice = ts.get_row("users", &Value::Integer(1)).unwrap();
            assert_eq!(
                alice,
                Some(make_row(vec![Value::Integer(1), Value::Text("Alice".into())]))
            );
            let bob = ts.get_row("users", &Value::Integer(2)).unwrap();
            assert_eq!(
                bob,
                Some(make_row(vec![Value::Integer(2), Value::Text("Bob".into())]))
            );

            // Verify orders data
            let order = ts.get_row("orders", &Value::Integer(100)).unwrap();
            assert_eq!(
                order,
                Some(make_row(vec![Value::Integer(100), Value::Integer(1)]))
            );

            // Non-existent data is still absent
            assert_eq!(ts.get_row("users", &Value::Integer(999)).unwrap(), None);
        }

        cleanup(&path);
    }

    #[test]
    fn test_persistence_with_many_rows() {
        let path = temp_db_path("ts_persist_many.db");
        cleanup(&path);

        let n = 300i64;

        // Write
        {
            let mut ts = TableStorage::open(&path, 50).unwrap();
            ts.create_table("data").unwrap();

            for i in 0..n {
                ts.insert_row("data", Value::Integer(i),
                    make_row(vec![Value::Integer(i), Value::Text(format!("row_{}", i))])).unwrap();
            }
            ts.flush().unwrap();
        }

        // Read back
        {
            let mut ts = TableStorage::open(&path, 50).unwrap();
            assert!(ts.has_table("data"));

            for i in 0..n {
                let result = ts.get_row("data", &Value::Integer(i)).unwrap();
                assert_eq!(
                    result,
                    Some(make_row(vec![Value::Integer(i), Value::Text(format!("row_{}", i))])),
                    "Row {} not found after reopen",
                    i
                );
            }

            let all = ts.scan_table("data").unwrap();
            assert_eq!(all.len(), n as usize);
        }

        cleanup(&path);
    }

    #[test]
    fn test_persistence_multiple_tables_many_rows() {
        let path = temp_db_path("ts_persist_multi_many.db");
        cleanup(&path);

        // Write multiple tables
        {
            let mut ts = TableStorage::open(&path, 40).unwrap();
            ts.create_table("alpha").unwrap();
            ts.create_table("beta").unwrap();
            ts.create_table("gamma").unwrap();

            for i in 0..100i64 {
                ts.insert_row("alpha", Value::Integer(i),
                    make_row(vec![Value::Integer(i), Value::Text("A".into())])).unwrap();
                ts.insert_row("beta", Value::Integer(i * 10),
                    make_row(vec![Value::Integer(i * 10), Value::Text("B".into())])).unwrap();
                ts.insert_row("gamma", Value::Text(format!("key_{:04}", i)),
                    make_row(vec![Value::Text(format!("key_{:04}", i)), Value::Integer(i)])).unwrap();
            }
            ts.flush().unwrap();
        }

        // Verify after reopen
        {
            let mut ts = TableStorage::open(&path, 40).unwrap();
            assert_eq!(ts.table_names(), vec!["alpha", "beta", "gamma"]);

            // Check alpha
            let alpha = ts.scan_table("alpha").unwrap();
            assert_eq!(alpha.len(), 100);

            // Check beta
            let beta_val = ts.get_row("beta", &Value::Integer(50)).unwrap();
            assert_eq!(
                beta_val,
                Some(make_row(vec![Value::Integer(50), Value::Text("B".into())]))
            );

            // Check gamma (text keys)
            let gamma_val = ts.get_row("gamma", &Value::Text("key_0042".into())).unwrap();
            assert_eq!(
                gamma_val,
                Some(make_row(vec![Value::Text("key_0042".into()), Value::Integer(42)]))
            );

            // Range scan gamma
            let gamma_range = ts.range_scan_table(
                "gamma",
                &Value::Text("key_0010".into()),
                &Value::Text("key_0015".into()),
            ).unwrap();
            assert_eq!(gamma_range.len(), 6);
        }

        cleanup(&path);
    }

    // ── Buffer pool eviction under pressure ──

    #[test]
    fn test_large_inserts_small_buffer_pool() {
        let path = temp_db_path("ts_small_pool.db");
        cleanup(&path);
        {
            // Very small buffer pool (5 frames) forces heavy eviction.
            let mut ts = TableStorage::open(&path, 5).unwrap();
            ts.create_table("stress").unwrap();

            let n = 500i64;
            for i in 0..n {
                ts.insert_row("stress", Value::Integer(i),
                    make_row(vec![Value::Integer(i), Value::Text(format!("v{}", i))])).unwrap();
            }

            // Verify all rows are retrievable despite eviction.
            for i in 0..n {
                let result = ts.get_row("stress", &Value::Integer(i)).unwrap();
                assert!(result.is_some(), "Key {} missing under small pool", i);
                assert_eq!(result.unwrap().values[0], Value::Integer(i));
            }

            let all = ts.scan_table("stress").unwrap();
            assert_eq!(all.len(), n as usize);
        }
        cleanup(&path);
    }

    #[test]
    fn test_multiple_tables_small_buffer_pool() {
        let path = temp_db_path("ts_multi_small_pool.db");
        cleanup(&path);
        {
            let mut ts = TableStorage::open(&path, 8).unwrap();
            ts.create_table("t1").unwrap();
            ts.create_table("t2").unwrap();
            ts.create_table("t3").unwrap();

            // Insert 50 rows into each table.
            for i in 0..50i64 {
                ts.insert_row("t1", Value::Integer(i), make_row(vec![Value::Integer(i)])).unwrap();
                ts.insert_row("t2", Value::Integer(i), make_row(vec![Value::Integer(i * 2)])).unwrap();
                ts.insert_row("t3", Value::Integer(i), make_row(vec![Value::Integer(i * 3)])).unwrap();
            }

            // Verify all tables.
            let t1 = ts.scan_table("t1").unwrap();
            let t2 = ts.scan_table("t2").unwrap();
            let t3 = ts.scan_table("t3").unwrap();

            assert_eq!(t1.len(), 50);
            assert_eq!(t2.len(), 50);
            assert_eq!(t3.len(), 50);

            // Spot check data integrity
            assert_eq!(
                ts.get_row("t2", &Value::Integer(25)).unwrap(),
                Some(make_row(vec![Value::Integer(50)]))
            );
            assert_eq!(
                ts.get_row("t3", &Value::Integer(30)).unwrap(),
                Some(make_row(vec![Value::Integer(90)]))
            );
        }
        cleanup(&path);
    }

    // ── Root page update after splits persists ──

    #[test]
    fn test_root_page_update_persists() {
        let path = temp_db_path("ts_root_update.db");
        cleanup(&path);

        // Insert enough data to cause B-tree splits (root page changes).
        {
            let mut ts = TableStorage::open(&path, 100).unwrap();
            ts.create_table("bigtable").unwrap();

            for i in 0..500i64 {
                ts.insert_row("bigtable", Value::Integer(i),
                    make_row(vec![Value::Integer(i)])).unwrap();
            }
            ts.flush().unwrap();
        }

        // Reopen and verify root was correctly persisted.
        {
            let mut ts = TableStorage::open(&path, 100).unwrap();

            for i in 0..500i64 {
                let result = ts.get_row("bigtable", &Value::Integer(i)).unwrap();
                assert!(
                    result.is_some(),
                    "Key {} missing after reopen (root update test)",
                    i
                );
            }

            let all = ts.scan_table("bigtable").unwrap();
            assert_eq!(all.len(), 500);
        }

        cleanup(&path);
    }

    // ── Catalog serialization roundtrip ──

    #[test]
    fn test_catalog_serialize_deserialize() {
        let mut tables = HashMap::new();
        tables.insert("users".to_string(), 1u32);
        tables.insert("orders".to_string(), 5u32);
        tables.insert("products".to_string(), 12u32);

        let bytes = TableStorage::serialize_catalog(&tables).unwrap();
        let recovered = TableStorage::deserialize_catalog(&bytes).unwrap();

        assert_eq!(recovered, tables);
    }

    #[test]
    fn test_catalog_empty_roundtrip() {
        let tables: HashMap<String, u32> = HashMap::new();
        let bytes = TableStorage::serialize_catalog(&tables).unwrap();
        let recovered = TableStorage::deserialize_catalog(&bytes).unwrap();
        assert!(recovered.is_empty());
    }

    // ── Open empty path ──

    #[test]
    fn test_open_fresh_database() {
        let path = temp_db_path("ts_fresh.db");
        cleanup(&path);
        {
            let ts = TableStorage::open(&path, 100).unwrap();
            assert!(ts.table_names().is_empty());
            assert!(!ts.has_table("anything"));
        }
        cleanup(&path);
    }

    // ── Reopen fresh database (no tables) ──

    #[test]
    fn test_reopen_empty_database() {
        let path = temp_db_path("ts_reopen_empty.db");
        cleanup(&path);

        {
            let ts = TableStorage::open(&path, 100).unwrap();
            assert!(ts.table_names().is_empty());
            // Just opening creates the meta page.
            drop(ts);
        }

        // Reopen
        {
            let ts = TableStorage::open(&path, 100).unwrap();
            assert!(ts.table_names().is_empty());
        }

        cleanup(&path);
    }

    // ── Create table after reopen ──

    #[test]
    fn test_create_table_after_reopen() {
        let path = temp_db_path("ts_create_reopen.db");
        cleanup(&path);

        {
            let mut ts = TableStorage::open(&path, 100).unwrap();
            ts.create_table("first").unwrap();
            ts.insert_row("first", Value::Integer(1), make_row(vec![Value::Integer(1)])).unwrap();
            ts.flush().unwrap();
        }

        {
            let mut ts = TableStorage::open(&path, 100).unwrap();
            ts.create_table("second").unwrap();
            ts.insert_row("second", Value::Integer(2), make_row(vec![Value::Integer(2)])).unwrap();

            // Both tables should work.
            assert_eq!(
                ts.get_row("first", &Value::Integer(1)).unwrap(),
                Some(make_row(vec![Value::Integer(1)]))
            );
            assert_eq!(
                ts.get_row("second", &Value::Integer(2)).unwrap(),
                Some(make_row(vec![Value::Integer(2)]))
            );

            ts.flush().unwrap();
        }

        // Verify both persist.
        {
            let mut ts = TableStorage::open(&path, 100).unwrap();
            assert_eq!(ts.table_names(), vec!["first", "second"]);
            assert!(ts.get_row("first", &Value::Integer(1)).unwrap().is_some());
            assert!(ts.get_row("second", &Value::Integer(2)).unwrap().is_some());
        }

        cleanup(&path);
    }

    // ─── WAL integration tests ───

    fn cleanup_wal(path: &std::path::Path) {
        cleanup(path);
        let wal = super::wal_path_for(path);
        let _ = fs::remove_file(&wal);
    }

    #[test]
    fn test_wal_open_creates_wal_file() {
        let path = temp_db_path("ts_wal_create.db");
        cleanup_wal(&path);

        {
            let ts = TableStorage::open_with_wal(&path, 100).unwrap();
            assert!(ts.wal_enabled());
            let wal_path = super::wal_path_for(&path);
            assert!(wal_path.exists(), "WAL file should be created");
        }

        cleanup_wal(&path);
    }

    #[test]
    fn test_wal_basic_crud() {
        let path = temp_db_path("ts_wal_crud.db");
        cleanup_wal(&path);

        {
            let mut ts = TableStorage::open_with_wal(&path, 100).unwrap();
            ts.create_table("users").unwrap();

            // Insert
            for i in 0..10i64 {
                ts.insert_row("users", Value::Integer(i),
                    make_row(vec![Value::Integer(i), Value::Text(format!("user_{}", i))])).unwrap();
            }

            // Read back
            for i in 0..10i64 {
                let row = ts.get_row("users", &Value::Integer(i)).unwrap();
                assert!(row.is_some(), "Row {} should exist", i);
                assert_eq!(row.unwrap().values[1], Value::Text(format!("user_{}", i)));
            }

            // Delete
            assert!(ts.delete_row("users", &Value::Integer(5)).unwrap());
            assert_eq!(ts.get_row("users", &Value::Integer(5)).unwrap(), None);

            ts.flush().unwrap();
        }

        cleanup_wal(&path);
    }

    #[test]
    fn test_wal_persistence_across_reopen() {
        let path = temp_db_path("ts_wal_persist.db");
        cleanup_wal(&path);

        // Write data with WAL enabled.
        {
            let mut ts = TableStorage::open_with_wal(&path, 100).unwrap();
            ts.create_table("items").unwrap();

            for i in 0..20i64 {
                ts.insert_row("items", Value::Integer(i),
                    make_row(vec![Value::Integer(i), Value::Text(format!("item_{}", i))])).unwrap();
            }
            ts.flush().unwrap();
        }

        // Reopen with WAL and verify data persists.
        {
            let mut ts = TableStorage::open_with_wal(&path, 100).unwrap();
            assert!(ts.has_table("items"));

            for i in 0..20i64 {
                let row = ts.get_row("items", &Value::Integer(i)).unwrap();
                assert_eq!(
                    row,
                    Some(make_row(vec![Value::Integer(i), Value::Text(format!("item_{}", i))])),
                    "Row {} missing after WAL reopen", i
                );
            }
        }

        cleanup_wal(&path);
    }

    #[test]
    fn test_wal_checkpoint_truncates_wal() {
        let path = temp_db_path("ts_wal_checkpoint.db");
        cleanup_wal(&path);

        let wal_path = super::wal_path_for(&path);

        {
            let mut ts = TableStorage::open_with_wal(&path, 100).unwrap();
            ts.create_table("data").unwrap();

            for i in 0..5i64 {
                ts.insert_row("data", Value::Integer(i),
                    make_row(vec![Value::Integer(i)])).unwrap();
            }
            ts.flush().unwrap();

            // WAL should have records now.
            let wal_size_before = fs::metadata(&wal_path).unwrap().len();
            assert!(wal_size_before > 0, "WAL should have data before checkpoint");

            // Checkpoint should flush + truncate WAL.
            ts.checkpoint().unwrap();

            let wal_size_after = fs::metadata(&wal_path).unwrap().len();
            assert_eq!(wal_size_after, 0, "WAL should be empty after checkpoint");
        }

        // Data should still be accessible after checkpoint.
        {
            let mut ts = TableStorage::open_with_wal(&path, 100).unwrap();
            for i in 0..5i64 {
                assert!(ts.get_row("data", &Value::Integer(i)).unwrap().is_some());
            }
        }

        cleanup_wal(&path);
    }

    #[test]
    fn test_wal_records_generated_on_flush() {
        use crate::storage::wal::WalReader;

        let path = temp_db_path("ts_wal_records.db");
        cleanup_wal(&path);

        let wal_path = super::wal_path_for(&path);

        {
            let mut ts = TableStorage::open_with_wal(&path, 100).unwrap();
            ts.create_table("test").unwrap();

            ts.insert_row("test", Value::Integer(1),
                make_row(vec![Value::Integer(1)])).unwrap();

            ts.flush().unwrap();
        }

        // Read the WAL and verify it has Write records.
        {
            let mut reader = WalReader::open(&wal_path).unwrap();
            let records = reader.read_all().unwrap();
            assert!(!records.is_empty(), "WAL should have records after flush");

            // All records should be Write records (no Begin/Commit in implicit mode).
            use crate::storage::wal::WalRecord;
            let write_count = records.iter().filter(|r| matches!(r, WalRecord::Write { .. })).count();
            assert!(write_count > 0, "WAL should have at least one Write record");
        }

        cleanup_wal(&path);
    }

    #[test]
    fn test_wal_multiple_tables_with_wal() {
        let path = temp_db_path("ts_wal_multi_tables.db");
        cleanup_wal(&path);

        {
            let mut ts = TableStorage::open_with_wal(&path, 50).unwrap();
            ts.create_table("alpha").unwrap();
            ts.create_table("beta").unwrap();

            for i in 0..30i64 {
                ts.insert_row("alpha", Value::Integer(i),
                    make_row(vec![Value::Integer(i), Value::Text("A".into())])).unwrap();
                ts.insert_row("beta", Value::Integer(i * 10),
                    make_row(vec![Value::Integer(i * 10), Value::Text("B".into())])).unwrap();
            }

            ts.flush().unwrap();
        }

        // Reopen and verify.
        {
            let mut ts = TableStorage::open_with_wal(&path, 50).unwrap();
            assert_eq!(ts.table_names(), vec!["alpha", "beta"]);

            let alpha = ts.scan_table("alpha").unwrap();
            assert_eq!(alpha.len(), 30);

            let beta = ts.scan_table("beta").unwrap();
            assert_eq!(beta.len(), 30);

            assert_eq!(
                ts.get_row("beta", &Value::Integer(100)).unwrap(),
                Some(make_row(vec![Value::Integer(100), Value::Text("B".into())]))
            );
        }

        cleanup_wal(&path);
    }

    #[test]
    fn test_wal_small_buffer_pool() {
        let path = temp_db_path("ts_wal_small_pool.db");
        cleanup_wal(&path);

        {
            // Very small buffer pool forces heavy eviction — WAL records
            // should be written for every evicted dirty page.
            let mut ts = TableStorage::open_with_wal(&path, 5).unwrap();
            ts.create_table("stress").unwrap();

            let n = 200i64;
            for i in 0..n {
                ts.insert_row("stress", Value::Integer(i),
                    make_row(vec![Value::Integer(i), Value::Text(format!("v{}", i))])).unwrap();
            }

            // All rows should be retrievable despite heavy eviction.
            for i in 0..n {
                let result = ts.get_row("stress", &Value::Integer(i)).unwrap();
                assert!(result.is_some(), "Key {} missing under WAL + small pool", i);
            }

            ts.flush().unwrap();
        }

        // Reopen and verify persistence.
        {
            let mut ts = TableStorage::open_with_wal(&path, 5).unwrap();
            let all = ts.scan_table("stress").unwrap();
            assert_eq!(all.len(), 200);
        }

        cleanup_wal(&path);
    }

    #[test]
    fn test_wal_path_derivation() {
        let path = PathBuf::from("/tmp/test.db");
        let wal = super::wal_path_for(&path);
        assert_eq!(wal, PathBuf::from("/tmp/test.db.wal"));

        let path2 = PathBuf::from("mydata");
        let wal2 = super::wal_path_for(&path2);
        assert_eq!(wal2, PathBuf::from("mydata.wal"));
    }

    #[test]
    fn test_wal_recovery_on_reopen() {
        // This test verifies that WAL recovery happens on reopen.
        // We write data, flush to WAL but DON'T checkpoint,
        // then reopen — recovery should apply the WAL.
        let path = temp_db_path("ts_wal_recovery.db");
        cleanup_wal(&path);

        {
            let mut ts = TableStorage::open_with_wal(&path, 100).unwrap();
            ts.create_table("recovered").unwrap();

            for i in 0..10i64 {
                ts.insert_row("recovered", Value::Integer(i),
                    make_row(vec![Value::Integer(i)])).unwrap();
            }

            // Flush writes WAL records AND data pages.
            ts.flush().unwrap();
            // Don't checkpoint — WAL still has records.
        }

        // Reopen — recovery runs, then we verify data.
        {
            let mut ts = TableStorage::open_with_wal(&path, 100).unwrap();
            assert!(ts.has_table("recovered"));

            for i in 0..10i64 {
                let row = ts.get_row("recovered", &Value::Integer(i)).unwrap();
                assert!(row.is_some(), "Row {} should survive WAL recovery", i);
            }
        }

        cleanup_wal(&path);
    }

    #[test]
    fn test_wal_no_wal_mode_still_works() {
        // Verify that opening without WAL still works as before.
        let path = temp_db_path("ts_no_wal_mode.db");
        cleanup(&path);

        {
            let mut ts = TableStorage::open(&path, 100).unwrap();
            assert!(!ts.wal_enabled());

            ts.create_table("plain").unwrap();
            ts.insert_row("plain", Value::Integer(1),
                make_row(vec![Value::Integer(1)])).unwrap();
            ts.flush().unwrap();

            // checkpoint is a no-op when WAL is not enabled.
            ts.checkpoint().unwrap();
        }

        {
            let mut ts = TableStorage::open(&path, 100).unwrap();
            assert!(ts.get_row("plain", &Value::Integer(1)).unwrap().is_some());
        }

        cleanup(&path);
    }

    #[test]
    fn test_wal_mixed_operations() {
        let path = temp_db_path("ts_wal_mixed.db");
        cleanup_wal(&path);

        {
            let mut ts = TableStorage::open_with_wal(&path, 100).unwrap();
            ts.create_table("ops").unwrap();

            // Insert
            for i in 0..20i64 {
                ts.insert_row("ops", Value::Integer(i),
                    make_row(vec![Value::Integer(i), Value::Text(format!("v{}", i))])).unwrap();
            }

            // Delete even numbers
            for i in (0..20i64).step_by(2) {
                ts.delete_row("ops", &Value::Integer(i)).unwrap();
            }

            ts.flush().unwrap();
        }

        // Reopen and verify.
        {
            let mut ts = TableStorage::open_with_wal(&path, 100).unwrap();
            let all = ts.scan_table("ops").unwrap();
            assert_eq!(all.len(), 10, "Should have 10 odd rows");

            for (key, _) in &all {
                let k = key.as_integer().unwrap();
                assert!(k % 2 == 1, "Only odd keys should remain, got {}", k);
            }
        }

        cleanup_wal(&path);
    }
}