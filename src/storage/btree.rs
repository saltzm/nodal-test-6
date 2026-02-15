//! Disk-backed B-tree using the page, disk, and buffer pool infrastructure.
//!
//! Leaf nodes store (key, row) pairs sorted by key.
//! Internal nodes store keys and child page IDs.
//!
//! ## Page layout
//!
//! ### Leaf node (PageType::Leaf)
//! ```text
//! [num_entries: u16][next_leaf: u32][entry0][entry1]...
//! ```
//! Each entry: `[key_len: u16][key_bytes][row_len: u16][row_bytes]`
//!
//! ### Internal node (PageType::Internal)
//! ```text
//! [num_keys: u16][child0: u32][key0][child1: u32][key1][child2: u32]...
//! ```
//! Each key: `[key_len: u16][key_bytes]`
//!
//! There are num_keys + 1 children and num_keys keys.

use crate::error::{Error, Result};
use crate::storage::buffer::BufferPool;
use crate::storage::page::{Page, PageId, PageType, INVALID_PAGE_ID, PAGE_PAYLOAD_SIZE};
use crate::storage::serde::{deserialize_row, deserialize_value, serialize_row, serialize_value};
use crate::types::{Row, Value};

// Note: The B-tree order is dynamically determined by whether entries fit in
// a page (PAGE_PAYLOAD_SIZE = 4091 bytes). No fixed MAX_KEYS constant is used;
// instead, `leaf_entries_fit` and `internal_entries_fit` check at runtime.

/// A disk-backed B-tree that stores (Value, Row) key-value pairs.
pub struct BTree {
    root_page_id: PageId,
}

impl BTree {
    /// Create a new B-tree by allocating a root leaf page.
    pub fn create(pool: &mut BufferPool) -> Result<Self> {
        let page = pool.new_page(PageType::Leaf)?;
        let root_page_id = page.id;

        // Initialize as empty leaf: num_entries=0, next_leaf=INVALID
        let mut leaf_page = page;
        write_leaf_header(&mut leaf_page, 0, INVALID_PAGE_ID);
        pool.write_page(leaf_page)?;

        Ok(Self { root_page_id })
    }

    /// Open an existing B-tree with the given root page ID.
    pub fn open(root_page_id: PageId) -> Self {
        Self { root_page_id }
    }

    /// Return the root page ID (used for persisting the tree location).
    pub fn root_page_id(&self) -> PageId {
        self.root_page_id
    }

    /// Insert a key-value pair into the B-tree.
    pub fn insert(&mut self, pool: &mut BufferPool, key: Value, row: Row) -> Result<()> {
        let result = self.insert_recursive(pool, self.root_page_id, &key, &row)?;
        if let Some((median_key, new_child_id)) = result {
            // Root was split — create a new root internal node.
            let new_root = pool.new_page(PageType::Internal)?;
            let new_root_id = new_root.id;

            let mut page = new_root;
            write_internal_node(
                &mut page,
                &[median_key],
                &[self.root_page_id, new_child_id],
            )?;
            pool.write_page(page)?;
            self.root_page_id = new_root_id;
        }
        Ok(())
    }

    /// Recursive insert helper. Returns `Some((median_key, new_page_id))` if the
    /// node was split, or `None` if no split occurred.
    fn insert_recursive(
        &mut self,
        pool: &mut BufferPool,
        page_id: PageId,
        key: &Value,
        row: &Row,
    ) -> Result<Option<(Value, PageId)>> {
        let page = pool.fetch_page(page_id)?;

        if page.page_type == PageType::Leaf {
            self.insert_into_leaf(pool, page_id, key, row)
        } else {
            // Internal node — find child to descend into.
            let (keys, children) = read_internal_node(&page)?;
            let child_idx = find_child_index(&keys, key);
            let child_id = children[child_idx];

            let result = self.insert_recursive(pool, child_id, key, row)?;

            if let Some((median_key, new_child_id)) = result {
                // Child was split, insert the median key into this internal node.
                self.insert_into_internal(pool, page_id, median_key, new_child_id)
            } else {
                Ok(None)
            }
        }
    }

    /// Insert a key-value pair into a leaf node. Splits if necessary.
    fn insert_into_leaf(
        &self,
        pool: &mut BufferPool,
        page_id: PageId,
        key: &Value,
        row: &Row,
    ) -> Result<Option<(Value, PageId)>> {
        let page = pool.fetch_page(page_id)?;
        let (mut entries, next_leaf) = read_leaf_entries(&page)?;

        // Find insertion position (sorted order).
        let pos = entries
            .iter()
            .position(|(k, _)| k.partial_cmp(key) == Some(std::cmp::Ordering::Greater)
                              || k.partial_cmp(key).is_none())
            .unwrap_or(entries.len());

        // Check for duplicate key — update in place if found.
        if pos < entries.len() && entries[pos].0 == *key {
            entries[pos].1 = row.clone();
            let mut page = pool.fetch_page(page_id)?;
            write_leaf_entries(&mut page, &entries, next_leaf)?;
            pool.write_page(page)?;
            return Ok(None);
        }
        // Also check previous entry for duplicate
        if pos > 0 && entries[pos - 1].0 == *key {
            entries[pos - 1].1 = row.clone();
            let mut page = pool.fetch_page(page_id)?;
            write_leaf_entries(&mut page, &entries, next_leaf)?;
            pool.write_page(page)?;
            return Ok(None);
        }

        entries.insert(pos, (key.clone(), row.clone()));

        // Check if entries fit in a single page.
        if leaf_entries_fit(&entries) {
            let mut page = pool.fetch_page(page_id)?;
            write_leaf_entries(&mut page, &entries, next_leaf)?;
            pool.write_page(page)?;
            Ok(None)
        } else {
            // Split: left half stays, right half goes to new page.
            let mid = entries.len() / 2;
            let right_entries: Vec<(Value, Row)> = entries.drain(mid..).collect();
            let left_entries = entries;

            let median_key = right_entries[0].0.clone();

            // Create new right leaf page.
            let new_leaf = pool.new_page(PageType::Leaf)?;
            let new_leaf_id = new_leaf.id;

            // Update left leaf: keep left entries, point next to new leaf.
            let mut left_page = pool.fetch_page(page_id)?;
            write_leaf_entries(&mut left_page, &left_entries, new_leaf_id)?;
            pool.write_page(left_page)?;

            // Write right leaf: right entries, point next to old next_leaf.
            let mut right_page = pool.fetch_page(new_leaf_id)?;
            write_leaf_entries(&mut right_page, &right_entries, next_leaf)?;
            pool.write_page(right_page)?;

            Ok(Some((median_key, new_leaf_id)))
        }
    }

    /// Insert a new key and child pointer into an internal node. Splits if necessary.
    fn insert_into_internal(
        &self,
        pool: &mut BufferPool,
        page_id: PageId,
        new_key: Value,
        new_child_id: PageId,
    ) -> Result<Option<(Value, PageId)>> {
        let page = pool.fetch_page(page_id)?;
        let (mut keys, mut children) = read_internal_node(&page)?;

        // Find position to insert the new key.
        let pos = keys
            .iter()
            .position(|k| k.partial_cmp(&new_key) == Some(std::cmp::Ordering::Greater)
                        || k.partial_cmp(&new_key).is_none())
            .unwrap_or(keys.len());

        keys.insert(pos, new_key);
        children.insert(pos + 1, new_child_id);

        // Check if it fits.
        if internal_entries_fit(&keys, &children) {
            let mut page = pool.fetch_page(page_id)?;
            write_internal_node(&mut page, &keys, &children)?;
            pool.write_page(page)?;
            Ok(None)
        } else {
            // Split internal node.
            let mid = keys.len() / 2;
            let median_key = keys[mid].clone();

            // Right half: keys[mid+1..], children[mid+1..]
            let right_keys: Vec<Value> = keys.drain(mid + 1..).collect();
            let right_children: Vec<PageId> = children.drain(mid + 1..).collect();
            // Remove the median key from left
            keys.pop(); // remove keys[mid]

            let left_keys = keys;
            let left_children = children;

            // Create new right internal page.
            let new_internal = pool.new_page(PageType::Internal)?;
            let new_internal_id = new_internal.id;

            // Write left node.
            let mut left_page = pool.fetch_page(page_id)?;
            write_internal_node(&mut left_page, &left_keys, &left_children)?;
            pool.write_page(left_page)?;

            // Write right node.
            let mut right_page = pool.fetch_page(new_internal_id)?;
            write_internal_node(&mut right_page, &right_keys, &right_children)?;
            pool.write_page(right_page)?;

            Ok(Some((median_key, new_internal_id)))
        }
    }

    /// Look up a key in the B-tree. Returns the associated Row if found.
    pub fn get(&self, pool: &mut BufferPool, key: &Value) -> Result<Option<Row>> {
        self.get_recursive(pool, self.root_page_id, key)
    }

    fn get_recursive(
        &self,
        pool: &mut BufferPool,
        page_id: PageId,
        key: &Value,
    ) -> Result<Option<Row>> {
        let page = pool.fetch_page(page_id)?;

        if page.page_type == PageType::Leaf {
            let (entries, _next_leaf) = read_leaf_entries(&page)?;
            for (k, row) in &entries {
                if k == key {
                    return Ok(Some(row.clone()));
                }
            }
            Ok(None)
        } else {
            let (keys, children) = read_internal_node(&page)?;
            let child_idx = find_child_index(&keys, key);
            self.get_recursive(pool, children[child_idx], key)
        }
    }

    /// Delete a key from the B-tree. Returns true if the key was found and removed.
    /// This is a simple implementation that just removes the entry from the leaf
    /// without merge/rebalance.
    pub fn delete(&mut self, pool: &mut BufferPool, key: &Value) -> Result<bool> {
        self.delete_recursive(pool, self.root_page_id, key)
    }

    fn delete_recursive(
        &mut self,
        pool: &mut BufferPool,
        page_id: PageId,
        key: &Value,
    ) -> Result<bool> {
        let page = pool.fetch_page(page_id)?;

        if page.page_type == PageType::Leaf {
            let (mut entries, next_leaf) = read_leaf_entries(&page)?;
            let original_len = entries.len();
            entries.retain(|(k, _)| k != key);
            if entries.len() == original_len {
                return Ok(false);
            }
            let mut page = pool.fetch_page(page_id)?;
            write_leaf_entries(&mut page, &entries, next_leaf)?;
            pool.write_page(page)?;
            Ok(true)
        } else {
            let (keys, children) = read_internal_node(&page)?;
            let child_idx = find_child_index(&keys, key);
            self.delete_recursive(pool, children[child_idx], key)
        }
    }

    /// Scan all entries in the B-tree in key order (leaf-level linked scan).
    pub fn scan(&self, pool: &mut BufferPool) -> Result<Vec<(Value, Row)>> {
        // Find the leftmost leaf by always going to the first child.
        let leftmost_leaf = self.find_leftmost_leaf(pool, self.root_page_id)?;
        let mut results = Vec::new();
        let mut current_leaf = leftmost_leaf;

        loop {
            let page = pool.fetch_page(current_leaf)?;
            let (entries, next_leaf) = read_leaf_entries(&page)?;
            results.extend(entries);

            if next_leaf == INVALID_PAGE_ID {
                break;
            }
            current_leaf = next_leaf;
        }

        Ok(results)
    }

    /// Range scan: returns entries where start <= key <= end, in key order.
    pub fn range_scan(
        &self,
        pool: &mut BufferPool,
        start: &Value,
        end: &Value,
    ) -> Result<Vec<(Value, Row)>> {
        // Find the leaf containing the start key.
        let leaf_id = self.find_leaf_for_key(pool, self.root_page_id, start)?;
        let mut results = Vec::new();
        let mut current_leaf = leaf_id;

        'outer: loop {
            let page = pool.fetch_page(current_leaf)?;
            let (entries, next_leaf) = read_leaf_entries(&page)?;

            for (k, row) in entries {
                if let Some(ord) = k.partial_cmp(end) {
                    if ord == std::cmp::Ordering::Greater {
                        break 'outer;
                    }
                }
                if let Some(ord) = k.partial_cmp(start) {
                    if ord == std::cmp::Ordering::Less {
                        continue;
                    }
                }
                results.push((k, row));
            }

            if next_leaf == INVALID_PAGE_ID {
                break;
            }
            current_leaf = next_leaf;
        }

        Ok(results)
    }

    /// Find the leftmost leaf page by descending through first children.
    fn find_leftmost_leaf(&self, pool: &mut BufferPool, page_id: PageId) -> Result<PageId> {
        let page = pool.fetch_page(page_id)?;
        if page.page_type == PageType::Leaf {
            return Ok(page_id);
        }
        let (_keys, children) = read_internal_node(&page)?;
        self.find_leftmost_leaf(pool, children[0])
    }

    /// Find the leaf page that should contain the given key.
    fn find_leaf_for_key(
        &self,
        pool: &mut BufferPool,
        page_id: PageId,
        key: &Value,
    ) -> Result<PageId> {
        let page = pool.fetch_page(page_id)?;
        if page.page_type == PageType::Leaf {
            return Ok(page_id);
        }
        let (keys, children) = read_internal_node(&page)?;
        let child_idx = find_child_index(&keys, key);
        self.find_leaf_for_key(pool, children[child_idx], key)
    }
}

// ─── Page layout helpers ───

/// Write the leaf header (num_entries + next_leaf pointer) into page data.
fn write_leaf_header(page: &mut Page, num_entries: u16, next_leaf: PageId) {
    page.data[0..2].copy_from_slice(&num_entries.to_le_bytes());
    page.data[2..6].copy_from_slice(&next_leaf.to_le_bytes());
}

/// Read leaf entries from a page.
/// Returns (entries, next_leaf_page_id).
fn read_leaf_entries(page: &Page) -> Result<(Vec<(Value, Row)>, PageId)> {
    let data = &page.data;
    if data.len() < 6 {
        return Err(Error::Storage("leaf page too small".into()));
    }

    let num_entries = u16::from_le_bytes([data[0], data[1]]) as usize;
    let next_leaf = u32::from_le_bytes([data[2], data[3], data[4], data[5]]);

    let mut offset = 6;
    let mut entries = Vec::with_capacity(num_entries);

    for _ in 0..num_entries {
        // Read key_len
        if offset + 2 > data.len() {
            return Err(Error::Storage("leaf entry: truncated key length".into()));
        }
        let key_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;

        // Read key bytes
        if offset + key_len > data.len() {
            return Err(Error::Storage("leaf entry: truncated key data".into()));
        }
        let (key, _) = deserialize_value(&data[offset..offset + key_len])?;
        offset += key_len;

        // Read row_len
        if offset + 2 > data.len() {
            return Err(Error::Storage("leaf entry: truncated row length".into()));
        }
        let row_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;

        // Read row bytes
        if offset + row_len > data.len() {
            return Err(Error::Storage("leaf entry: truncated row data".into()));
        }
        let (row, _) = deserialize_row(&data[offset..offset + row_len])?;
        offset += row_len;

        entries.push((key, row));
    }

    Ok((entries, next_leaf))
}

/// Write leaf entries into a page.
fn write_leaf_entries(
    page: &mut Page,
    entries: &[(Value, Row)],
    next_leaf: PageId,
) -> Result<()> {
    let num_entries = entries.len() as u16;
    page.data.fill(0);
    page.data[0..2].copy_from_slice(&num_entries.to_le_bytes());
    page.data[2..6].copy_from_slice(&next_leaf.to_le_bytes());

    let mut offset = 6;
    for (key, row) in entries {
        let key_bytes = serialize_value(key);
        let row_bytes = serialize_row(row);

        if offset + 2 + key_bytes.len() + 2 + row_bytes.len() > PAGE_PAYLOAD_SIZE {
            return Err(Error::Storage("leaf page overflow".into()));
        }

        // Write key_len + key_bytes
        page.data[offset..offset + 2]
            .copy_from_slice(&(key_bytes.len() as u16).to_le_bytes());
        offset += 2;
        page.data[offset..offset + key_bytes.len()].copy_from_slice(&key_bytes);
        offset += key_bytes.len();

        // Write row_len + row_bytes
        page.data[offset..offset + 2]
            .copy_from_slice(&(row_bytes.len() as u16).to_le_bytes());
        offset += 2;
        page.data[offset..offset + row_bytes.len()].copy_from_slice(&row_bytes);
        offset += row_bytes.len();
    }

    Ok(())
}

/// Check if the given entries fit into a single leaf page.
fn leaf_entries_fit(entries: &[(Value, Row)]) -> bool {
    let mut size = 6; // header: num_entries (2) + next_leaf (4)
    for (key, row) in entries {
        let key_bytes = serialize_value(key);
        let row_bytes = serialize_row(row);
        size += 2 + key_bytes.len() + 2 + row_bytes.len();
    }
    size <= PAGE_PAYLOAD_SIZE
}

/// Read internal node: returns (keys, children).
fn read_internal_node(page: &Page) -> Result<(Vec<Value>, Vec<PageId>)> {
    let data = &page.data;
    if data.len() < 2 {
        return Err(Error::Storage("internal node too small".into()));
    }

    let num_keys = u16::from_le_bytes([data[0], data[1]]) as usize;
    let mut offset = 2;

    // Read first child.
    if offset + 4 > data.len() {
        return Err(Error::Storage("internal node: truncated child".into()));
    }
    let mut children = Vec::with_capacity(num_keys + 1);
    children.push(u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]));
    offset += 4;

    let mut keys = Vec::with_capacity(num_keys);
    for _ in 0..num_keys {
        // Read key_len
        if offset + 2 > data.len() {
            return Err(Error::Storage("internal node: truncated key length".into()));
        }
        let key_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;

        // Read key bytes
        if offset + key_len > data.len() {
            return Err(Error::Storage("internal node: truncated key data".into()));
        }
        let (key, _) = deserialize_value(&data[offset..offset + key_len])?;
        offset += key_len;
        keys.push(key);

        // Read next child
        if offset + 4 > data.len() {
            return Err(Error::Storage("internal node: truncated child".into()));
        }
        children.push(u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]));
        offset += 4;
    }

    Ok((keys, children))
}

/// Write an internal node to a page.
fn write_internal_node(
    page: &mut Page,
    keys: &[Value],
    children: &[PageId],
) -> Result<()> {
    assert_eq!(children.len(), keys.len() + 1);

    let num_keys = keys.len() as u16;
    page.data.fill(0);
    page.data[0..2].copy_from_slice(&num_keys.to_le_bytes());

    let mut offset = 2;

    // Write first child.
    page.data[offset..offset + 4].copy_from_slice(&children[0].to_le_bytes());
    offset += 4;

    for i in 0..keys.len() {
        let key_bytes = serialize_value(&keys[i]);

        if offset + 2 + key_bytes.len() + 4 > PAGE_PAYLOAD_SIZE {
            return Err(Error::Storage("internal page overflow".into()));
        }

        // Write key_len + key_bytes
        page.data[offset..offset + 2]
            .copy_from_slice(&(key_bytes.len() as u16).to_le_bytes());
        offset += 2;
        page.data[offset..offset + key_bytes.len()].copy_from_slice(&key_bytes);
        offset += key_bytes.len();

        // Write child
        page.data[offset..offset + 4].copy_from_slice(&children[i + 1].to_le_bytes());
        offset += 4;
    }

    Ok(())
}

/// Check if keys and children fit into a single internal page.
fn internal_entries_fit(keys: &[Value], _children: &[PageId]) -> bool {
    let mut size = 2 + 4; // header: num_keys (2) + first child (4)
    for key in keys {
        let key_bytes = serialize_value(key);
        size += 2 + key_bytes.len() + 4; // key_len + key_bytes + child
    }
    size <= PAGE_PAYLOAD_SIZE
}

/// Find which child to descend into for a given key.
/// Returns the index into the children array.
fn find_child_index(keys: &[Value], key: &Value) -> usize {
    for (i, k) in keys.iter().enumerate() {
        match key.partial_cmp(k) {
            Some(std::cmp::Ordering::Less) => return i,
            None => return i, // incomparable types — go left
            _ => {}
        }
    }
    keys.len() // key >= all keys, go to rightmost child
}

// ───────────────────────── Tests ─────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::disk::DiskManager;
    use std::fs;
    use std::path::PathBuf;

    fn temp_db_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("toydb_test_btree");
        fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn cleanup(path: &std::path::Path) {
        let _ = fs::remove_file(path);
    }

    fn make_pool(name: &str) -> (BufferPool, PathBuf) {
        let path = temp_db_path(name);
        cleanup(&path);
        let dm = DiskManager::open(&path).unwrap();
        (BufferPool::new(dm, 100), path)
    }

    fn make_row(vals: Vec<Value>) -> Row {
        Row::new(vals)
    }

    #[test]
    fn test_btree_create() {
        let (mut pool, path) = make_pool("btree_create.db");
        let tree = BTree::create(&mut pool).unwrap();
        assert_eq!(tree.root_page_id(), 0);
        cleanup(&path);
    }

    #[test]
    fn test_btree_insert_get_single() {
        let (mut pool, path) = make_pool("btree_single.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        let key = Value::Integer(42);
        let row = make_row(vec![Value::Integer(42), Value::Text("hello".into())]);

        tree.insert(&mut pool, key.clone(), row.clone()).unwrap();

        let result = tree.get(&mut pool, &key).unwrap();
        assert_eq!(result, Some(row));

        cleanup(&path);
    }

    #[test]
    fn test_btree_get_nonexistent() {
        let (mut pool, path) = make_pool("btree_noexist.db");
        let tree = BTree::create(&mut pool).unwrap();

        let result = tree.get(&mut pool, &Value::Integer(999)).unwrap();
        assert_eq!(result, None);

        cleanup(&path);
    }

    #[test]
    fn test_btree_insert_get_multiple() {
        let (mut pool, path) = make_pool("btree_multi.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        for i in 0..20 {
            let key = Value::Integer(i);
            let row = make_row(vec![Value::Integer(i), Value::Text(format!("row_{}", i))]);
            tree.insert(&mut pool, key, row).unwrap();
        }

        for i in 0..20 {
            let result = tree.get(&mut pool, &Value::Integer(i)).unwrap();
            let expected = make_row(vec![Value::Integer(i), Value::Text(format!("row_{}", i))]);
            assert_eq!(result, Some(expected), "mismatch at key {}", i);
        }

        // Non-existent key
        assert_eq!(tree.get(&mut pool, &Value::Integer(100)).unwrap(), None);

        cleanup(&path);
    }

    #[test]
    fn test_btree_insert_reverse_order() {
        let (mut pool, path) = make_pool("btree_reverse.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        for i in (0..20).rev() {
            let key = Value::Integer(i);
            let row = make_row(vec![Value::Integer(i)]);
            tree.insert(&mut pool, key, row).unwrap();
        }

        for i in 0..20 {
            let result = tree.get(&mut pool, &Value::Integer(i)).unwrap();
            assert_eq!(result, Some(make_row(vec![Value::Integer(i)])));
        }

        cleanup(&path);
    }

    #[test]
    fn test_btree_update_existing_key() {
        let (mut pool, path) = make_pool("btree_update.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        let key = Value::Integer(1);
        let row1 = make_row(vec![Value::Text("first".into())]);
        let row2 = make_row(vec![Value::Text("second".into())]);

        tree.insert(&mut pool, key.clone(), row1).unwrap();
        tree.insert(&mut pool, key.clone(), row2.clone()).unwrap();

        let result = tree.get(&mut pool, &key).unwrap();
        assert_eq!(result, Some(row2));

        cleanup(&path);
    }

    #[test]
    fn test_btree_delete_existing() {
        let (mut pool, path) = make_pool("btree_delete.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        let key = Value::Integer(5);
        let row = make_row(vec![Value::Integer(5)]);
        tree.insert(&mut pool, key.clone(), row).unwrap();

        assert!(tree.delete(&mut pool, &key).unwrap());
        assert_eq!(tree.get(&mut pool, &key).unwrap(), None);

        cleanup(&path);
    }

    #[test]
    fn test_btree_delete_nonexistent() {
        let (mut pool, path) = make_pool("btree_delete_noexist.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        assert!(!tree.delete(&mut pool, &Value::Integer(99)).unwrap());

        cleanup(&path);
    }

    #[test]
    fn test_btree_scan_empty() {
        let (mut pool, path) = make_pool("btree_scan_empty.db");
        let tree = BTree::create(&mut pool).unwrap();

        let results = tree.scan(&mut pool).unwrap();
        assert!(results.is_empty());

        cleanup(&path);
    }

    #[test]
    fn test_btree_scan_sorted_order() {
        let (mut pool, path) = make_pool("btree_scan_sorted.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        // Insert in random-ish order
        for i in [5, 3, 8, 1, 9, 2, 7, 4, 6, 0] {
            let key = Value::Integer(i);
            let row = make_row(vec![Value::Integer(i)]);
            tree.insert(&mut pool, key, row).unwrap();
        }

        let results = tree.scan(&mut pool).unwrap();
        let keys: Vec<i64> = results
            .iter()
            .map(|(k, _)| k.as_integer().unwrap())
            .collect();
        assert_eq!(keys, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

        cleanup(&path);
    }

    #[test]
    fn test_btree_range_scan() {
        let (mut pool, path) = make_pool("btree_range.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        for i in 0..10 {
            let key = Value::Integer(i);
            let row = make_row(vec![Value::Integer(i)]);
            tree.insert(&mut pool, key, row).unwrap();
        }

        let results = tree
            .range_scan(&mut pool, &Value::Integer(3), &Value::Integer(7))
            .unwrap();
        let keys: Vec<i64> = results
            .iter()
            .map(|(k, _)| k.as_integer().unwrap())
            .collect();
        assert_eq!(keys, vec![3, 4, 5, 6, 7]);

        cleanup(&path);
    }

    #[test]
    fn test_btree_range_scan_empty_range() {
        let (mut pool, path) = make_pool("btree_range_empty.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        for i in 0..5 {
            let key = Value::Integer(i);
            let row = make_row(vec![Value::Integer(i)]);
            tree.insert(&mut pool, key, row).unwrap();
        }

        let results = tree
            .range_scan(&mut pool, &Value::Integer(10), &Value::Integer(20))
            .unwrap();
        assert!(results.is_empty());

        cleanup(&path);
    }

    #[test]
    fn test_btree_many_inserts_causes_splits() {
        let (mut pool, path) = make_pool("btree_splits.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        // Insert enough entries to cause multiple splits.
        // Each leaf entry with an integer key and simple row is about 22 bytes.
        // PAGE_PAYLOAD_SIZE = 4091. Leaf header = 6. So ~(4091-6)/22 ≈ 185 entries per leaf.
        // We'll insert 500 entries to trigger multiple splits.
        let n = 500;
        for i in 0..n {
            let key = Value::Integer(i);
            let row = make_row(vec![Value::Integer(i), Value::Text(format!("v{}", i))]);
            tree.insert(&mut pool, key, row).unwrap();
        }

        // Verify all entries can be retrieved.
        for i in 0..n {
            let result = tree.get(&mut pool, &Value::Integer(i)).unwrap();
            assert!(result.is_some(), "key {} not found", i);
            let expected = make_row(vec![Value::Integer(i), Value::Text(format!("v{}", i))]);
            assert_eq!(result.unwrap(), expected, "wrong value for key {}", i);
        }

        // Verify scan returns all entries in order.
        let results = tree.scan(&mut pool).unwrap();
        assert_eq!(results.len(), n as usize);
        for (idx, (key, _row)) in results.iter().enumerate() {
            assert_eq!(key, &Value::Integer(idx as i64));
        }

        cleanup(&path);
    }

    #[test]
    fn test_btree_text_keys() {
        let (mut pool, path) = make_pool("btree_text.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        let names = vec!["charlie", "alice", "bob", "dave", "eve"];
        for name in &names {
            let key = Value::Text(name.to_string());
            let row = make_row(vec![Value::Text(name.to_string())]);
            tree.insert(&mut pool, key, row).unwrap();
        }

        for name in &names {
            let result = tree
                .get(&mut pool, &Value::Text(name.to_string()))
                .unwrap();
            assert!(result.is_some());
        }

        // Scan should return in alphabetical order.
        let results = tree.scan(&mut pool).unwrap();
        let scanned_keys: Vec<&str> = results
            .iter()
            .map(|(k, _)| k.as_text().unwrap())
            .collect();
        assert_eq!(scanned_keys, vec!["alice", "bob", "charlie", "dave", "eve"]);

        cleanup(&path);
    }

    #[test]
    fn test_btree_delete_multiple() {
        let (mut pool, path) = make_pool("btree_delete_multi.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        for i in 0..10 {
            let key = Value::Integer(i);
            let row = make_row(vec![Value::Integer(i)]);
            tree.insert(&mut pool, key, row).unwrap();
        }

        // Delete even numbers.
        for i in (0..10).step_by(2) {
            assert!(tree.delete(&mut pool, &Value::Integer(i)).unwrap());
        }

        // Verify only odd numbers remain.
        let results = tree.scan(&mut pool).unwrap();
        let keys: Vec<i64> = results
            .iter()
            .map(|(k, _)| k.as_integer().unwrap())
            .collect();
        assert_eq!(keys, vec![1, 3, 5, 7, 9]);

        cleanup(&path);
    }

    #[test]
    fn test_btree_persistence() {
        let path = temp_db_path("btree_persist.db");
        cleanup(&path);

        let root_page_id;

        // Create tree and insert data.
        {
            let dm = DiskManager::open(&path).unwrap();
            let mut pool = BufferPool::new(dm, 100);
            let mut tree = BTree::create(&mut pool).unwrap();

            for i in 0..10 {
                let key = Value::Integer(i);
                let row = make_row(vec![Value::Integer(i)]);
                tree.insert(&mut pool, key, row).unwrap();
            }

            root_page_id = tree.root_page_id();
            pool.flush_all().unwrap();
        }

        // Reopen and verify data.
        {
            let dm = DiskManager::open(&path).unwrap();
            let mut pool = BufferPool::new(dm, 100);
            let tree = BTree::open(root_page_id);

            for i in 0..10 {
                let result = tree.get(&mut pool, &Value::Integer(i)).unwrap();
                assert_eq!(
                    result,
                    Some(make_row(vec![Value::Integer(i)])),
                    "key {} not found after reopen",
                    i
                );
            }
        }

        cleanup(&path);
    }
}