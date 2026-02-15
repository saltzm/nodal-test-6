//! Comprehensive tests for the B-tree module.

#[cfg(test)]
mod tests {
    use crate::storage::btree::BTree;
    use crate::storage::buffer::BufferPool;
    use crate::storage::disk::DiskManager;
    use crate::types::{Row, Value};
    use std::fs;
    use std::path::PathBuf;

    fn temp_db_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("toydb_test_btree_comprehensive");
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

    // ── Empty tree tests ──

    #[test]
    fn test_empty_tree_get_returns_none() {
        let (mut pool, path) = make_pool("comp_empty_get.db");
        let tree = BTree::create(&mut pool).unwrap();

        assert_eq!(tree.get(&mut pool, &Value::Integer(0)).unwrap(), None);
        assert_eq!(tree.get(&mut pool, &Value::Integer(999)).unwrap(), None);
        assert_eq!(tree.get(&mut pool, &Value::Text("hello".into())).unwrap(), None);
        assert_eq!(tree.get(&mut pool, &Value::Float(3.14)).unwrap(), None);
        assert_eq!(tree.get(&mut pool, &Value::Boolean(true)).unwrap(), None);
        assert_eq!(tree.get(&mut pool, &Value::Null).unwrap(), None);

        cleanup(&path);
    }

    #[test]
    fn test_empty_tree_scan_returns_empty() {
        let (mut pool, path) = make_pool("comp_empty_scan.db");
        let tree = BTree::create(&mut pool).unwrap();

        let results = tree.scan(&mut pool).unwrap();
        assert!(results.is_empty());

        cleanup(&path);
    }

    #[test]
    fn test_empty_tree_range_scan_returns_empty() {
        let (mut pool, path) = make_pool("comp_empty_range.db");
        let tree = BTree::create(&mut pool).unwrap();

        let results = tree
            .range_scan(&mut pool, &Value::Integer(0), &Value::Integer(100))
            .unwrap();
        assert!(results.is_empty());

        let results = tree
            .range_scan(&mut pool, &Value::Float(0.0), &Value::Float(100.0))
            .unwrap();
        assert!(results.is_empty());

        let results = tree
            .range_scan(&mut pool, &Value::Text("a".into()), &Value::Text("z".into()))
            .unwrap();
        assert!(results.is_empty());

        cleanup(&path);
    }

    // ── Single element tests ──

    #[test]
    fn test_single_element_insert_get_scan_range() {
        let (mut pool, path) = make_pool("comp_single.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        let key = Value::Integer(42);
        let row = make_row(vec![Value::Integer(42), Value::Text("only".into())]);
        tree.insert(&mut pool, key.clone(), row.clone()).unwrap();

        // get
        let result = tree.get(&mut pool, &key).unwrap();
        assert_eq!(result, Some(row.clone()));

        // scan
        let results = tree.scan(&mut pool).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (key.clone(), row.clone()));

        // range_scan including the element
        let results = tree
            .range_scan(&mut pool, &Value::Integer(0), &Value::Integer(100))
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (key.clone(), row.clone()));

        // range_scan exactly on the element
        let results = tree
            .range_scan(&mut pool, &Value::Integer(42), &Value::Integer(42))
            .unwrap();
        assert_eq!(results.len(), 1);

        // range_scan not including the element
        let results = tree
            .range_scan(&mut pool, &Value::Integer(43), &Value::Integer(100))
            .unwrap();
        assert!(results.is_empty());

        let results = tree
            .range_scan(&mut pool, &Value::Integer(0), &Value::Integer(41))
            .unwrap();
        assert!(results.is_empty());

        cleanup(&path);
    }

    // ── Key type tests ──

    #[test]
    fn test_float_keys_insert_lookup() {
        let (mut pool, path) = make_pool("comp_float.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        let floats = vec![3.14, 1.0, 2.718, 0.5, 42.0, 100.1];
        for &f in &floats {
            let key = Value::Float(f);
            let row = make_row(vec![Value::Float(f), Value::Text(format!("f{}", f))]);
            tree.insert(&mut pool, key, row).unwrap();
        }

        // Lookup each float key
        for &f in &floats {
            let result = tree.get(&mut pool, &Value::Float(f)).unwrap();
            assert!(result.is_some(), "Float key {} not found", f);
            let expected = make_row(vec![Value::Float(f), Value::Text(format!("f{}", f))]);
            assert_eq!(result.unwrap(), expected);
        }

        // Non-existent float
        assert_eq!(tree.get(&mut pool, &Value::Float(99.99)).unwrap(), None);

        // Scan should return in sorted order
        let results = tree.scan(&mut pool).unwrap();
        let keys: Vec<f64> = results
            .iter()
            .map(|(k, _)| match k {
                Value::Float(f) => *f,
                _ => panic!("expected float key"),
            })
            .collect();
        let mut sorted_floats = floats.clone();
        sorted_floats.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(keys, sorted_floats);

        cleanup(&path);
    }

    #[test]
    fn test_float_range_scan() {
        let (mut pool, path) = make_pool("comp_float_range.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        for i in 0..20 {
            let f = i as f64 * 0.5;
            let key = Value::Float(f);
            let row = make_row(vec![Value::Float(f)]);
            tree.insert(&mut pool, key, row).unwrap();
        }

        // Range scan [2.0, 5.0]
        let results = tree
            .range_scan(&mut pool, &Value::Float(2.0), &Value::Float(5.0))
            .unwrap();
        let keys: Vec<f64> = results
            .iter()
            .map(|(k, _)| match k {
                Value::Float(f) => *f,
                _ => panic!("expected float key"),
            })
            .collect();
        assert_eq!(keys, vec![2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]);

        cleanup(&path);
    }

    #[test]
    fn test_boolean_keys() {
        let (mut pool, path) = make_pool("comp_bool.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        let row_true = make_row(vec![Value::Boolean(true), Value::Text("yes".into())]);
        let row_false = make_row(vec![Value::Boolean(false), Value::Text("no".into())]);

        tree.insert(&mut pool, Value::Boolean(true), row_true.clone())
            .unwrap();
        tree.insert(&mut pool, Value::Boolean(false), row_false.clone())
            .unwrap();

        assert_eq!(
            tree.get(&mut pool, &Value::Boolean(true)).unwrap(),
            Some(row_true)
        );
        assert_eq!(
            tree.get(&mut pool, &Value::Boolean(false)).unwrap(),
            Some(row_false)
        );

        // Scan returns false before true (false < true)
        let results = tree.scan(&mut pool).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, Value::Boolean(false));
        assert_eq!(results[1].0, Value::Boolean(true));

        cleanup(&path);
    }

    #[test]
    fn test_null_key() {
        let (mut pool, path) = make_pool("comp_null.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        let key = Value::Null;
        let row = make_row(vec![Value::Null, Value::Text("null_row".into())]);
        tree.insert(&mut pool, key.clone(), row.clone()).unwrap();

        let result = tree.get(&mut pool, &Value::Null).unwrap();
        assert_eq!(result, Some(row));

        cleanup(&path);
    }

    // ── Large insert tests ──

    #[test]
    fn test_large_inserts_200_all_retrievable() {
        let (mut pool, path) = make_pool("comp_large200.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        let n = 200;
        for i in 0..n {
            let key = Value::Integer(i);
            let row = make_row(vec![Value::Integer(i), Value::Text(format!("data_{}", i))]);
            tree.insert(&mut pool, key, row).unwrap();
        }

        // Verify all via get
        for i in 0..n {
            let result = tree.get(&mut pool, &Value::Integer(i)).unwrap();
            assert!(result.is_some(), "key {} not found after 200 inserts", i);
            let row = result.unwrap();
            assert_eq!(row.values[0], Value::Integer(i));
            assert_eq!(row.values[1], Value::Text(format!("data_{}", i)));
        }

        // Verify full scan
        let results = tree.scan(&mut pool).unwrap();
        assert_eq!(results.len(), n as usize);
        for (idx, (key, _)) in results.iter().enumerate() {
            assert_eq!(key, &Value::Integer(idx as i64));
        }

        cleanup(&path);
    }

    #[test]
    fn test_random_order_inserts_100() {
        let (mut pool, path) = make_pool("comp_random100.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        // Create a deterministic pseudo-random ordering
        let n = 100i64;
        let mut order: Vec<i64> = (0..n).collect();
        for i in 0..order.len() {
            let j = ((i * 37 + 13) % order.len()) as usize;
            order.swap(i, j);
        }

        for &i in &order {
            let key = Value::Integer(i);
            let row = make_row(vec![Value::Integer(i)]);
            tree.insert(&mut pool, key, row).unwrap();
        }

        // All entries retrievable
        for i in 0..n {
            let result = tree.get(&mut pool, &Value::Integer(i)).unwrap();
            assert!(result.is_some(), "key {} not found after random insert", i);
        }

        // Scan returns sorted order
        let results = tree.scan(&mut pool).unwrap();
        assert_eq!(results.len(), n as usize);
        for (idx, (key, _)) in results.iter().enumerate() {
            assert_eq!(key, &Value::Integer(idx as i64));
        }

        cleanup(&path);
    }

    // ── Range scan variations ──

    #[test]
    fn test_range_scan_boundary_inclusive() {
        let (mut pool, path) = make_pool("comp_range_inclusive.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        for i in 0..10 {
            let key = Value::Integer(i);
            let row = make_row(vec![Value::Integer(i)]);
            tree.insert(&mut pool, key, row).unwrap();
        }

        // [3, 7] inclusive
        let results = tree
            .range_scan(&mut pool, &Value::Integer(3), &Value::Integer(7))
            .unwrap();
        let keys: Vec<i64> = results.iter().map(|(k, _)| k.as_integer().unwrap()).collect();
        assert_eq!(keys, vec![3, 4, 5, 6, 7]);

        // Single element range [5, 5]
        let results = tree
            .range_scan(&mut pool, &Value::Integer(5), &Value::Integer(5))
            .unwrap();
        let keys: Vec<i64> = results.iter().map(|(k, _)| k.as_integer().unwrap()).collect();
        assert_eq!(keys, vec![5]);

        // Edge: first element [0, 0]
        let results = tree
            .range_scan(&mut pool, &Value::Integer(0), &Value::Integer(0))
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, Value::Integer(0));

        // Edge: last element [9, 9]
        let results = tree
            .range_scan(&mut pool, &Value::Integer(9), &Value::Integer(9))
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, Value::Integer(9));

        // Range below data
        let results = tree
            .range_scan(&mut pool, &Value::Integer(-5), &Value::Integer(-1))
            .unwrap();
        assert!(results.is_empty());

        // Range above data
        let results = tree
            .range_scan(&mut pool, &Value::Integer(10), &Value::Integer(20))
            .unwrap();
        assert!(results.is_empty());

        cleanup(&path);
    }

    #[test]
    fn test_range_scan_full_range_equals_scan() {
        let (mut pool, path) = make_pool("comp_range_full.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        for i in 0..50 {
            let key = Value::Integer(i);
            let row = make_row(vec![Value::Integer(i)]);
            tree.insert(&mut pool, key, row).unwrap();
        }

        let range_results = tree
            .range_scan(&mut pool, &Value::Integer(0), &Value::Integer(49))
            .unwrap();
        let scan_results = tree.scan(&mut pool).unwrap();
        assert_eq!(range_results.len(), scan_results.len());
        assert_eq!(range_results, scan_results);

        // Superset range
        let range_results = tree
            .range_scan(&mut pool, &Value::Integer(-100), &Value::Integer(1000))
            .unwrap();
        assert_eq!(range_results.len(), 50);

        cleanup(&path);
    }

    #[test]
    fn test_text_range_scan() {
        let (mut pool, path) = make_pool("comp_text_range.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        let names = vec!["alice", "bob", "charlie", "dave", "eve", "frank", "grace"];
        for name in &names {
            let key = Value::Text(name.to_string());
            let row = make_row(vec![Value::Text(name.to_string())]);
            tree.insert(&mut pool, key, row).unwrap();
        }

        let results = tree
            .range_scan(
                &mut pool,
                &Value::Text("bob".into()),
                &Value::Text("eve".into()),
            )
            .unwrap();
        let keys: Vec<&str> = results.iter().map(|(k, _)| k.as_text().unwrap()).collect();
        assert_eq!(keys, vec!["bob", "charlie", "dave", "eve"]);

        cleanup(&path);
    }

    // ── Delete tests ──

    #[test]
    fn test_delete_then_scan_excludes_deleted() {
        let (mut pool, path) = make_pool("comp_delete_scan.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        for i in 0..10 {
            let key = Value::Integer(i);
            let row = make_row(vec![Value::Integer(i)]);
            tree.insert(&mut pool, key, row).unwrap();
        }

        assert!(tree.delete(&mut pool, &Value::Integer(5)).unwrap());
        assert_eq!(tree.get(&mut pool, &Value::Integer(5)).unwrap(), None);

        let results = tree.scan(&mut pool).unwrap();
        let keys: Vec<i64> = results.iter().map(|(k, _)| k.as_integer().unwrap()).collect();
        assert_eq!(keys, vec![0, 1, 2, 3, 4, 6, 7, 8, 9]);
        assert_eq!(results.len(), 9);

        // range_scan also excludes deleted
        let results = tree
            .range_scan(&mut pool, &Value::Integer(3), &Value::Integer(7))
            .unwrap();
        let keys: Vec<i64> = results.iter().map(|(k, _)| k.as_integer().unwrap()).collect();
        assert_eq!(keys, vec![3, 4, 6, 7]);

        cleanup(&path);
    }

    #[test]
    fn test_delete_nonexistent_returns_false() {
        let (mut pool, path) = make_pool("comp_delete_noexist.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        // Delete from empty tree
        assert!(!tree.delete(&mut pool, &Value::Integer(42)).unwrap());

        // Insert some, delete nonexistent
        tree.insert(&mut pool, Value::Integer(1), make_row(vec![Value::Integer(1)]))
            .unwrap();
        assert!(!tree.delete(&mut pool, &Value::Integer(42)).unwrap());

        // Existing data unchanged
        assert_eq!(
            tree.get(&mut pool, &Value::Integer(1)).unwrap(),
            Some(make_row(vec![Value::Integer(1)]))
        );

        cleanup(&path);
    }

    #[test]
    fn test_insert_after_delete() {
        let (mut pool, path) = make_pool("comp_insert_after_del.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        let key = Value::Integer(42);
        let row1 = make_row(vec![Value::Text("original".into())]);
        let row2 = make_row(vec![Value::Text("reinserted".into())]);

        tree.insert(&mut pool, key.clone(), row1).unwrap();
        assert!(tree.delete(&mut pool, &key).unwrap());
        assert_eq!(tree.get(&mut pool, &key).unwrap(), None);

        tree.insert(&mut pool, key.clone(), row2.clone()).unwrap();
        let result = tree.get(&mut pool, &key).unwrap();
        assert_eq!(result, Some(row2));

        cleanup(&path);
    }

    #[test]
    fn test_delete_all_entries() {
        let (mut pool, path) = make_pool("comp_delete_all.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        for i in 0..20 {
            tree.insert(&mut pool, Value::Integer(i), make_row(vec![Value::Integer(i)]))
                .unwrap();
        }

        for i in 0..20 {
            assert!(tree.delete(&mut pool, &Value::Integer(i)).unwrap());
        }

        assert!(tree.scan(&mut pool).unwrap().is_empty());
        for i in 0..20 {
            assert_eq!(tree.get(&mut pool, &Value::Integer(i)).unwrap(), None);
        }

        // Deleting again returns false
        for i in 0..20 {
            assert!(!tree.delete(&mut pool, &Value::Integer(i)).unwrap());
        }

        cleanup(&path);
    }

    #[test]
    fn test_delete_from_multiple_leaves() {
        let (mut pool, path) = make_pool("comp_delete_multi_leaf.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        let n = 200i64;
        for i in 0..n {
            tree.insert(&mut pool, Value::Integer(i), make_row(vec![Value::Integer(i)]))
                .unwrap();
        }

        // Delete every third entry across the range
        let mut deleted_count = 0;
        for i in (0..n).step_by(3) {
            assert!(tree.delete(&mut pool, &Value::Integer(i)).unwrap());
            deleted_count += 1;
        }

        let results = tree.scan(&mut pool).unwrap();
        assert_eq!(results.len(), (n as usize) - deleted_count);

        for i in 0..n {
            let result = tree.get(&mut pool, &Value::Integer(i)).unwrap();
            if i % 3 == 0 {
                assert_eq!(result, None, "key {} should have been deleted", i);
            } else {
                assert!(result.is_some(), "key {} should still exist", i);
            }
        }

        cleanup(&path);
    }

    // ── Duplicate key / update tests ──

    #[test]
    fn test_duplicate_key_updates_value() {
        let (mut pool, path) = make_pool("comp_dup_update.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        let key = Value::Integer(1);
        let row1 = make_row(vec![Value::Text("first".into())]);
        let row2 = make_row(vec![Value::Text("second".into())]);
        let row3 = make_row(vec![Value::Text("third".into())]);

        tree.insert(&mut pool, key.clone(), row1).unwrap();
        tree.insert(&mut pool, key.clone(), row2).unwrap();
        tree.insert(&mut pool, key.clone(), row3.clone()).unwrap();

        // Should have the latest value
        let result = tree.get(&mut pool, &key).unwrap();
        assert_eq!(result, Some(row3));

        // Only one entry in scan
        let results = tree.scan(&mut pool).unwrap();
        assert_eq!(results.len(), 1);

        cleanup(&path);
    }

    #[test]
    fn test_update_after_splits() {
        let (mut pool, path) = make_pool("comp_update_splits.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        let n = 200i64;
        for i in 0..n {
            tree.insert(
                &mut pool,
                Value::Integer(i),
                make_row(vec![Value::Integer(i), Value::Text("original".into())]),
            )
            .unwrap();
        }

        // Update entries spread across different leaves
        for i in (0..n).step_by(17) {
            tree.insert(
                &mut pool,
                Value::Integer(i),
                make_row(vec![Value::Integer(i), Value::Text("updated".into())]),
            )
            .unwrap();
        }

        for i in 0..n {
            let result = tree.get(&mut pool, &Value::Integer(i)).unwrap();
            assert!(result.is_some(), "key {} not found", i);
            let row = result.unwrap();
            if i % 17 == 0 {
                assert_eq!(
                    row.values[1],
                    Value::Text("updated".into()),
                    "key {} should be updated",
                    i
                );
            } else {
                assert_eq!(
                    row.values[1],
                    Value::Text("original".into()),
                    "key {} should be original",
                    i
                );
            }
        }

        // Total count unchanged
        let results = tree.scan(&mut pool).unwrap();
        assert_eq!(results.len(), n as usize);

        cleanup(&path);
    }

    // ── Persistence tests ──

    #[test]
    fn test_persistence_after_splits() {
        let path = temp_db_path("comp_persist_splits.db");
        cleanup(&path);

        let root_page_id;
        let n = 200i64;

        {
            let dm = DiskManager::open(&path).unwrap();
            let mut pool = BufferPool::new(dm, 100);
            let mut tree = BTree::create(&mut pool).unwrap();

            for i in 0..n {
                tree.insert(
                    &mut pool,
                    Value::Integer(i),
                    make_row(vec![Value::Integer(i), Value::Text(format!("v{}", i))]),
                )
                .unwrap();
            }

            root_page_id = tree.root_page_id();
            pool.flush_all().unwrap();
        }

        {
            let dm = DiskManager::open(&path).unwrap();
            let mut pool = BufferPool::new(dm, 100);
            let tree = BTree::open(root_page_id);

            for i in 0..n {
                let result = tree.get(&mut pool, &Value::Integer(i)).unwrap();
                assert_eq!(
                    result,
                    Some(make_row(vec![
                        Value::Integer(i),
                        Value::Text(format!("v{}", i))
                    ])),
                    "key {} not found after reopen",
                    i
                );
            }

            let results = tree.scan(&mut pool).unwrap();
            assert_eq!(results.len(), n as usize);
        }

        cleanup(&path);
    }

    // ── Large text values ──

    #[test]
    fn test_large_text_values() {
        let (mut pool, path) = make_pool("comp_large_text.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        for i in 0..50 {
            let key = Value::Integer(i);
            let long_text = format!("entry_{}_data_{}", i, "x".repeat(50));
            let row = make_row(vec![Value::Integer(i), Value::Text(long_text)]);
            tree.insert(&mut pool, key, row).unwrap();
        }

        for i in 0..50 {
            let result = tree.get(&mut pool, &Value::Integer(i)).unwrap();
            assert!(result.is_some(), "key {} not found", i);
            let expected_text = format!("entry_{}_data_{}", i, "x".repeat(50));
            assert_eq!(result.unwrap().values[1], Value::Text(expected_text));
        }

        let results = tree.scan(&mut pool).unwrap();
        assert_eq!(results.len(), 50);

        cleanup(&path);
    }

    // ── Mixed operations ──

    #[test]
    fn test_interleaved_insert_delete_get() {
        let (mut pool, path) = make_pool("comp_interleaved.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        // Insert 0..50
        for i in 0..50i64 {
            tree.insert(&mut pool, Value::Integer(i), make_row(vec![Value::Integer(i)]))
                .unwrap();
        }

        // Delete even numbers
        for i in (0..50i64).step_by(2) {
            assert!(tree.delete(&mut pool, &Value::Integer(i)).unwrap());
        }

        // Insert 50..100
        for i in 50..100i64 {
            tree.insert(&mut pool, Value::Integer(i), make_row(vec![Value::Integer(i)]))
                .unwrap();
        }

        // Verify: odds from 0..50, all from 50..100
        let results = tree.scan(&mut pool).unwrap();
        let keys: Vec<i64> = results.iter().map(|(k, _)| k.as_integer().unwrap()).collect();

        let mut expected: Vec<i64> = (0..50).filter(|i| i % 2 != 0).collect();
        expected.extend(50..100);
        assert_eq!(keys, expected);

        cleanup(&path);
    }

    #[test]
    fn test_scan_returns_correct_row_data() {
        let (mut pool, path) = make_pool("comp_scan_data.db");
        let mut tree = BTree::create(&mut pool).unwrap();

        // Insert rows with multiple column values
        for i in 0..10i64 {
            let row = make_row(vec![
                Value::Integer(i),
                Value::Text(format!("name_{}", i)),
                Value::Float(i as f64 * 1.5),
                Value::Boolean(i % 2 == 0),
            ]);
            tree.insert(&mut pool, Value::Integer(i), row).unwrap();
        }

        let results = tree.scan(&mut pool).unwrap();
        assert_eq!(results.len(), 10);

        // Verify complete row data integrity
        for (idx, (key, row)) in results.iter().enumerate() {
            let i = idx as i64;
            assert_eq!(key, &Value::Integer(i));
            assert_eq!(row.values.len(), 4);
            assert_eq!(row.values[0], Value::Integer(i));
            assert_eq!(row.values[1], Value::Text(format!("name_{}", i)));
            assert_eq!(row.values[2], Value::Float(i as f64 * 1.5));
            assert_eq!(row.values[3], Value::Boolean(i % 2 == 0));
        }

        cleanup(&path);
    }
}
