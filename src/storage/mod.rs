//! Storage layer — page management, disk I/O, and buffer pool.

pub mod btree;
#[cfg(test)]
mod btree_tests;
pub mod buffer;
pub mod disk;
pub mod page;
pub mod serde;
pub mod table;
pub mod wal;

pub use btree::BTree;
pub use buffer::BufferPool;
pub use disk::DiskManager;
pub use page::{Page, PageId, PageType, PAGE_SIZE, PAGE_HEADER_SIZE, PAGE_PAYLOAD_SIZE, INVALID_PAGE_ID};
pub use serde::{serialize_value, deserialize_value, serialize_row, deserialize_row};
pub use table::TableStorage;
pub use wal::{Wal, WalReader, WalRecord, WalWriter, TxnId};