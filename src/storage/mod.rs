//! Storage layer — page management, disk I/O, and buffer pool.

pub mod buffer;
pub mod disk;
pub mod page;

pub use buffer::BufferPool;
pub use disk::DiskManager;
pub use page::{Page, PageId, PageType, PAGE_SIZE, PAGE_HEADER_SIZE, PAGE_PAYLOAD_SIZE, INVALID_PAGE_ID};
