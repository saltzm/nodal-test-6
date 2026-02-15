//! Fixed-size page abstraction for the storage layer.
//!
//! Each page is 4096 bytes with a small header containing the page ID and page type,
//! followed by a byte payload.

use crate::error::{Error, Result};

/// Size of a single page in bytes.
pub const PAGE_SIZE: usize = 4096;

/// Size of the page header in bytes: 4 (page_id) + 1 (page_type) = 5.
pub const PAGE_HEADER_SIZE: usize = 5;

/// Maximum payload that can be stored in a page.
pub const PAGE_PAYLOAD_SIZE: usize = PAGE_SIZE - PAGE_HEADER_SIZE;

/// Unique identifier for a page (offset = page_id * PAGE_SIZE in the file).
pub type PageId = u32;

/// Sentinel value meaning "no page".
pub const INVALID_PAGE_ID: PageId = u32::MAX;

/// The type of a page, stored in the header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PageType {
    Meta = 0,
    Internal = 1,
    Leaf = 2,
    Free = 3,
}

impl PageType {
    /// Convert from a raw byte.
    pub fn from_byte(b: u8) -> Result<Self> {
        match b {
            0 => Ok(PageType::Meta),
            1 => Ok(PageType::Internal),
            2 => Ok(PageType::Leaf),
            3 => Ok(PageType::Free),
            _ => Err(Error::Storage(format!("Invalid page type byte: {}", b))),
        }
    }

    /// Convert to a raw byte.
    pub fn to_byte(self) -> u8 {
        self as u8
    }
}

/// A fixed-size page that can be serialized to/from a 4096-byte buffer.
#[derive(Clone)]
pub struct Page {
    /// The page identifier.
    pub id: PageId,
    /// The type of this page.
    pub page_type: PageType,
    /// The payload data (up to PAGE_PAYLOAD_SIZE bytes).
    pub data: Vec<u8>,
}

impl Page {
    /// Create a new page with the given id and type, initialized with zeroed payload.
    pub fn new(id: PageId, page_type: PageType) -> Self {
        Self {
            id,
            page_type,
            data: vec![0u8; PAGE_PAYLOAD_SIZE],
        }
    }

    /// Serialize this page into a fixed-size byte array.
    pub fn serialize(&self) -> [u8; PAGE_SIZE] {
        let mut buf = [0u8; PAGE_SIZE];
        // Write page id (4 bytes, little-endian)
        buf[0..4].copy_from_slice(&self.id.to_le_bytes());
        // Write page type (1 byte)
        buf[4] = self.page_type.to_byte();
        // Write payload
        let copy_len = self.data.len().min(PAGE_PAYLOAD_SIZE);
        buf[PAGE_HEADER_SIZE..PAGE_HEADER_SIZE + copy_len].copy_from_slice(&self.data[..copy_len]);
        buf
    }

    /// Deserialize a page from a fixed-size byte array.
    pub fn deserialize(buf: &[u8; PAGE_SIZE]) -> Result<Self> {
        let id = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let page_type = PageType::from_byte(buf[4])?;
        let data = buf[PAGE_HEADER_SIZE..].to_vec();
        Ok(Self {
            id,
            page_type,
            data,
        })
    }
}

impl std::fmt::Debug for Page {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Page")
            .field("id", &self.id)
            .field("page_type", &self.page_type)
            .field("data_len", &self.data.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_type_roundtrip() {
        for pt in [PageType::Meta, PageType::Internal, PageType::Leaf, PageType::Free] {
            let byte = pt.to_byte();
            let recovered = PageType::from_byte(byte).unwrap();
            assert_eq!(recovered, pt);
        }
    }

    #[test]
    fn test_page_type_invalid() {
        assert!(PageType::from_byte(255).is_err());
    }

    #[test]
    fn test_page_new() {
        let page = Page::new(42, PageType::Leaf);
        assert_eq!(page.id, 42);
        assert_eq!(page.page_type, PageType::Leaf);
        assert_eq!(page.data.len(), PAGE_PAYLOAD_SIZE);
        assert!(page.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_page_serialize_deserialize_roundtrip() {
        let mut page = Page::new(7, PageType::Internal);
        // Write some data into the payload.
        page.data[0] = 0xAA;
        page.data[1] = 0xBB;
        page.data[PAGE_PAYLOAD_SIZE - 1] = 0xFF;

        let buf = page.serialize();
        assert_eq!(buf.len(), PAGE_SIZE);

        let recovered = Page::deserialize(&buf).unwrap();
        assert_eq!(recovered.id, 7);
        assert_eq!(recovered.page_type, PageType::Internal);
        assert_eq!(recovered.data[0], 0xAA);
        assert_eq!(recovered.data[1], 0xBB);
        assert_eq!(recovered.data[PAGE_PAYLOAD_SIZE - 1], 0xFF);
    }

    #[test]
    fn test_page_serialize_all_types() {
        for pt in [PageType::Meta, PageType::Internal, PageType::Leaf, PageType::Free] {
            let page = Page::new(0, pt);
            let buf = page.serialize();
            let recovered = Page::deserialize(&buf).unwrap();
            assert_eq!(recovered.page_type, pt);
        }
    }

    #[test]
    fn test_page_data_isolation() {
        let mut page = Page::new(1, PageType::Leaf);
        page.data[100] = 42;
        let cloned = page.clone();
        assert_eq!(cloned.data[100], 42);
        // Ensure the clone is independent.
        assert_eq!(page.data[100], 42);
    }
}
