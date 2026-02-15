//! Write-Ahead Log (WAL) for crash recovery.
//!
//! The WAL records all modifications before they are applied to data pages,
//! ensuring durability and enabling crash recovery.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::storage::page::PAGE_SIZE;

/// Transaction identifier.
pub type TxnId = u64;

// ─── CRC32 (IEEE polynomial) ───

/// CRC32 lookup table for the IEEE polynomial (0xEDB88320, reflected).
const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0u32;
    while i < 256 {
        let mut crc = i;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i as usize] = crc;
        i += 1;
    }
    table
};

/// Compute CRC32 checksum of the given data.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        let idx = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32_TABLE[idx];
    }
    crc ^ 0xFFFFFFFF
}

// ─── WAL record types ───

/// Record type tags stored in the WAL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WalRecordType {
    Begin = 1,
    Commit = 2,
    Abort = 3,
    PageWrite = 4,
    Checkpoint = 5,
}

impl WalRecordType {
    fn from_byte(b: u8) -> Result<Self> {
        match b {
            1 => Ok(WalRecordType::Begin),
            2 => Ok(WalRecordType::Commit),
            3 => Ok(WalRecordType::Abort),
            4 => Ok(WalRecordType::PageWrite),
            5 => Ok(WalRecordType::Checkpoint),
            _ => Err(Error::Storage(format!("Invalid WAL record type: {}", b))),
        }
    }
}

/// A single WAL record.
#[derive(Debug, Clone, PartialEq)]
pub enum WalRecord {
    /// Begin a new transaction.
    Begin(TxnId),
    /// Commit a transaction (durable).
    Commit(TxnId),
    /// Abort a transaction.
    Abort(TxnId),
    /// A page write with before and after images for undo/redo.
    Write {
        txn_id: TxnId,
        page_id: u32,
        before_image: Box<[u8; PAGE_SIZE]>,
        after_image: Box<[u8; PAGE_SIZE]>,
    },
    /// Checkpoint marker.
    Checkpoint,
}

// ─── Serialization ───
//
// On-disk format for each record:
//   [record_length: u32 LE]  — length of (type_tag + payload), excluding this header and CRC
//   [type_tag: u8]
//   [payload: variable]
//   [crc32: u32 LE]          — CRC over (type_tag + payload)
//
// Total bytes on disk = 4 (length) + record_length + 4 (crc)

impl WalRecord {
    /// Serialize the record body (type tag + payload) into a byte vector.
    fn serialize_body(&self) -> Vec<u8> {
        match self {
            WalRecord::Begin(txn_id) => {
                let mut buf = Vec::with_capacity(1 + 8);
                buf.push(WalRecordType::Begin as u8);
                buf.extend_from_slice(&txn_id.to_le_bytes());
                buf
            }
            WalRecord::Commit(txn_id) => {
                let mut buf = Vec::with_capacity(1 + 8);
                buf.push(WalRecordType::Commit as u8);
                buf.extend_from_slice(&txn_id.to_le_bytes());
                buf
            }
            WalRecord::Abort(txn_id) => {
                let mut buf = Vec::with_capacity(1 + 8);
                buf.push(WalRecordType::Abort as u8);
                buf.extend_from_slice(&txn_id.to_le_bytes());
                buf
            }
            WalRecord::Write {
                txn_id,
                page_id,
                before_image,
                after_image,
            } => {
                let mut buf = Vec::with_capacity(1 + 8 + 4 + PAGE_SIZE * 2);
                buf.push(WalRecordType::PageWrite as u8);
                buf.extend_from_slice(&txn_id.to_le_bytes());
                buf.extend_from_slice(&page_id.to_le_bytes());
                buf.extend_from_slice(before_image.as_ref());
                buf.extend_from_slice(after_image.as_ref());
                buf
            }
            WalRecord::Checkpoint => {
                vec![WalRecordType::Checkpoint as u8]
            }
        }
    }

    /// Serialize the full on-disk representation: length + body + CRC.
    pub fn serialize(&self) -> Vec<u8> {
        let body = self.serialize_body();
        let length = body.len() as u32;
        let checksum = crc32(&body);

        let mut out = Vec::with_capacity(4 + body.len() + 4);
        out.extend_from_slice(&length.to_le_bytes());
        out.extend_from_slice(&body);
        out.extend_from_slice(&checksum.to_le_bytes());
        out
    }

    /// Deserialize a record from a byte slice that includes the full on-disk
    /// representation (length + body + CRC). Returns the record and the total
    /// number of bytes consumed.
    pub fn deserialize(data: &[u8]) -> Result<(Self, usize)> {
        if data.len() < 4 {
            return Err(Error::Storage("WAL record too short for length header".into()));
        }
        let length = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

        let total = 4 + length + 4; // length field + body + CRC
        if data.len() < total {
            return Err(Error::Storage(format!(
                "WAL record truncated: expected {} bytes, have {}",
                total,
                data.len()
            )));
        }

        let body = &data[4..4 + length];
        let stored_crc = u32::from_le_bytes([
            data[4 + length],
            data[4 + length + 1],
            data[4 + length + 2],
            data[4 + length + 3],
        ]);
        let computed_crc = crc32(body);
        if stored_crc != computed_crc {
            return Err(Error::Storage(format!(
                "WAL CRC mismatch: stored={:#010x}, computed={:#010x}",
                stored_crc, computed_crc
            )));
        }

        if body.is_empty() {
            return Err(Error::Storage("WAL record body is empty".into()));
        }

        let record_type = WalRecordType::from_byte(body[0])?;
        let payload = &body[1..];

        let record = match record_type {
            WalRecordType::Begin => {
                if payload.len() < 8 {
                    return Err(Error::Storage("Begin record too short".into()));
                }
                let txn_id = u64::from_le_bytes(payload[..8].try_into().unwrap());
                WalRecord::Begin(txn_id)
            }
            WalRecordType::Commit => {
                if payload.len() < 8 {
                    return Err(Error::Storage("Commit record too short".into()));
                }
                let txn_id = u64::from_le_bytes(payload[..8].try_into().unwrap());
                WalRecord::Commit(txn_id)
            }
            WalRecordType::Abort => {
                if payload.len() < 8 {
                    return Err(Error::Storage("Abort record too short".into()));
                }
                let txn_id = u64::from_le_bytes(payload[..8].try_into().unwrap());
                WalRecord::Abort(txn_id)
            }
            WalRecordType::PageWrite => {
                let expected = 8 + 4 + PAGE_SIZE * 2;
                if payload.len() < expected {
                    return Err(Error::Storage(format!(
                        "Write record too short: need {}, have {}",
                        expected,
                        payload.len()
                    )));
                }
                let txn_id = u64::from_le_bytes(payload[..8].try_into().unwrap());
                let page_id = u32::from_le_bytes(payload[8..12].try_into().unwrap());
                let mut before_image = Box::new([0u8; PAGE_SIZE]);
                before_image.copy_from_slice(&payload[12..12 + PAGE_SIZE]);
                let mut after_image = Box::new([0u8; PAGE_SIZE]);
                after_image.copy_from_slice(&payload[12 + PAGE_SIZE..12 + PAGE_SIZE * 2]);
                WalRecord::Write {
                    txn_id,
                    page_id,
                    before_image,
                    after_image,
                }
            }
            WalRecordType::Checkpoint => WalRecord::Checkpoint,
        };

        Ok((record, total))
    }
}

// ─── WAL Writer ───

/// Appends WAL records to a file sequentially.
pub struct WalWriter {
    file: File,
}

impl WalWriter {
    /// Open (or create) a WAL file for appending.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref())?;
        Ok(Self { file })
    }

    /// Append a record to the WAL file. If the record is a Commit, the file is
    /// fsynced to ensure durability.
    pub fn append(&mut self, record: &WalRecord) -> Result<()> {
        let data = record.serialize();
        self.file.write_all(&data)?;

        // Ensure commit records are durable.
        if matches!(record, WalRecord::Commit(_)) {
            self.file.sync_all()?;
        }

        Ok(())
    }

    /// Force an fsync of the WAL file.
    pub fn sync(&mut self) -> Result<()> {
        self.file.sync_all()?;
        Ok(())
    }
}

// ─── WAL Reader / Iterator ───

/// Reads WAL records sequentially from a file.
pub struct WalReader {
    data: Vec<u8>,
    offset: usize,
}

impl WalReader {
    /// Open a WAL file for reading. Reads the entire file into memory.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut file = OpenOptions::new().read(true).open(path.as_ref())?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Ok(Self { data, offset: 0 })
    }

    /// Create a reader from an in-memory byte buffer (useful for testing).
    pub fn from_bytes(data: Vec<u8>) -> Self {
        Self { data, offset: 0 }
    }

    /// Read the next record. Returns None at EOF, or an error on corruption.
    pub fn next_record(&mut self) -> Result<Option<WalRecord>> {
        if self.offset >= self.data.len() {
            return Ok(None);
        }
        // If there's not enough data for even a minimal record header, treat as EOF.
        if self.data.len() - self.offset < 4 {
            return Ok(None);
        }

        match WalRecord::deserialize(&self.data[self.offset..]) {
            Ok((record, consumed)) => {
                self.offset += consumed;
                Ok(Some(record))
            }
            Err(e) => Err(e),
        }
    }

    /// Read all remaining records into a vector.
    pub fn read_all(&mut self) -> Result<Vec<WalRecord>> {
        let mut records = Vec::new();
        while let Some(record) = self.next_record()? {
            records.push(record);
        }
        Ok(records)
    }
}

// ─── High-level Wal struct ───

/// High-level WAL interface that wraps a writer and provides convenient logging methods.
pub struct Wal {
    writer: WalWriter,
    path: PathBuf,
}

impl Wal {
    /// Open (or create) a WAL at the given path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let writer = WalWriter::open(&path)?;
        Ok(Self { writer, path })
    }

    /// Log a BEGIN record.
    pub fn log_begin(&mut self, txn_id: TxnId) -> Result<()> {
        self.writer.append(&WalRecord::Begin(txn_id))
    }

    /// Log a COMMIT record (fsynced for durability).
    pub fn log_commit(&mut self, txn_id: TxnId) -> Result<()> {
        self.writer.append(&WalRecord::Commit(txn_id))
    }

    /// Log an ABORT record.
    pub fn log_abort(&mut self, txn_id: TxnId) -> Result<()> {
        self.writer.append(&WalRecord::Abort(txn_id))
    }

    /// Log a page WRITE record with before and after images.
    pub fn log_write(
        &mut self,
        txn_id: TxnId,
        page_id: u32,
        before_image: [u8; PAGE_SIZE],
        after_image: [u8; PAGE_SIZE],
    ) -> Result<()> {
        self.writer.append(&WalRecord::Write {
            txn_id,
            page_id,
            before_image: Box::new(before_image),
            after_image: Box::new(after_image),
        })
    }

    /// Log a CHECKPOINT record.
    pub fn log_checkpoint(&mut self) -> Result<()> {
        self.writer.append(&WalRecord::Checkpoint)?;
        self.writer.sync()
    }

    /// Create a reader for this WAL file.
    pub fn reader(&self) -> Result<WalReader> {
        WalReader::open(&self.path)
    }

    /// Return the path to the WAL file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Force sync the WAL file.
    pub fn sync(&mut self) -> Result<()> {
        self.writer.sync()
    }

    /// Truncate the WAL file (discard all records).
    ///
    /// Typically called after a successful recovery or checkpoint to reclaim
    /// space. Re-opens the writer on the now-empty file.
    pub fn truncate(&mut self) -> Result<()> {
        // Truncate the file to zero length.
        let file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&self.path)?;
        drop(file);
        // Re-open the writer in append mode.
        self.writer = WalWriter::open(&self.path)?;
        Ok(())
    }
}


// ─── Recovery ───

use std::collections::HashSet;

/// Recover a database file using the WAL.
///
/// The algorithm uses a two-pass approach:
///
/// **Phase 1 (Analysis):** Read all WAL records to determine which
/// transactions committed. Also collect all write records in WAL order.
///
/// **Phase 2 (Redo/Undo):** Replay write records in WAL order.
/// - For committed transactions: apply after-images (redo).
/// - For uncommitted/aborted transactions: apply before-images (undo)
///   in reverse order to restore the original page contents.
///
/// Returns `(redo_count, undo_count)` — the number of committed and
/// uncommitted transactions that had write records.
pub fn recover(data_path: impl AsRef<Path>, wal_path: impl AsRef<Path>) -> Result<(usize, usize)> {
    let wal_path = wal_path.as_ref();
    if !wal_path.exists() {
        return Ok((0, 0));
    }

    // Phase 1: Analysis — read all WAL records.
    let mut reader = WalReader::open(wal_path)?;
    let mut committed: HashSet<TxnId> = HashSet::new();
    let mut txns_with_writes: HashSet<TxnId> = HashSet::new();
    // All write records in WAL order.
    let mut all_writes: Vec<(TxnId, u32, Box<[u8; PAGE_SIZE]>, Box<[u8; PAGE_SIZE]>)> = Vec::new();

    loop {
        match reader.next_record() {
            Ok(Some(record)) => match record {
                WalRecord::Begin(_) => {}
                WalRecord::Commit(txn_id) => {
                    committed.insert(txn_id);
                }
                WalRecord::Abort(_) => {
                    // Explicitly aborted — treated same as uncommitted.
                }
                WalRecord::Write {
                    txn_id,
                    page_id,
                    before_image,
                    after_image,
                } => {
                    txns_with_writes.insert(txn_id);
                    all_writes.push((txn_id, page_id, before_image, after_image));
                }
                WalRecord::Checkpoint => {}
            },
            Ok(None) => break,
            Err(_) => {
                // Truncated/corrupted tail — treat as end-of-log.
                break;
            }
        }
    }

    // Phase 2: Apply changes to the data file.
    let mut data_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(data_path.as_ref())?;

    let mut file_len = data_file.metadata()?.len();

    // Redo: replay committed transaction writes in WAL order.
    for (txn_id, page_id, _before, after) in &all_writes {
        if committed.contains(txn_id) {
            let offset = *page_id as u64 * PAGE_SIZE as u64;
            if offset + PAGE_SIZE as u64 > file_len {
                data_file.set_len(offset + PAGE_SIZE as u64)?;
                file_len = offset + PAGE_SIZE as u64;
            }
            data_file.seek(SeekFrom::Start(offset))?;
            data_file.write_all(after.as_ref())?;
        }
    }

    // Undo: replay uncommitted transaction writes in REVERSE order.
    for (txn_id, page_id, before, _after) in all_writes.iter().rev() {
        if !committed.contains(txn_id) {
            let offset = *page_id as u64 * PAGE_SIZE as u64;
            if offset + PAGE_SIZE as u64 <= file_len {
                data_file.seek(SeekFrom::Start(offset))?;
                data_file.write_all(before.as_ref())?;
            }
        }
    }

    data_file.sync_all()?;

    // Count committed/uncommitted transactions that had writes.
    let redo_count = txns_with_writes.iter().filter(|t| committed.contains(t)).count();
    let undo_count = txns_with_writes.iter().filter(|t| !committed.contains(t)).count();

    Ok((redo_count, undo_count))
}
// ─── Tests ───

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_wal_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("toydb_test_wal");
        fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn cleanup(path: &Path) {
        let _ = fs::remove_file(path);
    }

    // ── CRC32 tests ──

    #[test]
    fn test_crc32_empty() {
        assert_eq!(crc32(b""), 0x00000000);
    }

    #[test]
    fn test_crc32_known_values() {
        // CRC32 of "123456789" is 0xCBF43926 (well-known test vector).
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn test_crc32_different_inputs() {
        let a = crc32(b"hello");
        let b = crc32(b"world");
        assert_ne!(a, b);
    }

    // ── Serialization roundtrip tests ──

    #[test]
    fn test_serialize_roundtrip_begin() {
        let record = WalRecord::Begin(42);
        let data = record.serialize();
        let (decoded, consumed) = WalRecord::deserialize(&data).unwrap();
        assert_eq!(decoded, record);
        assert_eq!(consumed, data.len());
    }

    #[test]
    fn test_serialize_roundtrip_commit() {
        let record = WalRecord::Commit(100);
        let data = record.serialize();
        let (decoded, consumed) = WalRecord::deserialize(&data).unwrap();
        assert_eq!(decoded, record);
        assert_eq!(consumed, data.len());
    }

    #[test]
    fn test_serialize_roundtrip_abort() {
        let record = WalRecord::Abort(7);
        let data = record.serialize();
        let (decoded, consumed) = WalRecord::deserialize(&data).unwrap();
        assert_eq!(decoded, record);
        assert_eq!(consumed, data.len());
    }

    #[test]
    fn test_serialize_roundtrip_checkpoint() {
        let record = WalRecord::Checkpoint;
        let data = record.serialize();
        let (decoded, consumed) = WalRecord::deserialize(&data).unwrap();
        assert_eq!(decoded, record);
        assert_eq!(consumed, data.len());
    }

    #[test]
    fn test_serialize_roundtrip_write() {
        let mut before = Box::new([0u8; PAGE_SIZE]);
        let mut after = Box::new([0u8; PAGE_SIZE]);
        before[0] = 0xAA;
        before[PAGE_SIZE - 1] = 0xBB;
        after[0] = 0xCC;
        after[PAGE_SIZE - 1] = 0xDD;

        let record = WalRecord::Write {
            txn_id: 55,
            page_id: 10,
            before_image: before,
            after_image: after,
        };
        let data = record.serialize();
        let (decoded, consumed) = WalRecord::deserialize(&data).unwrap();
        assert_eq!(decoded, record);
        assert_eq!(consumed, data.len());
    }

    #[test]
    fn test_serialize_roundtrip_write_all_zeros() {
        let record = WalRecord::Write {
            txn_id: 0,
            page_id: 0,
            before_image: Box::new([0u8; PAGE_SIZE]),
            after_image: Box::new([0u8; PAGE_SIZE]),
        };
        let data = record.serialize();
        let (decoded, _) = WalRecord::deserialize(&data).unwrap();
        assert_eq!(decoded, record);
    }

    // ── CRC corruption detection ──

    #[test]
    fn test_crc_detects_corruption_in_body() {
        let record = WalRecord::Begin(42);
        let mut data = record.serialize();
        // Corrupt a byte in the body (type tag or payload).
        data[4] ^= 0xFF;
        assert!(WalRecord::deserialize(&data).is_err());
    }

    #[test]
    fn test_crc_detects_corruption_in_crc_field() {
        let record = WalRecord::Commit(99);
        let mut data = record.serialize();
        // Corrupt the CRC bytes at the end.
        let last = data.len() - 1;
        data[last] ^= 0x01;
        assert!(WalRecord::deserialize(&data).is_err());
    }

    #[test]
    fn test_crc_detects_corruption_in_write_record() {
        let record = WalRecord::Write {
            txn_id: 1,
            page_id: 5,
            before_image: Box::new([0xAA; PAGE_SIZE]),
            after_image: Box::new([0xBB; PAGE_SIZE]),
        };
        let mut data = record.serialize();
        // Flip a bit in the middle of the after_image area.
        let mid = 4 + 1 + 8 + 4 + PAGE_SIZE + PAGE_SIZE / 2;
        if mid < data.len() - 4 {
            data[mid] ^= 0x01;
        }
        assert!(WalRecord::deserialize(&data).is_err());
    }

    // ── Truncated data ──

    #[test]
    fn test_deserialize_too_short() {
        assert!(WalRecord::deserialize(&[0, 0]).is_err());
    }

    #[test]
    fn test_deserialize_truncated_body() {
        let record = WalRecord::Begin(42);
        let data = record.serialize();
        // Chop off last few bytes.
        assert!(WalRecord::deserialize(&data[..data.len() - 2]).is_err());
    }

    // ── WalWriter + WalReader integration ──

    #[test]
    fn test_writer_reader_single_record() {
        let path = temp_wal_path("wal_single.wal");
        cleanup(&path);

        {
            let mut writer = WalWriter::open(&path).unwrap();
            writer.append(&WalRecord::Begin(1)).unwrap();
        }

        {
            let mut reader = WalReader::open(&path).unwrap();
            let records = reader.read_all().unwrap();
            assert_eq!(records.len(), 1);
            assert_eq!(records[0], WalRecord::Begin(1));
        }

        cleanup(&path);
    }

    #[test]
    fn test_writer_reader_multiple_records() {
        let path = temp_wal_path("wal_multi.wal");
        cleanup(&path);

        let expected = vec![
            WalRecord::Begin(1),
            WalRecord::Write {
                txn_id: 1,
                page_id: 3,
                before_image: Box::new([0u8; PAGE_SIZE]),
                after_image: Box::new([0xFF; PAGE_SIZE]),
            },
            WalRecord::Commit(1),
            WalRecord::Begin(2),
            WalRecord::Abort(2),
            WalRecord::Checkpoint,
        ];

        {
            let mut writer = WalWriter::open(&path).unwrap();
            for record in &expected {
                writer.append(record).unwrap();
            }
            writer.sync().unwrap();
        }

        {
            let mut reader = WalReader::open(&path).unwrap();
            let records = reader.read_all().unwrap();
            assert_eq!(records.len(), expected.len());
            for (got, want) in records.iter().zip(expected.iter()) {
                assert_eq!(got, want);
            }
        }

        cleanup(&path);
    }

    #[test]
    fn test_writer_reader_empty_file() {
        let path = temp_wal_path("wal_empty.wal");
        cleanup(&path);

        // Create an empty file.
        {
            let _writer = WalWriter::open(&path).unwrap();
        }

        {
            let mut reader = WalReader::open(&path).unwrap();
            let records = reader.read_all().unwrap();
            assert!(records.is_empty());
        }

        cleanup(&path);
    }

    #[test]
    fn test_reader_incremental() {
        let path = temp_wal_path("wal_incremental.wal");
        cleanup(&path);

        {
            let mut writer = WalWriter::open(&path).unwrap();
            writer.append(&WalRecord::Begin(10)).unwrap();
            writer.append(&WalRecord::Commit(10)).unwrap();
            writer.append(&WalRecord::Begin(20)).unwrap();
        }

        {
            let mut reader = WalReader::open(&path).unwrap();
            assert_eq!(reader.next_record().unwrap(), Some(WalRecord::Begin(10)));
            assert_eq!(reader.next_record().unwrap(), Some(WalRecord::Commit(10)));
            assert_eq!(reader.next_record().unwrap(), Some(WalRecord::Begin(20)));
            assert_eq!(reader.next_record().unwrap(), None);
        }

        cleanup(&path);
    }

    // ── High-level Wal struct ──

    #[test]
    fn test_wal_high_level_api() {
        let path = temp_wal_path("wal_highlevel.wal");
        cleanup(&path);

        {
            let mut wal = Wal::open(&path).unwrap();
            wal.log_begin(1).unwrap();
            wal.log_write(1, 5, [0xAA; PAGE_SIZE], [0xBB; PAGE_SIZE]).unwrap();
            wal.log_commit(1).unwrap();
            wal.log_begin(2).unwrap();
            wal.log_abort(2).unwrap();
            wal.log_checkpoint().unwrap();
        }

        {
            let wal = Wal::open(&path).unwrap();
            let mut reader = wal.reader().unwrap();
            let records = reader.read_all().unwrap();
            assert_eq!(records.len(), 6);
            assert_eq!(records[0], WalRecord::Begin(1));
            match &records[1] {
                WalRecord::Write {
                    txn_id,
                    page_id,
                    before_image,
                    after_image,
                } => {
                    assert_eq!(*txn_id, 1);
                    assert_eq!(*page_id, 5);
                    assert!(before_image.iter().all(|&b| b == 0xAA));
                    assert!(after_image.iter().all(|&b| b == 0xBB));
                }
                other => panic!("Expected Write record, got {:?}", other),
            }
            assert_eq!(records[2], WalRecord::Commit(1));
            assert_eq!(records[3], WalRecord::Begin(2));
            assert_eq!(records[4], WalRecord::Abort(2));
            assert_eq!(records[5], WalRecord::Checkpoint);
        }

        cleanup(&path);
    }

    #[test]
    fn test_wal_append_after_reopen() {
        let path = temp_wal_path("wal_reopen.wal");
        cleanup(&path);

        // Write first batch.
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.log_begin(1).unwrap();
            wal.log_commit(1).unwrap();
        }

        // Reopen and append more.
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.log_begin(2).unwrap();
            wal.log_commit(2).unwrap();
        }

        // All 4 records should be present.
        {
            let mut reader = WalReader::open(&path).unwrap();
            let records = reader.read_all().unwrap();
            assert_eq!(records.len(), 4);
            assert_eq!(records[0], WalRecord::Begin(1));
            assert_eq!(records[1], WalRecord::Commit(1));
            assert_eq!(records[2], WalRecord::Begin(2));
            assert_eq!(records[3], WalRecord::Commit(2));
        }

        cleanup(&path);
    }

    // ── WalReader from bytes (in-memory) ──

    #[test]
    fn test_reader_from_bytes() {
        let records = vec![
            WalRecord::Begin(1),
            WalRecord::Commit(1),
        ];
        let mut data = Vec::new();
        for r in &records {
            data.extend_from_slice(&r.serialize());
        }

        let mut reader = WalReader::from_bytes(data);
        let decoded = reader.read_all().unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0], records[0]);
        assert_eq!(decoded[1], records[1]);
    }

    #[test]
    fn test_reader_from_bytes_corrupted() {
        let record = WalRecord::Begin(1);
        let mut data = record.serialize();
        // Corrupt a byte in the body.
        data[5] ^= 0xFF;

        let mut reader = WalReader::from_bytes(data);
        assert!(reader.next_record().is_err());
    }

    // ── Edge cases ──

    #[test]
    fn test_large_txn_id() {
        let record = WalRecord::Begin(u64::MAX);
        let data = record.serialize();
        let (decoded, _) = WalRecord::deserialize(&data).unwrap();
        assert_eq!(decoded, WalRecord::Begin(u64::MAX));
    }

    #[test]
    fn test_large_page_id() {
        let record = WalRecord::Write {
            txn_id: 1,
            page_id: u32::MAX,
            before_image: Box::new([0u8; PAGE_SIZE]),
            after_image: Box::new([0u8; PAGE_SIZE]),
        };
        let data = record.serialize();
        let (decoded, _) = WalRecord::deserialize(&data).unwrap();
        assert_eq!(decoded, record);
    }

    #[test]
    fn test_multiple_write_records_preserve_data() {
        let path = temp_wal_path("wal_multi_write.wal");
        cleanup(&path);

        let mut writer = WalWriter::open(&path).unwrap();
        for i in 0u8..5 {
            let mut before = [0u8; PAGE_SIZE];
            let mut after = [0u8; PAGE_SIZE];
            before[0] = i;
            after[0] = i + 100;
            writer
                .append(&WalRecord::Write {
                    txn_id: i as u64,
                    page_id: i as u32,
                    before_image: Box::new(before),
                    after_image: Box::new(after),
                })
                .unwrap();
        }
        writer.sync().unwrap();

        let mut reader = WalReader::open(&path).unwrap();
        let records = reader.read_all().unwrap();
        assert_eq!(records.len(), 5);
        for (idx, record) in records.iter().enumerate() {
            match record {
                WalRecord::Write {
                    txn_id,
                    page_id,
                    before_image,
                    after_image,
                } => {
                    assert_eq!(*txn_id, idx as u64);
                    assert_eq!(*page_id, idx as u32);
                    assert_eq!(before_image[0], idx as u8);
                    assert_eq!(after_image[0], (idx as u8) + 100);
                }
                other => panic!("Expected Write, got {:?}", other),
            }
        }

        cleanup(&path);
    }

    // ════════════════════════════════════════════════════════════
    //  Recovery tests
    // ════════════════════════════════════════════════════════════

    fn temp_db_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("toydb_test_wal_recovery");
        fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    /// Helper: read a raw page image from a data file.
    fn read_page_raw(data_path: &Path, page_id: u32) -> [u8; PAGE_SIZE] {
        use std::io::{Seek, SeekFrom};
        let mut file = std::fs::File::open(data_path).unwrap();
        let offset = page_id as u64 * PAGE_SIZE as u64;
        file.seek(SeekFrom::Start(offset)).unwrap();
        let mut buf = [0u8; PAGE_SIZE];
        file.read_exact(&mut buf).unwrap();
        buf
    }

    /// Helper: write a raw page image to a data file.
    fn write_page_raw(data_path: &Path, page_id: u32, data: &[u8; PAGE_SIZE]) {
        use std::io::{Seek, SeekFrom, Write as IoWrite};
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(data_path)
            .unwrap();
        let offset = page_id as u64 * PAGE_SIZE as u64;
        let end = offset + PAGE_SIZE as u64;
        if file.metadata().unwrap().len() < end {
            file.set_len(end).unwrap();
        }
        file.seek(SeekFrom::Start(offset)).unwrap();
        file.write_all(data).unwrap();
        file.flush().unwrap();
    }

    // ── Test 1: Uncommitted transaction changes are NOT applied ──

    #[test]
    fn test_recover_uncommitted_not_applied() {
        let db_path = temp_db_path("recover_uncommitted.db");
        let wal_path_f = temp_db_path("recover_uncommitted.wal");
        cleanup(&db_path);
        cleanup(&wal_path_f);

        let original = [0xAAu8; PAGE_SIZE];
        write_page_raw(&db_path, 0, &original);

        {
            let mut writer = WalWriter::open(&wal_path_f).unwrap();
            writer.append(&WalRecord::Begin(1)).unwrap();
            let mut after = [0xBBu8; PAGE_SIZE];
            after[0] = 0xFF;
            writer
                .append(&WalRecord::Write {
                    txn_id: 1,
                    page_id: 0,
                    before_image: Box::new(original),
                    after_image: Box::new(after),
                })
                .unwrap();
            writer.sync().unwrap();
        }

        let (redo, undo) = super::recover(&db_path, &wal_path_f).unwrap();
        assert_eq!(redo, 0);
        assert_eq!(undo, 1);

        let page = read_page_raw(&db_path, 0);
        assert_eq!(page, original);

        cleanup(&db_path);
        cleanup(&wal_path_f);
    }

    // ── Test 2: Committed transaction changes ARE applied ──

    #[test]
    fn test_recover_committed_applied() {
        let db_path = temp_db_path("recover_committed.db");
        let wal_path_f = temp_db_path("recover_committed.wal");
        cleanup(&db_path);
        cleanup(&wal_path_f);

        let original = [0x00u8; PAGE_SIZE];
        write_page_raw(&db_path, 0, &original);

        let mut after = [0x00u8; PAGE_SIZE];
        after[0] = 0xDE;
        after[1] = 0xAD;
        after[2] = 0xBE;
        after[3] = 0xEF;

        {
            let mut writer = WalWriter::open(&wal_path_f).unwrap();
            writer.append(&WalRecord::Begin(1)).unwrap();
            writer
                .append(&WalRecord::Write {
                    txn_id: 1,
                    page_id: 0,
                    before_image: Box::new(original),
                    after_image: Box::new(after),
                })
                .unwrap();
            writer.append(&WalRecord::Commit(1)).unwrap();
            writer.sync().unwrap();
        }

        let (redo, undo) = super::recover(&db_path, &wal_path_f).unwrap();
        assert_eq!(redo, 1);
        assert_eq!(undo, 0);

        let page = read_page_raw(&db_path, 0);
        assert_eq!(page[0], 0xDE);
        assert_eq!(page[1], 0xAD);
        assert_eq!(page[2], 0xBE);
        assert_eq!(page[3], 0xEF);

        cleanup(&db_path);
        cleanup(&wal_path_f);
    }

    // ── Test 3: Interleaved transactions ──

    #[test]
    fn test_recover_interleaved_transactions() {
        let db_path = temp_db_path("recover_interleaved.db");
        let wal_path_f = temp_db_path("recover_interleaved.wal");
        cleanup(&db_path);
        cleanup(&wal_path_f);

        let page0_original = [0x11u8; PAGE_SIZE];
        let page1_original = [0x22u8; PAGE_SIZE];
        write_page_raw(&db_path, 0, &page0_original);
        write_page_raw(&db_path, 1, &page1_original);

        let page0_after = [0xAAu8; PAGE_SIZE];
        let page1_after = [0xBBu8; PAGE_SIZE];

        {
            let mut writer = WalWriter::open(&wal_path_f).unwrap();
            writer.append(&WalRecord::Begin(1)).unwrap();
            writer.append(&WalRecord::Begin(2)).unwrap();
            writer.append(&WalRecord::Write {
                txn_id: 1, page_id: 0,
                before_image: Box::new(page0_original),
                after_image: Box::new(page0_after),
            }).unwrap();
            writer.append(&WalRecord::Write {
                txn_id: 2, page_id: 1,
                before_image: Box::new(page1_original),
                after_image: Box::new(page1_after),
            }).unwrap();
            writer.append(&WalRecord::Commit(1)).unwrap();
            writer.sync().unwrap();
        }

        let (redo, undo) = super::recover(&db_path, &wal_path_f).unwrap();
        assert_eq!(redo, 1);
        assert_eq!(undo, 1);

        let p0 = read_page_raw(&db_path, 0);
        assert_eq!(p0, page0_after);

        let p1 = read_page_raw(&db_path, 1);
        assert_eq!(p1, page1_original);

        cleanup(&db_path);
        cleanup(&wal_path_f);
    }

    // ── Test 4: Recovery is idempotent ──

    #[test]
    fn test_recover_idempotent() {
        let db_path = temp_db_path("recover_idempotent.db");
        let wal_path_f = temp_db_path("recover_idempotent.wal");
        cleanup(&db_path);
        cleanup(&wal_path_f);

        let page0_original = [0x00u8; PAGE_SIZE];
        let page1_original = [0x00u8; PAGE_SIZE];
        write_page_raw(&db_path, 0, &page0_original);
        write_page_raw(&db_path, 1, &page1_original);

        let mut page0_after = [0x00u8; PAGE_SIZE];
        page0_after[0] = 0xCA;
        page0_after[1] = 0xFE;

        let mut page1_after = [0x00u8; PAGE_SIZE];
        page1_after[0] = 0xBA;
        page1_after[1] = 0xBE;

        {
            let mut writer = WalWriter::open(&wal_path_f).unwrap();
            writer.append(&WalRecord::Begin(1)).unwrap();
            writer.append(&WalRecord::Write {
                txn_id: 1, page_id: 0,
                before_image: Box::new(page0_original),
                after_image: Box::new(page0_after),
            }).unwrap();
            writer.append(&WalRecord::Commit(1)).unwrap();
            writer.append(&WalRecord::Begin(2)).unwrap();
            writer.append(&WalRecord::Write {
                txn_id: 2, page_id: 1,
                before_image: Box::new(page1_original),
                after_image: Box::new(page1_after),
            }).unwrap();
            writer.sync().unwrap();
        }

        // First recovery.
        let (redo1, undo1) = super::recover(&db_path, &wal_path_f).unwrap();
        assert_eq!(redo1, 1);
        assert_eq!(undo1, 1);

        let p0_first = read_page_raw(&db_path, 0);
        let p1_first = read_page_raw(&db_path, 1);

        // Second recovery — same result.
        let (redo2, undo2) = super::recover(&db_path, &wal_path_f).unwrap();
        assert_eq!(redo2, 1);
        assert_eq!(undo2, 1);

        let p0_second = read_page_raw(&db_path, 0);
        let p1_second = read_page_raw(&db_path, 1);

        assert_eq!(p0_first, p0_second);
        assert_eq!(p1_first, p1_second);
        assert_eq!(p0_second[0], 0xCA);
        assert_eq!(p0_second[1], 0xFE);
        assert_eq!(p1_second, page1_original);

        cleanup(&db_path);
        cleanup(&wal_path_f);
    }

    // ── Test 5: Aborted transaction ──

    #[test]
    fn test_recover_aborted_transaction() {
        let db_path = temp_db_path("recover_aborted.db");
        let wal_path_f = temp_db_path("recover_aborted.wal");
        cleanup(&db_path);
        cleanup(&wal_path_f);

        let original = [0x55u8; PAGE_SIZE];
        write_page_raw(&db_path, 0, &original);

        let after = [0xFFu8; PAGE_SIZE];

        {
            let mut writer = WalWriter::open(&wal_path_f).unwrap();
            writer.append(&WalRecord::Begin(1)).unwrap();
            writer.append(&WalRecord::Write {
                txn_id: 1, page_id: 0,
                before_image: Box::new(original),
                after_image: Box::new(after),
            }).unwrap();
            writer.append(&WalRecord::Abort(1)).unwrap();
            writer.sync().unwrap();
        }

        let (redo, undo) = super::recover(&db_path, &wal_path_f).unwrap();
        assert_eq!(redo, 0);
        assert_eq!(undo, 1);

        let page = read_page_raw(&db_path, 0);
        assert_eq!(page, original);

        cleanup(&db_path);
        cleanup(&wal_path_f);
    }

    // ── Test 6: Multiple writes in one committed transaction ──

    #[test]
    fn test_recover_multiple_writes_committed() {
        let db_path = temp_db_path("recover_multi_write.db");
        let wal_path_f = temp_db_path("recover_multi_write.wal");
        cleanup(&db_path);
        cleanup(&wal_path_f);

        let zero = [0u8; PAGE_SIZE];
        for i in 0..3u32 {
            write_page_raw(&db_path, i, &zero);
        }

        let mut a0 = [0u8; PAGE_SIZE]; a0[0] = 0x10;
        let mut a1 = [0u8; PAGE_SIZE]; a1[0] = 0x20;
        let mut a2 = [0u8; PAGE_SIZE]; a2[0] = 0x30;

        {
            let mut writer = WalWriter::open(&wal_path_f).unwrap();
            writer.append(&WalRecord::Begin(1)).unwrap();
            writer.append(&WalRecord::Write {
                txn_id: 1, page_id: 0,
                before_image: Box::new(zero), after_image: Box::new(a0),
            }).unwrap();
            writer.append(&WalRecord::Write {
                txn_id: 1, page_id: 1,
                before_image: Box::new(zero), after_image: Box::new(a1),
            }).unwrap();
            writer.append(&WalRecord::Write {
                txn_id: 1, page_id: 2,
                before_image: Box::new(zero), after_image: Box::new(a2),
            }).unwrap();
            writer.append(&WalRecord::Commit(1)).unwrap();
            writer.sync().unwrap();
        }

        let (redo, undo) = super::recover(&db_path, &wal_path_f).unwrap();
        assert_eq!(redo, 1);
        assert_eq!(undo, 0);

        assert_eq!(read_page_raw(&db_path, 0)[0], 0x10);
        assert_eq!(read_page_raw(&db_path, 1)[0], 0x20);
        assert_eq!(read_page_raw(&db_path, 2)[0], 0x30);

        cleanup(&db_path);
        cleanup(&wal_path_f);
    }

    // ── Test 7: No WAL file ──

    #[test]
    fn test_recover_no_wal_file() {
        let db_path = temp_db_path("recover_no_wal.db");
        let wal_path_f = temp_db_path("recover_no_wal_gone.wal");
        cleanup(&db_path);
        cleanup(&wal_path_f);

        let data = [0x42u8; PAGE_SIZE];
        write_page_raw(&db_path, 0, &data);

        let (redo, undo) = super::recover(&db_path, &wal_path_f).unwrap();
        assert_eq!(redo, 0);
        assert_eq!(undo, 0);

        let page = read_page_raw(&db_path, 0);
        assert_eq!(page, data);

        cleanup(&db_path);
    }

    // ── Test 8: Empty WAL file ──

    #[test]
    fn test_recover_empty_wal() {
        let db_path = temp_db_path("recover_empty_wal.db");
        let wal_path_f = temp_db_path("recover_empty_wal.wal");
        cleanup(&db_path);
        cleanup(&wal_path_f);

        let data = [0x42u8; PAGE_SIZE];
        write_page_raw(&db_path, 0, &data);

        { let _w = WalWriter::open(&wal_path_f).unwrap(); }

        let (redo, undo) = super::recover(&db_path, &wal_path_f).unwrap();
        assert_eq!(redo, 0);
        assert_eq!(undo, 0);

        let page = read_page_raw(&db_path, 0);
        assert_eq!(page, data);

        cleanup(&db_path);
        cleanup(&wal_path_f);
    }

    // ── Test 9: WAL truncation ──

    #[test]
    fn test_wal_truncate() {
        let path = temp_wal_path("wal_truncate.wal");
        cleanup(&path);

        {
            let mut wal = Wal::open(&path).unwrap();
            wal.log_begin(1).unwrap();
            wal.log_write(1, 0, [0u8; PAGE_SIZE], [0xFFu8; PAGE_SIZE]).unwrap();
            wal.log_commit(1).unwrap();

            let mut reader = wal.reader().unwrap();
            assert_eq!(reader.read_all().unwrap().len(), 3);

            wal.truncate().unwrap();

            let mut reader = wal.reader().unwrap();
            assert!(reader.read_all().unwrap().is_empty());

            wal.log_begin(2).unwrap();
            wal.log_commit(2).unwrap();

            let mut reader = wal.reader().unwrap();
            let records = reader.read_all().unwrap();
            assert_eq!(records.len(), 2);
            assert_eq!(records[0], WalRecord::Begin(2));
            assert_eq!(records[1], WalRecord::Commit(2));
        }

        cleanup(&path);
    }

    // ── Test 10: Checkpoint + truncation after recovery ──

    #[test]
    fn test_checkpoint_and_truncate_after_recovery() {
        let db_path = temp_db_path("recover_ckpt_trunc.db");
        let wal_path_f = temp_db_path("recover_ckpt_trunc.wal");
        cleanup(&db_path);
        cleanup(&wal_path_f);

        let original = [0u8; PAGE_SIZE];
        write_page_raw(&db_path, 0, &original);

        let mut after = [0u8; PAGE_SIZE];
        after[0] = 0x77;

        {
            let mut wal = Wal::open(&wal_path_f).unwrap();
            wal.log_begin(1).unwrap();
            wal.log_write(1, 0, original, after).unwrap();
            wal.log_commit(1).unwrap();
            wal.log_checkpoint().unwrap();
        }

        let (redo, _) = super::recover(&db_path, &wal_path_f).unwrap();
        assert_eq!(redo, 1);
        assert_eq!(read_page_raw(&db_path, 0)[0], 0x77);

        {
            let mut wal = Wal::open(&wal_path_f).unwrap();
            wal.truncate().unwrap();
        }

        let (redo2, undo2) = super::recover(&db_path, &wal_path_f).unwrap();
        assert_eq!(redo2, 0);
        assert_eq!(undo2, 0);
        assert_eq!(read_page_raw(&db_path, 0)[0], 0x77);

        cleanup(&db_path);
        cleanup(&wal_path_f);
    }

    // ── Test 11: Many interleaved transactions ──

    #[test]
    fn test_recover_many_interleaved() {
        let db_path = temp_db_path("recover_many_il.db");
        let wal_path_f = temp_db_path("recover_many_il.wal");
        cleanup(&db_path);
        cleanup(&wal_path_f);

        for i in 0..4u32 {
            let mut data = [0u8; PAGE_SIZE];
            data[0] = i as u8;
            write_page_raw(&db_path, i, &data);
        }

        {
            let mut w = WalWriter::open(&wal_path_f).unwrap();
            w.append(&WalRecord::Begin(1)).unwrap();
            w.append(&WalRecord::Begin(2)).unwrap();

            let mut b0 = [0u8; PAGE_SIZE]; b0[0] = 0;
            let mut a0 = [0u8; PAGE_SIZE]; a0[0] = 0x10;
            w.append(&WalRecord::Write {
                txn_id: 1, page_id: 0,
                before_image: Box::new(b0), after_image: Box::new(a0),
            }).unwrap();

            let mut b1 = [0u8; PAGE_SIZE]; b1[0] = 1;
            let mut a1 = [0u8; PAGE_SIZE]; a1[0] = 0x11;
            w.append(&WalRecord::Write {
                txn_id: 1, page_id: 1,
                before_image: Box::new(b1), after_image: Box::new(a1),
            }).unwrap();

            let mut b2 = [0u8; PAGE_SIZE]; b2[0] = 2;
            let mut a2 = [0u8; PAGE_SIZE]; a2[0] = 0x22;
            w.append(&WalRecord::Write {
                txn_id: 2, page_id: 2,
                before_image: Box::new(b2), after_image: Box::new(a2),
            }).unwrap();

            let mut b3 = [0u8; PAGE_SIZE]; b3[0] = 3;
            let mut a3 = [0u8; PAGE_SIZE]; a3[0] = 0x33;
            w.append(&WalRecord::Write {
                txn_id: 2, page_id: 3,
                before_image: Box::new(b3), after_image: Box::new(a3),
            }).unwrap();

            w.append(&WalRecord::Commit(1)).unwrap();

            // Txn 3: overwrites page 1
            w.append(&WalRecord::Begin(3)).unwrap();
            let mut a1_v2 = [0u8; PAGE_SIZE]; a1_v2[0] = 0x99;
            w.append(&WalRecord::Write {
                txn_id: 3, page_id: 1,
                before_image: Box::new(a1), after_image: Box::new(a1_v2),
            }).unwrap();
            w.append(&WalRecord::Commit(3)).unwrap();
            w.sync().unwrap();
        }

        let (redo, undo) = super::recover(&db_path, &wal_path_f).unwrap();
        assert_eq!(redo, 2);
        assert_eq!(undo, 1);

        assert_eq!(read_page_raw(&db_path, 0)[0], 0x10);
        assert_eq!(read_page_raw(&db_path, 1)[0], 0x99);
        assert_eq!(read_page_raw(&db_path, 2)[0], 2);
        assert_eq!(read_page_raw(&db_path, 3)[0], 3);

        cleanup(&db_path);
        cleanup(&wal_path_f);
    }

    // ── Test 12: Truncated WAL tail (crash mid-write) ──

    #[test]
    fn test_recover_truncated_wal_tail() {
        let db_path = temp_db_path("recover_trunc_tail.db");
        let wal_path_f = temp_db_path("recover_trunc_tail.wal");
        cleanup(&db_path);
        cleanup(&wal_path_f);

        let original = [0u8; PAGE_SIZE];
        write_page_raw(&db_path, 0, &original);

        let mut after = [0u8; PAGE_SIZE];
        after[0] = 0xEE;

        {
            let mut writer = WalWriter::open(&wal_path_f).unwrap();
            writer.append(&WalRecord::Begin(1)).unwrap();
            writer.append(&WalRecord::Write {
                txn_id: 1, page_id: 0,
                before_image: Box::new(original),
                after_image: Box::new(after),
            }).unwrap();
            writer.append(&WalRecord::Commit(1)).unwrap();
            writer.sync().unwrap();
        }

        // Append garbage (partial record).
        {
            use std::io::Write as IoWrite;
            let mut file = OpenOptions::new().append(true).open(&wal_path_f).unwrap();
            file.write_all(&1000u32.to_le_bytes()).unwrap();
            file.write_all(b"garbage").unwrap();
            file.flush().unwrap();
        }

        let (redo, undo) = super::recover(&db_path, &wal_path_f).unwrap();
        assert_eq!(redo, 1);
        assert_eq!(undo, 0);
        assert_eq!(read_page_raw(&db_path, 0)[0], 0xEE);

        cleanup(&db_path);
        cleanup(&wal_path_f);
    }
}
