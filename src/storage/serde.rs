//! Compact binary serialization/deserialization for `Value` and `Row` types.
//!
//! ## Encoding format
//!
//! ### Value encoding
//! - 1-byte type tag: 0=Null, 1=Integer, 2=Float, 3=Text, 4=Boolean
//! - Payload (varies by type):
//!   - Null: no payload
//!   - Integer: 8-byte little-endian i64
//!   - Float: 8-byte little-endian f64
//!   - Text: 4-byte little-endian length prefix + UTF-8 bytes
//!   - Boolean: 1 byte (0 = false, 1 = true)
//!
//! ### Row encoding
//! - 4-byte little-endian column count (u32)
//! - Each value serialized sequentially

use crate::error::{Error, Result};
use crate::types::{Row, Value};

// Type tags
const TAG_NULL: u8 = 0;
const TAG_INTEGER: u8 = 1;
const TAG_FLOAT: u8 = 2;
const TAG_TEXT: u8 = 3;
const TAG_BOOLEAN: u8 = 4;

/// Serialize a `Value` into a compact binary representation.
pub fn serialize_value(v: &Value) -> Vec<u8> {
    match v {
        Value::Null => vec![TAG_NULL],
        Value::Integer(i) => {
            let mut buf = Vec::with_capacity(9);
            buf.push(TAG_INTEGER);
            buf.extend_from_slice(&i.to_le_bytes());
            buf
        }
        Value::Float(f) => {
            let mut buf = Vec::with_capacity(9);
            buf.push(TAG_FLOAT);
            buf.extend_from_slice(&f.to_le_bytes());
            buf
        }
        Value::Text(s) => {
            let bytes = s.as_bytes();
            let len = bytes.len() as u32;
            let mut buf = Vec::with_capacity(1 + 4 + bytes.len());
            buf.push(TAG_TEXT);
            buf.extend_from_slice(&len.to_le_bytes());
            buf.extend_from_slice(bytes);
            buf
        }
        Value::Boolean(b) => {
            vec![TAG_BOOLEAN, if *b { 1 } else { 0 }]
        }
    }
}

/// Deserialize a `Value` from binary data starting at `data[0]`.
///
/// Returns the deserialized value and the number of bytes consumed.
pub fn deserialize_value(data: &[u8]) -> Result<(Value, usize)> {
    if data.is_empty() {
        return Err(Error::Storage("cannot deserialize value: empty data".into()));
    }

    let tag = data[0];
    match tag {
        TAG_NULL => Ok((Value::Null, 1)),
        TAG_INTEGER => {
            if data.len() < 9 {
                return Err(Error::Storage(format!(
                    "cannot deserialize integer: need 9 bytes, have {}",
                    data.len()
                )));
            }
            let i = i64::from_le_bytes(data[1..9].try_into().unwrap());
            Ok((Value::Integer(i), 9))
        }
        TAG_FLOAT => {
            if data.len() < 9 {
                return Err(Error::Storage(format!(
                    "cannot deserialize float: need 9 bytes, have {}",
                    data.len()
                )));
            }
            let f = f64::from_le_bytes(data[1..9].try_into().unwrap());
            Ok((Value::Float(f), 9))
        }
        TAG_TEXT => {
            if data.len() < 5 {
                return Err(Error::Storage(format!(
                    "cannot deserialize text: need at least 5 bytes for header, have {}",
                    data.len()
                )));
            }
            let len = u32::from_le_bytes(data[1..5].try_into().unwrap()) as usize;
            let total = 1 + 4 + len;
            if data.len() < total {
                return Err(Error::Storage(format!(
                    "cannot deserialize text: need {} bytes, have {}",
                    total,
                    data.len()
                )));
            }
            let s = std::str::from_utf8(&data[5..5 + len])
                .map_err(|e| Error::Storage(format!("invalid UTF-8 in text value: {}", e)))?;
            Ok((Value::Text(s.to_string()), total))
        }
        TAG_BOOLEAN => {
            if data.len() < 2 {
                return Err(Error::Storage(format!(
                    "cannot deserialize boolean: need 2 bytes, have {}",
                    data.len()
                )));
            }
            let b = data[1] != 0;
            Ok((Value::Boolean(b), 2))
        }
        _ => Err(Error::Storage(format!(
            "unknown value type tag: {}",
            tag
        ))),
    }
}

/// Serialize a `Row` into a compact binary representation.
///
/// Format: 4-byte LE column count, then each value serialized sequentially.
pub fn serialize_row(r: &Row) -> Vec<u8> {
    let count = r.values.len() as u32;
    let mut buf = Vec::new();
    buf.extend_from_slice(&count.to_le_bytes());
    for v in &r.values {
        buf.extend(serialize_value(v));
    }
    buf
}

/// Deserialize a `Row` from binary data starting at `data[0]`.
///
/// Returns the deserialized row and the number of bytes consumed.
pub fn deserialize_row(data: &[u8]) -> Result<(Row, usize)> {
    if data.len() < 4 {
        return Err(Error::Storage(format!(
            "cannot deserialize row: need at least 4 bytes for column count, have {}",
            data.len()
        )));
    }

    let count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let mut offset = 4;
    let mut values = Vec::with_capacity(count);

    for i in 0..count {
        if offset >= data.len() {
            return Err(Error::Storage(format!(
                "cannot deserialize row: unexpected end of data at value {}",
                i
            )));
        }
        let (value, consumed) = deserialize_value(&data[offset..])?;
        values.push(value);
        offset += consumed;
    }

    Ok((Row::new(values), offset))
}

// ───────────────────────── Tests ─────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_null() {
        let v = Value::Null;
        let bytes = serialize_value(&v);
        assert_eq!(bytes, vec![TAG_NULL]);
        let (decoded, consumed) = deserialize_value(&bytes).unwrap();
        assert_eq!(decoded, v);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_roundtrip_integer() {
        for &val in &[0i64, 1, -1, i64::MIN, i64::MAX, 42, -9999] {
            let v = Value::Integer(val);
            let bytes = serialize_value(&v);
            assert_eq!(bytes.len(), 9);
            assert_eq!(bytes[0], TAG_INTEGER);
            let (decoded, consumed) = deserialize_value(&bytes).unwrap();
            assert_eq!(decoded, v);
            assert_eq!(consumed, 9);
        }
    }

    #[test]
    fn test_roundtrip_float() {
        for &val in &[0.0f64, 1.5, -1.5, f64::MIN, f64::MAX, std::f64::consts::PI] {
            let v = Value::Float(val);
            let bytes = serialize_value(&v);
            assert_eq!(bytes.len(), 9);
            assert_eq!(bytes[0], TAG_FLOAT);
            let (decoded, consumed) = deserialize_value(&bytes).unwrap();
            assert_eq!(decoded, v);
            assert_eq!(consumed, 9);
        }
    }

    #[test]
    fn test_roundtrip_float_nan() {
        // NaN != NaN, so we check the bit pattern
        let v = Value::Float(f64::NAN);
        let bytes = serialize_value(&v);
        let (decoded, consumed) = deserialize_value(&bytes).unwrap();
        assert_eq!(consumed, 9);
        match decoded {
            Value::Float(f) => assert!(f.is_nan()),
            _ => panic!("expected Float"),
        }
    }

    #[test]
    fn test_roundtrip_text() {
        let v = Value::Text("hello, world!".into());
        let bytes = serialize_value(&v);
        assert_eq!(bytes[0], TAG_TEXT);
        // 1 tag + 4 length + 13 bytes of text
        assert_eq!(bytes.len(), 1 + 4 + 13);
        let (decoded, consumed) = deserialize_value(&bytes).unwrap();
        assert_eq!(decoded, v);
        assert_eq!(consumed, 18);
    }

    #[test]
    fn test_roundtrip_text_empty() {
        let v = Value::Text(String::new());
        let bytes = serialize_value(&v);
        assert_eq!(bytes.len(), 5); // tag + 4-byte length (0)
        let (decoded, consumed) = deserialize_value(&bytes).unwrap();
        assert_eq!(decoded, v);
        assert_eq!(consumed, 5);
    }

    #[test]
    fn test_roundtrip_text_unicode() {
        let v = Value::Text("日本語テスト 🎉".into());
        let bytes = serialize_value(&v);
        let (decoded, consumed) = deserialize_value(&bytes).unwrap();
        assert_eq!(decoded, v);
        assert_eq!(consumed, bytes.len());
    }

    #[test]
    fn test_roundtrip_boolean() {
        for &val in &[true, false] {
            let v = Value::Boolean(val);
            let bytes = serialize_value(&v);
            assert_eq!(bytes.len(), 2);
            assert_eq!(bytes[0], TAG_BOOLEAN);
            let (decoded, consumed) = deserialize_value(&bytes).unwrap();
            assert_eq!(decoded, v);
            assert_eq!(consumed, 2);
        }
    }

    #[test]
    fn test_roundtrip_row_mixed() {
        let row = Row::new(vec![
            Value::Integer(42),
            Value::Text("hello".into()),
            Value::Float(3.14),
            Value::Boolean(true),
            Value::Null,
        ]);
        let bytes = serialize_row(&row);
        let (decoded, consumed) = deserialize_row(&bytes).unwrap();
        assert_eq!(decoded, row);
        assert_eq!(consumed, bytes.len());
    }

    #[test]
    fn test_roundtrip_row_empty() {
        let row = Row::new(vec![]);
        let bytes = serialize_row(&row);
        assert_eq!(bytes.len(), 4); // just the column count
        let (decoded, consumed) = deserialize_row(&bytes).unwrap();
        assert_eq!(decoded, row);
        assert_eq!(consumed, 4);
    }

    #[test]
    fn test_roundtrip_row_all_nulls() {
        let row = Row::new(vec![Value::Null, Value::Null, Value::Null]);
        let bytes = serialize_row(&row);
        // 4 (count) + 3 * 1 (null tags) = 7
        assert_eq!(bytes.len(), 7);
        let (decoded, consumed) = deserialize_row(&bytes).unwrap();
        assert_eq!(decoded, row);
        assert_eq!(consumed, 7);
    }

    #[test]
    fn test_roundtrip_row_single_value() {
        let row = Row::new(vec![Value::Integer(999)]);
        let bytes = serialize_row(&row);
        let (decoded, consumed) = deserialize_row(&bytes).unwrap();
        assert_eq!(decoded, row);
        assert_eq!(consumed, bytes.len());
    }

    #[test]
    fn test_deserialize_value_empty_data() {
        let result = deserialize_value(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_value_unknown_tag() {
        let result = deserialize_value(&[0xFF]);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_value_truncated_integer() {
        let result = deserialize_value(&[TAG_INTEGER, 0, 0, 0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_value_truncated_float() {
        let result = deserialize_value(&[TAG_FLOAT, 0, 0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_value_truncated_text() {
        // Length says 10 but only 3 bytes of text
        let mut data = vec![TAG_TEXT];
        data.extend_from_slice(&10u32.to_le_bytes());
        data.extend_from_slice(b"abc");
        let result = deserialize_value(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_value_truncated_boolean() {
        let result = deserialize_value(&[TAG_BOOLEAN]);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_row_truncated_header() {
        let result = deserialize_row(&[0, 0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_row_truncated_values() {
        // Says 2 columns but only enough data for 1
        let mut data = Vec::new();
        data.extend_from_slice(&2u32.to_le_bytes());
        data.push(TAG_NULL); // first value OK
        // second value missing
        let result = deserialize_row(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_values_sequential() {
        // Serialize multiple values into one buffer and deserialize sequentially
        let values = vec![
            Value::Integer(1),
            Value::Text("test".into()),
            Value::Boolean(false),
        ];
        let mut buf = Vec::new();
        for v in &values {
            buf.extend(serialize_value(v));
        }

        let mut offset = 0;
        for expected in &values {
            let (decoded, consumed) = deserialize_value(&buf[offset..]).unwrap();
            assert_eq!(&decoded, expected);
            offset += consumed;
        }
        assert_eq!(offset, buf.len());
    }

    #[test]
    fn test_row_with_long_text() {
        let long_text = "a".repeat(10_000);
        let row = Row::new(vec![Value::Text(long_text)]);
        let bytes = serialize_row(&row);
        let (decoded, consumed) = deserialize_row(&bytes).unwrap();
        assert_eq!(decoded, row);
        assert_eq!(consumed, bytes.len());
    }
}
