/// Core data types for the database.
use std::fmt;

/// The supported column data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    Integer,
    Float,
    Text,
    Boolean,
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::Integer => write!(f, "INTEGER"),
            DataType::Float => write!(f, "FLOAT"),
            DataType::Text => write!(f, "TEXT"),
            DataType::Boolean => write!(f, "BOOLEAN"),
        }
    }
}

/// A runtime value stored in a row.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Integer(i64),
    Float(f64),
    Text(String),
    Boolean(bool),
}

impl Value {
    /// Returns the data type of this value, or None if Null.
    pub fn data_type(&self) -> Option<DataType> {
        match self {
            Value::Null => None,
            Value::Integer(_) => Some(DataType::Integer),
            Value::Float(_) => Some(DataType::Float),
            Value::Text(_) => Some(DataType::Text),
            Value::Boolean(_) => Some(DataType::Boolean),
        }
    }

    /// Returns true if this value is NULL.
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Try to interpret this value as an i64.
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Value::Integer(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to interpret this value as an f64.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Float(v) => Some(*v),
            Value::Integer(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Try to interpret this value as a string reference.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Value::Text(v) => Some(v.as_str()),
            _ => None,
        }
    }

    /// Try to interpret this value as a bool.
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            Value::Boolean(v) => Some(*v),
            _ => None,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => write!(f, "NULL"),
            Value::Integer(v) => write!(f, "{}", v),
            Value::Float(v) => write!(f, "{}", v),
            Value::Text(v) => write!(f, "{}", v),
            Value::Boolean(v) => write!(f, "{}", v),
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Value::Null, Value::Null) => Some(std::cmp::Ordering::Equal),
            (Value::Null, _) => Some(std::cmp::Ordering::Less),
            (_, Value::Null) => Some(std::cmp::Ordering::Greater),
            (Value::Integer(a), Value::Integer(b)) => a.partial_cmp(b),
            (Value::Float(a), Value::Float(b)) => a.partial_cmp(b),
            (Value::Integer(a), Value::Float(b)) => (*a as f64).partial_cmp(b),
            (Value::Float(a), Value::Integer(b)) => a.partial_cmp(&(*b as f64)),
            (Value::Text(a), Value::Text(b)) => a.partial_cmp(b),
            (Value::Boolean(a), Value::Boolean(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}

/// A single row of data, consisting of an ordered list of values.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub values: Vec<Value>,
}

impl Row {
    pub fn new(values: Vec<Value>) -> Self {
        Self { values }
    }

    /// Get the value at the given column index.
    pub fn get(&self, index: usize) -> Option<&Value> {
        self.values.get(index)
    }

    /// Number of columns in this row.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns true if the row has no values.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Definition of a single column in a table.
#[derive(Debug, Clone, PartialEq)]
pub struct Column {
    /// Column name.
    pub name: String,
    /// Column data type.
    pub data_type: DataType,
    /// Whether this column allows NULL values.
    pub nullable: bool,
}

impl Column {
    pub fn new(name: impl Into<String>, data_type: DataType, nullable: bool) -> Self {
        Self {
            name: name.into(),
            data_type,
            nullable,
        }
    }
}

/// Schema for a table — an ordered list of columns.
#[derive(Debug, Clone, PartialEq)]
pub struct Schema {
    pub columns: Vec<Column>,
}

impl Schema {
    pub fn new(columns: Vec<Column>) -> Self {
        Self { columns }
    }

    /// Look up a column index by name (case-insensitive).
    pub fn column_index(&self, name: &str) -> Option<usize> {
        let lower = name.to_lowercase();
        self.columns
            .iter()
            .position(|c| c.name.to_lowercase() == lower)
    }

    /// Get a column definition by index.
    pub fn column(&self, index: usize) -> Option<&Column> {
        self.columns.get(index)
    }

    /// Number of columns.
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    /// Returns true if the schema has no columns.
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// Validate that a row matches this schema (correct number of columns and compatible types).
    pub fn validate_row(&self, row: &Row) -> Result<(), String> {
        if row.len() != self.len() {
            return Err(format!(
                "Row has {} values but schema has {} columns",
                row.len(),
                self.len()
            ));
        }
        for (i, (value, col)) in row.values.iter().zip(self.columns.iter()).enumerate() {
            if value.is_null() {
                if !col.nullable {
                    return Err(format!("Column '{}' (index {}) does not allow NULL", col.name, i));
                }
                continue;
            }
            if let Some(vtype) = value.data_type() {
                if vtype != col.data_type {
                    return Err(format!(
                        "Column '{}' (index {}) expects {:?} but got {:?}",
                        col.name, i, col.data_type, vtype
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Full table definition including name and schema.
#[derive(Debug, Clone, PartialEq)]
pub struct TableDef {
    /// Table name.
    pub name: String,
    /// Table schema.
    pub schema: Schema,
}

impl TableDef {
    pub fn new(name: impl Into<String>, schema: Schema) -> Self {
        Self {
            name: name.into(),
            schema,
        }
    }
}

// ───────────────────────── Tests ─────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_data_type() {
        assert_eq!(Value::Integer(1).data_type(), Some(DataType::Integer));
        assert_eq!(Value::Float(1.5).data_type(), Some(DataType::Float));
        assert_eq!(Value::Text("hi".into()).data_type(), Some(DataType::Text));
        assert_eq!(Value::Boolean(true).data_type(), Some(DataType::Boolean));
        assert_eq!(Value::Null.data_type(), None);
    }

    #[test]
    fn test_value_is_null() {
        assert!(Value::Null.is_null());
        assert!(!Value::Integer(0).is_null());
    }

    #[test]
    fn test_value_accessors() {
        assert_eq!(Value::Integer(42).as_integer(), Some(42));
        assert_eq!(Value::Float(3.14).as_float(), Some(3.14));
        assert_eq!(Value::Integer(5).as_float(), Some(5.0));
        assert_eq!(Value::Text("hello".into()).as_text(), Some("hello"));
        assert_eq!(Value::Boolean(true).as_boolean(), Some(true));

        // Wrong type returns None
        assert_eq!(Value::Text("hi".into()).as_integer(), None);
        assert_eq!(Value::Integer(1).as_text(), None);
    }

    #[test]
    fn test_value_display() {
        assert_eq!(format!("{}", Value::Null), "NULL");
        assert_eq!(format!("{}", Value::Integer(42)), "42");
        assert_eq!(format!("{}", Value::Float(3.14)), "3.14");
        assert_eq!(format!("{}", Value::Text("hello".into())), "hello");
        assert_eq!(format!("{}", Value::Boolean(true)), "true");
    }

    #[test]
    fn test_value_ordering() {
        assert!(Value::Integer(1) < Value::Integer(2));
        assert!(Value::Float(1.0) < Value::Float(2.0));
        assert!(Value::Text("a".into()) < Value::Text("b".into()));
        assert!(Value::Null < Value::Integer(0));

        // Cross-type numeric comparison
        assert_eq!(
            Value::Integer(1).partial_cmp(&Value::Float(1.0)),
            Some(std::cmp::Ordering::Equal)
        );

        // Incompatible types
        assert_eq!(Value::Integer(1).partial_cmp(&Value::Text("a".into())), None);
    }

    #[test]
    fn test_row_basic() {
        let row = Row::new(vec![Value::Integer(1), Value::Text("hello".into())]);
        assert_eq!(row.len(), 2);
        assert!(!row.is_empty());
        assert_eq!(row.get(0), Some(&Value::Integer(1)));
        assert_eq!(row.get(1), Some(&Value::Text("hello".into())));
        assert_eq!(row.get(2), None);
    }

    #[test]
    fn test_schema_column_index() {
        let schema = Schema::new(vec![
            Column::new("id", DataType::Integer, false),
            Column::new("name", DataType::Text, true),
        ]);
        assert_eq!(schema.column_index("id"), Some(0));
        assert_eq!(schema.column_index("ID"), Some(0));
        assert_eq!(schema.column_index("name"), Some(1));
        assert_eq!(schema.column_index("nonexistent"), None);
    }

    #[test]
    fn test_schema_validate_row_ok() {
        let schema = Schema::new(vec![
            Column::new("id", DataType::Integer, false),
            Column::new("name", DataType::Text, true),
        ]);
        let row = Row::new(vec![Value::Integer(1), Value::Text("Alice".into())]);
        assert!(schema.validate_row(&row).is_ok());
    }

    #[test]
    fn test_schema_validate_row_null_allowed() {
        let schema = Schema::new(vec![
            Column::new("id", DataType::Integer, false),
            Column::new("name", DataType::Text, true),
        ]);
        let row = Row::new(vec![Value::Integer(1), Value::Null]);
        assert!(schema.validate_row(&row).is_ok());
    }

    #[test]
    fn test_schema_validate_row_null_not_allowed() {
        let schema = Schema::new(vec![
            Column::new("id", DataType::Integer, false),
            Column::new("name", DataType::Text, true),
        ]);
        let row = Row::new(vec![Value::Null, Value::Text("Alice".into())]);
        assert!(schema.validate_row(&row).is_err());
    }

    #[test]
    fn test_schema_validate_row_wrong_type() {
        let schema = Schema::new(vec![
            Column::new("id", DataType::Integer, false),
            Column::new("name", DataType::Text, true),
        ]);
        let row = Row::new(vec![Value::Text("oops".into()), Value::Text("Alice".into())]);
        assert!(schema.validate_row(&row).is_err());
    }

    #[test]
    fn test_schema_validate_row_wrong_count() {
        let schema = Schema::new(vec![
            Column::new("id", DataType::Integer, false),
        ]);
        let row = Row::new(vec![Value::Integer(1), Value::Integer(2)]);
        assert!(schema.validate_row(&row).is_err());
    }

    #[test]
    fn test_table_def() {
        let schema = Schema::new(vec![
            Column::new("id", DataType::Integer, false),
            Column::new("name", DataType::Text, true),
        ]);
        let table = TableDef::new("users", schema.clone());
        assert_eq!(table.name, "users");
        assert_eq!(table.schema, schema);
    }

    #[test]
    fn test_datatype_display() {
        assert_eq!(format!("{}", DataType::Integer), "INTEGER");
        assert_eq!(format!("{}", DataType::Float), "FLOAT");
        assert_eq!(format!("{}", DataType::Text), "TEXT");
        assert_eq!(format!("{}", DataType::Boolean), "BOOLEAN");
    }
}
