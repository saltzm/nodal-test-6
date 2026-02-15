/// In-memory catalog that stores table metadata (schemas).
use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::types::TableDef;

/// The catalog keeps track of all table definitions in the database.
#[derive(Debug, Clone)]
pub struct Catalog {
    tables: HashMap<String, TableDef>,
}

impl Catalog {
    /// Create a new empty catalog.
    pub fn new() -> Self {
        Self {
            tables: HashMap::new(),
        }
    }

    /// Register a new table. Returns an error if a table with the same name already exists.
    pub fn create_table(&mut self, table: TableDef) -> Result<()> {
        let name_lower = table.name.to_lowercase();
        if self.tables.contains_key(&name_lower) {
            return Err(Error::Catalog(format!(
                "Table '{}' already exists",
                table.name
            )));
        }
        self.tables.insert(name_lower, table);
        Ok(())
    }

    /// Look up a table definition by name (case-insensitive).
    pub fn get_table(&self, name: &str) -> Result<&TableDef> {
        self.tables
            .get(&name.to_lowercase())
            .ok_or_else(|| Error::Catalog(format!("Table '{}' not found", name)))
    }

    /// Drop a table by name. Returns an error if the table does not exist.
    pub fn drop_table(&mut self, name: &str) -> Result<TableDef> {
        self.tables
            .remove(&name.to_lowercase())
            .ok_or_else(|| Error::Catalog(format!("Table '{}' not found", name)))
    }

    /// List all table names.
    pub fn table_names(&self) -> Vec<&str> {
        self.tables.keys().map(|s| s.as_str()).collect()
    }

    /// Check whether a table exists.
    pub fn has_table(&self, name: &str) -> bool {
        self.tables.contains_key(&name.to_lowercase())
    }
}

impl Default for Catalog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Column, DataType, Schema};

    fn sample_table(name: &str) -> TableDef {
        TableDef::new(
            name,
            Schema::new(vec![
                Column::new("id", DataType::Integer, false),
                Column::new("name", DataType::Text, true),
            ]),
        )
    }

    #[test]
    fn test_create_and_get() {
        let mut catalog = Catalog::new();
        catalog.create_table(sample_table("users")).unwrap();
        let table = catalog.get_table("users").unwrap();
        assert_eq!(table.name, "users");
        assert_eq!(table.schema.len(), 2);
    }

    #[test]
    fn test_case_insensitive() {
        let mut catalog = Catalog::new();
        catalog.create_table(sample_table("Users")).unwrap();
        assert!(catalog.get_table("USERS").is_ok());
        assert!(catalog.get_table("users").is_ok());
    }

    #[test]
    fn test_duplicate_table() {
        let mut catalog = Catalog::new();
        catalog.create_table(sample_table("users")).unwrap();
        assert!(catalog.create_table(sample_table("users")).is_err());
    }

    #[test]
    fn test_drop_table() {
        let mut catalog = Catalog::new();
        catalog.create_table(sample_table("users")).unwrap();
        assert!(catalog.has_table("users"));
        catalog.drop_table("users").unwrap();
        assert!(!catalog.has_table("users"));
    }

    #[test]
    fn test_drop_nonexistent() {
        let mut catalog = Catalog::new();
        assert!(catalog.drop_table("nope").is_err());
    }

    #[test]
    fn test_table_names() {
        let mut catalog = Catalog::new();
        catalog.create_table(sample_table("alpha")).unwrap();
        catalog.create_table(sample_table("beta")).unwrap();
        let mut names = catalog.table_names();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_get_nonexistent() {
        let catalog = Catalog::new();
        assert!(catalog.get_table("nope").is_err());
    }
}
