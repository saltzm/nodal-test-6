//! Query executor — expression evaluation, result types, and query execution.

use crate::catalog::Catalog;
use crate::error::{Error, Result};
use crate::sql::ast::{
    BinaryOperator, Expression, Literal, SelectColumn, SortDirection, Statement, UnaryOperator,
};
use crate::storage::table::TableStorage;
use crate::types::{Column, Row, Schema, TableDef, Value};

// ───────────────────────── ExecutionResult ─────────────────────────

/// The result of executing a SQL statement.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionResult {
    /// A table was created successfully.
    Created,
    /// Rows were inserted.
    Inserted { count: usize },
    /// Rows were selected (query result).
    Selected {
        columns: Vec<String>,
        rows: Vec<Row>,
    },
    /// Rows were deleted.
    Deleted { count: usize },
    /// Rows were updated.
    Updated { count: usize },
}

// ───────────────────────── Expression Evaluator ─────────────────────────

/// Evaluate an expression against a row with the given column names.
///
/// `columns` provides the mapping from column name to index in the row.
/// Column lookup is case-insensitive.
pub fn evaluate_expr(expr: &Expression, columns: &[String], row: &Row) -> Result<Value> {
    match expr {
        Expression::Literal(lit) => Ok(literal_to_value(lit)),

        Expression::ColumnRef { table: _, column } => {
            // Case-insensitive column lookup (ignore table qualifier for now)
            let col_lower = column.to_lowercase();
            let index = columns
                .iter()
                .position(|c| c.to_lowercase() == col_lower)
                .ok_or_else(|| {
                    Error::Execution(format!("Column '{}' not found", column))
                })?;
            row.get(index)
                .cloned()
                .ok_or_else(|| {
                    Error::Execution(format!(
                        "Column index {} out of range (row has {} values)",
                        index,
                        row.len()
                    ))
                })
        }

        Expression::BinaryOp { left, op, right } => {
            let lval = evaluate_expr(left, columns, row)?;
            let rval = evaluate_expr(right, columns, row)?;
            eval_binary_op(*op, &lval, &rval)
        }

        Expression::UnaryOp { op, operand } => {
            let val = evaluate_expr(operand, columns, row)?;
            eval_unary_op(*op, &val)
        }

        Expression::Aggregate { func: _, arg: _ } => Err(Error::Execution(
            "Aggregate not supported in this context".to_string(),
        )),
    }
}

/// Convert an AST literal to a runtime Value.
fn literal_to_value(lit: &Literal) -> Value {
    match lit {
        Literal::Integer(v) => Value::Integer(*v),
        Literal::Float(v) => Value::Float(*v),
        Literal::String(v) => Value::Text(v.clone()),
        Literal::Boolean(v) => Value::Boolean(*v),
        Literal::Null => Value::Null,
    }
}

/// Evaluate a binary operation on two Values.
fn eval_binary_op(op: BinaryOperator, left: &Value, right: &Value) -> Result<Value> {
    match op {
        // Comparison operators
        BinaryOperator::Equal => eval_comparison(left, right, |ord| {
            ord == std::cmp::Ordering::Equal
        }),
        BinaryOperator::NotEqual => eval_comparison(left, right, |ord| {
            ord != std::cmp::Ordering::Equal
        }),
        BinaryOperator::LessThan => eval_comparison(left, right, |ord| {
            ord == std::cmp::Ordering::Less
        }),
        BinaryOperator::GreaterThan => eval_comparison(left, right, |ord| {
            ord == std::cmp::Ordering::Greater
        }),
        BinaryOperator::LessEqual => eval_comparison(left, right, |ord| {
            ord == std::cmp::Ordering::Less || ord == std::cmp::Ordering::Equal
        }),
        BinaryOperator::GreaterEqual => eval_comparison(left, right, |ord| {
            ord == std::cmp::Ordering::Greater || ord == std::cmp::Ordering::Equal
        }),

        // Arithmetic operators
        BinaryOperator::Add => eval_arithmetic(left, right, |a, b| a + b, |a, b| a + b),
        BinaryOperator::Subtract => {
            eval_arithmetic(left, right, |a, b| a - b, |a, b| a - b)
        }
        BinaryOperator::Multiply => {
            eval_arithmetic(left, right, |a, b| a * b, |a, b| a * b)
        }
        BinaryOperator::Divide => eval_division(left, right),

        // Logical operators
        BinaryOperator::And => eval_logical_and(left, right),
        BinaryOperator::Or => eval_logical_or(left, right),
    }
}

/// Evaluate a comparison operation using partial_cmp.
fn eval_comparison(
    left: &Value,
    right: &Value,
    predicate: impl Fn(std::cmp::Ordering) -> bool,
) -> Result<Value> {
    // NULL comparisons: NULL = NULL is true, NULL compared to non-NULL follows
    // the PartialOrd implementation on Value.
    match left.partial_cmp(right) {
        Some(ord) => Ok(Value::Boolean(predicate(ord))),
        None => Err(Error::Execution(format!(
            "Cannot compare {:?} with {:?}",
            left, right
        ))),
    }
}

/// Evaluate arithmetic (add, subtract, multiply) with type promotion.
///
/// Integer+Integer→Integer, Float+Float→Float, Integer+Float→Float.
fn eval_arithmetic(
    left: &Value,
    right: &Value,
    int_op: impl Fn(i64, i64) -> i64,
    float_op: impl Fn(f64, f64) -> f64,
) -> Result<Value> {
    match (left, right) {
        (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(int_op(*a, *b))),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(float_op(*a, *b))),
        // Type promotion: Int + Float → Float
        (Value::Integer(a), Value::Float(b)) => {
            Ok(Value::Float(float_op(*a as f64, *b)))
        }
        (Value::Float(a), Value::Integer(b)) => {
            Ok(Value::Float(float_op(*a, *b as f64)))
        }
        (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
        _ => Err(Error::Execution(format!(
            "Cannot perform arithmetic on {:?} and {:?}",
            left, right
        ))),
    }
}

/// Evaluate division with division-by-zero check.
fn eval_division(left: &Value, right: &Value) -> Result<Value> {
    // Check for division by zero first
    match right {
        Value::Integer(0) => {
            return Err(Error::Execution("Division by zero".to_string()));
        }
        Value::Float(f) if *f == 0.0 => {
            return Err(Error::Execution("Division by zero".to_string()));
        }
        _ => {}
    }

    match (left, right) {
        (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a / b)),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),
        (Value::Integer(a), Value::Float(b)) => Ok(Value::Float(*a as f64 / b)),
        (Value::Float(a), Value::Integer(b)) => Ok(Value::Float(a / *b as f64)),
        (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
        _ => Err(Error::Execution(format!(
            "Cannot perform division on {:?} and {:?}",
            left, right
        ))),
    }
}

/// Evaluate logical AND on boolean values.
fn eval_logical_and(left: &Value, right: &Value) -> Result<Value> {
    match (left, right) {
        (Value::Boolean(a), Value::Boolean(b)) => Ok(Value::Boolean(*a && *b)),
        (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
        _ => Err(Error::Execution(format!(
            "AND requires boolean operands, got {:?} and {:?}",
            left, right
        ))),
    }
}

/// Evaluate logical OR on boolean values.
fn eval_logical_or(left: &Value, right: &Value) -> Result<Value> {
    match (left, right) {
        (Value::Boolean(a), Value::Boolean(b)) => Ok(Value::Boolean(*a || *b)),
        (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
        _ => Err(Error::Execution(format!(
            "OR requires boolean operands, got {:?} and {:?}",
            left, right
        ))),
    }
}

/// Evaluate a unary operation on a Value.
fn eval_unary_op(op: UnaryOperator, val: &Value) -> Result<Value> {
    match op {
        UnaryOperator::Not => match val {
            Value::Boolean(b) => Ok(Value::Boolean(!b)),
            Value::Null => Ok(Value::Null),
            _ => Err(Error::Execution(format!(
                "NOT requires a boolean operand, got {:?}",
                val
            ))),
        },
        UnaryOperator::Negate => match val {
            Value::Integer(v) => Ok(Value::Integer(-v)),
            Value::Float(v) => Ok(Value::Float(-v)),
            Value::Null => Ok(Value::Null),
            _ => Err(Error::Execution(format!(
                "Cannot negate {:?}",
                val
            ))),
        },
    }
}

// ───────────────────────── Database ─────────────────────────

/// The main database engine. Owns a catalog (in-memory schema info) and
/// table storage (disk-backed B-tree storage). Provides a high-level
/// `execute(sql)` interface for running SQL statements.
pub struct Database {
    catalog: Catalog,
    storage: TableStorage,
    /// Auto-incrementing key counter for inserts.
    next_id: i64,
}

impl Database {
    /// Open (or create) a database at the given path.
    pub fn open(path: &str, pool_size: usize) -> Result<Self> {
        let storage = TableStorage::open(path, pool_size)?;
        Ok(Self {
            catalog: Catalog::new(),
            storage,
            next_id: 1,
        })
    }

    /// Parse and execute a SQL statement, returning the result.
    pub fn execute(&mut self, sql: &str) -> Result<ExecutionResult> {
        let stmt = crate::sql::parse(sql)?;
        match stmt {
            Statement::CreateTable(ct) => self.execute_create_table(ct),
            Statement::Insert(ins) => self.execute_insert(ins),
            Statement::Select(sel) => self.execute_select(sel),
            Statement::Delete(del) => self.execute_delete(del),
            Statement::Update(upd) => self.execute_update(upd),
            Statement::Begin | Statement::Commit | Statement::Rollback => {
                Err(Error::Transaction("not yet implemented".to_string()))
            }
        }
    }

    // ─── CREATE TABLE ───

    fn execute_create_table(
        &mut self,
        ct: crate::sql::ast::CreateTable,
    ) -> Result<ExecutionResult> {
        // Convert AST ColumnDef list to types::Schema
        let columns: Vec<Column> = ct
            .columns
            .iter()
            .map(|cd| {
                Column::new(cd.name.clone(), cd.data_type, cd.nullable)
            })
            .collect();

        let schema = Schema::new(columns);
        let table_def = TableDef::new(ct.name.clone(), schema);

        // Register in catalog
        self.catalog.create_table(table_def)?;

        // Create storage table
        self.storage.create_table(&ct.name)?;

        Ok(ExecutionResult::Created)
    }

    // ─── INSERT ───

    fn execute_insert(
        &mut self,
        ins: crate::sql::ast::Insert,
    ) -> Result<ExecutionResult> {
        let table_def = self.catalog.get_table(&ins.table)?.clone();
        let schema = &table_def.schema;

        let mut count = 0;

        for row_exprs in &ins.values {
            // Evaluate each value expression (using empty column context — these are literals)
            let empty_cols: Vec<String> = vec![];
            let empty_row = Row::new(vec![]);

            let values: Vec<Value> = row_exprs
                .iter()
                .map(|expr| evaluate_expr(expr, &empty_cols, &empty_row))
                .collect::<Result<Vec<_>>>()?;

            // If explicit columns are specified, reorder values to match schema order
            let final_values = if let Some(ref col_names) = ins.columns {
                let mut ordered = vec![Value::Null; schema.len()];
                if col_names.len() != values.len() {
                    return Err(Error::Execution(format!(
                        "Column count ({}) does not match value count ({})",
                        col_names.len(),
                        values.len()
                    )));
                }
                for (i, col_name) in col_names.iter().enumerate() {
                    let idx = schema.column_index(col_name).ok_or_else(|| {
                        Error::Execution(format!(
                            "Column '{}' not found in table '{}'",
                            col_name, ins.table
                        ))
                    })?;
                    ordered[idx] = values[i].clone();
                }
                ordered
            } else {
                values
            };

            let row = Row::new(final_values);

            // Validate row against schema
            schema.validate_row(&row).map_err(|e| Error::Type(e))?;

            // Use auto-incrementing integer key
            let key = Value::Integer(self.next_id);
            self.next_id += 1;

            self.storage.insert_row(&ins.table, key, row)?;
            count += 1;
        }

        Ok(ExecutionResult::Inserted { count })
    }

    // ─── SELECT (single table, no joins/aggregates) ───

    fn execute_select(
        &mut self,
        sel: crate::sql::ast::Select,
    ) -> Result<ExecutionResult> {
        // Get the table name from FROM clause
        let from = sel.from.as_ref().ok_or_else(|| {
            Error::Execution("SELECT without FROM is not supported".to_string())
        })?;

        if !from.joins.is_empty() {
            return Err(Error::Execution(
                "JOINs are not yet supported".to_string(),
            ));
        }

        let table_name = &from.table.name;
        let table_def = self.catalog.get_table(table_name)?.clone();
        let schema = &table_def.schema;

        // Build column name list from schema
        let col_names: Vec<String> = schema
            .columns
            .iter()
            .map(|c| c.name.clone())
            .collect();

        // Scan table
        let entries = self.storage.scan_table(table_name)?;

        // Apply WHERE filter
        let mut filtered_rows: Vec<Row> = Vec::new();
        for (_key, row) in entries {
            if let Some(ref where_expr) = sel.r#where {
                let val = evaluate_expr(where_expr, &col_names, &row)?;
                match val {
                    Value::Boolean(true) => filtered_rows.push(row),
                    Value::Boolean(false) | Value::Null => {} // skip
                    _ => {
                        return Err(Error::Execution(
                            "WHERE clause must evaluate to a boolean".to_string(),
                        ))
                    }
                }
            } else {
                filtered_rows.push(row);
            }
        }

        // Apply ORDER BY (in-memory sort)
        if !sel.order_by.is_empty() {
            let order_by = sel.order_by.clone();
            let col_names_clone = col_names.clone();
            filtered_rows.sort_by(|a, b| {
                for ob in &order_by {
                    let va = evaluate_expr(&ob.expr, &col_names_clone, a)
                        .unwrap_or(Value::Null);
                    let vb = evaluate_expr(&ob.expr, &col_names_clone, b)
                        .unwrap_or(Value::Null);
                    let cmp = va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal);
                    let cmp = match ob.direction {
                        SortDirection::Ascending => cmp,
                        SortDirection::Descending => cmp.reverse(),
                    };
                    if cmp != std::cmp::Ordering::Equal {
                        return cmp;
                    }
                }
                std::cmp::Ordering::Equal
            });
        }

        // Apply LIMIT
        if let Some(ref limit_expr) = sel.limit {
            let limit_val =
                evaluate_expr(limit_expr, &col_names, &Row::new(vec![]))?;
            if let Value::Integer(n) = limit_val {
                if n >= 0 {
                    filtered_rows.truncate(n as usize);
                }
            } else {
                return Err(Error::Execution(
                    "LIMIT must evaluate to an integer".to_string(),
                ));
            }
        }

        // Check for aggregates — reject for now
        for sc in &sel.columns {
            if let SelectColumn::Expression { expr, .. } = sc {
                if Self::contains_aggregate(expr) {
                    return Err(Error::Execution(
                        "Aggregate functions are not yet supported".to_string(),
                    ));
                }
            }
        }

        // Project columns (handle * and named columns)
        let (result_columns, result_rows) =
            self.project_columns(&sel.columns, &col_names, &filtered_rows)?;

        Ok(ExecutionResult::Selected {
            columns: result_columns,
            rows: result_rows,
        })
    }

    /// Check if an expression contains an aggregate function call.
    fn contains_aggregate(expr: &Expression) -> bool {
        match expr {
            Expression::Aggregate { .. } => true,
            Expression::BinaryOp { left, right, .. } => {
                Self::contains_aggregate(left) || Self::contains_aggregate(right)
            }
            Expression::UnaryOp { operand, .. } => Self::contains_aggregate(operand),
            _ => false,
        }
    }

    /// Project columns from rows based on the SELECT column list.
    fn project_columns(
        &self,
        select_cols: &[SelectColumn],
        schema_cols: &[String],
        rows: &[Row],
    ) -> Result<(Vec<String>, Vec<Row>)> {
        // Determine output column names and the expressions to evaluate
        let mut out_col_names: Vec<String> = Vec::new();
        let mut col_exprs: Vec<Option<Expression>> = Vec::new(); // None means "all columns"

        for sc in select_cols {
            match sc {
                SelectColumn::AllColumns => {
                    for name in schema_cols {
                        out_col_names.push(name.clone());
                        col_exprs.push(None);
                    }
                }
                SelectColumn::Expression { expr, alias } => {
                    let name = if let Some(alias) = alias {
                        alias.clone()
                    } else {
                        Self::expr_display_name(expr)
                    };
                    out_col_names.push(name);
                    col_exprs.push(Some(expr.clone()));
                }
            }
        }

        // Now project each row
        let mut result_rows = Vec::with_capacity(rows.len());
        for row in rows {
            let mut values = Vec::with_capacity(col_exprs.len());
            let mut all_col_idx = 0usize;
            for (i, ce) in col_exprs.iter().enumerate() {
                match ce {
                    None => {
                        // AllColumns — the out_col_names were added in schema order
                        // Find which schema column index this corresponds to
                        let col_name = &out_col_names[i];
                        let idx = schema_cols
                            .iter()
                            .position(|c| c == col_name)
                            .unwrap_or(all_col_idx);
                        values.push(
                            row.get(idx).cloned().unwrap_or(Value::Null),
                        );
                        all_col_idx += 1;
                    }
                    Some(expr) => {
                        values.push(evaluate_expr(expr, schema_cols, row)?);
                    }
                }
            }
            result_rows.push(Row::new(values));
        }

        Ok((out_col_names, result_rows))
    }

    /// Generate a display name for an expression (used when no alias is given).
    fn expr_display_name(expr: &Expression) -> String {
        match expr {
            Expression::ColumnRef { table: Some(t), column } => {
                format!("{}.{}", t, column)
            }
            Expression::ColumnRef { table: None, column } => column.clone(),
            Expression::Literal(lit) => format!("{:?}", lit),
            _ => "?".to_string(),
        }
    }

    // ─── DELETE ───

    fn execute_delete(
        &mut self,
        del: crate::sql::ast::Delete,
    ) -> Result<ExecutionResult> {
        let table_def = self.catalog.get_table(&del.table)?.clone();
        let schema = &table_def.schema;
        let col_names: Vec<String> = schema
            .columns
            .iter()
            .map(|c| c.name.clone())
            .collect();

        // Scan table
        let entries = self.storage.scan_table(&del.table)?;

        // Find keys to delete
        let mut keys_to_delete: Vec<Value> = Vec::new();
        for (key, row) in &entries {
            if let Some(ref where_expr) = del.r#where {
                let val = evaluate_expr(where_expr, &col_names, row)?;
                match val {
                    Value::Boolean(true) => keys_to_delete.push(key.clone()),
                    Value::Boolean(false) | Value::Null => {}
                    _ => {
                        return Err(Error::Execution(
                            "WHERE clause must evaluate to a boolean".to_string(),
                        ))
                    }
                }
            } else {
                keys_to_delete.push(key.clone());
            }
        }

        let count = keys_to_delete.len();
        for key in &keys_to_delete {
            self.storage.delete_row(&del.table, key)?;
        }

        Ok(ExecutionResult::Deleted { count })
    }

    // ─── UPDATE ───

    fn execute_update(
        &mut self,
        upd: crate::sql::ast::Update,
    ) -> Result<ExecutionResult> {
        let table_def = self.catalog.get_table(&upd.table)?.clone();
        let schema = &table_def.schema;
        let col_names: Vec<String> = schema
            .columns
            .iter()
            .map(|c| c.name.clone())
            .collect();

        // Scan table
        let entries = self.storage.scan_table(&upd.table)?;

        // Find rows to update
        let mut updates: Vec<(Value, Row)> = Vec::new(); // (key, new_row)
        for (key, row) in &entries {
            let matches = if let Some(ref where_expr) = upd.r#where {
                let val = evaluate_expr(where_expr, &col_names, row)?;
                match val {
                    Value::Boolean(true) => true,
                    Value::Boolean(false) | Value::Null => false,
                    _ => {
                        return Err(Error::Execution(
                            "WHERE clause must evaluate to a boolean".to_string(),
                        ))
                    }
                }
            } else {
                true
            };

            if matches {
                // Apply SET assignments
                let mut new_values = row.values.clone();
                for assignment in &upd.assignments {
                    let col_idx = schema
                        .column_index(&assignment.column)
                        .ok_or_else(|| {
                            Error::Execution(format!(
                                "Column '{}' not found in table '{}'",
                                assignment.column, upd.table
                            ))
                        })?;
                    // Evaluate the SET expression using the current row context
                    let new_val =
                        evaluate_expr(&assignment.value, &col_names, row)?;
                    new_values[col_idx] = new_val;
                }
                updates.push((key.clone(), Row::new(new_values)));
            }
        }

        let count = updates.len();

        // Apply updates via delete + reinsert
        for (key, new_row) in updates {
            self.storage.delete_row(&upd.table, &key)?;
            self.storage.insert_row(&upd.table, key, new_row)?;
        }

        Ok(ExecutionResult::Updated { count })
    }
}

// ───────────────────────── Tests ─────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::ast::{AggregateFunction, BinaryOperator, Expression, Literal, UnaryOperator};

    /// Helper to build a simple row with known columns.
    fn test_columns() -> Vec<String> {
        vec![
            "id".to_string(),
            "name".to_string(),
            "age".to_string(),
            "score".to_string(),
            "active".to_string(),
        ]
    }

    fn test_row() -> Row {
        Row::new(vec![
            Value::Integer(1),
            Value::Text("Alice".to_string()),
            Value::Integer(30),
            Value::Float(95.5),
            Value::Boolean(true),
        ])
    }

    // ── Literal evaluation ──

    #[test]
    fn test_literal_integer() {
        let expr = Expression::Literal(Literal::Integer(42));
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Integer(42));
    }

    #[test]
    fn test_literal_float() {
        let expr = Expression::Literal(Literal::Float(3.14));
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Float(3.14));
    }

    #[test]
    fn test_literal_string() {
        let expr = Expression::Literal(Literal::String("hello".to_string()));
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Text("hello".to_string()));
    }

    #[test]
    fn test_literal_boolean() {
        let expr = Expression::Literal(Literal::Boolean(true));
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Boolean(true));
    }

    #[test]
    fn test_literal_null() {
        let expr = Expression::Literal(Literal::Null);
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Null);
    }

    // ── Column reference lookup ──

    #[test]
    fn test_column_ref_basic() {
        let cols = test_columns();
        let row = test_row();
        let expr = Expression::ColumnRef {
            table: None,
            column: "name".to_string(),
        };
        let result = evaluate_expr(&expr, &cols, &row).unwrap();
        assert_eq!(result, Value::Text("Alice".to_string()));
    }

    #[test]
    fn test_column_ref_case_insensitive() {
        let cols = test_columns();
        let row = test_row();
        let expr = Expression::ColumnRef {
            table: None,
            column: "NAME".to_string(),
        };
        let result = evaluate_expr(&expr, &cols, &row).unwrap();
        assert_eq!(result, Value::Text("Alice".to_string()));
    }

    #[test]
    fn test_column_ref_with_table_qualifier_ignored() {
        let cols = test_columns();
        let row = test_row();
        let expr = Expression::ColumnRef {
            table: Some("users".to_string()),
            column: "id".to_string(),
        };
        let result = evaluate_expr(&expr, &cols, &row).unwrap();
        assert_eq!(result, Value::Integer(1));
    }

    #[test]
    fn test_column_ref_not_found() {
        let cols = test_columns();
        let row = test_row();
        let expr = Expression::ColumnRef {
            table: None,
            column: "nonexistent".to_string(),
        };
        let result = evaluate_expr(&expr, &cols, &row);
        assert!(result.is_err());
    }

    // ── Comparison operators ──

    #[test]
    fn test_equal_integers() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Integer(5))),
            op: BinaryOperator::Equal,
            right: Box::new(Expression::Literal(Literal::Integer(5))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Boolean(true));
    }

    #[test]
    fn test_not_equal_integers() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Integer(5))),
            op: BinaryOperator::NotEqual,
            right: Box::new(Expression::Literal(Literal::Integer(3))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Boolean(true));
    }

    #[test]
    fn test_less_than() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Integer(3))),
            op: BinaryOperator::LessThan,
            right: Box::new(Expression::Literal(Literal::Integer(5))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Boolean(true));
    }

    #[test]
    fn test_greater_than() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Integer(10))),
            op: BinaryOperator::GreaterThan,
            right: Box::new(Expression::Literal(Literal::Integer(5))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Boolean(true));
    }

    #[test]
    fn test_less_equal() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Integer(5))),
            op: BinaryOperator::LessEqual,
            right: Box::new(Expression::Literal(Literal::Integer(5))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Boolean(true));

        let expr2 = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Integer(4))),
            op: BinaryOperator::LessEqual,
            right: Box::new(Expression::Literal(Literal::Integer(5))),
        };
        let result2 = evaluate_expr(&expr2, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result2, Value::Boolean(true));

        let expr3 = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Integer(6))),
            op: BinaryOperator::LessEqual,
            right: Box::new(Expression::Literal(Literal::Integer(5))),
        };
        let result3 = evaluate_expr(&expr3, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result3, Value::Boolean(false));
    }

    #[test]
    fn test_greater_equal() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Integer(5))),
            op: BinaryOperator::GreaterEqual,
            right: Box::new(Expression::Literal(Literal::Integer(5))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Boolean(true));
    }

    #[test]
    fn test_comparison_float() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Float(1.5))),
            op: BinaryOperator::LessThan,
            right: Box::new(Expression::Literal(Literal::Float(2.5))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Boolean(true));
    }

    #[test]
    fn test_comparison_mixed_int_float() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Integer(2))),
            op: BinaryOperator::Equal,
            right: Box::new(Expression::Literal(Literal::Float(2.0))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Boolean(true));
    }

    #[test]
    fn test_comparison_strings() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::String("abc".to_string()))),
            op: BinaryOperator::LessThan,
            right: Box::new(Expression::Literal(Literal::String("def".to_string()))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Boolean(true));
    }

    #[test]
    fn test_comparison_incompatible_types_error() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Integer(1))),
            op: BinaryOperator::Equal,
            right: Box::new(Expression::Literal(Literal::String("hello".to_string()))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![]));
        assert!(result.is_err());
    }

    // ── Arithmetic operators ──

    #[test]
    fn test_add_integers() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Integer(3))),
            op: BinaryOperator::Add,
            right: Box::new(Expression::Literal(Literal::Integer(4))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Integer(7));
    }

    #[test]
    fn test_add_floats() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Float(1.5))),
            op: BinaryOperator::Add,
            right: Box::new(Expression::Literal(Literal::Float(2.5))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Float(4.0));
    }

    #[test]
    fn test_add_int_float_promotion() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Integer(3))),
            op: BinaryOperator::Add,
            right: Box::new(Expression::Literal(Literal::Float(1.5))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Float(4.5));
    }

    #[test]
    fn test_add_float_int_promotion() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Float(1.5))),
            op: BinaryOperator::Add,
            right: Box::new(Expression::Literal(Literal::Integer(3))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Float(4.5));
    }

    #[test]
    fn test_subtract() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Integer(10))),
            op: BinaryOperator::Subtract,
            right: Box::new(Expression::Literal(Literal::Integer(3))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Integer(7));
    }

    #[test]
    fn test_multiply() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Integer(5))),
            op: BinaryOperator::Multiply,
            right: Box::new(Expression::Literal(Literal::Integer(6))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Integer(30));
    }

    #[test]
    fn test_divide_integers() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Integer(10))),
            op: BinaryOperator::Divide,
            right: Box::new(Expression::Literal(Literal::Integer(3))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Integer(3)); // integer division
    }

    #[test]
    fn test_divide_floats() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Float(10.0))),
            op: BinaryOperator::Divide,
            right: Box::new(Expression::Literal(Literal::Float(4.0))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Float(2.5));
    }

    #[test]
    fn test_divide_by_zero_integer() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Integer(10))),
            op: BinaryOperator::Divide,
            right: Box::new(Expression::Literal(Literal::Integer(0))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![]));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{}", err).contains("Division by zero"));
    }

    #[test]
    fn test_divide_by_zero_float() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Float(10.0))),
            op: BinaryOperator::Divide,
            right: Box::new(Expression::Literal(Literal::Float(0.0))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![]));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{}", err).contains("Division by zero"));
    }

    #[test]
    fn test_arithmetic_with_null_returns_null() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Integer(5))),
            op: BinaryOperator::Add,
            right: Box::new(Expression::Literal(Literal::Null)),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_arithmetic_type_error() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Integer(5))),
            op: BinaryOperator::Add,
            right: Box::new(Expression::Literal(Literal::String("hello".to_string()))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![]));
        assert!(result.is_err());
    }

    // ── Logical operators ──

    #[test]
    fn test_and_true_true() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Boolean(true))),
            op: BinaryOperator::And,
            right: Box::new(Expression::Literal(Literal::Boolean(true))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Boolean(true));
    }

    #[test]
    fn test_and_true_false() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Boolean(true))),
            op: BinaryOperator::And,
            right: Box::new(Expression::Literal(Literal::Boolean(false))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Boolean(false));
    }

    #[test]
    fn test_or_false_true() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Boolean(false))),
            op: BinaryOperator::Or,
            right: Box::new(Expression::Literal(Literal::Boolean(true))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Boolean(true));
    }

    #[test]
    fn test_or_false_false() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Boolean(false))),
            op: BinaryOperator::Or,
            right: Box::new(Expression::Literal(Literal::Boolean(false))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Boolean(false));
    }

    #[test]
    fn test_logical_and_type_error() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Integer(1))),
            op: BinaryOperator::And,
            right: Box::new(Expression::Literal(Literal::Boolean(true))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![]));
        assert!(result.is_err());
    }

    // ── Unary operators ──

    #[test]
    fn test_not_true() {
        let expr = Expression::UnaryOp {
            op: UnaryOperator::Not,
            operand: Box::new(Expression::Literal(Literal::Boolean(true))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Boolean(false));
    }

    #[test]
    fn test_not_false() {
        let expr = Expression::UnaryOp {
            op: UnaryOperator::Not,
            operand: Box::new(Expression::Literal(Literal::Boolean(false))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Boolean(true));
    }

    #[test]
    fn test_negate_integer() {
        let expr = Expression::UnaryOp {
            op: UnaryOperator::Negate,
            operand: Box::new(Expression::Literal(Literal::Integer(42))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Integer(-42));
    }

    #[test]
    fn test_negate_float() {
        let expr = Expression::UnaryOp {
            op: UnaryOperator::Negate,
            operand: Box::new(Expression::Literal(Literal::Float(3.14))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Float(-3.14));
    }

    #[test]
    fn test_not_type_error() {
        let expr = Expression::UnaryOp {
            op: UnaryOperator::Not,
            operand: Box::new(Expression::Literal(Literal::Integer(1))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![]));
        assert!(result.is_err());
    }

    #[test]
    fn test_negate_type_error() {
        let expr = Expression::UnaryOp {
            op: UnaryOperator::Negate,
            operand: Box::new(Expression::Literal(Literal::String("hello".to_string()))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![]));
        assert!(result.is_err());
    }

    // ── Aggregate error ──

    #[test]
    fn test_aggregate_returns_error() {
        let expr = Expression::Aggregate {
            func: AggregateFunction::Count,
            arg: None,
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![]));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{}", err).contains("Aggregate not supported"));
    }

    // ── Complex / nested expressions ──

    #[test]
    fn test_nested_expression_with_column() {
        // (age > 25) AND (score >= 90.0)
        let cols = test_columns();
        let row = test_row(); // age=30, score=95.5
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::BinaryOp {
                left: Box::new(Expression::ColumnRef {
                    table: None,
                    column: "age".to_string(),
                }),
                op: BinaryOperator::GreaterThan,
                right: Box::new(Expression::Literal(Literal::Integer(25))),
            }),
            op: BinaryOperator::And,
            right: Box::new(Expression::BinaryOp {
                left: Box::new(Expression::ColumnRef {
                    table: None,
                    column: "score".to_string(),
                }),
                op: BinaryOperator::GreaterEqual,
                right: Box::new(Expression::Literal(Literal::Float(90.0))),
            }),
        };
        let result = evaluate_expr(&expr, &cols, &row).unwrap();
        assert_eq!(result, Value::Boolean(true));
    }

    #[test]
    fn test_arithmetic_expression_with_columns() {
        // age + 10
        let cols = test_columns();
        let row = test_row(); // age=30
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::ColumnRef {
                table: None,
                column: "age".to_string(),
            }),
            op: BinaryOperator::Add,
            right: Box::new(Expression::Literal(Literal::Integer(10))),
        };
        let result = evaluate_expr(&expr, &cols, &row).unwrap();
        assert_eq!(result, Value::Integer(40));
    }

    #[test]
    fn test_not_with_comparison() {
        // NOT (5 > 10)
        let expr = Expression::UnaryOp {
            op: UnaryOperator::Not,
            operand: Box::new(Expression::BinaryOp {
                left: Box::new(Expression::Literal(Literal::Integer(5))),
                op: BinaryOperator::GreaterThan,
                right: Box::new(Expression::Literal(Literal::Integer(10))),
            }),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Boolean(true));
    }

    #[test]
    fn test_mixed_type_promotion_in_multiply() {
        // 3 * 2.5 -> 7.5 (Float)
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Integer(3))),
            op: BinaryOperator::Multiply,
            right: Box::new(Expression::Literal(Literal::Float(2.5))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Float(7.5));
    }

    #[test]
    fn test_divide_int_by_float_promotion() {
        // 7 / 2.0 -> 3.5 (Float)
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Integer(7))),
            op: BinaryOperator::Divide,
            right: Box::new(Expression::Literal(Literal::Float(2.0))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Float(3.5));
    }

    // ── ExecutionResult ──

    #[test]
    fn test_execution_result_created() {
        let r = ExecutionResult::Created;
        assert_eq!(r, ExecutionResult::Created);
    }

    #[test]
    fn test_execution_result_inserted() {
        let r = ExecutionResult::Inserted { count: 5 };
        if let ExecutionResult::Inserted { count } = r {
            assert_eq!(count, 5);
        } else {
            panic!("Expected Inserted");
        }
    }

    #[test]
    fn test_execution_result_selected() {
        let r = ExecutionResult::Selected {
            columns: vec!["id".to_string(), "name".to_string()],
            rows: vec![
                Row::new(vec![Value::Integer(1), Value::Text("Alice".to_string())]),
            ],
        };
        if let ExecutionResult::Selected { columns, rows } = &r {
            assert_eq!(columns.len(), 2);
            assert_eq!(rows.len(), 1);
        } else {
            panic!("Expected Selected");
        }
    }

    #[test]
    fn test_execution_result_deleted() {
        let r = ExecutionResult::Deleted { count: 3 };
        if let ExecutionResult::Deleted { count } = r {
            assert_eq!(count, 3);
        } else {
            panic!("Expected Deleted");
        }
    }

    #[test]
    fn test_execution_result_updated() {
        let r = ExecutionResult::Updated { count: 2 };
        if let ExecutionResult::Updated { count } = r {
            assert_eq!(count, 2);
        } else {
            panic!("Expected Updated");
        }
    }

    #[test]
    fn test_unary_not_null() {
        let expr = Expression::UnaryOp {
            op: UnaryOperator::Not,
            operand: Box::new(Expression::Literal(Literal::Null)),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_unary_negate_null() {
        let expr = Expression::UnaryOp {
            op: UnaryOperator::Negate,
            operand: Box::new(Expression::Literal(Literal::Null)),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_or_with_null() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Boolean(true))),
            op: BinaryOperator::Or,
            right: Box::new(Expression::Literal(Literal::Null)),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_and_with_null() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Literal(Literal::Null)),
            op: BinaryOperator::And,
            right: Box::new(Expression::Literal(Literal::Boolean(false))),
        };
        let result = evaluate_expr(&expr, &[], &Row::new(vec![])).unwrap();
        assert_eq!(result, Value::Null);
    }
}
#[cfg(test)]
mod db_tests {
    use super::*;

    fn temp_db_path(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("toydb_test_executor");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn cleanup(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_db_create_table() {
        let path = temp_db_path("db_create.db");
        cleanup(&path);
        let mut db = Database::open(path.to_str().unwrap(), 100).unwrap();
        let result = db
            .execute("CREATE TABLE users (id INTEGER, name TEXT)")
            .unwrap();
        assert_eq!(result, ExecutionResult::Created);
        cleanup(&path);
    }

    #[test]
    fn test_db_create_duplicate_table() {
        let path = temp_db_path("db_create_dup.db");
        cleanup(&path);
        let mut db = Database::open(path.to_str().unwrap(), 100).unwrap();
        db.execute("CREATE TABLE users (id INTEGER, name TEXT)")
            .unwrap();
        let result = db.execute("CREATE TABLE users (id INTEGER)");
        assert!(result.is_err());
        cleanup(&path);
    }

    #[test]
    fn test_db_insert_and_select() {
        let path = temp_db_path("db_ins_sel.db");
        cleanup(&path);
        let mut db = Database::open(path.to_str().unwrap(), 100).unwrap();
        db.execute("CREATE TABLE users (id INTEGER, name TEXT)")
            .unwrap();
        let ins = db
            .execute("INSERT INTO users VALUES (1, 'Alice')")
            .unwrap();
        assert_eq!(ins, ExecutionResult::Inserted { count: 1 });

        let sel = db.execute("SELECT * FROM users").unwrap();
        if let ExecutionResult::Selected { columns, rows } = sel {
            assert_eq!(columns, vec!["id", "name"]);
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].values[0], Value::Integer(1));
            assert_eq!(rows[0].values[1], Value::Text("Alice".to_string()));
        } else {
            panic!("Expected Selected");
        }
        cleanup(&path);
    }

    #[test]
    fn test_db_select_with_where() {
        let path = temp_db_path("db_sel_where.db");
        cleanup(&path);
        let mut db = Database::open(path.to_str().unwrap(), 100).unwrap();
        db.execute("CREATE TABLE users (id INTEGER, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob')").unwrap();
        db.execute("INSERT INTO users VALUES (3, 'Charlie')").unwrap();

        let sel = db.execute("SELECT * FROM users WHERE id > 1").unwrap();
        if let ExecutionResult::Selected { rows, .. } = sel {
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Selected");
        }
        cleanup(&path);
    }

    #[test]
    fn test_db_select_named_columns() {
        let path = temp_db_path("db_sel_named.db");
        cleanup(&path);
        let mut db = Database::open(path.to_str().unwrap(), 100).unwrap();
        db.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice', 30)")
            .unwrap();

        let sel = db.execute("SELECT name, age FROM users").unwrap();
        if let ExecutionResult::Selected { columns, rows } = sel {
            assert_eq!(columns, vec!["name", "age"]);
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].values[0], Value::Text("Alice".to_string()));
            assert_eq!(rows[0].values[1], Value::Integer(30));
        } else {
            panic!("Expected Selected");
        }
        cleanup(&path);
    }

    #[test]
    fn test_db_select_order_by_asc() {
        let path = temp_db_path("db_sel_ord_asc.db");
        cleanup(&path);
        let mut db = Database::open(path.to_str().unwrap(), 100).unwrap();
        db.execute("CREATE TABLE users (id INTEGER, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (3, 'Charlie')").unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob')").unwrap();

        let sel = db
            .execute("SELECT * FROM users ORDER BY id ASC")
            .unwrap();
        if let ExecutionResult::Selected { rows, .. } = sel {
            assert_eq!(rows.len(), 3);
            assert_eq!(rows[0].values[0], Value::Integer(1));
            assert_eq!(rows[1].values[0], Value::Integer(2));
            assert_eq!(rows[2].values[0], Value::Integer(3));
        } else {
            panic!("Expected Selected");
        }
        cleanup(&path);
    }

    #[test]
    fn test_db_select_order_by_desc() {
        let path = temp_db_path("db_sel_ord_desc.db");
        cleanup(&path);
        let mut db = Database::open(path.to_str().unwrap(), 100).unwrap();
        db.execute("CREATE TABLE users (id INTEGER, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (3, 'Charlie')").unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob')").unwrap();

        let sel = db
            .execute("SELECT * FROM users ORDER BY id DESC")
            .unwrap();
        if let ExecutionResult::Selected { rows, .. } = sel {
            assert_eq!(rows[0].values[0], Value::Integer(3));
            assert_eq!(rows[1].values[0], Value::Integer(2));
            assert_eq!(rows[2].values[0], Value::Integer(1));
        } else {
            panic!("Expected Selected");
        }
        cleanup(&path);
    }

    #[test]
    fn test_db_select_limit() {
        let path = temp_db_path("db_sel_limit.db");
        cleanup(&path);
        let mut db = Database::open(path.to_str().unwrap(), 100).unwrap();
        db.execute("CREATE TABLE users (id INTEGER, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob')").unwrap();
        db.execute("INSERT INTO users VALUES (3, 'Charlie')").unwrap();

        let sel = db.execute("SELECT * FROM users LIMIT 2").unwrap();
        if let ExecutionResult::Selected { rows, .. } = sel {
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Selected");
        }
        cleanup(&path);
    }

    #[test]
    fn test_db_delete() {
        let path = temp_db_path("db_delete.db");
        cleanup(&path);
        let mut db = Database::open(path.to_str().unwrap(), 100).unwrap();
        db.execute("CREATE TABLE users (id INTEGER, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob')").unwrap();
        db.execute("INSERT INTO users VALUES (3, 'Charlie')").unwrap();

        let del = db
            .execute("DELETE FROM users WHERE id = 2")
            .unwrap();
        assert_eq!(del, ExecutionResult::Deleted { count: 1 });

        let sel = db.execute("SELECT * FROM users").unwrap();
        if let ExecutionResult::Selected { rows, .. } = sel {
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Selected");
        }
        cleanup(&path);
    }

    #[test]
    fn test_db_update() {
        let path = temp_db_path("db_update.db");
        cleanup(&path);
        let mut db = Database::open(path.to_str().unwrap(), 100).unwrap();
        db.execute("CREATE TABLE users (id INTEGER, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob')").unwrap();

        let upd = db
            .execute("UPDATE users SET name = 'Bobby' WHERE id = 2")
            .unwrap();
        assert_eq!(upd, ExecutionResult::Updated { count: 1 });

        let sel = db
            .execute("SELECT * FROM users WHERE id = 2")
            .unwrap();
        if let ExecutionResult::Selected { rows, .. } = sel {
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].values[1], Value::Text("Bobby".to_string()));
        } else {
            panic!("Expected Selected");
        }
        cleanup(&path);
    }

    #[test]
    fn test_db_transaction_not_implemented() {
        let path = temp_db_path("db_txn.db");
        cleanup(&path);
        let mut db = Database::open(path.to_str().unwrap(), 100).unwrap();
        assert!(db.execute("BEGIN").is_err());
        assert!(db.execute("COMMIT").is_err());
        assert!(db.execute("ROLLBACK").is_err());
        cleanup(&path);
    }
}
