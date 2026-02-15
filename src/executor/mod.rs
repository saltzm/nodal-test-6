//! Query executor — expression evaluation, result types, and query execution.

use crate::error::{Error, Result};
use crate::sql::ast::{BinaryOperator, Expression, Literal, UnaryOperator};
use crate::types::{Row, Value};

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