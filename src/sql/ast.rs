//! SQL Abstract Syntax Tree (AST) type definitions.
//!
//! These types represent the parsed structure of SQL statements.
//! They are produced by the parser and consumed by the query executor.

use crate::types::DataType;

// ───────────────────────── Top-level Statement ─────────────────────────

/// A parsed SQL statement.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// CREATE TABLE ...
    CreateTable(CreateTable),
    /// INSERT INTO ... VALUES ...
    Insert(Insert),
    /// SELECT ...
    Select(Select),
    /// UPDATE ... SET ... WHERE ...
    Update(Update),
    /// DELETE FROM ... WHERE ...
    Delete(Delete),
    /// BEGIN
    Begin,
    /// COMMIT
    Commit,
    /// ROLLBACK
    Rollback,
}

// ───────────────────────── CREATE TABLE ─────────────────────────

/// CREATE TABLE statement.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateTable {
    /// Name of the table to create.
    pub name: String,
    /// Column definitions for the new table.
    pub columns: Vec<ColumnDef>,
}

/// A column definition within a CREATE TABLE statement.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnDef {
    /// Column name.
    pub name: String,
    /// Column data type.
    pub data_type: DataType,
    /// Whether the column allows NULL values.
    pub nullable: bool,
    /// Whether this column is the primary key.
    pub primary_key: bool,
}

// ───────────────────────── INSERT ─────────────────────────

/// INSERT INTO statement.
#[derive(Debug, Clone, PartialEq)]
pub struct Insert {
    /// Target table name.
    pub table: String,
    /// Optional explicit column list. If `None`, values are assumed to be in
    /// table-definition order.
    pub columns: Option<Vec<String>>,
    /// Rows to insert; each inner Vec corresponds to one row of values.
    pub values: Vec<Vec<Expression>>,
}

// ───────────────────────── SELECT ─────────────────────────

/// SELECT statement.
#[derive(Debug, Clone, PartialEq)]
pub struct Select {
    /// The columns / expressions to return.
    pub columns: Vec<SelectColumn>,
    /// The FROM clause (table references and joins).
    pub from: Option<FromClause>,
    /// Optional WHERE filter.
    pub r#where: Option<Expression>,
    /// Optional GROUP BY columns.
    pub group_by: Vec<Expression>,
    /// Optional ORDER BY specification.
    pub order_by: Vec<OrderByClause>,
    /// Optional LIMIT count.
    pub limit: Option<Expression>,
}

/// A single item in the SELECT column list.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectColumn {
    /// `*` — select all columns.
    AllColumns,
    /// A single expression, optionally aliased (`expr AS alias`).
    Expression {
        expr: Expression,
        alias: Option<String>,
    },
}

/// The FROM clause of a SELECT.
#[derive(Debug, Clone, PartialEq)]
pub struct FromClause {
    /// The primary table reference.
    pub table: TableRef,
    /// Zero or more JOINs.
    pub joins: Vec<Join>,
}

/// A table reference in a FROM clause.
#[derive(Debug, Clone, PartialEq)]
pub struct TableRef {
    /// Table name.
    pub name: String,
    /// Optional alias.
    pub alias: Option<String>,
}

/// A JOIN clause.
#[derive(Debug, Clone, PartialEq)]
pub struct Join {
    /// The type of join.
    pub join_type: JoinType,
    /// The table being joined.
    pub table: TableRef,
    /// The ON condition.
    pub on: Expression,
}

/// Supported join types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Cross,
}

/// ORDER BY clause entry.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderByClause {
    /// Expression to order by.
    pub expr: Expression,
    /// Sort direction.
    pub direction: SortDirection,
}

/// Sort direction for ORDER BY.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    Ascending,
    Descending,
}

// ───────────────────────── UPDATE ─────────────────────────

/// UPDATE statement.
#[derive(Debug, Clone, PartialEq)]
pub struct Update {
    /// Target table name.
    pub table: String,
    /// SET assignments.
    pub assignments: Vec<Assignment>,
    /// Optional WHERE filter.
    pub r#where: Option<Expression>,
}

/// A single `column = expression` assignment in an UPDATE SET clause.
#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    /// Column name being assigned.
    pub column: String,
    /// The value expression.
    pub value: Expression,
}

// ───────────────────────── DELETE ─────────────────────────

/// DELETE FROM statement.
#[derive(Debug, Clone, PartialEq)]
pub struct Delete {
    /// Target table name.
    pub table: String,
    /// Optional WHERE filter.
    pub r#where: Option<Expression>,
}

// ───────────────────────── Expressions ─────────────────────────

/// An expression that can appear in SELECT columns, WHERE clauses, etc.
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    /// A literal value.
    Literal(Literal),

    /// A column reference, optionally table-qualified (e.g. `users.id` or just `id`).
    ColumnRef {
        /// Optional table name or alias.
        table: Option<String>,
        /// Column name.
        column: String,
    },

    /// A binary operation (e.g. `a + b`, `x = 1`, `a AND b`).
    BinaryOp {
        left: Box<Expression>,
        op: BinaryOperator,
        right: Box<Expression>,
    },

    /// A unary operation (e.g. `NOT x`, `-5`).
    UnaryOp {
        op: UnaryOperator,
        operand: Box<Expression>,
    },

    /// An aggregate function call (e.g. `COUNT(*)`, `SUM(price)`).
    Aggregate {
        func: AggregateFunction,
        /// The argument expression. For `COUNT(*)` this is `None`.
        arg: Option<Box<Expression>>,
    },
}

/// A literal value in SQL.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    /// An integer literal (e.g. `42`).
    Integer(i64),
    /// A floating-point literal (e.g. `3.14`).
    Float(f64),
    /// A string literal (e.g. `'hello'`).
    String(String),
    /// A boolean literal (`TRUE` or `FALSE`).
    Boolean(bool),
    /// The NULL literal.
    Null,
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    // Comparison
    Equal,
    NotEqual,
    LessThan,
    GreaterThan,
    LessEqual,
    GreaterEqual,

    // Logical
    And,
    Or,

    // Arithmetic
    Add,
    Subtract,
    Multiply,
    Divide,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperator {
    /// Logical NOT.
    Not,
    /// Arithmetic negation (`-`).
    Negate,
}

/// Supported aggregate functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateFunction {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

// ───────────────────────── Tests ─────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statement_create_table() {
        let stmt = Statement::CreateTable(CreateTable {
            name: "users".to_string(),
            columns: vec![
                ColumnDef {
                    name: "id".to_string(),
                    data_type: DataType::Integer,
                    nullable: false,
                    primary_key: true,
                },
                ColumnDef {
                    name: "name".to_string(),
                    data_type: DataType::Text,
                    nullable: true,
                    primary_key: false,
                },
            ],
        });
        // Verify Debug and Clone work
        let _cloned = stmt.clone();
        let _debug = format!("{:?}", stmt);
    }

    #[test]
    fn test_statement_insert() {
        let stmt = Statement::Insert(Insert {
            table: "users".to_string(),
            columns: Some(vec!["id".to_string(), "name".to_string()]),
            values: vec![vec![
                Expression::Literal(Literal::Integer(1)),
                Expression::Literal(Literal::String("Alice".to_string())),
            ]],
        });
        let _cloned = stmt.clone();
        assert!(matches!(stmt, Statement::Insert(_)));
    }

    #[test]
    fn test_statement_select_simple() {
        let stmt = Statement::Select(Select {
            columns: vec![SelectColumn::AllColumns],
            from: Some(FromClause {
                table: TableRef {
                    name: "users".to_string(),
                    alias: None,
                },
                joins: vec![],
            }),
            r#where: None,
            group_by: vec![],
            order_by: vec![],
            limit: None,
        });
        assert!(matches!(stmt, Statement::Select(_)));
    }

    #[test]
    fn test_statement_select_with_join() {
        let select = Select {
            columns: vec![
                SelectColumn::Expression {
                    expr: Expression::ColumnRef {
                        table: Some("u".to_string()),
                        column: "name".to_string(),
                    },
                    alias: None,
                },
                SelectColumn::Expression {
                    expr: Expression::ColumnRef {
                        table: Some("o".to_string()),
                        column: "total".to_string(),
                    },
                    alias: Some("order_total".to_string()),
                },
            ],
            from: Some(FromClause {
                table: TableRef {
                    name: "users".to_string(),
                    alias: Some("u".to_string()),
                },
                joins: vec![Join {
                    join_type: JoinType::Inner,
                    table: TableRef {
                        name: "orders".to_string(),
                        alias: Some("o".to_string()),
                    },
                    on: Expression::BinaryOp {
                        left: Box::new(Expression::ColumnRef {
                            table: Some("u".to_string()),
                            column: "id".to_string(),
                        }),
                        op: BinaryOperator::Equal,
                        right: Box::new(Expression::ColumnRef {
                            table: Some("o".to_string()),
                            column: "user_id".to_string(),
                        }),
                    },
                }],
            }),
            r#where: Some(Expression::BinaryOp {
                left: Box::new(Expression::ColumnRef {
                    table: Some("o".to_string()),
                    column: "total".to_string(),
                }),
                op: BinaryOperator::GreaterThan,
                right: Box::new(Expression::Literal(Literal::Integer(100))),
            }),
            group_by: vec![],
            order_by: vec![OrderByClause {
                expr: Expression::ColumnRef {
                    table: Some("o".to_string()),
                    column: "total".to_string(),
                },
                direction: SortDirection::Descending,
            }],
            limit: Some(Expression::Literal(Literal::Integer(10))),
        };
        let stmt = Statement::Select(select);
        let _debug = format!("{:?}", stmt);
    }

    #[test]
    fn test_statement_select_with_aggregates() {
        let select = Select {
            columns: vec![
                SelectColumn::Expression {
                    expr: Expression::ColumnRef {
                        table: None,
                        column: "department".to_string(),
                    },
                    alias: None,
                },
                SelectColumn::Expression {
                    expr: Expression::Aggregate {
                        func: AggregateFunction::Count,
                        arg: None,
                    },
                    alias: Some("count".to_string()),
                },
                SelectColumn::Expression {
                    expr: Expression::Aggregate {
                        func: AggregateFunction::Avg,
                        arg: Some(Box::new(Expression::ColumnRef {
                            table: None,
                            column: "salary".to_string(),
                        })),
                    },
                    alias: Some("avg_salary".to_string()),
                },
            ],
            from: Some(FromClause {
                table: TableRef {
                    name: "employees".to_string(),
                    alias: None,
                },
                joins: vec![],
            }),
            r#where: None,
            group_by: vec![Expression::ColumnRef {
                table: None,
                column: "department".to_string(),
            }],
            order_by: vec![],
            limit: None,
        };
        let _stmt = Statement::Select(select);
    }

    #[test]
    fn test_statement_update() {
        let stmt = Statement::Update(Update {
            table: "users".to_string(),
            assignments: vec![
                Assignment {
                    column: "name".to_string(),
                    value: Expression::Literal(Literal::String("Bob".to_string())),
                },
                Assignment {
                    column: "age".to_string(),
                    value: Expression::BinaryOp {
                        left: Box::new(Expression::ColumnRef {
                            table: None,
                            column: "age".to_string(),
                        }),
                        op: BinaryOperator::Add,
                        right: Box::new(Expression::Literal(Literal::Integer(1))),
                    },
                },
            ],
            r#where: Some(Expression::BinaryOp {
                left: Box::new(Expression::ColumnRef {
                    table: None,
                    column: "id".to_string(),
                }),
                op: BinaryOperator::Equal,
                right: Box::new(Expression::Literal(Literal::Integer(1))),
            }),
        });
        assert!(matches!(stmt, Statement::Update(_)));
    }

    #[test]
    fn test_statement_delete() {
        let stmt = Statement::Delete(Delete {
            table: "users".to_string(),
            r#where: Some(Expression::BinaryOp {
                left: Box::new(Expression::ColumnRef {
                    table: None,
                    column: "active".to_string(),
                }),
                op: BinaryOperator::Equal,
                right: Box::new(Expression::Literal(Literal::Boolean(false))),
            }),
        });
        assert!(matches!(stmt, Statement::Delete(_)));
    }

    #[test]
    fn test_transaction_statements() {
        let begin = Statement::Begin;
        let commit = Statement::Commit;
        let rollback = Statement::Rollback;
        assert_eq!(begin, Statement::Begin);
        assert_eq!(commit, Statement::Commit);
        assert_eq!(rollback, Statement::Rollback);
    }

    #[test]
    fn test_expression_unary() {
        let expr = Expression::UnaryOp {
            op: UnaryOperator::Not,
            operand: Box::new(Expression::Literal(Literal::Boolean(true))),
        };
        let _debug = format!("{:?}", expr);

        let neg = Expression::UnaryOp {
            op: UnaryOperator::Negate,
            operand: Box::new(Expression::Literal(Literal::Integer(42))),
        };
        let _debug2 = format!("{:?}", neg);
    }

    #[test]
    fn test_expression_nested_binary() {
        // (a > 1) AND (b < 10)
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::BinaryOp {
                left: Box::new(Expression::ColumnRef {
                    table: None,
                    column: "a".to_string(),
                }),
                op: BinaryOperator::GreaterThan,
                right: Box::new(Expression::Literal(Literal::Integer(1))),
            }),
            op: BinaryOperator::And,
            right: Box::new(Expression::BinaryOp {
                left: Box::new(Expression::ColumnRef {
                    table: None,
                    column: "b".to_string(),
                }),
                op: BinaryOperator::LessThan,
                right: Box::new(Expression::Literal(Literal::Integer(10))),
            }),
        };
        let cloned = expr.clone();
        assert_eq!(expr, cloned);
    }

    #[test]
    fn test_all_binary_operators() {
        let ops = [
            BinaryOperator::Equal,
            BinaryOperator::NotEqual,
            BinaryOperator::LessThan,
            BinaryOperator::GreaterThan,
            BinaryOperator::LessEqual,
            BinaryOperator::GreaterEqual,
            BinaryOperator::And,
            BinaryOperator::Or,
            BinaryOperator::Add,
            BinaryOperator::Subtract,
            BinaryOperator::Multiply,
            BinaryOperator::Divide,
        ];
        for op in &ops {
            let _debug = format!("{:?}", op);
            assert_eq!(*op, op.clone());
        }
    }

    #[test]
    fn test_all_aggregate_functions() {
        let funcs = [
            AggregateFunction::Count,
            AggregateFunction::Sum,
            AggregateFunction::Avg,
            AggregateFunction::Min,
            AggregateFunction::Max,
        ];
        for f in &funcs {
            let _debug = format!("{:?}", f);
            assert_eq!(*f, f.clone());
        }
    }

    #[test]
    fn test_all_join_types() {
        let types = [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Cross,
        ];
        for t in &types {
            let _debug = format!("{:?}", t);
            assert_eq!(*t, t.clone());
        }
    }

    #[test]
    fn test_all_literals() {
        let lits = [
            Literal::Integer(42),
            Literal::Float(3.14),
            Literal::String("hello".to_string()),
            Literal::Boolean(true),
            Literal::Null,
        ];
        for lit in &lits {
            let _debug = format!("{:?}", lit);
            assert_eq!(*lit, lit.clone());
        }
    }

    #[test]
    fn test_sort_direction() {
        assert_ne!(SortDirection::Ascending, SortDirection::Descending);
        assert_eq!(SortDirection::Ascending, SortDirection::Ascending.clone());
    }

    #[test]
    fn test_insert_without_columns() {
        let stmt = Statement::Insert(Insert {
            table: "users".to_string(),
            columns: None,
            values: vec![
                vec![
                    Expression::Literal(Literal::Integer(1)),
                    Expression::Literal(Literal::String("Alice".to_string())),
                ],
                vec![
                    Expression::Literal(Literal::Integer(2)),
                    Expression::Literal(Literal::String("Bob".to_string())),
                ],
            ],
        });
        if let Statement::Insert(insert) = &stmt {
            assert!(insert.columns.is_none());
            assert_eq!(insert.values.len(), 2);
        } else {
            panic!("Expected Insert statement");
        }
    }

    #[test]
    fn test_delete_without_where() {
        let stmt = Statement::Delete(Delete {
            table: "logs".to_string(),
            r#where: None,
        });
        if let Statement::Delete(del) = &stmt {
            assert!(del.r#where.is_none());
        }
    }

    #[test]
    fn test_select_count_star() {
        let select = Select {
            columns: vec![SelectColumn::Expression {
                expr: Expression::Aggregate {
                    func: AggregateFunction::Count,
                    arg: None,
                },
                alias: None,
            }],
            from: Some(FromClause {
                table: TableRef {
                    name: "users".to_string(),
                    alias: None,
                },
                joins: vec![],
            }),
            r#where: None,
            group_by: vec![],
            order_by: vec![],
            limit: None,
        };
        if let SelectColumn::Expression { expr, .. } = &select.columns[0] {
            assert!(matches!(
                expr,
                Expression::Aggregate {
                    func: AggregateFunction::Count,
                    arg: None,
                }
            ));
        }
    }
}
