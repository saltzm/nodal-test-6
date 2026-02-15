//! SQL Recursive Descent Parser.
//!
//! Consumes tokens from the lexer and produces AST nodes.

use crate::error::{Error, Result};
use crate::sql::ast::*;
use crate::sql::lexer::{tokenize, Keyword, Token, TokenKind};
use crate::types::DataType;

// ───────────────────────── Parser struct ─────────────────────────

/// A recursive descent parser for SQL statements.
struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    // ─── Token navigation helpers ───

    fn peek(&self) -> Option<&TokenKind> {
        self.tokens.get(self.pos).map(|t| &t.kind)
    }

    fn advance(&mut self) -> Result<&Token> {
        if self.pos >= self.tokens.len() {
            return Err(Error::Parse("Unexpected end of input".to_string()));
        }
        let token = &self.tokens[self.pos];
        self.pos += 1;
        Ok(token)
    }

    fn at_end(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    fn expect_keyword(&mut self, kw: Keyword) -> Result<()> {
        match self.peek() {
            Some(TokenKind::Keyword(k)) if *k == kw => {
                self.advance()?;
                Ok(())
            }
            Some(other) => Err(Error::Parse(format!(
                "Expected keyword {:?}, got {:?}", kw, other
            ))),
            None => Err(Error::Parse(format!(
                "Expected keyword {:?}, got end of input", kw
            ))),
        }
    }

    fn expect_token(&mut self, expected: &TokenKind) -> Result<()> {
        match self.peek() {
            Some(kind) if kind == expected => {
                self.advance()?;
                Ok(())
            }
            Some(other) => Err(Error::Parse(format!(
                "Expected {:?}, got {:?}", expected, other
            ))),
            None => Err(Error::Parse(format!(
                "Expected {:?}, got end of input", expected
            ))),
        }
    }

    fn expect_identifier(&mut self) -> Result<String> {
        match self.peek().cloned() {
            Some(TokenKind::Identifier(name)) => {
                self.advance()?;
                Ok(name)
            }
            Some(other) => Err(Error::Parse(format!(
                "Expected identifier, got {:?}", other
            ))),
            None => Err(Error::Parse(
                "Expected identifier, got end of input".to_string(),
            )),
        }
    }

    fn peek_keyword(&self, kw: Keyword) -> bool {
        matches!(self.peek(), Some(TokenKind::Keyword(k)) if *k == kw)
    }

    fn match_keyword(&mut self, kw: Keyword) -> bool {
        if self.peek_keyword(kw) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn match_token(&mut self, kind: &TokenKind) -> bool {
        if self.peek() == Some(kind) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    // ─── Statement parsing ───

    fn parse_statement(&mut self) -> Result<Statement> {
        match self.peek() {
            Some(TokenKind::Keyword(Keyword::Select)) => self.parse_select(),
            Some(TokenKind::Keyword(Keyword::Create)) => self.parse_create_table(),
            Some(TokenKind::Keyword(Keyword::Insert)) => self.parse_insert(),
            Some(TokenKind::Keyword(Keyword::Delete)) => self.parse_delete(),
            Some(TokenKind::Keyword(Keyword::Update)) => self.parse_update(),
            Some(TokenKind::Keyword(Keyword::Begin)) => {
                self.advance()?;
                Ok(Statement::Begin)
            }
            Some(TokenKind::Keyword(Keyword::Commit)) => {
                self.advance()?;
                Ok(Statement::Commit)
            }
            Some(TokenKind::Keyword(Keyword::Rollback)) => {
                self.advance()?;
                Ok(Statement::Rollback)
            }
            Some(other) => Err(Error::Parse(format!(
                "Expected statement, got {:?}", other
            ))),
            None => Err(Error::Parse("Empty input".to_string())),
        }
    }

    // ─── SELECT ───

    fn parse_select(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Select)?;
        let columns = self.parse_select_columns()?;

        let from = if self.match_keyword(Keyword::From) {
            Some(self.parse_from_clause()?)
        } else {
            None
        };

        let where_clause = if self.match_keyword(Keyword::Where) {
            Some(self.parse_expression()?)
        } else {
            None
        };

        let group_by = if self.match_keyword(Keyword::Group) {
            self.expect_keyword(Keyword::By)?;
            self.parse_expression_list()?
        } else {
            vec![]
        };

        let order_by = if self.match_keyword(Keyword::Order) {
            self.expect_keyword(Keyword::By)?;
            self.parse_order_by_list()?
        } else {
            vec![]
        };

        let limit = if self.match_keyword(Keyword::Limit) {
            Some(self.parse_expression()?)
        } else {
            None
        };

        Ok(Statement::Select(Select {
            columns,
            from,
            r#where: where_clause,
            group_by,
            order_by,
            limit,
        }))
    }

    fn parse_select_columns(&mut self) -> Result<Vec<SelectColumn>> {
        let mut columns = Vec::new();

        if matches!(self.peek(), Some(TokenKind::Asterisk)) {
            self.advance()?;
            columns.push(SelectColumn::AllColumns);
            if !self.match_token(&TokenKind::Comma) {
                return Ok(columns);
            }
        }

        if columns.is_empty() {
            columns.push(self.parse_select_column()?);
        } else {
            columns.push(self.parse_select_column()?);
        }

        while self.match_token(&TokenKind::Comma) {
            columns.push(self.parse_select_column()?);
        }

        Ok(columns)
    }

    fn parse_select_column(&mut self) -> Result<SelectColumn> {
        let expr = self.parse_expression()?;
        let alias = if self.match_keyword(Keyword::As) {
            Some(self.expect_identifier()?)
        } else {
            None
        };
        Ok(SelectColumn::Expression { expr, alias })
    }

    fn parse_from_clause(&mut self) -> Result<FromClause> {
        let table = self.parse_table_ref()?;
        let mut joins = Vec::new();

        loop {
            let join_type = if self.match_keyword(Keyword::Join) {
                Some(JoinType::Inner)
            } else if self.match_keyword(Keyword::Inner) {
                self.expect_keyword(Keyword::Join)?;
                Some(JoinType::Inner)
            } else if self.match_keyword(Keyword::Left) {
                self.expect_keyword(Keyword::Join)?;
                Some(JoinType::Left)
            } else if self.match_keyword(Keyword::Right) {
                self.expect_keyword(Keyword::Join)?;
                Some(JoinType::Right)
            } else if self.match_keyword(Keyword::Cross) {
                self.expect_keyword(Keyword::Join)?;
                Some(JoinType::Cross)
            } else {
                None
            };

            match join_type {
                Some(jt) => {
                    let join_table = self.parse_table_ref()?;
                    let on = if jt == JoinType::Cross {
                        if self.match_keyword(Keyword::On) {
                            self.parse_expression()?
                        } else {
                            Expression::Literal(Literal::Boolean(true))
                        }
                    } else {
                        self.expect_keyword(Keyword::On)?;
                        self.parse_expression()?
                    };
                    joins.push(Join {
                        join_type: jt,
                        table: join_table,
                        on,
                    });
                }
                None => break,
            }
        }

        Ok(FromClause { table, joins })
    }

    fn parse_table_ref(&mut self) -> Result<TableRef> {
        let name = self.expect_identifier()?;
        let alias = if self.match_keyword(Keyword::As) {
            Some(self.expect_identifier()?)
        } else {
            match self.peek() {
                Some(TokenKind::Identifier(_)) => {
                    Some(self.expect_identifier()?)
                }
                _ => None,
            }
        };
        Ok(TableRef { name, alias })
    }

    fn parse_expression_list(&mut self) -> Result<Vec<Expression>> {
        let mut exprs = vec![self.parse_expression()?];
        while self.match_token(&TokenKind::Comma) {
            exprs.push(self.parse_expression()?);
        }
        Ok(exprs)
    }

    fn parse_order_by_list(&mut self) -> Result<Vec<OrderByClause>> {
        let mut clauses = vec![self.parse_order_by_item()?];
        while self.match_token(&TokenKind::Comma) {
            clauses.push(self.parse_order_by_item()?);
        }
        Ok(clauses)
    }

    fn parse_order_by_item(&mut self) -> Result<OrderByClause> {
        let expr = self.parse_expression()?;
        let direction = if self.match_keyword(Keyword::Desc) {
            SortDirection::Descending
        } else {
            self.match_keyword(Keyword::Asc);
            SortDirection::Ascending
        };
        Ok(OrderByClause { expr, direction })
    }

    // ─── CREATE TABLE ───

    fn parse_create_table(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Create)?;
        self.expect_keyword(Keyword::Table)?;
        let name = self.expect_identifier()?;
        self.expect_token(&TokenKind::LeftParen)?;
        let columns = self.parse_column_defs()?;
        self.expect_token(&TokenKind::RightParen)?;
        Ok(Statement::CreateTable(CreateTable { name, columns }))
    }

    fn parse_column_defs(&mut self) -> Result<Vec<ColumnDef>> {
        let mut cols = vec![self.parse_column_def()?];
        while self.match_token(&TokenKind::Comma) {
            cols.push(self.parse_column_def()?);
        }
        Ok(cols)
    }

    fn parse_column_def(&mut self) -> Result<ColumnDef> {
        let name = self.expect_identifier()?;
        let data_type = self.parse_data_type()?;
        let mut nullable = true;
        let mut primary_key = false;

        loop {
            if self.match_keyword(Keyword::Primary) {
                self.expect_keyword(Keyword::Key)?;
                primary_key = true;
            } else if self.match_keyword(Keyword::Not) {
                self.expect_keyword(Keyword::Null)?;
                nullable = false;
            } else if self.match_keyword(Keyword::Null) {
                nullable = true;
            } else {
                break;
            }
        }

        Ok(ColumnDef {
            name,
            data_type,
            nullable,
            primary_key,
        })
    }

    fn parse_data_type(&mut self) -> Result<DataType> {
        match self.peek() {
            Some(TokenKind::Keyword(Keyword::Integer)) => {
                self.advance()?;
                Ok(DataType::Integer)
            }
            Some(TokenKind::Keyword(Keyword::Float)) => {
                self.advance()?;
                Ok(DataType::Float)
            }
            Some(TokenKind::Keyword(Keyword::Text)) => {
                self.advance()?;
                Ok(DataType::Text)
            }
            Some(TokenKind::Keyword(Keyword::Boolean)) => {
                self.advance()?;
                Ok(DataType::Boolean)
            }
            Some(other) => Err(Error::Parse(format!(
                "Expected data type, got {:?}", other
            ))),
            None => Err(Error::Parse(
                "Expected data type, got end of input".to_string(),
            )),
        }
    }

    // ─── INSERT ───

    fn parse_insert(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Insert)?;
        self.expect_keyword(Keyword::Into)?;
        let table = self.expect_identifier()?;

        let columns = if self.match_token(&TokenKind::LeftParen) {
            let cols = self.parse_identifier_list()?;
            self.expect_token(&TokenKind::RightParen)?;
            Some(cols)
        } else {
            None
        };

        self.expect_keyword(Keyword::Values)?;

        let mut values = vec![self.parse_value_row()?];
        while self.match_token(&TokenKind::Comma) {
            values.push(self.parse_value_row()?);
        }

        Ok(Statement::Insert(Insert {
            table,
            columns,
            values,
        }))
    }

    fn parse_identifier_list(&mut self) -> Result<Vec<String>> {
        let mut ids = vec![self.expect_identifier()?];
        while self.match_token(&TokenKind::Comma) {
            ids.push(self.expect_identifier()?);
        }
        Ok(ids)
    }

    fn parse_value_row(&mut self) -> Result<Vec<Expression>> {
        self.expect_token(&TokenKind::LeftParen)?;
        let exprs = self.parse_expression_list()?;
        self.expect_token(&TokenKind::RightParen)?;
        Ok(exprs)
    }

    // ─── DELETE ───

    fn parse_delete(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Delete)?;
        self.expect_keyword(Keyword::From)?;
        let table = self.expect_identifier()?;
        let where_clause = if self.match_keyword(Keyword::Where) {
            Some(self.parse_expression()?)
        } else {
            None
        };
        Ok(Statement::Delete(Delete {
            table,
            r#where: where_clause,
        }))
    }

    // ─── UPDATE ───

    fn parse_update(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Update)?;
        let table = self.expect_identifier()?;
        self.expect_keyword(Keyword::Set)?;
        let assignments = self.parse_assignments()?;
        let where_clause = if self.match_keyword(Keyword::Where) {
            Some(self.parse_expression()?)
        } else {
            None
        };
        Ok(Statement::Update(Update {
            table,
            assignments,
            r#where: where_clause,
        }))
    }

    fn parse_assignments(&mut self) -> Result<Vec<Assignment>> {
        let mut assignments = vec![self.parse_assignment()?];
        while self.match_token(&TokenKind::Comma) {
            assignments.push(self.parse_assignment()?);
        }
        Ok(assignments)
    }

    fn parse_assignment(&mut self) -> Result<Assignment> {
        let column = self.expect_identifier()?;
        self.expect_token(&TokenKind::Equals)?;
        let value = self.parse_expression()?;
        Ok(Assignment { column, value })
    }

    // ─── Expression parsing with precedence ───
    // OR < AND < NOT < comparison < addition < multiplication < unary < primary

    fn parse_expression(&mut self) -> Result<Expression> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expression> {
        let mut left = self.parse_and()?;
        while self.match_keyword(Keyword::Or) {
            let right = self.parse_and()?;
            left = Expression::BinaryOp {
                left: Box::new(left),
                op: BinaryOperator::Or,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expression> {
        let mut left = self.parse_not()?;
        while self.match_keyword(Keyword::And) {
            let right = self.parse_not()?;
            left = Expression::BinaryOp {
                left: Box::new(left),
                op: BinaryOperator::And,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> Result<Expression> {
        if self.match_keyword(Keyword::Not) {
            let operand = self.parse_not()?;
            Ok(Expression::UnaryOp {
                op: UnaryOperator::Not,
                operand: Box::new(operand),
            })
        } else {
            self.parse_comparison()
        }
    }

    fn parse_comparison(&mut self) -> Result<Expression> {
        let left = self.parse_addition()?;
        let op = match self.peek() {
            Some(TokenKind::Equals) => Some(BinaryOperator::Equal),
            Some(TokenKind::NotEquals) => Some(BinaryOperator::NotEqual),
            Some(TokenKind::LessThan) => Some(BinaryOperator::LessThan),
            Some(TokenKind::GreaterThan) => Some(BinaryOperator::GreaterThan),
            Some(TokenKind::LessEqual) => Some(BinaryOperator::LessEqual),
            Some(TokenKind::GreaterEqual) => Some(BinaryOperator::GreaterEqual),
            _ => None,
        };
        if let Some(op) = op {
            self.advance()?;
            let right = self.parse_addition()?;
            Ok(Expression::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            })
        } else {
            Ok(left)
        }
    }

    fn parse_addition(&mut self) -> Result<Expression> {
        let mut left = self.parse_multiplication()?;
        loop {
            let op = match self.peek() {
                Some(TokenKind::Plus) => Some(BinaryOperator::Add),
                Some(TokenKind::Minus) => Some(BinaryOperator::Subtract),
                _ => None,
            };
            if let Some(op) = op {
                self.advance()?;
                let right = self.parse_multiplication()?;
                left = Expression::BinaryOp {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn parse_multiplication(&mut self) -> Result<Expression> {
        let mut left = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                Some(TokenKind::Asterisk) => Some(BinaryOperator::Multiply),
                Some(TokenKind::Slash) => Some(BinaryOperator::Divide),
                _ => None,
            };
            if let Some(op) = op {
                self.advance()?;
                let right = self.parse_unary()?;
                left = Expression::BinaryOp {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expression> {
        if self.match_token(&TokenKind::Minus) {
            let operand = self.parse_unary()?;
            Ok(Expression::UnaryOp {
                op: UnaryOperator::Negate,
                operand: Box::new(operand),
            })
        } else if self.match_keyword(Keyword::Not) {
            let operand = self.parse_unary()?;
            Ok(Expression::UnaryOp {
                op: UnaryOperator::Not,
                operand: Box::new(operand),
            })
        } else {
            self.parse_primary()
        }
    }

    fn parse_primary(&mut self) -> Result<Expression> {
        match self.peek().cloned() {
            Some(TokenKind::Integer(n)) => {
                self.advance()?;
                Ok(Expression::Literal(Literal::Integer(n)))
            }
            Some(TokenKind::Float(f)) => {
                self.advance()?;
                Ok(Expression::Literal(Literal::Float(f)))
            }
            Some(TokenKind::String(s)) => {
                self.advance()?;
                Ok(Expression::Literal(Literal::String(s)))
            }
            Some(TokenKind::Keyword(Keyword::True)) => {
                self.advance()?;
                Ok(Expression::Literal(Literal::Boolean(true)))
            }
            Some(TokenKind::Keyword(Keyword::False)) => {
                self.advance()?;
                Ok(Expression::Literal(Literal::Boolean(false)))
            }
            Some(TokenKind::Keyword(Keyword::Null)) => {
                self.advance()?;
                Ok(Expression::Literal(Literal::Null))
            }
            Some(TokenKind::Keyword(kw))
                if matches!(
                    kw,
                    Keyword::Count | Keyword::Sum | Keyword::Avg | Keyword::Min | Keyword::Max
                ) =>
            {
                self.parse_aggregate()
            }
            Some(TokenKind::LeftParen) => {
                self.advance()?;
                let expr = self.parse_expression()?;
                self.expect_token(&TokenKind::RightParen)?;
                Ok(expr)
            }
            Some(TokenKind::Identifier(_)) => self.parse_column_ref(),
            Some(other) => Err(Error::Parse(format!(
                "Expected expression, got {:?}", other
            ))),
            None => Err(Error::Parse(
                "Expected expression, got end of input".to_string(),
            )),
        }
    }

    fn parse_aggregate(&mut self) -> Result<Expression> {
        let func = match self.peek() {
            Some(TokenKind::Keyword(Keyword::Count)) => AggregateFunction::Count,
            Some(TokenKind::Keyword(Keyword::Sum)) => AggregateFunction::Sum,
            Some(TokenKind::Keyword(Keyword::Avg)) => AggregateFunction::Avg,
            Some(TokenKind::Keyword(Keyword::Min)) => AggregateFunction::Min,
            Some(TokenKind::Keyword(Keyword::Max)) => AggregateFunction::Max,
            _ => unreachable!(),
        };
        self.advance()?;
        self.expect_token(&TokenKind::LeftParen)?;

        let arg = if func == AggregateFunction::Count && self.match_token(&TokenKind::Asterisk) {
            None
        } else {
            Some(Box::new(self.parse_expression()?))
        };

        self.expect_token(&TokenKind::RightParen)?;
        Ok(Expression::Aggregate { func, arg })
    }

    fn parse_column_ref(&mut self) -> Result<Expression> {
        let name = self.expect_identifier()?;
        if self.match_token(&TokenKind::Dot) {
            let column = self.expect_identifier()?;
            Ok(Expression::ColumnRef {
                table: Some(name),
                column,
            })
        } else {
            Ok(Expression::ColumnRef {
                table: None,
                column: name,
            })
        }
    }
}

// ───────────────────────── Public API ─────────────────────────

/// Parse a SQL string into a `Statement`.
pub fn parse(input: &str) -> Result<Statement> {
    let tokens = tokenize(input)?;
    if tokens.is_empty() {
        return Err(Error::Parse("Empty input".to_string()));
    }
    let mut parser = Parser::new(tokens);
    let stmt = parser.parse_statement()?;

    // Skip optional trailing semicolon
    if matches!(parser.peek(), Some(TokenKind::Semicolon)) {
        parser.advance()?;
    }

    if !parser.at_end() {
        return Err(Error::Parse(format!(
            "Unexpected token after statement: {:?}",
            parser.peek()
        )));
    }

    Ok(stmt)
}

// ───────────────────────── Tests ─────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn col(name: &str) -> Expression {
        Expression::ColumnRef {
            table: None,
            column: name.to_string(),
        }
    }

    fn tcol(table: &str, column: &str) -> Expression {
        Expression::ColumnRef {
            table: Some(table.to_string()),
            column: column.to_string(),
        }
    }

    fn int(n: i64) -> Expression {
        Expression::Literal(Literal::Integer(n))
    }

    fn float(f: f64) -> Expression {
        Expression::Literal(Literal::Float(f))
    }

    fn str_lit(s: &str) -> Expression {
        Expression::Literal(Literal::String(s.to_string()))
    }

    fn bool_lit(b: bool) -> Expression {
        Expression::Literal(Literal::Boolean(b))
    }

    fn null_lit() -> Expression {
        Expression::Literal(Literal::Null)
    }

    fn binop(left: Expression, op: BinaryOperator, right: Expression) -> Expression {
        Expression::BinaryOp {
            left: Box::new(left),
            op,
            right: Box::new(right),
        }
    }

    // ─── SELECT tests ───

    #[test]
    fn test_select_simple() {
        let stmt = parse("SELECT a, b FROM t WHERE x > 1 AND y = 'hello'").unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.columns.len(), 2);
                assert_eq!(s.from.as_ref().unwrap().table.name, "t");
                let expected_where = binop(
                    binop(col("x"), BinaryOperator::GreaterThan, int(1)),
                    BinaryOperator::And,
                    binop(col("y"), BinaryOperator::Equal, str_lit("hello")),
                );
                assert_eq!(s.r#where, Some(expected_where));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_star() {
        let stmt = parse("SELECT * FROM users").unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.columns, vec![SelectColumn::AllColumns]);
                assert_eq!(s.from.as_ref().unwrap().table.name, "users");
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_aggregates_group_by() {
        let stmt = parse("SELECT COUNT(*), SUM(x) FROM t GROUP BY y").unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.columns.len(), 2);
                match &s.columns[0] {
                    SelectColumn::Expression { expr, alias } => {
                        assert_eq!(*expr, Expression::Aggregate {
                            func: AggregateFunction::Count,
                            arg: None,
                        });
                        assert_eq!(*alias, None);
                    }
                    _ => panic!("Expected expression"),
                }
                match &s.columns[1] {
                    SelectColumn::Expression { expr, alias } => {
                        assert_eq!(*expr, Expression::Aggregate {
                            func: AggregateFunction::Sum,
                            arg: Some(Box::new(col("x"))),
                        });
                        assert_eq!(*alias, None);
                    }
                    _ => panic!("Expected expression"),
                }
                assert_eq!(s.group_by, vec![col("y")]);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_join() {
        let stmt = parse("SELECT * FROM t1 JOIN t2 ON t1.id = t2.id WHERE t1.x > 0").unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.columns, vec![SelectColumn::AllColumns]);
                let from = s.from.as_ref().unwrap();
                assert_eq!(from.table.name, "t1");
                assert_eq!(from.joins.len(), 1);
                assert_eq!(from.joins[0].join_type, JoinType::Inner);
                assert_eq!(from.joins[0].table.name, "t2");
                let expected_on = binop(tcol("t1", "id"), BinaryOperator::Equal, tcol("t2", "id"));
                assert_eq!(from.joins[0].on, expected_on);
                let expected_where = binop(tcol("t1", "x"), BinaryOperator::GreaterThan, int(0));
                assert_eq!(s.r#where, Some(expected_where));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_inner_join() {
        let stmt = parse("SELECT a FROM t1 INNER JOIN t2 ON t1.id = t2.id").unwrap();
        match stmt {
            Statement::Select(s) => {
                let from = s.from.as_ref().unwrap();
                assert_eq!(from.joins[0].join_type, JoinType::Inner);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_left_join() {
        let stmt = parse("SELECT a FROM t1 LEFT JOIN t2 ON t1.id = t2.id").unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.from.as_ref().unwrap().joins[0].join_type, JoinType::Left);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_order_by_asc_desc() {
        let stmt = parse("SELECT a, b FROM t ORDER BY a ASC, b DESC LIMIT 10").unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.order_by.len(), 2);
                assert_eq!(s.order_by[0].direction, SortDirection::Ascending);
                assert_eq!(s.order_by[0].expr, col("a"));
                assert_eq!(s.order_by[1].direction, SortDirection::Descending);
                assert_eq!(s.order_by[1].expr, col("b"));
                assert_eq!(s.limit, Some(int(10)));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_order_by_default_asc() {
        let stmt = parse("SELECT a FROM t ORDER BY a").unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.order_by[0].direction, SortDirection::Ascending);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_with_alias() {
        let stmt = parse("SELECT a AS x FROM t").unwrap();
        match stmt {
            Statement::Select(s) => {
                match &s.columns[0] {
                    SelectColumn::Expression { expr, alias } => {
                        assert_eq!(*expr, col("a"));
                        assert_eq!(alias.as_deref(), Some("x"));
                    }
                    _ => panic!("Expected expression"),
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_table_alias() {
        let stmt = parse("SELECT u.name FROM users u").unwrap();
        match stmt {
            Statement::Select(s) => {
                let from = s.from.as_ref().unwrap();
                assert_eq!(from.table.name, "users");
                assert_eq!(from.table.alias.as_deref(), Some("u"));
                match &s.columns[0] {
                    SelectColumn::Expression { expr, .. } => {
                        assert_eq!(*expr, tcol("u", "name"));
                    }
                    _ => panic!("Expected expression"),
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_table_alias_with_as() {
        let stmt = parse("SELECT u.name FROM users AS u").unwrap();
        match stmt {
            Statement::Select(s) => {
                let from = s.from.as_ref().unwrap();
                assert_eq!(from.table.name, "users");
                assert_eq!(from.table.alias.as_deref(), Some("u"));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    // ─── CREATE TABLE tests ───

    #[test]
    fn test_create_table() {
        let stmt = parse("CREATE TABLE users (id INTEGER, name TEXT, active BOOLEAN)").unwrap();
        match stmt {
            Statement::CreateTable(ct) => {
                assert_eq!(ct.name, "users");
                assert_eq!(ct.columns.len(), 3);
                assert_eq!(ct.columns[0].name, "id");
                assert_eq!(ct.columns[0].data_type, DataType::Integer);
                assert!(ct.columns[0].nullable);
                assert!(!ct.columns[0].primary_key);
                assert_eq!(ct.columns[1].name, "name");
                assert_eq!(ct.columns[1].data_type, DataType::Text);
                assert_eq!(ct.columns[2].name, "active");
                assert_eq!(ct.columns[2].data_type, DataType::Boolean);
            }
            _ => panic!("Expected CREATE TABLE"),
        }
    }

    #[test]
    fn test_create_table_with_constraints() {
        let stmt = parse(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, bio TEXT NULL)",
        ).unwrap();
        match stmt {
            Statement::CreateTable(ct) => {
                assert!(ct.columns[0].primary_key);
                assert!(!ct.columns[1].nullable);
                assert!(ct.columns[2].nullable);
            }
            _ => panic!("Expected CREATE TABLE"),
        }
    }

    #[test]
    fn test_create_table_float_column() {
        let stmt = parse("CREATE TABLE data (val FLOAT)").unwrap();
        match stmt {
            Statement::CreateTable(ct) => {
                assert_eq!(ct.columns[0].data_type, DataType::Float);
            }
            _ => panic!("Expected CREATE TABLE"),
        }
    }

    // ─── INSERT tests ───

    #[test]
    fn test_insert_with_columns() {
        let stmt = parse("INSERT INTO t (a, b) VALUES (1, 'hello')").unwrap();
        match stmt {
            Statement::Insert(i) => {
                assert_eq!(i.table, "t");
                assert_eq!(i.columns, Some(vec!["a".to_string(), "b".to_string()]));
                assert_eq!(i.values.len(), 1);
                assert_eq!(i.values[0], vec![int(1), str_lit("hello")]);
            }
            _ => panic!("Expected INSERT"),
        }
    }

    #[test]
    fn test_insert_without_columns() {
        let stmt = parse("INSERT INTO t VALUES (1, 'hello')").unwrap();
        match stmt {
            Statement::Insert(i) => {
                assert_eq!(i.table, "t");
                assert!(i.columns.is_none());
                assert_eq!(i.values[0], vec![int(1), str_lit("hello")]);
            }
            _ => panic!("Expected INSERT"),
        }
    }

    #[test]
    fn test_insert_multiple_rows() {
        let stmt = parse("INSERT INTO t VALUES (1, 'a'), (2, 'b')").unwrap();
        match stmt {
            Statement::Insert(i) => {
                assert_eq!(i.values.len(), 2);
                assert_eq!(i.values[0], vec![int(1), str_lit("a")]);
                assert_eq!(i.values[1], vec![int(2), str_lit("b")]);
            }
            _ => panic!("Expected INSERT"),
        }
    }

    #[test]
    fn test_insert_with_null_and_bool() {
        let stmt = parse("INSERT INTO t VALUES (NULL, TRUE, FALSE)").unwrap();
        match stmt {
            Statement::Insert(i) => {
                assert_eq!(i.values[0], vec![null_lit(), bool_lit(true), bool_lit(false)]);
            }
            _ => panic!("Expected INSERT"),
        }
    }

    // ─── DELETE tests ───

    #[test]
    fn test_delete_with_where() {
        let stmt = parse("DELETE FROM t WHERE id = 5").unwrap();
        match stmt {
            Statement::Delete(d) => {
                assert_eq!(d.table, "t");
                let expected = binop(col("id"), BinaryOperator::Equal, int(5));
                assert_eq!(d.r#where, Some(expected));
            }
            _ => panic!("Expected DELETE"),
        }
    }

    #[test]
    fn test_delete_without_where() {
        let stmt = parse("DELETE FROM t").unwrap();
        match stmt {
            Statement::Delete(d) => {
                assert_eq!(d.table, "t");
                assert!(d.r#where.is_none());
            }
            _ => panic!("Expected DELETE"),
        }
    }

    // ─── UPDATE tests ───

    #[test]
    fn test_update_with_where() {
        let stmt = parse("UPDATE t SET x = 1, y = 'hi' WHERE id = 5").unwrap();
        match stmt {
            Statement::Update(u) => {
                assert_eq!(u.table, "t");
                assert_eq!(u.assignments.len(), 2);
                assert_eq!(u.assignments[0].column, "x");
                assert_eq!(u.assignments[0].value, int(1));
                assert_eq!(u.assignments[1].column, "y");
                assert_eq!(u.assignments[1].value, str_lit("hi"));
                let expected_where = binop(col("id"), BinaryOperator::Equal, int(5));
                assert_eq!(u.r#where, Some(expected_where));
            }
            _ => panic!("Expected UPDATE"),
        }
    }

    #[test]
    fn test_update_without_where() {
        let stmt = parse("UPDATE t SET x = 1").unwrap();
        match stmt {
            Statement::Update(u) => {
                assert!(u.r#where.is_none());
            }
            _ => panic!("Expected UPDATE"),
        }
    }

    // ─── Transaction tests ───

    #[test]
    fn test_begin() {
        assert_eq!(parse("BEGIN").unwrap(), Statement::Begin);
    }

    #[test]
    fn test_commit() {
        assert_eq!(parse("COMMIT").unwrap(), Statement::Commit);
    }

    #[test]
    fn test_rollback() {
        assert_eq!(parse("ROLLBACK").unwrap(), Statement::Rollback);
    }

    #[test]
    fn test_begin_with_semicolon() {
        assert_eq!(parse("BEGIN;").unwrap(), Statement::Begin);
    }

    // ─── Expression precedence tests ───

    #[test]
    fn test_precedence_and_or() {
        let stmt = parse("SELECT a FROM t WHERE a OR b AND c").unwrap();
        match stmt {
            Statement::Select(s) => {
                let w = s.r#where.unwrap();
                match w {
                    Expression::BinaryOp { op, right, .. } => {
                        assert_eq!(op, BinaryOperator::Or);
                        match *right {
                            Expression::BinaryOp { op, .. } => {
                                assert_eq!(op, BinaryOperator::And);
                            }
                            _ => panic!("Expected AND"),
                        }
                    }
                    _ => panic!("Expected OR"),
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_precedence_arithmetic() {
        let stmt = parse("SELECT a + b * c FROM t").unwrap();
        match stmt {
            Statement::Select(s) => {
                match &s.columns[0] {
                    SelectColumn::Expression { expr, .. } => match expr {
                        Expression::BinaryOp { op, right, .. } => {
                            assert_eq!(*op, BinaryOperator::Add);
                            match right.as_ref() {
                                Expression::BinaryOp { op, .. } => {
                                    assert_eq!(*op, BinaryOperator::Multiply);
                                }
                                _ => panic!("Expected Multiply"),
                            }
                        }
                        _ => panic!("Expected BinaryOp"),
                    },
                    _ => panic!("Expected expression"),
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_precedence_comparison_vs_arithmetic() {
        let stmt = parse("SELECT x FROM t WHERE a + 1 > b * 2").unwrap();
        match stmt {
            Statement::Select(s) => {
                match s.r#where.unwrap() {
                    Expression::BinaryOp { left, op, right } => {
                        assert_eq!(op, BinaryOperator::GreaterThan);
                        assert!(matches!(*left, Expression::BinaryOp { op: BinaryOperator::Add, .. }));
                        assert!(matches!(*right, Expression::BinaryOp { op: BinaryOperator::Multiply, .. }));
                    }
                    _ => panic!("Expected BinaryOp"),
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_unary_negate() {
        let stmt = parse("SELECT -5 FROM t").unwrap();
        match stmt {
            Statement::Select(s) => match &s.columns[0] {
                SelectColumn::Expression { expr, .. } => {
                    assert_eq!(*expr, Expression::UnaryOp {
                        op: UnaryOperator::Negate,
                        operand: Box::new(int(5)),
                    });
                }
                _ => panic!("Expected expression"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_unary_not() {
        let stmt = parse("SELECT x FROM t WHERE NOT active").unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.r#where.unwrap(), Expression::UnaryOp {
                    op: UnaryOperator::Not,
                    operand: Box::new(col("active")),
                });
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parenthesized_expression() {
        let stmt = parse("SELECT (a + b) * c FROM t").unwrap();
        match stmt {
            Statement::Select(s) => match &s.columns[0] {
                SelectColumn::Expression { expr, .. } => match expr {
                    Expression::BinaryOp { left, op, .. } => {
                        assert_eq!(*op, BinaryOperator::Multiply);
                        assert!(matches!(left.as_ref(), Expression::BinaryOp { op: BinaryOperator::Add, .. }));
                    }
                    _ => panic!("Expected BinaryOp"),
                },
                _ => panic!("Expected expression"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_complex_where_and_or() {
        let stmt = parse("SELECT x FROM t WHERE a = 1 AND b = 2 OR c = 3").unwrap();
        match stmt {
            Statement::Select(s) => {
                match s.r#where.unwrap() {
                    Expression::BinaryOp { op, left, .. } => {
                        assert_eq!(op, BinaryOperator::Or);
                        assert!(matches!(*left, Expression::BinaryOp { op: BinaryOperator::And, .. }));
                    }
                    _ => panic!("Expected OR"),
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_aggregate_avg() {
        let stmt = parse("SELECT AVG(price) FROM products").unwrap();
        match stmt {
            Statement::Select(s) => match &s.columns[0] {
                SelectColumn::Expression { expr, .. } => {
                    assert_eq!(*expr, Expression::Aggregate {
                        func: AggregateFunction::Avg,
                        arg: Some(Box::new(col("price"))),
                    });
                }
                _ => panic!("Expected expression"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_aggregate_min_max() {
        let stmt = parse("SELECT MIN(a), MAX(b) FROM t").unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.columns.len(), 2);
                assert!(matches!(&s.columns[0], SelectColumn::Expression { expr: Expression::Aggregate { func: AggregateFunction::Min, .. }, .. }));
                assert!(matches!(&s.columns[1], SelectColumn::Expression { expr: Expression::Aggregate { func: AggregateFunction::Max, .. }, .. }));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_qualified_columns() {
        let stmt = parse("SELECT t1.a, t2.b FROM t1 JOIN t2 ON t1.id = t2.id").unwrap();
        match stmt {
            Statement::Select(s) => {
                match &s.columns[0] {
                    SelectColumn::Expression { expr, .. } => assert_eq!(*expr, tcol("t1", "a")),
                    _ => panic!("Expected expression"),
                }
                match &s.columns[1] {
                    SelectColumn::Expression { expr, .. } => assert_eq!(*expr, tcol("t2", "b")),
                    _ => panic!("Expected expression"),
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_insert_with_float() {
        let stmt = parse("INSERT INTO t VALUES (3.14)").unwrap();
        match stmt {
            Statement::Insert(i) => {
                assert_eq!(i.values[0], vec![float(3.14)]);
            }
            _ => panic!("Expected INSERT"),
        }
    }

    #[test]
    fn test_select_all_comparison_operators() {
        for (op_str, op) in [
            ("=", BinaryOperator::Equal),
            ("!=", BinaryOperator::NotEqual),
            ("<", BinaryOperator::LessThan),
            (">", BinaryOperator::GreaterThan),
            ("<=", BinaryOperator::LessEqual),
            (">=", BinaryOperator::GreaterEqual),
        ] {
            let sql = format!("SELECT x FROM t WHERE a {} 1", op_str);
            let stmt = parse(&sql).unwrap();
            match stmt {
                Statement::Select(s) => match s.r#where.unwrap() {
                    Expression::BinaryOp { op: actual_op, .. } => {
                        assert_eq!(actual_op, op, "Failed for operator {}", op_str);
                    }
                    _ => panic!("Expected BinaryOp for {}", op_str),
                },
                _ => panic!("Expected SELECT"),
            }
        }
    }

    // ─── Error cases ───

    #[test]
    fn test_error_empty_input() {
        assert!(parse("").is_err());
    }

    #[test]
    fn test_error_select_from_no_columns() {
        let result = parse("SELECT FROM t");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_insert_hello() {
        let result = parse("INSERT hello");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_create_table_no_columns() {
        let result = parse("CREATE TABLE t ()");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_trailing_tokens() {
        let result = parse("SELECT a FROM t extra1 extra2");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_missing_from_in_delete() {
        let result = parse("DELETE t WHERE id = 1");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_missing_set_in_update() {
        let result = parse("UPDATE t WHERE id = 1");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_missing_values_in_insert() {
        let result = parse("INSERT INTO t (a, b)");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_invalid_data_type() {
        let result = parse("CREATE TABLE t (a BIGINT)");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_unclosed_paren() {
        let result = parse("SELECT (a + b FROM t");
        assert!(result.is_err());
    }

    #[test]
    fn test_select_with_semicolon() {
        let stmt = parse("SELECT a FROM t;").unwrap();
        assert!(matches!(stmt, Statement::Select(_)));
    }

    #[test]
    fn test_select_multiple_aggregates_group_by() {
        let stmt = parse(
            "SELECT department, COUNT(*), AVG(salary) FROM employees GROUP BY department",
        ).unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.columns.len(), 3);
                assert_eq!(s.group_by.len(), 1);
                assert_eq!(s.group_by[0], col("department"));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_cross_join() {
        let stmt = parse("SELECT a FROM t1 CROSS JOIN t2").unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.from.as_ref().unwrap().joins[0].join_type, JoinType::Cross);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_right_join() {
        let stmt = parse("SELECT a FROM t1 RIGHT JOIN t2 ON t1.id = t2.id").unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.from.as_ref().unwrap().joins[0].join_type, JoinType::Right);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_multiple_joins() {
        let stmt = parse(
            "SELECT a FROM t1 JOIN t2 ON t1.id = t2.id JOIN t3 ON t2.id = t3.id",
        ).unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.from.as_ref().unwrap().joins.len(), 2);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_nested_not() {
        let stmt = parse("SELECT x FROM t WHERE NOT NOT active").unwrap();
        match stmt {
            Statement::Select(s) => match s.r#where.unwrap() {
                Expression::UnaryOp { op, operand } => {
                    assert_eq!(op, UnaryOperator::Not);
                    assert!(matches!(*operand, Expression::UnaryOp { op: UnaryOperator::Not, .. }));
                }
                _ => panic!("Expected NOT"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_literal_values() {
        let stmt = parse("SELECT 1, 3.14, 'hello', TRUE, FALSE, NULL FROM t").unwrap();
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.columns.len(), 6);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_update_with_expression() {
        let stmt = parse("UPDATE t SET x = x + 1 WHERE id = 5").unwrap();
        match stmt {
            Statement::Update(u) => {
                assert_eq!(u.assignments[0].value, binop(col("x"), BinaryOperator::Add, int(1)));
            }
            _ => panic!("Expected UPDATE"),
        }
    }

    #[test]
    fn test_select_aggregate_with_alias() {
        let stmt = parse("SELECT COUNT(*) AS cnt FROM t").unwrap();
        match stmt {
            Statement::Select(s) => match &s.columns[0] {
                SelectColumn::Expression { alias, .. } => {
                    assert_eq!(alias.as_deref(), Some("cnt"));
                }
                _ => panic!("Expected expression"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_no_from() {
        let stmt = parse("SELECT 1").unwrap();
        match stmt {
            Statement::Select(s) => {
                assert!(s.from.is_none());
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_complex_expression_in_where() {
        let stmt = parse(
            "SELECT x FROM t WHERE (a > 1 AND b < 10) OR (c = 'hello' AND d != 0)",
        ).unwrap();
        match stmt {
            Statement::Select(s) => {
                assert!(matches!(s.r#where.unwrap(), Expression::BinaryOp { op: BinaryOperator::Or, .. }));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_insert_negative_value() {
        let stmt = parse("INSERT INTO t VALUES (-42)").unwrap();
        match stmt {
            Statement::Insert(i) => {
                assert_eq!(i.values[0][0], Expression::UnaryOp {
                    op: UnaryOperator::Negate,
                    operand: Box::new(int(42)),
                });
            }
            _ => panic!("Expected INSERT"),
        }
    }
}
