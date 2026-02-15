//! SQL Tokenizer/Lexer.
//!
//! Converts a SQL input string into a stream of [`Token`]s for the parser.

use crate::error::{Error, Result};

// ───────────────────────── Token types ─────────────────────────

/// A token with its position in the source string.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// The kind of token.
    pub kind: TokenKind,
    /// Byte offset where this token starts in the source string.
    pub offset: usize,
}

/// All possible token kinds.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // ── Keywords ──
    Keyword(Keyword),

    // ── Literals ──
    /// An identifier (table name, column name, etc.).
    Identifier(String),
    /// An integer literal.
    Integer(i64),
    /// A floating-point literal.
    Float(f64),
    /// A single-quoted string literal (contents without quotes).
    String(String),

    // ── Operators ──
    /// `=`
    Equals,
    /// `!=`
    NotEquals,
    /// `<`
    LessThan,
    /// `>`
    GreaterThan,
    /// `<=`
    LessEqual,
    /// `>=`
    GreaterEqual,
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Asterisk,
    /// `/`
    Slash,

    // ── Punctuation ──
    /// `(`
    LeftParen,
    /// `)`
    RightParen,
    /// `,`
    Comma,
    /// `;`
    Semicolon,
    /// `.`
    Dot,
}

/// SQL keywords recognized by the lexer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Keyword {
    Select,
    From,
    Where,
    Insert,
    Into,
    Values,
    Create,
    Table,
    Delete,
    Update,
    Set,
    Join,
    On,
    Group,
    By,
    Order,
    Asc,
    Desc,
    Limit,
    And,
    Or,
    Not,
    Begin,
    Commit,
    Rollback,
    Null,
    True,
    False,
    Integer,
    Float,
    Text,
    Boolean,
    Count,
    Sum,
    Avg,
    Min,
    Max,
    Inner,
    Left,
    Right,
    Cross,
    Primary,
    Key,
    As,
}

impl Keyword {
    /// Try to match an uppercase string to a keyword.
    fn from_str(s: &str) -> Option<Keyword> {
        match s {
            "SELECT" => Some(Keyword::Select),
            "FROM" => Some(Keyword::From),
            "WHERE" => Some(Keyword::Where),
            "INSERT" => Some(Keyword::Insert),
            "INTO" => Some(Keyword::Into),
            "VALUES" => Some(Keyword::Values),
            "CREATE" => Some(Keyword::Create),
            "TABLE" => Some(Keyword::Table),
            "DELETE" => Some(Keyword::Delete),
            "UPDATE" => Some(Keyword::Update),
            "SET" => Some(Keyword::Set),
            "JOIN" => Some(Keyword::Join),
            "ON" => Some(Keyword::On),
            "GROUP" => Some(Keyword::Group),
            "BY" => Some(Keyword::By),
            "ORDER" => Some(Keyword::Order),
            "ASC" => Some(Keyword::Asc),
            "DESC" => Some(Keyword::Desc),
            "LIMIT" => Some(Keyword::Limit),
            "AND" => Some(Keyword::And),
            "OR" => Some(Keyword::Or),
            "NOT" => Some(Keyword::Not),
            "BEGIN" => Some(Keyword::Begin),
            "COMMIT" => Some(Keyword::Commit),
            "ROLLBACK" => Some(Keyword::Rollback),
            "NULL" => Some(Keyword::Null),
            "TRUE" => Some(Keyword::True),
            "FALSE" => Some(Keyword::False),
            "INTEGER" => Some(Keyword::Integer),
            "FLOAT" => Some(Keyword::Float),
            "TEXT" => Some(Keyword::Text),
            "BOOLEAN" => Some(Keyword::Boolean),
            "COUNT" => Some(Keyword::Count),
            "SUM" => Some(Keyword::Sum),
            "AVG" => Some(Keyword::Avg),
            "MIN" => Some(Keyword::Min),
            "MAX" => Some(Keyword::Max),
            "INNER" => Some(Keyword::Inner),
            "LEFT" => Some(Keyword::Left),
            "RIGHT" => Some(Keyword::Right),
            "CROSS" => Some(Keyword::Cross),
            "PRIMARY" => Some(Keyword::Primary),
            "KEY" => Some(Keyword::Key),
            "AS" => Some(Keyword::As),
            _ => None,
        }
    }
}

// ───────────────────────── Lexer ─────────────────────────

/// SQL tokenizer that converts an input string into a sequence of tokens.
pub struct Lexer<'a> {
    input: &'a str,
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    /// Create a new lexer for the given input string.
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            bytes: input.as_bytes(),
            pos: 0,
        }
    }

    /// Tokenize the entire input and return all tokens.
    pub fn tokenize(&mut self) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();
        while let Some(token) = self.next_token()? {
            tokens.push(token);
        }
        Ok(tokens)
    }

    /// Return the next token, or `None` if input is exhausted.
    pub fn next_token(&mut self) -> Result<Option<Token>> {
        self.skip_whitespace();

        if self.pos >= self.bytes.len() {
            return Ok(None);
        }

        let start = self.pos;
        let ch = self.bytes[self.pos];

        let kind = match ch {
            // Single-character punctuation
            b'(' => { self.pos += 1; TokenKind::LeftParen }
            b')' => { self.pos += 1; TokenKind::RightParen }
            b',' => { self.pos += 1; TokenKind::Comma }
            b';' => { self.pos += 1; TokenKind::Semicolon }
            b'.' => { self.pos += 1; TokenKind::Dot }
            b'+' => { self.pos += 1; TokenKind::Plus }
            b'-' => { self.pos += 1; TokenKind::Minus }
            b'*' => { self.pos += 1; TokenKind::Asterisk }
            b'/' => { self.pos += 1; TokenKind::Slash }
            b'=' => { self.pos += 1; TokenKind::Equals }

            // Multi-character operators starting with `!`
            b'!' => {
                self.pos += 1;
                if self.pos < self.bytes.len() && self.bytes[self.pos] == b'=' {
                    self.pos += 1;
                    TokenKind::NotEquals
                } else {
                    return Err(Error::Parse(format!(
                        "Unexpected character '!' at position {}; expected '!='",
                        start
                    )));
                }
            }

            // `<` or `<=`
            b'<' => {
                self.pos += 1;
                if self.pos < self.bytes.len() && self.bytes[self.pos] == b'=' {
                    self.pos += 1;
                    TokenKind::LessEqual
                } else {
                    TokenKind::LessThan
                }
            }

            // `>` or `>=`
            b'>' => {
                self.pos += 1;
                if self.pos < self.bytes.len() && self.bytes[self.pos] == b'=' {
                    self.pos += 1;
                    TokenKind::GreaterEqual
                } else {
                    TokenKind::GreaterThan
                }
            }

            // String literal (single-quoted)
            b'\'' => {
                self.pos += 1; // skip opening quote
                let string_start = self.pos;
                let mut s = String::new();
                loop {
                    if self.pos >= self.bytes.len() {
                        return Err(Error::Parse(format!(
                            "Unterminated string literal starting at position {}",
                            start
                        )));
                    }
                    if self.bytes[self.pos] == b'\'' {
                        // Check for escaped quote ('')
                        if self.pos + 1 < self.bytes.len() && self.bytes[self.pos + 1] == b'\'' {
                            s.push('\'');
                            self.pos += 2;
                        } else {
                            self.pos += 1; // skip closing quote
                            break;
                        }
                    } else {
                        s.push(self.bytes[self.pos] as char);
                        self.pos += 1;
                    }
                }
                let _ = string_start; // suppress unused warning
                TokenKind::String(s)
            }

            // Number literal (integer or float)
            b'0'..=b'9' => {
                self.scan_number()?
            }

            // Identifier or keyword
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                self.scan_identifier_or_keyword()
            }

            _ => {
                return Err(Error::Parse(format!(
                    "Unexpected character '{}' at position {}",
                    ch as char, start
                )));
            }
        };

        Ok(Some(Token { kind, offset: start }))
    }

    /// Skip whitespace characters.
    fn skip_whitespace(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    /// Scan a numeric literal (integer or float).
    fn scan_number(&mut self) -> Result<TokenKind> {
        let start = self.pos;
        // Consume digits
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        // Check for decimal point
        let is_float = self.pos < self.bytes.len() && self.bytes[self.pos] == b'.';
        if is_float {
            self.pos += 1; // skip the dot
            // Consume fractional digits
            if self.pos >= self.bytes.len() || !self.bytes[self.pos].is_ascii_digit() {
                return Err(Error::Parse(format!(
                    "Expected digit after decimal point at position {}",
                    self.pos
                )));
            }
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }

        let text = &self.input[start..self.pos];
        if is_float {
            let val: f64 = text.parse().map_err(|e| {
                Error::Parse(format!("Invalid float literal '{}': {}", text, e))
            })?;
            Ok(TokenKind::Float(val))
        } else {
            let val: i64 = text.parse().map_err(|e| {
                Error::Parse(format!("Invalid integer literal '{}': {}", text, e))
            })?;
            Ok(TokenKind::Integer(val))
        }
    }

    /// Scan an identifier or keyword.
    fn scan_identifier_or_keyword(&mut self) -> TokenKind {
        let start = self.pos;
        while self.pos < self.bytes.len()
            && (self.bytes[self.pos].is_ascii_alphanumeric() || self.bytes[self.pos] == b'_')
        {
            self.pos += 1;
        }
        let text = &self.input[start..self.pos];
        let upper = text.to_uppercase();
        if let Some(kw) = Keyword::from_str(&upper) {
            TokenKind::Keyword(kw)
        } else {
            TokenKind::Identifier(text.to_string())
        }
    }
}

// ───────────────────────── Convenience function ─────────────────────────

/// Tokenize an input string into a vector of tokens.
pub fn tokenize(input: &str) -> Result<Vec<Token>> {
    Lexer::new(input).tokenize()
}

// ───────────────────────── Tests ─────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: tokenize and return just the token kinds.
    fn kinds(input: &str) -> Vec<TokenKind> {
        tokenize(input).unwrap().into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn test_select_star_from_users() {
        let tokens = kinds("SELECT * FROM users WHERE id = 1");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Keyword(Keyword::Select),
                TokenKind::Asterisk,
                TokenKind::Keyword(Keyword::From),
                TokenKind::Identifier("users".to_string()),
                TokenKind::Keyword(Keyword::Where),
                TokenKind::Identifier("id".to_string()),
                TokenKind::Equals,
                TokenKind::Integer(1),
            ]
        );
    }

    #[test]
    fn test_case_insensitive_keywords() {
        let tokens = kinds("select FROM Where");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Keyword(Keyword::Select),
                TokenKind::Keyword(Keyword::From),
                TokenKind::Keyword(Keyword::Where),
            ]
        );
    }

    #[test]
    fn test_mixed_case_keywords() {
        let tokens = kinds("SeLeCt InSeRt");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Keyword(Keyword::Select),
                TokenKind::Keyword(Keyword::Insert),
            ]
        );
    }

    #[test]
    fn test_integer_literal() {
        let tokens = kinds("42 0 12345");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Integer(42),
                TokenKind::Integer(0),
                TokenKind::Integer(12345),
            ]
        );
    }

    #[test]
    fn test_float_literal() {
        let tokens = kinds("3.14 0.5 100.0");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Float(3.14),
                TokenKind::Float(0.5),
                TokenKind::Float(100.0),
            ]
        );
    }

    #[test]
    fn test_string_literal() {
        let tokens = kinds("'hello' 'world'");
        assert_eq!(
            tokens,
            vec![
                TokenKind::String("hello".to_string()),
                TokenKind::String("world".to_string()),
            ]
        );
    }

    #[test]
    fn test_string_literal_with_escaped_quote() {
        let tokens = kinds("'it''s'");
        assert_eq!(
            tokens,
            vec![TokenKind::String("it's".to_string())]
        );
    }

    #[test]
    fn test_empty_string_literal() {
        let tokens = kinds("''");
        assert_eq!(tokens, vec![TokenKind::String("".to_string())]);
    }

    #[test]
    fn test_unterminated_string() {
        let result = tokenize("'hello");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{}", err).contains("Unterminated string"));
    }

    #[test]
    fn test_all_operators() {
        let tokens = kinds("= != < > <= >= + - * /");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Equals,
                TokenKind::NotEquals,
                TokenKind::LessThan,
                TokenKind::GreaterThan,
                TokenKind::LessEqual,
                TokenKind::GreaterEqual,
                TokenKind::Plus,
                TokenKind::Minus,
                TokenKind::Asterisk,
                TokenKind::Slash,
            ]
        );
    }

    #[test]
    fn test_punctuation() {
        let tokens = kinds("( ) , ; .");
        assert_eq!(
            tokens,
            vec![
                TokenKind::LeftParen,
                TokenKind::RightParen,
                TokenKind::Comma,
                TokenKind::Semicolon,
                TokenKind::Dot,
            ]
        );
    }

    #[test]
    fn test_all_keywords() {
        let input = "SELECT FROM WHERE INSERT INTO VALUES CREATE TABLE DELETE UPDATE \
                      SET JOIN ON GROUP BY ORDER ASC DESC LIMIT AND OR NOT BEGIN COMMIT \
                      ROLLBACK NULL TRUE FALSE INTEGER FLOAT TEXT BOOLEAN COUNT SUM AVG MIN MAX \
                      INNER LEFT RIGHT CROSS PRIMARY KEY AS";
        let tokens = kinds(input);
        let expected_keywords = vec![
            Keyword::Select, Keyword::From, Keyword::Where, Keyword::Insert,
            Keyword::Into, Keyword::Values, Keyword::Create, Keyword::Table,
            Keyword::Delete, Keyword::Update, Keyword::Set, Keyword::Join,
            Keyword::On, Keyword::Group, Keyword::By, Keyword::Order,
            Keyword::Asc, Keyword::Desc, Keyword::Limit, Keyword::And,
            Keyword::Or, Keyword::Not, Keyword::Begin, Keyword::Commit,
            Keyword::Rollback, Keyword::Null, Keyword::True, Keyword::False,
            Keyword::Integer, Keyword::Float, Keyword::Text, Keyword::Boolean,
            Keyword::Count, Keyword::Sum, Keyword::Avg, Keyword::Min, Keyword::Max,
            Keyword::Inner, Keyword::Left, Keyword::Right, Keyword::Cross,
            Keyword::Primary, Keyword::Key, Keyword::As,
        ];
        let expected: Vec<TokenKind> = expected_keywords.into_iter().map(TokenKind::Keyword).collect();
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_identifiers() {
        let tokens = kinds("my_table column1 _private");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Identifier("my_table".to_string()),
                TokenKind::Identifier("column1".to_string()),
                TokenKind::Identifier("_private".to_string()),
            ]
        );
    }

    #[test]
    fn test_create_table() {
        let input = "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)";
        let tokens = kinds(input);
        assert_eq!(
            tokens,
            vec![
                TokenKind::Keyword(Keyword::Create),
                TokenKind::Keyword(Keyword::Table),
                TokenKind::Identifier("users".to_string()),
                TokenKind::LeftParen,
                TokenKind::Identifier("id".to_string()),
                TokenKind::Keyword(Keyword::Integer),
                TokenKind::Keyword(Keyword::Primary),
                TokenKind::Keyword(Keyword::Key),
                TokenKind::Comma,
                TokenKind::Identifier("name".to_string()),
                TokenKind::Keyword(Keyword::Text),
                TokenKind::Keyword(Keyword::Not),
                TokenKind::Keyword(Keyword::Null),
                TokenKind::RightParen,
            ]
        );
    }

    #[test]
    fn test_insert_statement() {
        let input = "INSERT INTO users VALUES (1, 'Alice', 3.14, TRUE)";
        let tokens = kinds(input);
        assert_eq!(
            tokens,
            vec![
                TokenKind::Keyword(Keyword::Insert),
                TokenKind::Keyword(Keyword::Into),
                TokenKind::Identifier("users".to_string()),
                TokenKind::Keyword(Keyword::Values),
                TokenKind::LeftParen,
                TokenKind::Integer(1),
                TokenKind::Comma,
                TokenKind::String("Alice".to_string()),
                TokenKind::Comma,
                TokenKind::Float(3.14),
                TokenKind::Comma,
                TokenKind::Keyword(Keyword::True),
                TokenKind::RightParen,
            ]
        );
    }

    #[test]
    fn test_complex_select() {
        let input = "SELECT u.name, COUNT(*) FROM users u \
                      INNER JOIN orders o ON u.id = o.user_id \
                      WHERE o.total >= 100 \
                      GROUP BY u.name \
                      ORDER BY u.name ASC \
                      LIMIT 10";
        let tokens = kinds(input);
        // Just verify it tokenizes without error and has expected start/end
        assert_eq!(tokens[0], TokenKind::Keyword(Keyword::Select));
        assert_eq!(*tokens.last().unwrap(), TokenKind::Integer(10));
        // Verify some key tokens in the middle
        assert!(tokens.contains(&TokenKind::Keyword(Keyword::Inner)));
        assert!(tokens.contains(&TokenKind::Keyword(Keyword::Join)));
        assert!(tokens.contains(&TokenKind::Keyword(Keyword::Count)));
        assert!(tokens.contains(&TokenKind::GreaterEqual));
    }

    #[test]
    fn test_invalid_character() {
        let result = tokenize("SELECT @ FROM");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{}", err).contains("Unexpected character"));
    }

    #[test]
    fn test_bang_without_equals() {
        let result = tokenize("!");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{}", err).contains("!"));
    }

    #[test]
    fn test_empty_input() {
        let tokens = kinds("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_whitespace_only() {
        let tokens = kinds("   \t\n\r  ");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_position_tracking() {
        let tokens = tokenize("SELECT *").unwrap();
        assert_eq!(tokens[0].offset, 0); // SELECT at position 0
        assert_eq!(tokens[1].offset, 7); // * at position 7
    }

    #[test]
    fn test_operators_without_spaces() {
        let tokens = kinds("a<=b");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Identifier("a".to_string()),
                TokenKind::LessEqual,
                TokenKind::Identifier("b".to_string()),
            ]
        );
    }

    #[test]
    fn test_negative_number_as_separate_tokens() {
        // The lexer treats `-42` as Minus followed by Integer(42);
        // the parser can combine them.
        let tokens = kinds("-42");
        assert_eq!(
            tokens,
            vec![TokenKind::Minus, TokenKind::Integer(42)]
        );
    }

    #[test]
    fn test_string_with_spaces() {
        let tokens = kinds("'hello world'");
        assert_eq!(
            tokens,
            vec![TokenKind::String("hello world".to_string())]
        );
    }

    #[test]
    fn test_transaction_keywords() {
        let tokens = kinds("BEGIN; COMMIT; ROLLBACK;");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Keyword(Keyword::Begin),
                TokenKind::Semicolon,
                TokenKind::Keyword(Keyword::Commit),
                TokenKind::Semicolon,
                TokenKind::Keyword(Keyword::Rollback),
                TokenKind::Semicolon,
            ]
        );
    }

    #[test]
    fn test_update_statement() {
        let input = "UPDATE users SET name = 'Bob' WHERE id = 1";
        let tokens = kinds(input);
        assert_eq!(
            tokens,
            vec![
                TokenKind::Keyword(Keyword::Update),
                TokenKind::Identifier("users".to_string()),
                TokenKind::Keyword(Keyword::Set),
                TokenKind::Identifier("name".to_string()),
                TokenKind::Equals,
                TokenKind::String("Bob".to_string()),
                TokenKind::Keyword(Keyword::Where),
                TokenKind::Identifier("id".to_string()),
                TokenKind::Equals,
                TokenKind::Integer(1),
            ]
        );
    }

    #[test]
    fn test_delete_statement() {
        let input = "DELETE FROM users WHERE active = FALSE";
        let tokens = kinds(input);
        assert_eq!(
            tokens,
            vec![
                TokenKind::Keyword(Keyword::Delete),
                TokenKind::Keyword(Keyword::From),
                TokenKind::Identifier("users".to_string()),
                TokenKind::Keyword(Keyword::Where),
                TokenKind::Identifier("active".to_string()),
                TokenKind::Equals,
                TokenKind::Keyword(Keyword::False),
            ]
        );
    }

    #[test]
    fn test_dot_qualified_column() {
        let tokens = kinds("users.id");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Identifier("users".to_string()),
                TokenKind::Dot,
                TokenKind::Identifier("id".to_string()),
            ]
        );
    }

    #[test]
    fn test_float_without_fractional_digits_is_error() {
        // "42." with nothing after the dot should be an error
        let result = tokenize("42. ");
        assert!(result.is_err());
    }
}
