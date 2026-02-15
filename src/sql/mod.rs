//! SQL parser and AST definitions.

pub mod ast;
pub mod lexer;
pub mod parser;

// Re-export all AST types for convenient access.
pub use ast::*;

// Re-export the parse function for convenient access.
pub use parser::parse;