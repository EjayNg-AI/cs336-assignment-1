use anyhow::{bail, Result};

use crate::errors::BpeError;

pub fn python_bytes_repr(bytes: &[u8]) -> String {
    let quote = if bytes.contains(&b'\'') && !bytes.contains(&b'"') {
        b'"'
    } else {
        b'\''
    };

    let mut out = String::new();
    out.push('b');
    out.push(quote as char);
    for &byte in bytes {
        match byte {
            b'\\' => out.push_str("\\\\"),
            b'\n' => out.push_str("\\n"),
            b'\t' => out.push_str("\\t"),
            b'\r' => out.push_str("\\r"),
            b if b == quote => {
                out.push('\\');
                out.push(quote as char);
            }
            0x20..=0x7e => out.push(byte as char),
            _ => out.push_str(&format!("\\x{byte:02x}")),
        }
    }
    out.push(quote as char);
    out
}

pub fn parse_python_bytes_literal(input: &str) -> Result<Vec<u8>> {
    let literal = input.trim();
    let bytes = literal.as_bytes();
    if bytes.len() < 3 || (bytes[0] != b'b' && bytes[0] != b'B') {
        return Err(BpeError::InvalidByteLiteral(literal.to_string()).into());
    }
    let quote = bytes[1];
    if quote != b'\'' && quote != b'"' {
        return Err(BpeError::InvalidByteLiteral(literal.to_string()).into());
    }
    if *bytes.last().unwrap() != quote {
        return Err(BpeError::InvalidByteLiteral(literal.to_string()).into());
    }

    let mut out = Vec::new();
    let mut i = 2;
    let end = bytes.len() - 1;
    while i < end {
        let byte = bytes[i];
        if byte != b'\\' {
            out.push(byte);
            i += 1;
            continue;
        }

        i += 1;
        if i >= end {
            bail!(BpeError::InvalidByteLiteral(literal.to_string()));
        }
        match bytes[i] {
            b'\\' => out.push(b'\\'),
            b'\'' => out.push(b'\''),
            b'"' => out.push(b'"'),
            b'n' => out.push(b'\n'),
            b't' => out.push(b'\t'),
            b'r' => out.push(b'\r'),
            b'a' => out.push(0x07),
            b'b' => out.push(0x08),
            b'f' => out.push(0x0c),
            b'v' => out.push(0x0b),
            b'x' => {
                if i + 2 >= end {
                    bail!(BpeError::InvalidByteLiteral(literal.to_string()));
                }
                let high = hex_value(bytes[i + 1])
                    .ok_or_else(|| BpeError::InvalidByteLiteral(literal.to_string()))?;
                let low = hex_value(bytes[i + 2])
                    .ok_or_else(|| BpeError::InvalidByteLiteral(literal.to_string()))?;
                out.push((high << 4) | low);
                i += 2;
            }
            other => out.push(other),
        }
        i += 1;
    }

    Ok(out)
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_python_bytes_literal, python_bytes_repr};

    #[test]
    fn python_repr_matches_common_cases() {
        assert_eq!(python_bytes_repr(&[0]), "b'\\x00'");
        assert_eq!(python_bytes_repr(b"\n\t\r"), "b'\\n\\t\\r'");
        assert_eq!(python_bytes_repr(b"'"), "b\"'\"");
        assert_eq!(python_bytes_repr(b"\""), "b'\"'");
        assert_eq!(python_bytes_repr(b"'\""), "b'\\'\"'");
        assert_eq!(python_bytes_repr("é".as_bytes()), "b'\\xc3\\xa9'");
    }

    #[test]
    fn parses_python_bytes_literals() {
        let cases = [
            b"\x00".as_slice(),
            b"\n\t\r".as_slice(),
            b"'".as_slice(),
            b"\"".as_slice(),
            b"'\"".as_slice(),
            "é".as_bytes(),
            b"\\".as_slice(),
        ];
        for case in cases {
            let repr = python_bytes_repr(case);
            assert_eq!(parse_python_bytes_literal(&repr).unwrap(), case);
        }
    }
}
