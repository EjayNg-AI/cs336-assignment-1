use std::fs::File;
use std::io::{copy, BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};

pub fn write_npy_header<W: Write>(writer: &mut W, token_count: u64) -> Result<()> {
    let mut header =
        format!("{{'descr': '<u2', 'fortran_order': False, 'shape': ({token_count},), }}");
    header.push('\n');
    while (10 + header.len()) % 16 != 0 {
        let newline = header.pop();
        header.push(' ');
        if newline == Some('\n') {
            header.push('\n');
        }
    }
    let header_len: u16 = header
        .len()
        .try_into()
        .context("NumPy v1 header is too large")?;

    writer.write_all(b"\x93NUMPY")?;
    writer.write_all(&[1, 0])?;
    writer.write_all(&header_len.to_le_bytes())?;
    writer.write_all(header.as_bytes())?;
    Ok(())
}

pub fn copy_raw_uint16_to_npy(raw_path: &Path, npy_path: &Path, token_count: u64) -> Result<()> {
    let expected_bytes = token_count
        .checked_mul(2)
        .context("uint16 token stream byte length overflow")?;
    let raw_len = raw_path
        .metadata()
        .with_context(|| format!("failed to stat raw token stream {}", raw_path.display()))?
        .len();
    if raw_len != expected_bytes {
        bail!(
            "raw token stream has {raw_len} bytes but expected {expected_bytes} for {token_count} uint16 tokens"
        );
    }

    let mut output = BufWriter::new(
        File::create(npy_path)
            .with_context(|| format!("failed to create NumPy output {}", npy_path.display()))?,
    );
    write_npy_header(&mut output, token_count)?;
    let mut input = BufReader::new(
        File::open(raw_path)
            .with_context(|| format!("failed to open raw token stream {}", raw_path.display()))?,
    );
    copy(&mut input, &mut output)?;
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::write_npy_header;

    #[test]
    fn header_has_numpy_magic_and_alignment() {
        let mut bytes = Vec::new();
        write_npy_header(&mut bytes, 3).unwrap();
        assert_eq!(&bytes[..6], b"\x93NUMPY");
        assert_eq!(bytes[6], 1);
        assert_eq!(bytes[7], 0);
        assert_eq!(bytes.len() % 16, 0);
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
        assert!(header.contains("'descr': '<u2'"));
        assert!(header.contains("'shape': (3,)"));
    }
}
