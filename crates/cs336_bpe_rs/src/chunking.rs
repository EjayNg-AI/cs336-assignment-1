use std::fs;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{bail, Result};

use crate::config::DEFAULT_CHUNK_BYTES;

pub fn chunk_ranges(
    input_path: &Path,
    num_workers: usize,
    chunk_bytes: Option<usize>,
    special_tokens: &[String],
) -> Result<Vec<(u64, u64)>> {
    let file_size = fs::metadata(input_path)?.len();
    if file_size == 0 {
        return Ok(vec![(0, 0)]);
    }
    if special_tokens.is_empty() {
        return Ok(vec![(0, file_size)]);
    }

    let target_chunk_bytes = chunk_bytes.unwrap_or(DEFAULT_CHUNK_BYTES);
    if target_chunk_bytes < 1 {
        bail!("chunk_bytes must be at least 1");
    }

    let desired_by_size = (file_size as f64 / target_chunk_bytes as f64).ceil() as usize;
    let desired_chunks = num_workers.max(desired_by_size).max(1);
    let mut file = File::open(input_path)?;
    let boundaries =
        find_chunk_boundaries(&mut file, desired_chunks, special_tokens[0].as_bytes())?;
    let ranges: Vec<(u64, u64)> = boundaries
        .windows(2)
        .filter_map(|window| {
            let start = window[0];
            let end = window[1];
            (end > start).then_some((start, end))
        })
        .collect();

    if ranges.is_empty() {
        Ok(vec![(0, file_size)])
    } else {
        Ok(ranges)
    }
}

pub fn find_chunk_boundaries(
    file: &mut File,
    desired_num_chunks: usize,
    split_special_token: &[u8],
) -> Result<Vec<u64>> {
    if split_special_token.is_empty() {
        bail!("split special token must not be empty");
    }

    file.seek(SeekFrom::End(0))?;
    let file_size = file.stream_position()?;
    file.seek(SeekFrom::Start(0))?;
    if file_size == 0 {
        return Ok(vec![0]);
    }

    let desired_num_chunks = desired_num_chunks.max(1).min(file_size as usize);
    let chunk_size = (file_size / desired_num_chunks as u64).max(1);
    let mut boundaries = Vec::with_capacity(desired_num_chunks + 1);
    for i in 0..=desired_num_chunks {
        boundaries.push((i as u64 * chunk_size).min(file_size));
    }
    *boundaries.last_mut().unwrap() = file_size;

    let mini_chunk_size = 4096usize;
    let mut buffer = vec![0; mini_chunk_size];
    for boundary in boundaries.iter_mut().take(desired_num_chunks).skip(1) {
        let mut position = *boundary;
        file.seek(SeekFrom::Start(position))?;
        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                *boundary = file_size;
                break;
            }

            if let Some(found_at) = find_subslice(&buffer[..bytes_read], split_special_token) {
                *boundary = position + found_at as u64;
                break;
            }
            position += bytes_read as u64;
        }
    }

    boundaries.sort_unstable();
    boundaries.dedup();
    Ok(boundaries)
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.len() > haystack.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::NamedTempFile;

    use super::find_chunk_boundaries;

    #[test]
    fn finds_special_token_aligned_boundaries() {
        let mut file = NamedTempFile::new().unwrap();
        let content = b"aaa<|endoftext|>bbb<|endoftext|>ccc";
        file.write_all(content).unwrap();
        let mut readable = file.reopen().unwrap();
        let boundaries = find_chunk_boundaries(&mut readable, 3, b"<|endoftext|>").unwrap();
        assert_eq!(boundaries.first(), Some(&0));
        assert_eq!(boundaries.last(), Some(&(content.len() as u64)));
        assert!(boundaries[1..boundaries.len() - 1]
            .iter()
            .all(|boundary| [3, 19].contains(boundary)));
    }
}
