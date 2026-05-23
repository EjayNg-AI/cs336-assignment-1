pub mod artifacts;
pub mod counts;
pub mod heap;
pub mod merge;
pub mod state;

use std::collections::{HashMap, HashSet};
use std::fs;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use serde_json::json;

use crate::chunking::chunk_ranges;
use crate::config::{METADATA_FILENAME, MIN_PARALLEL_BYTES};
use crate::pretokenizer::pretoken_byte_counts;

use self::artifacts::write_training_artifacts;
use self::counts::{build_initial_pair_state, word_pair_frequencies};
use self::heap::{pop_best_pair, push_pair, rebuild_heap};
use self::merge::merge_word;
use self::state::{BytePair, Count, TokenId, TokenPair, TrainerState};

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub input_path: PathBuf,
    pub vocab_size: usize,
    pub special_tokens: Vec<String>,
    pub num_workers: Option<usize>,
    pub chunk_bytes: Option<usize>,
    pub heap_rebuild_factor: f64,
    pub output_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct TrainOutput {
    pub vocab: Vec<Vec<u8>>,
    pub merges: Vec<BytePair>,
    pub output_dir: PathBuf,
}

pub fn train_bpe(config: TrainConfig) -> Result<TrainOutput> {
    let start_time = Instant::now();
    let mut phase_durations: HashMap<String, f64> = HashMap::new();
    let resolved_num_workers = resolve_num_workers(config.num_workers)?;
    let pool = ThreadPoolBuilder::new()
        .num_threads(resolved_num_workers)
        .build()
        .context("failed to build Rayon thread pool")?;
    let input_file_bytes = fs::metadata(&config.input_path)?.len();

    let phase_start = Instant::now();
    let mut id_to_bytes: Vec<Vec<u8>> = (0u16..=255).map(|value| vec![value as u8]).collect();
    let mut vocab_values: HashSet<Vec<u8>> = id_to_bytes.iter().cloned().collect();
    for special_token in &config.special_tokens {
        let special_bytes = special_token.as_bytes().to_vec();
        if vocab_values.insert(special_bytes.clone()) {
            id_to_bytes.push(special_bytes);
        }
    }
    record_phase(&mut phase_durations, "vocab_setup", phase_start);

    let phase_start = Instant::now();
    let pretoken_counts = pretoken_counts_from_path(
        &config.input_path,
        &config.special_tokens,
        resolved_num_workers,
        config.chunk_bytes,
        input_file_bytes,
        &pool,
    )?;
    record_phase(&mut phase_durations, "pretoken_counting", phase_start);
    let unique_pretoken_count = pretoken_counts.len();
    let total_pretoken_count: Count = pretoken_counts.values().sum();

    let phase_start = Instant::now();
    let mut words: Vec<Vec<TokenId>> = Vec::with_capacity(pretoken_counts.len());
    let mut word_counts: Vec<Count> = Vec::with_capacity(pretoken_counts.len());
    for (pretoken, count) in pretoken_counts {
        words.push(pretoken.into_iter().map(TokenId::from).collect());
        word_counts.push(count);
    }
    record_phase(&mut phase_durations, "word_materialization", phase_start);

    let phase_start = Instant::now();
    let (pair_counts, pair_to_word_ids) =
        pool.install(|| build_initial_pair_state(&words, &word_counts, resolved_num_workers));
    record_phase(&mut phase_durations, "initial_pair_state", phase_start);
    let initial_pair_count = pair_counts.len();

    let phase_start = Instant::now();
    let heap = rebuild_heap(&pair_counts, &id_to_bytes);
    record_phase(&mut phase_durations, "initial_heap_build", phase_start);
    let initial_heap_size = heap.len();

    let mut state = TrainerState {
        id_to_bytes,
        words,
        word_counts,
        pair_counts,
        pair_to_word_ids,
        heap,
        merges: Vec::new(),
    };

    let merge_loop_start = Instant::now();
    let mut merge_pop_best_pair_seconds = 0.0;
    let mut merge_word_update_seconds = 0.0;
    let mut merge_heap_push_seconds = 0.0;
    let mut heap_rebuild_seconds = 0.0;
    let mut heap_rebuild_count = 0usize;

    while state.id_to_bytes.len() < config.vocab_size {
        let pop_start = Instant::now();
        let best_pair = pop_best_pair(&mut state.heap, &state.pair_counts);
        merge_pop_best_pair_seconds += pop_start.elapsed().as_secs_f64();
        let Some(best_pair) = best_pair else {
            break;
        };

        let update_start = Instant::now();
        let changed_pairs = apply_merge(&mut state, best_pair);
        merge_word_update_seconds += update_start.elapsed().as_secs_f64();

        let heap_push_start = Instant::now();
        for pair in changed_pairs {
            push_pair(
                &mut state.heap,
                &state.pair_counts,
                &state.id_to_bytes,
                pair,
            );
        }
        merge_heap_push_seconds += heap_push_start.elapsed().as_secs_f64();

        if should_rebuild_heap(
            config.heap_rebuild_factor,
            state.heap.len(),
            state.pair_counts.len(),
        ) {
            let rebuild_start = Instant::now();
            state.heap = rebuild_heap(&state.pair_counts, &state.id_to_bytes);
            heap_rebuild_seconds += rebuild_start.elapsed().as_secs_f64();
            heap_rebuild_count += 1;
        }
    }
    record_phase(&mut phase_durations, "merge_loop", merge_loop_start);

    let phase_start = Instant::now();
    let output_dir = write_training_artifacts(
        &state.id_to_bytes,
        &state.merges,
        config.output_dir.as_deref(),
        &config.input_path,
        config.vocab_size,
    )?;
    record_phase(&mut phase_durations, "artifact_writing", phase_start);
    phase_durations.insert(
        "total_training".to_string(),
        start_time.elapsed().as_secs_f64(),
    );

    let metadata = json!({
        "format": "cs336_basics.enhanced_bpe.rust.metadata.v1",
        "compatibility_target": "cs336_basics.train_bpe_enhanced",
        "input_path": config.input_path.to_string_lossy(),
        "output_dir": output_dir.to_string_lossy(),
        "requested_vocab_size": config.vocab_size,
        "vocab_size": state.id_to_bytes.len(),
        "merge_count": state.merges.len(),
        "special_tokens": config.special_tokens,
        "num_workers": resolved_num_workers,
        "chunk_bytes": config.chunk_bytes,
        "heap_rebuild_factor": config.heap_rebuild_factor,
        "input_file_bytes": input_file_bytes,
        "unique_pretoken_count": unique_pretoken_count,
        "total_pretoken_count": total_pretoken_count,
        "initial_pair_count": initial_pair_count,
        "final_pair_count": state.pair_counts.len(),
        "initial_heap_size": initial_heap_size,
        "final_heap_size": state.heap.len(),
        "heap_rebuild_count": heap_rebuild_count,
        "phase_durations_seconds": phase_durations,
        "merge_loop_subphase_durations_seconds": {
            "pop_best_pair": merge_pop_best_pair_seconds,
            "word_rewrite_and_pair_update": merge_word_update_seconds,
            "changed_pair_heap_push": merge_heap_push_seconds,
            "heap_rebuild": heap_rebuild_seconds,
        },
    });
    let mut metadata_text = serde_json::to_string_pretty(&metadata)?;
    metadata_text.push('\n');
    fs::write(output_dir.join(METADATA_FILENAME), metadata_text)?;

    Ok(TrainOutput {
        vocab: state.id_to_bytes,
        merges: state.merges,
        output_dir,
    })
}

fn resolve_num_workers(num_workers: Option<usize>) -> Result<usize> {
    match num_workers {
        Some(0) => anyhow::bail!("num_workers must be at least 1"),
        Some(value) => Ok(value),
        None => Ok(std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(8)
            .max(1)),
    }
}

fn record_phase(durations: &mut HashMap<String, f64>, name: &str, phase_start: Instant) {
    durations.insert(name.to_string(), phase_start.elapsed().as_secs_f64());
}

fn pretoken_counts_from_path(
    input_path: &Path,
    special_tokens: &[String],
    num_workers: usize,
    chunk_bytes: Option<usize>,
    input_file_bytes: u64,
    pool: &rayon::ThreadPool,
) -> Result<HashMap<Vec<u8>, Count>> {
    if num_workers == 1 || input_file_bytes < MIN_PARALLEL_BYTES || special_tokens.is_empty() {
        let text = fs::read_to_string(input_path)?;
        return pretoken_byte_counts(&text, special_tokens);
    }

    let ranges = chunk_ranges(input_path, num_workers, chunk_bytes, special_tokens)?;
    if ranges.len() == 1 {
        let text = fs::read_to_string(input_path)?;
        return pretoken_byte_counts(&text, special_tokens);
    }

    let partials: Vec<Result<HashMap<Vec<u8>, Count>>> = pool.install(|| {
        ranges
            .par_iter()
            .map(|&(start, end)| pretoken_counts_for_range(input_path, start, end, special_tokens))
            .collect()
    });

    let mut counts = HashMap::new();
    for partial in partials {
        for (pretoken, count) in partial? {
            *counts.entry(pretoken).or_insert(0) += count;
        }
    }
    Ok(counts)
}

fn pretoken_counts_for_range(
    input_path: &Path,
    start: u64,
    end: u64,
    special_tokens: &[String],
) -> Result<HashMap<Vec<u8>, Count>> {
    let mut file = File::open(input_path)?;
    file.seek(SeekFrom::Start(start))?;
    let mut buffer = vec![0; (end - start) as usize];
    file.read_exact(&mut buffer)?;
    let text = String::from_utf8(buffer).context("corpus chunk is not valid UTF-8")?;
    pretoken_byte_counts(&text, special_tokens)
}

fn apply_merge(state: &mut TrainerState, best_pair: TokenPair) -> HashSet<TokenPair> {
    let mut changed_pairs = HashSet::new();
    let mut merged_token = state.id_to_bytes[best_pair.0 as usize].clone();
    merged_token.extend(&state.id_to_bytes[best_pair.1 as usize]);
    let merged_token_id = state.id_to_bytes.len() as TokenId;
    state.merges.push((
        state.id_to_bytes[best_pair.0 as usize].clone(),
        state.id_to_bytes[best_pair.1 as usize].clone(),
    ));
    state.id_to_bytes.push(merged_token);

    let affected_word_ids: Vec<_> = state
        .pair_to_word_ids
        .get(&best_pair)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .collect();

    for word_id in affected_word_ids {
        let word_count = state.word_counts[word_id];
        let old_word = state.words[word_id].clone();
        let old_pairs = word_pair_frequencies(&old_word);

        for (pair, frequency) in old_pairs {
            changed_pairs.insert(pair);
            let decrement = frequency * word_count;
            match state.pair_counts.get_mut(&pair) {
                Some(count) if *count > decrement => *count -= decrement,
                Some(_) => {
                    state.pair_counts.remove(&pair);
                }
                None => {}
            }

            if let Some(word_ids) = state.pair_to_word_ids.get_mut(&pair) {
                word_ids.remove(&word_id);
                if word_ids.is_empty() {
                    state.pair_to_word_ids.remove(&pair);
                }
            }
        }

        let new_word = merge_word(&old_word, best_pair, merged_token_id);
        state.words[word_id] = new_word.clone();
        for (pair, frequency) in word_pair_frequencies(&new_word) {
            changed_pairs.insert(pair);
            *state.pair_counts.entry(pair).or_insert(0) += frequency * word_count;
            state
                .pair_to_word_ids
                .entry(pair)
                .or_default()
                .insert(word_id);
        }
    }
    changed_pairs.insert(best_pair);
    changed_pairs
}

fn should_rebuild_heap(heap_rebuild_factor: f64, heap_len: usize, pair_count_len: usize) -> bool {
    heap_rebuild_factor > 0.0
        && pair_count_len > 0
        && (heap_len as f64) > heap_rebuild_factor * pair_count_len as f64
}
