use std::path::PathBuf;

use cli::{ArgAction, Args, Subcommand};

use super::parse_positive_count;

extern crate cli as clap;

#[derive(Debug, Args)]
pub(in crate::cli) struct DatasetArgs {
    #[command(subcommand)]
    pub(in crate::cli) target: DatasetTarget,
}

#[derive(Debug, Subcommand)]
pub(in crate::cli) enum DatasetTarget {
    /// Pack persona directories into multi-frame AVIF plus JSONL manifests.
    Pack(DatasetPackArgs),
    /// Unpack multi-frame AVIF personas back to an AVIF file layout.
    Unpack(DatasetUnpackArgs),
    /// Verify packed AVIF frame hashes and JSONL metadata.
    Verify(DatasetVerifyArgs),
}

#[derive(Clone, Debug, Args)]
pub(in crate::cli) struct DatasetPackArgs {
    /// Root containing persona_*/images, glyphs, and poses.
    #[arg(
        long = "data-root",
        value_name = "DIR",
        default_value = "data/pose-obc"
    )]
    pub(in crate::cli) data_root: PathBuf,
    /// Output root for packed persona directories.
    #[arg(
        long = "output-root",
        value_name = "DIR",
        default_value = "data/pose-obc-packed"
    )]
    pub(in crate::cli) output_root: PathBuf,
    /// Pack only the named persona directory. Can be provided more than once.
    #[arg(long = "persona", value_name = "NAME")]
    pub(in crate::cli) personas: Vec<String>,
    /// AVIF encoder speed, 1 slowest/smallest through 10 fastest/largest.
    #[arg(long = "speed", default_value_t = 4)]
    pub(in crate::cli) speed: u8,
    /// AVIF encoder worker threads. Defaults to the encoder's thread pool.
    #[arg(long = "threads", value_name = "N", value_parser = parse_positive_count)]
    pub(in crate::cli) threads: Option<usize>,
    /// Replace existing packed files.
    #[arg(long, action = ArgAction::SetTrue)]
    pub(in crate::cli) force: bool,
}

#[derive(Clone, Debug, Args)]
pub(in crate::cli) struct DatasetUnpackArgs {
    /// Root containing packed persona directories.
    #[arg(
        long = "packed-root",
        value_name = "DIR",
        default_value = "data/pose-obc-packed"
    )]
    pub(in crate::cli) packed_root: PathBuf,
    /// Output root for unpacked persona directories.
    #[arg(
        long = "output-root",
        value_name = "DIR",
        default_value = "data/pose-obc"
    )]
    pub(in crate::cli) output_root: PathBuf,
    /// Unpack only the named persona directory. Can be provided more than once.
    #[arg(long = "persona", value_name = "NAME")]
    pub(in crate::cli) personas: Vec<String>,
    /// AVIF encoder speed used when writing per-sample AVIF files.
    #[arg(long = "speed", default_value_t = 4)]
    pub(in crate::cli) speed: u8,
    /// AVIF encoder worker threads. Defaults to the encoder's thread pool.
    #[arg(long = "threads", value_name = "N", value_parser = parse_positive_count)]
    pub(in crate::cli) threads: Option<usize>,
    /// Replace existing unpacked files.
    #[arg(long, action = ArgAction::SetTrue)]
    pub(in crate::cli) force: bool,
}

#[derive(Clone, Debug, Args)]
pub(in crate::cli) struct DatasetVerifyArgs {
    /// Root containing packed persona directories.
    #[arg(
        long = "packed-root",
        value_name = "DIR",
        default_value = "data/pose-obc-packed"
    )]
    pub(in crate::cli) packed_root: PathBuf,
    /// Verify only the named persona directory. Can be provided more than once.
    #[arg(long = "persona", value_name = "NAME")]
    pub(in crate::cli) personas: Vec<String>,
}
