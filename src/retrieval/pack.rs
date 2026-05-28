use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};

use super::{RetrievalError, canonical_or_original};

pub const DATASET_PACK_VERSION: u32 = 1;

#[derive(Clone, Debug)]
pub struct DatasetPackOptions {
    pub data_root: PathBuf,
    pub output_root: PathBuf,
    pub personas: Vec<String>,
    pub speed: u8,
    pub threads: Option<usize>,
    pub force: bool,
}

#[derive(Clone, Debug)]
pub struct DatasetUnpackOptions {
    pub packed_root: PathBuf,
    pub output_root: PathBuf,
    pub personas: Vec<String>,
    pub speed: u8,
    pub threads: Option<usize>,
    pub force: bool,
}

#[derive(Clone, Debug)]
pub struct DatasetVerifyOptions {
    pub packed_root: PathBuf,
    pub personas: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct DatasetPersonaReport {
    pub persona: String,
    pub samples: usize,
}

#[derive(Clone, Debug, Default)]
pub struct DatasetPackReport {
    pub personas: Vec<DatasetPersonaReport>,
}

impl DatasetPackReport {
    pub fn total_samples(&self) -> usize {
        self.personas.iter().map(|persona| persona.samples).sum()
    }
}

pub fn pack_dataset(options: &DatasetPackOptions) -> Result<DatasetPackReport, RetrievalError> {
    pack_dataset_impl(options)
}

pub fn unpack_dataset(options: &DatasetUnpackOptions) -> Result<DatasetPackReport, RetrievalError> {
    unpack_dataset_impl(options)
}

pub fn verify_packed_dataset(
    options: &DatasetVerifyOptions,
) -> Result<DatasetPackReport, RetrievalError> {
    verify_packed_dataset_impl(options)
}

fn pack_dataset_impl(options: &DatasetPackOptions) -> Result<DatasetPackReport, RetrievalError> {
    ensure_avif_speed(options.speed)?;
    let personas = collect_persona_dirs(&options.data_root, &options.personas)?;
    fs::create_dir_all(&options.output_root)?;

    let mut report = DatasetPackReport::default();
    for (persona, persona_dir) in personas {
        let source_records = collect_source_records(&persona, &persona_dir)?;
        let output_dir = options.output_root.join(&persona);
        create_clean_pack_dir(&output_dir, options.force)?;

        let image_paths = source_records
            .iter()
            .map(|record| record.image_path.clone())
            .collect::<Vec<_>>();
        let glyph_paths = source_records
            .iter()
            .map(|record| record.glyph_path.clone())
            .collect::<Vec<_>>();

        let image_entries = encode_image_sequence(
            &image_paths,
            &output_dir.join("images.avif"),
            options.speed,
            options.threads,
        )?;
        let glyph_entries = encode_image_sequence(
            &glyph_paths,
            &output_dir.join("glyphs.avif"),
            options.speed,
            options.threads,
        )?;

        let mut manifest_records = Vec::with_capacity(source_records.len());
        let mut pose_records = Vec::with_capacity(source_records.len());
        for (index, source) in source_records.iter().enumerate() {
            let mut pose_bytes = fs::read(&source.pose_path)?;
            let pose_source_sha256 = sha256_hex(&pose_bytes);
            let pose_json: json::OwnedValue = json::from_slice(&mut pose_bytes)?;
            let pose_json_sha256 = json_value_sha256(&pose_json)?;
            pose_records.push(PackedPoseRecord {
                version: DATASET_PACK_VERSION,
                id: source.id.clone(),
                json_sha256: pose_json_sha256.clone(),
                pose: pose_json,
            });

            let metadata = parse_glyph_id(&source.id);
            manifest_records.push(PackedManifestRecord {
                version: DATASET_PACK_VERSION,
                id: source.id.clone(),
                codepoint: metadata.codepoint,
                character: metadata.character,
                persona: persona.clone(),
                image: PackedImageEntry {
                    file: "images.avif".to_string(),
                    frame: index,
                    source: relative_slash_path(&source.image_path, &persona_dir),
                    source_sha256: image_entries[index].source_sha256.clone(),
                    rgba_sha256: image_entries[index].rgba_sha256.clone(),
                    width: image_entries[index].width,
                    height: image_entries[index].height,
                },
                glyph: PackedImageEntry {
                    file: "glyphs.avif".to_string(),
                    frame: index,
                    source: relative_slash_path(&source.glyph_path, &persona_dir),
                    source_sha256: glyph_entries[index].source_sha256.clone(),
                    rgba_sha256: glyph_entries[index].rgba_sha256.clone(),
                    width: glyph_entries[index].width,
                    height: glyph_entries[index].height,
                },
                pose: PackedPoseEntry {
                    file: "poses.jsonl".to_string(),
                    line: index,
                    source: relative_slash_path(&source.pose_path, &persona_dir),
                    source_sha256: pose_source_sha256,
                    json_sha256: pose_json_sha256,
                },
            });
        }

        write_jsonl(output_dir.join("manifest.jsonl"), &manifest_records)?;
        write_jsonl(output_dir.join("poses.jsonl"), &pose_records)?;
        report.personas.push(DatasetPersonaReport {
            persona,
            samples: manifest_records.len(),
        });
    }

    Ok(report)
}

fn unpack_dataset_impl(
    options: &DatasetUnpackOptions,
) -> Result<DatasetPackReport, RetrievalError> {
    ensure_avif_speed(options.speed)?;
    let personas = collect_packed_persona_dirs(&options.packed_root, &options.personas)?;
    fs::create_dir_all(&options.output_root)?;

    let mut report = DatasetPackReport::default();
    for (persona, packed_dir) in personas {
        let manifest = read_manifest(&packed_dir)?;
        let poses = read_pose_records(&packed_dir)?;
        validate_manifest(&persona, &packed_dir, &manifest, &poses)?;

        let image_frames = decode_animation_or_still(&packed_dir.join("images.avif"))?;
        let glyph_frames = decode_animation_or_still(&packed_dir.join("glyphs.avif"))?;
        ensure_frame_count("images.avif", image_frames.len(), manifest.len())?;
        ensure_frame_count("glyphs.avif", glyph_frames.len(), manifest.len())?;

        let output_dir = options.output_root.join(&persona);
        let image_dir = output_dir.join("images");
        let glyph_dir = output_dir.join("glyphs");
        let pose_dir = output_dir.join("poses");
        fs::create_dir_all(&image_dir)?;
        fs::create_dir_all(&glyph_dir)?;
        fs::create_dir_all(&pose_dir)?;

        let pose_by_id = poses
            .into_iter()
            .map(|pose| (pose.id.clone(), pose))
            .collect::<BTreeMap<_, _>>();
        let encode_config = lossless_encoder_config(options.speed, options.threads);
        for record in &manifest {
            let image_frame = image_frames.get(record.image.frame).ok_or_else(|| {
                RetrievalError::InvalidData(format!(
                    "{} image frame {} is out of range",
                    record.id, record.image.frame
                ))
            })?;
            let glyph_frame = glyph_frames.get(record.glyph.frame).ok_or_else(|| {
                RetrievalError::InvalidData(format!(
                    "{} glyph frame {} is out of range",
                    record.id, record.glyph.frame
                ))
            })?;
            let image_path = image_dir.join(format!("{}.avif", record.id));
            let glyph_path = glyph_dir.join(format!("{}.avif", record.id));
            write_still_avif_checked(&image_path, image_frame, &encode_config, options.force)?;
            write_still_avif_checked(&glyph_path, glyph_frame, &encode_config, options.force)?;

            let pose = pose_by_id.get(&record.id).ok_or_else(|| {
                RetrievalError::InvalidData(format!("missing pose record for {}", record.id))
            })?;
            let pose_path = pose_dir.join(format!("{}.json", record.id));
            let pose_json = json::to_string_pretty(&pose.pose)?;
            write_checked(&pose_path, pose_json.as_bytes(), options.force)?;
        }

        report.personas.push(DatasetPersonaReport {
            persona,
            samples: manifest.len(),
        });
    }

    Ok(report)
}

fn verify_packed_dataset_impl(
    options: &DatasetVerifyOptions,
) -> Result<DatasetPackReport, RetrievalError> {
    let personas = collect_packed_persona_dirs(&options.packed_root, &options.personas)?;
    let mut report = DatasetPackReport::default();

    for (persona, packed_dir) in personas {
        let manifest = read_manifest(&packed_dir)?;
        let poses = read_pose_records(&packed_dir)?;
        validate_manifest(&persona, &packed_dir, &manifest, &poses)?;

        let image_frames = decode_animation_or_still(&packed_dir.join("images.avif"))?;
        let glyph_frames = decode_animation_or_still(&packed_dir.join("glyphs.avif"))?;
        ensure_frame_count("images.avif", image_frames.len(), manifest.len())?;
        ensure_frame_count("glyphs.avif", glyph_frames.len(), manifest.len())?;

        for record in &manifest {
            verify_frame_hash(
                "image",
                record,
                &image_frames[record.image.frame],
                &record.image,
            )?;
            verify_frame_hash(
                "glyph",
                record,
                &glyph_frames[record.glyph.frame],
                &record.glyph,
            )?;
        }

        report.personas.push(DatasetPersonaReport {
            persona,
            samples: manifest.len(),
        });
    }

    Ok(report)
}

#[derive(Clone, Debug)]
struct SourceRecord {
    id: String,
    image_path: PathBuf,
    glyph_path: PathBuf,
    pose_path: PathBuf,
}

#[derive(Clone, Debug)]
struct EncodedImageEntry {
    source_sha256: String,
    rgba_sha256: String,
    width: u32,
    height: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackedManifestRecord {
    pub version: u32,
    pub id: String,
    pub codepoint: Option<String>,
    pub character: Option<String>,
    pub persona: String,
    pub image: PackedImageEntry,
    pub glyph: PackedImageEntry,
    pub pose: PackedPoseEntry,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackedImageEntry {
    pub file: String,
    pub frame: usize,
    pub source: String,
    pub source_sha256: String,
    pub rgba_sha256: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackedPoseEntry {
    pub file: String,
    pub line: usize,
    pub source: String,
    pub source_sha256: String,
    pub json_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackedPoseRecord {
    pub version: u32,
    pub id: String,
    pub json_sha256: String,
    pub pose: json::OwnedValue,
}

fn collect_persona_dirs(
    root: &Path,
    selected: &[String],
) -> Result<Vec<(String, PathBuf)>, RetrievalError> {
    let mut personas = Vec::new();
    let selected = selected_set(selected);
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with("persona_") || !selected_persona(&selected, &name) {
            continue;
        }
        personas.push((name, canonical_or_original(entry.path())));
    }
    personas.sort_by(|left, right| left.0.cmp(&right.0));
    ensure_found_personas(root, selected.as_ref(), &personas)?;
    Ok(personas)
}

fn collect_packed_persona_dirs(
    root: &Path,
    selected: &[String],
) -> Result<Vec<(String, PathBuf)>, RetrievalError> {
    let mut personas = Vec::new();
    let selected = selected_set(selected);
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with("persona_") || !selected_persona(&selected, &name) {
            continue;
        }
        let path = canonical_or_original(entry.path());
        if path.join("manifest.jsonl").is_file() {
            personas.push((name, path));
        }
    }
    personas.sort_by(|left, right| left.0.cmp(&right.0));
    ensure_found_personas(root, selected.as_ref(), &personas)?;
    Ok(personas)
}

fn selected_set(selected: &[String]) -> Option<BTreeSet<String>> {
    if selected.is_empty() {
        None
    } else {
        Some(selected.iter().cloned().collect())
    }
}

fn selected_persona(selected: &Option<BTreeSet<String>>, name: &str) -> bool {
    selected
        .as_ref()
        .is_none_or(|selected| selected.contains(name))
}

fn ensure_found_personas(
    root: &Path,
    selected: Option<&BTreeSet<String>>,
    personas: &[(String, PathBuf)],
) -> Result<(), RetrievalError> {
    if personas.is_empty() {
        return Err(RetrievalError::InvalidData(format!(
            "no persona directories found under {}",
            root.display()
        )));
    }
    if let Some(selected) = selected {
        let found = personas
            .iter()
            .map(|(name, _)| name)
            .cloned()
            .collect::<BTreeSet<_>>();
        let missing = selected.difference(&found).cloned().collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(RetrievalError::InvalidData(format!(
                "requested persona directories not found under {}: {}",
                root.display(),
                missing.join(", ")
            )));
        }
    }
    Ok(())
}

fn collect_source_records(
    persona: &str,
    persona_dir: &Path,
) -> Result<Vec<SourceRecord>, RetrievalError> {
    let image_dir = persona_dir.join("images");
    let glyph_dir = persona_dir.join("glyphs");
    let pose_dir = persona_dir.join("poses");
    if !image_dir.is_dir() || !glyph_dir.is_dir() || !pose_dir.is_dir() {
        return Err(RetrievalError::InvalidData(format!(
            "{persona} must contain images, glyphs, and poses directories"
        )));
    }

    let glyphs = collect_files_by_stem(&glyph_dir)?;
    let poses = collect_files_by_stem_with_extension(&pose_dir, "json")?;
    let mut image_paths = Vec::new();
    for entry in fs::read_dir(&image_dir)? {
        let entry = entry?;
        if entry.file_type()?.is_file() && is_supported_image_path(&entry.path()) {
            image_paths.push(entry.path());
        }
    }
    image_paths.sort();

    let mut records = Vec::new();
    for image_path in image_paths {
        let id = file_stem_string(&image_path)?;
        let glyph_path = glyphs.get(&id).ok_or_else(|| {
            RetrievalError::InvalidData(format!("{persona} missing glyph for sample {id}"))
        })?;
        let pose_path = poses.get(&id).ok_or_else(|| {
            RetrievalError::InvalidData(format!("{persona} missing pose JSON for sample {id}"))
        })?;
        records.push(SourceRecord {
            id,
            image_path,
            glyph_path: glyph_path.clone(),
            pose_path: pose_path.clone(),
        });
    }

    if records.is_empty() {
        return Err(RetrievalError::InvalidData(format!(
            "no image/glyph/pose records found in {}",
            persona_dir.display()
        )));
    }

    Ok(records)
}

fn collect_files_by_stem(dir: &Path) -> Result<BTreeMap<String, PathBuf>, RetrievalError> {
    let mut files = BTreeMap::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() || !is_supported_image_path(&entry.path()) {
            continue;
        }
        let stem = file_stem_string(&entry.path())?;
        files.entry(stem).or_insert_with(|| entry.path());
    }
    Ok(files)
}

fn collect_files_by_stem_with_extension(
    dir: &Path,
    expected_extension: &str,
) -> Result<BTreeMap<String, PathBuf>, RetrievalError> {
    let mut files = BTreeMap::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let matches_extension = entry
            .path()
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case(expected_extension));
        if !matches_extension {
            continue;
        }
        let stem = file_stem_string(&entry.path())?;
        files.entry(stem).or_insert_with(|| entry.path());
    }
    Ok(files)
}

fn file_stem_string(path: &Path) -> Result<String, RetrievalError> {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .map(str::to_owned)
        .ok_or_else(|| {
            RetrievalError::InvalidData(format!("path has no UTF-8 file stem: {}", path.display()))
        })
}

fn is_supported_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("avif"))
}

fn encode_image_sequence(
    paths: &[PathBuf],
    output_path: &Path,
    speed: u8,
    threads: Option<usize>,
) -> Result<Vec<EncodedImageEntry>, RetrievalError> {
    let mut frames = Vec::with_capacity(paths.len());
    let mut entries = Vec::with_capacity(paths.len());
    let mut expected_size = None;

    for path in paths {
        let source_bytes = fs::read(path)?;
        let source_sha256 = sha256_hex(&source_bytes);
        let image = crate::image::open_dynamic_image(path)?.to_rgba8();
        let (width, height) = image.dimensions();
        if let Some((expected_width, expected_height)) = expected_size {
            if (width, height) != (expected_width, expected_height) {
                return Err(RetrievalError::InvalidData(format!(
                    "AVIF sequence frames must have identical dimensions: {} is {}x{}, expected {}x{}",
                    path.display(),
                    width,
                    height,
                    expected_width,
                    expected_height
                )));
            }
        } else {
            expected_size = Some((width, height));
        }
        let rgba_bytes = image.as_raw().clone();
        let rgba_sha256 = sha256_hex(&rgba_bytes);
        frames.push(LosslessRgbaFrame {
            rgba: rgba_bytes,
            duration_ms: 1,
        });
        entries.push(EncodedImageEntry {
            source_sha256,
            rgba_sha256,
            width,
            height,
        });
    }

    let (width, height) = expected_size.ok_or_else(|| {
        RetrievalError::InvalidData("cannot encode empty AVIF sequence".to_string())
    })?;
    let encoded =
        encode_lossless_animation_rgba8(&frames, width as usize, height as usize, speed, threads)?;
    fs::write(output_path, encoded)?;
    let encoded_frames = decode_animation_or_still(output_path)?;
    ensure_frame_count(
        &output_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("encoded AVIF sequence"),
        encoded_frames.len(),
        entries.len(),
    )?;
    for (index, (entry, pixels)) in entries.iter_mut().zip(encoded_frames.iter()).enumerate() {
        let (actual_width, actual_height, rgba) = crate::image::pixel_buffer_to_rgba8(pixels)?;
        if (actual_width, actual_height) != (entry.width, entry.height) {
            return Err(RetrievalError::InvalidData(format!(
                "{} frame {index} decoded as {}x{}, expected {}x{}",
                output_path.display(),
                actual_width,
                actual_height,
                entry.width,
                entry.height
            )));
        }
        entry.rgba_sha256 = sha256_hex(&rgba);
    }
    Ok(entries)
}

struct LosslessRgbaFrame {
    rgba: Vec<u8>,
    duration_ms: u32,
}

struct EncodedAv1Sequence {
    sequence_header: Vec<u8>,
    frames: Vec<Vec<u8>>,
}

fn encode_lossless_animation_rgba8(
    frames: &[LosslessRgbaFrame],
    width: usize,
    height: usize,
    speed: u8,
    threads: Option<usize>,
) -> Result<Vec<u8>, RetrievalError> {
    if frames.is_empty() {
        return Err(RetrievalError::InvalidData(
            "cannot encode empty AVIF sequence".to_string(),
        ));
    }
    let expected_len = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| {
            RetrievalError::InvalidData(format!(
                "AVIF sequence dimensions are too large: {width}x{height}"
            ))
        })?;
    for frame in frames {
        if frame.duration_ms == 0 {
            return Err(RetrievalError::InvalidData(
                "AVIF animation frame duration must be > 0".to_string(),
            ));
        }
        if frame.rgba.len() != expected_len {
            return Err(RetrievalError::InvalidData(format!(
                "AVIF animation frame has {} RGBA bytes, expected {expected_len}",
                frame.rgba.len()
            )));
        }
    }

    let has_alpha = frames
        .iter()
        .any(|frame| frame.rgba.chunks_exact(4).any(|pixel| pixel[3] != 255));
    let color = encode_lossless_av1_sequence(
        width,
        height,
        frames.len(),
        speed,
        threads,
        false,
        |index, frame| fill_color_frame_bt601_12(frame, width, height, &frames[index].rgba),
    )?;
    let alpha = if has_alpha {
        Some(encode_lossless_av1_sequence(
            width,
            height,
            frames.len(),
            speed,
            threads,
            true,
            |index, frame| fill_alpha_frame(frame, width, height, &frames[index].rgba),
        )?)
    } else {
        None
    };

    let serialize_frames = color
        .frames
        .iter()
        .zip(frames.iter())
        .enumerate()
        .map(|(index, (color_data, source_frame))| {
            let frame =
                avif_serialize::animated::AnimFrame::new(color_data, source_frame.duration_ms)
                    .with_sync(true);
            if let Some(alpha) = &alpha {
                frame.with_alpha(&alpha.frames[index])
            } else {
                frame
            }
        })
        .collect::<Vec<_>>();

    let mut animation = avif_serialize::animated::AnimatedImage::new();
    animation.set_color_config(av1c_config_color_444());
    let mut colr = avif_serialize::ColrBox::default();
    colr.color_primaries = avif_serialize::constants::ColorPrimaries::Bt709;
    colr.transfer_characteristics = avif_serialize::constants::TransferCharacteristics::Srgb;
    colr.matrix_coefficients = avif_serialize::constants::MatrixCoefficients::Bt601;
    colr.full_range_flag = true;
    animation.set_colr(colr);
    if alpha.is_some() {
        animation.set_alpha_config(av1c_config_alpha_400());
    }

    let mut avif = animation.serialize(
        width as u32,
        height as u32,
        &serialize_frames,
        &color.sequence_header,
        alpha
            .as_ref()
            .map(|sequence| sequence.sequence_header.as_slice()),
    );
    inject_animation_color_track_colr(&mut avif)?;
    Ok(avif)
}

fn encode_lossless_av1_sequence(
    width: usize,
    height: usize,
    frame_count: usize,
    speed: u8,
    threads: Option<usize>,
    is_alpha: bool,
    mut init_frame: impl FnMut(usize, &mut av1_encoder::Frame<u16>) -> Result<(), RetrievalError>,
) -> Result<EncodedAv1Sequence, RetrievalError> {
    use av1_encoder::prelude::*;

    let encoder_config = lossless_animation_encoder_config(width, height, speed, is_alpha);
    let mut config = Config::new().with_encoder_config(encoder_config);
    if let Some(threads) = threads {
        config = config.with_threads(threads);
    }
    let mut context: Context<u16> = config.new_context().map_err(|error| {
        RetrievalError::InvalidData(format!("AVIF encoder config failed: {error}"))
    })?;
    let sequence_header = context.container_sequence_header();

    for index in 0..frame_count {
        let mut frame = context.new_frame();
        init_frame(index, &mut frame)?;
        context.send_frame(frame).map_err(avif_encoder_status)?;
    }
    context.flush();

    let mut packets = (0..frame_count)
        .map(|_| None)
        .collect::<Vec<Option<Vec<u8>>>>();
    loop {
        match context.receive_packet() {
            Ok(packet) => {
                let index = packet.input_frameno as usize;
                if let Some(slot) = packets.get_mut(index) {
                    *slot = Some(packet.data);
                }
            }
            Err(EncoderStatus::Encoded | EncoderStatus::NeedMoreData) => continue,
            Err(EncoderStatus::LimitReached) => break,
            Err(status) => return Err(avif_encoder_status(status)),
        }
    }

    let mut encoded_frames = Vec::with_capacity(frame_count);
    for (index, packet) in packets.into_iter().enumerate() {
        encoded_frames.push(packet.ok_or_else(|| {
            RetrievalError::InvalidData(format!("AVIF encoder did not emit frame {index}"))
        })?);
    }

    Ok(EncodedAv1Sequence {
        sequence_header,
        frames: encoded_frames,
    })
}

fn lossless_animation_encoder_config(
    width: usize,
    height: usize,
    speed: u8,
    is_alpha: bool,
) -> av1_encoder::EncoderConfig {
    use av1_encoder::prelude::*;

    let mut config = EncoderConfig::with_speed_preset(speed);
    config.width = width;
    config.height = height;
    config.sample_aspect_ratio = Rational::new(1, 1);
    config.time_base = Rational::new(1, 1000);
    config.bit_depth = 12;
    config.chroma_sampling = if is_alpha {
        ChromaSampling::Cs400
    } else {
        ChromaSampling::Cs444
    };
    config.chroma_sample_position = ChromaSamplePosition::Unknown;
    config.pixel_range = PixelRange::Full;
    config.color_description = if is_alpha {
        None
    } else {
        Some(ColorDescription {
            color_primaries: ColorPrimaries::BT709,
            transfer_characteristics: TransferCharacteristics::SRGB,
            matrix_coefficients: MatrixCoefficients::BT601,
        })
    };
    config.mastering_display = None;
    config.content_light = None;
    config.level_idx = None;
    config.enable_timing_info = false;
    config.still_picture = false;
    config.error_resilient = false;
    config.switch_frame_interval = 0;
    config.min_key_frame_interval = 1;
    config.max_key_frame_interval = 1;
    config.reservoir_frame_delay = None;
    config.low_latency = true;
    config.quantizer = 0;
    config.min_quantizer = 0;
    config.bitrate = 0;
    config.tune = Tune::Psychovisual;
    config.tile_cols = 0;
    config.tile_rows = 0;
    config.tiles = 0;
    config.film_grain_params = None;
    config.enable_qm = false;
    config.enable_vaq = false;
    config.vaq_strength = 1.0;
    config.seg_boost = 1.0;
    config.enable_trellis = false;
    config.max_pixel_count = u64::MAX;
    config
}

fn fill_color_frame_bt601_12(
    frame: &mut av1_encoder::Frame<u16>,
    width: usize,
    height: usize,
    rgba: &[u8],
) -> Result<(), RetrievalError> {
    let mut planes = frame.planes.iter_mut();
    let mut y_plane = planes.next().unwrap().mut_slice(Default::default());
    let mut cb_plane = planes.next().unwrap().mut_slice(Default::default());
    let mut cr_plane = planes.next().unwrap().mut_slice(Default::default());
    let mut color_cache = HashMap::<[u8; 3], (u16, u16, u16)>::new();

    for row_index in 0..height {
        let y_row = &mut y_plane.rows_iter_mut().nth(row_index).ok_or_else(|| {
            RetrievalError::InvalidData(format!("AVIF encoder missing Y row {row_index}"))
        })?[..width];
        let cb_row = &mut cb_plane.rows_iter_mut().nth(row_index).ok_or_else(|| {
            RetrievalError::InvalidData(format!("AVIF encoder missing Cb row {row_index}"))
        })?[..width];
        let cr_row = &mut cr_plane.rows_iter_mut().nth(row_index).ok_or_else(|| {
            RetrievalError::InvalidData(format!("AVIF encoder missing Cr row {row_index}"))
        })?[..width];
        let rgba_row = &rgba[row_index * width * 4..][..width * 4];
        for column_index in 0..width {
            let offset = column_index * 4;
            let rgb = [rgba_row[offset], rgba_row[offset + 1], rgba_row[offset + 2]];
            let (y, cb, cr) = *color_cache
                .entry(rgb)
                .or_insert_with(|| rgb_to_bt601_12(rgb[0], rgb[1], rgb[2]));
            y_row[column_index] = y;
            cb_row[column_index] = cb;
            cr_row[column_index] = cr;
        }
    }
    Ok(())
}

fn fill_alpha_frame(
    frame: &mut av1_encoder::Frame<u16>,
    width: usize,
    height: usize,
    rgba: &[u8],
) -> Result<(), RetrievalError> {
    let mut alpha_plane = frame.planes[0].mut_slice(Default::default());
    for row_index in 0..height {
        let alpha_row = &mut alpha_plane.rows_iter_mut().nth(row_index).ok_or_else(|| {
            RetrievalError::InvalidData(format!("AVIF encoder missing alpha row {row_index}"))
        })?[..width];
        let rgba_row = &rgba[row_index * width * 4..][..width * 4];
        for column_index in 0..width {
            alpha_row[column_index] = scale_u8_to_12(rgba_row[column_index * 4 + 3]);
        }
    }
    Ok(())
}

fn rgb_to_bt601_12(red: u8, green: u8, blue: u8) -> (u16, u16, u16) {
    let estimated = rgb_to_bt601_12_estimate(red, green, blue);
    let decoded = decoded_bt601_12(estimated);
    if rgb12_to_rgb8(decoded) == (red, green, blue)
        && rgb12_bucket_margin(decoded, (red, green, blue)) >= 3
    {
        return estimated;
    }
    corrected_bt601_12(red, green, blue, estimated).unwrap_or(estimated)
}

fn rgb_to_bt601_12_estimate(red: u8, green: u8, blue: u8) -> (u16, u16, u16) {
    const HALF_12: f64 = 2048.0;
    const KR: f64 = 0.2990;
    const KG: f64 = 0.5870;
    const KB: f64 = 0.1140;

    let red = f64::from(red) * 16.0 + 8.0;
    let green = f64::from(green) * 16.0 + 8.0;
    let blue = f64::from(blue) * 16.0 + 8.0;
    let y = KR * red + KG * green + KB * blue;
    let cb = (blue - y) * 0.5 / (1.0 - KB) + HALF_12;
    let cr = (red - y) * 0.5 / (1.0 - KR) + HALF_12;

    (round_to_12(y), round_to_12(cb), round_to_12(cr))
}

fn corrected_bt601_12(
    red: u8,
    green: u8,
    blue: u8,
    estimated: (u16, u16, u16),
) -> Option<(u16, u16, u16)> {
    const Y_COEF: i32 = 8192;
    const CB_COEF: i32 = 14516;
    const CR_COEF: i32 = 11485;
    const UV_BIAS: i32 = 2048;

    let target_r = i32::from(red) * 16 + 8;
    let target_g = i32::from(green) * 16 + 8;
    let target_b = i32::from(blue) * 16 + 8;
    let mut best = None;
    let mut best_score = i64::MAX;

    for radius in [2i32, 8, 32] {
        let y_min = (i32::from(estimated.0) - radius).max(0);
        let y_max = (i32::from(estimated.0) + radius).min(4095);
        for y in y_min..=y_max {
            let cb_center = clamp_to_12(UV_BIAS + round_div((target_b - y) * Y_COEF, CB_COEF));
            let cr_center = clamp_to_12(UV_BIAS + round_div((target_r - y) * Y_COEF, CR_COEF));

            for cb_delta in -4..=4 {
                let cb = clamp_to_12(i32::from(cb_center) + cb_delta);
                for cr_delta in -4..=4 {
                    let cr = clamp_to_12(i32::from(cr_center) + cr_delta);
                    let candidate = (y as u16, cb, cr);
                    let decoded = decoded_bt601_12(candidate);
                    if rgb12_to_rgb8(decoded) != (red, green, blue) {
                        continue;
                    }
                    let center_error = rgb12_center_error(decoded, (target_r, target_g, target_b));
                    let distance = ycbcr_distance(estimated, candidate);
                    let score = center_error.saturating_mul(1_000_000) + distance;
                    if score < best_score {
                        best_score = score;
                        best = Some(candidate);
                    }
                }
            }
        }
        if best.is_some() {
            return best;
        }
    }

    best
}

fn decoded_bt601_12((y, cb, cr): (u16, u16, u16)) -> (i32, i32, i32) {
    const PRECISION: i32 = 13;
    const Y_COEF: i32 = 8192;
    const CR_COEF: i32 = 11485;
    const CB_COEF: i32 = 14516;
    const G_COEF_1: i32 = 5850;
    const G_COEF_2: i32 = 2819;
    const UV_BIAS: i32 = 2048;

    let y_value = i32::from(y) * Y_COEF;
    let cb_value = i32::from(cb) - UV_BIAS;
    let cr_value = i32::from(cr) - UV_BIAS;
    let red_12 = qrshr_12::<PRECISION>(y_value + CR_COEF * cr_value);
    let blue_12 = qrshr_12::<PRECISION>(y_value + CB_COEF * cb_value);
    let green_12 = qrshr_12::<PRECISION>(y_value - G_COEF_1 * cr_value - G_COEF_2 * cb_value);
    (red_12, green_12, blue_12)
}

fn rgb12_to_rgb8((red, green, blue): (i32, i32, i32)) -> (u8, u8, u8) {
    ((red >> 4) as u8, (green >> 4) as u8, (blue >> 4) as u8)
}

fn rgb12_bucket_margin((red, green, blue): (i32, i32, i32), target: (u8, u8, u8)) -> i32 {
    [
        bucket_margin(red, target.0),
        bucket_margin(green, target.1),
        bucket_margin(blue, target.2),
    ]
    .into_iter()
    .min()
    .unwrap_or(0)
}

fn bucket_margin(value: i32, target: u8) -> i32 {
    let low = i32::from(target) * 16;
    let high = low + 15;
    (value - low).min(high - value)
}

fn qrshr_12<const PRECISION: i32>(value: i32) -> i32 {
    let rounding = 1 << (PRECISION - 1);
    ((value + rounding) >> PRECISION).clamp(0, 4095)
}

fn round_div(numerator: i32, denominator: i32) -> i32 {
    if numerator >= 0 {
        (numerator + denominator / 2) / denominator
    } else {
        (numerator - denominator / 2) / denominator
    }
}

fn clamp_to_12(value: i32) -> u16 {
    value.clamp(0, 4095) as u16
}

fn ycbcr_distance(left: (u16, u16, u16), right: (u16, u16, u16)) -> i64 {
    let dy = i64::from(left.0) - i64::from(right.0);
    let dcb = i64::from(left.1) - i64::from(right.1);
    let dcr = i64::from(left.2) - i64::from(right.2);
    dy * dy + dcb * dcb + dcr * dcr
}

fn rgb12_center_error(actual: (i32, i32, i32), target: (i32, i32, i32)) -> i64 {
    let dr = i64::from(actual.0 - target.0);
    let dg = i64::from(actual.1 - target.1);
    let db = i64::from(actual.2 - target.2);
    dr * dr + dg * dg + db * db
}

fn scale_u8_to_12(value: u8) -> u16 {
    ((u16::from(value) << 4) | (u16::from(value) >> 4)).min(4095)
}

fn round_to_12(value: f64) -> u16 {
    value.round().clamp(0.0, 4095.0) as u16
}

fn av1c_config_color_444() -> avif_serialize::Av1CBox {
    let mut config = avif_serialize::Av1CBox::default();
    config.seq_profile = 2;
    config.seq_level_idx_0 = 31;
    config.seq_tier_0 = false;
    config.high_bitdepth = true;
    config.twelve_bit = true;
    config.monochrome = false;
    config.chroma_subsampling_x = false;
    config.chroma_subsampling_y = false;
    config.chroma_sample_position = 0;
    config
}

fn av1c_config_alpha_400() -> avif_serialize::Av1CBox {
    let mut config = avif_serialize::Av1CBox::default();
    config.seq_profile = 2;
    config.seq_level_idx_0 = 31;
    config.seq_tier_0 = false;
    config.high_bitdepth = true;
    config.twelve_bit = true;
    config.monochrome = true;
    config.chroma_subsampling_x = true;
    config.chroma_subsampling_y = false;
    config.chroma_sample_position = 0;
    config
}

fn inject_animation_color_track_colr(avif: &mut Vec<u8>) -> Result<(), RetrievalError> {
    let moov = find_child_box(avif, 0, avif.len(), b"moov")?;
    let moov_end = box_end(avif, moov)?;
    let trak = find_child_box(avif, moov + 8, moov_end, b"trak")?;
    let trak_end = box_end(avif, trak)?;
    let mdia = find_child_box(avif, trak + 8, trak_end, b"mdia")?;
    let mdia_end = box_end(avif, mdia)?;
    let minf = find_child_box(avif, mdia + 8, mdia_end, b"minf")?;
    let minf_end = box_end(avif, minf)?;
    let stbl = find_child_box(avif, minf + 8, minf_end, b"stbl")?;
    let stbl_end = box_end(avif, stbl)?;
    let stsd = find_child_box(avif, stbl + 8, stbl_end, b"stsd")?;
    let stsd_end = box_end(avif, stsd)?;
    let av01 = find_child_box(avif, stsd + 16, stsd_end, b"av01")?;
    let av01_end = box_end(avif, av01)?;
    let av01_child_start = av01.checked_add(86).ok_or_else(|| {
        RetrievalError::InvalidData("AVIF av01 sample entry offset overflow".to_string())
    })?;
    if av01_child_start > av01_end {
        return Err(RetrievalError::InvalidData(
            "AVIF av01 sample entry is truncated".to_string(),
        ));
    }
    if find_child_box_optional(avif, av01_child_start, av01_end, b"colr")?.is_some() {
        return Ok(());
    }

    let av1c = find_child_box(avif, av01_child_start, av01_end, b"av1C")?;
    let insert_at = av1c;
    let colr = bt601_colr_box();
    let delta = colr.len();
    avif.splice(insert_at..insert_at, colr);

    for box_start in [moov, trak, mdia, minf, stbl, stsd, av01] {
        increment_box_size(avif, box_start, delta)?;
    }
    adjust_offsets_after_insert(avif, insert_at, delta)?;
    Ok(())
}

fn bt601_colr_box() -> Vec<u8> {
    let mut colr = Vec::with_capacity(19);
    colr.extend_from_slice(&(19u32).to_be_bytes());
    colr.extend_from_slice(b"colr");
    colr.extend_from_slice(b"nclx");
    colr.extend_from_slice(
        &(avif_serialize::constants::ColorPrimaries::Bt709 as u16).to_be_bytes(),
    );
    colr.extend_from_slice(
        &(avif_serialize::constants::TransferCharacteristics::Srgb as u16).to_be_bytes(),
    );
    colr.extend_from_slice(
        &(avif_serialize::constants::MatrixCoefficients::Bt601 as u16).to_be_bytes(),
    );
    colr.push(0x80);
    colr
}

fn adjust_offsets_after_insert(
    bytes: &mut [u8],
    insert_at: usize,
    delta: usize,
) -> Result<(), RetrievalError> {
    let mut offset = 0;
    while offset + 8 <= bytes.len() {
        let end = box_end(bytes, offset)?;
        match &bytes[offset + 4..offset + 8] {
            b"meta" => adjust_meta_iloc_offsets(bytes, offset, end, insert_at, delta)?,
            b"moov" => adjust_stco_offsets_recursive(bytes, offset + 8, end, insert_at, delta)?,
            _ => {}
        }
        offset = end;
    }
    Ok(())
}

fn adjust_meta_iloc_offsets(
    bytes: &mut [u8],
    meta_start: usize,
    meta_end: usize,
    insert_at: usize,
    delta: usize,
) -> Result<(), RetrievalError> {
    let iloc = find_child_box(bytes, meta_start + 12, meta_end, b"iloc")?;
    let iloc_end = box_end(bytes, iloc)?;
    if iloc + 16 > iloc_end {
        return Err(RetrievalError::InvalidData(
            "AVIF iloc box is truncated".to_string(),
        ));
    }

    let sizes = bytes[iloc + 12];
    let offset_size = (sizes >> 4) as usize;
    let length_size = (sizes & 0x0f) as usize;
    let base_offset_size = (bytes[iloc + 13] >> 4) as usize;
    if offset_size != 4 || length_size != 4 || base_offset_size != 0 {
        return Err(RetrievalError::InvalidData(
            "AVIF iloc layout is unsupported".to_string(),
        ));
    }

    let item_count = read_u16(bytes, iloc + 14)? as usize;
    let mut cursor = iloc + 16;
    for _ in 0..item_count {
        cursor = cursor.checked_add(4).ok_or_else(offset_overflow)?;
        if cursor + 2 > iloc_end {
            return Err(RetrievalError::InvalidData(
                "AVIF iloc item is truncated".to_string(),
            ));
        }
        let extent_count = read_u16(bytes, cursor)? as usize;
        cursor += 2;
        for _ in 0..extent_count {
            adjust_u32_offset(bytes, cursor, insert_at, delta)?;
            cursor = cursor
                .checked_add(offset_size + length_size)
                .ok_or_else(offset_overflow)?;
            if cursor > iloc_end {
                return Err(RetrievalError::InvalidData(
                    "AVIF iloc extent is truncated".to_string(),
                ));
            }
        }
    }
    Ok(())
}

fn adjust_stco_offsets_recursive(
    bytes: &mut [u8],
    start: usize,
    end: usize,
    insert_at: usize,
    delta: usize,
) -> Result<(), RetrievalError> {
    let mut offset = start;
    while offset + 8 <= end {
        let child_end = box_end(bytes, offset)?;
        match &bytes[offset + 4..offset + 8] {
            b"stco" => adjust_stco_offsets(bytes, offset, child_end, insert_at, delta)?,
            b"trak" | b"mdia" | b"minf" | b"stbl" => {
                adjust_stco_offsets_recursive(bytes, offset + 8, child_end, insert_at, delta)?;
            }
            b"stsd" => {
                adjust_stco_offsets_recursive(bytes, offset + 16, child_end, insert_at, delta)?;
            }
            _ => {}
        }
        offset = child_end;
    }
    Ok(())
}

fn adjust_stco_offsets(
    bytes: &mut [u8],
    stco_start: usize,
    stco_end: usize,
    insert_at: usize,
    delta: usize,
) -> Result<(), RetrievalError> {
    if stco_start + 16 > stco_end {
        return Err(RetrievalError::InvalidData(
            "AVIF stco box is truncated".to_string(),
        ));
    }
    let entry_count = read_u32(bytes, stco_start + 12)? as usize;
    let mut cursor = stco_start + 16;
    for _ in 0..entry_count {
        if cursor + 4 > stco_end {
            return Err(RetrievalError::InvalidData(
                "AVIF stco entry is truncated".to_string(),
            ));
        }
        adjust_u32_offset(bytes, cursor, insert_at, delta)?;
        cursor += 4;
    }
    Ok(())
}

fn adjust_u32_offset(
    bytes: &mut [u8],
    offset_position: usize,
    insert_at: usize,
    delta: usize,
) -> Result<(), RetrievalError> {
    let current = read_u32(bytes, offset_position)? as usize;
    if current >= insert_at {
        let adjusted = current.checked_add(delta).ok_or_else(offset_overflow)?;
        write_u32_checked(bytes, offset_position, adjusted)?;
    }
    Ok(())
}

fn increment_box_size(
    bytes: &mut [u8],
    box_start: usize,
    delta: usize,
) -> Result<(), RetrievalError> {
    let current = read_u32(bytes, box_start)? as usize;
    let adjusted = current.checked_add(delta).ok_or_else(offset_overflow)?;
    write_u32_checked(bytes, box_start, adjusted)?;
    Ok(())
}

fn write_u32_checked(bytes: &mut [u8], offset: usize, value: usize) -> Result<(), RetrievalError> {
    let value = u32::try_from(value).map_err(|_| offset_overflow())?;
    bytes[offset..offset + 4].copy_from_slice(&value.to_be_bytes());
    Ok(())
}

fn find_child_box(
    bytes: &[u8],
    start: usize,
    end: usize,
    box_type: &[u8; 4],
) -> Result<usize, RetrievalError> {
    find_child_box_optional(bytes, start, end, box_type)?.ok_or_else(|| {
        RetrievalError::InvalidData(format!(
            "AVIF box {} not found",
            String::from_utf8_lossy(box_type)
        ))
    })
}

fn find_child_box_optional(
    bytes: &[u8],
    start: usize,
    end: usize,
    box_type: &[u8; 4],
) -> Result<Option<usize>, RetrievalError> {
    let mut offset = start;
    while offset + 8 <= end {
        let child_end = box_end(bytes, offset)?;
        if &bytes[offset + 4..offset + 8] == box_type {
            return Ok(Some(offset));
        }
        offset = child_end;
    }
    Ok(None)
}

fn box_end(bytes: &[u8], box_start: usize) -> Result<usize, RetrievalError> {
    if box_start + 8 > bytes.len() {
        return Err(RetrievalError::InvalidData(
            "AVIF box header is truncated".to_string(),
        ));
    }
    let size = read_u32(bytes, box_start)? as usize;
    if size < 8 {
        return Err(RetrievalError::InvalidData(format!(
            "AVIF box at byte {box_start} has invalid size {size}"
        )));
    }
    let end = box_start.checked_add(size).ok_or_else(offset_overflow)?;
    if end > bytes.len() {
        return Err(RetrievalError::InvalidData(format!(
            "AVIF box at byte {box_start} extends past file end"
        )));
    }
    Ok(end)
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, RetrievalError> {
    let data = bytes.get(offset..offset + 2).ok_or_else(|| {
        RetrievalError::InvalidData(format!("AVIF u16 at byte {offset} is truncated"))
    })?;
    Ok(u16::from_be_bytes([data[0], data[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, RetrievalError> {
    let data = bytes.get(offset..offset + 4).ok_or_else(|| {
        RetrievalError::InvalidData(format!("AVIF u32 at byte {offset} is truncated"))
    })?;
    Ok(u32::from_be_bytes([data[0], data[1], data[2], data[3]]))
}

fn offset_overflow() -> RetrievalError {
    RetrievalError::InvalidData("AVIF offset overflow".to_string())
}

fn avif_encoder_status(status: av1_encoder::EncoderStatus) -> RetrievalError {
    RetrievalError::InvalidData(format!("AVIF encode failed: {status}"))
}

fn lossless_encoder_config(speed: u8, threads: Option<usize>) -> avif::EncoderConfig {
    let mut config = avif::EncoderConfig::new()
        .quality(100.0)
        .alpha_quality(100.0)
        .speed(speed)
        .bit_depth(avif::EncodeBitDepth::Eight)
        .color_model(avif::EncodeColorModel::YCbCr)
        .alpha_color_mode(avif::EncodeAlphaMode::UnassociatedDirty)
        .with_qm(false)
        .with_lossless(true);
    if let Some(threads) = threads {
        config = config.threads(Some(threads));
    }
    config
}

fn decode_animation_or_still(path: &Path) -> Result<Vec<avif::PixelBuffer>, RetrievalError> {
    let bytes = fs::read(path)?;
    let config = avif::DecoderConfig::new().prefer_8bit(true).threads(1);
    match avif::decode_animation_with(&bytes, &config, &avif::Unstoppable) {
        Ok(animation) => Ok(animation
            .frames
            .into_iter()
            .map(|frame| frame.pixels)
            .collect()),
        Err(animation_error) => match avif::decode_with(&bytes, &config, &avif::Unstoppable) {
            Ok(image) => Ok(vec![image]),
            Err(still_error) => Err(RetrievalError::InvalidData(format!(
                "failed to decode {} as AVIF animation ({animation_error}) or still image ({still_error})",
                path.display()
            ))),
        },
    }
}

fn write_still_avif_checked(
    path: &Path,
    pixels: &avif::PixelBuffer,
    config: &avif::EncoderConfig,
    force: bool,
) -> Result<(), RetrievalError> {
    use almost_enough::StopExt;

    let encoded = avif::encode_with(pixels, config, almost_enough::Unstoppable.into_token())
        .map_err(|error| RetrievalError::InvalidData(format!("AVIF encode failed: {error}")))?;
    write_checked(path, &encoded.avif_file, force)
}

fn verify_frame_hash(
    kind: &str,
    record: &PackedManifestRecord,
    pixels: &avif::PixelBuffer,
    entry: &PackedImageEntry,
) -> Result<(), RetrievalError> {
    let (width, height, rgba) = crate::image::pixel_buffer_to_rgba8(pixels)?;
    if width != entry.width || height != entry.height {
        return Err(RetrievalError::InvalidData(format!(
            "{} {kind} frame dimensions for {} are {}x{}, expected {}x{}",
            record.persona, record.id, width, height, entry.width, entry.height
        )));
    }
    let actual = sha256_hex(&rgba);
    if actual != entry.rgba_sha256 {
        return Err(RetrievalError::InvalidData(format!(
            "{} {kind} frame hash mismatch for {}: expected {}, got {}",
            record.persona, record.id, entry.rgba_sha256, actual
        )));
    }
    Ok(())
}

fn read_manifest(dir: &Path) -> Result<Vec<PackedManifestRecord>, RetrievalError> {
    read_jsonl(dir.join("manifest.jsonl"))
}

fn read_pose_records(dir: &Path) -> Result<Vec<PackedPoseRecord>, RetrievalError> {
    read_jsonl(dir.join("poses.jsonl"))
}

fn validate_manifest(
    persona: &str,
    packed_dir: &Path,
    manifest: &[PackedManifestRecord],
    poses: &[PackedPoseRecord],
) -> Result<(), RetrievalError> {
    if manifest.is_empty() {
        return Err(RetrievalError::InvalidData(format!(
            "empty manifest: {}",
            packed_dir.join("manifest.jsonl").display()
        )));
    }
    let mut ids = BTreeSet::new();
    let mut pose_ids = BTreeSet::new();
    for pose in poses {
        if pose.version != DATASET_PACK_VERSION {
            return Err(RetrievalError::InvalidData(format!(
                "{} pose {} has unsupported version {}",
                persona, pose.id, pose.version
            )));
        }
        let actual = json_value_sha256(&pose.pose)?;
        if actual != pose.json_sha256 {
            return Err(RetrievalError::InvalidData(format!(
                "{} pose {} JSON hash mismatch: expected {}, got {}",
                persona, pose.id, pose.json_sha256, actual
            )));
        }
        if !pose_ids.insert(pose.id.clone()) {
            return Err(RetrievalError::InvalidData(format!(
                "{persona} duplicate pose id {}",
                pose.id
            )));
        }
    }

    for (index, record) in manifest.iter().enumerate() {
        if record.version != DATASET_PACK_VERSION {
            return Err(RetrievalError::InvalidData(format!(
                "{} manifest record {} has unsupported version {}",
                persona, record.id, record.version
            )));
        }
        if record.persona != persona {
            return Err(RetrievalError::InvalidData(format!(
                "{} manifest record {} belongs to {}",
                persona, record.id, record.persona
            )));
        }
        if !ids.insert(record.id.clone()) {
            return Err(RetrievalError::InvalidData(format!(
                "{persona} duplicate manifest id {}",
                record.id
            )));
        }
        if record.image.file != "images.avif" || record.glyph.file != "glyphs.avif" {
            return Err(RetrievalError::InvalidData(format!(
                "{} manifest record {} points to unexpected AVIF files",
                persona, record.id
            )));
        }
        if record.pose.file != "poses.jsonl" {
            return Err(RetrievalError::InvalidData(format!(
                "{} manifest record {} points to unexpected pose file {}",
                persona, record.id, record.pose.file
            )));
        }
        if record.image.frame != index || record.glyph.frame != index || record.pose.line != index {
            return Err(RetrievalError::InvalidData(format!(
                "{} manifest record {} is not in deterministic frame/line order",
                persona, record.id
            )));
        }
        if !pose_ids.contains(&record.id) {
            return Err(RetrievalError::InvalidData(format!(
                "{persona} missing pose JSONL record for {}",
                record.id
            )));
        }
        let pose = poses.get(record.pose.line).ok_or_else(|| {
            RetrievalError::InvalidData(format!(
                "{} pose line {} for {} is out of range",
                persona, record.pose.line, record.id
            ))
        })?;
        if pose.id != record.id || pose.json_sha256 != record.pose.json_sha256 {
            return Err(RetrievalError::InvalidData(format!(
                "{} pose line {} does not match manifest record {}",
                persona, record.pose.line, record.id
            )));
        }
    }

    for required in [
        "images.avif",
        "glyphs.avif",
        "manifest.jsonl",
        "poses.jsonl",
    ] {
        let path = packed_dir.join(required);
        if !path.is_file() {
            return Err(RetrievalError::InvalidData(format!(
                "{persona} missing packed file {}",
                path.display()
            )));
        }
    }

    Ok(())
}

fn ensure_frame_count(label: &str, actual: usize, expected: usize) -> Result<(), RetrievalError> {
    if actual == expected {
        Ok(())
    } else {
        Err(RetrievalError::InvalidData(format!(
            "{label} contains {actual} frames, expected {expected}"
        )))
    }
}

fn create_clean_pack_dir(path: &Path, force: bool) -> Result<(), RetrievalError> {
    if path.exists() {
        if !force {
            return Err(RetrievalError::InvalidData(format!(
                "output directory already exists: {} (pass --force to replace files)",
                path.display()
            )));
        }
        for file in [
            "images.avif",
            "glyphs.avif",
            "manifest.jsonl",
            "poses.jsonl",
        ] {
            let candidate = path.join(file);
            if candidate.exists() {
                fs::remove_file(candidate)?;
            }
        }
    }
    fs::create_dir_all(path)?;
    Ok(())
}

fn write_jsonl<T: Serialize>(path: impl AsRef<Path>, values: &[T]) -> Result<(), RetrievalError> {
    let mut file = fs::File::create(path)?;
    for value in values {
        let line = json::to_string(value)?;
        file.write_all(line.as_bytes())?;
        file.write_all(b"\n")?;
    }
    Ok(())
}

fn read_jsonl<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<Vec<T>, RetrievalError> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut values = Vec::new();
    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let mut bytes = line.into_bytes();
        let value = json::from_slice(&mut bytes).map_err(|error| {
            RetrievalError::InvalidData(format!(
                "invalid JSONL at line {}: {error}",
                line_index + 1
            ))
        })?;
        values.push(value);
    }
    Ok(values)
}

fn write_checked(path: &Path, bytes: &[u8], force: bool) -> Result<(), RetrievalError> {
    if path.exists() && !force {
        return Err(RetrievalError::InvalidData(format!(
            "output file already exists: {} (pass --force to replace files)",
            path.display()
        )));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, bytes)?;
    Ok(())
}

fn json_value_sha256(value: &json::OwnedValue) -> Result<String, RetrievalError> {
    Ok(sha256_hex(json::to_string(value)?.as_bytes()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn relative_slash_path(path: &Path, base: &Path) -> String {
    path.strip_prefix(base)
        .unwrap_or(path)
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

fn ensure_avif_speed(speed: u8) -> Result<(), RetrievalError> {
    if (1..=10).contains(&speed) {
        Ok(())
    } else {
        Err(RetrievalError::InvalidData(format!(
            "AVIF speed must be in 1..=10, got {speed}"
        )))
    }
}

struct GlyphIdMetadata {
    codepoint: Option<String>,
    character: Option<String>,
}

fn parse_glyph_id(id: &str) -> GlyphIdMetadata {
    let codepoint = id
        .split("_u")
        .nth(1)
        .and_then(|rest| rest.split('_').next())
        .map(str::to_owned);
    let character = codepoint
        .as_deref()
        .and_then(|hex| u32::from_str_radix(hex, 16).ok())
        .and_then(char::from_u32)
        .map(|ch| ch.to_string());
    GlyphIdMetadata {
        codepoint: codepoint.map(|value| format!("U+{}", value.to_uppercase())),
        character,
    }
}
