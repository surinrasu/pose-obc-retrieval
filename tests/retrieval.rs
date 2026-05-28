use std::{
    fs,
    path::{Path, PathBuf},
};

use almost_enough::StopExt;
use ann::backend::{Autodiff, Flex};
use json::{OwnedValue, json};
use pose_obc_retrieval::{
    CandidateEntry, CandidateIndex, RetrievalModelConfig, RetrievalPairDataset,
    RetrievalTrainingConfig, build_candidate_index, extract_glyph_features_from_path,
    extract_pose_features_from_path, read_candidate_index, search_index, train_retrieval_dataset,
    write_candidate_index,
};
use rgb::Rgb;

type AB = Autodiff<Flex>;

fn fixture_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "pose_obc_retrieval_retrieval_{name}_{}",
        std::process::id()
    ))
}

fn write_fixture_image(path: &Path, variant: u8) {
    let width = 32;
    let height = 32;
    let mut pixels = vec![
        Rgb {
            r: 255,
            g: 255,
            b: 255
        };
        width * height
    ];
    for y in 6..26 {
        for x in 6..26 {
            let draw = if variant == 0 {
                (14..=18).contains(&x)
            } else {
                (14..=18).contains(&y)
            };
            if draw {
                pixels[y * width + x] = Rgb {
                    r: 20,
                    g: 20,
                    b: 20,
                };
            }
        }
    }
    write_avif(path, pixels, width as u32, height as u32);
}

fn write_avif(path: &Path, pixels: Vec<Rgb<u8>>, width: u32, height: u32) {
    let buffer: avif::PixelBuffer =
        avif::PixelBuffer::<Rgb<u8>>::from_pixels(pixels, width, height)
            .expect("fixture pixel buffer")
            .into();
    let encoded = avif::encode_with(
        &buffer,
        &lossless_avif_config(),
        almost_enough::Unstoppable.into_token(),
    )
    .expect("fixture AVIF encode");
    fs::write(path, encoded.avif_file).expect("fixture image write");
}

fn lossless_avif_config() -> avif::EncoderConfig {
    avif::EncoderConfig::new()
        .quality(100.0)
        .alpha_quality(100.0)
        .speed(10)
        .bit_depth(avif::EncodeBitDepth::Eight)
        .color_model(avif::EncodeColorModel::Rgb)
        .alpha_color_mode(avif::EncodeAlphaMode::UnassociatedDirty)
        .with_qm(false)
        .with_lossless(true)
}

fn write_fixture_pose(path: &Path, variant: u8) {
    let mut keypoints = vec![0.0; 37 * 3];
    for joint in 0..37 {
        let offset = joint * 3;
        keypoints[offset] = if variant == 0 {
            16.0
        } else {
            8.0 + joint as f32 * 0.4
        };
        keypoints[offset + 1] = if variant == 0 {
            8.0 + joint as f32 * 0.4
        } else {
            16.0
        };
        keypoints[offset + 2] = 1.0;
    }

    let pose: OwnedValue = json!({
        "version": 1.0,
        "people": [{
            "pose_keypoints_2d": keypoints
        }]
    });
    fs::write(path, json::to_string(&pose).expect("pose json")).expect("pose write");
}

fn write_fixture_dataset(root: &Path) -> PathBuf {
    let data_root = root.join("data");
    let image_dir = data_root.join("persona_fixture").join("images");
    let glyph_dir = data_root.join("persona_fixture").join("glyphs");
    let pose_dir = data_root.join("persona_fixture").join("poses");
    fs::create_dir_all(&image_dir).expect("image dir");
    fs::create_dir_all(&glyph_dir).expect("glyph dir");
    fs::create_dir_all(&pose_dir).expect("pose dir");

    for (name, variant) in [("U+4E00.avif", 0), ("U+4E8C.avif", 1)] {
        write_fixture_image(&image_dir.join(name), variant);
        write_fixture_image(&glyph_dir.join(name), variant);
        write_fixture_pose(&pose_dir.join(name.replace(".avif", ".json")), variant);
    }

    data_root
}

#[test]
fn retrieval_training_index_and_search_workflow_runs() {
    let root = fixture_dir("workflow");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("root");
    let data_root = write_fixture_dataset(&root);

    let dataset = RetrievalPairDataset::from_data_root(&data_root).expect("dataset");
    assert_eq!(dataset.len(), 2);

    let pose_features =
        extract_pose_features_from_path(&dataset.pairs()[0].image_path).expect("pose features");
    let glyph_features =
        extract_glyph_features_from_path(&dataset.pairs()[0].glyph_path).expect("glyph features");
    assert_eq!(
        pose_features.len(),
        pose_obc_retrieval::RETRIEVAL_FEATURE_DIM
    );
    assert_eq!(
        glyph_features.len(),
        pose_obc_retrieval::RETRIEVAL_FEATURE_DIM
    );

    let checkpoint_dir = root.join("checkpoints");
    let config = RetrievalTrainingConfig {
        model: RetrievalModelConfig {
            input_dim: pose_obc_retrieval::RETRIEVAL_FEATURE_DIM,
            hidden_dim: 8,
            embedding_dim: 4,
        },
        epochs: 1,
        batch_size: 2,
        learning_rate: 1e-3,
        temperature: 0.07,
        shuffle: false,
        seed: 7,
        max_pairs: None,
        checkpoint_dir,
        log_every: 0,
        save_every_epoch: false,
    };

    let device = Default::default();
    let (model, report) =
        train_retrieval_dataset::<AB, _>(config.clone(), &dataset, &device, |_| {})
            .expect("train retrieval");
    assert_eq!(report.epochs.len(), 1);

    let index = build_candidate_index(&model, config.model.clone(), &dataset, true, &device)
        .expect("candidate index");
    assert_eq!(index.entries.len(), 2);

    let index_path = root.join("glyph_index.json");
    write_candidate_index(&index_path, &index).expect("write index");
    let index = read_candidate_index(&index_path).expect("read index");
    let query = pose_obc_retrieval::encode_pose_features(&model, &pose_features, &device)
        .expect("query embedding");
    let hits = search_index(&index, &query, 1).expect("search index");
    assert_eq!(hits.len(), 1);

    let _ = fs::remove_dir_all(root);
}

#[test]
fn search_rejects_embedding_dimension_mismatch() {
    let index = CandidateIndex {
        version: pose_obc_retrieval::CANDIDATE_INDEX_VERSION,
        model: RetrievalModelConfig {
            input_dim: pose_obc_retrieval::RETRIEVAL_FEATURE_DIM,
            hidden_dim: 8,
            embedding_dim: 4,
        },
        entries: vec![CandidateEntry {
            id: "U+4E00".to_string(),
            codepoint: Some("U+4E00".to_string()),
            character: Some("一".to_string()),
            persona: "persona_fixture".to_string(),
            glyph_path: PathBuf::from("glyph.avif"),
            embedding: vec![0.1, 0.2, 0.3, 0.4],
        }],
    };

    let error = search_index(&index, &[0.1, 0.2], 1).expect_err("dimension mismatch");

    assert!(error.to_string().contains("query embedding"));
    assert!(error.to_string().contains("expected 4, got 2"));
}
