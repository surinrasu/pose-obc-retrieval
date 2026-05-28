use std::{
    fs,
    path::{Path, PathBuf},
};

use almost_enough::StopExt;
use json::{OwnedValue, json};
use pose_obc_retrieval::{
    DatasetPackOptions, DatasetUnpackOptions, DatasetVerifyOptions, RetrievalPairDataset,
    extract_glyph_features_from_path, pack_dataset, unpack_dataset, verify_packed_dataset,
};
use rgb::Rgb;

fn fixture_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "pose_obc_retrieval_dataset_pack_{name}_{}",
        std::process::id()
    ))
}

fn write_fixture_image(path: &Path, variant: u8) {
    let width = 16;
    let height = 16;
    let mut pixels = vec![
        Rgb {
            r: 255,
            g: 255,
            b: 255
        };
        width * height
    ];
    for y in 3..13 {
        for x in 3..13 {
            let draw = if variant == 0 {
                (7..=8).contains(&x)
            } else {
                (7..=8).contains(&y)
            };
            if draw {
                pixels[y * width + x] = Rgb { r: 0, g: 0, b: 0 };
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
        .color_model(avif::EncodeColorModel::YCbCr)
        .alpha_color_mode(avif::EncodeAlphaMode::UnassociatedDirty)
        .with_qm(false)
        .with_lossless(true)
}

fn write_fixture_pose(path: &Path, variant: u8) {
    let mut keypoints = vec![0.0; 37 * 3];
    for joint in 0..37 {
        let offset = joint * 3;
        keypoints[offset] = if variant == 0 {
            8.0
        } else {
            2.0 + joint as f32 * 0.2
        };
        keypoints[offset + 1] = if variant == 0 {
            2.0 + joint as f32 * 0.2
        } else {
            8.0
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
    let persona = data_root.join("persona_01");
    let image_dir = persona.join("images");
    let glyph_dir = persona.join("glyphs");
    let pose_dir = persona.join("poses");
    fs::create_dir_all(&image_dir).expect("image dir");
    fs::create_dir_all(&glyph_dir).expect("glyph dir");
    fs::create_dir_all(&pose_dir).expect("pose dir");

    for (name, variant) in [("0001_u4E00.avif", 0), ("0002_u4E8C.avif", 1)] {
        write_fixture_image(&image_dir.join(name), variant);
        write_fixture_image(&glyph_dir.join(name), variant);
        write_fixture_pose(&pose_dir.join(name.replace(".avif", ".json")), variant);
    }

    data_root
}

#[test]
fn pack_verify_and_unpack_roundtrip_runs() {
    let root = fixture_dir("roundtrip");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("root");
    let data_root = write_fixture_dataset(&root);
    let packed_root = root.join("packed");
    let unpacked_root = root.join("unpacked");

    let report = pack_dataset(&DatasetPackOptions {
        data_root: data_root.clone(),
        output_root: packed_root.clone(),
        personas: Vec::new(),
        speed: 10,
        threads: Some(1),
        force: true,
    })
    .expect("pack dataset");
    assert_eq!(report.total_samples(), 2);
    assert!(packed_root.join("persona_01/images.avif").is_file());
    assert!(packed_root.join("persona_01/glyphs.avif").is_file());
    assert!(packed_root.join("persona_01/manifest.jsonl").is_file());
    assert!(packed_root.join("persona_01/poses.jsonl").is_file());

    verify_packed_dataset(&DatasetVerifyOptions {
        packed_root: packed_root.clone(),
        personas: Vec::new(),
    })
    .expect("verify packed dataset");

    let report = unpack_dataset(&DatasetUnpackOptions {
        packed_root,
        output_root: unpacked_root.clone(),
        personas: Vec::new(),
        speed: 10,
        threads: Some(1),
        force: true,
    })
    .expect("unpack dataset");
    assert_eq!(report.total_samples(), 2);
    assert!(
        unpacked_root
            .join("persona_01/images/0001_u4E00.avif")
            .is_file()
    );
    assert!(
        unpacked_root
            .join("persona_01/glyphs/0001_u4E00.avif")
            .is_file()
    );
    assert!(
        unpacked_root
            .join("persona_01/poses/0001_u4E00.json")
            .is_file()
    );

    let dataset = RetrievalPairDataset::from_data_root(&unpacked_root).expect("unpacked dataset");
    assert_eq!(dataset.len(), 2);
    let glyph_features =
        extract_glyph_features_from_path(&dataset.pairs()[0].glyph_path).expect("glyph features");
    assert_eq!(
        glyph_features.len(),
        pose_obc_retrieval::RETRIEVAL_FEATURE_DIM
    );

    let _ = fs::remove_dir_all(root);
}
