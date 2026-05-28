use std::path::PathBuf;

use cli::Parser;

use super::*;

#[test]
fn parses_pose_train_command() {
    let cli = Cli::parse_from([
        "pose-obc-retrieval",
        "train",
        "pose",
        "--annotations",
        "person_keypoints_train.json",
        "--images",
        "train2017",
        "--pose-dir",
        "poses/train2017",
        "--input-size",
        "128x96",
        "--model",
        "litehrnet30",
    ]);

    let Command::Train(args) = cli.command else {
        panic!("expected train command");
    };
    let TrainTarget::Pose(args) = args.target else {
        panic!("expected train pose command");
    };
    assert_eq!(args.train_ann, PathBuf::from("person_keypoints_train.json"));
    assert_eq!(args.train_images, PathBuf::from("train2017"));
    assert_eq!(args.train_pose_dir, Some(PathBuf::from("poses/train2017")));
    assert_eq!(args.input_size.height, 128);
    assert_eq!(args.input_size.width, 96);
    assert_eq!(args.model, ModelArg::LiteHrNet30);
}

#[test]
fn parses_retrieval_train_command() {
    let cli = Cli::parse_from([
        "pose-obc-retrieval",
        "train",
        "retrieval",
        "--backend",
        "metal",
        "--epochs",
        "2",
        "--data-root",
        "data/pose-obc",
    ]);

    let Command::Train(args) = cli.command else {
        panic!("expected train command");
    };
    let TrainTarget::Retrieval(args) = args.target else {
        panic!("expected train retrieval command");
    };
    assert_eq!(args.backend, BackendArg::Metal);
    assert_eq!(args.epochs, 2);
    assert_eq!(args.data_root, PathBuf::from("data/pose-obc"));
}

#[test]
fn parses_dataset_pack_command() {
    let cli = Cli::parse_from([
        "pose-obc-retrieval",
        "dataset",
        "pack",
        "--data-root",
        "data/pose-obc",
        "--output-root",
        "data/pose-obc-packed",
        "--persona",
        "persona_01",
        "--speed",
        "6",
        "--force",
    ]);

    let Command::Dataset(args) = cli.command else {
        panic!("expected dataset command");
    };
    let DatasetTarget::Pack(args) = args.target else {
        panic!("expected dataset pack command");
    };
    assert_eq!(args.data_root, PathBuf::from("data/pose-obc"));
    assert_eq!(args.output_root, PathBuf::from("data/pose-obc-packed"));
    assert_eq!(args.personas, vec!["persona_01"]);
    assert_eq!(args.speed, 6);
    assert!(args.force);
}

#[test]
fn parses_retrieval_backend_argument() {
    let cli = Cli::parse_from([
        "pose-obc-retrieval",
        "search",
        "--backend",
        "metal",
        "--sample",
        "1",
    ]);

    let Command::Search(args) = cli.command else {
        panic!("expected retrieval search command");
    };
    assert_eq!(args.backend, BackendArg::Metal);
    assert_eq!(args.sample, Some(1));
}

#[test]
fn parses_retrieval_serve_live_argument() {
    let cli = Cli::parse_from(["pose-obc-retrieval", "serve", "--live", "--top-k", "5"]);

    let Command::Serve(args) = cli.command else {
        panic!("expected retrieval serve command");
    };
    assert!(args.live);
    assert_eq!(args.top_k, 5);
}

#[test]
fn rejects_invalid_input_size() {
    let error = Cli::try_parse_from(["pose-obc-retrieval", "train", "pose", "--input-size", "64"])
        .expect_err("invalid input size should fail");

    assert!(error.to_string().contains("HEIGHTxWIDTH"));
}

#[test]
fn rejects_validation_images_without_validation_annotations() {
    let error = Cli::try_parse_from([
        "pose-obc-retrieval",
        "train",
        "pose",
        "--annotations",
        "person_keypoints_train.json",
        "--images",
        "train2017",
        "--validation-images",
        "val2017",
    ])
    .expect_err("validation images without annotations should fail");

    assert!(error.to_string().contains("--validation-annotations"));
}
