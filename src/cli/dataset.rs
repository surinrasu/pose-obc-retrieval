use std::error::Error;

use pose_obc_retrieval::{
    DatasetPackOptions, DatasetPackReport, DatasetUnpackOptions, DatasetVerifyOptions,
    pack_dataset, unpack_dataset, verify_packed_dataset,
};

use super::args::{
    DatasetArgs, DatasetPackArgs, DatasetTarget, DatasetUnpackArgs, DatasetVerifyArgs,
};

pub(super) fn run_dataset(args: DatasetArgs) -> Result<(), Box<dyn Error>> {
    match args.target {
        DatasetTarget::Pack(args) => run_dataset_pack(args),
        DatasetTarget::Unpack(args) => run_dataset_unpack(args),
        DatasetTarget::Verify(args) => run_dataset_verify(args),
    }
}

fn run_dataset_pack(args: DatasetPackArgs) -> Result<(), Box<dyn Error>> {
    let report = pack_dataset(&DatasetPackOptions {
        data_root: args.data_root.clone(),
        output_root: args.output_root.clone(),
        personas: args.personas,
        speed: args.speed,
        threads: args.threads,
        force: args.force,
    })?;
    println!("Packed dataset");
    println!("  input: {}", args.data_root.display());
    println!("  output: {}", args.output_root.display());
    print_report(&report);
    Ok(())
}

fn run_dataset_unpack(args: DatasetUnpackArgs) -> Result<(), Box<dyn Error>> {
    let report = unpack_dataset(&DatasetUnpackOptions {
        packed_root: args.packed_root.clone(),
        output_root: args.output_root.clone(),
        personas: args.personas,
        speed: args.speed,
        threads: args.threads,
        force: args.force,
    })?;
    println!("Unpacked dataset");
    println!("  input: {}", args.packed_root.display());
    println!("  output: {}", args.output_root.display());
    print_report(&report);
    Ok(())
}

fn run_dataset_verify(args: DatasetVerifyArgs) -> Result<(), Box<dyn Error>> {
    let report = verify_packed_dataset(&DatasetVerifyOptions {
        packed_root: args.packed_root.clone(),
        personas: args.personas,
    })?;
    println!("Verified packed dataset");
    println!("  input: {}", args.packed_root.display());
    print_report(&report);
    Ok(())
}

fn print_report(report: &DatasetPackReport) {
    println!("  personas: {}", report.personas.len());
    println!("  samples: {}", report.total_samples());
    for persona in &report.personas {
        println!("    {}: {}", persona.persona, persona.samples);
    }
}
