use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut output_path = std::env::var("ECHOFORGE_DOCTOR_OUTPUT")
        .ok()
        .map(PathBuf::from);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => {
                let Some(path) = args.next() else {
                    return Err("--json requires a path".into());
                };
                output_path = Some(PathBuf::from(path));
            }
            "--help" | "-h" => {
                eprintln!("usage: gpu-doctor [--json PATH]");
                return Ok(());
            }
            _ => {}
        }
    }

    let report = echoforge_gpu_doctor::build_gpu_doctor_report();
    let payload = serde_json::to_string_pretty(&report)?;
    if let Some(path) = output_path {
        std::fs::write(path, payload.as_bytes())?;
    }
    println!("{payload}");
    Ok(())
}
