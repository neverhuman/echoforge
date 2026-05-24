use echoforge_cli::manifest::EchoSigManifest;
use std::path::PathBuf;

fn main() {
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/sample.echosig/manifest.json");
    let manifest = EchoSigManifest::load(manifest_path);
    print!("{}", manifest.inspect().render());
}
