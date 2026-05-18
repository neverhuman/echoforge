#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    echoforge_studio::serve_from_env().await?;
    Ok(())
}
