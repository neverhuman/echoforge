fn main() {
    std::process::exit(match echoforge_cli::run(std::env::args_os()) {
        Ok(code) => i32::from(code),
        Err(err) => {
            eprintln!("{err}");
            2
        }
    });
}
