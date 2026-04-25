mod cli;
mod engine;
mod progress;

use clap::Parser;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    // Preserve the old no-args output exactly; clap would print a different help text.
    if args.len() < 2 {
        cli::print_legacy_usage(&args[0]);
        std::process::exit(1);
    }

    // Preserve the positional target form (`pdf-sizer 20`) before clap sees it.
    if args[1].parse::<f64>().is_ok() {
        return cli::run_legacy(&args[1..]);
    }

    let cli = cli::Cli::parse();
    cli::dispatch(cli)
}
