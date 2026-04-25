mod cli;
mod engine;
mod gui;
mod progress;

use clap::Parser;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        return gui::run_gui();
    }

    // Preserve the positional target form (`pdf-sizer 20`) before clap sees it.
    if args[1].parse::<f64>().is_ok() {
        return cli::run_legacy(&args[1..]);
    }

    let cli = cli::Cli::parse();
    cli::dispatch(cli)
}
