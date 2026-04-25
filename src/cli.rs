use std::env;
use std::path::{Path, PathBuf};

use chrono::Local;
use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::engine::{self, CodecChoice, PreprocOpts, RunOpts};

#[derive(Parser, Debug)]
#[command(
    name = "pdf-sizer",
    about = "Подбор PDF-вариантов из папки изображений"
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Подобрать PDF-варианты из указанной папки изображений
    Run(RunArgs),
}

#[derive(Args, Debug)]
struct RunArgs {
    /// Папка с исходными изображениями
    #[arg(long)]
    input: PathBuf,
    /// Папка для PDF-вариантов
    #[arg(long)]
    output: Option<PathBuf>,
    /// Целевой размер PDF в мегабайтах
    #[arg(long)]
    target_mb: f64,
    /// 3x3 median filter для удаления одиночного шума
    #[arg(long)]
    despeckle: bool,
    /// Background flattening; без значения используется 30
    #[arg(long, num_args = 0..=1, default_missing_value = "30")]
    flatten: Option<u8>,
    /// Выравнивание наклона страницы
    #[arg(long)]
    deskew: bool,
    /// Кодек/режим подбора вариантов
    #[arg(long, alias = "mode", value_enum, default_value_t = CodecArg::Jpeg)]
    codec: CodecArg,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CodecArg {
    Jpeg,
    Jp2,
    Mrc,
    Auto,
}

impl From<CodecArg> for CodecChoice {
    fn from(value: CodecArg) -> Self {
        match value {
            CodecArg::Jpeg => Self::Jpeg,
            CodecArg::Jp2 => Self::Jp2,
            CodecArg::Mrc => Self::Mrc,
            CodecArg::Auto => Self::Auto,
        }
    }
}

pub(crate) fn print_legacy_usage(program: &str) {
    eprintln!(
        "Использование: {} <МБ> [--despeckle] [--flatten[=<0-255>]] [--deskew] [--codec=jpeg|jp2|mrc|auto]",
        program
    );
    eprintln!("Изображения берутся из папки images/ рядом с исполняемым файлом");
}

pub(crate) fn dispatch(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Command::Run(args) => run_command(args),
    }
}

pub(crate) fn run_legacy(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let (target_mb, preproc, codec) = parse_legacy_args(args)?;
    let exe_dir = env::current_exe()?
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let opts = RunOpts {
        input: exe_dir.join("images"),
        output: PathBuf::from("."),
        target_mb,
        preproc,
        codec,
    };
    engine::run_search(&opts)?;
    Ok(())
}

fn run_command(args: RunArgs) -> Result<(), Box<dyn std::error::Error>> {
    let output = args
        .output
        .unwrap_or_else(|| default_output_dir(&args.input));
    let opts = RunOpts {
        input: args.input,
        output,
        target_mb: args.target_mb,
        preproc: PreprocOpts {
            despeckle: args.despeckle,
            flatten_threshold: args.flatten.unwrap_or(0),
            deskew: args.deskew,
        },
        codec: args.codec.into(),
    };
    engine::run_search(&opts)?;
    Ok(())
}

fn parse_legacy_args(
    args: &[String],
) -> Result<(f64, PreprocOpts, CodecChoice), Box<dyn std::error::Error>> {
    let mut target_mb_arg: Option<&str> = None;
    let mut preproc = PreprocOpts {
        despeckle: false,
        flatten_threshold: 0,
        deskew: false,
    };
    let mut codec = CodecChoice::Jpeg;

    for arg in args {
        if arg == "--despeckle" {
            preproc.despeckle = true;
        } else if arg == "--deskew" {
            preproc.deskew = true;
        } else if arg == "--flatten" {
            preproc.flatten_threshold = 30;
        } else if let Some(val) = arg.strip_prefix("--flatten=") {
            preproc.flatten_threshold = val
                .parse::<u8>()
                .map_err(|_| format!("--flatten: '{}' не является числом от 0 до 255", val))?;
        } else if let Some(val) = arg
            .strip_prefix("--codec=")
            .or_else(|| arg.strip_prefix("--mode="))
        {
            codec = match val {
                "jpeg" => CodecChoice::Jpeg,
                "jp2" => CodecChoice::Jp2,
                "mrc" => CodecChoice::Mrc,
                "auto" => CodecChoice::Auto,
                _ => {
                    return Err(
                        format!("--codec: '{}' — ожидается jpeg, jp2, mrc или auto", val).into(),
                    );
                }
            };
        } else if target_mb_arg.is_none() {
            target_mb_arg = Some(arg);
        }
    }

    let target_mb = target_mb_arg
        .unwrap_or_else(|| {
            eprintln!("Ошибка: не задан целевой размер");
            std::process::exit(1)
        })
        .parse()
        .map_err(|_| "целевой размер не является числом")?;

    Ok((target_mb, preproc, codec))
}

fn default_output_dir(input: &Path) -> PathBuf {
    let ts = Local::now().format("%Y%m%d-%H%M%S").to_string();
    input.join("pdf-sizer-output").join(ts)
}
