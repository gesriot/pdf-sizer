use std::env;
use std::path::{Path, PathBuf};

use chrono::Local;
use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::engine::{self, CodecChoice, PreprocOpts, RunOpts};
use crate::progress::{
    CancellationToken, FinishStatus, LogLevel, ProgressEvent, ProgressReporter, RecommendationKind,
};

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
    /// Открыть native GUI
    Gui,
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

pub(crate) struct StdoutReporter;

impl ProgressReporter for StdoutReporter {
    fn report(&self, event: ProgressEvent) {
        match event {
            ProgressEvent::Log { level, message } => match level {
                LogLevel::Info => println!("{message}"),
                LogLevel::Warning | LogLevel::Error => eprintln!("{message}"),
            },
            ProgressEvent::Phase(phase) => {
                let _ = phase;
            }
            ProgressEvent::ImagesFound { count } => {
                let _ = count;
            }
            ProgressEvent::CurrentPage { path, index, total } => {
                let _ = (path, index, total);
            }
            ProgressEvent::SettingStarted {
                codec,
                setting,
                index,
                total,
            } => {
                let _ = (codec, setting, index, total);
            }
            ProgressEvent::Probe {
                setting,
                scale,
                size_bytes,
            } => {
                let _ = (setting, scale, size_bytes);
            }
            ProgressEvent::VariantReady(info) => {
                let _ = (
                    info.id,
                    info.filename,
                    info.codec,
                    info.setting,
                    info.scale_pct,
                    info.size_bytes,
                );
            }
            ProgressEvent::SettingSkipped {
                codec,
                setting,
                reason,
            } => {
                let _ = (codec, setting, reason);
            }
            ProgressEvent::Recommendations(recommendations) => {
                for recommendation in recommendations {
                    match recommendation {
                        RecommendationKind::BestForText(id)
                        | RecommendationKind::Balanced(id)
                        | RecommendationKind::MaxDetail(id)
                        | RecommendationKind::Smallest(id) => {
                            let _ = id;
                        }
                    }
                }
            }
            ProgressEvent::Finished { status } => match status {
                FinishStatus::Success | FinishStatus::Cancelled => {}
                FinishStatus::Failed(message) => {
                    let _ = message;
                }
            },
        }
    }
}

pub(crate) fn dispatch(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Command::Run(args) => run_command(args),
        Command::Gui => crate::gui::run_gui(),
    }
}

pub(crate) fn run_legacy(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let (target_mb, preproc, codec) = parse_legacy_args(args)?;
    let reporter = StdoutReporter;
    reporter.report(ProgressEvent::Log {
        level: LogLevel::Warning,
        message: format!(
            "pdf-sizer {} устарело, используйте pdf-sizer run --target-mb {} --input <DIR>",
            args[0], args[0]
        ),
    });
    let exe_dir = env::current_exe()?
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let opts = RunOpts {
        input: exe_dir.join("images"),
        output: PathBuf::from("."),
        target_mb,
        tolerance_mb: default_tolerance(target_mb),
        preproc,
        codec,
    };
    run_engine_with_reporter(&opts, &reporter)
}

fn run_command(args: RunArgs) -> Result<(), Box<dyn std::error::Error>> {
    let output = args
        .output
        .unwrap_or_else(|| default_output_dir(&args.input));
    let opts = RunOpts {
        target_mb: args.target_mb,
        tolerance_mb: default_tolerance(args.target_mb),
        input: args.input,
        output,
        preproc: PreprocOpts {
            despeckle: args.despeckle,
            flatten_threshold: args.flatten.unwrap_or(0),
            deskew: args.deskew,
        },
        codec: args.codec.into(),
    };
    run_engine(&opts)
}

fn run_engine(opts: &RunOpts) -> Result<(), Box<dyn std::error::Error>> {
    let reporter = StdoutReporter;
    run_engine_with_reporter(opts, &reporter)
}

fn run_engine_with_reporter(
    opts: &RunOpts,
    reporter: &dyn ProgressReporter,
) -> Result<(), Box<dyn std::error::Error>> {
    let cancel = CancellationToken::default();
    match engine::run_search(opts, reporter, &cancel) {
        Ok(_) => Ok(()),
        Err(_) => std::process::exit(1),
    }
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

/// 10 % of target — same formula the GUI uses before the user edits the field.
fn default_tolerance(target_mb: f64) -> f64 {
    target_mb * 0.1
}

fn default_output_dir(input: &Path) -> PathBuf {
    let ts = Local::now().format("%Y%m%d-%H%M%S").to_string();
    input.join("pdf-sizer-output").join(ts)
}
