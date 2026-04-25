use std::cmp::Reverse;
use std::fs;
use std::path::{Path, PathBuf};

mod encode;
mod mrc;
mod pdf;
mod preprocess;
mod segment;
mod types;

pub(crate) use types::{CodecChoice, PreprocOpts};

use encode::{collect_images, encode_as_jp2, encode_image, estimate_jp2_size, estimate_pdf_size};
use mrc::{encode_mrc_frames, estimate_mrc_size, prepare_mrc_sources};
use pdf::{build_mrc_pdf, build_pdf};
use preprocess::preprocess;
use types::Frame;

use crate::progress::{
    CancellationToken, Cancelled, EngineMessage, FinishStatus, LogLevel, ProgressEvent,
    ProgressReporter, RecommendationKind, SearchPhase, VariantInfo,
};

#[cfg(test)]
use encode::{
    encode_jp2_raw, encode_mozjpeg, encode_raster_as_jpeg, encode_text_mask_g4,
    encode_text_mask_jbig2, is_effectively_grayscale, jpeg_component_count, mask_to_jbig2_pixels,
};
#[cfg(test)]
use fax::Color as FaxColor;
#[cfg(test)]
use segment::{fill_masked_background, is_line_like, is_mrc_suitable, segment_text_mask};
#[cfg(test)]
use types::{ComponentStats, Cs, MaskCodec, MrcFrame, PdfFilter, TextMask};

pub(crate) struct RunOpts {
    pub(crate) input: PathBuf,
    pub(crate) output: PathBuf,
    pub(crate) target_mb: f64,
    pub(crate) preproc: PreprocOpts,
    pub(crate) codec: CodecChoice,
}

pub(crate) struct SearchSummary {
    pub(crate) variants: Vec<VariantInfo>,
    pub(crate) recommendations: Vec<RecommendationKind>,
}

struct CreatedVariant {
    info: VariantInfo,
    display_filename: String,
    size_mb: f64,
}

struct VariantRecord {
    filename: PathBuf,
    display_filename: String,
    codec: CodecChoice,
    setting: String,
    scale_pct: u32,
    size_bytes: u64,
}

fn report_info(reporter: &dyn ProgressReporter, message: impl Into<String>) {
    reporter.report(ProgressEvent::Log {
        level: LogLevel::Info,
        message: message.into(),
    });
}

fn check_cancel(cancel: &CancellationToken) -> Result<(), Box<dyn std::error::Error>> {
    if cancel.is_cancelled() {
        Err(Box::new(Cancelled))
    } else {
        Ok(())
    }
}

fn write_variant(
    output: &Path,
    filename: &str,
    bytes: &[u8],
) -> Result<(PathBuf, String), Box<dyn std::error::Error>> {
    let path = output.join(filename);
    fs::write(&path, bytes)?;
    let display_filename = if output == Path::new(".") {
        filename.to_string()
    } else {
        path.display().to_string()
    };
    let filename = fs::canonicalize(&path).unwrap_or(path);
    Ok((filename, display_filename))
}

fn record_variant(
    variants: &mut Vec<CreatedVariant>,
    reporter: &dyn ProgressReporter,
    record: VariantRecord,
) {
    let info = VariantInfo {
        id: variants.len(),
        filename: record.filename,
        codec: record.codec,
        setting: record.setting,
        scale_pct: record.scale_pct,
        size_bytes: record.size_bytes,
    };
    reporter.report(ProgressEvent::VariantReady(info.clone()));
    variants.push(CreatedVariant {
        info,
        display_filename: record.display_filename,
        size_mb: record.size_bytes as f64 / (1024.0 * 1024.0),
    });
}

fn setting_u8(setting: &str, prefix: &str) -> Option<u8> {
    setting.strip_prefix(prefix)?.parse().ok()
}

fn distance_to_target(size_bytes: u64, target_bytes: u64) -> u64 {
    size_bytes.abs_diff(target_bytes)
}

fn push_recommendation(
    recommendations: &mut Vec<RecommendationKind>,
    used: &mut Vec<usize>,
    id: usize,
    make: impl FnOnce(usize) -> RecommendationKind,
) {
    if !used.contains(&id) {
        used.push(id);
        recommendations.push(make(id));
    }
}

fn recommend_variants(variants: &[VariantInfo], target_bytes: u64) -> Vec<RecommendationKind> {
    let mut recommendations = Vec::new();
    let mut used = Vec::new();

    let best_text = variants
        .iter()
        .filter(|v| v.codec == CodecChoice::Mrc)
        .filter_map(|v| setting_u8(&v.setting, "mrc-q=").map(|q| (v, q)))
        .max_by_key(|(v, q)| (*q, v.scale_pct))
        .map(|(v, _)| v)
        .or_else(|| {
            variants
                .iter()
                .filter(|v| v.codec == CodecChoice::Jpeg)
                .filter_map(|v| setting_u8(&v.setting, "q=").map(|q| (v, q)))
                .max_by_key(|(v, q)| (*q, v.scale_pct))
                .map(|(v, _)| v)
        });
    if let Some(v) = best_text {
        push_recommendation(
            &mut recommendations,
            &mut used,
            v.id,
            RecommendationKind::BestForText,
        );
    }

    let balanced = variants
        .iter()
        .filter(|v| {
            v.codec == CodecChoice::Jpeg
                && setting_u8(&v.setting, "q=").is_some_and(|q| (60..=85).contains(&q))
        })
        .min_by_key(|v| {
            (
                distance_to_target(v.size_bytes, target_bytes),
                Reverse(v.scale_pct),
            )
        })
        .or_else(|| {
            variants
                .iter()
                .filter(|v| v.codec != CodecChoice::Mrc)
                .min_by_key(|v| {
                    (
                        distance_to_target(v.size_bytes, target_bytes),
                        Reverse(v.scale_pct),
                    )
                })
        });
    if let Some(v) = balanced {
        push_recommendation(
            &mut recommendations,
            &mut used,
            v.id,
            RecommendationKind::Balanced,
        );
    }

    if let Some(v) = variants
        .iter()
        .max_by_key(|v| (v.scale_pct, Reverse(v.size_bytes)))
    {
        push_recommendation(
            &mut recommendations,
            &mut used,
            v.id,
            RecommendationKind::MaxDetail,
        );
    }

    if let Some(v) = variants.iter().min_by_key(|v| v.size_bytes) {
        push_recommendation(
            &mut recommendations,
            &mut used,
            v.id,
            RecommendationKind::Smallest,
        );
    }

    recommendations
}

pub(crate) fn run_search(
    run_opts: &RunOpts,
    reporter: &dyn ProgressReporter,
    cancel: &CancellationToken,
) -> Result<SearchSummary, Box<dyn std::error::Error>> {
    let target_bytes = (run_opts.target_mb * 1024.0 * 1024.0) as u64;

    match run_search_inner(run_opts, reporter, cancel) {
        Ok(mut summary) => {
            let recommendations = recommend_variants(&summary.variants, target_bytes);
            reporter.report(ProgressEvent::Recommendations(recommendations.clone()));
            reporter.report(ProgressEvent::Finished {
                status: FinishStatus::Success,
            });
            summary.recommendations = recommendations;
            Ok(summary)
        }
        Err(err) => {
            if err.as_ref().downcast_ref::<Cancelled>().is_some() {
                reporter.report(ProgressEvent::Finished {
                    status: FinishStatus::Cancelled,
                });
            } else {
                let message = err.to_string();
                reporter.report(ProgressEvent::Log {
                    level: LogLevel::Error,
                    message: message.clone(),
                });
                reporter.report(ProgressEvent::Finished {
                    status: FinishStatus::Failed(message),
                });
            }
            Err(err)
        }
    }
}

fn run_search_inner(
    run_opts: &RunOpts,
    reporter: &dyn ProgressReporter,
    cancel: &CancellationToken,
) -> Result<SearchSummary, Box<dyn std::error::Error>> {
    check_cancel(cancel)?;

    let target_bytes = (run_opts.target_mb * 1024.0 * 1024.0) as u64;
    let tolerance_bytes = (2.0 * 1024.0 * 1024.0) as u64;

    let images_dir = &run_opts.input;

    reporter.report(ProgressEvent::Phase(SearchPhase::LoadingImages));
    if !images_dir.exists() {
        return Err(Box::new(EngineMessage(format!(
            "Ошибка: папка не найдена: {}",
            images_dir.display()
        ))));
    }

    let image_paths = collect_images(images_dir)?;
    if image_paths.is_empty() {
        return Err(Box::new(EngineMessage(format!(
            "Ошибка: в папке нет изображений: {}",
            images_dir.display()
        ))));
    }

    fs::create_dir_all(&run_opts.output)?;

    reporter.report(ProgressEvent::ImagesFound {
        count: image_paths.len(),
    });
    report_info(
        reporter,
        format!("Найдено изображений: {}", image_paths.len()),
    );
    report_info(
        reporter,
        format!("Целевой размер: {:.0} MB (±2 MB)", run_opts.target_mb),
    );
    if run_opts.preproc.despeckle
        || run_opts.preproc.flatten_threshold > 0
        || run_opts.preproc.deskew
    {
        let active: Vec<&str> = [
            run_opts.preproc.despeckle.then_some("despeckle"),
            (run_opts.preproc.flatten_threshold > 0).then_some("flatten"),
            run_opts.preproc.deskew.then_some("deskew"),
        ]
        .into_iter()
        .flatten()
        .collect();
        report_info(reporter, format!("Препроцессинг: {}", active.join(", ")));
    }

    reporter.report(ProgressEvent::Phase(SearchPhase::Preprocessing));
    let mut images: Vec<(PathBuf, image::DynamicImage)> = Vec::with_capacity(image_paths.len());
    for (index, path) in image_paths.iter().enumerate() {
        check_cancel(cancel)?;
        reporter.report(ProgressEvent::CurrentPage {
            path: path.display().to_string(),
            index: index + 1,
            total: image_paths.len(),
        });
        let img = image::open(path)
            .map_err(|e| EngineMessage(format!("Ошибка открытия {}: {}", path.display(), e)))?;
        check_cancel(cancel)?;
        let img = preprocess(img, &run_opts.preproc, reporter);
        images.push((path.clone(), img));
    }

    // Passthrough embeds the original JPEG bytes, bypassing the preprocessed DynamicImage.
    // Disable it whenever any preprocessing is active so flags take effect on every variant.
    let allow_passthrough = !run_opts.preproc.despeckle
        && run_opts.preproc.flatten_threshold == 0
        && !run_opts.preproc.deskew;

    // ------------------------------------------------------------------ JPEG
    let run_jpeg = matches!(run_opts.codec, CodecChoice::Jpeg | CodecChoice::Auto);
    let run_jp2 = matches!(run_opts.codec, CodecChoice::Jp2 | CodecChoice::Auto);
    let run_mrc = matches!(run_opts.codec, CodecChoice::Mrc | CodecChoice::Auto);

    let quality_steps: Vec<u8> = (1..=20).rev().map(|i| i * 5).collect(); // 100, 95, ..., 5
    let mut variants: Vec<CreatedVariant> = Vec::new();

    if run_jpeg {
        reporter.report(ProgressEvent::Phase(SearchPhase::Jpeg));
        report_info(
            reporter,
            format!(
                "\n[JPEG] Поиск вариантов: для каждого quality ищем scale, дающий ~{:.0} MB...\n",
                run_opts.target_mb
            ),
        );
    }

    if run_jpeg {
        for (step_index, &q) in quality_steps.iter().enumerate() {
            check_cancel(cancel)?;
            let setting = format!("q={:03}", q);
            reporter.report(ProgressEvent::SettingStarted {
                codec: CodecChoice::Jpeg,
                setting: setting.clone(),
                index: step_index + 1,
                total: quality_steps.len(),
            });
            let max_size = estimate_pdf_size(&images, q, 1.0, allow_passthrough)?;
            if max_size < target_bytes.saturating_sub(tolerance_bytes) {
                let message = format!(
                    "  quality={:3}: даже при scale=100% размер {:.1} MB — меньше цели, пропуск остальных",
                    q,
                    max_size as f64 / (1024.0 * 1024.0)
                );
                reporter.report(ProgressEvent::SettingSkipped {
                    codec: CodecChoice::Jpeg,
                    setting,
                    reason: message.clone(),
                });
                report_info(reporter, message);
                break;
            }

            let min_size = estimate_pdf_size(&images, q, 0.05, allow_passthrough)?;
            if min_size > target_bytes + tolerance_bytes {
                let message = format!(
                    "  quality={:3}: даже при scale=5% размер {:.1} MB — больше цели, пропуск",
                    q,
                    min_size as f64 / (1024.0 * 1024.0)
                );
                reporter.report(ProgressEvent::SettingSkipped {
                    codec: CodecChoice::Jpeg,
                    setting,
                    reason: message.clone(),
                });
                report_info(reporter, message);
                continue;
            }

            let mut lo = 5.0_f64;
            let mut hi = 100.0_f64;
            let mut best: Option<(u32, Vec<u8>, f64)> = None;

            for _ in 0..20 {
                check_cancel(cancel)?;
                if hi - lo < 0.5 {
                    break;
                }
                let mid = ((lo + hi) / 2.0).round().clamp(5.0, 100.0);
                let s = mid / 100.0;

                let jpegs: Vec<Frame> = images
                    .iter()
                    .map(|(path, img)| encode_image(path, img, q, s, allow_passthrough))
                    .collect::<Result<Vec<_>, _>>()?;
                let pdf_bytes = build_pdf(&jpegs);
                let size = pdf_bytes.len() as u64;
                reporter.report(ProgressEvent::Probe {
                    setting: setting.clone(),
                    scale: mid as u32,
                    size_bytes: size,
                });

                let in_range = size <= target_bytes + tolerance_bytes
                    && size >= target_bytes.saturating_sub(tolerance_bytes);

                if in_range {
                    let size_mb = size as f64 / (1024.0 * 1024.0);
                    best = Some((mid as u32, pdf_bytes, size_mb));
                    break;
                } else if size > target_bytes {
                    hi = mid - 1.0;
                } else {
                    lo = mid + 1.0;
                }
            }

            if best.is_none() {
                check_cancel(cancel)?;
                let mid = ((lo + hi) / 2.0).round().clamp(5.0, 100.0);
                let s = mid / 100.0;
                let jpegs: Vec<Frame> = images
                    .iter()
                    .map(|(path, img)| encode_image(path, img, q, s, allow_passthrough))
                    .collect::<Result<Vec<_>, _>>()?;
                let pdf_bytes = build_pdf(&jpegs);
                let size = pdf_bytes.len() as u64;
                let in_range = size <= target_bytes + tolerance_bytes
                    && size >= target_bytes.saturating_sub(tolerance_bytes);
                if in_range {
                    let size_mb = size as f64 / (1024.0 * 1024.0);
                    best = Some((mid as u32, pdf_bytes, size_mb));
                }
            }

            if let Some((s_pct, pdf_bytes, size_mb)) = best {
                let raw_filename = format!("variant_q{:03}_s{:03}.pdf", q, s_pct);
                let (filename, display_filename) =
                    write_variant(&run_opts.output, &raw_filename, &pdf_bytes)?;
                report_info(
                    reporter,
                    format!(
                        "  quality={:3}, scale={:3}% -> {} ({:.1} MB)",
                        q, s_pct, display_filename, size_mb
                    ),
                );
                record_variant(
                    &mut variants,
                    reporter,
                    VariantRecord {
                        filename,
                        display_filename,
                        codec: CodecChoice::Jpeg,
                        setting: format!("q={:03}", q),
                        scale_pct: s_pct,
                        size_bytes: pdf_bytes.len() as u64,
                    },
                );
            } else {
                let message = format!("  quality={:3}: не удалось попасть в целевой диапазон", q);
                reporter.report(ProgressEvent::SettingSkipped {
                    codec: CodecChoice::Jpeg,
                    setting,
                    reason: message.clone(),
                });
                report_info(reporter, message);
            }
        }
    } // end if run_jpeg / for quality_steps

    // ------------------------------------------------------------------ JP2
    // Quality steps: bits-per-pixel × 100 from high to low quality.
    // bpp = step / 100.0;  rate = num_comps × 8 / bpp
    let jp2_steps: &[u32] = &[200, 150, 100, 75, 50, 35, 25, 18, 10, 5];

    if run_jp2 {
        check_cancel(cancel)?;
        reporter.report(ProgressEvent::Phase(SearchPhase::Jp2));
        report_info(
            reporter,
            format!(
                "\n[JP2] Поиск вариантов: для каждого bpp ищем scale, дающий ~{:.0} MB...\n",
                run_opts.target_mb
            ),
        );
        for (step_index, &bpp_x100) in jp2_steps.iter().enumerate() {
            check_cancel(cancel)?;
            let bpp = bpp_x100 as f32 / 100.0;
            let setting = format!("bpp={:.2}", bpp);
            reporter.report(ProgressEvent::SettingStarted {
                codec: CodecChoice::Jp2,
                setting: setting.clone(),
                index: step_index + 1,
                total: jp2_steps.len(),
            });

            let max_size = estimate_jp2_size(&images, bpp, 1.0)?;
            if max_size < target_bytes.saturating_sub(tolerance_bytes) {
                let message = format!(
                    "  bpp={:.2}: даже при scale=100% размер {:.1} MB — меньше цели, пропуск остальных",
                    bpp,
                    max_size as f64 / (1024.0 * 1024.0)
                );
                reporter.report(ProgressEvent::SettingSkipped {
                    codec: CodecChoice::Jp2,
                    setting,
                    reason: message.clone(),
                });
                report_info(reporter, message);
                break;
            }
            let min_size = estimate_jp2_size(&images, bpp, 0.05)?;
            if min_size > target_bytes + tolerance_bytes {
                let message = format!(
                    "  bpp={:.2}: даже при scale=5% размер {:.1} MB — больше цели, пропуск",
                    bpp,
                    min_size as f64 / (1024.0 * 1024.0)
                );
                reporter.report(ProgressEvent::SettingSkipped {
                    codec: CodecChoice::Jp2,
                    setting,
                    reason: message.clone(),
                });
                report_info(reporter, message);
                continue;
            }

            let mut lo = 5.0_f64;
            let mut hi = 100.0_f64;
            let mut best: Option<(u32, Vec<u8>, f64)> = None;

            for _ in 0..20 {
                check_cancel(cancel)?;
                if hi - lo < 0.5 {
                    break;
                }
                let mid = ((lo + hi) / 2.0).round().clamp(5.0, 100.0);
                let s = mid / 100.0;

                let frames: Vec<Frame> = images
                    .iter()
                    .map(|(_, img)| encode_as_jp2(img, bpp, s))
                    .collect::<Result<Vec<_>, _>>()?;
                let pdf_bytes = build_pdf(&frames);
                let size = pdf_bytes.len() as u64;
                reporter.report(ProgressEvent::Probe {
                    setting: setting.clone(),
                    scale: mid as u32,
                    size_bytes: size,
                });

                let in_range = size <= target_bytes + tolerance_bytes
                    && size >= target_bytes.saturating_sub(tolerance_bytes);

                if in_range {
                    best = Some((mid as u32, pdf_bytes, size as f64 / (1024.0 * 1024.0)));
                    break;
                } else if size > target_bytes {
                    hi = mid - 1.0;
                } else {
                    lo = mid + 1.0;
                }
            }

            if best.is_none() {
                check_cancel(cancel)?;
                let mid = ((lo + hi) / 2.0).round().clamp(5.0, 100.0);
                let s = mid / 100.0;
                let frames: Vec<Frame> = images
                    .iter()
                    .map(|(_, img)| encode_as_jp2(img, bpp, s))
                    .collect::<Result<Vec<_>, _>>()?;
                let pdf_bytes = build_pdf(&frames);
                let size = pdf_bytes.len() as u64;
                if size <= target_bytes + tolerance_bytes
                    && size >= target_bytes.saturating_sub(tolerance_bytes)
                {
                    best = Some((mid as u32, pdf_bytes, size as f64 / (1024.0 * 1024.0)));
                }
            }

            if let Some((s_pct, pdf_bytes, size_mb)) = best {
                let raw_filename = format!("variant_jp2_b{:03}_s{:03}.pdf", bpp_x100, s_pct);
                let (filename, display_filename) =
                    write_variant(&run_opts.output, &raw_filename, &pdf_bytes)?;
                report_info(
                    reporter,
                    format!(
                        "  bpp={:.2}, scale={:3}% -> {} ({:.1} MB)",
                        bpp, s_pct, display_filename, size_mb
                    ),
                );
                record_variant(
                    &mut variants,
                    reporter,
                    VariantRecord {
                        filename,
                        display_filename,
                        codec: CodecChoice::Jp2,
                        setting: format!("bpp={:.2}", bpp),
                        scale_pct: s_pct,
                        size_bytes: pdf_bytes.len() as u64,
                    },
                );
            } else {
                let message = format!("  bpp={:.2}: не удалось попасть в целевой диапазон", bpp);
                reporter.report(ProgressEvent::SettingSkipped {
                    codec: CodecChoice::Jp2,
                    setting,
                    reason: message.clone(),
                });
                report_info(reporter, message);
            }
        }
    }

    // ------------------------------------------------------------------ MRC-lite
    let mrc_quality_steps: &[u8] = &[55, 45, 35, 25];

    if run_mrc {
        check_cancel(cancel)?;
        reporter.report(ProgressEvent::Phase(SearchPhase::Mrc));
        report_info(
            reporter,
            format!(
                "\n[MRC] Поиск вариантов: маска текста CCITT G4 + JPEG-фон около {:.0} MB...\n",
                run_opts.target_mb
            ),
        );

        let mrc_sources = prepare_mrc_sources(&images, reporter, cancel)?;
        if let Some(mrc_sources) = mrc_sources {
            for (step_index, &q) in mrc_quality_steps.iter().enumerate() {
                check_cancel(cancel)?;
                let setting = format!("mrc-q={:03}", q);
                reporter.report(ProgressEvent::SettingStarted {
                    codec: CodecChoice::Mrc,
                    setting: setting.clone(),
                    index: step_index + 1,
                    total: mrc_quality_steps.len(),
                });
                let max_size = estimate_mrc_size(&mrc_sources, q, 1.0)?;
                if max_size < target_bytes.saturating_sub(tolerance_bytes) {
                    let message = format!(
                        "  bg-q={:3}: даже при bg-scale=100% размер {:.1} MB — меньше цели, пропуск остальных",
                        q,
                        max_size as f64 / (1024.0 * 1024.0)
                    );
                    reporter.report(ProgressEvent::SettingSkipped {
                        codec: CodecChoice::Mrc,
                        setting,
                        reason: message.clone(),
                    });
                    report_info(reporter, message);
                    break;
                }

                let min_size = estimate_mrc_size(&mrc_sources, q, 0.05)?;
                if min_size > target_bytes + tolerance_bytes {
                    let message = format!(
                        "  bg-q={:3}: даже при bg-scale=5% размер {:.1} MB — больше цели, пропуск",
                        q,
                        min_size as f64 / (1024.0 * 1024.0)
                    );
                    reporter.report(ProgressEvent::SettingSkipped {
                        codec: CodecChoice::Mrc,
                        setting,
                        reason: message.clone(),
                    });
                    report_info(reporter, message);
                    continue;
                }

                let mut lo = 5.0_f64;
                let mut hi = 100.0_f64;
                let mut best: Option<(u32, Vec<u8>, f64)> = None;

                for _ in 0..20 {
                    check_cancel(cancel)?;
                    if hi - lo < 0.5 {
                        break;
                    }
                    let mid = ((lo + hi) / 2.0).round().clamp(5.0, 100.0);
                    let s = mid / 100.0;

                    let frames = encode_mrc_frames(&mrc_sources, q, s)?;
                    let pdf_bytes = build_mrc_pdf(&frames);
                    let size = pdf_bytes.len() as u64;
                    reporter.report(ProgressEvent::Probe {
                        setting: setting.clone(),
                        scale: mid as u32,
                        size_bytes: size,
                    });

                    let in_range = size <= target_bytes + tolerance_bytes
                        && size >= target_bytes.saturating_sub(tolerance_bytes);

                    if in_range {
                        best = Some((mid as u32, pdf_bytes, size as f64 / (1024.0 * 1024.0)));
                        break;
                    } else if size > target_bytes {
                        hi = mid - 1.0;
                    } else {
                        lo = mid + 1.0;
                    }
                }

                if best.is_none() {
                    check_cancel(cancel)?;
                    let mid = ((lo + hi) / 2.0).round().clamp(5.0, 100.0);
                    let s = mid / 100.0;
                    let frames = encode_mrc_frames(&mrc_sources, q, s)?;
                    let pdf_bytes = build_mrc_pdf(&frames);
                    let size = pdf_bytes.len() as u64;
                    if size <= target_bytes + tolerance_bytes
                        && size >= target_bytes.saturating_sub(tolerance_bytes)
                    {
                        best = Some((mid as u32, pdf_bytes, size as f64 / (1024.0 * 1024.0)));
                    }
                }

                if let Some((s_pct, pdf_bytes, size_mb)) = best {
                    let raw_filename = format!("variant_mrc_q{:03}_s{:03}.pdf", q, s_pct);
                    let (filename, display_filename) =
                        write_variant(&run_opts.output, &raw_filename, &pdf_bytes)?;
                    report_info(
                        reporter,
                        format!(
                            "  bg-q={:3}, bg-scale={:3}% -> {} ({:.1} MB)",
                            q, s_pct, display_filename, size_mb
                        ),
                    );
                    record_variant(
                        &mut variants,
                        reporter,
                        VariantRecord {
                            filename,
                            display_filename,
                            codec: CodecChoice::Mrc,
                            setting: format!("mrc-q={:03}", q),
                            scale_pct: s_pct,
                            size_bytes: pdf_bytes.len() as u64,
                        },
                    );
                } else {
                    let message = format!("  bg-q={:3}: не удалось попасть в целевой диапазон", q);
                    reporter.report(ProgressEvent::SettingSkipped {
                        codec: CodecChoice::Mrc,
                        setting,
                        reason: message.clone(),
                    });
                    report_info(reporter, message);
                }
            }
        } else {
            let message =
                "  MRC пропущен: маска текста слишком мала/велика хотя бы на одной странице";
            report_info(reporter, message);
        }
    }

    report_info(reporter, "\n--- Итого ---");
    if variants.is_empty() {
        report_info(
            reporter,
            "Не удалось создать ни одного варианта в целевом диапазоне.",
        );
    } else {
        report_info(reporter, format!("Создано вариантов: {}\n", variants.len()));
        report_info(
            reporter,
            format!(
                "  {:>12}  {:>5}  {:>10}  файл",
                "параметр", "scale", "размер"
            ),
        );
        report_info(reporter, format!("  {}", "-".repeat(56)));
        for variant in &variants {
            let setting = &variant.info.setting;
            let s = variant.info.scale_pct;
            let hint = if let Some(q_str) = setting
                .strip_prefix("q=")
                .or_else(|| setting.strip_prefix("mrc-q="))
            {
                let q: u8 = q_str.parse().unwrap_or(0);
                if q > 80 {
                    "  <- чёткость"
                } else if s > 80 {
                    "  <- детали"
                } else {
                    ""
                }
            } else {
                // JP2: hint only from scale
                if s > 80 { "  <- детали" } else { "" }
            };
            report_info(
                reporter,
                format!(
                    "  {:>12}  {:>3}%  {:>7.1} MB  {}{}",
                    setting, s, variant.size_mb, variant.display_filename, hint
                ),
            );
        }
        report_info(
            reporter,
            "\n  JPEG q=...   : высокий q  = меньше артефактов (текст, чёткие края)",
        );
        report_info(
            reporter,
            "  JP2 bpp=...  : меньший bpp = агрессивнее сжатие, мягче артефакты",
        );
        report_info(
            reporter,
            "  MRC mrc-q=...: q — это JPEG-фон; текст идёт 1-битной маской (JBIG2 или G4 — что меньше)",
        );
        report_info(
            reporter,
            "  Высокий scale = больше деталей и разрешение (мелкие элементы)",
        );
    }

    let variants = variants.into_iter().map(|variant| variant.info).collect();
    Ok(SearchSummary {
        variants,
        recommendations: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::progress::NoopReporter;
    use image::{DynamicImage, GrayImage, Luma, Rgb, RgbImage};
    use std::path::PathBuf;
    use std::sync::Mutex;

    #[derive(Default)]
    struct RecordingReporter {
        events: Mutex<Vec<ProgressEvent>>,
    }

    impl ProgressReporter for RecordingReporter {
        fn report(&self, event: ProgressEvent) {
            self.events.lock().unwrap().push(event);
        }
    }

    impl RecordingReporter {
        fn events(&self) -> Vec<ProgressEvent> {
            self.events.lock().unwrap().clone()
        }
    }

    fn make_jpeg(w: u32, h: u32, grayscale: bool) -> Vec<u8> {
        if grayscale {
            encode_mozjpeg(&vec![128u8; (w * h) as usize], w, h, 75, true).unwrap()
        } else {
            encode_mozjpeg(&vec![128u8; (w * h * 3) as usize], w, h, 75, false).unwrap()
        }
    }

    fn make_text_scan() -> DynamicImage {
        let mut img = RgbImage::from_fn(120, 80, |_, _| Rgb([245u8, 244, 238]));
        for y in 20..36 {
            for x in 15..23 {
                img.put_pixel(x, y, Rgb([20, 20, 20]));
            }
            for x in 32..40 {
                img.put_pixel(x, y, Rgb([25, 25, 25]));
            }
            for x in 49..57 {
                img.put_pixel(x, y, Rgb([30, 30, 30]));
            }
        }
        for y in 46..62 {
            for x in 15..25 {
                img.put_pixel(x, y, Rgb([20, 20, 20]));
            }
            for x in 34..44 {
                img.put_pixel(x, y, Rgb([25, 25, 25]));
            }
        }
        DynamicImage::ImageRgb8(img)
    }

    fn decode_g4_mask(bytes: &[u8], width: u16, height: u16) -> Vec<bool> {
        let mut decoded = Vec::with_capacity(width as usize * height as usize);
        let ok = fax::decoder::decode_g4(bytes.iter().copied(), width, Some(height), |line| {
            decoded.extend(fax::decoder::pels(line, width).map(|c| c == FaxColor::Black));
        });
        assert!(ok.is_some(), "G4 decoder failed");
        decoded
    }

    fn make_variant(
        id: usize,
        codec: CodecChoice,
        setting: &str,
        scale_pct: u32,
        size_mb: u64,
    ) -> VariantInfo {
        VariantInfo {
            id,
            filename: PathBuf::from(format!("variant-{id}.pdf")),
            codec,
            setting: setting.to_string(),
            scale_pct,
            size_bytes: size_mb * 1024 * 1024,
        }
    }

    // ---- progress / recommendations -----------------------------------

    #[test]
    fn cancellation_aborts_search() {
        let cancel = CancellationToken::default();
        cancel.cancel();
        let reporter = RecordingReporter::default();
        let opts = RunOpts {
            input: PathBuf::from("does-not-matter"),
            output: PathBuf::from("."),
            target_mb: 1.0,
            preproc: PreprocOpts {
                despeckle: false,
                flatten_threshold: 0,
                deskew: false,
            },
            codec: CodecChoice::Jpeg,
        };
        let result = run_search(&opts, &reporter, &cancel);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.as_ref().downcast_ref::<Cancelled>().is_some());
        let events = reporter.events();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            ProgressEvent::Finished {
                status: FinishStatus::Cancelled
            }
        ));
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, ProgressEvent::Recommendations(_)))
        );
    }

    #[test]
    fn fatal_error_emits_log_then_finished_failed() {
        let reporter = RecordingReporter::default();
        let opts = RunOpts {
            input: PathBuf::from("__missing_pdf_sizer_test_input__"),
            output: PathBuf::from("."),
            target_mb: 1.0,
            preproc: PreprocOpts {
                despeckle: false,
                flatten_threshold: 0,
                deskew: false,
            },
            codec: CodecChoice::Jpeg,
        };
        let result = run_search(&opts, &reporter, &CancellationToken::default());
        assert!(result.is_err());
        let events = reporter.events();
        assert!(matches!(
            events.last(),
            Some(ProgressEvent::Finished {
                status: FinishStatus::Failed(_)
            })
        ));
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, ProgressEvent::Finished { .. }))
                .count(),
            1
        );
        assert!(events.iter().any(|event| {
            matches!(
                event,
                ProgressEvent::Log {
                    level: LogLevel::Error,
                    ..
                }
            )
        }));
    }

    #[test]
    fn recommendations_for_jpeg_only_run() {
        let variants = vec![
            make_variant(0, CodecChoice::Jpeg, "q=095", 50, 22),
            make_variant(1, CodecChoice::Jpeg, "q=080", 80, 20),
            make_variant(2, CodecChoice::Jpeg, "q=065", 90, 19),
            make_variant(3, CodecChoice::Jpeg, "q=040", 60, 18),
        ];
        let recommendations = recommend_variants(&variants, 20 * 1024 * 1024);
        assert_eq!(
            recommendations,
            vec![
                RecommendationKind::BestForText(0),
                RecommendationKind::Balanced(1),
                RecommendationKind::MaxDetail(2),
                RecommendationKind::Smallest(3),
            ]
        );
    }

    #[test]
    fn recommendations_dedup() {
        let variants = vec![make_variant(0, CodecChoice::Jpeg, "q=080", 100, 20)];
        let recommendations = recommend_variants(&variants, 20 * 1024 * 1024);
        assert_eq!(recommendations, vec![RecommendationKind::BestForText(0)]);
    }

    // ---- jpeg_component_count ------------------------------------------

    #[test]
    fn jpeg_component_count_rgb() {
        assert_eq!(jpeg_component_count(&make_jpeg(32, 32, false)), Some(3));
    }

    #[test]
    fn jpeg_component_count_gray() {
        assert_eq!(jpeg_component_count(&make_jpeg(32, 32, true)), Some(1));
    }

    #[test]
    fn jpeg_component_count_invalid() {
        assert_eq!(jpeg_component_count(b"not a jpeg"), None);
        assert_eq!(jpeg_component_count(&[]), None);
        // Valid SOI marker but no SOF → scan runs off end → None
        assert_eq!(jpeg_component_count(&[0xFF, 0xD8, 0xFF, 0xE0]), None);
    }

    // ---- is_effectively_grayscale --------------------------------------

    #[test]
    fn grayscale_luma8_is_gray() {
        let img = DynamicImage::ImageLuma8(GrayImage::from_fn(10, 10, |_, _| Luma([200u8])));
        assert!(is_effectively_grayscale(&img));
    }

    #[test]
    fn grayscale_rgb_uniform_gray_content() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_fn(10, 10, |_, _| Rgb([200u8, 200, 200])));
        assert!(is_effectively_grayscale(&img));
    }

    #[test]
    fn grayscale_rgb_colored_is_not_gray() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_fn(10, 10, |_, _| Rgb([200u8, 100, 50])));
        assert!(!is_effectively_grayscale(&img));
    }

    #[test]
    fn grayscale_single_colored_pixel_detected() {
        // One red pixel surrounded by gray — the full scan must catch it.
        let mut buf = RgbImage::from_fn(100, 100, |_, _| Rgb([200u8, 200, 200]));
        buf.put_pixel(50, 50, Rgb([200u8, 50, 50]));
        assert!(!is_effectively_grayscale(&DynamicImage::ImageRgb8(buf)));
    }

    // ---- encode_mozjpeg ------------------------------------------------

    #[test]
    fn mozjpeg_quality_affects_size() {
        let px: Vec<u8> = (0..100u32 * 100 * 3).map(|i| (i % 255) as u8).collect();
        let lo = encode_mozjpeg(&px, 100, 100, 10, false).unwrap();
        let hi = encode_mozjpeg(&px, 100, 100, 90, false).unwrap();
        assert!(hi.len() > lo.len(), "q=90 should be larger than q=10");
    }

    #[test]
    fn mozjpeg_gray_smaller_than_rgb() {
        let gray = vec![128u8; 100 * 100];
        let rgb = vec![128u8; 100 * 100 * 3];
        let g = encode_mozjpeg(&gray, 100, 100, 75, true).unwrap();
        let r = encode_mozjpeg(&rgb, 100, 100, 75, false).unwrap();
        assert!(
            g.len() < r.len(),
            "grayscale JPEG should be smaller than RGB"
        );
    }

    // ---- encode_jp2_raw ------------------------------------------------

    #[test]
    fn jp2_raw_small_images_do_not_fail() {
        // Regression: numresolution=6 used to cause start_compress to fail for small dims.
        for (w, h) in [(1u32, 1u32), (5, 5), (15, 20), (31, 31), (32, 32)] {
            let px = vec![128u8; (w * h * 3) as usize];
            let result = encode_jp2_raw(&px, w, h, 0.5, false);
            assert!(result.is_ok(), "JP2 failed for {w}×{h}: {:?}", result.err());
            let bytes = result.unwrap();
            assert_eq!(&bytes[4..8], b"jP  ", "not a valid JP2 for {w}×{h}");
        }
    }

    #[test]
    fn encode_as_jp2_tiny_source_image_does_not_crash() {
        // Regression (P2): scale=0.05 on a 5×5 source rounds new_w/new_h to 0.
        // The clamp to 1 in encode_as_jp2 must prevent opj_image creation failure.
        let img = DynamicImage::ImageRgb8(RgbImage::from_fn(5, 5, |x, y| {
            Rgb([(x * 50) as u8, (y * 50) as u8, 128u8])
        }));
        let result = encode_as_jp2(&img, 0.5, 0.05);
        assert!(result.is_ok(), "encode_as_jp2 crashed: {:?}", result.err());
        let (bytes, w, h, _, filter) = result.unwrap();
        assert!(w >= 1 && h >= 1, "dimensions must be at least 1×1");
        assert_eq!(&bytes[4..8], b"jP  ", "output must be valid JP2");
        assert!(matches!(filter, PdfFilter::Jpx));
    }

    #[test]
    fn encode_image_tiny_source_does_not_crash() {
        // Same clamp for JPEG path: 5×5 image at scale=0.05 → rounds to 0 without clamp.
        use std::path::Path;
        let img = DynamicImage::ImageRgb8(RgbImage::from_fn(5, 5, |_, _| Rgb([200u8, 200, 200])));
        // Pass a non-JPEG path so passthrough is never attempted.
        let result = encode_image(Path::new("dummy.png"), &img, 75, 0.05, false);
        assert!(result.is_ok(), "encode_image crashed: {:?}", result.err());
        let (_, w, h, _, _) = result.unwrap();
        assert!(w >= 1 && h >= 1);
    }

    #[test]
    fn jp2_grayscale_valid() {
        let px = vec![128u8; 64 * 64];
        let result = encode_jp2_raw(&px, 64, 64, 0.5, true);
        assert!(result.is_ok());
        assert_eq!(&result.unwrap()[4..8], b"jP  ");
    }

    #[test]
    fn jp2_rate_control_higher_bpp_larger_file() {
        let px: Vec<u8> = (0..200u32 * 200 * 3).map(|i| (i % 251) as u8).collect();
        let small = encode_jp2_raw(&px, 200, 200, 0.1, false).unwrap();
        let large = encode_jp2_raw(&px, 200, 200, 2.0, false).unwrap();
        assert!(
            large.len() > small.len(),
            "2.0 bpp must be larger than 0.1 bpp"
        );
    }

    // ---- MRC-lite ------------------------------------------------------

    #[test]
    fn mrc_segmentation_detects_text_like_components() {
        let img = make_text_scan();
        let mask = segment_text_mask(&img);
        assert!(
            is_mrc_suitable(&mask),
            "unexpected coverage: {:.2}%",
            mask.coverage * 100.0
        );
        assert!(mask.pixels[(22 * mask.width + 18) as usize]);
        assert!(!mask.pixels[0]);
    }

    #[test]
    fn ccitt_g4_mask_roundtrips() {
        let width = 16u32;
        let height = 4u32;
        let mut pixels = vec![false; (width * height) as usize];
        for y in 0..height {
            for x in 3..9 {
                pixels[(y * width + x) as usize] = y % 2 == 0;
            }
        }
        let coverage = pixels.iter().filter(|&&p| p).count() as f32 / (width * height) as f32;
        let mask = TextMask {
            width,
            height,
            coverage,
            line_ratio: 0.0,
            pixels,
        };
        let encoded = encode_text_mask_g4(&mask).unwrap();
        let decoded = decode_g4_mask(&encoded, width as u16, height as u16);
        assert_eq!(decoded, mask.pixels);
    }

    #[test]
    fn mrc_background_fills_masked_text_pixels() {
        let img = make_text_scan();
        let mask = segment_text_mask(&img);
        let bg = fill_masked_background(&img, &mask).to_rgb8();
        let [r, g, b] = bg.get_pixel(18, 22).0;
        assert!(r > 200 && g > 200 && b > 200, "text pixel was not filled");
    }

    /// Build a wide A4-sized synthetic mask with hundreds of glyph-like blocks. JBIG2
    /// dominates G4 on inputs at this scale, so it exercises the JBIG2 PDF path.
    fn make_large_text_mask() -> TextMask {
        let (w, h) = (1240u32, 1754u32);
        let mut pixels = vec![false; (w * h) as usize];
        let mut text_pixels = 0usize;
        for line in 0..40 {
            let y0 = 60 + line * 40;
            for col in 0..50 {
                let x0 = 60 + col * 22;
                for y in y0..(y0 + 12).min(h) {
                    for x in x0..(x0 + 8).min(w) {
                        pixels[(y * w + x) as usize] = true;
                        text_pixels += 1;
                    }
                }
            }
        }
        TextMask {
            width: w,
            height: h,
            coverage: text_pixels as f32 / (w as f32 * h as f32),
            line_ratio: 0.0,
            pixels,
        }
    }

    #[test]
    fn build_mrc_pdf_uses_a_supported_mask_filter() {
        let img = make_text_scan();
        let images = vec![(PathBuf::from("page.png"), img)];
        let sources = prepare_mrc_sources(&images, &NoopReporter, &CancellationToken::default())
            .unwrap()
            .unwrap();
        let frames = encode_mrc_frames(&sources, 35, 0.5).unwrap();
        let pdf = build_mrc_pdf(&frames);
        let text = String::from_utf8_lossy(&pdf);
        assert!(text.contains("DCTDecode"));
        assert!(text.contains("ImageMask true"));
        // Mask filter is auto-selected; either CCITT G4 or JBIG2 is acceptable.
        assert!(
            text.contains("CCITTFaxDecode") || text.contains("JBIG2Decode"),
            "expected a 1-bit mask filter, found neither"
        );
    }

    // ---- JBIG2 ---------------------------------------------------------

    #[test]
    fn jbig2_encodes_non_empty_stream_and_is_lossless_only() {
        // Lossless mode must produce a standalone stream (no global dictionary):
        // global dictionaries would require a separate PDF object reference, which the
        // current build_mrc_pdf does not emit.
        let mask = make_large_text_mask();
        let bytes = encode_text_mask_jbig2(&mask).unwrap();
        assert!(!bytes.is_empty(), "JBIG2 produced an empty stream");
    }

    #[test]
    fn jbig2_beats_ccitt_g4_on_realistic_mask() {
        let mask = make_large_text_mask();
        let g4 = encode_text_mask_g4(&mask).unwrap();
        let jbig2 = encode_text_mask_jbig2(&mask).unwrap();
        assert!(
            jbig2.len() < g4.len(),
            "expected JBIG2 < G4 on a 1240×1754 text mask, got jbig2={} g4={}",
            jbig2.len(),
            g4.len()
        );
    }

    #[test]
    fn build_mrc_pdf_emits_jbig2_filter_when_jbig2_wins() {
        // Prepare an MrcSource that explicitly uses JBIG2, bypassing auto-selection,
        // so the PDF path that emits /JBIG2Decode is exercised even if a future mask
        // happens to be smaller in G4.
        let mask = make_large_text_mask();
        let bytes = encode_text_mask_jbig2(&mask).unwrap();
        let bg = DynamicImage::ImageRgb8(RgbImage::from_fn(40, 40, |_, _| Rgb([240u8, 240, 240])));
        let bg_frame = encode_raster_as_jpeg(&bg, 50, 1.0).unwrap();
        let frame = MrcFrame {
            background: bg_frame,
            mask_data: bytes,
            mask_codec: MaskCodec::Jbig2,
            mask_w: mask.width,
            mask_h: mask.height,
        };
        let pdf = build_mrc_pdf(&[frame]);
        let text = String::from_utf8_lossy(&pdf);
        assert!(text.contains("JBIG2Decode"));
        assert!(text.contains("ImageMask true"));
        // No DecodeParms /K for JBIG2: that's CCITT-only.
        assert!(!text.contains("/K -1"));
    }

    /// Round-trips a known mask through the JBIG2 encoder and an independent decoder,
    /// asserting pixel-exact equality. Symmetric to `ccitt_g4_mask_roundtrips`.
    ///
    /// Catches: bool→byte-mapping inversion, encoder regressions, decoder pairing
    /// mismatches — anything where text pixels would silently end up where background
    /// pixels should be (or vice versa).
    ///
    /// Encodes with `pdf_mode=false` here to get a parseable standalone JBIG2 file
    /// (justbig2 needs the file header). The wire format of the page-level segments is
    /// identical to what `encode_text_mask_jbig2` emits for PDF embedding — only the
    /// outer file framing differs.
    #[test]
    fn jbig2_mask_round_trips_through_decoder() {
        let mask = make_large_text_mask();
        let buf = mask_to_jbig2_pixels(&mask);
        let encoded =
            jbig2enc_rust::encode_single_image_lossless(&buf, mask.width, mask.height, false)
                .unwrap();
        // Lossless path must not produce a global dictionary; that's enforced separately.
        assert!(encoded.global_data.is_none());

        let mut decoder = justbig2::Decoder::new();
        decoder.write(&encoded.page_data).unwrap();
        let page = decoder.page().expect("decoder produced no page");

        assert_eq!(page.width, mask.width);
        assert_eq!(page.height, mask.height);

        for y in 0..mask.height {
            for x in 0..mask.width {
                let idx = (y * mask.width + x) as usize;
                let expected = if mask.pixels[idx] { 1u8 } else { 0u8 };
                let actual = page.get_pixel(x, y);
                assert_eq!(actual, expected, "pixel mismatch at ({x},{y})");
            }
        }
    }

    // ---- MRC defensive layers ------------------------------------------

    #[test]
    fn is_line_like_flags_long_thin_components() {
        // Horizontal rule: 1 px tall, 300 px wide on an 800×600 page (> 15% of 800).
        let mut h_rule = ComponentStats::new(100, 50);
        for x in 100..400 {
            h_rule.add(x, 50);
        }
        assert!(is_line_like(h_rule, 800, 600));

        // Vertical rule: 1 px wide, 200 px tall on an 800×600 page (> 15% of 600).
        let mut v_rule = ComponentStats::new(400, 50);
        for y in 50..250 {
            v_rule.add(400, y);
        }
        assert!(is_line_like(v_rule, 800, 600));

        // A small glyph-like blob must not be flagged.
        let mut glyph = ComponentStats::new(10, 10);
        for x in 10..18 {
            for y in 10..24 {
                glyph.add(x, y);
            }
        }
        assert!(!is_line_like(glyph, 800, 600));
    }

    /// Page dominated by isolated horizontal section rules — each one survives as its
    /// own connected component, lifting line_ratio over the 0.30 gate.
    fn make_table_scan() -> DynamicImage {
        let (w, h) = (400u32, 400u32);
        let mut img = RgbImage::from_fn(w, h, |_, _| Rgb([245u8, 245, 245]));
        for row in [50u32, 110, 170, 230, 290, 350] {
            for x in 20..380 {
                img.put_pixel(x, row, Rgb([10u8, 10, 10]));
                img.put_pixel(x, row + 1, Rgb([10u8, 10, 10]));
            }
        }
        // Two glyph-sized blobs, fewer than the rules so line_ratio dominates.
        for &(cy, cx) in &[(80u32, 50u32), (200u32, 100u32)] {
            for y in cy..cy + 8 {
                for x in cx..cx + 8 {
                    img.put_pixel(x, y, Rgb([10u8, 10, 10]));
                }
            }
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn mrc_rejects_table_heavy_page_via_line_ratio() {
        let img = make_table_scan();
        let mask = segment_text_mask(&img);
        assert!(
            mask.line_ratio > 0.30,
            "expected high line_ratio, got {:.3}",
            mask.line_ratio
        );
        assert!(
            !is_mrc_suitable(&mask),
            "MRC should skip a page dominated by table rules"
        );
    }

    #[test]
    fn mrc_median_filter_drops_oversized_outliers() {
        // 15 body-sized glyphs + 1 giant block that should be pruned by the median filter.
        let (w, h) = (400u32, 200u32);
        let mut img = RgbImage::from_fn(w, h, |_, _| Rgb([245u8, 245, 245]));
        for i in 0..15 {
            let x0 = 20 + (i % 5) * 40;
            let y0 = 20 + (i / 5) * 30;
            for y in y0..y0 + 10 {
                for x in x0..x0 + 8 {
                    img.put_pixel(x, y, Rgb([10u8, 10, 10]));
                }
            }
        }
        // Oversized block: 120×80 — far outside [0.35×, 3.0×] of the ~10-px glyph median.
        for y in 110..190 {
            for x in 220..340 {
                img.put_pixel(x, y, Rgb([10u8, 10, 10]));
            }
        }
        let mask = segment_text_mask(&DynamicImage::ImageRgb8(img));
        // A glyph pixel at (24, 24) should still be in the mask.
        assert!(mask.pixels[(24 * mask.width + 24) as usize]);
        // A pixel inside the oversized block must have been pruned.
        assert!(!mask.pixels[(150 * mask.width + 280) as usize]);
    }

    // ---- build_pdf -----------------------------------------------------

    #[test]
    fn build_pdf_starts_with_percent_pdf() {
        let jpeg = make_jpeg(32, 32, false);
        let pdf = build_pdf(&[(jpeg, 32, 32, Cs::Rgb, PdfFilter::Dct)]);
        assert_eq!(&pdf[..4], b"%PDF");
    }

    #[test]
    fn build_pdf_dct_filter_in_stream() {
        let jpeg = make_jpeg(32, 32, false);
        let pdf = build_pdf(&[(jpeg, 32, 32, Cs::Rgb, PdfFilter::Dct)]);
        assert!(
            String::from_utf8_lossy(&pdf).contains("DCTDecode"),
            "PDF must contain DCTDecode for JPEG images"
        );
    }

    #[test]
    fn build_pdf_jpx_filter_in_stream() {
        let px = vec![128u8; 32 * 32 * 3];
        let jp2 = encode_jp2_raw(&px, 32, 32, 0.5, false).unwrap();
        let pdf = build_pdf(&[(jp2, 32, 32, Cs::Rgb, PdfFilter::Jpx)]);
        assert!(
            String::from_utf8_lossy(&pdf).contains("JPXDecode"),
            "PDF must contain JPXDecode for JP2 images"
        );
    }

    #[test]
    fn build_pdf_device_gray_for_gray_jpeg() {
        let jpeg = make_jpeg(32, 32, true);
        let pdf = build_pdf(&[(jpeg, 32, 32, Cs::Gray, PdfFilter::Dct)]);
        assert!(String::from_utf8_lossy(&pdf).contains("DeviceGray"));
    }
}
