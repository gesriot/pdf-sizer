use std::cmp::Reverse;
use std::fs;
use std::path::{Path, PathBuf};

mod dpi;
mod encode;
mod jbig2sym;
mod mrc;
mod pdf;
mod preprocess;
mod raster;
mod segment;
mod types;

pub(crate) use types::{CodecChoice, PreprocOpts};

use encode::{collect_images, encode_as_jp2, encode_image, estimate_jp2_size, estimate_pdf_size};
use dpi::page_dpi;
use mrc::{encode_mrc_frames, estimate_mrc_size, prepare_mrc_sources};
use pdf::{build_mrc_pdf, build_pdf, page_size_pt};
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
use mrc::prepare_page;
#[cfg(test)]
use segment::segment_page;
#[cfg(test)]
use types::{Cs, FgFrame, Foreground, MaskCodec, MaskLayer, MrcFrame, PdfFilter, TextMask};

pub(crate) struct RunOpts {
    pub(crate) input: PathBuf,
    pub(crate) output: PathBuf,
    pub(crate) target_mb: f64,
    /// Accepted deviation from target_mb (±). Variants outside [target±tolerance] are skipped.
    pub(crate) tolerance_mb: f64,
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

/// Format a megabyte value with just enough decimal places to be non-zero.
fn fmt_mb(mb: f64) -> String {
    if mb < 0.05 {
        format!("{:.2}", mb)
    } else {
        format!("{:.1}", mb)
    }
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
    let tolerance_bytes = (run_opts.tolerance_mb * 1024.0 * 1024.0) as u64;

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
        format!(
            "Целевой размер: {} MB (±{} MB)",
            fmt_mb(run_opts.target_mb),
            fmt_mb(run_opts.tolerance_mb)
        ),
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
    // Physical page size from the scan resolution, identical for every variant.
    let dpis: Vec<f32> = images
        .iter()
        .map(|(path, img)| page_dpi(path, img.width(), img.height()))
        .collect();
    let page_sizes: Vec<(f32, f32)> = images
        .iter()
        .zip(&dpis)
        .map(|((_, img), &dpi)| page_size_pt(img.width(), img.height(), dpi))
        .collect();

    // Passthrough embeds the original JPEG bytes, bypassing the preprocessed DynamicImage.
    // Disable it whenever any preprocessing is active so flags take effect on every variant.
    let allow_passthrough = !run_opts.preproc.despeckle
        && run_opts.preproc.flatten_threshold == 0
        && !run_opts.preproc.deskew;

    // ------------------------------------------------------------------ JPEG
    let run_jpeg = matches!(run_opts.codec, CodecChoice::Jpeg | CodecChoice::Auto);
    let run_jp2 = matches!(run_opts.codec, CodecChoice::Jp2 | CodecChoice::Auto);
    let run_mrc = matches!(run_opts.codec, CodecChoice::Mrc | CodecChoice::Auto);

    // Step=1 gives the user adjacent quality levels (q=100, q=99, q=98…) rather
    // than coarse 5-step jumps. With a tight tolerance (≤10 % of target) the
    // early-exit break fires after only a handful of levels, so the extra cost
    // over step=5 is small in practice. A coarse+refine approach would give the
    // same result with fewer probes, but adds complexity without a clear win at
    // typical tolerances. Revisit if large batches become slow.
    let quality_steps: Vec<u8> = (1..=100).rev().collect(); // 100, 99, ..., 1
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
                let pdf_bytes = build_pdf(&jpegs, &page_sizes);
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
                let pdf_bytes = build_pdf(&jpegs, &page_sizes);
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
                let pdf_bytes = build_pdf(&frames, &page_sizes);
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
                let pdf_bytes = build_pdf(&frames, &page_sizes);
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

    // ------------------------------------------------------------------ MRC
    // q is the JPEG quality of pictures (background q/2, text colour max(q/2, 25)); scale is
    // the target resolution: pictures at scale, background at scale/2, text colour at
    // scale/4.
    let mrc_quality_steps: &[u8] = &[55, 45, 35, 25];

    if run_mrc {
        check_cancel(cancel)?;
        reporter.report(ProgressEvent::Phase(SearchPhase::Mrc));
        report_info(
            reporter,
            format!(
                "\n[MRC] Поиск вариантов: маска текста (JBIG2/G4) + цвет текста + фон + иллюстрации около {:.0} MB...\n",
                run_opts.target_mb
            ),
        );

        let mrc_sources = prepare_mrc_sources(&images, &dpis, reporter, cancel)?;
        // The mask stays at the text resolution unless even the smallest variant overshoots
        // the target; then the compact mask at half that resolution is used.
        let lowest_q = mrc_quality_steps[mrc_quality_steps.len() - 1];
        let mut mask_level = 0;
        if estimate_mrc_size(&mrc_sources, lowest_q, 0.05, 0)? > target_bytes + tolerance_bytes {
            mask_level = 1;
            report_info(
                reporter,
                "  маска текста не помещается в цель — разрешение маски уменьшено вдвое",
            );
        }

        for (step_index, &q) in mrc_quality_steps.iter().enumerate() {
            check_cancel(cancel)?;
            let setting = format!("mrc-q={:03}", q);
            reporter.report(ProgressEvent::SettingStarted {
                codec: CodecChoice::Mrc,
                setting: setting.clone(),
                index: step_index + 1,
                total: mrc_quality_steps.len(),
            });
            let max_size = estimate_mrc_size(&mrc_sources, q, 1.0, mask_level)?;
            if max_size < target_bytes.saturating_sub(tolerance_bytes) {
                let message = format!(
                    "  q={:3}: даже при scale=100% размер {:.1} MB — меньше цели, пропуск остальных",
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

            let min_size = estimate_mrc_size(&mrc_sources, q, 0.05, mask_level)?;
            if min_size > target_bytes + tolerance_bytes {
                let message = format!(
                    "  q={:3}: даже при scale=5% размер {:.1} MB — больше цели, пропуск",
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

                let frames = encode_mrc_frames(&mrc_sources, q, s, mask_level)?;
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
                let frames = encode_mrc_frames(&mrc_sources, q, s, mask_level)?;
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
                        "  q={:3}, scale={:3}% -> {} ({:.1} MB)",
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
                let message = format!("  q={:3}: не удалось попасть в целевой диапазон", q);
                reporter.report(ProgressEvent::SettingSkipped {
                    codec: CodecChoice::Mrc,
                    setting,
                    reason: message.clone(),
                });
                report_info(reporter, message);
            }
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
            "  MRC mrc-q=...: q — JPEG иллюстраций (фон сжимается сильнее); текст — 1-битная маска (JBIG2 со словарём символов или G4)",
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
            tolerance_mb: 0.1,
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
            tolerance_mb: 0.1,
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

    // ---- MRC -----------------------------------------------------------

    #[test]
    fn mrc_segmentation_detects_text_like_components() {
        let img = make_text_scan();
        let seg = segment_page(&img, 300.0);
        assert!(seg.ink[22 * 120 + 18]);
        assert!(!seg.ink[0]);
        assert!(seg.pictures.is_empty());
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
        let mask = TextMask {
            width,
            height,
            pixels,
        };
        let encoded = encode_text_mask_g4(&mask).unwrap();
        let decoded = decode_g4_mask(&encoded, width as u16, height as u16);
        assert_eq!(decoded, mask.pixels);
    }

    #[test]
    fn mrc_background_fills_masked_text_pixels() {
        let source = prepare_page(&make_text_scan(), 300.0).unwrap();
        // The background is stored at half resolution: (18, 22) -> (9, 11).
        let bg = source.background.to_rgb8();
        let [r, g, b] = bg.get_pixel(9, 11).0;
        assert!(r > 200 && g > 200 && b > 200, "text pixel was not filled: {r},{g},{b}");
    }

    #[test]
    fn mrc_black_text_uses_a_fill_colour() {
        let source = prepare_page(&make_text_scan(), 300.0).unwrap();
        match source.foreground {
            Foreground::Solid(c) => assert!(c.iter().all(|&v| v < 60), "{c:?}"),
            Foreground::Image(_) => panic!("uniform black text should not need a colour field"),
        }
    }

    #[test]
    fn mrc_coloured_text_gets_a_colour_field() {
        let mut img = RgbImage::from_fn(200, 120, |_, _| Rgb([245u8, 244, 238]));
        for (i, colour) in [[200u8, 20, 20], [20, 20, 200], [20, 20, 20]].into_iter().enumerate() {
            for y in 20..40 {
                for x in 20 + i as u32 * 60..44 + i as u32 * 60 {
                    img.put_pixel(x, y, Rgb(colour));
                }
            }
        }
        let source = prepare_page(&DynamicImage::ImageRgb8(img), 300.0).unwrap();
        let Foreground::Image(field) = source.foreground else {
            panic!("three text colours need a colour field");
        };
        // Colour field at 1/4 resolution: the red glyph (x 20..44) sits around (8, 7).
        let p = field.to_rgb8().get_pixel(8, 7).0;
        assert!(p[0] > 120 && p[2] < 100, "red glyph colour lost: {p:?}");
    }

    #[test]
    fn mrc_keeps_large_heading_glyphs_and_table_rules() {
        // The old segmentation dropped glyphs over 3× the median height and long rules.
        let (w, h) = (600u32, 400u32);
        let mut img = RgbImage::from_fn(w, h, |_, _| Rgb([245u8, 245, 240]));
        for i in 0..15 {
            let x0 = 20 + (i % 5) * 40;
            let y0 = 20 + (i / 5) * 30;
            for y in y0..y0 + 10 {
                for x in x0..x0 + 8 {
                    img.put_pixel(x, y, Rgb([15u8, 15, 15]));
                }
            }
        }
        for y in 150..240 {
            for x in 300..350 {
                img.put_pixel(x, y, Rgb([120u8, 20, 20]));
            }
        }
        for x in 50..550 {
            img.put_pixel(x, 300, Rgb([20u8, 20, 20]));
            img.put_pixel(x, 301, Rgb([20u8, 20, 20]));
        }
        let seg = segment_page(&DynamicImage::ImageRgb8(img), 300.0);
        assert!(seg.ink[(24 * w + 24) as usize], "body glyph");
        assert!(seg.ink[(195 * w + 325) as usize], "heading glyph");
        assert!(seg.ink[(300 * w + 300) as usize], "table rule");
    }

    #[test]
    fn mrc_cuts_out_photos_but_not_flat_boxes() {
        let (w, h) = (1200u32, 1600u32);
        let mut seed = 12345u32;
        let mut img = RgbImage::from_fn(w, h, |_, _| Rgb([240u8, 236, 225]));
        for y in 100..400 {
            for x in 100..500 {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                let n = (seed >> 24) as i32 % 30;
                let r = (40 + (x - 100) / 2) as i32 + n;
                let g = (200 - (y - 100) / 3) as i32 + n;
                img.put_pixel(x, y, Rgb([r.clamp(0, 255) as u8, g.clamp(0, 255) as u8, 90]));
            }
        }
        for y in 800..1000 {
            for x in 100..700 {
                img.put_pixel(x, y, Rgb([250u8, 236, 160]));
            }
        }
        for y in 880..900 {
            for x in 200..212 {
                img.put_pixel(x, y, Rgb([20u8, 20, 10]));
            }
        }
        let seg = segment_page(&DynamicImage::ImageRgb8(img), 300.0);
        assert_eq!(seg.pictures.len(), 1, "{:?}", seg.pictures);
        let p = seg.pictures[0];
        let near = |a: u32, b: u32| a.abs_diff(b) <= 8;
        assert!(near(p.x, 100) && near(p.y, 100) && near(p.w, 400) && near(p.h, 300), "{p:?}");
        assert!(seg.ink[(890 * w + 205) as usize], "text inside the tinted box stays in the mask");
    }

    /// Build a wide A4-sized synthetic mask with hundreds of glyph-like blocks. JBIG2
    /// dominates G4 on inputs at this scale, so it exercises the JBIG2 PDF path.
    fn make_large_text_mask() -> TextMask {
        let (w, h) = (1240u32, 1754u32);
        let mut pixels = vec![false; (w * h) as usize];
        for line in 0..40 {
            let y0 = 60 + line * 40;
            for col in 0..50 {
                let x0 = 60 + col * 22;
                for y in y0..(y0 + 12).min(h) {
                    for x in x0..(x0 + 8).min(w) {
                        pixels[(y * w + x) as usize] = true;
                    }
                }
            }
        }
        TextMask {
            width: w,
            height: h,
            pixels,
        }
    }

    #[test]
    fn build_mrc_pdf_uses_a_supported_mask_filter() {
        let img = make_text_scan();
        let images = vec![(PathBuf::from("page.png"), img)];
        let sources =
            prepare_mrc_sources(&images, &[300.0], &NoopReporter, &CancellationToken::default())
                .unwrap();
        let frames = encode_mrc_frames(&sources, 35, 0.5, 0).unwrap();
        let pdf = build_mrc_pdf(&frames);
        let text = String::from_utf8_lossy(&pdf);
        assert!(text.contains("DCTDecode"));
        assert!(text.contains("ImageMask true"));
        // Mask filter is auto-selected; either CCITT G4 or JBIG2 is acceptable.
        assert!(
            text.contains("CCITTFaxDecode") || text.contains("JBIG2Decode"),
            "expected a 1-bit mask filter, found neither"
        );
        // Both filters deliver text pixels as 0; an inverting Decode array would paint
        // everything except the text.
        assert!(!text.contains("/Decode ["), "mask must use the default Decode array");
    }

    #[test]
    fn mrc_page_size_follows_dpi() {
        let images = vec![(PathBuf::from("page.png"), make_text_scan())];
        let sources =
            prepare_mrc_sources(&images, &[150.0], &NoopReporter, &CancellationToken::default())
                .unwrap();
        let frames = encode_mrc_frames(&sources, 35, 1.0, 0).unwrap();
        // 120×80 px at 150 dpi = 57.6×38.4 pt.
        assert!((frames[0].page_size.0 - 57.6).abs() < 1e-3);
        assert!((frames[0].page_size.1 - 38.4).abs() < 1e-3);
        let (w, h) = page_size_pt(2480, 3508, 300.0);
        assert!((w - 595.2).abs() < 0.01 && (h - 841.92).abs() < 0.01);
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
    fn jbig2_stream_has_no_end_segments() {
        let mask = make_large_text_mask();
        let raw = jbig2enc_rust::encode_single_image_lossless(
            &mask_to_jbig2_pixels(&mask),
            mask.width,
            mask.height,
            true,
        )
        .unwrap()
        .page_data;
        // The encoder appends 11-byte end-of-page (49) and end-of-file (51) segments.
        assert_eq!(raw[raw.len() - 18] & 0x3F, 49);
        assert_eq!(raw[raw.len() - 7] & 0x3F, 51);
        let stripped = encode_text_mask_jbig2(&mask).unwrap();
        assert_eq!(raw.len() - stripped.len(), 22);
        assert_eq!(&raw[..stripped.len()], &stripped[..]);
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
        // Build a frame that explicitly uses JBIG2, bypassing auto-selection, so the PDF
        // path that emits /JBIG2Decode is exercised even if a future mask happens to be
        // smaller in G4.
        let mask = make_large_text_mask();
        let layer = MaskLayer {
            width: mask.width,
            height: mask.height,
            data: encode_text_mask_jbig2(&mask).unwrap(),
            codec: MaskCodec::Jbig2,
            globals: None,
        };
        let bg = DynamicImage::ImageRgb8(RgbImage::from_fn(40, 40, |_, _| Rgb([240u8, 240, 240])));
        let frame = MrcFrame {
            page_size: (595.2, 841.92),
            source_size: (mask.width, mask.height),
            background: encode_raster_as_jpeg(&bg, 50, 1.0).unwrap(),
            mask: Some(&layer),
            foreground: FgFrame::Solid([0, 0, 0]),
            pictures: Vec::new(),
        };
        let pdf = build_mrc_pdf(&[frame]);
        let text = String::from_utf8_lossy(&pdf);
        assert!(text.contains("JBIG2Decode"));
        assert!(text.contains("ImageMask true"));
        assert!(text.contains("/MediaBox [0 0 595.2 841.92]"));
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

    // ---- build_pdf -----------------------------------------------------

    #[test]
    fn build_pdf_starts_with_percent_pdf() {
        let jpeg = make_jpeg(32, 32, false);
        let pdf = build_pdf(&[(jpeg, 32, 32, Cs::Rgb, PdfFilter::Dct)], &[(32.0, 32.0)]);
        assert_eq!(&pdf[..4], b"%PDF");
    }

    #[test]
    fn build_pdf_dct_filter_in_stream() {
        let jpeg = make_jpeg(32, 32, false);
        let pdf = build_pdf(&[(jpeg, 32, 32, Cs::Rgb, PdfFilter::Dct)], &[(32.0, 32.0)]);
        assert!(
            String::from_utf8_lossy(&pdf).contains("DCTDecode"),
            "PDF must contain DCTDecode for JPEG images"
        );
    }

    #[test]
    fn build_pdf_jpx_filter_in_stream() {
        let px = vec![128u8; 32 * 32 * 3];
        let jp2 = encode_jp2_raw(&px, 32, 32, 0.5, false).unwrap();
        let pdf = build_pdf(&[(jp2, 32, 32, Cs::Rgb, PdfFilter::Jpx)], &[(32.0, 32.0)]);
        assert!(
            String::from_utf8_lossy(&pdf).contains("JPXDecode"),
            "PDF must contain JPXDecode for JP2 images"
        );
    }

    #[test]
    fn build_pdf_device_gray_for_gray_jpeg() {
        let jpeg = make_jpeg(32, 32, true);
        let pdf = build_pdf(&[(jpeg, 32, 32, Cs::Gray, PdfFilter::Dct)], &[(32.0, 32.0)]);
        assert!(String::from_utf8_lossy(&pdf).contains("DeviceGray"));
    }
}
