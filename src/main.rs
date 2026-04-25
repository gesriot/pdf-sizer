use fax::{Color as FaxColor, VecWriter};
use image::GenericImageView;
use imageproc::contrast::{ThresholdType, adaptive_threshold, otsu_level, threshold};
use imageproc::filter::median_filter;
use imageproc::geometric_transformations::{Interpolation, rotate_about_center_no_crop};
use imageproc::region_labelling::{Connectivity, connected_components};
use mozjpeg::{ColorSpace as MozCs, Compress};
use openjp2::{
    OPJ_CLRSPC_GRAY, OPJ_CLRSPC_SRGB, OPJ_CODEC_JP2,
    openjpeg::{
        opj_create_compress, opj_destroy_codec, opj_encode, opj_end_compress, opj_setup_encoder,
        opj_start_compress, opj_stream_create, opj_stream_destroy, opj_stream_set_seek_function,
        opj_stream_set_skip_function, opj_stream_set_user_data, opj_stream_set_write_function,
    },
    opj_cparameters_t, opj_image, opj_image_comptparm,
};
use pdf_writer::{Content, Finish, Name, Pdf, Rect, Ref};
use std::collections::{HashMap, HashSet};
use std::env;
use std::ffi::c_void;
use std::fs;
use std::path::{Path, PathBuf};

/// Encoded image frame ready to embed in a PDF page.
type Frame = (Vec<u8>, u32, u32, Cs, PdfFilter);

#[derive(Clone, Copy)]
enum Cs {
    Rgb,
    Gray,
}

#[derive(Clone, Copy)]
enum PdfFilter {
    Dct, // JPEG (DCTDecode)
    Jpx, // JPEG2000 (JPXDecode)
}

struct TextMask {
    width: u32,
    height: u32,
    pixels: Vec<bool>,
    coverage: f32,
    /// Share of structural (line-like) components among all non-trivial ones.
    /// High values flag tables/diagrams that MRC would misrepresent as a 1-bit mask.
    line_ratio: f32,
}

struct MrcSource {
    background: image::DynamicImage,
    mask: TextMask,
    mask_g4: Vec<u8>,
}

struct MrcFrame {
    background: Frame,
    mask_g4: Vec<u8>,
    mask_w: u32,
    mask_h: u32,
}

#[derive(Clone, Copy, PartialEq)]
enum CodecChoice {
    Jpeg,
    Jp2,
    Mrc,
    Auto,
}

struct PreprocOpts {
    despeckle: bool,
    /// Luma distance from background at which a pixel gets flattened (0 = off).
    flatten_threshold: u8,
    deskew: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!(
            "Использование: {} <МБ> [--despeckle] [--flatten[=<0-255>]] [--deskew] [--codec=jpeg|jp2|mrc|auto]",
            args[0]
        );
        eprintln!("Изображения берутся из папки images/ рядом с исполняемым файлом");
        std::process::exit(1);
    }

    let mut target_mb_arg: Option<&str> = None;
    let mut opts = PreprocOpts {
        despeckle: false,
        flatten_threshold: 0,
        deskew: false,
    };
    let mut codec_choice = CodecChoice::Jpeg;

    for arg in &args[1..] {
        if arg == "--despeckle" {
            opts.despeckle = true;
        } else if arg == "--deskew" {
            opts.deskew = true;
        } else if arg == "--flatten" {
            opts.flatten_threshold = 30;
        } else if let Some(val) = arg.strip_prefix("--flatten=") {
            opts.flatten_threshold = val
                .parse::<u8>()
                .map_err(|_| format!("--flatten: '{}' не является числом от 0 до 255", val))?;
        } else if let Some(val) = arg
            .strip_prefix("--codec=")
            .or_else(|| arg.strip_prefix("--mode="))
        {
            codec_choice = match val {
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

    let target_mb: f64 = target_mb_arg
        .unwrap_or_else(|| {
            eprintln!("Ошибка: не задан целевой размер");
            std::process::exit(1)
        })
        .parse()
        .map_err(|_| "целевой размер не является числом")?;

    let target_bytes = (target_mb * 1024.0 * 1024.0) as u64;
    let tolerance_bytes = (2.0 * 1024.0 * 1024.0) as u64;

    let exe_dir = env::current_exe()?
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let images_dir = exe_dir.join("images");

    if !images_dir.exists() {
        eprintln!(
            "Ошибка: папка images/ не найдена рядом с исполняемым файлом ({})",
            images_dir.display()
        );
        std::process::exit(1);
    }

    let image_paths = collect_images(&images_dir)?;
    if image_paths.is_empty() {
        eprintln!("Ошибка: в папке images/ нет изображений");
        std::process::exit(1);
    }

    println!("Найдено изображений: {}", image_paths.len());
    println!("Целевой размер: {:.0} MB (±2 MB)", target_mb);
    if opts.despeckle || opts.flatten_threshold > 0 || opts.deskew {
        let active: Vec<&str> = [
            opts.despeckle.then_some("despeckle"),
            (opts.flatten_threshold > 0).then_some("flatten"),
            opts.deskew.then_some("deskew"),
        ]
        .into_iter()
        .flatten()
        .collect();
        println!("Препроцессинг: {}", active.join(", "));
    }

    let images: Vec<(PathBuf, image::DynamicImage)> = image_paths
        .iter()
        .map(|p| {
            image::open(p)
                .map(|img| {
                    let img = preprocess(img, &opts);
                    (p.clone(), img)
                })
                .map_err(|e| {
                    eprintln!("Ошибка открытия {}: {}", p.display(), e);
                    e
                })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Passthrough embeds the original JPEG bytes, bypassing the preprocessed DynamicImage.
    // Disable it whenever any preprocessing is active so flags take effect on every variant.
    let allow_passthrough = !opts.despeckle && opts.flatten_threshold == 0 && !opts.deskew;

    // ------------------------------------------------------------------ JPEG
    let run_jpeg = matches!(codec_choice, CodecChoice::Jpeg | CodecChoice::Auto);
    let run_jp2 = matches!(codec_choice, CodecChoice::Jp2 | CodecChoice::Auto);
    let run_mrc = matches!(codec_choice, CodecChoice::Mrc | CodecChoice::Auto);

    let quality_steps: Vec<u8> = (1..=20).rev().map(|i| i * 5).collect(); // 100, 95, ..., 5
    // (filename, setting_label, scale%, size_mb)
    // setting_label: "q=080" for JPEG, "bpp=0.50" for JP2, "mrc-q=040" for MRC.
    let mut variants: Vec<(String, String, u32, f64)> = Vec::new();

    if run_jpeg {
        println!(
            "\n[JPEG] Поиск вариантов: для каждого quality ищем scale, дающий ~{:.0} MB...\n",
            target_mb
        );
    }

    if run_jpeg {
        for &q in &quality_steps {
            let max_size = estimate_pdf_size(&images, q, 1.0, allow_passthrough)?;
            if max_size < target_bytes.saturating_sub(tolerance_bytes) {
                println!(
                    "  quality={:3}: даже при scale=100% размер {:.1} MB — меньше цели, пропуск остальных",
                    q,
                    max_size as f64 / (1024.0 * 1024.0)
                );
                break;
            }

            let min_size = estimate_pdf_size(&images, q, 0.05, allow_passthrough)?;
            if min_size > target_bytes + tolerance_bytes {
                println!(
                    "  quality={:3}: даже при scale=5% размер {:.1} MB — больше цели, пропуск",
                    q,
                    min_size as f64 / (1024.0 * 1024.0)
                );
                continue;
            }

            let mut lo = 5.0_f64;
            let mut hi = 100.0_f64;
            let mut best: Option<(u32, Vec<u8>, f64)> = None;

            for _ in 0..20 {
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
                let filename = format!("variant_q{:03}_s{:03}.pdf", q, s_pct);
                fs::write(&filename, &pdf_bytes)?;
                println!(
                    "  quality={:3}, scale={:3}% -> {} ({:.1} MB)",
                    q, s_pct, filename, size_mb
                );
                variants.push((filename, format!("q={:03}", q), s_pct, size_mb));
            } else {
                println!("  quality={:3}: не удалось попасть в целевой диапазон", q);
            }
        }
    } // end if run_jpeg / for quality_steps

    // ------------------------------------------------------------------ JP2
    // Quality steps: bits-per-pixel × 100 from high to low quality.
    // bpp = step / 100.0;  rate = num_comps × 8 / bpp
    let jp2_steps: &[u32] = &[200, 150, 100, 75, 50, 35, 25, 18, 10, 5];

    if run_jp2 {
        println!(
            "\n[JP2] Поиск вариантов: для каждого bpp ищем scale, дающий ~{:.0} MB...\n",
            target_mb
        );
        for &bpp_x100 in jp2_steps {
            let bpp = bpp_x100 as f32 / 100.0;

            let max_size = estimate_jp2_size(&images, bpp, 1.0)?;
            if max_size < target_bytes.saturating_sub(tolerance_bytes) {
                println!(
                    "  bpp={:.2}: даже при scale=100% размер {:.1} MB — меньше цели, пропуск остальных",
                    bpp,
                    max_size as f64 / (1024.0 * 1024.0)
                );
                break;
            }
            let min_size = estimate_jp2_size(&images, bpp, 0.05)?;
            if min_size > target_bytes + tolerance_bytes {
                println!(
                    "  bpp={:.2}: даже при scale=5% размер {:.1} MB — больше цели, пропуск",
                    bpp,
                    min_size as f64 / (1024.0 * 1024.0)
                );
                continue;
            }

            let mut lo = 5.0_f64;
            let mut hi = 100.0_f64;
            let mut best: Option<(u32, Vec<u8>, f64)> = None;

            for _ in 0..20 {
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
                let filename = format!("variant_jp2_b{:03}_s{:03}.pdf", bpp_x100, s_pct);
                fs::write(&filename, &pdf_bytes)?;
                println!(
                    "  bpp={:.2}, scale={:3}% -> {} ({:.1} MB)",
                    bpp, s_pct, filename, size_mb
                );
                variants.push((filename, format!("bpp={:.2}", bpp), s_pct, size_mb));
            } else {
                println!("  bpp={:.2}: не удалось попасть в целевой диапазон", bpp);
            }
        }
    }

    // ------------------------------------------------------------------ MRC-lite
    let mrc_quality_steps: &[u8] = &[55, 45, 35, 25];

    if run_mrc {
        println!(
            "\n[MRC] Поиск вариантов: маска текста CCITT G4 + JPEG-фон около {:.0} MB...\n",
            target_mb
        );

        let mrc_sources = prepare_mrc_sources(&images)?;
        if let Some(mrc_sources) = mrc_sources {
            for &q in mrc_quality_steps {
                let max_size = estimate_mrc_size(&mrc_sources, q, 1.0)?;
                if max_size < target_bytes.saturating_sub(tolerance_bytes) {
                    println!(
                        "  bg-q={:3}: даже при bg-scale=100% размер {:.1} MB — меньше цели, пропуск остальных",
                        q,
                        max_size as f64 / (1024.0 * 1024.0)
                    );
                    break;
                }

                let min_size = estimate_mrc_size(&mrc_sources, q, 0.05)?;
                if min_size > target_bytes + tolerance_bytes {
                    println!(
                        "  bg-q={:3}: даже при bg-scale=5% размер {:.1} MB — больше цели, пропуск",
                        q,
                        min_size as f64 / (1024.0 * 1024.0)
                    );
                    continue;
                }

                let mut lo = 5.0_f64;
                let mut hi = 100.0_f64;
                let mut best: Option<(u32, Vec<u8>, f64)> = None;

                for _ in 0..20 {
                    if hi - lo < 0.5 {
                        break;
                    }
                    let mid = ((lo + hi) / 2.0).round().clamp(5.0, 100.0);
                    let s = mid / 100.0;

                    let frames = encode_mrc_frames(&mrc_sources, q, s)?;
                    let pdf_bytes = build_mrc_pdf(&frames);
                    let size = pdf_bytes.len() as u64;

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
                    let filename = format!("variant_mrc_q{:03}_s{:03}.pdf", q, s_pct);
                    fs::write(&filename, &pdf_bytes)?;
                    println!(
                        "  bg-q={:3}, bg-scale={:3}% -> {} ({:.1} MB)",
                        q, s_pct, filename, size_mb
                    );
                    variants.push((filename, format!("mrc-q={:03}", q), s_pct, size_mb));
                } else {
                    println!("  bg-q={:3}: не удалось попасть в целевой диапазон", q);
                }
            }
        } else {
            println!("  MRC пропущен: маска текста слишком мала/велика хотя бы на одной странице");
        }
    }

    println!("\n--- Итого ---");
    if variants.is_empty() {
        println!("Не удалось создать ни одного варианта в целевом диапазоне.");
    } else {
        println!("Создано вариантов: {}\n", variants.len());
        println!(
            "  {:>12}  {:>5}  {:>10}  файл",
            "параметр", "scale", "размер"
        );
        println!("  {}", "-".repeat(56));
        for (filename, setting, s, size_mb) in &variants {
            let hint = if let Some(q_str) = setting
                .strip_prefix("q=")
                .or_else(|| setting.strip_prefix("mrc-q="))
            {
                let q: u8 = q_str.parse().unwrap_or(0);
                if q > 80 {
                    "  <- чёткость"
                } else if *s > 80 {
                    "  <- детали"
                } else {
                    ""
                }
            } else {
                // JP2: hint only from scale
                if *s > 80 { "  <- детали" } else { "" }
            };
            println!(
                "  {:>12}  {:>3}%  {:>7.1} MB  {}{}",
                setting, s, size_mb, filename, hint
            );
        }
        println!("\n  JPEG q=...   : высокий q  = меньше артефактов (текст, чёткие края)");
        println!("  JP2 bpp=...  : меньший bpp = агрессивнее сжатие, мягче артефакты");
        println!("  MRC mrc-q=...: q управляет только JPEG-фоном; текст идёт 1-битной G4-маской");
        println!("  Высокий scale = больше деталей и разрешение (мелкие элементы)");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

fn preprocess(img: image::DynamicImage, opts: &PreprocOpts) -> image::DynamicImage {
    let mut img = img;
    if opts.deskew {
        img = apply_deskew(img);
    }
    if opts.despeckle {
        img = apply_despeckle(img);
    }
    if opts.flatten_threshold > 0 {
        img = apply_flatten_background(img, opts.flatten_threshold);
    }
    img
}

fn apply_despeckle(img: image::DynamicImage) -> image::DynamicImage {
    match img {
        image::DynamicImage::ImageLuma8(buf) => {
            image::DynamicImage::ImageLuma8(median_filter(&buf, 1, 1))
        }
        image::DynamicImage::ImageRgb8(buf) => {
            image::DynamicImage::ImageRgb8(median_filter(&buf, 1, 1))
        }
        other => {
            let rgb = other.to_rgb8();
            image::DynamicImage::ImageRgb8(median_filter(&rgb, 1, 1))
        }
    }
}

/// Estimate the dominant background colour by finding the most frequent luma bucket
/// and averaging the RGB values of all pixels near that bucket.
fn estimate_background(rgb: &image::RgbImage) -> [u8; 3] {
    let mut hist = [0u32; 256];
    for p in rgb.pixels() {
        let [r, g, b] = p.0;
        let luma = (77 * r as u32 + 150 * g as u32 + 29 * b as u32) >> 8;
        hist[luma as usize] += 1;
    }
    let mode = hist
        .iter()
        .enumerate()
        .max_by_key(|&(_, c)| c)
        .map(|(i, _)| i)
        .unwrap_or(255) as u8;

    let mut sr = 0u64;
    let mut sg = 0u64;
    let mut sb = 0u64;
    let mut n = 0u64;
    for p in rgb.pixels() {
        let [r, g, b] = p.0;
        let luma = ((77 * r as u32 + 150 * g as u32 + 29 * b as u32) >> 8) as u8;
        if luma.abs_diff(mode) <= 10 {
            sr += r as u64;
            sg += g as u64;
            sb += b as u64;
            n += 1;
        }
    }
    if n == 0 {
        return [255, 255, 255];
    }
    [(sr / n) as u8, (sg / n) as u8, (sb / n) as u8]
}

fn apply_flatten_background(img: image::DynamicImage, threshold: u8) -> image::DynamicImage {
    let rgb = img.to_rgb8();
    let [br, bg, bb] = estimate_background(&rgb);
    let bg_luma = (77 * br as u32 + 150 * bg as u32 + 29 * bb as u32) >> 8;

    let out = image::RgbImage::from_fn(rgb.width(), rgb.height(), |x, y| {
        let [r, g, b] = rgb.get_pixel(x, y).0;
        let luma = (77 * r as u32 + 150 * g as u32 + 29 * b as u32) >> 8;
        if (luma as i32 - bg_luma as i32).unsigned_abs() as u8 <= threshold {
            image::Rgb([br, bg, bb])
        } else {
            image::Rgb([r, g, b])
        }
    });
    image::DynamicImage::ImageRgb8(out)
}

/// Variance-of-Projection-Profile deskew.
/// Scans angles −10°…+10° (0.5° step); returns the angle (degrees) that maximises
/// the variance of the horizontal projection of dark pixels.
fn detect_skew_angle(luma: &image::GrayImage) -> f32 {
    let level = otsu_level(luma);
    let binary = threshold(luma, level, ThresholdType::Binary);

    // Subsample every 4th pixel in each axis — sufficient for line detection.
    let dark: Vec<(i32, i32)> = binary
        .enumerate_pixels()
        .filter(|(x, y, p)| x % 4 == 0 && y % 4 == 0 && p[0] < 128)
        .map(|(x, y, _)| (x as i32, y as i32))
        .collect();

    if dark.len() < 50 {
        return 0.0;
    }

    let max_p = (luma.width() as f32).hypot(luma.height() as f32) as usize + 1;
    let hist_len = 2 * max_p;

    let mut best_angle = 0.0f32;
    let mut best_variance = 0.0f64;

    // −10° to +10° in 0.5° steps
    for step in -20i32..=20 {
        let deg = step as f32 * 0.5;
        let (sin_t, cos_t) = (deg.to_radians().sin(), deg.to_radians().cos());

        let mut hist = vec![0u32; hist_len];
        for &(x, y) in &dark {
            let p = (-x as f32 * sin_t + y as f32 * cos_t).round() as i32 + max_p as i32;
            if (p as usize) < hist_len {
                hist[p as usize] += 1;
            }
        }

        let n = hist_len as f64;
        let mean = hist.iter().map(|&c| c as f64).sum::<f64>() / n;
        let variance = hist.iter().map(|&c| (c as f64 - mean).powi(2)).sum::<f64>() / n;

        if variance > best_variance {
            best_variance = variance;
            best_angle = deg;
        }
    }

    best_angle
}

fn apply_deskew(img: image::DynamicImage) -> image::DynamicImage {
    let angle = detect_skew_angle(&img.to_luma8());

    if angle.abs() < 0.25 {
        return img;
    }

    println!("  [deskew] обнаружен наклон {:.1}°, исправляю", angle);

    // Rotate clockwise by -angle to compensate.
    let theta = -angle.to_radians();

    match img {
        image::DynamicImage::ImageLuma8(buf) => image::DynamicImage::ImageLuma8(
            rotate_about_center_no_crop(&buf, theta, Interpolation::Bilinear, image::Luma([255u8])),
        ),
        image::DynamicImage::ImageRgb8(buf) => {
            image::DynamicImage::ImageRgb8(rotate_about_center_no_crop(
                &buf,
                theta,
                Interpolation::Bilinear,
                image::Rgb([255u8, 255u8, 255u8]),
            ))
        }
        other => {
            let rgb = other.to_rgb8();
            image::DynamicImage::ImageRgb8(rotate_about_center_no_crop(
                &rgb,
                theta,
                Interpolation::Bilinear,
                image::Rgb([255u8, 255u8, 255u8]),
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// MRC-lite segmentation and CCITT G4 mask encoding
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct ComponentStats {
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
    area: u32,
}

impl ComponentStats {
    fn new(x: u32, y: u32) -> Self {
        Self {
            min_x: x,
            min_y: y,
            max_x: x,
            max_y: y,
            area: 0,
        }
    }

    fn add(&mut self, x: u32, y: u32) {
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
        self.area += 1;
    }

    fn width(self) -> u32 {
        self.max_x - self.min_x + 1
    }

    fn height(self) -> u32 {
        self.max_y - self.min_y + 1
    }
}

fn adaptive_mrc_block_radius(w: u32, h: u32) -> u32 {
    (w.min(h) / 40).clamp(3, 35)
}

fn is_text_component(stats: ComponentStats, page_w: u32, page_h: u32) -> bool {
    let cw = stats.width();
    let ch = stats.height();
    if stats.area < 2 {
        return false;
    }

    let min_h = if page_h < 80 { 1 } else { 2 };
    let max_h = (page_h / 3).clamp(8, 160);
    let max_w = (page_w / 2).clamp(8, 240);
    let aspect = cw as f32 / ch.max(1) as f32;
    let fill = stats.area as f32 / (cw * ch).max(1) as f32;

    ch >= min_h && ch <= max_h && cw <= max_w && (0.05..=20.0).contains(&aspect) && fill >= 0.01
}

/// Table rules, frames, dividers: components that are thin (≤ 3 px) in one axis and
/// span a meaningful fraction of the page in the other. MRC would bake these into the
/// text mask as solid black; we drop them here and let the background layer render them.
fn is_line_like(stats: ComponentStats, page_w: u32, page_h: u32) -> bool {
    let cw = stats.width();
    let ch = stats.height();
    let long_h = page_w as f32 * 0.15;
    let long_v = page_h as f32 * 0.15;
    (ch <= 3 && cw as f32 > long_h) || (cw <= 3 && ch as f32 > long_v)
}

fn segment_text_mask(img: &image::DynamicImage) -> TextMask {
    let gray = img.to_luma8();
    let (w, h) = gray.dimensions();
    let radius = adaptive_mrc_block_radius(w, h);
    let binary = adaptive_threshold(&gray, radius, 15);
    let labels = connected_components(&binary, Connectivity::Eight, image::Luma([255u8]));

    let mut stats_by_label: HashMap<u32, ComponentStats> = HashMap::new();
    for (x, y, p) in labels.enumerate_pixels() {
        let label = p[0];
        if label == 0 {
            continue;
        }
        stats_by_label
            .entry(label)
            .or_insert_with(|| ComponentStats::new(x, y))
            .add(x, y);
    }

    // Pass 1: split components into line-like (table rules, frames) and text candidates.
    let mut candidates: Vec<(u32, ComponentStats)> = Vec::new();
    let mut line_like = 0usize;
    for (label, stats) in stats_by_label {
        if is_line_like(stats, w, h) {
            line_like += 1;
        } else if is_text_component(stats, w, h) {
            candidates.push((label, stats));
        }
    }

    // line_ratio is evaluated against the first-pass candidate count so that the later
    // median filter — which prunes size outliers — does not inflate the ratio.
    let line_ratio = if line_like + candidates.len() == 0 {
        0.0
    } else {
        line_like as f32 / (line_like + candidates.len()) as f32
    };

    // Pass 2: keep only glyph-sized components around the median candidate height.
    // Drops stray oversized/undersized components (banners, dust, page numbers in a
    // different size) that would otherwise be forced into the mask.
    if candidates.len() >= 4 {
        let mut heights: Vec<u32> = candidates.iter().map(|(_, s)| s.height()).collect();
        heights.sort_unstable();
        let median_h = heights[heights.len() / 2] as f32;
        let lo = (median_h * 0.35).floor().max(1.0) as u32;
        let hi = (median_h * 3.0).ceil() as u32;
        candidates.retain(|(_, s)| {
            let ch = s.height();
            ch >= lo && ch <= hi
        });
    }

    let keep: HashSet<u32> = candidates.iter().map(|(l, _)| *l).collect();

    let mut pixels = vec![false; (w * h) as usize];
    let mut text_pixels = 0usize;
    for (x, y, p) in labels.enumerate_pixels() {
        if keep.contains(&p[0]) {
            let idx = (y * w + x) as usize;
            pixels[idx] = true;
            text_pixels += 1;
        }
    }

    let coverage = if w == 0 || h == 0 {
        0.0
    } else {
        text_pixels as f32 / (w as f32 * h as f32)
    };

    TextMask {
        width: w,
        height: h,
        pixels,
        coverage,
        line_ratio,
    }
}

fn is_mrc_suitable(mask: &TextMask) -> bool {
    // Coverage gate rejects near-empty pages and pages where segmentation blew up.
    // line_ratio gate rejects pages dominated by table rules / diagram lines — those
    // should stay in the background JPEG rather than becoming a 1-bit mask.
    (0.005..=0.70).contains(&mask.coverage) && mask.line_ratio <= 0.30
}

fn fill_masked_background(img: &image::DynamicImage, mask: &TextMask) -> image::DynamicImage {
    let rgb = img.to_rgb8();
    let bg = estimate_background(&rgb);
    let out = image::RgbImage::from_fn(rgb.width(), rgb.height(), |x, y| {
        let idx = (y * mask.width + x) as usize;
        if mask.pixels.get(idx).copied().unwrap_or(false) {
            image::Rgb(bg)
        } else {
            *rgb.get_pixel(x, y)
        }
    });
    image::DynamicImage::ImageRgb8(out)
}

fn encode_text_mask_g4(mask: &TextMask) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    if mask.width > u16::MAX as u32 || mask.height > u16::MAX as u32 {
        return Err("MRC: CCITT G4 encoder supports dimensions up to 65535".into());
    }

    let width = mask.width as u16;
    let mut encoder = fax::encoder::Encoder::new(VecWriter::with_capacity(
        (mask.width * mask.height) as usize,
    ));

    for y in 0..mask.height {
        let row = (0..mask.width).map(|x| {
            let idx = (y * mask.width + x) as usize;
            if mask.pixels[idx] {
                FaxColor::Black
            } else {
                FaxColor::White
            }
        });
        encoder.encode_line(row, width)?;
    }

    Ok(encoder.finish()?.finish())
}

fn prepare_mrc_sources(
    images: &[(PathBuf, image::DynamicImage)],
) -> Result<Option<Vec<MrcSource>>, Box<dyn std::error::Error>> {
    let mut sources = Vec::with_capacity(images.len());
    for (path, img) in images {
        let mask = segment_text_mask(img);
        if !is_mrc_suitable(&mask) {
            println!(
                "  MRC: {} пропущен по покрытию маски {:.2}%",
                path.display(),
                mask.coverage * 100.0
            );
            return Ok(None);
        }
        let background = fill_masked_background(img, &mask);
        let mask_g4 = encode_text_mask_g4(&mask)?;
        sources.push(MrcSource {
            background,
            mask,
            mask_g4,
        });
    }
    Ok(Some(sources))
}

fn encode_mrc_frames(
    sources: &[MrcSource],
    bg_quality: u8,
    bg_scale: f64,
) -> Result<Vec<MrcFrame>, Box<dyn std::error::Error>> {
    sources
        .iter()
        .map(|source| {
            let background = encode_raster_as_jpeg(&source.background, bg_quality, bg_scale)?;
            Ok(MrcFrame {
                background,
                mask_g4: source.mask_g4.clone(),
                mask_w: source.mask.width,
                mask_h: source.mask.height,
            })
        })
        .collect()
}

fn estimate_mrc_size(
    sources: &[MrcSource],
    bg_quality: u8,
    bg_scale: f64,
) -> Result<u64, Box<dyn std::error::Error>> {
    let frames = encode_mrc_frames(sources, bg_quality, bg_scale)?;
    Ok(build_mrc_pdf(&frames).len() as u64)
}

// ---------------------------------------------------------------------------

fn estimate_pdf_size(
    images: &[(PathBuf, image::DynamicImage)],
    quality: u8,
    scale: f64,
    allow_passthrough: bool,
) -> Result<u64, Box<dyn std::error::Error>> {
    let mut total: u64 = 500;
    for (path, img) in images {
        let (jpeg_bytes, _, _, _, _) = encode_image(path, img, quality, scale, allow_passthrough)?;
        total += jpeg_bytes.len() as u64 + 300;
    }
    Ok(total)
}

fn collect_images(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let extensions = ["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"];

    let mut paths: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| extensions.contains(&ext.to_lowercase().as_str()))
                .unwrap_or(false)
        })
        .collect();

    paths.sort();
    Ok(paths)
}

fn is_effectively_grayscale(img: &image::DynamicImage) -> bool {
    // If the image already carries a grayscale type, trust it without pixel sampling.
    match img.color() {
        image::ColorType::L8
        | image::ColorType::L16
        | image::ColorType::La8
        | image::ColorType::La16 => return true,
        _ => {}
    }

    // Scan every pixel with early exit on the first coloured one.
    // Colour images (the common case) exit within the first few pixels.
    // Only truly grey images pay the full O(n) cost.
    let rgb = img.to_rgb8();
    for pixel in rgb.pixels() {
        let [r, g, b] = pixel.0;
        if r.abs_diff(g).max(g.abs_diff(b)).max(r.abs_diff(b)) > 2 {
            return false;
        }
    }
    true
}

/// Returns the number of colour components declared in the JPEG SOF marker (1 or 3 for
/// well-formed files). Returns None if the file is too short or malformed.
fn jpeg_component_count(data: &[u8]) -> Option<u8> {
    // A valid JPEG starts with FF D8.
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return None;
    }
    let mut pos = 2;
    while pos + 3 < data.len() {
        if data[pos] != 0xFF {
            return None;
        }
        let marker = data[pos + 1];
        pos += 2;
        // SOF markers: C0–C3, C5–C7, C9–CB, CD–CF (excludes C4=DHT, C8=JPEG ext, CC=DAC)
        let is_sof = matches!(marker, 0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF);
        let segment_len = ((data[pos] as usize) << 8) | data[pos + 1] as usize;
        if is_sof && segment_len >= 8 && pos + segment_len <= data.len() {
            // SOF payload: 2 len, 1 precision, 2 height, 2 width, 1 components
            return Some(data[pos + 7]);
        }
        pos += segment_len;
    }
    None
}

fn encode_mozjpeg(
    pixels: &[u8],
    w: u32,
    h: u32,
    quality: u8,
    grayscale: bool,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let colorspace = if grayscale {
        MozCs::JCS_GRAYSCALE
    } else {
        MozCs::JCS_RGB
    };

    let mut comp = Compress::new(colorspace);
    comp.set_size(w as usize, h as usize);
    comp.set_quality(quality as f32);
    comp.set_progressive_mode();
    comp.set_optimize_scans(true);
    comp.set_optimize_coding(true);
    if !grayscale {
        // 4:2:0 chroma subsampling: imperceptible on photos, ~30% savings on chroma channels
        comp.set_chroma_sampling_pixel_sizes((2, 2), (2, 2));
    }

    let mut comp = comp.start_compress(Vec::new())?;
    comp.write_scanlines(pixels)?;
    Ok(comp.finish()?)
}

fn encode_raster_as_jpeg(
    img: &image::DynamicImage,
    quality: u8,
    scale: f64,
) -> Result<Frame, Box<dyn std::error::Error>> {
    let (orig_w, orig_h) = img.dimensions();
    let new_w = ((orig_w as f64) * scale).round() as u32;
    let new_h = ((orig_h as f64) * scale).round() as u32;
    // Clamp to 1×1 so neither codec receives a zero-dimension image when scale is tiny.
    let new_w = new_w.max(1);
    let new_h = new_h.max(1);

    let resized = if scale < 1.0 {
        img.resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3)
    } else {
        img.clone()
    };

    if is_effectively_grayscale(&resized) {
        let gray = resized.to_luma8();
        let (w, h) = gray.dimensions();
        let bytes = encode_mozjpeg(gray.as_raw(), w, h, quality, true)?;
        Ok((bytes, w, h, Cs::Gray, PdfFilter::Dct))
    } else {
        let rgb = resized.to_rgb8();
        let (w, h) = rgb.dimensions();
        let bytes = encode_mozjpeg(rgb.as_raw(), w, h, quality, false)?;
        Ok((bytes, w, h, Cs::Rgb, PdfFilter::Dct))
    }
}

fn encode_image(
    path: &Path,
    img: &image::DynamicImage,
    quality: u8,
    scale: f64,
    allow_passthrough: bool,
) -> Result<Frame, Box<dyn std::error::Error>> {
    // Passthrough: embed the original JPEG bytes without re-encoding, avoiding generation
    // loss. Only at quality=100/scale≈100% and only when no preprocessing is active
    // (preprocessing transforms the in-memory image; reading the file would bypass it).
    let is_jpeg_path = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("jpeg"));

    if allow_passthrough && quality == 100 && scale > 0.995 && is_jpeg_path {
        let bytes = fs::read(path)?;
        // Derive colour space from the JPEG file itself (not the decoded image)
        // to handle CMYK and other atypical encodings correctly.
        match jpeg_component_count(&bytes) {
            Some(1) => {
                let (w, h) = img.dimensions();
                return Ok((bytes, w, h, Cs::Gray, PdfFilter::Dct));
            }
            Some(3) => {
                let (w, h) = img.dimensions();
                return Ok((bytes, w, h, Cs::Rgb, PdfFilter::Dct));
            }
            _ => {} // CMYK (4) or unknown — fall through to re-encode
        }
    }

    encode_raster_as_jpeg(img, quality, scale)
}

// ---------------------------------------------------------------------------
// JPEG2000 encoding
// ---------------------------------------------------------------------------

struct Jp2Buffer {
    data: Vec<u8>,
    pos: usize,
}

unsafe extern "C" fn jp2_write(buf: *mut c_void, nb: usize, ud: *mut c_void) -> usize {
    unsafe {
        let b = &mut *(ud as *mut Jp2Buffer);
        let src = std::slice::from_raw_parts(buf as *const u8, nb);
        let end = b.pos + nb;
        if end > b.data.len() {
            b.data.resize(end, 0);
        }
        b.data[b.pos..end].copy_from_slice(src);
        b.pos = end;
        nb
    }
}

unsafe extern "C" fn jp2_seek(off: i64, ud: *mut c_void) -> i32 {
    unsafe { (*(ud as *mut Jp2Buffer)).pos = off as usize };
    1
}

unsafe extern "C" fn jp2_skip(nb: i64, ud: *mut c_void) -> i64 {
    unsafe {
        let b = &mut *(ud as *mut Jp2Buffer);
        b.pos = (b.pos as i64 + nb) as usize;
    }
    nb
}

fn encode_as_jp2(
    img: &image::DynamicImage,
    bpp: f32,
    scale: f64,
) -> Result<Frame, Box<dyn std::error::Error>> {
    let (orig_w, orig_h) = img.dimensions();
    let new_w = ((orig_w as f64) * scale).round() as u32;
    let new_h = ((orig_h as f64) * scale).round() as u32;
    let new_w = new_w.max(1);
    let new_h = new_h.max(1);

    let resized = if scale < 1.0 {
        img.resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3)
    } else {
        img.clone()
    };

    let gray = is_effectively_grayscale(&resized);
    let bytes = if gray {
        let luma = resized.to_luma8();
        let (w, h) = luma.dimensions();
        encode_jp2_raw(luma.as_raw(), w, h, bpp, true)?
    } else {
        let rgb = resized.to_rgb8();
        let (w, h) = rgb.dimensions();
        encode_jp2_raw(rgb.as_raw(), w, h, bpp, false)?
    };

    let (w, h) = (new_w, new_h);
    let cs = if gray { Cs::Gray } else { Cs::Rgb };
    Ok((bytes, w, h, cs, PdfFilter::Jpx))
}

fn encode_jp2_raw(
    pixels: &[u8],
    w: u32,
    h: u32,
    bpp: f32,
    grayscale: bool,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let nc = if grayscale { 1u32 } else { 3u32 };
    let cs = if grayscale {
        OPJ_CLRSPC_GRAY
    } else {
        OPJ_CLRSPC_SRGB
    };

    let comp = opj_image_comptparm {
        dx: 1,
        dy: 1,
        w,
        h,
        x0: 0,
        y0: 0,
        prec: 8,
        bpp: 8,
        sgnd: 0,
    };
    let mut img =
        opj_image::create(&vec![comp; nc as usize], cs).ok_or("JP2: failed to create opj_image")?;
    img.x1 = w;
    img.y1 = h;

    let n = (w * h) as usize;
    if let Some(comps) = img.comps_mut() {
        for (c, comp) in comps.iter_mut().enumerate() {
            if let Some(d) = comp.data_mut() {
                if grayscale {
                    for i in 0..n {
                        d[i] = pixels[i] as i32;
                    }
                } else {
                    for i in 0..n {
                        d[i] = pixels[i * 3 + c] as i32;
                    }
                }
            }
        }
    }

    // compression rate = total_bits / target_bits = (nc * 8) / bpp
    let rate = (nc as f32 * 8.0) / bpp;

    // OpenJPEG requires 2^(numresolution-1) <= min(w, h).
    // Exceeding this causes start_compress to fail silently on small images.
    let max_res = (w.min(h) as f64).log2().floor() as i32 + 1;
    let numresolution = max_res.clamp(1, 6);

    let mut params = opj_cparameters_t {
        tcp_numlayers: 1,
        cp_disto_alloc: 1,
        numresolution,
        irreversible: 1,
        ..opj_cparameters_t::default()
    };
    params.tcp_rates[0] = rate;

    let mut buf = Jp2Buffer {
        data: Vec::new(),
        pos: 0,
    };
    unsafe {
        let codec = opj_create_compress(OPJ_CODEC_JP2);
        if codec.is_null() {
            return Err("JP2: failed to create codec".into());
        }
        if opj_setup_encoder(codec, &mut params, &mut *img as *mut _) == 0 {
            opj_destroy_codec(codec);
            return Err("JP2: setup_encoder failed".into());
        }
        let stream = opj_stream_create(1 << 16, 0);
        opj_stream_set_write_function(stream, Some(jp2_write));
        opj_stream_set_seek_function(stream, Some(jp2_seek));
        opj_stream_set_skip_function(stream, Some(jp2_skip));
        opj_stream_set_user_data(stream, &mut buf as *mut _ as *mut c_void, None);
        let r1 = opj_start_compress(codec, &mut *img, stream);
        let r2 = if r1 != 0 {
            opj_encode(codec, stream)
        } else {
            0
        };
        let r3 = if r2 != 0 {
            opj_end_compress(codec, stream)
        } else {
            0
        };
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        if r1 == 0 {
            return Err(format!(
                "JP2: start_compress failed ({}×{}, numres={})",
                w, h, numresolution
            )
            .into());
        }
        if r2 == 0 {
            return Err(
                format!("JP2: encode failed ({}×{}, numres={})", w, h, numresolution).into(),
            );
        }
        if r3 == 0 {
            return Err(format!(
                "JP2: end_compress failed ({}×{}, numres={})",
                w, h, numresolution
            )
            .into());
        }
    }
    Ok(buf.data)
}

fn estimate_jp2_size(
    images: &[(PathBuf, image::DynamicImage)],
    bpp: f32,
    scale: f64,
) -> Result<u64, Box<dyn std::error::Error>> {
    let mut total: u64 = 500;
    for (_, img) in images {
        let (bytes, _, _, _, _) = encode_as_jp2(img, bpp, scale)?;
        total += bytes.len() as u64 + 300;
    }
    Ok(total)
}

// ---------------------------------------------------------------------------

fn build_pdf(images: &[Frame]) -> Vec<u8> {
    let mut pdf = Pdf::new();

    let catalog_id = Ref::new(1);
    let page_tree_id = Ref::new(2);
    let base_ref = 3;

    let mut page_ids = Vec::new();

    for (i, (img_data, w, h, cs, filter)) in images.iter().enumerate() {
        let page_id = Ref::new((base_ref + i * 3) as i32);
        let content_id = Ref::new((base_ref + i * 3 + 1) as i32);
        let image_id = Ref::new((base_ref + i * 3 + 2) as i32);

        page_ids.push(page_id);

        let width_pt = *w as f32 * 72.0 / 150.0;
        let height_pt = *h as f32 * 72.0 / 150.0;

        let mut image_obj = pdf.image_xobject(image_id, img_data);
        image_obj.filter(match filter {
            PdfFilter::Dct => pdf_writer::Filter::DctDecode,
            PdfFilter::Jpx => pdf_writer::Filter::JpxDecode,
        });
        image_obj.width(*w as i32);
        image_obj.height(*h as i32);
        match cs {
            Cs::Gray => {
                image_obj.color_space().device_gray();
            }
            Cs::Rgb => {
                image_obj.color_space().device_rgb();
            }
        }
        image_obj.bits_per_component(8);
        image_obj.finish();

        let mut content = Content::new();
        content.save_state();
        content.transform([width_pt, 0.0, 0.0, height_pt, 0.0, 0.0]);
        content.x_object(Name(b"Img0"));
        content.restore_state();
        let content_data = content.finish();

        pdf.stream(content_id, &content_data);

        let mut page = pdf.page(page_id);
        page.media_box(Rect::new(0.0, 0.0, width_pt, height_pt));
        page.parent(page_tree_id);
        page.contents(content_id);
        page.resources().x_objects().pair(Name(b"Img0"), image_id);
        page.finish();
    }

    let mut pages = pdf.pages(page_tree_id);
    pages.count(images.len() as i32);
    pages.kids(page_ids);
    pages.finish();

    pdf.catalog(catalog_id).pages(page_tree_id);

    pdf.finish()
}

fn build_mrc_pdf(frames: &[MrcFrame]) -> Vec<u8> {
    let mut pdf = Pdf::new();

    let catalog_id = Ref::new(1);
    let page_tree_id = Ref::new(2);
    let base_ref = 3;

    let mut page_ids = Vec::new();

    for (i, frame) in frames.iter().enumerate() {
        let page_id = Ref::new((base_ref + i * 4) as i32);
        let content_id = Ref::new((base_ref + i * 4 + 1) as i32);
        let bg_id = Ref::new((base_ref + i * 4 + 2) as i32);
        let mask_id = Ref::new((base_ref + i * 4 + 3) as i32);

        page_ids.push(page_id);

        let page_width_pt = frame.mask_w as f32 * 72.0 / 150.0;
        let page_height_pt = frame.mask_h as f32 * 72.0 / 150.0;

        let (bg_data, bg_w, bg_h, bg_cs, bg_filter) = &frame.background;
        let mut bg_obj = pdf.image_xobject(bg_id, bg_data);
        bg_obj.filter(match bg_filter {
            PdfFilter::Dct => pdf_writer::Filter::DctDecode,
            PdfFilter::Jpx => pdf_writer::Filter::JpxDecode,
        });
        bg_obj.width(*bg_w as i32);
        bg_obj.height(*bg_h as i32);
        match bg_cs {
            Cs::Gray => {
                bg_obj.color_space().device_gray();
            }
            Cs::Rgb => {
                bg_obj.color_space().device_rgb();
            }
        }
        bg_obj.bits_per_component(8);
        bg_obj.finish();

        let mut mask_obj = pdf.image_xobject(mask_id, &frame.mask_g4);
        mask_obj.filter(pdf_writer::Filter::CcittFaxDecode);
        {
            let mut parms = mask_obj.decode_parms();
            parms
                .k(-1)
                .columns(frame.mask_w as i32)
                .rows(frame.mask_h as i32)
                .black_is_1(false);
        }
        mask_obj.width(frame.mask_w as i32);
        mask_obj.height(frame.mask_h as i32);
        mask_obj.image_mask(true);
        mask_obj.bits_per_component(1);
        // CCITT with BlackIs1=false decodes text pixels as 0. Invert the image mask
        // so text samples paint with the current fill color and background stays transparent.
        mask_obj.decode([1.0, 0.0]);
        mask_obj.finish();

        let mut content = Content::new();
        content.save_state();
        content.transform([page_width_pt, 0.0, 0.0, page_height_pt, 0.0, 0.0]);
        content.x_object(Name(b"Bg"));
        content.restore_state();
        content.save_state();
        content.set_fill_gray(0.0);
        content.transform([page_width_pt, 0.0, 0.0, page_height_pt, 0.0, 0.0]);
        content.x_object(Name(b"Mask"));
        content.restore_state();
        let content_data = content.finish();

        pdf.stream(content_id, &content_data);

        let mut page = pdf.page(page_id);
        page.media_box(Rect::new(0.0, 0.0, page_width_pt, page_height_pt));
        page.parent(page_tree_id);
        page.contents(content_id);
        {
            let mut resources = page.resources();
            let mut x_objects = resources.x_objects();
            x_objects.pair(Name(b"Bg"), bg_id);
            x_objects.pair(Name(b"Mask"), mask_id);
        }
        page.finish();
    }

    let mut pages = pdf.pages(page_tree_id);
    pages.count(frames.len() as i32);
    pages.kids(page_ids);
    pages.finish();

    pdf.catalog(catalog_id).pages(page_tree_id);

    pdf.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, GrayImage, Luma, Rgb, RgbImage};

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

    #[test]
    fn build_mrc_pdf_contains_background_and_ccitt_mask() {
        let img = make_text_scan();
        let images = vec![(PathBuf::from("page.png"), img)];
        let sources = prepare_mrc_sources(&images).unwrap().unwrap();
        let frames = encode_mrc_frames(&sources, 35, 0.5).unwrap();
        let pdf = build_mrc_pdf(&frames);
        let text = String::from_utf8_lossy(&pdf);
        assert!(text.contains("DCTDecode"));
        assert!(text.contains("CCITTFaxDecode"));
        assert!(text.contains("ImageMask true"));
        assert!(text.contains("/K -1"));
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
