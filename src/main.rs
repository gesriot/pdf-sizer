use image::GenericImageView;
use image::codecs::jpeg::JpegEncoder;
use pdf_writer::{Content, Finish, Name, Pdf, Rect, Ref};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Использование: {} <целевой_размер_МБ>", args[0]);
        eprintln!("Изображения берутся из папки images/ рядом с исполняемым файлом");
        std::process::exit(1);
    }

    let target_mb: f64 = args[1].parse().map_err(|_| {
        eprintln!("Ошибка: '{}' не является числом", args[1]);
        "invalid number"
    })?;

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

    let images: Vec<image::DynamicImage> = image_paths
        .iter()
        .map(|p| {
            image::open(p).map_err(|e| {
                eprintln!("Ошибка открытия {}: {}", p.display(), e);
                e
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // For each quality level, binary search for the scale that produces target size
    let quality_steps: Vec<u8> = (1..=20).rev().map(|i| i * 5).collect(); // 100, 95, ..., 5
    let mut variants: Vec<(String, u8, u32, f64)> = Vec::new(); // (filename, q, scale, size_mb)

    println!(
        "\nПоиск вариантов: для каждого quality ищем scale, дающий ~{:.0} MB...\n",
        target_mb
    );

    for &q in &quality_steps {
        // Quick check: at scale=100%, is the PDF already smaller than target-tolerance?
        // If so, skip this and all lower quality levels.
        let max_size = estimate_pdf_size(&images, q, 1.0)?;
        if max_size < target_bytes.saturating_sub(tolerance_bytes) {
            println!(
                "  quality={:3}: даже при scale=100% размер {:.1} MB — меньше цели, пропуск остальных",
                q,
                max_size as f64 / (1024.0 * 1024.0)
            );
            break;
        }

        // At scale=5%, is the PDF still bigger than target+tolerance?
        let min_size = estimate_pdf_size(&images, q, 0.05)?;
        if min_size > target_bytes + tolerance_bytes {
            println!(
                "  quality={:3}: даже при scale=5% размер {:.1} MB — больше цели, пропуск",
                q,
                min_size as f64 / (1024.0 * 1024.0)
            );
            continue;
        }

        // Binary search for the right scale
        let mut lo = 5.0_f64;
        let mut hi = 100.0_f64;
        let mut best: Option<(u32, Vec<u8>, f64)> = None;

        for _ in 0..20 {
            if hi - lo < 0.5 {
                break;
            }
            let mid = ((lo + hi) / 2.0).round().clamp(5.0, 100.0);
            let s = mid / 100.0;

            let jpegs: Vec<(Vec<u8>, u32, u32)> = images
                .iter()
                .map(|img| encode_image_as_jpeg(img, q, s))
                .collect::<Result<Vec<_>, _>>()?;
            let pdf_bytes = build_pdf_from_jpegs(&jpegs);
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

        // If exact hit not found, try the midpoint as last attempt
        if best.is_none() {
            let mid = ((lo + hi) / 2.0).round().clamp(5.0, 100.0);
            let s = mid / 100.0;
            let jpegs: Vec<(Vec<u8>, u32, u32)> = images
                .iter()
                .map(|img| encode_image_as_jpeg(img, q, s))
                .collect::<Result<Vec<_>, _>>()?;
            let pdf_bytes = build_pdf_from_jpegs(&jpegs);
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
            variants.push((filename, q, s_pct, size_mb));
        } else {
            println!("  quality={:3}: не удалось попасть в целевой диапазон", q);
        }
    }

    println!("\n--- Итого ---");
    if variants.is_empty() {
        println!("Не удалось создать ни одного варианта в целевом диапазоне.");
    } else {
        println!("Создано вариантов: {}\n", variants.len());
        println!(
            "  {:>7}  {:>5}  {:>10}  {}",
            "quality", "scale", "размер", "файл"
        );
        println!("  {}", "-".repeat(50));
        for (filename, q, s, size_mb) in &variants {
            let hint = if *q > 80 {
                "  <- чёткость"
            } else if *s > 80 {
                "  <- детали"
            } else {
                ""
            };
            println!(
                "  {:>5}    {:>3}%  {:>7.1} MB  {}{}",
                q, s, size_mb, filename, hint
            );
        }
        println!("\n  Высокий quality = меньше артефактов сжатия (чёткие края, текст)");
        println!("  Высокий scale   = больше деталей и разрешение (мелкие элементы)");
    }

    Ok(())
}

/// Quick estimate of PDF size without building the full PDF.
/// Just sums up JPEG sizes + small overhead per page.
fn estimate_pdf_size(
    images: &[image::DynamicImage],
    quality: u8,
    scale: f64,
) -> Result<u64, Box<dyn std::error::Error>> {
    let mut total: u64 = 500; // base PDF overhead
    for img in images {
        let (jpeg_bytes, _, _) = encode_image_as_jpeg(img, quality, scale)?;
        total += jpeg_bytes.len() as u64 + 300; // ~300 bytes overhead per page
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

fn encode_image_as_jpeg(
    img: &image::DynamicImage,
    quality: u8,
    scale: f64,
) -> Result<(Vec<u8>, u32, u32), Box<dyn std::error::Error>> {
    let (orig_w, orig_h) = img.dimensions();
    let new_w = ((orig_w as f64) * scale).round() as u32;
    let new_h = ((orig_h as f64) * scale).round() as u32;

    let resized = if scale < 1.0 {
        img.resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3)
    } else {
        img.clone()
    };

    let rgb = resized.to_rgb8();
    let (w, h) = rgb.dimensions();

    let mut jpeg_bytes = Vec::new();
    let encoder = JpegEncoder::new_with_quality(&mut jpeg_bytes, quality);
    image::ImageEncoder::write_image(encoder, &rgb, w, h, image::ExtendedColorType::Rgb8)?;

    Ok((jpeg_bytes, w, h))
}

fn build_pdf_from_jpegs(jpegs: &[(Vec<u8>, u32, u32)]) -> Vec<u8> {
    let mut pdf = Pdf::new();

    let catalog_id = Ref::new(1);
    let page_tree_id = Ref::new(2);

    // Reserve refs: for each image we need page + content stream + image xobject = 3 refs
    let base_ref = 3;

    let mut page_ids = Vec::new();

    for (i, (jpeg_data, w, h)) in jpegs.iter().enumerate() {
        let page_id = Ref::new((base_ref + i * 3) as i32);
        let content_id = Ref::new((base_ref + i * 3 + 1) as i32);
        let image_id = Ref::new((base_ref + i * 3 + 2) as i32);

        page_ids.push(page_id);

        // Page dimensions in points (assume 150 DPI)
        let width_pt = *w as f32 * 72.0 / 150.0;
        let height_pt = *h as f32 * 72.0 / 150.0;

        // Image XObject
        let mut image_obj = pdf.image_xobject(image_id, jpeg_data);
        image_obj.filter(pdf_writer::Filter::DctDecode);
        image_obj.width(*w as i32);
        image_obj.height(*h as i32);
        image_obj.color_space().device_rgb();
        image_obj.bits_per_component(8);
        image_obj.finish();

        // Content stream: draw image scaled to full page
        let mut content = Content::new();
        content.save_state();
        content.transform([width_pt, 0.0, 0.0, height_pt, 0.0, 0.0]);
        content.x_object(Name(b"Img0"));
        content.restore_state();
        let content_data = content.finish();

        pdf.stream(content_id, &content_data);

        // Page
        let mut page = pdf.page(page_id);
        page.media_box(Rect::new(0.0, 0.0, width_pt, height_pt));
        page.parent(page_tree_id);
        page.contents(content_id);
        page.resources()
            .x_objects()
            .pair(Name(b"Img0"), image_id);
        page.finish();
    }

    // Page tree
    let mut pages = pdf.pages(page_tree_id);
    pages.count(jpegs.len() as i32);
    pages.kids(page_ids);
    pages.finish();

    // Catalog
    pdf.catalog(catalog_id).pages(page_tree_id);

    pdf.finish()
}
