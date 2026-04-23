use image::GenericImageView;
use mozjpeg::{ColorSpace as MozCs, Compress};
use pdf_writer::{Content, Finish, Name, Pdf, Rect, Ref};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
enum Cs {
    Rgb,
    Gray,
}

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

    let images: Vec<(PathBuf, image::DynamicImage)> = image_paths
        .iter()
        .map(|p| {
            image::open(p)
                .map(|img| (p.clone(), img))
                .map_err(|e| {
                    eprintln!("Ошибка открытия {}: {}", p.display(), e);
                    e
                })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let quality_steps: Vec<u8> = (1..=20).rev().map(|i| i * 5).collect(); // 100, 95, ..., 5
    let mut variants: Vec<(String, u8, u32, f64)> = Vec::new();

    println!(
        "\nПоиск вариантов: для каждого quality ищем scale, дающий ~{:.0} MB...\n",
        target_mb
    );

    for &q in &quality_steps {
        let max_size = estimate_pdf_size(&images, q, 1.0)?;
        if max_size < target_bytes.saturating_sub(tolerance_bytes) {
            println!(
                "  quality={:3}: даже при scale=100% размер {:.1} MB — меньше цели, пропуск остальных",
                q,
                max_size as f64 / (1024.0 * 1024.0)
            );
            break;
        }

        let min_size = estimate_pdf_size(&images, q, 0.05)?;
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

            let jpegs: Vec<(Vec<u8>, u32, u32, Cs)> = images
                .iter()
                .map(|(path, img)| encode_image(path, img, q, s))
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
            let jpegs: Vec<(Vec<u8>, u32, u32, Cs)> = images
                .iter()
                .map(|(path, img)| encode_image(path, img, q, s))
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

fn estimate_pdf_size(
    images: &[(PathBuf, image::DynamicImage)],
    quality: u8,
    scale: f64,
) -> Result<u64, Box<dyn std::error::Error>> {
    let mut total: u64 = 500;
    for (path, img) in images {
        let (jpeg_bytes, _, _, _) = encode_image(path, img, quality, scale)?;
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
    let colorspace = if grayscale { MozCs::JCS_GRAYSCALE } else { MozCs::JCS_RGB };

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

fn encode_image(
    path: &Path,
    img: &image::DynamicImage,
    quality: u8,
    scale: f64,
) -> Result<(Vec<u8>, u32, u32, Cs), Box<dyn std::error::Error>> {
    // Passthrough: only at quality=100 and scale≈100%, so binary search sees consistent
    // mozjpeg output for every other quality level and duplicate variants are avoided.
    if quality == 100 && scale > 0.995 {
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg") {
                let bytes = fs::read(path)?;
                // Derive colour space from the JPEG file itself (not the decoded image)
                // to handle CMYK and other atypical encodings correctly.
                match jpeg_component_count(&bytes) {
                    Some(1) => {
                        let (w, h) = img.dimensions();
                        return Ok((bytes, w, h, Cs::Gray));
                    }
                    Some(3) => {
                        let (w, h) = img.dimensions();
                        return Ok((bytes, w, h, Cs::Rgb));
                    }
                    _ => {} // CMYK (4) or unknown — fall through to re-encode
                }
            }
        }
    }

    let (orig_w, orig_h) = img.dimensions();
    let new_w = ((orig_w as f64) * scale).round() as u32;
    let new_h = ((orig_h as f64) * scale).round() as u32;

    let resized = if scale < 1.0 {
        img.resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3)
    } else {
        img.clone()
    };

    if is_effectively_grayscale(&resized) {
        let gray = resized.to_luma8();
        let (w, h) = gray.dimensions();
        let bytes = encode_mozjpeg(gray.as_raw(), w, h, quality, true)?;
        Ok((bytes, w, h, Cs::Gray))
    } else {
        let rgb = resized.to_rgb8();
        let (w, h) = rgb.dimensions();
        let bytes = encode_mozjpeg(rgb.as_raw(), w, h, quality, false)?;
        Ok((bytes, w, h, Cs::Rgb))
    }
}

fn build_pdf(images: &[(Vec<u8>, u32, u32, Cs)]) -> Vec<u8> {
    let mut pdf = Pdf::new();

    let catalog_id = Ref::new(1);
    let page_tree_id = Ref::new(2);
    let base_ref = 3;

    let mut page_ids = Vec::new();

    for (i, (jpeg_data, w, h, cs)) in images.iter().enumerate() {
        let page_id = Ref::new((base_ref + i * 3) as i32);
        let content_id = Ref::new((base_ref + i * 3 + 1) as i32);
        let image_id = Ref::new((base_ref + i * 3 + 2) as i32);

        page_ids.push(page_id);

        let width_pt = *w as f32 * 72.0 / 150.0;
        let height_pt = *h as f32 * 72.0 / 150.0;

        let mut image_obj = pdf.image_xobject(image_id, jpeg_data);
        image_obj.filter(pdf_writer::Filter::DctDecode);
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
        page.resources()
            .x_objects()
            .pair(Name(b"Img0"), image_id);
        page.finish();
    }

    let mut pages = pdf.pages(page_tree_id);
    pages.count(images.len() as i32);
    pages.kids(page_ids);
    pages.finish();

    pdf.catalog(catalog_id).pages(page_tree_id);

    pdf.finish()
}
