use imageproc::contrast::{ThresholdType, otsu_level, threshold};
use imageproc::filter::median_filter;
use imageproc::geometric_transformations::{Interpolation, rotate_about_center_no_crop};

use super::types::PreprocOpts;

pub(crate) fn preprocess(img: image::DynamicImage, opts: &PreprocOpts) -> image::DynamicImage {
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
pub(crate) fn estimate_background(rgb: &image::RgbImage) -> [u8; 3] {
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
