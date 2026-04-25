use std::collections::{HashMap, HashSet};

use imageproc::contrast::adaptive_threshold;
use imageproc::region_labelling::{Connectivity, connected_components};

use super::preprocess::estimate_background;
use super::types::{ComponentStats, TextMask};

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
pub(crate) fn is_line_like(stats: ComponentStats, page_w: u32, page_h: u32) -> bool {
    let cw = stats.width();
    let ch = stats.height();
    let long_h = page_w as f32 * 0.15;
    let long_v = page_h as f32 * 0.15;
    (ch <= 3 && cw as f32 > long_h) || (cw <= 3 && ch as f32 > long_v)
}

pub(crate) fn segment_text_mask(img: &image::DynamicImage) -> TextMask {
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

pub(crate) fn is_mrc_suitable(mask: &TextMask) -> bool {
    // Coverage gate rejects near-empty pages and pages where segmentation blew up.
    // line_ratio gate rejects pages dominated by table rules / diagram lines — those
    // should stay in the background JPEG rather than becoming a 1-bit mask.
    (0.005..=0.70).contains(&mask.coverage) && mask.line_ratio <= 0.30
}

pub(crate) fn fill_masked_background(
    img: &image::DynamicImage,
    mask: &TextMask,
) -> image::DynamicImage {
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
