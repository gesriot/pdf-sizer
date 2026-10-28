//! Page segmentation for MRC: text and line art become a 1-bit mask, illustrations are cut
//! out as separate pictures, everything else is background.
//!
//! A pixel is ink when it is darker than the local paper level by more than ~42 % of the
//! local paper-to-ink contrast.

use super::raster::{area_resize, bilinear_resize, box_blur, dilate, label_components, max_filter, min_filter};

/// Share of the local paper-to-ink contrast a pixel must reach to count as ink.
const INK_THRESHOLD: f32 = 0.42;
/// Share of the contrast above which an ink pixel is solid stroke, not an antialiased edge.
const CORE_THRESHOLD: f32 = 0.6;
/// Minimum luma difference from the paper for ink, so paper noise never enters the mask.
const MIN_CONTRAST: f32 = 28.0;
/// RGB distance from the paper colour above which a pixel is "not paper".
const PAPER_DISTANCE: f32 = 30.0;

/// Illustration area in source pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PicRect {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) w: u32,
    pub(crate) h: u32,
}

pub(crate) struct PageSegmentation {
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// Text and line-art pixels at source resolution, pictures excluded.
    pub(crate) ink: Vec<bool>,
    /// Ink pixels that carry the full stroke colour (used for the foreground colour).
    pub(crate) core: Vec<bool>,
    pub(crate) pictures: Vec<PicRect>,
}

fn luma(rgb: &image::RgbImage) -> Vec<f32> {
    rgb.pixels()
        .map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)
        .collect()
}

/// Local paper level: grey-level dilation removes strokes narrower than ~5 mm, then
/// smoothing; computed at ~75 dpi and upsampled.
fn paper_level(gray: &[f32], w: usize, h: usize, dpi: f32) -> Vec<f32> {
    let k = (dpi / 75.0).round().max(1.0) as usize;
    let (lw, lh) = (w.div_ceil(k), h.div_ceil(k));
    let low = area_resize(gray, w, h, lw, lh);
    let r = ((32.0 * dpi / 300.0) / k as f32).round().max(2.0) as usize;
    let low = max_filter(&low, lw, lh, r);
    let low = box_blur(&box_blur(&low, lw, lh, r / 2), lw, lh, r / 2);
    bilinear_resize(&low, lw, lh, w, h)
}

/// Median colour of the brightest 40 % of the page: the paper colour.
fn paper_colour(rgb: &image::RgbImage, gray: &[f32]) -> [f32; 3] {
    let mut hist = [0u32; 256];
    for &g in gray {
        hist[g as usize] += 1;
    }
    let mut cut = 0;
    let mut seen = 0u64;
    let target = gray.len() as u64 * 60 / 100;
    for (v, &c) in hist.iter().enumerate() {
        seen += c as u64;
        if seen >= target {
            cut = v;
            break;
        }
    }
    let mut ch = [[0u32; 256]; 3];
    let mut n = 0u64;
    for (p, &g) in rgb.pixels().zip(gray) {
        if g as usize >= cut {
            for k in 0..3 {
                ch[k][p[k] as usize] += 1;
            }
            n += 1;
        }
    }
    let mut out = [255.0; 3];
    for k in 0..3 {
        let mut acc = 0u64;
        for (v, &c) in ch[k].iter().enumerate() {
            acc += c as u64;
            if acc * 2 >= n {
                out[k] = v as f32;
                break;
            }
        }
    }
    out
}

pub(crate) fn segment_page(img: &image::DynamicImage, dpi: f32) -> PageSegmentation {
    let rgb = img.to_rgb8();
    let (wu, hu) = rgb.dimensions();
    let (w, h) = (wu as usize, hu as usize);
    let gray = luma(&rgb);

    let paper = paper_level(&gray, w, h, dpi);
    let ir = (8.0 * dpi / 300.0).round().max(2.0) as usize;
    let ink_level = box_blur(&min_filter(&gray, w, h, ir), w, h, ir);

    let mut ink = vec![false; w * h];
    let mut core = vec![false; w * h];
    for i in 0..w * h {
        let contrast = (paper[i] - ink_level[i]).max(0.0);
        let dark = paper[i] - gray[i];
        if dark > (INK_THRESHOLD * contrast).max(MIN_CONTRAST) {
            ink[i] = true;
            core[i] = dark > CORE_THRESHOLD * contrast;
        }
    }
    drop(ink_level);
    drop(paper);

    // Specks: components of a few pixels.
    let min_area = (3.0 * (dpi / 300.0).powi(2)).round().max(2.0) as u32;
    let (labels, stats) = label_components(&ink, w, h);
    for (i, &l) in labels.iter().enumerate() {
        if l != 0 && stats[l as usize - 1].area < min_area {
            ink[i] = false;
            core[i] = false;
        }
    }
    drop(labels);

    let pictures = find_pictures(&rgb, &gray, &ink, dpi);
    for r in &pictures {
        for y in r.y..r.y + r.h {
            let row = y as usize * w;
            for x in r.x as usize..(r.x + r.w) as usize {
                ink[row + x] = false;
                core[row + x] = false;
            }
        }
    }

    PageSegmentation {
        width: wu,
        height: hu,
        ink,
        core,
        pictures,
    }
}

#[derive(Clone, Copy, Default)]
struct Cell {
    keep: u32,
    nonpaper: u32,
    sum: [f64; 3],
    luma: f64,
    luma2: f64,
}

/// Illustrations: connected areas of non-paper cells (~2.7 mm) that vary in colour or
/// texture. Flat tinted boxes stay in the background, their text stays in the mask.
fn find_pictures(rgb: &image::RgbImage, gray: &[f32], ink: &[bool], dpi: f32) -> Vec<PicRect> {
    let (wu, hu) = rgb.dimensions();
    let (w, h) = (wu as usize, hu as usize);
    let c = (32.0 * dpi / 300.0).round().max(4.0) as usize;
    let (cw, ch) = (w.div_ceil(c), h.div_ceil(c));
    let paper = paper_colour(rgb, gray);
    let near_ink = dilate(ink, w, h, 2);
    let dist = |p: &image::Rgb<u8>| {
        let d: f32 = (0..3).map(|k| (p[k] as f32 - paper[k]).powi(2)).sum();
        d.sqrt()
    };

    let mut cells = vec![Cell::default(); cw * ch];
    for (i, p) in rgb.pixels().enumerate() {
        if near_ink[i] {
            continue;
        }
        let cell = &mut cells[(i / w / c) * cw + (i % w) / c];
        cell.keep += 1;
        if dist(p) > PAPER_DISTANCE {
            cell.nonpaper += 1;
            for k in 0..3 {
                cell.sum[k] += p[k] as f64;
            }
            let g = gray[i] as f64;
            cell.luma += g;
            cell.luma2 += g * g;
        }
    }
    let cell_area = |i: usize| {
        let cx = i % cw;
        let cy = i / cw;
        ((w - cx * c).min(c) * (h - cy * c).min(c)) as u32
    };
    let ink_cell: Vec<bool> = (0..cw * ch).map(|i| cells[i].keep * 4 < cell_area(i)).collect();
    let nonpaper: Vec<bool> = (0..cw * ch)
        .map(|i| !ink_cell[i] && cells[i].nonpaper * 2 > cells[i].keep)
        .collect();
    let near_nonpaper = dilate(&nonpaper, cw, ch, 1);
    let cand: Vec<bool> = (0..cw * ch).map(|i| nonpaper[i] || (ink_cell[i] && near_nonpaper[i])).collect();

    let (labels, stats) = label_components(&cand, cw, ch);
    let min_side = (dpi * 0.5) as usize;
    let mut out = Vec::new();
    for (li, st) in stats.iter().enumerate() {
        if (st.width() as usize) * c < min_side || (st.height() as usize) * c < min_side {
            continue;
        }
        let members: Vec<&Cell> = labels
            .iter()
            .enumerate()
            .filter(|&(i, &l)| l as usize == li + 1 && nonpaper[i])
            .map(|(i, _)| &cells[i])
            .collect();
        if members.len() < 4 {
            continue;
        }
        let means: Vec<[f64; 3]> = members
            .iter()
            .map(|m| {
                let n = m.nonpaper.max(1) as f64;
                [m.sum[0] / n, m.sum[1] / n, m.sum[2] / n]
            })
            .collect();
        let mut spread = 0.0;
        for k in 0..3 {
            let mean = means.iter().map(|m| m[k]).sum::<f64>() / means.len() as f64;
            spread += means.iter().map(|m| (m[k] - mean).powi(2)).sum::<f64>() / means.len() as f64;
        }
        let spread = spread.sqrt();
        let mut inner: Vec<f64> = members
            .iter()
            .map(|m| {
                let n = m.nonpaper.max(1) as f64;
                let mean = m.luma / n;
                (m.luma2 / n - mean * mean).max(0.0).sqrt()
            })
            .collect();
        inner.sort_by(f64::total_cmp);
        let inner = inner[inner.len() / 2];
        if spread < 12.0 && inner < 6.0 {
            continue;
        }
        let x0 = st.min_x as usize * c;
        let y0 = st.min_y as usize * c;
        let x1 = (st.max_x as usize + 1) * c;
        let y1 = (st.max_y as usize + 1) * c;
        out.push(refine_rect(rgb, &paper, (x0, y0, x1, y1), c));
    }
    out
}

/// Grow the cell-aligned rectangle by one cell, then trim rows/columns that are mostly paper.
fn refine_rect(rgb: &image::RgbImage, paper: &[f32; 3], rect: (usize, usize, usize, usize), c: usize) -> PicRect {
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let (x0, y0, x1, y1) = rect;
    let (mut x0, mut y0) = (x0.saturating_sub(c), y0.saturating_sub(c));
    let (mut x1, mut y1) = ((x1 + c).min(w), (y1 + c).min(h));
    let not_paper = |x: usize, y: usize| {
        let p = rgb.get_pixel(x as u32, y as u32);
        let d: f32 = (0..3).map(|k| (p[k] as f32 - paper[k]).powi(2)).sum();
        d.sqrt() > PAPER_DISTANCE
    };
    let col_frac =
        |x: usize, y0: usize, y1: usize| (y0..y1).filter(|&y| not_paper(x, y)).count() as f32 / (y1 - y0) as f32;
    let row_frac =
        |y: usize, x0: usize, x1: usize| (x0..x1).filter(|&x| not_paper(x, y)).count() as f32 / (x1 - x0) as f32;
    while x1 - x0 > 1 && col_frac(x0, y0, y1) < 0.2 {
        x0 += 1;
    }
    while x1 - x0 > 1 && col_frac(x1 - 1, y0, y1) < 0.2 {
        x1 -= 1;
    }
    while y1 - y0 > 1 && row_frac(y0, x0, x1) < 0.2 {
        y0 += 1;
    }
    while y1 - y0 > 1 && row_frac(y1 - 1, x0, x1) < 0.2 {
        y1 -= 1;
    }
    PicRect {
        x: x0 as u32,
        y: y0 as u32,
        w: (x1 - x0) as u32,
        h: (y1 - y0) as u32,
    }
}
