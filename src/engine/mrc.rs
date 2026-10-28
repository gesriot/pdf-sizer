//! Layered MRC compression:
//!
//! * text mask — 1 bit at the text resolution: JBIG2 with a symbol dictionary shared by
//!   all pages, or per-page lossless JBIG2 / CCITT G4, whichever is smaller;
//! * text colour — a fill colour, or a colour field at 1/4 of the target resolution
//!   painted through the mask;
//! * background — text and pictures removed and filled in from their surroundings, at 1/2
//!   of the target resolution, compressed hard;
//! * pictures — cut out and stored as separate JPEGs at the target resolution.
//!
//! The probe scale `s` is the target resolution relative to the source.

use std::path::PathBuf;
use std::sync::Arc;

use super::encode::{encode_raster_as_jpeg, encode_text_mask_g4, encode_text_mask_jbig2};
use super::jbig2sym::SymbolCoder;
use super::pdf::build_mrc_pdf;
use super::raster::{area_resize, box_blur, dilate, push_pull};
use super::segment::{PageSegmentation, segment_page};
use super::types::{FgFrame, Foreground, MaskCodec, MaskLayer, MrcFrame, MrcSource, PictureLayer, TextMask};
use crate::progress::{CancellationToken, Cancelled, LogLevel, ProgressEvent, ProgressReporter};

/// The text mask never exceeds this resolution: finer scans gain nothing visible.
const MASK_MAX_DPI: f32 = 300.0;
/// The compact mask level never goes below this.
const MASK_MIN_DPI: f32 = 150.0;
/// Text pixels whose colour is within this RGB distance of the mean count as one colour.
const SOLID_COLOUR_DISTANCE: f32 = 40.0;

fn check_cancel(cancel: &CancellationToken) -> Result<(), Box<dyn std::error::Error>> {
    if cancel.is_cancelled() {
        Err(Box::new(Cancelled))
    } else {
        Ok(())
    }
}

/// JPEG qualities of the background and colour layers for picture quality `q`. Both
/// layers carry no fine detail, so they take much stronger compression than pictures.
fn layer_qualities(q: u8) -> (u8, u8) {
    ((q / 2).max(10), (q / 2).max(25))
}

pub(crate) fn prepare_mrc_sources(
    images: &[(PathBuf, image::DynamicImage)],
    dpis: &[f32],
    reporter: &dyn ProgressReporter,
    cancel: &CancellationToken,
) -> Result<Vec<MrcSource>, Box<dyn std::error::Error>> {
    let mut sources = Vec::with_capacity(images.len());
    let mut coders = [SymbolCoder::default(), SymbolCoder::default()];
    // Index of each page in the symbol coders (pages without text are not added).
    let mut coded_pages = Vec::with_capacity(images.len());
    for ((path, img), &dpi) in images.iter().zip(dpis) {
        check_cancel(cancel)?;
        let (source, coded) = prepare_page_with(img, dpi, Some(&mut coders))?;
        coded_pages.push(coded);
        let mask_info = match source.masks.first() {
            Some(m) => format!("маска {}×{} {:?} {} КБ", m.width, m.height, m.codec, m.data.len() / 1024),
            None => "текста нет".to_string(),
        };
        let fg = match source.foreground {
            Foreground::Solid(_) => "один цвет",
            Foreground::Image(_) => "цветной",
        };
        reporter.report(ProgressEvent::Log {
            level: LogLevel::Info,
            message: format!(
                "  MRC: {} ({:.0} dpi): {mask_info}, текст {fg}, иллюстраций {}",
                path.display(),
                dpi,
                source.pictures.len()
            ),
        });
        sources.push(source);
    }

    // One symbol dictionary for the whole document replaces the per-page masks when the
    // dictionary plus all page streams come out smaller.
    for (level, coder) in coders.iter().enumerate() {
        check_cancel(cancel)?;
        let Some(doc) = coder.encode() else {
            continue;
        };
        let per_page: usize = sources.iter().filter_map(|s| s.masks.get(level)).map(|m| m.data.len()).sum();
        let symbol: usize = doc.globals.len() + doc.pages.iter().map(Vec::len).sum::<usize>();
        let label = if level == 0 { "маска" } else { "компактная маска" };
        if symbol >= per_page {
            reporter.report(ProgressEvent::Log {
                level: LogLevel::Info,
                message: format!(
                    "  MRC: {label}: словарь символов не выигрывает ({} КБ против {} КБ)",
                    symbol / 1024,
                    per_page / 1024
                ),
            });
            continue;
        }
        reporter.report(ProgressEvent::Log {
            level: LogLevel::Info,
            message: format!(
                "  MRC: {label}: JBIG2 со словарём символов ({} символов): {} КБ вместо {} КБ",
                coder.symbol_count(),
                symbol / 1024,
                per_page / 1024
            ),
        });
        let globals = Arc::new(doc.globals);
        for (source, coded) in sources.iter_mut().zip(&coded_pages) {
            if let (Some(page), Some(mask)) = (coded[level], source.masks.get_mut(level)) {
                mask.data = doc.pages[page].clone();
                mask.codec = MaskCodec::Jbig2;
                mask.globals = Some(globals.clone());
            }
        }
    }
    Ok(sources)
}

#[cfg(test)]
pub(crate) fn prepare_page(img: &image::DynamicImage, dpi: f32) -> Result<MrcSource, Box<dyn std::error::Error>> {
    Ok(prepare_page_with(img, dpi, None)?.0)
}

/// Index of a page in the symbol coder of each mask level (`None`: page has no text).
type CodedPage = [Option<usize>; 2];

/// Prepare the layers of one page; text masks are also added to the symbol coders (one
/// per mask level), and the page's index in each coder is returned.
fn prepare_page_with(
    img: &image::DynamicImage,
    dpi: f32,
    mut coders: Option<&mut [SymbolCoder; 2]>,
) -> Result<(MrcSource, CodedPage), Box<dyn std::error::Error>> {
    let seg = segment_page(img, dpi);
    let rgb = img.to_rgb8();
    let (w, h) = (seg.width as usize, seg.height as usize);

    let has_text = seg.ink.iter().any(|&b| b);
    let mut masks = Vec::new();
    let mut coded = [None, None];
    if has_text {
        let full = (MASK_MAX_DPI / dpi).min(1.0);
        let compact = (full / 2.0).max((MASK_MIN_DPI / dpi).min(1.0));
        for (level, ratio) in [full, compact].into_iter().enumerate() {
            let mask = downsample_mask(&seg.ink, w, h, ratio);
            masks.push(encode_mask(&mask)?);
            if let Some(coders) = coders.as_deref_mut() {
                coded[level] = Some(coders[level].add_page(&mask));
            }
        }
    }

    let pictures = seg
        .pictures
        .iter()
        .map(|r| PictureLayer {
            rect: *r,
            image: img.crop_imm(r.x, r.y, r.w, r.h),
        })
        .collect();

    let source = MrcSource {
        width: seg.width,
        height: seg.height,
        dpi,
        masks,
        foreground: foreground(&rgb, &seg),
        background: background(&rgb, &seg, dpi),
        pictures,
    };
    Ok((source, coded))
}

/// Downscale a source-resolution mask: an output pixel is text when at least half of the
/// source pixels it covers are text (2 of 4 at a 2× reduction).
fn downsample_mask(ink: &[bool], w: usize, h: usize, ratio: f32) -> TextMask {
    if ratio >= 1.0 {
        return TextMask {
            width: w as u32,
            height: h as u32,
            pixels: ink.to_vec(),
        };
    }
    let nw = ((w as f32 * ratio).round() as usize).max(1);
    let nh = ((h as f32 * ratio).round() as usize).max(1);
    let plane: Vec<f32> = ink.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
    let pixels = area_resize(&plane, w, h, nw, nh).into_iter().map(|v| v >= 0.5 - 1e-4).collect();
    TextMask {
        width: nw as u32,
        height: nh as u32,
        pixels,
    }
}

fn encode_mask(mask: &TextMask) -> Result<MaskLayer, Box<dyn std::error::Error>> {
    // Encode both and pick the smaller. JBIG2 typically wins on A4-sized text masks
    // by 2–3×; G4 wins on tiny masks where JBIG2's fixed overhead dominates.
    let g4 = encode_text_mask_g4(mask)?;
    let jbig2 = encode_text_mask_jbig2(mask)?;
    let (data, codec) = if jbig2.len() < g4.len() {
        (jbig2, MaskCodec::Jbig2)
    } else {
        (g4, MaskCodec::Ccitt)
    };
    Ok(MaskLayer {
        width: mask.width,
        height: mask.height,
        data,
        codec,
        globals: None,
    })
}

/// Weighted area downscale of an RGB image followed by push-pull filling of the cells that
/// received no weight. A cell trusts its own colour completely once `full_weight` of it is
/// covered; returns the filled colours (3 interleaved channels) and the layer size.
fn weighted_layer(rgb: &image::RgbImage, weight: &[f32], ratio: f32, full_weight: f32) -> (Vec<f32>, usize, usize) {
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let nw = ((w as f32 * ratio).round() as usize).max(1);
    let nh = ((h as f32 * ratio).round() as usize).max(1);
    let wsum = area_resize(weight, w, h, nw, nh);
    let mut values = vec![0f32; nw * nh * 3];
    for k in 0..3 {
        let plane: Vec<f32> = rgb.pixels().zip(weight).map(|(p, &wt)| p[k] as f32 * wt).collect();
        let sum = area_resize(&plane, w, h, nw, nh);
        for i in 0..nw * nh {
            if wsum[i] > 0.0 {
                values[i * 3 + k] = sum[i] / wsum[i];
            }
        }
    }
    let conf: Vec<f32> = wsum.iter().map(|&v| (v / full_weight).min(1.0)).collect();
    push_pull(&mut values, &conf, nw, nh, 3);
    (values, nw, nh)
}

fn to_image(values: &[f32], w: usize, h: usize) -> image::DynamicImage {
    image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(w as u32, h as u32, |x, y| {
        let i = (y as usize * w + x as usize) * 3;
        image::Rgb([0, 1, 2].map(|k| values[i + k].round().clamp(0.0, 255.0) as u8))
    }))
}

fn foreground(rgb: &image::RgbImage, seg: &PageSegmentation) -> Foreground {
    // Solid strokes decide the colour; antialiased edges only fill gaps.
    let has_core = seg.core.iter().any(|&b| b);
    let weight: Vec<f32> = seg
        .ink
        .iter()
        .zip(&seg.core)
        .map(|(&ink, &core)| match (ink, core || !has_core) {
            (true, true) => 1.0,
            (true, false) => 0.002,
            _ => 0.0,
        })
        .collect();

    let mut sum = [0f64; 3];
    let mut n = 0f64;
    for (p, &wt) in rgb.pixels().zip(&weight) {
        if wt == 1.0 {
            for k in 0..3 {
                sum[k] += p[k] as f64;
            }
            n += 1.0;
        }
    }
    if n == 0.0 {
        return Foreground::Solid([0, 0, 0]);
    }
    let mean = sum.map(|v| (v / n) as f32);
    let near = rgb
        .pixels()
        .zip(&weight)
        .filter(|&(p, &wt)| {
            wt == 1.0 && (0..3).map(|k| (p[k] as f32 - mean[k]).powi(2)).sum::<f32>().sqrt() <= SOLID_COLOUR_DISTANCE
        })
        .count() as f64;
    if near >= 0.97 * n {
        return Foreground::Solid(mean.map(|v| v.round() as u8));
    }
    let (values, nw, nh) = weighted_layer(rgb, &weight, 0.25, 0.02);
    Foreground::Image(to_image(&values, nw, nh))
}

fn background(rgb: &image::RgbImage, seg: &PageSegmentation, dpi: f32) -> image::DynamicImage {
    let (w, h) = (seg.width as usize, seg.height as usize);
    // Grow the text so its antialiased fringe is filled too.
    let r = (dpi / 150.0).round().max(1.0) as usize;
    let mut hole = dilate(&seg.ink, w, h, r);
    for p in &seg.pictures {
        for y in p.y as usize..(p.y + p.h) as usize {
            hole[y * w + p.x as usize..y * w + (p.x + p.w) as usize].fill(true);
        }
    }
    let known: Vec<f32> = hole.iter().map(|&b| if b { 0.0 } else { 1.0 }).collect();
    drop(hole);
    let (mut values, nw, nh) = weighted_layer(rgb, &known, 0.5, 0.25);
    // Light smoothing of paper texture and scanner noise.
    for k in 0..3 {
        let plane: Vec<f32> = (0..nw * nh).map(|i| values[i * 3 + k]).collect();
        for (i, v) in box_blur(&plane, nw, nh, 1).into_iter().enumerate() {
            values[i * 3 + k] = v;
        }
    }
    to_image(&values, nw, nh)
}

/// Encode every page at picture quality `q` and target-resolution ratio `s`, using mask
/// level `mask_level` (see [`MrcSource::masks`]).
pub(crate) fn encode_mrc_frames(
    sources: &[MrcSource],
    q: u8,
    s: f64,
    mask_level: usize,
) -> Result<Vec<MrcFrame<'_>>, Box<dyn std::error::Error>> {
    let (q_bg, q_fg) = layer_qualities(q);
    sources
        .iter()
        .map(|src| {
            let mask = src.masks.get(mask_level).or(src.masks.last());
            let foreground = match &src.foreground {
                Foreground::Solid(c) => FgFrame::Solid(*c),
                Foreground::Image(img) => FgFrame::Image(encode_raster_as_jpeg(img, q_fg, s)?),
            };
            let pictures = src
                .pictures
                .iter()
                .map(|p| Ok((p.rect, encode_raster_as_jpeg(&p.image, q, s)?)))
                .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
            Ok(MrcFrame {
                page_size: (src.width as f32 * 72.0 / src.dpi, src.height as f32 * 72.0 / src.dpi),
                source_size: (src.width, src.height),
                background: encode_raster_as_jpeg(&src.background, q_bg, s)?,
                mask,
                foreground,
                pictures,
            })
        })
        .collect()
}

pub(crate) fn estimate_mrc_size(
    sources: &[MrcSource],
    q: u8,
    s: f64,
    mask_level: usize,
) -> Result<u64, Box<dyn std::error::Error>> {
    let frames = encode_mrc_frames(sources, q, s, mask_level)?;
    Ok(build_mrc_pdf(&frames).len() as u64)
}
