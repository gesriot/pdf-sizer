use std::path::PathBuf;

use super::encode::{encode_raster_as_jpeg, encode_text_mask_g4, encode_text_mask_jbig2};
use super::pdf::build_mrc_pdf;
use super::segment::{fill_masked_background, is_mrc_suitable, segment_text_mask};
use super::types::{MaskCodec, MrcFrame, MrcSource};
use crate::progress::{CancellationToken, Cancelled, LogLevel, ProgressEvent, ProgressReporter};

fn check_cancel(cancel: &CancellationToken) -> Result<(), Box<dyn std::error::Error>> {
    if cancel.is_cancelled() {
        Err(Box::new(Cancelled))
    } else {
        Ok(())
    }
}

pub(crate) fn prepare_mrc_sources(
    images: &[(PathBuf, image::DynamicImage)],
    reporter: &dyn ProgressReporter,
    cancel: &CancellationToken,
) -> Result<Option<Vec<MrcSource>>, Box<dyn std::error::Error>> {
    let mut sources = Vec::with_capacity(images.len());
    for (path, img) in images {
        check_cancel(cancel)?;
        let mask = segment_text_mask(img);
        if !is_mrc_suitable(&mask) {
            reporter.report(ProgressEvent::Log {
                level: LogLevel::Info,
                message: format!(
                    "  MRC: {} пропущен по покрытию маски {:.2}%",
                    path.display(),
                    mask.coverage * 100.0
                ),
            });
            return Ok(None);
        }
        let background = fill_masked_background(img, &mask);
        // Encode both and pick the smaller. JBIG2 typically wins on A4-sized text masks
        // by 2–3×; G4 wins on tiny masks where JBIG2's fixed overhead dominates.
        let g4 = encode_text_mask_g4(&mask)?;
        let jbig2 = encode_text_mask_jbig2(&mask)?;
        let (mask_data, mask_codec) = if jbig2.len() < g4.len() {
            (jbig2, MaskCodec::Jbig2)
        } else {
            (g4, MaskCodec::Ccitt)
        };
        sources.push(MrcSource {
            background,
            mask,
            mask_data,
            mask_codec,
        });
    }
    Ok(Some(sources))
}

pub(crate) fn encode_mrc_frames(
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
                mask_data: source.mask_data.clone(),
                mask_codec: source.mask_codec,
                mask_w: source.mask.width,
                mask_h: source.mask.height,
            })
        })
        .collect()
}

pub(crate) fn estimate_mrc_size(
    sources: &[MrcSource],
    bg_quality: u8,
    bg_scale: f64,
) -> Result<u64, Box<dyn std::error::Error>> {
    let frames = encode_mrc_frames(sources, bg_quality, bg_scale)?;
    Ok(build_mrc_pdf(&frames).len() as u64)
}
