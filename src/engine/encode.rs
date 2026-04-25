use std::ffi::c_void;
use std::fs;
use std::path::{Path, PathBuf};

use fax::{Color as FaxColor, VecWriter};
use image::GenericImageView;
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

use super::types::{Cs, Frame, PdfFilter, TextMask};

pub(crate) fn encode_text_mask_g4(mask: &TextMask) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
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

pub(crate) fn mask_to_jbig2_pixels(mask: &TextMask) -> Vec<u8> {
    let mut buf = Vec::with_capacity((mask.width * mask.height) as usize);
    for &p in &mask.pixels {
        buf.push(if p { 1u8 } else { 0u8 });
    }
    buf
}

/// Encode the mask as a standalone JBIG2 page stream (no global dictionary), ready for
/// `JBIG2Decode` in PDF. Uses the lossless variant: substitution errors are not safe for
/// documents that may contain digits or critical glyphs.
pub(crate) fn encode_text_mask_jbig2(
    mask: &TextMask,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let buf = mask_to_jbig2_pixels(mask);
    let result = jbig2enc_rust::encode_single_image_lossless(&buf, mask.width, mask.height, true)
        .map_err(|e| format!("JBIG2 encode failed: {:?}", e))?;
    // The lossless path forces standalone streams (no global dictionary). Asserting
    // surfaces upstream regressions early instead of producing broken PDFs.
    if result.global_data.is_some() {
        return Err("JBIG2: lossless path unexpectedly produced a global dictionary".into());
    }
    Ok(result.page_data)
}

pub(crate) fn estimate_pdf_size(
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

pub(crate) fn collect_images(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let extensions = ["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"];

    let mut paths: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            !path
                .components()
                .any(|component| component.as_os_str() == "pdf-sizer-output")
        })
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

pub(crate) fn is_effectively_grayscale(img: &image::DynamicImage) -> bool {
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
pub(crate) fn jpeg_component_count(data: &[u8]) -> Option<u8> {
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

pub(crate) fn encode_mozjpeg(
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

pub(crate) fn encode_raster_as_jpeg(
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

pub(crate) fn encode_image(
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

pub(crate) fn encode_as_jp2(
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

pub(crate) fn encode_jp2_raw(
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

pub(crate) fn estimate_jp2_size(
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
