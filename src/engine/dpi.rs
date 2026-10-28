//! Scan resolution from file metadata (JPEG JFIF/EXIF, PNG pHYs, TIFF, BMP), so pages keep
//! their physical size in the PDF.

use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Standard scanner resolutions an estimate snaps to.
const STANDARD_DPI: [f32; 9] = [72.0, 96.0, 100.0, 150.0, 200.0, 240.0, 300.0, 400.0, 600.0];
/// Long side of A4 in inches: pages without usable metadata are assumed to be A4.
const A4_LONG_IN: f32 = 11.69;

fn u16_at(d: &[u8], p: usize, le: bool) -> Option<u16> {
    let b = d.get(p..p + 2)?;
    Some(if le { u16::from_le_bytes([b[0], b[1]]) } else { u16::from_be_bytes([b[0], b[1]]) })
}

fn u32_at(d: &[u8], p: usize, le: bool) -> Option<u32> {
    let b: [u8; 4] = d.get(p..p + 4)?.try_into().ok()?;
    Some(if le { u32::from_le_bytes(b) } else { u32::from_be_bytes(b) })
}

/// XResolution/ResolutionUnit from a TIFF structure (TIFF files and JPEG EXIF).
fn tiff_dpi(d: &[u8]) -> Option<f32> {
    let le = match d.get(..4)? {
        b"II*\0" => true,
        b"MM\0*" => false,
        _ => return None,
    };
    let ifd = u32_at(d, 4, le)? as usize;
    let n = u16_at(d, ifd, le)? as usize;
    let mut res = None;
    let mut unit = 2;
    for i in 0..n {
        let e = ifd + 2 + i * 12;
        match u16_at(d, e, le)? {
            282 => {
                let off = u32_at(d, e + 8, le)? as usize;
                let (num, den) = (u32_at(d, off, le)?, u32_at(d, off + 4, le)?);
                if den != 0 {
                    res = Some(num as f32 / den as f32);
                }
            }
            296 => unit = u16_at(d, e + 8, le)?,
            _ => {}
        }
    }
    match unit {
        2 => res,
        3 => res.map(|r| r * 2.54),
        _ => None,
    }
}

fn jpeg_dpi(d: &[u8]) -> Option<f32> {
    let mut p = 2;
    while p + 4 <= d.len() && d[p] == 0xFF {
        let marker = d[p + 1];
        let len = u16_at(d, p + 2, false)? as usize;
        let seg = d.get(p + 4..p + 2 + len)?;
        if marker == 0xE0 && seg.starts_with(b"JFIF\0") && seg.len() >= 12 {
            let x = u16_at(seg, 8, false)? as f32;
            match seg[7] {
                1 => return Some(x),
                2 => return Some(x * 2.54),
                _ => {}
            }
        }
        if marker == 0xE1
            && seg.starts_with(b"Exif\0\0")
            && let Some(dpi) = tiff_dpi(&seg[6..])
        {
            return Some(dpi);
        }
        if marker == 0xDA || ((0xC0..=0xCF).contains(&marker) && ![0xC4, 0xC8, 0xCC].contains(&marker)) {
            break;
        }
        p += 2 + len;
    }
    None
}

fn png_dpi(d: &[u8]) -> Option<f32> {
    let mut p = 8;
    while p + 12 <= d.len() {
        let len = u32_at(d, p, false)? as usize;
        let kind = d.get(p + 4..p + 8)?;
        if kind == b"pHYs" && len >= 9 {
            let ppu = u32_at(d, p + 8, false)? as f32;
            return (d[p + 16] == 1).then_some(ppu * 0.0254);
        }
        if kind == b"IDAT" {
            break;
        }
        p += 12 + len;
    }
    None
}

fn bmp_dpi(d: &[u8]) -> Option<f32> {
    let ppm = u32_at(d, 38, true)? as i32;
    (ppm > 0).then_some(ppm as f32 * 0.0254)
}

/// Resolution stored in the file header, if any.
pub(crate) fn read_dpi(path: &Path) -> Option<f32> {
    let mut d = Vec::new();
    File::open(path).ok()?.take(256 * 1024).read_to_end(&mut d).ok()?;
    match d.get(..4)? {
        [0xFF, 0xD8, ..] => jpeg_dpi(&d),
        [0x89, b'P', b'N', b'G'] => png_dpi(&d),
        b"II*\0" | b"MM\0*" => tiff_dpi(&d),
        [b'B', b'M', ..] => bmp_dpi(&d),
        _ => None,
    }
}

/// Resolution for a `w`×`h` page: the file's own value when it is plausible for a scan,
/// otherwise the resolution at which the page would be A4, snapped to a standard value.
/// 72/96 dpi are editor defaults, not scan resolutions, and are ignored.
pub(crate) fn page_dpi(path: &Path, w: u32, h: u32) -> f32 {
    let long = w.max(h) as f32;
    if let Some(dpi) = read_dpi(path) {
        let inches = long / dpi;
        if (100.0..=2400.0).contains(&dpi) && (2.0..=60.0).contains(&inches) {
            return dpi;
        }
    }
    estimate_dpi(long)
}

fn estimate_dpi(long_px: f32) -> f32 {
    let raw = long_px / A4_LONG_IN;
    STANDARD_DPI
        .iter()
        .copied()
        .find(|s| (raw / s - 1.0).abs() <= 0.12)
        .unwrap_or(raw)
        .clamp(50.0, 1200.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jfif_density_is_read() {
        let mut d = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
        d.extend_from_slice(b"JFIF\0");
        d.extend_from_slice(&[1, 1, 1, 0x01, 0x2C, 0x01, 0x2C, 0, 0]);
        d.extend_from_slice(&[0xFF, 0xDA, 0, 2]);
        assert_eq!(jpeg_dpi(&d), Some(300.0));
    }

    #[test]
    fn png_phys_is_read() {
        let mut d = b"\x89PNG\r\n\x1a\n".to_vec();
        d.extend_from_slice(&9u32.to_be_bytes());
        d.extend_from_slice(b"pHYs");
        d.extend_from_slice(&11811u32.to_be_bytes());
        d.extend_from_slice(&11811u32.to_be_bytes());
        d.push(1);
        d.extend_from_slice(&[0; 4]);
        let dpi = png_dpi(&d).unwrap();
        assert!((dpi - 300.0).abs() < 0.1, "{dpi}");
    }

    #[test]
    fn estimate_snaps_to_standard_resolutions() {
        assert_eq!(estimate_dpi(3508.0), 300.0);
        assert_eq!(estimate_dpi(1754.0), 150.0);
        assert_eq!(estimate_dpi(7016.0), 600.0);
    }
}
