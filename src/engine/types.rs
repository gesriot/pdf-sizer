use super::segment::PicRect;

/// Encoded image frame ready to embed in a PDF page.
pub(crate) type Frame = (Vec<u8>, u32, u32, Cs, PdfFilter);

#[derive(Clone, Copy)]
pub(crate) enum Cs {
    Rgb,
    Gray,
}

#[derive(Clone, Copy)]
pub(crate) enum PdfFilter {
    Dct, // JPEG (DCTDecode)
    Jpx, // JPEG2000 (JPXDecode)
}

/// 1-bit text mask, `true` = text.
pub(crate) struct TextMask {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pixels: Vec<bool>,
}

/// Encoding chosen for a text mask. Picked per page as whichever produces the smaller
/// stream — JBIG2 wins on realistic A4 masks, G4 wins on tiny/sparse ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MaskCodec {
    Ccitt,
    Jbig2,
}

/// Encoded text mask at one resolution.
#[derive(Clone)]
pub(crate) struct MaskLayer {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) data: Vec<u8>,
    pub(crate) codec: MaskCodec,
    /// JBIG2 symbol dictionary shared by all pages (`/JBIG2Globals`), if symbol coded.
    pub(crate) globals: Option<std::sync::Arc<Vec<u8>>>,
}

/// Colour painted through the text mask.
pub(crate) enum Foreground {
    /// Every text pixel has (nearly) the same colour: a fill colour is enough.
    Solid([u8; 3]),
    /// Text colours vary: a low-resolution colour field (1/4 of the source resolution).
    Image(image::DynamicImage),
}

/// Illustration cut out of the page, kept at source resolution.
pub(crate) struct PictureLayer {
    pub(crate) rect: PicRect,
    pub(crate) image: image::DynamicImage,
}

/// Per-page MRC layers prepared once and re-encoded for every quality/scale probe.
pub(crate) struct MrcSource {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) dpi: f32,
    /// Text mask by level: 0 = text resolution (source, at most 300 dpi), 1 = half of it
    /// (used when the target size is too small). Empty: no text.
    pub(crate) masks: Vec<MaskLayer>,
    pub(crate) foreground: Foreground,
    /// Background with text and pictures filled in, at half the source resolution.
    pub(crate) background: image::DynamicImage,
    pub(crate) pictures: Vec<PictureLayer>,
}

pub(crate) enum FgFrame {
    Solid([u8; 3]),
    Image(Frame),
}

pub(crate) struct MrcFrame<'a> {
    /// Page size in points.
    pub(crate) page_size: (f32, f32),
    /// Source image size in pixels (maps picture rectangles onto the page).
    pub(crate) source_size: (u32, u32),
    pub(crate) background: Frame,
    pub(crate) mask: Option<&'a MaskLayer>,
    pub(crate) foreground: FgFrame,
    pub(crate) pictures: Vec<(PicRect, Frame)>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum CodecChoice {
    Jpeg,
    Jp2,
    Mrc,
    Auto,
}

pub(crate) struct PreprocOpts {
    pub(crate) despeckle: bool,
    /// Luma distance from background at which a pixel gets flattened (0 = off).
    pub(crate) flatten_threshold: u8,
    pub(crate) deskew: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct ComponentStats {
    pub(crate) min_x: u32,
    pub(crate) min_y: u32,
    pub(crate) max_x: u32,
    pub(crate) max_y: u32,
    pub(crate) area: u32,
}

impl ComponentStats {
    pub(crate) fn new(x: u32, y: u32) -> Self {
        Self {
            min_x: x,
            min_y: y,
            max_x: x,
            max_y: y,
            area: 0,
        }
    }

    pub(crate) fn add(&mut self, x: u32, y: u32) {
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
        self.area += 1;
    }

    pub(crate) fn width(self) -> u32 {
        self.max_x - self.min_x + 1
    }

    pub(crate) fn height(self) -> u32 {
        self.max_y - self.min_y + 1
    }
}
