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

pub(crate) struct TextMask {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pixels: Vec<bool>,
    pub(crate) coverage: f32,
    /// Share of structural (line-like) components among all non-trivial ones.
    /// High values flag tables/diagrams that MRC would misrepresent as a 1-bit mask.
    pub(crate) line_ratio: f32,
}

/// Encoding chosen for a text mask. Picked per page as whichever produces the smaller
/// stream — JBIG2 wins on realistic A4 masks, G4 wins on tiny/sparse ones.
#[derive(Clone, Copy)]
pub(crate) enum MaskCodec {
    Ccitt,
    Jbig2,
}

pub(crate) struct MrcSource {
    pub(crate) background: image::DynamicImage,
    pub(crate) mask: TextMask,
    pub(crate) mask_data: Vec<u8>,
    pub(crate) mask_codec: MaskCodec,
}

pub(crate) struct MrcFrame {
    pub(crate) background: Frame,
    pub(crate) mask_data: Vec<u8>,
    pub(crate) mask_codec: MaskCodec,
    pub(crate) mask_w: u32,
    pub(crate) mask_h: u32,
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
