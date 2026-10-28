//! JBIG2 symbol coding (ITU-T T.88) with one symbol dictionary shared by all pages of a
//! document, stored in the PDF as `/JBIG2Globals`.
//!
//! Every connected component of a text mask is matched against the symbols collected so
//! far. A component reuses a symbol only when the two shapes differ by boundary noise:
//! sizes within one pixel, every differing pixel lies on the outline of both shapes, no
//! 2×2 block of differences and few differences overall. Stroke-level differences (the
//! 6/8 kind of substitution) are rejected, so glyph identity is always kept; what is lost
//! is binarization jitter along the edges. Each symbol is stored as the per-pixel majority
//! of the instances that share its exact size, which also smooths that jitter.
//!
//! Components too large to be glyphs (rules, frames, drawings) go to a generic region.

use std::collections::HashMap;

use super::raster::label_components;
use super::types::TextMask;

/// Components larger than this in either dimension are line art, not glyphs.
const MAX_SYMBOL_SIDE: usize = 400;

// ------------------------------------------------------------------ arithmetic coder

/// (Qe, NMPS, NLPS, SWITCH) — T.88 Table E.1.
const QE: [(u32, u8, u8, bool); 47] = [
    (0x5601, 1, 1, true),
    (0x3401, 2, 6, false),
    (0x1801, 3, 9, false),
    (0x0AC1, 4, 12, false),
    (0x0521, 5, 29, false),
    (0x0221, 38, 33, false),
    (0x5601, 7, 6, true),
    (0x5401, 8, 14, false),
    (0x4801, 9, 14, false),
    (0x3801, 10, 14, false),
    (0x3001, 11, 17, false),
    (0x2401, 12, 18, false),
    (0x1C01, 13, 20, false),
    (0x1601, 29, 21, false),
    (0x5601, 15, 14, true),
    (0x5401, 16, 14, false),
    (0x5101, 17, 15, false),
    (0x4801, 18, 16, false),
    (0x3801, 19, 17, false),
    (0x3401, 20, 18, false),
    (0x3001, 21, 19, false),
    (0x2801, 22, 19, false),
    (0x2401, 23, 20, false),
    (0x2201, 24, 21, false),
    (0x1C01, 25, 22, false),
    (0x1801, 26, 23, false),
    (0x1601, 27, 24, false),
    (0x1401, 28, 25, false),
    (0x1201, 29, 26, false),
    (0x1101, 30, 27, false),
    (0x0AC1, 31, 28, false),
    (0x09C1, 32, 29, false),
    (0x08A1, 33, 30, false),
    (0x0521, 34, 31, false),
    (0x0441, 35, 32, false),
    (0x02A1, 36, 33, false),
    (0x0221, 37, 34, false),
    (0x0141, 38, 35, false),
    (0x0111, 39, 36, false),
    (0x0085, 40, 37, false),
    (0x0049, 41, 38, false),
    (0x0025, 42, 39, false),
    (0x0015, 43, 40, false),
    (0x0009, 44, 41, false),
    (0x0005, 45, 42, false),
    (0x0001, 45, 43, false),
    (0x5601, 46, 46, false),
];

/// Adaptive context: probability state index in the low 7 bits, MPS in bit 7.
type Ctx = u8;

/// MQ arithmetic encoder (T.88 Annex E.2).
struct ArithEncoder {
    a: u32,
    c: u32,
    ct: u32,
    b: u8,
    started: bool,
    out: Vec<u8>,
}

impl ArithEncoder {
    fn new() -> Self {
        Self {
            a: 0x8000,
            c: 0,
            ct: 12,
            b: 0,
            started: false,
            out: Vec::new(),
        }
    }

    fn emit(&mut self) {
        if self.started {
            self.out.push(self.b);
        }
        self.started = true;
    }

    fn byte_out(&mut self) {
        if self.b != 0xFF && self.c >= 0x800_0000 {
            self.b += 1;
            if self.b == 0xFF {
                self.c &= 0x7FF_FFFF;
            }
        }
        if self.b == 0xFF {
            // Bit stuffing: after 0xFF only seven bits go into the next byte.
            self.emit();
            self.b = (self.c >> 20) as u8;
            self.c &= 0xF_FFFF;
            self.ct = 7;
        } else {
            self.emit();
            self.b = (self.c >> 19) as u8;
            self.c &= 0x7_FFFF;
            self.ct = 8;
        }
    }

    fn encode(&mut self, ctx: &mut Ctx, bit: bool) {
        let i = (*ctx & 0x7F) as usize;
        let mps = *ctx >> 7 != 0;
        let (qe, nmps, nlps, switch) = QE[i];
        self.a -= qe;
        if bit == mps {
            if self.a & 0x8000 != 0 {
                self.c += qe;
                return;
            }
            if self.a < qe {
                self.a = qe;
            } else {
                self.c += qe;
            }
            *ctx = nmps | (*ctx & 0x80);
        } else {
            if self.a < qe {
                self.c += qe;
            } else {
                self.a = qe;
            }
            let mps = if switch { !mps } else { mps };
            *ctx = nlps | ((mps as u8) << 7);
        }
        loop {
            self.a <<= 1;
            self.c <<= 1;
            self.ct -= 1;
            if self.ct == 0 {
                self.byte_out();
            }
            if self.a & 0x8000 != 0 {
                break;
            }
        }
    }

    fn finish(mut self) -> Vec<u8> {
        let temp = self.c + self.a;
        self.c |= 0xFFFF;
        if self.c >= temp {
            self.c -= 0x8000;
        }
        self.c <<= self.ct;
        self.byte_out();
        self.c <<= self.ct;
        self.byte_out();
        self.emit();
        if self.b != 0xFF {
            self.out.push(0xFF);
        }
        self.out.push(0xAC);
        self.out
    }

    /// Integer arithmetic coding procedure (T.88 Annex A.2); `None` codes OOB.
    fn encode_int(&mut self, ctx: &mut [Ctx; 512], value: Option<i32>) {
        let (sign, mag) = match value {
            None => (true, 0),
            Some(v) => (v < 0, v.unsigned_abs()),
        };
        let (prefix, bits, base): (&[bool], u32, u32) = match mag {
            0..=3 => (&[false], 2, 0),
            4..=19 => (&[true, false], 4, 4),
            20..=83 => (&[true, true, false], 6, 20),
            84..=339 => (&[true, true, true, false], 8, 84),
            340..=4435 => (&[true, true, true, true, false], 12, 340),
            _ => (&[true, true, true, true, true], 32, 4436),
        };
        let mut prev = 1usize;
        let mut put = |enc: &mut Self, bit: bool| {
            enc.encode(&mut ctx[prev], bit);
            prev = if prev < 256 {
                (prev << 1) | bit as usize
            } else {
                (((prev << 1) | bit as usize) & 511) | 256
            };
        };
        put(self, sign);
        for &b in prefix {
            put(self, b);
        }
        let rest = mag - base;
        for i in (0..bits).rev() {
            put(self, (rest >> i) & 1 != 0);
        }
    }

    /// Symbol ID coding (T.88 Annex A.3).
    fn encode_iaid(&mut self, ctx: &mut [Ctx], code_len: u32, id: u32) {
        let mut prev = 1usize;
        for i in (0..code_len).rev() {
            let bit = (id >> i) & 1 != 0;
            self.encode(&mut ctx[prev], bit);
            prev = (prev << 1) | bit as usize;
        }
    }
}

// ------------------------------------------------------------------ bitmaps

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Bitmap {
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) px: Vec<u8>,
}

impl Bitmap {
    fn get(&self, x: isize, y: isize) -> u8 {
        if x < 0 || y < 0 || x as usize >= self.w || y as usize >= self.h {
            0
        } else {
            self.px[y as usize * self.w + x as usize]
        }
    }

    fn count(&self) -> usize {
        self.px.iter().filter(|&&p| p != 0).count()
    }

    /// Black pixels with a white 4-neighbour.
    fn perimeter(&self) -> usize {
        let mut n = 0;
        for y in 0..self.h as isize {
            for x in 0..self.w as isize {
                if self.get(x, y) != 0
                    && (self.get(x - 1, y) == 0
                        || self.get(x + 1, y) == 0
                        || self.get(x, y - 1) == 0
                        || self.get(x, y + 1) == 0)
                {
                    n += 1;
                }
            }
        }
        n
    }

    /// Whether any pixel in the 3×3 neighbourhood of (x, y) has value `v`.
    fn near(&self, x: isize, y: isize, v: u8) -> bool {
        (-1..=1).any(|dy| (-1..=1).any(|dx| self.get(x + dx, y + dy) == v))
    }
}

/// Generic region coding, template 0 with the nominal AT pixels, TPGDON off
/// (T.88 6.2.5.3). `gb` holds the 65536 adaptive contexts, shared across calls when the
/// bitmaps belong to one symbol dictionary.
fn encode_generic(enc: &mut ArithEncoder, gb: &mut [Ctx], bmp: &Bitmap) {
    let get = |x: isize, y: isize| bmp.get(x, y) as usize;
    for y in 0..bmp.h as isize {
        for x in 0..bmp.w as isize {
            let ctx = get(x - 1, y)
                | get(x - 2, y) << 1
                | get(x - 3, y) << 2
                | get(x - 4, y) << 3
                | get(x + 2, y - 1) << 4
                | get(x + 1, y - 1) << 5
                | get(x, y - 1) << 6
                | get(x - 1, y - 1) << 7
                | get(x - 2, y - 1) << 8
                | get(x + 3, y - 1) << 9
                | get(x - 3, y - 1) << 10
                | get(x + 1, y - 2) << 11
                | get(x, y - 2) << 12
                | get(x - 1, y - 2) << 13
                | get(x + 2, y - 2) << 14
                | get(x - 2, y - 2) << 15;
            enc.encode(&mut gb[ctx], bmp.px[y as usize * bmp.w + x as usize] != 0);
        }
    }
}

/// Nominal adaptive template pixels for GB template 0: A1..A4 as (x, y) pairs.
const GB_AT: [i8; 8] = [3, -1, -3, -1, 2, -2, -2, -2];

// ------------------------------------------------------------------ segments

const SEG_SYMBOL_DICT: u8 = 0;
const SEG_TEXT_REGION: u8 = 6;
const SEG_GENERIC_REGION: u8 = 38;
const SEG_PAGE_INFO: u8 = 48;

fn segment(number: u32, kind: u8, referred: &[u32], page: u8, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 16);
    out.extend_from_slice(&number.to_be_bytes());
    out.push(kind);
    out.push((referred.len() as u8) << 5);
    for &r in referred {
        match number {
            0..=256 => out.push(r as u8),
            257..=65536 => out.extend_from_slice(&(r as u16).to_be_bytes()),
            _ => out.extend_from_slice(&r.to_be_bytes()),
        }
    }
    out.push(page);
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(data);
    out
}

fn region_info(w: usize, h: usize, x: usize, y: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(17);
    for v in [w, h, x, y] {
        out.extend_from_slice(&(v as u32).to_be_bytes());
    }
    out.push(0); // external combination operator OR
    out
}

fn page_info(w: usize, h: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(19);
    for v in [w as u32, h as u32, 0, 0] {
        out.extend_from_slice(&v.to_be_bytes());
    }
    out.push(0); // default pixel 0, combination operator OR
    out.extend_from_slice(&0u16.to_be_bytes());
    out
}

fn generic_region_data(bmp: &Bitmap, x: usize, y: usize) -> Vec<u8> {
    let mut out = region_info(bmp.w, bmp.h, x, y);
    out.push(0); // arithmetic, template 0, TPGDON off
    out.extend(GB_AT.iter().map(|&v| v as u8));
    let mut enc = ArithEncoder::new();
    let mut gb = vec![0u8; 1 << 16];
    encode_generic(&mut enc, &mut gb, bmp);
    out.extend(enc.finish());
    out
}

/// Symbol dictionary (T.88 6.5) with all symbols exported. Symbols must be sorted by
/// height, then width.
fn symbol_dictionary_data(symbols: &[&Bitmap]) -> Vec<u8> {
    let n = symbols.len() as u32;
    let mut out = vec![0, 0]; // arithmetic, no refinement/aggregation, template 0
    out.extend(GB_AT.iter().map(|&v| v as u8));
    out.extend_from_slice(&n.to_be_bytes()); // SDNUMEXSYMS
    out.extend_from_slice(&n.to_be_bytes()); // SDNUMNEWSYMS
    let mut enc = ArithEncoder::new();
    let mut iadh = [0u8; 512];
    let mut iadw = [0u8; 512];
    let mut iaex = [0u8; 512];
    let mut gb = vec![0u8; 1 << 16];
    let mut height = 0;
    let mut i = 0;
    while i < symbols.len() {
        let h = symbols[i].h;
        enc.encode_int(&mut iadh, Some((h - height) as i32));
        height = h;
        let mut width = 0;
        while i < symbols.len() && symbols[i].h == h {
            let s = symbols[i];
            enc.encode_int(&mut iadw, Some(s.w as i32 - width));
            width = s.w as i32;
            encode_generic(&mut enc, &mut gb, s);
            i += 1;
        }
        enc.encode_int(&mut iadw, None);
    }
    // Export flags as alternating runs, starting with "not exported".
    enc.encode_int(&mut iaex, Some(0));
    enc.encode_int(&mut iaex, Some(n as i32));
    out.extend(enc.finish());
    out
}

/// Placement of dictionary symbol `id` with its top-left corner at (x, y).
#[derive(Clone, Copy, Debug)]
struct Placement {
    id: u32,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

/// log2 of the strip height of text regions.
const LOG_STRIPS: u32 = 3;

/// Text region (T.88 6.4) covering the page; symbols referenced by their bottom-left
/// corner and grouped into strips of 8 rows.
fn text_region_data(placements: &mut [Placement], w: usize, h: usize, num_symbols: u32) -> Vec<u8> {
    let strip = 1usize << LOG_STRIPS;
    let mut out = region_info(w, h, 0, 0);
    // SBHUFF 0, SBREFINE 0, LOGSBSTRIPS, REFCORNER 0 (bottom-left), not transposed, OR,
    // default pixel 0, SBDSOFFSET 0.
    out.extend_from_slice(&((LOG_STRIPS as u16) << 2).to_be_bytes());
    out.extend_from_slice(&(placements.len() as u32).to_be_bytes());

    let bottom = |p: &Placement| p.y + p.h - 1;
    placements.sort_by_key(|p| (bottom(p) / strip, p.x));
    let code_len = u32::BITS - (num_symbols - 1).leading_zeros();
    let mut enc = ArithEncoder::new();
    let (mut iadt, mut iafs, mut iads, mut iait) = ([0u8; 512], [0u8; 512], [0u8; 512], [0u8; 512]);
    let mut iaid = vec![0u8; 1 << (code_len + 1)];
    enc.encode_int(&mut iadt, Some(0));
    let mut strip_t = 0i64;
    let mut first_s = 0i64;
    let mut i = 0;
    while i < placements.len() {
        let t = (bottom(&placements[i]) / strip) as i64;
        enc.encode_int(&mut iadt, Some((t - strip_t) as i32));
        strip_t = t;
        let mut cur_s = 0i64;
        let mut first = true;
        while i < placements.len() && (bottom(&placements[i]) / strip) as i64 == t {
            let p = placements[i];
            if first {
                enc.encode_int(&mut iafs, Some((p.x as i64 - first_s) as i32));
                first_s = p.x as i64;
                first = false;
            } else {
                enc.encode_int(&mut iads, Some((p.x as i64 - cur_s) as i32));
            }
            enc.encode_int(&mut iait, Some((bottom(&p) as i64 - t * strip as i64) as i32));
            enc.encode_iaid(&mut iaid, code_len, p.id);
            cur_s = p.x as i64 + p.w as i64 - 1;
            i += 1;
        }
        enc.encode_int(&mut iads, None);
    }
    out.extend(enc.finish());
    out
}

// ------------------------------------------------------------------ classification

struct Class {
    bitmap: Bitmap,
    count: usize,
    perimeter: usize,
    centroid: (f32, f32),
    /// Per-pixel votes of the instances that matched with identical size and no offset.
    votes: Vec<u16>,
    voters: u16,
}

fn centroid(b: &Bitmap) -> (f32, f32) {
    let (mut sx, mut sy, mut n) = (0f32, 0f32, 0f32);
    for y in 0..b.h {
        for x in 0..b.w {
            if b.px[y * b.w + x] != 0 {
                sx += x as f32;
                sy += y as f32;
                n += 1.0;
            }
        }
    }
    (sx / n.max(1.0), sy / n.max(1.0))
}

/// Whether `b`, placed at offset (ox, oy) in the frame of `a`, differs from `a` only by
/// boundary noise.
fn same_shape(a: &Bitmap, b: &Bitmap, ox: isize, oy: isize, budget: usize) -> bool {
    let x0 = ox.min(0);
    let y0 = oy.min(0);
    let x1 = (a.w as isize).max(ox + b.w as isize);
    let y1 = (a.h as isize).max(oy + b.h as isize);
    let diff = |x: isize, y: isize| a.get(x, y) != b.get(x - ox, y - oy);
    let mut n = 0;
    for y in y0..y1 {
        for x in x0..x1 {
            if !diff(x, y) {
                continue;
            }
            n += 1;
            if n > budget {
                return false;
            }
            // Both shapes must have black and white pixels next to the difference: it sits
            // on the outline of each, so no stroke or hole appears or vanishes.
            let (bx, by) = (x - ox, y - oy);
            if !(a.near(x, y, 0) && a.near(x, y, 1) && b.near(bx, by, 0) && b.near(bx, by, 1)) {
                return false;
            }
            // No 2×2 block of differences: edges may move by one pixel, never more.
            if diff(x + 1, y) && diff(x, y + 1) && diff(x + 1, y + 1) {
                return false;
            }
        }
    }
    true
}

/// Rendered page of one mask: dictionary placements plus line art left as a bitmap.
struct PageSymbols {
    width: usize,
    height: usize,
    /// (class index, x, y) of each symbol's top-left corner.
    instances: Vec<(usize, usize, usize)>,
    /// Large components: encoded generic region segment data.
    residual: Option<Vec<u8>>,
}

/// Collects the symbols of all pages and encodes them with one shared dictionary.
#[derive(Default)]
pub(crate) struct SymbolCoder {
    classes: Vec<Class>,
    by_size: HashMap<(usize, usize), Vec<usize>>,
    pages: Vec<PageSymbols>,
}

/// Encoded document: the shared dictionary (`/JBIG2Globals`) and one stream per page.
pub(crate) struct SymbolDocument {
    pub(crate) globals: Vec<u8>,
    pub(crate) pages: Vec<Vec<u8>>,
}

impl SymbolCoder {
    fn classify(&mut self, b: Bitmap) -> (usize, isize, isize) {
        let count = b.count();
        let tiny = b.w < 4 || b.h < 4;
        let c = centroid(&b);
        for dh in -1isize..=1 {
            for dw in -1isize..=1 {
                if tiny && (dw != 0 || dh != 0) {
                    continue;
                }
                let key = ((b.w as isize + dw) as usize, (b.h as isize + dh) as usize);
                let Some(list) = self.by_size.get(&key) else {
                    continue;
                };
                for &ci in list {
                    let cl = &self.classes[ci];
                    let budget = if tiny { 0 } else { 2 + cl.perimeter / 5 };
                    if cl.count.abs_diff(count) > budget {
                        continue;
                    }
                    let ox = (cl.centroid.0 - c.0).round() as isize;
                    let oy = (cl.centroid.1 - c.1).round() as isize;
                    for (ox, oy) in [(ox, oy), (0, 0)] {
                        if ox.abs() <= 1 && oy.abs() <= 1 && same_shape(&cl.bitmap, &b, ox, oy, budget) {
                            if (ox, oy) == (0, 0) && key == (b.w, b.h) && cl.voters < u16::MAX {
                                let cl = &mut self.classes[ci];
                                for (v, &p) in cl.votes.iter_mut().zip(&b.px) {
                                    *v += p as u16;
                                }
                                cl.voters += 1;
                            }
                            return (ci, ox, oy);
                        }
                    }
                }
            }
        }
        let ci = self.classes.len();
        self.by_size.entry((b.w, b.h)).or_default().push(ci);
        self.classes.push(Class {
            perimeter: b.perimeter(),
            count,
            centroid: c,
            votes: b.px.iter().map(|&p| p as u16).collect(),
            voters: 1,
            bitmap: b,
        });
        (ci, 0, 0)
    }

    /// Add one page; returns its index in [`SymbolDocument::pages`].
    pub(crate) fn add_page(&mut self, mask: &TextMask) -> usize {
        let (w, h) = (mask.width as usize, mask.height as usize);
        let (labels, stats) = label_components(&mask.pixels, w, h);
        let mut instances = Vec::new();
        let mut residual: Option<(usize, usize, usize, usize)> = None;
        let mut big = vec![false; stats.len()];
        for (i, st) in stats.iter().enumerate() {
            let (bw, bh) = (st.width() as usize, st.height() as usize);
            if bw > MAX_SYMBOL_SIDE || bh > MAX_SYMBOL_SIDE {
                big[i] = true;
                let (x0, y0, x1, y1) = (st.min_x as usize, st.min_y as usize, st.max_x as usize, st.max_y as usize);
                residual = Some(match residual {
                    None => (x0, y0, x1, y1),
                    Some((a, b, c, d)) => (a.min(x0), b.min(y0), c.max(x1), d.max(y1)),
                });
                continue;
            }
            let mut px = vec![0u8; bw * bh];
            for y in 0..bh {
                let row = (st.min_y as usize + y) * w + st.min_x as usize;
                for x in 0..bw {
                    if labels[row + x] as usize == i + 1 {
                        px[y * bw + x] = 1;
                    }
                }
            }
            let (ci, ox, oy) = self.classify(Bitmap { w: bw, h: bh, px });
            let x = (st.min_x as isize - ox).max(0) as usize;
            let y = (st.min_y as isize - oy).max(0) as usize;
            instances.push((ci, x, y));
        }
        let residual = residual.map(|(x0, y0, x1, y1)| {
            let (rw, rh) = (x1 - x0 + 1, y1 - y0 + 1);
            let mut px = vec![0u8; rw * rh];
            for y in 0..rh {
                for x in 0..rw {
                    let l = labels[(y0 + y) * w + x0 + x];
                    if l != 0 && big[l as usize - 1] {
                        px[y * rw + x] = 1;
                    }
                }
            }
            generic_region_data(&Bitmap { w: rw, h: rh, px }, x0, y0)
        });
        self.pages.push(PageSymbols {
            width: w,
            height: h,
            instances,
            residual,
        });
        self.pages.len() - 1
    }

    /// Final symbol bitmaps: majority vote of the same-size instances.
    fn symbols(&self) -> Vec<Bitmap> {
        self.classes
            .iter()
            .map(|cl| {
                let mut b = cl.bitmap.clone();
                if cl.voters > 2 {
                    for (p, &v) in b.px.iter_mut().zip(&cl.votes) {
                        let v = v as u32 * 2;
                        let n = cl.voters as u32;
                        if v != n {
                            *p = (v > n) as u8;
                        }
                    }
                }
                b
            })
            .collect()
    }

    pub(crate) fn symbol_count(&self) -> usize {
        self.classes.len()
    }

    /// Encode the document. `None` when there are fewer than two symbols (a dictionary
    /// cannot pay off).
    pub(crate) fn encode(&self) -> Option<SymbolDocument> {
        if self.classes.len() < 2 {
            return None;
        }
        let symbols = self.symbols();
        let mut order: Vec<usize> = (0..symbols.len()).collect();
        order.sort_by_key(|&i| (symbols[i].h, symbols[i].w));
        let mut id_of = vec![0u32; symbols.len()];
        for (id, &ci) in order.iter().enumerate() {
            id_of[ci] = id as u32;
        }
        let sorted: Vec<&Bitmap> = order.iter().map(|&i| &symbols[i]).collect();
        let globals = segment(0, SEG_SYMBOL_DICT, &[], 0, &symbol_dictionary_data(&sorted));

        let pages = self
            .pages
            .iter()
            .map(|page| {
                let mut out = segment(1, SEG_PAGE_INFO, &[], 1, &page_info(page.width, page.height));
                if let Some(res) = &page.residual {
                    out.extend(segment(2, SEG_GENERIC_REGION, &[], 1, res));
                }
                if !page.instances.is_empty() {
                    let mut placements: Vec<Placement> = page
                        .instances
                        .iter()
                        .map(|&(ci, x, y)| Placement {
                            id: id_of[ci],
                            x,
                            y,
                            w: symbols[ci].w,
                            h: symbols[ci].h,
                        })
                        .collect();
                    let data = text_region_data(&mut placements, page.width, page.height, symbols.len() as u32);
                    out.extend(segment(3, SEG_TEXT_REGION, &[0], 1, &data));
                }
                out
            })
            .collect();
        Some(SymbolDocument { globals, pages })
    }

    /// The page as the decoder will reconstruct it (for tests).
    #[cfg(test)]
    pub(crate) fn render(&self, page: usize, mask: &TextMask) -> Vec<bool> {
        let p = &self.pages[page];
        let symbols = self.symbols();
        let (w, h) = (p.width, p.height);
        let mut out = vec![false; w * h];
        for &(ci, x, y) in &p.instances {
            let s = &symbols[ci];
            for sy in 0..s.h {
                for sx in 0..s.w {
                    if s.px[sy * s.w + sx] != 0 && x + sx < w && y + sy < h {
                        out[(y + sy) * w + x + sx] = true;
                    }
                }
            }
        }
        // Residual pixels: the large components, taken from the mask itself.
        let (labels, stats) = label_components(&mask.pixels, w, h);
        for (i, &l) in labels.iter().enumerate() {
            if l != 0 {
                let st = stats[l as usize - 1];
                if st.width() as usize > MAX_SYMBOL_SIDE || st.height() as usize > MAX_SYMBOL_SIDE {
                    out[i] = true;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glyph_page() -> TextMask {
        // Three rows of "letters" with boundary jitter, a rule and a unique blob.
        let (w, h) = (900u32, 700u32);
        let mut px = vec![false; (w * h) as usize];
        let mut put = |x: u32, y: u32| px[(y * w + x) as usize] = true;
        let mut seed = 7u32;
        for row in 0..3 {
            for col in 0..20 {
                let (x0, y0) = (30 + col * 40, 40 + row * 60);
                for y in 0..24 {
                    for x in 0..16 {
                        // An "o": ring with 4-px strokes.
                        let ring = !(4..12).contains(&x) || !(4..20).contains(&y);
                        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                        let jitter = (x == 0 || x == 15) && (seed >> 28) == 0;
                        if ring && !jitter {
                            put(x0 + x, y0 + y);
                        }
                    }
                }
                if col % 3 == 0 {
                    // A "c": the same ring with the right side open — must stay distinct.
                    for y in 0..24 {
                        for x in 0..16 {
                            let ring = !(4..12).contains(&x) || !(4..20).contains(&y);
                            let open = x >= 12 && (8..16).contains(&y);
                            if ring && !open {
                                put(x0 + x, y0 + 300 + y);
                            }
                        }
                    }
                }
            }
        }
        for x in 20..880 {
            put(x, 600);
            put(x, 601);
        }
        TextMask {
            width: w,
            height: h,
            pixels: px,
        }
    }

    /// Manual check against real decoders: `MASK_PNG=mask.png OUT_DIR=dir cargo test --release
    /// -- --ignored symbol_pdf_for_external_decoders`, then render `OUT_DIR/sym.pdf` at 72 dpi
    /// with Ghostscript / Poppler and compare with `OUT_DIR/expected.png`.
    #[test]
    #[ignore]
    fn symbol_pdf_for_external_decoders() {
        use pdf_writer::{Content, Finish, Name, Pdf, Rect, Ref};
        let path = std::env::var("MASK_PNG").expect("MASK_PNG");
        let out = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
        let img = image::open(path).unwrap().to_luma8();
        let mask = TextMask {
            width: img.width(),
            height: img.height(),
            pixels: img.pixels().map(|p| p[0] < 128).collect(),
        };
        let mut coder = SymbolCoder::default();
        coder.add_page(&mask);
        let doc = coder.encode().unwrap();
        let generic = super::super::encode::encode_text_mask_jbig2(&mask).unwrap();
        let g4 = super::super::encode::encode_text_mask_g4(&mask).unwrap();
        eprintln!(
            "symbols {} | globals {} + page {} = {} bytes | generic {} | G4 {}",
            coder.classes.len(),
            doc.globals.len(),
            doc.pages[0].len(),
            doc.globals.len() + doc.pages[0].len(),
            generic.len(),
            g4.len()
        );
        let expected = coder.render(0, &mask);
        let diff = expected.iter().zip(&mask.pixels).filter(|(a, b)| a != b).count();
        eprintln!("pixels changed against the mask: {diff} of {}", mask.pixels.iter().filter(|&&b| b).count());
        image::GrayImage::from_fn(mask.width, mask.height, |x, y| {
            image::Luma([if expected[(y * mask.width + x) as usize] { 0 } else { 255 }])
        })
        .save(out.join("expected.png"))
        .unwrap();

        let (w, h) = (mask.width as f32, mask.height as f32);
        let mut pdf = Pdf::new();
        let (catalog, tree, page, content, image, globals) =
            (Ref::new(1), Ref::new(2), Ref::new(3), Ref::new(4), Ref::new(5), Ref::new(6));
        pdf.stream(globals, &doc.globals);
        let mut obj = pdf.image_xobject(image, &doc.pages[0]);
        obj.filter(pdf_writer::Filter::Jbig2Decode);
        obj.decode_parms().jbig2_globals(globals);
        obj.width(mask.width as i32);
        obj.height(mask.height as i32);
        obj.image_mask(true);
        obj.bits_per_component(1);
        obj.finish();
        let mut c = Content::new();
        c.set_fill_gray(0.0);
        c.transform([w, 0.0, 0.0, h, 0.0, 0.0]);
        c.x_object(Name(b"M"));
        pdf.stream(content, &c.finish());
        let mut p = pdf.page(page);
        p.media_box(Rect::new(0.0, 0.0, w, h));
        p.parent(tree);
        p.contents(content);
        p.resources().x_objects().pair(Name(b"M"), image);
        p.finish();
        pdf.pages(tree).count(1).kids([page]);
        pdf.catalog(catalog).pages(tree);
        std::fs::write(out.join("sym.pdf"), pdf.finish()).expect("write sym.pdf");
    }

    #[test]
    fn jittered_glyphs_share_a_symbol_but_different_shapes_do_not() {
        let mut coder = SymbolCoder::default();
        coder.add_page(&glyph_page());
        // "o" (60 jittered copies) and "c" (21 copies) give two classes; the rule is residual.
        assert_eq!(coder.classes.len(), 2, "classes: {}", coder.classes.len());
        assert!(coder.pages[0].residual.is_some());
    }

    #[test]
    fn differences_inside_strokes_are_rejected() {
        let ring = |hole: bool| {
            let mut px = vec![0u8; 20 * 20];
            for y in 0..20 {
                for x in 0..20 {
                    let inside = (5..15).contains(&x) && (5..15).contains(&y);
                    px[y * 20 + x] = (!inside || !hole) as u8;
                }
            }
            Bitmap { w: 20, h: 20, px }
        };
        assert!(!same_shape(&ring(true), &ring(false), 0, 0, 1000), "filling a hole");
        let a = ring(true);
        let mut b = a.clone();
        b.px[0] = 0; // corner pixel: boundary noise
        assert!(same_shape(&a, &b, 0, 0, 10));
        let mut c = a.clone();
        c.px[2 * 20 + 2] = 0; // pixel inside the stroke
        assert!(!same_shape(&a, &c, 0, 0, 10));
    }

    #[test]
    fn multi_page_document_shares_the_dictionary() {
        let mut coder = SymbolCoder::default();
        let mask = glyph_page();
        coder.add_page(&mask);
        coder.add_page(&mask);
        assert_eq!(coder.classes.len(), 2);
        let doc = coder.encode().unwrap();
        assert_eq!(doc.pages.len(), 2);
        let symbols = reference::decode_dictionary(&doc.globals);
        assert_eq!(symbols.len(), 2);
        for (i, stream) in doc.pages.iter().enumerate() {
            let page = reference::decode_page(stream, &symbols);
            assert_eq!(page, coder.render(i, &mask), "page {i}");
        }
    }

    #[test]
    fn residual_line_art_decodes_with_independent_decoder() {
        // justbig2 decodes generic regions (not symbol dictionaries): check the residual.
        let mut coder = SymbolCoder::default();
        let mask = glyph_page();
        coder.add_page(&mask);
        let doc = coder.encode().unwrap();
        let mut stream = doc.pages[0].clone();
        stream.extend(segment(4, 49, &[], 1, &[]));
        let pages = justbig2::decode_embedded_with_globals(&stream, &doc.globals).unwrap();
        let rule: usize = (20..880).filter(|&x| pages[0].get_pixel(x, 600) != 0).count();
        assert_eq!(rule, 860);
    }

    /// Reference decoder following the T.88 decoding procedures (Annex E.3 arithmetic
    /// decoder, 6.5 symbol dictionary, 6.4 text region) for the subset the encoder emits.
    mod reference {
        use super::super::{Bitmap, Ctx, QE};

        struct Decoder<'a> {
            data: &'a [u8],
            pos: usize,
            c: u32,
            a: u32,
            ct: u32,
        }

        impl<'a> Decoder<'a> {
            fn new(data: &'a [u8]) -> Self {
                let mut d = Self { data, pos: 0, c: 0, a: 0x8000, ct: 0 };
                d.c = (d.byte(0) ^ 0xFF) << 16;
                d.byte_in();
                d.c <<= 7;
                d.ct -= 7;
                d
            }

            fn byte(&self, i: usize) -> u32 {
                self.data.get(i).copied().unwrap_or(0xFF) as u32
            }

            fn byte_in(&mut self) {
                if self.byte(self.pos) == 0xFF {
                    if self.byte(self.pos + 1) > 0x8F {
                        self.ct = 8;
                    } else {
                        self.pos += 1;
                        self.c += 0xFE00 - (self.byte(self.pos) << 9);
                        self.ct = 7;
                    }
                } else {
                    self.pos += 1;
                    self.c += 0xFF00 - (self.byte(self.pos) << 8);
                    self.ct = 8;
                }
            }

            fn decode(&mut self, cx: &mut Ctx) -> bool {
                let i = (*cx & 0x7F) as usize;
                let mps = *cx >> 7 != 0;
                let (qe, nmps, nlps, switch) = QE[i];
                self.a -= qe;
                let lps = |cx: &mut Ctx| {
                    let m = if switch { !mps } else { mps };
                    *cx = nlps | ((m as u8) << 7);
                    !mps
                };
                let d;
                if (self.c >> 16) < self.a {
                    if self.a & 0x8000 != 0 {
                        return mps;
                    }
                    if self.a < qe {
                        d = lps(cx);
                    } else {
                        d = mps;
                        *cx = nmps | ((mps as u8) << 7);
                    }
                } else {
                    self.c -= self.a << 16;
                    if self.a < qe {
                        self.a = qe;
                        d = mps;
                        *cx = nmps | ((mps as u8) << 7);
                    } else {
                        self.a = qe;
                        d = lps(cx);
                    }
                }
                loop {
                    if self.ct == 0 {
                        self.byte_in();
                    }
                    self.a <<= 1;
                    self.c <<= 1;
                    self.ct -= 1;
                    if self.a & 0x8000 != 0 {
                        break;
                    }
                }
                d
            }

            fn int(&mut self, cx: &mut [Ctx; 512]) -> Option<i32> {
                let mut prev = 1usize;
                let mut bit = |d: &mut Self| {
                    let b = d.decode(&mut cx[prev]);
                    prev = if prev < 256 { (prev << 1) | b as usize } else { (((prev << 1) | b as usize) & 511) | 256 };
                    b
                };
                let sign = bit(self);
                let (bits, base) = if !bit(self) {
                    (2, 0)
                } else if !bit(self) {
                    (4, 4)
                } else if !bit(self) {
                    (6, 20)
                } else if !bit(self) {
                    (8, 84)
                } else if !bit(self) {
                    (12, 340)
                } else {
                    (32, 4436)
                };
                let mut v = 0i64;
                for _ in 0..bits {
                    v = (v << 1) | bit(self) as i64;
                }
                let v = (v + base) as i32;
                if sign && v == 0 { None } else { Some(if sign { -v } else { v }) }
            }

            fn iaid(&mut self, cx: &mut [Ctx], len: u32) -> usize {
                let mut prev = 1usize;
                for _ in 0..len {
                    prev = (prev << 1) | self.decode(&mut cx[prev]) as usize;
                }
                prev - (1 << len)
            }

            fn generic(&mut self, gb: &mut [Ctx], w: usize, h: usize) -> Bitmap {
                let mut b = Bitmap { w, h, px: vec![0; w * h] };
                for y in 0..h as isize {
                    for x in 0..w as isize {
                        let get = |x: isize, y: isize| b.get(x, y) as usize;
                        let ctx = get(x - 1, y)
                            | get(x - 2, y) << 1
                            | get(x - 3, y) << 2
                            | get(x - 4, y) << 3
                            | get(x + 2, y - 1) << 4
                            | get(x + 1, y - 1) << 5
                            | get(x, y - 1) << 6
                            | get(x - 1, y - 1) << 7
                            | get(x - 2, y - 1) << 8
                            | get(x + 3, y - 1) << 9
                            | get(x - 3, y - 1) << 10
                            | get(x + 1, y - 2) << 11
                            | get(x, y - 2) << 12
                            | get(x - 1, y - 2) << 13
                            | get(x + 2, y - 2) << 14
                            | get(x - 2, y - 2) << 15;
                        b.px[y as usize * w + x as usize] = self.decode(&mut gb[ctx]) as u8;
                    }
                }
                b
            }
        }

        fn u32_at(d: &[u8], p: usize) -> u32 {
            u32::from_be_bytes(d[p..p + 4].try_into().unwrap())
        }

        /// (type, segment data) of every segment in a stream.
        fn segments(mut d: &[u8]) -> Vec<(u8, &[u8])> {
            let mut out = Vec::new();
            while !d.is_empty() {
                let kind = d[4] & 0x3F;
                let referred = (d[5] >> 5) as usize;
                let len_at = 6 + referred + 1;
                let len = u32_at(d, len_at) as usize;
                out.push((kind, &d[len_at + 4..len_at + 4 + len]));
                d = &d[len_at + 4 + len..];
            }
            out
        }

        pub(super) fn decode_dictionary(globals: &[u8]) -> Vec<Bitmap> {
            let segs = segments(globals);
            let (kind, data) = segs[0];
            assert_eq!(kind, 0);
            let n = u32_at(data, 14) as usize;
            let mut dec = Decoder::new(&data[18..]);
            let (mut iadh, mut iadw, mut iaex) = ([0; 512], [0; 512], [0; 512]);
            let mut gb = vec![0; 1 << 16];
            let mut out = Vec::new();
            let mut height = 0i32;
            while out.len() < n {
                height += dec.int(&mut iadh).unwrap();
                let mut width = 0i32;
                while let Some(dw) = dec.int(&mut iadw) {
                    width += dw;
                    out.push(dec.generic(&mut gb, width as usize, height as usize));
                }
            }
            assert_eq!(dec.int(&mut iaex), Some(0));
            assert_eq!(dec.int(&mut iaex), Some(n as i32));
            out
        }

        pub(super) fn decode_page(stream: &[u8], symbols: &[Bitmap]) -> Vec<bool> {
            let segs = segments(stream);
            let (w, h) = (u32_at(segs[0].1, 0) as usize, u32_at(segs[0].1, 4) as usize);
            let mut page = vec![false; w * h];
            let mut draw = |b: &Bitmap, x0: i64, y0: i64| {
                for y in 0..b.h {
                    for x in 0..b.w {
                        let (px, py) = (x0 + x as i64, y0 + y as i64);
                        if b.px[y * b.w + x] != 0 && px >= 0 && py >= 0 && (px as usize) < w && (py as usize) < h {
                            page[py as usize * w + px as usize] = true;
                        }
                    }
                }
            };
            for &(kind, data) in &segs[1..] {
                let (rx, ry) = (u32_at(data, 8) as i64, u32_at(data, 12) as i64);
                match kind {
                    38 => {
                        let (rw, rh) = (u32_at(data, 0) as usize, u32_at(data, 4) as usize);
                        let mut gb = vec![0; 1 << 16];
                        let b = Decoder::new(&data[26..]).generic(&mut gb, rw, rh);
                        draw(&b, rx, ry);
                    }
                    6 => {
                        let flags = u16::from_be_bytes([data[17], data[18]]);
                        let strips = 1i64 << ((flags >> 2) & 3);
                        assert_eq!((flags >> 4) & 3, 0, "bottom-left reference corner");
                        let count = u32_at(data, 19) as usize;
                        let len = u32::BITS - (symbols.len() as u32 - 1).leading_zeros();
                        let mut dec = Decoder::new(&data[23..]);
                        let (mut iadt, mut iafs, mut iads, mut iait) = ([0; 512], [0; 512], [0; 512], [0; 512]);
                        let mut iaid = vec![0; 1 << (len + 1)];
                        let mut strip_t = -(dec.int(&mut iadt).unwrap() as i64) * strips;
                        let mut first_s = 0i64;
                        let mut n = 0;
                        while n < count {
                            strip_t += dec.int(&mut iadt).unwrap() as i64 * strips;
                            first_s += dec.int(&mut iafs).unwrap() as i64;
                            let mut cur_s = first_s;
                            loop {
                                let cur_t = if strips > 1 { dec.int(&mut iait).unwrap() as i64 } else { 0 };
                                let t = strip_t + cur_t;
                                let b = &symbols[dec.iaid(&mut iaid, len)];
                                draw(b, rx + cur_s, ry + t - b.h as i64 + 1);
                                cur_s += b.w as i64 - 1;
                                n += 1;
                                match dec.int(&mut iads) {
                                    Some(ds) => cur_s += ds as i64,
                                    None => break,
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            page
        }
    }
}
