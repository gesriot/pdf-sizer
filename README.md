# pdf-sizer

Pack a folder of scanned images into compact PDFs that hit a target file size.
The tool generates multiple variants, each at the same size but with a different
quality/scale trade-off – you pick the one that looks best.

Supports JPEG, JPEG 2000, and MRC (Mixed Raster Content) compression, plus
optional scan preprocessing (despeckle, background flattening, deskew).

---

## Quick start

```bash
# GUI (default when launched without arguments)
pdf-sizer

# CLI
pdf-sizer run --input ./scans --target-mb 20

# Legacy shorthand (deprecated)
pdf-sizer 20          # reads images/ next to the binary
```

---

## How it works

For each quality level the tool runs a **binary search** over the scale factor
to find the exact downscale that produces a PDF within `target ± tolerance`.
Only variants that land inside the window are saved.

```
JPEG q=100  scale=77%  →  variant_q100_s077.pdf   (19.9 MB)  ← sharpest text
JPEG q=99   scale=83%  →  variant_q099_s083.pdf   (20.1 MB)
JPEG q=98   scale=89%  →  variant_q098_s089.pdf   (20.0 MB)
JPEG q=97   scale=95%  →  variant_q097_s095.pdf   (19.8 MB)  ← most detail
```

In the GUI, results are sorted by file size (largest first) so the
highest-quality variants appear at the top. A **Recommended** section
highlights four picks automatically: best for text, balanced, maximum
detail, smallest. The CLI prints variants in generation order without
recommendations.

---

## Compression modes

| Mode | Filter in PDF | Best for |
|---|---|---|
| `jpeg` (CLI default) | `DCTDecode` | General purpose; fastest |
| `jp2` | `JPXDecode` | Better at low bitrates; softer artefacts |
| `mrc` | `CCITTFaxDecode`/`JBIG2Decode` + `DCTDecode` | Scanned text: 1-bit mask for text, compressed JPEG background |
| `auto` (GUI default) | all three | Generates JPEG + JP2 + MRC variants in one run |

MRC stores text as a lossless 1-bit mask (JBIG2 or CCITT G4, whichever is
smaller) over a heavily compressed background — similar to what ABBYY
FineReader uses internally.

---

## Preprocessing

Optional scan cleanup applied before compression (does not affect source files):

| Flag | Effect |
|---|---|
| `--despeckle` | 3×3 median filter — removes scanner noise |
| `--flatten[=N]` | Background flattening — evens out paper colour/texture; N = aggressiveness 0–255, default 30 |
| `--deskew` | Auto-detects and corrects page tilt (±10°) |

---

## CLI reference

```
pdf-sizer run --input <DIR> [--output <DIR>] --target-mb <N>
              [--despeckle] [--flatten[=<0-255>]] [--deskew]
              [--codec=jpeg|jp2|mrc|auto]

Options:
  --input          Source folder with images
  --output         Destination folder for PDF variants
                   Default: <input>/pdf-sizer-output/<timestamp>/
  --target-mb      Target file size in megabytes
  --codec          Compression mode (default: jpeg)
  --despeckle      Apply median filter
  --flatten[=N]    Flatten background (N = strength, default 30)
  --deskew         Correct page skew
```

The tolerance window defaults to **10 % of target** (e.g. ±2 MB for a 20 MB
target). Adjust it in the GUI with the `±` field next to the size slider.

---

## GUI

Launch without arguments or with `pdf-sizer gui`. Requires a display.

On macOS, prefer launching via double-click or `open ./pdf-sizer` to get a
proper application lifecycle (running from a terminal that is later closed
will kill the process via SIGHUP).

---

## Build

Requires Rust 1.85+ (edition 2024) and a C compiler (for MozJPEG and
OpenJPEG, both built automatically from source).

```bash
git clone <repo>
cd pdf-sizer
cargo build --release
```

Binary: `target/release/pdf-sizer`

---

## Supported image formats

`jpg`, `jpeg`, `png`, `bmp`, `tiff`, `tif`, `webp`

Images are sorted alphabetically – that order becomes the page order in the PDF.

---

## Output files

| Codec | File name pattern |
|---|---|
| JPEG | `variant_q{quality}_s{scale}.pdf` |
| JPEG 2000 | `variant_jp2_b{bpp×100}_s{scale}.pdf` |
| MRC | `variant_mrc_q{bg-quality}_s{scale}.pdf` |
