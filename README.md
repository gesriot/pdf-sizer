# pdf-sizer

Pack a folder of scanned images into compact PDFs that hit a target file size.
The tool generates multiple variants, each at the same size but with a different
quality/scale trade-off – you pick the one that looks best.

Supports JPEG, JPEG 2000, and MRC (Mixed Raster Content) compression, plus
optional scan preprocessing (despeckle, background flattening, deskew).

---

## Quick start

```bash
# GUI
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
| `mrc` | `JBIG2Decode`/`CCITTFaxDecode` + `DCTDecode` | Scanned text: 1-bit text mask (JBIG2 symbol dictionary), text colour, low-res background, pictures cut out |
| `auto` (GUI default) | all three | Generates JPEG + JP2 + MRC variants in one run |

### MRC layers

| Layer | Content | Resolution | Encoding |
|---|---|---|---|
| Text mask | text, table rules, line art | source, at most 300 dpi | JBIG2 with one symbol dictionary for the whole document, or per-page JBIG2 / CCITT G4 — whichever is smaller |
| Text colour | one fill colour, or a colour field for multi-coloured text | scale / 4 | JPEG, q/2 (at least 25) |
| Background | page with text and pictures removed and filled in from their surroundings | scale / 2 | JPEG, q/2 |
| Pictures | photos and illustrations, cut out as rectangles | scale | JPEG, q |

`scale` is the target resolution relative to the scan and `q` the picture
quality; both are what the search varies. Text is binarized against the local
paper level (a pixel is ink when it is darker than the paper by more than 42 %
of the local paper-to-ink contrast). Tinted boxes stay in the background with
their text in the mask. If even the smallest variant does not fit the target,
the mask drops to half resolution (150 dpi for a 300 dpi scan).

In symbol mode every glyph-sized component of the mask is matched against the
symbols collected so far, across all pages, and the dictionary is stored once
in the PDF (`/JBIG2Globals`). A component reuses a symbol only when the shapes
differ by edge noise: sizes within one pixel, every differing pixel on the
outline of both shapes, no 2×2 block of differences and few differences
overall. Stroke-level differences are never merged, so a 6 cannot turn into an
8; only binarization jitter along the edges is lost (1–5 % of text pixels,
single pixels on glyph outlines), and each symbol is the per-pixel majority of
its instances.
Rules, frames and other large components stay in a generic region. On test
pages the mask shrinks 3–4× compared with per-page JBIG2.

Page sizes come from the scan resolution in the file (JPEG JFIF/EXIF, PNG
pHYs, TIFF, BMP); without it the page is assumed to be A4.

---

## Preprocessing

Optional scan cleanup applied before compression (does not affect source files):

| Flag | Effect |
|---|---|
| `--despeckle` | 3×3 median filter – removes scanner noise |
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

### macOS packaging

To build the macOS app bundle, ZIP archive, and DMG, run the packaging
script on macOS. It requires the build tools listed above, Python 3 with
Pillow, and the macOS utilities `plutil`, `zip`, and `hdiutil`:

```bash
./scripts/package-macos.sh
```

The script generates transparent rounded icons from `assets/icon.png`,
including the GUI runtime PNG and the bundle's `.icns` icon, then runs
`cargo build --release` and creates:

- `dist/pdf-sizer.app`
- `dist/pdf-sizer.app.zip`
- `dist/pdf-sizer.dmg`

The DMG includes an Applications shortcut for installation and is verified
with `hdiutil`. To package an existing release binary without rebuilding:

```bash
./scripts/package-macos.sh --skip-build
```

### Windows packaging

To embed the app icon in the Windows executable and create a portable ZIP,
run the packaging script from PowerShell. It requires Python with Pillow
and `rc.exe` from the Windows SDK:

```powershell
.\scripts\package-windows.ps1
```

The script converts the existing `macos/icon-runtime.png` artwork to a
multi-size Windows icon (16-256 px), embeds it in the release executable,
and creates:

- `dist/windows/pdf-sizer.exe`
- `dist/windows/pdf-sizer.ico` (also usable for shortcuts)
- `dist/pdf-sizer-windows.zip`

The embedded icon is used by Windows Explorer, shortcuts, and the taskbar.
The GUI continues to use the same artwork for its window icon.

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
| MRC | `variant_mrc_q{picture-quality}_s{scale}.pdf` |

## License

MIT – see [LICENSE](LICENSE).
