# pdf-sizer

A command-line tool that converts a folder of images into PDF files, automatically finding the best quality/scale combinations that hit a target file size.

## The Problem

When embedding images into a PDF, the resulting file size is often unpredictable. JPEG or PNG files that sum to your desired size may produce a much larger PDF when packaged together – because most PDF libraries decode images and re-store them as raw pixels. This tool solves that by embedding JPEG data directly into the PDF stream (DCTDecode filter), making the file size predictable and controllable.

## How It Works

Instead of producing a single PDF, the tool generates **multiple variants** – each with the same target file size but a different quality/scale tradeoff:

- **High quality, low scale** – fewer compression artifacts, sharper edges and text, but lower resolution
- **Low quality, high scale** – more compression artifacts, but higher resolution with more fine detail

For each JPEG quality level (100, 95, 90, ... down to 5), the tool runs a **binary search** over the scale factor to find the exact scale that produces a PDF within ±2 MB of your target. Only successful variants are saved to disk.

```
quality=100, scale=62%  →  variant_q100_s062.pdf  (19.8 MB)   ← sharper
quality= 90, scale=73%  →  variant_q090_s073.pdf  (19.5 MB)
quality= 80, scale=82%  →  variant_q080_s082.pdf  (19.9 MB)
quality= 70, scale=91%  →  variant_q070_s091.pdf  (19.7 MB)   ← more detail
```

You open all variants and pick whichever looks best for your specific images.

## Technical Details

- Images are resized using the **Lanczos3** filter (high-quality downscaling)
- JPEG bytes are embedded in the PDF with the **DCTDecode** filter – the original compressed data is stored as-is, not re-encoded
- PDF size ≈ sum of JPEG sizes + ~300 bytes overhead per page
- Built with [`pdf-writer`](https://crates.io/crates/pdf-writer) and [`image`](https://crates.io/crates/image)

## Installation

### Prerequisites

- [Rust](https://rustup.rs/) (edition 2024, Rust 1.85+)

### Build

```bash
git clone <repo>
cd pdf-converter
cargo build --release
```

The binary will be at `target/release/pdf-sizer` (or `pdf-sizer.exe` on Windows).

## Usage

1. Create an `images/` folder **next to the binary**:
   ```
   target/release/
   ├── pdf-sizer.exe
   └── images/
       ├── page_01.jpg
       ├── page_02.png
       └── ...
   ```

2. Run with your target size in megabytes:
   ```bash
   pdf-sizer 20
   ```

3. The tool prints progress and saves all variants in the current directory:
   ```
   Found images: 5
   Target size: 20 MB (±2 MB)

   Searching variants: finding scale for each quality level...

     quality=100, scale= 62% -> variant_q100_s062.pdf (19.8 MB)
     quality= 95, scale= 68% -> variant_q095_s068.pdf (20.3 MB)
     quality= 90, scale= 73% -> variant_q090_s073.pdf (19.5 MB)
     quality= 85, scale= 77% -> variant_q085_s077.pdf (20.1 MB)
     quality= 80, scale= 82% -> variant_q080_s082.pdf (19.9 MB)
     quality= 65: even at scale=100% size is 18.1 MB – below target, stopping

   --- Summary ---
   Variants created: 5
   ```

4. Open the variants and visually pick the one that looks best.

## Supported Image Formats

`jpg`, `jpeg`, `png`, `bmp`, `tiff`, `tif`, `webp`

Images are sorted alphabetically, which determines the page order in the PDF.

## Output Files

Output files are named `variant_q{quality}_s{scale}.pdf` and saved in the **working directory** (where you run the command from), not next to the binary.
