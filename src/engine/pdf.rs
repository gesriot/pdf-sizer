use pdf_writer::{Content, Finish, Name, Pdf, Rect, Ref};

use super::types::{Cs, FgFrame, Frame, MaskCodec, MaskLayer, MrcFrame, PdfFilter};

/// Page size in points of a `w`×`h` px image scanned at `dpi`.
pub(crate) fn page_size_pt(w: u32, h: u32, dpi: f32) -> (f32, f32) {
    (w as f32 * 72.0 / dpi, h as f32 * 72.0 / dpi)
}

fn write_image(pdf: &mut Pdf, id: Ref, frame: &Frame, mask: Option<Ref>) {
    let (data, w, h, cs, filter) = frame;
    let mut obj = pdf.image_xobject(id, data);
    obj.filter(match filter {
        PdfFilter::Dct => pdf_writer::Filter::DctDecode,
        PdfFilter::Jpx => pdf_writer::Filter::JpxDecode,
    });
    obj.width(*w as i32);
    obj.height(*h as i32);
    match cs {
        Cs::Gray => {
            obj.color_space().device_gray();
        }
        Cs::Rgb => {
            obj.color_space().device_rgb();
        }
    }
    obj.bits_per_component(8);
    obj.interpolate(true);
    if let Some(mask) = mask {
        obj.pair(Name(b"Mask"), mask);
    }
    obj.finish();
}

/// Stencil mask with text pixels painted. Both decoders deliver text as 0: CCITT encodes
/// text as black with BlackIs1 = false, and JBIG2Decode inverts JBIG2's 1 = black. The
/// default Decode [0 1] therefore paints text — no Decode array.
fn write_mask(pdf: &mut Pdf, id: Ref, mask: &MaskLayer, globals: Option<Ref>) {
    let mut obj = pdf.image_xobject(id, &mask.data);
    match mask.codec {
        MaskCodec::Ccitt => {
            obj.filter(pdf_writer::Filter::CcittFaxDecode);
            obj.decode_parms()
                .k(-1)
                .columns(mask.width as i32)
                .rows(mask.height as i32)
                .black_is_1(false);
        }
        MaskCodec::Jbig2 => {
            obj.filter(pdf_writer::Filter::Jbig2Decode);
            if let Some(globals) = globals {
                obj.decode_parms().jbig2_globals(globals);
            }
        }
    }
    obj.width(mask.width as i32);
    obj.height(mask.height as i32);
    obj.image_mask(true);
    obj.bits_per_component(1);
    obj.interpolate(true);
    obj.finish();
}

pub(crate) fn build_pdf(images: &[Frame], page_sizes: &[(f32, f32)]) -> Vec<u8> {
    let mut pdf = Pdf::new();

    let catalog_id = Ref::new(1);
    let page_tree_id = Ref::new(2);
    let base_ref = 3;

    let mut page_ids = Vec::new();

    for (i, (frame, &(width_pt, height_pt))) in images.iter().zip(page_sizes).enumerate() {
        let page_id = Ref::new((base_ref + i * 3) as i32);
        let content_id = Ref::new((base_ref + i * 3 + 1) as i32);
        let image_id = Ref::new((base_ref + i * 3 + 2) as i32);

        page_ids.push(page_id);
        write_image(&mut pdf, image_id, frame, None);

        let mut content = Content::new();
        content.save_state();
        content.transform([width_pt, 0.0, 0.0, height_pt, 0.0, 0.0]);
        content.x_object(Name(b"Img0"));
        content.restore_state();
        let content_data = content.finish();

        pdf.stream(content_id, &content_data);

        let mut page = pdf.page(page_id);
        page.media_box(Rect::new(0.0, 0.0, width_pt, height_pt));
        page.parent(page_tree_id);
        page.contents(content_id);
        page.resources().x_objects().pair(Name(b"Img0"), image_id);
        page.finish();
    }

    let mut pages = pdf.pages(page_tree_id);
    pages.count(images.len() as i32);
    pages.kids(page_ids);
    pages.finish();

    pdf.catalog(catalog_id).pages(page_tree_id);

    pdf.finish()
}

/// MRC page: background, then the text colour painted through the mask (a fill colour
/// with the mask as stencil, or the colour field with the mask as its /Mask), then the
/// pictures.
pub(crate) fn build_mrc_pdf(frames: &[MrcFrame]) -> Vec<u8> {
    let mut pdf = Pdf::new();

    let catalog_id = Ref::new(1);
    let page_tree_id = Ref::new(2);
    let mut next = 3;
    let mut alloc = || {
        let r = Ref::new(next);
        next += 1;
        r
    };

    let mut page_ids = Vec::new();
    // Shared JBIG2 symbol dictionaries, written once and referenced by every mask.
    let mut globals_ids: Vec<(*const Vec<u8>, Ref)> = Vec::new();

    for frame in frames {
        let page_id = alloc();
        let content_id = alloc();
        let bg_id = alloc();
        page_ids.push(page_id);

        let (page_w, page_h) = frame.page_size;
        let full_page = [page_w, 0.0, 0.0, page_h, 0.0, 0.0];
        let mut x_objects: Vec<(String, Ref)> = vec![("Bg".into(), bg_id)];
        let mut content = Content::new();

        write_image(&mut pdf, bg_id, &frame.background, None);
        content.save_state();
        content.transform(full_page);
        content.x_object(Name(b"Bg"));
        content.restore_state();

        if let Some(mask) = frame.mask {
            let mask_id = alloc();
            let globals = mask.globals.as_ref().map(|g| {
                let key = std::sync::Arc::as_ptr(g);
                match globals_ids.iter().find(|(k, _)| *k == key) {
                    Some(&(_, id)) => id,
                    None => {
                        let id = alloc();
                        pdf.stream(id, g);
                        globals_ids.push((key, id));
                        id
                    }
                }
            });
            write_mask(&mut pdf, mask_id, mask, globals);
            content.save_state();
            match &frame.foreground {
                FgFrame::Solid([r, g, b]) => {
                    content.set_fill_rgb(*r as f32 / 255.0, *g as f32 / 255.0, *b as f32 / 255.0);
                    content.transform(full_page);
                    content.x_object(Name(b"Mask"));
                    x_objects.push(("Mask".into(), mask_id));
                }
                FgFrame::Image(fg) => {
                    let fg_id = alloc();
                    write_image(&mut pdf, fg_id, fg, Some(mask_id));
                    content.transform(full_page);
                    content.x_object(Name(b"Fg"));
                    x_objects.push(("Fg".into(), fg_id));
                }
            }
            content.restore_state();
        }

        let (src_w, src_h) = frame.source_size;
        let sx = page_w / src_w as f32;
        let sy = page_h / src_h as f32;
        for (i, (rect, image)) in frame.pictures.iter().enumerate() {
            let id = alloc();
            let name = format!("P{i}");
            write_image(&mut pdf, id, image, None);
            content.save_state();
            content.transform([
                rect.w as f32 * sx,
                0.0,
                0.0,
                rect.h as f32 * sy,
                rect.x as f32 * sx,
                page_h - (rect.y + rect.h) as f32 * sy,
            ]);
            content.x_object(Name(name.as_bytes()));
            content.restore_state();
            x_objects.push((name, id));
        }

        pdf.stream(content_id, &content.finish());

        let mut page = pdf.page(page_id);
        page.media_box(Rect::new(0.0, 0.0, page_w, page_h));
        page.parent(page_tree_id);
        page.contents(content_id);
        {
            let mut resources = page.resources();
            let mut dict = resources.x_objects();
            for (name, id) in &x_objects {
                dict.pair(Name(name.as_bytes()), *id);
            }
        }
        page.finish();
    }

    let mut pages = pdf.pages(page_tree_id);
    pages.count(frames.len() as i32);
    pages.kids(page_ids);
    pages.finish();

    pdf.catalog(catalog_id).pages(page_tree_id);

    pdf.finish()
}
