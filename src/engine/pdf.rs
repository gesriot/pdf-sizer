use pdf_writer::{Content, Finish, Name, Pdf, Rect, Ref};

use super::types::{Cs, Frame, MaskCodec, MrcFrame, PdfFilter};

pub(crate) fn build_pdf(images: &[Frame]) -> Vec<u8> {
    let mut pdf = Pdf::new();

    let catalog_id = Ref::new(1);
    let page_tree_id = Ref::new(2);
    let base_ref = 3;

    let mut page_ids = Vec::new();

    for (i, (img_data, w, h, cs, filter)) in images.iter().enumerate() {
        let page_id = Ref::new((base_ref + i * 3) as i32);
        let content_id = Ref::new((base_ref + i * 3 + 1) as i32);
        let image_id = Ref::new((base_ref + i * 3 + 2) as i32);

        page_ids.push(page_id);

        let width_pt = *w as f32 * 72.0 / 150.0;
        let height_pt = *h as f32 * 72.0 / 150.0;

        let mut image_obj = pdf.image_xobject(image_id, img_data);
        image_obj.filter(match filter {
            PdfFilter::Dct => pdf_writer::Filter::DctDecode,
            PdfFilter::Jpx => pdf_writer::Filter::JpxDecode,
        });
        image_obj.width(*w as i32);
        image_obj.height(*h as i32);
        match cs {
            Cs::Gray => {
                image_obj.color_space().device_gray();
            }
            Cs::Rgb => {
                image_obj.color_space().device_rgb();
            }
        }
        image_obj.bits_per_component(8);
        image_obj.finish();

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

pub(crate) fn build_mrc_pdf(frames: &[MrcFrame]) -> Vec<u8> {
    let mut pdf = Pdf::new();

    let catalog_id = Ref::new(1);
    let page_tree_id = Ref::new(2);
    let base_ref = 3;

    let mut page_ids = Vec::new();

    for (i, frame) in frames.iter().enumerate() {
        let page_id = Ref::new((base_ref + i * 4) as i32);
        let content_id = Ref::new((base_ref + i * 4 + 1) as i32);
        let bg_id = Ref::new((base_ref + i * 4 + 2) as i32);
        let mask_id = Ref::new((base_ref + i * 4 + 3) as i32);

        page_ids.push(page_id);

        let page_width_pt = frame.mask_w as f32 * 72.0 / 150.0;
        let page_height_pt = frame.mask_h as f32 * 72.0 / 150.0;

        let (bg_data, bg_w, bg_h, bg_cs, bg_filter) = &frame.background;
        let mut bg_obj = pdf.image_xobject(bg_id, bg_data);
        bg_obj.filter(match bg_filter {
            PdfFilter::Dct => pdf_writer::Filter::DctDecode,
            PdfFilter::Jpx => pdf_writer::Filter::JpxDecode,
        });
        bg_obj.width(*bg_w as i32);
        bg_obj.height(*bg_h as i32);
        match bg_cs {
            Cs::Gray => {
                bg_obj.color_space().device_gray();
            }
            Cs::Rgb => {
                bg_obj.color_space().device_rgb();
            }
        }
        bg_obj.bits_per_component(8);
        bg_obj.finish();

        let mut mask_obj = pdf.image_xobject(mask_id, &frame.mask_data);
        match frame.mask_codec {
            MaskCodec::Ccitt => {
                mask_obj.filter(pdf_writer::Filter::CcittFaxDecode);
                let mut parms = mask_obj.decode_parms();
                parms
                    .k(-1)
                    .columns(frame.mask_w as i32)
                    .rows(frame.mask_h as i32)
                    .black_is_1(false);
            }
            MaskCodec::Jbig2 => {
                // Standalone JBIG2 page stream — no global dictionary, no DecodeParms.
                mask_obj.filter(pdf_writer::Filter::Jbig2Decode);
            }
        }
        mask_obj.width(frame.mask_w as i32);
        mask_obj.height(frame.mask_h as i32);
        mask_obj.image_mask(true);
        mask_obj.bits_per_component(1);
        // Both decoders place text on the "filled" side of the mask in our setup; the
        // [1.0, 0.0] decode array maps that to PDF's "0 = paint, 1 = transparent".
        mask_obj.decode([1.0, 0.0]);
        mask_obj.finish();

        let mut content = Content::new();
        content.save_state();
        content.transform([page_width_pt, 0.0, 0.0, page_height_pt, 0.0, 0.0]);
        content.x_object(Name(b"Bg"));
        content.restore_state();
        content.save_state();
        content.set_fill_gray(0.0);
        content.transform([page_width_pt, 0.0, 0.0, page_height_pt, 0.0, 0.0]);
        content.x_object(Name(b"Mask"));
        content.restore_state();
        let content_data = content.finish();

        pdf.stream(content_id, &content_data);

        let mut page = pdf.page(page_id);
        page.media_box(Rect::new(0.0, 0.0, page_width_pt, page_height_pt));
        page.parent(page_tree_id);
        page.contents(content_id);
        {
            let mut resources = page.resources();
            let mut x_objects = resources.x_objects();
            x_objects.pair(Name(b"Bg"), bg_id);
            x_objects.pair(Name(b"Mask"), mask_id);
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
