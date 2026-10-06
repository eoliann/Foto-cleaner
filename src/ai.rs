//! Segmentarea persoanei (portret) cu modelul MODNet, rulat local prin RTen.

use image::{GrayImage, Luma, RgbImage, imageops::FilterType};
use rten::{Model, ValueView};

pub const MODNET_MODEL: &[u8] = include_bytes!("../assets/modnet.onnx");

pub fn load_model() -> Result<Model, String> {
    Model::load_static_slice(MODNET_MODEL)
        .map_err(|error| format!("Modelul MODNet nu poate fi încărcat: {error}"))
}

/// Returnează masca persoanei (255 = persoană, 0 = fundal) la dimensiunile sursei.
pub fn infer_foreground_mask(model: &Model, source: &RgbImage) -> Result<GrayImage, String> {
    let (source_width, source_height) = source.dimensions();
    let short_edge = source_width.min(source_height) as f32;
    let long_edge = source_width.max(source_height) as f32;
    let scale = (512.0 / short_edge).min(1024.0 / long_edge);
    let input_width = round_to_multiple_of_32(source_width as f32 * scale);
    let input_height = round_to_multiple_of_32(source_height as f32 * scale);
    let resized = image::imageops::resize(source, input_width, input_height, FilterType::Triangle);

    let area = (input_width * input_height) as usize;
    let mut input_data = vec![0.0_f32; area * 3];
    for (index, pixel) in resized.pixels().enumerate() {
        input_data[index] = pixel[0] as f32 / 127.5 - 1.0;
        input_data[area + index] = pixel[1] as f32 / 127.5 - 1.0;
        input_data[2 * area + index] = pixel[2] as f32 / 127.5 - 1.0;
    }

    let input = ValueView::from_shape(
        [1, 3, input_height as usize, input_width as usize],
        &input_data,
    )
    .map_err(|error| format!("Intrare AI invalidă: {error}"))?;
    let output = model
        .run_one(input.into(), None)
        .map_err(|error| format!("Procesarea AI a eșuat: {error}"))?;
    let ([_, _, mask_height, mask_width], values) = output
        .into_shape_vec::<f32, 4>()
        .map_err(|error| format!("Rezultat AI invalid: {error}"))?;

    let mask = GrayImage::from_fn(mask_width as u32, mask_height as u32, |x, y| {
        let value = values[y as usize * mask_width + x as usize].clamp(0.0, 1.0);
        Luma([(value * 255.0).round() as u8])
    });
    Ok(image::imageops::resize(
        &mask,
        source_width,
        source_height,
        FilterType::Triangle,
    ))
}

fn round_to_multiple_of_32(value: f32) -> u32 {
    ((value / 32.0).round().max(1.0) as u32) * 32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_ai_model_produces_a_source_sized_mask() {
        let model = load_model().expect("embedded MODNet model should load");
        let source = RgbImage::from_pixel(96, 128, image::Rgb([180, 150, 120]));
        let mask = infer_foreground_mask(&model, &source).expect("MODNet inference should succeed");
        assert_eq!(mask.dimensions(), source.dimensions());
    }
}
