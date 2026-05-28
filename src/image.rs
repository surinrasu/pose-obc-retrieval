use std::{fs, path::Path};

use img::DynamicImage;

use crate::RetrievalError;

pub(crate) fn open_dynamic_image(path: impl AsRef<Path>) -> Result<DynamicImage, RetrievalError> {
    let path = path.as_ref();
    if !is_avif_path(path) {
        return Err(RetrievalError::InvalidData(format!(
            "expected AVIF image: {}",
            path.display()
        )));
    }
    load_avif_from_bytes(&fs::read(path)?)
}

pub(crate) fn load_dynamic_image_from_memory(bytes: &[u8]) -> Result<DynamicImage, RetrievalError> {
    if !looks_like_avif(bytes) {
        return Err(RetrievalError::InvalidData(
            "expected AVIF image bytes".to_string(),
        ));
    }
    load_avif_from_bytes(bytes)
}

fn is_avif_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("avif"))
}

fn looks_like_avif(bytes: &[u8]) -> bool {
    bytes.len() >= 12
        && &bytes[4..8] == b"ftyp"
        && (&bytes[8..12] == b"avif" || &bytes[8..12] == b"avis")
}

fn load_avif_from_bytes(bytes: &[u8]) -> Result<DynamicImage, RetrievalError> {
    let config = avif::DecoderConfig::new().prefer_8bit(true).threads(1);
    let pixels = avif::decode_with(bytes, &config, &avif::Unstoppable)
        .map_err(|error| RetrievalError::InvalidData(format!("AVIF decode failed: {error}")))?;
    let (width, height, rgba) = pixel_buffer_to_rgba8(&pixels)?;
    let image = img::RgbaImage::from_raw(width, height, rgba).ok_or_else(|| {
        RetrievalError::InvalidData(format!(
            "decoded AVIF pixel buffer has invalid {width}x{height} layout"
        ))
    })?;
    Ok(DynamicImage::ImageRgba8(image))
}

pub(crate) fn pixel_buffer_to_rgba8(
    pixels: &avif::PixelBuffer,
) -> Result<(u32, u32, Vec<u8>), RetrievalError> {
    if let Some(image) = pixels.try_as_imgref::<rgb::Rgba<u8>>() {
        let mut rgba = Vec::with_capacity(image.width() * image.height() * 4);
        for pixel in image.pixels() {
            rgba.extend_from_slice(&[pixel.r, pixel.g, pixel.b, pixel.a]);
        }
        return Ok((image.width() as u32, image.height() as u32, rgba));
    }

    if let Some(image) = pixels.try_as_imgref::<rgb::Rgb<u8>>() {
        let mut rgba = Vec::with_capacity(image.width() * image.height() * 4);
        for pixel in image.pixels() {
            rgba.extend_from_slice(&[pixel.r, pixel.g, pixel.b, 255]);
        }
        return Ok((image.width() as u32, image.height() as u32, rgba));
    }

    Err(RetrievalError::InvalidData(
        "decoded AVIF was not RGB8 or RGBA8 after 8-bit conversion".to_string(),
    ))
}
