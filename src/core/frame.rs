//! A captured frame: RGB, row-major, three bytes per pixel.
//!
//! This replaces the OpenCV `Mat` that used to flow from the capture stage through
//! inference and presentation. Two deliberate choices:
//!
//! * **RGB, not BGR.** The ONNX graphs want RGB, the iced preview wants RGBA, and
//!   every Rust capture library hands back RGB or BGRA. Holding the frame in BGR
//!   only ever made sense because OpenCV's capture backends produce BGR; with
//!   OpenCV gone there is no reason to swap channels twice a frame.
//! * **A plain owned buffer.** `pixels` is a `Vec<u8>` the capture stage refills
//!   in place, so the ~900 KB block is allocated once per run rather than once
//!   per frame. `image::ImageBuffer` would work too, but it makes buffer reuse
//!   awkward and it is a dependency we only need for the image *operations*.

use anyhow::Context;

#[derive(Debug, Clone)]
pub struct Frame {
    width: usize,
    height: usize,
    /// RGB, row-major: `pixels[(y * width + x) * 3]` is the red channel of (x, y).
    pixels: Vec<u8>,
}

impl Frame {
    pub fn new(width: usize, height: usize, pixels: Vec<u8>) -> Self {
        assert!(
            pixels.len() == width * height * 3,
            "frame {width}x{height} needs {} bytes, got {}",
            width * height * 3,
            pixels.len()
        );
        Frame {
            width,
            height,
            pixels,
        }
    }

    /// A blank frame. Only tests build one, hence the `cfg`.
    #[cfg(test)]
    pub fn solid(width: usize, height: usize, value: u8) -> Self {
        Frame {
            width,
            height,
            pixels: vec![value; width * height * 3],
        }
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    pub const fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    pub fn pixels_mut(&mut self) -> &mut [u8] {
        &mut self.pixels
    }

    /// Mirror the frame horizontally, in place.
    ///
    /// A raw webcam is seen from the camera's point of view, so the user's right
    /// hand lands on the *left* of the image; flipping it makes the stage behave
    /// like a mirror, which is how the keyboard reads (low notes left, high notes
    /// right). Reversing each row's three-byte groups is the whole operation: no
    /// scratch buffer, no extra allocation.
    pub fn flip_horizontal(&mut self) {
        let row_len = self.width * 3;
        for row in self.pixels.chunks_mut(row_len) {
            for x in 0..self.width / 2 {
                let mirror = (self.width - 1 - x) * 3;
                for channel in 0..3 {
                    row.swap(x * 3 + channel, mirror + channel);
                }
            }
        }
    }

    /// RGB -> RGBA, for the iced image handle.
    pub fn to_rgba(&self) -> Vec<u8> {
        let mut rgba = Vec::with_capacity(self.pixels.len() / 3 * 4);
        for pixel in self.pixels.as_chunks::<3>().0 {
            rgba.extend_from_slice(&[pixel[0], pixel[1], pixel[2], 255]);
        }
        rgba
    }

    /// Read a PNG/JPEG/whatever the `image` crate can decode as an RGB frame.
    pub fn read(path: &str) -> anyhow::Result<Frame> {
        let image = image::open(path).with_context(|| format!("Failed to read image {path}"))?;
        let rgb = image.to_rgb8();
        Ok(Frame::new(
            rgb.width() as usize,
            rgb.height() as usize,
            rgb.into_raw(),
        ))
    }

    /// Write the frame back out. The format is taken from the extension.
    pub fn write(&self, path: &str) -> anyhow::Result<()> {
        let image =
            image::RgbImage::from_raw(self.width as u32, self.height as u32, self.pixels.clone())
                .ok_or_else(|| anyhow::anyhow!("frame size does not match its pixel buffer"))?;
        image
            .save(path)
            .with_context(|| format!("Failed to write {path}"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn mirroring_reverses_the_columns() {
        // A 2x1 RGB frame: two pixels, so a reversal is unambiguous.
        let mut frame = Frame::new(2, 1, vec![1, 2, 3, 4, 5, 6]);
        frame.flip_horizontal();
        assert_eq!(frame.pixels(), [4, 5, 6, 1, 2, 3]);
    }

    #[test]
    fn flipping_twice_is_the_identity() {
        let mut frame = Frame::new(2, 1, vec![1, 2, 3, 4, 5, 6]);
        frame.flip_horizontal();
        frame.flip_horizontal();
        assert_eq!(frame.pixels(), [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    #[ignore]
    /// What mirroring a 640x480 frame costs. Run with
    /// `cargo test --release -- --include-ignored bench_mirror_flip --nocapture`.
    fn bench_mirror_flip() {
        const N: u64 = 200;
        let mut frame = Frame::new(
            640,
            480,
            (0..640 * 480 * 3).map(|i| (i % 256) as u8).collect(),
        );
        let start = Instant::now();
        for _ in 0..N {
            frame.flip_horizontal();
            std::hint::black_box(frame.pixels());
        }
        println!(
            "Frame::flip_horizontal 640x480: {:.4} ms/frame (in place, no scratch buffer)",
            start.elapsed().as_secs_f64() * 1000.0 / N as f64
        );
    }

    #[test]
    fn a_frame_rejects_a_pixel_buffer_that_is_not_its_size() {
        // The sampler, the ASCII grid and the annotator all index `pixels` as
        // `(y * width + x) * 3`, so a short buffer is an out-of-bounds read that
        // would only surface as a panic far from its cause.
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let outcome = std::panic::catch_unwind(|| Frame::new(2, 2, vec![0; 9]));
        std::panic::set_hook(previous);
        assert!(outcome.is_err(), "a 2x2 frame needs 12 bytes, not 9");
    }
}
