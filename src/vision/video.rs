//! A file-backed frame source that decodes MP4 in pure Rust.
//!
//! A recorded clip is the only way to make a run reproducible: the same frames,
//! the same detector output, the same note sequence, every time. That is what
//! makes a golden trace test possible, and it is why this sits next to the webcam
//! rather than replacing it.
//!
//! Two crates do the work, both dependency-free and `#![forbid(unsafe_code)]`:
//!
//! * `mp4io` parses the container and hands out the H.264 samples.
//! * `rusty_h264-decoder` decodes them. Its YUV planes are bit-exact against
//!   ffmpeg on the fixture (verified in the Phase 0 spike), and it runs at
//!   ~0.5 ms per 640x480 frame - comfortably inside the 33 ms budget.
//!
//! Frames are paced by a deadline clock at the clip's own frame rate, so the
//! telemetry a fixture produces is comparable with a live camera instead of a
//! "decode as fast as the CPU can" burst.

use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use mp4io::{Codec, Mp4, SampleEntry};
use rusty_h264_decoder::Decoder;

use crate::core::frame::Frame;

/// Frames of a clip with no usable rate metadata are paced like a camera.
const FALLBACK_FPS: f64 = 30.0;

/// The deadline interval a clip should be read at. Kept separate from the source
/// so the pacing rule itself can be tested.
fn cadence(reported_fps: f64) -> Duration {
    let fps = if reported_fps.is_finite() && reported_fps > 0.0 {
        reported_fps
    } else {
        FALLBACK_FPS
    };
    Duration::from_secs_f64(1.0 / fps)
}

/// H.264 pictures arrive in decode order; the pipeline needs display order.
///
/// Inside a GOP every picture with a picture-order-count below the newest one is
/// settled, because a later picture in the same GOP always has a higher POC. So
/// a handful of frames is enough to hold the window - three on the fixture - and
/// an IDR flushes whatever is left before the new GOP starts.
struct Reorder {
    pending: Vec<(i32, Frame)>,
}

impl Reorder {
    fn new() -> Self {
        Reorder {
            pending: Vec::new(),
        }
    }

    /// Drain everything held, oldest picture first.
    fn flush(&mut self, out: &mut Vec<Frame>) {
        self.pending.sort_by_key(|(poc, _)| *poc);
        out.extend(self.pending.drain(..).map(|(_, frame)| frame));
    }

    /// Record one decoded picture and return the frames that are now safe to
    /// emit, in display order.
    fn push(&mut self, poc: i32, frame: Frame, out: &mut Vec<Frame>) {
        let mut settled = Vec::new();
        let mut held = Vec::new();
        for (p, f) in self.pending.drain(..) {
            if p < poc {
                settled.push((p, f));
            } else {
                held.push((p, f));
            }
        }
        settled.sort_by_key(|(p, _)| *p);
        out.extend(settled.into_iter().map(|(_, f)| f));
        held.push((poc, frame));
        self.pending = held;
    }
}

/// `yuv420p` -> RGB, Rec. 601 limited range.
///
/// The chroma planes are half resolution and each sample is shared by the 2x2
/// block of luma pixels that surrounds it, so one lookup per pixel suffices.
/// The coefficients are the usual 10-bit fixed point (1.164, 1.596, 0.392,
/// 0.813, 2.001 scaled by 1024); they land within one LSB of ffmpeg's output,
/// which is far below the noise the detector already tolerates.
fn yuv420_to_rgb(y: &[u8], cb: &[u8], cr: &[u8], width: usize, height: usize, out: &mut [u8]) {
    let chroma_width = width / 2;
    for row in 0..height {
        let chroma_row = (row / 2) * chroma_width;
        let luma_row = row * width;
        for col in 0..width {
            let ci = col / 2;
            let yr = y[luma_row + col] as i32 - 16;
            let u = cb[chroma_row + ci] as i32 - 128;
            let v = cr[chroma_row + ci] as i32 - 128;

            let r = (1192 * yr + 1634 * v + 512) >> 10;
            let g = (1192 * yr - 400 * u - 813 * v + 512) >> 10;
            let b = (1192 * yr + 2065 * u + 512) >> 10;

            let pixel = (luma_row + col) * 3;
            out[pixel] = r.clamp(0, 255) as u8;
            out[pixel + 1] = g.clamp(0, 255) as u8;
            out[pixel + 2] = b.clamp(0, 255) as u8;
        }
    }
}

/// Turn one length-prefixed MP4 sample into Annex-B (4-byte start codes), which
/// is what the decoder reads.
fn to_annex_b(payload: &[u8], out: &mut Vec<u8>) {
    let mut pos = 0;
    while pos + 4 <= payload.len() {
        let len = u32::from_be_bytes([
            payload[pos],
            payload[pos + 1],
            payload[pos + 2],
            payload[pos + 3],
        ]) as usize;
        pos += 4;
        if pos + len > payload.len() {
            break;
        }
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&payload[pos..pos + len]);
        pos += len;
    }
}

/// True when the access unit opens a new GOP. NAL type 5 is an I slice with IDR.
fn is_intra(annex_b: &[u8]) -> bool {
    let mut i = 0;
    while i + 3 < annex_b.len() {
        if annex_b[i] == 0 && annex_b[i + 1] == 0 && annex_b[i + 2] == 1 {
            if (annex_b[i + 3] & 0x1f) == 5 {
                return true;
            }
            i += 4;
        } else {
            i += 1;
        }
    }
    false
}

pub struct VideoSource {
    /// Annex-B access units, in decode order. The whole clip is held in memory:
    /// fixtures are megabytes, and it keeps the decoder borrow-friendly.
    units: Vec<Vec<u8>>,
    /// The parameter sets that prefix the first picture.
    header: Vec<u8>,
    /// The decoder is ~5.4 KB of macroblock state, so it is kept on the heap:
    /// `VideoSource` is a variant of `FrameSource`, and an inline decoder would
    /// make that enum 5.4 KB wide for a branch that is usually a camera.
    decoder: Box<Decoder>,
    reorder: Reorder,
    /// Frames decoded but not yet handed to the pipeline.
    ready: Vec<Frame>,
    next_unit: usize,
    mirror: bool,
    cadence: Duration,
    /// Deadline of the next frame. One clock per source keeps the pacing honest.
    next: Instant,
    frames_read: u64,
}

impl VideoSource {
    pub fn new(path: &str, mirror: bool) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("Could not read {path}"))?;
        let mp4 = Mp4::parse(&bytes).with_context(|| format!("Could not parse {path}"))?;

        let track = mp4
            .tracks()
            .iter()
            .find(|t| t.codec() == Some(Codec::H264))
            .ok_or_else(|| anyhow!("{path} has no H.264 video track"))?;

        // The parameter sets live in the sample description, not in the picture
        // stream, so they have to be re-emitted in-band before the first frame.
        let mut header = Vec::new();
        if let Some(SampleEntry::Video(v)) = track.sample_entry() {
            if let mp4io::CodecConfig::Avc(avc) = &v.config {
                for nalu in avc.sps.iter().chain(avc.pps.iter()) {
                    header.extend_from_slice(&[0, 0, 0, 1]);
                    header.extend_from_slice(nalu);
                }
            }
        }

        let mut units = Vec::with_capacity(track.sample_count());
        for sample in track.samples() {
            let data = sample
                .with_context(|| format!("Could not read a sample from {path}"))?
                .data;
            let mut unit = Vec::with_capacity(data.len() + 4);
            to_annex_b(data, &mut unit);
            units.push(unit);
        }

        if units.is_empty() {
            return Err(anyhow!("{path} contains no frames"));
        }

        Ok(Self {
            units,
            header,
            decoder: Box::new(Decoder::new()),
            reorder: Reorder::new(),
            ready: Vec::new(),
            next_unit: 0,
            mirror,
            cadence: cadence(track.frame_rate()),
            next: Instant::now(),
            frames_read: 0,
        })
    }

    pub fn fps(&self) -> f64 {
        1.0 / self.cadence.as_secs_f64()
    }

    /// Decode access units until at least one display-ordered frame is available.
    fn fill(&mut self) -> Result<()> {
        while self.ready.is_empty() && self.next_unit < self.units.len() {
            let unit = &self.units[self.next_unit];
            self.next_unit += 1;

            if is_intra(unit) {
                self.reorder.flush(&mut self.ready);
            }

            let mut annex_b = Vec::with_capacity(unit.len() + self.header.len());
            if self.next_unit == 1 {
                annex_b.extend_from_slice(&self.header);
            }
            annex_b.extend_from_slice(unit);

            if let Some(pic) = self
                .decoder
                .decode(&annex_b)
                .with_context(|| "H.264 decode failed")?
            {
                let (width, height) = (pic.width, pic.height);
                let mut frame = Frame::new(width, height, vec![0u8; width * height * 3]);
                yuv420_to_rgb(&pic.y, &pic.u, &pic.v, width, height, frame.pixels_mut());
                if self.mirror {
                    frame.flip_horizontal();
                }
                self.reorder
                    .push(self.decoder.last_poc(), frame, &mut self.ready);
            }
        }

        // The stream is exhausted; whatever is still buffered is settled.
        if self.ready.is_empty() && self.next_unit >= self.units.len() {
            self.reorder.flush(&mut self.ready);
        }

        Ok(())
    }

    /// Sleep until the clip's frame deadline. Kept separate from `read` so the
    /// capture stage measures decoding, not pacing.
    pub fn wait_for_next(&mut self) {
        let now = Instant::now();
        if now < self.next {
            std::thread::sleep(self.next - now);
        }
        self.next += self.cadence;
    }

    /// The next frame, or `Err` when the clip is exhausted - which is how the
    /// capture thread knows the run is over.
    pub fn read(&mut self) -> Result<Frame> {
        self.fill()?;
        if self.ready.is_empty() {
            return Err(anyhow!("End of clip after {} frames", self.frames_read));
        }
        let frame = self.ready.remove(0);
        self.frames_read += 1;
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clip_with_a_rate_is_read_at_that_rate() {
        assert_eq!(cadence(60.0), Duration::from_secs_f64(1.0 / 60.0));
    }

    #[test]
    fn a_clip_without_a_rate_is_paced_like_a_camera() {
        for reported in [0.0, -1.0, f64::NAN] {
            assert_eq!(cadence(reported), Duration::from_secs_f64(1.0 / 30.0));
        }
    }

    #[test]
    fn a_missing_clip_names_the_file_it_could_not_open() {
        let error = VideoSource::new("definitely-not-a-clip.mp4", false)
            .err()
            .expect("a missing file must fail");
        assert!(
            error.to_string().contains("definitely-not-a-clip.mp4"),
            "the error should name the file, got {error}"
        );
    }

    #[test]
    fn length_prefixed_nals_get_start_codes() {
        let payload = [0, 0, 0, 3, 0x07, 0x42, 0x11];
        let mut out = Vec::new();
        to_annex_b(&payload, &mut out);
        assert_eq!(out, [0, 0, 0, 1, 0x07, 0x42, 0x11]);
    }

    #[test]
    fn an_i_slice_marks_a_new_gop() {
        let idr = [0, 0, 0, 1, 0x05, 0x00];
        let p = [0, 0, 0, 1, 0x01, 0x00];
        assert!(is_intra(&idr));
        assert!(!is_intra(&p));
    }

    #[test]
    fn neutral_chroma_is_grey() {
        // Rec. 601 limited range: Y=128 with neutral chroma expands to
        // 1.164 * (128 - 16) = 130.3, so mid grey lands at 130, not 128.
        let y = [128u8; 4];
        let cb = [128u8; 1];
        let cr = [128u8; 1];
        let mut rgb = [0u8; 12];
        yuv420_to_rgb(&y, &cb, &cr, 2, 2, &mut rgb);
        assert_eq!(rgb, [130; 12], "neutral chroma must give a flat grey");
    }

    #[test]
    fn saturation_is_clamped_not_wrapped() {
        // Y=255 with strong chroma would overflow u8; it must saturate at 255.
        let y = [255u8; 4];
        let cb = [255u8; 1];
        let cr = [255u8; 1];
        let mut rgb = [0u8; 12];
        yuv420_to_rgb(&y, &cb, &cr, 2, 2, &mut rgb);
        assert_eq!(rgb[0], 255, "red should saturate, not wrap");
        assert!(rgb[1] < 255, "green should be pulled down by the chroma");
    }

    #[test]
    #[ignore]
    /// Decode cost of the fixture, the number `capture_ms` reports for a clip.
    /// Run with
    /// `cargo test --release -- --include-ignored bench_clip_decode --nocapture`.
    fn bench_clip_decode() {
        const CLIP: &str = "fixtures/hand_clip.mp4";
        const N: u64 = 300;
        let mut source =
            VideoSource::new(CLIP, false).expect("record a clip with `make record` first");
        let start = Instant::now();
        for _ in 0..N {
            let frame = source
                .read()
                .expect("the clip has more frames than the bench asks for");
            std::hint::black_box(&frame);
        }
        println!(
            "VideoSource decode + RGB convert 640x480: {:.3} ms/frame",
            start.elapsed().as_secs_f64() * 1000.0 / N as f64
        );
    }

    #[test]
    fn reordering_puts_b_frames_before_their_reference() {
        // Decode order poc 0, 4, 2 - display order is 0, 2, 4.
        let mut reorder = Reorder::new();
        let mut out = Vec::new();
        for poc in [0, 4, 2] {
            reorder.push(poc, Frame::new(1, 1, vec![poc as u8; 3]), &mut out);
        }
        reorder.flush(&mut out);
        let seen: Vec<u8> = out.iter().map(|f| f.pixels()[0]).collect();
        assert_eq!(seen, [0, 2, 4]);
    }
}
