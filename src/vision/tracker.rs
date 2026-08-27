use anyhow::{anyhow, Result};
use opencv::core::Size;
use opencv::imgproc;
use opencv::prelude::*;
use ort::{inputs, session::{Session, SessionOutputs}, value::Value};
use crate::core::{HandDetectionResult, Landmark};
use ndarray::Array4;

pub struct HandTrackerImpl {
    session: Session,
    input_size: Size,
}

impl HandTrackerImpl {
    /// Initializes a new HandTracker using the provided ONNX model path.
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()?.commit_from_file(model_path)?;
 
        // Assuming a standard input size for hand landmark models, e.g., 256x256.
        let input_size = Size::new(256, 256);
 
        Ok(Self {
            session,
            input_size,
        })
    }
 
    /// Processes a single video frame to detect hand landmarks.
    pub fn process_frame(&mut self, frame: &Mat) -> Result<HandDetectionResult> {
        let mut resized = Mat::default();
        imgproc::resize(
            frame,
            &mut resized,
            self.input_size,
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;
 
        // Convert to RGB and then to float32 for normalization
        let mut rgb = Mat::default();
        imgproc::cvt_color(&resized, &mut rgb, imgproc::COLOR_BGR2RGB, 0, opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
 
        // Flatten the Mat into a Vec<f32> for ndarray conversion
        let mut float_data = vec![0.0f32; self.input_size.width as usize * self.input_size.height as usize * 3];
        for y in 0..self.input_size.height {
            for x in 0..self.input_size.width {
                let pixel = rgb.at_2d::<opencv::core::Vec3b>(y as i32, x as i32)?;
                let idx = (y * self.input_size.width + x) as usize * 3;
                float_data[idx]     = pixel[0] as f32 / 255.0; // R
                float_data[idx + 1] = pixel[1] as f32 / 255.0; // G
                float_data[idx + 2] = pixel[2] as f32 / 255.0; // B
            }
        }

        let mut input_tensor = Array4::<f32>::zeros((1, 3, self.input_size.height as usize, self.input_size.width as usize));
        for y in 0..self.input_size.height {
            for x in 0..self.input_size.width {
                let idx = (y * self.input_size.width + x) as usize;
                input_tensor[[0, 0, y as usize, x as usize]] = float_data[idx];
                input_tensor[[0, 1, y as usize, x as usize]] = float_data[idx+1];
                input_tensor[[0, 2, y as usize, x as usize]] = float_data[idx+2];
            }
        }

        // Use ort::Value to wrap the array for the session run call.
        let input_value = Value::from_array(input_tensor)?;
        
        // Run inference. Note: We pass a vec of (name, value) pairs.
        let outputs = self.session.run(vec![("input", input_value)])?;
 
        // Post-process the results into HandDetectionResult
        let landmarks = Self::parse_landmarks(outputs)?;
 
        Ok(HandDetectionResult {
            hand_landmarks: if landmarks.is_empty() { None } else { Some(landmarks) },
        })
    }
 
    fn parse_landmarks(_outputs: SessionOutputs) -> Result<Vec<Landmark>> {
        // This is a placeholder for actual tensor parsing logic.
        Ok(vec![])
    }
}
