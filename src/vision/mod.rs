pub mod tracker;
pub mod camera;

use anyhow::Result;
use crate::core::{Landmark, HandDetectionResult};

// Re-exporting for convenience
pub use tracker::HandTrackerImpl;
pub use camera::WebcamSource;
