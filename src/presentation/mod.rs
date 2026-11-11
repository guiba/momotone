//! Presentation layer: the parts of the app that exist only for a human to look at.
//!
//! `ascii` turns a frame into the quantised luminance grid behind the keyboard,
//! `presenter` runs that and the camera preview on its own thread, and `annotator`
//! draws the skeleton onto a saved frame for the offline `--image` check. Nothing
//! here is on the critical path to sound.

pub mod annotator;
pub mod ascii;
pub mod presenter;
