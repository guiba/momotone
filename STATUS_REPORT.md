# Hand Gesture Recognition Rust Port - Status Report

## Current Implementation Progress
The core architecture is successfully ported from Python to Rust. Key components are functional in `Mock` mode:
- **Core Domain**: Types (`GestureEvent`, `EventType`), Event Bus (`crossbeam-channel`), and modular traits (`Sink`, `Source`, `Tracker`) implemented.
- **Audio & MIDI**: Polyphonic synthesis via `cpal` and MIDI output via `midir` are ready.
- **Vision/HUD**: Infrastructure for OpenCV camera capture and HUD rendering is structured in `src/vision` and `src/visualizer`.
- **Orchestration**: Main loop supports Signal, Midi, and Mock modes to validate event flow without hardware dependency.

## Technical Blocker: `libclang` Issue
The project currently cannot be compiled due to a failure in the `opencv-rust` crate's build script.
- **Problem**: The `bindgen` process fails with error: `"a libclang shared library is not loaded on this thread"`.
- **Root Cause**: Even after installing LLVM via Homebrew and setting `LIBCLANG_PATH` to `/opt/homebrew/Cellar/llvm/22.1.8_2/lib`, the build environment fails to load the `.dylib`.
- **Attempted Fixes**: Switched from OpenCV 5 to OpenCV 4; manually exported `PKG_CONFIG_PATH` and `LIBCLANG_PATH`; attempted to prepend paths via `DYLD_LIBRARY_PATH`. The issue persists during the crate's compilation phase.

## Next Steps
1. **Resolve Binding Issue**: Troubleshoot `libclang` loading error (potentially involving `DYLD_LIBRARY_PATH` persistence or system-level LLVM configuration) to enable successful compilation.
2. **ML Integration**: Integrate ONNX runtime (`ort`) into `HandTrackerImpl` for real landmark detection.
3. **Strategy Layer**: Implement coordinate-to-gesture mapping logic.
4. **Live Integration**: Connect live OpenCV feed with the new ML and Strategy layers.
