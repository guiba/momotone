# Makefile — the local gate for momotone
#
# GitHub Actions (.github/workflows/ci.yml) runs the same checks on every push to
# master; this is the pre-commit version. Run
#
#     make verify        # fmt check + clippy (errors) + cargo check + tests
#
# before every commit. `make help` lists everything.
#
# Notes:
#   * `cargo check` is ~4 s warm. There is no heavy native build any more: the
#     vision path is pure Rust (see OPENCV_REMOVAL_PLAN.md).
#   * `make verify` treats clippy warnings as errors on purpose, and it is green.
#     Clear a lint rather than lowering the bar; if a suppression is ever needed,
#     say why in the commit message.
#   * `make trace-check` is a real regression gate, but only on an idle machine:
#     the detector's ROI anchor is seeded from the previous frame, so competing
#     load shifts the timings and therefore the detections.
#   * The `block v0.1.6` future-incompat notice comes from a transitive dep and is
#     not ours to fix.

SHELL      := /bin/sh
CARGO      ?= cargo
FRAMES     ?= 200
IMG        ?= output_frame.png
OUT        ?= out.png
CAMERA     ?= 0
COLS       ?= 70
CLIP       ?= fixtures/hand_clip.mp4
TRACE      ?= fixtures/clip_trace.json
MOCK_TRACE ?= fixtures/mock_trace.json
CLIP_FRAMES ?= 120

.DEFAULT_GOAL := help

.PHONY: help verify lint check clippy clippy-fix fmt fmt-check test bench build run \
        run-release mock ascii headless budget yolo image fixture trace-check record trace \
        mock-trace mock-trace-check \
        stats clean

## The gate: run this before every commit.
verify: check test fmt-check clippy
	@echo "verify: OK"

## --- quality ---------------------------------------------------------------

help:
	@echo "momotone — make targets"
	@echo ""
	@echo "  gate     verify | fmt-check | clippy | clippy-fix | check"
	@echo "  tests    test | bench"
	@echo "  builds   build | check"
	@echo "  run      run | run-release | mock | ascii | headless | yolo | image IMG=photo.png"
	@echo "  fixture  fixture | trace-check | record | trace"
	@echo "  mock     mock-trace | mock-trace-check"
	@echo "  misc     stats | budget | clean"

fmt:
	$(CARGO) fmt

fmt-check:
	$(CARGO) fmt --check

clippy:
	$(CARGO) clippy --all-targets -- -D warnings

clippy-fix:
	$(CARGO) clippy --all-targets --fix
	$(CARGO) fmt

## Non-fatal listing of what is currently failing, with file:line.
lint-count:
	$(CARGO) clippy --all-targets 2>&1 | grep -E '^warning:|^\s+--> src/'

check:
	$(CARGO) check --all-targets

test:
	$(CARGO) test

## The #[ignore]d benches (sampling, feed upload, mirror flip).
bench:
	$(CARGO) test --release -- --include-ignored

build:
	$(CARGO) build --release

## --- running -------------------------------------------------------------

run:
	$(CARGO) run

run-release:
	$(CARGO) run --release

## Synthetic hand, no webcam, no ONNX assets — the fastest smoke test.
mock:
	$(CARGO) run --release -- --mode mock --no-ui --max-frames $(FRAMES) --debug

## ASCII art in the terminal: quickest way to eyeball the renderer.
ascii:
	$(CARGO) run --release -- --mode mock --no-ui --ascii --ascii-cols $(COLS) --max-frames $(FRAMES)

## Headless live run: prints fps / capture / inference / dropped per frame.
headless:
	$(CARGO) run --release -- --no-ui --max-frames $(FRAMES) --debug

## Make the subsystem budget guard visible: inference's p95 is nowhere near 5 ms,
## so this fires the shed and logs why. `--budget-ms 0` turns the guard off.
budget:
	$(CARGO) run --release -- --no-ui --ascii --budget-ms 5 --max-frames 60 --debug

yolo:
	$(CARGO) run --release -- --model yolo --score-threshold 0.3 --no-ui --max-frames $(FRAMES) --debug

## Offline detection check on a still photo (no webcam needed).
image:
	$(CARGO) run --release -- --image $(IMG) --dump-frame $(OUT) --max-frames 1 --debug

## --- fixture runs ---------------------------------------------------------

## Record a 20 s webcam clip into fixtures/hand_clip.mp4. Needs ffmpeg and a
## free camera; run once, then use `make trace-check` from then on. The clip is
## not in the repo - it is a recording of a real hand, so every machine uses its
## own. `-pix_fmt yuv420p` is not cosmetic: the pure-Rust decoder in
## src/vision/video.rs only handles 4:2:0, and AVFoundation defaults to 4:2:2.
record:
	mkdir -p fixtures
	ffmpeg -y -f avfoundation -framerate 30 -video_size 640x480 -i "0:none" -t 20 -pix_fmt yuv420p $(CLIP)

## Run the recorded clip headlessly with per-stage timing.
fixture:
	@test -f $(CLIP) || { echo "no $(CLIP) - run \`make record\` first"; exit 1; }
	$(CARGO) run --release -- --no-ui --video $(CLIP) --max-frames $(CLIP_FRAMES) --debug

## Re-run the clip and compare against the golden trace.
trace-check:
	@test -f $(CLIP) || { echo "no $(CLIP) - run \`make record\` first"; exit 1; }
	@test -f $(TRACE) || { echo "no $(TRACE) - run \`make trace\` first"; exit 1; }
	$(CARGO) run --release -- --no-ui --video $(CLIP) --max-frames $(CLIP_FRAMES) --expect $(TRACE)

## Rewrite the golden trace from the current build. Only after an intentional
## change, and read the diff it produces first.
trace:
	@test -f $(CLIP) || { echo "no $(CLIP) - run \`make record\` first"; exit 1; }
	$(CARGO) run --release -- --no-ui --video $(CLIP) --max-frames $(CLIP_FRAMES) --trace $(TRACE)

## --- mock trace -----------------------------------------------------------

## The mock hand is driven by the frame index, so a mock run is reproducible on
## any machine with no camera and no ONNX assets. This is the trace CI checks.
mock-trace:
	$(CARGO) run --release -- --mode mock --no-ui --max-frames $(CLIP_FRAMES) --trace $(MOCK_TRACE)

mock-trace-check:
	$(CARGO) run --release -- --mode mock --no-ui --max-frames $(CLIP_FRAMES) --expect $(MOCK_TRACE)

## --- misc ---------------------------------------------------------------

stats:
	@find src -name '*.rs' | xargs wc -l | tail -1
	@$(CARGO) test 2>/dev/null | grep 'test result'

clean:
	$(CARGO) clean
