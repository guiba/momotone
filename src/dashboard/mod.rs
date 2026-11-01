//! The iced dashboard: a native HUD window replacing the OpenCV `imshow` view.
//!
//! Layout is two columns:
//!   * left  - the hand skeleton on a dark stage (the webcam feed is off by
//!     default; the `feed` button puts it back)
//!   * right - a telemetry rail with a **basic** view (note, volume, confidence,
//!     status) and an **advanced** view (performance, hand geometry, rolling
//!     history, event log)
//!
//! The header carries a Start/Stop button that pauses the pipeline thread: no
//! capture, no inference, and the sinks are released so the note goes silent.
//!
//! The pipeline thread writes a `FrameMetrics` into an `Arc<Mutex<_>>`; the UI
//! clones it on a 33 ms tick, so the camera loop never blocks on rendering.

pub mod gauges;
pub mod metrics;
pub mod skeleton;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use iced::widget::image::Handle;
use iced::widget::{
    button, canvas, column, container, pick_list, row, rule, space, text, Column, Scrollable,
};
use iced::{alignment, Background, Border, Color, Element, Length, Shadow, Size, Theme};
use iced::{time, Subscription};

use crate::core::geometry::FINGER_NAMES;
use crate::core::instrument::{InstrumentConfig, Waveform};
use crate::core::NOTE_COUNT;
use crate::dashboard::gauges::{Meter, Sparkline};
use crate::dashboard::metrics::{fade_trail, note_index, note_name, FrameMetrics};
use crate::dashboard::skeleton::Skeleton;

/// Number of samples kept for the rolling sparklines (~4 s at 30 fps).
const HISTORY: usize = 120;
const UI_TICK_MS: u64 = 33;
const BG: Color = Color::from_rgb8(16, 19, 24);
const CARD: Color = Color::from_rgb8(24, 28, 34);
const CARD_BORDER: Color = Color::from_rgb8(46, 53, 62);
const MUTED: Color = Color::from_rgb8(140, 150, 160);
const TEXT: Color = Color::from_rgb8(236, 240, 245);
const VOLUME: Color = Color::from_rgb8(6, 214, 160);
const CONFIDENCE: Color = Color::from_rgb8(78, 161, 255);
const NOTE: Color = Color::from_rgb8(255, 209, 102);
const ALARM: Color = Color::from_rgb8(255, 107, 107);

#[derive(Debug, Clone, Copy, PartialEq)]
enum Tab {
    Basic,
    Advanced,
}

#[derive(Debug, Clone)]
enum Message {
    Tick,
    ToggleRun,
    ShowBasic,
    ShowAdvanced,
    ToggleBox,
    ToggleLabels,
    ToggleFeed,
    ClearHistory,
    PickScale(String),
    PickRoot(String),
    PickOctaves(usize),
    PickWaveform(Waveform),
}

pub struct Dashboard {
    shared: Arc<Mutex<FrameMetrics>>,
    /// Cleared by the Stop button; the pipeline loop checks it every frame.
    active: Arc<AtomicBool>,
    /// Mirrors `show_feed` so the pipeline knows whether to build the RGBA
    /// upload at all.
    feed: Arc<AtomicBool>,
    /// The instrument the player is on. The pickers write to this and the
    /// pipeline picks the change up on the next frame.
    instrument: Arc<Mutex<InstrumentConfig>>,
    /// Local mirror of the config, so the pick lists show the current choice
    /// without locking on every `view`.
    instrument_view: InstrumentConfig,
    tab: Tab,
    metrics: FrameMetrics,
    frame_image: Option<Handle>,
    volume_history: Vec<f32>,
    confidence_history: Vec<f32>,
    show_box: bool,
    show_labels: bool,
    /// The stage is a backdrop + skeleton by default; the camera is only
    /// uploaded when this is on.
    show_feed: bool,
    /// Per-semitone brightness of the on-screen keyboard. Set to 1.0 when a
    /// note is played, exponentially faded on every UI tick.
    note_trail: [f32; NOTE_COUNT],
    /// Time constant of that fade, in seconds (`--fade-ms`).
    fade_tau: f32,
    last_tick: Instant,
}

impl Dashboard {
    fn new(
        shared: Arc<Mutex<FrameMetrics>>,
        active: Arc<AtomicBool>,
        feed: Arc<AtomicBool>,
        instrument: Arc<Mutex<InstrumentConfig>>,
        fade_ms: f32,
    ) -> Self {
        let instrument_view = instrument
            .lock()
            .map(|config| config.clone())
            .unwrap_or_default();
        Dashboard {
            shared,
            active,
            feed,
            instrument,
            instrument_view,
            tab: Tab::Basic,
            metrics: FrameMetrics::default(),
            frame_image: None,
            volume_history: Vec::new(),
            confidence_history: Vec::new(),
            show_box: true,
            show_labels: false,
            show_feed: false,
            note_trail: [0.0; NOTE_COUNT],
            fade_tau: fade_ms.max(1.0) / 1000.0,
            last_tick: Instant::now(),
        }
    }

    fn update(&mut self, message: Message) {
        match message {
            Message::Tick => {
                // Fade the keyboard trail against the wall clock, so the decay
                // rate is independent of the tick interval.
                let dt = self.last_tick.elapsed().as_secs_f32();
                self.last_tick = Instant::now();
                fade_trail(&mut self.note_trail, dt, self.fade_tau);

                if let Ok(mut shared) = self.shared.lock() {
                    // Only build an image handle when the feed is actually shown:
                    // a 640x480 RGBA frame is 1.2 MB per tick otherwise. Take
                    // ownership of the pixels so the clone below carries an empty
                    // buffer either way.
                    if self.show_feed && !shared.frame_pixels.is_empty() {
                        let (width, height) = shared.frame_size;
                        self.frame_image = Some(Handle::from_rgba(
                            width as u32,
                            height as u32,
                            std::mem::take(&mut shared.frame_pixels),
                        ));
                    } else {
                        shared.frame_pixels.clear();
                    }
                    self.metrics = shared.clone();
                }
                push(&mut self.volume_history, self.metrics.volume);
                push(&mut self.confidence_history, self.metrics.confidence);
                // Lighting a key restarts its fade from full brightness.
                if let Some(note) = self.metrics.note {
                    self.note_trail[note_index(note)] = 1.0;
                }
            }
            Message::ToggleRun => {
                self.active
                    .store(!self.active.load(Ordering::Relaxed), Ordering::Relaxed);
            }
            Message::ShowBasic => self.tab = Tab::Basic,
            Message::ShowAdvanced => self.tab = Tab::Advanced,
            Message::ToggleBox => self.show_box = !self.show_box,
            Message::ToggleLabels => self.show_labels = !self.show_labels,
            Message::ToggleFeed => {
                self.show_feed = !self.show_feed;
                self.feed.store(self.show_feed, Ordering::Relaxed);
            }
            Message::ClearHistory => {
                self.volume_history.clear();
                self.confidence_history.clear();
            }
            Message::PickScale(name) => self.apply(|config| config.set_scale(&name)),
            Message::PickRoot(name) => self.apply(|config| config.set_root(&name)),
            Message::PickOctaves(n) => self.apply(|config| config.set_octaves(n)),
            Message::PickWaveform(waveform) => self.apply(|config| config.set_waveform(waveform)),
        }
    }

    /// Write a picker change into the shared config and refresh the local mirror
    /// so the pick lists show the new value immediately. The pipeline applies it
    /// to the strategy and the sinks on its next frame.
    fn apply(&mut self, change: impl Fn(&mut InstrumentConfig) -> bool) {
        if let Ok(mut config) = self.instrument.lock() {
            change(&mut config);
            self.instrument_view = config.clone();
        }
    }

    fn subscription(&self) -> Subscription<Message> {
        time::every(Duration::from_millis(UI_TICK_MS)).map(|_| Message::Tick)
    }

    fn theme(&self) -> Theme {
        Theme::Dark
    }

    fn view(&self) -> Element<'_, Message> {
        let running = self.active.load(Ordering::Relaxed);

        let header = row![
            text("Hand Gesture HUD").size(20),
            text(format!(
                "{} · {} · {} {} · {}x{} · {} · trail {:.0} ms",
                self.metrics.mode,
                self.metrics.sink,
                self.metrics.scale,
                self.metrics.root,
                self.metrics.layout.rows,
                self.metrics.layout.cols,
                self.metrics.waveform,
                self.fade_tau * 1000.0
            ))
            .size(12)
            .color(MUTED),
            space::horizontal(),
            big_button(
                if running { "Stop" } else { "Start" },
                Message::ToggleRun,
                if running { ALARM } else { VOLUME },
            ),
            small_button(
                if self.show_feed { "hide feed" } else { "feed" },
                Message::ToggleFeed,
            ),
            small_button("box", Message::ToggleBox),
            small_button("labels", Message::ToggleLabels),
            small_button("clear", Message::ClearHistory),
        ]
        .spacing(10)
        .align_y(alignment::Vertical::Center);

        // The stage is the dominant panel: it fills the whole middle band, and
        // `Skeleton` maps the frame's coordinate space into a letterboxed
        // rectangle inside it.
        let stage = container(
            canvas(Skeleton {
                image: self.frame_image.clone(),
                show_feed: self.show_feed,
                landmarks: self.metrics.landmarks.clone(),
                frame_size: self.metrics.frame_size,
                hand_box: self.metrics.hand_box,
                show_box: self.show_box,
                show_labels: self.show_labels,
                detected: self.metrics.detected,
                lost_ms: self.metrics.lost_ms,
                note_trail: self.note_trail,
                layout: self.metrics.layout.clone(),
                current_note: self.metrics.note,
                volume: self.metrics.volume,
            })
            .width(Length::Fill)
            .height(Length::Fill),
        )
        .padding(8)
        .width(Length::Fill)
        .height(Length::Fill)
        .style(|_theme| container::Style {
            text_color: Some(TEXT),
            background: Some(Background::Color(BG)),
            border: Border {
                color: CARD_BORDER,
                width: 1.0,
                radius: 12.0.into(),
            },
            shadow: Shadow::default(),
            snap: false,
        });

        let tabs = row![
            tab_button("basic", self.tab == Tab::Basic, Message::ShowBasic),
            tab_button("advanced", self.tab == Tab::Advanced, Message::ShowAdvanced),
        ]
        .spacing(8);

        let cards: Column<'_, Message> = match self.tab {
            Tab::Basic => column![
                instrument_card(&self.instrument_view),
                note_card(&self.metrics),
                meter_card("Volume", self.metrics.volume, VOLUME),
                meter_card("Confidence", self.metrics.confidence, CONFIDENCE),
                status_card(&self.metrics),
            ]
            .spacing(10),
            Tab::Advanced => column![
                perf_card(&self.metrics),
                geometry_card(&self.metrics),
                sparkline_card("VOLUME", &self.volume_history, (0.0, 1.0), VOLUME),
                sparkline_card(
                    "CONFIDENCE",
                    &self.confidence_history,
                    (0.0, 1.0),
                    CONFIDENCE
                ),
                log_card(&self.metrics),
            ]
            .spacing(10),
        };

        // The rail scrolls, so the advanced tab can hold more cards than the
        // window height allows.
        let rail = column![tabs, cards].spacing(10).width(320);

        // `height(Length::Fill)` is what makes the stage take the whole middle
        // band: a `row!` defaults to `Length::Shrink`, which would collapse the
        // stage to the intrinsic height of the rail cards.
        let body = row![stage, Scrollable::new(rail).height(Length::Fill)]
            .spacing(12)
            .height(Length::Fill);

        column![header, rule::horizontal(1), body]
            .padding(14)
            .spacing(10)
            .into()
    }
}

fn app_title(_state: &Dashboard) -> String {
    "Hand Gesture HUD".to_string()
}

/// The Start/Stop control: deliberately chunky, it is the primary action.
fn big_button<'a>(label: &'a str, message: Message, color: Color) -> Element<'a, Message> {
    button(text(label).size(15))
        .on_press(message)
        .padding(12)
        .style(move |_theme, _status| button::Style {
            background: Some(Background::Color(color)),
            text_color: Color::from_rgb8(255, 255, 255),
            border: Border {
                color,
                width: 1.0,
                radius: 10.0.into(),
            },
            shadow: Shadow::default(),
            snap: false,
        })
        .into()
}

/// Secondary toggles: readable but not competing with Start/Stop.
fn small_button<'a>(label: &'a str, message: Message) -> Element<'a, Message> {
    button(text(label).size(13))
        .on_press(message)
        .padding(8)
        .style(move |_theme, _status| button::Style {
            background: Some(Background::Color(CARD)),
            text_color: TEXT,
            border: Border {
                color: CARD_BORDER,
                width: 1.0,
                radius: 8.0.into(),
            },
            shadow: Shadow::default(),
            snap: false,
        })
        .into()
}

fn tab_button<'a>(label: &'a str, selected: bool, message: Message) -> Element<'a, Message> {
    button(text(label).size(13))
        .on_press(message)
        .padding(8)
        .style(move |_theme, _status| button::Style {
            background: Some(Background::Color(if selected { CARD_BORDER } else { CARD })),
            text_color: if selected { TEXT } else { MUTED },
            border: Border {
                color: CARD_BORDER,
                width: 1.0,
                radius: 8.0.into(),
            },
            shadow: Shadow::default(),
            snap: false,
        })
        .into()
}

/// `feed` mirrors the stage's feed button so the pipeline can skip the per-frame
/// RGBA conversion when the camera is not shown.
pub fn run(
    shared: Arc<Mutex<FrameMetrics>>,
    active: Arc<AtomicBool>,
    feed: Arc<AtomicBool>,
    instrument: Arc<Mutex<InstrumentConfig>>,
    fade_ms: f32,
) -> iced::Result {
    iced::application(
        move || {
            Dashboard::new(
                shared.clone(),
                active.clone(),
                feed.clone(),
                instrument.clone(),
                fade_ms,
            )
        },
        Dashboard::update,
        Dashboard::view,
    )
    .subscription(Dashboard::subscription)
    .theme(Dashboard::theme)
    .title(app_title)
    .window_size(Size::new(1300.0, 860.0))
    .run()
}

fn push(history: &mut Vec<f32>, value: f32) {
    if history.len() >= HISTORY {
        history.remove(0);
    }
    history.push(value);
}

fn card<'a>(title: &'a str, body: impl Into<Element<'a, Message>>) -> Element<'a, Message> {
    container(column![text(title).size(11).color(MUTED), body.into()].spacing(6))
        .padding(12)
        .width(Length::Fill)
        .style(|_theme| container::Style {
            text_color: Some(TEXT),
            background: Some(Background::Color(CARD)),
            border: Border {
                color: CARD_BORDER,
                width: 1.0,
                radius: 10.0.into(),
            },
            shadow: Shadow::default(),
            snap: false,
        })
        .into()
}

fn kv<'a>(key: &'a str, value: String) -> Element<'a, Message> {
    row![
        text(key).size(11).color(MUTED),
        space::horizontal(),
        text(value).size(13).color(TEXT),
    ]
    .spacing(8)
    .into()
}

/// The instrument controls: scale, root, octave span and timbre. Each pick writes
/// straight into the shared config and the pipeline re-keys on its next frame, so
/// nothing here needs a restart.
fn instrument_card<'a>(config: &InstrumentConfig) -> Element<'a, Message> {
    let scales: Vec<String> = crate::core::scale::PRESETS
        .iter()
        .map(|(name, _)| name.to_string())
        .collect();
    let roots: Vec<String> = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ]
    .iter()
    .map(|&name| name.to_string())
    .collect();
    let octaves: Vec<usize> = (1..=6).collect();
    let waveforms: Vec<Waveform> = Waveform::all().to_vec();

    card(
        "INSTRUMENT",
        column![
            row![
                pick_list(
                    scales,
                    Some(config.scale.name.to_string()),
                    Message::PickScale
                )
                .width(Length::Fill),
                pick_list(roots, Some(config.root_name.clone()), Message::PickRoot)
                    .width(Length::Fill),
            ]
            .spacing(6),
            row![
                pick_list(octaves, Some(config.octaves), Message::PickOctaves).width(Length::Fill),
                pick_list(waveforms, Some(config.waveform), Message::PickWaveform)
                    .width(Length::Fill),
            ]
            .spacing(6),
            kv(
                "keyboard",
                format!(
                    "{} x {} - {} cells",
                    config.layout.rows,
                    config.layout.cols,
                    config.layout.len()
                ),
            ),
        ]
        .spacing(8),
    )
}

fn note_card(metrics: &FrameMetrics) -> Element<'_, Message> {
    let head: Element<'_, Message> = if metrics.paused {
        text("stopped").size(22).color(MUTED).into()
    } else {
        match metrics.note {
            Some(note) => row![
                text(note_name(note)).size(30).color(NOTE),
                space::horizontal(),
                text(format!("MIDI {note}")).size(12).color(MUTED),
            ]
            .spacing(10)
            .align_y(alignment::Vertical::Center)
            .into(),
            None => match metrics.frequency {
                Some(freq) => row![
                    text(format!("{freq:.0} Hz")).size(24).color(NOTE),
                    space::horizontal(),
                    text("signal").size(12).color(MUTED),
                ]
                .spacing(10)
                .align_y(alignment::Vertical::Center)
                .into(),
                None => text("no gesture").size(18).color(ALARM).into(),
            },
        }
    };

    card(
        "OUTPUT",
        column![
            head,
            kv("volume", format!("{:.0}%", metrics.volume * 100.0)),
        ]
        .spacing(6),
    )
}

fn meter_card<'a>(title: &'a str, value: f32, color: Color) -> Element<'a, Message> {
    column![
        text(title).size(11).color(MUTED),
        canvas(Meter { value, color })
            .width(Length::Fill)
            .height(14),
        text(format!("{:.0}%", value * 100.0)).size(13).color(TEXT),
    ]
    .spacing(6)
    .into()
}

fn perf_card<'a>(metrics: &'a FrameMetrics) -> Element<'a, Message> {
    card(
        "PERFORMANCE",
        column![
            kv("fps", format!("{:.1}", metrics.fps)),
            kv("inference", format!("{:.2} ms", metrics.frame_ms)),
            kv("capture", format!("{:.2} ms", metrics.capture_ms)),
            kv("latency", format!("{:.2} ms", metrics.latency_ms)),
            kv("detector", format!("{:.0} µs", metrics.detector_us)),
            kv("landmarker", format!("{:.0} µs", metrics.landmarker_us)),
            kv("frames", format!("{}", metrics.frames)),
            kv(
                "dropped",
                format!("{} of {} frames", metrics.dropped, metrics.captured),
            ),
            kv("uptime", format!("{} s", metrics.uptime_ms / 1000)),
        ]
        .spacing(4),
    )
}

fn status_card<'a>(metrics: &'a FrameMetrics) -> Element<'a, Message> {
    let status = if metrics.paused {
        "paused".to_string()
    } else if metrics.detected {
        "tracking".to_string()
    } else {
        format!("lost {} ms", metrics.lost_ms)
    };

    card(
        "STATUS",
        column![
            kv("state", status),
            kv(
                "pitch anchor",
                format!(
                    "tip {:.0}, {:.0} / {:.0} x {:.0} px",
                    metrics.tip_x, metrics.tip_y, metrics.frame_size.0, metrics.frame_size.1
                ),
            ),
            kv(
                "cell",
                format!(
                    "deg {} · octave {}",
                    metrics.grid_col,
                    metrics
                        .layout
                        .octaves
                        .get(metrics.grid_row)
                        .map_or(0, |o| o - 1)
                ),
            ),
            kv("landmarks", format!("{}", metrics.landmark_count)),
        ]
        .spacing(4),
    )
}

fn geometry_card<'a>(metrics: &'a FrameMetrics) -> Element<'a, Message> {
    let fingers: Vec<Element<'a, Message>> = metrics
        .extension
        .iter()
        .enumerate()
        .map(|(i, value)| {
            row![
                text(FINGER_NAMES[i]).size(10).color(MUTED),
                canvas(Meter {
                    value: (*value / 1.2).clamp(0.0, 1.0),
                    color: Color::from_rgb8(120, 170, 220),
                })
                .width(Length::Fill)
                .height(8),
                text(format!("{:.2}", value)).size(10).color(TEXT),
            ]
            .spacing(6)
            .into()
        })
        .collect();

    card(
        "HAND GEOMETRY",
        column![
            kv("spread", format!("{:.2}", metrics.spread)),
            kv("tilt", format!("{:.0}°", metrics.tilt_deg)),
            kv("jitter", format!("{:.1} px", metrics.jitter_px)),
            kv("palm", format!("{:.0} px", metrics.palm_length)),
            Column::with_children(fingers),
        ]
        .spacing(4),
    )
}

fn sparkline_card<'a>(
    title: &'a str,
    samples: &'a [f32],
    range: (f32, f32),
    color: Color,
) -> Element<'a, Message> {
    card(
        title,
        canvas(Sparkline {
            samples: samples.to_vec(),
            range,
            color,
        })
        .width(Length::Fill)
        .height(60),
    )
}

fn log_card<'a>(metrics: &'a FrameMetrics) -> Element<'a, Message> {
    let lines: Vec<Element<'a, Message>> = metrics
        .log
        .iter()
        .rev()
        .take(6)
        .map(|entry| {
            text(format!(
                "{}s  {}  {}",
                entry.uptime_ms / 1000,
                entry.text,
                entry.detail
            ))
            .size(11)
            .color(MUTED)
            .into()
        })
        .collect();

    card("EVENT LOG", Column::with_children(lines))
}
