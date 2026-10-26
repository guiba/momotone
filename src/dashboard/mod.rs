//! The iced dashboard: a native HUD window replacing the OpenCV `imshow` view.
//!
//! Layout is two columns:
//!   * left  - the hand skeleton on a dark stage (the webcam feed is off by
//!             default; the `feed` button puts it back)
//!   * right - a telemetry rail: note, volume, confidence, performance,
//!             hand geometry, rolling history, and the event log
//!
//! The pipeline thread writes a `FrameMetrics` into an `Arc<Mutex<_>>`; the UI
//! clones it on a 33 ms tick, so the camera loop never blocks on rendering.

pub mod gauges;
pub mod metrics;
pub mod skeleton;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use iced::widget::image::Handle;
use iced::widget::{button, canvas, column, container, row, rule, space, text, Column};
use iced::{alignment, Background, Border, Color, Element, Length, Shadow, Size, Theme};
use iced::{time, Subscription};

use crate::core::geometry::FINGER_NAMES;
use crate::dashboard::gauges::{Meter, Sparkline};
use crate::dashboard::metrics::{fade_trail, note_index, note_name, FrameMetrics, NOTE_COUNT};
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

#[derive(Debug, Clone, Copy)]
enum Message {
    Tick,
    ToggleBox,
    ToggleLabels,
    ToggleFeed,
    ClearHistory,
}

pub struct Dashboard {
    shared: Arc<Mutex<FrameMetrics>>,
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
    last_tick: Instant,
}

impl Dashboard {
    fn new(shared: Arc<Mutex<FrameMetrics>>) -> Self {
        Dashboard {
            shared,
            metrics: FrameMetrics::default(),
            frame_image: None,
            volume_history: Vec::new(),
            confidence_history: Vec::new(),
            show_box: true,
            show_labels: false,
            show_feed: false,
            note_trail: [0.0; NOTE_COUNT],
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
                fade_trail(&mut self.note_trail, dt);

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
            Message::ToggleBox => self.show_box = !self.show_box,
            Message::ToggleLabels => self.show_labels = !self.show_labels,
            Message::ToggleFeed => self.show_feed = !self.show_feed,
            Message::ClearHistory => {
                self.volume_history.clear();
                self.confidence_history.clear();
            }
        }
    }

    fn subscription(&self) -> Subscription<Message> {
        time::every(Duration::from_millis(UI_TICK_MS)).map(|_| Message::Tick)
    }

    fn theme(&self) -> Theme {
        Theme::Dark
    }

    fn view(&self) -> Element<'_, Message> {
        let header = row![
            text("Hand Gesture HUD").size(20),
            text(format!("{} · {}", self.metrics.mode, self.metrics.sink))
                .size(12)
                .color(MUTED),
            space::horizontal(),
            button(text(if self.show_feed { "hide feed" } else { "feed" }))
                .on_press(Message::ToggleFeed),
            button(text("box")).on_press(Message::ToggleBox),
            button(text("labels")).on_press(Message::ToggleLabels),
            button(text("clear")).on_press(Message::ClearHistory),
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

        let rail = column![
            note_card(&self.metrics),
            meter_card("Volume", self.metrics.volume, VOLUME),
            meter_card("Confidence", self.metrics.confidence, CONFIDENCE),
            perf_card(&self.metrics),
            geometry_card(&self.metrics),
        ]
        .spacing(10)
        .width(300);

        // `height(Length::Fill)` is what makes the stage take the whole middle
        // band: a `row!` defaults to `Length::Shrink`, which would collapse the
        // stage to the intrinsic height of the rail cards.
        let body = row![stage, rail].spacing(12).height(Length::Fill);

        // Bottom strip: time-series and log, away from the live numbers.
        let strip = row![
            sparkline_card("VOLUME", &self.volume_history, (0.0, 1.0), VOLUME),
            sparkline_card(
                "CONFIDENCE",
                &self.confidence_history,
                (0.0, 1.0),
                CONFIDENCE
            ),
            log_card(&self.metrics),
        ]
        .spacing(10);

        column![header, rule::horizontal(1), body, strip]
        .padding(14)
        .spacing(10)
        .into()
    }
}

fn app_title(_state: &Dashboard) -> String {
    "Hand Gesture HUD".to_string()
}

pub fn run(shared: Arc<Mutex<FrameMetrics>>) -> iced::Result {
    iced::application(
        move || Dashboard::new(shared.clone()),
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

fn note_card(metrics: &FrameMetrics) -> Element<'_, Message> {
    let head: Element<'_, Message> = match metrics.note {
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
            kv("frame", format!("{:.2} ms", metrics.frame_ms)),
            kv("detector", format!("{:.0} µs", metrics.detector_us)),
            kv("landmarker", format!("{:.0} µs", metrics.landmarker_us)),
            kv("frames", format!("{}", metrics.frames)),
            kv("uptime", format!("{} s", metrics.uptime_ms / 1000)),
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

    let status = if metrics.detected {
        "tracking".to_string()
    } else {
        format!("lost {} ms", metrics.lost_ms)
    };

    card(
        "HAND GEOMETRY",
        column![
            kv("status", status),
            kv("landmarks", format!("{}", metrics.landmark_count)),
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
