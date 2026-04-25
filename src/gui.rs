use std::fs;
use std::path::{Path, PathBuf};
#[cfg(any(target_os = "macos", target_os = "windows"))]
use std::process::Command;
use std::thread;

use chrono::Local;
use crossbeam_channel::{Receiver, Sender};
use eframe::egui;

use crate::engine::{self, CodecChoice, PreprocOpts, RunOpts};
use crate::progress::{
    CancellationToken, FinishStatus, LogLevel, ProgressEvent, ProgressReporter, RecommendationKind,
    SearchPhase, VariantInfo,
};

pub(crate) fn run_gui() -> Result<(), Box<dyn std::error::Error>> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "pdf-sizer",
        native_options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::light());
            Box::new(PdfSizerApp::new(cc))
        }),
    )?;
    Ok(())
}

#[derive(Clone, Debug)]
enum AppState {
    Idle,
    Running,
    Cancelling,
    Done,
    Failed(String),
    Cancelled,
}

#[derive(Clone)]
struct VariantRow {
    info: VariantInfo,
    kept: bool,
}

impl VariantRow {
    fn new(info: VariantInfo) -> Self {
        Self { info, kept: true }
    }
}

#[derive(Clone)]
struct LogLine {
    level: LogLevel,
    message: String,
}

enum VariantAction {
    Open(PathBuf),
    Reveal(PathBuf),
}

struct GuiReporter {
    tx: Sender<ProgressEvent>,
    ctx: egui::Context,
}

impl ProgressReporter for GuiReporter {
    fn report(&self, event: ProgressEvent) {
        let _ = self.tx.send(event);
        self.ctx.request_repaint();
    }
}

pub(crate) struct PdfSizerApp {
    input_dir: Option<PathBuf>,
    output_dir: Option<PathBuf>,
    target_mb: f64,
    /// Accepted deviation in MB. Shown and edited directly by the user.
    /// Automatically tracks 10 % of target_mb until the user edits it.
    tolerance_mb: f64,
    tolerance_user_set: bool,
    codec: CodecChoice,
    preproc: PreprocOpts,
    state: AppState,
    rx: Option<Receiver<ProgressEvent>>,
    cancel: CancellationToken,
    current_phase: Option<SearchPhase>,
    current_page: String,
    current_setting_label: String,
    setting_progress: Option<(usize, usize)>,
    current_probe_scale: Option<u32>,
    variants: Vec<VariantRow>,
    recommendations: Vec<RecommendationKind>,
    logs: Vec<LogLine>,
    confirm_delete: bool,
    last_output_dir: Option<PathBuf>,
}

impl PdfSizerApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            input_dir: None,
            output_dir: None,
            target_mb: 20.0,
            tolerance_mb: 2.0, // 10 % of the 20 MB default
            tolerance_user_set: false,
            codec: CodecChoice::Auto,
            preproc: PreprocOpts {
                despeckle: false,
                flatten_threshold: 0,
                deskew: false,
            },
            state: AppState::Idle,
            rx: None,
            cancel: CancellationToken::default(),
            current_phase: None,
            current_page: String::new(),
            current_setting_label: String::new(),
            setting_progress: None,
            current_probe_scale: None,
            variants: Vec::new(),
            recommendations: Vec::new(),
            logs: Vec::new(),
            confirm_delete: false,
            last_output_dir: None,
        }
    }

    fn is_busy(&self) -> bool {
        matches!(self.state, AppState::Running | AppState::Cancelling)
    }

    fn start_run(&mut self, ctx: &egui::Context) {
        let Some(input_dir) = self.input_dir.clone() else {
            self.log(LogLevel::Error, "Выберите папку с изображениями");
            return;
        };

        let output_dir = self
            .output_dir
            .clone()
            .unwrap_or_else(|| default_output_dir(&input_dir));

        let opts = RunOpts {
            input: input_dir,
            output: output_dir.clone(),
            target_mb: self.target_mb,
            tolerance_mb: self.tolerance_mb,
            preproc: PreprocOpts {
                despeckle: self.preproc.despeckle,
                flatten_threshold: self.preproc.flatten_threshold,
                deskew: self.preproc.deskew,
            },
            codec: self.codec,
        };

        let (tx, rx) = crossbeam_channel::unbounded();
        let cancel = CancellationToken::default();
        let worker_cancel = cancel.clone();
        let reporter = GuiReporter {
            tx,
            ctx: ctx.clone(),
        };

        self.state = AppState::Running;
        self.rx = Some(rx);
        self.cancel = cancel;
        self.current_phase = None;
        self.current_page.clear();
        self.current_setting_label.clear();
        self.setting_progress = None;
        self.current_probe_scale = None;
        self.variants.clear();
        self.recommendations.clear();
        self.logs.clear();
        self.confirm_delete = false;
        self.last_output_dir = Some(output_dir);

        thread::spawn(move || {
            let _ = engine::run_search(&opts, &reporter, &worker_cancel);
        });
    }

    fn cancel_run(&mut self) {
        if matches!(self.state, AppState::Running) {
            self.cancel.cancel();
            self.state = AppState::Cancelling;
            self.log(
                LogLevel::Warning,
                "Отмена запрошена: остановимся после текущей операции",
            );
        }
    }

    fn drain_messages(&mut self) {
        let Some(rx) = self.rx.clone() else {
            return;
        };

        let mut finished = false;
        for event in rx.try_iter() {
            match event {
                ProgressEvent::Phase(phase) => {
                    self.current_phase = Some(phase);
                    self.current_probe_scale = None;
                }
                ProgressEvent::ImagesFound { count } => {
                    self.current_page = format!("Найдено изображений: {count}");
                }
                ProgressEvent::CurrentPage { path, index, total } => {
                    self.current_page = format!("Страница {index}/{total}: {path}");
                }
                ProgressEvent::SettingStarted {
                    codec,
                    setting,
                    index,
                    total,
                } => {
                    self.current_setting_label = format!("{} — {setting}", codec_label(codec));
                    self.setting_progress = Some((index, total));
                    self.current_probe_scale = None;
                }
                ProgressEvent::Probe { scale, .. } => {
                    self.current_probe_scale = Some(scale);
                }
                ProgressEvent::VariantReady(info) => {
                    // Insert in descending size order so the largest (best quality) variants
                    // appear at the top of the list as they arrive.
                    let pos = self
                        .variants
                        .partition_point(|r| r.info.size_bytes >= info.size_bytes);
                    self.variants.insert(pos, VariantRow::new(info));
                }
                ProgressEvent::SettingSkipped { .. } => {}
                ProgressEvent::Recommendations(recommendations) => {
                    self.recommendations = recommendations;
                }
                ProgressEvent::Log { level, message } => {
                    self.logs.push(LogLine { level, message });
                }
                ProgressEvent::Finished { status } => {
                    self.current_phase = None;
                    self.setting_progress = None;
                    self.current_probe_scale = None;
                    self.state = match status {
                        FinishStatus::Success => AppState::Done,
                        FinishStatus::Cancelled => AppState::Cancelled,
                        FinishStatus::Failed(message) => AppState::Failed(message),
                    };
                    finished = true;
                }
            }
        }

        if finished {
            self.rx = None;
        }
    }

    fn log(&mut self, level: LogLevel, message: impl Into<String>) {
        self.logs.push(LogLine {
            level,
            message: message.into(),
        });
    }

    fn render_inputs(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let busy = self.is_busy();
        egui::Frame::group(ui.style())
            .fill(egui::Color32::from_rgb(248, 246, 239))
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 10.0;

                ui.horizontal(|ui| {
                    ui.label("Папка с изображениями:");
                    path_label(ui, self.input_dir.as_deref(), "не выбрана");
                    if ui
                        .add_enabled(!busy, egui::Button::new("Выбрать..."))
                        .clicked()
                        && let Some(path) = rfd::FileDialog::new().pick_folder()
                    {
                        self.input_dir = Some(path);
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Куда сохранить PDF:");
                    path_label(
                        ui,
                        self.output_dir.as_deref(),
                        "auto: внутри input/pdf-sizer-output",
                    );
                    if ui
                        .add_enabled(!busy, egui::Button::new("Выбрать..."))
                        .clicked()
                        && let Some(path) = rfd::FileDialog::new().pick_folder()
                    {
                        self.output_dir = Some(path);
                    }
                    if ui
                        .add_enabled(
                            !busy && self.output_dir.is_some(),
                            egui::Button::new("Auto"),
                        )
                        .clicked()
                    {
                        self.output_dir = None;
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Целевой размер:");
                    let old_target = self.target_mb;
                    ui.add_enabled(
                        !busy,
                        egui::Slider::new(&mut self.target_mb, 1.0..=200.0)
                            .text("MB")
                            .clamp_to_range(true),
                    );
                    // Auto-follow: if the user has never touched the tolerance field,
                    // keep it at 10 % of target so small and large targets both get a
                    // sensible default without the user having to adjust manually.
                    if !self.tolerance_user_set && self.target_mb != old_target {
                        self.tolerance_mb = (self.target_mb * 0.1 * 10.0).round() / 10.0;
                    }
                    ui.label("±");
                    let before = self.tolerance_mb;
                    ui.add_enabled(
                        !busy,
                        egui::DragValue::new(&mut self.tolerance_mb)
                            .speed(0.1)
                            .clamp_range(0.1..=50.0)
                            .min_decimals(1)
                            .max_decimals(1)
                            .suffix(" MB"),
                    );
                    if self.tolerance_mb != before {
                        self.tolerance_user_set = true;
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Режим:");
                    ui.add_enabled_ui(!busy, |ui| {
                        egui::ComboBox::from_id_source("codec_choice")
                            .selected_text(codec_label(self.codec))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.codec, CodecChoice::Auto, "Авто");
                                ui.selectable_value(
                                    &mut self.codec,
                                    CodecChoice::Jpeg,
                                    "Только JPEG",
                                );
                                ui.selectable_value(&mut self.codec, CodecChoice::Jp2, "JPEG2000");
                                ui.selectable_value(&mut self.codec, CodecChoice::Mrc, "MRC");
                            });
                    });
                });

                egui::CollapsingHeader::new("Препроцессинг")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.add_enabled_ui(!busy, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                ui.checkbox(&mut self.preproc.despeckle, "Despeckle");
                                ui.checkbox(&mut self.preproc.deskew, "Deskew");

                                let mut flatten = self.preproc.flatten_threshold > 0;
                                if ui.checkbox(&mut flatten, "Background flatten").changed() {
                                    self.preproc.flatten_threshold = if flatten { 30 } else { 0 };
                                }
                                ui.add_enabled(
                                    flatten,
                                    egui::DragValue::new(&mut self.preproc.flatten_threshold)
                                        .clamp_range(1..=255)
                                        .speed(1),
                                );
                            });
                        });
                    });

                ui.horizontal(|ui| {
                    let can_start = !busy && self.input_dir.is_some();
                    if ui
                        .add_enabled(
                            can_start,
                            egui::Button::new("Подобрать варианты")
                                .min_size(egui::vec2(180.0, 34.0)),
                        )
                        .clicked()
                    {
                        self.start_run(ctx);
                    }

                    if ui
                        .add_enabled(
                            matches!(self.state, AppState::Running),
                            egui::Button::new("Отмена").min_size(egui::vec2(90.0, 34.0)),
                        )
                        .clicked()
                    {
                        self.cancel_run();
                    }
                });
            });
    }

    fn render_status(&self, ui: &mut egui::Ui) {
        let phase = self
            .current_phase
            .map(phase_label)
            .unwrap_or_else(|| state_label(&self.state));
        let setting = if self.current_setting_label.is_empty() {
            String::new()
        } else if let Some(scale) = self.current_probe_scale {
            format!(" — {}, scale {}%", self.current_setting_label, scale)
        } else {
            format!(" — {}", self.current_setting_label)
        };

        egui::Frame::group(ui.style())
            .fill(egui::Color32::from_rgb(241, 247, 245))
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.strong(format!("Статус: {phase}{setting}"));
                    ui.label(format!("вариантов: {}", self.variants.len()));
                });
                if let AppState::Failed(message) = &self.state {
                    ui.colored_label(egui::Color32::from_rgb(170, 30, 30), message);
                }
                if !self.current_page.is_empty() {
                    ui.label(&self.current_page);
                }

                // The search uses early-exit and binary-search probes, so the total
                // planned steps rarely all execute. A deterministic percentage would
                // stall at a low value (e.g. 6%) and mislead. Instead: animate while
                // busy, show a step counter as text, fill to 100% on Done.
                match &self.state {
                    AppState::Running | AppState::Cancelling => {
                        let label = self
                            .setting_progress
                            .map(|(i, t)| format!("шаг {i}/{t}"))
                            .unwrap_or_default();
                        ui.add(
                            egui::ProgressBar::new(0.0)
                                .desired_width(f32::INFINITY)
                                .animate(true)
                                .text(label),
                        );
                    }
                    AppState::Done | AppState::Cancelled => {
                        ui.add(
                            egui::ProgressBar::new(1.0)
                                .desired_width(f32::INFINITY)
                                .text("готово"),
                        );
                    }
                    _ => {
                        ui.add(egui::ProgressBar::new(0.0).desired_width(f32::INFINITY));
                    }
                }
            });
    }

    fn render_recommendations(&mut self, ui: &mut egui::Ui) {
        if self.recommendations.is_empty() {
            return;
        }

        ui.heading("Рекомендованные");
        let mut actions = Vec::new();
        egui::Grid::new("recommendation_cards")
            .num_columns(2)
            .spacing([12.0, 12.0])
            .show(ui, |ui| {
                for (index, recommendation) in self.recommendations.clone().into_iter().enumerate()
                {
                    if let Some(row) = self.variant_for_recommendation(&recommendation).cloned() {
                        recommendation_card(ui, &recommendation, &row, &mut actions);
                    }
                    if index % 2 == 1 {
                        ui.end_row();
                    }
                }
            });
        self.handle_actions(actions);
    }

    fn render_variants(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading(format!("Все варианты ({})", self.variants.len()));
            let unmarked = self.variants.iter().filter(|row| !row.kept).count();
            if ui
                .add_enabled(unmarked > 0, egui::Button::new("Удалить непомеченные"))
                .clicked()
            {
                self.confirm_delete = true;
            }
        });

        if self.variants.is_empty() {
            ui.label("Пока вариантов нет. Они появятся здесь по мере поиска.");
            return;
        }

        let mut actions = Vec::new();
        egui::ScrollArea::vertical()
            .max_height(260.0)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                egui::Grid::new("variant_table")
                    .num_columns(6)
                    .spacing([12.0, 6.0])
                    .striped(true)
                    .show(ui, |ui| {
                        ui.strong("Оставить");
                        ui.strong("Параметр");
                        ui.strong("Scale");
                        ui.strong("Размер");
                        ui.strong("Файл");
                        ui.strong("Действия");
                        ui.end_row();

                        for row in &mut self.variants {
                            ui.checkbox(&mut row.kept, "");
                            ui.label(&row.info.setting);
                            ui.label(format!("{}%", row.info.scale_pct));
                            ui.label(format_size(row.info.size_bytes));
                            ui.label(file_name(&row.info.filename));
                            ui.horizontal(|ui| {
                                if ui.button("Открыть").clicked() {
                                    actions.push(VariantAction::Open(row.info.filename.clone()));
                                }
                                if ui.button("Показать").clicked() {
                                    actions.push(VariantAction::Reveal(row.info.filename.clone()));
                                }
                            });
                            ui.end_row();
                        }
                    });
            });
        self.handle_actions(actions);
    }

    fn render_logs(&self, ui: &mut egui::Ui) {
        if self.logs.is_empty() {
            return;
        }

        egui::CollapsingHeader::new("Лог")
            .default_open(false)
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .max_height(180.0)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for line in &self.logs {
                            let color = match line.level {
                                LogLevel::Info => egui::Color32::from_rgb(70, 70, 70),
                                LogLevel::Warning => egui::Color32::from_rgb(140, 91, 0),
                                LogLevel::Error => egui::Color32::from_rgb(170, 30, 30),
                            };
                            ui.colored_label(color, &line.message);
                        }
                    });
            });
    }

    fn render_delete_confirmation(&mut self, ctx: &egui::Context) {
        if !self.confirm_delete {
            return;
        }

        let count = self.variants.iter().filter(|row| !row.kept).count();
        egui::Window::new("Удалить файлы?")
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label(format!("Удалить {count} непомеченных PDF-файлов?"));
                ui.horizontal(|ui| {
                    if ui.button("Удалить").clicked() {
                        self.delete_unmarked();
                        self.confirm_delete = false;
                    }
                    if ui.button("Отмена").clicked() {
                        self.confirm_delete = false;
                    }
                });
            });
    }

    fn delete_unmarked(&mut self) {
        let mut kept = Vec::with_capacity(self.variants.len());
        for row in std::mem::take(&mut self.variants) {
            if row.kept {
                kept.push(row);
            } else if let Err(err) = fs::remove_file(&row.info.filename) {
                self.log(
                    LogLevel::Error,
                    format!("Не удалось удалить {}: {err}", row.info.filename.display()),
                );
                kept.push(row);
            }
        }
        self.variants = kept;
        self.recommendations.clear();
    }

    fn variant_for_recommendation(
        &self,
        recommendation: &RecommendationKind,
    ) -> Option<&VariantRow> {
        let id = recommendation_variant_id(recommendation);
        self.variants.iter().find(|row| row.info.id == id)
    }

    fn handle_actions(&mut self, actions: Vec<VariantAction>) {
        for action in actions {
            match action {
                VariantAction::Open(path) => {
                    if let Err(err) = opener::open(&path) {
                        self.log(LogLevel::Error, format!("Не удалось открыть PDF: {err}"));
                    }
                }
                VariantAction::Reveal(path) => {
                    if let Err(err) = reveal_path(&path) {
                        self.log(LogLevel::Error, format!("Не удалось показать файл: {err}"));
                    }
                }
            }
        }
    }
}

impl eframe::App for PdfSizerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_messages();

        egui::CentralPanel::default().show(ctx, |ui| {
            self.render_inputs(ui, ctx);
            ui.add_space(10.0);
            self.render_status(ui);
            ui.add_space(10.0);
            self.render_recommendations(ui);
            ui.add_space(10.0);
            self.render_variants(ui);
            ui.add_space(10.0);
            self.render_logs(ui);
        });

        self.render_delete_confirmation(ctx);
    }
}

fn path_label(ui: &mut egui::Ui, path: Option<&Path>, fallback: &str) {
    let text = path
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| fallback.to_string());
    ui.add_sized(
        [420.0, 20.0],
        egui::Label::new(egui::RichText::new(text).monospace()).truncate(true),
    );
}

fn recommendation_card(
    ui: &mut egui::Ui,
    recommendation: &RecommendationKind,
    row: &VariantRow,
    actions: &mut Vec<VariantAction>,
) {
    egui::Frame::group(ui.style())
        .fill(egui::Color32::from_rgb(255, 252, 242))
        .show(ui, |ui| {
            ui.set_min_width(260.0);
            ui.strong(recommendation_title(recommendation));
            ui.label(format!(
                "{} · scale {}% · {}",
                row.info.setting,
                row.info.scale_pct,
                format_size(row.info.size_bytes)
            ));
            ui.label(file_name(&row.info.filename));
            ui.horizontal(|ui| {
                if ui.button("Открыть").clicked() {
                    actions.push(VariantAction::Open(row.info.filename.clone()));
                }
                if ui.button("Показать").clicked() {
                    actions.push(VariantAction::Reveal(row.info.filename.clone()));
                }
            });
        });
}

fn recommendation_title(recommendation: &RecommendationKind) -> &'static str {
    match recommendation {
        RecommendationKind::BestForText(_) => "Лучший для текста",
        RecommendationKind::Balanced(_) => "Баланс",
        RecommendationKind::MaxDetail(_) => "Максимум деталей",
        RecommendationKind::Smallest(_) => "Самый маленький",
    }
}

fn recommendation_variant_id(recommendation: &RecommendationKind) -> usize {
    match recommendation {
        RecommendationKind::BestForText(id)
        | RecommendationKind::Balanced(id)
        | RecommendationKind::MaxDetail(id)
        | RecommendationKind::Smallest(id) => *id,
    }
}

fn state_label(state: &AppState) -> &'static str {
    match state {
        AppState::Idle => "готов",
        AppState::Running => "работаю",
        AppState::Cancelling => "останавливаюсь",
        AppState::Done => "готово",
        AppState::Failed(_) => "ошибка",
        AppState::Cancelled => "отменено",
    }
}

fn phase_label(phase: SearchPhase) -> &'static str {
    match phase {
        SearchPhase::LoadingImages => "загрузка изображений",
        SearchPhase::Preprocessing => "препроцессинг",
        SearchPhase::Jpeg => "JPEG",
        SearchPhase::Jp2 => "JP2",
        SearchPhase::Mrc => "MRC",
    }
}

fn codec_label(codec: CodecChoice) -> &'static str {
    match codec {
        CodecChoice::Auto => "Авто",
        CodecChoice::Jpeg => "JPEG",
        CodecChoice::Jp2 => "JP2",
        CodecChoice::Mrc => "MRC",
    }
}

fn format_size(size_bytes: u64) -> String {
    format!("{:.1} MB", size_bytes as f64 / (1024.0 * 1024.0))
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| path.display().to_string())
}

fn default_output_dir(input: &Path) -> PathBuf {
    let ts = Local::now().format("%Y%m%d-%H%M%S").to_string();
    input.join("pdf-sizer-output").join(ts)
}

fn reveal_path(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "macos")]
    {
        Command::new("open").arg("-R").arg(path).spawn()?;
        Ok(())
    }

    #[cfg(target_os = "windows")]
    {
        Command::new("explorer")
            .arg(format!("/select,{}", path.display()))
            .spawn()?;
        Ok(())
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let dir = path.parent().unwrap_or(path);
        opener::open(dir)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_variant(id: usize) -> VariantInfo {
        VariantInfo {
            id,
            filename: PathBuf::from(format!("variant-{id}.pdf")),
            codec: CodecChoice::Jpeg,
            setting: "q=080".to_string(),
            scale_pct: 75,
            size_bytes: 20 * 1024 * 1024,
        }
    }

    #[test]
    fn recommendation_to_card_label() {
        assert_eq!(
            recommendation_title(&RecommendationKind::BestForText(0)),
            "Лучший для текста"
        );
        assert_eq!(
            recommendation_title(&RecommendationKind::Balanced(0)),
            "Баланс"
        );
        assert_eq!(
            recommendation_title(&RecommendationKind::MaxDetail(0)),
            "Максимум деталей"
        );
        assert_eq!(
            recommendation_title(&RecommendationKind::Smallest(0)),
            "Самый маленький"
        );
    }

    #[test]
    fn variant_row_default_kept_is_true() {
        let row = VariantRow::new(make_variant(7));
        assert!(row.kept);
    }
}
