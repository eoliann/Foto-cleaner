#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod ai;
mod face;
mod processing;

use std::{
    path::{Path, PathBuf},
    sync::{Arc, mpsc},
};

use eframe::egui::{
    self, Color32, ColorImage, Context, FontId, RichText, Slider, Stroke, TextStyle, TextureHandle,
    TextureOptions, Vec2,
};
use face::Face;
use image::{DynamicImage, GrayImage, ImageDecoder, ImageReader, RgbImage, RgbaImage};
use jpeg_encoder::{ColorType, Density, Encoder};
use processing::{Adjustments, BackgroundMode, Framing, OutputFormat, RenderSettings, Spot};

const DPI: u16 = 300;
/// Latura lungă a copiei folosite pentru previzualizare (randare rapidă).
const PREVIEW_LONG_EDGE: u32 = 1400;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1180.0, 820.0])
            .with_min_inner_size([760.0, 640.0])
            .with_title("Foto Cleaner by eoliann on GitHub"),
        ..Default::default()
    };

    eframe::run_native(
        "Foto Cleaner by eoliann on GitHub",
        options,
        Box::new(|cc| Ok(Box::new(PhotoApp::new(cc)))),
    )
}

// ---------------------------------------------------------------------------
// Stare
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Background {
    Original,
    Transparent,
    White,
    LightGray,
    LightBlue,
    Custom,
    Blur,
    Image,
}

#[derive(Clone, Copy, PartialEq)]
enum ExportKind {
    Jpeg,
    Png,
}

enum AiState {
    Idle,
    Running,
    Ready,
    Error(String),
}

struct InferenceJob {
    generation: u64,
    source: Arc<RgbImage>,
}

struct InferenceResult {
    generation: u64,
    mask: Result<GrayImage, String>,
}

enum FaceState {
    Idle,
    Running,
    Ready,
    Error(String),
}

struct FaceJob {
    generation: u64,
    image: Arc<RgbImage>,
}

struct FaceResult {
    generation: u64,
    faces: Result<Vec<Face>, String>,
}

struct RenderJob {
    id: u64,
    base: Arc<RgbImage>,
    mask: Option<Arc<GrayImage>>,
    background_image: Option<Arc<RgbImage>>,
    settings: RenderSettings,
}

struct RenderResult {
    id: u64,
    image: ColorImage,
}

struct PhotoApp {
    source: Option<Arc<RgbImage>>,
    preview_base: Option<Arc<RgbImage>>,
    source_path: Option<PathBuf>,
    mask_full: Option<Arc<GrayImage>>,
    mask_preview: Option<Arc<GrayImage>>,
    background_image: Option<Arc<RgbImage>>,
    background_image_name: Option<String>,

    adjustments: Adjustments,
    spots: Vec<Spot>,
    framing: Framing,
    background: Background,
    custom_background: [u8; 3],
    blur_strength: f32,
    edge_firmness: f32,
    retouch_mode: bool,
    brush_size: f32,
    show_original: bool,
    export_kind: ExportKind,

    texture: Option<TextureHandle>,
    source_generation: u64,
    next_render_id: u64,
    shown_render_id: u64,
    dirty: bool,

    ai_state: AiState,
    inference_tx: mpsc::Sender<InferenceJob>,
    inference_rx: mpsc::Receiver<InferenceResult>,
    faces: Vec<Face>,
    face_state: FaceState,
    face_generation: u64,
    show_face: bool,
    face_tx: mpsc::Sender<FaceJob>,
    face_rx: mpsc::Receiver<FaceResult>,
    render_tx: mpsc::Sender<RenderJob>,
    render_rx: mpsc::Receiver<RenderResult>,
    busy: bool,
    task_tx: mpsc::Sender<String>,
    task_rx: mpsc::Receiver<String>,
    status: String,
}

impl Background {
    fn needs_mask(self) -> bool {
        self != Self::Original
    }
}

impl PhotoApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        configure_style(&cc.egui_ctx);
        let (inference_tx, job_rx) = mpsc::channel();
        let (result_tx, inference_rx) = mpsc::channel();
        start_inference_worker(job_rx, result_tx, cc.egui_ctx.clone());
        let (render_tx, render_job_rx) = mpsc::channel();
        let (render_result_tx, render_rx) = mpsc::channel();
        start_render_worker(render_job_rx, render_result_tx, cc.egui_ctx.clone());
        let (face_tx, face_job_rx) = mpsc::channel();
        let (face_result_tx, face_rx) = mpsc::channel();
        start_face_worker(face_job_rx, face_result_tx, cc.egui_ctx.clone());
        let (task_tx, task_rx) = mpsc::channel();
        Self {
            source: None,
            preview_base: None,
            source_path: None,
            mask_full: None,
            mask_preview: None,
            background_image: None,
            background_image_name: None,
            adjustments: Adjustments::default(),
            spots: Vec::new(),
            framing: Framing::default(),
            background: Background::Original,
            custom_background: [255, 255, 255],
            blur_strength: 50.0,
            edge_firmness: 30.0,
            retouch_mode: false,
            brush_size: 1.5,
            show_original: false,
            export_kind: ExportKind::Jpeg,
            texture: None,
            source_generation: 0,
            next_render_id: 0,
            shown_render_id: 0,
            dirty: false,
            ai_state: AiState::Idle,
            inference_tx,
            inference_rx,
            faces: Vec::new(),
            face_state: FaceState::Idle,
            face_generation: 0,
            show_face: false,
            face_tx,
            face_rx,
            render_tx,
            render_rx,
            busy: false,
            task_tx,
            task_rx,
            status: "Încarcă o fotografie pentru a începe.".to_owned(),
        }
    }

    // --- Încărcare ----------------------------------------------------------

    fn open_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Fotografii", &["jpg", "jpeg", "png", "webp", "bmp"])
            .pick_file()
        {
            self.load_image(path);
        }
    }

    fn load_image(&mut self, path: PathBuf) {
        match open_oriented(&path) {
            Ok(image) => {
                let rgb = image.to_rgb8();
                self.source_generation = self.source_generation.wrapping_add(1);
                self.preview_base = Some(Arc::new(downscale(&rgb, PREVIEW_LONG_EDGE)));
                self.source = Some(Arc::new(rgb));
                self.source_path = Some(path);
                self.adjustments = Adjustments::default();
                self.spots.clear();
                self.framing.zoom = 1.0;
                self.framing.offset_x = 0.0;
                self.framing.offset_y = 0.0;
                self.framing.rotation = 0.0;
                self.mask_full = None;
                self.mask_preview = None;
                self.ai_state = AiState::Idle;
                self.show_original = false;
                if self.background.needs_mask() {
                    self.request_background_removal();
                }
                self.faces.clear();
                self.face_state = FaceState::Idle;
                self.request_face_detection();
                self.dirty = true;
                self.status =
                    "Apasă „Auto-corecție” sau reglează manual, apoi salvează.".to_owned();
            }
            Err(error) => {
                self.status = format!("Fotografia nu a putut fi deschisă: {error}");
            }
        }
    }

    fn load_background_image(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Imagini", &["jpg", "jpeg", "png", "webp", "bmp"])
            .pick_file()
        else {
            return;
        };
        match open_oriented(&path) {
            Ok(image) => {
                let rgb = downscale(&image.to_rgb8(), 4000);
                self.background_image = Some(Arc::new(rgb));
                self.background_image_name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(str::to_owned);
                self.dirty = true;
            }
            Err(error) => {
                self.status = format!("Imaginea de fundal nu a putut fi deschisă: {error}")
            }
        }
    }

    // --- Transformări -------------------------------------------------------

    fn rotate(&mut self, clockwise: bool) {
        let rotate_rgb = |image: &RgbImage| {
            if clockwise {
                image::imageops::rotate90(image)
            } else {
                image::imageops::rotate270(image)
            }
        };
        let rotate_gray = |image: &GrayImage| {
            if clockwise {
                image::imageops::rotate90(image)
            } else {
                image::imageops::rotate270(image)
            }
        };
        let Some(source) = &self.source else {
            return;
        };
        self.source = Some(Arc::new(rotate_rgb(source)));
        self.preview_base = self
            .preview_base
            .as_deref()
            .map(|p| Arc::new(rotate_rgb(p)));
        self.mask_full = self.mask_full.as_deref().map(|m| Arc::new(rotate_gray(m)));
        self.mask_preview = self
            .mask_preview
            .as_deref()
            .map(|m| Arc::new(rotate_gray(m)));
        for spot in &mut self.spots {
            let (x, y) = (spot.x, spot.y);
            (spot.x, spot.y) = if clockwise {
                (1.0 - y, x)
            } else {
                (y, 1.0 - x)
            };
        }
        if matches!(self.ai_state, AiState::Running) {
            // Masca în curs de calcul corespunde orientării vechi: o cerem din nou.
            self.source_generation = self.source_generation.wrapping_add(1);
            self.ai_state = AiState::Idle;
            self.request_background_removal();
        }
        self.faces = self.faces.iter().map(|f| f.rotated90(clockwise)).collect();
        if matches!(self.face_state, FaceState::Running) {
            self.face_state = FaceState::Idle;
            self.request_face_detection();
        }
        self.dirty = true;
    }

    fn request_face_detection(&mut self) {
        let Some(image) = self.preview_base.clone() else {
            return;
        };
        self.face_generation = self.face_generation.wrapping_add(1);
        self.face_state = FaceState::Running;
        if self
            .face_tx
            .send(FaceJob {
                generation: self.face_generation,
                image,
            })
            .is_err()
        {
            self.face_state = FaceState::Error("Detectorul de fețe nu a putut porni.".to_owned());
        }
    }

    fn receive_face_results(&mut self) {
        while let Ok(result) = self.face_rx.try_recv() {
            if result.generation != self.face_generation {
                continue;
            }
            match result.faces {
                Ok(faces) => {
                    self.faces = faces;
                    self.face_state = FaceState::Ready;
                    self.dirty = true;
                }
                Err(error) => {
                    self.faces.clear();
                    self.face_state = FaceState::Error(error);
                }
            }
        }
    }

    fn straighten_automatically(&mut self) -> bool {
        let (Some(face), Some(preview)) = (self.faces.first(), &self.preview_base) else {
            return false;
        };
        self.framing.rotation =
            processing::auto_straighten_angle(face, preview.width(), preview.height());
        self.dirty = true;
        true
    }

    fn frame_automatically(&mut self) -> bool {
        let (Some(face), Some(preview)) = (self.faces.first(), &self.preview_base) else {
            return false;
        };
        if self.framing.format == OutputFormat::Original {
            return false;
        }
        self.framing =
            processing::auto_frame(preview.width(), preview.height(), face, &self.framing);
        self.dirty = true;
        true
    }

    fn reset_all(&mut self) {
        self.adjustments = Adjustments::default();
        self.spots.clear();
        self.framing = Framing {
            format: self.framing.format,
            ..Framing::default()
        };
        self.dirty = true;
    }

    fn auto_correct(&mut self) {
        let Some(preview) = &self.preview_base else {
            return;
        };
        let skin = self.adjustments.skin_smoothing;
        self.adjustments =
            processing::auto_adjust(preview, self.mask_preview.as_deref(), &self.faces);
        self.adjustments.skin_smoothing = skin;
        self.show_original = false;
        self.dirty = true;
        let mut done = vec!["culori și lumină"];
        if self.straighten_automatically() {
            done.push("îndreptare după ochi");
        }
        if self.frame_automatically() {
            done.push("încadrare după față");
        }
        let basis = if !self.faces.is_empty() {
            " (optimizată după față)"
        } else if self.mask_preview.is_some() {
            " (optimizată pentru persoană)"
        } else {
            ""
        };
        self.status = format!(
            "Auto-corecție aplicată{basis}: {}. Poți regla fin manual.",
            done.join(", ")
        );
    }

    // --- Randare ------------------------------------------------------------

    fn settings(&self, preview: bool) -> RenderSettings {
        let background = match self.background {
            Background::Original => BackgroundMode::Original,
            Background::Transparent => BackgroundMode::Transparent,
            Background::White => BackgroundMode::Color([255, 255, 255]),
            Background::LightGray => BackgroundMode::Color([232, 234, 237]),
            Background::LightBlue => BackgroundMode::Color([200, 225, 245]),
            Background::Custom => BackgroundMode::Color(self.custom_background),
            Background::Blur => BackgroundMode::Blur(self.blur_strength),
            Background::Image if self.background_image.is_some() => BackgroundMode::Image,
            Background::Image => BackgroundMode::Original,
        };
        RenderSettings {
            adjustments: self.adjustments,
            spots: self.spots.clone(),
            framing: self.framing,
            background,
            edge_firmness: self.edge_firmness,
            bypass: preview && self.show_original,
            faces: self.faces.clone(),
        }
    }

    fn request_render(&mut self) {
        let Some(base) = self.preview_base.clone() else {
            return;
        };
        self.next_render_id += 1;
        let job = RenderJob {
            id: self.next_render_id,
            base,
            mask: self.mask_preview.clone(),
            background_image: self.background_image.clone(),
            settings: self.settings(true),
        };
        let _ = self.render_tx.send(job);
    }

    fn receive_renders(&mut self, ctx: &Context) {
        while let Ok(result) = self.render_rx.try_recv() {
            if result.id <= self.shown_render_id {
                continue;
            }
            self.shown_render_id = result.id;
            if let Some(texture) = &mut self.texture {
                texture.set(result.image, TextureOptions::LINEAR);
            } else {
                self.texture =
                    Some(ctx.load_texture("previzualizare", result.image, TextureOptions::LINEAR));
            }
        }
    }

    fn ready_for_export(&self) -> bool {
        self.source.is_some()
            && !self.busy
            && (!self.background.needs_mask() || self.mask_full.is_some())
    }

    /// Randează imaginea finală la rezoluție completă (pe un fir separat).
    fn final_render_job(&self) -> Option<impl FnOnce() -> RgbaImage + Send + 'static> {
        let source = self.source.clone()?;
        let mask = self.mask_full.clone();
        let background_image = self.background_image.clone();
        let settings = self.settings(false);
        Some(move || {
            processing::render(
                &source,
                mask.as_deref(),
                background_image.as_deref(),
                &settings,
                true,
            )
        })
    }

    fn effective_export_kind(&self) -> ExportKind {
        if self.background == Background::Transparent {
            ExportKind::Png
        } else {
            self.export_kind
        }
    }

    fn save_dialog(&mut self, ctx: &Context) {
        if !self.ready_for_export() {
            return;
        }
        let kind = self.effective_export_kind();
        let extension = match kind {
            ExportKind::Jpeg => "jpg",
            ExportKind::Png => "png",
        };
        let suffix = if self.framing.format == OutputFormat::Act3x4 {
            "foto_act_3x4"
        } else {
            "editat"
        };
        let default_name = self
            .source_path
            .as_deref()
            .and_then(Path::file_stem)
            .and_then(|name| name.to_str())
            .map(|name| format!("{name}_{suffix}.{extension}"))
            .unwrap_or_else(|| format!("fotografie_{suffix}.{extension}"));

        let dialog = match kind {
            ExportKind::Jpeg => rfd::FileDialog::new().add_filter("Imagine JPEG", &["jpg", "jpeg"]),
            ExportKind::Png => rfd::FileDialog::new().add_filter("Imagine PNG", &["png"]),
        };
        let Some(path) = dialog.set_file_name(default_name).save_file() else {
            return;
        };
        let Some(job) = self.final_render_job() else {
            return;
        };

        self.busy = true;
        self.status = "Se procesează imaginea la rezoluție completă…".to_owned();
        let tx = self.task_tx.clone();
        let repaint = ctx.clone();
        std::thread::spawn(move || {
            let image = job();
            let result = match kind {
                ExportKind::Jpeg => save_jpeg_300_dpi(&flatten_on_white(&image), &path),
                ExportKind::Png => image
                    .save_with_format(&path, image::ImageFormat::Png)
                    .map_err(|error| error.to_string()),
            };
            let message = match result {
                Ok(()) => format!(
                    "Imagine salvată ({}x{} px): {}",
                    image.width(),
                    image.height(),
                    path.display()
                ),
                Err(error) => format!("Imaginea nu a putut fi salvată: {error}"),
            };
            let _ = tx.send(message);
            repaint.request_repaint();
        });
    }

    fn print_direct(&mut self, ctx: &Context) {
        if !self.ready_for_export() {
            return;
        }
        let Some(job) = self.final_render_job() else {
            return;
        };
        let mut temp_dir = std::env::temp_dir();
        temp_dir.push("Foto-cleaner");
        if let Err(error) = std::fs::create_dir_all(&temp_dir) {
            self.status = format!("Nu s-a putut pregăti fișierul pentru print: {error}");
            return;
        }
        let script_path = temp_dir.join("print_imagine.ps1");
        if let Err(error) = std::fs::write(&script_path, PRINT_SCRIPT) {
            self.status = format!("Scriptul de printare nu a putut fi creat: {error}");
            return;
        }
        let stem = self
            .source_path
            .as_deref()
            .and_then(Path::file_stem)
            .and_then(|name| name.to_str())
            .unwrap_or("imagine")
            .to_owned();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0);
        let path = temp_dir.join(format!("{stem}_print_{timestamp}.jpg"));
        // Foto act: tipărire la mărime reală 3x4 cm; altfel imaginea se încadrează în pagină.
        let real_size = (self.framing.format == OutputFormat::Act3x4).then_some((3.0, 4.0));

        self.busy = true;
        self.status = "Se pregătește imaginea pentru print…".to_owned();
        let tx = self.task_tx.clone();
        let repaint = ctx.clone();
        std::thread::spawn(move || {
            let image = flatten_on_white(&job());
            let message = match save_jpeg_300_dpi(&image, &path) {
                Ok(()) => run_print_dialog(&script_path, &path, real_size),
                Err(error) => format!("Fișierul pentru print nu a putut fi creat: {error}"),
            };
            let _ = tx.send(message);
            repaint.request_repaint();
        });
    }

    // --- AI -----------------------------------------------------------------

    fn request_background_removal(&mut self) {
        let Some(source) = self.source.clone() else {
            return;
        };
        if matches!(self.ai_state, AiState::Running) {
            return;
        }
        self.ai_state = AiState::Running;
        if self
            .inference_tx
            .send(InferenceJob {
                generation: self.source_generation,
                source,
            })
            .is_err()
        {
            self.ai_state = AiState::Error("Motorul AI nu a putut fi pornit.".to_owned());
        }
    }

    fn receive_inference_results(&mut self) {
        while let Ok(result) = self.inference_rx.try_recv() {
            if result.generation != self.source_generation {
                // Rezultat pentru o fotografie veche: dacă între timp s-a cerut
                // o mască nouă, cererea rămâne în așteptare în coadă.
                continue;
            }
            match result.mask {
                Ok(mask) => {
                    // Masca a fost calculată înainte de eventuale rotiri; o aliniem.
                    let mask = match (&self.source, mask) {
                        (Some(source), mask) => align_mask(mask, source.dimensions()),
                        (None, mask) => mask,
                    };
                    if let Some(preview) = &self.preview_base {
                        self.mask_preview = Some(Arc::new(processing::resize_mask(
                            &mask,
                            preview.width(),
                            preview.height(),
                        )));
                    }
                    self.mask_full = Some(Arc::new(mask));
                    self.ai_state = AiState::Ready;
                    self.status =
                        "Persoana a fost detectată. Fundalul poate fi eliminat sau înlocuit."
                            .to_owned();
                    self.dirty = true;
                }
                Err(error) => {
                    self.mask_full = None;
                    self.mask_preview = None;
                    self.ai_state = AiState::Error(error);
                }
            }
        }
    }
}

/// Dacă imaginea a fost rotită în timpul procesării AI, masca are dimensiunile
/// transpuse; o redimensionăm doar ca ultimă soluție.
fn align_mask(mask: GrayImage, dims: (u32, u32)) -> GrayImage {
    if mask.dimensions() == dims {
        mask
    } else {
        processing::resize_mask(&mask, dims.0, dims.1)
    }
}

// ---------------------------------------------------------------------------
// Interfață
// ---------------------------------------------------------------------------

fn black(text: impl Into<String>) -> RichText {
    RichText::new(text).color(Color32::BLACK)
}

fn heading(ui: &mut egui::Ui, text: &str) {
    ui.add_space(14.0);
    ui.label(RichText::new(text).heading().color(Color32::BLACK));
}

fn slider(
    ui: &mut egui::Ui,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    text: &str,
) -> bool {
    ui.add(
        Slider::new(value, range)
            .text(text)
            .text_color(Color32::BLACK)
            .fixed_decimals(0),
    )
    .changed()
}

impl eframe::App for PhotoApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        self.receive_inference_results();
        self.receive_face_results();
        self.receive_renders(ctx);
        while let Ok(message) = self.task_rx.try_recv() {
            self.busy = false;
            self.status = message;
        }

        if let Some(path) = ctx.input(|input| {
            input
                .raw
                .dropped_files
                .iter()
                .find_map(|file| file.path.clone())
        }) {
            self.load_image(path);
        }

        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("Foto Cleaner")
                        .heading()
                        .color(Color32::BLACK),
                );
                ui.label(black(format!("v{}", env!("CARGO_PKG_VERSION"))));
                ui.separator();
                ui.label(black(
                    "O singură imagine • corecții • retuș • fundal cu AI local",
                ));
            });
            ui.add_space(8.0);
        });

        egui::SidePanel::left("controls")
            .resizable(false)
            .exact_width(340.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.add_space(12.0);
                    self.controls(ui, ctx);
                    ui.add_space(10.0);
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| self.preview_panel(ui));

        if self.dirty && self.preview_base.is_some() {
            self.dirty = false;
            self.request_render();
        }
    }
}

impl PhotoApp {
    fn controls(&mut self, ui: &mut egui::Ui, ctx: &Context) {
        let has_image = self.source.is_some();
        let full = ui.available_width();

        if ui
            .add_sized([full, 42.0], egui::Button::new("Încarcă fotografia"))
            .clicked()
        {
            self.open_dialog();
        }
        ui.label(black("Poți și să tragi fotografia peste fereastră."));

        ui.add_enabled_ui(has_image, |ui| {
            ui.horizontal(|ui| {
                if ui.button("⟲ Rotește stânga").clicked() {
                    self.rotate(false);
                }
                if ui.button("⟳ Rotește dreapta").clicked() {
                    self.rotate(true);
                }
            });
            ui.horizontal(|ui| {
                if ui
                    .add(
                        Slider::new(&mut self.framing.rotation, -15.0..=15.0)
                            .text("Îndreptare °")
                            .text_color(Color32::BLACK)
                            .fixed_decimals(1),
                    )
                    .changed()
                {
                    self.dirty = true;
                }
            });
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!self.faces.is_empty(), egui::Button::new("Îndreaptă după ochi"))
                    .clicked()
                {
                    self.straighten_automatically();
                }
                if ui.button("0°").clicked() {
                    self.framing.rotation = 0.0;
                    self.dirty = true;
                }
            });

            // --- Față (AI) -------------------------------------------------
            match &self.face_state {
                FaceState::Running => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(black("Se caută fața..."));
                    });
                }
                FaceState::Ready => match self.faces.first() {
                    Some(face) => {
                        let extra = match self.faces.len() {
                            1 => String::new(),
                            n => format!(" ({n} fețe, se folosește cea mai mare)"),
                        };
                        ui.label(
                            RichText::new(format!(
                                "Față detectată, încredere {:.0}%{extra}",
                                face.score * 100.0
                            ))
                            .color(Color32::DARK_GREEN),
                        );
                        if ui
                            .checkbox(&mut self.show_face, black("Arată fața detectată"))
                            .changed()
                        {
                            self.dirty = true;
                        }
                    }
                    None => {
                        ui.label(black("Nicio față detectată; corecțiile folosesc toată imaginea."));
                    }
                },
                FaceState::Error(error) => {
                    ui.label(
                        RichText::new(format!("Detectare față: {error}")).color(Color32::DARK_RED),
                    );
                }
                FaceState::Idle => {}
            }

            // --- Auto ---------------------------------------------------
            heading(ui, "Corecție automată");
            ui.label(black(
                "Analizează expunerea, balansul de alb, contrastul, culorile, zgomotul și claritatea.",
            ));
            let auto = egui::Button::new(RichText::new("✨ Auto-corecție").strong())
                .fill(Color32::from_rgb(120, 70, 170));
            ui.horizontal(|ui| {
                if ui.add_sized([200.0, 36.0], auto).clicked() {
                    self.auto_correct();
                }
                if ui.add_sized([110.0, 36.0], egui::Button::new("Resetează tot")).clicked() {
                    self.reset_all();
                }
            });
            if ui
                .checkbox(&mut self.show_original, black("Arată originalul (comparare)"))
                .changed()
            {
                self.dirty = true;
            }

            // --- Lumină ---------------------------------------------------
            heading(ui, "Lumină");
            let a = &mut self.adjustments;
            let mut changed = false;
            changed |= slider(ui, &mut a.exposure, -100.0..=100.0, "Expunere");
            changed |= slider(ui, &mut a.brightness, -100.0..=100.0, "Luminozitate");
            changed |= slider(ui, &mut a.contrast, -100.0..=100.0, "Contrast");
            changed |= slider(ui, &mut a.highlights, -100.0..=100.0, "Lumini");
            changed |= slider(ui, &mut a.shadows, -100.0..=100.0, "Umbre");

            heading(ui, "Culoare");
            changed |= slider(ui, &mut a.saturation, -100.0..=100.0, "Saturație");
            changed |= slider(ui, &mut a.vibrance, -100.0..=100.0, "Vibranță");
            changed |= slider(ui, &mut a.temperature, -100.0..=100.0, "Temperatură");
            changed |= slider(ui, &mut a.tint, -100.0..=100.0, "Nuanță");

            heading(ui, "Detalii");
            changed |= slider(ui, &mut a.clarity, -100.0..=100.0, "Claritate");
            changed |= slider(ui, &mut a.sharpness, 0.0..=100.0, "Accentuare");
            changed |= slider(ui, &mut a.denoise, 0.0..=100.0, "Reducere zgomot");
            if ui.button("Resetează corecțiile").clicked() {
                let skin = a.skin_smoothing;
                *a = Adjustments {
                    skin_smoothing: skin,
                    ..Adjustments::default()
                };
                changed = true;
            }

            // --- Retuș ----------------------------------------------------
            heading(ui, "Eliminare impurități");
            changed |= slider(
                ui,
                &mut self.adjustments.skin_smoothing,
                0.0..=100.0,
                "Netezire piele",
            );
            if changed {
                self.dirty = true;
            }
            ui.checkbox(
                &mut self.retouch_mode,
                black("Pensulă pete: click pe imagine pentru a șterge"),
            );
            ui.add(
                Slider::new(&mut self.brush_size, 0.3..=8.0)
                    .text("Mărime pensulă")
                    .text_color(Color32::BLACK)
                    .fixed_decimals(1),
            );
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!self.spots.is_empty(), egui::Button::new("Anulează ultima"))
                    .clicked()
                {
                    self.spots.pop();
                    self.dirty = true;
                }
                if ui
                    .add_enabled(!self.spots.is_empty(), egui::Button::new("Șterge toate"))
                    .clicked()
                {
                    self.spots.clear();
                    self.dirty = true;
                }
                ui.label(black(format!("{} pete", self.spots.len())));
            });

            // --- Fundal ---------------------------------------------------
            heading(ui, "Fundal");
            ui.label(black(
                "Eliminarea folosește AI local; fotografia nu părăsește calculatorul.",
            ));
            let previous = self.background;
            ui.horizontal_wrapped(|ui| {
                for (value, label) in [
                    (Background::Original, "Original"),
                    (Background::Transparent, "Transparent"),
                    (Background::White, "Alb"),
                    (Background::LightGray, "Gri deschis"),
                    (Background::LightBlue, "Albastru deschis"),
                    (Background::Custom, "Personalizat"),
                    (Background::Blur, "Estompat"),
                    (Background::Image, "Imagine"),
                ] {
                    ui.radio_value(&mut self.background, value, black(label));
                }
            });
            match self.background {
                Background::Custom => {
                    ui.horizontal(|ui| {
                        ui.label(black("Culoare:"));
                        if ui.color_edit_button_srgb(&mut self.custom_background).changed() {
                            self.dirty = true;
                        }
                    });
                }
                Background::Blur => {
                    if slider(ui, &mut self.blur_strength, 0.0..=100.0, "Intensitate estompare") {
                        self.dirty = true;
                    }
                }
                Background::Image => {
                    ui.horizontal(|ui| {
                        if ui.button("Alege imaginea de fundal").clicked() {
                            self.load_background_image();
                        }
                        if let Some(name) = &self.background_image_name {
                            ui.label(black(name.clone()).small());
                        }
                    });
                }
                Background::Transparent => {
                    ui.label(black("Se salvează automat ca PNG cu transparență.").small());
                }
                _ => {}
            }
            if self.background.needs_mask()
                && slider(ui, &mut self.edge_firmness, 0.0..=100.0, "Fermitate margine")
            {
                self.dirty = true;
            }
            if self.background != previous {
                if self.background.needs_mask() && self.mask_full.is_none() {
                    self.request_background_removal();
                }
                self.dirty = true;
            }
            match &self.ai_state {
                AiState::Running => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(black("AI procesează fotografia..."));
                    });
                }
                AiState::Ready if self.background.needs_mask() => {
                    ui.label(RichText::new("Fundal detectat cu succes.").color(Color32::DARK_GREEN));
                }
                AiState::Error(error) => {
                    ui.label(
                        RichText::new(format!("Eroare AI: {error}")).color(Color32::DARK_RED),
                    );
                    if ui.button("Încearcă din nou").clicked() {
                        self.ai_state = AiState::Idle;
                        self.request_background_removal();
                    }
                }
                _ => {}
            }

            // --- Format ---------------------------------------------------
            heading(ui, "Format final");
            let previous_format = self.framing.format;
            egui::ComboBox::from_id_salt("format")
                .width(ui.available_width() - 8.0)
                .selected_text(self.framing.format.label())
                .show_ui(ui, |ui| {
                    for format in OutputFormat::ALL {
                        ui.selectable_value(&mut self.framing.format, format, format.label());
                    }
                });
            if self.framing.format != previous_format {
                // La schimbarea formatului, fața este încadrată automat.
                self.frame_automatically();
                self.dirty = true;
            }
            if self.framing.format != OutputFormat::Original {
                let f = &mut self.framing;
                let mut moved = false;
                moved |= ui
                    .add(
                        Slider::new(&mut f.zoom, 1.0..=4.0)
                            .text("Zoom")
                            .text_color(Color32::BLACK),
                    )
                    .changed();
                moved |= ui
                    .add(
                        Slider::new(&mut f.offset_x, -1.0..=1.0)
                            .text("Stânga / dreapta")
                            .text_color(Color32::BLACK),
                    )
                    .changed();
                moved |= ui
                    .add(
                        Slider::new(&mut f.offset_y, -1.0..=1.0)
                            .text("Sus / jos")
                            .text_color(Color32::BLACK),
                    )
                    .changed();
                let mut reset = false;
                let mut auto = false;
                ui.horizontal(|ui| {
                    auto = ui
                        .add_enabled(
                            !self.faces.is_empty(),
                            egui::Button::new("Încadrare automată după față"),
                        )
                        .clicked();
                    reset = ui.button("Resetează").clicked();
                });
                if reset {
                    let f = &mut self.framing;
                    f.zoom = 1.0;
                    f.offset_x = 0.0;
                    f.offset_y = 0.0;
                    moved = true;
                }
                if auto {
                    self.frame_automatically();
                }
                if moved {
                    self.dirty = true;
                }
            }
            if let Some(source) = &self.source {
                let (_, _, w, h) =
                    processing::crop_rect(source.width(), source.height(), &self.framing);
                let (w, h) = self.framing.format.fixed_size().unwrap_or((w, h));
                ui.label(black(format!("Rezultat: {w} x {h} px")).small());
            }

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label(black("Salvează ca:"));
                ui.add_enabled_ui(self.background != Background::Transparent, |ui| {
                    ui.radio_value(&mut self.export_kind, ExportKind::Jpeg, black("JPEG"));
                    ui.radio_value(&mut self.export_kind, ExportKind::Png, black("PNG"));
                });
            });
        });

        ui.add_space(14.0);
        let can_export = self.ready_for_export();
        let save_label = if self.busy {
            "Se procesează…"
        } else {
            "Salvează imaginea"
        };
        let save = egui::Button::new(RichText::new(save_label).strong())
            .fill(Color32::from_rgb(32, 103, 178));
        if ui
            .add_enabled_ui(can_export, |ui| ui.add_sized([full, 44.0], save).clicked())
            .inner
        {
            self.save_dialog(ctx);
        }
        ui.add_space(6.0);
        let print = egui::Button::new(RichText::new("Printează direct").strong())
            .fill(Color32::from_rgb(46, 125, 50));
        if ui
            .add_enabled_ui(can_export, |ui| ui.add_sized([full, 44.0], print).clicked())
            .inner
        {
            self.print_direct(ctx);
        }

        ui.add_space(12.0);
        ui.separator();
        ui.label(RichText::new(&self.status).small().color(Color32::BLACK));
        if self.framing.format == OutputFormat::Act3x4 {
            ui.label(
                RichText::new("Foto act: se printează la mărime reală 3x4 cm (100%).")
                    .small()
                    .strong()
                    .color(Color32::BLACK),
            );
        }
    }

    fn preview_panel(&mut self, ui: &mut egui::Ui) {
        ui.add_space(14.0);
        ui.label(
            RichText::new("Previzualizare")
                .heading()
                .color(Color32::BLACK),
        );
        let hint = if self.show_original {
            "Original, fără corecții."
        } else if self.retouch_mode {
            "Click pe pete, coșuri sau praf pentru a le elimina."
        } else {
            "Imaginea finală, așa cum va fi salvată."
        };
        ui.label(black(hint));
        ui.add_space(10.0);

        let Some(texture) = &self.texture else {
            let width = ui.available_width().min(720.0);
            let (rect, _) =
                ui.allocate_exact_size(Vec2::new(width, width * 2.0 / 3.0), egui::Sense::hover());
            ui.painter().rect_filled(rect, 3.0, Color32::from_gray(235));
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Previzualizarea va apărea aici",
                FontId::proportional(18.0),
                Color32::BLACK,
            );
            return;
        };

        let available = ui.available_size() - Vec2::new(8.0, 8.0);
        let ratio = texture.size_vec2().x / texture.size_vec2().y;
        let mut size = Vec2::new(available.x, available.x / ratio);
        if size.y > available.y {
            size = Vec2::new(available.y * ratio, available.y);
        }
        let sense = if self.retouch_mode && !self.show_original {
            egui::Sense::click()
        } else {
            egui::Sense::hover()
        };
        let response = ui.add(
            egui::Image::new(texture)
                .fit_to_exact_size(size)
                .corner_radius(2.0)
                .sense(sense),
        );
        ui.painter().rect_stroke(
            response.rect,
            2.0,
            Stroke::new(1.0_f32, Color32::from_gray(155)),
            egui::StrokeKind::Outside,
        );

        let Some(base) = &self.preview_base else {
            return;
        };
        let (bw, bh) = base.dimensions();
        let angle = self.framing.rotation;
        let (cx, cy, cw, ch) = processing::crop_rect(bw, bh, &self.framing);
        let rect = response.rect;
        let screen_per_pixel = rect.width() / cw as f32;
        // Coordonate normalizate ale sursei → punct pe ecran.
        let to_screen = |(x, y): (f32, f32)| {
            let (rx, ry) =
                processing::source_to_rotated(x * bw as f32, y * bh as f32, bw, bh, angle);
            egui::pos2(
                rect.left() + (rx - cx as f32) * screen_per_pixel,
                rect.top() + (ry - cy as f32) * screen_per_pixel,
            )
        };

        if self.show_face && !self.show_original {
            let painter = ui.painter().with_clip_rect(rect);
            let stroke = Stroke::new(2.0_f32, Color32::from_rgb(255, 196, 0));
            for face in &self.faces {
                let corners = [
                    (face.x, face.y),
                    (face.x + face.w, face.y),
                    (face.x + face.w, face.y + face.h),
                    (face.x, face.y + face.h),
                ]
                .map(to_screen);
                for i in 0..4 {
                    painter.line_segment([corners[i], corners[(i + 1) % 4]], stroke);
                }
                for point in face.landmarks {
                    painter.circle_filled(to_screen(point), 3.0, Color32::from_rgb(255, 196, 0));
                }
            }
        }

        if !(self.retouch_mode && !self.show_original) {
            return;
        }
        let k = processing::rotation_scale(bw, bh, angle);
        let radius_norm = self.brush_size / 100.0;
        let radius_screen = radius_norm * bw.max(bh) as f32 * k * screen_per_pixel;

        if let Some(pos) = response.hover_pos() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
            ui.painter()
                .circle_stroke(pos, radius_screen, Stroke::new(2.0_f32, Color32::WHITE));
            ui.painter().circle_stroke(
                pos,
                radius_screen + 1.5,
                Stroke::new(1.0_f32, Color32::BLACK),
            );
        }
        if response.clicked()
            && let Some(pos) = response.interact_pointer_pos()
        {
            let u = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
            let v = ((pos.y - rect.top()) / rect.height()).clamp(0.0, 1.0);
            let rx = cx as f32 + u * cw as f32;
            let ry = cy as f32 + v * ch as f32;
            let (px, py) = processing::rotated_to_source(rx, ry, bw, bh, angle);
            self.spots.push(Spot {
                x: px / bw as f32,
                y: py / bh as f32,
                radius: radius_norm,
            });
            self.dirty = true;
        }
    }
}

// ---------------------------------------------------------------------------
// Fire de lucru
// ---------------------------------------------------------------------------

fn start_inference_worker(
    job_rx: mpsc::Receiver<InferenceJob>,
    result_tx: mpsc::Sender<InferenceResult>,
    ctx: Context,
) {
    std::thread::spawn(move || {
        let model = ai::load_model();
        while let Ok(job) = job_rx.recv() {
            let mask = match &model {
                Ok(model) => ai::infer_foreground_mask(model, &job.source),
                Err(error) => Err(error.clone()),
            };
            if result_tx
                .send(InferenceResult {
                    generation: job.generation,
                    mask,
                })
                .is_err()
            {
                break;
            }
            ctx.request_repaint();
        }
    });
}

fn start_face_worker(
    job_rx: mpsc::Receiver<FaceJob>,
    result_tx: mpsc::Sender<FaceResult>,
    ctx: Context,
) {
    std::thread::spawn(move || {
        let model = face::load_model();
        while let Ok(mut job) = job_rx.recv() {
            while let Ok(newer) = job_rx.try_recv() {
                job = newer;
            }
            let faces = match &model {
                Ok(model) => face::detect_faces(model, &job.image),
                Err(error) => Err(error.clone()),
            };
            if result_tx
                .send(FaceResult {
                    generation: job.generation,
                    faces,
                })
                .is_err()
            {
                break;
            }
            ctx.request_repaint();
        }
    });
}

/// Randează previzualizările pe un fir separat; dacă se adună mai multe
/// cereri (slider tras rapid), o procesează doar pe cea mai recentă.
fn start_render_worker(
    job_rx: mpsc::Receiver<RenderJob>,
    result_tx: mpsc::Sender<RenderResult>,
    ctx: Context,
) {
    std::thread::spawn(move || {
        while let Ok(mut job) = job_rx.recv() {
            while let Ok(newer) = job_rx.try_recv() {
                job = newer;
            }
            let rendered = processing::render(
                &job.base,
                job.mask.as_deref(),
                job.background_image.as_deref(),
                &job.settings,
                false,
            );
            let image = checkerboard_color_image(&rendered);
            if result_tx.send(RenderResult { id: job.id, image }).is_err() {
                break;
            }
            ctx.request_repaint();
        }
    });
}

// ---------------------------------------------------------------------------
// Utilitare imagine
// ---------------------------------------------------------------------------

fn open_oriented(path: &Path) -> image::ImageResult<DynamicImage> {
    let mut decoder = ImageReader::open(path)?
        .with_guessed_format()?
        .into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut image = DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    Ok(image)
}

fn downscale(image: &RgbImage, long_edge: u32) -> RgbImage {
    let (w, h) = image.dimensions();
    let long = w.max(h);
    if long <= long_edge {
        return image.clone();
    }
    let scale = long_edge as f32 / long as f32;
    let nw = ((w as f32 * scale).round() as u32).max(1);
    let nh = ((h as f32 * scale).round() as u32).max(1);
    image::imageops::resize(image, nw, nh, image::imageops::FilterType::Triangle)
}

/// Pentru previzualizare: zonele transparente sunt afișate pe tablă de șah.
fn checkerboard_color_image(image: &RgbaImage) -> ColorImage {
    let size = [image.width() as usize, image.height() as usize];
    let cell = (image.width().max(image.height()) / 60).max(6);
    let mut pixels = Vec::with_capacity(size[0] * size[1]);
    for (x, y, p) in image.enumerate_pixels() {
        let a = p[3] as u32;
        if a == 255 {
            pixels.push(Color32::from_rgb(p[0], p[1], p[2]));
            continue;
        }
        let check = if ((x / cell) + (y / cell)).is_multiple_of(2) {
            255
        } else {
            204
        };
        let mix = |c: u8| ((c as u32 * a + check * (255 - a) + 127) / 255) as u8;
        pixels.push(Color32::from_rgb(mix(p[0]), mix(p[1]), mix(p[2])));
    }
    ColorImage::new(size, pixels)
}

fn flatten_on_white(image: &RgbaImage) -> RgbImage {
    RgbImage::from_fn(image.width(), image.height(), |x, y| {
        let p = image.get_pixel(x, y);
        let a = p[3] as u32;
        let mix = |c: u8| ((c as u32 * a + 255 * (255 - a) + 127) / 255) as u8;
        image::Rgb([mix(p[0]), mix(p[1]), mix(p[2])])
    })
}

fn save_jpeg_300_dpi(image: &RgbImage, path: &Path) -> Result<(), String> {
    if image.width() > u16::MAX as u32 || image.height() > u16::MAX as u32 {
        return Err("imaginea este prea mare pentru JPEG".to_owned());
    }
    let file = std::fs::File::create(path).map_err(|error| error.to_string())?;
    let mut encoder = Encoder::new(file, 95);
    encoder.set_density(Density::Inch { x: DPI, y: DPI });
    encoder
        .encode(
            image.as_raw(),
            image.width() as u16,
            image.height() as u16,
            ColorType::Rgb,
        )
        .map_err(|error| error.to_string())
}

// ---------------------------------------------------------------------------
// Printare (Windows)
// ---------------------------------------------------------------------------

/// Afișează dialogul nativ de printare (.NET WinForms). Cu `WidthCm`/`HeightCm`
/// imaginea se tipărește la mărime reală, centrată; altfel se încadrează în
/// pagină păstrând proporțiile. Orientarea paginii urmează orientarea imaginii.
const PRINT_SCRIPT: &str = r#"param(
    [Parameter(Mandatory = $true)][string]$ImagePath,
    [double]$WidthCm = 0,
    [double]$HeightCm = 0
)
$ErrorActionPreference = 'Stop'
try {
    Add-Type -AssemblyName System.Drawing
    Add-Type -AssemblyName System.Windows.Forms
    $script:printImage = [System.Drawing.Image]::FromFile($ImagePath)
    $script:widthCm = $WidthCm
    $script:heightCm = $HeightCm
    $doc = New-Object System.Drawing.Printing.PrintDocument
    $doc.DocumentName = 'Foto Cleaner'
    if ($WidthCm -le 0) {
        $doc.DefaultPageSettings.Landscape = ($script:printImage.Width -gt $script:printImage.Height)
    }
    try { $doc.DefaultPageSettings.Color = $true } catch { }
    $handler = {
        param($sender, $e)
        $g = $e.Graphics
        $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $page = $e.PageBounds
        $hx = $e.PageSettings.HardMarginX
        $hy = $e.PageSettings.HardMarginY
        if ($script:widthCm -gt 0) {
            $w = $script:widthCm / 2.54 * 100.0
            $h = $script:heightCm / 2.54 * 100.0
        } else {
            $areaW = $page.Width - 50.0
            $areaH = $page.Height - 50.0
            $scale = [Math]::Min($areaW / $script:printImage.Width, $areaH / $script:printImage.Height)
            $w = $script:printImage.Width * $scale
            $h = $script:printImage.Height * $scale
        }
        $x = ($page.Width - $w) / 2.0 - $hx
        $y = ($page.Height - $h) / 2.0 - $hy
        $g.DrawImage($script:printImage, [single]$x, [single]$y, [single]$w, [single]$h)
        $e.HasMorePages = $false
    }
    $doc.add_PrintPage($handler)
    $dlg = New-Object System.Windows.Forms.PrintDialog
    $dlg.UseEXDialog = $true
    $dlg.Document = $doc
    $result = $dlg.ShowDialog()
    if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
        $doc.Print()
        Write-Output 'PRINT_OK'
    } else {
        Write-Output 'PRINT_CANCELLED'
    }
} catch {
    Write-Output ('PRINT_ERROR: ' + $_.Exception.Message)
} finally {
    if ($script:printImage) { $script:printImage.Dispose() }
    if ($doc) { $doc.Dispose() }
    if ($dlg) { $dlg.Dispose() }
}
"#;

/// Ascunde fereastra de consolă a procesului PowerShell (Windows).
fn hide_console_window(command: &mut std::process::Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    #[cfg(not(windows))]
    {
        let _ = command;
    }
}

fn run_print_dialog(
    script_path: &Path,
    image_path: &Path,
    real_size: Option<(f32, f32)>,
) -> String {
    let mut command = std::process::Command::new("powershell");
    command.args([
        "-NoProfile",
        "-STA",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        &script_path.to_string_lossy(),
        "-ImagePath",
        &image_path.to_string_lossy(),
    ]);
    if let Some((w, h)) = real_size {
        command.args(["-WidthCm", &w.to_string(), "-HeightCm", &h.to_string()]);
    }
    hide_console_window(&mut command);

    match command.output() {
        Err(error) => open_with_default_viewer(
            image_path,
            &format!("dialogul de printare nu a putut porni ({error})"),
        ),
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if stdout.lines().any(|line| line.trim() == "PRINT_OK") {
                format!(
                    "Imaginea a fost trimisă la imprimantă. Fișierul rămâne salvat la: {}",
                    image_path.display()
                )
            } else if stdout.lines().any(|line| line.trim() == "PRINT_CANCELLED") {
                "Printare anulată.".to_owned()
            } else if let Some(error) = stdout
                .lines()
                .find_map(|line| line.strip_prefix("PRINT_ERROR:"))
            {
                open_with_default_viewer(
                    image_path,
                    &format!("dialogul a eșuat ({})", error.trim()),
                )
            } else if output.status.success() {
                "Dialogul de printare a fost închis.".to_owned()
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                open_with_default_viewer(
                    image_path,
                    &format!(
                        "dialogul a eșuat (cod {:?}): {}",
                        output.status.code(),
                        stderr.trim()
                    ),
                )
            }
        }
    }
}

/// Soluție de rezervă: deschide imaginea în aplicația implicită din Windows.
fn open_with_default_viewer(image_path: &Path, reason: &str) -> String {
    match std::process::Command::new("explorer")
        .arg(image_path)
        .spawn()
    {
        Ok(_) => format!(
            "Nu am putut deschide direct dialogul de printare, {reason}. Am deschis imaginea \
             în aplicația implicită: {}. De acolo folosește Print (Ctrl+P).",
            image_path.display()
        ),
        Err(error) => format!(
            "Printarea directă nu a funcționat ({reason}; nici imaginea nu s-a putut deschide: \
             {error}). Fișierul este salvat la: {}.",
            image_path.display()
        ),
    }
}

fn configure_style(ctx: &Context) {
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = Vec2::new(9.0, 8.0);
    style.spacing.slider_width = 150.0;
    style.visuals.panel_fill = Color32::from_rgb(248, 248, 246);
    style.visuals.window_fill = Color32::from_rgb(248, 248, 246);
    style.visuals.weak_text_color = Some(Color32::BLACK);
    style
        .text_styles
        .insert(TextStyle::Heading, FontId::proportional(21.0));
    style
        .text_styles
        .insert(TextStyle::Body, FontId::proportional(15.0));
    style
        .text_styles
        .insert(TextStyle::Button, FontId::proportional(15.0));
    style
        .text_styles
        .insert(TextStyle::Small, FontId::proportional(13.0));
    ctx.set_style(style);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transparent_pixels_are_flattened_on_white() {
        let mut image = RgbaImage::from_pixel(2, 1, image::Rgba([10, 20, 30, 255]));
        image.put_pixel(1, 0, image::Rgba([0, 0, 0, 0]));
        let flat = flatten_on_white(&image);
        assert_eq!(flat.get_pixel(0, 0).0, [10, 20, 30]);
        assert_eq!(flat.get_pixel(1, 0).0, [255, 255, 255]);
    }

    #[test]
    fn downscale_keeps_aspect_ratio() {
        let image = RgbImage::new(4000, 3000);
        assert_eq!(downscale(&image, 1400).dimensions(), (1400, 1050));
        assert_eq!(
            downscale(&RgbImage::new(800, 600), 1400).dimensions(),
            (800, 600)
        );
    }

    #[cfg(windows)]
    #[test]
    fn hidden_powershell_still_captures_output() {
        let mut command = std::process::Command::new("powershell");
        command.args(["-NoProfile", "-Command", "Write-Output PRINT_OK"]);
        super::hide_console_window(&mut command);
        let output = command.output().expect("powershell should run");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.lines().any(|line| line.trim() == "PRINT_OK"));
    }
}
