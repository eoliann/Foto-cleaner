//! Detectarea feței cu modelul YuNet (OpenCV Zoo, licență MIT), rulat local
//! prin RTen. Pe lângă dreptunghiul feței, modelul găsește 5 repere: ochii,
//! vârful nasului și colțurile gurii.

use image::{RgbImage, imageops::FilterType};
use rten::{Model, ValueView};

pub const YUNET_MODEL: &[u8] = include_bytes!("../assets/yunet.onnx");
const INPUT_SIZE: usize = 640;
const SCORE_THRESHOLD: f32 = 0.6;
const NMS_THRESHOLD: f32 = 0.3;

/// O față detectată, în coordonate normalizate (0..1) față de imagine.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Face {
    pub score: f32,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    /// Ochiul drept și ochiul stâng al persoanei (stânga/dreapta imaginii
    /// pentru o față dreaptă), vârful nasului, colțurile gurii.
    pub landmarks: [(f32, f32); 5],
}

impl Face {
    pub fn eye_left(&self) -> (f32, f32) {
        self.landmarks[0]
    }

    pub fn eye_right(&self) -> (f32, f32) {
        self.landmarks[1]
    }

    pub fn eyes_center(&self) -> (f32, f32) {
        let (a, b) = (self.eye_left(), self.eye_right());
        ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0)
    }

    pub fn mouth_center(&self) -> (f32, f32) {
        let (a, b) = (self.landmarks[3], self.landmarks[4]);
        ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0)
    }

    /// Unghiul liniei ochilor, în grade, pentru o imagine de dimensiunile date
    /// (coordonatele sunt normalizate, deci contează raportul laturilor).
    pub fn eye_angle_degrees(&self, width: u32, height: u32) -> f32 {
        let (a, b) = (self.eye_left(), self.eye_right());
        let dx = (b.0 - a.0) * width as f32;
        let dy = (b.1 - a.1) * height as f32;
        dy.atan2(dx).to_degrees()
    }

    /// Aceeași față după rotirea imaginii cu 90°.
    pub fn rotated90(&self, clockwise: bool) -> Self {
        let map = |(x, y): (f32, f32)| {
            if clockwise {
                (1.0 - y, x)
            } else {
                (y, 1.0 - x)
            }
        };
        let (ax, ay) = map((self.x, self.y));
        let (bx, by) = map((self.x + self.w, self.y + self.h));
        // Reperele își păstrează semnificația (ochiul drept al persoanei etc.),
        // astfel încât unghiul ochilor rămâne corect după rotire.
        let landmarks = self.landmarks.map(map);
        Self {
            score: self.score,
            x: ax.min(bx),
            y: ay.min(by),
            w: (ax - bx).abs(),
            h: (ay - by).abs(),
            landmarks,
        }
    }
}

pub fn load_model() -> Result<Model, String> {
    Model::load_static_slice(YUNET_MODEL)
        .map_err(|error| format!("Modelul YuNet nu poate fi încărcat: {error}"))
}

/// Detectează fețele. Rulează la două scări (fețe mici și fețe mari, de tip
/// portret) și combină rezultatele. Fețele sunt sortate descrescător după
/// mărime × încredere, deci prima este fața principală.
pub fn detect_faces(model: &Model, image: &RgbImage) -> Result<Vec<Face>, String> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Ok(Vec::new());
    }
    let mut candidates = Vec::new();
    for long_edge in [640.0_f32, 320.0] {
        candidates.extend(detect_at_scale(model, image, long_edge)?);
    }
    let mut faces = non_maximum_suppression(candidates);
    let (wf, hf) = (width as f32, height as f32);
    for face in &mut faces {
        face.x /= wf;
        face.w /= wf;
        face.y /= hf;
        face.h /= hf;
        for point in &mut face.landmarks {
            point.0 /= wf;
            point.1 /= hf;
        }
    }
    faces.sort_by(|a, b| (b.w * b.h * b.score).total_cmp(&(a.w * a.h * a.score)));
    Ok(faces)
}

/// Rulează modelul pe imaginea micșorată la `long_edge` și plasată în colțul
/// stâng-sus al unei pânze 640x640. Returnează fețe în pixeli sursă.
fn detect_at_scale(model: &Model, image: &RgbImage, long_edge: f32) -> Result<Vec<Face>, String> {
    let (width, height) = image.dimensions();
    let scale = long_edge / width.max(height) as f32;
    let rw = ((width as f32 * scale).round() as u32).clamp(1, INPUT_SIZE as u32);
    let rh = ((height as f32 * scale).round() as u32).clamp(1, INPUT_SIZE as u32);
    let resized = image::imageops::resize(image, rw, rh, FilterType::Triangle);

    // Intrarea YuNet: BGR, valori 0..255, NCHW.
    let area = INPUT_SIZE * INPUT_SIZE;
    let mut input = vec![0.0_f32; 3 * area];
    for (x, y, p) in resized.enumerate_pixels() {
        let i = y as usize * INPUT_SIZE + x as usize;
        input[i] = p[2] as f32;
        input[area + i] = p[1] as f32;
        input[2 * area + i] = p[0] as f32;
    }
    let input = ValueView::from_shape([1, 3, INPUT_SIZE, INPUT_SIZE], &input)
        .map_err(|error| format!("Intrare invalidă pentru detectarea feței: {error}"))?;
    let input_id = *model
        .input_ids()
        .first()
        .ok_or("Modelul YuNet nu are intrări.")?;

    let mut names = Vec::new();
    for stride in [8, 16, 32] {
        for kind in ["cls", "obj", "bbox", "kps"] {
            names.push(format!("{kind}_{stride}"));
        }
    }
    let ids = names
        .iter()
        .map(|name| model.node_id(name))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("Model YuNet incompatibil: {error}"))?;
    let outputs = model
        .run(vec![(input_id, input.into())], &ids, None)
        .map_err(|error| format!("Detectarea feței a eșuat: {error}"))?;
    let mut tensors = Vec::with_capacity(outputs.len());
    for output in outputs {
        let (_, values) = output
            .into_shape_vec::<f32, 3>()
            .map_err(|error| format!("Rezultat invalid la detectarea feței: {error}"))?;
        tensors.push(values);
    }

    let mut faces = Vec::new();
    for (level, stride) in [8_usize, 16, 32].into_iter().enumerate() {
        let [cls, obj, bbox, kps] = [0, 1, 2, 3].map(|k| &tensors[level * 4 + k]);
        let cols = INPUT_SIZE / stride;
        let s = stride as f32;
        for i in 0..cols * cols {
            let score = (cls[i].clamp(0.0, 1.0) * obj[i].clamp(0.0, 1.0)).sqrt();
            if score < SCORE_THRESHOLD {
                continue;
            }
            let (row, col) = ((i / cols) as f32, (i % cols) as f32);
            let cx = (col + bbox[i * 4]) * s;
            let cy = (row + bbox[i * 4 + 1]) * s;
            let w = bbox[i * 4 + 2].exp() * s;
            let h = bbox[i * 4 + 3].exp() * s;
            let landmarks = std::array::from_fn(|n| {
                (
                    (kps[i * 10 + 2 * n] + col) * s / scale,
                    (kps[i * 10 + 2 * n + 1] + row) * s / scale,
                )
            });
            faces.push(Face {
                score,
                x: (cx - w / 2.0) / scale,
                y: (cy - h / 2.0) / scale,
                w: w / scale,
                h: h / scale,
                landmarks,
            });
        }
    }
    Ok(faces)
}

fn iou(a: &Face, b: &Face) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.w).min(b.x + b.w);
    let y2 = (a.y + a.h).min(b.y + b.h);
    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let union = a.w * a.h + b.w * b.h - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

fn non_maximum_suppression(mut faces: Vec<Face>) -> Vec<Face> {
    faces.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Face> = Vec::new();
    for face in faces {
        if kept.iter().all(|k| iou(k, &face) < NMS_THRESHOLD) {
            kept.push(face);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_face_model_runs_on_blank_image() {
        let model = load_model().expect("YuNet should load");
        let image = RgbImage::from_pixel(300, 400, image::Rgb([128, 128, 128]));
        let faces = detect_faces(&model, &image).expect("YuNet inference should succeed");
        assert!(faces.is_empty());
    }

    #[test]
    fn rotation_round_trip_restores_face() {
        let face = Face {
            score: 0.9,
            x: 0.3,
            y: 0.2,
            w: 0.4,
            h: 0.5,
            landmarks: [(0.4, 0.4), (0.6, 0.4), (0.5, 0.5), (0.42, 0.6), (0.58, 0.6)],
        };
        let rotated = face.rotated90(true);
        assert!((rotated.w - 0.5).abs() < 1e-6 && (rotated.h - 0.4).abs() < 1e-6);
        // Fața rotită are linia ochilor verticală.
        assert!((rotated.eye_angle_degrees(100, 100).abs() - 90.0).abs() < 1e-3);
        let back = rotated.rotated90(false);
        assert!((back.x - face.x).abs() < 1e-6 && (back.y - face.y).abs() < 1e-6);
        assert!(back.eye_angle_degrees(100, 100).abs() < 1e-3);
    }

    #[test]
    fn overlapping_detections_are_merged() {
        let base = Face {
            score: 0.9,
            x: 10.0,
            y: 10.0,
            w: 100.0,
            h: 120.0,
            landmarks: [(0.0, 0.0); 5],
        };
        let shifted = Face {
            score: 0.8,
            x: 14.0,
            ..base
        };
        let far = Face {
            score: 0.7,
            x: 400.0,
            ..base
        };
        assert_eq!(non_maximum_suppression(vec![shifted, base, far]).len(), 2);
    }
}
