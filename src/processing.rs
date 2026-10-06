//! Procesarea imaginii: retuș, reducere zgomot, corecții de lumină și culoare,
//! claritate, compunerea fundalului și încadrarea finală.
//!
//! Toate razele filtrelor sunt exprimate relativ la dimensiunea imaginii, astfel
//! încât previzualizarea (micșorată) și exportul (rezoluție completă) arată la fel.

// Buclele cu index pe canale (0..3) sunt mai lizibile pentru calculele pe pixeli.
#![allow(clippy::needless_range_loop)]

use image::{GrayImage, RgbImage, RgbaImage, imageops::FilterType};

use crate::face::Face;

/// Dimensiunea fotografiei de act 3x4 cm la 300 DPI.
pub const ACT_WIDTH: u32 = 354;
pub const ACT_HEIGHT: u32 = 472;

// ---------------------------------------------------------------------------
// Setări
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OutputFormat {
    Original,
    Square,
    Portrait34,
    Landscape43,
    Portrait23,
    Wide169,
    Act3x4,
}

impl OutputFormat {
    pub const ALL: [Self; 7] = [
        Self::Original,
        Self::Act3x4,
        Self::Square,
        Self::Portrait34,
        Self::Landscape43,
        Self::Portrait23,
        Self::Wide169,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Original => "Original (fără decupare)",
            Self::Act3x4 => "Foto act 3x4 cm (300 DPI)",
            Self::Square => "Pătrat 1:1",
            Self::Portrait34 => "Portret 3:4",
            Self::Landscape43 => "Peisaj 4:3",
            Self::Portrait23 => "Portret 2:3",
            Self::Wide169 => "Panoramic 16:9",
        }
    }

    /// Raportul lățime:înălțime al decupării; `None` = imaginea întreagă.
    pub fn aspect(self) -> Option<(u32, u32)> {
        match self {
            Self::Original => None,
            Self::Act3x4 => Some((ACT_WIDTH, ACT_HEIGHT)),
            Self::Square => Some((1, 1)),
            Self::Portrait34 => Some((3, 4)),
            Self::Landscape43 => Some((4, 3)),
            Self::Portrait23 => Some((2, 3)),
            Self::Wide169 => Some((16, 9)),
        }
    }

    /// Dimensiunea fixă a exportului, dacă formatul o impune.
    pub fn fixed_size(self) -> Option<(u32, u32)> {
        match self {
            Self::Act3x4 => Some((ACT_WIDTH, ACT_HEIGHT)),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Framing {
    pub format: OutputFormat,
    pub zoom: f32,
    pub offset_x: f32,
    pub offset_y: f32,
    /// Îndreptare (rotire fină), în grade, aplicată înainte de decupare.
    pub rotation: f32,
}

impl Default for Framing {
    fn default() -> Self {
        Self {
            format: OutputFormat::Original,
            zoom: 1.0,
            offset_x: 0.0,
            offset_y: 0.0,
            rotation: 0.0,
        }
    }
}

/// Toate valorile sunt în intervalul -100..100 (sau 0..100 pentru filtre).
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Adjustments {
    pub exposure: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub saturation: f32,
    pub vibrance: f32,
    pub temperature: f32,
    pub tint: f32,
    pub clarity: f32,
    pub sharpness: f32,
    pub denoise: f32,
    pub skin_smoothing: f32,
}

/// O pată de retușat, în coordonate normalizate față de imaginea sursă.
/// `radius` este relativ la latura lungă a imaginii.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Spot {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BackgroundMode {
    Original,
    Transparent,
    Color([u8; 3]),
    /// Fundal estompat, intensitate 0..100.
    Blur(f32),
    /// Fundal înlocuit cu o imagine furnizată separat.
    Image,
}

#[derive(Clone, PartialEq, Debug)]
pub struct RenderSettings {
    pub adjustments: Adjustments,
    pub spots: Vec<Spot>,
    pub framing: Framing,
    pub background: BackgroundMode,
    /// 0..100: cât de „tare” este marginea dintre persoană și fundal.
    pub edge_firmness: f32,
    /// Afișează doar încadrarea, fără nicio corecție (comparare cu originalul).
    pub bypass: bool,
    /// Fețele detectate, în coordonate normalizate față de imaginea sursă.
    pub faces: Vec<Face>,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            adjustments: Adjustments::default(),
            spots: Vec::new(),
            framing: Framing::default(),
            background: BackgroundMode::Original,
            edge_firmness: 30.0,
            bypass: false,
            faces: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Randează imaginea finală. `mask` (dacă există) trebuie să aibă dimensiunile
/// lui `base`. Cu `final_size = true`, formatele cu dimensiune fixă sunt
/// redimensionate exact (de ex. 354x472 pentru foto act).
pub fn render(
    base: &RgbImage,
    mask: Option<&GrayImage>,
    background_image: Option<&RgbImage>,
    settings: &RenderSettings,
    final_size: bool,
) -> RgbaImage {
    let (width, height) = base.dimensions();
    let resized_mask;
    let mask = match mask {
        Some(mask) if mask.dimensions() != (width, height) => {
            resized_mask = image::imageops::resize(mask, width, height, FilterType::Triangle);
            Some(&resized_mask)
        }
        other => other,
    };

    let healed;
    let working = if !settings.bypass && !settings.spots.is_empty() {
        let mut copy = base.clone();
        heal_spots(&mut copy, &settings.spots);
        healed = copy;
        &healed
    } else {
        base
    };

    let angle = settings.framing.rotation;
    let rotated;
    let rotated_mask;
    let (working, mask) = if angle.abs() > 0.01 {
        rotated = rotate_fine(working, angle);
        rotated_mask = mask.map(|mask| rotate_fine(mask, angle));
        (&rotated, rotated_mask.as_ref())
    } else {
        (working, mask)
    };

    let (x, y, crop_w, crop_h) = crop_rect(width, height, &settings.framing);
    let cropped = image::imageops::crop_imm(working, x, y, crop_w, crop_h).to_image();
    let cropped_mask =
        mask.map(|mask| image::imageops::crop_imm(mask, x, y, crop_w, crop_h).to_image());

    let output = if settings.bypass {
        rgb_to_rgba(&cropped)
    } else {
        // Zonele fețelor în coordonatele imaginii decupate.
        let faces: Vec<FaceRegion> = settings
            .faces
            .iter()
            .map(|face| {
                let (cx, cy) = source_to_rotated(
                    (face.x + face.w / 2.0) * width as f32,
                    (face.y + face.h / 2.0) * height as f32,
                    width,
                    height,
                    angle,
                );
                let k = rotation_scale(width, height, angle);
                FaceRegion {
                    cx: cx - x as f32,
                    cy: cy - y as f32,
                    w: face.w * width as f32 * k,
                    h: face.h * height as f32 * k,
                }
            })
            .collect();
        process(
            &cropped,
            cropped_mask.as_ref(),
            background_image,
            settings,
            &faces,
        )
    };

    match (final_size, settings.framing.format.fixed_size()) {
        (true, Some((out_w, out_h))) if output.dimensions() != (out_w, out_h) => {
            image::imageops::resize(&output, out_w, out_h, FilterType::Lanczos3)
        }
        _ => output,
    }
}

fn process(
    image: &RgbImage,
    mask: Option<&GrayImage>,
    background_image: Option<&RgbImage>,
    settings: &RenderSettings,
    faces: &[FaceRegion],
) -> RgbaImage {
    let adj = &settings.adjustments;
    let mut img = FloatImage::from_rgb(image);
    let scale = (img.w.max(img.h) as f32 / 1000.0).max(0.2);
    let person = mask.map(|mask| {
        let k = 1.0 + settings.edge_firmness.clamp(0.0, 100.0) / 100.0 * 5.0;
        mask.pixels()
            .map(|p| ((p[0] as f32 / 255.0 - 0.5) * k + 0.5).clamp(0.0, 1.0))
            .collect::<Vec<f32>>()
    });

    if adj.denoise > 0.0 {
        denoise(&mut img, adj.denoise, scale);
    }
    if adj.skin_smoothing > 0.0 {
        smooth_skin(
            &mut img,
            person.as_deref(),
            faces,
            adj.skin_smoothing,
            scale,
        );
    }
    apply_tone_and_color(&mut img, adj);
    if adj.clarity != 0.0 {
        apply_clarity(&mut img, adj.clarity, scale);
    }
    if adj.sharpness > 0.0 {
        sharpen(&mut img, adj.sharpness, scale);
    }

    match (settings.background, person) {
        (BackgroundMode::Original, _) | (_, None) => img.to_rgba(None),
        (mode, Some(alpha)) => composite(img, &alpha, mode, background_image, scale),
    }
}

// ---------------------------------------------------------------------------
// Încadrare
// ---------------------------------------------------------------------------

/// Dreptunghiul de decupare (x, y, lățime, înălțime) în pixeli sursă.
pub fn crop_rect(width: u32, height: u32, framing: &Framing) -> (u32, u32, u32, u32) {
    let Some((aspect_w, aspect_h)) = framing.format.aspect() else {
        return (0, 0, width, height);
    };
    let target_ratio = aspect_w as f64 / aspect_h as f64;
    let source_ratio = width as f64 / height as f64;
    let (base_w, base_h) = if source_ratio > target_ratio {
        ((height as f64 * target_ratio).round() as u32, height)
    } else {
        (width, (width as f64 / target_ratio).round() as u32)
    };
    let zoom = framing.zoom.max(1.0);
    let crop_w = ((base_w as f32 / zoom).round() as u32).clamp(1, width);
    let crop_h = ((base_h as f32 / zoom).round() as u32).clamp(1, height);
    let max_x = width - crop_w;
    let max_y = height - crop_h;
    let x = (((framing.offset_x.clamp(-1.0, 1.0) + 1.0) * 0.5) * max_x as f32).round() as u32;
    let y = (((framing.offset_y.clamp(-1.0, 1.0) + 1.0) * 0.5) * max_y as f32).round() as u32;
    (x.min(max_x), y.min(max_y), crop_w, crop_h)
}

// ---------------------------------------------------------------------------
// Imagine în virgulă mobilă (planară, 0..1)
// ---------------------------------------------------------------------------

pub struct FloatImage {
    pub w: usize,
    pub h: usize,
    pub c: [Vec<f32>; 3],
}

impl FloatImage {
    pub fn from_rgb(image: &RgbImage) -> Self {
        let (w, h) = (image.width() as usize, image.height() as usize);
        let mut c = [vec![0.0; w * h], vec![0.0; w * h], vec![0.0; w * h]];
        for (i, p) in image.pixels().enumerate() {
            for ch in 0..3 {
                c[ch][i] = p[ch] as f32 / 255.0;
            }
        }
        Self { w, h, c }
    }

    fn luminance(&self) -> Vec<f32> {
        (0..self.w * self.h)
            .map(|i| luma(self.c[0][i], self.c[1][i], self.c[2][i]))
            .collect()
    }

    fn to_rgba(&self, alpha: Option<&[f32]>) -> RgbaImage {
        RgbaImage::from_fn(self.w as u32, self.h as u32, |x, y| {
            let i = y as usize * self.w + x as usize;
            let a = alpha.map_or(1.0, |a| a[i]);
            image::Rgba([
                to_u8(self.c[0][i]),
                to_u8(self.c[1][i]),
                to_u8(self.c[2][i]),
                to_u8(a),
            ])
        })
    }
}

fn to_u8(v: f32) -> u8 {
    (v * 255.0).round().clamp(0.0, 255.0) as u8
}

fn luma(r: f32, g: f32, b: f32) -> f32 {
    0.299 * r + 0.587 * g + 0.114 * b
}

fn rgb_to_rgba(image: &RgbImage) -> RgbaImage {
    RgbaImage::from_fn(image.width(), image.height(), |x, y| {
        let p = image.get_pixel(x, y);
        image::Rgba([p[0], p[1], p[2], 255])
    })
}

// ---------------------------------------------------------------------------
// Filtre de bază
// ---------------------------------------------------------------------------

/// Filtru „box” separabil, O(N) indiferent de rază, cu margini extinse.
pub fn box_blur(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    if r == 0 || w == 0 || h == 0 {
        return src.to_vec();
    }
    let mut tmp = vec![0.0; w * h];
    blur_pass(src, &mut tmp, h, w, w, 1, r);
    let mut out = vec![0.0; w * h];
    blur_pass(&tmp, &mut out, w, h, 1, w, r);
    out
}

fn blur_pass(
    src: &[f32],
    dst: &mut [f32],
    lines: usize,
    len: usize,
    line_stride: usize,
    pixel_stride: usize,
    r: usize,
) {
    let norm = 1.0 / (2 * r + 1) as f64;
    let last = len as isize - 1;
    for line in 0..lines {
        let base = line * line_stride;
        let at = |i: isize| src[base + i.clamp(0, last) as usize * pixel_stride] as f64;
        let mut sum = 0.0_f64;
        for i in -(r as isize)..=(r as isize) {
            sum += at(i);
        }
        for i in 0..len as isize {
            dst[base + i as usize * pixel_stride] = (sum * norm) as f32;
            sum += at(i + r as isize + 1) - at(i - r as isize);
        }
    }
}

/// Aproximare de blur gaussian prin două treceri box.
fn soft_blur(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let once = box_blur(src, w, h, r);
    box_blur(&once, w, h, r)
}

/// Filtru ghidat (He et al.) cu imaginea ca propriul ghid: netezește zonele
/// uniforme și păstrează marginile.
pub fn guided_self(p: &[f32], w: usize, h: usize, r: usize, eps: f32) -> Vec<f32> {
    let mean = box_blur(p, w, h, r);
    let squares: Vec<f32> = p.iter().map(|v| v * v).collect();
    let mean_sq = box_blur(&squares, w, h, r);
    let mut a = vec![0.0; p.len()];
    let mut b = vec![0.0; p.len()];
    for i in 0..p.len() {
        let var = (mean_sq[i] - mean[i] * mean[i]).max(0.0);
        a[i] = var / (var + eps);
        b[i] = mean[i] - a[i] * mean[i];
    }
    let mean_a = box_blur(&a, w, h, r);
    let mean_b = box_blur(&b, w, h, r);
    (0..p.len()).map(|i| mean_a[i] * p[i] + mean_b[i]).collect()
}

fn radius(base: f32, scale: f32, min: usize) -> usize {
    ((base * scale).round() as usize).max(min)
}

// ---------------------------------------------------------------------------
// Curățare: zgomot și piele
// ---------------------------------------------------------------------------

fn denoise(img: &mut FloatImage, amount: f32, scale: f32) {
    let s = (amount / 100.0).clamp(0.0, 1.0);
    let r = radius(1.0 + 2.0 * s, scale, 1);
    let eps = (0.012 + 0.05 * s).powi(2);
    let blend = (s * 1.6).min(1.0);
    let (w, h) = (img.w, img.h);
    for channel in &mut img.c {
        let smooth = guided_self(channel, w, h, r, eps);
        for (v, s) in channel.iter_mut().zip(smooth) {
            *v += (s - *v) * blend;
        }
    }
}

/// Probabilitatea (0..1) ca un pixel să fie piele, după culoare (YCbCr).
pub fn skin_likelihood(r: f32, g: f32, b: f32) -> f32 {
    let (r, g, b) = (r * 255.0, g * 255.0, b * 255.0);
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let cb = 128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b;
    let cr = 128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b;
    let band = |v: f32, lo: f32, hi: f32, soft: f32| {
        let up = ((v - (lo - soft)) / soft).clamp(0.0, 1.0);
        let down = (((hi + soft) - v) / soft).clamp(0.0, 1.0);
        up.min(down)
    };
    band(cb, 77.0, 127.0, 10.0) * band(cr, 133.0, 173.0, 10.0) * ((y - 35.0) / 25.0).clamp(0.0, 1.0)
}

/// Zona unei fețe în pixeli ai imaginii procesate (centru și dimensiuni).
#[derive(Clone, Copy, Debug)]
struct FaceRegion {
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
}

impl FaceRegion {
    /// 1 în interiorul feței (inclusiv frunte, bărbie, gât), cu tranziție lină spre 0.
    fn weight(&self, x: f32, y: f32) -> f32 {
        let dx = (x - self.cx) / (self.w * 0.75);
        let dy = (y - (self.cy + self.h * 0.12)) / (self.h * 0.85);
        let d = (dx * dx + dy * dy).sqrt();
        let t = ((d - 0.8) / 0.4).clamp(0.0, 1.0);
        1.0 - t * t * (3.0 - 2.0 * t)
    }
}

fn smooth_skin(
    img: &mut FloatImage,
    person: Option<&[f32]>,
    faces: &[FaceRegion],
    amount: f32,
    scale: f32,
) {
    let s = (amount / 100.0).clamp(0.0, 1.0);
    let (w, h) = (img.w, img.h);
    let mut skin: Vec<f32> = (0..w * h)
        .map(|i| {
            let likelihood = skin_likelihood(img.c[0][i], img.c[1][i], img.c[2][i]);
            // Cu fețe detectate, netezirea se aplică doar pe față; altfel pe
            // toată pielea persoanei.
            let region = if faces.is_empty() {
                1.0
            } else {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                faces.iter().map(|f| f.weight(x, y)).fold(0.0, f32::max)
            };
            likelihood * region * person.map_or(1.0, |p| p[i])
        })
        .collect();
    skin = soft_blur(&skin, w, h, radius(3.0, scale, 1));

    let r = radius(7.0, scale, 2);
    let eps = (0.025 + 0.045 * s).powi(2);
    let strength = 0.9 * s;
    for channel in &mut img.c {
        let smooth = guided_self(channel, w, h, r, eps);
        for i in 0..w * h {
            channel[i] += (smooth[i] - channel[i]) * skin[i] * strength;
        }
    }
}

// ---------------------------------------------------------------------------
// Lumină și culoare
// ---------------------------------------------------------------------------

/// Câștigurile pe canale pentru temperatură (-100 rece .. 100 cald) și nuanță
/// (-100 verde .. 100 magenta).
pub fn white_balance_gains(temperature: f32, tint: f32) -> [f32; 3] {
    let t = temperature / 100.0 * 0.25;
    let m = tint / 100.0 * 0.2;
    [1.0 + t, 1.0 - m, 1.0 - t]
}

fn apply_tone_and_color(img: &mut FloatImage, adj: &Adjustments) {
    let gains = white_balance_gains(adj.temperature, adj.tint);
    let exposure = 2.0_f32.powf(adj.exposure / 100.0 * 2.0);
    let shadows = adj.shadows / 100.0;
    let highlights = adj.highlights / 100.0;
    let gamma = 2.0_f32.powf(-adj.brightness / 100.0);
    let c = adj.contrast / 100.0;
    let contrast = if c >= 0.0 {
        1.0 + c * 1.2
    } else {
        1.0 + c * 0.8
    };
    let saturation = 1.0 + adj.saturation / 100.0;
    let vibrance = adj.vibrance / 100.0;

    let identity = *adj
        == Adjustments {
            clarity: adj.clarity,
            sharpness: adj.sharpness,
            denoise: adj.denoise,
            skin_smoothing: adj.skin_smoothing,
            ..Adjustments::default()
        };
    if identity {
        return;
    }

    for i in 0..img.w * img.h {
        let mut px = [
            img.c[0][i] * gains[0] * exposure,
            img.c[1][i] * gains[1] * exposure,
            img.c[2][i] * gains[2] * exposure,
        ];

        // Umbre și lumini: modifică luminanța și păstrează raportul culorilor.
        if shadows != 0.0 || highlights != 0.0 {
            let l = luma(px[0], px[1], px[2]).clamp(0.0, 1.0);
            let target = l
                + shadows * 1.2 * l * (1.0 - l).powi(3)
                + highlights * 1.2 * (1.0 - l) * l.powi(3);
            if l > 1e-4 {
                let ratio = target.max(0.0) / l;
                for v in &mut px {
                    *v *= ratio;
                }
            }
        }

        for v in &mut px {
            let mut x = v.clamp(0.0, 1.0);
            if gamma != 1.0 {
                x = x.powf(gamma);
            }
            x = 0.5 + (x - 0.5) * contrast;
            *v = x;
        }

        if saturation != 1.0 || vibrance != 0.0 {
            let l = luma(px[0], px[1], px[2]);
            let max = px[0].max(px[1]).max(px[2]);
            let min = px[0].min(px[1]).min(px[2]);
            let chroma = (max - min).clamp(0.0, 1.0);
            let factor = saturation * (1.0 + vibrance * (1.0 - chroma).powi(2));
            for v in &mut px {
                *v = l + (*v - l) * factor.max(0.0);
            }
        }

        for (ch, v) in px.into_iter().enumerate() {
            img.c[ch][i] = v.clamp(0.0, 1.0);
        }
    }
}

/// Contrast local (claritate) pe tonurile medii.
fn apply_clarity(img: &mut FloatImage, amount: f32, scale: f32) {
    let (w, h) = (img.w, img.h);
    let l = img.luminance();
    let blurred = soft_blur(&l, w, h, radius(18.0, scale, 3));
    let k = amount / 100.0 * 0.9;
    for i in 0..w * h {
        let midtone = 1.0 - (2.0 * l[i] - 1.0).powi(2);
        let delta = (l[i] - blurred[i]) * k * midtone;
        for ch in 0..3 {
            img.c[ch][i] = (img.c[ch][i] + delta).clamp(0.0, 1.0);
        }
    }
}

/// Mască de claritate (unsharp mask) pe luminanță, fără a amplifica zgomotul de culoare.
fn sharpen(img: &mut FloatImage, amount: f32, scale: f32) {
    let (w, h) = (img.w, img.h);
    let l = img.luminance();
    let blurred = soft_blur(&l, w, h, radius(1.0, scale, 1));
    let k = amount / 100.0 * 1.5;
    for i in 0..w * h {
        let detail = l[i] - blurred[i];
        // Prag mic: detaliile foarte fine (zgomot) sunt accentuate mai puțin.
        let detail = detail * (detail.abs() / 0.01).min(1.0);
        for ch in 0..3 {
            img.c[ch][i] = (img.c[ch][i] + detail * k).clamp(0.0, 1.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Fundal
// ---------------------------------------------------------------------------

/// Media culorilor dintr-o zonă, ponderată cu `weights` (blur „normalizat”).
fn weighted_blur(img: &FloatImage, weights: &[f32], r: usize) -> [Vec<f32>; 3] {
    let (w, h) = (img.w, img.h);
    let den = soft_blur(weights, w, h, r);
    let mut out: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (ch, slot) in out.iter_mut().enumerate() {
        let weighted: Vec<f32> = img.c[ch]
            .iter()
            .zip(weights)
            .map(|(v, wt)| v * wt)
            .collect();
        let num = soft_blur(&weighted, w, h, r);
        *slot = (0..w * h)
            .map(|i| {
                if den[i] > 1e-3 {
                    num[i] / den[i]
                } else {
                    img.c[ch][i]
                }
            })
            .collect();
    }
    out
}

fn composite(
    mut img: FloatImage,
    alpha: &[f32],
    mode: BackgroundMode,
    background_image: Option<&RgbImage>,
    scale: f32,
) -> RgbaImage {
    let (w, h) = (img.w, img.h);
    let inverse: Vec<f32> = alpha.iter().map(|a| 1.0 - a).collect();

    // Decontaminare: pe margini, culoarea pixelului conține și fundalul vechi.
    // Estimăm fundalul local și îl scădem: F = (I - (1 - a) * B) / a.
    let old_background = weighted_blur(&img, &inverse, radius(10.0, scale, 2));
    for i in 0..w * h {
        let a = alpha[i];
        if a > 0.02 && a < 0.98 {
            for ch in 0..3 {
                let fg = (img.c[ch][i] - (1.0 - a) * old_background[ch][i]) / a;
                img.c[ch][i] = fg.clamp(0.0, 1.0);
            }
        }
    }

    let backdrop: Option<[Vec<f32>; 3]> = match mode {
        BackgroundMode::Original | BackgroundMode::Transparent => None,
        BackgroundMode::Color(color) => Some(color.map(|v| vec![v as f32 / 255.0; w * h])),
        BackgroundMode::Blur(strength) => {
            let r = radius(6.0 + strength.clamp(0.0, 100.0) / 100.0 * 34.0, scale, 2);
            Some(weighted_blur(
                &img_with_background(&img, &old_background, alpha),
                &inverse,
                r,
            ))
        }
        BackgroundMode::Image => background_image.map(|bg| {
            let cover = cover_resize(bg, w as u32, h as u32);
            let f = FloatImage::from_rgb(&cover);
            f.c
        }),
    };

    match backdrop {
        None if mode == BackgroundMode::Transparent => img.to_rgba(Some(alpha)),
        None => img.to_rgba(None),
        Some(bg) => {
            for i in 0..w * h {
                let a = alpha[i];
                for ch in 0..3 {
                    img.c[ch][i] = img.c[ch][i] * a + bg[ch][i] * (1.0 - a);
                }
            }
            img.to_rgba(None)
        }
    }
}

/// Imaginea în care zona persoanei este înlocuită cu fundalul estimat,
/// ca estomparea fundalului să nu „sângereze” culorile persoanei.
fn img_with_background(img: &FloatImage, background: &[Vec<f32>; 3], alpha: &[f32]) -> FloatImage {
    let mut c: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (ch, slot) in c.iter_mut().enumerate() {
        *slot = (0..img.w * img.h)
            .map(|i| img.c[ch][i] * (1.0 - alpha[i]) + background[ch][i] * alpha[i])
            .collect();
    }
    FloatImage {
        w: img.w,
        h: img.h,
        c,
    }
}

/// Redimensionează imaginea ca să acopere complet (w, h), decupând centrat.
pub fn cover_resize(image: &RgbImage, w: u32, h: u32) -> RgbImage {
    let (iw, ih) = image.dimensions();
    let scale = (w as f32 / iw as f32).max(h as f32 / ih as f32);
    let rw = ((iw as f32 * scale).ceil() as u32).max(w);
    let rh = ((ih as f32 * scale).ceil() as u32).max(h);
    let resized = image::imageops::resize(image, rw, rh, FilterType::Triangle);
    image::imageops::crop_imm(&resized, (rw - w) / 2, (rh - h) / 2, w, h).to_image()
}

// ---------------------------------------------------------------------------
// Retuș pete (pensulă „healing”)
// ---------------------------------------------------------------------------

pub fn heal_spots(image: &mut RgbImage, spots: &[Spot]) {
    let (w, h) = image.dimensions();
    let long = w.max(h) as f32;
    for spot in spots {
        heal_one(
            image,
            spot.x * w as f32,
            spot.y * h as f32,
            (spot.radius * long).max(1.5),
        );
    }
}

fn sample(image: &RgbImage, x: f32, y: f32) -> [f32; 3] {
    let (w, h) = image.dimensions();
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let p00 = image.get_pixel(x0, y0);
    let p10 = image.get_pixel(x1, y0);
    let p01 = image.get_pixel(x0, y1);
    let p11 = image.get_pixel(x1, y1);
    let mut out = [0.0; 3];
    for ch in 0..3 {
        let top = p00[ch] as f32 * (1.0 - fx) + p10[ch] as f32 * fx;
        let bottom = p01[ch] as f32 * (1.0 - fx) + p11[ch] as f32 * fx;
        out[ch] = top * (1.0 - fy) + bottom * fy;
    }
    out
}

fn color_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    (0..3).map(|ch| (a[ch] - b[ch]).powi(2)).sum()
}

/// Umple discul (cx, cy, r) interpolând netezit inelul din jur și adaugă
/// textura (detaliile fine) unei zone donatoare asemănătoare din apropiere.
fn heal_one(image: &mut RgbImage, cx: f32, cy: f32, r: f32) {
    let (w, h) = image.dimensions();
    let ring_r = r * 1.15 + 1.0;
    const RING: usize = 48;
    let ring_offsets: Vec<(f32, f32)> = (0..RING)
        .map(|i| {
            let angle = i as f32 / RING as f32 * std::f32::consts::TAU;
            (ring_r * angle.cos(), ring_r * angle.sin())
        })
        .collect();
    let ring_at = |cx: f32, cy: f32| -> Vec<[f32; 3]> {
        ring_offsets
            .iter()
            .map(|(dx, dy)| sample(image, cx + dx, cy + dy))
            .collect()
    };
    let target_ring = ring_at(cx, cy);

    // Alegem zona donatoare al cărei inel seamănă cel mai bine cu al țintei
    // și al cărei interior este uniform (fără alte pete).
    let margin = ring_r + 1.0;
    let mut best: Option<((f32, f32), f32)> = None;
    for distance in [2.4_f32, 3.2, 4.2] {
        for k in 0..16 {
            let angle = k as f32 / 16.0 * std::f32::consts::TAU;
            let (dx, dy) = (distance * r * angle.cos(), distance * r * angle.sin());
            let (dcx, dcy) = (cx + dx, cy + dy);
            if dcx < margin
                || dcy < margin
                || dcx > w as f32 - 1.0 - margin
                || dcy > h as f32 - 1.0 - margin
            {
                continue;
            }
            let donor_ring = ring_at(dcx, dcy);
            let ring_score: f32 = target_ring
                .iter()
                .zip(&donor_ring)
                .map(|(a, b)| color_distance(*a, *b))
                .sum::<f32>()
                / RING as f32;
            let ring_mean = mean_color(&donor_ring);
            let interior_score: f32 = (0..12)
                .map(|i| {
                    let a = i as f32 / 12.0 * std::f32::consts::TAU;
                    let p = sample(image, dcx + 0.5 * r * a.cos(), dcy + 0.5 * r * a.sin());
                    color_distance(p, ring_mean)
                })
                .sum::<f32>()
                / 12.0;
            let score = ring_score + 0.5 * interior_score + distance * 2.0;
            if best.is_none_or(|(_, s)| score < s) {
                best = Some(((dx, dy), score));
            }
        }
    }
    let donor = best.map(|((dx, dy), _)| (dx, dy, ring_at(cx + dx, cy + dy)));

    let x0 = (cx - r).floor().max(0.0) as u32;
    let y0 = (cy - r).floor().max(0.0) as u32;
    let x1 = ((cx + r).ceil() as u32).min(w - 1);
    let y1 = ((cy + r).ceil() as u32).min(h - 1);
    let mut updates = Vec::new();
    for y in y0..=y1 {
        for x in x0..=x1 {
            let (px, py) = (x as f32, y as f32);
            let d = ((px - cx).powi(2) + (py - cy).powi(2)).sqrt();
            if d > r {
                continue;
            }
            let t = ((d - 0.65 * r) / (0.35 * r)).clamp(0.0, 1.0);
            let alpha = 1.0 - t * t * (3.0 - 2.0 * t);

            // Interpolare Shepard din inel (aceleași ponderi pentru țintă și donator).
            let mut total = 0.0;
            let mut fill = [0.0; 3];
            let mut donor_fill = [0.0; 3];
            for (i, (ox, oy)) in ring_offsets.iter().enumerate() {
                let dist2 = (cx + ox - px).powi(2) + (cy + oy - py).powi(2);
                let weight = 1.0 / (dist2 + 0.25);
                total += weight;
                for ch in 0..3 {
                    fill[ch] += target_ring[i][ch] * weight;
                    if let Some((_, _, ring)) = &donor {
                        donor_fill[ch] += ring[i][ch] * weight;
                    }
                }
            }
            let mut value = fill.map(|v| v / total);
            if let Some((dx, dy, _)) = &donor {
                let texture = sample(image, px + dx, py + dy);
                for ch in 0..3 {
                    value[ch] += texture[ch] - donor_fill[ch] / total;
                }
            }
            let original = image.get_pixel(x, y).0;
            let blended: [u8; 3] = std::array::from_fn(|ch| {
                (original[ch] as f32 * (1.0 - alpha) + value[ch] * alpha)
                    .round()
                    .clamp(0.0, 255.0) as u8
            });
            updates.push((x, y, blended));
        }
    }
    for (x, y, color) in updates {
        image.put_pixel(x, y, image::Rgb(color));
    }
}

fn mean_color(colors: &[[f32; 3]]) -> [f32; 3] {
    let n = colors.len().max(1) as f32;
    let mut sum = [0.0; 3];
    for c in colors {
        for ch in 0..3 {
            sum[ch] += c[ch];
        }
    }
    sum.map(|v| v / n)
}

// ---------------------------------------------------------------------------
// Îndreptare (rotire fină) și încadrare automată după față
// ---------------------------------------------------------------------------

/// Factorul de mărire necesar ca imaginea rotită să nu aibă colțuri goale.
pub fn rotation_scale(width: u32, height: u32, degrees: f32) -> f32 {
    let t = degrees.to_radians().abs();
    let (w, h) = (width as f32, height as f32);
    (t.cos() + h / w * t.sin()).max(t.cos() + w / h * t.sin())
}

/// Punctul din imaginea sursă care ajunge în (x, y) după îndreptare.
pub fn rotated_to_source(x: f32, y: f32, width: u32, height: u32, degrees: f32) -> (f32, f32) {
    let k = rotation_scale(width, height, degrees);
    let (cx, cy) = (width as f32 / 2.0, height as f32 / 2.0);
    let (s, c) = degrees.to_radians().sin_cos();
    let (dx, dy) = ((x - cx) / k, (y - cy) / k);
    (cx + c * dx - s * dy, cy + s * dx + c * dy)
}

/// Inversa lui [`rotated_to_source`].
pub fn source_to_rotated(x: f32, y: f32, width: u32, height: u32, degrees: f32) -> (f32, f32) {
    let k = rotation_scale(width, height, degrees);
    let (cx, cy) = (width as f32 / 2.0, height as f32 / 2.0);
    let (s, c) = degrees.to_radians().sin_cos();
    let (dx, dy) = (x - cx, y - cy);
    (cx + k * (c * dx + s * dy), cy + k * (-s * dx + c * dy))
}

/// Rotește imaginea cu un unghi mic, păstrând dimensiunile (cu mărire ușoară,
/// fără colțuri goale). Folosește interpolare biliniară.
fn rotate_fine<P>(
    image: &image::ImageBuffer<P, Vec<u8>>,
    degrees: f32,
) -> image::ImageBuffer<P, Vec<u8>>
where
    P: image::Pixel<Subpixel = u8>,
{
    let (w, h) = image.dimensions();
    let channels = P::CHANNEL_COUNT as usize;
    let src = image.as_raw();
    let mut out = image::ImageBuffer::<P, Vec<u8>>::new(w, h);
    let raw: &mut [u8] = &mut out;
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = rotated_to_source(x as f32 + 0.5, y as f32 + 0.5, w, h, degrees);
            let sx = (sx - 0.5).clamp(0.0, (w - 1) as f32);
            let sy = (sy - 0.5).clamp(0.0, (h - 1) as f32);
            let (x0, y0) = (sx.floor() as usize, sy.floor() as usize);
            let (x1, y1) = ((x0 + 1).min(w as usize - 1), (y0 + 1).min(h as usize - 1));
            let (fx, fy) = (sx - x0 as f32, sy - y0 as f32);
            let at = |xx: usize, yy: usize, ch: usize| {
                src[(yy * w as usize + xx) * channels + ch] as f32
            };
            let dst = (y as usize * w as usize + x as usize) * channels;
            for ch in 0..channels {
                let top = at(x0, y0, ch) * (1.0 - fx) + at(x1, y0, ch) * fx;
                let bottom = at(x0, y1, ch) * (1.0 - fx) + at(x1, y1, ch) * fx;
                raw[dst + ch] = (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Unghiul de îndreptare sugerat de linia ochilor (0 dacă fața e prea înclinată
/// pentru a fi o eroare de fotografiere).
pub fn auto_straighten_angle(face: &Face, width: u32, height: u32) -> f32 {
    let angle = face.eye_angle_degrees(width, height);
    if angle.abs() > 20.0 {
        0.0
    } else {
        (angle * 10.0).round() / 10.0
    }
}

/// Calculează zoom-ul și poziția astfel încât fața să fie încadrată corect:
/// pentru foto act, capul ocupă ~65% din înălțime, cu ochii la ~42% de sus.
pub fn auto_frame(width: u32, height: u32, face: &Face, framing: &Framing) -> Framing {
    let mut result = *framing;
    let Some((aspect_w, aspect_h)) = framing.format.aspect() else {
        return result;
    };
    let angle = framing.rotation;
    let to_px = |(x, y): (f32, f32)| {
        source_to_rotated(x * width as f32, y * height as f32, width, height, angle)
    };
    let eyes = to_px(face.eyes_center());
    let mouth = to_px(face.mouth_center());
    // Bărbia: puțin sub marginea de jos a feței; creștetul: deasupra ochilor
    // cu aproximativ aceeași distanță ca de la ochi la bărbie.
    let chin_y = (face.y + face.h) * height as f32;
    let chin = to_px((face.eyes_center().0, chin_y / height as f32)).1;
    let eye_to_chin = (chin - eyes.1).max((mouth.1 - eyes.1) * 1.6).max(1.0);
    let head = eye_to_chin * 1.95;

    let (head_share, eyes_from_top) = if framing.format == OutputFormat::Act3x4 {
        (0.65, 0.42)
    } else {
        (0.42, 0.38)
    };
    let crop_h = head / head_share;
    let crop_w = crop_h * aspect_w as f32 / aspect_h as f32;

    // Dimensiunea decupării la zoom 1 (maximul posibil); zoom-ul nu scade sub 1.
    let base = Framing {
        zoom: 1.0,
        offset_x: 0.0,
        offset_y: 0.0,
        ..*framing
    };
    let (_, _, base_w, base_h) = crop_rect(width, height, &base);
    let zoom = (base_w as f32 / crop_w).clamp(1.0, 4.0);
    let crop_w = base_w as f32 / zoom;
    let crop_h = base_h as f32 / zoom;
    let left = eyes.0 - crop_w / 2.0;
    let top = eyes.1 - crop_h * eyes_from_top;
    let max_x = width as f32 - crop_w;
    let max_y = height as f32 - crop_h;
    let offset = |pos: f32, max: f32| {
        if max < 1.0 {
            0.0
        } else {
            (pos.clamp(0.0, max) / max * 2.0 - 1.0).clamp(-1.0, 1.0)
        }
    };
    result.zoom = zoom;
    result.offset_x = offset(left, max_x);
    result.offset_y = offset(top, max_y);
    result
}

// ---------------------------------------------------------------------------
// Auto-corecție („mini AI”): analizează histograma, balansul de alb,
// zgomotul și claritatea, cu accent pe persoană când masca AI este disponibilă.
// ---------------------------------------------------------------------------

pub fn auto_adjust(image: &RgbImage, mask: Option<&GrayImage>, faces: &[Face]) -> Adjustments {
    let (w, h) = image.dimensions();
    let total = (w * h) as usize;
    let step = ((total / 250_000).max(1) as f32).sqrt().ceil() as u32;
    let mask = mask.filter(|m| m.dimensions() == (w, h));

    let mut samples: Vec<([f32; 3], f32)> = Vec::new();
    for y in (0..h).step_by(step as usize) {
        for x in (0..w).step_by(step as usize) {
            let p = image.get_pixel(x, y);
            let weight = mask.map_or(1.0, |m| 0.2 + 0.8 * m.get_pixel(x, y)[0] as f32 / 255.0);
            samples.push(([p[0], p[1], p[2]].map(|v| v as f32 / 255.0), weight));
        }
    }
    if samples.is_empty() {
        return Adjustments::default();
    }

    // 1. Balans de alb: media pixelilor aproape neutri (ignoră pielea și culorile vii).
    let mut neutral = [0.0_f32; 3];
    let mut neutral_weight = 0.0;
    let mut all = [0.0_f32; 3];
    let mut all_weight = 0.0;
    for (p, wt) in &samples {
        let l = luma(p[0], p[1], p[2]);
        let chroma = p[0].max(p[1]).max(p[2]) - p[0].min(p[1]).min(p[2]);
        for ch in 0..3 {
            all[ch] += p[ch] * wt;
        }
        all_weight += wt;
        if chroma < 0.12 && (0.15..0.97).contains(&l) {
            for ch in 0..3 {
                neutral[ch] += p[ch] * wt;
            }
            neutral_weight += wt;
        }
    }
    let (reference, strength) = if neutral_weight > all_weight * 0.03 {
        (neutral.map(|v| v / neutral_weight), 0.8)
    } else {
        (all.map(|v| v / all_weight), 0.35)
    };
    let [rr, rg, rb] = reference.map(|v| v.max(1e-3));
    let k = (rb - rr) / (rb + rr);
    let temperature = (k / 0.25 * 100.0 * strength).clamp(-50.0, 50.0);
    let gains = white_balance_gains(temperature, 0.0);
    let green_target = (rr * gains[0] + rb * gains[2]) / 2.0;
    let gg = green_target / rg;
    let tint = ((1.0 - gg) / 0.2 * 100.0 * strength).clamp(-40.0, 40.0);
    let gains = white_balance_gains(temperature, tint);

    // 2. Expunere, umbre, lumini și contrast din percentilele luminanței.
    let mut lum: Vec<(f32, f32)> = samples
        .iter()
        .map(|(p, wt)| (luma(p[0] * gains[0], p[1] * gains[1], p[2] * gains[2]), *wt))
        .collect();
    lum.sort_by(|a, b| a.0.total_cmp(&b.0));
    let weight_sum: f32 = lum.iter().map(|(_, wt)| wt).sum();
    let percentile = |q: f32| {
        let goal = q * weight_sum;
        let mut acc = 0.0;
        for (v, wt) in &lum {
            acc += wt;
            if acc >= goal {
                return *v;
            }
        }
        lum.last().map_or(0.5, |(v, _)| *v)
    };
    let (p01, p50, p95, p99) = (
        percentile(0.01),
        percentile(0.5),
        percentile(0.95),
        percentile(0.99),
    );
    // Țintim tonuri medii echilibrate, dar fără a „arde” zonele luminoase (fața).
    // Dacă există piele (portret), expunerea se reglează după față: tonul
    // pielii bine expus are luminanța în jur de 0.55.
    // Cu o față detectată, se folosește doar pielea din interiorul feței
    // (obraji, frunte, nas), care este cea mai sigură referință.
    let face_skin: Vec<(f32, f32)> = faces
        .first()
        .map(|face| {
            let x0 = ((face.x + face.w * 0.2) * w as f32).max(0.0) as u32;
            let x1 = (((face.x + face.w * 0.8) * w as f32) as u32).min(w);
            let y0 = ((face.y + face.h * 0.25) * h as f32).max(0.0) as u32;
            let y1 = (((face.y + face.h * 0.85) * h as f32) as u32).min(h);
            let mut out = Vec::new();
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = image.get_pixel(x, y).0.map(|v| v as f32 / 255.0);
                    let l = luma(p[0] * gains[0], p[1] * gains[1], p[2] * gains[2]);
                    out.push((l, skin_likelihood(p[0], p[1], p[2]) + 0.05));
                }
            }
            out
        })
        .unwrap_or_default();
    let mut skin: Vec<(f32, f32)> = if face_skin.len() > 50 {
        face_skin
    } else {
        samples
            .iter()
            .map(|(p, wt)| {
                let l = luma(p[0] * gains[0], p[1] * gains[1], p[2] * gains[2]);
                (l, skin_likelihood(p[0], p[1], p[2]) * wt)
            })
            .filter(|(_, wt)| *wt > 0.05)
            .collect()
    };
    let skin_weight: f32 = skin.iter().map(|(_, wt)| wt).sum();
    let ev = if skin_weight > weight_sum * 0.02 || (!faces.is_empty() && skin_weight > 1.0) {
        skin.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut acc = 0.0;
        let mut median = 0.5;
        for (v, wt) in &skin {
            acc += wt;
            if acc >= skin_weight / 2.0 {
                median = *v;
                break;
            }
        }
        (0.55 / median.max(0.02)).log2() * 0.8
    } else {
        (0.46 / p50.max(0.02)).log2() * 0.6
    };
    // Fără a „arde” zonele luminoase.
    let ev = ev.min((0.9 / p95.max(0.02)).log2()).clamp(-1.0, 1.0);
    let exposure = (ev / 2.0 * 100.0).round();
    let factor = 2.0_f32.powf(ev);
    let (low, high) = (p01 * factor, (p99 * factor).min(1.2));
    let highlights = if high > 0.96 {
        -((high - 0.9) * 250.0).clamp(0.0, 60.0)
    } else {
        0.0
    };
    let shadows = if low < 0.05 {
        ((0.07 - low) * 500.0).clamp(0.0, 35.0)
    } else {
        0.0
    };
    let spread = high.min(1.0) - low;
    let contrast = ((0.85 - spread) * 80.0).clamp(-15.0, 30.0);

    // 3. Saturație după croma medie.
    let mean_chroma = samples
        .iter()
        .map(|(p, wt)| (p[0].max(p[1]).max(p[2]) - p[0].min(p[1]).min(p[2])) * wt)
        .sum::<f32>()
        / all_weight;
    let saturation = if mean_chroma < 0.08 {
        18.0
    } else if mean_chroma < 0.15 {
        8.0
    } else if mean_chroma > 0.35 {
        -10.0
    } else {
        0.0
    };

    // 4. Zgomot și claritate estimate din diferența față de imaginea netezită.
    let float = FloatImage::from_rgb(image);
    let l = float.luminance();
    let blurred = box_blur(&l, w as usize, h as usize, 1);
    let mut residuals: Vec<f32> = l
        .iter()
        .zip(&blurred)
        .step_by((step * step) as usize)
        .map(|(a, b)| (a - b).abs())
        .collect();
    residuals.sort_by(f32::total_cmp);
    let noise = residuals.get(residuals.len() / 2).copied().unwrap_or(0.0) * 1.4826 * 255.0;
    let denoise = ((noise - 1.2) * 12.0).clamp(0.0, 55.0).round();
    let detail = residuals
        .get(residuals.len() * 9 / 10)
        .copied()
        .unwrap_or(0.0)
        * 255.0;
    let sharpness = if detail < 4.0 { 35.0 } else { 20.0 };

    Adjustments {
        exposure,
        brightness: 0.0,
        contrast: contrast.round(),
        highlights: highlights.round(),
        shadows: shadows.round(),
        saturation,
        vibrance: 15.0,
        temperature: temperature.round(),
        tint: tint.round(),
        clarity: 10.0,
        sharpness,
        denoise,
        skin_smoothing: 0.0,
    }
}

/// Micșorează o mască la dimensiunile date.
pub fn resize_mask(mask: &GrayImage, w: u32, h: u32) -> GrayImage {
    if mask.dimensions() == (w, h) {
        return mask.clone();
    }
    image::imageops::resize(mask, w, h, FilterType::Triangle)
}

/// Utilitar pentru teste: o mască plină.
#[cfg(test)]
pub fn full_mask(w: u32, h: u32, value: u8) -> GrayImage {
    GrayImage::from_pixel(w, h, image::Luma([value]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgb};

    fn settings() -> RenderSettings {
        RenderSettings::default()
    }

    #[test]
    fn original_format_keeps_full_resolution() {
        let base = RgbImage::from_pixel(640, 480, Rgb([120, 130, 140]));
        let out = render(&base, None, None, &settings(), true);
        assert_eq!(out.dimensions(), (640, 480));
        assert_eq!(out.get_pixel(10, 10).0, [120, 130, 140, 255]);
    }

    #[test]
    fn act_format_exports_exact_print_size() {
        let base = RgbImage::from_pixel(1200, 800, Rgb([90, 90, 90]));
        let mut s = settings();
        s.framing.format = OutputFormat::Act3x4;
        assert_eq!(
            render(&base, None, None, &s, true).dimensions(),
            (ACT_WIDTH, ACT_HEIGHT)
        );
        // Previzualizarea păstrează rezoluția decupării, cu același raport.
        let preview = render(&base, None, None, &s, false);
        let ratio = preview.width() as f32 / preview.height() as f32;
        assert!((ratio - 0.75).abs() < 0.01);
    }

    #[test]
    fn square_crop_has_square_output() {
        let base = RgbImage::new(900, 600);
        let mut s = settings();
        s.framing.format = OutputFormat::Square;
        s.framing.zoom = 2.0;
        assert_eq!(render(&base, None, None, &s, true).dimensions(), (300, 300));
    }

    #[test]
    fn color_background_replaces_only_background() {
        let base = RgbImage::from_pixel(40, 40, Rgb([200, 20, 10]));
        let mut mask = full_mask(40, 40, 0);
        for y in 10..30 {
            for x in 10..30 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        let mut s = settings();
        s.background = BackgroundMode::Color([255, 255, 255]);
        let out = render(&base, Some(&mask), None, &s, true);
        assert_eq!(out.get_pixel(0, 0).0, [255, 255, 255, 255]);
        assert_eq!(out.get_pixel(20, 20).0, [200, 20, 10, 255]);
    }

    #[test]
    fn transparent_background_uses_alpha() {
        let base = RgbImage::from_pixel(20, 20, Rgb([50, 60, 70]));
        let mut mask = full_mask(20, 20, 0);
        mask.put_pixel(10, 10, Luma([255]));
        let mut s = settings();
        s.background = BackgroundMode::Transparent;
        let out = render(&base, Some(&mask), None, &s, true);
        assert_eq!(out.get_pixel(0, 0)[3], 0);
        assert_eq!(out.get_pixel(10, 10)[3], 255);
    }

    #[test]
    fn image_background_is_used() {
        let base = RgbImage::from_pixel(30, 20, Rgb([0, 0, 0]));
        let mask = full_mask(30, 20, 0);
        let backdrop = RgbImage::from_pixel(7, 9, Rgb([10, 200, 30]));
        let mut s = settings();
        s.background = BackgroundMode::Image;
        let out = render(&base, Some(&mask), Some(&backdrop), &s, true);
        assert_eq!(out.get_pixel(15, 10).0, [10, 200, 30, 255]);
    }

    #[test]
    fn neutral_adjustments_do_not_change_pixels() {
        let base = RgbImage::from_fn(32, 32, |x, y| Rgb([(x * 8) as u8, (y * 8) as u8, 77]));
        let out = render(&base, None, None, &settings(), true);
        for (a, b) in base.pixels().zip(out.pixels()) {
            assert_eq!([a[0], a[1], a[2], 255], b.0);
        }
    }

    #[test]
    fn brightness_and_saturation_work() {
        let base = RgbImage::from_pixel(8, 8, Rgb([100, 120, 140]));
        let mut s = settings();
        s.adjustments.brightness = 50.0;
        let bright = render(&base, None, None, &s, true);
        assert!(bright.get_pixel(0, 0)[1] > 120);
        s.adjustments.brightness = 0.0;
        s.adjustments.saturation = -100.0;
        let gray = render(&base, None, None, &s, true);
        let p = gray.get_pixel(0, 0);
        assert!(p[0].abs_diff(p[2]) <= 1);
    }

    #[test]
    fn guided_filter_keeps_flat_image() {
        let flat = vec![0.4; 50 * 40];
        let out = guided_self(&flat, 50, 40, 3, 0.01);
        assert!(out.iter().all(|v| (v - 0.4).abs() < 1e-4));
    }

    #[test]
    fn denoise_reduces_noise() {
        let noisy = RgbImage::from_fn(64, 64, |x, y| {
            let n = ((x * 7919 + y * 104_729) % 41) as i32 - 20;
            let v = (128 + n) as u8;
            Rgb([v, v, v])
        });
        let mut s = settings();
        s.adjustments.denoise = 80.0;
        let out = render(&noisy, None, None, &s, true);
        let spread = |img: &dyn Fn(u32, u32) -> u8| {
            (0..64)
                .flat_map(|y| (0..64).map(move |x| (x, y)))
                .map(|(x, y)| (img(x, y) as f32 - 128.0).abs())
                .sum::<f32>()
        };
        let before = spread(&|x, y| noisy.get_pixel(x, y)[0]);
        let after = spread(&|x, y| out.get_pixel(x, y)[0]);
        assert!(after < before * 0.6, "{after} vs {before}");
    }

    #[test]
    fn healing_removes_a_dark_spot() {
        let mut image = RgbImage::from_pixel(200, 200, Rgb([210, 170, 150]));
        for y in 95..106 {
            for x in 95..106 {
                image.put_pixel(x, y, Rgb([60, 30, 30]));
            }
        }
        heal_spots(
            &mut image,
            &[Spot {
                x: 0.5,
                y: 0.5,
                radius: 0.05,
            }],
        );
        let p = image.get_pixel(100, 100);
        assert!(p[0] > 200 && p[1] > 160, "{p:?}");
    }

    #[test]
    fn auto_adjust_brightens_dark_photo() {
        let dark = RgbImage::from_fn(100, 100, |x, _| {
            let v = (20 + x / 3) as u8;
            Rgb([v, v, v])
        });
        let adj = auto_adjust(&dark, None, &[]);
        assert!(adj.exposure > 10.0, "{adj:?}");
        assert!(adj.temperature.abs() < 5.0 && adj.tint.abs() < 5.0);
    }

    #[test]
    fn auto_adjust_corrects_blue_cast() {
        let blue = RgbImage::from_fn(100, 100, |x, _| {
            let v = (90 + x) as u8;
            Rgb([v, v, v.saturating_add(14)])
        });
        assert!(auto_adjust(&blue, None, &[]).temperature > 5.0);
    }

    #[test]
    fn rotation_mapping_round_trips() {
        let (x, y) = source_to_rotated(123.0, 45.0, 400, 300, 7.5);
        let (bx, by) = rotated_to_source(x, y, 400, 300, 7.5);
        assert!((bx - 123.0).abs() < 1e-3 && (by - 45.0).abs() < 1e-3);
    }

    #[test]
    fn straightening_keeps_dimensions_and_center() {
        let image = RgbImage::from_fn(101, 81, |x, y| {
            if (x as i32 - 50).abs() < 3 && (y as i32 - 40).abs() < 3 {
                Rgb([255, 0, 0])
            } else {
                Rgb([0, 0, 255])
            }
        });
        let rotated = rotate_fine(&image, 10.0);
        assert_eq!(rotated.dimensions(), (101, 81));
        assert_eq!(rotated.get_pixel(50, 40).0, [255, 0, 0]);
        // Fără colțuri goale (negre).
        assert_eq!(rotated.get_pixel(0, 0).0, [0, 0, 255]);
    }

    #[test]
    fn auto_frame_centers_face_for_id_photo() {
        let face = Face {
            score: 0.9,
            x: 0.4,
            y: 0.3,
            w: 0.1,
            h: 0.15,
            landmarks: [
                (0.43, 0.36),
                (0.47, 0.36),
                (0.45, 0.4),
                (0.435, 0.42),
                (0.465, 0.42),
            ],
        };
        let framing = Framing {
            format: OutputFormat::Act3x4,
            ..Framing::default()
        };
        let (w, h) = (2000, 1500);
        let framed = auto_frame(w, h, &face, &framing);
        assert!(framed.zoom > 1.0);
        let (x, y, cw, ch) = crop_rect(w, h, &framed);
        let eyes_x = 0.45 * w as f32;
        let eyes_y = 0.36 * h as f32;
        assert!(((eyes_x - x as f32) / cw as f32 - 0.5).abs() < 0.02);
        assert!(((eyes_y - y as f32) / ch as f32 - 0.42).abs() < 0.03);
    }
}
