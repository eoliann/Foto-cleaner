//! Eliminarea watermark-urilor și a obiectelor mici: zona marcată de utilizator
//! este reconstruită cu modelul MI-GAN (Picsart AI Research, licență MIT),
//! rulat local prin RTen. Dacă modelul nu poate fi folosit, zona este umplută
//! cu o metodă clasică (difuzie piramidală de la margini).

use image::{GrayImage, Luma, Rgb, RgbImage, imageops::FilterType};
use rten::{Model, ValueView};

pub const MIGAN_MODEL: &[u8] = include_bytes!("../assets/migan.onnx");

/// Rezoluția de lucru a modelului.
const RES: u32 = 512;
/// Context păstrat în jurul zonei marcate, de fiecare parte (ca în pipeline-ul oficial).
const PADDING: u32 = 128;
/// Zonele marcate mai apropiate decât atât (în pixeli) sunt procesate împreună.
const GROUP_DISTANCE: u32 = 24;
/// Cu câți pixeli se lărgește zona marcată.
const EDGE_GROWTH: u32 = 3;

/// Dreptunghi (x0, y0, x1, y1), cu x1 și y1 exclusive.
type Rect = (u32, u32, u32, u32);

/// O trăsătură de pensulă, în coordonate normalizate ale sursei (0..1).
/// Raza este relativă la latura cea mai lungă a imaginii.
#[derive(Clone, Debug, PartialEq)]
pub struct Stroke {
    pub points: Vec<(f32, f32)>,
    pub radius: f32,
}

pub fn load_model() -> Result<Model, String> {
    Model::load_static_slice(MIGAN_MODEL)
        .map_err(|error| format!("Modelul MI-GAN nu poate fi încărcat: {error}"))
}

/// Desenează trăsăturile într-o mască de dimensiunile date (255 = de eliminat).
pub fn rasterize_strokes(strokes: &[Stroke], width: u32, height: u32) -> GrayImage {
    let mut mask = GrayImage::new(width, height);
    let long = width.max(height) as f32;
    for stroke in strokes {
        let r = (stroke.radius * long).max(1.0);
        let points: Vec<(f32, f32)> = stroke
            .points
            .iter()
            .map(|&(x, y)| (x * width as f32, y * height as f32))
            .collect();
        if points.len() == 1 {
            fill_capsule(&mut mask, points[0], points[0], r);
        }
        for pair in points.windows(2) {
            fill_capsule(&mut mask, pair[0], pair[1], r);
        }
    }
    mask
}

fn fill_capsule(mask: &mut GrayImage, a: (f32, f32), b: (f32, f32), r: f32) {
    let (w, h) = mask.dimensions();
    let x0 = (a.0.min(b.0) - r).floor().max(0.0) as u32;
    let y0 = (a.1.min(b.1) - r).floor().max(0.0) as u32;
    let x1 = ((a.0.max(b.0) + r).ceil().max(0.0) as u32).min(w);
    let y1 = ((a.1.max(b.1) + r).ceil().max(0.0) as u32).min(h);
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len2 = dx * dx + dy * dy;
    for y in y0..y1 {
        for x in x0..x1 {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            let t = if len2 > 0.0 {
                (((px - a.0) * dx + (py - a.1) * dy) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let (qx, qy) = (a.0 + t * dx - px, a.1 + t * dy - py);
            if qx * qx + qy * qy <= r * r {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
    }
}

/// Reconstruiește zonele marcate (mask > 127) ale imaginii.
/// Fără model (sau dacă modelul eșuează) se folosește umplerea clasică; în
/// acest caz se întoarce și eroarea modelului.
pub fn inpaint(
    model: Option<&Model>,
    image: &RgbImage,
    mask: &GrayImage,
) -> (RgbImage, Option<String>) {
    let mut result = image.clone();
    let mut ai_error = None;
    // Marginea zonei este lărgită puțin: pixelii semi-transparenți de pe
    // conturul watermark-ului nu trebuie să rămână „cunoscuți” pentru model.
    let mask = &grow_mask(mask, EDGE_GROWTH);
    for (x0, y0, x1, y1) in mask_groups(mask) {
        let crop = crop_box(x0, y0, x1, y1, image.width(), image.height());
        let group = (x0, y0, x1, y1);
        let outcome = match model {
            Some(model) => inpaint_region_ai(model, &mut result, mask, crop, group),
            None => Err("modelul AI nu este disponibil".to_owned()),
        };
        if let Err(error) = outcome {
            ai_error.get_or_insert(error);
            inpaint_region_classic(&mut result, mask, crop, group);
        }
    }
    (result, ai_error)
}

/// Dreptunghiurile (x0, y0, x1, y1 exclusiv) ale grupurilor de zone marcate.
fn mask_groups(mask: &GrayImage) -> Vec<Rect> {
    let (w, h) = mask.dimensions();
    // Componentele conexe se caută pe o grilă redusă: fiecare celulă are
    // latura GROUP_DISTANCE, deci zonele apropiate ajung în același grup.
    let cell = GROUP_DISTANCE;
    let (gw, gh) = (w.div_ceil(cell), h.div_ceil(cell));
    let mut grid = vec![false; (gw * gh) as usize];
    for (x, y, p) in mask.enumerate_pixels() {
        if p[0] > 127 {
            grid[((y / cell) * gw + x / cell) as usize] = true;
        }
    }
    let mut seen = vec![false; grid.len()];
    let mut groups = Vec::new();
    for start in 0..grid.len() {
        if !grid[start] || seen[start] {
            continue;
        }
        let mut stack = vec![start];
        seen[start] = true;
        let (mut gx0, mut gy0, mut gx1, mut gy1) = (u32::MAX, u32::MAX, 0, 0);
        while let Some(i) = stack.pop() {
            let (gx, gy) = (i as u32 % gw, i as u32 / gw);
            gx0 = gx0.min(gx);
            gy0 = gy0.min(gy);
            gx1 = gx1.max(gx);
            gy1 = gy1.max(gy);
            for ny in gy.saturating_sub(1)..=(gy + 1).min(gh - 1) {
                for nx in gx.saturating_sub(1)..=(gx + 1).min(gw - 1) {
                    let j = (ny * gw + nx) as usize;
                    if grid[j] && !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        // Dreptunghiul exact al pixelilor marcați din grup.
        let (cx0, cy0) = (gx0 * cell, gy0 * cell);
        let (cx1, cy1) = (((gx1 + 1) * cell).min(w), ((gy1 + 1) * cell).min(h));
        let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0, 0);
        for y in cy0..cy1 {
            for x in cx0..cx1 {
                if mask.get_pixel(x, y)[0] > 127 {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x + 1);
                    y1 = y1.max(y + 1);
                }
            }
        }
        if x0 < x1 {
            groups.push((x0, y0, x1, y1));
        }
    }
    groups
}

/// Pătratul de context din jurul zonei (latură ≥ 512, cu PADDING de fiecare
/// parte), deplasat cât să încapă în imagine.
fn crop_box(x0: u32, y0: u32, x1: u32, y1: u32, w: u32, h: u32) -> Rect {
    let size = ((x1 - x0).max(y1 - y0) + 2 * PADDING).max(RES);
    let place = |lo: u32, hi: u32, limit: u32| {
        let side = size.min(limit);
        let center = (lo + hi) / 2;
        let start = center.saturating_sub(side / 2).min(limit - side);
        (start, start + side)
    };
    let (cx0, cx1) = place(x0, x1, w);
    let (cy0, cy1) = place(y0, y1, h);
    (cx0, cy0, cx1, cy1)
}

fn inpaint_region_ai(
    model: &Model,
    image: &mut RgbImage,
    mask: &GrayImage,
    (x0, y0, x1, y1): Rect,
    group: Rect,
) -> Result<(), String> {
    let (cw, ch) = (x1 - x0, y1 - y0);
    let crop = image::imageops::crop_imm(image, x0, y0, cw, ch).to_image();
    let crop_mask = image::imageops::crop_imm(mask, x0, y0, cw, ch).to_image();
    let small = image::imageops::resize(&crop, RES, RES, FilterType::Triangle);
    let small_mask = image::imageops::resize(&crop_mask, RES, RES, FilterType::Nearest);

    // Intrare: [cunoscut - 0.5, imagine * cunoscut], imaginea în [-1, 1]
    // (cunoscut = 1 în afara zonei marcate, 0 în interior).
    let area = (RES * RES) as usize;
    let mut input = vec![0.0_f32; area * 4];
    for (i, (pixel, m)) in small.pixels().zip(small_mask.pixels()).enumerate() {
        let known = if m[0] > 127 { 0.0 } else { 1.0 };
        input[i] = known - 0.5;
        for c in 0..3 {
            input[(c + 1) * area + i] = (pixel[c] as f32 / 127.5 - 1.0) * known;
        }
    }
    let view = ValueView::from_shape([1, 4, RES as usize, RES as usize], &input)
        .map_err(|error| format!("Intrare MI-GAN invalidă: {error}"))?;
    let output = model
        .run_one(view.into(), None)
        .map_err(|error| format!("MI-GAN a eșuat: {error}"))?;
    let ([_, channels, oh, ow], values) = output
        .into_shape_vec::<f32, 4>()
        .map_err(|error| format!("Rezultat MI-GAN invalid: {error}"))?;
    if channels != 3 || oh != RES as usize || ow != RES as usize {
        return Err("Rezultat MI-GAN cu dimensiuni neașteptate.".to_owned());
    }
    let out_area = oh * ow;
    let generated = RgbImage::from_fn(RES, RES, |x, y| {
        let i = (y * RES + x) as usize;
        let v = |c: usize| ((values[c * out_area + i] * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
        Rgb([v(0), v(1), v(2)])
    });
    let generated = image::imageops::resize(&generated, cw, ch, FilterType::Triangle);
    blend_into(image, &generated, &crop_mask, (x0, y0), group);
    Ok(())
}

/// Copiază rezultatul doar în zona marcată a grupului curent. Pixelii din
/// afara zonei nu se modifică (ieșirea modelului de acolo nu e de încredere);
/// doar primul rând de pixeli din interior se amestecă ușor cu originalul.
fn blend_into(
    image: &mut RgbImage,
    filled: &RgbImage,
    crop_mask: &GrayImage,
    (x0, y0): (u32, u32),
    (gx0, gy0, gx1, gy1): Rect,
) {
    let (w, h) = crop_mask.dimensions();
    let known: Vec<f32> = crop_mask
        .pixels()
        .map(|p| if p[0] > 127 { 0.0 } else { 1.0 })
        .collect();
    let soft_known = crate::processing::box_blur(&known, w as usize, h as usize, 1);
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let (sx, sy) = (x0 + x, y0 + y);
            if known[i] > 0.5 || sx < gx0 || sx >= gx1 || sy < gy0 || sy >= gy1 {
                continue;
            }
            let a = 1.0 - 0.5 * soft_known[i].clamp(0.0, 1.0);
            let src = filled.get_pixel(x, y);
            let dst = image.get_pixel_mut(sx, sy);
            for c in 0..3 {
                dst[c] = (dst[c] as f32 * (1.0 - a) + src[c] as f32 * a).round() as u8;
            }
        }
    }
}

/// Lărgește zona marcată (mask > 127) cu `r` pixeli (filtru maxim pătrat, separabil).
fn grow_mask(mask: &GrayImage, r: u32) -> GrayImage {
    let (w, h) = mask.dimensions();
    let max_pass = |src: &GrayImage, horizontal: bool| {
        GrayImage::from_fn(w, h, |x, y| {
            let (pos, limit) = if horizontal { (x, w) } else { (y, h) };
            let mut m = 0;
            for k in pos.saturating_sub(r)..=(pos + r).min(limit - 1) {
                let p = if horizontal {
                    src.get_pixel(k, y)
                } else {
                    src.get_pixel(x, k)
                };
                m = m.max(p[0]);
            }
            Luma([m])
        })
    };
    max_pass(&max_pass(mask, true), false)
}

/// Umplere clasică (fără AI): valorile cunoscute sunt „împinse” pe o
/// piramidă de rezoluții și apoi „trase” înapoi, rezultând o tranziție
/// netedă de la marginile zonei spre interior.
fn inpaint_region_classic(
    image: &mut RgbImage,
    mask: &GrayImage,
    (x0, y0, x1, y1): Rect,
    group: Rect,
) {
    let (w, h) = ((x1 - x0) as usize, (y1 - y0) as usize);
    let mut colors = vec![[0.0_f32; 3]; w * h];
    let mut weights = vec![0.0_f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = (x0 + x as u32, y0 + y as u32);
            if mask.get_pixel(sx, sy)[0] <= 127 {
                let p = image.get_pixel(sx, sy);
                colors[y * w + x] = [p[0] as f32, p[1] as f32, p[2] as f32];
                weights[y * w + x] = 1.0;
            }
        }
    }
    let filled = push_pull(&colors, &weights, w, h);
    let crop_mask = image::imageops::crop_imm(mask, x0, y0, w as u32, h as u32).to_image();
    let filled_image = RgbImage::from_fn(w as u32, h as u32, |x, y| {
        let c = filled[y as usize * w + x as usize];
        Rgb(c.map(|v| v.round().clamp(0.0, 255.0) as u8))
    });
    blend_into(image, &filled_image, &crop_mask, (x0, y0), group);
}

fn push_pull(colors: &[[f32; 3]], weights: &[f32], w: usize, h: usize) -> Vec<[f32; 3]> {
    if w <= 1 && h <= 1 {
        return colors
            .iter()
            .zip(weights)
            .map(|(c, &wt)| {
                if wt > 0.0 {
                    c.map(|v| v / wt)
                } else {
                    [128.0; 3]
                }
            })
            .collect();
    }
    // Nivelul următor: sume ponderate pe blocuri 2x2.
    let (nw, nh) = (w.div_ceil(2), h.div_ceil(2));
    let mut c2 = vec![[0.0_f32; 3]; nw * nh];
    let mut w2 = vec![0.0_f32; nw * nh];
    for y in 0..h {
        for x in 0..w {
            let (i, j) = (y * w + x, (y / 2) * nw + x / 2);
            for c in 0..3 {
                c2[j][c] += colors[i][c] * weights[i];
            }
            w2[j] += weights[i];
        }
    }
    for j in 0..nw * nh {
        if w2[j] > 0.0 {
            c2[j] = c2[j].map(|v| v / w2[j]);
            w2[j] = w2[j].min(1.0);
        }
    }
    let coarse = push_pull(&c2, &w2, nw, nh);
    let mut out = vec![[0.0_f32; 3]; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            // Interpolare biliniară din nivelul grosier.
            let fx = ((x as f32 + 0.5) / 2.0 - 0.5).clamp(0.0, (nw - 1) as f32);
            let fy = ((y as f32 + 0.5) / 2.0 - 0.5).clamp(0.0, (nh - 1) as f32);
            let (ax, ay) = (fx.floor() as usize, fy.floor() as usize);
            let (bx, by) = ((ax + 1).min(nw - 1), (ay + 1).min(nh - 1));
            let (tx, ty) = (fx - ax as f32, fy - ay as f32);
            let mut up = [0.0_f32; 3];
            for (c, slot) in up.iter_mut().enumerate() {
                let top = coarse[ay * nw + ax][c] * (1.0 - tx) + coarse[ay * nw + bx][c] * tx;
                let bottom = coarse[by * nw + ax][c] * (1.0 - tx) + coarse[by * nw + bx][c] * tx;
                *slot = top * (1.0 - ty) + bottom * ty;
            }
            let wt = weights[i].min(1.0);
            let own = if weights[i] > 0.0 {
                colors[i]
            } else {
                [0.0; 3]
            };
            for c in 0..3 {
                out[i][c] = own[c] * wt + up[c] * (1.0 - wt);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Imagine de test: un fundal neted, ca un perete sau o piele luminată lateral.
    fn test_image(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| {
            let (u, v) = (x as f32 / w as f32, y as f32 / h as f32);
            Rgb([
                (150.0 + 50.0 * u) as u8,
                (120.0 + 30.0 * v) as u8,
                (100.0 + 20.0 * u * v) as u8,
            ])
        })
    }

    fn watermark(image: &mut RgbImage, mask: &GrayImage) {
        for (x, y, m) in mask.enumerate_pixels() {
            if m[0] > 127 && (x / 3 + y / 3) % 2 == 0 {
                image.put_pixel(x, y, Rgb([255, 255, 255]));
            }
        }
    }

    fn mean_error(a: &RgbImage, b: &RgbImage, mask: &GrayImage) -> f32 {
        let mut sum = 0.0;
        let mut n = 0.0;
        for (x, y, m) in mask.enumerate_pixels() {
            if m[0] > 127 {
                let (p, q) = (a.get_pixel(x, y), b.get_pixel(x, y));
                for c in 0..3 {
                    sum += (p[c] as f32 - q[c] as f32).abs();
                }
                n += 3.0;
            }
        }
        sum / n
    }

    fn sample_strokes() -> Vec<Stroke> {
        vec![
            Stroke {
                points: vec![(0.3, 0.5), (0.7, 0.55)],
                radius: 0.03,
            },
            Stroke {
                points: vec![(0.85, 0.15)],
                radius: 0.02,
            },
        ]
    }

    #[test]
    fn strokes_are_rasterized_where_painted() {
        let mask = rasterize_strokes(&sample_strokes(), 400, 300);
        assert_eq!(mask.get_pixel(200, 157)[0], 255);
        assert_eq!(mask.get_pixel(340, 45)[0], 255);
        assert_eq!(mask.get_pixel(20, 20)[0], 0);
        assert_eq!(mask_groups(&mask).len(), 2);
    }

    #[test]
    fn crop_box_stays_inside_the_image() {
        let (x0, y0, x1, y1) = crop_box(10, 380, 60, 395, 700, 400);
        assert_eq!((x1 - x0, y1 - y0), (512, 400));
        assert!(x0 <= 10 && x1 >= 60 && y1 <= 400);
    }

    #[test]
    fn classic_fill_removes_a_watermark() {
        let clean = test_image(400, 300);
        let mask = rasterize_strokes(&sample_strokes(), 400, 300);
        let mut marked = clean.clone();
        watermark(&mut marked, &mask);
        let (restored, error) = inpaint(None, &marked, &mask);
        assert!(error.is_some());
        let before = mean_error(&marked, &clean, &mask);
        let after = mean_error(&restored, &clean, &mask);
        assert!(after < before / 3.0, "înainte {before}, după {after}");
        // În afara zonei marcate imaginea rămâne neatinsă (cu excepția marginii de 1-2 px).
        assert_eq!(restored.get_pixel(10, 10), clean.get_pixel(10, 10));
    }

    #[test]
    fn embedded_migan_model_removes_a_watermark() {
        let model = load_model().expect("embedded MI-GAN model should load");
        let clean = test_image(640, 480);
        let mask = rasterize_strokes(&sample_strokes(), 640, 480);
        let mut marked = clean.clone();
        watermark(&mut marked, &mask);
        let (restored, error) = inpaint(Some(&model), &marked, &mask);
        assert_eq!(error, None);
        let before = mean_error(&marked, &clean, &mask);
        let after = mean_error(&restored, &clean, &mask);
        assert!(after < before / 3.0, "înainte {before}, după {after}");
        assert_eq!(restored.get_pixel(5, 5), clean.get_pixel(5, 5));
    }
}
