![Followers](https://img.shields.io/github/followers/eoliann?style=plastic&color=green)
![Watchers](https://img.shields.io/github/watchers/eoliann/Foto-cleaner?style=plastic)
![Stars](https://img.shields.io/github/stars/eoliann/Foto-cleaner?style=plastic)

[![Donate](https://img.shields.io/badge/Donate-PayPal-blue?style=plastic)](https://www.paypal.com/donate/?hosted_button_id=PTH2EXUDS423S)
[![Donate](https://img.shields.io/badge/Donate-Revolut-8A2BE2?style=plastic)](https://revolut.me/adriannm9)
[![Donate](https://img.shields.io/badge/Donate-KoFi-green?style=plastic)](https://ko-fi.com/eoliann)

![Release Date](https://img.shields.io/github/release-date/eoliann/Foto-cleaner?style=plastic)
![Last Commit](https://img.shields.io/github/last-commit/eoliann/Foto-cleaner?style=plastic)
![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/eoliann/Foto-cleaner/total?style=plastic)
[![Downloads latest](https://img.shields.io/github/downloads/eoliann/Foto-cleaner/latest/total?style=plastic)](https://github.com/eoliann/Foto-cleaner/releases/latest/download/Foto-cleaner-windows.zip)

![OS](https://img.shields.io/badge/OS-Windows-blue?style=plastic)
![Lang](https://img.shields.io/badge/Lang-Rust-magenta?style=plastic)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=plastic)](LICENSE.md)

<details open>
<summary><b>Downloads · daily (cumulative)</b></summary>

<img src=".github/downloads/chart.svg?v=2026111602" alt="Daily download progress" width="760">

</details>
<details>
<summary><b>Downloads · monthly</b></summary>

<img src=".github/downloads/chart-monthly.svg?v=2026111602" alt="Monthly downloads" width="760">

</details>
<details>
<summary><b>Downloads · yearly</b></summary>

<img src=".github/downloads/chart-yearly.svg?v=2026111602" alt="Yearly downloads" width="760">

</details>


# Foto Cleaner

A portable, offline Windows desktop app that cleans and corrects a photo and saves **a single** final image. It starts from [Foto acte 3x4](https://github.com/eoliann/Foto-acte-3x4), but exports one image instead of the sheet with 6 photos.

## Features

**Automatic correction.** The **✨ Auto-correction** (*Auto-corecție*) button analyzes the photo, straightens and frames it based on the face, and automatically sets exposure, white balance (temperature and tint), shadows, highlights, contrast, saturation, noise reduction and sharpening. If the person was detected by the AI, the correction is optimized for the skin tone. All values can then be adjusted manually.

**Manual corrections:**

- light: exposure, brightness, contrast, highlights, shadows;
- color: saturation, vibrance, temperature, tint;
- detail: clarity (local contrast), sharpening, noise reduction.

**Blemish removal:**

- skin smoothing, applied only to skin areas (and only on the person, when the AI detected one), preserving edges;
- spot brush: click on pimples, spots or dust and the area is reconstructed from the surrounding texture;
- noise / grain reduction with an edge-preserving filter.

**Watermark / object removal (local AI, MI-GAN):**

- paint with the brush over a watermark, text, the date stamped by the camera or a small unwanted object, then click **Remove marked area** (*Elimină zona marcată*);
- the area is reconstructed by the AI from its surroundings, at the photo's full resolution; marked areas far apart are processed separately;
- you can undo the last 5 removals; if the AI model cannot be used, the area is filled with the classic method;
- use this feature only on your own photos or photos you have the right to modify.

**Face detection (local AI, YuNet):**

- automatic straightening along the eye line, plus manual straightening of ±15°;
- automatic framing for ID photos (head takes ~65% of the height, eyes at ~42% from the top) and for the other formats; also applied when the format changes;
- auto-correction adjusts exposure based on the facial skin;
- skin smoothing is applied only on the face;
- the "Show detected face" option displays the face and landmarks (eyes, nose, mouth).

**Background (local AI, MODNet):**

- original, transparent (PNG), white, light gray, light blue, any color;
- blurred background (portrait effect);
- replacement with another image;
- edge firmness adjustment and automatic cleanup of the "halo" along the outline.

**Final format:**

- original resolution, no cropping;
- ID photo 3x4 cm (354x472 px, 300 DPI), with zoom and positioning;
- 1:1, 3:4, 4:3, 2:3 or 16:9 crop;
- save as JPEG (300 DPI) or PNG; transparency is automatically saved as PNG;
- direct printing: ID photo at real 3x4 cm size, other formats fitted to the page.

As with the original app, you can rotate the image, drag the photo onto the window, EXIF orientation is corrected automatically, and everything runs locally, without internet.

## Usage

1. Open `Foto-cleaner.exe` and load the photo (or drag it onto the window).
2. Click **✨ Auto-correction**, then fine-tune if needed. Check **Show original** (*Arată originalul*) to compare.
3. For spots: check **Spot brush** (*Pensulă pete*), choose the size and click on each spot.
4. For watermarks or objects: check **Watermark brush** (*Pensulă watermark*), paint over the area and click **Remove marked area**. For large areas, removing in pieces gives a better result.
5. Choose the background. The first AI processing takes a few seconds.
6. Choose the final format and click **Save image** (*Salvează imaginea*) or **Print directly** (*Printează direct*).

The preview uses a downscaled copy so it responds quickly. The export is processed at full resolution with the same settings.

## Development

[Rust](https://www.rust-lang.org/tools/install) 1.94 or newer is required.

```powershell
cargo run
cargo fmt -- --check
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
```

Code structure:

- `src/main.rs`: the interface (eframe/egui), worker threads, saving and printing;
- `src/processing.rs`: filters, retouching, auto-correction, background compositing and framing;
- `src/ai.rs`: person segmentation with MODNet;
- `src/face.rs`: face and landmark detection with YuNet;
- `src/inpaint.rs`: watermark and object removal with MI-GAN (plus a classic fallback fill).

## Portable build

```powershell
.\build-portable.ps1
```

The standalone executable will be created at `dist\Foto-cleaner.exe`. A `v*` tag (for example `v1.0.0`) automatically publishes a GitHub Release through the workflow in `.github/workflows/windows.yml`. The release text is taken from `release-notes/<tag>.md` (for example `release-notes/v1.2.0.md`), which must be added in the commit before the tag is created.

## Technologies

Rust with `eframe/egui`, the `image` crate and custom filters (guided filter, unsharp mask, Shepard interpolation for retouching). It runs three AI models through the `RTen` runtime: MODNet for background removal (Apache-2.0), YuNet for face detection (MIT, ~230 KB) and MI-GAN for watermark removal (MIT, ~27 MB). All are embedded in the executable; details are in [`assets/README.md`](assets/README.md).
