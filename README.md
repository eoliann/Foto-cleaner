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

# Foto Cleaner

Aplicație desktop Windows, portabilă și offline, care curăță și corectează o fotografie și salvează **o singură imagine** finală. Pornește de la [Foto acte 3x4](https://github.com/eoliann/Foto-acte-3x4), dar exportă o singură imagine în loc de coala cu 6 fotografii.

## Funcționalități

**Corecție automată.** Butonul **✨ Auto-corecție** analizează fotografia, o îndreaptă și o încadrează după față, și setează automat expunerea, balansul de alb (temperatură și nuanță), umbrele, luminile, contrastul, saturația, reducerea zgomotului și accentuarea. Dacă persoana a fost detectată de AI, corecția se optimizează după tonul pielii. Toate valorile pot fi apoi reglate manual.

**Corecții manuale:**

- lumină: expunere, luminozitate, contrast, lumini, umbre;
- culoare: saturație, vibranță, temperatură, nuanță;
- detalii: claritate (contrast local), accentuare, reducere zgomot.

**Eliminare impurități:**

- netezirea pielii, aplicată doar pe zonele de piele (și doar pe persoană, când AI-ul a detectat-o), cu păstrarea marginilor;
- pensulă pentru pete: click pe coșuri, pete sau praf și zona este reconstruită din textura din jur;
- reducerea zgomotului / granulației cu un filtru care păstrează marginile.

**Detectarea feței (AI local, YuNet):**

- îndreptare automată după linia ochilor, plus îndreptare manuală de ±15°;
- încadrare automată pentru foto act (capul ocupă ~65% din înălțime, ochii la ~42% de sus) și pentru celelalte formate; se aplică și la schimbarea formatului;
- auto-corecția reglează expunerea după pielea feței;
- netezirea pielii se aplică doar pe față;
- opțiunea „Arată fața detectată” afișează fața și reperele (ochi, nas, gură).

**Fundal (AI local, MODNet):**

- original, transparent (PNG), alb, gri deschis, albastru deschis, orice culoare;
- fundal estompat (efect portret);
- înlocuire cu o altă imagine;
- reglarea fermității marginii și curățarea automată a „halo”-ului de pe contur.

**Format final:**

- rezoluția originală, fără decupare;
- foto act 3x4 cm (354x472 px, 300 DPI), cu zoom și poziționare;
- decupare 1:1, 3:4, 4:3, 2:3 sau 16:9;
- salvare JPEG (300 DPI) sau PNG; transparența se salvează automat ca PNG;
- printare directă: foto act la mărime reală 3x4 cm, celelalte formate încadrate în pagină.

Ca la aplicația inițială, poți roti imaginea, poți trage fotografia peste fereastră, orientarea EXIF este corectată automat și totul rulează local, fără internet.

## Utilizare

1. Deschide `Foto-cleaner.exe` și încarcă fotografia (sau trage-o peste fereastră).
2. Apasă **✨ Auto-corecție**, apoi reglează fin dacă e nevoie. Bifează **Arată originalul** ca să compari.
3. Pentru pete: bifează **Pensulă pete**, alege mărimea și dă click pe fiecare pată.
4. Alege fundalul. Prima procesare AI durează câteva secunde.
5. Alege formatul final și apasă **Salvează imaginea** sau **Printează direct**.

Previzualizarea folosește o copie micșorată, ca să răspundă rapid. Exportul este procesat la rezoluție completă, cu aceleași setări.

## Dezvoltare

Este necesar [Rust](https://www.rust-lang.org/tools/install) 1.94 sau mai nou.

```powershell
cargo run
cargo fmt -- --check
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
```

Structura codului:

- `src/main.rs`: interfața (eframe/egui), firele de lucru, salvarea și printarea;
- `src/processing.rs`: filtrele, retușul, auto-corecția, compunerea fundalului și încadrarea;
- `src/ai.rs`: segmentarea persoanei cu MODNet;
- `src/face.rs`: detectarea feței și a reperelor cu YuNet.

## Build portabil

```powershell
.\build-portable.ps1
```

Executabilul independent va fi creat în `dist\Foto-cleaner.exe`. Un tag `v*` (de exemplu `v1.0.0`) publică automat un GitHub Release prin workflow-ul din `.github/workflows/windows.yml`.

## Tehnologii

Rust cu `eframe/egui`, crate-ul `image` și filtre proprii (filtru ghidat, unsharp mask, interpolare Shepard pentru retuș). Rulează două modele AI prin runtime-ul `RTen`: MODNet pentru eliminarea fundalului (Apache-2.0) și YuNet pentru detectarea feței (MIT, ~230 KB). Ambele sunt incluse în executabil; detaliile sunt în [`assets/README.md`](assets/README.md).
