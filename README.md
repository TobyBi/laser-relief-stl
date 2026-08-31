# Laser Relief STL

Convert a grayscale image into a binary STL. The repository includes the 70 mm
Aghosh memento grayscale master as a working example.

Dark pixels become raised geometry. `--features-only` traces smooth sub-pixel
contours and produces separate closed extrusions with no background plate,
matching a features-only vector STL. The original bas-relief slab and an
optional square-pixel extrusion mode remain available. The generated STL uses
millimetres by convention, although STL itself does not store a unit declaration.

## Install

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

If the `py` launcher or virtual environments are unavailable, install and run
with the Python command that works on your machine instead:

```powershell
python -m pip install --user -r requirements.txt
```

## Make the dark-background panels engravable

The binary-state and BPF source images use bright coloured detail on dark
backgrounds. A global grayscale conversion makes each background one solid
raised rectangle. The helper below uses colour-aware local contrast only in
those three panel regions, placing their bright rings and labels as dark
linework on white:

```powershell
python enhance_aghosh_dark_panels.py `
  examples\aghosh_memento.png `
  examples\aghosh_memento_grayscale_70mm_600dpi.png `
  examples\aghosh_memento_grayscale_enhanced_70mm_600dpi.png `
  --thin-px 1
```

Use the enhanced PNG as the input to `image_to_relief_stl.py`. This selective
conversion is important: inverting the complete artwork would also reverse all
ordinary black-on-white diagrams and text.

`--thin-px 1` removes one raster pixel from each edge of heavy linework
extracted from the binary-state and BPF panels only. Fine labels and
annotations are preserved at their original width. Omit it for the original
widths; `--thin-px 2` is more aggressive.

Add `--thin-full-image` to apply the same heavy-stroke-only thinning across
the complete artwork. The main name/header is restored at its original width,
while all four paper-title bands receive a light half-pixel-equivalent erosion
to reduce their binary-extrusion weight without breaking small words.

For the full-image version, the very small fourth-harmonic paper title needs a
finer contour grid. Use `--resolution 1200 --simplify 0.015 --smoothing 0.01
--min-feature-area 0.003` when creating its STL.

## Generate the 70 mm features-only memento

```powershell
python image_to_relief_stl.py `
  examples\aghosh_memento_grayscale_enhanced_70mm_600dpi.png `
  aghosh_memento_features_only_70mm.stl `
  --features-only `
  --threshold 0.12 `
  --size 70 `
  --relief 0.5 `
  --resolution 1024 `
  --simplify 0.035 `
  --smoothing 0.025
```

This creates smooth constant-height dark features between Z = 0 and Z = 0.5 mm;
there is no 70 mm square base. Raising `--threshold` removes more light-gray
content. `--simplify` and `--smoothing` control outline cleanup in millimetres.
For a horizontally mirrored back-face version, add `--mirror`.

```powershell
py image_to_relief_stl.py `
  examples\aghosh_memento_grayscale_70mm_600dpi.png `
  aghosh_memento_features_only_70mm_mirrored.stl `
  --features-only `
  --threshold 0.12 `
  --size 70 `
  --relief 0.5 `
  --resolution 1024 `
  --simplify 0.035 `
  --smoothing 0.025 `
  --mirror
```

With the supplied example and threshold 0.12, the high-quality 1024 preset
produces about 128,000 triangles and a 6.4 MB binary STL. Use resolution 512 for
a faster, smaller draft. Import the result as millimetres.

## Generate the original bas-relief slab

Omit `--features-only` and provide a base thickness:

```powershell
py image_to_relief_stl.py `
  examples\aghosh_memento_grayscale_70mm_600dpi.png `
  aghosh_memento_relief_70mm.stl `
  --size 70 `
  --base 0.8 `
  --relief 0.6 `
  --threshold 0.12 `
  --resolution 512
```

## Options

```text
--size         XY size in millimetres; default 70
--base         Base thickness in millimetres; default 0.8
--relief       Feature height or maximum added relief; default 0.6
--resolution   Height-map samples per axis; default 512
--threshold    Minimum retained darkness from 0 to 1; default 0.12
--features-only  Smoothly trace and extrude only dark features
--pixel-features  Use square raster cells instead of smooth contours
--simplify     Contour simplification tolerance in mm; default 0.035
--smoothing    Contour rounding radius in mm; default 0.025
--min-feature-area  Discard smaller features in mm squared; default 0.01
--mirror       Mirror horizontally for back-face use
```
