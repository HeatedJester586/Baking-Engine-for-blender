# PBR Atlas Baker for Blender

Combine **every material of a mesh into one material** with one set of
square PBR texture atlases: Base Color (+ Alpha), Roughness, Metallic,
Normal, Emission and Height (displacement).

No baking, no ray tracing, no re-unwrapping. The textures are **cut and
repacked texel for texel**:

1. The object is duplicated. The original is never touched.
2. For every material, the add-on finds which parts of its textures the UVs
   actually use (UV islands → texel rectangles, "chunks") and cuts exactly
   those parts out of every channel.
3. The chunks are packed as tightly as possible into one square atlas. The
   **smallest power-of-two size that holds everything at full resolution**
   is picked automatically (up to 8K or 16K).
4. The UVs are only **moved**, never re-unwrapped or resized, so every face
   still sees exactly the texels it saw before.
5. All materials are replaced by one material with the atlases plugged into
   its Principled BSDF. The atlases are stored inside the `.blend`.

![Blender 4.2+](https://img.shields.io/badge/Blender-4.2%2B-orange) ![GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)

## Features

- **Lossless.** Texels are copied byte for byte. Nothing is resampled
  unless it has to be (see below), and texel density is kept.
- **Only what is used.** A material that uses a quarter of its 4K texture
  only takes a quarter of the space in the atlas.
- **Deduplication.** Materials with identical textures (for example
  `MAT_Keytar.063` … `.074` whose images are copies of each other) are
  detected by their pixel content and stored once. Overlapping or mirrored
  UV islands share their texels too.
- **Flat colours shrink for free.** A material or chunk that is one solid
  colour becomes a tiny block, which loses nothing.
- **Understands real-world node trees.** Channel-packed ORM textures
  (Separate Color R/G/B), Color Ramps (evaluated exactly), Mapping nodes
  (tiling), UV Map nodes, the glTF importer's Mix/Math "factor" nodes,
  Invert, Normal Map strength, reroutes and muted nodes.
- **Looks like the original.** Base Color and Emission atlases are sRGB,
  data maps are Non-Color. Roughness / Metallic / Height textures left on
  sRGB are gamma-decoded exactly like Blender renders them, so the atlas
  looks the same as the original materials. Tick **Raw Data Maps** to keep
  their raw values instead (game-engine style). Normal maps are always read
  raw.
- **Copies miswired materials faithfully.** A normal texture plugged straight
  into Normal (no Normal Map node) is not a normal map to Blender: it uses the
  colour as a fixed direction and ignores that material's displacement. The
  atlas reproduces exactly that look (checked against Cycles to within about
  1°), while correctly wired normal maps are copied byte for byte. *Check
  Materials* points these textures out and says whether they look like real
  normal maps, in case the wiring was a mistake.
- **Padding** of real neighbouring texels around every island, so
  mipmaps do not bleed.
- **Material checker.** *Check Materials* lists every slot in the panel:
  which textures feed Base Color, Alpha, Roughness, Metallic and Normal,
  which materials have emission or displacement, and any warnings (missing
  files, unsupported nodes). A full report also goes to the Text Editor.
- **Displacement.** Textures wired into the Material Output's Displacement,
  directly or through a Displacement node, are packed into a Height atlas
  that drives one Displacement node. Materials without displacement stay
  flat; different midlevel/scale settings are converted exactly.
- Keeps custom split normals and every other mesh attribute.

## Installation

Requires **Blender 4.2 or newer**. The automated tests run on Blender 5.0.

1. Download `pbr_atlas_baker-<version>.zip` from the
   [Releases](https://github.com/HeatedJester586/Baking-Engine-for-blender/releases) page
   (or build it yourself, see [Building the zip](#building-the-zip)).
2. In Blender: **Edit > Preferences > Get Extensions**, open the drop-down
   menu (top right) and choose **Install from Disk...**, then pick the zip.
   Installing a newer zip replaces the old version.
3. The panel appears in the 3D Viewport sidebar (**N**) under the **Atlas** tab.

## Usage

1. Select the mesh (for example `guitar.036`).
2. Optional: click **Check Materials** and look through the **Material
   Check** list in the panel (full report: `PBR_ATLAS_REPORT.txt` in the Text
   Editor).
3. Click **Build Atlas**.

You get a new object `<name>_ATLAS` with one UV map (`UV_Atlas`) and one
material (`M_<name>_ATLAS`):

| Texture | Colour space | Plugged into |
|---|---|---|
| `<name>_ATLAS_BaseColor_<size>` | sRGB (alpha channel = Alpha, if any) | Base Color, Alpha |
| `<name>_ATLAS_Roughness_<size>` | Non-Color | Roughness |
| `<name>_ATLAS_Metallic_<size>` | Non-Color | Metallic |
| `<name>_ATLAS_Normal_<size>` | Non-Color, OpenGL (+Y) | Normal Map > Normal |
| `<name>_ATLAS_Emission_<size>` (only if something glows) | sRGB | Emission Color |
| `<name>_ATLAS_Height_<size>` (only if something is displaced) | Non-Color | Displacement node > Material Output |

Nothing is written to disk unless you tick **Also Save PNG Files**. To
export later, use *Image > Save As* in the Image Editor or
*File > External Data > Unpack Resources*. Building again replaces the
previous `_ATLAS` object and textures.

### Running it as a script

`pbr_atlas_baker/baker.py` also works without installing the add-on. Open it
in Blender's **Text Editor**, change the settings at the bottom of the file
(`OBJECT_NAME`, `MAX_ATLAS_SIZE`, `PADDING`, `OUTPUT_DIR`) and press
**Run Script**.

### Settings

| Setting | Default | Notes |
|---|---|---|
| Atlas Size | Auto | Auto picks the smallest power of two that keeps every texel (up to 16384). A number caps the size. |
| Lossless | on | Never shrink anything. If the textures do not fit, stop and list which materials need the room. |
| Raw Data Maps | off | Off: sRGB-tagged roughness/metallic/height look exactly like in Blender. On: keep their raw values. |
| Padding | 8 px | Real texels kept around every UV island. |
| Hide Original | on | Hides the source object afterwards. |
| Also Save PNG Files | off | Also write the atlases as PNGs to **Folder** (`//` = next to the `.blend`). |

## When texels are not copied 1:1

The tool only changes texels when there is no lossless option, and says so
in the console:

- **Different resolutions in one material** (for example a 2K Base Color
  with a 1K ORM): a face has only one UV, so the smaller texture is
  upscaled with bilinear filtering onto the larger one's grid. Nothing is
  lost, but those texels are interpolated, not copied.
- **Too big for the maximum size** (only with *Lossless* off): if all chunks
  together do not fit, everything is scaled down evenly and a warning tells
  you by how much. With *Lossless* on, nothing is built and you get a list of
  the materials that need the room instead.
- **Heavily tiled textures:** a texture repeated 10× across a face needs 10×
  its size in the atlas. *Check Materials* warns about this in advance. If
  it is larger than the maximum, that part is scaled down (averaged like a
  mipmap, so it stays smooth) with a warning.
- **Factor math, colour-space fixes and Normal Map strength** are applied
  to the texels, because the atlas has to look like the original material.

## Limitations

- Only Principled BSDF inputs are read. Procedural textures and vertex
  colours are reported and approximated.
- Ambient occlusion, transmission, subsurface, clearcoat and sheen are not
  transferred.
- UDIM (tiled) images are not supported.
- Normal maps are assumed to be OpenGL style (+Y), Blender's and glTF's
  convention.
- Atlas sizes are powers of two, so the atlas can be up to 4× larger than
  the texels it holds. The fill percentage is printed after every build.

## Troubleshooting

- **Pink areas:** a source image could not be loaded. The diagnostics report
  names the file. Use *File > External Data > Find Missing Files*.
- **Warnings:** open *Window > Toggle System Console* to see the
  `[PBR Atlas]` log.

## Development

### Tests

`tests/smoke_test.py` builds a mesh with random-noise textures covering ORM
packing, duplicated materials, partly used textures, tiling, a rotated
normal-mapped island, alpha, emission, flat materials, an empty slot and 8
UV maps. It then checks that every face samples exactly its original texels
through the new UVs.

`tests/normal_test.py` bakes the real shading normal with Cycles for an
original and its atlas (normal textures wired straight into Normal and through
a Normal Map node, on a rotated cube) and checks they match.

```bash
blender -b --factory-startup --python tests/smoke_test.py
blender -b --factory-startup --python tests/normal_test.py
# or, with the bpy module from PyPI:
pip install bpy && python tests/smoke_test.py
```

### Building the zip

```bash
blender --command extension build --source-dir pbr_atlas_baker --output-dir dist
```

## License

[GPL-3.0-or-later](LICENSE), the license Blender requires for add-ons
distributed on extensions.blender.org.
