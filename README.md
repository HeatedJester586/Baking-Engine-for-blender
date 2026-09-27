# PBR Atlas Baker for Blender

Merge **every material of a mesh into one set of PBR textures**: Base Color,
Roughness, Metallic and Normal, packed into a single square atlas
(1K to 8K) on a copy of the mesh that uses one master material.

It was built for imported game models (glTF/GLB rips) that come with dozens
of material slots, channel-packed ORM textures and floating detail geometry
(strings, bridges, knobs) that ruin ordinary "Selected to Active" bakes.

![Blender 4.2+](https://img.shields.io/badge/Blender-4.2%2B-orange) ![GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)

## Features

- **GPU rasterizer (default).** Each source texture is copied directly into
  the new UV layout on the graphics card. No rays are cast, so floating
  geometry cannot spray black noise or shadows onto the surface underneath.
- **Cycles surface bake (alternative).** A native Cycles bake of each surface
  onto itself. Base Color, Roughness and Metallic go through an Emission
  shader, so metal parts do not bake black.
- **Understands real-world node trees.** Channel-packed ORM textures
  (Separate Color R/G/B), Mapping nodes (tiling), UV Map nodes, the
  glTF importer's Mix/Math "factor" nodes, Invert, Normal Map strength, and
  reroutes and muted nodes.
- **Correct normal maps.** Tangent-space normals are rotated into the atlas
  UVs' tangent frame, including mirrored islands. Plain copying gives wrong
  lighting on every island that Smart UV Project rotated.
- **Correct colour.** Base Color is written as sRGB; data maps as Non-Color.
- **Edge padding** so mipmaps never pull in background colour, plus an edge
  pass that keeps sub-pixel parts (strings, fret wires) from disappearing.
- **Diagnostics report** that lists, for every slot, exactly which image and
  channel each PBR input comes from, with warnings for missing files,
  unsupported nodes and wrong colour spaces.
- Keeps custom split normals on the GPU path. The original object is never
  modified.

## Installation

Requires **Blender 4.2 or newer**. The automated tests run on Blender 5.0.

1. Download `pbr_atlas_baker-<version>.zip` from the
   [Releases](https://github.com/HeatedJester586/Baking-Engine-for-blender/releases) page
   (or build it yourself, see [Building the zip](#building-the-zip)).
2. In Blender: **Edit > Preferences > Get Extensions**, open the drop-down
   menu (top right) and choose **Install from Disk...**, then pick the zip.
   You can also drag the zip into the Blender window.
3. The panel appears in the 3D Viewport sidebar (**N**) under the **Atlas** tab.

## Usage

1. Save your `.blend` file (textures are written next to it by default).
2. Select the mesh (for example `guitar.036`) in Object Mode.
3. Open **Sidebar > Atlas** and click **Diagnose Materials**. Check the
   report in the Text Editor (`PBR_ATLAS_REPORT.txt`) and fix any warnings
   you care about, such as missing images.
4. Pick a resolution and click **Bake Atlas**.

You get a new object `<name>_ATLAS` with one UV map (`UV_Atlas`), one material
(`M_<name>_ATLAS`) and four PNGs in the output folder:

```
atlas_textures/
  guitar.036_ATLAS_BaseColor_4096.png   (sRGB)
  guitar.036_ATLAS_Roughness_4096.png   (Non-Color)
  guitar.036_ATLAS_Metallic_4096.png    (Non-Color)
  guitar.036_ATLAS_Normal_4096.png      (Non-Color, OpenGL / +Y)
```

Baking again replaces the previous `_ATLAS` object and images.

### Running it as a script

`pbr_atlas_baker/baker.py` also works without installing the add-on. Open it
in Blender's **Text Editor**, change the settings at the bottom of the file
(`OBJECT_NAME`, `METHOD`, `RESOLUTION`, ...) and press **Run Script**.

### Settings

| Setting | Default | Notes |
|---|---|---|
| Method | GPU Rasterizer | Use Cycles if you prefer a native bake (see below). |
| Resolution | 4096 | 1024, 2048, 4096 or 8192. The atlas is always square. |
| Edge Padding | 16 px | How far each UV island is extended outwards. |
| Angle Limit | 66° | Smart UV Project angle limit for the new UVs. |
| Island Margin | 0.005 | Space between UV islands. |
| Output Folder | `//atlas_textures/` | `//` means next to the `.blend` file. |
| Pack into .blend | off | Also embed the PNGs in the `.blend`. |
| Hide Original | on | Hides the source object afterwards. |
| Samples (Cycles) | 8 | Only for anti-aliasing; no lighting is baked. |
| Clear Custom Normals (Cycles) | on | Works around black normal bakes on meshes with custom split normals. |

### GPU or Cycles?

| | GPU Rasterizer | Cycles Surface Bake |
|---|---|---|
| Speed | Fast (one GPU draw per channel) | Slower (one Cycles bake per channel) |
| Floating geometry | Unaffected | Unaffected (no Selected to Active) |
| Node support | Common setups (listed above) | Anything Cycles can render |
| Custom split normals | Kept | Cleared on the copy (optional) |

If the diagnostics report warns about an unsupported node that matters (for
example a procedural texture or a Color Ramp), use the Cycles method.

## How it works

1. The object is duplicated (object and mesh data). Materials are shared,
   never edited permanently.
2. A new UV map `UV_Atlas` is created with **Smart UV Project**. The angle
   limit is passed in *radians*; passing degrees silently collapses every UV
   to (0, 0).
3. **GPU:** for each channel, every material slot's triangles are drawn at
   their atlas UV position, sampling the source texture at the original UV
   through a custom shader. The shader handles channel selection, the
   factor math, wrap mode, sRGB output and the normal-map tangent rotation.
   The result is read back, padded and saved.
   **Cycles:** an image node is added to every material and each channel is
   baked onto the surface itself.
4. All old UV maps and slots on the copy are replaced by one master material
   that uses the four atlases.

## Limitations

- The GPU method only reads Principled BSDF inputs. Procedural textures,
  Color Ramps, vertex colours and non-trivial math are reported and
  approximated. Use Cycles for those.
- Alpha / opacity, emission and ambient occlusion are not baked yet.
- UDIM (tiled) images are not supported.
- Normal maps are assumed to be OpenGL style (+Y), which is Blender's and
  glTF's convention.
- Tiny parts get few pixels in the atlas, because Smart UV Project sizes
  islands by surface area. Use a higher resolution if thin parts look blurry.

## Troubleshooting

- **Pink areas:** a source image could not be loaded. The diagnostics report
  names the file. Use *File > External Data > Find Missing Files*.
- **"GPU readback check failed":** the GPU backend could not draw. Try another
  backend in *Preferences > System*, or use the Cycles method.
- **Roughness/Metallic look too dark or too glossy:** the report warns when a
  data texture is tagged sRGB. Blender renders it that way too; set the
  source image to Non-Color if that is wrong.
- **Nothing happens / errors:** open *Window > Toggle System Console* to see
  the `[PBR Atlas]` log.

## Development

### Tests

`tests/smoke_test.py` builds a small multi-material mesh covering ORM
packing, metal, tiling, rotated and mirrored normal-mapped islands, floating
geometry and an empty slot. It bakes the mesh with both methods and checks
that the GPU result matches the Cycles result.

```bash
blender -b --factory-startup --python tests/smoke_test.py
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
