# The BOMBOCLAT Baking Engine

A Blender add-on that combines **every material of a mesh into one material**
with one set of square texture atlases: Base Color (+ Alpha), Roughness,
Metallic, Normal, Emission and Height, plus every other Principled BSDF input
the model uses (Subsurface, Specular, Anisotropic, Transmission, Coat, Sheen,
Thin Film, IOR, Diffuse Roughness).

No ray tracing and no re-unwrapping. The parts of each texture that the
model actually uses are cut out and packed together **texel for texel**, so
the result looks like the original.

## Features

- **Lossless.** Textures are copied byte for byte; UV islands are only moved,
  never resized.
- **Auto size.** Picks the smallest square atlas (1K, 2K, 4K, 8K, 16K) that
  keeps every texel.
- **Only what is used.** Unused parts of textures take no space.
- **Every Principled input.** An input only gets an atlas if some material
  textures it or uses different values. If every material uses the same value,
  it is simply set on the new material. Inputs nobody changed are skipped.
  Single-value inputs share textures (three per texture), values outside 0-1
  (IOR, thin film thickness...) are stored as a range and scaled back by the
  material, and Coat Normal is handled like Normal.
- **Deduplication.** Materials that are copies of each other are stored once,
  and flat-colour parts shrink to a tiny block.
- **Copies what you see.** Reads the node setup the way Blender renders it:
  channel-packed textures, Color Ramps, Mix/Math nodes, Mapping (tiling),
  Normal Map and Bump nodes, displacement, emission and alpha. Textures that
  are wired unusually are reproduced as Blender shows them.
- **Node groups.** Follows textures and settings into node groups (also
  nested ones), so Character Creator / CC4 and other imported shaders work.
  See-through overlay materials (eye occlusion, tear lines) stay invisible.
- **Node math.** Setups that combine several textures (AO multiply, masks,
  Hue/Saturation/Value, Gamma, Brightness/Contrast, Math, Mix with a texture
  factor, Color Ramps...) are calculated per texel exactly like Blender's
  nodes. Mix Shaders on the output (e.g. a blood layer over skin) are blended
  per texel too. Simple texture hookups are still copied byte for byte.
- **Procedural logic.** Texture coordinates computed with math (scaled irises,
  pivots, tiling groups), UV-based masks, colour attributes (vertex colours),
  RGB Curves and Map Range are followed per texel. Bump nodes are baked into
  the normal map, Normal Map strength can come from a texture, and normals
  blended by a Mix Shader are blended too.
- **Matches Blender's quirks.** Colour textures with transparency are dimmed
  by their alpha like Blender does, and glass-like materials (corneas) keep
  raytraced transmission. View-dependent nodes (Fresnel, Layer Weight) use
  their head-on value.
- **Fix Textures.** One click repairs common mistakes in the original
  materials: adds missing Normal Map nodes (normal maps plugged straight into
  Normal), adds a Displacement node for height maps plugged straight into the
  output, sets roughness / metallic / normal / height maps to Non-Color and
  colour maps to sRGB. Lists every change; Ctrl+Z undoes it.
- **Material check.** Lists every material and what feeds each channel, with
  warnings and hints about wiring that looks like a mistake.
- Everything is stored inside the `.blend`; PNG export is optional.
- The original object is never modified.

## Installation

Requires **Blender 4.2 or newer**.

1. Download `bomboclat_baking_engine-<version>.zip`.
2. In Blender: **Edit > Preferences > Get Extensions**, open the drop-down menu
   (top right), choose **Install from Disk...** and pick the zip.
3. The panel appears in the 3D Viewport sidebar (**N**) under the **BOMBOCLAT** tab.

## Usage

1. Select the mesh.
2. Optional: click **Check Materials** and look through the **Material Check**
   list.
3. Optional: click **Fix Textures** to repair wiring mistakes in the original
   (the list of changes is written to `BOMBOCLAT_FIX_REPORT.txt`).
4. Click **Build Atlas**.

You get a copy of the object named `<name>_ATLAS` with one UV map and one
material whose Principled BSDF uses the new textures. A build log is written
to the Text Editor as `PBR_ATLAS_BUILD_LOG.txt`.

## Settings

| Setting | Default | What it does |
|---|---|---|
| Atlas Size | Auto | Auto picks the smallest size that keeps every texel. A number caps the size. |
| Lossless | on | Never shrink anything. If the textures do not fit, stop and list the materials that need the room. |
| Raw Data Maps | off | Off: roughness/metallic/height textures left on sRGB look exactly like in Blender. On: use their raw values, like a game engine. |
| Padding | 8 px | Real texture kept around every piece so mipmaps do not bleed. |
| Hide Original | on | Hides the source object afterwards. |
| Also Save PNG Files | off | Also writes the atlases as PNG files to a folder. |

## License

Proprietary. Copyright (c) 2026 HeatedJester586. All rights reserved.
See [LICENSE](LICENSE).
