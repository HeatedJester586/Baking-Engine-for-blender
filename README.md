# The BOMBOCLAT Baking Engine

A Blender add-on that combines **every material of a mesh into one material**
with one set of square texture atlases: Base Color (+ Alpha), Roughness,
Metallic, Normal, Emission and Height.

No ray tracing and no re-unwrapping. The parts of each texture that the
model actually uses are cut out and packed together **texel for texel**, so
the result looks like the original.

## Features

- **Lossless.** Textures are copied byte for byte; UV islands are only moved,
  never resized.
- **Auto size.** Picks the smallest square atlas (1K, 2K, 4K, 8K, 16K) that
  keeps every texel.
- **Only what is used.** Unused parts of textures take no space.
- **Deduplication.** Materials that are copies of each other are stored once,
  and flat-colour parts shrink to a tiny block.
- **Copies what you see.** Reads the node setup the way Blender renders it:
  channel-packed textures, Color Ramps, Mix/Math nodes, Mapping (tiling),
  Normal Map and Bump nodes, displacement, emission and alpha. Textures that
  are wired unusually are reproduced as Blender shows them.
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
3. Click **Build Atlas**.

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
