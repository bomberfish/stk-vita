# SuperTuxKart for PlayStation Vita

Packaging for a homebrew-enabled ("modded") PS Vita or PSTV.

Title ID: `STKV00001`

## What you need on the Vita

* Custom firmware — HENkaku / h-encore / Ensō
* [VitaShell](https://github.com/TheOfficialFloW/VitaShell) to install the vpk
  and to copy files over USB or FTP
* Roughly 1.6 GB free on `ux0:`

## Installing

1. Copy `SuperTuxKart.vpk` to the Vita and open it in VitaShell, then press
   ✕ to install.

2. Create `ux0:data/stk/` and copy the game data into it, so that you end up
   with this layout:

   ```
   ux0:data/stk/data/          <- contents of stk-code/data
   ux0:data/stk/stk-assets/    <- contents of the stk-assets repository
   ux0:data/stk/save/          <- created on first run (config, saves, addons)
   ```

   `data/` is about 58 MB and `stk-assets/` about 1.4 GB, which is why they are
   not bundled in the vpk. Keeping them out also means you can update the
   assets without reinstalling the app.

   In VitaShell, press SELECT to start USB or FTP mode and copy the two
   directories across.

3. Launch SuperTuxKart from the LiveArea.

If the game exits immediately, `ux0:data/stk/data/` is the first thing to
check — the game looks for a version stamp inside it and will refuse to start
if the directory is missing or incomplete.

## Building

The Vita build needs five libraries that are not part of this tree. Install
each into `$VITASDK` before configuring SuperTuxKart:

| Library | Source | Notes |
| --- | --- | --- |
| [math-neon](https://github.com/Rinnegatamante/math-neon) | `make install` | |
| [SceShaccCgExt](https://github.com/bythos14/SceShaccCgExt) | CMake | Needs `-DCMAKE_C_FLAGS=-std=gnu17` on GCC 15: taihen's `TAI_CONTINUE` relies on old-style prototypes, which C23 rejects |
| [vitaShaRK](https://github.com/Rinnegatamante/vitaShaRK) | `make install` | Requires SceShaccCgExt |
| [vitaGL](https://github.com/Rinnegatamante/vitaGL) | `make NO_DEBUG=1 HAVE_GLSL_SUPPORT=1 install` | Provides OpenGL; requires vitaShaRK and math-neon |
| [SDL2 (Northfear fork)](https://github.com/Northfear/SDL/tree/vitagl) | CMake, `-DVIDEO_VITA_VGL=ON` | Uses vitaGL for its video backend. Installs over the `libSDL2.a` shipped with vitasdk — back that up first. Needs `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` on CMake 4 |

Then:

```sh
cmake -S . -B build \
  -DCMAKE_TOOLCHAIN_FILE=$VITASDK/share/vita.toolchain.cmake \
  -DVITA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

`build/SuperTuxKart.vpk` is produced as part of the default target.

## LiveArea assets

`sce_sys/` holds the icon and LiveArea artwork, derived from the icons already
in `data/`. The Vita requires these exact dimensions:

| File | Size |
| --- | --- |
| `icon0.png` | 128x128 |
| `livearea/contents/bg.png` | 840x500 |
| `livearea/contents/startup.png` | 280x158 |

## Renderers

There are two, selectable under Options → Graphics → Custom settings as
"Render driver", the same way desktop builds pick between OpenGL and Vulkan:

| Driver | What it is |
| --- | --- |
| `gxm` (default) | A native renderer talking to the GPU through libgxm. |
| `opengl` | The existing path, through vitaGL. |

`--render-driver=gxm` or `--render-driver=opengl` overrides the setting for one
run.

### The GXM renderer

The Vita's GPU is a PowerVR SGX543MP4+, a tile based deferred rasteriser. Three
of its limits shape the renderer:

* **No multiple render targets.** `sceGxmBeginScene()` takes one colour surface,
  so a deferred G-buffer pass cannot be expressed at all. The GXM renderer is
  therefore forward: one geometry pass that lights as it shades.
* **No compute shaders and no storage buffers.** Per-object data reaches the
  vertex shader through a vertex stream indexed by instance, and the environment
  map prefiltering the Vulkan backend does in compute is done on the CPU at
  track load instead.
* **A small, fixed per-draw uniform budget**, held in the pipeline's secondary
  attribute registers. See `lib/graphics_engine/src/ge_gxm_limits.hpp` for what
  that costs — notably a 40 joint skinning palette and 8 lights per draw.

What it gains in exchange is a large on-chip tile buffer. Depth is never written
to memory in the main pass, and with post-processing off the whole frame is a
single scene, so intermediate colour never round trips either.

Pipeline, in order: an orthographic shadow cascade (depth only, colour surface
disabled); the skybox; forward PBR opaque geometry with sun, shadow, image based
ambient and culled point lights; transparent geometry back to front; then bloom
and the composite when enabled; then the batched GUI.

The scene graph, mesh format, materials, culling and texture loading are all
shared with the Vulkan backend through the rest of `lib/graphics_engine` — only
the GPU-facing half is new (`ge_gxm_*`).

### Shader compilation

Sony's offline Cg compiler, `psp2cgc`, needs a licensed SDK, so the shaders are
compiled **at runtime** by the `SceShaccCg` module that ships on every retail
Vita, reached through vitaShaRK. The compiled binaries are cached in
`ux0:data/stk/save/gxm_shader_cache/`, so the first launch after an install (or
after the shader sources change) takes a few extra seconds and every launch
after that reads the cache.

If rendering ever looks wrong after editing a shader, deleting that directory
forces a full recompile — though the cache key includes the shader source, so
that should not be necessary.

The driver probes the compiler at startup rather than assuming what it can do,
because two things vary by system:

* **The SceShaccCgExt patches.** vitaShaRK enables them through taiHEN, and they
  are what raise the vertex uniform register limit from 128 to 512 — which is
  what makes a useful joint palette possible. On hardware they work; under
  Vita3K the patched compiler faults and fails every shader with "fatal internal
  error". So the driver compiles one trivial shader to check, and on failure
  releases the patches and carries on without them. The finding is recorded as
  `gxm_shader_cache/no_shacccg_ext` so later launches skip straight past it;
  delete that file to try again.
* **The joint palette size.** Rather than reason about how the compiler assigns
  registers to a dynamically indexed array, it compiles the skinned mesh shader
  at 40 joints and halves until the compiler agrees. Without the extensions the
  budget is much smaller, and if no size compiles at all, skinned meshes are
  drawn in bind pose rather than not at all.

Both results are logged, so `GXM: joint palette holds N joints` tells you which
path you are on.

Note that on a Vita there is no console, so STK's log also goes out through
`sceClibPrintf`. That is what makes it visible in the Vita3K log, and unlike
`stdout.log` it survives a crash — which is exactly when it is wanted.

## Notes

* The self is built `UNSAFE`, which the game needs in order to read and write
  `ux0:data/stk/`; a safe-mode homebrew is confined to its own sandbox.
* The `opengl` driver goes through vitaGL. `USE_GLES2` is ON for Vita
  (`CMakeLists.txt`), so that path runs irrlicht's `COGLES2Driver`
  (`EDT_OGLES2`) against vitaGL rather than the desktop GL driver.
* If the game runs out of memory, `EBOOT_APP_MEMSIZE` / the `MEMSIZE` argument
  to `vita_create_self()` in the top-level `CMakeLists.txt` is the knob to
  reach for. Note that the GXM renderer's GPU memory comes from its own kernel
  memory blocks (see `ge_gxm_memory.cpp`), not from the libc heap, so the two
  budgets are separate.
