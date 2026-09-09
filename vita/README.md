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

## Notes

* The self is built `UNSAFE`, which the game needs in order to read and write
  `ux0:data/stk/`; a safe-mode homebrew is confined to its own sandbox.
* OpenGL comes from vitaGL, which exports GL 2.1. SuperTuxKart reaches it
  through glad via `SDL_GL_GetProcAddress`, so the legacy/fallback renderer is
  what actually runs — `USE_GLES2` is deliberately left off for Vita.
* If the game runs out of memory, `EBOOT_APP_MEMSIZE` / the `MEMSIZE` argument
  to `vita_create_self()` in the top-level `CMakeLists.txt` is the knob to
  reach for.
