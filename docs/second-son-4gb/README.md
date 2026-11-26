# shadPS4 — inFAMOUS Second Son 4 GB mesh-reuse build

This is a small, focused fork of `axylpup/shadPS4` (branch `second-son`, commit `39b6f64b`) for **inFAMOUS Second Son** on **4 GB VRAM cards**. It keeps the core emulator only; the Qt launcher is not included.

## What changed

- **One mesh join, not repeated re-creations.** The game streams its ~2.8 GB mesh buffer in small steps. The old code re-created and copied the whole buffer when a write landed just past the joined range. `ResolveOverlaps()` now grows once with **256 MB of headroom** (default, `SHADPS4_MESH_HEADROOM_MB`) inside mapped guest memory, and stops at a neighbouring large buffer. The range scan was changed from one lookup per 4 KB page to a walk of the existing buffer ranges, removing a 141 ms hitch.
- **Host-memory fallback.** When the mesh buffer does not fit in video memory, it is allocated in system memory instead of asserting. It stays device-addressable; reads over PCIe are slower, but the game keeps running. Set `SHADPS4_MESH_HOST_FALLBACK=0` to restore the old assert for comparison.
- **Driver-budget-aware texture GC.** The texture cache can follow `VK_EXT_memory_budget` as it changes during play, or use a fixed budget. On small discrete cards it keeps extra headroom free for buffers, staging and the swapchain.
- **`SHADPS4_MESH_MAX_MB`.** Caps the mesh buffer before the fit probe (the release flags use `2048`, i.e. 2 GB, instead of the 2.75 GB tier).

## Recommended runtime flags

`shadps4-flags.txt` and `shadps4-flags-user.txt` are included in the binary ZIP and in this folder. Put both next to `shadPS4.exe`.

The user flags contain:

| Variable | Value | Why |
|---|---:|---|
| `SHADPS4_ASPECT` | `16:9` | Fixes subtitles/HUD going off-screen on 16:10 monitors. Without it, `auto` reported a `1912x1204` HUD space on a 1728x1080 screen. |
| `SHADPS4_PERF_MODE` | `low` | Lowers render cost to leave room for the mesh buffer. |
| `SHADPS4_SCREEN_PERCENT` | `50` | Renders at 50% resolution scale. |
| `SHADPS4_MESH_MAX_MB` | `2048` | Caps the mesh buffer at 2 GB. |
| `SHADPS4_VRAM_AUTO` | `0` | Disables automatic texture-budget tracking; fixed budget was more stable in the faithful 4 GB test. |
| `SHADPS4_VRAM_BUDGET_MB` | `2048` | Fixed 2 GB texture budget. |

`SHADPS4_SIM_VRAM_MB` is **not** included. It was only used to simulate 4 GB on a larger test card; on a real 4 GB card it is ignored and unnecessary.

Environment variables already set by a launcher override the values in these files.

## Running

1. Provide your own legal game dump; no game files are included in the release.
2. Extract the ZIP into a folder of your choice.
3. Launch directly, for example:

   ```powershell
   .\shadPS4.exe --game "<full path to your dump>\eboot.bin"
   ```

   A launcher can be used instead, as long as it passes the same flags or leaves the flags files in place.

## Test evidence

Faithful `SHADPS4_SIM_VRAM_MB=4096` run after the final logic:

- one `BIG_GROW` mesh join of ~**2984 MB**, extended in a single 15.88 ms CPU step;
- no repeated buffer re-creation, OOM, assert or `TEMP:` diagnostics;
- FPS settled around **59–74**;
- peak system heap ~6592 MB, garbage-collected back to ~3624 MB;
- subtitles visible after `SHADPS4_ASPECT=16:9` (user-confirmed).

The release EXE was rebuilt from this exact source after the temporary diagnostics were removed.

## Binary ZIP contents

- `shadPS4.exe`
- `shadps4-flags.txt`, `shadps4-flags-user.txt`
- `defaults/`, `dlss/`, `tex_replace/`
- empty `user/` runtime folders
- `README.md`, `LICENSE`

Excluded: game files, shader caches, logs, savedata, `shadps4-game.txt`, PDBs, build scripts and other build artifacts.

## Building from source

The tested build used Clang **22.1.8**, ThinLTO and PGO with `pgo/second-son-v1.1.1.profdata`. On Windows, with the portable toolchain already configured:

```bat
D:\shadps4-toolchain\build-release.cmd
```

The output is `Build\x64-Clang-RelWithDebInfo\shadps4.exe`. `Build\beta 1` is a frozen reference package and is never modified by this build.
