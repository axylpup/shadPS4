// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once
#include <cstring>
#include <string>
#include <vector>
#include "common/types.h"

#if defined(WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

namespace MemoryPatcher {

extern EXPORT uintptr_t g_eboot_address;
extern uint64_t g_eboot_image_size;
/// 64 bytes of read-write memory near the eboot's code (module.cpp: the trampoline area's tail).
extern uintptr_t g_eboot_scratch;
/// One loaded segment of the eboot (module.cpp, before OnGameLoaded): the built-in patches check
/// their 01.07 offsets against these before reading (GitHub #22).
void AddEbootSegment(uintptr_t address, uint64_t size);
/// Cheats (common/cheats.h): whether [offset, offset + size) of the eboot image is loaded, and
/// whether it lies in its first segment (the code).
bool EbootRangeLoaded(u64 offset, u64 size);
bool EbootRangeIsCode(u64 offset, u64 size);
extern std::string g_game_serial;
extern std::string patch_file;

enum PatchMask : uint8_t {
    None,
    Mask,
    Mask_Jump32,
};

struct patchInfo {
    std::string gameSerial;
    std::string modNameStr;
    std::string offsetStr;
    std::string valueStr;
    std::string targetStr;
    std::string sizeStr;
    bool isOffset;
    bool littleEndian;
    PatchMask patchMask;
    int maskOffset;
};

std::string convertValueToHex(const std::string type, const std::string valueStr);

void OnGameLoaded();

/// emulator.cpp, right after the window exists: the screen / window size in pixels, for the
/// dynamic aspect ratio (SHADPS4_ASPECT, ApplyAspectRatio).
void SetScreenSize(int width, int height);
/// The monitor the window opened on (pixels): 100% of the screen percentage.
void SetDisplaySize(int width, int height);

/// Screen percentage (1.1, the DLSS panel): the game's render size in percent of the screen,
/// decided at boot (ApplyAspectRatio).
struct RenderSize {
    u32 width;
    u32 height;
};
RenderSize CurrentRenderSize();
/// 100%: the largest box of the render shape on the monitor.
RenderSize PercentReference();
/// The percentage in effect (the game's 1080 lines as a percentage when none is set).
float CurrentScreenPercent();
/// The render size `percent` gives at the next start (0 = the game's 1080 lines).
RenderSize RenderSizeForPercent(float percent);
/// SHADPS4_RENDER_HEIGHT sets the size (the panel's value then waits).
bool RenderHeightFromFlags();
/// user/render-size.ini (0 = none saved).
float SavedScreenPercent();
void SaveScreenPercent(float percent);

/// Live screen percentage (study 10-20 route A): the game boots at the largest size and a change
/// rides its own mode switch. Available when a percentage was chosen at boot (normal mode).
bool LiveResizeAvailable();
/// The boot allocation: a live size is at most this per side.
RenderSize LiveMaxSize();
/// Queues `percent` for the next frame; false = more than the boot size (after a restart).
bool RequestScreenPercent(float percent);
/// Insert: the saved pick <-> 100% (live). Returns the percentage requested, 0 = not live.
float ToggleNativePercent();
/// sceGnmSubmitDone (Game:Main, once per frame): applies a queued size.
void OnSubmitDone();
/// The back, at a frame's end: true once after the game took a change (the texture cache then
/// drops the render-shaped targets that frame did not use).
bool TakeLivePurge(u64 sizes[2]); ///< sizes[0] = old, [1] = new (w << 32 | h)
/// The frame was still the old size: try again at the next frame's end (a few times).
void RequeueLivePurge();
/// The 3 s after a change was poked: the texture cache replaces re-sized render targets.
bool InLiveResizeWindow();
/// A size of the render shape (+-3%): the game's screen-sized targets and their half / quarter.
bool IsRenderShaped(u32 width, u32 height);
/// Milliseconds since the last change (INT64_MAX: none) - DLSS's grace window.
s64 MsSinceLiveResize();

/// sceSystemServiceGetDisplaySafeAreaInfo: the safe-area ratio the game gets (1.0 unless
/// SHADPS4_HUD narrows the HUD; ApplyHudWidth).
float HudSafeAreaRatio();

/// ApplyMovieFit: the full-screen movie quad's corner (x, y) in NDC when movies are fitted inside a
/// non-16:9 screen; false on 16:9 (the game's own draw). The rasterizer clears that draw's target
/// to black first, so the bars beside the movie do not show the last game frame.
bool MovieFitRect(float& x, float& y);

/// The game's own quality ladder level (0 = full quality ... 20; SHADPS4_PERF_MODE notes), read
/// live from the renderer (0x800f39978); -1 before the game is loaded. For the memory census.
int GameQualityLevel();
void AddPatchToQueue(const patchInfo& patchToAdd);

void PatchMemory(const patchInfo& patch);

static std::vector<int32_t> PatternToByte(const std::string& pattern);
uintptr_t PatternScan(const std::string& signature);

} // namespace MemoryPatcher