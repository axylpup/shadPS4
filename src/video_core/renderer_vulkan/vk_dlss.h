// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include "common/logging/log.h"
#include "common/types.h"
#include "shader_recompiler/info.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/renderer_vulkan/vk_common.h"

// SHADPS4_DLSS (2026-10-17, LATER-PLAN 2; plan: notes/study-dlss-plan-2026-10-14.md, build D1;
// Streamline facts: notes/dlss-streamline-checklist-2026-10-17.md).
// Second Son's SMAA T2x temporal resolve (fs_a6d23116) reads exactly DLSS's inputs and writes the
// anti-aliased frame the HUD is drawn onto; that one draw per frame becomes a back-side action
// (Rasterizer::DrawClass::DlssResolve), so the worker paths carry it like a resolve or a depth copy
// and nothing is added per draw beyond one address compare in ClassifyDraw.
//   =1: DLAA through NVIDIA Streamline. The player's own signed DLLs are loaded from dlss\ next to
//       the EXE (sl.interposer, sl.common, sl.dlss, nvngx_dlss); only MIT headers are compiled in.
//       Missing / unsigned DLLs, a GPU without DLSS or any failure: the draw stays the game's own.
//   =2: diagnostic - the action copies the un-anti-aliased tonemap output (the game without T2x).
// Live keys: Home = DLSS on / off (instant A/B with the game's AA), Page Up = jitter sign combination.
namespace Vulkan::Dlss {

constexpr u64 ResolvePsHash = 0xa6d23116; ///< T2x temporal resolve (writes 0x26bd70000)
// Measured in every capture since 09-24 (study sec. 1.1-1.2): fixed guest addresses.
constexpr VAddr ColorInAddr = 0x2f9e10000; ///< tonemap output, RGBA8 1920x1080, not anti-aliased
constexpr VAddr MotionAddr = 0x3059b0000;  ///< RG16F motion vectors (UV, current minus previous)
constexpr VAddr DepthAddr = 0x26b200000;   ///< D32S8, reversed Z

inline int Mode() {
    static const int mode = [] {
        const char* v = std::getenv("SHADPS4_DLSS");
        return v && (v[0] == '1' || v[0] == '2') ? v[0] - '0' : 0;
    }();
    return mode;
}
inline bool Enabled() {
    return Mode() != 0;
}

/// Streamline is up and the feature exists on this device (=1), or the diagnostic copy (=2).
inline std::atomic<bool> g_ready{false};
/// 1.1.1 (GitHub #19): why the dlss folder's files were not used, in a player's words, for the Home
/// notice ("DLSS not available: ..."); nullptr = no reason known (the log has the detail).
inline std::atomic<const char*> g_why_not{nullptr};
/// Home key: the player's on / off (the game's own T2x runs while off).
inline std::atomic<bool> g_user_on{true};
/// Page Up: jitter sign combination (bit 0 flips x, bit 1 flips y).
inline std::atomic<u32> g_jitter_combo{3}; ///< D (-x -y): the user's live pick, 2026-10-17
/// Insert (SHADPS4_DLSS_UPSCALE): 1 = DLSS writes the window size, 1.5 = 1.5x the window (3840x2160
/// on a 1440p screen), scaled down by the post pass (supersampling; user idea 2026-10-17).
inline std::atomic<float> g_output_scale{1.0f};
/// The panel's last output pick above the screen size (Insert switches between it and 1.0).
inline std::atomic<float> g_output_pick{1.5f};
/// The network (sl::DLSSPreset number; 0 = the DLL's default, 11 = K, 12 = L, 13 = M): the Page Up
/// panel, first from SHADPS4_DLSS_PRESET.
inline std::atomic<u32> g_preset{[] {
    const char* v = std::getenv("SHADPS4_DLSS_PRESET");
    return v ? static_cast<u32>(std::atoi(v)) : 0u;
}()};
/// What the last options call set (the panel shows it): render -> output size and the mode
/// (sl::DLSSMode: 0 off, 1 Performance, 2 Balanced, 3 Quality, 4 Ultra Performance, 5 Ultra
/// Quality, 6 DLAA).
inline std::atomic<u32> g_status_in_w{0}, g_status_in_h{0}, g_status_out_w{0}, g_status_out_h{0};
inline std::atomic<u32> g_status_mode{0};
inline std::atomic<u64> g_status_evaluates{0};

/// Guest code address of fs_a6d23116 once a pipeline lookup has met it (0 = not yet / off).
inline std::atomic<u64> g_resolve_ps{0};
/// A frame's SMAA edges draw has told where the colour input is ("Any render size" below): until
/// then the game's own resolve runs (the first frames of a session).
inline std::atomic<bool> g_inputs_seen{false};

/// PipelineCache::GetProgram: remembers where the resolve shader lives.
inline void NoteProgram(bool fragment, u64 hash, VAddr base) {
    if (fragment && hash == ResolvePsHash && Enabled()) [[unlikely]] {
        g_resolve_ps.store(base, std::memory_order_relaxed);
    }
}

/// ClassifyDraw: is this draw's pixel shader the resolve, and is DLSS to replace it now?
inline bool IsResolve(VAddr ps_address) {
    const u64 a = g_resolve_ps.load(std::memory_order_relaxed);
    return a != 0 && a == ps_address && g_ready.load(std::memory_order_relaxed) &&
           g_user_on.load(std::memory_order_relaxed) &&
           g_inputs_seen.load(std::memory_order_relaxed);
}

// The frame's jitter phase (study sec. 1.4): the game alternates two quarter-pixel offsets and
// writes the matching T2x subsample index (1 or 2; 0 = T2x off) into the SMAA blend-weights
// draw's 80-byte CB, float 12. That draw runs two draws before the resolve in the same command
// buffer, so the thread parsing it keeps the value and the resolve's record carries it to the back
// (Rasterizer::SplitDraw::dlss_phase): exact per frame, no guess from a frame counter that a
// loading screen could flip.
constexpr u64 WeightsPsHash = 0x961c93ba; ///< SMAA blend weights
inline std::atomic<u64> g_weights_ps{0};
inline std::atomic<u32> g_weights_ud{~0u}; ///< dword of its CB's V# in the flattened user data
/// When the V# is not in the 16 user-data registers: the shader's SRT walker (it reads the guest
/// pointer tables into the flattened buffer, as binding does) and that buffer's size.
inline std::atomic<Shader::PFN_SrtWalker> g_weights_walker{nullptr};
inline std::atomic<u32> g_weights_flat_dw{0};
/// The parsing thread's last weights draw: 1 / 2 = subsample index, 3 = T2x off, 0 = unknown,
/// 4 = an explicit offset (the emulator's Halton jitter, memory_patcher.cpp: t_jitter).
inline thread_local u8 t_phase = 0;
inline thread_local float t_jitter[2] = {}; ///< phase 4: the frame's (jx, jy) in render px

/// Where the weights program's Info is known: remember its address and its CB's user-data slot.
inline void NoteWeightsInfo(const Shader::Info& info, VAddr base) {
    if (info.pgm_hash != WeightsPsHash || info.stage != Shader::Stage::Fragment || !Enabled())
        [[likely]] {
        return;
    }
    if (g_weights_ps.load(std::memory_order_relaxed) == base) {
        return;
    }
    for (const auto& b : info.buffers) {
        if (b.IsSpecial() || b.inline_cbuf) {
            continue;
        }
        if (b.sharp_idx + 4 <= 16) {
            g_weights_walker.store(nullptr, std::memory_order_relaxed);
        } else if (info.srt_info.walker_func != nullptr &&
                   b.sharp_idx + 4 <= info.srt_info.flattened_bufsize_dw &&
                   info.srt_info.flattened_bufsize_dw <= 1024) {
            // Play log 2026-10-17: the phase stayed "unknown" on every frame - the V# comes
            // through a pointer table, not the 16 registers.
            g_weights_flat_dw.store(info.srt_info.flattened_bufsize_dw, std::memory_order_relaxed);
            g_weights_walker.store(info.srt_info.walker_func, std::memory_order_relaxed);
        } else {
            continue;
        }
        g_weights_ud.store(b.sharp_idx, std::memory_order_relaxed);
        g_weights_ps.store(base, std::memory_order_release);
        LOG_INFO(Render_Vulkan, "DLSS: jitter source fs_961c93ba at {:#x}, CB V# at dword {} ({})",
                 base, b.sharp_idx,
                 b.sharp_idx + 4 <= 16 ? "user data" : "pointer table");
        return;
    }
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true)) {
        LOG_WARNING(Render_Vulkan, "DLSS: fs_961c93ba has no readable CB ({} buffers) - jitter "
                                   "phase unknown",
                    info.buffers.size());
    }
}

/// Next to every ClassifyDraw call that parses the stream in order (front, lean workers, inline).
// ---- Any render size (2026-10-20, v1.1 part C) ----
// The addresses above hold at the game's 1920x1080. When the game renders at another size
// (SHADPS4_ASPECT, SHADPS4_RENDER_HEIGHT, a community resolution patch) its targets sit elsewhere
// (a 1440p patch: no input at 0x2f9e10000, a yellow picture - RTX 5070 Ti report, 10-20). So the
// inputs are read every frame from the draws that use them, on the parsing thread (NoteDraw):
//  - SMAA edges fs_abc3c768 samples the tonemap output (its RGBA8 texture) with the scene depth
//    attached (rooftop replay EID 39235, measured);
//  - the resolve fs_a6d23116 samples the motion vectors (its one RG16F texture; its size = the
//    game's render size) and writes the HUD layer (its colour target).
// Until a frame has run them, the 1080p addresses stand (the values learned at 1080p equal them).
constexpr u64 EdgesPsHash = 0xabc3c768;
struct ShaderTap {
    std::atomic<u64> ps{0};
    std::atomic<Shader::PFN_SrtWalker> walker{nullptr};
    std::atomic<u32> flat_dw{0};
    std::atomic<u32> count{0};
    std::array<std::atomic<u32>, 8> sharps{}; ///< sharp_idx | 0x80000000 for a 128-bit T#
};
inline ShaderTap g_edges_tap;
inline ShaderTap g_resolve_tap;
inline std::atomic<VAddr> g_color_addr{ColorInAddr};
inline std::atomic<VAddr> g_motion_addr{MotionAddr};
inline std::atomic<VAddr> g_depth_addr{DepthAddr};
inline std::atomic<VAddr> g_hud_addr{0x26bd70000}; ///< the resolve's target = the HUD layer
inline std::atomic<u32> g_game_width{1920};
inline std::atomic<u32> g_game_height{1080};

/// Where a tapped program's Info is known: its texture slots in the (flattened) user data.
inline void NoteTapInfo(ShaderTap& tap, const Shader::Info& info, VAddr base) {
    if (tap.ps.load(std::memory_order_relaxed) == base) [[likely]] {
        return;
    }
    const bool flat = info.srt_info.walker_func != nullptr &&
                      info.srt_info.flattened_bufsize_dw <= 1024;
    const u32 limit = flat ? info.srt_info.flattened_bufsize_dw : 16u;
    u32 n = 0;
    for (const auto& img : info.images) {
        const u32 size_dw = img.is_r128 ? 4u : 8u;
        if (img.sharp_idx + size_dw > limit || n >= tap.sharps.size()) {
            continue;
        }
        tap.sharps[n++].store(img.sharp_idx | (img.is_r128 ? 0x80000000u : 0u),
                              std::memory_order_relaxed);
    }
    tap.count.store(n, std::memory_order_relaxed);
    tap.walker.store(flat ? info.srt_info.walker_func : nullptr, std::memory_order_relaxed);
    tap.flat_dw.store(flat ? info.srt_info.flattened_bufsize_dw : 16u, std::memory_order_relaxed);
    tap.ps.store(base, std::memory_order_release);
    LOG_INFO(Render_Vulkan, "DLSS: input tap fs_{:08x} at {:#x}, {} of {} textures readable ({})",
             info.pgm_hash, base, n, info.images.size(), flat ? "pointer table" : "user data");
}

/// The pipeline lookup's Info sites: the weights CB and the two input taps.
/// SHADPS4_SKIP_PS=<hex hash>[,<hex hash>...] (test, 10-21): draws with these pixel shaders are
/// skipped, e.g. 204f45fe = the screen-space reflection ray march (study-delsin-mv-ssr-2026-10-21).
/// A list and every guest copy of each program (10-26, First Light's film grain search: the game
/// loads some programs at two addresses).
constexpr size_t MaxSkipPs = 16;
inline std::array<std::atomic<u64>, MaxSkipPs> g_skip_ps{};
inline std::atomic<u32> g_skip_ps_count{0};
/// The hashes skipped; emulator.cpp adds a game's own (First Light's film grain) before the
/// renderer starts, so the list is read-only once draws run.
inline std::vector<u64>& SkipPsHashes() {
    static std::vector<u64> hashes = [] {
        std::vector<u64> out;
        if (const char* v = std::getenv("SHADPS4_SKIP_PS")) {
            for (const char* p = v; *p;) {
                char* end = nullptr;
                const u64 h = std::strtoull(p, &end, 16);
                if (end == p) {
                    break;
                }
                out.push_back(h);
                p = *end == ',' ? end + 1 : end;
            }
        }
        return out;
    }();
    return hashes;
}

inline bool IsSkippedPs(VAddr ps_address) {
    const u32 n = g_skip_ps_count.load(std::memory_order_acquire);
    for (u32 i = 0; i < n; ++i) {
        if (g_skip_ps[i].load(std::memory_order_relaxed) == ps_address) {
            return true;
        }
    }
    return false;
}

inline void NoteShaderInfo(const Shader::Info& info, VAddr base) {
    if (info.stage == Shader::Stage::Fragment && !SkipPsHashes().empty() &&
        std::ranges::find(SkipPsHashes(), info.pgm_hash) != SkipPsHashes().end()) [[unlikely]] {
        static std::mutex skip_mutex;
        std::scoped_lock lock{skip_mutex};
        const u32 n = g_skip_ps_count.load(std::memory_order_relaxed);
        bool known = false;
        for (u32 i = 0; i < n; ++i) {
            known = known || g_skip_ps[i].load(std::memory_order_relaxed) == base;
        }
        if (!known && n < MaxSkipPs) {
            g_skip_ps[n].store(base, std::memory_order_relaxed);
            g_skip_ps_count.store(n + 1, std::memory_order_release);
            LOG_INFO(Render_Vulkan, "SHADPS4_SKIP_PS: fs_{:08x} at {:#x} skipped", info.pgm_hash,
                     base);
        }
    }
    if (info.stage != Shader::Stage::Fragment || !Enabled()) [[likely]] {
        return;
    }
    if (info.pgm_hash == EdgesPsHash) {
        NoteTapInfo(g_edges_tap, info, base);
    } else if (info.pgm_hash == ResolvePsHash) {
        NoteTapInfo(g_resolve_tap, info, base);
    }
}

/// A tapped draw's textures (T#), read from its user data as binding does.
template <typename Fn>
inline void ForEachTapImage(const ShaderTap& tap, const AmdGpu::Regs& regs, Fn&& fn) {
    const u32 n = tap.count.load(std::memory_order_relaxed);
    const auto walker = tap.walker.load(std::memory_order_relaxed);
    const u32* ud = regs.ps_program.user_data.data();
    thread_local std::vector<u32> flat;
    if (walker != nullptr) {
        flat.assign(std::max(tap.flat_dw.load(std::memory_order_relaxed), 16u), 0u);
        std::memcpy(flat.data(), ud, 16 * sizeof(u32));
        walker(ud, flat.data());
        ud = flat.data();
    }
    for (u32 i = 0; i < n; ++i) {
        const u32 s = tap.sharps[i].load(std::memory_order_relaxed);
        AmdGpu::Image image{};
        std::memcpy(&image, ud + (s & 0x7fffffffu), (s & 0x80000000u) ? 16 : sizeof(image));
        if (s & 0x80000000u) {
            image.pitch = image.width;
        }
        if (image.Valid() && image.Address() != 0) {
            fn(image);
        }
    }
}

template <typename T>
inline void StoreIfChanged(std::atomic<T>& a, T v) {
    if (a.load(std::memory_order_relaxed) != v) {
        a.store(v, std::memory_order_relaxed);
    }
}

/// NoteDraw: the edges draw -> colour input + depth; the resolve -> motion vectors, render size,
/// HUD layer.
inline void NoteInputDraw(const AmdGpu::Regs& regs, VAddr ps) {
    if (ps == g_edges_tap.ps.load(std::memory_order_acquire)) {
        bool found = false;
        ForEachTapImage(g_edges_tap, regs, [&](const AmdGpu::Image& image) {
            if (!found && image.GetDataFmt() == AmdGpu::DataFormat::Format8_8_8_8) {
                StoreIfChanged(g_color_addr, image.Address());
                StoreIfChanged(g_inputs_seen, true);
                found = true;
            }
        });
        if (const VAddr z = regs.depth_buffer.DepthAddress(); z != 0) {
            StoreIfChanged(g_depth_addr, z);
        }
    } else if (ps == g_resolve_tap.ps.load(std::memory_order_acquire)) {
        bool found = false;
        ForEachTapImage(g_resolve_tap, regs, [&](const AmdGpu::Image& image) {
            if (!found && image.GetDataFmt() == AmdGpu::DataFormat::Format16_16 &&
                image.GetNumberFmt() == AmdGpu::NumberFormat::Float) {
                StoreIfChanged(g_motion_addr, image.Address());
                StoreIfChanged(g_game_width, static_cast<u32>(image.width) + 1);
                StoreIfChanged(g_game_height, static_cast<u32>(image.height) + 1);
                found = true;
            }
        });
        StoreIfChanged(g_hud_addr, static_cast<VAddr>(regs.color_buffers[0].Address()));
    }
}

inline void NoteDraw(const AmdGpu::Regs& regs) {
    const VAddr ps = regs.ps_program.Address<VAddr>();
    const u64 w = g_weights_ps.load(std::memory_order_relaxed);
    if (w == 0 || ps != w) [[likely]] {
        if (ps == g_edges_tap.ps.load(std::memory_order_relaxed) ||
            ps == g_resolve_tap.ps.load(std::memory_order_relaxed)) [[unlikely]] {
            if (ps != 0) {
                NoteInputDraw(regs, ps);
            }
        }
        return;
    }
    const u32 ud = g_weights_ud.load(std::memory_order_relaxed);
    AmdGpu::Buffer vsharp;
    if (ud + 4 <= 16) {
        std::memcpy(&vsharp, &regs.ps_program.user_data[ud], sizeof(vsharp));
    } else {
        const auto walker = g_weights_walker.load(std::memory_order_relaxed);
        const u32 flat_dw = g_weights_flat_dw.load(std::memory_order_relaxed);
        if (walker == nullptr || ud + 4 > flat_dw) {
            t_phase = 0;
            return;
        }
        thread_local std::vector<u32> flat;
        flat.assign(flat_dw, 0u);
        std::memcpy(flat.data(), regs.ps_program.user_data.data(), 16 * sizeof(u32));
        walker(regs.ps_program.user_data.data(), flat.data());
        std::memcpy(&vsharp, &flat[ud], sizeof(vsharp));
    }
    if (vsharp.base_address == 0 || vsharp.GetSize() < 60) {
        t_phase = 0;
        return;
    }
    float f;
    std::memcpy(&f, reinterpret_cast<const void*>(vsharp.base_address + 48), sizeof(f));
    float tag;
    std::memcpy(&tag, reinterpret_cast<const void*>(vsharp.base_address + 56), sizeof(tag));
    if (tag >= 99.5f) {
        t_phase = 4;
        t_jitter[0] = f;
        std::memcpy(&t_jitter[1], reinterpret_cast<const void*>(vsharp.base_address + 52),
                    sizeof(float));
        return;
    }
    t_phase = (f > 0.5f && f < 1.5f)   ? 1
              : (f > 1.5f && f < 2.5f) ? 2
              : (f > -0.5f && f < 0.5f) ? 3
                                        : 0;
}

// ---- Streamline (vk_dlss.cpp; Windows only, stubs elsewhere) ----

/// vk_platform GetInstanceExtensions: loads + verifies the interposer, slInit, and appends the
/// instance extensions DLSS needs. Before vkCreateInstance (Streamline's required order).
void AddInstanceExtensions(std::vector<const char*>& extensions);
/// vk_instance, device selection: slIsFeatureSupported for this GPU; false = DLSS stays off.
bool DeviceSupported(vk::PhysicalDevice physical_device);
/// The device extensions DLSS asked for (valid until Shutdown).
const std::vector<const char*>& DeviceExtensions();
/// vk_instance, after the device and its queues exist: slSetVulkanInfo, present proxy, DLSS
/// functions; sets g_ready on success.
void DeviceCreated(vk::Instance instance, vk::PhysicalDevice physical_device, vk::Device device,
                   u32 graphics_family, u32 graphics_index);
/// vk_swapchain: the present to use (Streamline's proxy runs its per-frame bookkeeping and
/// forwards); null = present as usual.
PFN_vkQueuePresentKHR PresentFunction();
/// Instance destruction (GPU idle): slShutdown before the device goes.
void Shutdown();

struct EvalImage {
    vk::Image image;
    vk::ImageView view;
    vk::Format format;
    vk::ImageLayout layout;
    vk::ImageUsageFlags usage;
    u32 width;
    u32 height;
    bool depth;
};
/// One DLAA evaluate into `cmdbuf` (outside rendering; the images already in `layout`, with
/// their producer barriers). jitter in pixels. False = not run (the caller falls back).
bool Evaluate(vk::CommandBuffer cmdbuf, u64 frame, const EvalImage& color, const EvalImage& motion,
              const EvalImage& depth, const EvalImage& output, float jitter_x, float jitter_y,
              bool reset);

} // namespace Vulkan::Dlss
