// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include "common/types.h"
#include "video_core/amdgpu/cp_split.h"

namespace VideoCore {

// SHADPS4_NATIVE_CENSUS=1 (diagnostic, default off; native-path study 2026-10-09): the host-side
// vertex / index bindings of the current draw, filled by the buffer cache on the translating
// thread, read by the rasterizer's batchability census at the end of the draw.
struct NativeCensusVi {
    u32 n{};
    std::array<u64, 16> buf{};
    std::array<u64, 16> off{};
    std::array<u32, 16> stride{};
    u64 ib_buf{};
    u64 ib_off{};
    u32 ib_type{};
    bool gpu_mod{}; ///< some vertex range of this draw is GPU-written (SHADPS4_NATIVE_COLOUR)
};
extern const bool g_native_census;
/// SHADPS4_NATIVE_PASS: 0 off, 1 native shadow batches, 2 verify (generic path renders).
extern const int g_native_pass;
/// SHADPS4_NATIVE_COLOUR=1 (with SHADPS4_NATIVE_PASS): textured pipelines get native variants
/// too (notes/colour-pass-2026-10-09.md sec. 3). Not const: emulator.cpp turns it off for First
/// Light before the renderer starts.
extern bool g_native_colour;
/// SHADPS4_NATIVE_COLOUR_DIAG (bisect, default 0): 1 = colour batches take no joins, 2 = packed
/// 10:10:10:2 vertex formats not pulled, 4 = buffers with 8 / 16-bit loads ineligible, 8 =
/// formatted buffers ineligible, 16 = PS stays non-native (no colour variants at all).
extern const u32 g_native_colour_diag;

/// SHADPS4_NATIVE_COLOUR: the key of a vertex description (what a colour variant bakes:
/// format, stride, location, rate per attribute). Never 0 (0 = free slot).
inline u64 NativeVertexKey(const AmdGpu::CpSplit::VertexInputAttrib* attribs, u32 count) {
    u64 h = 0xcbf29ce484222325ull ^ count;
    for (u32 i = 0; i < count; ++i) {
        const auto& d = attribs[i].desc;
        const u64 words[2] = {u64(d.vk_format) | (u64(d.stride) << 32),
                              u64(d.divisor) | (u64(d.location) << 32) | (u64(d.instanced) << 40)};
        for (const u64 w : words) {
            h = (h ^ w) * 0x100000001b3ull;
            h ^= h >> 29;
        }
    }
    return h | 1;
}

/// SHADPS4_NATIVE_PASS join: one draw's vertex / index bindings as the batch would use them.
struct NativeJoinVi {
    u32 n{};
    std::array<u64, 16> buf{};
    std::array<u64, 16> off{};
    std::array<u32, 16> stride{};
    u64 ib_buf{};
    u64 ib_off{};
};
inline thread_local NativeCensusVi t_native_vi{};

} // namespace VideoCore
 