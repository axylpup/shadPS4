// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <immintrin.h>
#include <set>
#include <chrono>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "video_core/texture_cache/tex_table.h"
#include "common/memory_patcher.h"
#include "common/thread.h"
#include "common/debug.h"
#include "common/buffer_sample_probe.h"
#include "common/draw_sample_probe.h"
#include "common/fault_monitor.h"
#include "common/lean.h"
#include "common/rdtsc.h"
#include "common/scope_exit.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/split_hazard.h"
#include "video_core/buffer_cache/back_clean.h"
#include "video_core/buffer_cache/dirty_epoch.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/pb_census.h"
#include "video_core/native_census.h"
#include "video_core/host_shaders/neo_id_derive_comp.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_draw_packet.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/amdgpu/mtb_census.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_dlss.h"
#include "video_core/renderer_vulkan/vk_dlss_upscale.h"
#include "video_core/renderer_vulkan/vk_dlss5.h"
#include "video_core/amdgpu/native_rec.h"
#include "video_core/amdgpu/lean_front_api.h"
#include "video_core/renderer_vulkan/vk_a2.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/water_scale.h"

#ifdef MemoryBarrier
#undef MemoryBarrier
#endif

namespace AmdGpu {
extern thread_local u64 g_back_idle_cyc; // liverpool.cpp, SHADPS4_DRAW_COST
}

namespace Vulkan {

namespace {
// Per-draw / per-binding environment flags, read once at startup. A function-local
// `static const` flag costs a guard check on every call (under the MSVC ABI: load the guard,
// _tls_index, gs:[0x58], the thread's TLS block, compare with _Init_thread_epoch); guard lines
// drew ~2.5% of the command processor's stack samples (a19, 2026-09-25). Values unchanged.
bool EnvIs1(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] == '1';
}
bool EnvExactly1(const char* name) {
    const char* value = std::getenv(name);
    return value && std::string_view{value} == "1";
}
const bool g_prepared_targets = EnvIs1("SHADPS4_PREPARED_TARGETS");
// SHADPS4_TEX_MEMO_LOCAL_GEN=1: texture bind memo validity per image (overlap_epoch) instead of the
// texture cache's global structure generation (scripts/patch/tex-memo-local-gen-2026-10-05.py).
const bool g_tex_memo_local = EnvIs1("SHADPS4_TEX_MEMO_LOCAL_GEN");
const bool g_verify_targets = EnvIs1("SHADPS4_PREPARED_TARGETS_VERIFY");
const bool g_raw_write_invalidate = EnvIs1("SHADPS4_RAW_WRITE_INVALIDATE");
// SHADPS4_READ_MEMO_PREFETCH (was a function-local static in BindBuffers' pass-1 loop: a guard
// check per binding).
const bool g_read_memo_prefetch = EnvIs1("SHADPS4_READ_MEMO_PREFETCH");
const bool g_cache_decoded_images = [] {
    const char* value = std::getenv("SHADPS4_DECODE_IMAGE_CACHE");
    return !value || value[0] != '0';
}();
const bool g_verify_decoded_images = EnvIs1("SHADPS4_DECODE_IMAGE_VERIFY");
const std::vector<u64> g_trace_hashes = [] {
    std::vector<u64> out;
    if (const char* value = std::getenv("SHADPS4_TRACE_SHADER")) {
        char* cursor = const_cast<char*>(value);
        while (*cursor) {
            out.push_back(std::strtoull(cursor, &cursor, 16));
            if (*cursor == ',') {
                ++cursor;
            } else {
                break;
            }
        }
    }
    return out;
}();
const s64 g_idbuf_fill = [] {
    const char* value = std::getenv("SHADPS4_NEO_IDBUF_FILL");
    return value ? static_cast<s64>(std::strtoll(value, nullptr, 0)) : s64{-1};
}();
const bool g_idbuf_derive = EnvIs1("SHADPS4_NEO_IDBUF_DERIVE");
const int g_viewport_memo = [] {
    const char* value = std::getenv("SHADPS4_VIEWPORT_MEMO");
    return value ? value[0] - '0' : 0;
}();
// SHADPS4_RENDER_STATE_MEMO=1 (=2 verify), default off; see RenderStateMemo in vk_rasterizer.h.
// Forced off with SHADPS4_STENCIL_PLANE_MERGE (FindDepthTarget merges stencil planes per draw).
const int g_render_state_memo = [] {
    const char* value = std::getenv("SHADPS4_RENDER_STATE_MEMO");
    const int mode = value ? value[0] - '0' : 0;
    return (mode == 1 || mode == 2) && !VideoCore::TextureCache::StencilPlaneMerge() ? mode : 0;
}();
const bool g_hair_stencil_trace = EnvExactly1("SHADPS4_HAIR_STENCIL_TRACE");
const bool g_replace_op_reference = EnvExactly1("SHADPS4_STENCIL_REPLACE_OP");

// SHADPS4_DRAW_PACKET=1 (=2 verify), default off (vk_draw_packet.h; notes/back-split-design-
// 2026-09-28.md phase 1): the direct draw and dispatch paths record buffer barriers, push
// constants and push descriptors as ONE worker record of flat bytes, folding the draw / dispatch
// in when the legacy path would record nothing between the descriptors and it; the recording
// worker rebuilds the WriteDescriptorSets from the pipeline's binding table. =2: the writes are
// rebuilt on the back too, compared field by field with the legacy set_writes (DrawPacket:
// MISMATCH, first 16), and the legacy path records.
/// SHADPS4_DRAW_REPEAT (0 off / 1 skip / 2 verify): see Rasterizer::DrawRepeatMatch.
const int g_draw_repeat = [] {
    const char* value = std::getenv("SHADPS4_DRAW_REPEAT");
    return value ? std::atoi(value) : 0;
}();

const int g_draw_packet = [] {
    const char* value = std::getenv("SHADPS4_DRAW_PACKET");
    const int mode = value ? value[0] - '0' : 0;
    return (mode == 1 || mode == 2) ? mode : 0;
}();
// Counters of the thread that executes draws (one thread). Cumulative; logged every ~10 s.
struct DrawPacketStats {
    u64 draws{};        // EmitPipelineResourcesOrPacket calls (draws + direct dispatches)
    u64 packets{};      // sent as a packet
    u64 folded{};       // packets that also carried the draw / dispatch
    u64 fb_oversize{};  // legacy: payload over DrawPacketMaxBytes
    u64 fb_occlusion{}; // legacy: draw inside an occlusion query
    u64 fb_other{};     // legacy: no push descriptors / native bindings / sync recording / shape
    u64 verify_checked{};
    u64 mismatches{};
    u64 fold_misses{};  // a "would record nothing" prediction that recorded (see DrawTail)
    u64 packet_bytes{}; // chunk bytes of the packets plus, when not folded, the draw record
    u64 legacy_bytes{}; // chunk bytes the legacy records of every draw would take (estimate)
    std::chrono::steady_clock::time_point last_log{};
};
DrawPacketStats g_dp_stats;

void DrawPacketLog() {
    const auto& s = g_dp_stats;
    LOG_INFO(Render_Vulkan,
             "DrawPacket: mode={} draws={} packets={} folded_draws={} "
             "fallbacks(oversize/occlusion/other)={}/{}/{} bytes_per_packet={} "
             "legacy_bytes_per_draw={} (both estimated from the chunk layout) verify_checked={} "
             "mismatches={} fold_misses={}",
             g_draw_packet, s.draws, s.packets, s.folded, s.fb_oversize, s.fb_occlusion,
             s.fb_other, s.packet_bytes / std::max<u64>(s.packets, 1),
             s.legacy_bytes / std::max<u64>(s.draws, 1), s.verify_checked, s.mismatches,
             s.fold_misses);
}

void NoteDrawPacketFoldMiss(const char* what) {
    if (++g_dp_stats.fold_misses <= 16) {
        LOG_CRITICAL(Render_Vulkan,
                     "DrawPacket: fold prediction missed, {} recorded while the packet was "
                     "pending (packet recorded after it)",
                     what);
    }
}

// SHADPS4_DRAW_COST=1 (diagnostic, default off): rdtsc cycles of each DrawTail phase per graphics
// pipeline on the thread that executes draws (the CP back with the split), logged every 2^20
// draws as the top pipelines by total cost plus a depth-only/no-image class total. Answers which
// kinds of draws the CP time goes to (frame map 2026-09-27). Direct draws only (DrawTail).
const bool g_draw_cost = EnvIs1("SHADPS4_DRAW_COST");
enum DrawCostPhase : u32 {
    DcPrepare,
    DcBindBuffers,
    DcBindTextures,
    DcBindOther,
    DcBeginRendering,
    DcVertexIndex,
    DcEmitResources,
    DcRecordTail,
    DcCount
};
constexpr std::array<const char*, DcCount> kDrawCostNames{
    "prep", "buf", "tex", "bind_other", "begin", "vtx_idx", "emit", "tail"};
thread_local u64 g_dc_bind_buf = 0;
thread_local u64 g_dc_bind_tex = 0;
// Buffer bindings (BindBuffers, draws and dispatches): [0] read-only, [1] written; phases:
// 0 find pass (per stage, split evenly), 1 sync walk, 2 sync rest, 3 other ObtainBuffer work,
// 4 barrier + written-branch invalidation + descriptor info; plus counts and special bindings.
thread_local std::array<std::array<u64, 5>, 2> g_bb_cyc{};
thread_local std::array<u64, 2> g_bb_n{};
thread_local u64 g_bb_special_n = 0;
// Repeat census for read bindings (DRAW_COST): same (buffer, base, size) already bound since the
// last scheduler tick change (= submission), and whether the dirty epoch is unchanged since then.
thread_local u64 g_rep_total = 0;
thread_local u64 g_rep_hit = 0;
thread_local u64 g_rep_hit_clean = 0;
thread_local u64 g_rep_distinct = 0;
void NoteReadBindingRepeat(u64 tick, u32 buffer_id, u64 base, u64 size) {
    thread_local u64 last_tick = 0;
    thread_local std::unordered_map<u64, u64> seen; // key -> dirty epoch at first bind
    if (tick != last_tick) {
        g_rep_distinct += seen.size();
        seen.clear();
        last_tick = tick;
    }
    const u64 key = (base * 0x9E3779B97F4A7C15ULL) ^ (size << 20) ^ buffer_id;
    const u64 epoch = VideoCore::g_dirty_epoch.load(std::memory_order_relaxed);
    ++g_rep_total;
    const auto [it, inserted] = seen.try_emplace(key, epoch);
    if (!inserted) {
        ++g_rep_hit;
        if (it->second == epoch) {
            ++g_rep_hit_clean;
        } else {
            it->second = epoch;
        }
    }
}
thread_local u64 g_dc_disp_cyc = 0; // ExecuteDispatch on the back (direct + indirect)
thread_local u64 g_dc_disp_n = 0;
// Inside BindTextures (DRAW_COST only): [0] first pass (FindImage per binding), [1] FindTexture
// (view lookup, refresh/upload), [2] Transit (layout changes, may end rendering), plus the single
// slowest FindTexture+Transit image of the draw. DrawTail logs the split for draws whose texture
// phase exceeds ~50 us (first 40 such draws).
thread_local std::array<u64, 3> g_dc_tex_sub{};
struct SlowImage {
    u64 cyc{};
    u64 addr{};
    u32 w{};
    u32 h{};
    u32 layers{};
    u32 fmt{};
    bool depth{};
    bool tiled{};
    u32 layout_before{};
};
thread_local SlowImage g_dc_slow_img{};
// Program 5 gate (2026-10-09 evening, dcb-parallel note sec. 6): which part of a draw's binding
// work is a pure function of guest data (could run on the game's job thread) and which needs the
// back's state. Per draw, reset in BindResources: 0 buffer pass 1 (V# decode + FindBuffer),
// 1 special bindings (flat-buffer copy, null clip planes, ...), 2 cache bindings (Obtain, barrier,
// descriptor info), 3 texture memo key (T# / S# decode, hash, way search), 4 texture memo
// validation (memo section, per-image checks, LRU / sampler touches), 5 texture memo hit emit,
// 6 texture memo miss (full BindTextures), 7 stages without images or samplers.
enum P5Part : u32 { P5Find, P5Special, P5Cache, P5TexKey, P5TexValid, P5TexEmit, P5TexMiss,
                    P5TexEmpty, P5Count };
constexpr std::array<const char*, P5Count> kP5Names{"buf_find", "buf_special", "buf_cache",
                                                    "tex_key",  "tex_valid",   "tex_emit",
                                                    "tex_miss", "tex_empty"};
thread_local std::array<u64, P5Count> g_p5{};
// SHADPS4_DRAW_REPEAT_HEAP=1 (2026-10-09, dcb note sec. 9): a draw repeats the previous one when
// only its per-frame-heap V#s moved (same buffer, new offset); identical formatted reads allowed.
const bool g_repeat_heap = [] {
    const char* v = std::getenv("SHADPS4_DRAW_REPEAT_HEAP");
    return v && v[0] == '1';
}();
constexpr VAddr RepeatHeapLo = 0x248000000ULL;
constexpr VAddr RepeatHeapHi = 0x24c000000ULL;
// DRAW_REPEAT=2 diagnostic: which binding of the full path recorded commands (a skip omits them).
thread_local std::string g_rp_seq_who;

// SHADPS4_BIND_REGION_CENSUS=1 (2026-10-09, native resource model, dcb-parallel note sec. 8): which
// guest memory regions (16 MB granules) the graphics bindings read, what each costs on the back and
// how often its data had to be uploaded (CPU-written since the last upload). Back thread only.
const bool g_bind_region_census = [] {
    const char* v = std::getenv("SHADPS4_BIND_REGION_CENSUS");
    return v && v[0] == '1';
}();
struct RegionStats {
    u64 n = 0, cyc = 0, uploaded = 0, formatted = 0, written = 0, bytes = 0, img = 0;
};
struct RegionCensus {
    std::unordered_map<u32, RegionStats> by;
    u64 bindings = 0, images = 0, draws = 0;
    u64 tsc0 = 0;
    std::chrono::steady_clock::time_point t0{};
};
// SHADPS4_STATIC_KEY_CENSUS=1 (2026-10-09, dcb note sec. 8, s355): does the STATIC part of a full
// draw's bindings repeat? Key = pipeline + every V# outside the per-frame heap (granules
// 0x248-0x24b; heap V#s contribute only their size / stride / format, not the address) + T#s + S#s.
// Hit = the key was seen in this frame or the previous one. Back thread only.
const bool g_static_key_census = [] {
    const char* v = std::getenv("SHADPS4_STATIC_KEY_CENSUS");
    return v && v[0] == '1';
}();
constexpr VAddr HeapLo = 0x248000000ULL;
constexpr VAddr HeapHi = 0x24c000000ULL;
struct StaticKeyRec {
    u64 frame = 0;
    u64 clock = 0; // granule clock when stored (before this draw's resolution)
    u64 img_epoch = 0, tex_gen = 0, smp_gen = 0, buf_gen = 0;
};
struct StaticKeyCensus {
    std::unordered_map<u64, StaticKeyRec> seen; // key hash -> last occurrence
    std::vector<std::pair<VAddr, VAddr>> ranges; // this draw's static buffer ranges
    u64 hit_buf_ok = 0, hit_img_ok = 0, hit_both_ok = 0, big_bindings = 0, hit_textured_ok = 0;
    // Previous FULL draw (BindResources) had the same static key; and the previous draw of any
    // kind was that full draw (no repeat / join in between), the draw-repeat shape.
    u64 prev_key = 0, prev_full_seq = 0, prev_match = 0, prev_match_textured = 0,
        prev_match_adjacent = 0;
    std::unordered_set<u32> heap_ids;  // heap buffer ids this window
    u64 frame = 0;
    u64 draws = 0, textured = 0, hit = 0, hit_textured = 0, hit_same_frame = 0;
    u64 heap_bindings = 0, static_bindings = 0, frames = 0, distinct_new = 0;
    u64 keys_this_frame = 0, keys_frames_sum = 0;
};
StaticKeyCensus& GetStaticKeyCensus() {
    static StaticKeyCensus c;
    return c;
}
inline u64 KeyMix(u64 h, const void* data, size_t bytes) {
    const auto* p = static_cast<const u8*>(data);
    for (size_t i = 0; i < bytes; ++i) {
        h = (h ^ p[i]) * 0x100000001b3ULL;
    }
    return h;
}
thread_local bool g_skey_last_hit = false; // the current full draw's static key was seen recently
thread_local bool g_skey_last_prev = false; // ... and equals the previous full draw's
void NoteStaticKeyHeapId(u32 id) {
    GetStaticKeyCensus().heap_ids.insert(id);
}
RegionCensus& GetRegionCensus() {
    static RegionCensus c;
    return c;
}
void RegionCensusLog(RegionCensus& c) {
    const u64 now_tsc = __rdtsc();
    const double wall_ns = double(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() - c.t0)
                                      .count());
    const double ns_per_cyc = wall_ns / double(std::max<u64>(now_tsc - c.tsc0, 1));
    std::vector<std::pair<u32, RegionStats>> v(c.by.begin(), c.by.end());
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
        return a.second.cyc + a.second.img > b.second.cyc + b.second.img;
    });
    u64 all_cyc = 0;
    for (const auto& [k, r] : v) {
        all_cyc += r.cyc;
    }
    std::string line;
    for (size_t i = 0; i < std::min<size_t>(v.size(), 24); ++i) {
        const auto& [k, r] = v[i];
        line += fmt::format(" | {:#011x}: buf n={} {:.0f}ns {:.1f}%cyc up={:.1f}% fmt={:.0f}% wr={:.0f}% "
                            "avgKB={:.0f} img={}",
                            u64(k) << 24, r.n, r.n ? double(r.cyc) * ns_per_cyc / double(r.n) : 0.0,
                            all_cyc ? 100.0 * double(r.cyc) / double(all_cyc) : 0.0,
                            r.n ? 100.0 * double(r.uploaded) / double(r.n) : 0.0,
                            r.n ? 100.0 * double(r.formatted) / double(r.n) : 0.0,
                            r.n ? 100.0 * double(r.written) / double(r.n) : 0.0,
                            r.n ? double(r.bytes) / 1024.0 / double(r.n) : 0.0, r.img);
    }
    LOG_INFO(Render_Vulkan,
             "BindRegionCensus: window {:.0f} ms, {} graphics buffer bindings ({:.0f} ns total each), "
             "{} image bindings, {} regions:{}",
             wall_ns / 1e6, c.bindings,
             c.bindings ? double(all_cyc) * ns_per_cyc / double(c.bindings) : 0.0, c.images,
             v.size(), line);
    c.by.clear();
    c.bindings = c.images = 0;
    c.tsc0 = __rdtsc();
    c.t0 = std::chrono::steady_clock::now();
}
// The per-binding repeat census below allocates in a hash map inside the timed buffer phase;
// SHADPS4_DRAW_COST_REPEAT=1 keeps it, off by default so the phases are not inflated.
const bool g_dc_repeat_census = [] {
    const char* v = std::getenv("SHADPS4_DRAW_COST_REPEAT");
    return v && v[0] == '1';
}();

struct DrawCostStats {
    u64 n = 0;
    std::array<u64, DcCount> cyc{};
    std::array<u64, P5Count> p5{};
};

bool IsDepthOnlyNoImage(const GraphicsPipeline* pipeline) {
    // mrt_mask is the shader's export mask (0x1 even for the empty depth-only shader); the
    // attachment count is what the capture shows as "no colour output".
    if (pipeline->GetGraphicsKey().num_color_attachments != 0) {
        return false;
    }
    for (const auto* stage : pipeline->GetStages()) {
        if (stage && !stage->images.empty()) {
            return false;
        }
    }
    return true;
}

void NoteDrawCost(const GraphicsPipeline* pipeline, const std::array<u64, DcCount>& cyc,
                  bool repeat) {
    // 2026-10-09 afternoon (colour-pass note sec. 6): phases by class, bit 0 = draw-repeat hit,
    // bit 1 = the pipeline samples images.
    thread_local std::array<DrawCostStats, 4> by_class{};
    {
        bool images = false;
        for (const auto* stage : pipeline->GetStages()) {
            images = images || (stage && !stage->images.empty());
        }
        auto& c = by_class[(repeat ? 1u : 0u) | (images ? 2u : 0u)];
        ++c.n;
        for (u32 i = 0; i < DcCount; ++i) {
            c.cyc[i] += cyc[i];
        }
        if (!repeat) {
            for (u32 i = 0; i < P5Count; ++i) {
                c.p5[i] += g_p5[i];
            }
        }
    }
    thread_local std::unordered_map<const GraphicsPipeline*, DrawCostStats> stats;
    thread_local u64 draws = 0;
    thread_local u64 win_tsc = 0;
    thread_local std::chrono::steady_clock::time_point win_start{};
    if (win_tsc == 0) {
        win_tsc = __rdtsc();
        win_start = std::chrono::steady_clock::now();
    }
    auto& s = stats[pipeline];
    ++s.n;
    for (u32 i = 0; i < DcCount; ++i) {
        s.cyc[i] += cyc[i];
    }
    if ((++draws & 0xFFFFF) != 0) {
        return;
    }
    const u64 now_tsc = __rdtsc();
    const double wall_ns = double(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() - win_start)
                                      .count());
    const double ns_per_cyc = wall_ns / double(now_tsc - win_tsc);
    std::vector<std::pair<const GraphicsPipeline*, const DrawCostStats*>> order;
    order.reserve(stats.size());
    u64 all_cyc = 0;
    u64 depth_cyc = 0;
    u64 depth_n = 0;
    std::array<u64, DcCount> all_phase{};
    std::array<u64, DcCount> depth_phase{};
    for (const auto& [p, st] : stats) {
        order.emplace_back(p, &st);
        const bool depth_only = IsDepthOnlyNoImage(p);
        for (u32 i = 0; i < DcCount; ++i) {
            all_cyc += st.cyc[i];
            all_phase[i] += st.cyc[i];
            if (depth_only) {
                depth_cyc += st.cyc[i];
                depth_phase[i] += st.cyc[i];
            }
        }
        if (depth_only) {
            depth_n += st.n;
        }
    }
    const auto total_of = [](const DrawCostStats& st) {
        u64 t = 0;
        for (const u64 c : st.cyc) {
            t += c;
        }
        return t;
    };
    std::sort(order.begin(), order.end(),
              [&](const auto& a, const auto& b) { return total_of(*a.second) > total_of(*b.second); });
    const auto phases = [&](const std::array<u64, DcCount>& ph, u64 n) {
        std::string out;
        for (u32 i = 0; i < DcCount; ++i) {
            out += fmt::format(" {}={:.0f}", kDrawCostNames[i],
                               n ? double(ph[i]) * ns_per_cyc / double(n) : 0.0);
        }
        return out;
    };
    LOG_INFO(Render_Vulkan,
             "DrawCost: window {:.0f} ms, {} draws, draw time {:.1f}% of wall, {:.0f} ns/draw;{}",
             wall_ns / 1e6, draws, 100.0 * double(all_cyc) * ns_per_cyc / wall_ns,
             double(all_cyc) * ns_per_cyc / double(draws), phases(all_phase, draws));
    {
        static constexpr std::array<const char*, 4> kClass{"full_untextured", "repeat_untextured",
                                                           "full_textured", "repeat_textured"};
        std::string line;
        for (u32 k = 0; k < 4; ++k) {
            const auto& c = by_class[k];
            u64 t = 0;
            for (const u64 v : c.cyc) {
                t += v;
            }
            line += fmt::format(" | {} n={} {:.0f} ns/draw:{}", kClass[k], c.n,
                                c.n ? double(t) * ns_per_cyc / double(c.n) : 0.0,
                                phases(c.cyc, c.n));
        }
        LOG_INFO(Render_Vulkan, "DrawCostClass:{}", line);
        // Program 5 gate: binding parts of the full (non-repeat) draws, ns per draw.
        std::string p5line;
        for (const u32 k : {0u, 2u}) {
            const auto& c = by_class[k];
            p5line += fmt::format(" | {} n={}:", kClass[k], c.n);
            for (u32 i = 0; i < P5Count; ++i) {
                p5line += fmt::format(" {}={:.0f}", kP5Names[i],
                                      c.n ? double(c.p5[i]) * ns_per_cyc / double(c.n) : 0.0);
            }
        }
        LOG_INFO(Render_Vulkan, "DrawCostP5:{}", p5line);
    }
    LOG_INFO(Render_Vulkan,
             "DrawCost: back idle (WaitForWork) {:.1f}% of wall; dispatches {} = {:.1f}% of wall, "
             "{:.0f} ns each",
             100.0 * double(AmdGpu::g_back_idle_cyc) * ns_per_cyc / wall_ns, g_dc_disp_n,
             100.0 * double(g_dc_disp_cyc) * ns_per_cyc / wall_ns,
             g_dc_disp_n ? double(g_dc_disp_cyc) * ns_per_cyc / double(g_dc_disp_n) : 0.0);
    for (u32 w = 0; w < 2; ++w) {
        const double n = double(std::max<u64>(g_bb_n[w], 1));
        const auto& c = g_bb_cyc[w];
        LOG_INFO(Render_Vulkan,
                 "DrawCost: buffer bindings {}: {} ({:.0f}/draw) ns each: find={:.0f} walk={:.0f} "
                 "sync_rest={:.0f} obtain_other={:.0f} barrier_written={:.0f} total={:.0f}; "
                 "{:.1f}% of wall",
                 w ? "written" : "read", g_bb_n[w], double(g_bb_n[w]) / double(draws),
                 double(c[0]) * ns_per_cyc / n, double(c[1]) * ns_per_cyc / n,
                 double(c[2]) * ns_per_cyc / n, double(c[3]) * ns_per_cyc / n,
                 double(c[4]) * ns_per_cyc / n,
                 double(c[0] + c[1] + c[2] + c[3] + c[4]) * ns_per_cyc / n,
                 100.0 * double(c[0] + c[1] + c[2] + c[3] + c[4]) * ns_per_cyc / wall_ns);
    }
    LOG_INFO(Render_Vulkan, "DrawCost: special (non-cache) buffer bindings {}", g_bb_special_n);
    LOG_INFO(Render_Vulkan,
             "DrawCost: read-binding repeats within a submission: {} of {} ({:.1f}%), with no dirty-epoch "
             "change since the previous bind {} ({:.1f}%); distinct per submission (summed) {}",
             g_rep_hit, g_rep_total, 100.0 * double(g_rep_hit) / double(std::max<u64>(g_rep_total, 1)),
             g_rep_hit_clean, 100.0 * double(g_rep_hit_clean) / double(std::max<u64>(g_rep_total, 1)),
             g_rep_distinct);
    g_rep_total = g_rep_hit = g_rep_hit_clean = g_rep_distinct = 0;
    g_bb_cyc = {};
    g_bb_n = {};
    g_bb_special_n = 0;
    AmdGpu::g_back_idle_cyc = 0;
    g_dc_disp_cyc = 0;
    g_dc_disp_n = 0;
    LOG_INFO(Render_Vulkan,
             "DrawCost: depth-only/no-image class: {} draws ({:.1f}%), {:.1f}% of draw time, "
             "{:.0f} ns/draw;{}",
             depth_n, 100.0 * double(depth_n) / double(draws),
             all_cyc ? 100.0 * double(depth_cyc) / double(all_cyc) : 0.0,
             depth_n ? double(depth_cyc) * ns_per_cyc / double(depth_n) : 0.0,
             phases(depth_phase, depth_n));
    for (size_t k = 0; k < std::min<size_t>(order.size(), 30); ++k) {
        const auto* p = order[k].first;
        const auto& st = *order[k].second;
        u64 vs = 0;
        u64 fs = 0;
        for (const auto* stage : p->GetStages()) {
            if (!stage) {
                continue;
            }
            if (stage->l_stage == Shader::LogicalStage::Vertex) {
                vs = stage->pgm_hash;
            } else if (stage->l_stage == Shader::LogicalStage::Fragment) {
                fs = stage->pgm_hash;
            }
        }
        LOG_INFO(Render_Vulkan,
                 "DrawCost: #{} fs_{:08x} vs_{:08x} mrt={:#x} depth_only={} n={} share={:.1f}% "
                 "ns/draw={:.0f};{}",
                 k, fs, vs, p->GetGraphicsKey().mrt_mask, IsDepthOnlyNoImage(p) ? 1 : 0, st.n,
                 all_cyc ? 100.0 * double(total_of(st)) / double(all_cyc) : 0.0,
                 double(total_of(st)) * ns_per_cyc / double(st.n), phases(st.cyc, st.n));
    }
    by_class = {};
    stats.clear();
    draws = 0;
    win_tsc = __rdtsc();
    win_start = std::chrono::steady_clock::now();
}

// SHADPS4_DRAW_COST=1, dispatches: per compute pipeline, total cycles and the BindResources
// buffer/texture split (from BindResources' own counters); top 20 every 2^18 dispatches.
struct DispatchCostStats {
    u64 n = 0;
    u64 total = 0;
    u64 buf = 0;
    u64 tex = 0;
};
void NoteDispatchCost(const ComputePipeline* pipeline, u64 total, u64 buf, u64 tex) {
    thread_local std::unordered_map<const ComputePipeline*, DispatchCostStats> stats;
    thread_local u64 dispatches = 0;
    thread_local u64 win_tsc = 0;
    thread_local std::chrono::steady_clock::time_point win_start{};
    if (win_tsc == 0) {
        win_tsc = __rdtsc();
        win_start = std::chrono::steady_clock::now();
    }
    auto& st = stats[pipeline];
    ++st.n;
    st.total += total;
    st.buf += buf;
    st.tex += tex;
    if ((++dispatches & 0x3FFFF) != 0) {
        return;
    }
    const double wall_ns = double(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() - win_start)
                                      .count());
    const double ns_per_cyc = wall_ns / double(__rdtsc() - win_tsc);
    std::vector<std::pair<const ComputePipeline*, const DispatchCostStats*>> order;
    u64 all = 0;
    for (const auto& [p, v] : stats) {
        order.emplace_back(p, &v);
        all += v.total;
    }
    std::sort(order.begin(), order.end(),
              [](const auto& a, const auto& b) { return a.second->total > b.second->total; });
    LOG_INFO(Render_Vulkan, "DispatchCost: window {:.0f} ms, {} dispatches, {:.1f}% of wall, {:.0f} ns each",
             wall_ns / 1e6, dispatches, 100.0 * double(all) * ns_per_cyc / wall_ns,
             double(all) * ns_per_cyc / double(dispatches));
    for (size_t k = 0; k < std::min<size_t>(order.size(), 20); ++k) {
        const auto* p = order[k].first;
        const auto& v = *order[k].second;
        u64 cs = 0;
        for (const auto* stage : p->GetStages()) {
            if (stage) {
                cs = stage->pgm_hash;
            }
        }
        LOG_INFO(Render_Vulkan,
                 "DispatchCost: #{} cs_{:08x} n={} share={:.1f}% ns/disp={:.0f} buf={:.0f} tex={:.0f} "
                 "other={:.0f} buffers={} images={}",
                 k, cs, v.n, all ? 100.0 * double(v.total) / double(all) : 0.0,
                 double(v.total) * ns_per_cyc / double(v.n), double(v.buf) * ns_per_cyc / double(v.n),
                 double(v.tex) * ns_per_cyc / double(v.n),
                 double(v.total - v.buf - v.tex) * ns_per_cyc / double(v.n),
                 p->GetStages()[0] ? p->GetStages()[0]->buffers.size() : 0,
                 p->GetStages()[0] ? p->GetStages()[0]->images.size() : 0);
    }
    stats.clear();
    dispatches = 0;
    win_tsc = __rdtsc();
    win_start = std::chrono::steady_clock::now();
}

// SHADPS4_MERGE_PROBE=1 (diagnostic, default off): for each direct draw that uses the same
// graphics pipeline as the previous direct draw, record which inputs differ from that draw.
// Bits: user-data regs, flattened user data, buffer V# base, buffer V# size/stride, image T#,
// vertex V# base, index base, index offset (VGT index_offset / vertex offset), index count,
// instance count, render-target change (a BeginRendering in between). Per pipeline: draws,
// same-pipeline repeats, top diff masks, OR of differing user-data regs and buffer slots.
// Logged every 2^20 draws. Answers whether consecutive draws can merge into one call.
const bool g_merge_probe = EnvIs1("SHADPS4_MERGE_PROBE");
enum MergeDiffBit : u32 {
    MdUserData = 1u << 0,
    MdFlatUd = 1u << 1,
    MdBufBase = 1u << 2,
    MdBufShape = 1u << 3,
    MdImage = 1u << 4,
    MdVertexBase = 1u << 5,
    MdIndexBase = 1u << 6,
    MdIndexOffset = 1u << 7,
    MdCount = 1u << 8,
    MdInstances = 1u << 9,
    MdTarget = 1u << 10,
};
constexpr std::array<const char*, 11> kMergeDiffNames{
    "ud", "flat", "bufbase", "bufshape", "img", "vbase", "ibase", "ioff", "count", "inst", "rt"};

struct MergeSig {
    const GraphicsPipeline* pipeline{};
    std::vector<u32> ud;
    std::vector<u32> flat;
    std::vector<u64> buf_base;
    std::vector<u64> buf_shape;
    std::vector<u64> img;
    std::vector<u64> vbase;
    u64 ibase{};
    u32 ioff{};
    u32 count{};
    u32 inst{};
    u64 rt_epoch{};
};

struct MergeStats {
    u64 n{};
    u64 repeats{};
    u32 ud_regs{};  // OR of differing user-data register indices (VS in low 16, FS in high 16)
    u64 buf_slots{}; // OR of differing buffer slots (VS slots 0-31, FS slots 32-63)
    std::unordered_map<u32, u64> masks;
};

u64 g_merge_rt_epoch = 0; // bumped by BeginRendering changes (draw thread only)

void NoteMergeProbe(const GraphicsPipeline* pipeline, std::span<const Shader::BoundStage> stages,
                    const AmdGpu::Regs& regs, bool is_indexed, u32 index_offset) {
    thread_local MergeSig prev;
    thread_local MergeSig cur;
    thread_local std::unordered_map<const GraphicsPipeline*, MergeStats> stats;
    thread_local u64 draws = 0;
    cur.pipeline = pipeline;
    cur.ud.clear();
    cur.flat.clear();
    cur.buf_base.clear();
    cur.buf_shape.clear();
    cur.img.clear();
    cur.vbase.clear();
    u32 ud_diff = 0;
    u64 slot_diff = 0;
    const bool comparable = prev.pipeline == pipeline;
    u32 stage_no = 0;
    for (const auto& bound : stages) {
        const auto* info = bound.info;
        if (!info) {
            continue;
        }
        const bool is_vs = info->l_stage == Shader::LogicalStage::Vertex;
        const u32 ud_begin = static_cast<u32>(cur.ud.size());
        for (u32 i = 0; i < bound.user_data.size() && i < 16; ++i) {
            cur.ud.push_back(bound.user_data[i]);
        }
        cur.flat.insert(cur.flat.end(), bound.flattened_ud_buf.begin(),
                        bound.flattened_ud_buf.end());
        const u32 buf_begin = static_cast<u32>(cur.buf_base.size());
        for (const auto& desc : info->buffers) {
            const auto vsharp = desc.GetSharp(bound);
            cur.buf_base.push_back(vsharp.base_address);
            cur.buf_shape.push_back(u64(vsharp.GetSize()) << 32 | vsharp.GetStride());
        }
        for (const auto& desc : info->images) {
            const auto tsharp = desc.GetSharp(bound);
            u64 h = 0;
            const auto* words = reinterpret_cast<const u64*>(&tsharp);
            for (size_t w = 0; w < sizeof(tsharp) / sizeof(u64); ++w) {
                h = h * 0x9E3779B97F4A7C15ULL ^ words[w];
            }
            cur.img.push_back(h);
        }
        if (is_vs) {
            if (const auto* fetch = pipeline->GetFetchShader()) {
                for (const auto& attrib : fetch->attributes) {
                    cur.vbase.push_back(attrib.GetSharp(bound).base_address);
                }
            }
        }
        if (comparable) {
            for (u32 i = ud_begin; i < cur.ud.size() && i < prev.ud.size(); ++i) {
                if (cur.ud[i] != prev.ud[i]) {
                    ud_diff |= 1u << ((i - ud_begin) + (is_vs ? 0 : 16));
                }
            }
            for (u32 i = buf_begin; i < cur.buf_base.size() && i < prev.buf_base.size(); ++i) {
                if (cur.buf_base[i] != prev.buf_base[i] || cur.buf_shape[i] != prev.buf_shape[i]) {
                    slot_diff |= u64{1} << (((i - buf_begin) & 31) + (is_vs ? 0 : 32));
                }
            }
        }
        ++stage_no;
    }
    cur.ibase = is_indexed ? regs.index_base_address.Address() : 0;
    cur.ioff = index_offset;
    cur.count = regs.num_indices;
    cur.inst = regs.num_instances.NumInstances();
    cur.rt_epoch = g_merge_rt_epoch;
    auto& st = stats[pipeline];
    ++st.n;
    if (comparable) {
        u32 mask = 0;
        mask |= cur.ud != prev.ud ? MdUserData : 0;
        mask |= cur.flat != prev.flat ? MdFlatUd : 0;
        mask |= cur.buf_base != prev.buf_base ? MdBufBase : 0;
        mask |= cur.buf_shape != prev.buf_shape ? MdBufShape : 0;
        mask |= cur.img != prev.img ? MdImage : 0;
        mask |= cur.vbase != prev.vbase ? MdVertexBase : 0;
        mask |= cur.ibase != prev.ibase ? MdIndexBase : 0;
        mask |= cur.ioff != prev.ioff ? MdIndexOffset : 0;
        mask |= cur.count != prev.count ? MdCount : 0;
        mask |= cur.inst != prev.inst ? MdInstances : 0;
        mask |= cur.rt_epoch != prev.rt_epoch ? MdTarget : 0;
        ++st.repeats;
        ++st.masks[mask];
        st.ud_regs |= ud_diff;
        st.buf_slots |= slot_diff;
    }
    std::swap(prev, cur);
    if ((++draws & 0xFFFFF) != 0) {
        return;
    }
    std::vector<std::pair<const GraphicsPipeline*, const MergeStats*>> order;
    for (const auto& [p, s] : stats) {
        order.emplace_back(p, &s);
    }
    std::sort(order.begin(), order.end(),
              [](const auto& a, const auto& b) { return a.second->n > b.second->n; });
    const auto mask_name = [](u32 mask) {
        if (mask == 0) {
            return std::string("none");
        }
        std::string out;
        for (u32 b = 0; b < kMergeDiffNames.size(); ++b) {
            if (mask & (1u << b)) {
                out += out.empty() ? "" : "+";
                out += kMergeDiffNames[b];
            }
        }
        return out;
    };
    for (size_t k = 0; k < std::min<size_t>(order.size(), 16); ++k) {
        const auto* p = order[k].first;
        const auto& s = *order[k].second;
        u64 vs = 0;
        u64 fs = 0;
        for (const auto* stage : p->GetStages()) {
            if (stage && stage->l_stage == Shader::LogicalStage::Vertex) {
                vs = stage->pgm_hash;
            } else if (stage && stage->l_stage == Shader::LogicalStage::Fragment) {
                fs = stage->pgm_hash;
            }
        }
        std::vector<std::pair<u32, u64>> masks(s.masks.begin(), s.masks.end());
        std::sort(masks.begin(), masks.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        std::string top;
        for (size_t m = 0; m < std::min<size_t>(masks.size(), 5); ++m) {
            top += fmt::format(" {}:{:.1f}%", mask_name(masks[m].first),
                               100.0 * double(masks[m].second) / double(s.repeats));
        }
        LOG_INFO(Render_Vulkan,
                 "MergeProbe: #{} fs_{:08x} vs_{:08x} n={} repeats={} ({:.1f}%) ud_regs={:#010x} "
                 "buf_slots={:#018x} masks:{}",
                 k, fs, vs, s.n, s.repeats, 100.0 * double(s.repeats) / double(s.n), s.ud_regs,
                 s.buf_slots, top);
    }
    stats.clear();
    draws = 0;
}
// SHADPS4_NATIVE_CENSUS=1 (diagnostic, default off; native-path study 2026-10-09): could this
// draw join the previous draw in ONE host multi-draw-indirect call? Host-level conditions: same
// pipeline, same rendering state, nothing recorded between the two draws, no dynamic-state
// change, no buffer barrier, identical image / sampler bindings, the same VkBuffer in every
// buffer slot (offsets may differ: they would travel in a per-draw record), the same VkBuffer in
// every vertex slot with offsets one common multiple of the strides apart (a vertexOffset), the
// same index VkBuffer / type, the same viewport push data, no occlusion query. The first failing
// condition is counted. Groups: depth-only (no colour attachment) vs colour. Logged every 2^20
// draws.
enum NativeBreak : u32 {
    NbFirst,
    NbBetween,
    NbPipeline,
    NbRendering,
    NbDynState,
    NbBarrier,
    NbImages,
    NbBuffers,
    NbVertex,
    NbIndex,
    NbViewport,
    NbOcclusion,
    NbGroup,
    NbCount
};
constexpr std::array<const char*, NbCount> kNativeBreakNames{
    "first",  "between", "pipeline", "rendering", "dynstate", "barrier",   "images",
    "buffers", "vertex", "index",    "viewport",  "occlusion", "group"};
struct NativeCensusGroup {
    u64 draws{};
    u64 joins{};
    u64 runs{};
    u64 run_hist[6]{}; // run lengths 1, 2-3, 4-7, 8-15, 16-63, 64+
    u64 cur_run{};
    std::array<u64, NbCount> breaks{};
};
struct NativeCensusPrev {
    const void* pipeline{};
    RenderState state{};
    u64 end_seq{~0ULL};
    std::vector<vk::DescriptorImageInfo> images;
    std::vector<VkBuffer> buffers;
    VideoCore::NativeCensusVi vi{};
    std::array<float, 4> vp{};
    u32 group{};
};
thread_local NativeCensusPrev t_nc_prev;
thread_local std::array<NativeCensusGroup, 2> t_nc_groups;
thread_local std::unordered_map<u64, std::array<u64, 3>> t_nc_pipes; // fs -> draws, joins, runs
thread_local u64 t_nc_draws = 0;

void NativeCensusCloseRun(NativeCensusGroup& g) {
    if (g.cur_run == 0) {
        return;
    }
    ++g.runs;
    const u64 r = g.cur_run;
    g.run_hist[r == 1 ? 0 : r < 4 ? 1 : r < 8 ? 2 : r < 16 ? 3 : r < 64 ? 4 : 5]++;
    g.cur_run = 0;
}

u32 NativeCensusWhy(const NativeCensusPrev& prev, const GraphicsPipeline* pipeline,
                    const RenderState& state, u64 entry_seq, bool dyn_change, bool barriers,
                    std::span<const vk::DescriptorImageInfo> images,
                    std::span<const vk::DescriptorBufferInfo> buffers,
                    const Shader::PushData& push, bool occlusion, u32 group) {
    const auto& vi = VideoCore::t_native_vi;
    if (prev.pipeline == nullptr) {
        return NbFirst;
    }
    if (entry_seq != prev.end_seq) {
        return NbBetween;
    }
    if (prev.pipeline != pipeline) {
        return NbPipeline;
    }
    if (prev.group != group || !(prev.state == state)) {
        return NbRendering;
    }
    if (dyn_change) {
        return NbDynState;
    }
    if (barriers) {
        return NbBarrier;
    }
    if (occlusion) {
        return NbOcclusion;
    }
    if (images.size() != prev.images.size() ||
        !std::equal(images.begin(), images.end(), prev.images.begin())) {
        return NbImages;
    }
    if (buffers.size() != prev.buffers.size()) {
        return NbBuffers;
    }
    for (size_t i = 0; i < buffers.size(); ++i) {
        if (static_cast<VkBuffer>(buffers[i].buffer) != prev.buffers[i]) {
            return NbBuffers;
        }
    }
    if (vi.n != prev.vi.n) {
        return NbVertex;
    }
    s64 q = 0;
    bool q_set = false;
    for (u32 k = 0; k < vi.n; ++k) {
        if (vi.buf[k] != prev.vi.buf[k] || vi.stride[k] != prev.vi.stride[k]) {
            return NbVertex;
        }
        if (vi.buf[k] == 0 || vi.off[k] == prev.vi.off[k]) {
            continue;
        }
        const s64 d = s64(vi.off[k]) - s64(prev.vi.off[k]);
        if (vi.stride[k] == 0 || d % s64(vi.stride[k]) != 0) {
            return NbVertex;
        }
        const s64 qk = d / s64(vi.stride[k]);
        if (q_set && qk != q) {
            return NbVertex;
        }
        q = qk;
        q_set = true;
    }
    if (vi.ib_buf != prev.vi.ib_buf || vi.ib_type != prev.vi.ib_type ||
        (vi.ib_type != 0 && (s64(vi.ib_off) - s64(prev.vi.ib_off)) % s64(vi.ib_type) != 0)) {
        return NbIndex;
    }
    if (push.xoffset != prev.vp[0] || push.yoffset != prev.vp[1] || push.xscale != prev.vp[2] ||
        push.yscale != prev.vp[3]) {
        return NbViewport;
    }
    return NbCount;
}

// Census v2 (colour-pass study 2026-10-09): every condition evaluated independently (a bit
// mask, not the first failure), then joinability per cumulative relaxation level:
//   L0 strict (= v1), L1 vertex / index OFFSETS free (in-shader fetch, per-draw offsets in the
//   record, same VkBuffers), L2 + vertex / index VkBuffers free (per-draw addresses / vertex
//   pulling), L3 + every guest buffer VkBuffer free, L4 + images free (bindless), L5 + viewport
//   push data free. Groups: untextured (no image bindings: shadow / depth passes) vs textured.
enum NativeBit : u32 {
    NbbBetween = 1u << 0,
    NbbPipeline = 1u << 1,
    NbbRendering = 1u << 2,
    NbbDyn = 1u << 3,
    NbbBarrier = 1u << 4,
    NbbOcclusion = 1u << 5,
    NbbImages = 1u << 6,
    NbbBufCount = 1u << 7,
    NbbBufVk = 1u << 8,
    NbbVtxLayout = 1u << 9, // stream count / stride differs
    NbbVtxBuf = 1u << 10,   // some vertex stream in another VkBuffer
    NbbVtxOff = 1u << 11,   // same VkBuffers, offsets not one common stride multiple
    NbbIdxBuf = 1u << 12,   // index VkBuffer or type differs
    NbbIdxOff = 1u << 13,   // index offset not a multiple of the index size
    NbbViewport = 1u << 14,
    NbbBits = 15
};
constexpr std::array<const char*, NbbBits> kNativeBitNames{
    "between", "pipeline", "rendering", "dyn",    "barrier", "occlusion", "images", "bufcount",
    "bufvk",   "vlayout",  "vbuf",      "voff",   "ibuf",    "ioff",      "viewport"};
constexpr u32 kNbLevels = 6;
constexpr std::array<u32, kNbLevels> kNbLevelFree{
    0u,
    NbbVtxOff | NbbIdxOff,
    NbbVtxOff | NbbIdxOff | NbbVtxBuf | NbbIdxBuf,
    NbbVtxOff | NbbIdxOff | NbbVtxBuf | NbbIdxBuf | NbbBufVk,
    NbbVtxOff | NbbIdxOff | NbbVtxBuf | NbbIdxBuf | NbbBufVk | NbbImages,
    NbbVtxOff | NbbIdxOff | NbbVtxBuf | NbbIdxBuf | NbbBufVk | NbbImages | NbbViewport};
struct NativeCensus2Level {
    u64 joins{};
    u64 runs{};
    u64 cur_run{};
    u64 run_hist[6]{};
    std::array<u64, NbbBits> first{}; // first remaining failing bit, per break
};
struct NativeCensus2Group {
    u64 draws{};
    std::array<u64, NbbBits> any{}; // draws with this bit set (independent)
    std::array<NativeCensus2Level, kNbLevels> lv{};
    u64 pipe_vs_same{};  // pipeline break with the same VS
    u64 pipe_abab{};     // pipeline break back to the pipeline two draws ago
    std::array<u64, 4> vk_per_draw{}; // draws whose vertex streams use 0 / 1 / 2 / 3+ VkBuffers
    u64 valigned{};      // every stream offset and stride 4-byte aligned
    u64 hint{};          // same pipeline + render state + no dyn change + nothing between as the
                         // previous draw (what a front-side "next draw may join" hint can see)
    u64 hint_join{};     // ... and the draw joins at L1
};
thread_local std::array<NativeCensus2Group, 2> t_nc2;
thread_local const void* t_nc2_prev2_pipeline = nullptr;
thread_local u64 t_nc2_prev_vs = 0;
thread_local u32 t_nc2_prev_group = 0;
thread_local u64 t_nc2_draws = 0;

u32 NativeCensusBits(const NativeCensusPrev& prev, const GraphicsPipeline* pipeline,
                     const RenderState& state, u64 entry_seq, bool dyn_change, bool barriers,
                     std::span<const vk::DescriptorImageInfo> images,
                     std::span<const vk::DescriptorBufferInfo> buffers,
                     const Shader::PushData& push, bool occlusion, u32 group) {
    const auto& vi = VideoCore::t_native_vi;
    u32 m = 0;
    if (entry_seq != prev.end_seq) {
        m |= NbbBetween;
    }
    if (prev.pipeline != pipeline) {
        m |= NbbPipeline;
    }
    if (!(prev.state == state)) {
        m |= NbbRendering; // the group change is added by the caller (v2 groups)
    }
    if (dyn_change) {
        m |= NbbDyn;
    }
    if (barriers) {
        m |= NbbBarrier;
    }
    if (occlusion) {
        m |= NbbOcclusion;
    }
    if (images.size() != prev.images.size() ||
        !std::equal(images.begin(), images.end(), prev.images.begin())) {
        m |= NbbImages;
    }
    if (buffers.size() != prev.buffers.size()) {
        m |= NbbBufCount;
    } else {
        for (size_t i = 0; i < buffers.size(); ++i) {
            if (static_cast<VkBuffer>(buffers[i].buffer) != prev.buffers[i]) {
                m |= NbbBufVk;
                break;
            }
        }
    }
    if (vi.n != prev.vi.n) {
        m |= NbbVtxLayout;
    } else {
        s64 q = 0;
        bool q_set = false;
        for (u32 k = 0; k < vi.n; ++k) {
            if (vi.stride[k] != prev.vi.stride[k]) {
                m |= NbbVtxLayout;
                continue;
            }
            if (vi.buf[k] != prev.vi.buf[k]) {
                m |= NbbVtxBuf;
                continue;
            }
            if (vi.buf[k] == 0 || vi.off[k] == prev.vi.off[k]) {
                continue;
            }
            const s64 d = s64(vi.off[k]) - s64(prev.vi.off[k]);
            if (vi.stride[k] == 0 || d % s64(vi.stride[k]) != 0) {
                m |= NbbVtxOff;
                continue;
            }
            const s64 qk = d / s64(vi.stride[k]);
            if (q_set && qk != q) {
                m |= NbbVtxOff;
            }
            q = qk;
            q_set = true;
        }
    }
    if (vi.ib_buf != prev.vi.ib_buf || vi.ib_type != prev.vi.ib_type) {
        m |= NbbIdxBuf;
    } else if (vi.ib_type != 0 && (s64(vi.ib_off) - s64(prev.vi.ib_off)) % s64(vi.ib_type) != 0) {
        m |= NbbIdxOff;
    }
    if (push.xoffset != prev.vp[0] || push.yoffset != prev.vp[1] || push.xscale != prev.vp[2] ||
        push.yscale != prev.vp[3]) {
        m |= NbbViewport;
    }
    return m;
}

void NativeCensus2CloseRuns() {
    for (auto& g : t_nc2) {
        for (auto& l : g.lv) {
            if (l.cur_run == 0) {
                continue;
            }
            ++l.runs;
            const u64 r = l.cur_run;
            l.run_hist[r == 1 ? 0 : r < 4 ? 1 : r < 8 ? 2 : r < 16 ? 3 : r < 64 ? 4 : 5]++;
            l.cur_run = 0;
        }
    }
}

void NativeCensus2Note(const NativeCensusPrev& prev, const GraphicsPipeline* pipeline,
                       const RenderState& state, u64 entry_seq, bool dyn_change, bool barriers,
                       std::span<const vk::DescriptorImageInfo> images,
                       std::span<const vk::DescriptorBufferInfo> buffers,
                       const Shader::PushData& push, bool occlusion) {
    const u32 group = images.empty() ? 0 : 1;
    u64 vs = 0;
    for (const auto* stage : pipeline->GetStages()) {
        if (stage && stage->l_stage == Shader::LogicalStage::Vertex) {
            vs = stage->pgm_hash;
        }
    }
    auto& g = t_nc2[group];
    ++g.draws;
    {
        const auto& vi = VideoCore::t_native_vi;
        std::array<u64, 16> seen{};
        u32 nvk = 0;
        bool aligned = true;
        for (u32 k = 0; k < vi.n; ++k) {
            if (vi.buf[k] == 0) {
                continue;
            }
            aligned = aligned && (vi.off[k] & 3) == 0 && (vi.stride[k] & 3) == 0;
            if (std::find(seen.begin(), seen.begin() + nvk, vi.buf[k]) == seen.begin() + nvk) {
                seen[nvk++] = vi.buf[k];
            }
        }
        ++g.vk_per_draw[std::min<u32>(nvk, 3)];
        g.valigned += aligned;
    }
    if (prev.pipeline != nullptr) {
        u32 m = NativeCensusBits(prev, pipeline, state, entry_seq, dyn_change, barriers, images,
                                 buffers, push, occlusion, group);
        if (t_nc2_prev_group != group) {
            m |= NbbRendering;
        }
        for (u32 b = 0; b < NbbBits; ++b) {
            g.any[b] += (m >> b) & 1u;
        }
        if ((m & (NbbPipeline | NbbRendering | NbbDyn | NbbBetween)) == 0) {
            ++g.hint;
            g.hint_join += (m & ~kNbLevelFree[1]) == 0;
        }
        if (m & NbbPipeline) {
            g.pipe_vs_same += vs == t_nc2_prev_vs;
            g.pipe_abab += pipeline == t_nc2_prev2_pipeline;
        }
        for (u32 li = 0; li < kNbLevels; ++li) {
            auto& l = g.lv[li];
            const u32 rest = m & ~kNbLevelFree[li];
            if (rest == 0) {
                ++l.joins;
                ++l.cur_run;
                continue;
            }
            // A break at this level closes the open run of this level in both groups.
            for (auto& gg : t_nc2) {
                auto& ll = gg.lv[li];
                if (ll.cur_run != 0) {
                    ++ll.runs;
                    const u64 r = ll.cur_run;
                    ll.run_hist[r == 1 ? 0 : r < 4 ? 1 : r < 8 ? 2 : r < 16 ? 3 : r < 64 ? 4 : 5]++;
                    ll.cur_run = 0;
                }
            }
            ++l.first[std::countr_zero(rest)];
            l.cur_run = 1;
        }
    }
    t_nc2_prev2_pipeline = prev.pipeline;
    t_nc2_prev_vs = vs;
    t_nc2_prev_group = group;
    if ((++t_nc2_draws & 0xFFFFF) != 0) {
        return;
    }
    for (u32 gi = 0; gi < 2; ++gi) {
        auto& gg = t_nc2[gi];
        const double n = double(std::max<u64>(gg.draws, 1));
        std::string any;
        for (u32 b = 0; b < NbbBits; ++b) {
            if (gg.any[b]) {
                any += fmt::format(" {}={:.1f}%", kNativeBitNames[b], 100.0 * double(gg.any[b]) / n);
            }
        }
        LOG_INFO(Render_Vulkan,
                 "NativeCensus2: {} draws={} conditions failing (independent):{} | pipeline "
                 "breaks with same VS={:.1f}% back-to-2-ago={:.1f}%",
                 gi == 0 ? "untextured" : "textured", gg.draws, any,
                 100.0 * double(gg.pipe_vs_same) / double(std::max<u64>(gg.any[1], 1)),
                 100.0 * double(gg.pipe_abab) / double(std::max<u64>(gg.any[1], 1)));
        LOG_INFO(Render_Vulkan,
                 "NativeCensus2: {} vertex VkBuffers per draw [0,1,2,3+]={:.1f}%,{:.1f}%,{:.1f}%,"
                 "{:.1f}% aligned={:.1f}% | front hint (same pipeline/state/dyn, nothing between) "
                 "{:.1f}% of draws, of which L1-joinable {:.1f}%",
                 gi == 0 ? "untextured" : "textured", 100.0 * double(gg.vk_per_draw[0]) / n,
                 100.0 * double(gg.vk_per_draw[1]) / n, 100.0 * double(gg.vk_per_draw[2]) / n,
                 100.0 * double(gg.vk_per_draw[3]) / n, 100.0 * double(gg.valigned) / n,
                 100.0 * double(gg.hint) / n,
                 100.0 * double(gg.hint_join) / double(std::max<u64>(gg.hint, 1)));
        for (u32 li = 0; li < kNbLevels; ++li) {
            auto& l = gg.lv[li];
            std::string first;
            for (u32 b = 0; b < NbbBits; ++b) {
                if (l.first[b]) {
                    first += fmt::format(" {}={:.1f}%", kNativeBitNames[b],
                                         100.0 * double(l.first[b]) / n);
                }
            }
            LOG_INFO(Render_Vulkan,
                     "NativeCensus2: {} L{} joinable={:.1f}% mean_run={:.2f} "
                     "runs[1,2-3,4-7,8-15,16-63,64+]={},{},{},{},{},{} first-break:{}",
                     gi == 0 ? "untextured" : "textured", li, 100.0 * double(l.joins) / n,
                     n / double(std::max<u64>(l.runs, 1)), l.run_hist[0], l.run_hist[1],
                     l.run_hist[2], l.run_hist[3], l.run_hist[4], l.run_hist[5], first);
        }
    }
    NativeCensus2CloseRuns();
    for (auto& gg : t_nc2) {
        gg = NativeCensus2Group{};
    }
}

void NativeCensusNote(const GraphicsPipeline* pipeline, const RenderState& state, u64 entry_seq,
                      u64 end_seq, bool dyn_change, bool barriers,
                      std::span<const vk::DescriptorImageInfo> images,
                      std::span<const vk::DescriptorBufferInfo> buffers,
                      const Shader::PushData& push, bool occlusion) {
    auto& prev = t_nc_prev;
    NativeCensus2Note(prev, pipeline, state, entry_seq, dyn_change, barriers, images, buffers,
                      push, occlusion);
    const u32 group = state.num_color_attachments == 0 ? 0 : 1;
    const u32 why = NativeCensusWhy(prev, pipeline, state, entry_seq, dyn_change, barriers, images,
                                    buffers, push, occlusion, group);
    u64 fs = 0;
    for (const auto* stage : pipeline->GetStages()) {
        if (stage && stage->l_stage == Shader::LogicalStage::Fragment) {
            fs = stage->pgm_hash;
        }
    }
    auto& pp = t_nc_pipes[fs];
    ++pp[0];
    auto& g = t_nc_groups[group];
    ++g.draws;
    if (why == NbCount) {
        ++g.joins;
        ++pp[1];
        ++g.cur_run;
    } else {
        NativeCensusCloseRun(t_nc_groups[0]);
        NativeCensusCloseRun(t_nc_groups[1]);
        ++g.breaks[why];
        g.cur_run = 1;
        ++pp[2];
    }
    prev.pipeline = pipeline;
    prev.state = state;
    prev.end_seq = end_seq;
    prev.images.assign(images.begin(), images.end());
    prev.buffers.clear();
    for (const auto& b : buffers) {
        prev.buffers.push_back(static_cast<VkBuffer>(b.buffer));
    }
    prev.vi = VideoCore::t_native_vi;
    prev.vp = {push.xoffset, push.yoffset, push.xscale, push.yscale};
    prev.group = group;
    if ((++t_nc_draws & 0xFFFFF) != 0) {
        return;
    }
    for (u32 gi = 0; gi < 2; ++gi) {
        auto& gg = t_nc_groups[gi];
        std::string br;
        for (u32 b = 0; b < NbCount; ++b) {
            if (gg.breaks[b]) {
                br += fmt::format(" {}={:.1f}%", kNativeBreakNames[b],
                                  100.0 * double(gg.breaks[b]) / double(std::max<u64>(gg.draws, 1)));
            }
        }
        LOG_INFO(Render_Vulkan,
                 "NativeCensus: {} draws={} joinable={:.1f}% runs={} mean_run={:.2f} "
                 "runs[1,2-3,4-7,8-15,16-63,64+]={},{},{},{},{},{} breaks:{}",
                 gi == 0 ? "depth-only" : "colour", gg.draws,
                 100.0 * double(gg.joins) / double(std::max<u64>(gg.draws, 1)), gg.runs,
                 double(gg.draws) / double(std::max<u64>(gg.runs, 1)), gg.run_hist[0],
                 gg.run_hist[1], gg.run_hist[2], gg.run_hist[3], gg.run_hist[4], gg.run_hist[5],
                 br);
        const u64 carry = gg.cur_run;
        gg = NativeCensusGroup{};
        gg.cur_run = carry;
    }
    std::vector<std::pair<u64, std::array<u64, 3>>> order(t_nc_pipes.begin(), t_nc_pipes.end());
    std::sort(order.begin(), order.end(),
              [](const auto& a, const auto& b) { return a.second[0] > b.second[0]; });
    std::string top;
    for (size_t k = 0; k < std::min<size_t>(order.size(), 12); ++k) {
        top += fmt::format(" fs_{:08x}:{}/{:.0f}%/run{:.1f}", order[k].first, order[k].second[0],
                           100.0 * double(order[k].second[1]) / double(order[k].second[0]),
                           double(order[k].second[0]) /
                               double(std::max<u64>(order[k].second[2], 1)));
    }
    LOG_INFO(Render_Vulkan, "NativeCensus: top fs (draws/joinable/mean run):{}", top);
    t_nc_pipes.clear();
}

} // namespace

// SHADPS4_FLUSH_EVERY_OPS=N: submit once N draws + dispatches have been recorded since the last
// submission of any kind (a fault flush resets the count too). A readback copy runs after
// everything submitted before it; measured 2026-09-27 in the fire scene's slow mode, readbacks
// queued behind ~5 ms batches. SHADPS4_FLUSH_EVERY_DRAWS counts draws only and is not reset by
// other flushes. 0 (default) = off. Runs on the command processor thread that records.
static void MaybeFlushEveryOps(Scheduler& scheduler) {
    static const u32 flush_every_ops = [] {
        const char* value = std::getenv("SHADPS4_FLUSH_EVERY_OPS");
        return value ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 0u;
    }();
    if (flush_every_ops != 0 && scheduler.BatchOps() >= flush_every_ops) {
        scheduler.NoteCadenceFlush();
        scheduler.Flush();
    }
}

// Shape-only diagnostic; enable for one run, then compile it out for FPS comparisons.
#ifndef SHADPS4_BINDING_PROBE
#define SHADPS4_BINDING_PROBE 0
#endif
#if SHADPS4_BINDING_PROBE
static void ProbeBindingChanges(const Pipeline* pipeline,
                                std::span<const Shader::BoundStage> bound_stages) {
    if (pipeline->IsCompute()) {
        return;
    }
    struct PreviousStage {
        const Shader::Info* info{};
        size_t num_buffers{}, num_images{};
        std::array<AmdGpu::Buffer, Shader::NUM_BUFFERS> buffers{};
        std::array<AmdGpu::Image, Shader::NUM_IMAGES> images{};
    };
    static std::array<PreviousStage, Shader::MaxStageTypes> previous{};
    static u64 draws{}, stages{}, same_program{}, buffers{}, same_buffers{}, special{}, images{},
        same_images{}, written{};
    ++draws;
    for (const auto& bound : bound_stages) {
        const auto* stage = bound.info;
        if (!stage) {
            continue;
        }
        auto& prev = previous[static_cast<u32>(stage->l_stage)];
        ++stages;
        same_program += prev.info == stage;
        size_t i = 0;
        for (const auto& desc : stage->buffers) {
            const auto sharp = desc.GetSharp(bound);
            if (desc.IsSpecial()) {
                ++special;
            } else {
                ++buffers;
                written += desc.is_written;
                same_buffers += prev.info && i < prev.num_buffers &&
                                std::memcmp(&sharp, &prev.buffers[i], sizeof(sharp)) == 0;
            }
            prev.buffers[i++] = sharp;
        }
        prev.num_buffers = i;
        i = 0;
        for (const auto& desc : stage->images) {
            const auto sharp = desc.GetSharp(bound);
            ++images;
            same_images += prev.info && i < prev.num_images &&
                           std::memcmp(&sharp, &prev.images[i], sizeof(sharp)) == 0;
            prev.images[i++] = sharp;
        }
        prev.num_images = i;
        prev.info = stage;
    }
    if (draws >= 200000) {
        LOG_INFO(Render_Vulkan,
                 "BindingChanges: draws={} buffers/draw={:.2f} raw_same_buffers={:.1f}% "
                 "special/draw={:.2f} written_buffers={:.1f}% images/draw={:.2f} "
                 "raw_same_images={:.1f}% same_stage_program={:.1f}%",
                 draws, double(buffers) / draws, buffers ? 100.0 * same_buffers / buffers : 0.0,
                 double(special) / draws, buffers ? 100.0 * written / buffers : 0.0,
                 double(images) / draws, images ? 100.0 * same_images / images : 0.0,
                 stages ? 100.0 * same_program / stages : 0.0);
        draws = stages = same_program = buffers = same_buffers = special = images = same_images =
            written = 0;
    }
    // Equal descriptors do not prove equal contents or cache state. This probe
    // never skips synchronization, resource lookup, barriers, or descriptor writes.
}
#endif

static Shader::PushData MakeUserData(const AmdGpu::Regs& regs) {
    // TODO(roamic): Add support for multiple viewports and geometry shaders when ViewportIndex
    // is encountered and implemented in the recompiler.
    Shader::PushData push_data{};
    push_data.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push_data.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push_data.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push_data.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
    return push_data;
}

Rasterizer::Shared::Shared(Rasterizer* owner, const Instance& instance, Scheduler& scheduler,
                           AmdGpu::Liverpool* liverpool)
    : native_bindings{NativeBindings::Create(instance, scheduler)}, page_manager{owner},
      buffer_cache{instance, scheduler, liverpool, texture_cache, page_manager},
      texture_cache{instance, scheduler, liverpool, buffer_cache, page_manager},
      pipeline_cache{instance, scheduler, liverpool} {}

Rasterizer::Rasterizer(Rasterizer& primary, WorkerTag)
    : instance{primary.instance}, scheduler{primary.scheduler}, shared{primary.shared},
      native_bindings{shared.native_bindings}, page_manager{shared.page_manager},
      buffer_cache{shared.buffer_cache}, texture_cache{shared.texture_cache},
      liverpool{primary.liverpool}, memory{primary.memory}, mapped_ranges{shared.mapped_ranges},
      mapped_ranges_mutex{shared.mapped_ranges_mutex}, pipeline_cache{shared.pipeline_cache},
      occlusion{shared.occlusion}, fault_process_pending{shared.fault_process_pending} {
    // The settings the primary's constructor derived; everything else starts empty.
    build_set_writes = primary.build_set_writes;
    bind_memo_stats = primary.bind_memo_stats;
    tex_bind_memo = primary.tex_bind_memo;
    native_pt_on = primary.native_pt_on;
    native_pt_extra = primary.native_pt_extra;
    native_seg_on = primary.native_seg_on;
}

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_,
                       AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_},
      owned_shared{std::make_unique<Shared>(this, instance_, scheduler_, liverpool_)},
      shared{*owned_shared}, native_bindings{shared.native_bindings},
      page_manager{shared.page_manager}, buffer_cache{shared.buffer_cache},
      texture_cache{shared.texture_cache}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, mapped_ranges{shared.mapped_ranges},
      mapped_ranges_mutex{shared.mapped_ranges_mutex}, pipeline_cache{shared.pipeline_cache},
      occlusion{shared.occlusion}, fault_process_pending{shared.fault_process_pending} {
    if (AmdGpu::NativeRec::g_gpu_mode == 1) {
        NativeGpuRingInit(this); // SHADPS4_NATIVE_GPU_DIRECT=1: the job threads' ring hook
    }
    native_pt_on = Shader::g_native_pt != 0 &&
                   (AmdGpu::NativeRec::g_gpu_mode == 1 || AmdGpu::NativeRec::g_gpu_mode == 3);
    native_pt_extra = native_pt_on && Shader::g_native_pt == 2 ? 1u : 0u; // the compare SSBO
    if (Shader::g_native_pt != 0) {
        LOG_INFO(Render_Vulkan, "NativePt: mode {}{}", Shader::g_native_pt,
                 native_pt_on ? "" : " (needs SHADPS4_NATIVE_GPU_DIRECT=1; off)");
    }
    native_seg_on = Shader::g_native_seg != 0 && native_pt_on;
    if (Shader::g_native_seg != 0) {
        LOG_INFO(Render_Vulkan, "NativeSeg: mode {}{}{}", Shader::g_native_seg,
                 native_seg_on ? "" : " (needs SHADPS4_NATIVE_GPU_DIRECT=1 and SHADPS4_NATIVE_PT; off)",
                 Shader::g_native_seg == 2 && !Shader::g_native_seg_compare
                     ? " (the compare needs SHADPS4_NATIVE_PT=2; running as 1)"
                     : "");
    }
    if (!EmulatorSettings.IsNullGPU()) {
        liverpool->BindRasterizer(this);
    }
    memory->SetRasterizer(this);
    if (OcclusionQueries::Requested() && !EmulatorSettings.IsNullGPU()) {
        if (instance.IsHostQueryResetSupported()) {
            occlusion = std::make_unique<OcclusionQueries>(instance, scheduler);
        } else {
            LOG_WARNING(Render_Vulkan, "OcclusionQ: off, hostQueryReset is not supported");
        }
    }
    build_set_writes = !native_bindings || native_bindings->IsVerifying();
    const char* memo_stats = std::getenv("SHADPS4_BIND_MEMO_STATS");
    bind_memo_stats = memo_stats && memo_stats[0] == '1';
    if (bind_memo_stats) {
        LOG_INFO(Render_Vulkan, "BindMemoStats: enabled (counters only, behaviour unchanged)");
    }
    if (const char* value = std::getenv("SHADPS4_TEX_BIND_MEMO");
        value && (value[0] == '1' || value[0] == '2')) {
        if (bind_memo_stats) {
            LOG_WARNING(Render_Vulkan, "TexBindMemo: off, SHADPS4_BIND_MEMO_STATS takes precedence");
        } else if (native_bindings) {
            LOG_WARNING(Render_Vulkan, "TexBindMemo: off, not supported with native bindings");
        } else {
            tex_bind_memo = static_cast<u8>(value[0] - '0');
            LOG_INFO(Render_Vulkan, "TexBindMemo: enabled, mode={}",
                     tex_bind_memo == 2 ? "verify (full path + compare)" : "memo");
        }
    }
    if (g_mtb_p1) {
        // P1d (notes/parrec-build-2026-10-08.md sec. 14a): the Pro-only stencil scratch and Neo
        // ID derive keep per-instance state that both instances' draws would need.
        const auto on = [](const char* name) {
            const char* value = std::getenv(name);
            return value && value[0] == '1';
        };
        if (on("SHADPS4_STENCIL_STORAGE") || on("SHADPS4_NEO_IDBUF_DERIVE")) {
            LOG_WARNING(Render_Vulkan, "MtbP1: no worker rasterizer (Pro stencil storage / Neo "
                                       "ID derive are per instance)");
        } else if (native_bindings) {
            LOG_WARNING(Render_Vulkan, "MtbP1: no worker rasterizer with native bindings (ring "
                                       "space per segment is P3 work)");
        } else {
            mtb_worker = std::make_unique<Rasterizer>(*this, WorkerTag{});
            LOG_INFO(Render_Vulkan, "MtbP1: worker rasterizer instance created");
        }
    }
}

Rasterizer::~Rasterizer() {
    scheduler.DrainRecording();
}

void Rasterizer::CpSync() {
    scheduler.EndRendering();
    auto command_access = scheduler.AcquireCommandBuffer();
    auto cmdbuf = command_access.Handle();

    const vk::MemoryBarrier ib_barrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead,
    };
    cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                           vk::PipelineStageFlagBits::eDrawIndirect,
                           vk::DependencyFlagBits::eByRegion, ib_barrier, {}, {});
}

void Rasterizer::WaitBarrier() {
    scheduler.EndRendering();
    auto command_access = scheduler.AcquireCommandBuffer();
    auto cmdbuf = command_access.Handle();
    const vk::MemoryBarrier barrier{
        .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
        .dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
    };
    cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                           vk::PipelineStageFlagBits::eAllCommands, {}, barrier, {}, {});
}

Rasterizer::DrawClass Rasterizer::ClassifyDraw(const AmdGpu::Regs& regs) {
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::EliminateFastClear) {
        return DrawClass::EliminateFastClear;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::FmaskDecompress) {
        return DrawClass::FmaskDecompress;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Resolve) {
        return DrawClass::Resolve;
    }
    if (regs.primitive_type == AmdGpu::PrimitiveType::None) {
        return DrawClass::PrimitiveNone;
    }

    const bool cb_disabled =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    const auto depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    const auto stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    if (cb_disabled && (depth_copy || stencil_copy)) {
        return depth_copy && stencil_copy ? DrawClass::DepthStencilCopy
               : depth_copy               ? DrawClass::DepthCopy
                                          : DrawClass::StencilCopy;
    }

    // SHADPS4_DLSS (vk_dlss.h): the T2x resolve; one relaxed load + compare (0 while off).
    if (Dlss::IsResolve(regs.ps_program.Address<VAddr>())) [[unlikely]] {
        return DrawClass::DlssResolve;
    }
    // SHADPS4_SKIP_PS (test): that pixel shader's draws are dropped (as primitive type None).
    if (Dlss::IsSkippedPs(regs.ps_program.Address<VAddr>())) [[unlikely]] {
        return DrawClass::PrimitiveNone;
    }

    return DrawClass::Draw;
}

void Rasterizer::ExecuteFilteredDraw(DrawClass draw_class) {
    if (native_batch.open) [[unlikely]] {
        NativeClose(12);
    }
    if (draw_class != DrawClass::Draw) {
        // RENDER_STATE_MEMO: these touch render-target metadata outside DrawTail.
        rs_memo.valid = false;
    }
    switch (draw_class) {
    case DrawClass::Draw:
        break;
    case DrawClass::EliminateFastClear:
        // Clears the render target if FCE is launched before any draws
        EliminateFastClear();
        break;
    case DrawClass::FmaskDecompress:
        // TODO: check for a valid MRT1 to promote the draw to the resolve pass.
        LOG_TRACE(Render_Vulkan, "FMask decompression pass skipped");
        ScopedMarkerInsert("FmaskDecompress");
        break;
    case DrawClass::Resolve:
        LOG_TRACE(Render_Vulkan, "Resolve pass");
        Resolve();
        break;
    case DrawClass::PrimitiveNone:
        LOG_TRACE(Render_Vulkan, "Primitive type 'None' skipped");
        ScopedMarkerInsert("PrimitiveTypeNone");
        break;
    case DrawClass::DepthCopy:
    case DrawClass::StencilCopy:
    case DrawClass::DepthStencilCopy:
        // Games may disable color buffer and enable force depth/stencil dirty and valid to
        // do a copy from one depth-stencil surface to another, without a pixel shader.
        // We need to detect this case and perform the copy, otherwise it will have no effect.
        LOG_TRACE(Render_Vulkan, "Performing depth-stencil override copy");
        DepthStencilCopy(draw_class != DrawClass::StencilCopy,
                         draw_class != DrawClass::DepthCopy);
        break;
    case DrawClass::DlssResolve:
        DlssResolve();
        break;
    }
}

void Rasterizer::DlssResolve() {
    // SHADPS4_DLSS (vk_dlss.h): the game's T2x resolve replaced. =1: DLAA from the tonemap output
    // 0x2f9e10000 + motion vectors + depth into the resolve's target; =2 or any failure this frame:
    // a plain copy of the tonemap output (the game without its temporal AA for that frame).
    ++dlss_calls;
    ++dlss_phases[dlss_phase_in & 3];
    Dlss5::g_resolves.fetch_add(1, std::memory_order_release); // a game-rendered frame
    const auto& regs = liverpool->BackRegs();
    VideoCore::TextureCache::ImageDesc out_desc{regs.color_buffers[0], liverpool->BackCbExtent(0)};
    auto& out = texture_cache.GetImage(texture_cache.FindImage(out_desc, true));
    // The inputs come from this frame's SMAA edges draw and the resolve itself (vk_dlss.h "Any
    // render size"; the 1080p addresses until those ran). Found by address and size: 0x26b200000
    // also holds the minimap's 768x768 depth after the AA (play log 2026-10-17: depth "missing"
    // on 109,135 of 109,569 resolves).
    const VAddr color_addr = Dlss::g_color_addr.load(std::memory_order_relaxed);
    const VAddr motion_addr = Dlss::g_motion_addr.load(std::memory_order_relaxed);
    const VAddr depth_addr = Dlss::g_depth_addr.load(std::memory_order_relaxed);
    const auto find = [&](VAddr address) -> VideoCore::Image* {
        const VideoCore::ImageId id = texture_cache.FindImageAtSize(
            address, out.info.size.width, out.info.size.height);
        return id ? &texture_cache.GetImage(id) : nullptr;
    };
    VideoCore::Image* in = find(color_addr);
    if (!in || in->info.pixel_format != out.info.pixel_format) [[unlikely]] {
        // A target that keeps stale memory would show (a 1440p patch before 1.1: a yellow
        // picture, RTX 5070 Ti report 10-20). Off the 1080p frame DLSS then steps aside for the
        // session: g_ready off stops the classification (the game's own T2x resolve runs from the
        // next frame on, on every path), the upscaler and DLSS 5.
        // A live screen percentage change: the 1-2 frames around it mix the sizes (the learned
        // addresses are last frame's) - skip them, DLSS stays on.
        if (MemoryPatcher::MsSinceLiveResize() < 1000) {
            return;
        }
        if (out.info.size.width != 1920 || out.info.size.height != 1080) {
            // Resolves already classified before the switch land here too: one line is enough.
            if (Dlss::g_ready.exchange(false, std::memory_order_relaxed)) {
                LOG_WARNING(Render_Vulkan,
                            "DLSS steps aside: no colour input at {:#x} for the game's {}x{} "
                            "frame; the game's own anti-aliasing runs",
                            color_addr, out.info.size.width, out.info.size.height);
            }
            return;
        }
        if (++dlss_missing <= 8) {
            LOG_WARNING(Render_Vulkan, "DLSS: no matching input at {:#x} for output {:#x} ({}x{}); "
                                       "the resolve's target keeps last frame's picture",
                        color_addr, regs.color_buffers[0].Address(), out.info.size.width,
                        out.info.size.height);
        }
        return;
    }
    if (dlss_calls == 1 || (dlss_calls & 1023) == 1) {
        LOG_INFO(Render_Vulkan, "DLSS: inputs at {}x{}: colour {:#x}, motion {:#x}, depth {:#x}, "
                                "HUD layer {:#x}",
                 out.info.size.width, out.info.size.height, color_addr, motion_addr, depth_addr,
                 regs.color_buffers[0].Address());
    }
    scheduler.EndRendering();
    bool evaluated = false;
    bool hud_layer = false;
    if (Dlss::Mode() == 1) {
        VideoCore::Image* motion = find(motion_addr);
        VideoCore::Image* depth = find(depth_addr);
        if (motion && depth) [[likely]] {
            in->Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead, {});
            motion->Transit(vk::ImageLayout::eShaderReadOnlyOptimal,
                            vk::AccessFlagBits2::eShaderRead, {});
            depth->Transit(vk::ImageLayout::eDepthStencilReadOnlyOptimal,
                           vk::AccessFlagBits2::eShaderRead, {});
            out.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite, {});
            const auto eval_image = [](VideoCore::Image& image, vk::ImageLayout layout,
                                       bool storage, bool is_depth) {
                VideoCore::ImageViewInfo view_info{};
                view_info.type = AmdGpu::ImageType::Color2D;
                view_info.format = image.info.pixel_format;
                view_info.is_storage = storage;
                const auto& view = image.FindView(view_info);
                return Dlss::EvalImage{
                    .image = image.GetImage(),
                    .view = *view.image_view,
                    .format = image.info.pixel_format,
                    .layout = Vulkan::GuestLayout(layout),
                    .usage = image.usage_flags,
                    .width = image.info.size.width,
                    .height = image.info.size.height,
                    .depth = is_depth,
                };
            };
            // The game's two jitter phases are +-0.25 px diagonally (study sec. 1.4); which sign
            // goes with which phase is calibrated live (Page Up cycles the four combinations).
            // +-0.25 render px at every render size: the game divides its +-0.5 by the slot-2
            // render target's size (FUN_800472810, notes/study-jitter-2026-10-20.md); the 1/1920
            // constants have no reader. SHADPS4_DLSS_JITTER_SCALE=1 (test) scales by size / 1080p.
            float jx = 0.0f, jy = 0.0f;
            // Phase 4: the emulator's Halton offset, exact for this frame (memory_patcher.cpp
            // JitterFrame). A switch between it and the game's 2 phases starts a new history.
            const bool explicit_jitter = dlss_phase_in == 4;
            dlss_reset_pending |= explicit_jitter != dlss_last_explicit;
            dlss_last_explicit = explicit_jitter;
            if (explicit_jitter) {
                jx = dlss_jitter_in[0];
                jy = dlss_jitter_in[1];
            } else if (dlss_phase_in == 1 || dlss_phase_in == 2) {
                const float j = dlss_phase_in == 1 ? 0.25f : -0.25f;
                const u32 combo = Dlss::g_jitter_combo.load(std::memory_order_relaxed);
                static const bool scale_jitter = [] {
                    const char* v = std::getenv("SHADPS4_DLSS_JITTER_SCALE");
                    return v && v[0] == '1';
                }();
                const float sx =
                    scale_jitter ? static_cast<float>(out.info.size.width) / 1920.0f : 1.0f;
                const float sy =
                    scale_jitter ? static_cast<float>(out.info.size.height) / 1080.0f : 1.0f;
                jx = ((combo & 1) ? -j : j) * sx;
                jy = ((combo & 2) ? -j : j) * sy;
                const bool combo_changed = combo != dlss_last_combo;
                dlss_last_combo = combo;
                dlss_reset_pending |= combo_changed;
            }
            const auto cmdbuf = scheduler.AcquireCommandBuffer().Handle();
            // SHADPS4_DLSS_UPSCALE (vk_dlss_upscale.h): DLSS writes the window size into its own
            // image; this target becomes the HUD layer below.
            Dlss::Upscaler* const upscaler =
                Dlss::HudLayerActive() ? Dlss::GetUpscaler(instance) : nullptr;
            const Dlss::EvalImage target =
                upscaler ? upscaler->SceneTarget(cmdbuf)
                         : eval_image(out, vk::ImageLayout::eGeneral, true, false);
            const Dlss::EvalImage motion_eval =
                eval_image(*motion, vk::ImageLayout::eShaderReadOnlyOptimal, false, false);
            const Dlss::EvalImage depth_eval =
                eval_image(*depth, vk::ImageLayout::eDepthStencilReadOnlyOptimal, false, true);
            Dlss::EvalImage color_eval =
                eval_image(*in, vk::ImageLayout::eShaderReadOnlyOptimal, false, false);
            // DLSS 5 (vk_dlss5.h): the model edits the 1080p frame first (no HUD in it yet) and
            // DLSS 4.5 takes its answer; off / missing / failed = the game's frame as before.
            if (const auto nr = Dlss5::Apply(instance, cmdbuf, color_eval, motion_eval,
                                             depth_eval, dlss_reset_pending)) {
                color_eval = *nr;
            }
            evaluated = Dlss::Evaluate(cmdbuf, dlss_calls, color_eval, motion_eval, depth_eval,
                                       target, jx, jy, dlss_reset_pending);
            dlss_reset_pending = false;
            if (upscaler && evaluated) {
                upscaler->SceneWritten(cmdbuf);
                hud_layer = true;
            }
            // Streamline / NGX bind their own compute state into this command buffer (checklist
            // sec. 4): nothing the next draw relies on may be assumed still bound.
            scheduler.GetDynamicState().Invalidate();
            GraphicsBindTracker::Invalidate();
            // DLSS 5's passes push descriptors, which unbind the native descriptor heaps (as the
            // scheduler's own push descriptors do): the next native draw must bind them again.
            // Without this the GPU read / wrote through stale heap descriptors - device lost
            // ~2 s into gameplay, 2026-10-18 runs 02:43 / 03:09 (fault: invalid read / write).
            scheduler.InvalidateNativeHeaps();
        } else if (++dlss_missing <= 8) {
            LOG_WARNING(Render_Vulkan, "DLSS: motion vectors {:#x} {} / depth {:#x} {} - copy this frame",
                        motion_addr, motion ? "found" : "missing", depth_addr,
                        depth ? "found" : "missing");
        }
    }
    if (hud_layer) {
        // The HUD draws blend premultiplied onto 0 and write alpha (ApplyHudLayerMask): this
        // target ends as HUD colour + coverage; the presenter lays it over the DLSS output.
        ++dlss_evaluated;
        if (Dlss::UpscaleTestBits() & 4) {
            out.CopyImage(*in);
        } else {
            out.Clear(
                vk::ClearValue{.color = vk::ClearColorValue{std::array{0.0f, 0.0f, 0.0f, 0.0f}}},
                VideoCore::SubresourceRange{});
        }
    } else if (evaluated) {
        ++dlss_evaluated;
    } else {
        out.CopyImage(*in);
    }
    out.flags |= VideoCore::ImageFlagBits::GpuModified;
    out.flags &= ~VideoCore::ImageFlagBits::Dirty;
    if ((dlss_calls & 1023) == 1) {
        LOG_INFO(Render_Vulkan,
                 "DLSS: resolve actions {} (evaluated {}, no input {}), output {:#x}; jitter phase "
                 "1 {} / 2 {} / T2x off {} / unknown {}, signs {}",
                 dlss_calls, dlss_evaluated, dlss_missing, regs.color_buffers[0].Address(),
                 dlss_phases[1], dlss_phases[2], dlss_phases[3], dlss_phases[0],
                 Dlss::g_jitter_combo.load(std::memory_order_relaxed));
    }
}

bool Rasterizer::FilterDraw() {
    // The classifier reads the front's registers (it runs where the pipeline lookup runs); the
    // actions read the back's through BackRegs().
    const DrawClass draw_class = ClassifyDraw(liverpool->regs);
    Dlss::NoteDraw(liverpool->regs);
    if (draw_class == DrawClass::Draw) {
        return true;
    }
    dlss_phase_in = Dlss::t_phase;
    dlss_jitter_in[0] = Dlss::t_jitter[0];
    dlss_jitter_in[1] = Dlss::t_jitter[1];
    ExecuteFilteredDraw(draw_class);
    return false;
}

static bool SamePreparedTarget(const VideoCore::TextureCache::ImageDesc& a,
                               const VideoCore::TextureCache::ImageDesc& b) {
    const auto& x = a.info;
    const auto& y = b.info;
    if (a.type != b.type || a.view_info != b.view_info || x.pixel_format != y.pixel_format ||
        x.type != y.type || x.resources != y.resources || x.size != y.size ||
        x.num_bits != y.num_bits || x.num_samples != y.num_samples || x.pitch != y.pitch ||
        x.tile_mode != y.tile_mode || x.array_mode != y.array_mode ||
        x.guest_address != y.guest_address || x.guest_size != y.guest_size ||
        x.bank_swizzle != y.bank_swizzle || x.alt_tile != y.alt_tile ||
        x.stencil_addr != y.stencil_addr || x.stencil_size != y.stencil_size ||
        x.meta_info.cmask_addr != y.meta_info.cmask_addr ||
        x.meta_info.fmask_addr != y.meta_info.fmask_addr ||
        x.meta_info.htile_addr != y.meta_info.htile_addr ||
        x.meta_info.htile_clear_mask != y.meta_info.htile_clear_mask ||
        x.props.is_volume != y.props.is_volume || x.props.is_tiled != y.props.is_tiled ||
        x.props.is_pow2 != y.props.is_pow2 || x.props.is_block != y.props.is_block ||
        x.props.is_depth != y.props.is_depth || x.props.has_stencil != y.props.has_stencil) {
        return false;
    }
    for (u32 mip = 0; mip < x.resources.levels; ++mip) {
        const auto& xm = x.mips_layout[mip];
        const auto& ym = y.mips_layout[mip];
        if (xm.size != ym.size || xm.pitch != ym.pitch || xm.height != ym.height ||
            xm.offset != ym.offset) {
            return false;
        }
    }
    return true;
}

void Rasterizer::PrepareRenderState(const GraphicsPipeline* pipeline) {
    const bool prepared_targets = g_prepared_targets;
    const bool verify_targets = g_verify_targets;
    const auto verify = [](bool hit, const auto& prepared, const auto& reference) {
        ASSERT(SamePreparedTarget(prepared, reference));
        static thread_local u64 calls{}, hits{};
        hits += hit;
        if (++calls % 500000 == 0) {
            LOG_INFO(Render_Vulkan, "PreparedTargetVerify: calls={} hits={}", calls, hits);
        }
    };
    // Prefetch render targets to handle overlaps with bound textures (e.g. mipgen)
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = liverpool->BackRegs();
    if (regs.color_control.degamma_enable) {
        LOG_WARNING(Render_Vulkan, "Color buffers require gamma correction");
    }

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    for (s32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (skip_cb_binding || !col_buf || !target_mask || (key.mrt_mask & (1 << cb)) == 0) {
            image_id = {};
            continue;
        }
        const auto& hint = liverpool->BackCbExtent(cb);
        if (prepared_targets) {
            auto& entry = prepared_color_targets[cb];
            const bool hit = entry.valid &&
                             std::memcmp(&entry.buffer, &col_buf, sizeof(col_buf)) == 0 &&
                             std::memcmp(&entry.extent, &hint, sizeof(hint)) == 0;
            if (!hit) {
                entry.buffer = col_buf;
                entry.extent = hint;
                std::construct_at(&entry.value, col_buf, hint);
                entry.valid = true;
            }
            if (verify_targets) {
                verify(hit, entry.value, VideoCore::TextureCache::ImageDesc{col_buf, hint});
            }
            desc = entry.value;
        } else {
            std::construct_at(&desc, col_buf, hint);
        }
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    }

    if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())) {
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& hint = liverpool->BackDbExtent();
        auto& [image_id, desc] = db_desc;
        if (prepared_targets) {
            auto& entry = prepared_depth_target;
            const bool hit =
                entry.valid && entry.htile_address == htile_address &&
                std::memcmp(&entry.buffer, &regs.depth_buffer, sizeof(entry.buffer)) == 0 &&
                std::memcmp(&entry.view, &regs.depth_view, sizeof(entry.view)) == 0 &&
                std::memcmp(&entry.control, &regs.depth_control, sizeof(entry.control)) == 0 &&
                std::memcmp(&entry.extent, &hint, sizeof(hint)) == 0;
            if (!hit) {
                entry.buffer = regs.depth_buffer;
                entry.view = regs.depth_view;
                entry.control = regs.depth_control;
                entry.extent = hint;
                entry.htile_address = htile_address;
                std::construct_at(&entry.value, regs.depth_buffer, regs.depth_view,
                                  regs.depth_control, htile_address, hint);
                entry.valid = true;
            }
            if (verify_targets) {
                verify(hit, entry.value,
                       VideoCore::TextureCache::ImageDesc{regs.depth_buffer, regs.depth_view,
                                                          regs.depth_control, htile_address, hint});
            }
            desc = entry.value;
        } else {
            std::construct_at(&desc, regs.depth_buffer, regs.depth_view, regs.depth_control,
                              htile_address, hint);
        }
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    } else {
        db_desc.first = {};
    }
}

static std::pair<u32, u32> GetDrawOffsets(
    const AmdGpu::Regs& regs, const Shader::BoundStage& info,
    const Shader::Gcn::FetchShaderData* fetch_shader) {
    u32 vertex_offset = regs.index_offset;
    u32 instance_offset = 0;
    if (fetch_shader) {
        if (vertex_offset == 0 && fetch_shader->vertex_offset_sgpr != -1) {
            vertex_offset = info.user_data[fetch_shader->vertex_offset_sgpr];
        }
        if (fetch_shader->instance_offset_sgpr != -1) {
            instance_offset = info.user_data[fetch_shader->instance_offset_sgpr];
        }
    }
    return {vertex_offset, instance_offset};
}

void Rasterizer::EliminateFastClear() {
    auto& col_buf = liverpool->BackRegs().color_buffers[0];
    if (!col_buf || !col_buf.info.fast_clear) {
        return;
    }
    VideoCore::TextureCache::ImageDesc desc(col_buf, liverpool->BackCbExtent(0));
    const auto image_id = texture_cache.FindImage(desc);
    const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
    if (!texture_cache.IsMetaCleared(col_buf.CmaskAddress(), col_buf.view.slice_start)) {
        return;
    }
    for (u32 slice = col_buf.view.slice_start; slice <= col_buf.view.slice_max; ++slice) {
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);
    }
    auto& image = texture_cache.GetImage(image_id);
    const auto clear_value = LiverpoolToVK::ColorBufferClearValue(col_buf);

    ScopeMarkerBegin(fmt::format("EliminateFastClear:MRT={:#x}:M={:#x}", col_buf.Address(),
                                 col_buf.CmaskAddress()));
    image.Clear(clear_value, desc.view_info.range);
    ScopeMarkerEnd();
}

// SHADPS4_REG_SCAN=<hex>[,<hex>...] (diagnostic): every 512th draw, scan all registers for the
// given 32-bit values (e.g. a surface address >> 8) and log each register index that holds one,
// once. Finds which register points at a surface the emulator does not model (Pro ID buffer).
void Rasterizer::ScanRegistersForValue() {
    static const std::vector<u32> targets = [] {
        std::vector<u32> out;
        const char* value = std::getenv("SHADPS4_REG_SCAN");
        if (!value) {
            return out;
        }
        std::string_view list{value};
        while (!list.empty()) {
            const size_t comma = list.find(',');
            const auto item = list.substr(0, comma);
            out.push_back(u32(std::strtoull(std::string(item).c_str(), nullptr, 16)));
            list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1);
        }
        return out;
    }();
    if (targets.empty()) {
        return;
    }
    static u64 draws = 0;
    if ((draws++ & 511) != 0) {
        return;
    }
    static std::unordered_set<u64> seen;
    const auto& regs = liverpool->BackRegs().reg_array;
    for (u32 i = 0; i < regs.size(); ++i) {
        for (const u32 target : targets) {
            if (regs[i] == target && seen.insert((u64(i) << 32) | target).second) {
                LOG_INFO(Render_Vulkan, "REG_SCAN: value {:#x} in register word {:#x} (draw {})",
                         target, i, draws);
            }
        }
    }
}

struct Rasterizer::NeoIdDerive {
    vk::UniqueDescriptorSetLayout set_layout;
    vk::UniquePipelineLayout layout;
    vk::UniquePipeline pipeline;
    vk::Image depth_image{};
    vk::ImageView depth_view{};
    vk::Image id_image{};
    vk::ImageView id_view{};
    // Views are never destroyed while possibly in flight: superseded ones are kept (rare, small).
    std::vector<vk::UniqueImageView> views;
    u64 runs{};
};

void Rasterizer::DeriveNeoIdBuffer(VideoCore::Image& id_image) {
    static const s64 parity_override = [] {
        const char* value = std::getenv("SHADPS4_NEO_IDBUF_PARITY");
        return value ? static_cast<s64>(std::strtoll(value, nullptr, 0)) : s64{-1};
    }();
    static const u32 bucket_mask = [] {
        const char* value = std::getenv("SHADPS4_NEO_IDBUF_MASK");
        return value ? static_cast<u32>(std::strtoull(value, nullptr, 0)) : 0xFFFFF000u;
    }();
    if (!neo_depth_addr) {
        return;
    }
    const auto depth_id = texture_cache.FindDepthImageBySize(neo_depth_addr, neo_depth_size);
    if (!depth_id) {
        return;
    }
    auto& depth = texture_cache.GetImage(depth_id);
    if (depth.info.num_samples != 2 || depth.info.size.width != id_image.info.size.width ||
        depth.info.size.height != id_image.info.size.height) {
        return;
    }
    const auto device = instance.GetDevice();
    if (!neo_id_derive) {
        neo_id_derive = std::make_shared<NeoIdDerive>();
        auto& d = *neo_id_derive;
        const std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {{
            {.binding = 0,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 1,
             .descriptorType = vk::DescriptorType::eStorageImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
        }};
        const vk::DescriptorSetLayoutCreateInfo set_ci{
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = 2,
            .pBindings = bindings.data(),
        };
        d.set_layout = Check(device.createDescriptorSetLayoutUnique(set_ci));
        const vk::PushConstantRange push_range{
            .stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = 16};
        const vk::PipelineLayoutCreateInfo layout_ci{
            .setLayoutCount = 1,
            .pSetLayouts = &*d.set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range,
        };
        d.layout = Check(device.createPipelineLayoutUnique(layout_ci));
        const auto module =
            Compile(HostShaders::NEO_ID_DERIVE_COMP, vk::ShaderStageFlagBits::eCompute, device);
        const vk::ComputePipelineCreateInfo pipeline_ci{
            .stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"},
            .layout = *d.layout,
        };
        d.pipeline = Check(device.createComputePipelineUnique({}, pipeline_ci));
        device.destroyShaderModule(module);
        LOG_INFO(Render_Vulkan, "NeoIdDerive: pipeline created");
    }
    auto& d = *neo_id_derive;
    if (d.depth_image != depth.GetImage()) {
        d.depth_image = depth.GetImage();
        const vk::ImageViewCreateInfo view_ci{
            .image = d.depth_image,
            .viewType = vk::ImageViewType::e2D,
            .format = depth.info.pixel_format,
            .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1},
        };
        d.views.push_back(Check(device.createImageViewUnique(view_ci)));
        d.depth_view = *d.views.back();
    }
    if (d.id_image != id_image.GetImage()) {
        d.id_image = id_image.GetImage();
        const vk::ImageViewCreateInfo view_ci{
            .image = d.id_image,
            .viewType = vk::ImageViewType::e2D,
            .format = vk::Format::eR32Uint,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        d.views.push_back(Check(device.createImageViewUnique(view_ci)));
        d.id_view = *d.views.back();
    }
    const u32 s0_x = neo_depth_x0y0 & 0xF;
    const u32 parity = parity_override >= 0 ? static_cast<u32>(parity_override & 1)
                                            : ((s0_x != 0 && s0_x < 8) ? 1u : 0u);
    const std::array<u32, 4> push{id_image.info.size.width, id_image.info.size.height, parity,
                                  bucket_mask};

    scheduler.EndRendering();
    depth.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead, {});
    id_image.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite, {});
    const vk::DescriptorImageInfo depth_info{.imageView = d.depth_view,
                                             .imageLayout = vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo id_info{.imageView = d.id_view,
                                          .imageLayout = vk::ImageLayout::eGeneral};
    const std::array<vk::WriteDescriptorSet, 2> writes = {{
        {.dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eSampledImage,
         .pImageInfo = &depth_info},
        {.dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageImage,
         .pImageInfo = &id_info},
    }};
    {
        auto command_access = scheduler.AcquireCommandBuffer();
        const auto cmdbuf = command_access.Handle();
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *d.pipeline);
        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *d.layout, 0, writes);
        cmdbuf.pushConstants(*d.layout, vk::ShaderStageFlagBits::eCompute, 0,
                             static_cast<u32>(sizeof(push)), push.data());
        cmdbuf.dispatch((push[0] + 7) / 8, (push[1] + 7) / 8, 1);
    }
    ++d.runs;
    if (d.runs <= 20 || (d.runs & 1023) == 0) {
        LOG_INFO(Render_Vulkan,
                 "NeoIdDerive: depth {:#x} {}x{} 2s -> id {:#x} parity {} (x0y0 {:#x}) mask {:#x} "
                 "run {}",
                 neo_depth_addr, depth.info.size.width, depth.info.size.height,
                 id_image.info.guest_address, parity, neo_depth_x0y0, bucket_mask, d.runs);
    }
}

void Rasterizer::Draw(bool is_indexed, u32 index_offset) {
    RENDERER_TRACE;
    Common::SampleCommandProcessorCore();
    ScanRegistersForValue();

    auto sample = Common::DrawSampleProbe::Begin();
    const auto t_entry = Common::FaultMonitor::Now();
    buffer_cache.CommitDeletes(); // r13
    scheduler.PopPendingOperations();
    if (native_bindings) {
        // The only point where the heap backend may flush or wait; nothing is assembled yet.
        native_bindings->BeginDraw();
    }

    if (!FilterDraw()) {
        return;
    }

    AmdGpu::SplitHazard::NoteDraw(liverpool->regs.ps_program, liverpool->regs.vs_program);
    if (SkipEmptyDraw(liverpool->regs)) {
        return;
    }

    // Scoped timers: the stack scanner reports inclusive frequencies with false positives and
    // cannot apportion cost here. See Common::FaultMonitor for why these exist.
    const auto t_pipeline = Common::FaultMonitor::Now();
    sample.BeginPhase(Common::DrawSampleProbe::Phase::Pipeline);
    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline();
    sample.EndPhase(Common::DrawSampleProbe::Phase::Pipeline);
    const auto pipeline_ns = Common::FaultMonitor::SinceNs(t_pipeline);
    if (!pipeline) {
        return;
    }
    BindStageData(pipeline);
    DrawTail(pipeline, is_indexed, index_offset, sample, t_entry, t_pipeline, pipeline_ns);
}

namespace {
/// SHADPS4_EMPTY_DRAWS mode, read once (shared by SkipEmptyDraw and SkipEmptyDrawRegs).
int EmptyDrawsMode() {
    static const int empty_draws = [] {
        const char* value = std::getenv("SHADPS4_EMPTY_DRAWS");
        return value ? value[0] - '0' : 0;
    }();
    return empty_draws;
}
} // namespace

bool Rasterizer::SkipEmptyDraw(const AmdGpu::Regs& r) {
    // SHADPS4_EMPTY_DRAWS=1 counts direct draws with zero indices or zero instances (they render
    // nothing but still bind every resource); =2 also skips them. Clears and resolves are
    // handled in FilterDraw before this point. Register reads and statics only (front-safe).
    const int empty_draws = EmptyDrawsMode();
    if (empty_draws > 0) {
        static u64 draws = 0, empty = 0;
        const bool is_empty = r.num_indices == 0 || r.num_instances.NumInstances() == 0;
        ++draws;
        empty += is_empty;
        if ((draws & 0xfffff) == 0) {
            LOG_INFO(Render_Vulkan, "EMPTY_DRAWS: {} of {} direct draws ({:.2f}%)", empty, draws,
                     100.0 * empty / draws);
        }
        if (is_empty && empty_draws == 2) {
            return true;
        }
    }
    return false;
}

bool Rasterizer::SkipEmptyDrawRegs(const AmdGpu::Regs& r) {
    // SkipEmptyDraw's decision only: mode 2 and a zero index / instance count. The front's
    // non-atomic EMPTY_DRAWS counters stay the front's.
    return EmptyDrawsMode() == 2 && (r.num_indices == 0 || r.num_instances.NumInstances() == 0);
}

bool Rasterizer::NativeColourVariantReady(const GraphicsPipeline* pipeline,
                                          const AmdGpu::CpSplit::VertexInputsDerived& vi) {
    // Mirrors the no-change exits of PipelineCache::PrepareNativeColourLocked without its lock:
    // ineligible pipeline (2) or a slot already claimed for this description -> the front would
    // return without writing. Undecided (0), or eligible with a free slot for a new description
    // -> the front would write (decide / claim; its 2 ms throttle is not visible here, so the
    // free-slot case is conservative). No free slot -> the front returns without a claim.
    if (!pipeline) {
        return true; // PrepareNativeColourFront does nothing for a null pipeline
    }
    const u8 eligibility = pipeline->native_colour_static; // front-written once, 0 -> 1 or 2
    if (eligibility == 2) {
        return true;
    }
    if (eligibility == 0) {
        return false;
    }
    const u64 key = VideoCore::NativeVertexKey(vi.attribs.data(), vi.header.count);
    bool free_slot = false;
    for (const auto& slot : pipeline->native_colour) {
        const u64 k = slot.key.load(std::memory_order_acquire);
        if (k == key) {
            return true; // claimed: built, building or failed
        }
        free_slot |= k == 0;
    }
    return !free_slot;
}

bool Rasterizer::NoParticlesSkip(const GraphicsPipeline* pipeline) {
    if (!VideoCore::g_no_particles || !pipeline) {
        return false;
    }
    bool vs = false, fs = false;
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        vs |= stage->l_stage == Shader::LogicalStage::Vertex && stage->pgm_hash == 0xbbd2d927;
        fs |= stage->l_stage == Shader::LogicalStage::Fragment && stage->pgm_hash == 0x24dad4be;
    }
    if (vs && fs) {
        VideoCore::NoParticlesNote(true);
        return true;
    }
    return false;
}

void Rasterizer::PrepareDrawFront(SplitDraw& draw) {
    // CP split, front thread: the part of Draw/DrawIndirect before PopPendingOperations that
    // reads only the front's registers and the pipeline cache, in the same order as the inline
    // path. Only order change: the lookup now runs before the back's PopPendingOperations and
    // native_bindings->BeginDraw instead of after them.
    // SPLIT-TODO: confirm the pipeline constructor does not depend on BeginDraw (heap backend).
    front_capture_valid = false;
    draw.draw_class = ClassifyDraw(liverpool->regs);
    Dlss::NoteDraw(liverpool->regs);
    if (draw.draw_class != DrawClass::Draw) {
        draw.dlss_phase = Dlss::t_phase;
        draw.dlss_jitter[0] = Dlss::t_jitter[0];
        draw.dlss_jitter[1] = Dlss::t_jitter[1];
        return;
    }
    if (!draw.args.is_indirect) {
        AmdGpu::SplitHazard::NoteDraw(liverpool->regs.ps_program, liverpool->regs.vs_program);
        draw.skip = SkipEmptyDraw(liverpool->regs);
        if (draw.skip) {
            return;
        }
    }
    if (AmdGpu::LeanMode() == 2) [[unlikely]] {
        // Lean workers may build pipelines (LookupOrBuildGraphics), which repoints the shared
        // Infos' per-draw fields: copy this draw's under the same exclusive hold, then release
        // (never across guest reads: the back's GetComputePipeline takes this lock too).
        auto lk = pipeline_cache.AcquireExclusive();
        AmdGpu::t_front_in_lock = true;
        draw.pipeline = pipeline_cache.GetGraphicsPipelineLocked();
        if (VideoCore::g_native_pass != 0 && draw.pipeline && !draw.pipeline->native_checked) {
            pipeline_cache.PrepareNativeVariantLocked(draw.pipeline, liverpool->regs);
        }
        pipeline_cache.CaptureFront(front_capture, draw.pipeline);
        AmdGpu::t_front_in_lock = false;
        front_capture_valid = true;
        draw.skip = NoParticlesSkip(draw.pipeline); // q40
        return;
    }
    draw.pipeline = pipeline_cache.GetGraphicsPipeline();
    if (VideoCore::g_native_pass != 0 && draw.pipeline && !draw.pipeline->native_checked)
        [[unlikely]] {
        pipeline_cache.PrepareNativeVariant(draw.pipeline);
    }
    draw.skip = NoParticlesSkip(draw.pipeline); // q40
}

bool Rasterizer::PrepareMemberFront(SplitDraw& draw, const GraphicsPipeline* pipeline) {
    // As PrepareDrawFront for a direct draw, without the lookup: the job thread proved the
    // shaders, vertex layout, specialization and context equal to the run head's.
    draw.draw_class = ClassifyDraw(liverpool->regs);
    Dlss::NoteDraw(liverpool->regs);
    if (draw.draw_class != DrawClass::Draw || draw.args.is_indirect) {
        return false;
    }
    AmdGpu::SplitHazard::NoteDraw(liverpool->regs.ps_program, liverpool->regs.vs_program);
    draw.skip = SkipEmptyDraw(liverpool->regs) || NoParticlesSkip(pipeline); // q40
    draw.pipeline = pipeline;
    return true;
}

const ComputePipeline* Rasterizer::PrepareDispatchFront() {
    // CP split, front thread: reads the front's current queue cs registers (GetCsRegs).
    return pipeline_cache.GetComputePipeline();
}

const GraphicsPipeline* Rasterizer::RelookupDrawFront() {
    // Liverpool::ExecuteDrawRecord (VERIFY): the front is parked, so its registers and the
    // pipeline cache are stable and published to this thread (see DrawRelookup).
    return pipeline_cache.GetGraphicsPipeline();
}

// SHADPS4_DRAW_TAG=1 (with SHADPS4_NATIVE_TIMING=1): the back groups draw cycles by the game's
// job-thread runs (DrawTagNote).
static const bool g_draw_tag_back = [] {
    const char* e = std::getenv("SHADPS4_DRAW_TAG");
    return e && e[0] == '1';
}();

void Rasterizer::ExecuteDraw(const SplitDraw& draw,
                             const std::array<Shader::BoundStage, Shader::MaxStageTypes>& stages,
                             const AmdGpu::CpSplit::VertexInputsPayload* front_vi) {
    // CP split, back thread: Draw/DrawIndirect with the front's classification, lookup and
    // per-stage data. Registers come from the back's mirror through BackRegs().
    // With run-ahead the front runs concurrently: nothing here may read front state (regs,
    // Info::user_data/flattened_ud_buf/pgm_base, pipeline-cache tables). Indirect args are read
    // here, in stream order, so they need no front drain.
    // SPLIT-TODO(run-ahead): FaultMonitor's single-writer counters must keep one writer each
    // (checked: draw/dispatch counts are back-only, WaitRegMem/CpWait/CpActive front-only; the
    // other Count* sites are unchecked).
    RENDERER_TRACE;
    // Same prologue as the inline Draw/DrawIndirect, minus what the front already did.
    const auto begin = [&] {
        buffer_cache.CommitDeletes(); // r13
        scheduler.PopPendingOperations();
        if (native_bindings) {
            native_bindings->BeginDraw();
        }
        if (draw.draw_class != DrawClass::Draw) {
            dlss_phase_in = draw.dlss_phase;
            dlss_jitter_in[0] = draw.dlss_jitter[0];
            dlss_jitter_in[1] = draw.dlss_jitter[1];
            ExecuteFilteredDraw(draw.draw_class);
            return false;
        }
        if (draw.skip || !draw.pipeline) {
            return false;
        }
        bound_stages = stages;
        // SHADPS4_DRAW_REPEAT_STATS=1: census of draws whose binding inputs repeat the previous
        // draw's (scripts/patch/draw-repeat-census-2026-10-02.py).
        static const bool repeat_stats = [] {
            const char* value = std::getenv("SHADPS4_DRAW_REPEAT_STATS");
            return value && value[0] == '1';
        }();
        if (repeat_stats && !draw.args.is_indirect) [[unlikely]] {
            struct State {
                const GraphicsPipeline* pipeline{};
                std::vector<u32> inputs;
                u64 epoch{}, buf_gen{}, tex_gen{}, tick{};
                u64 draws{}, same_pipeline{}, same_inputs{}, all_same{};
                u64 broke_epoch{}, broke_buf_gen{}, broke_tex_gen{}, broke_tick{};
                std::vector<u32> scratch;
            };
            static State st;
            st.scratch.clear();
            for (const auto& stage : stages) {
                if (!stage.info) {
                    continue;
                }
                // Only what the binding resolution reads: every V# / T# / S# of the stage and the
                // user-data registers it pushes (the raw registers also hold per-draw values the
                // bindings never read, e.g. the vertex-buffer table pointer: 0.0% repeats).
                const auto add = [&](const auto& sharp) {
                    const size_t words = sizeof(sharp) / sizeof(u32);
                    const size_t at = st.scratch.size();
                    st.scratch.resize(at + words);
                    std::memcpy(&st.scratch[at], &sharp, words * sizeof(u32));
                };
                for (const auto& desc : stage.info->buffers) {
                    add(desc.GetSharp(stage));
                }
                for (const auto& desc : stage.info->images) {
                    add(desc.GetSharp(stage));
                }
                for (const auto& desc : stage.info->samplers) {
                    add(desc.GetSharp(stage));
                }
                u32 mask = stage.info->ud_mask.mask;
                while (mask) {
                    const u32 index = std::countr_zero(mask);
                    mask &= ~(1U << index);
                    st.scratch.push_back(stage.user_data[index]);
                }
            }
            const u64 epoch = VideoCore::g_dirty_epoch.load(std::memory_order_acquire);
            const u64 buf_gen = buffer_cache.StructGeneration();
            const u64 tex_gen = texture_cache.StructGeneration();
            const u64 tick = scheduler.CurrentTick();
            ++st.draws;
            const bool same_pipeline = st.pipeline == draw.pipeline;
            st.same_pipeline += same_pipeline;
            if (same_pipeline && st.scratch == st.inputs) {
                ++st.same_inputs;
                const bool e = epoch == st.epoch, b = buf_gen == st.buf_gen,
                           t = tex_gen == st.tex_gen, k = tick == st.tick;
                st.all_same += e && b && t && k;
                st.broke_epoch += !e;
                st.broke_buf_gen += !b;
                st.broke_tex_gen += !t;
                st.broke_tick += !k;
            }
            st.pipeline = draw.pipeline;
            st.inputs.swap(st.scratch);
            st.epoch = epoch;
            st.buf_gen = buf_gen;
            st.tex_gen = tex_gen;
            st.tick = tick;
            if ((st.draws & 0x1fffff) == 0) {
                const double d = static_cast<double>(st.draws);
                LOG_INFO(Render_Vulkan,
                         "DrawRepeat: draws={} same_pipeline={:.1f}% same_inputs={:.1f}% "
                         "same_inputs_and_unchanged={:.1f}% | of same-input draws broken by: "
                         "dirty_epoch={:.1f}% buffer_gen={:.1f}% texture_gen={:.1f}% tick={:.1f}% "
                         "| input words per draw={}",
                         st.draws, 100.0 * st.same_pipeline / d, 100.0 * st.same_inputs / d,
                         100.0 * st.all_same / d,
                         st.same_inputs ? 100.0 * st.broke_epoch / st.same_inputs : 0.0,
                         st.same_inputs ? 100.0 * st.broke_buf_gen / st.same_inputs : 0.0,
                         st.same_inputs ? 100.0 * st.broke_tex_gen / st.same_inputs : 0.0,
                         st.same_inputs ? 100.0 * st.broke_tick / st.same_inputs : 0.0,
                         st.inputs.size());
                st.draws = st.same_pipeline = st.same_inputs = st.all_same = 0;
                st.broke_epoch = st.broke_buf_gen = st.broke_tex_gen = st.broke_tick = 0;
            }
        }
        if (VideoCore::BufferCache::CensusEnabled()) [[unlikely]] {
            u64 id = 0;
            VideoCore::g_census_vs = 0; // q35
            VideoCore::g_census_args = 0;
            for (const auto* stage : draw.pipeline->GetStages()) {
                if (stage && stage->l_stage == Shader::LogicalStage::Fragment) {
                    id = stage->pgm_hash;
                }
                if (stage && stage->l_stage == Shader::LogicalStage::Vertex) {
                    VideoCore::g_census_vs = stage->pgm_hash;
                }
            }
            buffer_cache.SetWriterTag(1, id);
        }
        return true;
    };
    if (draw.args.is_indirect) {
        if (begin()) {
            DrawIndirectTail(draw.pipeline, draw.args.is_indexed, draw.args.arg_address,
                             draw.args.offset, draw.args.stride, draw.args.max_count,
                             draw.args.count_address);
        }
        return;
    }
    Common::SampleCommandProcessorCore();
    ScanRegistersForValue();
    auto sample = Common::DrawSampleProbe::Begin();
    const auto t_entry = Common::FaultMonitor::Now();
    if (begin()) {
        static const bool native_timing = [] {
            const char* e = std::getenv("SHADPS4_NATIVE_TIMING");
            return e && e[0] == '1';
        }();
        if (VideoCore::g_native_pass != 0 && native_batch.open) {
            const u64 t0 = native_timing ? __rdtsc() : 0;
            const bool was_colour = native_batch.colour;
            const bool joined = NativeJoin(draw, front_vi);
            if (native_timing) {
                const u64 dt = __rdtsc() - t0;
                (joined ? native_stats.c_join_ok : native_stats.c_join_fail) += dt;
                ++(joined ? native_stats.n_join_ok : native_stats.n_join_fail);
                if (joined && was_colour) {
                    native_stats.c_cjoin_ok += dt;
                    ++native_stats.n_cjoin_ok;
                }
                if (joined && g_draw_tag_back) [[unlikely]] {
                    DrawTagNote(draw.pipeline, was_colour ? 2u : 0u, dt);
                }
            }
            if (joined) {
                rec_outcome = was_colour ? 2u : 1u;
                return;
            }
        }
        rec_outcome = 3;
        const u64 t1 = native_timing ? __rdtsc() : 0;
        native_stats.last_open_colour = false;
        // The lookup ran on the front: the Pipeline phase and pipeline_ns are 0 here.
        DrawTail(draw.pipeline, draw.args.is_indexed, draw.args.index_offset, sample, t_entry,
                 t_entry, 0, front_vi);
        if (native_timing) {
            const u64 dt = __rdtsc() - t1;
            if (native_stats.last_open_colour) {
                native_stats.seg_last_native = true;
                native_stats.c_tail_copen += dt;
                ++native_stats.n_tail_copen;
                if (native_stats.last_open_direct) {
                    native_stats.c_tail_dopen += dt;
                    ++native_stats.n_tail_dopen;
                } else {
                    native_stats.c_bind_copen += native_stats.last_bind_cyc;
                    ++native_stats.n_bind_copen;
                }
                {
                    const u32 a = native_stats.close_key ? 1u : 0u;
                    const u32 fl = native_stats.last_flags & 31;
                    native_stats.c_cop_fl[a][fl] += dt;
                    ++native_stats.n_cop_fl[a][fl];
                    native_stats.close_key = false;
                }
                if (g_draw_tag_back) [[unlikely]] {
                    DrawTagNote(draw.pipeline, 3u, dt);
                }
            } else if (draw.pipeline->native_variant.load(std::memory_order_relaxed)) {
                native_stats.seg_last_native = true;
                native_stats.c_tail_nat += dt;
                ++native_stats.n_tail_nat;
                if (g_draw_tag_back) [[unlikely]] {
                    DrawTagNote(draw.pipeline, 1u, dt);
                }
            } else {
                native_stats.c_tail_other += dt;
                ++native_stats.n_tail_other;
                native_stats.seg_to_other += native_stats.seg_last_native;
                native_stats.seg_last_native = false;
                native_stats.seg_generic_since = true;
                {
                    auto& w = native_stats.why_other[native_stats.last_why
                                                         ? native_stats.last_why
                                                         : "unknown"];
                    ++w.first;
                    w.second += dt;
                }
                bool images = false;
                for (const auto* stage : draw.pipeline->GetStages()) {
                    images = images || (stage && !stage->images.empty());
                }
                const u32 cls = (native_stats.last_tail_repeat ? 1u : 0u) | (images ? 2u : 0u);
                native_stats.c_tail_cls[cls] += dt;
                ++native_stats.n_tail_cls[cls];
                if (cls == 2) {
                    native_stats.c_flags[native_stats.last_flags & 127] += dt;
                    ++native_stats.n_flags[native_stats.last_flags & 127];
                }
                if (g_draw_tag_back) [[unlikely]] {
                    DrawTagNote(draw.pipeline, 4u + cls, dt);
                }
            }
        }
    }
}

// SHADPS4_NATIVE_DIAG=1 (diagnostic): which context registers changed when a join is refused
// for a context-epoch change (close reason 6). Copies the context block at every batch open.
static const bool g_native_diag = [] {
    const char* e = std::getenv("SHADPS4_NATIVE_DIAG");
    return e && e[0] == '1';
}();
static std::vector<u32> g_native_diag_ctx;
static std::unordered_map<u32, u64> g_native_diag_regs;
static u64 g_native_diag_n = 0;

bool Rasterizer::NativeImageKey(
    const GraphicsPipeline* base, std::array<u32, 512>& out, u32& n,
    const std::array<Shader::BoundStage, Shader::MaxStageTypes>* stages) const {
    // SHADPS4_NATIVE_COLOUR: the raw T# / S# of every image and sampler of every stage, as the
    // draw-repeat key reads them (DrawRepeatMatch). False on overflow. `stages` (brick C v3
    // item 4): another draw's bound stages (a segment material's representative) in place of
    // the rasterizer's.
    n = 0;
    const auto add = [&](const auto& sharp) {
        constexpr u32 words = static_cast<u32>(sizeof(sharp) / sizeof(u32));
        if (n + words > out.size()) {
            return false;
        }
        std::memcpy(&out[n], &sharp, sizeof(sharp));
        n += words;
        return true;
    };
    const std::span<const Shader::BoundStage> bound_span =
        stages ? std::span<const Shader::BoundStage>{*stages}.first(Shader::MaxStageTypes - 1)
               : BoundStages(base);
    for (const auto& bound : bound_span) {
        if (!bound.info) {
            continue;
        }
        for (const auto& desc : bound.info->images) {
            if (!add(desc.GetSharp(bound))) {
                return false;
            }
        }
        for (const auto& desc : bound.info->samplers) {
            if (!add(desc.GetSharp(bound))) {
                return false;
            }
        }
    }
    return true;
}

// SHADPS4_NATIVE_REC_RUN: 1 = members of a validated shadow run take their bindings from the
// run's spans, 2 = verify (both paths, compare handle / offset; the normal result is used).
static const int g_native_rec_run = [] {
    const char* e = std::getenv("SHADPS4_NATIVE_REC_RUN");
    return e ? std::atoi(e) : 0;
}();

void Rasterizer::NativeRunValidate(const GraphicsPipeline* base,
                                   Pipeline::BufferBarriers* barriers) {
    // SHADPS4_NATIVE_REC_RUN (stage c1, notes/native-path-2026-10-09.md sec. 7): at a shadow
    // batch's open, if the draw heads a job-thread run, read the members' records and validate one
    // span per guest buffer slot, vertex stream and the index buffer, each inside ONE existing
    // buffer; members then take offsets inside the spans (NativeJoin) while the dirty epoch and
    // the buffer structure generation stay put. Back thread, before the multi-draw is recorded.
    namespace NR = AmdGpu::NativeRec;
    auto& b = native_batch;
    auto& st = native_stats;
    b.run_valid = false;
    if (g_native_rec_run == 0 || nr_head_tag == 0 || NR::g_mode < 2) {
        return;
    }
    const auto fail = [&](u32 why) { ++st.run_fail[std::min<u32>(why, 7)]; };
    const u32 slot = nr_head_tag >> 28;
    const NR::Record* ring = NR::g_rings[slot].load(std::memory_order_acquire);
    const u8* payload = NR::g_payloads[slot].load(std::memory_order_acquire);
    if (!ring || !payload) {
        return fail(0);
    }
    const u32 idx = nr_head_idx;
    const NR::Record& head = ring[idx & (NR::RingSize - 1)];
    const u32 count = head.count.load(std::memory_order_acquire);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (head.stamp.load(std::memory_order_relaxed) != idx + 1 || count < 2) {
        return fail(0);
    }
    const u32 n = std::min<u32>(count, NativeMaxDraws);
    const auto base_stages = base->GetStages();
    struct Span {
        VAddr lo{~VAddr{0}}, hi{};
        void Add(VAddr a, u64 size) {
            lo = std::min(lo, a);
            hi = std::max<VAddr>(hi, a + size);
        }
    };
    std::array<Span, Shader::NUM_BUFFERS> sb{};
    std::array<Span, 16> sv{};
    Span si{};
    u64 first_pos = 0;
    for (u32 k = 1; k < n; ++k) {
        const u32 mi = idx + k;
        const NR::Record& r = ring[mi & (NR::RingSize - 1)];
        if (r.stamp.load(std::memory_order_acquire) != mi + 1) {
            return fail(1);
        }
        const u32 run_k = r.run_k, head_idx = r.head_idx;
        const bool derived = r.derived != 0;
        const void* pipeline = r.pipeline;
        const u32 nvs = r.flat_vs_dw, nps = r.flat_ps_dw, vi_bytes = r.vi_bytes;
        const u64 pos = r.payload_pos;
        const auto ud_vs = r.ud_vs;
        const auto ud_ps = r.ud_ps;
        const VAddr vs_pgm = VAddr{r.vs_pgm} << 8, ps_pgm = VAddr{r.ps_pgm} << 8;
        const u64 index_addr = r.index_addr;
        const u32 index_count = r.index_count;
        std::atomic_thread_fence(std::memory_order_acquire);
        if (r.stamp.load(std::memory_order_relaxed) != mi + 1 || run_k != k || head_idx != idx ||
            !derived || !pipeline || nvs > 1024 || nps > 1024 ||
            vi_bytes < sizeof(AmdGpu::CpSplit::VertexInputsPayload)) {
            return fail(1);
        }
        const auto rstages = static_cast<const GraphicsPipeline*>(pipeline)->GetStages();
        const u32 vs_i = static_cast<u32>(Shader::LogicalStage::Vertex);
        const u32 ps_i = static_cast<u32>(Shader::LogicalStage::Fragment);
        if (rstages[vs_i] != base_stages[vs_i] || rstages[ps_i] != base_stages[ps_i]) {
            return fail(2);
        }
        if (k == 1) {
            first_pos = pos;
        }
        const u8* pl = payload + (pos & (NR::PayloadSize - 1));
        const u32* flat_vs = reinterpret_cast<const u32*>(pl);
        const u32* flat_ps = flat_vs + nvs;
        u32 bi = 0;
        for (u32 i = 0; i < base_stages.size(); ++i) {
            const Shader::Info* info = base_stages[i];
            if (!info) {
                continue;
            }
            if (i != vs_i && i != ps_i) {
                return fail(2);
            }
            Shader::BoundStage bs{};
            bs.info = info;
            bs.user_data = std::span<const u32>{i == vs_i ? ud_vs.data() : ud_ps.data(), 16};
            bs.flattened_ud_buf =
                std::span<const u32>{i == vs_i ? flat_vs : flat_ps, i == vs_i ? nvs : nps};
            bs.pgm_base = i == vs_i ? vs_pgm : ps_pgm;
            for (const auto& desc : info->buffers) {
                if (bi >= sb.size()) {
                    return fail(3);
                }
                if (b.guest_mask & (u64{1} << bi)) {
                    const auto vsharp = desc.GetSharp(bs);
                    if (vsharp.base_address == 0 || vsharp.GetSize() == 0) {
                        return fail(3);
                    }
                    sb[bi].Add(vsharp.base_address,
                               memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize()));
                }
                ++bi;
            }
        }
        const auto* hdr = reinterpret_cast<const AmdGpu::CpSplit::VertexInputsPayload*>(
            pl + (u64(nvs + nps) * 4 + 7) / 8 * 8);
        if (hdr->count != b.vb_n || hdr->count > sv.size()) {
            return fail(4);
        }
        const auto* att = reinterpret_cast<const AmdGpu::CpSplit::VertexInputAttrib*>(hdr + 1);
        for (u32 a = 0; a < hdr->count; ++a) {
            if (att[a].range != AmdGpu::CpSplit::VertexInputNoRange && att[a].base_address != 0 &&
                att[a].size != 0) {
                sv[a].Add(att[a].base_address, att[a].size);
            }
        }
        si.Add(index_addr, u64(index_count) * b.index_size);
    }
    // The payload is intact only if the job thread has not claimed past the first member's.
    std::atomic_thread_fence(std::memory_order_acquire);
    if (NR::g_payload_claim[slot].load(std::memory_order_relaxed) - first_pos > NR::PayloadSize) {
        return fail(1);
    }
    // Validate each span once, inside one existing buffer, against the batch's handles.
    const auto validate = [&](const Span& s, u64 want_handle, VideoCore::NativeRunAccess access,
                              NativeRunSpan& out) {
        if (s.hi <= s.lo) {
            out = {};
            return true; // unused
        }
        const VideoCore::Buffer* buf =
            buffer_cache.NativeRunSpan(s.lo, s.hi, access, barriers, cur_dispatch_run);
        if (!buf) {
            return false;
        }
        const u64 handle = std::bit_cast<u64>(static_cast<VkBuffer>(buf->Handle()));
        if (handle != want_handle) {
            return false;
        }
        out = {.lo = s.lo, .hi = s.hi, .cpu = buf->CpuAddr(), .handle = handle};
        return true;
    };
    for (u32 bi = 0; bi < sb.size(); ++bi) {
        if ((b.guest_mask & (u64{1} << bi)) &&
            !validate(sb[bi], b.buf[bi], VideoCore::NativeRunAccess::Storage, b.run_buf[bi])) {
            return fail(5);
        }
    }
    for (u32 a = 0; a < b.vb_n; ++a) {
        if (!validate(sv[a], b.vb[a], VideoCore::NativeRunAccess::Vertex, b.run_vb[a])) {
            return fail(6);
        }
    }
    if (!validate(si, b.ib, VideoCore::NativeRunAccess::Index, b.run_ib)) {
        return fail(7);
    }
    b.run_valid = true;
    b.run_epoch = VideoCore::g_dirty_epoch.load(std::memory_order_acquire);
    b.run_gen = buffer_cache.StructGeneration();
    ++st.run_ok;
}

bool Rasterizer::NativeRunVi(const AmdGpu::CpSplit::VertexInputsPayload& vi, u32 index_offset,
                             VideoCore::NativeJoinVi& out) const {
    // NativeJoinVertexIndex's results from the run's spans, or false when a range is outside.
    const auto& b = native_batch;
    const auto* attribs = AmdGpu::CpSplit::VertexInputAttribs(vi);
    out.n = std::min<u32>(vi.count, static_cast<u32>(out.buf.size()));
    for (u32 k = 0; k < out.n; ++k) {
        const auto& a = attribs[k];
        if (a.range == AmdGpu::CpSplit::VertexInputNoRange) {
            out.buf[k] = 0;
            out.off[k] = 0;
        } else {
            const auto& s = b.run_vb[k];
            if (s.handle == 0 || a.base_address < s.lo || a.base_address + a.size > s.hi) {
                return false;
            }
            out.buf[k] = s.handle;
            out.off[k] = a.base_address - s.cpu;
        }
        out.stride[k] = a.desc.stride;
    }
    const auto& regs = liverpool->BackRegs();
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    const VAddr index_address =
        regs.index_base_address.Address<VAddr>() + index_offset * index_size;
    const u64 index_bytes = u64(regs.num_indices) * index_size;
    const auto& s = b.run_ib;
    if (s.handle == 0 || index_address < s.lo || index_address + index_bytes > s.hi) {
        return false;
    }
    out.ib_buf = s.handle;
    out.ib_off = index_address - s.cpu;
    return true;
}

// SHADPS4_NATIVE_REC_PREP (notes/native-path-2026-10-09.md sec. 9): 1 = the members of a
// job-thread run are prepared at the batch open from the job records (records + indirect
// commands in the batch's slots) and confirmed one by one at their own draw records (count + 1);
// 2 = verify (prepared on the side, never confirmed; the normal join's record and command are
// compared field by field).
static const int g_native_prep = [] {
    const char* e = std::getenv("SHADPS4_NATIVE_REC_PREP");
    return e ? std::atoi(e) : 0;
}();
// SHADPS4_NATIVE_RING_JOIN (sec. 11): 1 (default) = a draw the join accepts joins a ring batch
// when its job record is the batch's next ring entry; 2 = also compare the join's record with the
// job record the GPU reads (slow: reads write-combined memory); 0 = off (close 13).
static const int g_native_ring_join = [] {
    const char* e = std::getenv("SHADPS4_NATIVE_RING_JOIN");
    return e ? std::atoi(e) : 1;
}();
// SHADPS4_NATIVE_DIRECT (sec. 15): 1 = native run start (no BindResources for bindless colour
// openers with a job record), 2 = verify (direct bind built, then the generic bind; compared
// binding by binding; the generic renders).
static const int g_native_direct = [] {
    const char* e = std::getenv("SHADPS4_NATIVE_DIRECT");
    return e ? std::atoi(e) : 0;
}();
// SHADPS4_NATIVE_SPAN_CHECK=1 (Opus G2 #1): count ring span-cache hits over CPU-dirty pages.
static const bool g_native_span_check = [] {
    const char* e = std::getenv("SHADPS4_NATIVE_SPAN_CHECK");
    return e && std::atoi(e) != 0;
}();

namespace {
struct PrepMember {
    std::array<u32, Shader::NUM_USER_DATA_REGS> ud;
    std::array<u64, Shader::NUM_BUFFERS> addr; ///< guest slot V# base (0: the opener's, formatted)
    std::array<u32, Shader::NUM_BUFFERS> size;
    std::array<u64, 16> vaddr; ///< 0: null stream
    std::array<u32, 16> vsize;
    u64 iaddr;
    u32 icount, instances, vertex_offset;
    const void* cmd;
    std::array<u32, Shader::NativeRecordDwords> rec;
    std::array<u32, 5> cmdw;
};
thread_local std::vector<PrepMember> t_prep;
thread_local u32 t_prep_n = 0;
} // namespace

bool Rasterizer::NativeCtxSame() {
    // The epoch moves on any value-changing SET_CONTEXT_REG, also when the game sets a register
    // and sets it back before this draw (s285: at 97% of such refusals only the per-draw index
    // DMA slots VGT_DMA_BASE_HI / BASE / MAX_SIZE / INDEX_TYPE differed). Compare the block
    // itself, without those per-draw slots.
    auto& b = native_batch;
    if (liverpool->BackContextEpoch() == b.ctx_epoch) {
        return true;
    }
    const auto& regs = liverpool->BackRegs();
    const u32* now = &regs.reg_array[AmdGpu::Regs::ContextRegWordOffset];
    static_assert(std::tuple_size_v<decltype(native_ctx)> == AmdGpu::NativeCtxBlockWords);
    const bool same = AmdGpu::NativeCtxBlockSame(now, native_ctx.data());
    if (same) {
        b.ctx_epoch = liverpool->BackContextEpoch();
        ++native_stats.epoch_same;
    }
    return same;
}

void Rasterizer::NativePrepare(const GraphicsPipeline* base,
                               const AmdGpu::CpSplit::VertexInputAttrib* attribs,
                               const std::array<u64, Shader::NUM_BUFFERS>& o_addr,
                               const std::array<u64, Shader::NUM_BUFFERS>& o_size,
                               const std::array<u32, Shader::NUM_BUFFERS>& o_off,
                               const std::array<u32, Shader::NUM_BUFFERS>& o_rsize) {
    // Back thread, NativeOpen of a live batch, after the opener's buffer slots were captured and
    // BEFORE the stream reservation (span barriers go into buffer_barriers, emitted before the
    // multi-draw). Every per-member value comes from the job record: the mirror still holds the
    // opener's user data (the members' SET_SH_REG writes have not been applied yet).
    namespace NR = AmdGpu::NativeRec;
    auto& b = native_batch;
    auto& st = native_stats;
    native_prep.active = false;
    t_prep_n = 0;
    if (g_native_prep == 0 || NR::g_mode < 2 || draw_job_tag == 0 || g_native_rec_run != 0 ||
        b.verify || !attribs || NR::g_gpu_mode == 1) {
        return;
    }
    ++st.prep_try;
    const u64 t0 = __rdtsc();
    const auto stop = [&](u32 why) {
        ++st.prep_stop[std::min<u32>(why, 15)];
        st.c_prep += __rdtsc() - t0;
    };
    const u32 slot = draw_job_tag >> 28;
    const NR::Record* ring = NR::g_rings[slot].load(std::memory_order_acquire);
    const u8* payload = NR::g_payloads[slot].load(std::memory_order_acquire);
    if (!ring || !payload) {
        return stop(0);
    }
    constexpr u32 mask = NR::RingSize - 1;
    const u32 idx = draw_job_idx;
    const NR::Record& r0 = ring[idx & mask];
    const u32 run_k0 = r0.run_k, head_idx0 = r0.head_idx, tag0 = r0.tag;
    const u32* cmd0 = r0.cmd;
    std::atomic_thread_fence(std::memory_order_acquire);
    if (r0.stamp.load(std::memory_order_relaxed) != idx + 1 || tag0 != draw_job_tag ||
        static_cast<const void*>(cmd0) != draw_job_cmd) {
        return stop(0);
    }
    const u32 head = run_k0 == 0 ? idx : head_idx0;
    const NR::Record& hr = ring[head & mask];
    const u32 count = hr.count.load(std::memory_order_acquire);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (hr.stamp.load(std::memory_order_relaxed) != head + 1) {
        return stop(0);
    }
    if (count <= run_k0 + 1 || b.n != 0) {
        return stop(1); // the opener is the run's last draw
    }
    // Slot 0 = the opener (written after the stream reservation), members from slot 1.
    const u32 want = std::min<u32>(count - run_k0 - 1, NativeMaxDraws - 1);
    if (t_prep.size() < NativeMaxDraws) {
        t_prep.resize(NativeMaxDraws);
    }
    const auto& regs = liverpool->BackRegs();
    const auto base_stages = base->GetStages();
    constexpr u32 vs_i = static_cast<u32>(Shader::LogicalStage::Vertex);
    constexpr u32 ps_i = static_cast<u32>(Shader::LogicalStage::Fragment);
    const u32 index_size = b.index_size;
    u64 first_pos = 0;
    u32 n = 0;
    u32 why = 15;
    std::array<u32, 512> key;
    // Exact truncation (s398: halving retries discarded 37% of the members looked at): every
    // member range must lie inside the batch's buffer for its slot, checked by arithmetic against
    // that buffer's bounds (page-table lookup once per slot, no sync); the stretch ends at the
    // first member that does not fit.
    // s402: per-prepare page-table lookups and array zeroing were fixed costs over ~4.4 members;
    // bounds come from a small cache keyed by VkBuffer (the same few buffers recur every batch,
    // valid while the buffer structure generation stays put) and the arrays start uninitialised.
    struct Bounds {
        VAddr lo, hi;
        u64 handle;
    };
    std::array<Bounds, Shader::NUM_BUFFERS> slot_b;
    std::array<Bounds, 16> vb_b;
    Bounds ib_b{0, 0, 0};
    // Running per-slot unions: the stretch also ends before any one range passes NativeRunSpan's
    // 64 MB, so the span sync below cannot refuse the whole prepare.
    struct Run {
        VAddr lo, hi;
        bool Fits(VAddr a, u64 size) const {
            return std::max<VAddr>(hi, a + size) - std::min(lo, a) <= (u64{64} << 20);
        }
        void Add(VAddr a, u64 size) {
            lo = std::min(lo, a);
            hi = std::max<VAddr>(hi, a + size);
        }
    };
    std::array<Run, Shader::NUM_BUFFERS> slot_u;
    std::array<Run, 16> vb_u;
    Run ib_u{~VAddr{0}, 0};
    // Slots and streams start uninitialised; set on first use.
    u64 slot_seen = 0;
    u32 vb_seen = 0;
    const auto slot_init = [&](u32 bi) {
        if (!((slot_seen >> bi) & 1)) {
            slot_seen |= u64{1} << bi;
            slot_b[bi] = {0, 0, 0};
            slot_u[bi] = {~VAddr{0}, 0};
        }
    };
    const auto vb_init = [&](u32 a) {
        if (!((vb_seen >> a) & 1)) {
            vb_seen |= 1u << a;
            vb_b[a] = {0, 0, 0};
            vb_u[a] = {~VAddr{0}, 0};
        }
    };
    const u64 bgen = buffer_cache.StructGeneration();
    if (native_bcache_gen != bgen) {
        native_bcache_gen = bgen;
        native_bcache_n = 0;
    }
    const auto inside = [&](Bounds& bd, VAddr a, u64 size, u64 want) {
        if (a >= bd.lo && a + size <= bd.hi && bd.handle == want) {
            return true;
        }
        for (u32 i = 0; i < native_bcache_n; ++i) {
            const auto& e = native_bcache[i];
            if (e.handle == want && a >= e.lo && a + size <= e.hi) {
                bd = {e.lo, e.hi, want};
                return true;
            }
        }
        const VideoCore::Buffer* buf = buffer_cache.NativeBufferAt(a, size);
        if (!buf || !buf->IsInBounds(a, size)) {
            return false;
        }
        const u64 handle = std::bit_cast<u64>(static_cast<VkBuffer>(buf->Handle()));
        if (handle != want) {
            return false;
        }
        bd = {buf->CpuAddr(), buf->CpuAddr() + buf->SizeBytes(), handle};
        auto& e = native_bcache[native_bcache_n < native_bcache.size()
                                    ? native_bcache_n++
                                    : (native_bcache_rr++ % native_bcache.size())];
        e = {handle, bd.lo, bd.hi};
        return true;
    };
    u64 tl = __rdtsc();
    const auto lap = [&](u32 ph) {
        const u64 now = __rdtsc();
        st.c_prep_ph[ph] += now - tl;
        tl = now;
    };
    lap(0); // ring, opener and head records
    for (u32 j = 0; j < want; ++j) {
        const u32 k = run_k0 + 1 + j;
        const u32 mi = head + k;
        const NR::Record& r = ring[mi & mask];
        if (r.stamp.load(std::memory_order_acquire) != mi + 1) {
            why = 2;
            break;
        }
        // Opus review 1 (stage b): copy every field, fence, re-check the stamp, then use.
        const u32 run_k = r.run_k, head_idx = r.head_idx;
        const bool derived = r.derived != 0;
        const void* pipeline = r.pipeline;
        const u32 nvs = r.flat_vs_dw, nps = r.flat_ps_dw, vi_bytes = r.vi_bytes;
        const u64 pos = r.payload_pos;
        const auto ud_vs = r.ud_vs;
        const auto ud_ps = r.ud_ps;
        const VAddr vs_pgm = VAddr{r.vs_pgm} << 8, ps_pgm = VAddr{r.ps_pgm} << 8;
        const u64 index_addr = r.index_addr;
        const u32 index_count = r.index_count, instances = r.instances,
                  inst_known = r.inst_known;
        const u32* cmd = r.cmd;
        std::atomic_thread_fence(std::memory_order_acquire);
        if (r.stamp.load(std::memory_order_relaxed) != mi + 1 || run_k != k || head_idx != head ||
            !derived || !pipeline || nvs > 1024 || nps > 1024 ||
            vi_bytes < sizeof(AmdGpu::CpSplit::VertexInputsPayload) || !inst_known ||
            index_count == 0) {
            why = 2;
            break;
        }
        const auto rstages = static_cast<const GraphicsPipeline*>(pipeline)->GetStages();
        if (rstages[vs_i] != base_stages[vs_i] || rstages[ps_i] != base_stages[ps_i]) {
            why = 3;
            break;
        }
        if (j == 0) {
            first_pos = pos;
        }
        lap(4); // the job ring record: copy, fence, re-check (cross-core reads)
        const u8* pl = payload + (pos & (NR::PayloadSize - 1));
        const u32* flat_vs = reinterpret_cast<const u32*>(pl);
        const u32* flat_ps = flat_vs + nvs;
        auto& m = t_prep[j];
        std::array<Shader::BoundStage, Shader::MaxStageTypes> bs{};
        Shader::PushData push{};
        Shader::Backend::Bindings bnd{};
        bool ok = true;
        u32 bi = 0;
        u32 kn = 0;
        const auto add_key = [&](const auto& sharp) {
            constexpr u32 words = static_cast<u32>(sizeof(sharp) / sizeof(u32));
            if (kn + words > key.size()) {
                return false;
            }
            std::memcpy(&key[kn], &sharp, sizeof(sharp));
            kn += words;
            return true;
        };
        for (u32 i = 0; ok && i < base_stages.size(); ++i) {
            const Shader::Info* info = base_stages[i];
            if (!info) {
                continue;
            }
            if (i != vs_i && i != ps_i) {
                ok = false;
                why = 3;
                break;
            }
            auto& s = bs[i];
            s.info = info;
            s.user_data = std::span<const u32>{i == vs_i ? ud_vs.data() : ud_ps.data(), 16};
            s.flattened_ud_buf =
                std::span<const u32>{i == vs_i ? flat_vs : flat_ps, i == vs_i ? nvs : nps};
            s.pgm_base = i == vs_i ? vs_pgm : ps_pgm;
            s.PushUd(bnd, push);
            for (const auto& desc : info->buffers) {
                if (bi >= Shader::NUM_BUFFERS) {
                    ok = false;
                    why = 4;
                    break;
                }
                if (b.guest_mask & (u64{1} << bi)) {
                    const auto vsharp = desc.GetSharp(s);
                    if (vsharp.base_address == 0 || vsharp.GetSize() == 0) {
                        ok = false;
                        why = 4;
                        break;
                    }
                    const u64 size =
                        memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
                    if (desc.is_formatted) {
                        // Formatted reads (e.g. the 2.4 GB window): only the opener's own V#.
                        if (vsharp.base_address != o_addr[bi] || size != o_size[bi]) {
                            ok = false;
                            why = 5;
                            break;
                        }
                        m.addr[bi] = 0;
                    } else {
                        m.addr[bi] = vsharp.base_address;
                        m.size[bi] = static_cast<u32>(std::min<u64>(size, 0xffffffffull));
                    }
                }
                ++bi;
            }
            if (ok && b.colour) {
                for (const auto& desc : info->images) {
                    ok = ok && add_key(desc.GetSharp(s));
                }
                for (const auto& desc : info->samplers) {
                    ok = ok && add_key(desc.GetSharp(s));
                }
                if (!ok) {
                    why = 7;
                }
            }
        }
        if (!ok) {
            break;
        }
        lap(5); // stages: user data push, V# reads from the payload flat buffers, image key
        if (b.colour &&
            (kn != b.img_n || std::memcmp(key.data(), native_img_key.data(), kn * 4) != 0)) {
            why = 7; // the material changed: the next batch starts here
            break;
        }
        std::memcpy(m.ud.data(), push.ud_regs.data(), sizeof(m.ud));
        const auto [vo, io] = GetDrawOffsets(regs, bs[vs_i], base->GetFetchShader());
        if (io != 0) {
            why = 6;
            break;
        }
        m.vertex_offset = vo;
        const auto* hdr = reinterpret_cast<const AmdGpu::CpSplit::VertexInputsPayload*>(
            pl + (u64(nvs + nps) * 4 + 7) / 8 * 8);
        if (hdr->count != b.vb_n || hdr->count > 16) {
            why = 6;
            break;
        }
        const auto* att = reinterpret_cast<const AmdGpu::CpSplit::VertexInputAttrib*>(hdr + 1);
        for (u32 a = 0; a < hdr->count; ++a) {
            const bool null = att[a].range == AmdGpu::CpSplit::VertexInputNoRange;
            if (null != (b.vb[a] == 0) || att[a].desc.stride != b.vb_stride[a]) {
                ok = false;
                break;
            }
            m.vaddr[a] = null ? 0 : att[a].base_address;
            m.vsize[a] = null ? 0 : att[a].size;
            if (m.vaddr[a] != 0 && m.vsize[a] == 0) {
                ok = false; // a stream with an address but no size: not prepared
                break;
            }
        }
        if (!ok) {
            why = 6;
            break;
        }
        lap(6); // vertex block (payload)
        m.iaddr = index_addr;
        m.icount = index_count;
        m.instances = instances;
        m.cmd = cmd;
        for (u32 bi = 0; ok && bi < Shader::NUM_BUFFERS; ++bi) {
            if ((b.guest_mask & (u64{1} << bi)) && m.addr[bi] != 0) {
                slot_init(bi);
                ok = inside(slot_b[bi], m.addr[bi], m.size[bi], b.buf[bi]) &&
                     slot_u[bi].Fits(m.addr[bi], m.size[bi]);
            }
        }
        for (u32 a = 0; ok && a < b.vb_n; ++a) {
            if (m.vaddr[a] != 0 && m.vsize[a] != 0) {
                vb_init(a);
                ok = inside(vb_b[a], m.vaddr[a], m.vsize[a], b.vb[a]) &&
                     vb_u[a].Fits(m.vaddr[a], m.vsize[a]);
            }
        }
        const u64 ibytes = u64(index_count) * index_size;
        ok = ok && inside(ib_b, index_addr, ibytes, b.ib) && ib_u.Fits(index_addr, ibytes);
        if (!ok) {
            why = 9; // a range outside the batch's buffer: the next batch starts here
            break;
        }
        for (u32 bi = 0; bi < Shader::NUM_BUFFERS; ++bi) {
            if ((b.guest_mask & (u64{1} << bi)) && m.addr[bi] != 0) {
                slot_u[bi].Add(m.addr[bi], m.size[bi]);
            }
        }
        for (u32 a = 0; a < b.vb_n; ++a) {
            if (m.vaddr[a] != 0 && m.vsize[a] != 0) {
                vb_u[a].Add(m.vaddr[a], m.vsize[a]);
            }
        }
        ib_u.Add(index_addr, ibytes);
        lap(7); // range checks against the bounds
        n = j + 1;
    }
    st.prep_looked += n;
    lap(1); // per-member compute from the job records
    if (n < want) {
        ++st.prep_stop[why];
    }
    // The payload is intact only if the job thread has not claimed past the first member's.
    std::atomic_thread_fence(std::memory_order_acquire);
    if (n != 0 &&
        NR::g_payload_claim[slot].load(std::memory_order_relaxed) - first_pos > NR::PayloadSize) {
        n = 0;
        ++st.prep_stop[8];
    }
    if (n == 0) {
        st.c_prep += __rdtsc() - t0;
        return;
    }
    // Snapshot BEFORE the spans sync: a CPU write after the sync moves the epoch past it.
    const u64 epoch = VideoCore::g_dirty_epoch.load(std::memory_order_acquire);
    const u64 gen = buffer_cache.StructGeneration();
    struct Span {
        VAddr lo{~VAddr{0}}, hi{};
        void Add(VAddr a, u64 size) {
            lo = std::min(lo, a);
            hi = std::max<VAddr>(hi, a + size);
        }
    };
    // One span per distinct (buffer, access) over the kept members (heap slots share one buffer),
    // each synced once through NativeRunSpan; the ranges were checked against the bounds above.
    struct USpan {
        u64 handle;
        VideoCore::NativeRunAccess access;
        VAddr lo, hi;
    };
    std::array<USpan, Shader::NUM_BUFFERS + 17> us;
    u32 nus = 0;
    bool spans_ok = true;
    const auto add_span = [&](u64 handle, VideoCore::NativeRunAccess access, VAddr a, u64 size) {
        for (u32 i = 0; i < nus; ++i) {
            // Merge only while the union stays within NativeRunSpan's 64 MB (s400: merging every
            // slot of one buffer failed 1.3 M prepares).
            if (us[i].handle == handle && us[i].access == access &&
                std::max<VAddr>(us[i].hi, a + size) - std::min(us[i].lo, a) <= (u64{64} << 20)) {
                us[i].lo = std::min(us[i].lo, a);
                us[i].hi = std::max<VAddr>(us[i].hi, a + size);
                return;
            }
        }
        if (nus == us.size()) {
            spans_ok = false;
            return;
        }
        us[nus] = {handle, access, a, a + size};
        ++nus;
    };
    // Colour batches read vertex data as storage (vertex pulling).
    const auto vaccess =
        b.colour ? VideoCore::NativeRunAccess::Storage : VideoCore::NativeRunAccess::Vertex;
    for (u32 j = 0; j < n; ++j) {
        const auto& m = t_prep[j];
        for (u32 bi = 0; bi < Shader::NUM_BUFFERS; ++bi) {
            if ((b.guest_mask & (u64{1} << bi)) && m.addr[bi] != 0) {
                add_span(b.buf[bi], VideoCore::NativeRunAccess::Storage, m.addr[bi], m.size[bi]);
            }
        }
        for (u32 a = 0; a < b.vb_n; ++a) {
            if (m.vaddr[a] != 0 && m.vsize[a] != 0) {
                add_span(b.vb[a], vaccess, m.vaddr[a], m.vsize[a]);
            }
        }
        add_span(b.ib, VideoCore::NativeRunAccess::Index, m.iaddr, u64(m.icount) * index_size);
    }
    for (u32 i = 0; spans_ok && i < nus; ++i) {
        const VideoCore::Buffer* buf = buffer_cache.NativeRunSpan(
            us[i].lo, us[i].hi, us[i].access, &buffer_barriers, cur_dispatch_run);
        spans_ok = buf && std::bit_cast<u64>(static_cast<VkBuffer>(buf->Handle())) == us[i].handle;
    }
    if (!spans_ok) {
        ++st.prep_stop[11];
    }
    st.prep_spans += nus;
    lap(2); // spans
    if (!spans_ok) {
        st.c_prep += __rdtsc() - t0;
        return;
    }
    // Final records and commands, as NativeJoin writes them.
    u32 done = 0;
    for (u32 j = 0; j < n; ++j) {
        auto& m = t_prep[j];
        m.rec.fill(0);
        std::memcpy(m.rec.data(), m.ud.data(), sizeof(m.ud));
        bool ok = true;
        for (u32 bi = 0; bi < Shader::NUM_BUFFERS; ++bi) {
            if (!(b.guest_mask & (u64{1} << bi))) {
                continue;
            }
            u64 off;
            u32 size;
            if (m.addr[bi] == 0) {
                off = o_off[bi];
                size = o_rsize[bi];
            } else {
                off = m.addr[bi] - slot_b[bi].lo;
                size = m.size[bi];
            }
            if (off > 0xffffffffull) {
                ok = false;
                break;
            }
            m.rec[Shader::NativeRecordOffOffset + bi] = static_cast<u32>(off);
            m.rec[Shader::NativeRecordSizeOffset + bi] = size;
        }
        s64 q = 0;
        bool q_set = false;
        for (u32 a = 0; ok && a < b.vb_n; ++a) {
            if (m.vaddr[a] == 0) {
                continue;
            }
            const u64 off = m.vaddr[a] - vb_b[a].lo;
            if (b.colour) {
                const u32 loc = attribs[a].desc.location;
                if (loc >= Shader::NativeMaxPulledLocations || off > 0xffffffffull) {
                    ok = false;
                    break;
                }
                m.rec[Shader::NativeRecordVoffOffset + loc] = static_cast<u32>(off);
                m.rec[Shader::NativeRecordVsizeOffset + loc] = 0xffffffffu;
                continue;
            }
            const s64 d = s64(off) - s64(b.vb_off[a]);
            if (attribs[a].desc.instanced) {
                // Opus #3: vertexOffset cannot rebase an instance-rate binding.
                if (d != 0) {
                    ok = false;
                }
                continue;
            }
            const u32 stride = b.vb_stride[a];
            if (stride == 0 ? d != 0 : d % s64(stride) != 0) {
                ok = false;
                break;
            }
            if (stride != 0) {
                const s64 qa = d / s64(stride);
                if (q_set && qa != q) {
                    ok = false;
                    break;
                }
                q = qa;
                q_set = true;
            }
        }
        const u64 ib_off = m.iaddr - ib_b.lo;
        const s64 vo = s64(s32(m.vertex_offset)) + q;
        if (!ok || ib_off % index_size != 0 || vo < 0 || vo > INT32_MAX) {
            ++st.prep_stop[10];
            break;
        }
        m.cmdw = {m.icount, m.instances, static_cast<u32>(ib_off / index_size),
                  static_cast<u32>(s32(vo)), 0u};
        done = j + 1;
    }
    lap(3); // final records and commands
    t_prep_n = done;
    native_prep.epoch = epoch;
    native_prep.gen = gen;
    native_prep.first_idx = head + run_k0 + 1;
    st.c_prep += __rdtsc() - t0;
}

void Rasterizer::NativeGpuVerify(const u32* ud, const std::array<u32, Shader::NUM_BUFFERS>& off,
                                 const std::array<u32, Shader::NUM_BUFFERS>& size,
                                 const std::array<u64, Shader::NUM_BUFFERS>& cpu,
                                 const u32* voff, const AmdGpu::CpSplit::VertexInputAttrib* attribs,
                                 const u32* cmd, u32 vertex_offset, u32 instance_offset) {
    // SHADPS4_NATIVE_GPU_DIRECT=2 (sec. 10, stage G1): the job thread's GPU record and run union
    // of the draw a join / open is writing now, against what it writes. Back thread.
    namespace NR = AmdGpu::NativeRec;
    auto& st = native_stats;
    auto& b = native_batch;
    if (draw_job_tag == 0) {
        return;
    }
    const u32 slot = draw_job_tag >> 28;
    const NR::Record* ring = NR::g_rings[slot].load(std::memory_order_acquire);
    const u32* gpu = NR::g_gpu_recs[slot].load(std::memory_order_acquire);
    const NR::RunUnion* unions = NR::g_unions[slot].load(std::memory_order_acquire);
    if (!ring || !gpu || !unions) {
        ++st.gv[0];
        return;
    }
    const u32 idx = draw_job_idx;
    const NR::Record& r = ring[idx & (NR::RingSize - 1)];
    const u32 gpu_ok = r.gpu_ok, union_idx = r.union_idx, vo_ud = r.vo_ud, io_ud = r.io_ud;
    const u32 run_k = r.run_k, head_idx = r.head_idx, vcount = r.vcount;
    const u32 index_count = r.index_count, instances = r.instances, inst_known = r.inst_known;
    const u64 index_addr = r.index_addr;
    const auto vaddr = r.vaddr;
    const u32 push_inh = r.push_inh;
    const u64 inh_slots = r.inh_slots;
    const bool cmd_ok = static_cast<const void*>(r.cmd) == draw_job_cmd;
    std::array<u32, NR::GpuRecDwords> g;
    std::memcpy(g.data(), gpu + size_t(idx & (NR::RingSize - 1)) * NR::GpuRecDwords, sizeof(g));
    std::atomic_thread_fence(std::memory_order_acquire);
    if (r.stamp.load(std::memory_order_relaxed) != idx + 1 || !cmd_ok) {
        ++st.gv[0];
        return;
    }
    if (gpu_ok != 1) { // q18 B: 2 = A2 translator only
        ++st.gv[1];
        return;
    }
    std::string why;
    // Inherited registers / binding slots (sec. 11) come from the batch opener in ring mode:
    // stale in the job record on purpose, not compared here.
    for (u32 i = 0; i < Shader::NUM_USER_DATA_REGS && why.empty(); ++i) {
        if (!((push_inh >> i) & 1) && g[i] != ud[i]) {
            why = "user data";
        }
    }
    for (u32 bi = 0; why.empty() && bi < Shader::NUM_BUFFERS; ++bi) {
        if (!(b.guest_mask & (u64{1} << bi)) || (inh_slots & (u64{1} << bi))) {
            continue;
        }
        const u32 want = static_cast<u32>(g[Shader::NativeRecordOffOffset + bi] -
                                          static_cast<u32>(cpu[bi]));
        if (want != off[bi] || g[Shader::NativeRecordSizeOffset + bi] != size[bi]) {
            why = fmt::format("buffer {}: job {:#x}-{:#x} = {:#x} size {:#x}, join {:#x} size {:#x}",
                              bi, g[Shader::NativeRecordOffOffset + bi], cpu[bi], want,
                              g[Shader::NativeRecordSizeOffset + bi], off[bi], size[bi]);
        }
    }
    if (why.empty() && voff && attribs) {
        for (u32 a = 0; a < vcount && why.empty(); ++a) {
            if (vaddr[a] == 0) {
                continue;
            }
            const u32 loc = attribs[a].desc.location;
            const auto* vb = buffer_cache.NativeBufferAt(vaddr[a], 0, 27);
            if (loc >= Shader::NativeMaxPulledLocations || !vb) {
                continue;
            }
            const u64 want = vaddr[a] - vb->CpuAddr();
            if (g[Shader::NativeRecordVoffOffset + loc] != static_cast<u32>(vaddr[a]) ||
                want != voff[loc]) {
                why = fmt::format("vertex location {}: job {:#x} join {:#x}", loc, want, voff[loc]);
            }
        }
    }
    if (why.empty()) {
        const auto* ib = buffer_cache.NativeBufferAt(index_addr, 0, 28);
        const u32 want_first =
            ib ? static_cast<u32>((index_addr - ib->CpuAddr()) / b.index_size) : ~0u;
        const auto& regs = liverpool->BackRegs();
        const u32 want_vo = regs.index_offset != 0 ? regs.index_offset : vo_ud;
        if (index_count != cmd[0] || (inst_known && instances != cmd[1]) ||
            want_first != cmd[2] || want_vo != vertex_offset || io_ud != instance_offset) {
            why = fmt::format("command: job count {} inst {} first {} vo {} io {}, join {} {} {} "
                              "{} {}",
                              index_count, instances, want_first, want_vo, io_ud, cmd[0], cmd[1],
                              cmd[2], vertex_offset, instance_offset);
        }
    }
    // The run union must cover this draw (k < n, every range inside).
    const NR::RunUnion& u = unions[union_idx & NR::UnionMask];
    const u32 ustamp = u.stamp.load(std::memory_order_acquire);
    const u32 un = u.n.load(std::memory_order_acquire);
    bool union_ok = ustamp == head_idx + 1 && run_k < un;
    if (!union_ok) {
        ++st.gv[2]; // union gone (overwritten / killed) or not covering yet
    } else if (why.empty()) {
        for (u32 bi = 0; bi < Shader::NUM_BUFFERS; ++bi) {
            if (!(b.guest_mask & (u64{1} << bi)) || (inh_slots & (u64{1} << bi))) {
                continue;
            }
            const u64 full = cpu[bi] + off[bi];
            if (full < u.lo[bi] || full + size[bi] > u.hi[bi]) {
                why = fmt::format("union buffer {}: {:#x}+{:#x} outside {:#x}-{:#x}", bi, full,
                                  size[bi], u.lo[bi], u.hi[bi]);
                break;
            }
        }
        for (u32 a = 0; why.empty() && a < vcount; ++a) {
            if (vaddr[a] != 0 && (vaddr[a] < u.vlo[a] || vaddr[a] >= u.vhi[a])) {
                why = fmt::format("union vertex {}", a);
            }
        }
        if (why.empty() && (index_addr < u.ilo ||
                            index_addr + u64(index_count) * b.index_size > u.ihi)) {
            why = "union index";
        }
    }
    if (why.empty()) {
        ++st.gv[3];
    } else if (++st.gv[4] <= 16) {
        LOG_CRITICAL(Render_Vulkan, "NativeGpu verify MISMATCH ({}, k {}): {}",
                     b.colour ? "colour" : "shadow", run_k, why);
    }
}

// SHADPS4_NATIVE_GPU_DIRECT=1: the job threads' GPU record rings (mapped Vulkan buffers, one per
// job-thread slot, allocated on the job thread's first derived draw through
// AmdGpu::NativeRec::g_gpu_alloc; never freed - they live as long as the process).
namespace {
Rasterizer* g_ring_owner = nullptr;
std::mutex g_ring_mutex;
std::array<VideoCore::StreamBuffer*, AmdGpu::NativeRec::MaxSlots> g_ring_bufs{};
} // namespace
// SHADPS4_NATIVE_SEG: the shaders' ring mask and pseudo-slot mirror the job ring's constants.
static_assert(AmdGpu::NativeRec::RingSize - 1 == Shader::NativeSegRingMask);
static_assert(AmdGpu::NativeRec::MaxSlots == Shader::NativeSegLegacySlot);

void Rasterizer::NativeGpuRingInit(Rasterizer* owner) {
    std::scoped_lock lk{g_ring_mutex};
    g_ring_owner = owner;
    AmdGpu::NativeRec::g_gpu_alloc.store(&Rasterizer::NativeGpuRingAlloc, std::memory_order_release);
}

u32* Rasterizer::NativeGpuRingAlloc(u32 slot) {
    namespace NR = AmdGpu::NativeRec;
    std::scoped_lock lk{g_ring_mutex};
    Rasterizer* r = g_ring_owner;
    if (!r || slot >= NR::MaxSlots) {
        return nullptr;
    }
    auto*& buf = g_ring_bufs[slot];
    if (!buf) {
        // eShaderDeviceAddress: SHADPS4_NATIVE_SEG reads the records by pointer.
        buf = new VideoCore::StreamBuffer(r->instance, r->scheduler, VideoCore::MemoryUsage::Stream,
                                          u64(NR::RingSize) * NR::GpuRecDwords * sizeof(u32),
                                          Shader::g_native_seg != 0
                                              ? vk::BufferUsageFlagBits::eShaderDeviceAddress
                                              : vk::BufferUsageFlags{});
    }
    if (buf->mapped_data.empty() || !buf->IsCoherent()) {
        LOG_WARNING(Render_Vulkan, "NativeGpu: ring of slot {} is not mapped coherent memory", slot);
        return nullptr;
    }
    if (Shader::g_native_seg != 0) {
        NR::g_gpu_bda[slot].store(buf->BufferDeviceAddress(), std::memory_order_release);
    }
    NR::g_gpu_handles[slot].store(std::bit_cast<u64>(static_cast<VkBuffer>(buf->Handle())),
                                  std::memory_order_release);
    LOG_INFO(Render_Vulkan, "NativeGpu: record ring of job slot {} ({} MB) allocated", slot,
             (u64(NR::RingSize) * NR::GpuRecDwords * 4) >> 20);
    return reinterpret_cast<u32*>(buf->mapped_data.data());
}

bool Rasterizer::NativeRingPrepare(const GraphicsPipeline* base,
                                   const AmdGpu::CpSplit::VertexInputAttrib* attribs,
                                   const std::array<u64, Shader::NUM_BUFFERS>& o_addr,
                                   const std::array<u32, Shader::NUM_BUFFERS>& o_off,
                                   const std::array<u32, Shader::NUM_BUFFERS>& o_size,
                                   const u32* o_ud) {
    // Back thread, NativeOpen of a live batch, before the stream reservation. Ring mode: the
    // multi-draw reads every draw's record from the opener's job-thread ring (entries idx ..
    // idx + n); the back validates the run's binding unions once (the job thread kept them),
    // fills the table (ring index, mask, per-binding / per-location buffer bases) and writes the
    // members' indirect commands; their records it never touches. False: the batch stays in
    // the legacy mode (the back's own records, joins).
    namespace NR = AmdGpu::NativeRec;
    auto& b = native_batch;
    auto& st = native_stats;
    t_prep_n = 0;
    const u64 t0 = __rdtsc();
    const auto fail = [&](u32 why) {
        ++st.ring_fail[std::min<u32>(why, 15)];
        st.c_ring += __rdtsc() - t0;
        return false;
    };
    if (draw_job_tag == 0) {
        return fail(0);
    }
    const u32 slot = draw_job_tag >> 28;
    const NR::Record* ring = NR::g_rings[slot].load(std::memory_order_acquire);
    const NR::RunUnion* unions = NR::g_unions[slot].load(std::memory_order_acquire);
    const u64 ring_handle = NR::g_gpu_handles[slot].load(std::memory_order_acquire);
    const bool ring_verify = NR::g_gpu_mode == 3;
    const u32* gpu_cpu = ring_verify ? NR::g_gpu_recs[slot].load(std::memory_order_acquire) : nullptr;
    if (!ring || !unions || (ring_handle == 0 && !ring_verify) || (ring_verify && !gpu_cpu)) {
        return fail(1);
    }
    constexpr u32 mask = NR::RingSize - 1;
    const u32 idx = draw_job_idx;
    // Ring lifetime: the GPU reads entries idx .. idx + 63 when it executes the batch; the job
    // thread must not have written more than a quarter ring past them by now (it would need
    // another ~12 k draws, ~7 frames of one job thread, before the GPU ran the batch).
    if (NR::g_job_next[slot].load(std::memory_order_relaxed) - idx > NR::LifetimeWindow()) {
        return fail(2);
    }
    const NR::Record& r0 = ring[idx & mask];
    const u32 run_k0 = r0.run_k, head_idx0 = r0.head_idx, gpu_ok0 = r0.gpu_ok;
    const u32 union0 = r0.union_idx;
    const void* cmd0 = r0.cmd;
    const u32 img_n0 = r0.img_n;
    const u64 img_h10 = r0.img_h1, img_h20 = r0.img_h2;
    const u32 push_inh0 = r0.push_inh, inh_vs0 = r0.inh_vs, inh_ps0 = r0.inh_ps;
    const u64 inh_slots0 = r0.inh_slots;
    std::atomic_thread_fence(std::memory_order_acquire);
    if (r0.stamp.load(std::memory_order_relaxed) != idx + 1 || cmd0 != draw_job_cmd || gpu_ok0 != 1) {
        return fail(3);
    }
    // Colour: every member must sample the opener's images and samplers (raw T# / S# words, as
    // the join's NativeImageKey); the job record carries their hashes. The opener's own record
    // must agree with the back's key - unless a T# / S# sits in inherited registers (ImgInhBit:
    // zeros in the job key); members are compared with the opener's job key either way and the
    // inherited registers with the opener's at every confirm (sec. 11).
    const u64 key_h1 = img_h10, key_h2 = img_h20;
    if (b.colour) {
        if ((img_n0 & ~NR::ImgInhBit) != b.img_n) {
            return fail(13);
        }
        if (!(img_n0 & NR::ImgInhBit)) {
            u64 h1 = 0, h2 = 0;
            NR::ImageKeyHash(native_img_key.data(), b.img_n, h1, h2);
            if (h1 != img_h10 || h2 != img_h20) {
                return fail(13);
            }
        }
    }
    const u32 head = run_k0 == 0 ? idx : head_idx0;
    const NR::RunUnion& u = unions[union0 & NR::UnionMask];
    const u32 un = u.n.load(std::memory_order_acquire);
    if (u.stamp.load(std::memory_order_relaxed) != head + 1 || un <= run_k0 + 1) {
        return fail(4); // no member covered by the union yet
    }
    // Only the batch's Guest slots and streams are copied (s408: the whole union, ~1.2 KB of
    // another core's cache lines, cost ~670 cycles per batch).
    const u64 guest_mask = u.guest_mask, eq_mask = u.eq_mask;
    std::array<u64, Shader::NUM_BUFFERS> lo, hi, first_addr;
    for (u64 m = b.guest_mask; m != 0; m &= m - 1) {
        const u32 bi = static_cast<u32>(std::countr_zero(m));
        lo[bi] = u.lo[bi];
        hi[bi] = u.hi[bi];
        first_addr[bi] = u.first_addr[bi];
    }
    std::array<u64, 16> vlo, vhi;
    for (u32 a = 0; a < b.vb_n; ++a) {
        vlo[a] = u.vlo[a];
        vhi[a] = u.vhi[a];
    }
    const u64 ilo = u.ilo, ihi = u.ihi;
    std::atomic_thread_fence(std::memory_order_acquire);
    if (u.stamp.load(std::memory_order_relaxed) != head + 1) {
        return fail(4);
    }
    const u32 want = std::min<u32>(un - run_k0 - 1, NativeMaxDraws - 1);
    if (t_prep.size() < NativeMaxDraws) {
        t_prep.resize(NativeMaxDraws);
    }
    u64 tl = __rdtsc();
    const auto lap = [&](u32 ph) {
        const u64 now = __rdtsc();
        st.c_ring_ph[ph] += now - tl;
        tl = now;
    };
    lap(0); // records, union copy
    // Snapshot BEFORE the span syncs (a CPU write after a sync moves the epoch past it).
    const u64 epoch = VideoCore::g_dirty_epoch.load(std::memory_order_acquire);
    const u64 gen = buffer_cache.StructGeneration();
    // Buffer bases and spans. A buffer is found by the page table at the union's low end and
    // must be the batch's buffer for that binding and hold the whole union.
    struct USpan {
        u64 handle;
        VideoCore::NativeRunAccess access;
        VAddr lo, hi;
    };
    std::array<USpan, Shader::NUM_BUFFERS + 17> us;
    u32 nus = 0;
    const u64 bgen = buffer_cache.StructGeneration();
    if (native_bcache_gen != bgen) {
        native_bcache_gen = bgen;
        native_bcache_n = 0;
    }
    const auto base_of = [&](VAddr a_lo, VAddr a_hi, u64 want_handle, VAddr& cpu) {
        if (a_lo == 0 || a_hi <= a_lo || a_hi - a_lo > (u64{64} << 20)) {
            return false;
        }
        // Bounds by VkBuffer, valid while the buffer structure generation stays put.
        for (u32 i = 0; i < native_bcache_n; ++i) {
            const auto& e = native_bcache[i];
            if (e.handle == want_handle && a_lo >= e.lo && a_hi <= e.hi) {
                cpu = e.lo;
                return true;
            }
        }
        const VideoCore::Buffer* buf = buffer_cache.NativeBufferAt(a_lo, a_hi - a_lo);
        if (!buf || !buf->IsInBounds(a_lo, a_hi - a_lo) ||
            std::bit_cast<u64>(static_cast<VkBuffer>(buf->Handle())) != want_handle) {
            return false;
        }
        cpu = buf->CpuAddr();
        auto& e = native_bcache[native_bcache_n < native_bcache.size()
                                    ? native_bcache_n++
                                    : (native_bcache_rr++ % native_bcache.size())];
        e = {want_handle, cpu, cpu + buf->SizeBytes()};
        return true;
    };
    const auto add_span = [&](u64 handle, VideoCore::NativeRunAccess access, VAddr a_lo,
                              VAddr a_hi) {
        for (u32 i = 0; i < nus; ++i) {
            if (us[i].handle == handle && us[i].access == access &&
                std::max(us[i].hi, a_hi) - std::min(us[i].lo, a_lo) <= (u64{64} << 20)) {
                us[i].lo = std::min(us[i].lo, a_lo);
                us[i].hi = std::max(us[i].hi, a_hi);
                return true;
            }
        }
        if (nus == us.size()) {
            return false;
        }
        us[nus++] = {handle, access, a_lo, a_hi};
        return true;
    };
    auto& table = native_ring_table;
    table.fill(0);
    table[Shader::NativeTableFirst] = idx & mask;
    table[Shader::NativeTableMask] = mask;
    if (native_pt_on) {
        // SHADPS4_NATIVE_PT: where a page-table miss points (always mapped memory).
        const u64 fb = buffer_cache.GetBdaPageTableBuffer()->BufferDeviceAddress();
        table[Shader::NativeTablePtFallback] = static_cast<u32>(fb);
        table[Shader::NativeTablePtFallback + 1] = static_cast<u32>(fb >> 32);
        const u64 fa = buffer_cache.GetFaultBuffer()->BufferDeviceAddress();
        table[Shader::NativeTableFaultAddr] = static_cast<u32>(fa);
        table[Shader::NativeTableFaultAddr + 1] = static_cast<u32>(fa >> 32);
    }
    // Coverage (sec. 11): inherited user-data registers - every member's equal the opener's
    // (checked at each confirm); the shaders take them from here.
    table[Shader::NativeTablePushInh] = push_inh0;
    for (u32 pm = push_inh0; pm != 0; pm &= pm - 1) {
        const u32 pi = static_cast<u32>(std::countr_zero(pm));
        if (pi >= Shader::NUM_USER_DATA_REGS) {
            return fail(14);
        }
        table[Shader::NativeTableUd + pi] = o_ud[pi];
    }
    for (u32 bi = 0; bi < Shader::NUM_BUFFERS; ++bi) {
        const u64 bit = u64{1} << bi;
        if (!(b.guest_mask & bit)) {
            continue;
        }
        if (inh_slots0 & bit) {
            // The V# sits in inherited registers: every member binds the opener's range (the
            // opener's bind synced it). The record word is 0, so the base makes the offset the
            // opener's; the size comes from the table too.
            table[Shader::NativeTableBase + bi] = 0u - o_off[bi];
            table[Shader::NativeTableSize + bi] = o_size[bi];
            table[Shader::NativeTableInhSlots + bi / 32] |= 1u << (bi % 32);
            if (native_pt_on) {
                // SHADPS4_NATIVE_PT: the job record's words are stale here; the opener's guest
                // address comes from the table.
                table[Shader::NativeTablePtLo + bi] = static_cast<u32>(o_addr[bi]);
                table[Shader::NativeTablePtHi + bi / 4] |=
                    static_cast<u32>((o_addr[bi] >> 32) & 0xffu) << ((bi % 4) * 8);
            }
            continue;
        }
        if (!(guest_mask & bit)) {
            return fail(5);
        }
        VAddr cpu = 0;
        if ((eq_mask & bit) && first_addr[bi] == o_addr[bi]) {
            // Every draw of the run binds the opener's own V# here (e.g. the 2.4 GB window): the
            // opener's generic bind synced it; no span.
            const VideoCore::Buffer* buf = buffer_cache.NativeBufferAt(o_addr[bi], 0, 29);
            if (!buf || std::bit_cast<u64>(static_cast<VkBuffer>(buf->Handle())) != b.buf[bi]) {
                return fail(5);
            }
            cpu = buf->CpuAddr();
        } else if (!base_of(lo[bi], hi[bi], b.buf[bi], cpu) ||
                   !add_span(b.buf[bi], VideoCore::NativeRunAccess::Storage, lo[bi], hi[bi])) {
            return fail(6);
        }
        table[Shader::NativeTableBase + bi] = static_cast<u32>(cpu);
    }
    std::array<VAddr, 16> vcpu{};
    const auto vaccess =
        b.colour ? VideoCore::NativeRunAccess::Storage : VideoCore::NativeRunAccess::Vertex;
    for (u32 a = 0; a < b.vb_n; ++a) {
        if (b.vb[a] == 0) {
            if (vhi[a] > vlo[a]) {
                return fail(7); // a member has a stream where the opener has none
            }
            continue;
        }
        if (!base_of(vlo[a], vhi[a], b.vb[a], vcpu[a]) || !add_span(b.vb[a], vaccess, vlo[a], vhi[a])) {
            return fail(7);
        }
        if (b.colour) {
            const u32 loc = attribs[a].desc.location;
            if (loc >= Shader::NativeMaxPulledLocations) {
                return fail(7);
            }
            table[Shader::NativeTableVbase + loc] = static_cast<u32>(vcpu[a]);
        }
    }
    VAddr icpu = 0;
    if (!base_of(ilo, ihi, b.ib, icpu) ||
        !add_span(b.ib, VideoCore::NativeRunAccess::Index, ilo, ihi)) {
        return fail(8);
    }
    lap(1); // bases, span list
    // The members' indirect commands (their records are the job thread's).
    const auto& regs = liverpool->BackRegs();
    const u32 index_size = b.index_size;
    u32 n = 0;
    u32 why = 0;
    for (u32 j = 0; j < want; ++j) {
        const u32 k = run_k0 + 1 + j;
        const u32 mi = head + k;
        const NR::Record& r = ring[mi & mask];
        const u32 run_k = r.run_k, head_idx = r.head_idx, gpu_ok = r.gpu_ok;
        const u32 uidx = r.union_idx, inst_known = r.inst_known, instances = r.instances;
        const u32 index_count = r.index_count, vo_ud = r.vo_ud, io_ud = r.io_ud;
        const u64 index_addr = r.index_addr;
        const auto vaddr = r.vaddr;
        const void* cmd = r.cmd;
        const u32 img_n = r.img_n;
        const u64 img_h1 = r.img_h1, img_h2 = r.img_h2;
        const u32 push_inh = r.push_inh, inh_vs = r.inh_vs, inh_ps = r.inh_ps;
        const u64 inh_slots = r.inh_slots;
        std::atomic_thread_fence(std::memory_order_acquire);
        if (r.stamp.load(std::memory_order_relaxed) != mi + 1 || run_k != k || head_idx != head ||
            gpu_ok != 1 || uidx != union0 || !inst_known || io_ud != 0 || index_count == 0) {
            why = 10;
            break;
        }
        if (push_inh != push_inh0 || inh_slots != inh_slots0 || inh_vs != inh_vs0 ||
            inh_ps != inh_ps0) {
            why = 14; // another inherited set (the job thread ends runs there too)
            break;
        }
        if (b.colour && (img_n != img_n0 || img_h1 != key_h1 || img_h2 != key_h2)) {
            why = 13; // the material changed: the batch ends here
            break;
        }
        const u64 ib_off = index_addr - icpu;
        if (index_addr < ilo || ib_off % index_size != 0) {
            why = 11;
            break;
        }
        s64 q = 0;
        bool ok = true;
        bool q_set = false;
        for (u32 a = 0; !b.colour && a < b.vb_n; ++a) {
            if (b.vb[a] == 0 || vaddr[a] == 0) {
                ok = ok && (b.vb[a] == 0) == (vaddr[a] == 0);
                continue;
            }
            const s64 d = s64(vaddr[a] - vcpu[a]) - s64(b.vb_off[a]);
            if (attribs[a].desc.instanced) {
                ok = ok && d == 0;
                continue;
            }
            const u32 stride = b.vb_stride[a];
            if (stride == 0 ? d != 0 : d % s64(stride) != 0) {
                ok = false;
                break;
            }
            if (stride != 0) {
                const s64 qa = d / s64(stride);
                if (q_set && qa != q) {
                    ok = false;
                    break;
                }
                q = qa;
                q_set = true;
            }
        }
        const u32 base_vo = regs.index_offset != 0 ? regs.index_offset : vo_ud;
        const s64 vo = s64(s32(base_vo)) + q;
        if (!ok || vo < 0 || vo > INT32_MAX) {
            why = 12;
            break;
        }
        auto& m = t_prep[j];
        m.cmdw = {index_count, instances, static_cast<u32>(ib_off / index_size),
                  static_cast<u32>(s32(vo)), 0u};
        m.cmd = cmd;
        m.iaddr = index_addr;
        if (ring_verify) {
            // The record the GPU would compute: job words minus the table's bases; inherited
            // registers and binding sizes from the table (what the ring shaders select).
            const u32* g = gpu_cpu + size_t(mi & mask) * NR::GpuRecDwords;
            m.rec.fill(0);
            std::memcpy(m.rec.data(), g, sizeof(u32) * Shader::NUM_USER_DATA_REGS);
            for (u32 pm = push_inh0; pm != 0; pm &= pm - 1) {
                const u32 pi = static_cast<u32>(std::countr_zero(pm));
                m.rec[Shader::NativeRecordUdOffset + pi] = table[Shader::NativeTableUd + pi];
            }
            for (u32 bi = 0; bi < Shader::NUM_BUFFERS; ++bi) {
                if (b.guest_mask & (u64{1} << bi)) {
                    m.rec[Shader::NativeRecordOffOffset + bi] =
                        g[Shader::NativeRecordOffOffset + bi] - table[Shader::NativeTableBase + bi];
                    m.rec[Shader::NativeRecordSizeOffset + bi] =
                        (inh_slots0 & (u64{1} << bi)) ? table[Shader::NativeTableSize + bi]
                                                      : g[Shader::NativeRecordSizeOffset + bi];
                }
            }
            for (u32 loc = 0; b.colour && loc < Shader::NativeMaxPulledLocations; ++loc) {
                if (g[Shader::NativeRecordVsizeOffset + loc] != 0) {
                    m.rec[Shader::NativeRecordVoffOffset + loc] =
                        g[Shader::NativeRecordVoffOffset + loc] - table[Shader::NativeTableVbase + loc];
                    m.rec[Shader::NativeRecordVsizeOffset + loc] =
                        g[Shader::NativeRecordVsizeOffset + loc];
                }
            }
        }
        n = j + 1;
    }
    lap(3); // member commands
    if (n < want) {
        ++st.ring_fail[why];
    }
    if (n == 0) {
        st.c_ring += __rdtsc() - t0;
        return false; // no member: legacy mode, before any span work
    }
    // Validity cache (s410: the global dirty epoch moved between most batches): a superset of
    // the range in the same buffer was validated at granule clock T, no 2 MB granule of the range
    // was marked CPU-dirty or GPU-written since T (dirty_epoch.h: every such marking bumps the
    // clock of its granules; the global epoch moves only together with it) and the buffer
    // structure is unchanged -> no sync walk, no GPU-modified walk (cached answer), the access
    // barrier as always.
    const u64 clock0 = VideoCore::g_granule_clock.load(std::memory_order_acquire);
    for (u32 i = 0; i < nus; ++i) {
        auto* hit = static_cast<NativeSpanEntry*>(nullptr);
        for (auto& e : native_span_cache) {
            if (e.handle == us[i].handle && e.access == static_cast<u32>(us[i].access) &&
                e.gen == gen && us[i].lo >= e.lo && us[i].hi <= e.hi &&
                VideoCore::GranulesUnchangedSince(us[i].lo, us[i].hi, e.epoch)) {
                hit = &e;
                break;
            }
        }
        int stale_kind = -1;
        if (hit && g_native_span_check) [[unlikely]] {
            // Opus G2 #1 compare mode: a hit must not cover CPU-dirty pages (no sync runs). On
            // one: classify, and take the real sync instead (protective).
            if (buffer_cache.IsRegionCpuModified(us[i].lo, us[i].hi - us[i].lo)) {
                stale_kind =
                    VideoCore::GranulesUnchangedSince(us[i].lo, us[i].hi, hit->epoch) ? 1 : 0;
                hit = nullptr;
            }
        }
        bool gpu_mod = false;
        const VideoCore::Buffer* buf = buffer_cache.NativeRunSpan(
            us[i].lo, us[i].hi, us[i].access, &buffer_barriers, cur_dispatch_run, hit == nullptr,
            hit ? (hit->gpu_mod ? 1 : 0) : -1, &gpu_mod);
        if (stale_kind == 1 && !buffer_cache.IsRegionCpuModified(us[i].lo, us[i].hi - us[i].lo)) {
            stale_kind = 2;
        }
        if (stale_kind >= 0) {
            ++st.ring_span_stale[stale_kind];
        }
        if (!buf || std::bit_cast<u64>(static_cast<VkBuffer>(buf->Handle())) != us[i].handle) {
            return fail(9);
        }
        st.ring_span_bytes += us[i].hi - us[i].lo;
        if (hit) {
            ++st.ring_span_hits;
        } else {
            // Grow the entry of this buffer if it is still current, else replace the oldest.
            NativeSpanEntry* slot = nullptr;
            for (auto& e : native_span_cache) {
                if (e.handle == us[i].handle && e.access == static_cast<u32>(us[i].access)) {
                    slot = &e;
                    break;
                }
            }
            // Grow while the old part is still unchanged since its clock (the grown entry keeps
            // the older clock: conservative), else replace.
            // Opus G2 #1: only an overlapping / touching range grows the entry (a merged gap was
            // never validated).
            if (slot && slot->gen == gen && us[i].lo <= slot->hi && us[i].hi >= slot->lo &&
                std::max(slot->hi, us[i].hi) - std::min(slot->lo, us[i].lo) <= (u64{64} << 20) &&
                VideoCore::GranulesUnchangedSince(slot->lo, slot->hi, slot->epoch)) {
                slot->lo = std::min(slot->lo, us[i].lo);
                slot->hi = std::max(slot->hi, us[i].hi);
                slot->gpu_mod = slot->gpu_mod || gpu_mod;
            } else {
                if (!slot) {
                    slot = &native_span_cache[native_span_rr++ % native_span_cache.size()];
                }
                *slot = {us[i].handle, static_cast<u32>(us[i].access), us[i].lo, us[i].hi, clock0,
                         gen, gpu_mod};
            }
        }
    }
    st.ring_spans += nus;
    lap(2); // span syncs
    t_prep_n = n;
    native_prep.epoch = epoch;
    native_prep.gen = gen;
    native_prep.first_idx = idx + 1;
    native_prep.inh_vs = inh_vs0 & 0xffffu;
    native_prep.inh_ps = inh_ps0 & 0xffffu;
    for (u32 i = 0; i < 16; ++i) {
        native_prep.inh_val_vs[i] = regs.vs_program.user_data[i];
        native_prep.inh_val_ps[i] = regs.ps_program.user_data[i];
    }
    st.ring_inh_runs += (inh_vs0 | inh_ps0) != 0;
    native_ring_handle = ring_handle;
    st.c_ring += __rdtsc() - t0;
    return true;
}

bool Rasterizer::NativeSegPrepare(const GraphicsPipeline* base,
                                  const AmdGpu::CpSplit::VertexInputAttrib* attribs,
                                  const std::array<u64, Shader::NUM_BUFFERS>& o_addr,
                                  const std::array<u32, Shader::NUM_BUFFERS>& o_off,
                                  const std::array<u32, Shader::NUM_BUFFERS>& o_size,
                                  const u32* o_ud) {
    // Back thread, NativeOpen of member 0 of a Segment record (SHADPS4_NATIVE_SEGX=1), in place
    // of NativeRingPrepare: the batch's bases and spans come from the record's unions (index,
    // pulled streams, the Guest slots' job-run unions), each found in ONE cached buffer (no
    // creation, no merge: a buffer made here would invalidate the opener's bindings) and synced
    // once - no member is read. Reason codes as NativeSegEvaluate (segx_result): 2 lifetime,
    // 3 index, 4 vertex, 5 vertex buffers, 6 Guest, 8 member 0's record / material / spans.
    namespace NR = AmdGpu::NativeRec;
    auto& b = native_batch;
    auto& st = native_stats;
    const auto& seg = *segx_pending;
    const auto fail = [&](u32 why) {
        segx_result = why;
        return false;
    };
    if (!b.colour || !native_seg_on || !native_pt_on || draw_job_tag == 0 || seg.n == 0 ||
        seg.n > NativeMaxDraws) {
        return fail(8);
    }
    const u32 slot0 = draw_job_tag >> 28;
    const NR::Record* ring = NR::g_rings[slot0].load(std::memory_order_acquire);
    const u64 ring_handle = NR::g_gpu_handles[slot0].load(std::memory_order_acquire);
    if (!ring || ring_handle == 0) {
        return fail(8);
    }
    constexpr u32 mask = NR::RingSize - 1;
    const u32 idx = draw_job_idx;
    const NR::Record& r0 = ring[idx & mask];
    const u32 gpu_ok0 = r0.gpu_ok;
    const void* cmd0 = r0.cmd;
    const u32 img_n0 = r0.img_n;
    const u64 img_h10 = r0.img_h1, img_h20 = r0.img_h2;
    const u32 push_inh0 = r0.push_inh;
    const u64 inh_slots0 = r0.inh_slots;
    std::atomic_thread_fence(std::memory_order_acquire);
    if (r0.stamp.load(std::memory_order_relaxed) != idx + 1 || cmd0 != draw_job_cmd || gpu_ok0 != 1) {
        return fail(8);
    }
    // Material 0: member 0's job key is the segment's first and agrees with the back's key.
    // v3 item 4: the further materials were resolved from their representatives' records
    // (NativeOpen, after the opener's resolve); their count must be the record's.
    if ((img_n0 & ~NR::ImgInhBit) != b.img_n || seg.n_mat == 0 || seg.n_mat != segx_n_mat ||
        img_h10 != seg.mat_h1[0] || img_h20 != seg.mat_h2[0]) {
        return fail(8);
    }
    if (!(img_n0 & NR::ImgInhBit)) {
        u64 h1 = 0, h2 = 0;
        NR::ImageKeyHash(native_img_key.data(), b.img_n, h1, h2);
        if (h1 != img_h10 || h2 != img_h20) {
            return fail(8);
        }
    }
    // Every member's job ring is alive and its entry inside the safe window.
    for (u32 j = 0; j < seg.n; ++j) {
        const u32 slot = (segx_list[j] >> 24) & Shader::NativeSegSlotMask;
        const u32 mi = segx_list[j] & 0xffffffu;
        if (slot >= NR::MaxSlots || NR::g_gpu_bda[slot].load(std::memory_order_acquire) == 0 ||
            ((NR::g_job_next[slot].load(std::memory_order_relaxed) - mi) & 0xffffffu) >
                NR::LifetimeWindow()) {
            return fail(2);
        }
    }
    const u64 gen = buffer_cache.StructGeneration();
    struct USpan {
        u64 handle;
        VideoCore::NativeRunAccess access;
        VAddr lo, hi;
    };
    std::array<USpan, Shader::NUM_BUFFERS + 17> us;
    u32 nus = 0;
    const auto base_of = [&](VAddr a_lo, VAddr a_hi, u64 want_handle, VAddr& cpu) {
        if (a_lo == 0 || a_hi <= a_lo || a_hi - a_lo > (u64{64} << 20)) {
            return false;
        }
        const VideoCore::Buffer* buf = buffer_cache.NativeBufferAt(a_lo, a_hi - a_lo);
        if (!buf || !buf->IsInBounds(a_lo, a_hi - a_lo) ||
            std::bit_cast<u64>(static_cast<VkBuffer>(buf->Handle())) != want_handle) {
            return false;
        }
        cpu = buf->CpuAddr();
        return true;
    };
    const auto add_span = [&](u64 handle, VideoCore::NativeRunAccess access, VAddr a_lo,
                              VAddr a_hi) {
        for (u32 i = 0; i < nus; ++i) {
            if (us[i].handle == handle && us[i].access == access &&
                std::max(us[i].hi, a_hi) - std::min(us[i].lo, a_lo) <= (u64{64} << 20)) {
                us[i].lo = std::min(us[i].lo, a_lo);
                us[i].hi = std::max(us[i].hi, a_hi);
                return true;
            }
        }
        if (nus == us.size()) {
            return false;
        }
        us[nus++] = {handle, access, a_lo, a_hi};
        return true;
    };
    auto& table = native_ring_table;
    table.fill(0);
    table[Shader::NativeTableFirst] = idx & mask;
    table[Shader::NativeTableMask] = mask;
    {
        const u64 fb = buffer_cache.GetBdaPageTableBuffer()->BufferDeviceAddress();
        table[Shader::NativeTablePtFallback] = static_cast<u32>(fb);
        table[Shader::NativeTablePtFallback + 1] = static_cast<u32>(fb >> 32);
        const u64 fa = buffer_cache.GetFaultBuffer()->BufferDeviceAddress();
        table[Shader::NativeTableFaultAddr] = static_cast<u32>(fa);
        table[Shader::NativeTableFaultAddr + 1] = static_cast<u32>(fa >> 32);
    }
    // Inherited user-data registers: the front admitted only members with the opener's set and
    // values (SegB v2), so the shaders take them from here.
    table[Shader::NativeTablePushInh] = push_inh0;
    for (u32 pm = push_inh0; pm != 0; pm &= pm - 1) {
        const u32 pi = static_cast<u32>(std::countr_zero(pm));
        if (pi >= Shader::NUM_USER_DATA_REGS) {
            return fail(8);
        }
        table[Shader::NativeTableUd + pi] = o_ud[pi];
    }
    for (u32 bi = 0; bi < Shader::NUM_BUFFERS; ++bi) {
        const u64 bit = u64{1} << bi;
        if (!(b.guest_mask & bit)) {
            continue;
        }
        if (inh_slots0 & bit) {
            // The V# sits in inherited registers: every member binds the opener's range.
            table[Shader::NativeTableBase + bi] = 0u - o_off[bi];
            table[Shader::NativeTableSize + bi] = o_size[bi];
            table[Shader::NativeTableInhSlots + bi / 32] |= 1u << (bi % 32);
            table[Shader::NativeTablePtLo + bi] = static_cast<u32>(o_addr[bi]);
            table[Shader::NativeTablePtHi + bi / 4] |=
                static_cast<u32>((o_addr[bi] >> 32) & 0xffu) << ((bi % 4) * 8);
            continue;
        }
        VAddr cpu = 0;
        if ((seg.gmask & bit) && base_of(seg.glo[bi], seg.ghi[bi], b.buf[bi], cpu) &&
            add_span(b.buf[bi], VideoCore::NativeRunAccess::Storage, seg.glo[bi], seg.ghi[bi])) {
            table[Shader::NativeTableBase + bi] = static_cast<u32>(cpu);
            continue;
        }
        // The union is not inside the opener's buffer (brick C v2): the base stays the opener's
        // (the descriptor path: scalar reads of this slot by members outside it are a known gap,
        // sec. 19); the pt loads address each member's range through the page table and flag
        // uncached / non-contiguous pages for ProcessFaultBuffer.
        const VideoCore::Buffer* ob = buffer_cache.NativeBufferAt(o_addr[bi], 0, 30);
        if (!ob) {
            return fail(6);
        }
        table[Shader::NativeTableBase + bi] = static_cast<u32>(ob->CpuAddr());
        ++st.segx_guest_gpu;
    }
    for (u32 a = 0; a < b.vb_n; ++a) {
        const bool in_union = ((seg.vmask >> a) & 1) != 0;
        if (b.vb[a] == 0) {
            if (in_union) {
                return fail(4); // a member has a stream where the opener has none
            }
            continue;
        }
        VAddr vcpu = 0;
        const u32 loc = attribs[a].desc.location;
        if (!in_union || loc >= Shader::NativeMaxPulledLocations ||
            !base_of(seg.vlo[a], seg.vhi[a], b.vb[a], vcpu) ||
            !add_span(b.vb[a], VideoCore::NativeRunAccess::Storage, seg.vlo[a], seg.vhi[a])) {
            return fail(4);
        }
        table[Shader::NativeTableVbase + loc] = static_cast<u32>(vcpu);
    }
    VAddr icpu = 0;
    if (!base_of(seg.ilo, seg.ihi, b.ib, icpu) ||
        !add_span(b.ib, VideoCore::NativeRunAccess::Index, seg.ilo, seg.ihi) ||
        (seg.ilo - icpu) % b.index_size != 0) {
        return fail(3);
    }
    // The spans, through the validity cache as NativeRingPrepare (one sync walk per buffer).
    const u64 clock0 = VideoCore::g_granule_clock.load(std::memory_order_acquire);
    for (u32 i = 0; i < nus; ++i) {
        NativeSpanEntry* hit = nullptr;
        for (auto& e : native_span_cache) {
            if (e.handle == us[i].handle && e.access == static_cast<u32>(us[i].access) &&
                e.gen == gen && us[i].lo >= e.lo && us[i].hi <= e.hi &&
                VideoCore::GranulesUnchangedSince(us[i].lo, us[i].hi, e.epoch)) {
                hit = &e;
                break;
            }
        }
        bool gpu_mod = false;
        const VideoCore::Buffer* buf = buffer_cache.NativeRunSpan(
            us[i].lo, us[i].hi, us[i].access, &buffer_barriers, cur_dispatch_run, hit == nullptr,
            hit ? (hit->gpu_mod ? 1 : 0) : -1, &gpu_mod);
        if (!buf || std::bit_cast<u64>(static_cast<VkBuffer>(buf->Handle())) != us[i].handle) {
            return fail(8);
        }
        st.ring_span_bytes += us[i].hi - us[i].lo;
        if (hit) {
            ++st.ring_span_hits;
            continue;
        }
        NativeSpanEntry* slot = nullptr;
        for (auto& e : native_span_cache) {
            if (e.handle == us[i].handle && e.access == static_cast<u32>(us[i].access)) {
                slot = &e;
                break;
            }
        }
        if (slot && slot->gen == gen && us[i].lo <= slot->hi && us[i].hi >= slot->lo &&
            std::max(slot->hi, us[i].hi) - std::min(slot->lo, us[i].lo) <= (u64{64} << 20) &&
            VideoCore::GranulesUnchangedSince(slot->lo, slot->hi, slot->epoch)) {
            slot->lo = std::min(slot->lo, us[i].lo);
            slot->hi = std::max(slot->hi, us[i].hi);
            slot->gpu_mod = slot->gpu_mod || gpu_mod;
        } else {
            if (!slot) {
                slot = &native_span_cache[native_span_rr++ % native_span_cache.size()];
            }
            *slot = {us[i].handle, static_cast<u32>(us[i].access), us[i].lo, us[i].hi, clock0,
                     gen, gpu_mod};
        }
    }
    st.ring_spans += nus;
    t_segx_delta = static_cast<u32>((seg.ilo - icpu) / b.index_size);
    t_segx_ranges = 0;
    for (u32 i = 0; i < seg.n_ops; ++i) {
        t_segx_ranges += segx_ops[i].kind == AmdGpu::CpSplit::SegOpDrawRange;
    }
    native_prep.active = false;
    native_ring_handle = ring_handle;
    return true;
}

void Rasterizer::NativeSegWrite(u8* map_ptr, u64 map_off, u64 table_bytes, u64 tables_rel,
                                u8* mat_ptr) {
    // NativeOpen, after member 0's record and command: the segment's members into the batch -
    // the commands (firstIndex + the index buffer delta) into slots 0..n-1, the list entries
    // (slot | ring index | material index), the slot table (one row of 32 words per distinct
    // material: the opener's, then those resolved from the representatives - no per-member
    // write, brick C v3 item 4), the count. One table copy per DRAW_RANGE after the first (its
    // list address at the range's first member). The batch is sealed.
    auto& b = native_batch;
    const auto& seg = *segx_pending;
    const u32 n = seg.n;
    for (u32 j = 0; j < n; ++j) {
        const u32* c = segx_cmds + size_t(j) * 5;
        const u32 w[5] = {c[0], c[1], c[2] + t_segx_delta, c[3], 0u};
        std::memcpy(b.cmds + u64(j) * NativeCmdBytes, w, sizeof(w));
    }
    if (b.list) {
        std::memcpy(b.list, segx_list, size_t(n) * sizeof(u32));
    }
    if (b.bindless && mat_ptr) {
        constexpr size_t row = Shader::NativeSegMatWords * sizeof(u32);
        static_assert(sizeof(b.bl_slots) == row);
        std::memcpy(mat_ptr, b.bl_slots.data(), row);
        for (u32 m = 1; m < segx_n_mat; ++m) {
            std::memcpy(mat_ptr + size_t(m) * row, segx_mat_slots[m].data(), row);
        }
    }
    *b.count = n;
    b.n = n;
    b.sealed = true;
    native_prep.active = false;
    const u64 list_addr = u64{native_ring_table[Shader::NativeTableSegList]} |
                          (u64{native_ring_table[Shader::NativeTableSegList + 1]} << 32);
    u32 k = 0;
    for (u32 i = 0; i < seg.n_ops; ++i) {
        if (segx_ops[i].kind != AmdGpu::CpSplit::SegOpDrawRange) {
            continue;
        }
        if (k > 0) {
            std::array<u32, Shader::NativeTableDwords> t = native_ring_table;
            const u64 la = list_addr + u64(segx_ops[i].a) * sizeof(u32);
            t[Shader::NativeTableSegList] = static_cast<u32>(la);
            t[Shader::NativeTableSegList + 1] = static_cast<u32>(la >> 32);
            std::memcpy(map_ptr + tables_rel + u64(k - 1) * table_bytes, t.data(), table_bytes);
            segx_table_off[k] = map_off + tables_rel + u64(k - 1) * table_bytes;
        }
        ++k;
    }
    segx_result = 1;
    ++native_stats.segx_opens;
    native_stats.segx_members += n;
}

namespace {
/// SHADPS4_NATIVE_SEGX: the dynamic stencil values UpdateDynamicState derives, for an op's raw
/// DB_STENCILREFMASK / _BF (everything else from the mirror): front / back reference, write
/// mask, compare mask.
void SegxStencilValues(const AmdGpu::Regs& regs, u32 raw_f, u32 raw_b, std::array<u32, 6>& out) {
    AmdGpu::StencilRefMask front{}, back_raw{};
    std::memcpy(&front, &raw_f, sizeof(u32));
    std::memcpy(&back_raw, &raw_b, sizeof(u32));
    const auto back = regs.depth_control.backface_enable ? back_raw : front;
    const bool stencil_clear = regs.depth_render_control.stencil_clear_enable;
    const auto reference = [&](const AmdGpu::StencilRefMask& face, AmdGpu::CompareFunc compare,
                               AmdGpu::StencilFunc fail, AmdGpu::StencilFunc pass,
                               AmdGpu::StencilFunc depth_fail) -> u32 {
        if (g_replace_op_reference && compare == AmdGpu::CompareFunc::Always &&
            fail == AmdGpu::StencilFunc::Keep && pass == AmdGpu::StencilFunc::ReplaceOp &&
            depth_fail == AmdGpu::StencilFunc::Keep &&
            !regs.depth_shader_control.stencil_test_val_export_enable &&
            !regs.depth_shader_control.stencil_op_val_export_enable) {
            return face.stencil_op_val;
        }
        return face.stencil_test_val;
    };
    out[0] = reference(front, regs.depth_control.stencil_ref_func,
                       regs.stencil_control.stencil_fail_front,
                       regs.stencil_control.stencil_zpass_front,
                       regs.stencil_control.stencil_zfail_front);
    out[1] = regs.depth_control.backface_enable
                 ? reference(back, regs.depth_control.stencil_bf_func,
                             regs.stencil_control.stencil_fail_back,
                             regs.stencil_control.stencil_zpass_back,
                             regs.stencil_control.stencil_zfail_back)
                 : out[0];
    out[2] = !stencil_clear ? front.stencil_write_mask : 0u;
    out[3] = !stencil_clear ? back.stencil_write_mask : 0u;
    out[4] = front.stencil_mask;
    out[5] = back.stencil_mask;
}

void SegxCullValues(const AmdGpu::Regs& regs, u32 raw_pc, vk::CullModeFlags& cull,
                    vk::FrontFace& ff) {
    AmdGpu::PolygonControl pc{};
    std::memcpy(&pc, &raw_pc, sizeof(u32));
    cull = LiverpoolToVK::IsPrimitiveCulled(regs.primitive_type)
               ? LiverpoolToVK::CullMode(pc.CullingMode())
               : vk::CullModeFlagBits::eNone;
    ff = LiverpoolToVK::FrontFace(pc.front_face);
}
} // namespace

void Rasterizer::NativeSegRecord(const GraphicsPipeline* pipeline, vk::Buffer sb, u64 cmd_off,
                                 vk::PipelineLayout bl_layout, vk::DescriptorSet bl_set) {
    // DrawTail, in place of the batch's drawIndexedIndirectCount: the op list - indirect draw
    // ranges over the members' commands with stencil / cull dynamic state between them. A range
    // after the first re-pushes the table (its list address at the range's first member:
    // DrawIndex restarts at 0 for every indirect draw). The dynamic-state tracker is told the
    // stencil / cull state is unknown afterwards.
    struct Baked {
        u32 kind{};
        u32 first{}, count{};
        std::array<u32, 6> st{};
        vk::CullModeFlags cull{};
        vk::FrontFace ff{};
        u64 table_off{};
    };
    const auto& seg = *segx_pending;
    const auto& regs = liverpool->BackRegs();
    std::vector<Baked> baked;
    baked.reserve(seg.n_ops);
    u32 k = 0;
    bool any_state = false;
    for (u32 i = 0; i < seg.n_ops; ++i) {
        const auto& op = segx_ops[i];
        Baked bk{};
        bk.kind = op.kind;
        switch (op.kind) {
        case AmdGpu::CpSplit::SegOpDrawRange:
            bk.first = op.a;
            bk.count = op.b;
            bk.table_off = k > 0 ? segx_table_off[k] : 0;
            ++k;
            break;
        case AmdGpu::CpSplit::SegOpStencil:
            SegxStencilValues(regs, op.a, op.b, bk.st);
            any_state = true;
            break;
        case AmdGpu::CpSplit::SegOpCull:
            SegxCullValues(regs, op.a, bk.cull, bk.ff);
            any_state = true;
            break;
        default:
            continue;
        }
        baked.push_back(bk);
    }
    const auto& b = native_batch;
    scheduler.Record([handle = pipeline->Handle(), sb, cmd_off, bl_layout, bl_set,
                      layout = pipeline->GetLayout(), tbind = pipeline->native_table_binding,
                      table_bytes = u64(Shader::NativeTableDwords) * sizeof(u32),
                      ib = vk::Buffer{std::bit_cast<VkBuffer>(b.ib)},
                      type = b.index_size == 2 ? vk::IndexType::eUint16 : vk::IndexType::eUint32,
                      baked = std::move(baked)](vk::CommandBuffer cmdbuf) {
        if (!GraphicsBindTracker::Enabled() || GraphicsBindTracker::NeedBind(cmdbuf, handle)) {
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
        }
        if (bl_set) {
            cmdbuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, bl_layout, 1, bl_set, {});
        }
        cmdbuf.bindIndexBuffer(ib, 0, type);
        for (const auto& op : baked) {
            switch (op.kind) {
            case AmdGpu::CpSplit::SegOpDrawRange: {
                if (op.table_off != 0) {
                    vk::DescriptorBufferInfo info{sb, op.table_off, table_bytes};
                    vk::WriteDescriptorSet w{};
                    w.dstSet = VK_NULL_HANDLE;
                    w.dstBinding = tbind;
                    w.dstArrayElement = 0;
                    w.descriptorCount = 1;
                    w.descriptorType = vk::DescriptorType::eStorageBuffer;
                    w.pBufferInfo = &info;
                    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, layout, 0, w);
                }
                cmdbuf.drawIndexedIndirect(sb, cmd_off + u64(op.first) * NativeCmdBytes, op.count,
                                           NativeCmdBytes);
                break;
            }
            case AmdGpu::CpSplit::SegOpStencil:
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFront, op.st[0]);
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eBack, op.st[1]);
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFront, op.st[2]);
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eBack, op.st[3]);
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFront, op.st[4]);
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eBack, op.st[5]);
                break;
            case AmdGpu::CpSplit::SegOpCull:
                cmdbuf.setCullMode(op.cull);
                cmdbuf.setFrontFace(op.ff);
                break;
            default:
                break;
            }
        }
    });
    if (any_state) {
        auto& ds = scheduler.GetDynamicState();
        ds.dirty_state.stencil_front_reference = true;
        ds.dirty_state.stencil_back_reference = true;
        ds.dirty_state.stencil_front_write_mask = true;
        ds.dirty_state.stencil_back_write_mask = true;
        ds.dirty_state.stencil_front_compare_mask = true;
        ds.dirty_state.stencil_back_compare_mask = true;
        ds.dirty_state.cull_mode = true;
        ds.dirty_state.front_face = true;
    }
}

void Rasterizer::NativeRingVerifyWrite() {
    // SHADPS4_NATIVE_GPU_DIRECT=3: the batch stays legacy; the ring's predictions (records and
    // commands) go to the prepare-verify side copy that NativeJoin compares with each join.
    auto& p = native_prep;
    const u32 n = t_prep_n;
    t_prep_n = 0;
    p.side.resize(size_t(n) * (Shader::NativeRecordDwords + 5));
    for (u32 j = 0; j < n; ++j) {
        const auto& m = t_prep[j];
        u32* out = p.side.data() + size_t(j) * (Shader::NativeRecordDwords + 5);
        std::memcpy(out, m.rec.data(), sizeof(m.rec));
        std::memcpy(out + Shader::NativeRecordDwords, m.cmdw.data(), sizeof(m.cmdw));
        p.cmd[j] = m.cmd;
        p.icount[j] = m.cmdw[0];
        p.inst[j] = m.cmdw[1];
        p.iaddr[j] = m.iaddr;
    }
    p.ring_src = true;
    p.verify = true;
    p.tag = draw_job_tag;
    p.next_idx = p.first_idx;
    p.end_idx = p.first_idx + n;
    p.slot0 = native_batch.n;
    p.n = n;
    p.active = n != 0;
    ++native_stats.ring_runs;
    native_stats.ring_members += n;
}

void Rasterizer::NativeRingWrite() {
    // NativeOpen, after draw 0's command: the members' commands into slots 1..n, the prepared
    // members for NativePrepConfirm.
    auto& b = native_batch;
    auto& p = native_prep;
    const u32 n = t_prep_n;
    t_prep_n = 0;
    for (u32 j = 0; j < n; ++j) {
        const auto& m = t_prep[j];
        std::memcpy(b.cmds + u64(b.n + j) * NativeCmdBytes, m.cmdw.data(), NativeCmdBytes);
        p.cmd[j] = m.cmd;
        p.icount[j] = m.cmdw[0];
        p.inst[j] = m.cmdw[1];
        p.iaddr[j] = m.iaddr;
    }
    if (b.bindless) {
        for (u32 j = 0; j <= n; ++j) {
            BindlessRingWrite(draw_job_tag, p.first_idx - 1 + j, b.bl_slots.data());
        }
    }
    p.ring_src = true;
    p.verify = false;
    b.ring_tag = draw_job_tag;
    b.ring_idx0 = p.first_idx - 1;
    b.ring_push_inh = native_ring_table[Shader::NativeTablePushInh];
    b.ring_inh_slots = u64{native_ring_table[Shader::NativeTableInhSlots]} |
                       (u64{native_ring_table[Shader::NativeTableInhSlots + 1]} << 32);
    b.ring_inh_vs = p.inh_vs;
    b.ring_inh_ps = p.inh_ps;
    p.tag = draw_job_tag;
    p.next_idx = p.first_idx;
    p.end_idx = p.first_idx + n;
    p.slot0 = b.n;
    p.n = n;
    p.active = true;
    ++native_stats.ring_runs;
    native_stats.ring_members += n;
}

void Rasterizer::NativePrepWrite() {
    // NativeOpen, after draw 0's record and command: the prepared members go into slots 1..n.
    auto& b = native_batch;
    auto& p = native_prep;
    const u32 n = t_prep_n;
    t_prep_n = 0;
    p.active = false;
    if (n == 0) {
        return;
    }
    p.ring_src = false;
    p.verify = g_native_prep == 2;
    if (p.verify) {
        p.side.resize(size_t(n) * (Shader::NativeRecordDwords + 5));
    }
    for (u32 j = 0; j < n; ++j) {
        const auto& m = t_prep[j];
        if (p.verify) {
            u32* out = p.side.data() + size_t(j) * (Shader::NativeRecordDwords + 5);
            std::memcpy(out, m.rec.data(), sizeof(m.rec));
            std::memcpy(out + Shader::NativeRecordDwords, m.cmdw.data(), sizeof(m.cmdw));
        } else {
            std::memcpy(b.recs + u64(b.n + j) * NativeRecBytes, m.rec.data(), NativeRecBytes);
            std::memcpy(b.cmds + u64(b.n + j) * NativeCmdBytes, m.cmdw.data(), NativeCmdBytes);
        }
        p.cmd[j] = m.cmd;
    }
    p.tag = draw_job_tag;
    p.next_idx = p.first_idx;
    p.end_idx = p.first_idx + n;
    p.slot0 = b.n;
    p.n = n;
    p.active = true;
    ++native_stats.prep_runs;
    native_stats.prep_members += n;
}

bool Rasterizer::NativePrepConfirm(const SplitDraw& draw, bool plain,
                                   const AmdGpu::CpSplit::VertexInputsPayload* vi) {
    // Back thread, Liverpool::ExecuteDrawRecord: the join's per-draw checks (NativeJoin) for a
    // member whose record and command the open already wrote; true = count + 1.
    auto& p = native_prep;
    auto& b = native_batch;
    auto& st = native_stats;
    const auto refuse = [&](u32 why) {
        ++st.prep_refuse[std::min<u32>(why, 15)];
        p.active = false;
        return false;
    };
    if (draw_job_tag != p.tag || draw_job_idx != p.next_idx) {
        return refuse(0);
    }
    const u32 j = p.next_idx - p.first_idx;
    if (j >= p.n || draw_job_cmd != p.cmd[j]) {
        return refuse(1);
    }
    if (!b.open || b.verify) {
        return refuse(2);
    }
    if (!plain || draw.draw_class != DrawClass::Draw || draw.skip || draw.args.is_indirect ||
        !draw.args.is_indexed || draw.pipeline != b.base) {
        return refuse(3);
    }
    if (!vi || vi->desc_id != b.desc_id) {
        return refuse(4);
    }
    buffer_cache.CommitDeletes(); // r13
    scheduler.PopPendingOperations();
    if (b.n != p.slot0 + j || b.n >= NativeMaxDraws) {
        return refuse(5);
    }
    if (scheduler.RecordSeq() != b.seq || scheduler.CurrentTick() != b.tick) {
        return refuse(6);
    }
    if (texture_cache.StructGeneration() != b.tex_gen ||
        texture_cache.MetaClearEpoch() != b.meta_epoch ||
        VideoCore::g_image_epoch.load(std::memory_order_acquire) != b.image_epoch) {
        return refuse(7);
    }
    if (!NativeCtxSame()) {
        return refuse(8);
    }
    const auto& regs = liverpool->BackRegs();
    for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
        if (liverpool->BackCbExtent(cb).raw != b.extents[cb]) {
            return refuse(9);
        }
    }
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    if (liverpool->BackDbExtent().raw != b.extents[AmdGpu::NUM_COLOR_BUFFERS] ||
        (is_index16 ? 2u : 4u) != b.index_size || (occlusion && occlusion->Open())) {
        return refuse(9);
    }
    if (b.colour && ((VideoCore::g_native_colour_diag & 1) ||
                     texture_cache.SamplerGeneration() != b.smp_gen ||
                     regs.ta_bc_base.base_addr != b.border || !NativeColourGates())) {
        return refuse(10);
    }
    if (VideoCore::g_dirty_epoch.load(std::memory_order_acquire) != p.epoch ||
        buffer_cache.StructGeneration() != p.gen) {
        return refuse(11);
    }
    if (b.ring && (regs.num_indices != p.icount[j] ||
                   regs.num_instances.NumInstances() != p.inst[j])) {
        return refuse(12); // the job record's counts disagree with the back's registers
    }
    if (p.ring_src) {
        // Opus G2 #3: the member's index buffer is the one the job thread saw.
        if (regs.index_base_address.Address<VAddr>() != p.iaddr[j]) {
            return refuse(13);
        }
        // Coverage (sec. 11): inherited registers must hold the opener's values.
        for (u32 m = p.inh_vs; m != 0; m &= m - 1) {
            const u32 i = static_cast<u32>(std::countr_zero(m));
            if (regs.vs_program.user_data[i] != p.inh_val_vs[i]) {
                return refuse(14);
            }
        }
        for (u32 m = p.inh_ps; m != 0; m &= m - 1) {
            const u32 i = static_cast<u32>(std::countr_zero(m));
            if (regs.ps_program.user_data[i] != p.inh_val_ps[i]) {
                return refuse(14);
            }
        }
    }
    if (p.verify) {
        ++st.prep_conf; // would confirm; NativeJoin compares its record with the prepared one
        st.prep_conf_col += b.colour;
        return false;
    }
    ++b.n;
    *b.count = b.n;
    if (++p.next_idx == p.end_idx) {
        p.active = false;
    }
    ++st.prep_conf;
    st.prep_conf_col += b.colour;
    ++st.joins;
    st.col_joins += b.colour;
    rec_outcome = b.colour ? 5u : 4u;
    DebugState.IncDrawCall();
    scheduler.NoteDraw();
    MaybeFlushEveryOps(scheduler);
    if ((st.joins & 0x3ffff) == 0) {
        NativeLog();
    }
    return true;
}

void Rasterizer::NoteRecordCycles(u64 dt) {
    const u32 i = std::min<u32>(rec_outcome, 5) + (draw_job_member ? 6u : 0u);
    native_stats.c_rec[i] += dt;
    ++native_stats.n_rec[i];
}

bool Rasterizer::NativeOpen(const GraphicsPipeline* base, const GraphicsPipeline* native,
                            bool is_indexed, u32 index_offset,
                            const AmdGpu::CpSplit::VertexInputsPayload* front_vi, bool colour) {
    std::array<u64, 6> no_ph{};
    u64 no_t = __rdtsc();
    const auto no_lap = [&](u32 ph) {
        const u64 now = __rdtsc();
        no_ph[ph] += now - no_t;
        no_t = now;
    };
    // Back thread, inside DrawTail after BindResources and the vertex / index binds of `base`.
    // native == nullptr: verify mode (only the batch keys are captured; nothing changes).
    auto& b = native_batch;
    b = NativeBatch{.verify = native == nullptr};
    b.colour = colour;
    native_prep.active = false;
    const auto fail = [&](u32 line) {
        ++native_stats.open_fail;
        native_stats.col_open_fail += colour;
        for (auto& [l, n] : native_stats.fail_lines) {
            if (l == line || l == 0) {
                l = line;
                ++n;
                break;
            }
        }
        b.open = false;
        return false;
    };
    // Opus review 2026-10-09: the context epoch only moves with the viewport memo (#2); without
    // dynamic vertex input the bound sizes would clip rebased reads (#3); attributes past 16 are
    // not tracked (#7).
    // Opus code review #3: RecordSeq only sees records through the recording worker, and the
    // stream-size / deferred-upload modes upload without a record; MTB translators share the
    // stream buffer.
    static const bool native_config_ok = [] {
        const char* stream = std::getenv("SHADPS4_STREAM_MAX_SIZE");
        const bool stream_off = !stream || std::strtoull(stream, nullptr, 0) == 0;
        const char* mtb = std::getenv("SHADPS4_MTB_P1");
        const bool mtb_off = !mtb || mtb[0] != '1';
        return stream_off && mtb_off && !VideoCore::g_epoch_defer_upload;
    }();
    if ((AmdGpu::NativeRec::g_gpu_mode == 1 || AmdGpu::NativeRec::g_gpu_mode == 3) &&
        (!native || native->native_table_binding == GraphicsPipeline::NoNativeRecords ||
         buffer_infos.size() + 4 + native_pt_extra > Shader::NUM_BUFFERS)) {
        return fail(__LINE__);
    }
    if (!front_vi || !is_indexed || g_viewport_memo < 1 || !instance.IsVertexInputDynamicState() ||
        front_vi->count > 16 || !native_config_ok || !scheduler.IsAsyncRecording()) {
        return fail(__LINE__);
    }
    const auto& vi = VideoCore::t_native_vi;
    if (vi.ib_type == 0 || vi.ib_off % vi.ib_type != 0 || vi.n > b.vb.size()) {
        return fail(__LINE__);
    }
    const auto& regs = liverpool->BackRegs();
    no_lap(0); // pre-checks
    b.base = base;
    b.ctx_epoch = liverpool->BackContextEpoch();
    std::memcpy(native_ctx.data(), &regs.reg_array[AmdGpu::Regs::ContextRegWordOffset],
                sizeof(u32) * native_ctx.size());
    b.tex_gen = texture_cache.StructGeneration();
    b.meta_epoch = texture_cache.MetaClearEpoch();
    b.image_epoch = VideoCore::g_image_epoch.load(std::memory_order_acquire);
    if (g_native_diag) [[unlikely]] {
        g_native_diag_ctx.assign(regs.reg_array.begin() + AmdGpu::Regs::ContextRegWordOffset,
                                 regs.reg_array.begin() + AmdGpu::Regs::ContextRegWordOffset + 0x400);
    }
    b.desc_id = front_vi->desc_id;
    for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
        b.extents[cb] = liverpool->BackCbExtent(cb).raw;
    }
    b.extents[AmdGpu::NUM_COLOR_BUFFERS] = liverpool->BackDbExtent().raw;
    b.index_size = vi.ib_type;
    b.ib = vi.ib_buf;
    b.vb_n = vi.n;
    for (u32 k = 0; k < vi.n; ++k) {
        b.vb[k] = vi.buf[k];
        b.vb_off[k] = vi.off[k];
        b.vb_stride[k] = vi.stride[k];
    }
    // SHADPS4_NATIVE_COLOUR: the opener's image key + the draw-repeat config gates; every
    // non-null vertex stream in ONE VkBuffer (s301: 92.6% of textured draws), per-vertex only;
    // the record carries each location's offset (any alignment), so no rebase.
    std::array<u32, Shader::NativeMaxPulledLocations> col_voff{};
    std::array<u32, Shader::NativeMaxPulledLocations> col_vsize{};
    if (colour) {
        if (!NativeColourGates() || !NativeImageKey(base, native_img_key, b.img_n)) {
            return fail(__LINE__);
        }
        // Opus code review 2026-10-09 #1: a draw that samples a bound target (feedback loop)
        // records a Transit barrier per draw on the generic path; a join would skip it.
        if (attachment_feedback_loop) {
            return fail(__LINE__);
        }
        for (const auto id : bound_images) {
            const auto& image = texture_cache.PeekImage(id);
            if ((image.binding.is_target && image.binding.is_bound) ||
                image.binding.force_general) {
                return fail(__LINE__);
            }
        }
        b.smp_gen = texture_cache.SamplerGeneration();
        b.border = regs.ta_bc_base.base_addr;
        no_lap(1); // keys, context copy, image key, gates
        if (native && native->native_bindless) {
            // SHADPS4_NATIVE_BINDLESS: the opener's slots, checked against what BindTextures
            // just bound for it (the generic writes: views + layouts, samplers, in order).
            std::array<vk::ImageView, Shader::NativeBindlessImgCap> views{};
            std::array<vk::Sampler, Shader::NativeBindlessSmpCap> smps{};
            u32 ni = 0, ns = 0;
            const u64 s0 = scheduler.RecordSeq();
            if (!BindlessResolve(base, b.bl_slots.data(), views.data(), smps.data(), ni, ns) ||
                scheduler.RecordSeq() != s0) {
                ++native_stats.bl_open_fail;
                return fail(__LINE__);
            }
            u32 ki = 0, ks = 0;
            bool bad = false;
            // SHADPS4_NATIVE_BINDLESS=2: verify every opener (speed arms: =1, no verify).
            static const bool bl_verify = [] {
                const char* e = std::getenv("SHADPS4_NATIVE_BINDLESS");
                return e && e[0] == '2';
            }();
            for (u32 w = 0; bl_verify && w < set_write_index; ++w) {
                const auto& sw = set_writes[w];
                if (sw.descriptorType == vk::DescriptorType::eSampledImage && ki < ni) {
                    bad = bad || sw.pImageInfo[0].imageView != views[ki];
                    ++ki;
                } else if (sw.descriptorType == vk::DescriptorType::eSampler && ks < ns) {
                    bad = bad || sw.pImageInfo[0].sampler != smps[ks];
                    ++ks;
                }
            }
            if (!bl_verify) {
            } else if (ki != ni || ks != ns) {
                ++native_stats.bl_vfy_none;
            } else if (bad) {
                if (++native_stats.bl_vfy_bad <= 16) {
                    LOG_CRITICAL(Render_Vulkan,
                                 "NativeBindless verify MISMATCH: the opener's resolved views / "
                                 "samplers differ from BindTextures' ({} images, {} samplers)",
                                 ni, ns);
                }
                return fail(__LINE__);
            } else {
                ++native_stats.bl_vfy_ok;
            }
            b.bindless = true;
            b.bl_set = bindless.cur;
            bindless.last_tick[b.bl_set] = scheduler.CurrentTick();
            ++native_stats.bl_open;
            if (segx_pending != nullptr && segx_n_mat > 1) {
                // Brick C v3 item 4: every further DISTINCT material of the segment, resolved
                // once from its representative member's stage blocks exactly as the opener's
                // (image refresh included: a dirty image misses the slot cache and goes through
                // FindImage), into the segment's slot table; the set copy must not rotate and no
                // barrier may be recorded meanwhile (the join rule).
                namespace NR = AmdGpu::NativeRec;
                const auto& seg = *segx_pending;
                const u64 s1 = scheduler.RecordSeq();
                for (u32 m = 1; m < segx_n_mat; ++m) {
                    std::array<u32, 512> key;
                    u32 n = 0;
                    u64 h1 = 0, h2 = 0;
                    if (!NativeImageKey(base, key, n, &segx_mat_stages[m]) || n != b.img_n) {
                        return fail(__LINE__);
                    }
                    NR::ImageKeyHash(key.data(), n, h1, h2);
                    if (h1 != seg.mat_h1[m] || h2 != seg.mat_h2[m]) {
                        // The job's key may hold zeros for inherited T# / S# (ImgInhBit on its
                        // record); any other disagreement refuses the open (replay).
                        const u32 e = segx_list[seg.mat_rep[m]];
                        const u32 slot = (e >> 24) & Shader::NativeSegSlotMask;
                        const u32 ridx = e & 0xffffffu;
                        const NR::Record* ring =
                            slot < NR::MaxSlots ? NR::g_rings[slot].load(std::memory_order_acquire)
                                                : nullptr;
                        bool inherited = false;
                        if (ring) {
                            const NR::Record& r = ring[ridx & (NR::RingSize - 1)];
                            const u32 img_n = r.img_n;
                            std::atomic_thread_fence(std::memory_order_acquire);
                            inherited = ((r.stamp.load(std::memory_order_relaxed) - 1) & 0xffffffu) ==
                                            ridx &&
                                        (img_n & NR::ImgInhBit) != 0;
                        }
                        if (!inherited) {
                            return fail(__LINE__);
                        }
                    }
                    u32 ni = 0, ns = 0;
                    const auto& fs_m = segx_mat_stages[m][u32(Shader::LogicalStage::Fragment)];
                    if (!BindlessResolve(base, segx_mat_slots[m].data(), nullptr, nullptr, ni, ns,
                                         true, &fs_m) ||
                        bindless.cur != b.bl_set || scheduler.RecordSeq() != s1) {
                        ++native_stats.bl_open_fail;
                        return fail(__LINE__);
                    }
                }
            }
        }
        no_lap(2); // bindless resolve
        const auto* attribs = AmdGpu::CpSplit::VertexInputAttribs(*front_vi);
        for (u32 k = 0; k < vi.n; ++k) {
            const u32 loc = attribs[k].desc.location;
            if (attribs[k].desc.instanced || loc >= Shader::NativeMaxPulledLocations) {
                return fail(__LINE__);
            }
            if (vi.buf[k] == 0) {
                continue; // null stream: vsize 0 -> the shader reads (0, 0, 0, 1)
            }
            if ((b.vbuf != 0 && vi.buf[k] != b.vbuf) || vi.off[k] > 0xffffffffull) {
                return fail(__LINE__);
            }
            b.vbuf = vi.buf[k];
            b.vmask |= 1u << loc;
            col_voff[loc] = static_cast<u32>(vi.off[k]);
            col_vsize[loc] = 0xffffffffu; // bound to the VkBuffer's end, as bindVertexBuffers
        }
        if (b.verify) {
            native_open_images.assign(image_infos.begin(), image_infos.end());
        }
        // GPU-written vertex data (compute skinning): the generic vertex barrier made it visible
        // to vertex-input reads only (and the tracker may have skipped it); the native VS reads it
        // as storage. One explicit whole-buffer barrier at the open (s329: doing this for every
        // vertex barrier of every draw cost 2.4%). Joins refuse on any later write (RecordSeq).
        if (!b.verify && vi.gpu_mod && b.vbuf != 0) {
            buffer_barriers.push_back(vk::BufferMemoryBarrier2{
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eVertexShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead,
                .buffer = vk::Buffer{std::bit_cast<VkBuffer>(b.vbuf)},
                .offset = 0,
                .size = VK_WHOLE_SIZE,
            });
        }
    }
    // Rebase the per-vertex bindings to the lowest offset of the same stride residue, so a
    // later draw's vertex data below draw 0's still has a non-negative vertexOffset (s287: 3.1 M
    // refusals). Needs one common quotient over the per-vertex bindings; instance-rate bindings
    // keep draw 0's offset (they may not move).
    if (!colour) {
        const auto* attribs = AmdGpu::CpSplit::VertexInputAttribs(*front_vi);
        bool ok = true;
        bool any = false;
        u64 q0 = 0;
        for (u32 k = 0; k < vi.n; ++k) {
            if (vi.buf[k] == 0 || attribs[k].desc.instanced || vi.stride[k] == 0) {
                continue;
            }
            const u64 qk = vi.off[k] / vi.stride[k];
            if (any && qk != q0) {
                ok = false;
            }
            q0 = qk;
            any = true;
        }
        if (ok && any && q0 <= 0x7fffffffull) {
            for (u32 k = 0; k < vi.n; ++k) {
                if (vi.buf[k] == 0 || attribs[k].desc.instanced || vi.stride[k] == 0) {
                    continue;
                }
                b.vb_off[k] = vi.off[k] % vi.stride[k];
            }
            b.vb_rebased = true;
            b.vb_q0 = static_cast<u32>(q0);
        }
    }
    // Buffer bindings, in BindResources order.
    std::array<u32, Shader::NUM_BUFFERS> rec_off{};
    std::array<u32, Shader::NUM_BUFFERS> rec_size{};
    std::array<u64, Shader::NUM_BUFFERS> open_addr{}; // SHADPS4_NATIVE_REC_PREP: the opener's V#s
    std::array<u64, Shader::NUM_BUFFERS> open_size{};
    u32 bi = 0;
    for (const auto& bound : BoundStages(base)) {
        if (!bound.info) {
            continue;
        }
        for (const auto& desc : bound.info->buffers) {
            if (bi >= buffer_infos.size()) {
                return fail(__LINE__);
            }
            if (desc.buffer_type == Shader::BufferType::Guest &&
                (g_native_prep != 0 || AmdGpu::NativeRec::g_gpu_mode != 0) &&
                bi < Shader::NUM_BUFFERS) {
                const auto vsharp = desc.GetSharp(bound);
                open_addr[bi] = vsharp.base_address;
                open_size[bi] = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
            }
            if (desc.buffer_type == Shader::BufferType::Guest) {
                const u64 full = buffer_infos[bi].offset + push_data.buf_offsets[bi];
                if (full > 0xffffffffull || !buffer_infos[bi].buffer) {
                    return fail(__LINE__);
                }
                rec_off[bi] = static_cast<u32>(full);
                const u64 bound_size = buffer_infos[bi].range - push_data.buf_offsets[bi];
                rec_size[bi] = static_cast<u32>(std::min<u64>(bound_size, 0xffffffffull));
                b.buf[bi] = std::bit_cast<u64>(static_cast<VkBuffer>(buffer_infos[bi].buffer));
                b.guest_mask |= u64{1} << bi;
            }
            ++bi;
        }
    }
    if (bi != buffer_infos.size()) {
        return fail(__LINE__);
    }
    const auto& vs_bound = GetBoundStage(Shader::LogicalStage::Vertex);
    if (GetDrawOffsets(regs, vs_bound, base->GetFetchShader()).second != 0) {
        // Opus code review #2: firstInstance != 0 in an indirect command needs
        // drawIndirectFirstInstance, which is not enabled.
        return fail(__LINE__);
    }
    if (b.verify) {
        b.open = true;
        return true;
    }
    if (!colour && g_native_rec_run != 0) [[unlikely]] {
        NativeRunValidate(base, &buffer_barriers);
    }
    no_lap(3); // vertex locations, buffers / record words
    const bool gpu_direct =
        AmdGpu::NativeRec::g_gpu_mode == 1 || AmdGpu::NativeRec::g_gpu_mode == 3;
    // Opus G2 #7: NATIVE_REC_PREP with GPU_DIRECT 1 / 3 would overwrite the ring's prepared
    // members - the ring wins, the prepare is off.
    const bool prep_on = g_native_prep != 0 && !gpu_direct;
    if (prep_on) [[unlikely]] {
        NativePrepare(base, AmdGpu::CpSplit::VertexInputAttribs(*front_vi), open_addr, open_size,
                      rec_off, rec_size);
    }
    bool ring_verify = false;
    bool segx_prepared = false;
    if (gpu_direct) [[unlikely]] {
        // SHADPS4_NATIVE_SEGX=1: a Segment record's member 0 takes the segment's unions instead
        // of the opener's job run (NativeSegPrepare); a refusal leaves a legacy opener.
        const bool prepared =
            segx_pending != nullptr
                ? NativeSegPrepare(base, AmdGpu::CpSplit::VertexInputAttribs(*front_vi),
                                   open_addr, rec_off, rec_size, push_data.ud_regs.data())
                : NativeRingPrepare(base, AmdGpu::CpSplit::VertexInputAttribs(*front_vi),
                                    open_addr, rec_off, rec_size, push_data.ud_regs.data());
        segx_prepared = prepared && segx_pending != nullptr;
        ring_verify = prepared && !segx_prepared && AmdGpu::NativeRec::g_gpu_mode == 3;
        b.ring = prepared && !ring_verify;
        native_stats.ring_opens_legacy += !prepared;
    }
    no_lap(4); // ring prepare
    const u64 table_bytes = gpu_direct ? Shader::NativeTableDwords * 4 : 0;
    // SHADPS4_NATIVE_SEG: the batch's record list (a u32 per draw) after the count.
    const u64 seg_bytes = native_seg_on ? u64(NativeMaxDraws) * 4 : 0;
    // SHADPS4_NATIVE_SEGX: one table copy per DRAW_RANGE after the first (NativeSegWrite).
    const u64 segx_bytes =
        segx_prepared && t_segx_ranges > 1 ? u64(t_segx_ranges - 1) * table_bytes : 0;
    // Brick C v3 item 4: the segment's slot table (one row per distinct material) last.
    const u64 mat_bytes =
        segx_prepared ? u64(segx_n_mat) * Shader::NativeSegMatWords * sizeof(u32) : 0;
    auto& stream = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
    if (!stream.IsCoherent()) {
        return fail(__LINE__);
    }
    const u64 total = u64(NativeMaxDraws) * (NativeRecBytes + NativeCmdBytes) + 16 + table_bytes +
                      seg_bytes + segx_bytes + mat_bytes;
    const u64 seq_before_map = scheduler.RecordSeq();
    const u64 tick_before_map = scheduler.CurrentTick();
    const auto [map_ptr, map_off] = stream.Map(total, instance.StorageMinAlignment(), false);
    if (!map_ptr || scheduler.RecordSeq() != seq_before_map ||
        scheduler.CurrentTick() != tick_before_map) {
        b.ring = false;
        return fail(__LINE__);
    }
    // SHADPS4_NATIVE_GPU_DIRECT: the batch table first (256 bytes keeps the records aligned),
    // then the records, commands and count as before.
    u8* const ptr = map_ptr + table_bytes;
    const u64 offset = map_off + table_bytes;
    if (gpu_direct) [[unlikely]] {
        if (!b.ring) {
            native_ring_table.fill(0);
            native_ring_table[Shader::NativeTableFirst] = 0;
            native_ring_table[Shader::NativeTableMask] = NativeMaxDraws - 1;
            if (native_pt_on) {
                // SHADPS4_NATIVE_PT: the records hold guest addresses (NativePtRecord); the
                // slot's base = its VkBuffer's CPU address (record - base = the offset).
                for (u64 m = b.guest_mask; m != 0; m &= m - 1) {
                    const u32 i = static_cast<u32>(std::countr_zero(m));
                    native_ring_table[Shader::NativeTableBase + i] =
                        static_cast<u32>(open_addr[i]) - rec_off[i];
                }
                const u64 fb = buffer_cache.GetBdaPageTableBuffer()->BufferDeviceAddress();
                native_ring_table[Shader::NativeTablePtFallback] = static_cast<u32>(fb);
                native_ring_table[Shader::NativeTablePtFallback + 1] = static_cast<u32>(fb >> 32);
                const u64 fa = buffer_cache.GetFaultBuffer()->BufferDeviceAddress();
                native_ring_table[Shader::NativeTableFaultAddr] = static_cast<u32>(fa);
                native_ring_table[Shader::NativeTableFaultAddr + 1] = static_cast<u32>(fa >> 32);
            }
        }
        if (native_seg_on) [[unlikely]] {
            const u64 list_off = u64(NativeMaxDraws) * (NativeRecBytes + NativeCmdBytes) + 16;
            NativeSegOpen(stream.BufferDeviceAddress() + offset,
                          stream.BufferDeviceAddress() + offset + list_off,
                          reinterpret_cast<u32*>(ptr + list_off));
            b.list = reinterpret_cast<u32*>(ptr + list_off);
        }
        // Brick C v3 item 4: the segment's slot table address, 0 for every other batch (written
        // on every open so no earlier segment's address can survive in the persistent table).
        const u64 ma =
            segx_prepared ? stream.BufferDeviceAddress() + map_off + total - mat_bytes : 0;
        native_ring_table[Shader::NativeTableSegMat] = static_cast<u32>(ma);
        native_ring_table[Shader::NativeTableSegMat + 1] = static_cast<u32>(ma >> 32);
        std::memcpy(map_ptr, native_ring_table.data(), table_bytes);
    }
    stream.Commit();
    b.stream_off = offset;
    b.recs = ptr;
    b.cmds = ptr + u64(NativeMaxDraws) * NativeRecBytes;
    b.count = reinterpret_cast<u32*>(b.cmds + u64(NativeMaxDraws) * NativeCmdBytes);
    // Draw 0's record and command.
    auto* rec = reinterpret_cast<u32*>(b.recs);
    std::memcpy(rec, push_data.ud_regs.data(), sizeof(u32) * Shader::NUM_USER_DATA_REGS);
    std::memcpy(rec + Shader::NUM_USER_DATA_REGS, rec_off.data(), sizeof(u32) * Shader::NUM_BUFFERS);
    if (native_pt_on) [[unlikely]] {
        NativePtRecord(rec, open_addr);
    }
    std::memcpy(rec + Shader::NUM_USER_DATA_REGS + Shader::NUM_BUFFERS, rec_size.data(),
                sizeof(u32) * Shader::NUM_BUFFERS);
    if (colour) {
        std::memcpy(rec + Shader::NativeRecordVoffOffset, col_voff.data(), sizeof(col_voff));
        std::memcpy(rec + Shader::NativeRecordVsizeOffset, col_vsize.data(), sizeof(col_vsize));
    }
    if (b.bindless) {
        std::memcpy(rec + Shader::NativeRecordImgOffset, b.bl_slots.data(), sizeof(b.bl_slots));
    }
    b.open_tick = tick_before_map;
    const auto& vs_info = GetBoundStage(Shader::LogicalStage::Vertex);
    const auto [vertex_offset, instance_offset] =
        GetDrawOffsets(regs, vs_info, base->GetFetchShader());
    const s64 vo0 = s64(s32(vertex_offset)) + (b.vb_rebased ? s64(b.vb_q0) : 0);
    if (vo0 < 0 || vo0 > INT32_MAX) {
        return fail(__LINE__);
    }
    const u32 cmd[5] = {regs.num_indices, regs.num_instances.NumInstances(),
                        static_cast<u32>(vi.ib_off / vi.ib_type), static_cast<u32>(vo0),
                        instance_offset};
    std::memcpy(b.cmds, cmd, sizeof(cmd));
    b.n = 1;
    *b.count = 1;
    // Rewrite the guest bindings to whole buffers (the record carries each draw's offset) and
    // add the records SSBO as the last binding.
    for (u32 i = 0; i < bi; ++i) {
        if (b.guest_mask & (u64{1} << i)) {
            buffer_infos[i].offset = 0;
            buffer_infos[i].range = VK_WHOLE_SIZE;
            push_data.buf_offsets[i] = 0;
        }
    }
    if (Shader::g_pt_vertex != 0) {
        // q18b: the colour VS reads its page-table vertex mask from buf_offsets[0..1]; the back pulls from its bound
        // vertex buffer only.
        push_data.buf_offsets[0] = 0;
        push_data.buf_offsets[1] = 0;
    }
    if (b.ring) {
        buffer_infos.emplace_back(vk::Buffer{std::bit_cast<VkBuffer>(native_ring_handle)}, 0,
                                  VK_WHOLE_SIZE);
    } else {
        buffer_infos.emplace_back(stream.Handle(), offset, u64(NativeMaxDraws) * NativeRecBytes);
    }
    auto& w = set_writes.emplace_back();
    w.dstSet = VK_NULL_HANDLE;
    w.dstBinding = native->native_records_binding;
    w.dstArrayElement = 0;
    w.descriptorCount = 1;
    w.descriptorType = vk::DescriptorType::eStorageBuffer;
    w.pBufferInfo = &buffer_infos.back();
    ++set_write_index;
    vk::Buffer vb2{}; // q53: the second vertex SSBO's buffer (the back binds its one vertex buffer twice)
    if (colour) {
        // The whole vertex VkBuffer (a stream-buffer stand-in when every stream is null).
        const vk::Buffer vb = b.vbuf != 0 ? vk::Buffer{std::bit_cast<VkBuffer>(b.vbuf)}
                                          : stream.Handle();
        buffer_infos.emplace_back(vb, 0, VK_WHOLE_SIZE);
        auto& wv = set_writes.emplace_back();
        wv.dstSet = VK_NULL_HANDLE;
        wv.dstBinding = native->native_vertex_binding;
        wv.dstArrayElement = 0;
        wv.descriptorCount = 1;
        wv.descriptorType = vk::DescriptorType::eStorageBuffer;
        wv.pBufferInfo = &buffer_infos.back();
        ++set_write_index;
        vb2 = vb; // q53: written last (binding order), below
        if (VideoCore::g_native_colour_diag & 4096) {
            if (!native_dbg) {
                native_dbg = std::make_unique<VideoCore::Buffer>(
                    instance, scheduler, VideoCore::MemoryUsage::Download, 0,
                    vk::BufferUsageFlagBits::eStorageBuffer, 64 * 1024);
                std::memset(native_dbg->mapped_data.data(), 0, native_dbg->mapped_data.size());
            }
            buffer_infos.emplace_back(native_dbg->Handle(), 0, VK_WHOLE_SIZE);
            auto& wd = set_writes.emplace_back();
            wd.dstSet = VK_NULL_HANDLE;
            wd.dstBinding = native->native_vertex_binding + 1;
            wd.dstArrayElement = 0;
            wd.descriptorCount = 1;
            wd.descriptorType = vk::DescriptorType::eStorageBuffer;
            wd.pBufferInfo = &buffer_infos.back();
            ++set_write_index;
        }
    }
    if (gpu_direct) [[unlikely]] {
        buffer_infos.emplace_back(stream.Handle(), map_off, table_bytes);
        auto& wt = set_writes.emplace_back();
        wt.dstSet = VK_NULL_HANDLE;
        wt.dstBinding = native->native_table_binding;
        wt.dstArrayElement = 0;
        wt.descriptorCount = 1;
        wt.descriptorType = vk::DescriptorType::eStorageBuffer;
        wt.pBufferInfo = &buffer_infos.back();
        ++set_write_index;
        if (native_pt_on && native->native_pt_debug_binding != GraphicsPipeline::NoNativeRecords)
            [[unlikely]] {
            // SHADPS4_NATIVE_PT: the page table is read by pointer at the table's fallback (its
            // device address): no binding, no push; =2 the compare SSBO.
            {
                if (!native_pt_dbg) {
                    native_pt_dbg = std::make_unique<VideoCore::Buffer>(
                        instance, scheduler, VideoCore::MemoryUsage::Download, 0,
                        vk::BufferUsageFlagBits::eStorageBuffer, 64 * 1024);
                    std::memset(native_pt_dbg->mapped_data.data(), 0,
                                native_pt_dbg->mapped_data.size());
                }
                buffer_infos.emplace_back(native_pt_dbg->Handle(), 0, VK_WHOLE_SIZE);
                auto& wd = set_writes.emplace_back();
                wd.dstSet = VK_NULL_HANDLE;
                wd.dstBinding = native->native_pt_debug_binding;
                wd.dstArrayElement = 0;
                wd.descriptorCount = 1;
                wd.descriptorType = vk::DescriptorType::eStorageBuffer;
                wd.pBufferInfo = &buffer_infos.back();
                ++set_write_index;
            }
        }
        if (segx_prepared) [[unlikely]] {
            NativeSegWrite(map_ptr, map_off, table_bytes, total - segx_bytes - mat_bytes,
                           map_ptr + total - mat_bytes);
        } else if (b.ring) {
            NativeRingWrite();
        } else if (ring_verify) {
            NativeRingVerifyWrite();
        }
    }
    if (colour && native->native_vertex2_binding != GraphicsPipeline::NoNativeRecords) {
        // q53: the second vertex SSBO = the back's one vertex buffer, so any mask bit reads the same bytes; =2 / =3
        // (bit-exact gate): every / every other location through it. LAST: its binding follows the table's, and
        // the draw packet (SHADPS4_DRAW_PACKET) pairs buffer infos with the layout's bindings in binding order
        // (s882 / s883: the write placed before the table bound the table to the vertex buffer -> device lost).
        buffer_infos.emplace_back(vb2, 0, VK_WHOLE_SIZE);
        auto& wv2 = set_writes.emplace_back();
        wv2.dstSet = VK_NULL_HANDLE;
        wv2.dstBinding = native->native_vertex2_binding;
        wv2.dstArrayElement = 0;
        wv2.descriptorCount = 1;
        wv2.descriptorType = vk::DescriptorType::eStorageBuffer;
        wv2.pBufferInfo = &buffer_infos.back();
        ++set_write_index;
        const u8 m = Shader::g_vbuf2 == 2 ? 0xffu : Shader::g_vbuf2 == 3 ? 0x55u : 0u;
        push_data.buf_offsets[2] = m;
        push_data.buf_offsets[3] = m;
    }
    if (prep_on) [[unlikely]] {
        NativePrepWrite();
    }
    if (AmdGpu::NativeRec::g_gpu_mode == 2) [[unlikely]] {
        std::array<u64, Shader::NUM_BUFFERS> cpu{};
        for (u32 i = 0; i < Shader::NUM_BUFFERS; ++i) {
            if (b.guest_mask & (u64{1} << i)) {
                const auto* ob = buffer_cache.NativeBufferAt(open_addr[i], 0, 31);
                cpu[i] = ob ? ob->CpuAddr() : 0;
            }
        }
        NativeGpuVerify(push_data.ud_regs.data(), rec_off, rec_size, cpu,
                        colour ? col_voff.data() : nullptr,
                        AmdGpu::CpSplit::VertexInputAttribs(*front_vi), cmd, vertex_offset,
                        instance_offset);
    }
    {
        // Segment continuation census (sec. 14).
        auto& st = native_stats;
        const auto mix = [](u64 h, u64 v) { return (h ^ v) * 0x100000001B3ull; };
        const auto& key = base->GetGraphicsKey();
        constexpr size_t fixed0 = offsetof(GraphicsPipelineKey, patch_control_points);
        u64 hf = 0xcbf29ce484222325ull;
        for (size_t i = fixed0; i < sizeof(key); ++i) {
            hf = mix(hf, reinterpret_cast<const u8*>(&key)[i]);
        }
        const auto& ds = scheduler.GetDynamicState();
        u64 hd = 0xcbf29ce484222325ull;
        const auto add = [&](const void* p, size_t n) {
            for (size_t i = 0; i < n; ++i) {
                hd = mix(hd, static_cast<const u8*>(p)[i]);
            }
        };
        add(ds.viewports.data(), ds.viewports.size() * sizeof(ds.viewports[0]));
        add(ds.scissors.data(), ds.scissors.size() * sizeof(ds.scissors[0]));
        const u32 flags = u32(ds.depth_test_enabled) | u32(ds.depth_write_enabled) << 1 |
                          u32(ds.depth_bounds_test_enabled) << 2 | u32(ds.depth_bias_enabled) << 3 |
                          u32(ds.stencil_test_enabled) << 4 | u32(ds.primitive_restart_enable) << 5 |
                          u32(ds.rasterizer_discard_enable) << 6 | u32(ds.feedback_loop_enabled) << 7;
        add(&flags, sizeof(flags));
        add(&ds.depth_compare_op, sizeof(ds.depth_compare_op));
        add(&ds.depth_bounds_min, sizeof(float));
        add(&ds.depth_bounds_max, sizeof(float));
        add(&ds.depth_bias_constant, sizeof(float));
        add(&ds.depth_bias_clamp, sizeof(float));
        add(&ds.depth_bias_slope, sizeof(float));
        add(&ds.stencil_front_ops, sizeof(ds.stencil_front_ops));
        add(&ds.stencil_back_ops, sizeof(ds.stencil_back_ops));
        const std::array<u32, 6> sv{ds.stencil_front_reference, ds.stencil_front_write_mask,
                                    ds.stencil_front_compare_mask, ds.stencil_back_reference,
                                    ds.stencil_back_write_mask, ds.stencil_back_compare_mask};
        add(sv.data(), sizeof(sv));
        add(&ds.cull_mode, sizeof(ds.cull_mode));
        add(&ds.front_face, sizeof(ds.front_face));
        add(ds.blend_constants.data(), sizeof(ds.blend_constants));
        add(&ds.color_write_masks, sizeof(ds.color_write_masks));
        add(&ds.line_width, sizeof(float));
        std::array<u64, 11> g{};
        {
            const auto gh = [&](u32 gi, const void* p, size_t n) {
                u64 h = g[gi] ? g[gi] : 0xcbf29ce484222325ull;
                for (size_t i = 0; i < n; ++i) {
                    h = mix(h, static_cast<const u8*>(p)[i]);
                }
                g[gi] = h;
            };
            gh(0, ds.viewports.data(), ds.viewports.size() * sizeof(ds.viewports[0]));
            gh(1, ds.scissors.data(), ds.scissors.size() * sizeof(ds.scissors[0]));
            const u32 zf = u32(ds.depth_test_enabled) | u32(ds.depth_write_enabled) << 1 |
                           u32(ds.depth_bounds_test_enabled) << 2;
            gh(2, &zf, 4);
            gh(2, &ds.depth_compare_op, sizeof(ds.depth_compare_op));
            gh(2, &ds.depth_bounds_min, 4);
            gh(2, &ds.depth_bounds_max, 4);
            const u32 bf = ds.depth_bias_enabled;
            gh(3, &bf, 4);
            gh(3, &ds.depth_bias_constant, 4);
            gh(3, &ds.depth_bias_clamp, 4);
            gh(3, &ds.depth_bias_slope, 4);
            const u32 sf = ds.stencil_test_enabled;
            gh(4, &sf, 4);
            gh(4, &ds.stencil_front_ops, sizeof(ds.stencil_front_ops));
            gh(4, &ds.stencil_back_ops, sizeof(ds.stencil_back_ops));
            gh(5, &ds.stencil_front_reference, 4);
            gh(5, &ds.stencil_back_reference, 4);
            gh(6, &ds.stencil_front_write_mask, 4);
            gh(6, &ds.stencil_front_compare_mask, 4);
            gh(6, &ds.stencil_back_write_mask, 4);
            gh(6, &ds.stencil_back_compare_mask, 4);
            gh(7, &ds.cull_mode, sizeof(ds.cull_mode));
            gh(7, &ds.front_face, sizeof(ds.front_face));
            gh(8, ds.blend_constants.data(), sizeof(ds.blend_constants));
            gh(9, &ds.color_write_masks, sizeof(ds.color_write_masks));
            const u32 of = u32(ds.primitive_restart_enable) | u32(ds.rasterizer_discard_enable) << 1 |
                           u32(ds.feedback_loop_enabled) << 2;
            gh(10, &of, 4);
            gh(10, &ds.line_width, 4);
        }
        u64 hr = 0xcbf29ce484222325ull;
        for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
            hr = mix(hr, regs.color_buffers[cb].Address());
        }
        hr = mix(hr, regs.depth_buffer.DepthAddress());
        if (st.seg_prev_valid) {
            ++st.seg_opens;
            const u32 why = (hf != st.seg_prev_fixed ? 1u : 0u) | (hd != st.seg_prev_dyn ? 2u : 0u) |
                            (hr != st.seg_prev_rt ? 4u : 0u) | (st.seg_generic_since ? 8u : 0u) |
                            (b.index_size != st.seg_prev_index ? 16u : 0u);
            st.seg_cont += why == 0;
            ++st.seg_break[why];
            if (why & 2) {
                for (u32 gi = 0; gi < 11; ++gi) {
                    st.seg_dgroup[gi] += g[gi] != st.seg_prev_g[gi];
                }
            }
        }
        st.seg_prev_g = g;
        st.seg_prev_valid = true;
        st.seg_prev_fixed = hf;
        st.seg_prev_dyn = hd;
        st.seg_prev_rt = hr;
        st.seg_prev_index = b.index_size;
        st.seg_generic_since = false;
    }
    no_lap(5); // stream map, table, records, writes
    if (colour) {
        for (u32 ph = 0; ph < 6; ++ph) {
            native_stats.c_no_ph[ph] += no_ph[ph];
        }
        ++native_stats.n_no;
    }
    segb_opened = !b.verify;
    if (native_pt_on && !b.verify) {
        fault_process_pending = true; // the pt shaders may have flagged pages (brick C v2)
    }
    return true;
}

void Rasterizer::A2BeginStretch(const GraphicsPipeline* pipeline) {
    if (native_batch.open) {
        NativeClose(15);
    }
    draw_repeat.valid = false;
    rs_memo.valid = false;
    PrepareRenderState(pipeline);
    const RenderState state = BeginRendering(pipeline);
    scheduler.BeginRendering(state);
}

void Rasterizer::A2EndStretch() {
    ResetBindings();
    auto& ds = scheduler.GetDynamicState();
    ds.Invalidate();
    ds.vertex_input_unknown = true;
    ds.packet_state_unknown = true;
    GraphicsBindTracker::Invalidate();
    scheduler.InvalidateNativeHeaps();
    viewport_epoch = 0;
    if (native_pt_on) {
        fault_process_pending = true; // the pt shaders may have flagged pages
    }
}

bool Rasterizer::A2SyncTexel(VAddr addr, u64 vsize) {
    const u64 size = memory->ClampRangeSize(addr, vsize);
    if (size == 0 || size > 0xffffffffull) {
        return false;
    }
    const auto [vb, vo] = buffer_cache.ObtainBuffer(addr, static_cast<u32>(size), false, true,
                                                    VideoCore::BufferId{},
                                                    Common::BufferSampleCaller::Storage);
    if (!vb || vb->CpuAddr() + vo != addr) {
        return false;
    }
    if (auto barrier = vb->GetBarrier(vk::AccessFlagBits2::eShaderRead,
                                      vk::PipelineStageFlagBits2::eAllCommands, 0,
                                      cur_dispatch_run)) {
        scheduler.EndRendering();
        scheduler.Record([b = *barrier](vk::CommandBuffer cmdbuf) {
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .bufferMemoryBarrierCount = 1,
                .pBufferMemoryBarriers = &b,
            });
        });
    }
    return true;
}

void Rasterizer::A2GlobalBarrier() {
    scheduler.EndRendering();
    scheduler.Record([](vk::CommandBuffer cmdbuf) {
        const vk::MemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &barrier,
        });
    });
}

bool Rasterizer::A2OcclusionOpen() const {
    return occlusion && occlusion->Open();
}

namespace {
int MatMemoMode() {
    static const int m = [] {
        const char* e = std::getenv("SHADPS4_A2_MAT_MEMO");
        return e ? std::atoi(e) : 0;
    }();
    return m;
}
constexpr size_t MatMemoSets = 4096; // l4e: 2 ways each
constexpr size_t MatMemoSize = MatMemoSets * 2;
size_t MatMemoSet(u64 h1) {
    return ((h1 ^ (h1 >> 31)) & (MatMemoSets - 1)) * 2;
}
} // namespace

bool Rasterizer::A2MaterialMemoValid(const MatMemo& m) {
    // The conditions under which BindlessResolve would hit every image and sampler entry.
    const auto& bl = bindless;
    if (m.img_gen != bl.img_gen || m.smp_gen != bl.smp_gen ||
        texture_cache.MetaClearEpoch() != bl.meta_epoch ||
        texture_cache.SamplerGeneration() != bl.sampler_gen) {
        return false;
    }
    for (u32 i = 0; i < m.n_img; ++i) {
        const auto& e = m.img[i];
        if (!texture_cache.BindlessEntryValid(e.id, e.uid, e.tex_gen, e.local, e.overlap)) {
            return false;
        }
        const auto& image = texture_cache.PeekImage(e.id);
        if (image.backing != e.backing || image.backing->state_serial != e.serial) {
            return false;
        }
    }
    return true;
}

bool Rasterizer::A2MaterialMemo(const GraphicsPipeline* base, u64 h1, u64 h2, u32 kn, u32* slots,
                                u32& n_img, u32& n_smp, std::vector<VideoCore::ImageId>& ids) {
    const int mode = MatMemoMode();
    if (mode == 0 || mode == 2 || !base || (h1 == 0 && h2 == 0) || mat_memo.empty() ||
        (VideoCore::TexTable::Mode() != 0 && VideoCore::TexTable::Mode() != 3)) {
        // q8: the table replaces the memo (q10: =3 keeps the memo for table misses)
        return false; // =2: A2ResolveMaterial verifies every hit against the full resolve
    }
    auto& st = native_stats;
    const u64 t0 = __rdtsc();
    const size_t set = MatMemoSet(h1);
    MatMemo* hit = nullptr;
    for (size_t w = 0; w < 2 && !hit; ++w) {
        auto& c = mat_memo[set + w];
        if (c.base == base && c.h1 == h1 && c.h2 == h2 && c.kn == kn) {
            hit = &c;
        }
    }
    if (!hit) {
        ++st.mm_miss;
        return false;
    }
    auto& m = *hit;
    if (!A2MaterialMemoValid(m)) {
        ++st.mm_stale;
        return false;
    }
    // l4e: the touches write gc_tick / AgeTick; skipped when this entry already wrote these
    // same values (the sampler generation is unchanged: A2MaterialMemoValid).
    const u64 gc = texture_cache.GcTickNow();
    const u64 age = texture_cache.AgeTickNow();
    const bool touch = m.touch_gc != gc || m.touch_age != age;
    if (touch && m.n_smp != 0 &&
        !texture_cache.TouchSamplers(std::span{m.lru.data(), m.n_smp}, bindless.sampler_gen)) {
        ++st.mm_stale;
        return false;
    }
    if (native_batch.open) {
        NativeClose(15); // as A2ResolveMaterial
    }
    std::array<VideoCore::ImageId, Shader::NativeBindlessImgCap> q4ids{};
    for (u32 i = 0; i < m.n_img; ++i) {
        if (touch) {
            texture_cache.TouchImageAccess(texture_cache.PeekImage(m.img[i].id));
        }
        ids.push_back(m.img[i].id);
        q4ids[i] = m.img[i].id;
    }
    Q4NoteMaterial(q4ids.data(), m.n_img);
    if (touch) {
        m.touch_gc = gc;
        m.touch_age = age;
    } else {
        ++st.mm_touch_skip;
    }
    m.used = ++mat_memo_clock;
    bindless.tex_gen = texture_cache.StructGeneration();
    n_img = m.n_img;
    n_smp = m.n_smp;
    std::memcpy(slots, m.slots.data(), sizeof(m.slots));
    ++st.mm_hit;
    st.c_mm_hit += __rdtsc() - t0;
    return true;
}

void Rasterizer::Q4NoteMaterial(const VideoCore::ImageId* ids, u32 n) {
    // q4 census: is every image of this material table-eligible (sampled only, ever)?
    static std::array<u64, 6> c{}; // materials, all eligible, images, rt/depth/storage, depth assoc, msaa
    ++c[0];
    bool all = true;
    for (u32 i = 0; i < n; ++i) {
        const auto& im = texture_cache.PeekImage(ids[i]);
        ++c[2];
        const bool used = im.usage.render_target || im.usage.depth_target || im.usage.storage;
        const bool dep = static_cast<bool>(im.depth_id);
        const bool ms = im.info.num_samples > 1;
        c[3] += used;
        c[4] += dep;
        c[5] += ms;
        all = all && !used && !dep && !ms;
    }
    c[1] += all;
    if ((c[0] & 0x3ffff) == 0) {
        LOG_INFO(Render_Vulkan,
                 "q4 table eligibility (cumulative): materials {} all-eligible {} | images {} "
                 "rt/depth/storage-used {} depth-associated {} msaa {}",
                 c[0], c[1], c[2], c[3], c[4], c[5]);
    }
}

bool Rasterizer::A2ResolveMaterial(const GraphicsPipeline* base, const Shader::BoundStage& fs,
                                   u32* slots, u32& n_img, u32& n_smp,
                                   std::vector<VideoCore::ImageId>& ids, u64 h1, u64 h2, u32 kn,
                                   const Shader::BoundStage* vs) {
    if (!base || !fs.info) {
        return false;
    }
    if (native_batch.open) {
        NativeClose(15); // a native batch may hold the current set's slots
    }
    const int mode = MatMemoMode();
    const bool memo = mode != 0 && !(h1 == 0 && h2 == 0);
    if (memo && mat_memo.empty()) {
        mat_memo.resize(MatMemoSize);
    }
    MatMemo* mm = nullptr;
    if (memo) {
        // l4e: the way holding this key, else the least recently used of the two.
        const size_t set = MatMemoSet(h1);
        for (size_t w = 0; w < 2 && !mm; ++w) {
            auto& c = mat_memo[set + w];
            if (c.base == base && c.h1 == h1 && c.h2 == h2 && c.kn == kn) {
                mm = &c;
            }
        }
        if (!mm) {
            mm = mat_memo[set].used <= mat_memo[set + 1].used ? &mat_memo[set] : &mat_memo[set + 1];
        }
    }
    // =2: a valid entry is compared with the full resolve below (taken before it refreshes).
    const bool vfy = mode == 2 && mm && mm->base == base && mm->h1 == h1 && mm->h2 == h2 &&
                     mm->kn == kn && A2MaterialMemoValid(*mm);
    MatMemo old{};
    if (vfy) {
        old = *mm;
    }
    std::array<VideoCore::ImageId, Shader::NativeBindlessImgCap> got{};
    MatMemo fill{};
    std::array<BindlessImgEntry, Shader::NativeBindlessImgCap> cap{};
    std::array<BindlessSmpEntry, Shader::NativeBindlessSmpCap> cap_s{}; // q8
    const bool tab = VideoCore::TexTable::Mode() != 0;
    const u64 tg0 = VideoCore::TexTable::Gen(); // q8b (D): before the resolve
    if (memo || tab) {
        bindless.cap_img = cap.data();
        bindless.cap_lru = fill.lru.data();
        bindless.cap_smp = cap_s.data();
    }
    const bool ok =
        BindlessResolve(base, slots, nullptr, nullptr, n_img, n_smp, true, &fs, got.data(), vs);
    bindless.cap_img = nullptr;
    bindless.cap_lru = nullptr;
    bindless.cap_smp = nullptr;
    if (!ok) {
        return false;
    }
    if (tab) {
        // q8: insert what the workers may look up - sampled-only images (never a render / depth
        // target or storage image, no depth association, one sample, not bound as a target).
        namespace TT = VideoCore::TexTable;
        const u64 tg = tg0;
        for (u32 i = 0; i < n_img; ++i) {
            const auto& e = cap[i];
            auto& im = texture_cache.GetImage(e.id);
            const bool ok_img = !im.usage.render_target && !im.usage.depth_target &&
                                !im.usage.storage && !im.depth_id && im.info.num_samples <= 1 &&
                                !im.info.props.is_depth && e.local && // q8b (C) + depth
                                !im.binding.is_target && !im.binding.force_general &&
                                !im.binding.needs_rebind;
            if (!ok_img) {
                TT::g_c[7].fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            // q8b (D): publish tab_gen, fence, then re-check what a concurrent mark could have
            // changed - a dirty mark after this point sees tab_gen and kills.
            // q10: the epoch is read first; a kill after it moves the epoch (entry stale).
            const u32 iid = static_cast<u32>(e.id.index);
            const u32 ep0 = TT::Ep(iid);
            std::atomic_ref<u32>{im.tab_id}.store(iid, std::memory_order_relaxed);
            std::atomic_ref<u64>{im.tab_gen}.store(tg, std::memory_order_release); // q10b
            std::atomic_thread_fence(std::memory_order_seq_cst);
            const auto flags = std::atomic_ref<VideoCore::ImageFlagBits>{im.flags}.load(
                std::memory_order_relaxed);
            const bool still =
                True(flags & VideoCore::ImageFlagBits::Registered) &&
                False(flags & VideoCore::ImageFlagBits::Dirty) && !im.depth_id &&
                im.backing == e.backing && im.backing->state_serial == e.serial &&
                std::atomic_ref<u64>{im.overlap_epoch}.load(std::memory_order_relaxed) ==
                    e.overlap &&
                TT::Gen() == tg && TT::Ep(iid) == ep0;
            if (!still) {
                TT::g_c[7].fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            TT::InsertImg(static_cast<u32>(reinterpret_cast<uintptr_t>(e.desc)), e.words, e.slot,
                          iid, tg, ep0);
            TT::g_c[4].fetch_add(1, std::memory_order_relaxed);
        }
        for (u32 j = 0; j < n_smp; ++j) {
            const auto& e = cap_s[j];
            TT::InsertSmp(e.words, e.slot, e.lru, tg);
        }
    }
    if (vfy) {
        auto& st = native_stats;
        ++st.mm_vfy;
        bool bad = old.n_img != n_img || old.n_smp != n_smp;
        for (u32 i = 0; !bad && i < n_img; ++i) {
            bad = old.slots[i] != slots[i] || old.img[i].id != got[i];
        }
        for (u32 j = 0; !bad && j < n_smp; ++j) {
            bad = old.slots[Shader::g_bl_img_max + j] != slots[Shader::g_bl_img_max + j];
        }
        if (bad && ++st.mm_bad <= 16) {
            LOG_WARNING(Render_Vulkan,
                        "A2 material memo verify: BAD ({} / {} images, {} / {} samplers)",
                        old.n_img, n_img, old.n_smp, n_smp);
        }
    }
    if (mm) {
        fill.base = base;
        fill.h1 = h1;
        fill.h2 = h2;
        fill.kn = kn;
        fill.img_gen = bindless.img_gen;
        fill.smp_gen = bindless.smp_gen;
        fill.n_img = n_img;
        fill.n_smp = n_smp;
        std::memcpy(fill.slots.data(), slots, sizeof(fill.slots));
        for (u32 i = 0; i < n_img; ++i) {
            const auto& e = cap[i];
            fill.img[i] = {e.id, e.local, e.uid, e.tex_gen, e.overlap, e.backing, e.serial};
        }
        // The full resolve just touched everything at these ticks.
        fill.touch_gc = texture_cache.GcTickNow();
        fill.touch_age = texture_cache.AgeTickNow();
        fill.used = ++mat_memo_clock;
        *mm = fill;
    }
    ids.insert(ids.end(), got.begin(), got.begin() + n_img);
    Q4NoteMaterial(got.data(), n_img);
    return true;
}

void Rasterizer::A2TexTableTouch() {
    // q8: the table's materials skip the per-material LRU touches; touch every current entry's
    // image / sampler at most every 4 ms instead (an evicted image is unregistered: a kill).
    namespace TT = VideoCore::TexTable;
    static auto last = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::milliseconds(16)) { // q11b: 16384 entries (was 4096 / 4 ms)
        return;
    }
    last = now;
    const u64 tg = TT::Gen();
    for (const auto& e : TT::g_img) {
        if (e.seq.load(std::memory_order_relaxed) != 0 && e.gen == tg &&
            (!TT::V2() || TT::Ep(e.id) == e.ep)) { // q10b: not a killed entry
            const VideoCore::ImageId id{e.id};
            if (texture_cache.IsImageAllocated(id)) {
                texture_cache.TouchImageAccess(texture_cache.PeekImage(id));
            }
        }
    }
    std::vector<size_t> lru;
    lru.reserve(256);
    for (const auto& e : TT::g_smp) {
        if (e.seq.load(std::memory_order_relaxed) != 0 && e.gen == tg) {
            lru.push_back(static_cast<size_t>(e.lru));
        }
    }
    if (!lru.empty() &&
        !texture_cache.TouchSamplers(std::span{lru.data(), lru.size()}, bindless.sampler_gen)) {
        TT::Kill();
    }
}

vk::DescriptorSet Rasterizer::A2BindlessCommit() {
    bindless.last_tick[bindless.cur] = scheduler.CurrentTick();
    return Bindless::G().sets[bindless.cur];
}

bool Rasterizer::A2ImagesSafe(const std::vector<VideoCore::ImageId>& ids) const {
    // NativeOpen's colour rule (Opus review 2026-10-09 #1): a sampled image that is a bound
    // target (feedback loop) or forced to the general layout needs per-draw barriers.
    for (const auto id : ids) {
        const auto& image = texture_cache.PeekImage(id);
        if ((image.binding.is_target && image.binding.is_bound) || image.binding.force_general) {
            return false;
        }
    }
    return true;
}

void Rasterizer::NativeClose(u32 why) {
    native_batch.open = false;
    native_batch.cand_valid = false;
    native_stats.close_key = false;
    ++native_stats.close[std::min<u32>(why, 15)];
    if (native_prep.active) [[unlikely]] {
        if (native_prep.verify && draw_job_tag == native_prep.tag &&
            draw_job_idx == native_prep.next_idx) {
            ++native_stats.prep_join_refused; // the join refused a member the open prepared
            ++native_stats.prep_jref[std::min<u32>(why, 14)];
            // 15: of the close-10 refusals (the join recorded), those after a CPU write since the
            // prepare (a race the confirm check would have seen a moment later, inferred).
            if (why == 10 &&
                VideoCore::g_dirty_epoch.load(std::memory_order_acquire) != native_prep.epoch) {
                ++native_stats.prep_jref[15];
            }
        }
        native_prep.active = false;
    }
}

bool Rasterizer::BindlessRotate() {
    // The next set copy, once the GPU finished every batch that bound it; caches and slot maps
    // start over.
    auto& bl = bindless;
    const u32 next = (bl.cur + 1) % Bindless::NumSets;
    if (!scheduler.IsFree(bl.last_tick[next])) {
        ++native_stats.bl_fail[3];
        return false;
    }
    bl.cur = next;
    A2::g_bindless_cur.store(next, std::memory_order_release); // p2: the workers' secondaries
    bl.img_next = bl.smp_next = 0;
    bl.view_slot.clear();
    bl.smp_slot.clear();
    ++bl.img_gen;
    ++bl.smp_gen;
    ++native_stats.bl_rot;
    VideoCore::TexTable::Kill(); // q8: every slot is reassigned
    return true;
}

void Rasterizer::BindlessRingWrite(u32 tag, u32 idx, const u32* slots) {
    // Ring mode: the GPU reads this draw's record from the job ring; the slot words (128-159)
    // are the back's (the job thread writes dwords 0-127 and the guest-address hi bytes
    // 160-175). Write-only (write-combined).
    namespace NR = AmdGpu::NativeRec;
    u32* gr = NR::g_gpu_recs[tag >> 28].load(std::memory_order_acquire);
    if (gr) {
        std::memcpy(gr + size_t(idx & (NR::RingSize - 1)) * NR::GpuRecDwords +
                        Shader::NativeRecordImgOffset,
                    slots, sizeof(u32) * Shader::NativeBindlessSlots);
    }
    static const bool a2_verify = Vulkan::A2::Mode() == 2;
    if (a2_verify) [[unlikely]] {
        a2_probe.tag = tag;
        a2_probe.idx = idx;
        std::memcpy(a2_probe.slots.data(), slots, sizeof(u32) * Shader::NativeBindlessSlots);
        ++a2_probe.n;
        if (const auto it = a2_expect.find(u64{tag} << 32 | idx); it != a2_expect.end()) {
            const auto& ex = it->second;
            bool bad = false;
            for (u32 i = 0; !bad && i < ex.n_img; ++i) {
                bad = slots[i] != ex.slots[i];
            }
            for (u32 j = 0; !bad && j < ex.n_smp; ++j) {
                bad = slots[Shader::g_bl_img_max + j] != ex.slots[Shader::g_bl_img_max + j];
            }
            ++a2_slot_cmp;
            if (bad && ++a2_slot_bad <= 16) {
                LOG_WARNING(Render_Vulkan, "A2 verify: slot words differ (tag {:#x} idx {}, {} images {} samplers)",
                            tag, idx, ex.n_img, ex.n_smp);
            }
            a2_expect.erase(it);
        }
    }
}

bool Rasterizer::BindlessResolve(const GraphicsPipeline* base, u32* slots, vk::ImageView* views,
                                 vk::Sampler* samplers, u32& n_img, u32& n_smp, bool retry,
                                 const Shader::BoundStage* fs_override, VideoCore::ImageId* ids,
                                 const Shader::BoundStage* vs_override) {
    // SHADPS4_NATIVE_BINDLESS (sec. 12), back thread. Per fragment image: the cache entry for
    // (descriptor, raw T#) if every validity condition holds (then only the LRU touch), else
    // what BindTextures does for one sampled image - FindImage, FindTexture, the sampled-layout
    // Transit (may record a barrier: callers check RecordSeq) - and a slot for (view, layout).
    // Per sampler: the S# (with BindTextures' aniso rule) -> GetSampler -> a slot.
    auto& bl = bindless;
    auto& st = native_stats;
    const auto& g = Bindless::G();
    const u64 tg = texture_cache.StructGeneration();
    const u64 ie = VideoCore::g_image_epoch.load(std::memory_order_acquire);
    const u64 me = texture_cache.MetaClearEpoch();
    const u64 sg = texture_cache.SamplerGeneration();
    // Per-image validity (s431: a global texture-cache generation / image epoch wiped every
    // entry a few times per frame, 30% misses, -7% city): BindlessEntryValid per entry; only a
    // meta clear invalidates everything. Slot keys carry the image uid and backing, so a dead
    // view's handle value cannot alias a live one.
    (void)ie;
    if (me != bl.meta_epoch) {
        ++bl.img_gen;
        bl.meta_epoch = me;
        VideoCore::TexTable::Kill(); // q8
    }
    bl.tex_gen = tg;
    if (sg != bl.sampler_gen) {
        VideoCore::TexTable::Kill(); // q8
        bl.smp_slot.clear();
        ++bl.smp_gen;
        bl.sampler_gen = sg;
    }
    if (bl.img_cache.empty()) {
        bl.img_cache.resize(16384);
        bl.smp_cache.resize(1024);
    }
    // Brick C v3 item 4: a segment material's representative member supplies its own fragment
    // stage (user data + flat buffer from its record); the Info is the pipeline's.
    const Shader::BoundStage* fs = fs_override && fs_override->info ? fs_override : nullptr;
    for (const auto& bound : BoundStages(base)) {
        if (!fs && bound.info && bound.info->l_stage == Shader::LogicalStage::Fragment) {
            fs = &bound;
        }
    }
    if (!fs) {
        ++st.bl_fail[0];
        return false;
    }
    // q3 (SHADPS4_NATIVE_BINDLESS_VS): the VS's images / samplers after the FS's.
    const Shader::BoundStage* vsb = nullptr;
    if (Shader::g_native_bindless_vs) {
        vsb = vs_override && vs_override->info ? vs_override : nullptr;
        if (!vsb) {
            for (const auto& bound : BoundStages(base)) {
                if (bound.info && bound.info->l_stage == Shader::LogicalStage::Vertex) {
                    vsb = &bound;
                }
            }
        }
        if (vsb && vsb->info->images.empty() && vsb->info->samplers.empty()) {
            vsb = nullptr;
        }
    }
    std::array<const Shader::ImageResource*, Shader::NativeBindlessImgCap> img_d{};
    std::array<const Shader::BoundStage*, Shader::NativeBindlessImgCap> img_b{};
    std::array<const Shader::SamplerResource*, Shader::NativeBindlessSmpCap> smp_d{};
    std::array<const Shader::BoundStage*, Shader::NativeBindlessSmpCap> smp_b{};
    n_img = 0;
    n_smp = 0;
    for (const Shader::BoundStage* sb : {fs, vsb}) {
        if (!sb) {
            continue;
        }
        if (n_img + sb->info->images.size() > Shader::g_bl_img_max ||
            n_smp + sb->info->samplers.size() > Shader::g_bl_smp_max) { // q54 split
            ++st.bl_fail[0];
            return false;
        }
        for (const auto& d : sb->info->images) {
            img_d[n_img] = &d;
            img_b[n_img++] = sb;
        }
        for (const auto& d : sb->info->samplers) {
            smp_d[n_smp] = &d;
            smp_b[n_smp++] = sb;
        }
    }
    const auto device = instance.GetDevice();
    const auto rotate_retry = [&] {
        return retry && BindlessRotate() &&
               BindlessResolve(base, slots, views, samplers, n_img, n_smp, false, fs_override,
                               ids, vs_override);
    };
    for (u32 i = 0; i < n_img; ++i) {
        const auto& desc = *img_d[i];
        const auto tsharp = desc.GetSharp(*img_b[i]);
        if (tsharp.Address() == 0 || tsharp.GetDataFmt() == AmdGpu::DataFormat::FormatInvalid) {
            ++st.bl_fail[1];
            return false;
        }
        std::array<u32, 8> words;
        static_assert(sizeof(tsharp) == sizeof(words));
        std::memcpy(words.data(), &tsharp, sizeof(words));
        // Keyed by the T# words + the descriptor bits ImageDesc depends on (BindTextures'
        // decoded-image cache uses the same three), so pipelines sharing a texture share it.
        const void* dkey = reinterpret_cast<const void*>(
            uintptr_t(desc.is_depth) | uintptr_t(desc.is_array) << 1 | uintptr_t(desc.is_r128) << 2 |
            uintptr_t(1) << 8);
        u64 hsh = std::bit_cast<u64>(dkey) * 0x9E3779B97F4A7C15ull;
        for (const u32 w : words) {
            hsh = (hsh ^ w) * 0x100000001B3ull;
        }
        auto& e = bl.img_cache[(hsh ^ (hsh >> 29)) & (bl.img_cache.size() - 1)];
        const u64 tc0 = __rdtsc();
        if (e.gen == bl.img_gen && e.desc == dkey && e.words == words &&
            texture_cache.BindlessEntryValid(e.id, e.uid, e.tex_gen, e.local, e.overlap)) {
            auto& image = texture_cache.PeekImage(e.id);
            if (image.backing == e.backing && image.backing->state_serial == e.serial) {
                texture_cache.TouchImageAccess(image);
                slots[i] = e.slot;
                if (views) {
                    views[i] = e.view;
                }
                if (ids) {
                    ids[i] = e.id;
                }
                if (bl.cap_img) {
                    bl.cap_img[i] = e;
                }
                ++st.bl_img_hit;
                st.c_bl_hit += __rdtsc() - tc0;
                continue;
            }
        }
        ++st.bl_img_miss;
        VideoCore::TextureCache::ImageDesc idesc{tsharp, desc};
        const auto image_id = texture_cache.FindImage(idesc);
        auto& image = texture_cache.GetImage(image_id);
        if (texture_cache.GetAssociatedDepth(image) || image.binding.needs_rebind ||
            image.binding.is_target || image.binding.force_general) {
            ++st.bl_fail[2];
            return false;
        }
        auto& view = texture_cache.FindTexture(image_id, idesc);
        image.Transit(image.info.props.is_depth ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                                : vk::ImageLayout::eShaderReadOnlyOptimal,
                      vk::AccessFlagBits2::eShaderRead, idesc.view_info.range);
        image.usage.texture |= true;
        const vk::ImageView vh = *view.image_view;
        const vk::ImageLayout lay = image.backing->state.layout;
        const BindlessState::ViewKey vkey{image.image_uid, image.backing,
                                          std::bit_cast<u64>(static_cast<VkImageView>(vh)),
                                          static_cast<u32>(lay)};
        u32 slot;
        if (const auto it = bl.view_slot.find(vkey); it != bl.view_slot.end()) {
            slot = it->second;
        } else {
            if (bl.img_next >= Bindless::ImageSlots) {
                return rotate_retry();
            }
            slot = bl.img_next++;
            const vk::DescriptorImageInfo ii{{}, vh, lay};
            const vk::WriteDescriptorSet w{
                .dstSet = g.sets[bl.cur],
                .dstBinding = 0,
                .dstArrayElement = slot,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eSampledImage,
                .pImageInfo = &ii,
            };
            device.updateDescriptorSets(w, {});
            bl.view_slot.emplace(vkey, slot);
        }
        e = {dkey,
             words,
             bl.img_gen,
             slot,
             image_id,
             image.image_uid,
             texture_cache.StructGeneration(),
             std::atomic_ref<u64>{image.overlap_epoch}.load(std::memory_order_relaxed),
             // Contained (not only exact): an image registered on any page of this T#'s range
             // is on the found image's pages too and bumps its overlap epoch.
             idesc.info.guest_address >= image.info.guest_address &&
                 idesc.info.guest_address + idesc.info.guest_size <=
                     image.info.guest_address + image.info.guest_size,
             image.backing,
             image.backing->state_serial,
             vh,
             lay};
        if (bl.cap_img) {
            bl.cap_img[i] = e;
        }
        slots[i] = slot;
        if (views) {
            views[i] = vh;
        }
        if (ids) {
            ids[i] = image_id;
        }
        st.c_bl_miss += __rdtsc() - tc0;
    }
    std::array<size_t, Shader::NativeBindlessSmpCap> lru{};
    u32 n_lru = 0;
    const u64 ts0 = __rdtsc();
    for (u32 j = 0; j < n_smp; ++j) {
        const auto& sdesc = *smp_d[j];
        const auto& sbound = *smp_b[j];
        auto ssharp = sdesc.GetSharp(sbound);
        if (sdesc.disable_aniso) {
            const auto& tsharp = sbound.info->images[sdesc.associated_image].GetSharp(sbound);
            if (tsharp.base_level == 0 && tsharp.last_level == 0) {
                ssharp.max_aniso.Assign(AmdGpu::AnisoRatio::One);
            }
        }
        std::array<u32, 4> words;
        static_assert(sizeof(ssharp) == sizeof(words));
        std::memcpy(words.data(), &ssharp, sizeof(words));
        u64 hsh = 0xC2B2AE3D27D4EB4Full;
        for (const u32 w : words) {
            hsh = (hsh ^ w) * 0x100000001B3ull;
        }
        auto& e = bl.smp_cache[(hsh ^ (hsh >> 29)) & (bl.smp_cache.size() - 1)];
        if (e.gen == bl.smp_gen && e.words == words) {
            slots[Shader::g_bl_img_max + j] = e.slot;
            if (samplers) {
                samplers[j] = e.sampler;
            }
            lru[n_lru++] = e.lru;
            if (bl.cap_lru) {
                bl.cap_lru[j] = e.lru;
            }
            if (bl.cap_smp) {
                bl.cap_smp[j] = e; // q8
            }
            ++st.bl_smp_hit;
            continue;
        }
        ++st.bl_smp_miss;
        const vk::Sampler sh =
            texture_cache.GetSampler(ssharp, liverpool->BackRegs().ta_bc_base, nullptr);
        const u64 skey = std::bit_cast<u64>(static_cast<VkSampler>(sh));
        u32 slot;
        if (const auto it = bl.smp_slot.find(skey); it != bl.smp_slot.end()) {
            slot = it->second;
        } else {
            if (bl.smp_next >= Bindless::SamplerSlots) {
                return rotate_retry();
            }
            slot = bl.smp_next++;
            const vk::DescriptorImageInfo ii{sh, {}, vk::ImageLayout::eUndefined};
            const vk::WriteDescriptorSet w{
                .dstSet = g.sets[bl.cur],
                .dstBinding = 1,
                .dstArrayElement = slot,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eSampler,
                .pImageInfo = &ii,
            };
            device.updateDescriptorSets(w, {});
            bl.smp_slot.emplace(skey, slot);
        }
        e = {words, bl.smp_gen, slot, texture_cache.SamplerLruId(ssharp), sh};
        if (bl.cap_lru) {
            bl.cap_lru[j] = e.lru;
        }
        if (bl.cap_smp) {
            bl.cap_smp[j] = e; // q8
        }
        slots[Shader::g_bl_img_max + j] = slot;
        if (samplers) {
            samplers[j] = sh;
        }
    }
    // Cache hits skipped GetSampler's LRU touch; keep those samplers alive the same way.
    const u64 ts1 = __rdtsc();
    const bool touched =
        n_lru == 0 || texture_cache.TouchSamplers(std::span{lru.data(), n_lru}, sg);
    st.c_bl_smp += ts1 - ts0;
    st.c_bl_touch += __rdtsc() - ts1;
    ++st.n_bl_res;
    if (!touched) {
        ++st.bl_fail[4];
        return false;
    }
    return true;
}

const VideoCore::Buffer* Rasterizer::NativeSpanSync(VAddr lo, VAddr hi,
                                                   VideoCore::NativeRunAccess access) {
    // One range validated through the ring's validity cache (granule clocks, sec. 10): a
    // superset validated since its granules last changed -> no sync walk; else the lean sync.
    // The access barrier always. Null: not inside one cached buffer (the generic path creates /
    // joins buffers).
    const VideoCore::Buffer* b0 = buffer_cache.NativeBufferAt(lo, hi - lo);
    if (!b0 || !b0->IsInBounds(lo, hi - lo)) {
        return nullptr;
    }
    if (buffer_cache.CoherentSkip(false, false)) {
        // SHADPS4_NATIVE_COHERENT (sec. 17): memory is coherent for the submission - no span
        // cache, no granule clocks, no sync; the lookup above and the access barrier.
        buffer_cache.NoteCoherentSpanSkip();
        const VideoCore::Buffer* buf = buffer_cache.NativeRunSpan(
            lo, hi, access, &buffer_barriers, cur_dispatch_run, false, -1, nullptr);
        if (buf != b0) {
            return nullptr;
        }
        if (VideoCore::g_native_coherent == 2) [[unlikely]] {
            buffer_cache.CoherentVerify(const_cast<VideoCore::Buffer&>(*buf), lo,
                                        static_cast<u32>(hi - lo));
        }
        return buf;
    }
    const u64 handle = std::bit_cast<u64>(static_cast<VkBuffer>(b0->Handle()));
    const u64 gen = buffer_cache.StructGeneration();
    const u64 clock0 = VideoCore::g_granule_clock.load(std::memory_order_acquire);
    NativeSpanEntry* hit = nullptr;
    for (auto& e : native_span_cache) {
        if (e.handle == handle && e.access == static_cast<u32>(access) && e.gen == gen &&
            lo >= e.lo && hi <= e.hi && VideoCore::GranulesUnchangedSince(lo, hi, e.epoch)) {
            hit = &e;
            break;
        }
    }
    bool gpu_mod = false;
    const VideoCore::Buffer* buf =
        buffer_cache.NativeRunSpan(lo, hi, access, &buffer_barriers, cur_dispatch_run,
                                   hit == nullptr, hit ? (hit->gpu_mod ? 1 : 0) : -1, &gpu_mod);
    if (!buf || buf != b0) {
        return nullptr;
    }
    if (!hit) {
        NativeSpanEntry* slot = nullptr;
        for (auto& e : native_span_cache) {
            if (e.handle == handle && e.access == static_cast<u32>(access)) {
                slot = &e;
                break;
            }
        }
        if (slot && slot->gen == gen && lo <= slot->hi && hi >= slot->lo &&
            std::max(slot->hi, hi) - std::min(slot->lo, lo) <= (u64{64} << 20) &&
            VideoCore::GranulesUnchangedSince(slot->lo, slot->hi, slot->epoch)) {
            slot->lo = std::min(slot->lo, lo);
            slot->hi = std::max(slot->hi, hi);
            slot->gpu_mod = slot->gpu_mod || gpu_mod;
        } else {
            if (!slot) {
                slot = &native_span_cache[native_span_rr++ % native_span_cache.size()];
            }
            *slot = {handle, static_cast<u32>(access), lo, hi, clock0, gen, gpu_mod};
        }
    }
    return buf;
}

const GraphicsPipeline* Rasterizer::NativeDirectCandidate(
    const GraphicsPipeline* pipeline, bool is_indexed,
    const AmdGpu::CpSplit::VertexInputsPayload* vi) {
    if (g_native_direct == 0 || VideoCore::g_native_pass != 1 || !is_indexed || !vi ||
        !VideoCore::g_native_colour || AmdGpu::NativeRec::g_gpu_mode != 1 || draw_job_tag == 0 ||
        native_bindings || !build_set_writes || (occlusion && occlusion->Open()) ||
        pipeline->native_variant.load(std::memory_order_relaxed) != nullptr ||
        pipeline->native_colour[0].key.load(std::memory_order_relaxed) == 0) {
        return nullptr;
    }
    const auto* attribs = AmdGpu::CpSplit::VertexInputAttribs(*vi);
    const auto* v = pipeline->NativeColourVariant(VideoCore::NativeVertexKey(attribs, vi->count));
    if (!v || !v->native_bindless ||
        v->native_table_binding == GraphicsPipeline::NoNativeRecords) {
        return nullptr;
    }
    // The normal path's single-draw rule (advisor ~01:15: without it, singles became openers
    // at ~13 k instead of generic draws at ~7.9 k): open only if the next queued draw shares the
    // pipeline.
    if (segx_left == 0 && liverpool->NativePeekNextDraw(pipeline) == 0) {
        ++native_stats.nd_fail[10];
        return nullptr;
    }
    // NativeOpen's colour rule (every vertex stream in one VkBuffer), checked here so a refusal
    // costs no bind.
    const VideoCore::Buffer* vb0 = nullptr;
    for (u32 k = 0; k < vi->count && k < 16; ++k) {
        if (attribs[k].base_address == 0 || attribs[k].size == 0) {
            continue;
        }
        const VideoCore::Buffer* vb = buffer_cache.NativeBufferAt(attribs[k].base_address, attribs[k].size);
        if (!vb || (vb0 && vb != vb0)) {
            ++native_stats.nd_fail[11];
            return nullptr;
        }
        vb0 = vb;
    }
    ++native_stats.nd_cand;
    return v;
}

bool Rasterizer::NativeDirectBind(const GraphicsPipeline* base, const GraphicsPipeline* native,
                                  u32 index_offset,
                                  const AmdGpu::CpSplit::VertexInputsPayload* front_vi,
                                  bool open) {
    // Back thread, DrawTail in place of BindResources (sec. 15). The draw's own V#s give set 0
    // (whole-range Guest buffers, the generic alignment / push offsets so NativeOpen reads them
    // like a generic bind); images and samplers are not written (the bindless PS samples the
    // global arrays at the record's slots). Then the vertex / index bindings from the front
    // and NativeOpen (ring mode, bindless) - all before the rendering state, so any failure
    // leaves the generic BindResources to run.
    namespace NR = AmdGpu::NativeRec;
    auto& st = native_stats;
    const auto fail = [&](u32 why) {
        ++st.nd_fail[std::min<u32>(why, 15)];
        return false;
    };
    const NR::Record* ring = NR::g_rings[draw_job_tag >> 28].load(std::memory_order_acquire);
    if (!ring) {
        return fail(1);
    }
    const NR::Record& r0 = ring[draw_job_idx & (NR::RingSize - 1)];
    const u32 gpu_ok = r0.gpu_ok;
    const u64 inh_slots = r0.inh_slots;
    const void* cmd = r0.cmd;
    std::atomic_thread_fence(std::memory_order_acquire);
    if (r0.stamp.load(std::memory_order_relaxed) != draw_job_idx + 1 || cmd != draw_job_cmd ||
        gpu_ok != 1) {
        return fail(2);
    }
    if (inh_slots != 0) {
        return fail(3); // v1: inherited V#s are outside the union and not synced here
    }
    if (!pending_stencil_writebacks.empty()) {
        return fail(4);
    }
    if (open && native_batch.open) {
        NativeClose(1);
    }
    set_write_index = 0;
    set_writes.clear();
    p1c_bind_serial = p1c_dispatch_serial;
    buffer_barriers.clear();
    if (!native_carry.empty()) [[unlikely]] {
        buffer_barriers.insert(buffer_barriers.end(), native_carry.begin(), native_carry.end());
        native_carry.clear();
    }
    buffer_infos.clear();
    image_infos.clear();
    binding_compute = false;
    const auto& regs = liverpool->BackRegs();
    Shader::Backend::Bindings binding{};
    u64 nd_t = __rdtsc();
    const auto nd_lap = [&](u32 ph) {
        const u64 now = __rdtsc();
        st.c_nd_ph[ph] += now - nd_t;
        nd_t = now;
    };
    push_data = MakeUserData(regs);
    for (const auto& bound : BoundStages(base)) {
        const auto* stage = bound.info;
        if (!stage) {
            continue;
        }
        bound.PushUd(binding, push_data);
        for (const auto& desc : stage->buffers) {
            if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                // As BindBuffers: a null descriptor when no user plane is enabled, else a copy.
                const u32 alignment = instance.UniformMinAlignment();
                if (regs.clipper_control.user_clip_plane_enable == 0) {
                    buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
                } else {
                    auto& sb = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                    std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = regs.clip_user_data[i];
                        planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
                        planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
                        planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
                        planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
                    }
                    const u32 ubo_size = static_cast<u32>(sizeof(planes));
                    const u64 off = sb.Copy(planes.data(), ubo_size, alignment);
                    buffer_infos.emplace_back(sb.Handle(), off, ubo_size);
                }
                auto& w = set_writes.emplace_back();
                w.dstSet = VK_NULL_HANDLE;
                w.dstBinding = binding.unified++;
                w.dstArrayElement = 0;
                w.descriptorCount = 1;
                w.descriptorType = desc.IsStorage(desc.GetSharp(bound))
                                       ? vk::DescriptorType::eStorageBuffer
                                       : vk::DescriptorType::eUniformBuffer;
                w.pBufferInfo = &buffer_infos.back();
                ++binding.buffer;
                continue;
            }
            if (desc.buffer_type != Shader::BufferType::Guest || desc.is_written ||
                desc.inline_cbuf) {
                return fail(5);
            }
            const auto vsharp = desc.GetSharp(bound);
            if (vsharp.base_address == 0 || vsharp.GetSize() == 0) {
                return fail(6);
            }
            const u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
            const bool is_storage = desc.IsStorage(vsharp);
            const u32 alignment =
                is_storage ? instance.StorageMinAlignment() : instance.UniformMinAlignment();
            const VideoCore::Buffer* buf = nullptr;
            if (desc.is_formatted) {
                // Typed reads: the full lookup (an image -> buffer sync of overlapping image
                // data), as the join does; a recorded copy is caught by NativeOpen's checks.
                const auto [vb, vo] = buffer_cache.ObtainBuffer(
                    vsharp.base_address, static_cast<u32>(size), false, true, VideoCore::BufferId{},
                    Common::BufferSampleCaller::Storage);
                if (vb && vb->CpuAddr() + vo == vsharp.base_address) {
                    if (auto barrier = vb->GetBarrier(vk::AccessFlagBits2::eShaderRead,
                                                      vk::PipelineStageFlagBits2::eAllCommands, 0,
                                                      cur_dispatch_run)) {
                        buffer_barriers.emplace_back(*barrier);
                    }
                    buf = vb;
                }
            } else {
                buf = NativeSpanSync(vsharp.base_address, vsharp.base_address + size,
                                     VideoCore::NativeRunAccess::Storage);
            }
            if (!buf) {
                return fail(7);
            }
            const u64 offset = vsharp.base_address - buf->CpuAddr();
            const u64 offset_aligned = Common::AlignDown(offset, alignment);
            const u32 adjust = static_cast<u32>(offset - offset_aligned);
            if (adjust % 4 != 0 || adjust >= 256 || offset_aligned > 0xffffffffull) {
                return fail(8);
            }
            push_data.AddOffset(binding.buffer, adjust);
            buffer_infos.emplace_back(buf->Handle(), offset_aligned, size + adjust);
            auto& w = set_writes.emplace_back();
            w.dstSet = VK_NULL_HANDLE;
            w.dstBinding = binding.unified++;
            w.dstArrayElement = 0;
            w.descriptorCount = 1;
            w.descriptorType =
                is_storage ? vk::DescriptorType::eStorageBuffer : vk::DescriptorType::eUniformBuffer;
            w.pBufferInfo = &buffer_infos.back();
            ++binding.buffer;
        }
        for (const auto& desc : stage->images) {
            binding.unified += desc.NumBindings(bound);
        }
        binding.unified += static_cast<u32>(stage->samplers.size());
    }
    set_write_index = static_cast<u32>(set_writes.size());
    native_bound_count = binding.unified;
    nd_lap(0);
    // Vertex / index bindings as a join computes them (syncs + GPU-written barriers).
    VideoCore::NativeJoinVi jvi{};
    buffer_cache.NativeJoinVertexIndex(*front_vi, index_offset, jvi, &buffer_barriers);
    auto& tvi = VideoCore::t_native_vi;
    tvi.n = jvi.n;
    tvi.buf = jvi.buf;
    tvi.off = jvi.off;
    tvi.stride = jvi.stride;
    tvi.ib_buf = jvi.ib_buf;
    tvi.ib_off = jvi.ib_off;
    tvi.ib_type =
        regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16 ? 2u : 4u;
    tvi.gpu_mod = false;
    const auto* attribs = AmdGpu::CpSplit::VertexInputAttribs(*front_vi);
    for (u32 k = 0; k < front_vi->count && k < 16; ++k) {
        if (attribs[k].base_address != 0 && attribs[k].size != 0 &&
            buffer_cache.IsRegionGpuModified(attribs[k].base_address, attribs[k].size)) {
            tvi.gpu_mod = true;
        }
    }
    nd_lap(1);
    if (!open) {
        return true;
    }
    attachment_feedback_loop = false; // BeginRendering recomputes it; the slot resolve refuses
                                      // any image that is this draw's target
    const bool opened = NativeOpen(base, native, true, index_offset, front_vi, true);
    nd_lap(2);
    if (!opened) {
        return fail(9);
    }
    draw_repeat.valid = false;
    ++st.nd_ok;
    return true;
}

bool Rasterizer::NativeRingJoinOk() {
    // Back thread, NativeJoin of a ring batch: the draw's job record must be the batch's next
    // ring entry (the multi-draw reads draw i at first + i) with a GPU record and the opener's
    // inherited set / values. The join's own checks (same VkBuffers, image key, vertex / index)
    // follow as for a legacy batch; then the job record's guest addresses minus the table's
    // per-buffer bases are the join's offsets.
    namespace NR = AmdGpu::NativeRec;
    auto& b = native_batch;
    auto& st = native_stats;
    if (g_native_ring_join == 0 || draw_job_tag != b.ring_tag ||
        draw_job_idx != b.ring_idx0 + b.n) {
        ++st.rj_fail[0];
        return false;
    }
    const NR::Record* ring = NR::g_rings[draw_job_tag >> 28].load(std::memory_order_acquire);
    if (!ring) {
        ++st.rj_fail[1];
        return false;
    }
    const NR::Record& r = ring[draw_job_idx & (NR::RingSize - 1)];
    const u32 gpu_ok = r.gpu_ok, push_inh = r.push_inh, inh_vs = r.inh_vs, inh_ps = r.inh_ps;
    const u64 inh_slots = r.inh_slots;
    const void* cmd = r.cmd;
    std::atomic_thread_fence(std::memory_order_acquire);
    if (r.stamp.load(std::memory_order_relaxed) != draw_job_idx + 1 || cmd != draw_job_cmd ||
        gpu_ok != 1) {
        ++st.rj_fail[1];
        return false;
    }
    if (push_inh != b.ring_push_inh || inh_slots != b.ring_inh_slots ||
        (inh_vs & 0xffffu) != b.ring_inh_vs || (inh_ps & 0xffffu) != b.ring_inh_ps) {
        ++st.rj_fail[2];
        return false;
    }
    const auto& regs = liverpool->BackRegs();
    const auto& p = native_prep;
    for (u32 m = b.ring_inh_vs; m != 0; m &= m - 1) {
        const u32 i = static_cast<u32>(std::countr_zero(m));
        if (regs.vs_program.user_data[i] != p.inh_val_vs[i]) {
            ++st.rj_fail[3];
            return false;
        }
    }
    for (u32 m = b.ring_inh_ps; m != 0; m &= m - 1) {
        const u32 i = static_cast<u32>(std::countr_zero(m));
        if (regs.ps_program.user_data[i] != p.inh_val_ps[i]) {
            ++st.rj_fail[3];
            return false;
        }
    }
    return true;
}

bool Rasterizer::NativeJoin(const SplitDraw& draw,
                            const AmdGpu::CpSplit::VertexInputsPayload* front_vi) {
    // Back thread, ExecuteDraw after the prologue. True: the draw went into the open batch.
    auto& b = native_batch;
    const auto& regs = liverpool->BackRegs();
    b.cand_valid = false;
    if (b.sealed) [[unlikely]] {
        // SHADPS4_NATIVE_SEGX: an executed segment is complete; the next draw opens its own.
        ++native_stats.segx_join_refused;
        NativeClose(15);
        return false;
    }
    if (draw.pipeline != b.base) {
        // Segment census (sec. 14): would a GPU-driven segment survive this pipeline change?
        const auto& ka = b.base->GetGraphicsKey();
        const auto& kb = draw.pipeline->GetGraphicsKey();
        constexpr size_t fixed0 = offsetof(GraphicsPipelineKey, patch_control_points);
        const bool fixed_eq =
            std::memcmp(reinterpret_cast<const u8*>(&ka) + fixed0,
                        reinterpret_cast<const u8*>(&kb) + fixed0, sizeof(ka) - fixed0) == 0;
        const bool vf_eq = ka.vertex_buffer_formats == kb.vertex_buffer_formats;
        const bool nat = draw.pipeline->native_variant.load(std::memory_order_relaxed) != nullptr ||
                         draw.pipeline->native_colour[0].key.load(std::memory_order_relaxed) != 0;
        const bool ctx = liverpool->BackContextEpoch() != b.ctx_epoch;
        ++native_stats.seg_c2[(fixed_eq ? 1u : 0u) | (vf_eq ? 2u : 0u) | (nat ? 4u : 0u) |
                              (ctx ? 8u : 0u)];
        NativeClose(2);
        return false;
    }
    if (b.ring && !NativeRingJoinOk()) {
        // SHADPS4_NATIVE_GPU_DIRECT: a ring batch reads every record from the job ring; only a
        // draw whose job record is the next ring entry can join (ring-join, sec. 11).
        NativeClose(13);
        return false;
    }
    if (!draw.args.is_indexed || !front_vi || front_vi->desc_id != b.desc_id) {
        NativeClose(3);
        return false;
    }
    if (!b.verify && (b.n >= NativeMaxDraws || scheduler.RecordSeq() != b.seq ||
                      scheduler.CurrentTick() != b.tick)) {
        NativeClose(b.n >= NativeMaxDraws ? 4 : 5);
        return false;
    }
    if (texture_cache.StructGeneration() != b.tex_gen ||
        texture_cache.MetaClearEpoch() != b.meta_epoch ||
        VideoCore::g_image_epoch.load(std::memory_order_acquire) != b.image_epoch) {
        NativeClose(11);
        return false;
    }
    NativeCtxSame(); // the block compare when the epoch moved (refreshes b.ctx_epoch)
    if (liverpool->BackContextEpoch() != b.ctx_epoch) {
        if (g_native_diag && g_native_diag_ctx.size() == 0x400) [[unlikely]] {
            for (u32 i = 0; i < 0x400; ++i) {
                if (regs.reg_array[AmdGpu::Regs::ContextRegWordOffset + i] != g_native_diag_ctx[i]) {
                    ++g_native_diag_regs[i];
                }
            }
            if ((++g_native_diag_n & 0xffff) == 0) {
                std::vector<std::pair<u32, u64>> top(g_native_diag_regs.begin(),
                                                     g_native_diag_regs.end());
                std::sort(top.begin(), top.end(),
                          [](const auto& x, const auto& y) { return x.second > y.second; });
                std::string line;
                for (size_t k = 0; k < std::min<size_t>(top.size(), 16); ++k) {
                    line += fmt::format(" {:#x}={}", top[k].first, top[k].second);
                }
                LOG_INFO(Render_Vulkan, "NativeDiag: {} epoch closes, changed context regs:{}",
                         g_native_diag_n, line);
            }
        }
        NativeClose(6);
        return false;
    }
    for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
        if (liverpool->BackCbExtent(cb).raw != b.extents[cb]) {
            NativeClose(7);
            return false;
        }
    }
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    if (liverpool->BackDbExtent().raw != b.extents[AmdGpu::NUM_COLOR_BUFFERS] ||
        (is_index16 ? 2u : 4u) != b.index_size || (occlusion && occlusion->Open())) {
        NativeClose(7);
        return false;
    }
    if (b.colour && (VideoCore::g_native_colour_diag & 1)) {
        NativeClose(12);
        return false;
    }
    std::array<u32, Shader::NativeBindlessSlots> jslots{};
    bool jslots_own = false;
    if (b.colour) {
        // SHADPS4_NATIVE_COLOUR: the draw-repeat conditions for "the same images": identical
        // raw T# / S#, no sampler / border / structure / image epoch change since the opener.
        // SHADPS4_NATIVE_BINDLESS: other images join through the slot cache (sec. 12).
        std::array<u32, 512> key;
        u32 n = 0;
        const bool gates = texture_cache.SamplerGeneration() == b.smp_gen &&
                           regs.ta_bc_base.base_addr == b.border && NativeColourGates() &&
                           NativeImageKey(b.base, key, n) && n == b.img_n;
        if (!gates || std::memcmp(key.data(), native_img_key.data(), n * sizeof(u32)) != 0) {
            bool joined = false;
            if (gates && b.bindless) {
                u32 ni = 0, ns = 0;
                const u64 s0 = scheduler.RecordSeq();
                joined = BindlessResolve(b.base, jslots.data(), nullptr, nullptr, ni, ns) &&
                         bindless.cur == b.bl_set && scheduler.RecordSeq() == s0 &&
                         scheduler.CurrentTick() == b.tick;
                if (!joined) {
                    ++native_stats.bl_join_fail;
                    native_stats.bl_fail[5] += bindless.cur != b.bl_set ||
                                               scheduler.RecordSeq() != s0 ||
                                               scheduler.CurrentTick() != b.tick;
                }
            }
            if (!joined) {
                NativeClose(12);
                native_stats.close_key = true; // bindless census: the next opener follows a key change
                return false;
            }
            jslots_own = true;
            ++native_stats.bl_join;
        }
    }
    static const bool join_timing = [] {
        const char* e = std::getenv("SHADPS4_NATIVE_TIMING");
        return e && e[0] == '1';
    }();
    u64 tp = join_timing ? __rdtsc() : 0;
    const auto lap = [&](u32 phase) {
        if (join_timing) {
            const u64 now = __rdtsc();
            native_stats.c_phase[phase] += now - tp;
            if (b.colour) {
                native_stats.c_cphase[phase] += now - tp;
            }
            tp = now;
        }
    };
    // User data, as BindResources pushes it.
    Shader::PushData push = MakeUserData(regs);
    Shader::Backend::Bindings bnd{};
    for (const auto& bound : BoundStages(b.base)) {
        if (bound.info) {
            bound.PushUd(bnd, push);
        }
    }
    // Buffers: the same lookups / syncs as BindBuffers.
    const u64 seq0 = scheduler.RecordSeq();
    u32 rec_where = 0; // SHADPS4_NATIVE_REC_PREP verify: where this join recorded first
    auto* barriers = b.verify ? nullptr : &native_carry;
    const size_t carry0 = native_carry.size();
    std::array<u32, Shader::NUM_BUFFERS> rec_off{};
    std::array<u32, Shader::NUM_BUFFERS> rec_size{};
    std::array<u64, Shader::NUM_BUFFERS> join_cpu{}; // GPU_DIRECT=2: each slot's buffer base
    std::array<u64, Shader::NUM_BUFFERS> join_addr{}; // SHADPS4_NATIVE_PT: each slot's V# address
    bool ok = true;
    u32 bi = 0;
    // SHADPS4_NATIVE_REC_RUN: a run head that joins validates its run's spans here; anything
    // that records or needs a barrier closes the batch below (carry / RecordSeq checks), as for
    // the join's own lookups.
    if (g_native_rec_run != 0 && !b.colour && !b.verify && nr_head_tag != 0) [[unlikely]] {
        NativeRunValidate(b.base, &native_carry);
    }
    // SHADPS4_NATIVE_REC_RUN: spans validated at the open or by this run's head, still valid.
    const bool run_ok = b.run_valid && !b.verify &&
                        VideoCore::g_dirty_epoch.load(std::memory_order_acquire) == b.run_epoch &&
                        buffer_cache.StructGeneration() == b.run_gen;
    for (const auto& bound : BoundStages(b.base)) {
        if (!bound.info) {
            continue;
        }
        for (const auto& desc : bound.info->buffers) {
            if (b.guest_mask & (u64{1} << bi)) {
                const auto vsharp = desc.GetSharp(bound);
                if (vsharp.base_address == 0 || vsharp.GetSize() == 0) {
                    ok = false;
                    break;
                }
                const u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
                join_addr[bi] = vsharp.base_address;
                if (run_ok && !desc.is_formatted) {
                    const auto& s = b.run_buf[bi];
                    if (s.handle != 0 && vsharp.base_address >= s.lo &&
                        vsharp.base_address + size <= s.hi) {
                        const u64 fast_off = vsharp.base_address - s.cpu;
                        ++native_stats.run_buf_fast; // candidates (verify: compared)
                        if (g_native_rec_run == 1) {
                            b.cand_buf[bi] = s.handle;
                            b.cand_off[bi] = fast_off;
                            b.cand_size[bi] = size;
                            rec_size[bi] = static_cast<u32>(std::min<u64>(size, 0xffffffffull));
                            if (s.handle != b.buf[bi] || fast_off > 0xffffffffull) {
                                ok = false;
                            }
                            rec_off[bi] = static_cast<u32>(fast_off);
                            ++bi;
                            continue;
                        }
                        // Verify: the normal lookup below must give the same handle / offset.
                        const auto [vbuf, voff] = buffer_cache.ObtainBuffer(
                            vsharp.base_address, static_cast<u32>(size), false, desc.is_formatted,
                            VideoCore::BufferId{}, Common::BufferSampleCaller::Storage);
                        if (std::bit_cast<u64>(static_cast<VkBuffer>(vbuf->Handle())) != s.handle ||
                            voff != fast_off) {
                            if (++native_stats.run_mismatch <= 16) {
                                LOG_CRITICAL(Render_Vulkan,
                                             "NativeRun verify MISMATCH: buffer slot {} offset {} "
                                             "vs {}",
                                             bi, voff, fast_off);
                            }
                        }
                    } else {
                        ++native_stats.run_buf_slow;
                    }
                } else if (run_ok) {
                    ++native_stats.run_buf_slow;
                }
                const auto [vk_buffer, offset] =
                    buffer_cache.ObtainBuffer(vsharp.base_address, static_cast<u32>(size), false,
                                              desc.is_formatted, VideoCore::BufferId{},
                                              Common::BufferSampleCaller::Storage);
                const u64 handle = std::bit_cast<u64>(static_cast<VkBuffer>(vk_buffer->Handle()));
                join_cpu[bi] = vk_buffer->CpuAddr();
                if (barriers) {
                    if (auto barrier = vk_buffer->GetBarrier(vk::AccessFlagBits2::eShaderRead,
                                                             vk::PipelineStageFlagBits2::eAllCommands,
                                                             0, cur_dispatch_run)) {
                        barriers->emplace_back(*barrier);
                    }
                }
                b.cand_buf[bi] = handle;
                b.cand_off[bi] = offset;
                b.cand_size[bi] = size;
                rec_size[bi] = static_cast<u32>(std::min<u64>(size, 0xffffffffull));
                if (handle != b.buf[bi]) {
                    ok = false;
                }
                rec_off[bi] = offset;
                if (rec_where == 0 && scheduler.RecordSeq() != seq0) {
                    rec_where = desc.is_formatted ? 1u : 2u;
                }
            }
            ++bi;
        }
        if (!ok) {
            break;
        }
    }
    lap(0); // checks + user data + buffers
    VideoCore::NativeJoinVi jvi{};
    if (ok) {
        const bool vi_fast =
            run_ok && NativeRunVi(*front_vi, draw.args.index_offset, jvi);
        native_stats.run_vi_fast += vi_fast; // candidates (verify: compared)
        if (vi_fast && g_native_rec_run == 1) {
        } else {
            VideoCore::NativeJoinVi ref{};
            buffer_cache.NativeJoinVertexIndex(*front_vi, draw.args.index_offset, ref, barriers);
            if (vi_fast) {
                bool same = ref.n == jvi.n && ref.ib_buf == jvi.ib_buf && ref.ib_off == jvi.ib_off;
                for (u32 k = 0; same && k < ref.n; ++k) {
                    same = ref.buf[k] == jvi.buf[k] && ref.off[k] == jvi.off[k] &&
                           ref.stride[k] == jvi.stride[k];
                }
                if (!same && ++native_stats.run_mismatch <= 16) {
                    LOG_CRITICAL(Render_Vulkan, "NativeRun verify MISMATCH: vertex / index");
                }
            } else if (run_ok) {
                ++native_stats.run_vi_slow;
            }
            jvi = ref;
        }
    }
    if (rec_where == 0 && scheduler.RecordSeq() != seq0) {
        rec_where = 3;
    }
    lap(1); // vertex / index
    s64 q = 0;
    bool q_set = false;
    if (ok && (jvi.n != b.vb_n || jvi.ib_buf != b.ib || jvi.ib_off % b.index_size != 0)) {
        ok = false;
    }
    const auto* join_attribs = AmdGpu::CpSplit::VertexInputAttribs(*front_vi);
    std::array<u32, Shader::NativeMaxPulledLocations> col_voff{};
    std::array<u32, Shader::NativeMaxPulledLocations> col_vsize{};
    for (u32 k = 0; ok && b.colour && k < jvi.n; ++k) {
        // SHADPS4_NATIVE_COLOUR: the same VkBuffer per stream (null where the opener's was
        // null), any offset.
        const u32 loc = join_attribs[k].desc.location;
        const bool had = loc < Shader::NativeMaxPulledLocations && (b.vmask & (1u << loc)) != 0;
        if (jvi.buf[k] == 0) {
            ok = !had && loc < Shader::NativeMaxPulledLocations;
            continue;
        }
        if (!had || jvi.buf[k] != b.vbuf || jvi.off[k] > 0xffffffffull) {
            ok = false;
            break;
        }
        col_voff[loc] = static_cast<u32>(jvi.off[k]);
        col_vsize[loc] = 0xffffffffu;
    }
    for (u32 k = 0; ok && !b.colour && k < jvi.n; ++k) {
        if (jvi.buf[k] != b.vb[k] || jvi.stride[k] != b.vb_stride[k]) {
            ok = false;
            break;
        }
        if (jvi.buf[k] == 0) {
            continue;
        }
        const s64 d = s64(jvi.off[k]) - s64(b.vb_off[k]);
        if (join_attribs[k].desc.instanced) {
            // Opus #3: vertexOffset cannot rebase an instance-rate binding.
            if (d != 0) {
                ok = false;
                break;
            }
            continue; // Opus code review #6: no quotient from an instance-rate binding.
        }
        if (b.vb_stride[k] == 0 ? d != 0 : d % s64(b.vb_stride[k]) != 0) {
            ok = false;
            break;
        }
        const s64 qk = b.vb_stride[k] == 0 ? 0 : d / s64(b.vb_stride[k]);
        if (b.vb_stride[k] != 0) {
            if (q_set && qk != q) {
                ok = false;
                break;
            }
            q = qk;
            q_set = true;
        }
    }
    const auto& vs_info = GetBoundStage(Shader::LogicalStage::Vertex);
    const auto [vertex_offset, instance_offset] =
        GetDrawOffsets(regs, vs_info, b.base->GetFetchShader());
    if (b.verify) {
        // The generic path renders; keep the candidate for NativeVerifyCompare.
        b.cand_valid = ok;
        std::memcpy(b.cand_ud.data(), push.ud_regs.data(), sizeof(u32) * Shader::NUM_USER_DATA_REGS);
        b.cand_vi = jvi;
        b.cand_vertex_offset = vertex_offset;
        b.cand_instance_offset = instance_offset;
        if (!ok) {
            NativeClose(8);
        }
        return false;
    }
    if (!ok || native_carry.size() != carry0 || scheduler.RecordSeq() != seq0 ||
        scheduler.CurrentTick() != b.tick) {
        // A barrier, an upload record or a different VkBuffer: the generic path takes the draw
        // (with the carried barriers) and may open a new batch.
        if (ok && native_carry.size() == carry0 && native_prep.active && native_prep.verify &&
            draw_job_tag == native_prep.tag && draw_job_idx == native_prep.next_idx) {
            ++native_stats.prep_c10[std::min<u32>(rec_where, 3)];
        }
        NativeClose(!ok ? 8 : native_carry.size() != carry0 ? 9 : 10);
        return false;
    }
    const s64 vo = s64(s32(vertex_offset)) + q;
    if (instance_offset != 0) {
        NativeClose(3); // Opus code review #2 (firstInstance)
        return false;
    }
    if (vo < 0 || vo > INT32_MAX) {
        // Opus #3: a rebased vertexOffset must stay a valid non-negative vertex index base.
        NativeClose(3);
        return false;
    }
    auto* rec = reinterpret_cast<u32*>(b.recs + u64(b.n) * NativeRecBytes);
    std::memcpy(rec, push.ud_regs.data(), sizeof(u32) * Shader::NUM_USER_DATA_REGS);
    std::memcpy(rec + Shader::NUM_USER_DATA_REGS, rec_off.data(), sizeof(u32) * Shader::NUM_BUFFERS);
    if (native_pt_on) [[unlikely]] {
        NativePtRecord(rec, join_addr);
    }
    std::memcpy(rec + Shader::NUM_USER_DATA_REGS + Shader::NUM_BUFFERS, rec_size.data(),
                sizeof(u32) * Shader::NUM_BUFFERS);
    if (b.colour) {
        std::memcpy(rec + Shader::NativeRecordVoffOffset, col_voff.data(), sizeof(col_voff));
        std::memcpy(rec + Shader::NativeRecordVsizeOffset, col_vsize.data(), sizeof(col_vsize));
    }
    if (b.bindless) {
        const u32* sl = jslots_own ? jslots.data() : b.bl_slots.data();
        std::memcpy(rec + Shader::NativeRecordImgOffset, sl, sizeof(u32) * jslots.size());
        if (b.ring) {
            BindlessRingWrite(draw_job_tag, draw_job_idx, sl);
        }
    }
    const u32 cmd[5] = {regs.num_indices, regs.num_instances.NumInstances(),
                        static_cast<u32>(jvi.ib_off / b.index_size), static_cast<u32>(s32(vo)),
                        instance_offset};
    std::memcpy(b.cmds + u64(b.n) * NativeCmdBytes, cmd, sizeof(cmd));
    if (b.ring) [[unlikely]] {
        ++native_stats.rj_ok;
        if (g_native_ring_join == 2) {
            // The record the GPU reads for this draw (job words minus the table's bases,
            // inherited registers / sizes from the table) against the join's own.
            namespace NR = AmdGpu::NativeRec;
            const u32* gr = NR::g_gpu_recs[draw_job_tag >> 28].load(std::memory_order_acquire);
            std::array<u32, NR::GpuRecDwords> g{};
            if (gr) {
                std::memcpy(g.data(),
                            gr + size_t(draw_job_idx & (NR::RingSize - 1)) * NR::GpuRecDwords,
                            sizeof(g));
            }
            const auto& t = native_ring_table;
            std::string why = gr ? "" : "no ring";
            for (u32 i = 0; why.empty() && i < Shader::NUM_USER_DATA_REGS; ++i) {
                const u32 want = ((b.ring_push_inh >> i) & 1) ? t[Shader::NativeTableUd + i] : g[i];
                if (want != push.ud_regs[i]) {
                    why = fmt::format("ud {}: ring {:#x} join {:#x}", i, want, push.ud_regs[i]);
                }
            }
            for (u32 i = 0; why.empty() && i < Shader::NUM_BUFFERS; ++i) {
                const u64 bit = u64{1} << i;
                if (!(b.guest_mask & bit)) {
                    continue;
                }
                const u32 off = g[Shader::NativeRecordOffOffset + i] - t[Shader::NativeTableBase + i];
                const u32 sz = (b.ring_inh_slots & bit) ? t[Shader::NativeTableSize + i]
                                                        : g[Shader::NativeRecordSizeOffset + i];
                if (off != rec_off[i] || sz != rec_size[i]) {
                    why = fmt::format("buffer {}: ring {:#x}+{:#x} join {:#x}+{:#x}", i, off, sz,
                                      rec_off[i], rec_size[i]);
                }
            }
            for (u32 loc = 0; why.empty() && b.colour && loc < Shader::NativeMaxPulledLocations;
                 ++loc) {
                if (col_vsize[loc] == 0) {
                    continue;
                }
                const u32 off =
                    g[Shader::NativeRecordVoffOffset + loc] - t[Shader::NativeTableVbase + loc];
                if (off != col_voff[loc] || g[Shader::NativeRecordVsizeOffset + loc] != col_vsize[loc]) {
                    why = fmt::format("vertex location {}: ring {:#x} join {:#x}", loc, off,
                                      col_voff[loc]);
                }
            }
            if (why.empty()) {
                ++native_stats.rj_cmp_ok;
            } else if (++native_stats.rj_cmp_bad <= 16) {
                LOG_CRITICAL(Render_Vulkan, "NativeRingJoin compare MISMATCH ({}, slot {}): {}",
                             b.colour ? "colour" : "shadow", b.n, why);
            }
        }
    }
    if (AmdGpu::NativeRec::g_gpu_mode == 2) [[unlikely]] {
        NativeGpuVerify(push.ud_regs.data(), rec_off, rec_size, join_cpu,
                        b.colour ? col_voff.data() : nullptr, join_attribs, cmd, vertex_offset,
                        instance_offset);
    }
    if (native_prep.active && native_prep.verify) [[unlikely]] {
        // SHADPS4_NATIVE_REC_PREP=2: the open's prepared record / command vs this join's.
        auto& p = native_prep;
        const u32 j = p.next_idx - p.first_idx;
        if (draw_job_tag == p.tag && draw_job_idx == p.next_idx && j < p.n) {
            // Compare with the join's local copies: `rec` lives in write-combined stream memory
            // (a read there costs ~10 k cycles, s395).
            const u32* pr = p.side.data() + size_t(j) * (Shader::NativeRecordDwords + 5);
            std::string why;
            if (std::memcmp(pr, push.ud_regs.data(), sizeof(u32) * Shader::NUM_USER_DATA_REGS) !=
                0) {
                why = "user data";
            }
            for (u32 i = 0; why.empty() && i < Shader::NUM_BUFFERS; ++i) {
                const u32 oo = Shader::NativeRecordOffOffset + i;
                const u32 so = Shader::NativeRecordSizeOffset + i;
                if ((b.guest_mask & (u64{1} << i)) &&
                    (pr[oo] != rec_off[i] || pr[so] != rec_size[i])) {
                    why = fmt::format("buffer {} ({:#x}+{:#x} vs {:#x}+{:#x})", i, pr[oo], pr[so],
                                      rec_off[i], rec_size[i]);
                }
            }
            if (why.empty() && b.colour &&
                (std::memcmp(pr + Shader::NativeRecordVoffOffset, col_voff.data(),
                             sizeof(col_voff)) != 0 ||
                 std::memcmp(pr + Shader::NativeRecordVsizeOffset, col_vsize.data(),
                             sizeof(col_vsize)) != 0)) {
                why = "vertex offsets";
            }
            if (why.empty() &&
                std::memcmp(pr + Shader::NativeRecordDwords, cmd, sizeof(cmd)) != 0) {
                const u32* pc = pr + Shader::NativeRecordDwords;
                why = fmt::format("command ({} {} {} {} {} vs {} {} {} {} {})", pc[0], pc[1],
                                  pc[2], pc[3], pc[4], cmd[0], cmd[1], cmd[2], cmd[3], cmd[4]);
            }
            if (why.empty()) {
                ++native_stats.prep_cmp_ok;
            } else if (++native_stats.prep_cmp_bad <= 16) {
                LOG_CRITICAL(Render_Vulkan, "NativePrep verify MISMATCH (member {} of {}, {}): {}",
                             j, p.n, b.colour ? "colour" : "shadow", why);
            }
            if (++p.next_idx == p.end_idx) {
                p.active = false;
            }
        } else {
            p.active = false;
        }
    }
    ++b.n;
    *b.count = b.n;
    lap(2); // compare + record write
    ++native_stats.joins;
    native_stats.col_joins += b.colour;
    DebugState.IncDrawCall();
    scheduler.NoteDraw();
    MaybeFlushEveryOps(scheduler);
    if ((native_stats.joins & 0x3ffff) == 0) {
        NativeLog();
    }
    return true;
}

void Rasterizer::NativeVerifyCompare(const GraphicsPipeline* pipeline, bool is_indexed,
                                     u32 index_offset) {
    // Verify mode, end of the generic draw: the join's candidate vs what the generic path bound.
    auto& b = native_batch;
    b.cand_valid = false;
    ++native_stats.verify_draws;
    native_stats.col_verify += b.colour;
    std::string why;
    if (std::memcmp(b.cand_ud.data(), push_data.ud_regs.data(),
                    sizeof(u32) * Shader::NUM_USER_DATA_REGS) != 0) {
        why = "user data";
    }
    for (u32 i = 0; why.empty() && i < buffer_infos.size() && i < Shader::NUM_BUFFERS; ++i) {
        if (!(b.guest_mask & (u64{1} << i))) {
            continue;
        }
        const u64 handle = std::bit_cast<u64>(static_cast<VkBuffer>(buffer_infos[i].buffer));
        const u64 full = buffer_infos[i].offset + push_data.buf_offsets[i];
        const u64 size = buffer_infos[i].range - push_data.buf_offsets[i];
        if (handle != b.cand_buf[i] || full != b.cand_off[i] || size != b.cand_size[i]) {
            why = fmt::format("buffer {} ({:#x}+{:#x} vs {:#x}+{:#x})", i, handle, full,
                              b.cand_buf[i], b.cand_off[i]);
        }
    }
    const auto& vi = VideoCore::t_native_vi;
    if (why.empty() && (vi.n != b.cand_vi.n || vi.ib_buf != b.cand_vi.ib_buf ||
                        vi.ib_off != b.cand_vi.ib_off)) {
        why = fmt::format("index / vertex count (ib {:#x}+{:#x} vs {:#x}+{:#x}, n {} vs {})",
                          vi.ib_buf, vi.ib_off, b.cand_vi.ib_buf, b.cand_vi.ib_off, vi.n,
                          b.cand_vi.n);
    }
    for (u32 k = 0; why.empty() && k < vi.n; ++k) {
        if (vi.buf[k] != b.cand_vi.buf[k] || vi.off[k] != b.cand_vi.off[k] ||
            vi.stride[k] != b.cand_vi.stride[k]) {
            why = fmt::format("vertex {}", k);
        }
    }
    if (why.empty() && b.colour &&
        (image_infos.size() != native_open_images.size() ||
         !std::equal(image_infos.begin(), image_infos.end(), native_open_images.begin()))) {
        // SHADPS4_NATIVE_COLOUR: equal image keys must resolve to the opener's images.
        why = fmt::format("images ({} vs {})", image_infos.size(), native_open_images.size());
    }
    const auto& vs_info = GetBoundStage(Shader::LogicalStage::Vertex);
    const auto [vertex_offset, instance_offset] =
        GetDrawOffsets(liverpool->BackRegs(), vs_info, pipeline->GetFetchShader());
    if (why.empty() && (vertex_offset != b.cand_vertex_offset ||
                        instance_offset != b.cand_instance_offset)) {
        why = "draw offsets";
    }
    if (!why.empty()) {
        if (++native_stats.verify_mismatch <= 16) {
            LOG_CRITICAL(Render_Vulkan, "NativePass verify MISMATCH: {}", why);
        }
    }
    if ((native_stats.verify_draws & 0x3ffff) == 0) {
        NativeLog();
    }
}

void Rasterizer::DrawTagNote(const GraphicsPipeline* pipeline, u32 cls, u64 dt) {
    auto& st = draw_tag_stats;
    const u32 tag = draw_tag;
    if (tag == 0 || tag != st.tag) {
        DrawTagFlush();
    }
    if (tag == 0) {
        ++st.untagged;
        st.c_untagged += dt;
    } else {
        const u64 epoch = liverpool->BackContextEpoch();
        if (st.n != 0) {
            st.pipe_changes += pipeline != st.last_pipe;
            st.epoch_changes += epoch != st.last_epoch;
            // Full textured draw whose static key differs from the previous full draw's (bit 6
            // = PREV, set only with SHADPS4_STATIC_KEY_CENSUS).
            st.key_changes += cls == 6 && (native_stats.last_flags & 64) == 0;
        }
        st.tag = tag;
        ++st.n;
        st.cycles += dt;
        ++st.cls[cls];
        st.cls_cycles[cls] += dt;
        if (cls == 6) {
            st.texmiss += (native_stats.last_flags & 1) != 0;
            st.uploads += (native_stats.last_flags & 4) != 0;
        }
        st.last_pipe = pipeline;
        st.last_epoch = epoch;
    }
    if ((++st.total & 0xfffff) != 0) {
        return;
    }
    static constexpr std::array<const char*, 3> names{"shadow", "textured", "untextured"};
    const auto per = [](u64 c, u64 n) { return n ? double(c) / double(n) : 0.0; };
    u64 tagged = 0;
    for (const auto& k : st.kind) {
        tagged += k.draws;
    }
    LOG_INFO(Render_Vulkan,
             "DrawTag: {} draws, tagged {:.1f}%, untagged {} at {:.0f} cycles (whole run "
             "since the last line)",
             tagged + st.untagged, 100.0 * per(tagged, tagged + st.untagged), st.untagged,
             per(st.c_untagged, st.untagged));
    for (u32 i = 0; i < 3; ++i) {
        const auto& k = st.kind[i];
        std::string h, c;
        for (u32 b = 0; b < 8; ++b) {
            h += fmt::format(" {:.1f}", 100.0 * per(k.hist[b], k.draws));
            c += fmt::format(" {}:{}@{:.0f}", b, k.cls[b], per(k.cls_cycles[b], k.cls[b]));
        }
        LOG_INFO(Render_Vulkan,
                 "DrawTag {}: runs={} draws={} mean_run={:.2f} cycles/draw={:.0f} | per run: "
                 "pipeline changes {:.2f} ctx-epoch changes {:.2f} static-key changes {:.2f} "
                 "texmiss draws {:.2f} upload draws {:.3f} | draws % in runs of 1/2-3/4-7/8-15/"
                 "16-31/32-63/64-127/128+:{} | classes n@cycles:{}",
                 names[i], k.runs, k.draws, per(k.draws, k.runs), per(k.cycles, k.draws),
                 per(k.pipe_changes, k.runs), per(k.epoch_changes, k.runs),
                 per(k.key_changes, k.runs), per(k.texmiss, k.runs), per(k.uploads, k.runs), h,
                 c);
    }
    st.kind = {};
    st.untagged = st.c_untagged = 0;
}

void Rasterizer::DrawTagFlush() {
    auto& st = draw_tag_stats;
    if (st.n != 0) {
        const u32 i = (st.cls[0] + st.cls[1]) != 0                              ? 0u
                      : (st.cls[2] + st.cls[3] + st.cls[6] + st.cls[7]) != 0 ? 1u
                                                                             : 2u;
        auto& k = st.kind[i];
        ++k.runs;
        k.draws += st.n;
        k.cycles += st.cycles;
        k.pipe_changes += st.pipe_changes;
        k.epoch_changes += st.epoch_changes;
        k.key_changes += st.key_changes;
        k.texmiss += st.texmiss;
        k.uploads += st.uploads;
        for (u32 b = 0; b < 8; ++b) {
            k.cls[b] += st.cls[b];
            k.cls_cycles[b] += st.cls_cycles[b];
        }
        k.hist[std::min<u32>(std::bit_width(st.n) - 1, 7u)] += st.n;
    }
    st.tag = 0;
    st.n = 0;
    st.cycles = 0;
    st.cls = {};
    st.cls_cycles = {};
    st.pipe_changes = st.epoch_changes = st.key_changes = st.texmiss = st.uploads = 0;
    st.last_pipe = nullptr;
}

void Rasterizer::NativePtRecord(u32* rec, const std::array<u64, Shader::NUM_BUFFERS>& addr) const {
    // SHADPS4_NATIVE_PT: a legacy batch's record holds each Guest slot's guest address (low 32
    // bits in the offset word, bits 32..39 in the hi bytes; the table's base for the slot is the
    // buffer's CPU address low 32 bits, so record - base is still the VkBuffer offset).
    const auto& b = native_batch;
    std::array<u32, Shader::NativeRecordDwords - Shader::NativeRecordHiOffset> hi{};
    for (u64 m = b.guest_mask; m != 0; m &= m - 1) {
        const u32 bi = static_cast<u32>(std::countr_zero(m));
        rec[Shader::NativeRecordOffOffset + bi] = static_cast<u32>(addr[bi]);
        hi[bi / 4] |= static_cast<u32>((addr[bi] >> 32) & 0xffu) << ((bi % 4) * 8);
    }
    std::memcpy(rec + Shader::NativeRecordHiOffset, hi.data(), sizeof(hi));
}

void Rasterizer::NativeSegOpen(u64 recs_addr, u64 list_addr, u32* list) {
    // SHADPS4_NATIVE_SEG (sec. 20 brick A), NativeOpen before the table copy: the table's slot
    // addresses (every job ring's; NativeSegLegacySlot = this batch's own records at recs_addr)
    // and the list's address; then the whole list, once: member i's record is ring entry
    // ring_idx0 + i of the opener's job slot (a ring join appends at ring_idx0 + n;
    // NativeRingPrepare set first_idx = idx + 1) or record i of the batch's own buffer.
    // Sequential stores into write-combined memory only.
    namespace NR = AmdGpu::NativeRec;
    auto& table = native_ring_table;
    const auto put = [&](u32 slot, u64 addr) {
        table[Shader::NativeTableSlotAddr + 2 * slot] = static_cast<u32>(addr);
        table[Shader::NativeTableSlotAddr + 2 * slot + 1] = static_cast<u32>(addr >> 32);
    };
    for (u32 s = 0; s < NR::MaxSlots; ++s) {
        put(s, NR::g_gpu_bda[s].load(std::memory_order_acquire));
    }
    put(Shader::NativeSegLegacySlot, recs_addr);
    table[Shader::NativeTableSegList] = static_cast<u32>(list_addr);
    table[Shader::NativeTableSegList + 1] = static_cast<u32>(list_addr >> 32);
    const auto& b = native_batch;
    const u32 slot_bits = (b.ring ? draw_job_tag >> 28 : Shader::NativeSegLegacySlot) << 24;
    const u32 idx0 = b.ring ? draw_job_idx : 0u;
    for (u32 i = 0; i < NativeMaxDraws; ++i) {
        list[i] = slot_bits | ((idx0 + i) & Shader::NativeSegRingMask);
    }
}

bool Rasterizer::NativeSegEvaluate(const AmdGpu::CpSplit::SplitSegmentRecord& seg,
                                   const GraphicsPipeline* base,
                                   const AmdGpu::CpSplit::VertexInputsPayload* vi,
                                   const u32* list, u32& why) {
    // Back thread, at a Segment record, before anything runs: what the segment open would check
    // without touching a cache (no sync, no barrier, no creation): the variant is published in
    // ring + bindless form, every member's job record is still inside its ring's safe window,
    // and each union (index, pulled streams, Guest job-run unions) lies inside ONE cached buffer
    // of at most 64 MB. The material's bindless resolve and the opener's bind are not evaluated
    // here (they have side effects; today's nd_fail / open_fail stats measure them).
    namespace NR = AmdGpu::NativeRec;
    if (!base || !vi || base->native_variant.load(std::memory_order_relaxed) != nullptr) {
        why = 1;
        return false;
    }
    const auto* attribs = AmdGpu::CpSplit::VertexInputAttribs(*vi);
    const auto* v = base->NativeColourVariant(VideoCore::NativeVertexKey(attribs, vi->count));
    if (!v || !v->native_bindless || v->native_table_binding == GraphicsPipeline::NoNativeRecords ||
        !v->UsesPushDescriptors()) {
        why = 1;
        return false;
    }
    for (u32 j = 0; j < seg.n; ++j) {
        // The entry's low 24 bits are the job's ring counter (unmasked); the age against the
        // job's next counter, modulo 2^24, must leave the GPU its safe window (r3: NR::LifetimeWindow()).
        const u32 slot = (list[j] >> 24) & Shader::NativeSegSlotMask;
        const u32 idx = list[j] & 0xffffffu;
        const u32 next = slot < NR::MaxSlots ? NR::g_job_next[slot].load(std::memory_order_relaxed) : 0u;
        const u32 age = (next - idx) & 0xffffffu;
        if (slot >= NR::MaxSlots || age > NR::LifetimeWindow()) {
            static u32 logged = 0;
            if (++logged <= 10) {
                LOG_INFO(Render_Vulkan,
                         "NativeSegX lifetime refusal: segment {} member {} slot {} ring index "
                         "{} job next {} age {}",
                         seg.id, j, slot, idx, next, age);
            }
            why = 2;
            return false;
        }
    }
    const auto inside = [&](u64 lo, u64 hi, const VideoCore::Buffer*& buf) {
        if (lo == 0 || hi <= lo || hi - lo > (u64{64} << 20)) {
            return false;
        }
        buf = buffer_cache.NativeBufferAt(lo, hi - lo);
        return buf != nullptr && buf->IsInBounds(lo, hi - lo);
    };
    const VideoCore::Buffer* buf = nullptr;
    if (!inside(seg.ilo, seg.ihi, buf)) {
        why = 3;
        return false;
    }
    const VideoCore::Buffer* vb0 = nullptr;
    for (u32 a = 0; a < 16; ++a) {
        if (!((seg.vmask >> a) & 1)) {
            continue;
        }
        if (!inside(seg.vlo[a], seg.vhi[a], buf)) {
            why = 4;
            return false;
        }
        if (vb0 && buf != vb0) {
            why = 5; // the colour rule: every stream in one VkBuffer
            return false;
        }
        vb0 = buf;
    }
    // Guest unions: not a refusal any more (brick C v2) - the pt shaders address every member's
    // range through the page table and flag what is not cached.
    if (seg.n_ops > AmdGpu::CpSplit::SegMaxOps + 3 || seg.n == 0 || seg.n > NativeMaxDraws) {
        why = 7;
        return false;
    }
    why = 0;
    return true;
}

void Rasterizer::NativeLog() {
    const auto& st = native_stats;
    std::string closes;
    for (u32 i = 0; i < st.close.size(); ++i) {
        if (st.close[i]) {
            closes += fmt::format(" c{}={}", i, st.close[i]);
        }
    }
    LOG_INFO(Render_Vulkan,
             "NativePass: opens={} joins={} open_fail={} verify_draws={} verify_mismatch={} "
             "epoch_same={} tick_moved={} closes:{}",
             st.opens, st.joins, st.open_fail, st.verify_draws, st.verify_mismatch,
             st.epoch_same, st.tick_moved, closes);
    if (native_pt_dbg) {
        // SHADPS4_NATIVE_PT=2: the count every time (a clean run shows 0), the entries once.
        const auto* d = reinterpret_cast<const u32*>(native_pt_dbg->mapped_data.data());
        LOG_INFO(Render_Vulkan, "NativePt compare: {} mismatching loads", d[0]);
        if (d[0] != 0 && native_pt_logged < 4) {
            ++native_pt_logged;
            for (u32 e = 0; e < 64 && e < d[0]; ++e) {
                const u32* w = d + 16 + e * 16;
                LOG_WARNING(Render_Vulkan,
                            "NativePt compare: {}{} binding {} width {} rel {:#x} pt {:08x} desc "
                            "{:08x} draw {} guest {:02x}_{:08x}",
                            (w[7] & 2) ? "ps" : "vs", (w[7] & 4) ? " scalar" : "", w[0],
                            w[7] >> 8, w[1], w[2], w[3], w[4], w[6], w[5]);
            }
        }
        if (native_seg_on && Shader::g_native_seg_compare) {
            // SHADPS4_NATIVE_SEG=2: the same buffer, counter dword 1, entries from dword 2048.
            LOG_INFO(Render_Vulkan, "NativeSeg compare: {} mismatching record words", d[1]);
            if (d[1] != 0 && native_seg_logged < 4) {
                ++native_seg_logged;
                for (u32 e = 0; e < 64 && e < d[1]; ++e) {
                    const u32* w = d + 2048 + e * 16;
                    LOG_WARNING(Render_Vulkan,
                                "NativeSeg compare: {} word {} entry {:08x} ptr {:08x} ssbo "
                                "{:08x} draw {} record {:08x}_{:08x}",
                                (w[7] & 0xff) == 2 ? "ps" : "vs", w[0], w[1], w[2], w[3], w[4],
                                w[6], w[5]);
                }
            }
        }
    }
    if (native_dbg && native_dbg_logged < 8) {
        const auto* d = reinterpret_cast<const u32*>(native_dbg->mapped_data.data());
        if (d[0] != 0) {
            ++native_dbg_logged;
            LOG_WARNING(Render_Vulkan, "NativeCompare: {} mismatching attribute fetches", d[0]);
            for (u32 e = 0; e < 64 && e < d[0]; ++e) {
                const u32* w = d + 16 + e * 16;
                LOG_WARNING(Render_Vulkan,
                            "NativeCompare: vs {:08x} loc {} kind {} bits {} count {} vertex {} "
                            "draw {} voff {:#x} pulled {:08x} {:08x} {:08x} {:08x} real {:08x} "
                            "{:08x} {:08x} {:08x}",
                            w[12], w[0], w[13], w[14], w[15], w[1], w[2], w[3], w[4], w[5], w[6],
                            w[7], w[8], w[9], w[10], w[11]);
            }
        }
    }
    if (g_native_rec_run != 0) {
        LOG_INFO(Render_Vulkan,
                 "NativeRun: runs validated={} refused (ring/record/infos/buffers/vertex count/"
                 "buffer span/vertex span/index span)={} {} {} {} {} {} {} {} | member bindings "
                 "fast={} slow={} | vertex-index fast={} slow={} | verify MISMATCH={}",
                 st.run_ok, st.run_fail[0], st.run_fail[1], st.run_fail[2], st.run_fail[3],
                 st.run_fail[4], st.run_fail[5], st.run_fail[6], st.run_fail[7], st.run_buf_fast,
                 st.run_buf_slow, st.run_vi_fast, st.run_vi_slow, st.run_mismatch);
    }
    if (g_native_prep != 0) {
        const auto arr = [](const auto& a) {
            std::string out;
            for (u32 i = 0; i < a.size(); ++i) {
                if (a[i]) {
                    out += fmt::format(" {}={}", i, a[i]);
                }
            }
            return out;
        };
        LOG_INFO(Render_Vulkan,
                 "NativePrep: tries={} runs={} members={} ({:.2f} per run) prepare cycles per "
                 "member {:.0f} (phases per member looked at ({}): head {:.0f} compute {:.0f} spans "
                 "{:.0f} final {:.0f}; spans per batch {:.2f}; inside compute: ring record {:.0f} "
                 "stages {:.0f} vertex block {:.0f} range checks {:.0f}) | stops:{} | confirmed={} "
                 "(colour {}) "
                 "refused:{} | verify ok={} MISMATCH={} join refused a prepared member={} (by "
                 "close:{}; close 10 recorded at formatted slot / other slot / vertex-index: {} {} "
                 "{})",
                 st.prep_try, st.prep_runs, st.prep_members,
                 st.prep_runs ? double(st.prep_members) / double(st.prep_runs) : 0.0,
                 st.prep_members ? double(st.c_prep) / double(st.prep_members) : 0.0,
                 st.prep_looked, double(st.c_prep_ph[0]) / double(std::max<u64>(st.prep_looked, 1)),
                 double(st.c_prep_ph[1]) / double(std::max<u64>(st.prep_looked, 1)),
                 double(st.c_prep_ph[2]) / double(std::max<u64>(st.prep_looked, 1)),
                 double(st.c_prep_ph[3]) / double(std::max<u64>(st.prep_looked, 1)),
                 st.prep_runs ? double(st.prep_spans) / double(st.prep_runs) : 0.0,
                 double(st.c_prep_ph[4]) / double(std::max<u64>(st.prep_looked, 1)),
                 double(st.c_prep_ph[5]) / double(std::max<u64>(st.prep_looked, 1)),
                 double(st.c_prep_ph[6]) / double(std::max<u64>(st.prep_looked, 1)),
                 double(st.c_prep_ph[7]) / double(std::max<u64>(st.prep_looked, 1)),
                 arr(st.prep_stop), st.prep_conf, st.prep_conf_col, arr(st.prep_refuse),
                 st.prep_cmp_ok, st.prep_cmp_bad, st.prep_join_refused, arr(st.prep_jref),
                 st.prep_c10[1], st.prep_c10[2], st.prep_c10[3]);
    }
    if (AmdGpu::NativeRec::g_gpu_mode != 0) {
        LOG_INFO(Render_Vulkan,
                 "NativeGpu verify: ok={} MISMATCH={} | no ring / stale={} no GPU record={} union "
                 "gone or short={}",
                 st.gv[3], st.gv[4], st.gv[0], st.gv[1], st.gv[2]);
        std::string rf;
        for (u32 i = 0; i < st.ring_fail.size(); ++i) {
            if (st.ring_fail[i]) {
                rf += fmt::format(" {}={}", i, st.ring_fail[i]);
            }
        }
        LOG_INFO(Render_Vulkan,
                 "NativeGpu ring: batches={} members={} ({:.2f} per batch) legacy opens={} | ring "
                 "cycles per member {:.0f} spans per batch {:.2f} (phases per batch: records {:.0f} "
                 "bases {:.0f} span syncs {:.0f} commands {:.0f}; KB per span {:.0f}; span cache "
                 "hits {:.1f}%, stale hits race/normal-too/MISSED {}/{}/{}, inherited batches {}) "
                 "| ring-join ok={} refused "
                 "{}/{}/{}/{} compare ok={} MISMATCH={} | refusals:{} "
                 "| confirmed={} (colour {}) | ring verify: ok={} MISMATCH={} join refused a "
                 "predicted member={} (image key c12={}, recorded c10={})",
                 st.ring_runs, st.ring_members,
                 st.ring_runs ? double(st.ring_members) / double(st.ring_runs) : 0.0,
                 st.ring_opens_legacy,
                 st.ring_members ? double(st.c_ring) / double(st.ring_members) : 0.0,
                 st.ring_runs ? double(st.ring_spans) / double(st.ring_runs) : 0.0,
                 double(st.c_ring_ph[0]) / double(std::max<u64>(st.ring_runs, 1)),
                 double(st.c_ring_ph[1]) / double(std::max<u64>(st.ring_runs, 1)),
                 double(st.c_ring_ph[2]) / double(std::max<u64>(st.ring_runs, 1)),
                 double(st.c_ring_ph[3]) / double(std::max<u64>(st.ring_runs, 1)),
                 double(st.ring_span_bytes) / 1024.0 / double(std::max<u64>(st.ring_spans, 1)),
                 100.0 * double(st.ring_span_hits) / double(std::max<u64>(st.ring_spans, 1)),
                 st.ring_span_stale[0], st.ring_span_stale[1], st.ring_span_stale[2],
                 st.ring_inh_runs, st.rj_ok, st.rj_fail[0], st.rj_fail[1],
                 st.rj_fail[2], st.rj_fail[3], st.rj_cmp_ok, st.rj_cmp_bad, rf,
                 st.prep_conf, st.prep_conf_col, st.prep_cmp_ok, st.prep_cmp_bad,
                 st.prep_join_refused, st.prep_jref[12], st.prep_jref[10]);
    }
    {
        std::string sc;
        for (u32 i = 0; i < 16; ++i) {
            if (st.seg_c2[i]) {
                sc += fmt::format(" {}{}{}{}={}", (i & 1) ? "F" : "f", (i & 2) ? "V" : "v",
                                  (i & 4) ? "N" : "n", (i & 8) ? "C" : "c", st.seg_c2[i]);
            }
        }
        std::string sb;
        for (u32 i = 1; i < 32; ++i) {
            if (st.seg_break[i]) {
                sb += fmt::format(" {}{}{}{}{}={}", (i & 1) ? "F" : "", (i & 2) ? "D" : "",
                                  (i & 4) ? "R" : "", (i & 8) ? "G" : "", (i & 16) ? "I" : "",
                                  st.seg_break[i]);
            }
        }
        LOG_INFO(Render_Vulkan,
                 "NativeSegCont: batch opens after the first={} could continue one segment={} "
                 "| breaks by F fixed / D dynamic values / R render targets / G generic draw "
                 "between / I index type:{}",
                 st.seg_opens, st.seg_cont, sb);
        LOG_INFO(Render_Vulkan,
                 "NativeSegDyn: dynamic-value breaks by group: viewport {} scissor {} depth {} "
                 "bias {} stencil ops {} stencil ref {} stencil masks {} cull/face {} blend const "
                 "{} write masks {} other {}",
                 st.seg_dgroup[0], st.seg_dgroup[1], st.seg_dgroup[2], st.seg_dgroup[3],
                 st.seg_dgroup[4], st.seg_dgroup[5], st.seg_dgroup[6], st.seg_dgroup[7],
                 st.seg_dgroup[8], st.seg_dgroup[9], st.seg_dgroup[10]);
        LOG_INFO(Render_Vulkan,
                 "NativeSegCensus: pipeline-change closes by F fixed state equal / V vertex "
                 "formats equal / N next native / C context moved (lower case = not):{} | native "
                 "-> other transitions {}",
                 sc, st.seg_to_other);
        std::vector<std::pair<const char*, std::pair<u64, u64>>> wv(st.why_other.begin(),
                                                                     st.why_other.end());
        std::sort(wv.begin(), wv.end(), [](const auto& x, const auto& y) {
            return x.second.second > y.second.second;
        });
        std::string wl;
        for (const auto& [w, v] : wv) {
            wl += fmt::format(" | {}: n={} cycles {:.0f}", w, v.first,
                              double(v.second) / double(std::max<u64>(v.first, 1)));
        }
        LOG_INFO(Render_Vulkan, "NativeWhyOther: generic DrawTail draws by reason (most cycles "
                                "first):{}",
                 wl);
    }
    if (st.n_no != 0) {
        const auto per = [&](u32 ph) { return double(st.c_no_ph[ph]) / double(st.n_no); };
        LOG_INFO(Render_Vulkan,
                 "NativeOpenPhases (colour, successful, cycles): pre-checks {:.0f} keys+ctx+image "
                 "key {:.0f} bindless resolve {:.0f} vertex+buffers {:.0f} ring prepare {:.0f} "
                 "stream+records {:.0f} (n={}) | generic colour openers' BindResources {:.0f} (n={})",
                 per(0), per(1), per(2), per(3), per(4), per(5), st.n_no,
                 double(st.c_bind_copen) / double(std::max<u64>(st.n_bind_copen, 1)),
                 st.n_bind_copen);
    }
    if (st.nd_cand != 0) {
        LOG_INFO(Render_Vulkan,
                 "NativeDirect: candidates={} direct openers={} (DrawTail {:.0f} cycles each) | "
                 "fails 1..11: {} {} {} {} {} {} {} {} {} {} {} | verify ok={} MISMATCH={} "
                 "(restructured between the binds {}) | cycles per direct bind: buffers {:.0f} "
                 "vertex+index {:.0f} NativeOpen {:.0f} | generic colour openers: BindResources "
                 "{:.0f} (n={})",
                 st.nd_cand, st.nd_ok,
                 double(st.c_tail_dopen) / double(std::max<u64>(st.n_tail_dopen, 1)),
                 st.nd_fail[1], st.nd_fail[2], st.nd_fail[3], st.nd_fail[4], st.nd_fail[5],
                 st.nd_fail[6], st.nd_fail[7], st.nd_fail[8], st.nd_fail[9], st.nd_fail[10],
                 st.nd_fail[11], st.nd_vfy_ok,
                 st.nd_vfy_bad, st.nd_fail[15],
                 double(st.c_nd_ph[0]) / double(std::max<u64>(st.nd_cand, 1)),
                 double(st.c_nd_ph[1]) / double(std::max<u64>(st.nd_cand, 1)),
                 double(st.c_nd_ph[2]) / double(std::max<u64>(st.nd_cand, 1)),
                 double(st.c_bind_copen) / double(std::max<u64>(st.n_bind_copen, 1)),
                 st.n_bind_copen);
    }
    if (st.bl_open + st.bl_open_fail != 0) {
        LOG_INFO(Render_Vulkan,
                 "NativeBindless: batches={} open fail={} | joins with other images={} failed={} "
                 "| fails 0..5: {} {} {} {} {} {} | images hit/miss {}/{} samplers hit/miss {}/{} "
                 "| opener verify ok={} MISMATCH={} unverified={} | rotations={} slots {}/{} | "
                 "cycles per image hit {:.0f} miss {:.0f} | per resolve: samplers {:.0f} touch {:.0f}",
                 st.bl_open, st.bl_open_fail, st.bl_join, st.bl_join_fail, st.bl_fail[0],
                 st.bl_fail[1], st.bl_fail[2], st.bl_fail[3], st.bl_fail[4], st.bl_fail[5],
                 st.bl_img_hit, st.bl_img_miss, st.bl_smp_hit, st.bl_smp_miss, st.bl_vfy_ok,
                 st.bl_vfy_bad, st.bl_vfy_none, st.bl_rot, bindless.img_next, bindless.smp_next,
                 double(st.c_bl_hit) / double(std::max<u64>(st.bl_img_hit, 1)),
                 double(st.c_bl_miss) / double(std::max<u64>(st.bl_img_miss, 1)),
                 double(st.c_bl_smp) / double(std::max<u64>(st.n_bl_res, 1)),
                 double(st.c_bl_touch) / double(std::max<u64>(st.n_bl_res, 1)));
    }
    {
        const auto& cl = PipelineCache::g_nc_claims;
        LOG_INFO(Render_Vulkan,
                 "NativeColour claims (cumulative): claimed {} refused with a reason {} vertex format {} "
                 "records binding {} pushed {} built {} built without push descriptors {} (kept as "
                 "set variants {})",
                 cl[0].load(), cl[1].load(), cl[2].load(), cl[3].load(), cl[4].load(), cl[5].load(),
                 cl[6].load(), cl[7].load());
    }
    if (st.mm_hit + st.mm_miss + st.mm_stale + st.mm_vfy != 0) {
        LOG_INFO(Render_Vulkan,
                 "A2 material memo (SHADPS4_A2_MAT_MEMO): hits {} misses {} stale {} | cycles per "
                 "hit {:.0f} | touches skipped (same ticks) {} | verify compares {} BAD {}",
                 st.mm_hit, st.mm_miss, st.mm_stale,
                 double(st.c_mm_hit) / double(std::max<u64>(st.mm_hit, 1)), st.mm_touch_skip,
                 st.mm_vfy, st.mm_bad);
    }
    if (st.n_rec[0] + st.n_rec[3] + st.n_rec[9] != 0) {
        const auto per = [](u64 c, u64 n) { return n ? double(c) / double(n) : 0.0; };
        std::string out;
        static constexpr const char* names[6] = {"other", "shadow_join", "colour_join", "tail",
                                                 "shadow_confirm", "colour_confirm"};
        for (u32 i = 0; i < 12; ++i) {
            if (st.n_rec[i]) {
                out += fmt::format(" | {}{} {:.0f} (n={})", i >= 6 ? "member " : "", names[i % 6],
                                   per(st.c_rec[i], st.n_rec[i]), st.n_rec[i]);
            }
        }
        LOG_INFO(Render_Vulkan, "NativeRecTiming: whole draw records, cycles per record:{}", out);
    }
    if (st.open_fail != 0) {
        std::string lines;
        for (const auto& [l, n] : st.fail_lines) {
            if (l != 0) {
                lines += fmt::format(" L{}={}", l, n);
            }
        }
        LOG_INFO(Render_Vulkan, "NativeOpenFail (vk_rasterizer.cpp line=count):{}", lines);
    }
    if (VideoCore::g_native_colour) {
        LOG_INFO(Render_Vulkan,
                 "NativeColour: opens={} joins={} open_fail={} verify={} | textured draws={} "
                 "with a variant {:.1f}% | peek yes={} no={} unknown={}",
                 st.col_opens, st.col_joins, st.col_open_fail, st.col_verify, st.col_tex_draws,
                 100.0 * double(st.col_tex_variant) / double(std::max<u64>(st.col_tex_draws, 1)),
                 st.peek_yes, st.peek_no, st.peek_unknown);
    }
    if (st.n_join_ok + st.n_tail_other != 0) {
        const auto per = [](u64 c, u64 n) { return n ? double(c) / double(n) : 0.0; };
        LOG_INFO(Render_Vulkan,
                 "NativeTiming: cycles per draw: join_ok {:.0f} (n={}) join_fail {:.0f} (n={}) "
                 "tail_native_pipe {:.0f} (n={}) tail_other {:.0f} (n={}) | join phases per "
                 "attempt: checks+ud+buffers {:.0f} vertex+index {:.0f} record {:.0f}",
                 per(st.c_join_ok, st.n_join_ok), st.n_join_ok, per(st.c_join_fail, st.n_join_fail),
                 st.n_join_fail, per(st.c_tail_nat, st.n_tail_nat), st.n_tail_nat,
                 per(st.c_tail_other, st.n_tail_other), st.n_tail_other,
                 per(st.c_phase[0], st.n_join_ok + st.n_join_fail),
                 per(st.c_phase[1], st.n_join_ok + st.n_join_fail),
                 per(st.c_phase[2], st.n_join_ok));
        LOG_INFO(Render_Vulkan,
                 "NativeTiming: colour join_ok {:.0f} (n={}) colour opener DrawTail {:.0f} (n={}) | "
                 "colour join phases per join: checks+ud+buffers {:.0f} vertex+index {:.0f} "
                 "record {:.0f}",
                 per(st.c_cjoin_ok, st.n_cjoin_ok), st.n_cjoin_ok,
                 per(st.c_tail_copen, st.n_tail_copen), st.n_tail_copen,
                 per(st.c_cphase[0], st.n_cjoin_ok), per(st.c_cphase[1], st.n_cjoin_ok),
                 per(st.c_cphase[2], st.n_cjoin_ok));
        LOG_INFO(Render_Vulkan,
                 "NativeTiming: other DrawTail by class, cycles per draw: full_untextured {:.0f} "
                 "(n={}) repeat_untextured {:.0f} (n={}) full_textured {:.0f} (n={}) "
                 "repeat_textured {:.0f} (n={})",
                 per(st.c_tail_cls[0], st.n_tail_cls[0]), st.n_tail_cls[0],
                 per(st.c_tail_cls[1], st.n_tail_cls[1]), st.n_tail_cls[1],
                 per(st.c_tail_cls[2], st.n_tail_cls[2]), st.n_tail_cls[2],
                 per(st.c_tail_cls[3], st.n_tail_cls[3]), st.n_tail_cls[3]);
        std::string fl;
        for (u32 f = 0; f < 128; ++f) {
            if (st.n_flags[f] != 0) {
                fl += fmt::format(" | {}{}{}{}{}{}{}: n={} {:.0f}", (f & 64) ? "PREV " : "",
                                  (f & 32) ? "KEYHIT " : "",
                                  (f & 1) ? "texmiss " : "",
                                  (f & 2) ? "recorded " : "", (f & 4) ? "uploaded " : "",
                                  (f & 8) ? "giant " : "", (f & 16) ? "clip " : "",
                                  st.n_flags[f], per(st.c_flags[f], st.n_flags[f]));
            }
        }
        LOG_INFO(Render_Vulkan, "NativeTiming: full textured DrawTail by event (cycles):{}", fl);
        for (u32 a = 0; a < 2; ++a) {
            std::string cl;
            for (u32 f = 0; f < 32; ++f) {
                if (st.n_cop_fl[a][f] != 0) {
                    cl += fmt::format(" | {}{}{}{}{}: n={} {:.0f}", (f & 1) ? "texmiss " : "",
                                      (f & 2) ? "recorded " : "", (f & 4) ? "uploaded " : "",
                                      (f & 8) ? "giant " : "", (f & 16) ? "clip " : "",
                                      st.n_cop_fl[a][f], per(st.c_cop_fl[a][f], st.n_cop_fl[a][f]));
                }
            }
            LOG_INFO(Render_Vulkan, "NativeTiming: colour openers {} by event (cycles):{}",
                     a ? "AFTER AN IMAGE-KEY CLOSE" : "other", cl);
        }
    }
}

void Rasterizer::ExecuteDispatch(const ComputePipeline* pipeline,
                                 const std::array<Shader::BoundStage, Shader::MaxStageTypes>& stages,
                                 bool is_indirect, VAddr address, u32 offset, u32 size) {
    ++p1c_dispatch_serial;
    if (native_batch.open) [[unlikely]] {
        NativeClose(13);
    }
    // CP split, back thread: DispatchDirect/DispatchIndirect after the front's lookup. The caller
    // points Liverpool::BackCsRegs() at the record's ComputeProgram snapshot.
    RENDERER_TRACE;
    const auto t_dispatch = Common::FaultMonitor::Now();
    const u64 dc_disp_t0 = g_draw_cost ? __rdtsc() : 0;
    if (g_draw_cost) {
        g_dc_bind_buf = 0; // an HLE dispatch skips BindResources, which resets these
        g_dc_bind_tex = 0;
    }
    SCOPE_EXIT {
        Common::FaultMonitor::CountDispatch(Common::FaultMonitor::SinceNs(t_dispatch));
        if (g_draw_cost) {
            const u64 cyc = __rdtsc() - dc_disp_t0;
            g_dc_disp_cyc += cyc;
            ++g_dc_disp_n;
            if (pipeline) {
                NoteDispatchCost(pipeline, cyc, g_dc_bind_buf, g_dc_bind_tex);
            }
        }
    };
    // SHADPS4_ASC_LANE: an early ASC dispatch takes the ASC stream's routing onto the lane.
    const bool lane_disp = scheduler.HasLane() && VideoCore::t_force_binding_sync;
    const bool asc_stream = scheduler.HasAscStream() || lane_disp;
    if (asc_stream) {
        // Pending operations, HLE shaders, stencil writebacks and buffer creation/joins (pass 1)
        // record on the main stream; BindBuffers moves an eligible dispatch to the ASC stream.
        scheduler.SwitchStream(0);
    }
    buffer_cache.CommitDeletes(); // r13
    scheduler.PopPendingOperations();
    if (native_bindings) {
        native_bindings->BeginDraw();
    }
    if (!pipeline) {
        return;
    }
    bound_stages = stages;
    dispatch_target = asc_stream && dispatch_queue != 0 && AscCandidate(pipeline) ? 1u : 0u;
    if (lane_disp && dispatch_target == 0) {
        scheduler.LaneFallbackMain(); // not eligible: on main, after the lane's work so far
        if (VideoCore::g_lane_who) [[unlikely]] {
            const auto* ci = bound_stages[static_cast<u32>(Shader::LogicalStage::Compute)].info;
            VideoCore::LaneWhoKept(dispatch_queue == 0 ? 8u : asc_why, ci ? ci->pgm_hash : 0);
        }
    }
    if (VideoCore::BufferCache::CensusEnabled()) [[unlikely]] {
        u64 id = 0;
        for (const auto* stage : pipeline->GetStages()) {
            if (stage) {
                id = stage->pgm_hash;
            }
        }
        buffer_cache.SetWriterTag(dispatch_queue == 0 ? 2 : 3, id);
        VideoCore::g_census_vs = 0; // q35
        VideoCore::g_census_args = 0;
    }
    dispatch_on_asc = false;
    // An indirect dispatch reads its argument buffer: BindBuffers checks it with the bindings.
    buffer_cache.SetAscIndirectArgs(is_indirect ? address + offset : 0, is_indirect ? size : 0);
    const u32 ts = dispatch_target;
    // SHADPS4_GUEST_DISPATCH_ORDER: continue the previous dispatch's run only if the guest issued
    // no sync packet or draw in between (same nonzero guest run) and nothing was recorded since
    // that dispatch ended (no copy, upload, readback, pending operation or borrowed buffer).
    // With the ASC stream the state is per command stream (barriers order one stream only).
    if (dispatch_run != 0 && scheduler.IsAsyncRecording()) {
        const bool same_guest = dispatch_run == (asc_stream ? s_prev_run[ts] : prev_dispatch_run);
        const u64 seq = asc_stream ? scheduler.RecordSeqOf(ts) : scheduler.RecordSeq();
        // A flush between the two dispatches may run pending operations without moving the
        // record sequence (Opus review 2026-09-27, ASC_YIELD): require the same batch too.
        const bool cont =
            asc_stream ? same_guest && seq == s_prev_end_seq[ts] &&
                             StreamBatchId(ts) == s_prev_end_batch[ts]
                       : same_guest && seq == prev_dispatch_end_seq &&
                             scheduler.CurrentTick() == prev_dispatch_end_tick;
        const u64 prev_eff = asc_stream ? s_prev_effective[ts] : prev_effective_run;
        cur_dispatch_run = cont ? prev_eff : ++effective_run_counter;
        // Counters (log only): continued runs vs breaks by cause, and recorded commands between.
        static u64 n_cont = 0, n_guest = 0, n_seq = 0, seq_gap = 0, n_all = 0;
        ++n_all;
        n_cont += cont;
        n_guest += !same_guest;
        if (same_guest && !cont) {
            ++n_seq;
            seq_gap += seq - (asc_stream ? s_prev_end_seq[ts] : prev_dispatch_end_seq);
        }
        if ((n_all & 0xffff) == 0) {
            LOG_INFO(Render_Vulkan,
                     "GuestDispatchOrder: dispatches={} continued={} new_guest_run={} "
                     "recorded_between={} (avg {:.1f} records)",
                     n_all, n_cont, n_guest, n_seq, n_seq ? double(seq_gap) / double(n_seq) : 0.0);
        }
    } else {
        cur_dispatch_run = 0;
    }
    const u64 tick_before_tail = scheduler.CurrentTick();
    if (is_indirect) {
        DispatchIndirectTail(pipeline, address, offset, size);
    } else {
        DispatchDirectTail(pipeline);
    }
    // A flush inside the tail runs pending operations after the dispatch; the next dispatch's
    // sequence check could not see what they recorded (Opus review 2026-09-27): end the run.
    const bool flushed = scheduler.CurrentTick() != tick_before_tail;
    if (asc_stream) {
        // dispatch_on_asc: BindBuffers really moved it (HLE / skipped dispatches never get there).
        if (lane_disp) {
            if (dispatch_on_asc) {
                scheduler.LaneCount(5);
            } else if (ts == 1) {
                scheduler.LaneCount(12);
                scheduler.LaneFallbackMain();
                if (VideoCore::g_lane_who) [[unlikely]] {
                    const auto* ci =
                        bound_stages[static_cast<u32>(Shader::LogicalStage::Compute)].info;
                    VideoCore::LaneWhoKept(20, ci ? ci->pgm_hash : 0); // demoted in BindBuffers
                }
            }
        }
        const u32 s = dispatch_on_asc ? 1u : 0u;
        s_prev_run[s] = (cur_dispatch_run != 0 && !flushed) ? dispatch_run : 0;
        s_prev_effective[s] = cur_dispatch_run;
        s_prev_end_seq[s] = scheduler.RecordSeqOf(s);
        s_prev_end_batch[s] = StreamBatchId(s);
        if (s == 1) {
            // Stamp the dispatch's ASC accesses with the segment that holds its tail.
            const u64 asc_value = scheduler.OpenAscValue();
            ASSERT(asc_value != 0 || buffer_cache.pending_asc.empty());
            buffer_cache.StampPendingAsc(asc_value);
            ++asc_dispatches;
            if ((asc_dispatches & 0xffff) == 0) {
                LOG_INFO(Render_Vulkan,
                         "AscStream: dispatches={} demoted_to_main={} reasons[struct/written/"
                         "read/args/gds/texel_image]={}/{}/{}/{}/{}/{} main_flushes={}",
                         asc_dispatches, asc_demoted, asc_demote_reason[1], asc_demote_reason[2],
                         asc_demote_reason[3], asc_demote_reason[4], asc_demote_reason[5],
                         asc_demote_reason[6], asc_flushes);
            }
        }
        dispatch_target = 0;
        dispatch_on_asc = false;
        cur_dispatch_run = 0;
        buffer_cache.SetAscIndirectArgs(0, 0);
        // Nothing between records may land in the ASC stream (markers, CP-self work).
        scheduler.SwitchStream(0);
        return;
    }
    prev_dispatch_run = (cur_dispatch_run != 0 && !flushed) ? dispatch_run : 0;
    prev_effective_run = cur_dispatch_run;
    prev_dispatch_end_seq = scheduler.RecordSeq();
    prev_dispatch_end_tick = scheduler.CurrentTick();
    cur_dispatch_run = 0;
}

u64 Rasterizer::StreamBatchId(u32 stream) const {
    return stream == 1 ? scheduler.OpenAscValue() : scheduler.CurrentTick();
}

bool Rasterizer::AscCandidate(const ComputePipeline* pipeline) const {
    // Buffers only: no images, samplers, texel buffers (may alias images), DMA / fault / BDA
    // buffers. HLE shaders and image copy / clear shaders never reach BindResources.
    // Descriptor sets from the heap are recycled by the main timeline (asc-i1-relax design R0
    // table): ASC pipelines must push their descriptors.
    asc_why = 0;
    if (!pipeline->UsesPushDescriptors()) {
        asc_why = 1;
        static std::atomic<u64> no_push{0};
        const u64 n = no_push.fetch_add(1) + 1;
        if ((n & (n - 1)) == 0) {
            LOG_INFO(Render_Vulkan, "AscCandidate: {} dispatches rejected (heap descriptors)", n);
        }
        return false;
    }
    const auto* info = bound_stages[static_cast<u32>(Shader::LogicalStage::Compute)].info;
    if (info && VideoCore::BufferCache::CensusEnabled()) [[unlikely]] {
        // SHADPS4_ASC_V2_CENSUS: why an async-queue pipeline is not eligible, once per shader.
        static std::unordered_set<u64> logged;
        if (logged.insert(info->pgm_hash).second) {
            std::string bufs;
            for (const auto& desc : info->buffers) {
                bufs += fmt::format(" [type={} formatted={} written={}]",
                                    static_cast<u32>(desc.buffer_type), desc.is_formatted,
                                    desc.is_written);
            }
            LOG_INFO(Render_Vulkan, "AscCensusPipe: cs_{:08x} images={} samplers={} uses_dma={} "
                                    "readconst_types={} buffers:{}",
                     info->pgm_hash, info->images.size(), info->samplers.size(), info->uses_dma,
                     static_cast<u32>(info->readconst_types), bufs);
        }
    }
    if (!info || !info->images.empty() || !info->samplers.empty() || info->uses_dma) {
        asc_why = !info ? 2 : !info->images.empty() ? 3 : !info->samplers.empty() ? 4 : 5;
        return false;
    }
    // SHADPS4_ASC_TEXEL=1: formatted (texel) buffers are allowed; BindBuffers keeps a dispatch on
    // main only if a read texel binding starts where a cached image does (the one case whose
    // SynchronizeBufferFromImage records an image->buffer copy). Census 2026-09-28: most particle
    // shaders bind one formatted buffer, and the smoke sort's formatted output on main blocked
    // 38% of the control-block copies from the ASC path.
    static const bool asc_texel = [] {
        const char* value = std::getenv("SHADPS4_ASC_TEXEL");
        return value && value[0] == '1';
    }();
    for (const auto& desc : info->buffers) {
        if (desc.is_formatted && !asc_texel) {
            asc_why = 6;
            return false;
        }
        if (desc.IsSpecial() && desc.buffer_type != Shader::BufferType::GdsBuffer &&
            desc.buffer_type != Shader::BufferType::Flatbuf &&
            desc.buffer_type != Shader::BufferType::SharedMemory) {
            asc_why = 7;
            return false;
        }
    }
    return true;
}

void Rasterizer::DrawTail(const GraphicsPipeline* pipeline, bool is_indexed, u32 index_offset,
                          Common::DrawSampleProbe& sample,
                          std::chrono::steady_clock::time_point t_entry,
                          std::chrono::steady_clock::time_point t_pipeline, u64 pipeline_ns,
                          const AmdGpu::CpSplit::VertexInputsPayload* front_vi) {
    const auto& regs = liverpool->BackRegs();
    const bool pb_census = VideoCore::g_pb_census != 0;
    const u64 nc_entry_seq = VideoCore::g_native_census ? scheduler.RecordSeq() : 0;
    if (VideoCore::g_native_census || VideoCore::g_native_pass != 0) [[unlikely]] {
        VideoCore::t_native_vi.n = 0;
        VideoCore::t_native_vi.ib_buf = 0;
        VideoCore::t_native_vi.ib_type = 0;
        VideoCore::t_native_vi.gpu_mod = false;
    }
    if (pb_census) [[unlikely]] {
        PbCensusBegin();
    }
    std::array<u64, DcCount> dc{};
    u64 dc_t = g_draw_cost ? __rdtsc() : 0;
    const auto dc_lap = [&](DrawCostPhase phase) {
        if (g_draw_cost) {
            const u64 now = __rdtsc();
            dc[phase] += now - dc_t;
            dc_t = now;
        }
    };
    const auto t_prepare = Common::FaultMonitor::Now();
    sample.BeginPhase(Common::DrawSampleProbe::Phase::Prepare);
    const int rs_mode = g_render_state_memo;
    const size_t rs_bound_begin = bound_images.size();
    u64 rs_gen_at_entry = 0;
    bool rs_hit = false;   // mode 1: the memo stood in for PrepareRenderState
    bool rs_would = false; // mode 2: it would have
    if (rs_mode != 0) {
        if ((++rs_stats.draws & 0x3fffff) == 0) {
            RenderStateMemoLog();
        }
        rs_gen_at_entry = texture_cache.StructGeneration();
        if (rs_mode == 1) {
            rs_hit = RenderStateMemoMatch(pipeline, true);
            if (!rs_hit) {
                // The full path rewrites cb_descs / db_desc.
                rs_memo.valid = false;
            }
        } else {
            rs_would = RenderStateMemoMatch(pipeline, false);
            if (!rs_would) {
                rs_memo.valid = false;
            }
        }
    }
    if (!rs_hit) {
        PrepareRenderState(pipeline);
    }
    if (rs_would) {
        // Verify: the full path must have resolved the remembered targets, in order.
        ++rs_stats.verify_prepare;
        bool same = bound_images.size() == rs_bound_begin + rs_memo.targets.size();
        for (size_t i = 0; same && i < rs_memo.targets.size(); ++i) {
            same = bound_images[rs_bound_begin + i] == rs_memo.targets[i].id;
        }
        if (!same) {
            rs_would = false;
            rs_memo.valid = false;
            if (++rs_stats.mismatches <= 16) {
                LOG_CRITICAL(Render_Vulkan,
                             "RenderStateMemo mismatch: prepare resolved {} targets, memo {}",
                             bound_images.size() - rs_bound_begin, rs_memo.targets.size());
            }
        }
    }
    sample.EndPhase(Common::DrawSampleProbe::Phase::Prepare);
    dc_lap(DcPrepare);
    if (pb_census) [[unlikely]] {
        PbCensusPoint(1);
    }
    const auto t_bindres = Common::FaultMonitor::Now();
    sample.BeginPhase(Common::DrawSampleProbe::Phase::BindResources);
    // SHADPS4_DRAW_REPEAT: the previous draw's resolved bindings stand when this draw repeats
    // its binding inputs and nothing changed in between (DrawRepeatMatch).
    const int repeat_mode = g_draw_repeat;
    // SHADPS4_NATIVE_DIRECT (sec. 15): the native run start takes the place of BindResources.
    const GraphicsPipeline* direct_native =
        g_native_direct != 0 ? NativeDirectCandidate(pipeline, is_indexed, front_vi) : nullptr;
    bool direct_ok = false;
    if (direct_native && g_native_direct == 1) {
        direct_ok = NativeDirectBind(pipeline, direct_native, index_offset, front_vi, true);
    } else if (direct_native && g_native_direct == 2) [[unlikely]] {
        // Verify: the direct set-0 buffer writes + push offsets vs the generic bind's.
        if (NativeDirectBind(pipeline, direct_native, index_offset, front_vi, false)) {
            struct W {
                u32 binding;
                u64 buffer, offset, range;
                vk::DescriptorType type;
            };
            std::vector<W> dw;
            for (u32 i = 0; i < set_write_index; ++i) {
                const auto& w = set_writes[i];
                dw.push_back({w.dstBinding,
                              std::bit_cast<u64>(static_cast<VkBuffer>(w.pBufferInfo->buffer)),
                              w.pBufferInfo->offset, w.pBufferInfo->range, w.descriptorType});
            }
            const auto dpush = push_data;
            const auto dvi = VideoCore::t_native_vi;
            const u64 dgen = buffer_cache.StructGeneration();
            BindResources(pipeline);
            const bool restructured = buffer_cache.StructGeneration() != dgen;
            std::string why;
            u32 found = 0;
            for (u32 i = 0; i < set_write_index && why.empty(); ++i) {
                const auto& w = set_writes[i];
                if (w.descriptorType != vk::DescriptorType::eStorageBuffer &&
                    w.descriptorType != vk::DescriptorType::eUniformBuffer) {
                    continue;
                }
                const vk::Buffer stream_h =
                    buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream).Handle();
                for (const auto& d : dw) {
                    if (d.binding != w.dstBinding) {
                        continue;
                    }
                    ++found;
                    if (w.pBufferInfo->buffer == stream_h || !w.pBufferInfo->buffer) {
                        continue; // clip planes: a fresh stream copy in each bind
                    }
                    if (d.type != w.descriptorType ||
                        d.buffer != std::bit_cast<u64>(static_cast<VkBuffer>(w.pBufferInfo->buffer)) ||
                        d.offset != w.pBufferInfo->offset || d.range != w.pBufferInfo->range) {
                        why = fmt::format("binding {}: direct {:#x}+{:#x}/{:#x} generic {:#x}+{:#x}/{:#x}",
                                          d.binding, d.buffer, d.offset, d.range,
                                          std::bit_cast<u64>(static_cast<VkBuffer>(w.pBufferInfo->buffer)),
                                          w.pBufferInfo->offset, w.pBufferInfo->range);
                    }
                }
            }
            if (why.empty() && found != dw.size()) {
                why = fmt::format("{} direct buffer writes, {} matched", dw.size(), found);
            }
            if (why.empty() && (std::memcmp(dpush.buf_offsets.data(), push_data.buf_offsets.data(),
                                            sizeof(dpush.buf_offsets)) != 0 ||
                                std::memcmp(dpush.ud_regs.data(), push_data.ud_regs.data(),
                                            sizeof(dpush.ud_regs)) != 0)) {
                why = "push data";
            }
            if (why.empty()) {
                ++native_stats.nd_vfy_ok;
            } else if (restructured) {
                ++native_stats.nd_fail[15]; // the generic bind changed the buffer structure
            } else if (++native_stats.nd_vfy_bad <= 16) {
                LOG_CRITICAL(Render_Vulkan, "NativeDirect verify MISMATCH: {}", why);
            }
            (void)dvi;
            direct_native = nullptr; // the generic bind stands; the normal flow follows
        }
    }
    const bool repeat = !direct_ok && repeat_mode != 0 && native_carry.empty() &&
                        DrawRepeatMatch(pipeline, rs_hit);
    native_stats.last_tail_repeat = repeat && repeat_mode == 1;
    bool bound;
    if (direct_ok) {
        bound = true;
    } else if (g_native_direct == 2 && direct_native == nullptr && native_stats.nd_cand != 0 &&
               false) {
        bound = true;
    } else if (repeat && repeat_mode == 1) {
        DrawRepeatApply(pipeline);
        bound = true;
    } else if (repeat) {
        draw_repeat.v_buffers.assign(buffer_infos.begin(), buffer_infos.end());
        for (size_t k = 0; k < draw_repeat.heap_new.size(); ++k) {
            const u32 idx = draw_repeat.heap_base[k].info_idx;
            if (idx < draw_repeat.v_buffers.size()) {
                draw_repeat.v_buffers[idx] = draw_repeat.heap_new[k].info;
            }
        }
        if (draw_repeat.clip_patch && draw_repeat.clip_base_idx < draw_repeat.v_buffers.size()) {
            draw_repeat.v_buffers[draw_repeat.clip_base_idx] = draw_repeat.clip_new;
        }
        draw_repeat.v_images.assign(image_infos.begin(), image_infos.end());
        draw_repeat.v_seq = scheduler.RecordSeq();
        bound = BindResources(pipeline);
        DrawRepeatVerify(pipeline);
    } else {
        const u64 fc_seq0 = scheduler.RecordSeq();
        const u64 fc_up0 = buffer_cache.SyncUploadBytesTotal();
        native_stats.giant_bound = false;
        const u64 br0 = __rdtsc();
        bound = BindResources(pipeline);
        native_stats.last_bind_cyc = __rdtsc() - br0;
        native_stats.last_flags = (draw_repeat.tex_nonhit_stages != 0 ? 1u : 0u) |
                                  (scheduler.RecordSeq() != fc_seq0 ? 2u : 0u) |
                                  (buffer_cache.SyncUploadBytesTotal() != fc_up0 ? 4u : 0u) |
                                  (native_stats.giant_bound ? 8u : 0u) |
                                  (draw_repeat.clip_info_idx != ~0u ? 16u : 0u) |
                                  (g_static_key_census && g_skey_last_hit ? 32u : 0u) |
                                  (g_static_key_census && g_skey_last_prev ? 64u : 0u);
    }
    sample.EndPhase(Common::DrawSampleProbe::Phase::BindResources);
    const auto bindres_ns = Common::FaultMonitor::SinceNs(t_bindres);
    if (!bound) {
        draw_repeat.valid = false;
        return;
    }
    if (pb_census) [[unlikely]] {
        PbCensusPoint(2);
    }
    if (g_draw_cost) {
        dc_lap(DcBindOther);
        dc[DcBindBuffers] = g_dc_bind_buf;
        dc[DcBindTextures] = g_dc_bind_tex;
        dc[DcBindOther] -= std::min(dc[DcBindOther], g_dc_bind_buf + g_dc_bind_tex);
    }
    const auto t_begin = Common::FaultMonitor::Now();
    sample.BeginPhase(Common::DrawSampleProbe::Phase::BuildRenderingState);
    {
        // Diagnostic SHADPS4_NEO_REG_DIAG=1: raw depth / AA / CB0 registers, logged once per
        // distinct combination (max 60), to see how PS4 Pro checkerboard programs the DB.
        static const bool neo_reg_diag = [] {
            const char* value = std::getenv("SHADPS4_NEO_REG_DIAG");
            return value && value[0] == '1';
        }();
        if (neo_reg_diag) {
            // Sample-location registers (PA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y0_0 .. X1Y1_0) and
            // PA_SC_AA_CONFIG: log each change with a timestamp (first 200), to see whether the
            // Pro checkerboard alternates sample positions per frame.
            const auto& ra = liverpool->BackRegs().reg_array;
            constexpr u32 base = AmdGpu::Regs::ContextRegWordOffset;
            const std::array<u32, 5> locs{ra[base + 0x2F8], ra[base + 0x2FE], ra[base + 0x302],
                                          ra[base + 0x306], ra[base + 0x30A]};
            static std::array<u32, 5> last_locs{};
            static u32 loc_lines = 0;
            if (locs != last_locs && loc_lines < 200) {
                ++loc_lines;
                last_locs = locs;
                const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count();
                LOG_WARNING(Render_Vulkan,
                            "NeoSampleLocs: t={}us AA_CONFIG={:#x} X0Y0={:#010x} X1Y0={:#010x} "
                            "X0Y1={:#010x} X1Y1={:#010x} rt0={}x{}",
                            us, locs[0], locs[1], locs[2], locs[3], locs[4],
                            liverpool->BackRegs().color_buffers[0].Pitch(),
                            liverpool->BackRegs().color_buffers[0].Height());
            }
        }
        if (neo_reg_diag && liverpool->BackRegs().depth_buffer.DepthValid()) {
            static constexpr std::array<u32, 16> offsets{
                0x000, 0x003, 0x010, 0x011, 0x012, 0x014, 0x016, 0x017,
                0x201, 0x203, 0x292, 0x293, 0x2F8, 0x2F9, 0x31D, 0x31C};
            std::array<u32, 16> values{};
            for (size_t i = 0; i < offsets.size(); ++i) {
                values[i] = liverpool->BackRegs().reg_array[AmdGpu::Regs::ContextRegWordOffset +
                                                            offsets[i]];
            }
            static std::set<std::array<u32, 16>> seen;
            if (seen.size() < 60 && seen.insert(values).second) {
                LOG_WARNING(Render_Vulkan,
                            "NeoRegs: DB_RENDER_CONTROL={:#x} DB_RENDER_OVERRIDE={:#x} "
                            "Z_INFO={:#x} STENCIL_INFO={:#x} Z_READ_BASE={:#x} Z_WRITE_BASE={:#x} "
                            "DEPTH_SIZE={:#x} DEPTH_SLICE={:#x} DB_EQAA={:#x} "
                            "DB_SHADER_CONTROL={:#x} PA_SC_MODE_CNTL_0={:#x} MODE_CNTL_1={:#x} "
                            "PA_SC_AA_CONFIG={:#x} AA_MASK?={:#x} CB0_ATTRIB={:#x} CB0_INFO={:#x}",
                            values[0], values[1], values[2], values[3], values[4], values[5],
                            values[6], values[7], values[8], values[9], values[10], values[11],
                            values[12], values[13], values[14], values[15]);
            }
        }
    }
    RenderState state;
    if (rs_hit && RenderStateMemoBeginOk()) {
        RenderStateMemoBegin(state);
        ++rs_stats.begin_hits;
    } else {
        const bool rs_begin_would = rs_would && RenderStateMemoBeginOk();
        rs_stats.begin_fallbacks += rs_hit;
        RenderState rs_probe;
        if (rs_begin_would) {
            // Verify: run the hit-side begin as well (its effects are repeated by the full
            // path right after) and compare what it would have returned.
            RenderStateMemoBegin(rs_probe);
        }
        state = BeginRendering(pipeline);
        if (rs_mode != 0) {
            if (rs_begin_would && !attachment_feedback_loop) {
                // Verify: the remembered state is what the full path just built.
                ++rs_stats.verify_begin;
                const bool depth_same = !db_desc.first ||
                                        rs_last_depth_layout == rs_memo.depth_layout;
                if (!(state == rs_memo.state) || !depth_same || !(state == rs_probe)) {
                    if (++rs_stats.mismatches <= 16) {
                        u32 first_diff = 99;
                        for (u32 i = 0; i < state.color_attachments.size(); ++i) {
                            if (std::memcmp(&state.color_attachments[i],
                                            &rs_memo.state.color_attachments[i],
                                            sizeof(state.color_attachments[i])) != 0) {
                                first_diff = i;
                                break;
                            }
                        }
                        LOG_CRITICAL(Render_Vulkan,
                                     "RenderStateMemo mismatch: state differs (probe same {}, "
                                     "first colour "
                                     "attachment {}, depth attachment same {}, depth layout same "
                                     "{}, extent {}x{}x{} vs {}x{}x{}, colours {} vs {})",
                                     state == rs_probe, first_diff,
                                     std::memcmp(&state.depth_stencil_attachment,
                                                 &rs_memo.state.depth_stencil_attachment,
                                                 sizeof(state.depth_stencil_attachment)) == 0,
                                     depth_same, state.width, state.height, state.num_layers,
                                     rs_memo.state.width, rs_memo.state.height,
                                     rs_memo.state.num_layers, state.num_color_attachments,
                                     rs_memo.state.num_color_attachments);
                    }
                }
            }
            RenderStateMemoStore(pipeline, state, rs_bound_begin, rs_gen_at_entry);
        }
    }
    sample.EndPhase(Common::DrawSampleProbe::Phase::BuildRenderingState);
    dc_lap(DcBeginRendering);
    if (pb_census) [[unlikely]] {
        PbCensusPoint(3);
    }
    if (g_merge_probe) {
        thread_local RenderState last_state{};
        if (!(state == last_state)) {
            ++g_merge_rt_epoch;
            last_state = state;
        }
        NoteMergeProbe(pipeline, BoundStages(pipeline), regs, is_indexed, index_offset);
    }
    const auto t_vtxbuf = Common::FaultMonitor::Now();
    Common::FaultMonitor::CountDraw(pipeline_ns, bindres_ns,
                                    Common::FaultMonitor::DiffNs(t_begin, t_vtxbuf));

    sample.BeginPhase(Common::DrawSampleProbe::Phase::VertexIndexBuffers);
    buffer_cache.BindVertexBuffers(*pipeline, GetBoundStage(Shader::LogicalStage::Vertex),
                                   buffer_barriers, front_vi);
    if (is_indexed) {
        buffer_cache.BindIndexBuffer(index_offset, buffer_barriers);
    }
    sample.EndPhase(Common::DrawSampleProbe::Phase::VertexIndexBuffers);
    dc_lap(DcVertexIndex);
    if (pb_census) [[unlikely]] {
        PbCensusPoint(4);
    }
    const auto t_descwrite = Common::FaultMonitor::Now();

    // SHADPS4_NATIVE_PASS: this draw opens a native batch (its own records go first).
    bool native_open = false;
    native_stats.last_why = is_indexed ? "native pass off" : "not indexed";
    native_stats.last_open_direct = direct_ok;
    if (direct_ok) [[unlikely]] {
        native_open = true;
        pipeline = direct_native;
    } else if (VideoCore::g_native_pass == 1 && is_indexed) [[unlikely]] {
        if (native_batch.open) {
            NativeClose(1);
        }
        const auto* native = pipeline->native_variant.load(std::memory_order_acquire);
        bool colour = false;
        if (!native) {
            const char* cw = pipeline->native_colour_why.load(std::memory_order_relaxed);
            native_stats.last_why =
                pipeline->native_colour[0].key.load(std::memory_order_relaxed) != 0
                    ? "colour: no variant for this vertex description"
                : cw                       ? cw
                : !front_vi                ? "no front vertex inputs"
                : !VideoCore::g_native_colour ? "colour pass off"
                                           : "no variant yet / undecided";
        }
        // Opus code review #7: slots fill in order, so slot 0 empty = no variant at all.
        if (!native && VideoCore::g_native_colour && front_vi &&
            pipeline->native_colour[0].key.load(std::memory_order_relaxed) != 0) [[unlikely]] {
            native = pipeline->NativeColourVariant(VideoCore::NativeVertexKey(
                AmdGpu::CpSplit::VertexInputAttribs(*front_vi), front_vi->count));
            colour = native != nullptr;
            if (!image_infos.empty()) {
                ++native_stats.col_tex_draws;
                native_stats.col_tex_variant += colour;
            }
            // SHADPS4_NATIVE_COLOUR_PEEK (default on): open only if the next queued draw uses the
            // same pipeline (s327 / s328: 62% of colour batches stayed single draws, each paying
            // the open and an indirect draw).
            static const bool peek = [] {
                const char* e = std::getenv("SHADPS4_NATIVE_COLOUR_PEEK");
                return !e || e[0] != '0';
            }();
            if (colour && peek && segx_left == 0) {
                const int next = liverpool->NativePeekNextDraw(pipeline);
                ++(next > 0 ? native_stats.peek_yes
                            : next == 0 ? native_stats.peek_no : native_stats.peek_unknown);
                if (next == 0) {
                    native = nullptr;
                    colour = false;
                    native_stats.last_why = "colour: peek says single draw";
                }
            }
        }
        if (native) {
            native_stats.last_why = (repeat && repeat_mode == 1) ? "draw repeat"
                                    : (occlusion && occlusion->Open()) ? "occlusion query open"
                                    : !build_set_writes || native_bindings ? "no set writes"
                                                                           : "NativeOpen failed";
        }
        if (native && !(repeat && repeat_mode == 1) && !(occlusion && occlusion->Open()) &&
            build_set_writes && !native_bindings &&
            NativeOpen(pipeline, native, is_indexed, index_offset, front_vi, colour)) {
            native_open = true;
            pipeline = native;
        }
    }

    sample.BeginPhase(Common::DrawSampleProbe::Phase::PipelineResources);
    // SHADPS4_DRAW_PACKET: `packet` = the resources are pending as one record that must go out
    // before anything else of this draw is recorded (the legacy order is barrier, push
    // constants, descriptors, dynamic state, beginRendering, draw). A draw wrapped in an
    // occlusion query takes the legacy path; Open() only changes between draws.
    const bool occlusion_draw = occlusion && occlusion->Open();
    bool packet = EmitPipelineResourcesOrPacket(pipeline, occlusion_draw);
    sample.EndPhase(Common::DrawSampleProbe::Phase::PipelineResources);
    dc_lap(DcEmitResources);
    const auto t_dynstate = Common::FaultMonitor::Now();
    sample.BeginPhase(Common::DrawSampleProbe::Phase::DynamicState);
    UpdateDynamicState(pipeline, is_indexed, !packet);
    const bool nc_dyn =
        VideoCore::g_native_census && (!packet || scheduler.CommitDynamicStateWouldRecord());
    if (packet) {
        if (scheduler.CommitDynamicStateWouldRecord()) {
            RecordPacket(pipeline, DrawPacketHeader{});
            packet = false;
            scheduler.CommitDynamicState();
        } else {
            // Predicted no record: check it (the one thing verify mode cannot see). A miss
            // only reorders the dynamic state before the descriptors (valid); count and log.
            const u64 seq = scheduler.RecordSeq();
            scheduler.CommitDynamicState();
            if (scheduler.RecordSeq() != seq) [[unlikely]] {
                NoteDrawPacketFoldMiss("dynamic state");
                RecordPacket(pipeline, DrawPacketHeader{});
                packet = false;
            }
        }
    }
    sample.EndPhase(Common::DrawSampleProbe::Phase::DynamicState);
    const auto t_schedbegin = Common::FaultMonitor::Now();
    sample.BeginPhase(Common::DrawSampleProbe::Phase::SchedulerBeginRendering);
    if (packet && !scheduler.RenderingMatches(state)) {
        RecordPacket(pipeline, DrawPacketHeader{});
        packet = false;
    }
    if (packet) {
        const u64 seq = scheduler.RecordSeq();
        scheduler.BeginRendering(state);
        if (scheduler.RecordSeq() != seq) [[unlikely]] {
            NoteDrawPacketFoldMiss("beginRendering");
            RecordPacket(pipeline, DrawPacketHeader{});
            packet = false;
        }
    } else {
        scheduler.BeginRendering(state);
    }
    sample.EndPhase(Common::DrawSampleProbe::Phase::SchedulerBeginRendering);
    const auto t_emit = Common::FaultMonitor::Now();

    sample.BeginPhase(Common::DrawSampleProbe::Phase::ActualDraw);
    const auto& vs_info = GetBoundStage(Shader::LogicalStage::Vertex);
    const auto& fetch_shader = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);

    // SHADPS4_OCCLUSION_QUERIES: a draw between the guest's two counter dumps is wrapped in its
    // own occlusion query, recorded right around the draw (same render pass, same command buffer).
    u32 occlusion_slot = OcclusionQueries::NoSlot;
    if (occlusion && occlusion->Open()) [[unlikely]] {
        occlusion_slot = occlusion->DrawSlot(OcclusionDrawSig(pipeline));
        AmdGpu::MtbCensus::Note(AmdGpu::MtbCensus::OcclusionSlot);
        if (occlusion_slot != OcclusionQueries::NoSlot) {
            scheduler.Record([query_pool = occlusion->Pool(), occlusion_slot,
                              query_flags = occlusion->Flags()](vk::CommandBuffer cmdbuf) {
                cmdbuf.beginQuery(query_pool, occlusion_slot, query_flags);
            });
        }
    }
    if (native_open) [[unlikely]] {
        if (packet) {
            RecordPacket(pipeline, DrawPacketHeader{});
            packet = false;
        }
        auto& stream = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
        const u64 base_off = native_batch.stream_off;
        const u64 cmd_off = base_off + u64(NativeMaxDraws) * NativeRecBytes;
        const u64 count_off = cmd_off + u64(NativeMaxDraws) * NativeCmdBytes;
        std::array<vk::Buffer, 16> vbufs{};
        std::array<vk::DeviceSize, 16> voffs{};
        for (u32 k = 0; k < native_batch.vb_n; ++k) {
            vbufs[k] = vk::Buffer{std::bit_cast<VkBuffer>(native_batch.vb[k])};
            voffs[k] = native_batch.vb_off[k];
        }
        const vk::PipelineLayout bl_layout =
            native_batch.bindless ? pipeline->GetLayout() : vk::PipelineLayout{};
        const vk::DescriptorSet bl_set =
            native_batch.bindless ? Bindless::G().sets[native_batch.bl_set] : vk::DescriptorSet{};
        if (segx_pending != nullptr && segx_result == 1 && native_batch.sealed) [[unlikely]] {
            // SHADPS4_NATIVE_SEGX=1: the segment's op list in place of the batch draw.
            NativeSegRecord(pipeline, stream.Handle(), cmd_off, bl_layout, bl_set);
        } else {
        scheduler.Record([handle = pipeline->Handle(), sb = stream.Handle(), cmd_off, count_off,
                          bl_layout, bl_set,
                          rebind = native_batch.vb_rebased, vb_n = native_batch.vb_n, vbufs, voffs,
                          ib = vk::Buffer{std::bit_cast<VkBuffer>(native_batch.ib)},
                          type = native_batch.index_size == 2 ? vk::IndexType::eUint16
                                                              : vk::IndexType::eUint32](
                             vk::CommandBuffer cmdbuf) {
            if (!GraphicsBindTracker::Enabled() || GraphicsBindTracker::NeedBind(cmdbuf, handle)) {
                cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
            }
            if (bl_set) {
                cmdbuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, bl_layout, 1, bl_set,
                                          {});
            }
            cmdbuf.bindIndexBuffer(ib, 0, type);
            if (rebind && vb_n != 0) {
                cmdbuf.bindVertexBuffers(0, vb_n, vbufs.data(), voffs.data());
            }
            cmdbuf.drawIndexedIndirectCount(sb, cmd_off, sb, count_off, NativeMaxDraws,
                                            NativeCmdBytes);
        });
        }
        scheduler.GetDynamicState().packet_state_unknown = true;
        native_batch.seq = scheduler.RecordSeq();
        native_batch.tick = scheduler.CurrentTick();
        if (native_batch.tick != native_batch.open_tick) [[unlikely]] {
            // Opus code review #4: a flush between the reservation and this record; the region's
            // watch says the earlier tick. No joins into this batch; counted.
            ++native_stats.tick_moved;
            native_batch.tick = ~0ULL;
        }
        native_batch.open = true;
        ++native_stats.opens;
        native_stats.col_opens += native_batch.colour;
        native_stats.last_open_colour = native_batch.colour;
        if ((native_stats.opens & 0xffff) == 0) {
            NativeLog();
        }
    } else if (packet) {
        // Nothing was recorded since the descriptors: the draw rides in the packet.
        DrawPacketHeader tail{};
        tail.flags = DrawPacketDraw | (is_indexed ? DrawPacketIndexed : 0u);
        tail.count = regs.num_indices;
        tail.instances = regs.num_instances.NumInstances();
        tail.vertex_offset = vertex_offset;
        tail.instance_offset = instance_offset;
        RecordPacket(pipeline, tail);
    } else {
        scheduler.Record([handle = pipeline->Handle(), is_indexed, count = regs.num_indices,
                          instances = regs.num_instances.NumInstances(), vertex_offset,
                          instance_offset](vk::CommandBuffer cmdbuf) {
            if (!GraphicsBindTracker::Enabled() ||
                GraphicsBindTracker::NeedBind(cmdbuf, handle)) {
                cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
            }
            if (is_indexed) {
                cmdbuf.drawIndexed(count, instances, 0, s32(vertex_offset), instance_offset);
            } else {
                cmdbuf.draw(count, instances, vertex_offset, instance_offset);
            }
        });
    }
    if (occlusion_slot != OcclusionQueries::NoSlot) [[unlikely]] {
        scheduler.Record(
            [query_pool = occlusion->Pool(), occlusion_slot](vk::CommandBuffer cmdbuf) {
                cmdbuf.endQuery(query_pool, occlusion_slot);
            });
    }
    DebugState.IncDrawCall();
    scheduler.NoteDraw();
    if (native_batch.verify && native_batch.cand_valid) [[unlikely]] {
        NativeVerifyCompare(pipeline, is_indexed, index_offset);
    }
    if (native_open) [[unlikely]] {
        draw_repeat.valid = false;
    }
    if (VideoCore::g_native_pass == 2 && is_indexed) [[unlikely]] {
        // Verify: key the next draw's candidate on this one.
        native_batch.open = false;
        const bool colour_cand =
            VideoCore::g_native_colour && front_vi &&
            pipeline->native_colour[0].key.load(std::memory_order_relaxed) != 0 &&
            pipeline->NativeColourVariant(VideoCore::NativeVertexKey(
                AmdGpu::CpSplit::VertexInputAttribs(*front_vi), front_vi->count)) != nullptr;
        if ((pipeline->native_variant.load(std::memory_order_acquire) || colour_cand) &&
            occlusion_slot == OcclusionQueries::NoSlot) {
            NativeOpen(pipeline, nullptr, is_indexed, index_offset, front_vi,
                       colour_cand && !pipeline->native_variant.load(std::memory_order_acquire));
        }
    }
    if (VideoCore::g_native_census) [[unlikely]] {
        NativeCensusNote(pipeline, state, nc_entry_seq, scheduler.RecordSeq(), nc_dyn,
                         !buffer_barriers.empty(), image_infos, buffer_infos, push_data,
                         occlusion_slot != OcclusionQueries::NoSlot);
    }

    sample.EndPhase(Common::DrawSampleProbe::Phase::ActualDraw);
    sample.BeginPhase(Common::DrawSampleProbe::Phase::ResetImageBindings);
    ResetBindings();
    if (repeat_mode != 0 && !native_open) {
        // An occlusion-query draw goes through the legacy records: not a base for a repeat.
        DrawRepeatStore(pipeline, occlusion_slot == OcclusionQueries::NoSlot &&
                                      !(occlusion && occlusion->Open()));
    }
    sample.EndPhase(Common::DrawSampleProbe::Phase::ResetImageBindings);
    sample.Finish();
    if (g_draw_cost) {
        dc_lap(DcRecordTail);
        NoteDrawCost(pipeline, dc, repeat && repeat_mode == 1);
        // ~50 us at ~3.5 GHz TSC: log where a slow texture phase went (first 5 per shader).
        static thread_local std::unordered_map<u64, u32> slow_logged;
        u64 fs = 0;
        if (dc[DcBindTextures] > 175000) {
            for (const auto* stage : pipeline->GetStages()) {
                if (stage && stage->l_stage == Shader::LogicalStage::Fragment) {
                    fs = stage->pgm_hash;
                }
            }
        }
        if (dc[DcBindTextures] > 175000 && slow_logged[fs]++ < 5) {
            const auto& si = g_dc_slow_img;
            LOG_INFO(Render_Vulkan,
                     "DrawCostSlowTex: fs_{:08x} tex={} cyc first_pass={} find_texture={} "
                     "transit={} | slowest image {:#x} {}x{}x{} fmt={} depth={} tiled={} "
                     "layout_before={} cyc={}",
                     fs, dc[DcBindTextures], g_dc_tex_sub[0], g_dc_tex_sub[1], g_dc_tex_sub[2],
                     si.addr, si.w, si.h, si.layers, vk::to_string(vk::Format(si.fmt)), si.depth,
                     si.tiled, vk::to_string(vk::ImageLayout(si.layout_before)), si.cyc);
        }
    }

    // SHADPS4_FLUSH_EVERY_DRAWS=N: hand recorded work to the GPU every N draws. Guest readback
    // faults wait for everything queued before their copy; measured 2026-09-23 Game:Main spent
    // 51% of its wall time in that wait while the CP sat 41% idle, so earlier submission trades
    // spare CP time for shorter readback latency. 0 (default) keeps the old cadence.
    static const u32 flush_every_draws = [] {
        const char* value = std::getenv("SHADPS4_FLUSH_EVERY_DRAWS");
        return value ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 0u;
    }();
    if (flush_every_draws != 0) {
        static thread_local u32 draws_since_flush = 0;
        if (++draws_since_flush >= flush_every_draws) {
            draws_since_flush = 0;
            scheduler.NoteCadenceFlush();
            scheduler.Flush();
        }
    }
    // SHADPS4_READBACK_EARLY_FLUSH=1: this draw wrote into memory the guest recently read
    // back; submit now so the next readback of it waits less (see BufferCache::NoteGpuWrite).
    if (buffer_cache.TakeReadbackFlushRequest()) {
        static thread_local u32 early_flushes = 0;
        ++early_flushes;
        scheduler.NoteCadenceFlush();
        scheduler.Flush();
    }
    MaybeFlushEveryOps(scheduler);
    if (pb_census) [[unlikely]] {
        PbCensusEnd(pipeline, rs_hit, repeat && repeat_mode == 1, front_vi, is_indexed,
                    index_offset);
    }

    Common::FaultMonitor::CountDrawTail(Common::FaultMonitor::DiffNs(t_entry, t_pipeline) +
                                            Common::FaultMonitor::DiffNs(t_prepare, t_bindres),
                                        Common::FaultMonitor::DiffNs(t_vtxbuf, t_descwrite),
                                        Common::FaultMonitor::DiffNs(t_descwrite, t_dynstate),
                                        Common::FaultMonitor::DiffNs(t_dynstate, t_schedbegin),
                                        Common::FaultMonitor::DiffNs(t_schedbegin, t_emit),
                                        Common::FaultMonitor::SinceNs(t_emit));
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address) {
    RENDERER_TRACE;

    buffer_cache.CommitDeletes(); // r13
    scheduler.PopPendingOperations();
    if (native_bindings) {
        native_bindings->BeginDraw();
    }

    if (!FilterDraw()) {
        return;
    }

    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline();
    if (!pipeline) {
        return;
    }
    BindStageData(pipeline);
    DrawIndirectTail(pipeline, is_indexed, arg_address, offset, stride, max_count, count_address);
}

void Rasterizer::DrawIndirectTail(const GraphicsPipeline* pipeline, bool is_indexed,
                                  VAddr arg_address, u32 offset, u32 stride, u32 max_count,
                                  VAddr count_address) {
    if (native_batch.open) [[unlikely]] {
        NativeClose(14);
    }
    rs_memo.valid = false; // RENDER_STATE_MEMO: this path rewrites cb_descs / db_desc
    PrepareRenderState(pipeline);
    if (VideoCore::g_lane_seg_census) [[unlikely]] {
        VideoCore::g_census_args = arg_address + offset; // q35 H5
    }
    if (!BindResources(pipeline)) {
        return;
    }
    const auto state = BeginRendering(pipeline);

    buffer_cache.BindVertexBuffers(*pipeline, GetBoundStage(Shader::LogicalStage::Vertex),
                                   buffer_barriers);
    if (is_indexed) {
        // r4 (SHADPS4_IB_INDIRECT_CAP): an indexed indirect draw reads up to the capacity
        // INDEX_BUFFER_SIZE set (VGT_DMA_MAX_SIZE), not the last direct draw's index count.
        buffer_cache.BindIndexBuffer(0, buffer_barriers,
                                     VideoCore::IndirectIndexCapacity(liverpool->BackRegs(), 0));
    }

    const auto& [buffer, base] =
        buffer_cache.ObtainBuffer(arg_address + offset, stride * max_count, false);

    VideoCore::Buffer* count_buffer{};
    u32 count_base{};
    if (count_address != 0) {
        std::tie(count_buffer, count_base) = buffer_cache.ObtainBuffer(count_address, 4, false);
    }

    bool q1_barrier = false;
    if (auto barrier = buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                          vk::PipelineStageFlagBits2::eDrawIndirect)) {
        buffer_barriers.emplace_back(*barrier);
        q1_barrier = true;
    }
    {
        // q1 census: indirect draws on the old path, of which the args buffer needed a barrier
        // (GPU-written since its last barrier), indexed, with a count buffer, max_count > 1.
        static std::array<u64, 5> q1{};
        ++q1[0];
        q1[1] += q1_barrier;
        q1[2] += is_indexed;
        q1[3] += count_address != 0;
        q1[4] += max_count > 1;
        if ((q1[0] & 0xffff) == 0) {
            LOG_INFO(Render_Vulkan,
                     "q1 indirect (old path, cumulative): draws {} args barrier {} indexed {} "
                     "count buffer {} max_count>1 {}",
                     q1[0], q1[1], q1[2], q1[3], q1[4]);
        }
    }
    if (count_buffer) {
        if (auto barrier = count_buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                                    vk::PipelineStageFlagBits2::eDrawIndirect)) {
            buffer_barriers.emplace_back(*barrier);
        }
    }

    EmitPipelineResources(pipeline);
    UpdateDynamicState(pipeline, is_indexed);
    scheduler.BeginRendering(state);

    // We can safely ignore both SGPR UD indices and results of fetch shader parsing, as vertex and
    // instance offsets will be automatically applied by Vulkan from indirect args buffer.

    ASSERT((is_indexed ? sizeof(VkDrawIndexedIndirectCommand) : sizeof(VkDrawIndirectCommand)) ==
           stride);
    u32 occlusion_slot = OcclusionQueries::NoSlot;
    if (occlusion && occlusion->Open()) [[unlikely]] {
        occlusion_slot = occlusion->DrawSlot(OcclusionDrawSig(pipeline));
        AmdGpu::MtbCensus::Note(AmdGpu::MtbCensus::OcclusionSlot);
        if (occlusion_slot != OcclusionQueries::NoSlot) {
            scheduler.Record([query_pool = occlusion->Pool(), occlusion_slot,
                              query_flags = occlusion->Flags()](vk::CommandBuffer cmdbuf) {
                cmdbuf.beginQuery(query_pool, occlusion_slot, query_flags);
            });
        }
    }
    scheduler.Record([handle = pipeline->Handle(), buffer = buffer->Handle(), base,
                      count_handle = count_buffer ? count_buffer->Handle() : vk::Buffer{},
                      has_count = count_address != 0, count_base, max_count, stride,
                      is_indexed](vk::CommandBuffer cmdbuf) {
        if (!GraphicsBindTracker::Enabled() || GraphicsBindTracker::NeedBind(cmdbuf, handle)) {
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
        }
        if (is_indexed) {
            if (has_count) {
                cmdbuf.drawIndexedIndirectCount(buffer, base, count_handle, count_base, max_count,
                                                stride);
            } else {
                cmdbuf.drawIndexedIndirect(buffer, base, max_count, stride);
            }
        } else {
            if (has_count) {
                cmdbuf.drawIndirectCount(buffer, base, count_handle, count_base, max_count, stride);
            } else {
                cmdbuf.drawIndirect(buffer, base, max_count, stride);
            }
        }
    });
    if (occlusion_slot != OcclusionQueries::NoSlot) [[unlikely]] {
        scheduler.Record(
            [query_pool = occlusion->Pool(), occlusion_slot](vk::CommandBuffer cmdbuf) {
                cmdbuf.endQuery(query_pool, occlusion_slot);
            });
    }
    DebugState.IncDrawCall();
    scheduler.NoteOp();

    ResetBindings();
    MaybeFlushEveryOps(scheduler);
}

void Rasterizer::DispatchDirect() {
    RENDERER_TRACE;
    // q11a: dispatches on the CP thread (and those inside the early ASC service).
    const bool lb_cp = VideoCore::g_lb_on && Common::FaultMonitor::t_is_cp_thread;
    const bool lb_srv = lb_cp && VideoCore::g_lb_in_service;
    const u64 lb_t0 = lb_cp ? __rdtsc() : 0;
    SCOPE_EXIT {
        if (lb_cp) {
            const u64 d = __rdtsc() - lb_t0;
            VideoCore::g_lb[2] += d;
            ++VideoCore::g_lb[3];
            if (lb_srv) {
                VideoCore::g_lb[4] += d;
                ++VideoCore::g_lb[5];
            }
        }
    };

    // 526 dispatches per forest frame, and they sit in the 27% of the frame that Draw's timers do
    // not cover. SCOPE_EXIT so the several early returns below are still counted.
    const auto t_dispatch = Common::FaultMonitor::Now();
    SCOPE_EXIT {
        Common::FaultMonitor::CountDispatch(Common::FaultMonitor::SinceNs(t_dispatch));
    };

    buffer_cache.CommitDeletes(); // r13
    scheduler.PopPendingOperations();
    if (native_bindings) {
        native_bindings->BeginDraw();
    }

    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }
    BindStageData(pipeline);
    DispatchDirectTail(pipeline);
}

int Rasterizer::EarlyCutMode() {
    static const int m = [] {
        const char* e = std::getenv("SHADPS4_READBACK_EARLY_CUT");
        return e ? std::atoi(e) : 0;
    }();
    return m;
}

void Rasterizer::MaybeEarlyCut(bool run_end) {
    // q7 (SHADPS4_READBACK_EARLY_CUT): this dispatch wrote a recently read-back window - end the
    // batch now so the readback waits for the GPU only up to here, not for the rest of the frame.
    // q7b: mode 2 cuts once, at the end of the dispatch run (A2Execute, the next non-dispatch
    // record); mode 1 after every such dispatch.
    if ((EarlyCutMode() == 2) != run_end || !buffer_cache.EarlyCutPending()) {
        return;
    }
    if (Scheduler::t_lane_mode || scheduler.ActiveStream() != 0) {
        static std::atomic<u64> blocked{0}; // q7c (1): kept pending, counted
        if ((blocked.fetch_add(1, std::memory_order_relaxed) & 0xfff) == 0) {
            LOG_INFO(Render_Vulkan, "q7c early cut deferred (lane / stream): {}", blocked.load());
        }
        return;
    }
    (void)buffer_cache.TakeEarlyCut();
    scheduler.EndRendering();
    scheduler.Flush();
    static std::atomic<u64> cuts{0};
    static std::atomic<s64> last{0};
    const u64 n = cuts.fetch_add(1, std::memory_order_relaxed) + 1;
    const s64 now = std::chrono::steady_clock::now().time_since_epoch().count();
    if (now - last.load(std::memory_order_relaxed) > 10'000'000'000ll) {
        last.store(now, std::memory_order_relaxed);
        LOG_INFO(Render_Vulkan, "q7 readback early cuts (cumulative): {}", n);
    }
}

// r20 SHADPS4_EXPOSURE_STALE_READ: the game's exposure metering shader (REFERENCE cs_a8b68717).
static bool IsExposureMetering(const ComputePipeline* pipeline) {
    if (!VideoCore::BufferCache::ExposureStaleRead() || !pipeline) {
        return false;
    }
    for (const auto* stage : pipeline->GetStages()) {
        if (stage && stage->pgm_hash == 0xa8b68717ULL) {
            return true;
        }
    }
    return false;
}

void Rasterizer::DispatchDirectTail(const ComputePipeline* pipeline) {
    VideoCore::t_in_dispatch = true; // q7c
    VideoCore::t_tolerant_writer = IsExposureMetering(pipeline); // r20
    SCOPE_EXIT {
        VideoCore::t_in_dispatch = false;
        VideoCore::t_tolerant_writer = false;
        buffer_cache.HotPrefetchAfterWrite(); // q56
        MaybeEarlyCut();
    };
    if (native_batch.open) [[unlikely]] {
        NativeClose(15);
    }
    const auto& cs_program = liverpool->BackCsRegs();
    const auto& cs = GetBoundStage(Shader::LogicalStage::Compute);
    if (ExecuteShaderHLE(cs, liverpool->BackRegs(), cs_program, *this)) {
        return;
    }
    const bool pb_census = VideoCore::g_pb_census != 0;
    if (pb_census) [[unlikely]] {
        PbCensusDispatch(pipeline, 0);
    }

    if (!BindResources(pipeline)) {
        return;
    }
    if (pb_census) [[unlikely]] {
        PbCensusDispatch(pipeline, 1);
    }

    scheduler.EndRendering();
    if (EmitPipelineResourcesOrPacket(pipeline, false)) {
        // SHADPS4_DRAW_PACKET: nothing is recorded between the descriptors and the dispatch.
        DrawPacketHeader tail{};
        tail.flags = DrawPacketDispatch;
        tail.dim_x = cs_program.dim_x;
        tail.dim_y = cs_program.dim_y;
        tail.dim_z = cs_program.dim_z;
        RecordPacket(pipeline, tail);
    } else {
        scheduler.Record([handle = pipeline->Handle(), x = cs_program.dim_x, y = cs_program.dim_y,
                          z = cs_program.dim_z](vk::CommandBuffer cmdbuf) {
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
            cmdbuf.dispatch(x, y, z);
        });
    }
    DebugState.IncDispatch();
    scheduler.NoteDispatch();

    ResetBindings();
    if (buffer_cache.TakeReadbackFlushRequest()) {
        scheduler.NoteCadenceFlush();
        scheduler.Flush();
    }
    MaybeFlushEveryOps(scheduler);
    if (pb_census) [[unlikely]] {
        PbCensusDispatch(pipeline, 2);
    }
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    RENDERER_TRACE;
    // q11a: dispatches on the CP thread (and those inside the early ASC service).
    const bool lb_cp = VideoCore::g_lb_on && Common::FaultMonitor::t_is_cp_thread;
    const bool lb_srv = lb_cp && VideoCore::g_lb_in_service;
    const u64 lb_t0 = lb_cp ? __rdtsc() : 0;
    SCOPE_EXIT {
        if (lb_cp) {
            const u64 d = __rdtsc() - lb_t0;
            VideoCore::g_lb[2] += d;
            ++VideoCore::g_lb[3];
            if (lb_srv) {
                VideoCore::g_lb[4] += d;
                ++VideoCore::g_lb[5];
            }
        }
    };

    const auto t_dispatch = Common::FaultMonitor::Now();
    SCOPE_EXIT {
        Common::FaultMonitor::CountDispatch(Common::FaultMonitor::SinceNs(t_dispatch));
    };

    buffer_cache.CommitDeletes(); // r13
    scheduler.PopPendingOperations();
    if (native_bindings) {
        native_bindings->BeginDraw();
    }

    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }
    BindStageData(pipeline);
    DispatchIndirectTail(pipeline, address, offset, size);
}

void Rasterizer::DispatchIndirectTail(const ComputePipeline* pipeline, VAddr address, u32 offset,
                                      u32 size) {
    VideoCore::t_in_dispatch = true; // q7c
    VideoCore::t_tolerant_writer = IsExposureMetering(pipeline); // r20
    SCOPE_EXIT {
        VideoCore::t_in_dispatch = false;
        VideoCore::t_tolerant_writer = false;
        buffer_cache.HotPrefetchAfterWrite(); // q56
        MaybeEarlyCut(); // q7
    };
    if (!BindResources(pipeline)) {
        return;
    }

    const auto [buffer, base] = buffer_cache.ObtainBuffer(address + offset, size, false);

    if (auto barrier = buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                          vk::PipelineStageFlagBits2::eDrawIndirect)) {
        buffer_barriers.emplace_back(*barrier);
    }

    scheduler.EndRendering();
    EmitPipelineResources(pipeline);

    scheduler.Record(
        [handle = pipeline->Handle(), buffer = buffer->Handle(), base](vk::CommandBuffer cmdbuf) {
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
            cmdbuf.dispatchIndirect(buffer, base);
        });
    DebugState.IncDispatch();
    scheduler.NoteDispatch();

    ResetBindings();
    MaybeFlushEveryOps(scheduler);
}

u64 Rasterizer::Flush() {
    const u64 current_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return current_tick;
}

void Rasterizer::Finish() {
    scheduler.Finish();
}

void Rasterizer::DeferToGpu(Common::UniqueFunction<void>&& func, bool flush) {
    // Queued at the current tick first, then that tick is submitted (or, lazily, left to the
    // next flush: every SHADPS4_FLUSH_EVERY_DRAWS draws, a CP memory wait or CP idle).
    scheduler.DeferPriorityOperation(std::move(func));
    if (flush) {
        Flush();
    } else {
        lazy_fence_tick = scheduler.CurrentTick();
    }
}

void Rasterizer::FlushLazyFences() {
    if (lazy_fence_tick != 0 && lazy_fence_tick == scheduler.CurrentTick()) {
        Flush();
    }
    lazy_fence_tick = 0;
}

void Rasterizer::OnSubmit() {
    {
        // Diagnostic: SHADPS4_DUMP_IMAGES=addr[,addr...] (hex) with SHADPS4_DUMP_IMAGES_AT_S=N
        // dumps those surfaces at the first submit N s after the first one, twice (a frame apart).
        static const std::vector<VAddr> dump_addresses = [] {
            std::vector<VAddr> out;
            if (const char* value = std::getenv("SHADPS4_DUMP_IMAGES")) {
                std::string list{value};
                size_t pos = 0;
                while (pos < list.size()) {
                    const size_t comma = list.find(',', pos);
                    out.push_back(std::strtoull(list.substr(pos, comma - pos).c_str(), nullptr, 16));
                    pos = comma == std::string::npos ? list.size() : comma + 1;
                }
            }
            return out;
        }();
        if (!dump_addresses.empty()) {
            static const double at_s = [] {
                const char* value = std::getenv("SHADPS4_DUMP_IMAGES_AT_S");
                return value ? std::atof(value) : 120.0;
            }();
            static const auto first = std::chrono::steady_clock::now();
            static u32 dumps_done = 0;
            if (dumps_done < 2 &&
                std::chrono::duration<double>(std::chrono::steady_clock::now() - first).count() >=
                    at_s + dumps_done * 0.05) {
                texture_cache.DebugDumpImages(dump_addresses, dumps_done++);
            }
        }
    }
    if (fault_process_pending) {
        VideoCore::G1Scope g1{2}; // q19
        fault_process_pending = false;
        buffer_cache.ProcessFaultBuffer();
        if (native_pt_on) [[unlikely]] {
            // SHADPS4_NATIVE_PT: pages the pt shaders faulted (brick C v2), every ~10 s.
            static auto window = std::chrono::steady_clock::now();
            static u64 last = 0;
            const auto now = std::chrono::steady_clock::now();
            if (now - window >= std::chrono::seconds(10)) {
                const u64 pages = buffer_cache.FaultPagesProcessed();
                LOG_INFO(Render_Vulkan, "NativePt faults: pages={}", pages - last);
                last = pages;
                window = now;
            }
        }
    }
    {
        VideoCore::G1Scope g1{3}; // q19
        texture_cache.ProcessDownloadImages();
    }
    {
        VideoCore::G1Scope g1{4};
        texture_cache.RunGarbageCollector();
    }
    VideoCore::G1Scope g1{5};
    buffer_cache.RunGarbageCollector();
}

bool Rasterizer::DrawRepeatMatch(const GraphicsPipeline* pipeline, bool rs_hit) {
    auto& st = draw_repeat;
    // global, dma, written, formatted, special (other), image written, flat buffer (SRT copy)
    static std::array<u64, 7> why{};
    // SHADPS4_DRAW_REPEAT_FLAT_CENSUS=1 (2026-10-09, dcb-parallel note sec. 7): draws ineligible
    // only because of flat-buffer bindings whose binding inputs - including the flat buffer's
    // CONTENT - equal the immediately previous draw's, with the same epochs and generations.
    static const bool flat_census = [] {
        const char* v = std::getenv("SHADPS4_DRAW_REPEAT_FLAT_CENSUS");
        return v && v[0] == '1';
    }();
    struct FlatCensus {
        std::vector<u32> key[2];
        u32 kidx = 0;
        const GraphicsPipeline* pipeline = nullptr;
        u64 draw = 0; // st.draws of the stored draw
        u64 dirty = 0, image = 0, buf_gen = 0, tex_gen = 0, smp_gen = 0, border = 0;
        u64 draws = 0, flat_only = 0, textured = 0, same_pipe_prev = 0, match = 0,
            match_textured = 0, fmt_bindings = 0, fmt_image_backed = 0;
        std::array<u64, 8> special_type{}; // ineligible draws by first special BufferType
        std::array<u64, 2> c_draws{}, c_same{}, c_match{}; // [0] liftable, [1] eligible control
    };
    static FlatCensus fc;
    if ((++st.draws & 0x1fffff) == 0) {
        const double d = static_cast<double>(st.draws);
        LOG_INFO(Render_Vulkan,
                 "DrawRepeatSkip: mode={} draws={} repeats={} ({:.1f}%) | not a repeat: "
                 "ineligible={:.1f}% pipeline={:.1f}% key={:.1f}% render_state={:.1f}% "
                 "recorded_between={:.1f}% dirty_epoch={:.1f}% image_epoch={:.1f}% "
                 "cache_gen={:.1f}% other={:.1f}% gated={:.1f}% heap={:.1f}% | verified={} "
                 "mismatches={} heap_repeats={} clip_refused={} clip_repeats={}",
                 g_draw_repeat, st.draws, st.repeats, 100.0 * st.repeats / d,
                 100.0 * st.f_ineligible / d, 100.0 * st.f_pipeline / d, 100.0 * st.f_key / d,
                 100.0 * st.f_rs / d, 100.0 * st.f_seq / d, 100.0 * st.f_epoch / d,
                 100.0 * st.f_image / d, 100.0 * st.f_gen / d, 100.0 * st.f_other / d,
                 100.0 * st.f_gated / d, 100.0 * st.f_heap / d, st.verified, st.mismatches,
                 st.heap_repeats, st.f_clip, st.clip_repeats);
        st.f_heap = st.heap_repeats = st.f_clip = st.clip_repeats = 0;
        LOG_INFO(Render_Vulkan,
                 "DrawRepeatWhy: ineligible draws by global={} dma={} written={} formatted={} "
                 "special={} image_written={} flatbuf={}",
                 why[0], why[1], why[2], why[3], why[4], why[5], why[6]);
        why = {};
        if (flat_census) {
            LOG_INFO(Render_Vulkan,
                     "DrawRepeatFlatCensus: draws={} liftable_ineligible={} (textured {}) "
                     "previous_draw_same_pipeline={} would_repeat={} (textured {}) "
                     "formatted_bindings={} image_backed={} | special first type "
                     "guest/flat/bda/fault/gds/lds/clip/other={}/{}/{}/{}/{}/{}/{}/{}",
                     st.draws, fc.flat_only, fc.textured, fc.same_pipe_prev, fc.match,
                     fc.match_textured, fc.fmt_bindings, fc.fmt_image_backed,
                     fc.special_type[0], fc.special_type[1], fc.special_type[2],
                     fc.special_type[3], fc.special_type[4], fc.special_type[5],
                     fc.special_type[6], fc.special_type[7]);
            fc.flat_only = fc.textured = fc.same_pipe_prev = fc.match = fc.match_textured = 0;
            LOG_INFO(Render_Vulkan,
                     "DrawRepeatFlatControl: eligible draws={} prev same pipeline={} census key "
                     "match={} (real repeats this window {}); liftable draws={} same={} match={}",
                     fc.c_draws[1], fc.c_same[1], fc.c_match[1], st.repeats, fc.c_draws[0],
                     fc.c_same[0], fc.c_match[0]);
            fc.fmt_bindings = fc.fmt_image_backed = 0;
            fc.special_type = {};
            fc.c_draws = fc.c_same = fc.c_match = {};
        }
        st.draws = st.repeats = st.f_ineligible = st.f_pipeline = st.f_key = st.f_rs = st.f_seq =
            st.f_epoch = st.f_image = st.f_gen = st.f_other = st.verified = st.f_gated = 0;
    }
    // The state of this draw, sampled BEFORE its bindings are resolved (a change during the
    // resolution must break the next draw's repeat).
    st.c_dirty_epoch = VideoCore::g_dirty_epoch.load(std::memory_order_acquire);
    st.c_image_epoch = VideoCore::g_image_epoch.load(std::memory_order_acquire);
    st.c_buf_gen = buffer_cache.StructGeneration();
    st.c_tex_gen = texture_cache.StructGeneration();
    st.c_smp_gen = texture_cache.SamplerGeneration();
    st.c_border = liverpool->BackRegs().ta_bc_base.base_addr;
    // Key: everything the binding resolution reads from the stages.
    // Configurations in which the claim does not hold (review 2026-10-02): no recording worker
    // (RecordSeq does not see synchronous records), deferred uploads keyed on the submit epoch,
    // the untracked stream-copy path.
    static const bool config_ok = [] {
        const char* stream = std::getenv("SHADPS4_STREAM_MAX_SIZE");
        const bool stream_off = !stream || std::strtoull(stream, nullptr, 0) == 0;
        return stream_off && !VideoCore::g_epoch_defer_upload;
    }();
    bool eligible = tex_bind_memo == 1 && build_set_writes && !native_bindings &&
                    !bind_memo_stats && pending_stencil_writebacks.empty() && config_ok &&
                    scheduler.IsAsyncRecording();
    // Why a draw is ineligible (diagnostic split of f_ineligible; first reason per pipeline).
    u32 reason = eligible ? ~0u : 0u;
    if (pipeline->repeat_static == 0xff) {
        // Static per pipeline: the stage infos are the pipeline's own (BoundStages). Special
        // buffers: only the clip-plane slot with clip planes off (BindBuffers binds a null
        // descriptor for it; the enable bits are tested on both draws). Flatbuf / LDS copies get
        // a new stream offset per draw; the others are left to the full path.
        u8 cls = 1;
        u8 first = 0;
        const auto bad = [&](u8 r) {
            cls = 0;
            first = first != 0 ? first : r;
        };
        for (const auto& bound : BoundStages(pipeline)) {
            const auto* stage = bound.info;
            if (!stage) {
                continue;
            }
            if (stage->uses_dma) {
                bad(1);
            }
            for (const auto& desc : stage->buffers) {
                if (desc.is_written) {
                    bad(2);
                } else if (desc.is_formatted && !g_repeat_heap) {
                    bad(3);
                } else if (desc.IsSpecial()) {
                    if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                        bad(6);
                    } else if (desc.buffer_type != Shader::BufferType::ClipPlanes) {
                        if (first == 0) {
                            pipeline->repeat_special_type =
                                static_cast<u8>(std::min<u32>(u32(desc.buffer_type), 7u));
                        }
                        bad(4);
                    } else if (cls == 1) {
                        cls = 2;
                    }
                }
            }
            for (const auto& desc : stage->images) {
                if (desc.is_written) {
                    bad(5);
                }
            }
        }
        pipeline->repeat_static = cls;
        pipeline->repeat_why = first;
        if (cls == 0) {
            // Flat-only: every bad reason is a flat buffer (re-run of the walk, reasons only).
            bool flat = true, any = false;
            for (const auto& bound : BoundStages(pipeline)) {
                const auto* stage = bound.info;
                if (!stage) {
                    continue;
                }
                flat = flat && !stage->uses_dma;
                for (const auto& desc : stage->buffers) {
                    if (desc.buffer_type == Shader::BufferType::Flatbuf ||
                        (desc.is_formatted && !desc.is_written && !desc.IsSpecial())) {
                        any = true;
                    } else if (desc.is_written || desc.is_formatted ||
                               (desc.IsSpecial() &&
                                desc.buffer_type != Shader::BufferType::ClipPlanes)) {
                        flat = false;
                    }
                }
                for (const auto& desc : stage->images) {
                    flat = flat && !desc.is_written;
                }
            }
            pipeline->repeat_flat_only = flat && any ? 1 : 0;
        }
    }
    // Positive control: the same census key on draws the repeat path accepts as eligible.
    const bool census_control = eligible && pipeline->repeat_static != 0 &&
                                !(pipeline->repeat_static == 2 &&
                                  liverpool->BackRegs().clipper_control.user_clip_plane_enable != 0);
    if (flat_census && eligible &&
        ((pipeline->repeat_static == 0 && pipeline->repeat_flat_only != 0) || census_control))
        [[unlikely]] {
        // Census only: the key a flat-capable repeat would compare, never used for the draw.
        auto& key = fc.key[fc.kidx ^ 1];
        key.clear();
        bool textured = false;
        for (const auto& bound : BoundStages(pipeline)) {
            const auto* stage = bound.info;
            if (!stage) {
                continue;
            }
            const auto push = [&](const auto& sharp) {
                const auto* w = reinterpret_cast<const u32*>(&sharp);
                key.insert(key.end(), w, w + sizeof(sharp) / sizeof(u32));
            };
            for (const auto& desc : stage->buffers) {
                push(desc.GetSharp(bound));
            }
            for (const auto& desc : stage->images) {
                push(desc.GetSharp(bound));
                textured = true;
            }
            for (const auto& desc : stage->buffers) {
                if (desc.is_formatted && !desc.IsSpecial()) {
                    const auto vsharp = desc.GetSharp(bound);
                    ++fc.fmt_bindings;
                    if (vsharp.base_address != 0 &&
                        texture_cache.FindImageFromRange(vsharp.base_address, vsharp.GetSize())) {
                        ++fc.fmt_image_backed;
                    }
                }
            }
            for (const auto& desc : stage->samplers) {
                push(desc.GetSharp(bound));
            }
            u32 mask = stage->ud_mask.mask;
            while (mask) {
                const u32 index = std::countr_zero(mask);
                mask &= ~(1U << index);
                key.push_back(bound.user_data[index]);
            }
            // The flat buffer starts with a copy of ALL user-data registers, unread ones included
            // (per-draw values): compare only what the SRT walk appended after them.
            key.push_back(0xf1a7f1a7u);
            constexpr size_t walked = Shader::NUM_USER_DATA_REGS;
            if (bound.flattened_ud_buf.size() > walked) {
                key.insert(key.end(), bound.flattened_ud_buf.begin() + walked,
                           bound.flattened_ud_buf.end());
            }
        }
        const u32 g = census_control ? 1 : 0;
        ++fc.c_draws[g];
        if (!census_control) {
            ++fc.flat_only;
            fc.textured += textured;
        }
        const u64 dirty = st.c_dirty_epoch, image = st.c_image_epoch;
        if (fc.pipeline == pipeline && fc.draw + 1 == st.draws) {
            ++fc.c_same[g];
            if (!census_control) {
                ++fc.same_pipe_prev;
            }
            if (fc.key[fc.kidx] == key && fc.dirty == dirty && fc.image == image &&
                fc.buf_gen == st.c_buf_gen && fc.tex_gen == st.c_tex_gen &&
                fc.smp_gen == st.c_smp_gen && fc.border == st.c_border) {
                ++fc.c_match[g];
                if (!census_control) {
                    ++fc.match;
                    fc.match_textured += textured;
                }
            }
        }
        fc.kidx ^= 1;
        fc.pipeline = pipeline;
        fc.draw = st.draws;
        fc.dirty = dirty;
        fc.image = image;
        fc.buf_gen = st.c_buf_gen;
        fc.tex_gen = st.c_tex_gen;
        fc.smp_gen = st.c_smp_gen;
        fc.border = st.c_border;
    }
    if (eligible) {
        if (pipeline->repeat_static == 0) {
            eligible = false;
            reason = pipeline->repeat_why;
        } else if (pipeline->repeat_static == 2 &&
                   liverpool->BackRegs().clipper_control.user_clip_plane_enable != 0 &&
                   !g_repeat_heap) {
            eligible = false;
            reason = 4;
        }
    }
    if (eligible && pipeline->repeat_skip != 0) {
        // Adaptive gate: this pipeline's keys kept changing. No test, no base.
        --pipeline->repeat_skip;
        st.eligible = false;
        ++st.f_gated;
        return false;
    }
    auto& cand = st.words[st.kidx ^ 1];
    u32 n = 0;
    bool overflow = false;
    const auto add = [&](const auto& sharp) {
        constexpr u32 words = static_cast<u32>(sizeof(sharp) / sizeof(u32));
        if (n + words > cand.size()) {
            overflow = true;
            return;
        }
        std::memcpy(&cand[n], &sharp, sizeof(sharp));
        n += words;
    };
    st.heap_cand.clear();
    if (eligible) {
        bool heap_formatted = false;
        u64 heap_positions = 0; // which buffer positions were heap (review item 5)
        u32 buf_pos = 0;
        for (const auto& bound : BoundStages(pipeline)) {
            const auto* stage = bound.info;
            if (!stage) {
                continue;
            }
            for (const auto& desc : stage->buffers) {
                auto v = desc.GetSharp(bound);
                if (g_repeat_heap && desc.is_formatted && !desc.IsSpecial() &&
                    v.base_address != 0 &&
                    (v.GetSize() > (u64{64} << 20) || texture_cache.AnyImageAt(v.base_address))) {
                    // A formatted read over an image: its full path syncs buffer <- image
                    // (SynchronizeBufferFromImage) on every bind; a repeat must not skip that
                    // (s359: the 2.4 GB view; review item 2: any size).
                    heap_formatted = true;
                }
                if (g_repeat_heap && !desc.IsSpecial() && v.base_address >= RepeatHeapLo &&
                    v.base_address < RepeatHeapHi && v.GetSize() > 0) {
                    // The address is re-pointed at the repeat; size / stride / format stay keyed.
                    heap_formatted = heap_formatted || desc.is_formatted;
                    st.heap_cand.push_back(v);
                    v.base_address = 1;
                    if (buf_pos < 64) {
                        heap_positions |= u64{1} << buf_pos;
                    }
                }
                ++buf_pos;
                add(v);
            }
            for (const auto& desc : stage->images) {
                add(desc.GetSharp(bound));
            }
            for (const auto& desc : stage->samplers) {
                add(desc.GetSharp(bound));
            }
            if (g_repeat_heap) {
                // Heap mode: user-data registers reach the bindings only through the sharps keyed
                // above; DrawRepeatApply pushes this draw's own registers (verify: mode 2).
                continue;
            }
            u32 mask = stage->ud_mask.mask;
            while (mask) {
                const u32 index = std::countr_zero(mask);
                mask &= ~(1U << index);
                add(bound.user_data[index]);
            }
        }
        if (g_repeat_heap) {
            add(heap_positions);
            add(buf_pos);
        }
        if (overflow) {
            eligible = false;
            reason = 0;
        }
        if (heap_formatted) {
            eligible = false;
            reason = 3;
        }
    }
    st.sizes[st.kidx ^ 1] = n;
    st.eligible = eligible;
    if (!eligible) {
        ++st.f_ineligible;
        ++why[reason < why.size() ? reason : 0];
        if (reason == 4) {
            ++fc.special_type[pipeline->repeat_special_type & 7];
        }
        if ((st.f_ineligible & 0xfffff) == 0) {
            LOG_INFO(Render_Vulkan,
                     "DrawRepeatWhy: ineligible by global={} dma={} written={} formatted={} "
                     "special={} image_written={}",
                     why[0], why[1], why[2], why[3], why[4], why[5]);
        }
        return false;
    }
    if (!st.valid) {
        ++st.f_other;
        return false;
    }
    if (st.pipeline != pipeline) {
        ++st.f_pipeline;
        return false;
    }
    if (st.sizes[st.kidx] != n ||
        std::memcmp(st.words[st.kidx].data(), cand.data(), n * sizeof(u32)) != 0) {
        ++st.f_key;
        if (++pipeline->repeat_miss_run >= 4) {
            pipeline->repeat_miss_run = 0;
            pipeline->repeat_skip = 28;
        }
        return false;
    }
    pipeline->repeat_miss_run = 0;
    if (!rs_hit) {
        ++st.f_rs;
        return false;
    }
    if (scheduler.RecordSeq() != st.end_seq) {
        ++st.f_seq;
        return false;
    }
    if (st.c_dirty_epoch != st.dirty_epoch) {
        ++st.f_epoch;
        return false;
    }
    if (st.c_image_epoch != st.image_epoch) {
        ++st.f_image;
        return false;
    }
    if (st.c_buf_gen != st.buf_gen || st.c_tex_gen != st.tex_gen ||
        st.c_smp_gen != st.smp_gen || st.c_border != st.border) {
        ++st.f_gen;
        return false;
    }
    st.heap_new.clear();
    if (g_repeat_heap && pipeline->repeat_static == 2) {
        // Clip-plane slot: with planes enabled BindBuffers copies the planes into the stream
        // buffer; the base's copy stands only for the same enable bits and plane data.
        const auto& regs = liverpool->BackRegs();
        st.clip_cur[0] = regs.clipper_control.user_clip_plane_enable;
        for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
            const auto& plane = regs.clip_user_data[i];
            st.clip_cur[1 + i * 4 + 0] = plane.data_x;
            st.clip_cur[1 + i * 4 + 1] = plane.data_y;
            st.clip_cur[1 + i * 4 + 2] = plane.data_z;
            st.clip_cur[1 + i * 4 + 3] = plane.data_w;
        }
        // The planes are copied afresh below (Opus delta review item 1: the base's stream copy
        // may be overwritten by a ring wrap inside the tick), so only the enable bits must match.
        if (st.clip_cur[0] != st.clip_base[0] ||
            (st.clip_cur[0] != 0 && st.clip_base_idx == ~0u)) {
            ++st.f_clip;
            return false;
        }
    }
    if (g_repeat_heap) {
        if (st.heap_cand.size() != st.heap_base.size()) {
            ++st.f_heap;
            return false;
        }
        const bool lean = VideoCore::g_lean_bind != 0;
        if (!st.heap_cand.empty() && (!lean || !buffer_cache.LeanReadOk())) {
            // The full path would be ObtainBuffer (range notes, ASC deps, host mirror, other
            // sync modes): only the lean read path's side effects are reproduced (review item 1).
            ++st.f_heap;
            return false;
        }
        for (size_t k = 0; k < st.heap_cand.size(); ++k) {
            const auto& v = st.heap_cand[k];
            const auto& slot = st.heap_base[k];
            const VAddr addr = v.base_address;
            const u64 vsize = v.GetSize();
            const u64 size =
                lean && vsize < (u64{1} << 30) ? vsize : memory->ClampRangeSize(addr, vsize);
            // Same Buffer as the base binding (the generation test above keeps it alive), its
            // pages verified clean (no upload due), as BindBuffers' lean read would find.
            if (!slot.buffer->IsInBounds(addr, size) || !buffer_cache.ReadRangeClean(addr, size)) {
                ++st.f_heap;
                st.heap_new.clear();
                return false;
            }
            const u32 offset = static_cast<u32>(slot.buffer->Offset(addr));
            const u32 offset_aligned = Common::AlignDown(offset, slot.alignment);
            const u32 adjust = offset - offset_aligned;
            if (adjust % 4 != 0 || adjust >= 256) {
                ++st.f_heap;
                st.heap_new.clear();
                return false;
            }
            st.heap_new.push_back(
                {vk::DescriptorBufferInfo{slot.buffer->Handle(), offset_aligned, size + adjust},
                 adjust});
        }
        st.heap_repeats += !st.heap_new.empty();
    }
    st.clip_patch = false;
    if (g_repeat_heap && pipeline->repeat_static == 2 && st.clip_cur[0] != 0) {
        // As BindBuffers' ClipPlanes branch: this draw's planes into the stream buffer. A Map that
        // waits (ring wrap) submits work, which moves RecordSeq: then take the full path.
        const u64 seq0 = scheduler.RecordSeq();
        const auto& regs = liverpool->BackRegs();
        std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
        for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
            const auto& plane = regs.clip_user_data[i];
            planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
            planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
            planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
            planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
        }
        auto& vk_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
        const u32 ubo_size = static_cast<u32>(sizeof(planes));
        const u64 offset = vk_buffer.Copy(planes.data(), ubo_size, instance.StorageMinAlignment());
        if (scheduler.RecordSeq() != seq0) {
            ++st.f_clip;
            st.heap_new.clear();
            return false;
        }
        st.clip_new = vk::DescriptorBufferInfo{vk_buffer.Handle(), offset, ubo_size};
        st.clip_patch = true;
        ++st.clip_repeats;
    }
    ++st.repeats;
    return true;
}

void Rasterizer::DrawRepeatApply(const GraphicsPipeline* pipeline, bool patch_infos) {
    if (VideoCore::g_lane_war && !Scheduler::t_lane_mode) [[unlikely]] {
        // q9c (2b): no bindings resolved. q29b: a heap repeat binds new addresses -> every probe.
        VideoCore::LaneProbeReadAllMain(scheduler.CurrentTick(), draw_repeat.heap_new.empty() ? 19u : 25u);
    }
    // What BindResources leaves besides the resolved infos (still those of the previous draw):
    // push_data. The viewport transform and the pushed registers come from this draw, the
    // buffer offsets from the bindings, which are the previous draw's.
    buffer_barriers.clear();
    binding_compute = false;
    g_dc_bind_buf = 0;
    g_dc_bind_tex = 0;
    Shader::Backend::Bindings binding{};
    push_data = MakeUserData(liverpool->BackRegs());
    for (const auto& bound : BoundStages(pipeline)) {
        if (bound.info) {
            bound.PushUd(binding, push_data);
        }
    }
    push_data.buf_offsets = draw_repeat.push_data.buf_offsets;
    // SHADPS4_DRAW_REPEAT_HEAP: the heap bindings point at this draw's data.
    const auto& st = draw_repeat;
    for (size_t k = 0; k < st.heap_new.size(); ++k) {
        const auto& slot = st.heap_base[k];
        ASSERT(slot.bind_idx < push_data.buf_offsets.size() &&
               (!patch_infos || slot.info_idx < buffer_infos.size()));
        push_data.buf_offsets[slot.bind_idx] = st.heap_new[k].adjust;
        if (patch_infos) {
            buffer_infos[slot.info_idx] = st.heap_new[k].info;
        }
    }
    if (st.clip_patch && patch_infos) {
        ASSERT(st.clip_base_idx < buffer_infos.size());
        buffer_infos[st.clip_base_idx] = st.clip_new;
    }
}

void Rasterizer::DrawRepeatVerify(const GraphicsPipeline* pipeline) {
    auto& st = draw_repeat;
    ++st.verified;
    const char* what = nullptr;
    if (scheduler.RecordSeq() != st.v_seq) {
        what = "the full path recorded commands";
    } else if (!buffer_barriers.empty()) {
        what = "buffer barriers";
    } else if (st.v_buffers.size() != buffer_infos.size() ||
               st.v_images.size() != image_infos.size()) {
        what = "info counts";
    }
    for (size_t i = 0; !what && i < buffer_infos.size(); ++i) {
        const auto& a = st.v_buffers[i];
        const auto& b = buffer_infos[i];
        // Both sides made a fresh copy of the same registers at this point: offsets differ.
        const bool clip_slot = st.clip_patch && i == st.clip_base_idx;
        if (a.buffer != b.buffer || (!clip_slot && a.offset != b.offset) || a.range != b.range) {
            what = clip_slot ? "clip-plane buffer info" : "buffer info";
        }
    }
    for (size_t i = 0; !what && i < image_infos.size(); ++i) {
        const auto& a = st.v_images[i];
        const auto& b = image_infos[i];
        if (a.sampler != b.sampler || a.imageView != b.imageView ||
            a.imageLayout != b.imageLayout) {
            what = "image info";
        }
    }
    if (!what) {
        // push_data as the skip would have built it against the full path's.
        const Shader::PushData full = push_data;
        DrawRepeatApply(pipeline, false);
        if (push_data.xoffset != full.xoffset || push_data.yoffset != full.yoffset ||
            push_data.xscale != full.xscale || push_data.yscale != full.yscale ||
            push_data.ud_regs != full.ud_regs || push_data.buf_offsets != full.buf_offsets) {
            what = "push data";
        }
        push_data = full;
    }
    if (what && ++st.mismatches <= 64) {
        LOG_ERROR(Render_Vulkan,
                  "DrawRepeat: MISMATCH {} (pipeline {:#x}, {} buffer infos, {} image infos, {} "
                  "heap slots) recorded by:{}",
                  what, reinterpret_cast<u64>(pipeline), buffer_infos.size(), image_infos.size(),
                  st.heap_new.size(), g_rp_seq_who.empty() ? " (outside the bindings)" : g_rp_seq_who);
    }
}

namespace {

// SHADPS4_PB_CENSUS state (back thread only). See scripts/patch/pb-census-2026-10-05.py.
constexpr std::array<u64, 6> kPbLeads{1, 2, 4, 8, 16, 64};
constexpr u32 kPbLeadCount = static_cast<u32>(kPbLeads.size());
constexpr u32 kPbStamps = 11;            // 0-8 the design set, 9 BackContextEpoch, 10 CurrentTick
constexpr u32 kPbDesign = 9;
constexpr u32 kPbMemo = kPbStamps;       // component index of the memo stamp
constexpr u32 kPbComps = kPbStamps + 1;
constexpr std::array<const char*, kPbComps> kPbCompNames{
    "dirty", "image", "bufgen", "texgen", "smpgen", "border",
    "gpuclear", "faultjobs", "metaclear", "ctx", "tick", "memo"};
enum PbClass : u32 {
    PbRepeat,
    PbStatic,
    PbRsMiss,
    PbTexMiss,
    PbGen,
    PbUpload,
    PbWritten,
    PbReadMiss,
    PbPureRec,
    PbPure,
    PbClasses
};
constexpr std::array<const char*, PbClasses> kPbClassNames{
    "repeat", "static",     "rs_miss",  "tex_miss", "gen",
    "upload", "written",    "rmemo_miss", "pure_rec", "pure"};
constexpr u32 kPbPhases = 5; // rs, bind, begin, vi, rest

struct PbCensus {
    u64 n{};
    std::array<u64, kPbStamps> prev{};
    std::array<u64, kPbStamps> last_change{};
    bool sampled{};
    u64 rs_w{};
    std::array<u64, 5> tsc{};
    u64 self{};
    u64 seq0{}, seq2{}, seq3{}, seq4{};
    std::array<u8, 16 + 32 * 32 + 32 * 16> vi_prev{};
    u32 vi_prev_bytes{};
    std::array<u64, 4> ib_prev{};
    u64 up0{}, up2{}, up3{}, up4{};
    u64 bg0{}, bg2{}, bg3{}, bg4{};
    u64 tg0{}, tg2{}, tg3{}, tg4{};
    u64 rm0{}, rm2{};
    const GraphicsPipeline* prev_pipeline{};
    // Window.
    u64 w_start_ns{}, w_start_tsc{}, w_start_idle{};
    std::array<u64, PbClasses> cls{};
    std::array<std::array<u64, kPbLeadCount>, PbClasses> acc{};    // per class, with the memo stamp
    std::array<std::array<u64, kPbLeadCount>, PbClasses> acc_nm{}; // per class, design set only
    // [all draws, resolvable draws (pure, pure_rec, rmemo_miss)]
    std::array<std::array<std::array<u64, kPbLeadCount>, kPbComps>, 2> inv{};
    std::array<std::array<u64, kPbLeadCount>, kPbComps> only{}; // resolvable, the only bad one
    // Commit side, [repeat, other]: draws, vi_same, begin_rec0, vi_rec0, rest_rec1, draw_only.
    std::array<std::array<u64, 6>, 2> vi{};
    // v3, [repeat, other]: draws, indexed, ib_same, vb_same, all_same, gpu-modified vertex range.
    std::array<std::array<u64, 6>, 2> vk{};
    std::array<u64, PbClasses> obt{}; // sampled vertex / index lookup cycles per class
    std::array<u64, 32> vk_prev{};
    u32 vk_prev_n{};
    u64 ib_prev_buf{};
    u32 ib_prev_type{};
    // Dispatches: count, sampled, bind cyc, total cyc, sampled written, bind cyc without written.
    std::array<u64, 6> disp{};
    u64 nd{};
    bool d_sampled{};
    u64 d_t0{}, d_t1{};
    std::array<std::array<u64, kPbPhases>, PbClasses> cyc{};
    std::array<u64, PbClasses> smp{};
    u64 self_cyc{}, self_n{};
    std::array<u64, 2> pure_attr{}; // pure draws with a written buffer / image, with a formatted one
    // Keys (SHADPS4_PB_CENSUS & 2).
    u64 k_draws{}, k_prev_same{}, k_prev_valid{}, k_pipe_any{}, k_pipe_valid{}, k_pipe_valid_pure{};
    std::array<u64, 5> k_prev_by{}; // not taken by the skip: formatted, written, dma/special, rs miss, other
    u64 k_diff{}, k_bind_total{}, k_bind_eq{}, k_stage_total{}, k_stage_eq{};
};
PbCensus g_pb;

template <size_t N>
std::string PbJoin(const std::array<u64, N>& values) {
    std::string out;
    for (size_t i = 0; i < N; ++i) {
        out += fmt::format("{}{}", i ? "/" : "", values[i]);
    }
    return out;
}

void PbCensusMaybeLog(PbCensus& pb) {
    const u64 now_ns = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            std::chrono::steady_clock::now().time_since_epoch())
                                            .count());
    const u64 now_tsc = __rdtsc();
    const u64 idle = Common::FaultMonitor::back_idle_total_ticks.load(std::memory_order_relaxed);
    if (pb.w_start_ns == 0) {
        pb.w_start_ns = now_ns;
        pb.w_start_tsc = now_tsc;
        pb.w_start_idle = idle;
        return;
    }
    if (now_ns - pb.w_start_ns < 5'000'000'000ull) {
        return;
    }
    const double wall_ns = static_cast<double>(now_ns - pb.w_start_ns);
    const double ns_per_tick = wall_ns / static_cast<double>(std::max<u64>(1, now_tsc - pb.w_start_tsc));
    const double idle_ms = static_cast<double>(idle - pb.w_start_idle) * ns_per_tick / 1e6;
    std::string line = fmt::format(
        "PbCensus: v=2 n={} wall_ms={:.1f} idle_ms={:.1f} ns_per_tick={:.5f} frames={} cls={} "
        "self_cyc={} self_n={} pure_attr={} vi_repeat={} vi_other={}",
        pb.n, wall_ns / 1e6, idle_ms, ns_per_tick,
        Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed), PbJoin(pb.cls),
        pb.self_cyc, pb.self_n, PbJoin(pb.pure_attr), PbJoin(pb.vi[0]), PbJoin(pb.vi[1]));
    line += fmt::format(" v3=1 vk_repeat={} vk_other={} disp={}", PbJoin(pb.vk[0]),
                        PbJoin(pb.vk[1]), PbJoin(pb.disp));
    for (u32 k = 0; k < PbClasses; ++k) {
        line += fmt::format(" obt_{}={}", kPbClassNames[k], pb.obt[k]);
    }
    for (u32 k = 0; k < PbClasses; ++k) {
        line += fmt::format(" acc_{}={} accnm_{}={}", kPbClassNames[k], PbJoin(pb.acc[k]),
                            kPbClassNames[k], PbJoin(pb.acc_nm[k]));
    }
    for (u32 c = 0; c < kPbComps; ++c) {
        line += fmt::format(" inv_all_{}={} inv_res_{}={} only_res_{}={}", kPbCompNames[c],
                            PbJoin(pb.inv[0][c]), kPbCompNames[c], PbJoin(pb.inv[1][c]),
                            kPbCompNames[c], PbJoin(pb.only[c]));
    }
    for (u32 k = 0; k < PbClasses; ++k) {
        line += fmt::format(" cyc_{}={}", kPbClassNames[k], PbJoin(pb.cyc[k]));
    }
    line += fmt::format(" smp={}", PbJoin(pb.smp));
    if (VideoCore::g_pb_census & 2) {
        line += fmt::format(" keys={}/{}/{}/{}/{}/{} prev_by={} diff={}/{}/{}/{}/{}", pb.k_draws,
                            pb.k_prev_same, pb.k_prev_valid, pb.k_pipe_any, pb.k_pipe_valid,
                            pb.k_pipe_valid_pure, PbJoin(pb.k_prev_by), pb.k_diff,
                            pb.k_bind_total, pb.k_bind_eq, pb.k_stage_total, pb.k_stage_eq);
    }
    LOG_INFO(Render_Vulkan, "{}", line);
    const u64 n = pb.n;
    const auto prev = pb.prev;
    const auto last_change = pb.last_change;
    const auto* prev_pipeline = pb.prev_pipeline;
    const auto vi_prev = pb.vi_prev;
    const u32 vi_prev_bytes = pb.vi_prev_bytes;
    const auto ib_prev = pb.ib_prev;
    const auto vk_prev = pb.vk_prev;
    const u32 vk_prev_n = pb.vk_prev_n;
    const u64 ib_prev_buf = pb.ib_prev_buf;
    const u32 ib_prev_type = pb.ib_prev_type;
    const u64 nd = pb.nd;
    pb = PbCensus{};
    pb.n = n;
    pb.prev = prev;
    pb.last_change = last_change;
    pb.prev_pipeline = prev_pipeline;
    pb.vi_prev = vi_prev;
    pb.vi_prev_bytes = vi_prev_bytes;
    pb.ib_prev = ib_prev;
    pb.vk_prev = vk_prev;
    pb.vk_prev_n = vk_prev_n;
    pb.ib_prev_buf = ib_prev_buf;
    pb.ib_prev_type = ib_prev_type;
    pb.nd = nd;
    pb.w_start_ns = now_ns;
    pb.w_start_tsc = now_tsc;
    pb.w_start_idle = idle;
}

} // Anonymous namespace

void Rasterizer::PbCensusBegin() {
    auto& pb = g_pb;
    const u64 n = ++pb.n;
    VideoCore::g_pb_draw.store(n, std::memory_order_relaxed);
    VideoCore::t_pb_memo_max = 0;
    pb.sampled = (n & 63) == 0;
    const u64 c0 = pb.sampled ? __rdtsc() : 0;
    const auto& regs = liverpool->BackRegs();
    const std::array<u64, kPbStamps> s{
        VideoCore::g_dirty_epoch.load(std::memory_order_acquire),
        VideoCore::g_image_epoch.load(std::memory_order_acquire),
        buffer_cache.StructGeneration(),
        texture_cache.StructGeneration(),
        texture_cache.SamplerGeneration(),
        static_cast<u64>(regs.ta_bc_base.base_addr),
        VideoCore::MemoryTracker::gpu_clear_epoch.load(std::memory_order_acquire),
        buffer_cache.FaultJobsAdmitted(),
        texture_cache.MetaClearEpoch(),
        liverpool->BackContextEpoch(),
        scheduler.CurrentTick()};
    for (u32 c = 0; c < kPbStamps; ++c) {
        if (s[c] != pb.prev[c]) {
            pb.last_change[c] = n;
            pb.prev[c] = s[c];
        }
    }
    pb.rs_w = rs_memo_pb_written;
    pb.seq0 = scheduler.RecordSeq();
    pb.up0 = buffer_cache.SyncUploadBytesTotal();
    pb.bg0 = s[2];
    pb.tg0 = s[3];
    pb.rm0 = buffer_cache.ReadMemoMissesTotal();
    VideoCore::t_pb_vi = {};
    VideoCore::t_pb_sampled = pb.sampled;
    if (pb.sampled) {
        const u64 c1 = __rdtsc();
        pb.self = c1 - c0;
        pb.tsc[0] = c1;
    }
}

void Rasterizer::PbCensusDispatch(const ComputePipeline* pipeline, u32 point) {
    auto& pb = g_pb;
    if (point == 0) {
        ++pb.disp[0];
        pb.d_sampled = (++pb.nd & 63) == 0;
        if (pb.d_sampled) {
            pb.d_t0 = __rdtsc();
        }
        return;
    }
    if (!pb.d_sampled) {
        return;
    }
    if (point == 1) {
        pb.d_t1 = __rdtsc();
        return;
    }
    const u64 t2 = __rdtsc();
    bool written = false;
    for (const auto& bound : BoundStages(pipeline)) {
        const auto* stage = bound.info;
        if (!stage) {
            continue;
        }
        for (const auto& desc : stage->buffers) {
            written |= desc.is_written;
        }
        for (const auto& desc : stage->images) {
            written |= desc.is_written;
        }
    }
    ++pb.disp[1];
    pb.disp[2] += pb.d_t1 - pb.d_t0;
    pb.disp[3] += t2 - pb.d_t0;
    pb.disp[4] += written;
    if (!written) {
        pb.disp[5] += pb.d_t1 - pb.d_t0;
    }
    pb.d_sampled = false;
}

void Rasterizer::PbCensusPoint(u32 point) {
    auto& pb = g_pb;
    switch (point) {
    case 2:
        pb.seq2 = scheduler.RecordSeq();
        pb.up2 = buffer_cache.SyncUploadBytesTotal();
        pb.bg2 = buffer_cache.StructGeneration();
        pb.tg2 = texture_cache.StructGeneration();
        pb.rm2 = buffer_cache.ReadMemoMissesTotal();
        break;
    case 3:
        pb.seq3 = scheduler.RecordSeq();
        pb.up3 = buffer_cache.SyncUploadBytesTotal();
        pb.bg3 = buffer_cache.StructGeneration();
        pb.tg3 = texture_cache.StructGeneration();
        break;
    case 4:
        pb.seq4 = scheduler.RecordSeq();
        pb.up4 = buffer_cache.SyncUploadBytesTotal();
        pb.bg4 = buffer_cache.StructGeneration();
        pb.tg4 = texture_cache.StructGeneration();
        break;
    default:
        break;
    }
    if (pb.sampled && point < pb.tsc.size()) {
        pb.tsc[point] = __rdtsc();
    }
}

void Rasterizer::PbCensusEnd(const GraphicsPipeline* pipeline, bool rs_hit, bool repeat_skip,
                             const AmdGpu::CpSplit::VertexInputsPayload* front_vi,
                             bool is_indexed, u32 index_offset) {
    auto& pb = g_pb;
    const u64 t_end = pb.sampled ? __rdtsc() : 0;
    const u64 seq_end = scheduler.RecordSeq();
    const u64 n = pb.n;
    if (pipeline->pb_static == 0xff) {
        u8 bits = 0;
        for (const auto& bound : BoundStages(pipeline)) {
            const auto* stage = bound.info;
            if (!stage) {
                continue;
            }
            if (stage->uses_dma) {
                bits |= 1;
            }
            for (const auto& desc : stage->buffers) {
                if (desc.IsSpecial()) {
                    bits |= desc.buffer_type == Shader::BufferType::ClipPlanes ? 16 : 8;
                } else {
                    bits |= desc.is_written ? 2 : 0;
                    bits |= desc.is_formatted ? 4 : 0;
                }
            }
            for (const auto& desc : stage->images) {
                bits |= desc.is_written ? 32 : 0;
            }
        }
        pipeline->pb_static = bits;
    }
    const u8 sbits = pipeline->pb_static;
    const bool clip_on = liverpool->BackRegs().clipper_control.user_clip_plane_enable != 0;
    PbClass cls;
    if (repeat_skip) {
        cls = PbRepeat;
    } else if ((sbits & (1 | 8)) != 0 || ((sbits & 16) != 0 && clip_on)) {
        cls = PbStatic;
    } else if (!rs_hit) {
        cls = PbRsMiss;
    } else if (draw_repeat.tex_nonhit_stages != 0) {
        cls = PbTexMiss;
    } else if (pb.bg2 != pb.bg0 || pb.tg2 != pb.tg0 || pb.bg4 != pb.bg3 || pb.tg4 != pb.tg3) {
        cls = PbGen;
    } else if (pb.up2 != pb.up0 || pb.up4 != pb.up3) {
        cls = PbUpload;
    } else if ((sbits & (2 | 32)) != 0) {
        cls = PbWritten;
    } else if (pb.rm2 != pb.rm0) {
        cls = PbReadMiss;
    } else if (pb.seq2 != pb.seq0) {
        cls = PbPureRec;
    } else {
        cls = PbPure;
    }
    const bool pure = cls == PbPure || cls == PbPureRec;
    const bool resolvable = pure || cls == PbReadMiss;
    ++pb.cls[cls];
    {
        // v3: the Vulkan buffers the vertex / index binding resolved, against the previous draw.
        const auto& v = VideoCore::t_pb_vi;
        const bool vb_same = v.nranges == pb.vk_prev_n &&
                             std::equal(v.bufs.begin(), v.bufs.begin() + v.nranges,
                                        pb.vk_prev.begin());
        const bool ib_same = is_indexed && v.ib_buf == pb.ib_prev_buf &&
                             v.ib_type == pb.ib_prev_type;
        auto& k = pb.vk[repeat_skip ? 0 : 1];
        ++k[0];
        k[1] += is_indexed;
        k[2] += ib_same;
        k[3] += vb_same;
        k[4] += vb_same && (ib_same || !is_indexed);
        k[5] += v.gpu_mod != 0;
        pb.vk_prev = v.bufs;
        pb.vk_prev_n = v.nranges;
        pb.ib_prev_buf = is_indexed ? v.ib_buf : 0;
        pb.ib_prev_type = is_indexed ? v.ib_type : 0;
        if (pb.sampled) {
            pb.obt[cls] += v.obtain_cyc;
        }
    }
    {
        // Commit side: vertex inputs equal to the previous draw's, records per window.
        bool vi_same = false;
        if (front_vi) {
            const u32 bytes = AmdGpu::CpSplit::VertexInputsBytes(front_vi->count,
                                                                 front_vi->range_count);
            if (bytes <= pb.vi_prev.size()) {
                vi_same = bytes == pb.vi_prev_bytes &&
                          std::memcmp(pb.vi_prev.data(), front_vi, bytes) == 0;
                std::memcpy(pb.vi_prev.data(), front_vi, bytes);
                pb.vi_prev_bytes = bytes;
            } else {
                pb.vi_prev_bytes = 0;
            }
        } else {
            pb.vi_prev_bytes = 0;
        }
        const auto& r = liverpool->BackRegs();
        const std::array<u64, 4> ib{
            is_indexed ? r.index_base_address.Address<u64>() : ~0ull,
            is_indexed ? u64{index_offset} : 0ull, is_indexed ? u64{r.num_indices} : 0ull,
            is_indexed && r.index_buffer_type.index_type == AmdGpu::IndexType::Index16 ? 1ull : 0ull};
        vi_same = vi_same && ib == pb.ib_prev;
        pb.ib_prev = ib;
        const bool begin0 = pb.seq3 == pb.seq2;
        const bool vi0 = pb.seq4 == pb.seq3;
        const bool rest1 = seq_end - pb.seq4 == 1;
        auto& v = pb.vi[repeat_skip ? 0 : 1];
        ++v[0];
        v[1] += vi_same;
        v[2] += begin0;
        v[3] += vi0;
        v[4] += rest1;
        v[5] += vi_same && begin0 && rest1;
    }
    if (pure) {
        pb.pure_attr[0] += (sbits & (2 | 32)) != 0;
        pb.pure_attr[1] += (sbits & 4) != 0;
    }

    // (a) acceptance at each lead.
    u64 memo_w = VideoCore::t_pb_memo_max;
    if (rs_hit) {
        memo_w = std::max(memo_w, pb.rs_w);
    }
    u64 design_max = 0;
    for (u32 c = 0; c < kPbDesign; ++c) {
        design_max = std::max(design_max, pb.last_change[c]);
    }
    for (u32 li = 0; li < kPbLeadCount; ++li) {
        const u64 lead = kPbLeads[li];
        if (n <= lead) {
            continue;
        }
        const u64 base = n - lead;
        std::array<bool, kPbComps> bad{};
        for (u32 c = 0; c < kPbStamps; ++c) {
            bad[c] = pb.last_change[c] > base;
        }
        bad[kPbMemo] = memo_w >= base;
        // v3: `only` over the design stamps; the memo stamp is a retry counter, not validity.
        u32 nbad_d = 0;
        u32 which = 0;
        for (u32 c = 0; c < kPbDesign; ++c) {
            if (bad[c]) {
                ++nbad_d;
                which = c;
            }
        }
        const bool design_ok = nbad_d == 0;
        const u32 nbad = nbad_d + (bad[kPbMemo] ? 1u : 0u);
        for (u32 k = 0; k < (resolvable ? 2u : 1u); ++k) {
            for (u32 c = 0; c < kPbComps; ++c) {
                pb.inv[k][c][li] += bad[c];
            }
        }
        pb.acc[cls][li] += nbad == 0;
        pb.acc_nm[cls][li] += design_ok;
        if (resolvable && nbad_d == 1) {
            ++pb.only[which][li];
        }
    }

    // (d) timing of the sampled draw.
    if (pb.sampled) {
        auto& cyc = pb.cyc[cls];
        cyc[0] += pb.tsc[1] - pb.tsc[0];
        cyc[1] += pb.tsc[2] - pb.tsc[1];
        cyc[2] += pb.tsc[3] - pb.tsc[2];
        cyc[3] += pb.tsc[4] - pb.tsc[3];
        cyc[4] += t_end - pb.tsc[4];
        ++pb.smp[cls];
    }

    // (c) binding inputs against the previous draw of the same pipeline.
    if (VideoCore::g_pb_census & 2) {
        auto& key = pipeline->pb_key;
        const bool have = pipeline->pb_last_draw != 0;
        u32 pos = 0;
        u32 nb = 0;
        u32 neq = 0;
        u32 ns = 0;
        u32 nseq = 0;
        const auto cmp = [&](const void* data, u32 words) {
            bool eq = false;
            if (pos + words <= key.size()) {
                eq = have && std::memcmp(&key[pos], data, words * sizeof(u32)) == 0;
            } else {
                key.resize(pos + words);
            }
            std::memcpy(&key[pos], data, words * sizeof(u32));
            pos += words;
            ++nb;
            neq += eq;
            return eq;
        };
        for (const auto& bound : BoundStages(pipeline)) {
            const auto* stage = bound.info;
            if (!stage) {
                continue;
            }
            bool stage_eq = true;
            bool any = false;
            for (const auto& desc : stage->buffers) {
                const auto sharp = desc.GetSharp(bound);
                stage_eq &= cmp(&sharp, static_cast<u32>(sizeof(sharp) / sizeof(u32)));
                any = true;
            }
            for (const auto& desc : stage->images) {
                const auto sharp = desc.GetSharp(bound);
                stage_eq &= cmp(&sharp, static_cast<u32>(sizeof(sharp) / sizeof(u32)));
                any = true;
            }
            for (const auto& desc : stage->samplers) {
                const auto sharp = desc.GetSharp(bound);
                stage_eq &= cmp(&sharp, static_cast<u32>(sizeof(sharp) / sizeof(u32)));
                any = true;
            }
            u32 mask = stage->ud_mask.mask;
            while (mask) {
                const u32 index = std::countr_zero(mask);
                mask &= ~(1U << index);
                const u32 value = bound.user_data[index];
                stage_eq &= cmp(&value, 1);
                any = true;
            }
            if (any) {
                ++ns;
                nseq += stage_eq;
            }
        }
        const bool key_eq = have && pos == key.size() && neq == nb;
        ++pb.k_draws;
        if (pb.prev_pipeline == pipeline) {
            if (key_eq) {
                ++pb.k_prev_same;
                if (n > 1 && design_max <= n - 1) {
                    ++pb.k_prev_valid;
                    if (!repeat_skip) {
                        const u32 why = (sbits & 4) != 0            ? 0
                                        : (sbits & (2 | 32)) != 0   ? 1
                                        : (sbits & (1 | 8 | 16)) != 0 ? 2
                                        : !rs_hit                    ? 3
                                                                     : 4;
                        ++pb.k_prev_by[why];
                    }
                }
            }
        } else if (have && key_eq) {
            ++pb.k_pipe_any;
            if (design_max <= pipeline->pb_last_draw) {
                ++pb.k_pipe_valid;
                pb.k_pipe_valid_pure += pipeline->pb_last_pure != 0;
            }
        }
        if (have && !key_eq) {
            ++pb.k_diff;
            pb.k_bind_total += nb;
            pb.k_bind_eq += neq;
            pb.k_stage_total += ns;
            pb.k_stage_eq += nseq;
        }
        pipeline->pb_last_draw = n;
        pipeline->pb_last_pure = pure ? 1 : 0;
    }
    pb.prev_pipeline = pipeline;
    if (pb.sampled) {
        pb.self_cyc += pb.self + (__rdtsc() - t_end);
        ++pb.self_n;
    }
    if ((n & 4095) == 0) {
        PbCensusMaybeLog(pb);
    }
}

void Rasterizer::DrawRepeatStore(const GraphicsPipeline* pipeline, bool completed) {
    auto& st = draw_repeat;
    // A base must have resolved every textured stage through a memo hit (a repeat inherits the
    // base's count: BindResources did not run).
    st.valid = completed && st.eligible && st.tex_nonhit_stages == 0;
    if (!st.valid) {
        return;
    }
    st.pipeline = pipeline;
    st.kidx ^= 1; // the candidate becomes the key
    st.dirty_epoch = st.c_dirty_epoch;
    st.image_epoch = st.c_image_epoch;
    st.buf_gen = st.c_buf_gen;
    st.tex_gen = st.c_tex_gen;
    st.smp_gen = st.c_smp_gen;
    st.border = st.c_border;
    st.push_data = push_data;
    st.end_seq = scheduler.RecordSeq();
    st.heap_base = st.heap_pending;
    st.clip_base_idx = st.clip_info_idx;
    if (g_repeat_heap && pipeline->repeat_static == 2) {
        const auto& regs = liverpool->BackRegs();
        st.clip_base[0] = regs.clipper_control.user_clip_plane_enable;
        for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
            const auto& plane = regs.clip_user_data[i];
            st.clip_base[1 + i * 4 + 0] = plane.data_x;
            st.clip_base[1 + i * 4 + 1] = plane.data_y;
            st.clip_base[1 + i * 4 + 2] = plane.data_z;
            st.clip_base[1 + i * 4 + 3] = plane.data_w;
        }
    }
}

bool Rasterizer::BindResources(const Pipeline* pipeline) {
    if (!pending_stencil_writebacks.empty()) {
        FlushStencilWritebacks();
    }
    if (IsComputeImageCopy(pipeline) || IsComputeMetaClear(pipeline) ||
        IsComputeImageClear(pipeline)) {
        return false;
    }

#if SHADPS4_BINDING_PROBE
    ProbeBindingChanges(pipeline, BoundStages(pipeline));
#endif
    // q29 E6: collect this graphics bind's probe read stamps for a later draw repeat.
    const bool q29_collect = VideoCore::g_lane_repeat_exact && !pipeline->IsCompute() &&
                             Common::FaultMonitor::t_is_cp_thread && !Scheduler::t_lane_mode;
    if (q29_collect) {
        VideoCore::g_draw_stamps_n = 0;
        VideoCore::g_draw_stamps_over = false;
        VideoCore::g_draw_stamps_valid = false;
        VideoCore::g_draw_stamps_on = true;
    }
    SCOPE_EXIT {
        if (q29_collect) {
            VideoCore::g_draw_stamps_on = false;
            VideoCore::g_draw_stamps_valid = true;
        }
    };

    set_write_index = 0;
    set_writes.clear();
    p1c_bind_serial = p1c_dispatch_serial;
    buffer_barriers.clear();
    if (!native_carry.empty()) [[unlikely]] {
        // SHADPS4_NATIVE_PASS: barriers a failed join already took from the hazard state.
        buffer_barriers.insert(buffer_barriers.end(), native_carry.begin(), native_carry.end());
        native_carry.clear();
    }
    buffer_infos.clear();
    image_infos.clear();
    draw_repeat.tex_nonhit_stages = 0;
    draw_repeat.heap_pending.clear();
    draw_repeat.clip_info_idx = ~0u;
    if (g_draw_repeat == 2) {
        g_rp_seq_who.clear();
    }
    g_dc_bind_buf = 0;
    g_dc_bind_tex = 0;
    g_dc_tex_sub = {};
    g_dc_slow_img = {};
    g_p5 = {};

    bool uses_dma = false;
    binding_compute = pipeline->IsCompute();
    if (g_static_key_census && !binding_compute) [[unlikely]] {
        auto& c = GetStaticKeyCensus();
        const u64 frame = Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
        if (frame != c.frame) {
            c.frame = frame;
            ++c.frames;
            c.keys_frames_sum += c.keys_this_frame;
            c.keys_this_frame = 0;
        }
        u64 h = 0xcbf29ce484222325ULL;
        h = KeyMix(h, &pipeline, sizeof(pipeline));
        bool textured = false;
        c.ranges.clear();
        bool big = false;
        for (const auto& bound : BoundStages(pipeline)) {
            const auto* stage = bound.info;
            if (!stage) {
                continue;
            }
            for (const auto& desc : stage->buffers) {
                auto vsharp = desc.GetSharp(bound);
                if (vsharp.base_address >= HeapLo && vsharp.base_address < HeapHi) {
                    vsharp.base_address = 1; // heap: a dynamic offset, not part of the key
                    ++c.heap_bindings;
                } else if (!desc.IsSpecial()) {
                    ++c.static_bindings;
                    const u64 sz = vsharp.GetSize();
                    if (sz > (u64{64} << 20)) {
                        big = true;
                        ++c.big_bindings;
                    } else if (vsharp.base_address != 0 && sz != 0) {
                        const VAddr lo = vsharp.base_address;
                        c.ranges.emplace_back(lo, lo + sz);
                    }
                }
                h = KeyMix(h, &vsharp, sizeof(vsharp));
            }
            for (const auto& desc : stage->images) {
                const auto t = desc.GetSharp(bound);
                h = KeyMix(h, &t, sizeof(t));
                textured = true;
            }
            for (const auto& desc : stage->samplers) {
                const auto smp = desc.GetSharp(bound);
                h = KeyMix(h, &smp, sizeof(smp));
            }
        }
        ++c.draws;
        c.textured += textured;
        {
            const u64 seq = draw_repeat.draws; // DrawRepeatMatch counts every draw
            g_skey_last_prev = h == c.prev_key;
            if (h == c.prev_key) {
                ++c.prev_match;
                c.prev_match_textured += textured;
                c.prev_match_adjacent += seq == c.prev_full_seq + 1;
            }
            c.prev_key = h;
            c.prev_full_seq = seq;
        }
        StaticKeyRec now{};
        now.frame = frame;
        now.clock = VideoCore::g_granule_clock.load(std::memory_order_acquire);
        now.img_epoch = VideoCore::g_image_epoch.load(std::memory_order_acquire);
        now.tex_gen = texture_cache.StructGeneration();
        now.smp_gen = texture_cache.SamplerGeneration();
        now.buf_gen = buffer_cache.StructGeneration();
        const auto [it, inserted] = c.seen.try_emplace(h, now);
        g_skey_last_hit = false;
        if (!inserted) {
            const auto& old = it->second;
            g_skey_last_hit = old.frame + 1 >= frame;
            if (old.frame + 1 >= frame) {
                ++c.hit;
                c.hit_textured += textured;
                c.hit_same_frame += old.frame == frame;
                // Would the cached record still be valid? (a) no static range marked since it was
                // stored, (b) no image / texture / sampler / buffer structure change since.
                bool buf_ok = !big && old.buf_gen == now.buf_gen;
                for (const auto& [lo, hi] : c.ranges) {
                    buf_ok = buf_ok && VideoCore::GranulesUnchangedSince(lo, hi, old.clock);
                }
                const bool img_ok = old.img_epoch == now.img_epoch &&
                                    old.tex_gen == now.tex_gen && old.smp_gen == now.smp_gen;
                c.hit_buf_ok += buf_ok;
                c.hit_img_ok += img_ok;
                c.hit_both_ok += buf_ok && img_ok;
                c.hit_textured_ok += buf_ok && img_ok && textured;
            }
            if (old.frame != frame) {
                ++c.keys_this_frame;
            }
            it->second = now;
        } else {
            ++c.distinct_new;
            ++c.keys_this_frame;
        }
        if ((c.draws & 0x1fffff) == 0) {
            LOG_INFO(Render_Vulkan,
                     "StaticKeyCensus: full draws={} (textured {}) hit (seen this or last frame)={} "
                     "({:.1f}%, textured {:.1f}%) same_frame={} | distinct keys per frame={:.0f}, new "
                     "keys={} | per draw: heap bindings {:.2f} static {:.2f} | distinct heap buffer "
                     "ids this window={} | keys stored={}",
                     c.draws, c.textured, c.hit, 100.0 * double(c.hit) / double(c.draws),
                     100.0 * double(c.hit_textured) / double(std::max<u64>(c.textured, 1)),
                     c.hit_same_frame,
                     double(c.keys_frames_sum) / double(std::max<u64>(c.frames, 1)), c.distinct_new,
                     double(c.heap_bindings) / double(c.draws),
                     double(c.static_bindings) / double(c.draws), c.heap_ids.size(), c.seen.size());
            LOG_INFO(Render_Vulkan,
                     "StaticKeyValid: of {} hits: static buffers unchanged {:.1f}%, image / texture / "
                     "sampler / buffer structure unchanged {:.1f}%, both {:.1f}% (textured hits "
                     "valid {:.1f}% of textured full draws) | draws with a > 64 MB static binding {}",
                     c.hit, 100.0 * double(c.hit_buf_ok) / double(std::max<u64>(c.hit, 1)),
                     100.0 * double(c.hit_img_ok) / double(std::max<u64>(c.hit, 1)),
                     100.0 * double(c.hit_both_ok) / double(std::max<u64>(c.hit, 1)),
                     100.0 * double(c.hit_textured_ok) / double(std::max<u64>(c.textured, 1)),
                     c.big_bindings);
            LOG_INFO(Render_Vulkan,
                     "StaticKeyPrev: same static key as the previous full draw {} ({:.1f}% of full "
                     "draws; textured {:.1f}% of textured full draws), adjacent (no repeat / join "
                     "between) {}",
                     c.prev_match, 100.0 * double(c.prev_match) / double(c.draws),
                     100.0 * double(c.prev_match_textured) / double(std::max<u64>(c.textured, 1)),
                     c.prev_match_adjacent);
            c.prev_match = c.prev_match_textured = c.prev_match_adjacent = 0;
            c.hit_buf_ok = c.hit_img_ok = c.hit_both_ok = c.big_bindings = c.hit_textured_ok = 0;
            c.draws = c.textured = c.hit = c.hit_textured = c.hit_same_frame = 0;
            c.heap_bindings = c.static_bindings = c.frames = c.distinct_new = 0;
            c.keys_frames_sum = 0;
            c.heap_ids.clear();
            if (c.seen.size() > (1u << 20)) {
                c.seen.clear();
            }
        }
    }

    // Bind resource buffers and textures.
    Shader::Backend::Bindings binding{};
    push_data = MakeUserData(liverpool->BackRegs());
    u8 memo_draw_bits = 3;
    for (const auto& bound : BoundStages(pipeline)) {
        const auto* stage = bound.info;
        if (!stage) {
            continue;
        }
        if (build_set_writes) {
            set_writes.resize(set_writes.size() + stage->buffers.size() + stage->images.size() +
                              stage->samplers.size());
        }
        bound.PushUd(binding, push_data);
        if (bind_memo_stats) {
            // Runs the same BindBuffers/BindTextures, with the memo's key and checks around them.
            memo_draw_bits &= BindStageMemoStats(pipeline, bound, binding);
        } else {
            // Split measured because BindResources is 6.8 us of the 8.0 us spent per draw - 85% -
            // and a stack scan could not tell these two apart from anything else.
            const auto t_buf = Common::FaultMonitor::Now();
            if (VideoCore::BufferCache::ReadProducerStatsEnabled()) [[unlikely]] {
                buffer_cache.SetDiagProducer(stage->pgm_hash);
            }
            const u64 dc0 = g_draw_cost ? __rdtsc() : 0;
            BindBuffers(bound, binding, push_data);
            const u64 dc1 = g_draw_cost ? __rdtsc() : 0;
            const auto t_tex = Common::FaultMonitor::Now();
            const u64 rp_tseq0 = g_draw_repeat == 2 ? scheduler.RecordSeq() : 0;
            if (tex_bind_memo) {
                BindTexturesMemo(pipeline, bound, binding);
            } else {
                BindTextures(bound, binding);
            }
            if (g_draw_repeat == 2 && scheduler.RecordSeq() != rp_tseq0) [[unlikely]] {
                g_rp_seq_who += fmt::format(" tex(stage {})", static_cast<u32>(stage->l_stage));
            }
            if (g_draw_cost) {
                g_dc_bind_buf += dc1 - dc0;
                g_dc_bind_tex += __rdtsc() - dc1;
            }
            Common::FaultMonitor::CountBindSplit(Common::FaultMonitor::DiffNs(t_buf, t_tex),
                                                 Common::FaultMonitor::SinceNs(t_tex));
        }
        uses_dma |= stage->uses_dma;
    }
    native_bound_count = binding.unified;
    if (VideoCore::BufferCache::ReadProducerStatsEnabled()) [[unlikely]] {
        buffer_cache.SetDiagProducer(0); // Later written bindings (DMA, HLE, fills): untracked.
    }
    if (bind_memo_stats) {
        BindMemoEndDraw(pipeline->IsCompute(), memo_draw_bits);
    }

    if (uses_dma && VideoCore::g_lane_war && Common::FaultMonitor::t_is_cp_thread &&
        !Scheduler::t_lane_mode) [[unlikely]] {
        VideoCore::LaneProbeReadAllMain(scheduler.CurrentTick(), 20); // q9c (2a): BDA reads any address
    }
    if (uses_dma) {
        // We only use fault buffer for DMA right now.
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (auto& range : mapped_ranges) {
            buffer_cache.SynchronizeBuffersInRange(range.lower(), range.upper() - range.lower());
        }
        fault_process_pending = true;
    }
    if (AmdGpu::MtbCensus::g_enabled) [[unlikely]] {
        // Census v3: images this draw binds (textures and, for draws, render targets).
        for (const auto image_id : bound_images) {
            const auto& image = texture_cache.PeekImage(image_id);
            AmdGpu::MtbCensus::NoteUse(image.info.guest_address, image.info.guest_size);
        }
    }

    return true;
}

namespace {

// SHADPS4_BIND_MEMO_STATS state, CP thread only. One record per (pipeline, stage), which is where
// the memo itself would keep it. Generations are the values read before the full path that
// produced the outputs, so a bump during that path makes the next occurrence miss.
struct BindMemoRecord {
    struct Buf {
        VideoCore::BufferId id{};
        VAddr addr{};
        u64 size{};
        vk::Buffer handle{};
        vk::DeviceSize offset{};
        vk::DeviceSize range{};
        u8 adjust{};
        bool memo{}; // read-only guest binding: a hit would skip it
    };
    struct Img {
        VideoCore::ImageId id{};
        u64 uid{};
        const VideoCore::Image::BackingImage* backing{};
        vk::ImageView view{};
        vk::ImageLayout layout{};
        VideoCore::Image::State state{}; // after the full path's Transit
    };
    bool valid{};
    std::vector<AmdGpu::Buffer> vsharps;
    std::vector<AmdGpu::Image> tsharps;
    std::vector<AmdGpu::Sampler> ssharps;
    u64 border{};
    u64 buf_gen{};
    u64 buf_tex_gen{};
    bool buf_excluded{};
    bool buf_texel{};
    std::vector<Buf> bufs;
    u64 tex_gen{};
    u64 smp_gen{};
    bool tex_excluded{};
    std::vector<Img> imgs;
    std::vector<vk::Sampler> samplers;
};

enum BindMemoBufMiss : u32 { BufExcluded, BufGen, BufTexGen, BufDirtyPage, NumBufMiss };
enum BindMemoTexMiss : u32 {
    TexExcluded,
    TexGen,
    TexSamplerGen,
    TexUid,
    TexBacking,
    TexDirtyImage,
    TexFlags,
    NumTexMiss
};

// Hit work that is not replayed: GetBarrier per skipped buffer, the sampler LRU touch per
// sampler. Guesses, subtracted so the estimate is not flattered by leaving them out.
constexpr double BindMemoAllowBufNs = 5.0;
constexpr double BindMemoAllowSamplerNs = 5.0;

struct BindMemoStats {
    std::unordered_map<u64, BindMemoRecord> records;
    // Key under test and dry-replay scratch; capacity is kept so timed spans do not allocate.
    std::vector<AmdGpu::Buffer> vsharps;
    std::vector<AmdGpu::Image> tsharps;
    std::vector<AmdGpu::Sampler> ssharps;
    std::vector<vk::DescriptorBufferInfo> replay_buffers;
    std::vector<vk::DescriptorImageInfo> replay_images;
    std::array<u8, Shader::NUM_BUFFERS> replay_adjust{};
    u64 stage_counter{};
    bool started{};
    // Window origin.
    u64 frames0{}, buf_gen0{}, tex_gen0{}, smp_gen0{}, tsc0{};
    std::chrono::steady_clock::time_point clock0{};
    // Window counters. Stage types: 0 vs, 1 ps, 2 cs, 3 other.
    u64 binds{}, stages{}, draws{}, draw_key{}, draw_would{}, dma_stages{};
    std::array<u64, 4> type_stages{}, type_key{}, type_would{};
    u64 buf_key{}, buf_would{}, tex_key{}, tex_would{}, transit_needed{};
    std::array<u64, NumBufMiss> buf_miss{};
    std::array<u64, NumTexMiss> tex_miss{};
    u64 slot_buf{}, slot_buf_key{}, slot_img{}, slot_img_key{}, slot_smp{}, slot_smp_key{};
    u64 mis_buf{}, mis_upload{}, mis_img{}, mis_smp{}, layout_diff{};
    u64 memo_bufs_on_hit{}, samplers_on_hit{};
    // Timed stages (1 in 16), TSC ticks. t_empty is two back-to-back reads, subtracted from
    // every span average so the fenced reads themselves are not counted as work.
    u64 timed{}, t_empty{}, t_key_buf{}, t_key_tex{};
    u64 t_check_buf_n{}, t_check_buf{}, t_check_tex_n{}, t_check_tex{};
    u64 t_replay_buf_n{}, t_replay_buf{}, t_replay_tex_n{}, t_replay_tex{};
    u64 t_buf_hit_n{}, t_buf_hit{}, t_fixed_hit{}, t_buf_miss_n{}, t_buf_miss{};
    u64 t_tex_hit_n{}, t_tex_hit{}, t_tex_miss_n{}, t_tex_miss{};
};

BindMemoStats& GetBindMemoStats() {
    static BindMemoStats stats;
    return stats;
}

template <typename T>
bool SameBytes(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

template <typename T>
bool SameSlot(const std::vector<T>& a, const std::vector<T>& b, size_t i) {
    return i < a.size() && i < b.size() && std::memcmp(&a[i], &b[i], sizeof(T)) == 0;
}

} // Anonymous namespace

u8 Rasterizer::BindStageMemoStats(const Pipeline* pipeline, const Shader::BoundStage& bound,
                                  Shader::Backend::Bindings& binding) {
    const auto& stage = *bound.info;
    auto& st = GetBindMemoStats();
    if (!st.started) {
        st.started = true;
        st.frames0 = Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
        st.buf_gen0 = buffer_cache.StructGeneration();
        st.tex_gen0 = texture_cache.StructGeneration();
        st.smp_gen0 = texture_cache.SamplerGeneration();
        st.clock0 = std::chrono::steady_clock::now();
        st.tsc0 = Common::FencedRDTSC();
    }
    const bool timed = (++st.stage_counter & 15) == 0;
    const auto tsc = [timed] { return timed ? Common::FencedRDTSC() : u64{0}; };
    // The lookup stays outside the timed spans: the memo would own its record.
    const u64 record_key =
        (static_cast<u64>(reinterpret_cast<uintptr_t>(pipeline)) << 3) | u64(stage.l_stage);
    auto& rec = st.records[record_key];
    const u32 type = stage.l_stage == Shader::LogicalStage::Vertex     ? 0
                     : stage.l_stage == Shader::LogicalStage::Fragment ? 1
                     : stage.l_stage == Shader::LogicalStage::Compute  ? 2
                                                                       : 3;

    const u64 t_empty = tsc();
    // Key, buffer half: every guest V#, read as BindBuffers reads it.
    const u64 t_key = tsc();
    st.vsharps.clear();
    for (const auto& desc : stage.buffers) {
        st.vsharps.push_back(desc.IsSpecial() ? AmdGpu::Buffer{} : desc.GetSharp(bound));
    }
    const bool buf_key = rec.valid && SameBytes(st.vsharps, rec.vsharps);
    const u64 t_key_tex = tsc();
    // Key, texture half: every T#, every S# after BindTextures' aniso adjustment, border base.
    st.tsharps.clear();
    for (const auto& desc : stage.images) {
        st.tsharps.push_back(desc.GetSharp(bound));
    }
    st.ssharps.clear();
    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(bound);
        if (sampler.disable_aniso) {
            const auto& tsharp = stage.images[sampler.associated_image].GetSharp(bound);
            if (tsharp.base_level == 0 && tsharp.last_level == 0) {
                ssharp.max_aniso.Assign(AmdGpu::AnisoRatio::One);
            }
        }
        st.ssharps.push_back(ssharp);
    }
    const u64 border = liverpool->BackRegs().ta_bc_base.base_addr;
    const bool tex_key = rec.valid && border == rec.border && SameBytes(st.tsharps, rec.tsharps) &&
                         SameBytes(st.ssharps, rec.ssharps);

    // Validation, buffer half (design section 3): structure generation, texel aliasing, exact
    // CPU-dirty pages of every skipped range. Written and special bindings run in full anyway.
    const u64 t_check_buf = tsc();
    const u64 buf_gen = buffer_cache.StructGeneration();
    const u64 tex_gen = texture_cache.StructGeneration();
    const u64 smp_gen = texture_cache.SamplerGeneration();
    s32 buf_miss = -1;
    if (buf_key) {
        if (rec.buf_excluded || stage.uses_dma || native_bindings) {
            buf_miss = BufExcluded;
        } else if (buf_gen != rec.buf_gen) {
            buf_miss = BufGen;
        } else if (rec.buf_texel && tex_gen != rec.buf_tex_gen) {
            buf_miss = BufTexGen;
        } else {
            for (const auto& b : rec.bufs) {
                if (b.memo && b.id && buffer_cache.IsRegionCpuDirtyRelaxed(b.addr, b.size)) {
                    buf_miss = BufDirtyPage;
                    break;
                }
            }
        }
    }
    const bool buf_would = buf_key && buf_miss < 0;
    // Validation, texture half: image structure and sampler generations, then per image the
    // identity, backing, UpdateImageLockFree predicate and binding flags. A state difference is
    // not a miss: the memo would call Transit.
    const u64 t_check_tex = tsc();
    s32 tex_miss = -1;
    bool transit = false;
    if (tex_key) {
        if (rec.tex_excluded || native_bindings) {
            tex_miss = TexExcluded;
        } else if (tex_gen != rec.tex_gen) {
            tex_miss = TexGen;
        } else if (!rec.samplers.empty() && smp_gen != rec.smp_gen) {
            tex_miss = TexSamplerGen;
        } else {
            for (const auto& img : rec.imgs) {
                if (!img.id) {
                    continue;
                }
                auto& image = texture_cache.PeekImage(img.id);
                if (image.image_uid != img.uid) {
                    tex_miss = TexUid;
                } else if (image.backing != img.backing) {
                    tex_miss = TexBacking;
                } else if (!texture_cache.ImageUpToDateRelaxed(img.id)) {
                    tex_miss = TexDirtyImage;
                } else if (image.binding.is_target || image.binding.force_general ||
                           image.binding.needs_rebind) {
                    tex_miss = TexFlags;
                }
                if (tex_miss >= 0) {
                    break;
                }
                const auto& backing = *image.backing;
                transit |= !(backing.state == img.state) || !backing.subresource_runs.empty();
            }
        }
    }
    const bool tex_would = tex_key && tex_miss < 0;
    const u64 t_checked = tsc();

    // Dry replay of the stored outputs into scratch, timed only: what a hit writes instead.
    u64 replay_buf_ticks = 0;
    u64 replay_tex_ticks = 0;
    if (timed && buf_would) {
        const u64 t0 = tsc();
        st.replay_buffers.clear();
        for (size_t i = 0; i < rec.bufs.size(); ++i) {
            const auto& b = rec.bufs[i];
            st.replay_buffers.emplace_back(b.handle, b.offset, b.range);
            st.replay_adjust[i] = b.adjust;
        }
        replay_buf_ticks = tsc() - t0;
    }
    if (timed && tex_would) {
        const u64 t0 = tsc();
        st.replay_images.clear();
        for (const auto& img : rec.imgs) {
            if (img.id) {
                static_cast<void>(texture_cache.GetImage(img.id)); // the LRU touch a hit keeps
            }
            st.replay_images.emplace_back(vk::Sampler{}, img.view, img.layout);
        }
        for (const auto sampler : rec.samplers) {
            st.replay_images.emplace_back(sampler, vk::ImageView{}, vk::ImageLayout::eGeneral);
        }
        replay_tex_ticks = tsc() - t0;
    }

    // The unchanged full path.
    const size_t buf_begin = buffer_infos.size();
    const size_t img_begin = image_infos.size();
    const u32 buf_binding = binding.buffer;
    bind_memo_active = true;
    bind_memo_timing = timed;
    bind_memo_fixed_ticks = 0;
    bind_memo_redirected = 0;
    bind_memo_uploaded.fill(0);
    const u64 t_full = tsc();
    BindBuffers(bound, binding, push_data);
    const u64 t_full_tex = tsc();
    BindTextures(bound, binding);
    const u64 t_done = tsc();
    bind_memo_active = false;
    bind_memo_timing = false;

    if (timed) {
        ++st.timed;
        st.t_empty += t_key - t_empty;
        st.t_key_buf += t_key_tex - t_key;
        st.t_key_tex += t_check_buf - t_key_tex;
        if (buf_key) {
            ++st.t_check_buf_n;
            st.t_check_buf += t_check_tex - t_check_buf;
        }
        if (tex_key) {
            ++st.t_check_tex_n;
            st.t_check_tex += t_checked - t_check_tex;
        }
        if (buf_would) {
            ++st.t_replay_buf_n;
            st.t_replay_buf += replay_buf_ticks;
            ++st.t_buf_hit_n;
            st.t_buf_hit += t_full_tex - t_full;
            st.t_fixed_hit += bind_memo_fixed_ticks;
        } else {
            ++st.t_buf_miss_n;
            st.t_buf_miss += t_full_tex - t_full;
        }
        if (tex_would) {
            ++st.t_replay_tex_n;
            st.t_replay_tex += replay_tex_ticks;
            ++st.t_tex_hit_n;
            st.t_tex_hit += t_done - t_full_tex;
        } else {
            ++st.t_tex_miss_n;
            st.t_tex_miss += t_done - t_full_tex;
        }
    }

    ++st.stages;
    ++st.type_stages[type];
    st.type_key[type] += buf_key && tex_key;
    st.type_would[type] += buf_would && tex_would;
    st.dma_stages += stage.uses_dma;
    st.buf_key += buf_key;
    st.buf_would += buf_would;
    st.tex_key += tex_key;
    st.tex_would += tex_would;
    st.transit_needed += tex_would && transit;
    if (buf_miss >= 0) {
        ++st.buf_miss[static_cast<u32>(buf_miss)];
    }
    if (tex_miss >= 0) {
        ++st.tex_miss[static_cast<u32>(tex_miss)];
    }
    // Per-slot repeats against the previous occurrence (the per-binding fallback's key).
    for (size_t i = 0; i < stage.buffers.size(); ++i) {
        if (!stage.buffers[i].IsSpecial()) {
            ++st.slot_buf;
            st.slot_buf_key += rec.valid && SameSlot(st.vsharps, rec.vsharps, i);
        }
    }
    for (size_t i = 0; i < st.tsharps.size(); ++i) {
        ++st.slot_img;
        st.slot_img_key += rec.valid && SameSlot(st.tsharps, rec.tsharps, i);
    }
    for (size_t i = 0; i < st.ssharps.size(); ++i) {
        ++st.slot_smp;
        st.slot_smp_key +=
            rec.valid && border == rec.border && SameSlot(st.ssharps, rec.ssharps, i);
    }

    // Soundness: on a predicted hit the full path must have produced exactly the stored outputs
    // and uploaded nothing. Anything else means the invalidation list is incomplete.
    if (buf_would) {
        if (rec.bufs.size() != stage.buffers.size()) {
            ++st.mis_buf;
        } else {
            for (size_t i = 0; i < rec.bufs.size(); ++i) {
                const auto& b = rec.bufs[i];
                if (!b.memo) {
                    continue;
                }
                const auto& info = buffer_infos[buf_begin + i];
                const auto buffer_id = std::get<0>(buffer_bindings[i]);
                st.mis_buf += buffer_id != b.id || info.buffer != b.handle ||
                              info.offset != b.offset || info.range != b.range ||
                              push_data.buf_offsets[buf_binding + i] != b.adjust;
                st.mis_upload += bind_memo_uploaded[i] != 0;
                ++st.memo_bufs_on_hit;
            }
        }
    }
    if (tex_would) {
        const size_t num_images = image_bindings.size();
        if (num_images != rec.imgs.size() || stage.samplers.size() != rec.samplers.size()) {
            ++st.mis_img;
        } else {
            for (size_t i = 0; i < num_images; ++i) {
                const auto& info = image_infos[img_begin + i];
                const auto& img = rec.imgs[i];
                st.mis_img += image_bindings[i].first != img.id || info.imageView != img.view;
                st.layout_diff += info.imageLayout != img.layout;
            }
            for (size_t i = 0; i < rec.samplers.size(); ++i) {
                st.mis_smp += image_infos[img_begin + num_images + i].sampler != rec.samplers[i];
            }
            st.samplers_on_hit += rec.samplers.size();
        }
    }

    // Record this occurrence (outside every timed span).
    rec.valid = true;
    rec.vsharps = st.vsharps;
    rec.tsharps = st.tsharps;
    rec.ssharps = st.ssharps;
    rec.border = border;
    rec.buf_gen = buf_gen;
    rec.buf_tex_gen = tex_gen;
    rec.tex_gen = tex_gen;
    rec.smp_gen = smp_gen;
    rec.buf_excluded = false;
    rec.buf_texel = false;
    rec.bufs.resize(stage.buffers.size());
    for (size_t i = 0; i < stage.buffers.size(); ++i) {
        const auto& desc = stage.buffers[i];
        const auto& [buffer_id, vsharp, size] = buffer_bindings[i];
        const auto& info = buffer_infos[buf_begin + i];
        auto& b = rec.bufs[i];
        b.id = buffer_id;
        b.addr = vsharp.base_address;
        b.size = size;
        b.handle = info.buffer;
        b.offset = info.offset;
        b.range = info.range;
        b.adjust = push_data.buf_offsets[buf_binding + i];
        b.memo = !desc.IsSpecial() && !desc.is_written;
        if (b.memo && buffer_id) {
            // Stream copies and host mirrors are not the cache buffer; a texel range aliasing
            // an image is re-tiled on every bind (SynchronizeBufferFromImage).
            if (info.buffer != buffer_cache.GetBuffer(buffer_id).Handle()) {
                rec.buf_excluded = true;
            }
            if (desc.is_formatted) {
                rec.buf_texel = true;
                if (texture_cache.FindImageFromRange(vsharp.base_address, size)) {
                    rec.buf_excluded = true;
                }
            }
        }
    }
    rec.tex_excluded = native_bindings != nullptr;
    for (const auto& desc : stage.images) {
        rec.tex_excluded |= desc.mip_fallback_mode != Shader::MipStorageFallbackMode::None;
    }
    rec.imgs.resize(image_bindings.size());
    for (size_t i = 0; i < image_bindings.size(); ++i) {
        const auto& [image_id, desc] = image_bindings[i];
        const auto& info = image_infos[img_begin + i];
        auto& img = rec.imgs[i];
        img = {};
        img.id = image_id;
        img.view = info.imageView;
        img.layout = info.imageLayout;
        if (!image_id) {
            continue;
        }
        auto& image = texture_cache.PeekImage(image_id);
        img.uid = image.image_uid;
        img.backing = image.backing;
        img.state = image.backing->state;
        // Storage images, depth/stencil redirects and the Pro ID buffer stay on the full path.
        rec.tex_excluded |= desc.type == VideoCore::TextureCache::BindingType::Storage ||
                            ((bind_memo_redirected >> (i & 63)) & 1) != 0 ||
                            (!stencil_scratches.empty() && IsStencilScratch(image_id)) ||
                            (image.info.num_samples == 2 &&
                             image.info.pixel_format == vk::Format::eR32Uint);
    }
    rec.samplers.resize(stage.samplers.size());
    for (size_t i = 0; i < stage.samplers.size(); ++i) {
        rec.samplers[i] = image_infos[img_begin + image_bindings.size() + i].sampler;
    }
    return u8(buf_key && tex_key) | u8((buf_would && tex_would) << 1);
}

void Rasterizer::BindMemoEndDraw(bool is_compute, u8 draw_bits) {
    auto& st = GetBindMemoStats();
    ++st.binds;
    if (!is_compute) {
        ++st.draws;
        st.draw_key += (draw_bits & 1) != 0;
        st.draw_would += (draw_bits & 2) != 0;
    }
    if (st.binds < (u64{1} << 18)) {
        return;
    }
    const u64 tsc_now = Common::FencedRDTSC();
    const auto clock_now = std::chrono::steady_clock::now();
    const double wall_ns = double(
        std::chrono::duration_cast<std::chrono::nanoseconds>(clock_now - st.clock0).count());
    const double ns_per_tick = tsc_now > st.tsc0 ? wall_ns / double(tsc_now - st.tsc0) : 0.0;
    const u64 frames_now = Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
    const u64 frames = frames_now - st.frames0;
    const double per_frame = frames ? 1.0 / double(frames) : 0.0;
    const u64 buf_gen = buffer_cache.StructGeneration();
    const u64 tex_gen = texture_cache.StructGeneration();
    const u64 smp_gen = texture_cache.SamplerGeneration();
    const auto pct = [](u64 part, u64 whole) {
        return whole ? 100.0 * double(part) / double(whole) : 0.0;
    };
    const double empty_ns = st.timed ? double(st.t_empty) * ns_per_tick / double(st.timed) : 0.0;
    const auto avg_ns = [ns_per_tick, empty_ns](u64 ticks, u64 n) {
        return n ? std::max(0.0, double(ticks) * ns_per_tick / double(n) - empty_ns) : 0.0;
    };
    const double full_buf_hit = avg_ns(st.t_buf_hit, st.t_buf_hit_n);
    // Summed per always-full binding inside BindBuffers, so no single-span correction here.
    const double fixed_buf_hit =
        st.t_buf_hit_n ? double(st.t_fixed_hit) * ns_per_tick / double(st.t_buf_hit_n) : 0.0;
    const double full_buf_miss = avg_ns(st.t_buf_miss, st.t_buf_miss_n);
    const double full_tex_hit = avg_ns(st.t_tex_hit, st.t_tex_hit_n);
    const double full_tex_miss = avg_ns(st.t_tex_miss, st.t_tex_miss_n);
    const double key_buf = avg_ns(st.t_key_buf, st.timed);
    const double key_tex = avg_ns(st.t_key_tex, st.timed);
    const double check_buf = avg_ns(st.t_check_buf, st.t_check_buf_n);
    const double check_tex = avg_ns(st.t_check_tex, st.t_check_tex_n);
    const double replay_buf = avg_ns(st.t_replay_buf, st.t_replay_buf_n);
    const double replay_tex = avg_ns(st.t_replay_tex, st.t_replay_tex_n);
    // Per predicted hit: the half's full path minus what a hit still runs (always-full buffer
    // bindings, the checks, the replay, the unreplayed allowances).
    const double allow_buf =
        st.buf_would ? BindMemoAllowBufNs * double(st.memo_bufs_on_hit) / double(st.buf_would)
                     : 0.0;
    const double allow_tex =
        st.tex_would ? BindMemoAllowSamplerNs * double(st.samplers_on_hit) / double(st.tex_would)
                     : 0.0;
    const double save_buf_hit = full_buf_hit - fixed_buf_hit - check_buf - replay_buf - allow_buf;
    const double save_tex_hit = full_tex_hit - check_tex - replay_tex - allow_tex;
    // Paid on every stage (key), on every key hit that fails validation (check) and on every
    // miss (record update, costed like a replay).
    const double gross_buf = double(st.buf_would) * save_buf_hit;
    const double gross_tex = double(st.tex_would) * save_tex_hit;
    const double cost_buf = double(st.stages) * key_buf +
                            double(st.buf_key - st.buf_would) * check_buf +
                            double(st.stages - st.buf_would) * replay_buf;
    const double cost_tex = double(st.stages) * key_tex +
                            double(st.tex_key - st.tex_would) * check_tex +
                            double(st.stages - st.tex_would) * replay_tex;
    const double net_buf_us = (gross_buf - cost_buf) * per_frame / 1000.0;
    const double net_tex_us = (gross_tex - cost_tex) * per_frame / 1000.0;
    const u64 mismatches = st.mis_buf + st.mis_upload + st.mis_img + st.mis_smp;

    LOG_INFO(Render_Vulkan,
             "BindMemoStats: binds={} draws={} frames={} stages={} "
             "stage_key(vs/ps/cs/other)={:.1f}/{:.1f}/{:.1f}/{:.1f}% "
             "stage_would(vs/ps/cs/other)={:.1f}/{:.1f}/{:.1f}/{:.1f}% "
             "whole_draw(key/would)={:.1f}/{:.1f}% slot_key(buf/img/smp)={:.1f}/{:.1f}/{:.1f}% "
             "dma_stages={}",
             st.binds, st.draws, frames, st.stages, pct(st.type_key[0], st.type_stages[0]),
             pct(st.type_key[1], st.type_stages[1]), pct(st.type_key[2], st.type_stages[2]),
             pct(st.type_key[3], st.type_stages[3]), pct(st.type_would[0], st.type_stages[0]),
             pct(st.type_would[1], st.type_stages[1]), pct(st.type_would[2], st.type_stages[2]),
             pct(st.type_would[3], st.type_stages[3]), pct(st.draw_key, st.draws),
             pct(st.draw_would, st.draws), pct(st.slot_buf_key, st.slot_buf),
             pct(st.slot_img_key, st.slot_img), pct(st.slot_smp_key, st.slot_smp),
             st.dma_stages);
    LOG_INFO(Render_Vulkan,
             "BindMemoStats: buf key={:.1f}% would={:.1f}% "
             "miss(excluded/buf_gen/tex_gen/dirty_page)={}/{}/{}/{} tex key={:.1f}% "
             "would={:.1f}% transit_needed={} "
             "miss(excluded/tex_gen/smp_gen/uid/backing/dirty_image/flags)={}/{}/{}/{}/{}/{}/{} "
             "mismatches={} (buf={} upload={} img={} smp={}) layout_diff={} "
             "gens/frame(buf/tex/smp)={:.2f}/{:.3f}/{:.3f}",
             pct(st.buf_key, st.stages), pct(st.buf_would, st.stages), st.buf_miss[BufExcluded],
             st.buf_miss[BufGen], st.buf_miss[BufTexGen], st.buf_miss[BufDirtyPage],
             pct(st.tex_key, st.stages), pct(st.tex_would, st.stages), st.transit_needed,
             st.tex_miss[TexExcluded], st.tex_miss[TexGen], st.tex_miss[TexSamplerGen],
             st.tex_miss[TexUid], st.tex_miss[TexBacking], st.tex_miss[TexDirtyImage],
             st.tex_miss[TexFlags], mismatches, st.mis_buf, st.mis_upload, st.mis_img,
             st.mis_smp, st.layout_diff, double(buf_gen - st.buf_gen0) * per_frame,
             double(tex_gen - st.tex_gen0) * per_frame, double(smp_gen - st.smp_gen0) * per_frame);
    LOG_INFO(Render_Vulkan,
             "BindMemoCost: timed={} tsc_read_ns={:.1f} ns full_buf(hit/miss)={:.0f}/{:.0f} "
             "fixed_buf_hit={:.0f} "
             "full_tex(hit/miss)={:.0f}/{:.0f} key(buf/tex)={:.1f}/{:.1f} "
             "check(buf/tex)={:.1f}/{:.1f} replay(buf/tex)={:.1f}/{:.1f} "
             "save_per_hit(buf/tex)={:.0f}/{:.0f} gross_us_frame(buf/tex)={:.0f}/{:.0f} "
             "est_save_us_frame(buf/tex/total)={:.0f}/{:.0f}/{:.0f}",
             st.timed, empty_ns, full_buf_hit, full_buf_miss, fixed_buf_hit, full_tex_hit,
             full_tex_miss,
             key_buf, key_tex, check_buf, check_tex, replay_buf, replay_tex, save_buf_hit,
             save_tex_hit, gross_buf * per_frame / 1000.0, gross_tex * per_frame / 1000.0,
             net_buf_us, net_tex_us, net_buf_us + net_tex_us);

    // Cumulative-and-reset: the next window starts here. Records are kept.
    auto records = std::move(st.records);
    const u64 stage_counter = st.stage_counter;
    st = {};
    st.records = std::move(records);
    st.stage_counter = stage_counter;
    st.started = true;
    st.frames0 = frames_now;
    st.buf_gen0 = buf_gen;
    st.tex_gen0 = tex_gen;
    st.smp_gen0 = smp_gen;
    st.clock0 = clock_now;
    st.tsc0 = tsc_now;
}

// SHADPS4_TEX_BIND_MEMO: the texture half of the binding memo (notes/binding-memo-design-
// 2026-09-25.md sections 3-4). Key and validity checks are the stats mode's above (measured
// 2026-09-25: texture key hit 94.5%, 0 mismatches), plus a live depth-association check and a
// perfect-match requirement at record time. One record per (pipeline, logical stage), owned by
// the Pipeline. Binding thread only.
struct TexMemoRecord {
    struct Img {
        VideoCore::ImageId id{};
        u64 uid{};
        const VideoCore::Image::BackingImage* backing{};
        vk::ImageView view{};
        vk::DescriptorType type{};
        // Backing state after the recorded full path. With a full-range view and no subresource
        // runs this state is a fixed point of the stage's Transit calls (same layout, access and
        // stage every time), so a hit skips Transit only when the live state equals it.
        VideoCore::Image::State state{};
        VideoCore::SubresourceRange range{}; // the range the full path passed to Transit
        bool full_range{};
        /// SHADPS4_TEX_TRANSIT_SKIP: BackingImage::state_serial at which this image's Transit call
        /// on a hit was a pure no-op (0 = not known). Written on hits, hence mutable.
        mutable u64 ok_serial{};
        u64 overlap_epoch{}; ///< SHADPS4_TEX_MEMO_LOCAL_GEN: Image::overlap_epoch when recorded
    };
    bool valid{};
    bool excluded{};
    /// SHADPS4_TEX_MEMO_LOCAL_GEN: every T# resolved to an image and the structure generation did
    /// not move during the recording full path, so per-image overlap epochs stand in for tex_gen.
    bool local_ok{};
    u64 border{};
    u64 tex_gen{};
    u64 smp_gen{};
    u64 pb_written{}; ///< SHADPS4_PB_CENSUS: census draw index of the last record
    std::vector<AmdGpu::Image> tsharps;
    std::vector<AmdGpu::Sampler> ssharps;
    std::vector<Img> imgs;
    std::vector<vk::Sampler> samplers;
    std::vector<size_t> sampler_lru;
    u64 last_use{}; ///< SHADPS4_TEX_MEMO_WAYS: LRU stamp of this way.
};

/// SHADPS4_TEX_MEMO_WAYS=N (1-8, default 1): each (pipeline, stage) keeps N records, so a shader
/// that alternates between materials hits on each of them instead of missing on the key.
struct TexMemoSet {
    static constexpr u32 MaxWays = 32;
    /// SHADPS4_TEX_MEMO_HASH: key hash (0 = never recorded) and LRU stamp of every way, kept flat
    /// so the search does not touch the records.
    std::array<u64, MaxWays> key_hash{};
    std::array<u64, MaxWays> hash_last_use{};
    std::array<TexMemoRecord, MaxWays> ways{};
};

/// SHADPS4_TEX_MEMO_HASH=1 (default off): hash-first search of the ways (see TexMemoSet).
static bool TexMemoHash() {
    static const bool on = [] {
        const char* value = std::getenv("SHADPS4_TEX_MEMO_HASH");
        return value && value[0] == '1';
    }();
    return on;
}

/// 64-bit mix of a byte string (never returns 0). Only narrows the search; SameBytes decides.
static u64 TexMemoHashBytes(u64 h, const void* data, size_t size) {
    const u8* p = static_cast<const u8*>(data);
    while (size >= 8) {
        u64 v;
        std::memcpy(&v, p, 8);
        h = (h ^ v) * 0x9E3779B97F4A7C15ULL;
        h ^= h >> 29;
        p += 8;
        size -= 8;
    }
    if (size != 0) {
        u64 v = 0;
        std::memcpy(&v, p, size);
        h = (h ^ v) * 0xC2B2AE3D27D4EB4FULL;
        h ^= h >> 31;
    }
    return h;
}

static u32 TexMemoWays() {
    static const u32 ways = [] {
        const char* value = std::getenv("SHADPS4_TEX_MEMO_WAYS");
        const u32 v = value ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 1u;
        return std::clamp<u32>(v, 1u, TexMemoSet::MaxWays);
    }();
    return ways;
}

namespace {

enum TexMemoMiss : u32 {
    TmKey,
    TmExcluded,
    TmTexGen,
    TmSmpGen,
    TmUid,
    TmBacking,
    TmDirtyImage,
    TmFlags,
    TmAssoc,
    TmRefused,
    NumTexMemoMiss
};

struct TexMemoState {
    // Key under test; capacity is kept.
    std::vector<AmdGpu::Image> tsharps;
    std::vector<AmdGpu::Sampler> ssharps;
    // Verify scratch: a hit's outputs, captured before they are rolled back.
    struct Write {
        u32 binding;
        u32 count;
        vk::DescriptorType type;
        const vk::DescriptorImageInfo* info;
    };
    std::vector<vk::DescriptorImageInfo> hit_infos;
    std::vector<Write> hit_writes;
    std::vector<VideoCore::ImageId> hit_bound;
    std::vector<std::pair<VideoCore::Image::State, size_t>> hit_states;
    // Window counters (reset at every log line) and run totals.
    bool started{};
    u64 frames0{};
    u64 stages{}, empty_stages{}, key_hits{}, hits{}, hit_transits{}, hit_images{}, hit_samplers{};
    std::array<u64, NumTexMemoMiss> miss{};
    u64 verified{}, mismatched{};
    u64 local_saved{}, local_overlap_miss{}; ///< SHADPS4_TEX_MEMO_LOCAL_GEN (window)
    u64 mis_shape{}, mis_id{}, mis_view{}, mis_layout{}, mis_sampler{}, mis_write{}, mis_bound{},
        mis_state{};
    u64 total_stages{}, total_hits{}, total_verified{}, total_mismatched{};
    u32 logged{};
};

TexMemoState& GetTexMemoState() {
    static TexMemoState state;
    return state;
}

void LogTexMemo(TexMemoState& st, u8 mode) {
    const u64 frames_now = Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
    const u64 frames = frames_now - st.frames0;
    const auto pct = [](u64 part, u64 whole) {
        return whole ? 100.0 * double(part) / double(whole) : 0.0;
    };
    const auto per = [](u64 part, u64 whole) { return whole ? double(part) / double(whole) : 0.0; };
    const u64 tex_stages = st.stages - st.empty_stages;
    LOG_INFO(Render_Vulkan,
             "TexBindMemo: mode={} frames={} stages={} tex_stages={} key={:.1f}% hit={:.1f}% "
             "hit_all={:.1f}% hits/frame={:.0f} transits/hit={:.3f} images/hit={:.2f} "
             "samplers/hit={:.2f} miss(key/excluded/tex_gen/smp_gen/uid/backing/dirty_image/"
             "flags/assoc/refused)={}/{}/{}/{}/{}/{}/{}/{}/{}/{} verified={} mismatches={} "
             "(shape={} id={} view={} layout={} sampler={} write={} bound={} state={}) "
             "total(stages/hits/verified/mismatches)={}/{}/{}/{}",
             mode == 2 ? "verify" : "memo", frames, st.stages, tex_stages,
             pct(st.key_hits, tex_stages), pct(st.hits, tex_stages),
             pct(st.hits + st.empty_stages, st.stages), per(st.hits, frames),
             per(st.hit_transits, st.hits), per(st.hit_images, st.hits),
             per(st.hit_samplers, st.hits), st.miss[TmKey], st.miss[TmExcluded],
             st.miss[TmTexGen], st.miss[TmSmpGen], st.miss[TmUid], st.miss[TmBacking],
             st.miss[TmDirtyImage], st.miss[TmFlags], st.miss[TmAssoc], st.miss[TmRefused],
             st.verified, st.mismatched, st.mis_shape, st.mis_id, st.mis_view, st.mis_layout,
             st.mis_sampler, st.mis_write, st.mis_bound, st.mis_state, st.total_stages,
             st.total_hits, st.total_verified, st.total_mismatched);
    if (g_tex_memo_local) {
        LOG_INFO(Render_Vulkan,
                 "TexMemoLocal: hits the global generation test would have missed={} "
                 "overlap-epoch misses={}",
                 st.local_saved, st.local_overlap_miss);
        st.local_saved = st.local_overlap_miss = 0;
    }
    st.frames0 = frames_now;
    st.stages = st.empty_stages = st.key_hits = st.hits = st.hit_transits = st.hit_images =
        st.hit_samplers = 0;
    st.miss = {};
    st.verified = st.mismatched = 0;
    st.mis_shape = st.mis_id = st.mis_view = st.mis_layout = st.mis_sampler = st.mis_write =
        st.mis_bound = st.mis_state = 0;
}

} // Anonymous namespace

void Rasterizer::BindTexturesMemo(const Pipeline* pipeline, const Shader::BoundStage& bound,
                                  Shader::Backend::Bindings& binding) {
    const auto& stage = *bound.info;
    // Program 5 gate (DRAW_COST): key / validation / hit-emit / miss split, accounted on exit.
    struct P5Timer {
        u64 t0 = 0, t1 = 0, t2 = 0;
        bool hit = false;
        ~P5Timer() {
            if (t0 == 0) {
                return;
            }
            const u64 end = __rdtsc();
            if (t1 == 0) {
                g_p5[P5TexEmpty] += end - t0;
                return;
            }
            g_p5[P5TexKey] += t1 - t0;
            g_p5[P5TexValid] += t2 - t1;
            g_p5[hit ? P5TexEmit : P5TexMiss] += end - t2;
        }
    } p5;
    p5.t0 = g_draw_cost ? __rdtsc() : 0;
    auto& st = GetTexMemoState();
    if (!st.started) {
        st.started = true;
        st.frames0 = Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
    }
    if (st.stages >= (u64{1} << 18)) {
        LogTexMemo(st, tex_bind_memo);
    }
    ++st.stages;
    ++st.total_stages;
    if (stage.images.empty() && stage.samplers.empty()) {
        // Nothing to skip; BindTextures only walks empty lists.
        ++st.empty_stages;
        BindTextures(bound, binding);
        return;
    }
    auto& slot = pipeline->tex_bind_memo[static_cast<u32>(stage.l_stage)];
    if (!slot) {
        slot = std::make_shared<TexMemoSet>();
    }
    auto& set = *static_cast<TexMemoSet*>(slot.get());
    const u32 ways = TexMemoWays();

    // Key: every T#, every S# after BindTextures' aniso adjustment, the border colour base.
    st.tsharps.clear();
    for (const auto& desc : stage.images) {
        st.tsharps.push_back(desc.GetSharp(bound));
    }
    st.ssharps.clear();
    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(bound);
        if (sampler.disable_aniso) {
            const auto& tsharp = stage.images[sampler.associated_image].GetSharp(bound);
            if (tsharp.base_level == 0 && tsharp.last_level == 0) {
                ssharp.max_aniso.Assign(AmdGpu::AnisoRatio::One);
            }
        }
        st.ssharps.push_back(ssharp);
    }
    const u64 border = liverpool->BackRegs().ta_bc_base.base_addr;
    if (g_bind_region_census && !pipeline->IsCompute()) [[unlikely]] {
        auto& c = GetRegionCensus();
        for (const auto& t : st.tsharps) {
            ++c.by[static_cast<u32>(t.Address() >> 24)].img;
            ++c.images;
        }
    }

    // The way whose key matches, else the least recently used (or an empty) way, which then
    // misses on the key below and is re-recorded by the full path.
    TexMemoRecord* found = nullptr;
    TexMemoRecord* victim = &set.ways[0];
    const bool hash_search = TexMemoHash();
    if (hash_search) {
        u64 h = TexMemoHashBytes(border | 1, st.tsharps.data(),
                                 st.tsharps.size() * sizeof(AmdGpu::Image));
        h = TexMemoHashBytes(h, st.ssharps.data(), st.ssharps.size() * sizeof(AmdGpu::Sampler));
        h |= 1; // 0 marks a way that was never recorded
        u32 victim_way = 0;
        u64 oldest = ~0ULL;
        u32 found_way = ways;
        for (u32 w = 0; w < ways; ++w) {
            if (set.key_hash[w] == h) {
                auto& cand = set.ways[w];
                if (cand.valid && border == cand.border && SameBytes(st.tsharps, cand.tsharps) &&
                    SameBytes(st.ssharps, cand.ssharps)) {
                    found = &cand;
                    found_way = w;
                    break;
                }
            }
            // Never-used ways have stamp 0 and are taken first, then the least recently used.
            if (set.hash_last_use[w] < oldest) {
                oldest = set.hash_last_use[w];
                victim_way = w;
            }
        }
        if (!found) {
            // The miss path below re-records this way with the key under test.
            victim = &set.ways[victim_way];
            set.key_hash[victim_way] = h;
            found_way = victim_way;
        }
        set.hash_last_use[found_way] = st.total_stages;
    } else {
        for (u32 w = 0; w < ways; ++w) {
            auto& cand = set.ways[w];
            if (cand.valid && border == cand.border && SameBytes(st.tsharps, cand.tsharps) &&
                SameBytes(st.ssharps, cand.ssharps)) {
                found = &cand;
                break;
            }
            if (!cand.valid ? victim->valid
                            : (victim->valid && cand.last_use < victim->last_use)) {
                victim = &cand;
            }
        }
    }
    auto& rec = found ? *found : *victim;
    rec.last_use = st.total_stages;
    if (p5.t0 != 0) {
        p5.t1 = __rdtsc();
    }

    // Validation (design section 3). Image checks and the LRU touches a hit keeps run inside
    // one memo section, so a foreign structural change cannot interleave with them; the
    // structure generation is re-read inside it for the same reason.
    s32 miss = -1;
    // Hash search: a found way already passed the full key compare; a victim never matches.
    if (hash_search ? !found
                    : (!rec.valid || border != rec.border ||
                       !SameBytes(st.tsharps, rec.tsharps) ||
                       !SameBytes(st.ssharps, rec.ssharps))) {
        miss = TmKey;
    } else if (rec.excluded) {
        miss = TmExcluded;
    } else {
        const auto section = texture_cache.EnterMemoSection();
        if (section == VideoCore::TextureCache::MemoSection::Refused) {
            miss = TmRefused;
        } else {
            const bool local = g_tex_memo_local && rec.local_ok;
            const bool gen_moved = texture_cache.StructGeneration() != rec.tex_gen;
            if (!local && gen_moved) {
                miss = TmTexGen;
            } else {
                for (const auto& img : rec.imgs) {
                    if (!img.id) {
                        continue;
                    }
                    if (local && !texture_cache.ImageAllocated(img.id)) {
                        // Without the generation test an erased slot is reachable here.
                        miss = TmUid;
                        break;
                    }
                    const auto& image = texture_cache.PeekImage(img.id);
                    if (image.image_uid != img.uid) {
                        miss = TmUid;
                    } else if (image.backing != img.backing) {
                        miss = TmBacking;
                    } else if (local &&
                               std::atomic_ref<u64>{const_cast<u64&>(image.overlap_epoch)}.load(
                                   std::memory_order_relaxed) != img.overlap_epoch) {
                        miss = TmTexGen;
                        ++st.local_overlap_miss;
                    } else if (!texture_cache.ImageUpToDateRelaxed(img.id)) {
                        miss = TmDirtyImage;
                    } else if (image.binding.is_target || image.binding.force_general ||
                               image.binding.needs_rebind) {
                        miss = TmFlags;
                    } else if (image.depth_id) {
                        // AssociateDepth bumps no generation; the full path would redirect.
                        miss = TmAssoc;
                    }
                    if (miss >= 0) {
                        break;
                    }
                }
            }
            if (miss < 0) {
                // FindImage and UpdateImage fast paths: access tick and LRU touch.
                for (const auto& img : rec.imgs) {
                    if (img.id) {
                        texture_cache.TouchImageAccess(texture_cache.PeekImage(img.id));
                    }
                }
            }
            if (miss < 0 && local && gen_moved) {
                ++st.local_saved;
            }
            texture_cache.LeaveMemoSection(section);
        }
        // GetSampler's LRU touch, so the sampler GC cannot free a sampler this draw uses.
        if (miss < 0 && !texture_cache.TouchSamplers(rec.sampler_lru, rec.smp_gen)) {
            miss = TmSmpGen;
        }
    }
    st.key_hits += miss != static_cast<s32>(TmKey);
    if (miss >= 0) {
        ++st.miss[static_cast<u32>(miss)];
    }
    if (p5.t0 != 0) {
        p5.t2 = __rdtsc();
        p5.hit = miss < 0 && tex_bind_memo == 1;
    }

    if (miss < 0 && tex_bind_memo == 1) {
        if (VideoCore::g_pb_census != 0) [[unlikely]] {
            VideoCore::PbFoldMemo(rec.pb_written);
        }
        st.hit_transits += EmitTexMemoHit(rec, binding);
        ++st.hits;
        ++st.total_hits;
        st.hit_images += rec.imgs.size();
        st.hit_samplers += rec.samplers.size();
        return;
    }
    // SHADPS4_DRAW_REPEAT: this stage takes the full texture path (or the verify path).
    ++draw_repeat.tex_nonhit_stages;

    const size_t img_begin = image_infos.size();
    const u32 sw_begin = set_write_index;
    if (miss < 0) {
        // Verify: run the hit for real, capture its outputs, roll the outputs back and run the
        // full path over the same slots. Binding flags, LRU touches and any Transit the hit made
        // stay; the full path repeats them idempotently when the hit was exact, and redoes
        // whatever the hit skipped wrongly (then caught below as a state mismatch).
        const u32 unified_begin = binding.unified;
        const size_t bound_begin = bound_images.size();
        st.hit_transits += EmitTexMemoHit(rec, binding);
        ++st.hits;
        ++st.total_hits;
        st.hit_images += rec.imgs.size();
        st.hit_samplers += rec.samplers.size();
        st.hit_infos.assign(image_infos.begin() + img_begin, image_infos.end());
        st.hit_writes.clear();
        for (u32 i = sw_begin; i < set_write_index; ++i) {
            const auto& w = set_writes[i];
            st.hit_writes.push_back({w.dstBinding, w.descriptorCount, w.descriptorType,
                                     w.pImageInfo});
        }
        st.hit_bound.assign(bound_images.begin() + bound_begin, bound_images.end());
        const u32 hit_unified = binding.unified;
        st.hit_states.clear();
        for (const auto& img : rec.imgs) {
            if (img.id) {
                const auto& backing = *texture_cache.PeekImage(img.id).backing;
                st.hit_states.emplace_back(backing.state, backing.subresource_runs.size());
            }
        }
        image_infos.erase(image_infos.begin() + img_begin, image_infos.end());
        set_write_index = sw_begin;
        binding.unified = unified_begin;
        bound_images.erase(bound_images.begin() + bound_begin, bound_images.end());

        const u64 tex_gen = texture_cache.StructGeneration();
        const u64 smp_gen = texture_cache.SamplerGeneration();
        bind_memo_redirected = 0;
        bind_memo_rebound = false;
        BindTextures(bound, binding);

        u64 shape = 0, id = 0, view = 0, layout = 0, sampler = 0, write = 0, bound_diff = 0,
            state = 0;
        const size_t num_infos = image_infos.size() - img_begin;
        if (num_infos != st.hit_infos.size() ||
            set_write_index - sw_begin != st.hit_writes.size() || binding.unified != hit_unified ||
            image_bindings.size() != rec.imgs.size()) {
            shape = 1;
        } else {
            for (size_t i = 0; i < num_infos; ++i) {
                const auto& full = image_infos[img_begin + i];
                const auto& hit = st.hit_infos[i];
                view += full.imageView != hit.imageView;
                layout += full.imageLayout != hit.imageLayout;
                sampler += full.sampler != hit.sampler;
            }
            for (size_t i = 0; i < st.hit_writes.size(); ++i) {
                const auto& full = set_writes[sw_begin + i];
                const auto& hit = st.hit_writes[i];
                write += full.dstBinding != hit.binding || full.descriptorCount != hit.count ||
                         full.descriptorType != hit.type || full.pImageInfo != hit.info;
            }
            for (size_t i = 0; i < rec.imgs.size(); ++i) {
                id += image_bindings[i].first != rec.imgs[i].id;
            }
        }
        bound_diff = bound_images.size() - bound_begin != st.hit_bound.size() ||
                     !std::equal(st.hit_bound.begin(), st.hit_bound.end(),
                                 bound_images.begin() + bound_begin);
        size_t k = 0;
        for (const auto& img : rec.imgs) {
            if (!img.id) {
                continue;
            }
            const auto& backing = *texture_cache.PeekImage(img.id).backing;
            state += !(backing.state == st.hit_states[k].first) ||
                     backing.subresource_runs.size() != st.hit_states[k].second;
            ++k;
        }
        ++st.verified;
        ++st.total_verified;
        if (shape | id | view | layout | sampler | write | bound_diff | state) {
            ++st.mismatched;
            ++st.total_mismatched;
            st.mis_shape += shape != 0;
            st.mis_id += id != 0;
            st.mis_view += view != 0;
            st.mis_layout += layout != 0;
            st.mis_sampler += sampler != 0;
            st.mis_write += write != 0;
            st.mis_bound += bound_diff != 0;
            st.mis_state += state != 0;
            if (st.logged < 32) {
                ++st.logged;
                LOG_ERROR(Render_Vulkan,
                          "TexBindMemo verify MISMATCH #{}: pgm={:#x} stage={} images={} "
                          "samplers={} shape={} id={} view={} layout={} sampler={} write={} "
                          "bound={} state={}",
                          st.total_mismatched, stage.pgm_hash, static_cast<u32>(stage.l_stage),
                          rec.imgs.size(), rec.samplers.size(), shape, id, view, layout, sampler,
                          write, bound_diff, state);
            }
            // Re-record from the full path so one hole is not reported on every draw.
            RecordTexMemo(rec, stage, tex_gen, smp_gen, border, img_begin, sw_begin);
        }
        return;
    }

    // Miss: the unchanged full path, then record it. Generations are read first, so a bump
    // during BindTextures makes the next occurrence miss.
    const u64 tex_gen = texture_cache.StructGeneration();
    const u64 smp_gen = texture_cache.SamplerGeneration();
    bind_memo_redirected = 0;
    bind_memo_rebound = false;
    BindTextures(bound, binding);
    RecordTexMemo(rec, stage, tex_gen, smp_gen, border, img_begin, sw_begin);
}

u32 Rasterizer::EmitTexMemoHit(const TexMemoRecord& rec, Shader::Backend::Bindings& binding) {
    // BindTextures pass 1: binding flags. No memoized image is written, so force_general stays.
    for (const auto& img : rec.imgs) {
        if (img.id) {
            texture_cache.PeekImage(img.id).binding.is_bound = 1u;
        }
    }
    // Pass 2: bound_images, state, usage and image infos, in binding order.
    const u32 first_image_idx = image_infos.size();
    u32 transits = 0;
    for (const auto& img : rec.imgs) {
        if (!img.id) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
            continue;
        }
        bound_images.emplace_back(img.id);
        auto& image = texture_cache.PeekImage(img.id);
        const auto& backing = *image.backing;
        if (!img.full_range || !backing.subresource_runs.empty() || !(backing.state == img.state)) {
            // SHADPS4_TEX_TRANSIT_SKIP (0 off / 1 skip / 2 verify): the same call on the same state
            // was a pure no-op before (the serial did not move), so it is one again.
            static const int transit_skip = [] {
                const char* value = std::getenv("SHADPS4_TEX_TRANSIT_SKIP");
                return value ? std::atoi(value) : 0;
            }();
            static u64 skip_calls = 0, skip_skipped = 0, skip_mismatches = 0;
            const bool stamped = transit_skip != 0 && img.ok_serial == backing.state_serial &&
                                 img.backing == image.backing;
            if (stamped && transit_skip == 1) {
                ++skip_skipped;
            } else {
                const u64 serial_before = backing.state_serial;
                // Exactly the full path's call for a sampled image with no feedback/general flags.
                const auto new_layout = image.info.props.is_depth
                                            ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                            : vk::ImageLayout::eShaderReadOnlyOptimal;
                image.Transit(new_layout, vk::AccessFlagBits2::eShaderRead, img.range);
                ++transits;
                if (transit_skip != 0) {
                    if (image.backing->state_serial == serial_before) {
                        img.ok_serial = serial_before;
                        skip_skipped += stamped; // verify mode: would have been skipped
                    } else if (stamped) {
                        if (skip_mismatches++ < 16) {
                            LOG_ERROR(Render_Vulkan,
                                      "TexTransitSkip: MISMATCH image {:#x} serial {} -> {} "
                                      "full_range={} runs={}",
                                      image.info.guest_address, serial_before,
                                      image.backing->state_serial, img.full_range,
                                      image.backing->subresource_runs.size());
                        }
                    }
                }
            }
            if (transit_skip != 0 && (++skip_calls & 0x3fffff) == 0) {
                LOG_INFO(Render_Vulkan,
                         "TexTransitSkip: mode={} hit_transit_sites={} skipped={} ({:.1f}%) "
                         "mismatches={}",
                         transit_skip, skip_calls, skip_skipped,
                         100.0 * static_cast<double>(skip_skipped) / static_cast<double>(skip_calls),
                         skip_mismatches);
                skip_calls = 0;
                skip_skipped = 0;
            }
        }
        image.usage.texture |= 1u;
        image_infos.emplace_back(VK_NULL_HANDLE, img.view, image.backing->state.layout);
    }
    // Descriptor writes: one per image (array size 1; mip-fallback arrays are never memoized),
    // then one per sampler, advancing the unified binding exactly as BindTextures does.
    u32 image_info_idx = first_image_idx;
    for (const auto& img : rec.imgs) {
        if (build_set_writes) {
            auto& set_write = set_writes[set_write_index++];
            set_write.dstSet = VK_NULL_HANDLE;
            set_write.dstBinding = binding.unified;
            set_write.dstArrayElement = 0;
            set_write.descriptorCount = 1;
            set_write.descriptorType = img.type;
            set_write.pImageInfo = &image_infos[image_info_idx];
        }
        ++image_info_idx;
        ++binding.unified;
    }
    for (const auto sampler : rec.samplers) {
        image_infos.emplace_back(sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        const u32 unified_binding = binding.unified++;
        if (build_set_writes) {
            auto& set_write = set_writes[set_write_index++];
            set_write.dstSet = VK_NULL_HANDLE;
            set_write.dstBinding = unified_binding;
            set_write.dstArrayElement = 0;
            set_write.descriptorCount = 1;
            set_write.descriptorType = vk::DescriptorType::eSampler;
            set_write.pImageInfo = &image_infos.back();
        }
    }
    return transits;
}

void Rasterizer::RecordTexMemo(TexMemoRecord& rec, const Shader::Info& stage, u64 tex_gen,
                               u64 smp_gen, u64 border, size_t img_begin, u32 sw_begin) {
    const auto& st = GetTexMemoState();
    rec.valid = true;
    rec.pb_written = VideoCore::PbDraw();
    rec.tsharps = st.tsharps;
    rec.ssharps = st.ssharps;
    rec.border = border;
    rec.tex_gen = tex_gen;
    rec.smp_gen = smp_gen;
    rec.local_ok = true; // decided inside a memo section after the image loop
    const size_t num_images = image_bindings.size();
    const size_t num_samplers = stage.samplers.size();
    // Stage-wide exclusions: native bindings, stencil redirects and scratch (Pro paths), images
    // re-found through needs_rebind, mip-fallback arrays, storage images, and any shape that is
    // not one descriptor per image and per sampler.
    rec.excluded = !build_set_writes || native_bindings != nullptr || bind_memo_redirected != 0 ||
                   bind_memo_rebound || num_images != stage.images.size() ||
                   image_infos.size() - img_begin != num_images + num_samplers ||
                   set_write_index - sw_begin != num_images + num_samplers;
    for (const auto& desc : stage.images) {
        rec.excluded |=
            desc.is_written || desc.mip_fallback_mode != Shader::MipStorageFallbackMode::None;
    }
    if (rec.excluded) {
        rec.imgs.clear();
        rec.samplers.clear();
        rec.sampler_lru.clear();
        return;
    }
    rec.imgs.resize(num_images);
    for (size_t i = 0; i < num_images; ++i) {
        const auto& [image_id, desc] = image_bindings[i];
        const auto& info = image_infos[img_begin + i];
        auto& img = rec.imgs[i];
        img = {};
        img.id = image_id;
        img.view = info.imageView;
        img.type = set_writes[sw_begin + i].descriptorType;
        img.range = desc.view_info.range;
        if (!image_id) {
            rec.local_ok = false;
            continue;
        }
        auto& image = texture_cache.PeekImage(image_id);
        img.uid = image.image_uid;
        img.backing = image.backing;
        img.state = image.backing->state;
        img.full_range = img.range.base == VideoCore::SubresourceBase{} &&
                         img.range.extent == image.info.resources;
        // Per-image exclusions: anything but a plain sampled image, a live depth association,
        // stencil scratch, the Pro ID buffer, and bindings FindImage reached through overlap
        // resolution (only perfect matches are stable while the structure generation is).
        rec.excluded |= desc.type != VideoCore::TextureCache::BindingType::Texture ||
                        img.type != vk::DescriptorType::eSampledImage ||
                        static_cast<bool>(image.depth_id) ||
                        (!stencil_scratches.empty() && IsStencilScratch(image_id)) ||
                        (image.info.num_samples == 2 &&
                         image.info.pixel_format == vk::Format::eR32Uint) ||
                        !texture_cache.IsPerfectMatch(desc.info, image_id);
    }
    if (g_tex_memo_local && !rec.excluded && rec.local_ok) {
        // The generation test and the epoch reads together, inside one memo section: a foreign
        // RegisterImage bumps the epochs before the generation, so separate reads outside a
        // section could pair an old generation with a new epoch (Opus review 2026-10-05).
        const auto section = texture_cache.EnterMemoSection();
        if (section == VideoCore::TextureCache::MemoSection::Refused) {
            rec.local_ok = false;
        } else {
            rec.local_ok = texture_cache.StructGeneration() == tex_gen;
            for (auto& img : rec.imgs) {
                if (img.id) {
                    img.overlap_epoch =
                        std::atomic_ref<u64>{
                            const_cast<u64&>(texture_cache.PeekImage(img.id).overlap_epoch)}
                            .load(std::memory_order_relaxed);
                }
            }
            texture_cache.LeaveMemoSection(section);
        }
    } else {
        rec.local_ok = false;
    }
    rec.samplers.resize(num_samplers);
    rec.sampler_lru.resize(num_samplers);
    for (size_t i = 0; i < num_samplers; ++i) {
        rec.samplers[i] = image_infos[img_begin + num_images + i].sampler;
        rec.sampler_lru[i] = texture_cache.SamplerLruId(st.ssharps[i]);
        rec.excluded |= rec.sampler_lru[i] == ~size_t{0} ||
                        set_writes[sw_begin + num_images + i].descriptorType !=
                            vk::DescriptorType::eSampler;
    }
}

void Rasterizer::P1cCheckWrites(const Pipeline* pipeline) {
    if (binding_compute || set_writes.empty()) {
        return;
    }
    // q3: the table's extent by its highest binding (a sparse layout skips numbers).
    size_t n = 0;
    for (const auto& pb : pipeline->PacketBindings()) {
        n = std::max<size_t>(n, size_t{pb.binding} + pb.count);
    }
    if (n == 0) {
        return;
    }
    u32 bad = 0, max_b = 0;
    for (const auto& w : set_writes) {
        bad += w.dstBinding >= n ? 1 : 0;
        max_b = std::max(max_b, w.dstBinding);
    }
    if (bad == 0) {
        return;
    }
    static std::atomic<u32> logged{0};
    if (logged.fetch_add(1) < 16) {
        LOG_CRITICAL(Render_Vulkan,
                     "p1c: graphics push with {} writes outside the {}-binding table (writes {}, max "
                     "binding {}) | dispatches since BindResources began {} | lane mode {} stream {} "
                     "early record {}",
                     bad, n, set_writes.size(), max_b, p1c_dispatch_serial - p1c_bind_serial,
                     Scheduler::t_lane_mode, scheduler.ActiveStream(),
                     VideoCore::t_force_binding_sync);
    }
    set_writes.erase(std::remove_if(set_writes.begin(), set_writes.end(),
                                    [n](const vk::WriteDescriptorSet& w) {
                                        return w.dstBinding >= n;
                                    }),
                     set_writes.end());
}

void Rasterizer::EmitPipelineResources(const Pipeline* pipeline) {
    P1cCheckWrites(pipeline);
    // SHADPS4_DRAW_PACKET_DELTA: the legacy records push descriptors outside the draw packets.
    scheduler.GetDynamicState().packet_state_unknown = true;
    if (!native_bindings) {
        pipeline->BindResources(set_writes, buffer_barriers, push_data);
        return;
    }
    // The mode is global, so every guest pipeline here was created with heap mappings. A
    // count mismatch would mean the draw-time binding walk disagrees with pipeline creation.
    ASSERT_MSG(pipeline->UsesNativeBindings() &&
                   native_bound_count == pipeline->NativeBindingCount(),
               "Native binding count {} does not match pipeline {}", native_bound_count,
               pipeline->NativeBindingCount());
    const std::span<const u32> indices{native_indices.data(), native_bound_count};
    if (native_bindings->IsVerifying()) {
        native_bindings->Verify(set_writes, indices);
    }
    pipeline->BindResourcesNative(*native_bindings, indices, buffer_barriers, push_data);
}

bool Rasterizer::EmitPipelineResourcesOrPacket(const Pipeline* pipeline, bool occlusion_draw) {
    P1cCheckWrites(pipeline);
    const int mode = g_draw_packet;
    if (mode == 0) {
        EmitPipelineResources(pipeline);
        return false;
    }
    auto& st = g_dp_stats;
    ++st.draws;
    if ((st.draws & 0x3fff) == 0) {
        const auto now = std::chrono::steady_clock::now();
        if (now - st.last_log >= std::chrono::seconds(10)) {
            st.last_log = now;
            DrawPacketLog();
        }
    }
    const std::span<const PacketBinding> table = pipeline->PacketBindings();
    const size_t num_barriers = buffer_barriers.size();
    const size_t num_writes = table.size();
    const size_t num_buffer_infos = buffer_infos.size();
    const size_t num_image_infos = image_infos.size();
    // Sampled (1 draw in 64, scaled): the estimate walks the binding table (review 2026-10-02).
    if ((st.draws & 63) == 0) {
        st.legacy_bytes += 64 * DrawPacketLegacyBytes(num_barriers, table, true);
    }

    // Legacy for everything the packet does not model: the descriptor-heap and native paths,
    // synchronous recording, a draw inside an occlusion query (three records around the draw),
    // a layout whose table does not match what BindResources resolved, oversize payloads.
    u64* fallback = nullptr;
    if (native_bindings || !build_set_writes || !pipeline->UsesPushDescriptors() ||
        !scheduler.IsAsyncRecording()) {
        fallback = &st.fb_other;
    } else if (occlusion_draw) {
        fallback = &st.fb_occlusion;
    } else if (num_writes > DrawPacketMaxWrites || num_writes != set_writes.size() ||
               pipeline->PacketBufferInfos() != num_buffer_infos ||
               pipeline->PacketImageInfos() != num_image_infos) {
        fallback = &st.fb_other;
    } else if (DrawPacketPayloadBytes(num_barriers, num_writes, num_buffer_infos,
                                      num_image_infos) > DrawPacketMaxBytes) {
        fallback = &st.fb_oversize;
    }

    if (!fallback && mode == 2) {
        // Verify: rebuild the writes the worker would build and compare with the legacy ones.
        ++st.verify_checked;
        std::array<vk::WriteDescriptorSet, DrawPacketMaxWrites> derived;
        const u32 count = BuildDrawPacketWrites(
            table, std::span<const vk::DescriptorBufferInfo>{buffer_infos.data(), num_buffer_infos},
            std::span<const vk::DescriptorImageInfo>{image_infos.data(), num_image_infos}, derived);
        std::string why;
        if (count != set_writes.size()) {
            why = fmt::format("count {} vs {}", count, set_writes.size());
        }
        for (size_t i = 0; why.empty() && i < set_writes.size(); ++i) {
            const auto& a = set_writes[i];
            const auto& b = derived[i];
            if (a.pNext != nullptr || a.dstSet != b.dstSet || a.dstBinding != b.dstBinding ||
                a.dstArrayElement != b.dstArrayElement || a.descriptorCount != b.descriptorCount ||
                a.descriptorType != b.descriptorType || a.pTexelBufferView != nullptr ||
                (a.pBufferInfo == nullptr) != (b.pBufferInfo == nullptr) ||
                (a.pImageInfo == nullptr) != (b.pImageInfo == nullptr)) {
                why = fmt::format("write {}: binding {} vs {}, count {} vs {}, type {} vs {}", i,
                                  a.dstBinding, b.dstBinding, a.descriptorCount,
                                  b.descriptorCount, vk::to_string(a.descriptorType),
                                  vk::to_string(b.descriptorType));
                break;
            }
            for (u32 e = 0; why.empty() && e < a.descriptorCount; ++e) {
                if (a.pBufferInfo) {
                    const auto& x = a.pBufferInfo[e];
                    const auto& y = b.pBufferInfo[e];
                    if (x.buffer != y.buffer || x.offset != y.offset || x.range != y.range) {
                        why = fmt::format("write {} buffer info {} differs", i, e);
                    }
                } else if (a.pImageInfo) {
                    const auto& x = a.pImageInfo[e];
                    const auto& y = b.pImageInfo[e];
                    if (x.sampler != y.sampler || x.imageView != y.imageView ||
                        x.imageLayout != y.imageLayout) {
                        why = fmt::format("write {} image info {} differs", i, e);
                    }
                }
            }
        }
        if (!why.empty()) {
            if (++st.mismatches <= 16) {
                LOG_CRITICAL(Render_Vulkan,
                             "DrawPacket: MISMATCH {} (writes {}, buffer infos {}, image infos "
                             "{}, compute {})",
                             why, set_writes.size(), num_buffer_infos, num_image_infos,
                             pipeline->IsCompute());
            }
        }
        EmitPipelineResources(pipeline);
        return false;
    }
    if (fallback) {
        ++*fallback;
        EmitPipelineResources(pipeline);
        return false;
    }

    // Packet. The command-processor side of RecordBufferBarriers happens now (it ends the
    // rendering scope before the barrier, as today); the barrier itself rides in the packet.
    pipeline->BeginBufferBarriers(buffer_barriers);
    ++st.packets;
    st.packet_bytes +=
        DrawPacketRecordBytes(num_barriers, num_writes, num_buffer_infos, num_image_infos);
    return true;
}

u64 Rasterizer::OcclusionDrawSig(const Pipeline* pipeline) const {
    // SHADPS4_OCCLUSION_QUERIES: what tells two queries on one result block apart (vk_occlusion.h):
    // the pipeline and the targets the draw tests against - stable from frame to frame.
    const auto& regs = liverpool->BackRegs();
    u64 h = reinterpret_cast<u64>(static_cast<VkPipeline>(pipeline->Handle()));
    const auto mix = [&h](u64 v) { h = (h ^ v) * 0x100000001B3ULL + (h >> 29); };
    mix(regs.depth_buffer.DepthAddress());
    mix(regs.color_buffers[0].Address());
    mix(std::bit_cast<u32>(regs.depth_control));
    return h;
}

void Rasterizer::RecordPacket(const Pipeline* pipeline, const DrawPacketHeader& tail) {
    DrawPacketDesc desc{};
    desc.header = tail;
    const bool compute = pipeline->IsCompute();
    desc.header.layout = pipeline->GetLayout();
    desc.header.pipeline = pipeline->Handle();
    desc.header.stage_flags =
        compute ? vk::ShaderStageFlags{vk::ShaderStageFlagBits::eCompute} : AllGraphicsStageBits;
    desc.header.bind_point =
        compute ? vk::PipelineBindPoint::eCompute : vk::PipelineBindPoint::eGraphics;
    desc.table = pipeline->PacketBindings();
    desc.barriers = std::span<const vk::BufferMemoryBarrier2>{buffer_barriers.data(),
                                                             buffer_barriers.size()};
    desc.buffer_infos =
        std::span<const vk::DescriptorBufferInfo>{buffer_infos.data(), buffer_infos.size()};
    desc.image_infos =
        std::span<const vk::DescriptorImageInfo>{image_infos.data(), image_infos.size()};
    desc.header.num_barriers = static_cast<u32>(desc.barriers.size());
    desc.header.num_writes = static_cast<u32>(desc.table.size());
    desc.header.num_buffer_infos = static_cast<u32>(desc.buffer_infos.size());
    desc.header.num_image_infos = static_cast<u32>(desc.image_infos.size());
    desc.header.push_data = push_data;
    if (desc.header.flags & (DrawPacketDraw | DrawPacketDispatch)) {
        ++g_dp_stats.folded;
    } else {
        // The draw / dispatch record that follows separately, so the bytes compare with
        // legacy_bytes (which always includes it).
        g_dp_stats.packet_bytes += 32 + 40;
    }
    scheduler.RecordDrawPacket(desc);
}

bool Rasterizer::IsComputeMetaClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Most of the time when a metadata is updated with a shader it gets cleared. It means
    // we can skip the whole dispatch and update the tracked state instead. Also, it is not
    // intended to be consumed and in such rare cases (e.g. HTile introspection, CRAA) we
    // will need its full emulation anyways.
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
    const auto& bound = GetBoundStage(Shader::LogicalStage::Compute);

    // Assume if a shader reads metadata, it is a copy shader.
    for (const auto& desc : info.buffers) {
        const VAddr address = desc.GetSharp(bound).base_address;
        if (!desc.IsSpecial() && !desc.is_written && texture_cache.IsMeta(address)) {
            return false;
        }
    }

    // Metadata surfaces are tiled and thus need address calculation to be written properly.
    // If a shader wants to encode HTILE, for example, from a depth image it will have to compute
    // proper tile address from dispatch invocation id. This address calculation contains an xor
    // operation so use it as a heuristic for metadata writes that are probably not clears.
    if (!info.has_bitwise_xor) {
        // Assume if a shader writes metadata without address calculation, it is a clear shader.
        for (const auto& desc : info.buffers) {
            const VAddr address = desc.GetSharp(bound).base_address;
            if (!desc.IsSpecial() && desc.is_written && texture_cache.ClearMeta(address)) {
                // Assume all slices were updates
                LOG_TRACE(Render_Vulkan, "Metadata update skipped");
                return true;
            }
        }
    }
    return false;
}

bool Rasterizer::IsComputeImageCopy(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->BackCsRegs();
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
    const auto& bound = GetBoundStage(Shader::LogicalStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // Those 2 buffers must both be formatted. One must be source and another destination.
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (!desc0.is_formatted || !desc1.is_formatted || desc0.is_written == desc1.is_written) {
        return false;
    }

    // Buffers must have the same size and each thread of the dispatch must copy 1 dword of data
    const AmdGpu::Buffer buf0 = desc0.GetSharp(bound);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(bound);
    if (buf0.GetSize() != buf1.GetSize() || cs_pgm.dim_x != (buf0.GetSize() / 256)) {
        return false;
    }

    // Find images the buffer alias
    const auto image0_id = texture_cache.FindImageFromRange(buf0.base_address, buf0.GetSize());
    if (!image0_id) {
        return false;
    }
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image copy must be valid
    VideoCore::Image& image0 = texture_cache.GetImage(image0_id);
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image0.info.guest_size != image1.info.guest_size ||
        image0.info.pitch != image1.info.pitch || image0.info.guest_size != buf0.GetSize() ||
        image0.info.num_bits != image1.info.num_bits) {
        return false;
    }

    // Perform image copy
    VideoCore::Image& src_image = desc0.is_written ? image1 : image0;
    VideoCore::Image& dst_image = desc0.is_written ? image0 : image1;
    if (instance.IsMaintenance8Supported() ||
        src_image.info.props.is_depth == dst_image.info.props.is_depth) {
        dst_image.CopyImage(src_image);
    } else {
        const auto& copy_buffer =
            buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::DeviceLocal);
        dst_image.CopyImageWithBuffer(src_image, copy_buffer.Handle(), 0);
    }
    dst_image.flags |= VideoCore::ImageFlagBits::GpuModified;
    dst_image.flags &= ~VideoCore::ImageFlagBits::Dirty;
    return true;
}

bool Rasterizer::IsComputeImageClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->BackCsRegs();
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
    const auto& bound = GetBoundStage(Shader::LogicalStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // From those 2 buffers, first must hold the clear vector and second the image being cleared
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (desc0.is_formatted || !desc1.is_formatted || desc0.is_written || !desc1.is_written) {
        return false;
    }

    // First buffer must have size of vec4 and second the size of a single layer
    const AmdGpu::Buffer buf0 = desc0.GetSharp(bound);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(bound);
    const u32 buf1_bpp = AmdGpu::NumBitsPerBlock(buf1.GetDataFmt());
    if (buf0.GetSize() != 16 || (cs_pgm.dim_x * 128ULL * (buf1_bpp / 8)) != buf1.GetSize()) {
        return false;
    }

    // Find image the buffer alias
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image clear must be valid
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image1.info.guest_size != buf1.GetSize() || image1.info.num_bits != buf1_bpp ||
        image1.info.props.is_depth) {
        return false;
    }

    // Perform image clear
    const float* values = reinterpret_cast<float*>(buf0.base_address);
    const vk::ClearValue clear = {
        .color = {.float32 = std::array<float, 4>{values[0], values[1], values[2], values[3]}},
    };
    const VideoCore::SubresourceRange range = {
        .base =
            {
                .level = 0,
                .layer = 0,
            },
        .extent = image1.info.resources,
    };
    image1.Clear(clear, range);
    image1.flags |= VideoCore::ImageFlagBits::GpuModified;
    image1.flags &= ~VideoCore::ImageFlagBits::Dirty;
    return true;
}

void Rasterizer::BindBuffers(const Shader::BoundStage& bound, Shader::Backend::Bindings& binding,
                             Shader::PushData& push_data) {
    const auto& stage = *bound.info;
    buffer_bindings.clear();
    const u64 struct_gen_before = buffer_cache.StructGeneration();

    // Pass 1 (FindBuffer) vs pass 2 (ObtainBuffer -> SynchronizeBuffer) timed separately: this
    // function is 62% of per-draw cost and a stack scan is not trustworthy for apportioning it.
    const auto t_find = Common::FaultMonitor::Now();
    const u64 bb_find_t0 = g_draw_cost ? __rdtsc() : 0;
    // SHADPS4_LEAN_BIND: ClampRangeSize returns small sizes unchanged (memory.cpp, < 1 GB) - test
    // that here instead of an out-of-line call per binding; buffers found through the hint.
    const bool lean_bind = VideoCore::g_lean_bind != 0;
    for (const auto& desc : stage.buffers) {
        const auto vsharp = desc.GetSharp(bound);
        if (!desc.IsSpecial() && vsharp.base_address != 0 && vsharp.GetSize() > 0) {
            const u64 vsize = vsharp.GetSize();
            const u64 size = lean_bind && vsize < (u64{1} << 30)
                                 ? vsize
                                 : memory->ClampRangeSize(vsharp.base_address, vsize);
            const auto buffer_id =
                lean_bind ? buffer_cache.FindBufferHinted(vsharp.base_address, size)
                          : buffer_cache.FindBuffer(vsharp.base_address, size);
            // SHADPS4_READ_MEMO_PREFETCH: pass 2 probes the read memo for this binding.
            if (g_read_memo_prefetch && buffer_id && !desc.is_written && !desc.is_formatted) {
                buffer_cache.PrefetchReadMemo(vsharp.base_address, static_cast<u32>(size),
                                              buffer_id);
            }
            buffer_bindings.emplace_back(buffer_id, vsharp, size);
            if (desc.is_formatted && size > (u64{64} << 20)) {
                native_stats.giant_bound = true;
                // s369 census (NATIVE_TIMING): what the giant formatted view is, once per
                // distinct (base, shader).
                static const bool giant_log = [] {
                    const char* e = std::getenv("SHADPS4_NATIVE_TIMING");
                    return e && e[0] == '1';
                }();
                static std::set<std::pair<u64, u64>> giant_seen;
                if (giant_log && giant_seen.size() < 24 &&
                    giant_seen.insert({vsharp.base_address, stage.pgm_hash}).second) [[unlikely]] {
                    std::string img = "none";
                    if (const auto id = texture_cache.FindImageFromRange(vsharp.base_address,
                                                                         size, false)) {
                        const auto& im = texture_cache.PeekImage(id);
                        img = fmt::format("{:#x}+{:#x} {}x{}x{} {} target={} ", im.info.guest_address,
                                          im.info.guest_size, im.info.size.width,
                                          im.info.size.height, im.info.size.depth,
                                          vk::to_string(im.info.pixel_format),
                                          im.binding.is_target ? 1 : 0);
                    }
                    LOG_WARNING(Render_Vulkan,
                                "GiantView: base {:#x} size {:#x} stride {} num_records {:#x} dfmt {} "
                                "nfmt {} swizzle {} written {} | shader {:#x} stage {} | image at base: {}",
                                u64(vsharp.base_address), size, u32(vsharp.stride), vsharp.num_records,
                                u32(vsharp.data_format), u32(vsharp.num_format),
                                u32(vsharp.swizzle_enable), desc.is_written ? 1 : 0, stage.pgm_hash,
                                static_cast<u32>(stage.l_stage), img);
                }
            }
            if (g_static_key_census && !binding_compute && buffer_id &&
                vsharp.base_address >= HeapLo && vsharp.base_address < HeapHi) [[unlikely]] {
                NoteStaticKeyHeapId(buffer_id.index);
            }
        } else {
            buffer_bindings.emplace_back(VideoCore::BufferId{}, vsharp, 0);
        }
    }
    const u64 find_ns = Common::FaultMonitor::SinceNs(t_find);
    if (binding_compute && dispatch_target == 1 && VideoCore::BufferCache::CensusEnabled())
        [[unlikely]] {
        for (u32 i = 0; i < buffer_bindings.size(); ++i) {
            const auto& [buffer_id, vsharp, size] = buffer_bindings[i];
            if (size != 0) {
                buffer_cache.CensusAscBinding(buffer_id, vsharp.base_address, size,
                                              stage.buffers[i].is_written);
            }
        }
        if (VideoCore::BufferCache::RangeShadowEnabled()) {
            // Step 2 shadow: per-Buffer open-batch conflicts vs real range overlaps (RAW: main wrote
            // a range this dispatch touches; WAR: main read a range it writes).
            struct Stats {
                u64 dispatches = 0, per_buffer = 0, range = 0;
                std::array<u64, 256> by_kind{};
                std::unordered_map<u64, u64> by_id;
            };
            static Stats st;
            const u64 now = scheduler.CurrentTick();
            bool pb = false;
            u8 first_kind = 0;
            u64 first_id = 0;
            for (u32 i = 0; i < buffer_bindings.size(); ++i) {
                const auto& [buffer_id, vsharp, size] = buffer_bindings[i];
                if (!buffer_id || size == 0) {
                    continue;
                }
                pb |= buffer_cache.GetBuffer(buffer_id).main_access_tick >= now;
                if (first_kind == 0) {
                    u64 id = 0;
                    first_kind = buffer_cache.RangeConflict(buffer_id, vsharp.base_address, size,
                                                            stage.buffers[i].is_written, now, id);
                    first_id = id;
                }
            }
            ++st.dispatches;
            st.per_buffer += pb;
            if (first_kind != 0) {
                ++st.range;
                ++st.by_kind[first_kind];
                ++st.by_id[(u64(first_kind) << 56) ^ first_id];
            }
            if ((st.dispatches & 0xffff) == 0) {
                std::string kinds;
                for (size_t k = 0; k < st.by_kind.size(); ++k) {
                    if (st.by_kind[k] != 0) {
                        kinds += fmt::format(" k{}={}", k, st.by_kind[k]);
                    }
                }
                std::vector<std::pair<u64, u64>> top(st.by_id.begin(), st.by_id.end());
                std::sort(top.begin(), top.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });
                std::string ids;
                for (size_t k = 0; k < std::min<size_t>(top.size(), 8); ++k) {
                    ids += fmt::format(" k{}:{:08x}={}", top[k].first >> 56,
                                       top[k].first & 0x00ffffffffffffffULL, top[k].second);
                }
                LOG_INFO(Render_Vulkan,
                         "AscRangeShadow: candidates={} per_buffer_open_conflicts={} "
                         "range_open_conflicts={} kinds:{} top:{}",
                         st.dispatches, st.per_buffer, st.range, kinds, ids);
            }
        }
    }
    if (binding_compute && dispatch_target == 1 && scheduler.HasLane() &&
        VideoCore::t_force_binding_sync) {
        // SHADPS4_ASC_LANE: an early ASC dispatch records pass 2 and the dispatch on the lane;
        // its waits on main are taken per binding range (ObtainBuffer, rule A). A read texel
        // binding over a cached image needs an image->buffer copy (images are main only).
        bool texel_block = false;
        for (u32 i = 0; i < buffer_bindings.size(); ++i) {
            const auto& desc = stage.buffers[i];
            const auto& [buffer_id, vsharp, size] = buffer_bindings[i];
            if (desc.is_formatted && !desc.is_written && buffer_id && size != 0 &&
                texture_cache.FindImageFromRange(vsharp.base_address, size)) {
                texel_block = true;
                break;
            }
        }
        if (texel_block) {
            dispatch_target = 0;
            cur_dispatch_run = 0;
            scheduler.LaneFallbackMain();
        } else {
            scheduler.SwitchStream(1);
            dispatch_on_asc = true;
            for (const auto& desc : stage.buffers) {
                if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
                    scheduler.LaneNeedMain(buffer_cache.GetGdsBuffer()->main_access_tick, 3, true);
                }
            }
        }
    }
    if (binding_compute && scheduler.HasAscStream()) {
        // SHADPS4_ASC_STREAM: the buffers are known now. A candidate goes to the ASC stream only if
        // no buffer it binds was touched by the main batch still being recorded (an ASC segment
        // may only wait on main batches already flushed) and pass 1 created or joined nothing.
        u32 target = dispatch_target;
        u64 wait_main = 0;
        if (target == 1) {
            const u64 now = scheduler.CurrentTick();
            bool ok = buffer_cache.StructGeneration() == struct_gen_before;
            u32 reason = ok ? 0 : 1;
            // SHADPS4_ASC_TEXEL: a read texel binding over a cached image needs an image->buffer
            // copy (images are main-stream only). Checked unconditionally: mode 2's flush must not
            // send such a dispatch to the ASC stream (Opus review 2026-09-28).
            bool texel_block = false;
            for (u32 i = 0; i < buffer_bindings.size(); ++i) {
                const auto& desc = stage.buffers[i];
                const auto& [buffer_id, vsharp, size] = buffer_bindings[i];
                if (desc.is_formatted && !desc.is_written && buffer_id && size != 0 &&
                    texture_cache.FindImageFromRange(vsharp.base_address, size)) {
                    texel_block = true;
                    break;
                }
            }
            static const bool asc_range = VideoCore::BufferCache::AscRangeEnabled();
            for (u32 i = 0; i < buffer_bindings.size(); ++i) {
                if (!ok) {
                    break;
                }
                const auto& [buffer_id, vsharp, size] = buffer_bindings[i];
                if (!buffer_id) {
                    continue;
                }
                const auto& buffer = buffer_cache.GetBuffer(buffer_id);
                if (asc_range) {
                    // SHADPS4_ASC_RANGE: conflict only if the open batch touched this binding's
                    // range (RAW / WAR / upload WAR); otherwise wait on the newest flushed batch
                    // that touched the Buffer (never on the open one: the ASC segment is
                    // submitted before it, I1).
                    buffer_cache.MarkAscBound(buffer_id);
                    u64 id = 0;
                    const u8 kind = buffer_cache.RangeConflict(
                        buffer_id, vsharp.base_address, size, stage.buffers[i].is_written, now, id);
                    ok = kind == 0;
                    if (!ok) {
                        reason = buffer.main_write_tick >= now ? 2 : 3;
                    }
                    // Step 3 (I2 precision): a read binding only depends on flushed main WRITES
                    // (RAW); a written one also on flushed reads (WAR) - the heap is read by every
                    // graphics batch, so access ticks made every segment chase main.
                    // A read binding over CPU-dirty pages uploads on the ASC stream (a write):
                    // it keeps the access-based wait (WAR vs flushed main reads).
                    const bool written =
                        stage.buffers[i].is_written ||
                        buffer_cache.IsRegionCpuModified(vsharp.base_address, size);
                    const u64 last = written ? buffer.main_access_tick : buffer.main_write_tick;
                    const u64 prev = written ? buffer.prev_access_tick : buffer.prev_write_tick;
                    const u64 flushed = last < now ? last : prev;
                    wait_main = std::max(wait_main, flushed);
                    if (VideoCore::BufferCache::CensusEnabled()) [[unlikely]] {
                        // Would range precision drop this flushed-batch wait? (I2 census)
                        static u64 waits = 0, logged = 0, avoidable = 0;
                        if (flushed != 0) {
                            ++waits;
                            const int c = buffer.PrevLogConflict(flushed, vsharp.base_address,
                                                                 size, written);
                            logged += c >= 0;
                            avoidable += c == 0;
                            if ((waits & 0xfffff) == 0) {
                                LOG_INFO(Render_Vulkan,
                                         "AscI2Census: flushed_waits={} with_log={} "
                                         "no_range_overlap={}",
                                         waits, logged, avoidable);
                            }
                        }
                    }
                    continue;
                }
                ok = buffer.main_access_tick < now;
                if (!ok) {
                    reason = buffer.main_write_tick >= now ? 2 : 3;
                }
                wait_main = std::max(wait_main, buffer.main_access_tick);
            }
            if (ok && buffer_cache.asc_indirect_size != 0) {
                const auto args_id = buffer_cache.FindBuffer(buffer_cache.asc_indirect_addr,
                                                             buffer_cache.asc_indirect_size);
                ok = buffer_cache.StructGeneration() == struct_gen_before;
                if (ok) {
                    const auto& args = buffer_cache.GetBuffer(args_id);
                    ok = args.main_access_tick < now;
                    wait_main = std::max(wait_main, args.main_access_tick);
                }
                if (!ok && reason == 0) {
                    reason = 4;
                }
            }
            for (const auto& desc : stage.buffers) {
                if (ok && desc.buffer_type == Shader::BufferType::GdsBuffer) {
                    const u64 gds_tick = buffer_cache.GetGdsBuffer()->main_access_tick;
                    ok = gds_tick < now;
                    wait_main = std::max(wait_main, gds_tick);
                    if (!ok) {
                        reason = 5;
                    }
                }
            }
            // SHADPS4_ASC_STREAM=2: instead of keeping the dispatch on main (whose writes then
            // demote every later dispatch of the span: 89% of demotions, run
            // diag-fire-ascstream-20260928-r), flush the open main batch once. Its submission
            // first submits the open ASC segment (I1); the new segment waits on the flushed batch.
            static const bool flush_mode = [] {
                const char* value = std::getenv("SHADPS4_ASC_STREAM");
                return value && value[0] == '2';
            }();
            // A texel read over an image must stay on main: a flush cannot move its copy.
            if (!ok && reason != 1 && !texel_block && flush_mode) {
                const u64 flushed = scheduler.CurrentTick();
                scheduler.Flush();
                wait_main = std::max(wait_main, flushed);
                ok = true;
                ++asc_flushes;
            }
            if (texel_block) {
                ok = false;
                reason = 6;
            }
            if (!ok) {
                target = 0;
                dispatch_target = 0;
                cur_dispatch_run = 0; // Its run state was the ASC stream's.
                ++asc_demoted;
                // Reasons: 1 pass 1 created/joined, 2 binding written by the open main batch,
                // 3 binding only read by it, 4 indirect args, 5 GDS.
                ++asc_demote_reason[std::min<u32>(reason, 6)];
            }
        }
        scheduler.SwitchStream(target);
        if (target == 1) {
            scheduler.AddAscWaitMain(wait_main);
            dispatch_on_asc = true;
        }
    }
    u64 obtain_ns = 0;
    const u64 bb_find_cyc = g_draw_cost ? __rdtsc() - bb_find_t0 : 0;
    g_p5[P5Find] += bb_find_cyc;
    const u64 bb_find_each = buffer_bindings.empty() ? 0 : bb_find_cyc / buffer_bindings.size();

    // SHADPS4_LEAN_BIND: plain read bindings take BufferCache::ObtainReadLean.
    // SHADPS4_ASC_LANE: lane bindings take ObtainBuffer (its rule-A lookup); q11c: ObtainReadLean runs the
    // same lookup, so lane read bindings take the lean path too (SHADPS4_LANE_LEAN_READ=0: the old way).
    static const bool lane_lean = [] {
        const char* env = std::getenv("SHADPS4_LANE_LEAN_READ");
        return env && env[0] == '1'; // q11d: off by default (no gain, s653)
    }();
    const bool lean_read = lean_bind && buffer_cache.LeanReadOk() &&
                           (lane_lean || !Vulkan::Scheduler::t_lane_mode);
    // Second pass to re-bind buffers that were updated after binding
    for (u32 i = 0; i < buffer_bindings.size(); i++) {
        const auto& [buffer_id, vsharp, size] = buffer_bindings[i];
        const auto& desc = stage.buffers[i];
        const u64 p5_t0 = g_draw_cost ? __rdtsc() : 0;
        const u64 rp_seq0 = g_draw_repeat == 2 ? scheduler.RecordSeq() : 0;
        const bool rc_on = g_bind_region_census && !binding_compute && buffer_id;
        const u64 rc_t0 = rc_on ? __rdtsc() : 0;
        const u64 rc_up0 = rc_on ? buffer_cache.SyncUploadBytesTotal() : 0;
        // SHADPS4_BIND_MEMO_STATS: bindings a memo hit would still run in full are timed.
        const bool memo_fixed = bind_memo_timing && (desc.IsSpecial() || desc.is_written);
        const u64 memo_t0 = memo_fixed ? Common::FencedRDTSC() : 0;
        const bool is_storage = desc.IsStorage(vsharp);
        const u32 alignment =
            is_storage ? instance.StorageMinAlignment() : instance.UniformMinAlignment();
        // The buffer object behind this binding's legacy payload; null for null descriptors.
        const VideoCore::Buffer* bound_buffer = nullptr;
        // Buffer is not from the cache, either a special buffer or unbound.
        if (!buffer_id) {
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                gds_buf->NoteMainAccess(scheduler.CurrentTick(), true);
                if (VideoCore::g_record_stream == 0) {
                    gds_buf->CheckMainVsAsc(gds_buf->CpuAddr(), gds_buf->SizeBytes(), true);
                } else if (VideoCore::BufferCache::AscDepsEnabled()) {
                    buffer_cache.pending_asc.push_back({VideoCore::BufferId{}, gds_buf,
                                                        gds_buf->CpuAddr(), gds_buf->SizeBytes(),
                                                        true});
                }
                buffer_infos.emplace_back(gds_buf->Handle(), 0, gds_buf->SizeBytes());
                bound_buffer = gds_buf;
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                AmdGpu::MtbCensus::Note(AmdGpu::MtbCensus::StreamAlloc);
                auto& vk_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                const u32 ubo_size = bound.flattened_ud_buf.size() * sizeof(u32);
                const u64 offset =
                    vk_buffer.Copy(bound.flattened_ud_buf.data(), ubo_size, alignment);
                buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
                bound_buffer = &vk_buffer;
            } else if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                // Permutations compiled without enabled planes never read the buffer, so the
                // declared binding is satisfied with a null descriptor instead of a copy.
                if (liverpool->BackRegs().clipper_control.user_clip_plane_enable == 0) {
                    buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
                } else {
                    AmdGpu::MtbCensus::Note(AmdGpu::MtbCensus::StreamAlloc);
                    auto& vk_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                    std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = liverpool->BackRegs().clip_user_data[i];
                        planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
                        planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
                        planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
                        planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
                    }
                    const u32 ubo_size = static_cast<u32>(sizeof(planes));
                    const u64 offset = vk_buffer.Copy(planes.data(), ubo_size, alignment);
                    buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
                    // DRAW_REPEAT=2 + HEAP: a clip-plane repeat reuses the base's copy of equal
                    // planes; verify compares this slot by range (its offset is a fresh copy).
                    draw_repeat.clip_info_idx = static_cast<u32>(buffer_infos.size() - 1);
                    bound_buffer = &vk_buffer;
                }
            } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) {
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                buffer_infos.emplace_back(bda_buffer->Handle(), 0, bda_buffer->SizeBytes());
                bound_buffer = bda_buffer;
            } else if (desc.buffer_type == Shader::BufferType::FaultBuffer) {
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                buffer_infos.emplace_back(fault_buffer->Handle(), 0, fault_buffer->SizeBytes());
                bound_buffer = fault_buffer;
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) {
                AmdGpu::MtbCensus::Note(AmdGpu::MtbCensus::StreamAlloc);
                auto& lds_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                const auto& cs_program = liverpool->BackCsRegs();
                const auto lds_size = cs_program.SharedMemSize() * cs_program.NumWorkgroups();
                const auto [data, offset] = lds_buffer.Map(lds_size, alignment);
                std::memset(data, 0, lds_size);
                buffer_infos.emplace_back(lds_buffer.Handle(), offset, lds_size);
                bound_buffer = &lds_buffer;
            } else {
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            }
        } else {
            const auto t_obtain = Common::FaultMonitor::Now();
            const u64 memo_uploads = bind_memo_active ? buffer_cache.SyncUploadBytesTotal() : 0;
            u64 bb_t0 = 0;
            if (g_draw_cost) {
                VideoCore::DiagSyncWalkCycles() = 0;
                VideoCore::DiagSyncRestCycles() = 0;
                bb_t0 = __rdtsc();
            }
            std::pair<VideoCore::Buffer*, u32> obtained{};
            if (lean_read && !desc.is_written && !desc.is_formatted) {
                obtained = buffer_cache.ObtainReadLean(vsharp.base_address, size, buffer_id);
            }
            if (!obtained.first) {
                // s370 census (NATIVE_TIMING): the memory-window binds (formatted, > 64 MB).
                static const bool window_census = [] {
                    const char* e = std::getenv("SHADPS4_NATIVE_TIMING");
                    return e && e[0] == '1';
                }();
                const bool window = window_census && desc.is_formatted && size > (u64{64} << 20);
                const u64 w0 = window ? __rdtsc() : 0;
                const u64 wup0 = window ? buffer_cache.SyncUploadBytesTotal() : 0;
                const u64 wseq0 = window ? scheduler.RecordSeq() : 0;
                obtained = buffer_cache.ObtainBuffer(
                    vsharp.base_address, size, desc.is_written, desc.is_formatted, buffer_id,
                    is_storage ? Common::BufferSampleCaller::Storage
                               : Common::BufferSampleCaller::Uniform);
                if (window) [[unlikely]] {
                    struct WindowStats {
                        u64 n = 0, cyc = 0, up_n = 0, up_bytes = 0, up_cyc = 0, rec_n = 0, max_cyc = 0;
                    };
                    static WindowStats ws;
                    const u64 dt = __rdtsc() - w0;
                    const u64 ub = buffer_cache.SyncUploadBytesTotal() - wup0;
                    ++ws.n;
                    ws.cyc += dt;
                    ws.max_cyc = std::max(ws.max_cyc, dt);
                    if (ub != 0) {
                        ++ws.up_n;
                        ws.up_bytes += ub;
                        ws.up_cyc += dt;
                    }
                    ws.rec_n += scheduler.RecordSeq() != wseq0;
                    if ((ws.n & 0x3ffff) == 0) {
                        LOG_INFO(Render_Vulkan,
                                 "WindowBind: {} binds, {:.0f} cycles each (max {}); uploading binds "
                                 "{} ({:.0f} cycles each, {:.0f} KB each); binds that recorded "
                                 "commands {}",
                                 ws.n, double(ws.cyc) / double(ws.n), ws.max_cyc, ws.up_n,
                                 ws.up_n ? double(ws.up_cyc) / double(ws.up_n) : 0.0,
                                 ws.up_n ? double(ws.up_bytes) / 1024.0 / double(ws.up_n) : 0.0,
                                 ws.rec_n);
                        ws = {};
                    }
                }
            }
            const auto [vk_buffer, offset] = obtained;
            obtain_ns += Common::FaultMonitor::SinceNs(t_obtain);
            const u64 bb_t1 = g_draw_cost ? __rdtsc() : 0;
            if (bind_memo_active) {
                bind_memo_uploaded[i] = buffer_cache.SyncUploadBytesTotal() != memo_uploads;
            }
            const u32 offset_aligned = Common::AlignDown(offset, alignment);
            const u32 adjust = offset - offset_aligned;
            ASSERT(adjust % 4 == 0);
            push_data.AddOffset(binding.buffer, adjust);
            buffer_infos.emplace_back(vk_buffer->Handle(), offset_aligned, size + adjust);
            bound_buffer = vk_buffer;
            if (g_repeat_heap && !binding_compute && vsharp.base_address >= RepeatHeapLo &&
                vsharp.base_address < RepeatHeapHi) {
                draw_repeat.heap_pending.push_back(
                    {static_cast<u32>(buffer_infos.size() - 1), binding.buffer, vk_buffer, alignment});
            }
            if (auto barrier =
                    vk_buffer->GetBarrier(desc.is_written ? vk::AccessFlagBits2::eShaderWrite
                                                          : vk::AccessFlagBits2::eShaderRead,
                                          vk::PipelineStageFlagBits2::eAllCommands, 0,
                                          cur_dispatch_run)) {
                buffer_barriers.emplace_back(*barrier);
            }
            if (desc.is_written && VideoCore::TextureCache::BufferImageAliasDiag()) {
                texture_cache.DiagBufferWriteOverImages(vsharp.base_address, size,
                                                        desc.is_formatted);
            }
            // SHADPS4_RAW_WRITE_INVALIDATE=1 (port of upstream #5092): raw storage-buffer writes
            // can also make an aliased cached image stale, not only formatted ones.
            const bool raw_write_invalidate = g_raw_write_invalidate;
            if (desc.is_written && (desc.is_formatted || raw_write_invalidate)) {
                texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size);
            }
            if (g_draw_cost) {
                const u32 w = desc.is_written ? 1 : 0;
                const u64 walk = VideoCore::DiagSyncWalkCycles();
                const u64 rest = VideoCore::DiagSyncRestCycles();
                const u64 obtain = bb_t1 - bb_t0;
                g_bb_cyc[w][0] += bb_find_each;
                g_bb_cyc[w][1] += walk;
                g_bb_cyc[w][2] += rest;
                g_bb_cyc[w][3] += obtain > walk + rest ? obtain - walk - rest : 0;
                g_bb_cyc[w][4] += __rdtsc() - bb_t1;
                ++g_bb_n[w];
                if (!w && g_dc_repeat_census) {
                    NoteReadBindingRepeat(scheduler.CurrentTick(), buffer_id.index,
                                          vsharp.base_address, size);
                }
            }
        }
        if (g_draw_cost && !buffer_id) {
            ++g_bb_special_n;
        }

        const u32 unified_binding = binding.unified++;
        if (native_bindings) {
            // Encode exactly the handle generation, offset and range recorded just above. A
            // later merge leaves this payload alone in the legacy path, and so does this.
            const auto& info = buffer_infos.back();
            native_indices[unified_binding] = native_bindings->BufferIndex(
                info.buffer, bound_buffer ? bound_buffer->BufferDeviceAddress() : 0, info.offset,
                info.range, is_storage);
        }
        if (build_set_writes) {
            auto& set_write = set_writes[set_write_index++];
            set_write.dstSet = VK_NULL_HANDLE;
            set_write.dstBinding = unified_binding;
            set_write.dstArrayElement = 0;
            set_write.descriptorCount = 1;
            set_write.descriptorType = is_storage ? vk::DescriptorType::eStorageBuffer
                                                  : vk::DescriptorType::eUniformBuffer;
            set_write.pBufferInfo = &buffer_infos.back();
        }
        ++binding.buffer;
        if (memo_fixed) {
            bind_memo_fixed_ticks += Common::FencedRDTSC() - memo_t0;
        }
        if (g_draw_cost) {
            g_p5[buffer_id ? P5Cache : P5Special] += __rdtsc() - p5_t0;
        }
        if (g_draw_repeat == 2 && scheduler.RecordSeq() != rp_seq0) [[unlikely]] {
            const VAddr a = vsharp.base_address;
            g_rp_seq_who += fmt::format(" b{}:{}{}{}{}@{:#x}+{:#x}", i,
                                        a >= RepeatHeapLo && a < RepeatHeapHi ? "heap" : "static",
                                        desc.is_formatted ? ",fmt" : "",
                                        desc.is_written ? ",wr" : "", desc.IsSpecial() ? ",sp" : "",
                                        a, size);
        }
        if (rc_on) {
            auto& c = GetRegionCensus();
            if (c.tsc0 == 0) {
                c.tsc0 = __rdtsc();
                c.t0 = std::chrono::steady_clock::now();
            }
            auto& r = c.by[static_cast<u32>(vsharp.base_address >> 24)];
            ++r.n;
            r.cyc += __rdtsc() - rc_t0;
            r.uploaded += buffer_cache.SyncUploadBytesTotal() != rc_up0;
            r.formatted += desc.is_formatted;
            r.written += desc.is_written;
            r.bytes += size;
            if ((++c.bindings & 0x3fffff) == 0) {
                RegionCensusLog(c);
            }
        }
    }
    Common::FaultMonitor::CountBindBufferPass(find_ns, obtain_ns);
}

VideoCore::ImageId Rasterizer::StencilScratchFor(VideoCore::ImageId depth_id,
                                                 const VideoCore::ImageInfo& info) {
    const u64 depth_uid = texture_cache.GetImage(depth_id).image_uid;
    for (auto& entry : stencil_scratches) {
        if (entry.depth_id == depth_id && entry.depth_uid == depth_uid) {
            return entry.scratch_id;
        }
    }
    // The depth image can be recreated at the same address (same surface): reuse the scratch.
    for (auto& entry : stencil_scratches) {
        const auto& scratch = texture_cache.GetImage(entry.scratch_id);
        if (scratch.info.guest_address == info.guest_address &&
            scratch.info.size == info.size && scratch.info.pixel_format == info.pixel_format) {
            entry.depth_id = depth_id;
            entry.depth_uid = depth_uid;
            return entry.scratch_id;
        }
    }
    const VideoCore::ImageId scratch_id = texture_cache.InsertScratchImage(info);
    stencil_scratches.push_back({depth_id, depth_uid, scratch_id});
    const auto& depth = texture_cache.GetImage(depth_id);
    const auto& scratch = texture_cache.GetImage(scratch_id);
    LOG_WARNING(Render_Vulkan,
                "StencilStorage: scratch {} ({}x{} {}) for depth {:#x} ({}x{} {}, {} samples)",
                scratch_id.index, scratch.info.size.width, scratch.info.size.height,
                vk::to_string(scratch.info.pixel_format), depth.info.guest_address,
                depth.info.size.width, depth.info.size.height,
                vk::to_string(depth.info.pixel_format), depth.info.num_samples);
    return scratch_id;
}

bool Rasterizer::IsStencilScratch(VideoCore::ImageId id) const {
    return std::ranges::any_of(stencil_scratches,
                               [id](const auto& entry) { return entry.scratch_id == id; });
}

void Rasterizer::CopyStencilPlane(VideoCore::ImageId src_id, VideoCore::ImageId dst_id) {
    auto& src = texture_cache.GetImage(src_id);
    auto& dst = texture_cache.GetImage(dst_id);
    const auto aspect_of = [](const VideoCore::Image& image) {
        return (image.aspect_mask & vk::ImageAspectFlagBits::eStencil)
                   ? vk::ImageAspectFlagBits::eStencil
                   : vk::ImageAspectFlagBits::eColor;
    };
    const vk::ImageCopy region{
        .srcSubresource = {aspect_of(src), 0, 0, 1},
        .dstSubresource = {aspect_of(dst), 0, 0, 1},
        .extent = {std::min(src.info.size.width, dst.info.size.width),
                   std::min(src.info.size.height, dst.info.size.height), 1},
    };
    scheduler.EndRendering();
    src.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {});
    dst.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {});
    auto command_access = scheduler.AcquireCommandBuffer();
    command_access.Handle().copyImage(src.GetImage(), src.backing->state.layout, dst.GetImage(),
                                      dst.backing->state.layout, region);
}

void Rasterizer::FlushStencilWritebacks() {
    for (const auto& [scratch_id, depth_id] : pending_stencil_writebacks) {
        CopyStencilPlane(scratch_id, depth_id);
    }
    pending_stencil_writebacks.clear();
}

void Rasterizer::BindTextures(const Shader::BoundStage& bound,
                              Shader::Backend::Bindings& binding) {
    const u64 dc_tex_t0 = g_draw_cost ? __rdtsc() : 0;
    const auto& stage = *bound.info;
    u32 lookup_count{};
    image_bindings.clear();
    const u32 first_image_idx = image_infos.size();
    // For loading/storing to explicit mip levels, when no native instruction support, bind an array
    // of descriptors consecutively, 1 for each mip level. The shader can index this with LOD
    // operand.
    // This array holds the size of each consecutive array with the number of bindings consumed.
    // This is currently always 1 for anything other than mip fallback arrays.
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;

    const auto t_findimage = Common::FaultMonitor::Now();

    const bool cache_decoded_images = g_cache_decoded_images;
    const bool verify_decoded_images = g_verify_decoded_images;
    for (size_t image_slot = 0; image_slot < stage.images.size(); ++image_slot) {
        const auto& image_desc = stage.images[image_slot];
        const auto tsharp = image_desc.GetSharp(bound);
        // Log-only: a hash lookup per image binding (a19 profile: 11 of 1279 CP samples).
        if (!::Common::Lean() && texture_cache.IsMeta(tsharp.Address())) {
            LOG_WARNING(Render_Vulkan, "Unexpected metadata read by a shader (texture)");
        }

        if (tsharp.Address() == 0 || tsharp.GetDataFmt() == AmdGpu::DataFormat::FormatInvalid) {
            image_bindings.emplace_back(std::piecewise_construct, std::tuple{}, std::tuple{});
            image_descriptor_array_sizes.push_back(1);
            continue;
        }

        const Shader::MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;
        const u32 num_bindings = image_desc.NumBindings(bound);

        const VideoCore::TextureCache::ImageDesc* decoded = nullptr;
        if (cache_decoded_images) {
            auto& entry = decoded_image_descs[static_cast<u32>(stage.l_stage)][image_slot];
            // These are the only ImageResource fields read by the ImageInfo and
            // ImageViewInfo constructors. Neo mode is fixed for the guest process.
            // Cache only the pure descriptor decode, before FindImage modifies its
            // view range. Image IDs, contents, bindings and barriers remain live.
            const u8 flags =
                image_desc.is_depth | (image_desc.is_array << 1) | (image_desc.is_written << 2);
            const bool hit = entry.valid && entry.flags == flags &&
                             std::memcmp(&entry.sharp, &tsharp, sizeof(tsharp)) == 0;
            if (!hit) {
                entry.sharp = tsharp;
                entry.flags = flags;
                entry.value = VideoCore::TextureCache::ImageDesc{tsharp, image_desc};
                entry.valid = true;
            }
            decoded = &entry.value;
            if (verify_decoded_images) {
                const VideoCore::TextureCache::ImageDesc reference{tsharp, image_desc};
                const auto& a = entry.value.info;
                const auto& b = reference.info;
                ASSERT(entry.value.type == reference.type &&
                       entry.value.view_info == reference.view_info &&
                       a.pixel_format == b.pixel_format && a.type == b.type &&
                       a.resources == b.resources && a.size == b.size && a.num_bits == b.num_bits &&
                       a.num_samples == b.num_samples && a.pitch == b.pitch &&
                       a.tile_mode == b.tile_mode && a.array_mode == b.array_mode &&
                       a.guest_address == b.guest_address && a.guest_size == b.guest_size &&
                       a.bank_swizzle == b.bank_swizzle && a.alt_tile == b.alt_tile &&
                       a.props.is_depth == b.props.is_depth &&
                       a.props.is_block == b.props.is_block && a.props.is_pow2 == b.props.is_pow2 &&
                       a.props.is_volume == b.props.is_volume &&
                       a.props.is_tiled == b.props.is_tiled);
                ASSERT(std::memcmp(a.mips_layout.data(), b.mips_layout.data(),
                                   a.resources.levels * sizeof(VideoCore::ImageInfo::MipInfo)) ==
                       0);
                static u64 calls{}, hits{};
                hits += hit;
                if (++calls >= 500000) {
                    LOG_INFO(Render_Vulkan, "DecodeImageVerify: calls={} hits={} ({:.1f}%)", calls,
                             hits, 100.0 * hits / calls);
                    calls = hits = 0;
                }
            }
        }

        for (auto i = 0; i < num_bindings; i++) {
            auto& [image_id, desc] =
                decoded ? image_bindings.emplace_back(VideoCore::ImageId{}, *decoded)
                        : image_bindings.emplace_back(std::piecewise_construct, std::tuple{},
                                                      std::tuple{tsharp, image_desc});

            if (mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
                ASSERT(num_bindings == 1);
                desc.view_info.range.base.level += image_desc.constant_mip_index;
                desc.view_info.range.extent.levels = 1;
            } else if (mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                desc.view_info.range.base.level += i;
                desc.view_info.range.extent.levels = 1;
            }

            image_id = texture_cache.FindImage(desc);
            ++lookup_count;
            auto* image = &texture_cache.GetImage(image_id);
            {
                // Diagnostic SHADPS4_TRACE_SHADER=<hex pgm hash>: log texture bindings of that
                // shader (first 16 binds): requested T# vs the cache image it resolved to.
                const auto& trace_hashes = g_trace_hashes;
                static u32 traced = 0;
                const u64 trace_hash = stage.pgm_hash;
                if (!trace_hashes.empty() && traced < 48 &&
                    std::ranges::find(trace_hashes, trace_hash) != trace_hashes.end()) {
                    ++traced;
                    LOG_WARNING(Render_Vulkan,
                                "TraceShader {:#x}: slot {} type {} T# {:#x} {}x{} {} -> image "
                                "{:#x} {}x{} {} samples {} uid {} flags {:#x}",
                                trace_hash, image_slot, static_cast<u32>(desc.type),
                                desc.info.guest_address, desc.info.size.width,
                                desc.info.size.height, vk::to_string(desc.info.pixel_format),
                                image->info.guest_address, image->info.size.width,
                                image->info.size.height, vk::to_string(image->info.pixel_format),
                                image->info.num_samples, image->image_uid,
                                static_cast<u32>(image->flags));
                }
            }
            if (auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
                // SHADPS4_BIND_MEMO_STATS: redirected bindings are excluded from the memo.
                bind_memo_redirected |= u64{1} << ((image_bindings.size() - 1) & 63);
                static const bool stencil_storage = [] {
                    const char* value = std::getenv("SHADPS4_STENCIL_STORAGE");
                    return value && value[0] == '1';
                }();
                if (stencil_storage &&
                    desc.type == VideoCore::TextureCache::BindingType::Storage) {
                    // The placeholder's association is whichever depth surface bound it last;
                    // prefer the depth image at that address whose extent matches this view.
                    const VAddr depth_addr = texture_cache.GetImage(depth_image_id).info.guest_address;
                    if (const auto sized = texture_cache.FindDepthImageBySize(depth_addr,
                                                                             desc.info.size)) {
                        depth_image_id = sized;
                    }
                    // Vulkan cannot bind a D32S8 stencil aspect as a storage image. Write an R8
                    // scratch seeded from the stencil plane; copied back before the next bind.
                    image_id = StencilScratchFor(depth_image_id, desc.info);
                    image = &texture_cache.GetImage(image_id);
                    const std::pair writeback{image_id, depth_image_id};
                    if (std::ranges::find(pending_stencil_writebacks, writeback) ==
                        pending_stencil_writebacks.end()) {
                        CopyStencilPlane(depth_image_id, image_id);
                        pending_stencil_writebacks.push_back(writeback);
                    }
                } else {
                    // If this image has an associated depth image, it's a stencil attachment.
                    // Redirect the access to the actual depth-stencil buffer.
                    image_id = depth_image_id;
                    image = &texture_cache.GetImage(image_id);
                }
            } else if (image_desc.is_written &&
                       desc.type == VideoCore::TextureCache::BindingType::Storage) {
                texture_cache.NoteStorageWrite(*image);
            }
            if (image->binding.is_bound) {
                // The image is already bound. In case if it is about to be used as storage we
                // need to force general layout on it.
                image->binding.force_general |= image_desc.is_written;
            }
            image->binding.is_bound = 1u;
        }

        image_descriptor_array_sizes.push_back(num_bindings);
    }

    if (Common::BufferSampleProbe::Enabled()) {
        struct LookupShape {
            u64 stages{}, nonempty{}, lookups{};
        };
        static thread_local LookupShape shape;
        ++shape.stages;
        shape.nonempty += lookup_count != 0;
        shape.lookups += lookup_count;
        if ((shape.stages & 0xfffff) == 0) {
            LOG_INFO(Render_Vulkan,
                     "TextureLookupShape: stages={} nonempty={} lookups={} (cumulative)",
                     shape.stages, shape.nonempty, shape.lookups);
        }
    }
    const auto t_view = Common::FaultMonitor::Now();
    u64 findtex_ns = 0;
    u64 transit_ns = 0;
    if (g_draw_cost) {
        g_dc_tex_sub[0] += __rdtsc() - dc_tex_t0;
    }

    // Second pass to re-bind images that were updated after binding
    for (auto& [image_id, desc] : image_bindings) {
        bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (!image_id) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
            if (native_bindings) {
                native_image_indices[image_infos.size() - 1] =
                    NativeBindings::NullImageIndex(is_storage);
            }
        } else {
            const bool is_scratch = !stencil_scratches.empty() && IsStencilScratch(image_id);
            if (auto& old_image = texture_cache.GetImage(image_id);
                !is_scratch && old_image.binding.needs_rebind) {
                bind_memo_rebound = true; // SHADPS4_TEX_BIND_MEMO: not memoized
                old_image.binding = {};
                image_id = texture_cache.FindImage(desc);
            }

            bound_images.emplace_back(image_id);

            auto& image = texture_cache.GetImage(image_id);
            const auto t_ft = Common::FaultMonitor::Now();
            const u64 dc_ft0 = g_draw_cost ? __rdtsc() : 0;
            const u32 dc_layout_before =
                g_draw_cost ? static_cast<u32>(image.backing->state.layout) : 0;
            auto& image_view = is_scratch ? image.FindView(desc.view_info)
                                          : texture_cache.FindTexture(image_id, desc);
            const u64 dc_ft1 = g_draw_cost ? __rdtsc() : 0;
            {
                // Experiment SHADPS4_NEO_IDBUF_FILL=<u32>: the PS4 Pro hardware ID buffer (a
                // 2-sample R32Uint surface written by the rasterizer) is not emulated; fill it with
                // a constant before a shader samples it, to test whether it gates the checkerboard
                // resolve. Not a fix.
                const s64 idbuf_fill = g_idbuf_fill;
                const bool idbuf_derive = g_idbuf_derive;
                if (idbuf_derive && !is_storage && image.info.num_samples == 2 &&
                    image.info.pixel_format == vk::Format::eR32Uint) {
                    DeriveNeoIdBuffer(image);
                } else if (idbuf_fill >= 0 && !is_storage && image.info.num_samples == 2 &&
                           image.info.pixel_format == vk::Format::eR32Uint) {
                    scheduler.EndRendering();
                    image.Transit(vk::ImageLayout::eTransferDstOptimal,
                                  vk::AccessFlagBits2::eTransferWrite, {});
                    const u32 fill = static_cast<u32>(idbuf_fill);
                    vk::ClearColorValue value{};
                    value.setUint32({fill, fill, fill, fill});
                    const vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0,
                                                          1};
                    auto command_access = scheduler.AcquireCommandBuffer();
                    command_access.Handle().clearColorImage(image.GetImage(),
                                                            image.backing->state.layout, value,
                                                            range);
                }
            }
            const auto t_tr = Common::FaultMonitor::Now();
            findtex_ns += Common::FaultMonitor::DiffNs(t_ft, t_tr);

            // The image is either bound as storage in a separate descriptor or bound as render
            // target in feedback loop. Depth images are excluded because they can't be bound as
            // storage and feedback loop doesn't make sense for them
            if ((image.binding.force_general || image.binding.is_target) &&
                !image.info.props.is_depth) {
                image.Transit(instance.IsAttachmentFeedbackLoopLayoutSupported() &&
                                      image.binding.is_target
                                  ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                  : vk::ImageLayout::eGeneral,
                              vk::AccessFlagBits2::eShaderRead |
                                  (image.info.props.is_depth
                                       ? vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                                       : vk::AccessFlagBits2::eColorAttachmentWrite |
                                             vk::AccessFlagBits2::eColorAttachmentRead),
                              {});
            } else {
                if (is_storage) {
                    image.Transit(vk::ImageLayout::eGeneral,
                                  vk::AccessFlagBits2::eShaderRead |
                                      vk::AccessFlagBits2::eShaderWrite,
                                  desc.view_info.range);
                } else {
                    const auto new_layout = image.info.props.is_depth
                                                ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                                : vk::ImageLayout::eShaderReadOnlyOptimal;
                    image.Transit(new_layout, vk::AccessFlagBits2::eShaderRead,
                                  desc.view_info.range);
                }
            }
            transit_ns += Common::FaultMonitor::SinceNs(t_tr);
            if (g_draw_cost) {
                const u64 dc_ft2 = __rdtsc();
                g_dc_tex_sub[1] += dc_ft1 - dc_ft0;
                g_dc_tex_sub[2] += dc_ft2 - dc_ft1;
                if (dc_ft2 - dc_ft0 > g_dc_slow_img.cyc) {
                    g_dc_slow_img = {dc_ft2 - dc_ft0,
                                     image.info.guest_address,
                                     image.info.size.width,
                                     image.info.size.height,
                                     image.info.resources.layers,
                                     static_cast<u32>(image.info.pixel_format),
                                     image.info.props.is_depth != 0,
                                     image.info.props.is_tiled != 0,
                                     dc_layout_before};
                }
            }
            image.usage.storage |= is_storage;
            image.usage.texture |= !is_storage;

            image_infos.emplace_back(VK_NULL_HANDLE, *image_view.image_view,
                                     image.backing->state.layout);
            if (native_bindings) {
                // Same view object and the layout observed at this moment, as recorded above.
                const auto& info = image_infos.back();
                native_image_indices[image_infos.size() - 1] = native_bindings->ImageIndex(
                    image_view.native, info.imageView, info.imageLayout, is_storage);
            }
        }
    }

    Common::FaultMonitor::CountBindTexView(findtex_ns, transit_ns);
    const auto t_sampler_start = Common::FaultMonitor::Now();

    u32 image_info_idx = first_image_idx;
    u32 image_binding_idx = 0;
    for (u32 array_size : image_descriptor_array_sizes) {
        const auto& [_, desc] = image_bindings[image_binding_idx];
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (native_bindings) {
            for (u32 element = 0; element < array_size; ++element) {
                native_indices[binding.unified + element] =
                    native_image_indices[image_info_idx + element];
            }
        }
        if (build_set_writes) {
            auto& set_write = set_writes[set_write_index++];
            set_write.dstSet = VK_NULL_HANDLE;
            set_write.dstBinding = binding.unified;
            set_write.dstArrayElement = 0;
            set_write.descriptorCount = array_size;
            set_write.descriptorType =
                is_storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage;
            set_write.pImageInfo = &image_infos[image_info_idx];
        }

        image_info_idx += array_size;
        image_binding_idx += array_size;
        binding.unified += array_size;
    }

    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(bound);
        if (sampler.disable_aniso) {
            const auto& tsharp = stage.images[sampler.associated_image].GetSharp(bound);
            if (tsharp.base_level == 0 && tsharp.last_level == 0) {
                ssharp.max_aniso.Assign(AmdGpu::AnisoRatio::One);
            }
        }
        u32 native_sampler_index{};
        const auto vk_sampler =
            texture_cache.GetSampler(ssharp, liverpool->BackRegs().ta_bc_base,
                                     native_bindings ? &native_sampler_index : nullptr);
        image_infos.emplace_back(vk_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        const u32 unified_binding = binding.unified++;
        if (native_bindings) {
            native_indices[unified_binding] = native_sampler_index;
        }
        if (build_set_writes) {
            auto& set_write = set_writes[set_write_index++];
            set_write.dstSet = VK_NULL_HANDLE;
            set_write.dstBinding = unified_binding;
            set_write.dstArrayElement = 0;
            set_write.descriptorCount = 1;
            set_write.descriptorType = vk::DescriptorType::eSampler;
            set_write.pImageInfo = &image_infos.back();
        }
    }

    // The descriptor-write loop between t_view's end and t_sampler_start is bookkeeping only and
    // is folded into the sampler span; if sampler time looks large with a high memo hit rate, that
    // loop is where to look next rather than GetSampler.
    Common::FaultMonitor::CountBindTexSplit(Common::FaultMonitor::DiffNs(t_findimage, t_view),
                                            Common::FaultMonitor::DiffNs(t_view, t_sampler_start),
                                            Common::FaultMonitor::SinceNs(t_sampler_start));
}

namespace {
template <typename T>
bool SameReg(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}
} // namespace

bool Rasterizer::RenderStateMemoRegsEqual(u32 num_cbs) const {
    const auto& regs = liverpool->BackRegs();
    const auto& m = rs_memo;
    return std::memcmp(m.color_buffers.data(), &regs.color_buffers[0],
                       num_cbs * sizeof(AmdGpu::ColorBuffer)) == 0 &&
           SameReg(m.color_target_mask, regs.color_target_mask) &&
           SameReg(m.color_control, regs.color_control) &&
           SameReg(m.depth_control, regs.depth_control) &&
           SameReg(m.depth_buffer, regs.depth_buffer) && SameReg(m.depth_view, regs.depth_view) &&
           SameReg(m.depth_htile_data_base, regs.depth_htile_data_base) &&
           SameReg(m.depth_render_control, regs.depth_render_control) &&
           SameReg(m.depth_clear, regs.depth_clear) && SameReg(m.stencil_clear, regs.stencil_clear);
}

bool Rasterizer::RenderStateMemoMatch(const GraphicsPipeline* pipeline, bool apply) {
    auto& m = rs_memo;
    auto& st = rs_stats;
    if (!m.valid) {
        ++st.miss_invalid;
        return false;
    }
    const auto& key = pipeline->GetGraphicsKey();
    if (key.mrt_mask != m.mrt_mask || key.color_samples != m.color_samples) {
        ++st.miss_key;
        return false;
    }
    const u32 num_cbs = std::bit_width(key.mrt_mask);
    // The extent hints come from NOP packets and can change without a register byte changing.
    for (u32 cb = 0; cb < num_cbs; ++cb) {
        if (liverpool->BackCbExtent(cb).raw != m.extents[cb]) {
            ++st.miss_regs;
            return false;
        }
    }
    if (liverpool->BackDbExtent().raw != m.extents[AmdGpu::NUM_COLOR_BUFFERS]) {
        ++st.miss_regs;
        return false;
    }
    // The context epoch only moves with SHADPS4_VIEWPORT_MEMO; an unchanged epoch means no
    // context register byte changed. Otherwise compare the registers the two functions read.
    const u64 epoch = liverpool->BackContextEpoch();
    if (g_viewport_memo >= 1 && epoch == m.ctx_epoch) {
        if (g_render_state_memo == 2 && !RenderStateMemoRegsEqual(num_cbs)) {
            if (++st.mismatches <= 16) {
                LOG_CRITICAL(Render_Vulkan,
                             "RenderStateMemo mismatch: registers changed under context epoch {}",
                             epoch);
            }
            return false;
        }
    } else {
        if (!RenderStateMemoRegsEqual(num_cbs)) {
            ++st.miss_regs;
            return false;
        }
        m.ctx_epoch = epoch;
        ++st.epoch_moved;
    }
    // Image checks and the touches a hit keeps run inside one memo section, like the texture
    // binding memo: a foreign structural change cannot interleave with them.
    const auto section = texture_cache.EnterMemoSection();
    if (section == VideoCore::TextureCache::MemoSection::Refused) {
        ++st.miss_refused;
        return false;
    }
    bool ok = true;
    if (texture_cache.StructGeneration() != m.struct_gen) {
        ++st.miss_struct;
        ok = false;
    } else if (texture_cache.MetaClearEpoch() != m.meta_epoch) {
        ++st.miss_meta;
        ok = false;
    } else {
        // An unchanged structure generation means no image was deleted: the ids are live.
        for (const auto& t : m.targets) {
            const auto& image = texture_cache.PeekImage(t.id);
            if (image.image_uid != t.uid || image.backing != t.backing) {
                ++st.miss_image;
                ok = false;
                break;
            }
        }
    }
    if (ok && apply) {
        // What PrepareRenderState leaves behind: FindImage's access tick and LRU touch, the
        // target flag and the bound_images entry (same order: colours by index, then depth).
        for (const auto& t : m.targets) {
            auto& image = texture_cache.PeekImage(t.id);
            texture_cache.TouchImageAccess(image);
            image.binding.is_target = 1u;
            bound_images.emplace_back(t.id);
        }
    }
    texture_cache.LeaveMemoSection(section);
    st.hits += ok;
    return ok;
}

bool Rasterizer::RenderStateMemoBeginOk() {
    const auto& m = rs_memo;
    // BindResources ran since the match: it can create, replace or re-back images, bind a
    // target as a texture (feedback loop) or ask for a rebind. Any of those takes the full path.
    const auto section = texture_cache.EnterMemoSection();
    if (section == VideoCore::TextureCache::MemoSection::Refused) {
        return false;
    }
    bool ok = m.valid && texture_cache.StructGeneration() == m.struct_gen &&
              texture_cache.MetaClearEpoch() == m.meta_epoch;
    if (ok) {
        for (const auto& t : m.targets) {
            const auto& image = texture_cache.PeekImage(t.id);
            if (image.image_uid != t.uid || image.backing != t.backing ||
                image.binding.needs_rebind || image.binding.is_bound ||
                image.binding.force_general) {
                ok = false;
                break;
            }
        }
    }
    texture_cache.LeaveMemoSection(section);
    return ok;
}

void Rasterizer::RenderStateMemoBegin(RenderState& state) {
    const auto& m = rs_memo;
    attachment_feedback_loop = false;
    state = m.state;
    for (const auto& t : m.targets) {
        const bool is_depth = t.cb == RenderStateMemo::DepthSlot;
        texture_cache.NoteTargetUse(t.id, is_depth);
        auto& image = texture_cache.PeekImage(t.id);
        if (!is_depth) {
            // An upload inside UpdateImage switches a re-sampled colour target to its guest
            // sample count; the full path switches back before it looks up the view, and
            // the remembered view belongs to the backing of the pipeline's sample count.
            image.SetBackingSamples(m.color_samples[t.cb]);
        }
        if (is_depth) {
            image.Transit(m.depth_layout,
                          vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                              vk::AccessFlagBits2::eDepthStencilAttachmentRead,
                          db_desc.second.view_info.range);
            state.depth_stencil_attachment.image_layout = image.backing->state.layout;
        } else {
            image.Transit(vk::ImageLayout::eColorAttachmentOptimal,
                          vk::AccessFlagBits2::eColorAttachmentWrite |
                              vk::AccessFlagBits2::eColorAttachmentRead,
                          cb_descs[t.cb].second.view_info.range);
            state.color_attachments[t.cb].image_layout = image.backing->state.layout;
        }
    }
    // After the sample-backing switches above (review 2026-10-14): see BeginRendering.
    scheduler.GetDynamicState().SetRenderScale(state.scale, state.skip != 0);
}

void Rasterizer::RenderStateMemoStore(const GraphicsPipeline* pipeline, const RenderState& state,
                                      size_t bound_begin, u64 struct_gen_at_entry) {
    auto& m = rs_memo;
    auto& st = rs_stats;
    m.valid = false;
    if (attachment_feedback_loop) {
        ++st.refused_feedback;
        return;
    }
    // A structural change during this draw (a texture binding created or replaced an image):
    // the targets were resolved against the older structure.
    const u64 struct_gen = texture_cache.StructGeneration();
    if (struct_gen != struct_gen_at_entry) {
        ++st.refused_gen;
        return;
    }
    m.targets.clear();
    for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
        const auto& [image_id, desc] = cb_descs[cb];
        if (!image_id) {
            continue;
        }
        // A cleared state must not repeat from the memo (the clear is consumed by this draw).
        if (state.color_attachments[cb].is_clear) {
            ++st.refused_clear;
            return;
        }
        const auto& image = texture_cache.PeekImage(image_id);
        if (!image.image_uid || !texture_cache.IsPerfectMatch(desc.info, image_id)) {
            ++st.refused_overlap;
            return;
        }
        m.targets.push_back({image_id, image.image_uid, image.backing, static_cast<u8>(cb)});
    }
    if (const auto image_id = db_desc.first; image_id) {
        const auto& attachment = state.depth_stencil_attachment;
        if (attachment.depth_clear || attachment.stencil_clear) {
            ++st.refused_clear;
            return;
        }
        const auto& image = texture_cache.PeekImage(image_id);
        if (image.info.num_samples == 2) {
            // Pro scene depth: BeginRendering keeps per-draw bookkeeping for it.
            ++st.refused_other;
            return;
        }
        if (!image.image_uid || !texture_cache.IsPerfectMatch(db_desc.second.info, image_id)) {
            ++st.refused_overlap;
            return;
        }
        m.targets.push_back({image_id, image.image_uid, image.backing, RenderStateMemo::DepthSlot});
    }
    // The ids PrepareRenderState resolved must be the ones BeginRendering used (it re-finds an
    // image whose binding asked for a rebind).
    if (bound_images.size() < bound_begin + m.targets.size()) {
        ++st.refused_rebind;
        return;
    }
    for (size_t i = 0; i < m.targets.size(); ++i) {
        if (bound_images[bound_begin + i] != m.targets[i].id) {
            ++st.refused_rebind;
            return;
        }
    }
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = liverpool->BackRegs();
    m.ctx_epoch = liverpool->BackContextEpoch();
    m.mrt_mask = key.mrt_mask;
    m.color_samples = key.color_samples;
    m.struct_gen = struct_gen;
    m.meta_epoch = texture_cache.MetaClearEpoch();
    std::memcpy(m.color_buffers.data(), &regs.color_buffers[0], sizeof(m.color_buffers));
    for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
        m.extents[cb] = liverpool->BackCbExtent(cb).raw;
    }
    m.extents[AmdGpu::NUM_COLOR_BUFFERS] = liverpool->BackDbExtent().raw;
    const auto copy_reg = [](auto& dst, const auto& src) {
        static_assert(sizeof(dst) == sizeof(src));
        std::memcpy(&dst, &src, sizeof(dst));
    };
    copy_reg(m.color_target_mask, regs.color_target_mask);
    copy_reg(m.color_control, regs.color_control);
    copy_reg(m.depth_control, regs.depth_control);
    copy_reg(m.depth_buffer, regs.depth_buffer);
    copy_reg(m.depth_view, regs.depth_view);
    copy_reg(m.depth_htile_data_base, regs.depth_htile_data_base);
    copy_reg(m.depth_render_control, regs.depth_render_control);
    copy_reg(m.depth_clear, regs.depth_clear);
    copy_reg(m.stencil_clear, regs.stencil_clear);
    m.state = state;
    m.depth_layout = rs_last_depth_layout;
    m.valid = true;
    rs_memo_pb_written = VideoCore::PbDraw();
}

void Rasterizer::RenderStateMemoLog() {
    const auto& s = rs_stats;
    LOG_INFO(Render_Vulkan,
             "RenderStateMemo: mode={} draws={} hits={} ({:.1f}%) begin_hits={} "
             "begin_fallbacks={} | miss invalid={} key={} regs={} refused={} struct={} meta={} "
             "image={} | store refused clear={} feedback={} overlap={} rebind={} gen={} other={} | "
             "epoch_moved={} verify_prepare={} verify_begin={} mismatches={}",
             g_render_state_memo, s.draws, s.hits, 100.0 * s.hits / std::max<u64>(s.draws, 1),
             s.begin_hits, s.begin_fallbacks, s.miss_invalid, s.miss_key, s.miss_regs,
             s.miss_refused, s.miss_struct, s.miss_meta, s.miss_image, s.refused_clear,
             s.refused_feedback, s.refused_overlap, s.refused_rebind, s.refused_gen,
             s.refused_other, s.epoch_moved, s.verify_prepare, s.verify_begin, s.mismatches);
}

namespace {
/// SHADPS4_WATER_REFLECTION_SCALE: the planar reflection's own depth - the frame's only 480x270
/// D32 target (water capture 2026-09-25: 36986).
bool WaterScaleDepth(const VideoCore::ImageInfo& d) {
    return d.props.is_depth && !d.props.has_stencil && d.pixel_format == vk::Format::eD32Sfloat &&
           d.size.width == VideoCore::WaterScale::kWidth &&
           d.size.height == VideoCore::WaterScale::kHeight && d.size.depth == 1 &&
           d.resources.levels == 1 && d.resources.layers == 1 && d.num_samples == 1;
}
/// ... and a single-level RGBA16F colour target of its size bound together with it (36982).
bool WaterScaleColour(const VideoCore::ImageInfo& c, const VideoCore::ImageInfo& d) {
    return !c.props.is_depth && c.pixel_format == vk::Format::eR16G16B16A16Sfloat &&
           c.size.width == d.size.width && c.size.height == d.size.height && c.size.depth == 1 &&
           c.resources.levels == 1 && c.resources.layers == 1 && c.num_samples == 1 &&
           c.type != AmdGpu::ImageType::Color3D;
}
/// v2b: the previous full BeginRendering rendered into a reflection target (water_scale_mode 1).
/// The reflection's blur pass follows the fog pass directly (09-25 capture: fog 28228 -> blur
/// 28238); the same blur shader also runs in the bloom chain, whose passes must stay 1x.
/// Command-processor back thread only.
bool g_ws_prev_reflection = false;
void WaterScaleLog(const char* what, const VideoCore::Image& image) {
    const auto& c = VideoCore::WaterScale::g_counters;
    LOG_INFO(Render_Vulkan,
             "WaterScale: x{} {} {:#x} {}x{} {} | switches depth={} colour={} scaled_passes={} "
             "mixed={} worker_refusals={} downloads={} copies={} clip_disabled={} "
             "scratch_switches={} skipped_blurs={}",
             VideoCore::WaterScale::g_scale, what, image.info.guest_address, image.info.size.width,
             image.info.size.height, vk::to_string(image.info.pixel_format),
             c.depth_switches.load(std::memory_order_relaxed),
             c.colour_switches.load(std::memory_order_relaxed),
             c.scaled_passes.load(std::memory_order_relaxed),
             c.mixed_passes.load(std::memory_order_relaxed),
             c.worker_refusals.load(std::memory_order_relaxed),
             c.downloads.load(std::memory_order_relaxed), c.copies.load(std::memory_order_relaxed),
             c.clip_disabled.load(std::memory_order_relaxed),
             c.scratch_switches.load(std::memory_order_relaxed),
             c.skipped_blurs.load(std::memory_order_relaxed));
}
} // namespace

RenderState Rasterizer::BeginRendering(const GraphicsPipeline* pipeline) {
    attachment_feedback_loop = false;
    const auto& regs = liverpool->BackRegs();
    const auto& key = pipeline->GetGraphicsKey();
    RenderState state{};
    state.width = instance.GetMaxFramebufferWidth();
    state.height = instance.GetMaxFramebufferHeight();
    state.num_layers = std::numeric_limits<u16>::max();
    state.num_color_attachments = std::bit_width(key.mrt_mask);
    // SHADPS4_WATER_REFLECTION_SCALE (video_core/water_scale.h): the reflection's depth gets its
    // scaled backing here, before any view of it is looked up; a qualifying colour target bound
    // with it gets one in the loop below. The switches are sticky (the images' only users are
    // the reflection chain); everything after reads the current backing.
    const u32 ws = VideoCore::WaterScale::g_scale;
    const bool ws_active = VideoCore::WaterScale::g_active;
    const VideoCore::Image* ws_depth = nullptr;
    if (ws_active && db_desc.first) [[unlikely]] {
        auto& dimg = texture_cache.GetImage(db_desc.first);
        // All or nothing (review 2026-10-14 risk 4): every bound colour target must qualify too.
        bool colours_ok = true;
        for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
            if (const auto id = cb_descs[cb].first;
                id && !WaterScaleColour(texture_cache.GetImage(id).info, dimg.info)) {
                colours_ok = false;
            }
        }
        if (WaterScaleDepth(dimg.info) && colours_ok) {
            ws_depth = &dimg;
            if (dimg.BackingScale() != ws) {
                dimg.SetBackingScale(ws);
                VideoCore::WaterScale::Publish(dimg.info.guest_address);
                VideoCore::WaterScale::g_counters.depth_switches.fetch_add(
                    1, std::memory_order_relaxed);
                WaterScaleLog("depth", dimg);
            }
        }
    }
    u32 ws_pass_scale = 0;
    bool ws_mixed = false;
    const bool ws_after_reflection = g_ws_prev_reflection;
    bool ws_this_reflection = false;
    const auto ws_note = [&](u32 sc) {
        if (ws_pass_scale == 0) {
            ws_pass_scale = sc;
        } else if (ws_pass_scale != sc) {
            ws_mixed = true;
        }
    };
    for (auto cb = 0u; cb < state.num_color_attachments; ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        if (!image_id) {
            state.color_attachments[cb] = {};
            continue;
        }
        auto* image = &texture_cache.GetImage(image_id);
        if (image->binding.needs_rebind) {
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            image = &texture_cache.GetImage(image_id);
        }
        if (ws_depth && image->water_scale_mode != 1 &&
            WaterScaleColour(image->info, ws_depth->info)) [[unlikely]] {
            image->SetBackingScale(ws);
            image->water_scale_mode = 1;
            VideoCore::WaterScale::Publish(image->info.guest_address);
            VideoCore::WaterScale::g_counters.colour_switches.fetch_add(1,
                                                                        std::memory_order_relaxed);
            WaterScaleLog("colour", *image);
        } else if (ws > 1 && image->water_scale_mode != 1) [[unlikely]] {
            // v2: the blur's scratch target follows the pass (scaled under the blur shader).
            const auto* fs = pipeline->GetStages()[int(Shader::LogicalStage::Fragment)];
            const bool blur = ws_after_reflection && fs &&
                              fs->pgm_hash == VideoCore::WaterScale::kBlurFs &&
                              image->info.size.width == VideoCore::WaterScale::kWidth &&
                              image->info.size.height == VideoCore::WaterScale::kHeight &&
                              WaterScaleColour(image->info, image->info);
            const u32 want = blur ? ws : 1u;
            if (image->BackingScale() != want) {
                image->SetBackingScale(want);
                if (blur && image->water_scale_mode == 0) {
                    image->water_scale_mode = 2;
                    VideoCore::WaterScale::Publish(image->info.guest_address);
                    WaterScaleLog("blur scratch", *image);
                }
                VideoCore::WaterScale::g_counters.scratch_switches.fetch_add(
                    1, std::memory_order_relaxed);
            }
        }
        texture_cache.UpdateImage(image_id);
        image->SetBackingSamples(key.color_samples[cb]);
        const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
        const auto slice = image_view.info.range.base.layer;
        const auto mip = image_view.info.range.base.level;

        const auto& col_buf = regs.color_buffers[cb];
        const bool is_clear = texture_cache.IsMetaCleared(col_buf.CmaskAddress(), slice);
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);

        if (image->binding.is_bound) {
            ASSERT_MSG(!image->binding.force_general,
                       "Having image both as storage and render target is unsupported");
            image->Transit(instance.IsAttachmentFeedbackLoopLayoutSupported()
                               ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                               : vk::ImageLayout::eGeneral,
                           vk::AccessFlagBits2::eColorAttachmentWrite, {});
            attachment_feedback_loop = true;
        } else {
            image->Transit(vk::ImageLayout::eColorAttachmentOptimal,
                           vk::AccessFlagBits2::eColorAttachmentWrite |
                               vk::AccessFlagBits2::eColorAttachmentRead,
                           desc.view_info.range);
        }

        // The render area is the current backing's (scaled by SHADPS4_WATER_REFLECTION_SCALE).
        const u32 cb_scale = image->BackingScale();
        ws_this_reflection |= image->water_scale_mode == 1;
        ws_note(cb_scale);
        state.width =
            std::min<u32>(state.width, std::max((image->info.size.width * cb_scale) >> mip, 1u));
        state.height =
            std::min<u32>(state.height, std::max((image->info.size.height * cb_scale) >> mip, 1u));
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        const auto clear_value =
            is_clear ? LiverpoolToVK::ColorBufferClearValue(col_buf) : vk::ClearValue{};
        auto& attachment = state.color_attachments[cb];
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image->backing->state.layout;
        attachment.clear_value = clear_value.color.uint32;
        attachment.is_clear = is_clear;

        image->usage.render_target = 1u;
    }
    for (u32 cb = state.num_color_attachments; cb < state.color_attachments.size(); ++cb) {
        state.color_attachments[cb] = {};
    }
    // 1.1.1 (user 10-22): black bars beside a fitted movie (MemoryPatcher::ApplyMovieFit). On the
    // PS4 the movie quad covered its target whole, so beside the smaller quad the target kept the
    // last game frame. The full-screen movie draw (vs b94242a8 + the NV12 ps b03b3012, s1597)
    // whose first vertex is the fitted corner clears its target to black first; menu videos use
    // the same shaders at other positions and are left alone. A cleared state never comes from
    // the render-state memo (RenderStateMemoStore refuses it), so every movie frame passes here.
    if (float fx, fy; state.num_color_attachments >= 1 && state.color_attachments[0].image_view &&
                      MemoryPatcher::MovieFitRect(fx, fy)) [[unlikely]] {
        const auto& stages = pipeline->GetStages();
        const auto* vs = stages[u32(Shader::LogicalStage::Vertex)];
        const auto* fs = stages[u32(Shader::LogicalStage::Fragment)];
        if (vs && fs && vs->pgm_hash == 0xb94242a8 && fs->pgm_hash == 0xb03b3012 &&
            !vs->buffers.empty()) {
            const AmdGpu::Buffer quad =
                vs->buffers[0].GetSharp(GetBoundStage(Shader::LogicalStage::Vertex));
            std::array<float, 16> pos{}; // 4 x vec4: vertex 0 = (right, top) ... 3 = (left, bottom)
            if (quad.GetSize() >= sizeof(pos) &&
                memory->TryReadBacking(quad.base_address, reinterpret_cast<u8*>(pos.data()),
                                       sizeof(pos)) &&
                std::abs(pos[0] - fx) < 1e-4f && std::abs(pos[1] - fy) < 1e-4f &&
                std::abs(pos[12] + fx) < 1e-4f && std::abs(pos[13] + fy) < 1e-4f) {
                auto& attachment = state.color_attachments[0];
                attachment.is_clear = true;
                attachment.clear_value = {0u, 0u, 0u, std::bit_cast<u32>(1.0f)};
                static std::atomic<bool> logged{false};
                if (!logged.exchange(true, std::memory_order_relaxed)) {
                    LOG_INFO(Render_Vulkan, "Movie fit: black bars - the movie's target is cleared "
                                            "before its draw (quad corner {:.3f}, {:.3f})",
                             fx, fy);
                }
            }
        }
    }

    if (auto image_id = db_desc.first; image_id) {
        auto& desc = db_desc.second;
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto t_fd = Common::FaultMonitor::Now();
        const auto& image_view = texture_cache.FindDepthTarget(image_id, desc);
        Common::FaultMonitor::CountFindDepthTarget(Common::FaultMonitor::SinceNs(t_fd));
        auto& image = texture_cache.GetImage(image_id);
        if (image.info.num_samples == 2) {
            // Pro scene depth (2 samples): source for SHADPS4_NEO_IDBUF_DERIVE, with the sample
            // locations active while it is rasterized (PA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y0_0).
            neo_depth_addr = image.info.guest_address;
            neo_depth_size = image.info.size;
            neo_depth_x0y0 = regs.reg_array[0xA2FE];
        }

        const auto slice = image_view.info.range.base.layer;
        const bool is_depth_clear =
            (regs.depth_render_control.depth_clear_enable && regs.depth_control.depth_enable &&
             regs.depth_control.depth_write_enable) ||
            texture_cache.IsMetaCleared(htile_address, slice);
        const bool is_stencil_clear = regs.depth_render_control.stencil_clear_enable;
        texture_cache.TouchMeta(htile_address, slice, false);
        ASSERT(desc.view_info.range.extent.levels == 1 && !image.binding.needs_rebind);

        const bool has_stencil = image.info.props.has_stencil;
        // Stencil writes can be enabled while depth writes are off.
        const bool stencil_write =
            has_stencil && regs.depth_control.stencil_enable && !desc.view_info.is_storage;
        const auto new_layout = desc.view_info.is_storage
                                    ? has_stencil ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                                                  : vk::ImageLayout::eDepthAttachmentOptimal
                                : stencil_write
                                    ? vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal
                                : has_stencil ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                              : vk::ImageLayout::eDepthReadOnlyOptimal;
        rs_last_depth_layout = new_layout;
        image.Transit(new_layout,
                      vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                          vk::AccessFlagBits2::eDepthStencilAttachmentRead,
                      desc.view_info.range);

        const u32 db_scale = image.BackingScale();
        ws_note(db_scale);
        state.width = std::min<u32>(state.width, image.info.size.width * db_scale);
        state.height = std::min<u32>(state.height, image.info.size.height * db_scale);
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        auto& attachment = state.depth_stencil_attachment;
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image.backing->state.layout;
        attachment.clear_value = {};

        if (regs.depth_buffer.DepthValid()) {
            attachment.clear_value[0] = is_depth_clear ? std::bit_cast<u32>(regs.depth_clear) : 0u;
            attachment.has_depth = true;
            attachment.depth_clear = is_depth_clear;
        }
        if (regs.depth_buffer.StencilValid()) {
            attachment.clear_value[1] = is_stencil_clear ? regs.stencil_clear : 0u;
            attachment.has_stencil = true;
            attachment.stencil_clear = is_stencil_clear;
        } else if (regs.depth_buffer.DepthValid() &&
                   LiverpoolToVK::PromoteDepthToStencil(regs.depth_buffer.z_info.format,
                                                        regs.depth_buffer.stencil_info.format)) {
            // Promoted D32S8: attach the stencil aspect (loaded, never cleared) so the attachment
            // matches the pipeline's stencil format.
            attachment.has_stencil = true;
            attachment.stencil_clear = false;
        }

        image.usage.depth_target = true;
    } else {
        state.depth_stencil_attachment = {};
    }

    if (state.num_layers == std::numeric_limits<u16>::max()) {
        state.num_layers = 1;
    }
    if (ws_active && VideoCore::WaterScale::g_no_blur) [[unlikely]] {
        // The reflection's own blur passes: the blur shader right after a reflection pass (blur
        // H into the scratch) or into the reflection itself (blur V).
        const auto* fs = pipeline->GetStages()[int(Shader::LogicalStage::Fragment)];
        if (fs && fs->pgm_hash == VideoCore::WaterScale::kBlurFs &&
            (ws_after_reflection || ws_this_reflection)) {
            state.skip = 1;
            VideoCore::WaterScale::g_counters.skipped_blurs.fetch_add(1,
                                                                      std::memory_order_relaxed);
        }
    }
    if (ws_pass_scale > 1 || ws_mixed) [[unlikely]] {
        // A pass on scaled backings: its viewports / scissors are scaled at commit. Mixed scales
        // (never expected) keep 1x: the render area is the smallest attachment's.
        auto& c = VideoCore::WaterScale::g_counters;
        if (ws_mixed) {
            c.mixed_passes.fetch_add(1, std::memory_order_relaxed);
        } else {
            state.scale = static_cast<u16>(ws_pass_scale);
            const u64 n = c.scaled_passes.fetch_add(1, std::memory_order_relaxed) + 1;
            if ((n & 0x3fff) == 0 && db_desc.first) {
                WaterScaleLog("periodic", texture_cache.GetImage(db_desc.first));
            }
        }
    }
    // The draw's dynamic state is committed before Scheduler::BeginRendering runs: the pass's
    // scale must be in place now (review 2026-10-14 BUG 1; a no-op while the flag is off).
    scheduler.GetDynamicState().SetRenderScale(state.scale, state.skip != 0);
    g_ws_prev_reflection = ws_this_reflection;

    // Diagnostic: a render pass with no usable attachment at all.
    // A null color attachment on its own is normal - MRT gaps, or the guest disabling colour
    // writes - but a pass where every colour attachment is null AND there is no depth/stencil
    // attachment renders into nothing. That is legal Vulkan and raises no validation error, so
    // it is invisible unless we say so here. Deduplicated by signature; each distinct case is
    // reported once per run.
    {
        bool any_color = false;
        for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
            if (state.color_attachments[cb].image_view) {
                any_color = true;
                break;
            }
        }
        if (!any_color && !state.depth_stencil_attachment.image_view) {
            u64 sig = u64(key.mrt_mask) << 1;
            for (const auto h : key.stage_hashes) {
                sig = sig * 0x100000001b3ULL ^ u64(h);
            }
            static std::unordered_set<u64> reported;
            if (reported.insert(sig).second) {
                const bool cb_disabled =
                    regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
                std::string detail;
                for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
                    const auto& col_buf = regs.color_buffers[cb];
                    detail += fmt::format(
                        " [cb{} valid={} addr={:#x} dfmt={} nfmt={} write_mask={:#x} in_mrt={}]",
                        cb, bool(col_buf), col_buf.Address(), u32(col_buf.GetDataFmt()),
                        u32(col_buf.GetNumberFmt()), regs.color_target_mask.GetMask(cb),
                        (key.mrt_mask >> cb) & 1);
                }
                std::string stages;
                for (u32 i = 0; i < key.stage_hashes.size(); ++i) {
                    if (key.stage_hashes[i] != 0) {
                        stages += fmt::format(" stage{}={:#x}", i, key.stage_hashes[i]);
                    }
                }
                // NOTE: this is usually the guest's intent, not a fault - Second Son issues
                // these with color_writes_disabled=true and write_mask=0. Reported so the case
                // stays visible, but do NOT read it as an error on its own. See
                // notes/renderdoc-capture-01-timbers.md.
                LOG_WARNING(Render_Vulkan,
                            "Attachment-less draw (usually guest intent, see notes): "
                            "mrt_mask={:#x} "
                            "color_writes_disabled={} depth_en={} depth_valid={} stencil_en={} "
                            "stencil_valid={}{} shaders:{}",
                            key.mrt_mask, cb_disabled, bool(regs.depth_control.depth_enable),
                            regs.depth_buffer.DepthValid(), bool(regs.depth_control.stencil_enable),
                            regs.depth_buffer.StencilValid(), detail, stages);
            }
        }
    }

    return state;
}

void Rasterizer::Resolve() {
    const auto& mrt0_hint = liverpool->BackCbExtent(0);
    const auto& mrt1_hint = liverpool->BackCbExtent(1);
    VideoCore::TextureCache::ImageDesc mrt0_desc{liverpool->BackRegs().color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{liverpool->BackRegs().color_buffers[1], mrt1_hint};
    auto& mrt0_image = texture_cache.GetImage(texture_cache.FindImage(mrt0_desc, true));
    auto& mrt1_image = texture_cache.GetImage(texture_cache.FindImage(mrt1_desc, true));

    ScopeMarkerBegin(fmt::format("Resolve:MRT0={:#x}:MRT1={:#x}",
                                 liverpool->BackRegs().color_buffers[0].Address(),
                                 liverpool->BackRegs().color_buffers[1].Address()));
    mrt1_image.Resolve(mrt0_image, mrt0_desc.view_info.range, mrt1_desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::DepthStencilCopy(bool is_depth, bool is_stencil) {
    auto& regs = liverpool->BackRegs();

    auto read_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->BackDbExtent(), false);
    auto write_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->BackDbExtent(), true);

    auto& read_image = texture_cache.GetImage(texture_cache.FindImage(read_desc));
    auto& write_image = texture_cache.GetImage(texture_cache.FindImage(write_desc));

    VideoCore::SubresourceRange sub_range;
    sub_range.base.layer = liverpool->BackRegs().depth_view.slice_start;
    sub_range.extent.layers = liverpool->BackRegs().depth_view.NumSlices() - sub_range.base.layer;

    ScopeMarkerBegin(fmt::format(
        "DepthStencilCopy:DR={:#x}:SR={:#x}:DW={:#x}:SW={:#x}", regs.depth_buffer.DepthAddress(),
        regs.depth_buffer.StencilAddress(), regs.depth_buffer.DepthWriteAddress(),
        regs.depth_buffer.StencilWriteAddress()));

    read_image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                       sub_range);
    write_image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
                        sub_range);

    auto aspect_mask = vk::ImageAspectFlags(0);
    if (is_depth) {
        aspect_mask |= vk::ImageAspectFlagBits::eDepth;
    }
    if (is_stencil) {
        aspect_mask |= vk::ImageAspectFlagBits::eStencil;
    }

    vk::ImageCopy region = {
        .srcSubresource =
            {
                .aspectMask = aspect_mask,
                .mipLevel = 0,
                .baseArrayLayer = sub_range.base.layer,
                .layerCount = sub_range.extent.layers,
            },
        .srcOffset = {0, 0, 0},
        .dstSubresource =
            {
                .aspectMask = aspect_mask,
                .mipLevel = 0,
                .baseArrayLayer = sub_range.base.layer,
                .layerCount = sub_range.extent.layers,
            },
        .dstOffset = {0, 0, 0},
        .extent = {write_image.info.size.width, write_image.info.size.height, 1},
    };
    scheduler.AcquireCommandBuffer().Handle().copyImage(
        read_image.GetImage(), Vulkan::GuestLayout(vk::ImageLayout::eTransferSrcOptimal),
        write_image.GetImage(), Vulkan::GuestLayout(vk::ImageLayout::eTransferDstOptimal), region);

    ScopeMarkerEnd();
}

void Rasterizer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    if (!is_gds) {
        AmdGpu::SplitHazard::NoteWrite(address, num_bytes);
    }
    buffer_cache.FillBuffer(address, num_bytes, value, is_gds);
}

void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    if (!dst_gds) {
        AmdGpu::SplitHazard::NoteWrite(dst, num_bytes);
    }
    buffer_cache.CopyBuffer(dst, src, num_bytes, dst_gds, src_gds);
}

u32 Rasterizer::ReadDataFromGds(u32 gds_offset) {
    auto* gds_buf = buffer_cache.GetGdsBuffer();
    u32 value;
    std::memcpy(&value, gds_buf->mapped_data.data() + gds_offset, sizeof(u32));
    return value;
}

bool Rasterizer::InvalidateMemory(VAddr addr, u64 size, VideoCore::InvalidateOrigin origin) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    if (AmdGpu::t_lean_worker) [[unlikely]] {
        AmdGpu::LeanNoteWorkerFault(addr); // counted; completes as a game thread's would
    }
    if (AmdGpu::t_front_in_lock) [[unlikely]] {
        if (AmdGpu::g_lean_l2.front_faults_in_lock.fetch_add(1, std::memory_order_relaxed) == 0) {
            LOG_CRITICAL(Render, "Lean L2: FRONT fault under the pipeline cache lock at {:#x}", addr);
        }
    }
    // CP split run-ahead: a fault on the front must not download or invalidate ahead of the
    // draws it has posted. No lock is held here. No-op on every other thread.
    liverpool->DrainForFrontFault(addr, size, false);
    buffer_cache.InvalidateMemory(addr, size, origin);
    texture_cache.InvalidateMemory(addr, size);
    return true;
}

bool Rasterizer::ReadMemory(VAddr addr, u64 size) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    // CP split run-ahead: see InvalidateMemory. SPLIT-TODO: a drain here waits for posted draws
    // the way today's guest faults wait for the CP's current packet (SendCommand<true>), so it
    // should add no new lock cycle - inferred, not checked at run time.
    if (AmdGpu::t_lean_worker) [[unlikely]] {
        // SHADPS4_LEAN fallback (the pre-check missed a racing protect): counted (must read 0);
        // the worker holds no cache lock here (its guest reads run outside the shared lock), so
        // completing as a game thread's fault cannot deadlock; its span stops before emitting.
        AmdGpu::LeanNoteWorkerFault(addr);
    }
    if (AmdGpu::t_front_in_lock) [[unlikely]] {
        if (AmdGpu::g_lean_l2.front_faults_in_lock.fetch_add(1, std::memory_order_relaxed) == 0) {
            LOG_CRITICAL(Render, "Lean L2: FRONT fault under the pipeline cache lock at {:#x}", addr);
        }
    }
    liverpool->DrainForFrontFault(addr, size, true);
    buffer_cache.ReadMemory(addr, size);
    return true;
}

void Rasterizer::ProcessDownloadImages() {
    texture_cache.ProcessDownloadImages();
}

bool Rasterizer::IsMapped(VAddr addr, u64 size) {
    if (size == 0) {
        // There is no memory, so not mapped.
        return false;
    }
    if (static_cast<u64>(addr) > std::numeric_limits<u64>::max() - size) {
        // Memory range wrapped the address space, cannot be mapped.
        return false;
    }
    const auto range = boost::icl::interval_set<VAddr>::interval_type::right_open(addr, addr + size);

    Common::RecursiveSharedLock lock{mapped_ranges_mutex};
    return boost::icl::contains(mapped_ranges, range);
}

bool Rasterizer::MappedInterval(VAddr addr, VAddr& lo, VAddr& hi) {
    Common::RecursiveSharedLock lock{mapped_ranges_mutex};
    const auto it = mapped_ranges.find(addr);
    if (it == mapped_ranges.end()) {
        return false;
    }
    lo = it->lower();
    hi = it->upper();
    return true;
}

void Rasterizer::MapMemory(VAddr addr, u64 size) {
    // PERF-PLAN 11 (2026-10-20): the GPU page table (BDA_PAGETABLE_SIZE, 512 MB of video memory)
    // covers 40 bits; how high does the game map GPU memory? Each new highest end is logged.
    static std::atomic<VAddr> highest_end{0};
    if (const VAddr end = addr + size; end > highest_end.load(std::memory_order_relaxed)) {
        highest_end.store(end, std::memory_order_relaxed);
        LOG_INFO(Render_Vulkan, "GPU map: highest mapped end now {:#x} ({} GB; map {:#x} + {:#x})",
                 end, end >> 30, addr, size);
    }
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges += boost::icl::interval_set<VAddr>::interval_type::right_open(addr, addr + size);
    }
    mapped_gen.fetch_add(1, std::memory_order_acq_rel);
    page_manager.OnGpuMap(addr, size);
    buffer_cache.EndGpuMap(addr, size);
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size) {
    // Before anything is torn down (a worker's cached interval stops validating), and again after.
    mapped_gen.fetch_add(1, std::memory_order_acq_rel);
    buffer_cache.BeginGpuUnmap(addr, size);
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.UnmapMemory(addr, size);
    page_manager.OnGpuUnmap(addr, size);
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges -= boost::icl::interval_set<VAddr>::interval_type::right_open(addr, addr + size);
    }
    mapped_gen.fetch_add(1, std::memory_order_acq_rel);
    // SHADPS4_BIND_MEMO_STATS: BeginGpuUnmap bumped before the unmap; bump again after it, so a
    // record taken while it was in progress cannot validate against the final mapping.
    buffer_cache.BumpStructGeneration();
}

void Rasterizer::UpdateDynamicState(const GraphicsPipeline* pipeline, const bool is_indexed,
                                    const bool commit) const {
    // SHADPS4_VIEWPORT_MEMO=1: viewports and scissors are a pure function of context registers
    // (and device limits), so skip recomputing them while no SET_CONTEXT_REG changed a byte.
    // =2 recomputes anyway and checks the tracked values did not change.
    const int viewport_memo = g_viewport_memo;
    const u64 epoch = liverpool->BackContextEpoch();
    if (viewport_memo == 1 && epoch == viewport_epoch) {
        ++viewport_memo_skips;
    } else if (viewport_memo == 2 && epoch == viewport_epoch) {
        const auto& ds = scheduler.GetDynamicState();
        const auto viewports = ds.viewports;
        const auto scissors = ds.scissors;
        UpdateViewportScissorState();
        const auto& after = scheduler.GetDynamicState();
        const bool same =
            viewports.size() == after.viewports.size() && scissors.size() == after.scissors.size() &&
            std::memcmp(viewports.data(), after.viewports.data(),
                        viewports.size() * sizeof(viewports[0])) == 0 &&
            std::memcmp(scissors.data(), after.scissors.data(),
                        scissors.size() * sizeof(scissors[0])) == 0;
        ++viewport_memo_skips;
        if (!same) {
            ++viewport_memo_mismatches;
        }
        if ((viewport_memo_skips & 0xfffff) == 0) {
            LOG_INFO(Render_Vulkan, "VIEWPORT_MEMO verify: skippable {} mismatches {}",
                     viewport_memo_skips, viewport_memo_mismatches);
        }
    } else {
        UpdateViewportScissorState();
        viewport_epoch = epoch;
    }
    UpdateDepthStencilState();

    // Bounded read-only diagnostic: distinguish PS4 test/operation stencil values
    // before selecting any correction for the captured hair composite failure.
    const bool hair_stencil_trace = g_hair_stencil_trace;
    if (hair_stencil_trace) {
        const auto* fs = pipeline->GetStages()[int(Shader::LogicalStage::Fragment)];
        if (fs && (fs->pgm_hash == 0xe59a987f || fs->pgm_hash == 0x1540dfa9 ||
                   fs->pgm_hash == 0x86bb04f5 || fs->pgm_hash == 0x6f5258e8)) {
            static std::unordered_set<std::string> seen;
            const auto& regs = liverpool->BackRegs();
            const auto state = fmt::format(
                "fs={:#x} control={:#x} ops={:#x} front={:#x} back={:#x} shader={:#x} "
                "render={:#x} stencil_valid={} depth_addr={:#x}",
                fs->pgm_hash, std::bit_cast<u32>(regs.depth_control),
                std::bit_cast<u32>(regs.stencil_control), std::bit_cast<u32>(regs.stencil_ref_front),
                std::bit_cast<u32>(regs.stencil_ref_back),
                std::bit_cast<u32>(regs.depth_shader_control),
                std::bit_cast<u32>(regs.depth_render_control), regs.depth_buffer.StencilValid(),
                regs.depth_buffer.DepthAddress());
            if (seen.size() < 128 && seen.insert(state).second) {
                LOG_INFO(Render_Vulkan, "HairStencil: {}", state);
            }
        }
    }
    UpdatePrimitiveState(is_indexed);
    UpdateRasterizationState();
    UpdateColorBlendingState(pipeline);

    if (commit) {
        scheduler.CommitDynamicState();
    }
}

void Rasterizer::UpdateViewportScissorState() const {
    const auto& regs = liverpool->BackRegs();

    const auto combined_scissor_value_tl = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::max({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const auto combined_scissor_value_br = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::min({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const bool enable_offset = !regs.window_scissor.window_offset_disable;

    AmdGpu::Scissor scsr{};
    scsr.top_left_x = combined_scissor_value_tl(
        regs.screen_scissor.top_left_x, s16(regs.window_scissor.top_left_x),
        s16(regs.generic_scissor.top_left_x),
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.top_left_y = combined_scissor_value_tl(
        regs.screen_scissor.top_left_y, s16(regs.window_scissor.top_left_y),
        s16(regs.generic_scissor.top_left_y),
        enable_offset ? regs.window_offset.window_y_offset : 0);
    scsr.bottom_right_x = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_x, regs.window_scissor.bottom_right_x,
        regs.generic_scissor.bottom_right_x,
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.bottom_right_y = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_y, regs.window_scissor.bottom_right_y,
        regs.generic_scissor.bottom_right_y,
        enable_offset ? regs.window_offset.window_y_offset : 0);

    boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS> viewports;
    boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS> scissors;

    if (regs.polygon_control.enable_window_offset &&
        (regs.window_offset.window_x_offset != 0 || regs.window_offset.window_y_offset != 0)) {
        LOG_ERROR(Render_Vulkan,
                  "PA_SU_SC_MODE_CNTL.VTX_WINDOW_OFFSET_ENABLE support is not yet implemented.");
    }

    const auto& vp_ctl = regs.viewport_control;
    for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS; i++) {
        const auto& vp = regs.viewports[i];
        const auto& vp_d = regs.viewport_depths[i];
        if (vp.xscale == 0) {
            continue;
        }

        const auto zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
        const auto zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;

        vk::Viewport viewport{};

        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_pipeline_graphics.c#L688-689
        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_cmd_buffer.c#L3103-3109
        // When the clip space is ranged [-1...1], the zoffset is centered.
        // By reversing the above viewport calculations, we get the following:
        if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
            viewport.minDepth = zoffset - zscale;
            viewport.maxDepth = zoffset + zscale;
        } else {
            viewport.minDepth = zoffset;
            viewport.maxDepth = zoffset + zscale;
        }

        if (!instance.IsDepthRangeUnrestrictedSupported()) {
            // Unrestricted depth range not supported by device. Restrict to valid range.
            viewport.minDepth = std::max(viewport.minDepth, 0.f);
            viewport.maxDepth = std::min(viewport.maxDepth, 1.f);
        }

        if (regs.IsClipDisabled()) {
            // In case if clipping is disabled we patch the shader to convert vertex position
            // from screen space coordinates to NDC by defining a render space as full hardware
            // window range [0..16383, 0..16383] and setting the viewport to its size.
            viewport.x = 0.f;
            viewport.y = 0.f;
            viewport.width = float(std::min<u32>(instance.GetMaxViewportWidth(), 16_KB));
            viewport.height = float(std::min<u32>(instance.GetMaxViewportHeight(), 16_KB));
        } else {
            const auto xoffset = vp_ctl.xoffset_enable ? vp.xoffset : 0.f;
            const auto xscale = vp_ctl.xscale_enable ? vp.xscale : 1.f;
            const auto yoffset = vp_ctl.yoffset_enable ? vp.yoffset : 0.f;
            const auto yscale = vp_ctl.yscale_enable ? vp.yscale : 1.f;

            viewport.x = xoffset - xscale;
            viewport.y = yoffset - yscale;
            viewport.width = xscale * 2.0f;
            viewport.height = yscale * 2.0f;
        }

        viewports.push_back(viewport);

        auto vp_scsr = scsr;
        if (regs.mode_control.vport_scissor_enable) {
            vp_scsr.top_left_x =
                std::max(vp_scsr.top_left_x, s16(regs.viewport_scissors[i].top_left_x));
            vp_scsr.top_left_y =
                std::max(vp_scsr.top_left_y, s16(regs.viewport_scissors[i].top_left_y));
            vp_scsr.bottom_right_x = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_x),
                                              regs.viewport_scissors[i].bottom_right_x);
            vp_scsr.bottom_right_y = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_y),
                                              regs.viewport_scissors[i].bottom_right_y);
        }
        scissors.push_back({
            .offset = {vp_scsr.top_left_x, vp_scsr.top_left_y},
            .extent = {vp_scsr.GetWidth(), vp_scsr.GetHeight()},
        });
    }

    if (viewports.empty()) {
        // Vulkan requires providing at least one viewport.
        constexpr vk::Viewport empty_viewport = {
            .x = -1.0f,
            .y = -1.0f,
            .width = 1.0f,
            .height = 1.0f,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        constexpr vk::Rect2D empty_scissor = {
            .offset = {0, 0},
            .extent = {1, 1},
        };
        viewports.push_back(empty_viewport);
        scissors.push_back(empty_scissor);
    }

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);
}

void Rasterizer::UpdateDepthStencilState() const {
    const auto& regs = liverpool->BackRegs();
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto depth_test_enabled =
        regs.depth_control.depth_enable && regs.depth_buffer.DepthValid();
    dynamic_state.SetDepthTestEnabled(depth_test_enabled);
    if (depth_test_enabled) {
        dynamic_state.SetDepthWriteEnabled(regs.depth_control.depth_write_enable &&
                                           !regs.depth_render_control.depth_clear_enable);
        dynamic_state.SetDepthCompareOp(LiverpoolToVK::CompareOp(regs.depth_control.depth_func));
    }

    const auto depth_bounds_test_enabled = regs.depth_control.depth_bounds_enable;
    dynamic_state.SetDepthBoundsTestEnabled(depth_bounds_test_enabled);
    if (depth_bounds_test_enabled) {
        dynamic_state.SetDepthBounds(regs.depth_bounds_min, regs.depth_bounds_max);
    }

    const auto depth_bias_enabled = regs.polygon_control.NeedsBias();
    dynamic_state.SetDepthBiasEnabled(depth_bias_enabled);
    if (depth_bias_enabled) {
        const bool front = regs.polygon_control.enable_polygon_offset_front;
        dynamic_state.SetDepthBias(
            front ? regs.poly_offset.front_offset : regs.poly_offset.back_offset,
            regs.poly_offset.depth_bias,
            (front ? regs.poly_offset.front_scale : regs.poly_offset.back_scale) / 16.f);
    }

    const auto stencil_test_enabled =
        regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid();
    dynamic_state.SetStencilTestEnabled(stencil_test_enabled);
    if (stencil_test_enabled) {
        const StencilOps front_ops{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_front),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_front),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_front),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_ref_func),
        };
        const StencilOps back_ops = regs.depth_control.backface_enable ? StencilOps{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_back),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_back),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_back),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_bf_func),
        } : front_ops;
        dynamic_state.SetStencilOps(front_ops, back_ops);

        const bool stencil_clear = regs.depth_render_control.stencil_clear_enable;
        const auto front = regs.stencil_ref_front;
        const auto back =
            regs.depth_control.backface_enable ? regs.stencil_ref_back : regs.stencil_ref_front;
        const bool replace_op_reference = g_replace_op_reference;
        const auto reference = [&](const AmdGpu::StencilRefMask& face, AmdGpu::CompareFunc compare,
                                   AmdGpu::StencilFunc fail, AmdGpu::StencilFunc pass,
                                   AmdGpu::StencilFunc depth_fail) -> u32 {
            // PS4 ReplaceOp writes stencil_op_val, independently of stencil_test_val.
            // Vulkan Replace uses one reference for both. With an unconditional test
            // and Keep for the other outcomes, changing that reference is equivalent.
            // Other combinations and shader-exported references need separate handling.
            if (replace_op_reference && compare == AmdGpu::CompareFunc::Always &&
                fail == AmdGpu::StencilFunc::Keep && pass == AmdGpu::StencilFunc::ReplaceOp &&
                depth_fail == AmdGpu::StencilFunc::Keep &&
                !regs.depth_shader_control.stencil_test_val_export_enable &&
                !regs.depth_shader_control.stencil_op_val_export_enable) {
                return face.stencil_op_val;
            }
            return face.stencil_test_val;
        };
        const u32 front_reference =
            reference(front, regs.depth_control.stencil_ref_func,
                      regs.stencil_control.stencil_fail_front, regs.stencil_control.stencil_zpass_front,
                      regs.stencil_control.stencil_zfail_front);
        const u32 back_reference = regs.depth_control.backface_enable
                                      ? reference(back, regs.depth_control.stencil_bf_func,
                                                  regs.stencil_control.stencil_fail_back,
                                                  regs.stencil_control.stencil_zpass_back,
                                                  regs.stencil_control.stencil_zfail_back)
                                      : front_reference;
        if (replace_op_reference && (front_reference != front.stencil_test_val ||
                                     back_reference != back.stencil_test_val)) {
            static bool reported{};
            if (!reported) {
                LOG_INFO(Render_Vulkan,
                         "Stencil ReplaceOp correction active: front test={} op={} reference={} "
                         "write_mask={:#x}; back test={} op={} reference={} write_mask={:#x}",
                         front.stencil_test_val, front.stencil_op_val, front_reference,
                         front.stencil_write_mask, back.stencil_test_val, back.stencil_op_val,
                         back_reference, back.stencil_write_mask);
                reported = true;
            }
        }
        dynamic_state.SetStencilReferences(front_reference, back_reference);
        dynamic_state.SetStencilWriteMasks(!stencil_clear ? front.stencil_write_mask : 0U,
                                           !stencil_clear ? back.stencil_write_mask : 0U);
        dynamic_state.SetStencilCompareMasks(front.stencil_mask, back.stencil_mask);
    }
}

void Rasterizer::UpdatePrimitiveState(const bool is_indexed) const {
    const auto& regs = liverpool->BackRegs();
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto is_list_topology = [](const AmdGpu::PrimitiveType type) {
        const auto topology = LiverpoolToVK::PrimitiveType(type);
        return topology == vk::PrimitiveTopology::ePointList ||
               topology == vk::PrimitiveTopology::eLineList ||
               topology == vk::PrimitiveTopology::eTriangleList ||
               topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
               topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
    };
    const auto is_patch_list_topology = [](const AmdGpu::PrimitiveType type) {
        // Quad and rect lists are emulated using tessellation.
        return type == AmdGpu::PrimitiveType::PatchPrimitive ||
               type == AmdGpu::PrimitiveType::QuadList || type == AmdGpu::PrimitiveType::RectList;
    };

    const auto prim_restart =
        (regs.enable_primitive_restart & 1) != 0 &&
        (instance.IsListRestartSupported() || !is_list_topology(regs.primitive_type)) &&
        (instance.IsPatchListRestartSupported() || !is_patch_list_topology(regs.primitive_type));
    ASSERT_MSG(!is_indexed || !prim_restart || regs.primitive_restart_index == 0xFFFF ||
                   regs.primitive_restart_index == 0xFFFFFFFF,
               "Primitive restart index other than -1 is not supported yet");

    const auto cull_mode = LiverpoolToVK::IsPrimitiveCulled(regs.primitive_type)
                               ? LiverpoolToVK::CullMode(regs.polygon_control.CullingMode())
                               : vk::CullModeFlagBits::eNone;
    const auto front_face = LiverpoolToVK::FrontFace(regs.polygon_control.front_face);

    dynamic_state.SetPrimitiveRestartEnabled(prim_restart);
    dynamic_state.SetRasterizerDiscardEnabled(regs.clipper_control.dx_rasterization_kill);
    dynamic_state.SetCullMode(cull_mode);
    dynamic_state.SetFrontFace(front_face);
}

void Rasterizer::UpdateRasterizationState() const {
    const auto& regs = liverpool->BackRegs();
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetLineWidth(regs.line_control.Width());
}

void Rasterizer::UpdateColorBlendingState(const GraphicsPipeline* pipeline) const {
    const auto& regs = liverpool->BackRegs();
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetBlendConstants(regs.blend_constants);
    if (Dlss::Mode() == 1) [[unlikely]] {
        auto masks = pipeline->GetGraphicsKey().write_masks;
        Dlss::ApplyHudLayerMask(masks, regs.color_buffers[0].Address());
        dynamic_state.SetColorWriteMasks(masks);
    } else {
        dynamic_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    }
    dynamic_state.SetAttachmentFeedbackLoopEnabled(attachment_feedback_loop);
}

void Rasterizer::ScopeMarkerBegin(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    auto command_access = scheduler.AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();
    cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopeMarkerEnd(bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    auto command_access = scheduler.AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();
    cmdbuf.endDebugUtilsLabelEXT();
}

void Rasterizer::ScopedMarkerInsert(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    auto command_access = scheduler.AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                         bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    auto command_access = scheduler.AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
        .color = std::array<f32, 4>(
            {(f32)((color >> 16) & 0xff) / 255.0f, (f32)((color >> 8) & 0xff) / 255.0f,
             (f32)(color & 0xff) / 255.0f, (f32)((color >> 24) & 0xff) / 255.0f})});
}

} // namespace Vulkan
