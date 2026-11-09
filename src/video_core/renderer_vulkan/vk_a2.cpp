// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include "video_core/texture_cache/tex_table.h"
#include "common/thread.h"
#include "common/logging/log.h"
#include "shader_recompiler/native_emit.h"
#include "video_core/amdgpu/cp_split.h"
#include "video_core/amdgpu/native_rec.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/native_census.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_a2.h"
#include "video_core/renderer_vulkan/vk_bindless.h"
#include "video_core/renderer_vulkan/vk_dlss_upscale.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/water_scale.h"

// Architecture A step A2, the stretch translator: v1 shadow draws, v2 colour draws (bindless
// textures + pulled vertices, "deferred material resolve"). Everything here runs on a WORKER
// thread except Replay (the recording lane).
// What it reproduces: NativeOpen (shadow and colour) + NativeRingPrepare's table and command
// math for a run of ring-described draws, and the generic draw tail's command-buffer state
// (pipeline, push constants, push descriptors, vertex input, index / vertex buffers, dynamic
// state) - as VALUES in a StretchCmds that the back replays in stream order (Replay below).
// Colour: the bindless slot words are the back's (BindlessResolve needs the texture cache), so
// the worker only keys each draw's material (raw T# / S# words as NativeImageKey) and leaves a
// SlotWrite per draw; the back resolves every material of the stretch once at publish and
// Replay writes the slot words into the job ring records as Rasterizer::BindlessRingWrite does.
//
// Thread-safety review - every non-local object this file reads, and why that is safe here:
//  - Config: Mode / NumWorkers (own statics), VideoCore::g_native_pass, g_native_coherent,
//    g_native_colour, g_native_colour_diag, AmdGpu::NativeRec::g_gpu_mode, Shader::g_native_ring /
//    g_native_pt / g_native_seg: all `const` / `inline const` initialised at load, never written.
//  - GraphicsPipeline: native_variant (atomic, acquire; the front stores it once), the
//    native_colour slots (key acquire, then variant acquire: the front claims the key, the
//    compile thread publishes the variant after setting native_vertex_binding /
//    native_table_binding; native_bindless is a constructor argument), and the variant's
//    immutable fields (Handle - may wait a deferred compile, GetLayout, GetGraphicsKey,
//    GetFetchShader, native_*_binding) plus the base's stage Infos' persistent lists (buffers,
//    images, samplers, ud_mask), which only the compile thread writes before the pipeline is
//    published.
//  - DrawIn::stages / vi: the Draw record's blocks in the CpSplit ring, valid while the worker
//    runs on this DCB (the caller's guarantee); read-only. The image / sampler sharps come from
//    the stage's flat buffer span inside that block (ReadStages output), as NativeImageKey's.
//  - NativeRec: g_rings / g_unions / g_gpu_handles (atomics, acquire), g_job_next (relaxed, a
//    lifetime bound only); Record (img_n / img_h1 / img_h2 included) and RunUnion through their
//    stamp seqlocks exactly as NativeRingPrepare reads them (fields first, acquire fence, stamp
//    re-check). g_gpu_recs (acquire) is written by Replay ONLY, on the recording lane: the slot
//    words 128-159 of a record, which the job thread never writes (GpuRecJobDwords) and the back
//    writes the same way (BindlessRingWrite); the back has published the stretch (slot words
//    resolved) before it records the Replay closure.
//  - BufferCache: A2LookupBuffer (page_table find_acquire + stable SlotVector; see its comment),
//    StructGeneration (atomic), GetBdaPageTableBuffer / GetFaultBuffer handles and device
//    addresses (set at construction, never change).
//  - Instance: device limits and feature flags (immutable after creation).
//  - Scheduler: GetMasterSemaphore()->KnownGpuTick() (atomic load) for arena reuse only.
//  - The arena: this worker's own buffer (created here, host-coherent); the GPU reads a region
//    only after the back published the stretch (happens-after this thread's writes through the
//    Stretch handoff), and the region is reused only after that batch's tick passed.
//  - Nothing here records into the scheduler, touches the stream buffer, the texture cache,
//    barriers or any Rasterizer member. Memory validity (CPU-dirty pages) is the submission's
//    coherence point (SHADPS4_NATIVE_COHERENT, required); GPU-written index / vertex data is
//    covered by the back's conservative per-stretch global barrier rule (architecture-a sec. 6);
//    a colour variant's VS reads the vertex buffer as STORAGE (the back's open adds a whole-buffer
//    vertex barrier when the stream is GPU-written), so that rule must cover storage reads too.
//  - What the worker does NOT check for a colour draw (the back does, per material, at
//    publish): NativeColourGates (tex_bind_memo / bind_memo_stats / pending stencil writebacks),
//    attachment_feedback_loop, images that are bound targets or force_general, the texture-cache
//    side of BindlessResolve (FindImage, layouts, sampler generation / border colour).

namespace Vulkan::A2 {

namespace NR = AmdGpu::NativeRec;

namespace {

constexpr u32 MaxDraws = 64;                              // Rasterizer::NativeMaxDraws
constexpr u32 CmdBytes = 20;                              // VkDrawIndexedIndirectCommand
constexpr u32 TableBytes = Shader::NativeTableDwords * 4; // the per-batch table
constexpr u32 ClipBytes = 96; // AmdGpu::NUM_CLIP_PLANES x 4 words (the back's planes copy)
constexpr u64 ArenaBytes = u64{8} << 20;                  // per worker
constexpr u64 MaxSpan = u64{64} << 20;                    // NativeRingPrepare's union bound

constexpr u64 AlignUp(u64 v, u64 a) {
    return (v + a - 1) & ~(a - 1);
}

/// SHADPS4_STENCIL_REPLACE_OP (vk_rasterizer.cpp g_replace_op_reference: exactly "1").
const bool g_replace_op = [] {
    const char* v = std::getenv("SHADPS4_STENCIL_REPLACE_OP");
    return v && std::strcmp(v, "1") == 0;
}();

constexpr std::array<const char*, NumRefusals> RefusalNames = {
    "off",           "not indexed",     "indirect",       "no variant",      "variant kind",
    "no vi",         "no job",          "no ring",        "stamp",           "lifetime",
    "record fields", "inherited slot",  "union",          "slot kind",       "guest lookup",
    "index lookup",  "vertex lookup",   "record vs regs", "instance offset", "vertex offset",
    "restart index", "arena",           "stages",         "layout",          "no cmd (once)",
    "colour gates",  "colour stages",   "colour images",  "colour key hash", "colour vertex",
    "colour stream", "materials",
};
static_assert(RefusalNames.size() == NumRefusals && NumRefusals <= 32);

/// The raw T# / S# words of a colour draw (Rasterizer::NativeImageKey, 512 words at most).
struct ImageKey {
    u32 n{};
    std::array<u32, 512> w{};
};

/// vk_rasterizer.cpp GetDrawOffsets, verbatim.
std::pair<u32, u32> GetDrawOffsets(const AmdGpu::Regs& regs, const Shader::BoundStage& info,
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

/// vk_rasterizer.cpp MakeUserData, verbatim.
Shader::PushData MakePushData(const AmdGpu::Regs& regs) {
    Shader::PushData push_data{};
    push_data.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push_data.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push_data.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push_data.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
    return push_data;
}

/// buffer_cache.cpp BuildVertexInputArrays, verbatim (it sits in that file's namespace).
void BuildVertexInputArrays(const AmdGpu::CpSplit::VertexInputsPayload& vi, SubBatch& out) {
    const auto* attribs = AmdGpu::CpSplit::VertexInputAttribs(vi);
    out.vi_count = vi.count;
    for (u32 i = 0; i < vi.count; ++i) {
        const auto& desc = attribs[i].desc;
        out.vi_attrs[i] = vk::VertexInputAttributeDescription2EXT{
            .location = desc.location,
            .binding = desc.location,
            .format = static_cast<vk::Format>(desc.vk_format),
            .offset = 0,
        };
        out.vi_bindings[i] = vk::VertexInputBindingDescription2EXT{
            .binding = desc.location,
            .stride = desc.stride,
            .inputRate = desc.instanced ? vk::VertexInputRate::eInstance
                                        : vk::VertexInputRate::eVertex,
            .divisor = desc.divisor,
        };
    }
}

// Rasterizer::UpdateViewportScissorState / UpdateDepthStencilState / UpdatePrimitiveState /
// UpdateRasterizationState / UpdateColorBlendingState as free functions of (regs, instance,
// pipeline): pure functions of the register mirror and device limits (the viewport memo and
// the hair trace are left out; attachment_feedback_loop is false: a shadow pass samples no
// bound target).

void ComputeViewportScissor(DynamicState& dynamic_state, const Instance& instance,
                            const AmdGpu::Regs& regs) {
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

    Viewports viewports;
    Scissors scissors;

    const auto& vp_ctl = regs.viewport_control;
    for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS; i++) {
        const auto& vp = regs.viewports[i];
        if (vp.xscale == 0) {
            continue;
        }

        const auto zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
        const auto zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;

        vk::Viewport viewport{};
        if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
            viewport.minDepth = zoffset - zscale;
            viewport.maxDepth = zoffset + zscale;
        } else {
            viewport.minDepth = zoffset;
            viewport.maxDepth = zoffset + zscale;
        }
        if (!instance.IsDepthRangeUnrestrictedSupported()) {
            viewport.minDepth = std::max(viewport.minDepth, 0.f);
            viewport.maxDepth = std::min(viewport.maxDepth, 1.f);
        }
        if (regs.IsClipDisabled()) {
            viewport.x = 0.f;
            viewport.y = 0.f;
            viewport.width = float(std::min<u32>(instance.GetMaxViewportWidth(), 16384u));
            viewport.height = float(std::min<u32>(instance.GetMaxViewportHeight(), 16384u));
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
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);
}

void ComputeDepthStencil(DynamicState& dynamic_state, const AmdGpu::Regs& regs) {
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
        const auto reference = [&](const AmdGpu::StencilRefMask& face, AmdGpu::CompareFunc compare,
                                   AmdGpu::StencilFunc fail, AmdGpu::StencilFunc pass,
                                   AmdGpu::StencilFunc depth_fail) -> u32 {
            if (g_replace_op && compare == AmdGpu::CompareFunc::Always &&
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
        dynamic_state.SetStencilReferences(front_reference, back_reference);
        dynamic_state.SetStencilWriteMasks(!stencil_clear ? front.stencil_write_mask : 0U,
                                           !stencil_clear ? back.stencil_write_mask : 0U);
        dynamic_state.SetStencilCompareMasks(front.stencil_mask, back.stencil_mask);
    }
}

/// False: primitive restart with an index other than -1 (the back ASSERTs; a worker refuses).
bool ComputePrimitive(DynamicState& dynamic_state, const Instance& instance,
                      const AmdGpu::Regs& regs) {
    const auto is_list_topology = [](const AmdGpu::PrimitiveType type) {
        const auto topology = LiverpoolToVK::PrimitiveType(type);
        return topology == vk::PrimitiveTopology::ePointList ||
               topology == vk::PrimitiveTopology::eLineList ||
               topology == vk::PrimitiveTopology::eTriangleList ||
               topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
               topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
    };
    const auto is_patch_list_topology = [](const AmdGpu::PrimitiveType type) {
        return type == AmdGpu::PrimitiveType::PatchPrimitive ||
               type == AmdGpu::PrimitiveType::QuadList || type == AmdGpu::PrimitiveType::RectList;
    };
    const auto prim_restart =
        (regs.enable_primitive_restart & 1) != 0 &&
        (instance.IsListRestartSupported() || !is_list_topology(regs.primitive_type)) &&
        (instance.IsPatchListRestartSupported() || !is_patch_list_topology(regs.primitive_type));
    if (prim_restart && regs.primitive_restart_index != 0xFFFF &&
        regs.primitive_restart_index != 0xFFFFFFFF) {
        return false;
    }
    const auto cull_mode = LiverpoolToVK::IsPrimitiveCulled(regs.primitive_type)
                               ? LiverpoolToVK::CullMode(regs.polygon_control.CullingMode())
                               : vk::CullModeFlagBits::eNone;
    const auto front_face = LiverpoolToVK::FrontFace(regs.polygon_control.front_face);
    dynamic_state.SetPrimitiveRestartEnabled(prim_restart);
    dynamic_state.SetRasterizerDiscardEnabled(regs.clipper_control.dx_rasterization_kill);
    dynamic_state.SetCullMode(cull_mode);
    dynamic_state.SetFrontFace(front_face);
    return true;
}

bool ComputeDynamicState(DynamicState& ds, const Instance& instance, const AmdGpu::Regs& regs,
                         const GraphicsPipeline* pipeline) {
    ComputeViewportScissor(ds, instance, regs);
    ComputeDepthStencil(ds, regs);
    if (!ComputePrimitive(ds, instance, regs)) {
        return false;
    }
    ds.SetLineWidth(regs.line_control.Width());
    ds.SetBlendConstants(regs.blend_constants);
    if (Dlss::Mode() == 1) [[unlikely]] {
        // SHADPS4_DLSS_UPSCALE: HUD draws into the HUD layer also write coverage.
        auto masks = pipeline->GetGraphicsKey().write_masks;
        Dlss::ApplyHudLayerMask(masks, regs.color_buffers[0].Address());
        ds.SetColorWriteMasks(masks);
    } else {
        ds.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    }
    ds.SetAttachmentFeedbackLoopEnabled(false);
    return true;
}

/// The push-constant bytes the native shaders read: the viewport scale / offset floats. The
/// ud_regs are per draw (the VS reads them from the record, the FS has none) and buf_offsets
/// are zero, so they are not a batching key.
bool PushKeySame(const Shader::PushData& a, const Shader::PushData& b) {
    static_assert(offsetof(Shader::PushData, ud_regs) == 4 * sizeof(float));
    return std::memcmp(&a, &b, offsetof(Shader::PushData, ud_regs)) == 0;
}

bool AnyDirty(const DynamicState& ds) {
    // The dirty bits are a bit-field struct whose bytes are only ever memset (Invalidate /
    // ClearDirty) or bit-set: compare them with zero.
    constexpr size_t Bytes = sizeof(DynamicState::dirty_state);
    std::array<u8, Bytes> zero{};
    return std::memcmp(&ds.dirty_state, zero.data(), Bytes) != 0;
}

void ClearDirty(DynamicState& ds) {
    std::memset(&ds.dirty_state, 0, sizeof(ds.dirty_state));
}

} // namespace

int Mode() {
    static const int mode = [] {
        const char* v = std::getenv("SHADPS4_A2");
        return v && (v[0] == '1' || v[0] == '2') && v[1] == 0 ? v[0] - '0' : 0;
    }();
    return mode;
}

u32 NumWorkers() {
    static const u32 n = [] {
        if (const u32 a = Common::AutoPoolWorkers(false)) {
            return a; // SHADPS4_POOL_AUTO
        }
        const char* v = std::getenv("SHADPS4_A2_WORKERS");
        const long w = v ? std::strtol(v, nullptr, 10) : 3;
        return static_cast<u32>(std::clamp<long>(w <= 0 ? 3 : w, 1, 8));
    }();
    return n;
}

const char* RefusalName(u32 reason) {
    return reason < NumRefusals ? RefusalNames[reason] : "?";
}

bool CommitSets(StretchCmds& cmds) {
    // l4g, the back's thread at publish: one set per non-push sub-batch.
    for (auto& b : cmds.batches) {
        if (!b.set_owner || !b.emit_descriptors || b.num_writes == 0) {
            continue;
        }
        std::array<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS + 4> infos;
        std::array<vk::WriteDescriptorSet, Shader::NUM_BUFFERS + 4> writes;
        for (u32 i = 0; i < b.num_writes; ++i) {
            const auto& w = b.writes[i];
            infos[i] = vk::DescriptorBufferInfo{w.buffer, w.offset, w.range};
            writes[i] = vk::WriteDescriptorSet{
                .dstSet = VK_NULL_HANDLE,
                .dstBinding = w.binding,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &infos[i],
            };
        }
        b.set = b.set_owner->CommitSetWrites(std::span{writes.data(), b.num_writes});
        if (!b.set) {
            return false;
        }
    }
    return true;
}

bool SecondaryMode() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_A2_SEC");
        return e && e[0] == '1';
    }();
    return on;
}

/// r9 (2026-10-14, notes/rooftop-2026-10-14.md sec. 6): SHADPS4_NR_WORKER. The job threads' draw
/// records (NativeRec: derive + GPU record per DrawIndex HLE call) cost the rooftop's render-job
/// span 2.85 ms per frame (s992 vs s986: span 4.80 -> 1.95 ms with SHADPS4_NATIVE_REC=0) - on
/// Game:Main's critical path, while everything the translator needs it already computes from its
/// own exact register state (user data, each Guest V#'s address and size, the vertex streams).
/// 1 = the translator writes each taken draw's GPU record into its own ring (no job record read;
/// run with SHADPS4_NATIVE_REC=0 so the job threads do no record work); 2 = verify: job records
/// stay the ones drawn, and the translator builds its own record for every taken draw without
/// inherited registers and compares it dword by dword with the job's (GPU record words 0-127,
/// 160-175; 128-159 are the back's slot words).
int NrWorkerMode() {
    static const int m = [] {
        const char* e = std::getenv("SHADPS4_NR_WORKER");
        const int v = e ? e[0] - '0' : 0;
        return v == 1 || v == 2 ? v : 0;
    }();
    return m;
}
constexpr u32 WorkerRecBits = 15;
constexpr u32 WorkerRecSize = 1u << WorkerRecBits; ///< records per translator (~23 MB)
constexpr u32 WorkerRecMask = WorkerRecSize - 1;

/// SHADPS4_LEAN_PERDRAW bit 2 (2026-10-15; also read by liverpool.cpp and lean_parse.cpp): the
/// translator's per-stretch container costs - a stretch's sub-batch list starts at a recent
/// high-water mark (8..16) and doubles when full (the 1.5x growth moved every SubBatch, ~3 KB
/// each, again at 12, 18, 27, ...), and the arena / record fence queues pop their front in O(1)
/// (erase(begin()) shifted every live fence - an estimated 1-3 thousand - once per stretch).
static bool LeanPerDrawOn() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_LEAN_PERDRAW");
        return e && ((std::strtoul(e, nullptr, 10) & 2) != 0); // bit 2 (s2 screen: L3 translate 1.35 -> 6.09 us per draw)
    }();
    return on;
}

/// SHADPS4_LEAN_PERDRAW bit 8 (2026-10-15 evening; bit 2 includes it): only the O(1) fence queue
/// pops, without the sub-batch reserve (the s2 screen's inferred cause). The value is a decimal
/// bit mask (liverpool.cpp: 1 memos, 2 every allocation part, 4 the register-delta swap).
static bool LeanFencePopOn() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_LEAN_PERDRAW");
        return e && ((std::strtoul(e, nullptr, 10) & (2 | 8)) != 0);
    }();
    return on;
}

/// SHADPS4_LEAN_PERDRAW bit 16 (2026-10-15 evening): a stretch's StretchCmds comes from a pool
/// instead of a fresh make_shared. s1268: the per-stretch batches.reserve(8) (~24 KB of SubBatches,
/// above the Windows heap's low-fragmentation size limit) went to the backend allocator and cost
/// 28% of a worker's wall once the heap layout changed (bit 4 parks the delta buffers); 0.6% in
/// s1251. The last owner (back, recorder lane) only returns the object; the translator resets it
/// (every field) when it takes it, so the lists keep their capacity.
static bool LeanCmdsPoolOn() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_LEAN_PERDRAW");
        return e && ((std::strtoul(e, nullptr, 10) & 16) != 0);
    }();
    return on;
}

namespace {
struct CmdsPool {
    std::mutex mu;
    std::vector<StretchCmds*> free;
};

CmdsPool& GetCmdsPool() {
    static auto* pool = new CmdsPool(); // never destroyed: a deleter may run during shutdown
    return *pool;
}

/// shared_ptr deleter: back to the pool, bounded in bytes (Opus review RISK): at most 1024
/// objects, and an object whose lists grew past ~96 KB of sub-batches (32 x ~3 KB) or 4096
/// slot writes / checks is freed instead (its next stretch allocates again).
void ReturnCmds(StretchCmds* c) {
    if (c->batches.capacity() <= 32 && c->slot_writes.capacity() <= 4096 &&
        c->inh_checks.capacity() <= 4096 && c->clip_checks.capacity() <= 4096 &&
        c->fmt_checks.capacity() <= 4096) {
        auto& pool = GetCmdsPool();
        std::scoped_lock lk{pool.mu};
        if (pool.free.size() < 1024) {
            pool.free.push_back(c);
            return;
        }
    }
    delete c;
}

/// A reset StretchCmds (as a fresh one) with the pool's capacity.
std::shared_ptr<StretchCmds> TakePooledCmds() {
    StretchCmds* c = nullptr;
    {
        auto& pool = GetCmdsPool();
        std::scoped_lock lk{pool.mu};
        if (!pool.free.empty()) {
            c = pool.free.back();
            pool.free.pop_back();
        }
    }
    if (c) {
        c->batches.clear();
        c->bindless_set = vk::DescriptorSet{};
        c->mat_slots = {};
        c->slot_writes.clear();
        c->inh_checks.clear();
        c->clip_checks.clear();
        c->fmt_checks.clear();
    } else {
        c = new StretchCmds();
    }
    return std::shared_ptr<StretchCmds>(c, &ReturnCmds);
}
} // namespace

/// LeanFencePopOn: pops a fence queue's front. The popped entry drops its tick reference now (as
/// erase did); the dead prefix is erased once it is at least half the vector (amortized O(1)).
/// `head` is the queue's first live entry (0 forever with the flag off).
template <typename Fences>
void PopFenceFront(Fences& fences, size_t& head) {
    fences[head].tick.reset();
    ++head;
    if (head == fences.size()) {
        fences.clear();
        head = 0;
    } else if (head >= 64 && head * 2 >= fences.size()) {
        fences.erase(fences.begin(), fences.begin() + static_cast<std::ptrdiff_t>(head));
        head = 0;
    }
}

/// r9: one draw's GPU record as NativeRecGpu writes it (words 0-127 and 160-175; the slot words
/// 128-159 stay zero here, Replay writes them).
void BuildWorkerRecord(std::array<u32, NR::GpuRecDwords>& g, const Shader::PushData& push,
                       u64 guest_mask, const std::array<VAddr, Shader::NUM_BUFFERS>& o_addr,
                       const std::array<u64, Shader::NUM_BUFFERS>& o_vsize, u32 vb_n,
                       const AmdGpu::CpSplit::VertexInputAttrib* attribs,
                       const std::array<VAddr, 16>& vaddr_vi) {
    g.fill(0);
    std::memcpy(g.data() + Shader::NativeRecordUdOffset, push.ud_regs.data(),
                sizeof(u32) * Shader::NUM_USER_DATA_REGS);
    for (u64 m = guest_mask; m != 0; m &= m - 1) {
        const u32 b = static_cast<u32>(std::countr_zero(m));
        g[Shader::NativeRecordOffOffset + b] = static_cast<u32>(o_addr[b]);
        g[Shader::NativeRecordSizeOffset + b] = static_cast<u32>(std::min<u64>(o_vsize[b], 0xffffffffull));
        g[Shader::NativeRecordHiOffset + b / 4] |= static_cast<u32>((o_addr[b] >> 32) & 0xffu)
                                                   << ((b % 4) * 8);
    }
    for (u32 k = 0; k < vb_n && k < 16; ++k) {
        const u32 loc = attribs[k].desc.location;
        if (vaddr_vi[k] == 0 || loc >= Shader::NativeMaxPulledLocations) {
            continue;
        }
        g[Shader::NativeRecordVoffOffset + loc] = static_cast<u32>(vaddr_vi[k]);
        g[Shader::NativeRecordVsizeOffset + loc] = 0xffffffffu;
        g[Shader::NativeRecordVhiOffset + loc / 4] |= static_cast<u32>((vaddr_vi[k] >> 32) & 0xffu)
                                                      << ((loc % 4) * 8);
    }
}

void Replay(const StretchCmds& cmds, vk::CommandBuffer cmdbuf, const Instance& instance) {
    WriteSlots(cmds);
    RecordBatches(cmds, cmds.bindless_set, cmdbuf, instance);
}

void WriteSlots(const StretchCmds& cmds) {
    // Colour draws: the back resolved every material of the stretch (mat_slots, bindless_set)
    // before recording this closure; write each draw's slot words into its job ring record
    // exactly as Rasterizer::BindlessRingWrite does (write-combined ring memory, 128-159).
    for (const auto& sw : cmds.slot_writes) {
        u32* gr = sw.ring ? sw.ring : NR::g_gpu_recs[sw.tag >> 28].load(std::memory_order_acquire);
        if (gr && sw.mat < MaxMaterials) {
            std::memcpy(gr + size_t(sw.idx & (sw.ring ? WorkerRecMask : NR::RingSize - 1)) * NR::GpuRecDwords +
                            Shader::NativeRecordImgOffset,
                        cmds.mat_slots[sw.mat].data(), sizeof(u32) * Shader::NativeBindlessSlots);
        }
    }
}

void RecordBatches(const StretchCmds& cmds, vk::DescriptorSet set1, vk::CommandBuffer cmdbuf,
                   const Instance& instance) {
    for (const auto& b : cmds.batches) {
        if (b.emit_pipeline) {
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, b.pipeline);
        }
        if (b.emit_push) {
            cmdbuf.pushConstants(b.layout, AllGraphicsStageBits, 0u, sizeof(b.push), &b.push);
        }
        if (b.emit_descriptors && b.num_writes != 0 && b.set_owner) {
            // l4g: the set the back committed at publish
            cmdbuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, b.layout, 0, b.set, {});
        } else if (b.emit_descriptors && b.num_writes != 0) {
            std::array<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS + 4> infos;
            std::array<vk::WriteDescriptorSet, Shader::NUM_BUFFERS + 4> writes;
            for (u32 i = 0; i < b.num_writes; ++i) {
                const auto& w = b.writes[i];
                infos[i] = vk::DescriptorBufferInfo{w.buffer, w.offset, w.range};
                writes[i] = vk::WriteDescriptorSet{
                    .dstSet = VK_NULL_HANDLE,
                    .dstBinding = w.binding,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo = &infos[i],
                };
            }
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, b.layout, 0,
                                        vk::ArrayProxy<const vk::WriteDescriptorSet>{
                                            b.num_writes, writes.data()});
        }
        if (b.bindless && set1) {
            // The global image / sampler arrays (set 1) with the variant's layout, as the
            // back's native colour batch binds them (vk_rasterizer DrawTail, bl_set).
            cmdbuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, b.layout, 1, set1, {});
        }
        if (b.emit_vertex_input) {
            cmdbuf.setVertexInputEXT(
                vk::ArrayProxy<const vk::VertexInputBindingDescription2EXT>{b.vi_count,
                                                                             b.vi_bindings.data()},
                vk::ArrayProxy<const vk::VertexInputAttributeDescription2EXT>{b.vi_count,
                                                                               b.vi_attrs.data()});
        }
        if (b.emit_index) {
            cmdbuf.bindIndexBuffer(b.ib, b.ib_off, b.ib_type);
        }
        if (b.emit_vertex_buffers && b.vb_n != 0) {
            cmdbuf.bindVertexBuffers(0, b.vb_n, b.vb.data(), b.vb_off.data());
        }
        DynamicState dyn = b.dyn; // Commit clears dirty bits: work on a copy
        dyn.Commit(instance, cmdbuf);
        if (b.ind_buf) {
            cmdbuf.drawIndexedIndirect(b.ind_buf, b.ind_off, 1, CmdBytes); // q2: the game's args
        } else {
            cmdbuf.drawIndexedIndirectCount(b.arena, b.cmd_off, b.arena, b.count_off, b.n,
                                            CmdBytes);
        }
    }
}

struct StretchTranslator::Impl {
    explicit Impl(StretchTranslator* owner_, const WorkerContext& ctx_);

    bool Take(const AmdGpu::Regs& regs, const DrawIn& draw);
    std::unique_ptr<Stretch> End();

    /// A region of `bytes` from the arena ring at `align`; never waits. False = none free.
    /// `start` = its unmasked start (the fence bookkeeping), `masked_off` its buffer offset.
    bool AllocRegion(u64 bytes, u64& start, u64& masked_off, u8*& ptr);

    /// p2 (SHADPS4_A2_SEC): records the open stretch into a secondary (End()).
    void RecordSecondary();
    /// A secondary from this worker's pools, tied to `tick` for reuse; null if none is free.
    vk::CommandBuffer AcquireSecondary(const std::shared_ptr<std::atomic<u64>>& tick);
    struct SecPool {
        vk::CommandPool pool{};
        std::vector<vk::CommandBuffer> cbs;
        u32 used{};
        std::vector<std::shared_ptr<std::atomic<u64>>> ticks;
    };
    static constexpr u32 SecPools = 8, SecPerPool = 64;
    std::array<SecPool, SecPools> sec_pools{};
    u32 sec_cur{};
    bool sec_on{};

    StretchTranslator* owner;
    WorkerContext ctx;
    std::unique_ptr<VideoCore::Buffer> arena_buffer; ///< the Arena's VkBuffer (owned here)
    bool enabled{};
    bool colour_on{}; ///< colour draws may be taken (SHADPS4_NATIVE_COLOUR=1, no diag 1 / 4096)
    bool cmd_warned{};
    const bool perdraw{LeanPerDrawOn()}; ///< SHADPS4_LEAN_PERDRAW bit 2 (sub-batch growth)
    const bool fence_pop{LeanFencePopOn()}; ///< SHADPS4_LEAN_PERDRAW bit 2 or 8 (O(1) fence pops)
    const bool cmds_pool{LeanCmdsPoolOn()}; ///< SHADPS4_LEAN_PERDRAW bit 16 (pooled StretchCmds)
    u32 batch_hint{8}; ///< perdraw: the next stretch's sub-batch reserve (decaying high-water mark)
    u64 align{256};
    u64 region_bytes{};
    u64 clip_off_in_region{}; ///< the sub-batch's clip planes (after table, commands, count)
    vk::Buffer dummy{}; ///< the always-valid Guest stand-in (BDA page table buffer)
    u64 pt_fallback{}, fault_addr{};

    // The open stretch.
    std::unique_ptr<Stretch> stretch;
    std::shared_ptr<StretchCmds> cmds;
    u64 stretch_start{};
    u64 last_desc_id{};
    DynamicState dyn{}; ///< tracked state of the stretch (dirty bits cleared per sub-batch)
    /// The stretch's materials (index = Stretch::materials index): the key the dedup compares
    /// and the base pipeline it was seen with (another base's FS may decode the same T# with
    /// other descriptor bits: a different material).
    std::array<ImageKey, MaxMaterials> mat_keys{};
    std::array<const GraphicsPipeline*, MaxMaterials> mat_base{};

    // The open sub-batch and its keys.
    bool batch_open{};
    bool b_colour{};
    u64 b_vbuf{};  ///< colour: the one vertex VkBuffer of the opener
    u32 b_vmask{}; ///< colour: the opener's non-null pulled locations
    u32 b_vptmask{}; ///< q18: the opener's page-table-pulled locations
    u64 b_vbuf2{};   ///< q53: the opener's second vertex VkBuffer (0 = none)
    u32 b_v2mask{};  ///< q53: the opener's locations pulled from it
    u32* count_ptr{};
    u8* cmd_ptr{};
    const GraphicsPipeline* b_native{};
    u32 b_slot{}, b_next_idx{}, b_union{}, b_head{};
    u32 b_push_inh{}, b_inh_vs{}, b_inh_ps{};
    std::array<u32, 16> b_ud_vs{}, b_ud_ps{};
    u64 b_ib{};
    u32 b_index_size{};
    u32 b_vb_n{};
    std::array<u64, 16> b_vb{}, b_vcpu{}, b_vb_off{};
    std::array<u32, 16> b_stride{};
    std::array<u8, 16> b_inst{};
    u64 b_guest_mask{};
    std::array<VAddr, Shader::NUM_BUFFERS> b_glo{}, b_ghi{};
    Shader::PushData b_push{};
    // Inherited Guest slots (the opener's V# for every member) and clip planes.
    u64 b_inh_slots{};
    std::array<VAddr, Shader::NUM_BUFFERS> b_iaddr{};
    std::array<u64, Shader::NUM_BUFFERS> b_isize{};
    std::array<u32, Shader::NUM_BUFFERS> b_ioff{}, b_isz{};
    bool b_clip{};
    std::array<u32, 24> b_planes{};
    u64 b_fmt_mask{};
    u64 b_fb_mask{}; ///< slots resolved by the opener's own V# (guest-lookup fallback)
    u64 b_big_mask{}; ///< r12 mode 1: slots with a V# over MaxSpan (members bind the identical V#)
    bool b_ind{};    ///< q2: the open sub-batch is a guest indirect draw (never continued)
    std::array<u32, Shader::NUM_BUFFERS> b_tbase{}; ///< the opener's table bases (checks)

    // r9 (SHADPS4_NR_WORKER=1): this translator's own GPU record ring (records are written by this
    // thread only; an entry is reused only after the GPU passed the tick of the stretch that last
    // wrote it - the arena's rule).
    std::unique_ptr<VideoCore::StreamBuffer> rec_ring;
    u32* rec_mapped{};
    u64 rec_handle{};
    u64 rec_head{};          ///< next record index (unmasked)
    u64 rec_stretch_start{}; ///< the open stretch's first record index
    struct RecFence {
        u64 start, end;
        std::shared_ptr<std::atomic<u64>> tick;
    };
    std::vector<RecFence> rec_fences;
    size_t rec_fence_head{}; ///< first live rec_fences entry (fence_pop; 0 otherwise)
    /// True if entry rec_head may be written now (never waits).
    bool RecReserve();
    /// r9 census (this translator): records written, verify compares, mismatching records by
    /// region (user data, address, size, vertex address, vertex size, high bytes), ring full.
    std::array<u64, 19> nrw{};
    void NrwLog();
};

StretchTranslator::Impl::Impl(StretchTranslator* owner_, const WorkerContext& ctx_)
    : owner{owner_}, ctx{ctx_} {
    const char* why = nullptr;
    if (Mode() == 0) {
        why = "SHADPS4_A2 off";
    } else if (VideoCore::g_native_pass != 1) {
        why = "SHADPS4_NATIVE_PASS != 1";
    } else if (NR::g_gpu_mode != 1 || !Shader::g_native_ring) {
        why = "SHADPS4_NATIVE_GPU_DIRECT != 1";
    } else if (Shader::g_native_pt != 1) {
        why = "SHADPS4_NATIVE_PT != 1";
    } else if (Shader::g_native_seg != 0) {
        why = "SHADPS4_NATIVE_SEG on (list-addressed records)";
    } else if (VideoCore::g_native_coherent == 0) {
        why = "SHADPS4_NATIVE_COHERENT off";
    } else if (!ctx.instance || !ctx.scheduler || !ctx.buffer_cache || !ctx.arena) {
        why = "context incomplete";
    } else if (!ctx.instance->IsVertexInputDynamicState()) {
        why = "no dynamic vertex input";
    }
    if (!why) {
        align = std::max<u64>(ctx.instance->StorageMinAlignment(), 16);
        clip_off_in_region = AlignUp(TableBytes + u64(MaxDraws) * CmdBytes + 16, align);
        region_bytes = clip_off_in_region + AlignUp(ClipBytes, align);
        if (!ctx.arena->buffer) {
            // Host-visible, coherent (Upload prefers HOST_COHERENT), storage + indirect, with a
            // device address as the Arena interface asks (its own dedicated allocation).
            arena_buffer = std::make_unique<VideoCore::Buffer>(
                *ctx.instance, *ctx.scheduler, VideoCore::MemoryUsage::Upload, 0,
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer |
                    vk::BufferUsageFlagBits::eShaderDeviceAddress,
                ArenaBytes);
            if (!arena_buffer->is_coherent || arena_buffer->mapped_data.empty()) {
                why = "arena memory not host-coherent / not mapped";
            } else {
                ctx.arena->buffer = arena_buffer->Handle();
                ctx.arena->mapped = arena_buffer->mapped_data.data();
                ctx.arena->device_address = arena_buffer->BufferDeviceAddress();
                ctx.arena->size = ArenaBytes;
                ctx.arena->head = 0;
            }
        }
    }
    if (!why && NrWorkerMode() == 1) {
        // r9: the record ring in mapped coherent memory, as the job slots' rings (NativeGpuRingAlloc).
        rec_ring = std::make_unique<VideoCore::StreamBuffer>(
            *ctx.instance, *ctx.scheduler, VideoCore::MemoryUsage::Stream,
            u64(WorkerRecSize) * NR::GpuRecDwords * sizeof(u32), vk::BufferUsageFlags{});
        if (rec_ring->mapped_data.empty() || !rec_ring->IsCoherent()) {
            why = "SHADPS4_NR_WORKER: record ring not mapped coherent memory";
        } else {
            rec_mapped = reinterpret_cast<u32*>(rec_ring->mapped_data.data());
            rec_handle = std::bit_cast<u64>(static_cast<VkBuffer>(rec_ring->Handle()));
        }
    }
    if (!why) {
        const auto* pt = ctx.buffer_cache->GetBdaPageTableBuffer();
        dummy = pt->Handle();
        pt_fallback = pt->BufferDeviceAddress();
        fault_addr = ctx.buffer_cache->GetFaultBuffer()->BufferDeviceAddress();
        enabled = true;
        // Colour draws: SHADPS4_NATIVE_COLOUR=1; diag bit 1 (the back takes no colour joins) and
        // bit 4096 (the compare SSBO shifts the variant's table binding) keep them on the back.
        colour_on = VideoCore::g_native_colour && (VideoCore::g_native_colour_diag & 4097u) == 0;
        LOG_INFO(Render_Vulkan,
                 "A2 worker {}: translator on (arena {} MB, region {} B, align {}, colour {})",
                 ctx.worker_index, ArenaBytes >> 20, region_bytes, align, colour_on ? "on" : "off");
        // p2b (review #10): only the L3 lean translators (worker_index >= 16) - the older A2
        // worker loop's publish never executes a secondary.
        if (SecondaryMode() && ctx.instance->IsMaintenance7Supported() && ctx.worker_index >= 16) {
            const auto dev = ctx.instance->GetDevice();
            for (auto& p : sec_pools) {
                p.pool = dev.createCommandPool(vk::CommandPoolCreateInfo{
                                                   .flags = vk::CommandPoolCreateFlagBits::eTransient,
                                                   .queueFamilyIndex =
                                                       ctx.instance->GetGraphicsQueueFamilyIndex(),
                                               })
                             .value;
            }
            sec_on = true;
            LOG_INFO(Render_Vulkan, "A2 worker {}: secondaries on ({} pools x {})",
                     ctx.worker_index, SecPools, SecPerPool);
        }
    } else {
        LOG_INFO(Render_Vulkan, "A2 worker {}: translator off ({})", ctx.worker_index, why);
    }
}

bool StretchTranslator::Impl::AllocRegion(u64 bytes, u64& start, u64& masked_off, u8*& ptr) {
    Arena& ar = *ctx.arena;
    u64 a = AlignUp(ar.head, align);
    if ((a % ar.size) + bytes > ar.size) {
        a += ar.size - (a % ar.size); // never straddle the wrap: pad to the next lap
    }
    const u64 end = a + bytes;
    // The open stretch's own regions must not alias (a stretch longer than the arena).
    if (stretch && end - stretch_start > ar.size) {
        return false;
    }
    // Every published region one lap back must be past its batch tick; never wait (the back
    // may be blocked on this worker's DCB).
    if (end > ar.size) {
        const u64 reuse_below = end - ar.size;
        u64 known = 0;
        bool known_read = false;
        size_t& head = ar.fence_head; // 0 forever unless fence_pop: [0] is the front
        while (head < ar.fences.size() && ar.fences[head].start < reuse_below) {
            const u64 tick = ar.fences[head].tick->load(std::memory_order_acquire);
            if (tick == 0) {
                return false; // not published yet
            }
            if (!known_read) {
                known = ctx.scheduler->GetMasterSemaphore()->KnownGpuTick();
                known_read = true;
            }
            if (known < tick) {
                return false; // the GPU has not passed it (the back refreshes the tick)
            }
            if (fence_pop) {
                PopFenceFront(ar.fences, head);
            } else {
                ar.fences.erase(ar.fences.begin());
            }
        }
    }
    ar.head = end;
    start = a;
    masked_off = a % ar.size;
    ptr = ar.mapped + masked_off;
    return true;
}

bool StretchTranslator::Impl::RecReserve() {
    const u64 next = rec_head + 1;
    if (next <= WorkerRecSize) {
        return true;
    }
    const u64 reuse_below = next - WorkerRecSize; // entries below it are overwritten from here
    if (stretch && rec_stretch_start < reuse_below) {
        return false; // the open stretch would overwrite its own records
    }
    u64 known = 0;
    bool known_read = false;
    size_t& head = rec_fence_head; // 0 forever unless fence_pop: [0] is the front
    while (head < rec_fences.size() && rec_fences[head].start < reuse_below) {
        const u64 tick = rec_fences[head].tick->load(std::memory_order_acquire);
        if (tick == 0) {
            return false; // not published yet
        }
        if (!known_read) {
            known = ctx.scheduler->GetMasterSemaphore()->KnownGpuTick();
            known_read = true;
        }
        if (known < tick) {
            return false; // the GPU has not passed that stretch
        }
        if (fence_pop) {
            PopFenceFront(rec_fences, head);
        } else {
            rec_fences.erase(rec_fences.begin());
        }
    }
    return true;
}

void StretchTranslator::Impl::NrwLog() {
    LOG_INFO(Render_Vulkan,
             "NrWorker (mode {}, worker {}): records written {} | verify compares {}, mismatching: "
             "user data {} address {} size {} vertex address {} vertex size {} high bytes {} | "
             "ring full refusals {} | guest lookup refusals: V# size 0 {} over MaxSpan {} no "
             "cached buffer {} below buffer {} past the end by <=4K {} <=64K {} more {} | mode-2 "
             "union binds {} with the own V# past the buffer end {} | mode-1 big-V# binds {}",
             NrWorkerMode(), ctx.worker_index, nrw[0], nrw[1], nrw[2], nrw[3], nrw[4], nrw[5],
             nrw[6], nrw[7], nrw[8], nrw[9], nrw[10], nrw[11], nrw[12], nrw[13], nrw[14], nrw[15], nrw[17], nrw[16], nrw[18]);
}

/// q2: SHADPS4_A2_INDIRECT=1 - indirect colour draws on the translator.
bool A2IndirectOn() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_A2_INDIRECT");
        return e && e[0] == '1';
    }();
    return on;
}

/// SHADPS4_WATER_REFLECTION_SCALE: a draw into the planar reflection's targets (a 480x270 depth,
/// or an image the back gave a scaled backing - published addresses, relaxed atomics). Its pass
/// needs the back's BeginRendering / viewport scaling, so the worker leaves it to the back
/// (under RefuseOff: the refusal table is full). Pure function of the mirror + atomics.
static bool WaterScaleTarget(const AmdGpu::Regs& regs, const GraphicsPipeline* pipeline) {
    const auto& db = regs.depth_buffer;
    const bool depth_used = regs.depth_control.depth_enable || regs.depth_control.depth_write_enable;
    if (depth_used && db.DepthValid() &&
        ((db.Pitch() == VideoCore::WaterScale::kWidth &&
          (db.Height() == VideoCore::WaterScale::kHeight || db.Height() == 272)) ||
         VideoCore::WaterScale::IsPublished(db.DepthAddress()))) {
        return true;
    }
    // Only the colour targets the pipeline writes (stale registers of unbound slots don't count).
    const u32 mrt = pipeline ? pipeline->GetGraphicsKey().mrt_mask : 0xffu;
    for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
        const auto& c = regs.color_buffers[cb];
        if ((mrt >> cb) & 1u && c && VideoCore::WaterScale::IsPublished(c.Address())) {
            return true;
        }
    }
    return false;
}

bool StretchTranslator::Impl::Take(const AmdGpu::Regs& regs, const DrawIn& draw) {
    const auto refuse = [&](u32 why) {
        ++owner->refusals[why];
        owner->last_refusal = why;
        return false;
    };
    if (!enabled) {
        return refuse(RefuseOff);
    }
    if (VideoCore::WaterScale::g_active && WaterScaleTarget(regs, draw.pipeline)) [[unlikely]] {
        VideoCore::WaterScale::g_counters.worker_refusals.fetch_add(1, std::memory_order_relaxed);
        return refuse(RefuseOff);
    }
    // SHADPS4_HUD_NATIVE (vk_dlss_upscale.h): draws into the HUD layer go to the back, whose
    // BeginRendering sends the HUD passes to the native HUD image. Decided by the HUD-layer mode
    // (stable across frames), not by the redirect itself: this runs ahead of the back's resolve.
    if (Dlss::HudNativeSetting() != 0 && Dlss::HudLayerActive() && regs.color_buffers[0] &&
        regs.color_buffers[0].Address() == Dlss::HudLayerAddr()) [[unlikely]] {
        return refuse(RefuseOff);
    }
    if (!draw.is_indexed) {
        return refuse(RefuseNotIndexed);
    }
    const bool ind = draw.is_indirect && A2IndirectOn();
    if (draw.is_indirect && !ind) {
        // m3 census: would a variant exist for it (native shadow kind or colour by description)?
        const GraphicsPipeline* b = draw.pipeline;
        const bool has = b && (b->native_variant.load(std::memory_order_relaxed) != nullptr ||
                               (draw.vi && draw.vi->count <= 16 &&
                                b->NativeColourVariant(VideoCore::NativeVertexKey(
                                    AmdGpu::CpSplit::VertexInputAttribs(*draw.vi),
                                    draw.vi->count)) != nullptr));
        ++owner->sub[has ? 32 : 33];
        if (!has) {
            // q1: why no variant - no vi / instanced attribute / the pipeline's colour state /
            // the slot of this vertex description.
            u32 r = 34;
            if (draw.vi && b) {
                const auto* at = AmdGpu::CpSplit::VertexInputAttribs(*draw.vi);
                bool inst = false;
                for (u32 k = 0; k < draw.vi->count && k < 16; ++k) {
                    inst = inst || at[k].desc.instanced;
                }
                const u8 s = b->native_colour_static;
                r = inst ? 35 : s == 0 ? 36 : s == 1 ? 37 : b->native_colour_untextured ? 38 : 39;
                if (s == 1 && draw.vi->count <= 16) {
                    const u64 vkey = VideoCore::NativeVertexKey(at, draw.vi->count);
                    u32 q = 43;
                    for (const auto& slot : b->native_colour) {
                        if (slot.key.load(std::memory_order_relaxed) == vkey) {
                            const u32 st = slot.state.load(std::memory_order_relaxed);
                            q = st == 2 ? 40
                                : slot.variant.load(std::memory_order_relaxed) ? 41
                                                                               : 42;
                            break;
                        }
                    }
                    ++owner->sub[q];
                }
            } else if (!b) {
                r = 39;
            }
            ++owner->sub[r];
            static std::atomic<u32> logged{0};
            if (b && logged.load(std::memory_order_relaxed) < 24) {
                static std::mutex mu;
                static std::vector<const void*> seen;
                std::scoped_lock lk{mu};
                if (std::find(seen.begin(), seen.end(), b) == seen.end() && seen.size() < 24) {
                    seen.push_back(b);
                    logged.fetch_add(1, std::memory_order_relaxed);
                    const char* why = b->native_colour_why.load(std::memory_order_relaxed);
                    LOG_INFO(Render_Vulkan,
                             "A2 q1 indirect pipeline {:#x}: vi {} attribs {} colour state {} "
                             "untextured {} why '{}' cause {}",
                             std::hash<GraphicsPipelineKey>{}(b->GetGraphicsKey()),
                             draw.vi ? 1 : 0, draw.vi ? draw.vi->count : 0u,
                             u32(b->native_colour_static), u32(b->native_colour_untextured),
                             why ? why : "-", r);
                }
            }
        }
        return refuse(RefuseIndirect);
    }
    const GraphicsPipeline* base = draw.pipeline;
    const GraphicsPipeline* native =
        base ? base->native_variant.load(std::memory_order_acquire) : nullptr;
    // Colour (the back's DrawTail rule): a pipeline without a shadow variant may have a colour
    // variant per vertex description (NativeColourVariant by NativeVertexKey); it must be the
    // bindless + pulled-vertex kind (NativeDirectCandidate).
    bool colour = false;
    // l4b census: "no variant" by the pipeline's colour state - 24 undecided, 25 eligible (no
    // variant for this vertex description yet), 26 ineligible as untextured, 27 ineligible other.
    // l4c: an eligible pipeline's draw by the slot state of its vertex key (0: not computed).
    const auto no_variant = [&](u64 vkey = 0) {
        if (base) {
            const u8 s = base->native_colour_static;
            ++owner->sub[s == 0 ? 24 : s == 1 ? 25 : base->native_colour_untextured ? 26 : 27];
            if (s == 1 && vkey != 0) {
                u32 r = 31;
                for (const auto& slot : base->native_colour) {
                    const u64 k = slot.key.load(std::memory_order_relaxed);
                    if (k == vkey) {
                        r = slot.state.load(std::memory_order_relaxed) == 2 ? 28 : 29;
                        break;
                    }
                    if (k == 0) {
                        r = 30;
                    }
                }
                ++owner->sub[r];
            }
        }
        return refuse(RefuseNoVariant);
    };
    if (!native) {
        if (!base || !colour_on) {
            return no_variant();
        }
        if (!draw.vi || draw.vi->count > 16) {
            return refuse(RefuseNoVi);
        }
        const u64 vkey = VideoCore::NativeVertexKey(AmdGpu::CpSplit::VertexInputAttribs(*draw.vi),
                                                    draw.vi->count);
        native = base->NativeColourVariant(vkey);
        if (!native) {
            native = base->NativeColourSetVariant(vkey); // l4g: null unless SET_VARIANTS
        }
        if (!native) {
            return no_variant(vkey);
        }
        colour = true;
        if (!native->native_bindless ||
            native->native_vertex_binding == GraphicsPipeline::NoNativeRecords) {
            return refuse(RefuseColourGates);
        }
    }
    if (ind && !colour) {
        return refuse(RefuseIndirect); // q2: colour variants only (no vertex rebase)
    }
    if (native->native_records_binding == GraphicsPipeline::NoNativeRecords ||
        native->native_table_binding == GraphicsPipeline::NoNativeRecords ||
        native->native_pt_debug_binding != GraphicsPipeline::NoNativeRecords) {
        return refuse(RefuseVariantKind);
    }
    if (!draw.vi || draw.vi->count > 16) {
        return refuse(RefuseNoVi);
    }
    if (!draw.stages) {
        return refuse(RefuseStages);
    }
    const Shader::BoundStage& vs = (*draw.stages)[u32(Shader::LogicalStage::Vertex)];
    if (!vs.info) {
        return refuse(RefuseStages);
    }
    if (colour) {
        // The colour variant numbers its bindings over {Fragment, Vertex} only (pipeline cache),
        // and BindlessResolve needs the fragment stage: no other stage may be bound.
        if (!(*draw.stages)[u32(Shader::LogicalStage::Fragment)].info) {
            return refuse(RefuseColourStages);
        }
        for (u32 s = 0; s < Shader::MaxStageTypes - 1; ++s) {
            if (s != u32(Shader::LogicalStage::Fragment) && s != u32(Shader::LogicalStage::Vertex) &&
                (*draw.stages)[s].info) {
                return refuse(RefuseColourStages);
            }
        }
    }
    // r9 (SHADPS4_NR_WORKER=1): no job record - this translator's own register state gives every
    // field (no inherited registers: the worker's mirror is complete) and its own ring holds the
    // GPU record (written at the commit below). Run / union: every draw is its own head; a
    // sub-batch continues on this ring's consecutive entries while the members' Guest ranges fit
    // the opener's buffers (the fallback rule below).
    const int wm = NrWorkerMode();
    const bool wrec = wm == 1;
    u32 slot = 0;
    const NR::Record* ring = nullptr;
    const NR::RunUnion* unions = nullptr;
    u64 ring_handle = 0;
    u32 mask = NR::RingSize - 1;
    u32 idx = 0;
    u32 run_k = 0, head_idx = 0, union_idx = 0;
    u32 index_count = 0, io_ud = 0, vcount = 0;
    u64 index_addr_rec = 0;
    std::array<u64, 16> vaddr{};
    u32 push_inh = 0, inh_vs = 0, inh_ps = 0;
    u64 inh_slots = 0;
    u32 img_n_rec = 0;
    u64 img_h1_rec = 0, img_h2_rec = 0;
    u32 n_inst = regs.num_instances.NumInstances();
    if (wrec) {
        if (!rec_mapped) {
            return refuse(RefuseNoRing);
        }
        if (!RecReserve()) {
            ++nrw[8];
            return refuse(RefuseArena);
        }
        ring_handle = rec_handle;
        mask = WorkerRecMask;
        idx = static_cast<u32>(rec_head);
        index_count = regs.num_indices;
        if (!ind && index_count == 0) {
            ++owner->sub[2];
            return refuse(RefuseRecordFields);
        }
    } else {
        if (draw.job_tag == 0) {
            return refuse(RefuseNoJob);
        }
        slot = draw.job_tag >> 28;
        if (slot > NR::MaxRecordSlot) {
            return refuse(RefuseNoRing);
        }
        ring = NR::g_rings[slot].load(std::memory_order_acquire);
        unions = NR::g_unions[slot].load(std::memory_order_acquire);
        ring_handle = NR::g_gpu_handles[slot].load(std::memory_order_acquire);
        if (!ring || !unions || ring_handle == 0) {
            return refuse(RefuseNoRing);
        }
        idx = draw.job_idx;
        // Ring lifetime (NativeRingPrepare fail 2): the job thread must not be a quarter ring ahead.
        if (NR::g_job_next[slot].load(std::memory_order_relaxed) - idx > NR::LifetimeWindow()) {
            return refuse(RefuseLifetime);
        }
        // The job record (seqlock: fields, acquire fence, stamp).
        const NR::Record& r = ring[idx & mask];
        run_k = r.run_k;
        head_idx = r.head_idx;
        const u32 gpu_ok = r.gpu_ok;
        union_idx = r.union_idx;
        const u32 inst_known = r.inst_known, instances = r.instances;
        index_count = r.index_count;
        io_ud = r.io_ud;
        vcount = r.vcount;
        index_addr_rec = r.index_addr;
        vaddr = r.vaddr;
        const void* cmd = r.cmd;
        push_inh = r.push_inh;
        inh_vs = r.inh_vs;
        inh_ps = r.inh_ps;
        inh_slots = r.inh_slots;
        img_n_rec = r.img_n;
        img_h1_rec = r.img_h1;
        img_h2_rec = r.img_h2;
        std::atomic_thread_fence(std::memory_order_acquire);
        if (const u32 stamp = r.stamp.load(std::memory_order_relaxed); stamp != idx + 1 || !gpu_ok) {
            ++owner->sub[stamp != idx + 1 ? (s32(stamp - (idx + 1)) < 0 ? 12 : 13) : 14];
            return refuse(RefuseStamp);
        }
        if (draw.job_cmd) {
            if (cmd != draw.job_cmd) {
                ++owner->sub[15];
                return refuse(RefuseStamp);
            }
        } else if (!cmd_warned) {
            cmd_warned = true;
            ++owner->refusals[RefuseNoCmd];
        }
        // q2: an indirect record has no index count, and its instance-offset register is the
        // CP's (the args' firstInstance, as DrawIndirectTail).
        if ((!ind && (io_ud != 0 || index_count == 0)) ||
            (push_inh >> Shader::NUM_USER_DATA_REGS) != 0) {
            ++owner->sub[io_ud != 0 ? 1 : index_count == 0 ? 2 : 3];
            return refuse(RefuseRecordFields);
        }
        // The record knows the instance count only after a NUM_INSTANCES packet in the same DCB;
        // the mirror always has it (as the generic draw path reads it).
        n_inst = inst_known ? instances : regs.num_instances.NumInstances();
        owner->sub[0] += inst_known ? 0 : 1; // census: instance count taken from the mirror
    }
    // Inherited Guest slots (l3e): every member binds the opener's V# through the table (Base =
    // 0 - offset, Size, the InhSlots bit, PtLo / PtHi), as NativeRingPrepare does; resolved in the
    // union loop below.

    // The mirror against the record (NativePrepConfirm refuse 12 / 13, Opus G2 #3).
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const u32 index_size = is_index16 ? 2u : 4u;
    const VAddr index_addr =
        regs.index_base_address.Address<VAddr>() +
        (ind ? 0 : u64(draw.index_offset) * index_size);
    if (!wrec && !ind && (index_addr != index_addr_rec || regs.num_indices != index_count ||
                 regs.num_instances.NumInstances() != n_inst)) {
        return refuse(RefuseRecordRegs);
    }
    // q2: the args (20-byte VkDrawIndexedIndirectCommand layout, 4-byte aligned) in one cached
    // buffer; the index range = INDEX_BUFFER_SIZE indices from INDEX_BASE (as BindIndexBuffer).
    u64 ind_handle = 0, ind_off = 0;
    if (ind) {
        const VAddr aa = draw.arg_address;
        const VideoCore::Buffer* ab =
            aa != 0 && (aa & 3) == 0 ? ctx.buffer_cache->A2LookupBuffer(aa) : nullptr;
        if (!ab || !ab->IsInBounds(aa, CmdBytes) || regs.num_indices == 0) {
            return refuse(RefuseIndirect);
        }
        ind_handle = std::bit_cast<u64>(static_cast<VkBuffer>(ab->Handle()));
        ind_off = aa - ab->CpuAddr();
    }
    const auto* attribs = AmdGpu::CpSplit::VertexInputAttribs(*draw.vi);
    const u32 vb_n = draw.vi->count;
    if (!wrec && vcount != vb_n) {
        return refuse(RefuseRecordRegs);
    }
    std::array<VAddr, 16> vaddr_vi{};
    for (u32 k = 0; k < vb_n; ++k) {
        const auto& a = attribs[k];
        vaddr_vi[k] = (a.base_address != 0 && a.size != 0) ? a.base_address : 0;
        if (!wrec && vaddr_vi[k] != vaddr[k]) {
            return refuse(RefuseRecordRegs);
        }
    }
    auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs, base->GetFetchShader());
    if (ind) {
        vertex_offset = 0; // q2: from the args (Vulkan applies vertexOffset / firstInstance)
        instance_offset = 0;
    }
    if (instance_offset != 0) {
        return refuse(RefuseInstanceOffset); // Opus code review #2 (drawIndirectFirstInstance)
    }

    // Set 0 in BindResources order (LogicalStage index order over the stages): every binding
    // must be fillable without the caches - Guest V#s (never read under PT: the dummy buffer),
    // unused clip planes (a null descriptor); then the records SSBO (the job ring), (colour) the
    // vertex SSBO, and the table. Colour: images and samplers (fragment stage only) take binding
    // numbers but no set-0 write (the bindless PS samples set 1 at the record's slots); their raw
    // T# / S# words form the draw's material key exactly as NativeImageKey builds it (stage index
    // order, images then samplers).
    Shader::Backend::Bindings bnd{};
    Shader::PushData push = MakePushData(regs);
    std::array<SubBatch::Write, Shader::NUM_BUFFERS + 4> writes{};
    u32 num_writes = 0;
    u64 guest_mask = 0;
    std::array<VAddr, Shader::NUM_BUFFERS> o_addr{};
    std::array<u64, Shader::NUM_BUFFERS> o_vsize{};
    bool clip_on = false;
    u32 clip_wi = 0;
    u64 fmt_mask = 0;
    std::array<u32, 24> planes{};
    ImageKey key{};
    u32 bi = 0;
    u32 tot_img = 0, tot_smp = 0; // q3
    // q8: the material's images / samplers in slot order (FS then VS) for the table lookup.
    std::array<std::pair<const Shader::ImageResource*, const Shader::BoundStage*>, 16> tl_img{};
    std::array<std::pair<const Shader::SamplerResource*, const Shader::BoundStage*>, 16> tl_smp{};
    u32 tl_ni = 0, tl_ns = 0;
    for (u32 s = 0; s < Shader::MaxStageTypes - 1; ++s) {
        const Shader::BoundStage& bound = (*draw.stages)[s];
        const Shader::Info* info = bound.info;
        if (!info) {
            continue;
        }
        if (!colour && (!info->images.empty() || !info->samplers.empty())) {
            ++owner->sub[4];
            return refuse(RefuseSlotKind);
        }
        if (colour && s != u32(Shader::LogicalStage::Fragment) &&
            (!info->images.empty() || !info->samplers.empty()) &&
            !(s == u32(Shader::LogicalStage::Vertex) && native->native_bindless_vs)) {
            return refuse(RefuseColourStages); // BindlessResolve resolves the FS (+ q3 VS)
        }
        if (bnd.user_data + info->ud_mask.NumRegs() > Shader::NUM_USER_DATA_REGS) {
            return refuse(RefuseLayout);
        }
        bound.PushUd(bnd, push);
        for (const auto& desc : info->buffers) {
            if (bi >= Shader::NUM_BUFFERS || num_writes >= Shader::NUM_BUFFERS) {
                return refuse(RefuseLayout);
            }
            if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                if (regs.clipper_control.user_clip_plane_enable != 0) {
                    // As BindResources: a copy of the planes (the opener's region, below).
                    clip_on = true;
                    clip_wi = num_writes;
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = regs.clip_user_data[i];
                        planes[i * 4 + 0] = plane.data_x;
                        planes[i * 4 + 1] = plane.data_y;
                        planes[i * 4 + 2] = plane.data_z;
                        planes[i * 4 + 3] = plane.data_w;
                    }
                }
                writes[num_writes++] = {bnd.unified++, vk::Buffer{}, 0, VK_WHOLE_SIZE};
                ++bnd.buffer;
                ++bi;
                continue;
            }
            // Formatted (typed) reads on colour draws: lowered to raw loads in the variant (the
            // pipeline cache allows them for colour only); the back syncs the range at publish.
            const bool fmt_ok = colour && desc.is_formatted &&
                                desc.buffer_type == Shader::BufferType::Guest && !desc.is_written;
            if (desc.buffer_type != Shader::BufferType::Guest || desc.is_written ||
                (desc.is_formatted && !fmt_ok)) {
                ++owner->sub[desc.buffer_type != Shader::BufferType::Guest ? 6
                             : desc.is_written                             ? 7
                                                                           : 8];
                return refuse(RefuseSlotKind);
            }
            const auto vsharp = desc.GetSharp(bound);
            if (colour && (desc.inline_cbuf || vsharp.base_address == 0 || vsharp.GetSize() == 0)) {
                ++owner->sub[9];
                return refuse(RefuseSlotKind); // NativeDirectBind fail 5 / 6
            }
            o_addr[bi] = vsharp.base_address;
            o_vsize[bi] = vsharp.GetSize();
            guest_mask |= u64{1} << bi;
            if (fmt_ok) {
                fmt_mask |= u64{1} << bi;
                bool known = false;
                if (stretch) {
                    for (const auto& [ta, ts] : stretch->texel) {
                        known = known || (ta == o_addr[bi] && ts == o_vsize[bi]);
                    }
                }
                if (!known && stretch && stretch->texel.size() >= 16) {
                    ++owner->sub[21];
                    return refuse(RefuseSlotKind);
                }
            }
            writes[num_writes++] = {bnd.unified++, dummy, 0, VK_WHOLE_SIZE};
            ++bnd.buffer;
            ++bi;
        }
        if (colour) {
            // BindlessResolve's own pure checks (bl_fail 0 / 1) and the numbering rule: the
            // variant counts one binding per image (Info::AddBindings), NativeDirectBind one per
            // mip level under the DynamicIndex fallback - refuse that mode rather than guess.
            tot_img += static_cast<u32>(info->images.size()); // q3: FS + VS <= 16 each
            tot_smp += static_cast<u32>(info->samplers.size());
            if (tot_img > Shader::g_bl_img_max || tot_smp > Shader::g_bl_smp_max) { // q54 split
                return refuse(RefuseColourImages);
            }
            for (const auto& desc : info->images) {
                if (desc.is_written || desc.is_atomic ||
                    desc.mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                    return refuse(RefuseColourImages);
                }
                const auto tsharp = desc.GetSharp(bound);
                if (tsharp.Address() == 0 ||
                    tsharp.GetDataFmt() == AmdGpu::DataFormat::FormatInvalid) {
                    return refuse(RefuseColourImages);
                }
                constexpr u32 words = static_cast<u32>(sizeof(tsharp) / sizeof(u32));
                if (key.n + words > key.w.size()) {
                    return refuse(RefuseColourImages);
                }
                std::memcpy(&key.w[key.n], &tsharp, sizeof(tsharp));
                key.n += words;
                if (tl_ni < tl_img.size()) {
                    tl_img[tl_ni++] = {&desc, &bound}; // q8
                }
            }
            for (const auto& desc : info->samplers) {
                if (tl_ns < tl_smp.size()) {
                    tl_smp[tl_ns++] = {&desc, &bound}; // q8
                }
                const auto ssharp = desc.GetSharp(bound);
                constexpr u32 words = static_cast<u32>(sizeof(ssharp) / sizeof(u32));
                if (key.n + words > key.w.size()) {
                    return refuse(RefuseColourImages);
                }
                std::memcpy(&key.w[key.n], &ssharp, sizeof(ssharp));
                key.n += words;
            }
            bnd.unified += static_cast<u32>(info->images.size() + info->samplers.size());
        }
    }
    if (colour) {
        if (native->native_records_binding != bnd.unified ||
            native->native_vertex_binding != bnd.unified + 1 ||
            native->native_table_binding != bnd.unified + 2 || guest_mask == 0) {
            return refuse(RefuseLayout);
        }
        // The job record's image count / key hashes must agree with the stages (NativeRingPrepare
        // fail 13), unless a T# / S# sat in inherited registers (zeros in the job's key).
        if (!wrec && (img_n_rec & ~NR::ImgInhBit) != key.n) {
            return refuse(RefuseColourKeyHash);
        }
        if (!wrec && !(img_n_rec & NR::ImgInhBit)) {
            u64 h1 = 0, h2 = 0;
            NR::ImageKeyHash(key.w.data(), key.n, h1, h2);
            if (h1 != img_h1_rec || h2 != img_h2_rec) {
                return refuse(RefuseColourKeyHash);
            }
        }
    } else if (native->native_records_binding != bnd.unified ||
               native->native_table_binding != bnd.unified + 1 || guest_mask == 0) {
        return refuse(RefuseLayout);
    }
    push.buf_offsets = {}; // whole-range bindings: the record carries each draw's offset

    // Index buffer: the whole index range in one cached buffer, bound at offset 0.
    // r4 (SHADPS4_IB_INDIRECT_CAP): an indirect draw's range is the INDEX_BUFFER_SIZE capacity.
    u32 ind_count = regs.num_indices;
    if (ind) {
        if (const u32 cap = VideoCore::IndirectIndexCapacity(regs, 1); cap != 0) {
            ind_count = cap;
        }
    }
    const VideoCore::Buffer* ib = ctx.buffer_cache->A2LookupBuffer(index_addr);
    if (!ib || !ib->IsInBounds(index_addr, u64(ind ? ind_count : index_count) * index_size)) {
        return refuse(RefuseIndexLookup);
    }
    const u64 ib_off = index_addr - ib->CpuAddr();
    if (ib_off % index_size != 0) {
        return refuse(RefuseRecordRegs); // NativeRingPrepare why 11
    }
    const u64 ib_handle = std::bit_cast<u64>(static_cast<VkBuffer>(ib->Handle()));
    // Vertex streams: each non-null stream's buffer and its offset in it. Colour (NativeOpen's
    // pulled-vertex rule): per-vertex attributes at locations < 16 only, every non-null stream
    // inside ONE VkBuffer, offsets below 2^32, at least one stream (the back binds a
    // stream-buffer stand-in when all are null: not reproduced, refused); each stream's whole
    // [base, base + size) inside the buffer (NativeRingPrepare fail 7 checks the run's span).
    std::array<u64, 16> vb_handle{}, vcpu{}, voff{};
    u64 vbuf = 0;
    u32 vmask = 0;
    u32 vptmask = 0; // q18: locations pulled through the page table (SHADPS4_A2_PT_VERTEX)
    // q53 (SHADPS4_A2_VBUF2): a second distinct vertex VkBuffer goes to the variant's second vertex SSBO (its
    // locations in v2mask); a third is refused (or page-table pulled with q18).
    const bool v2_on = colour && native->native_vertex2_binding != GraphicsPipeline::NoNativeRecords;
    u64 vbuf2 = 0;
    u32 v2mask = 0;
    for (u32 k = 0; k < vb_n; ++k) {
        const u32 loc = attribs[k].desc.location;
        if (colour && (attribs[k].desc.instanced || loc >= Shader::NativeMaxPulledLocations)) {
            ++owner->sub[44]; // q4: instanced / location
            return refuse(RefuseColourVertex);
        }
        if (vaddr_vi[k] == 0) {
            continue;
        }
        const VideoCore::Buffer* vb = ctx.buffer_cache->A2LookupBuffer(vaddr_vi[k]);
        if (!vb) {
            return refuse(RefuseVertexLookup);
        }
        vb_handle[k] = std::bit_cast<u64>(static_cast<VkBuffer>(vb->Handle()));
        vcpu[k] = vb->CpuAddr();
        voff[k] = vaddr_vi[k] - vcpu[k];
        if (colour) {
            bool pt_loc = false;
            bool v2_loc = false;
            const bool other = vbuf != 0 && vb_handle[k] != vbuf;
            const bool third = other && v2_on && vbuf2 != 0 && vb_handle[k] != vbuf2;
            if (other && v2_on && !third && voff[k] <= 0xffffffffull) {
                v2_loc = true; // q53: the second vertex SSBO
            } else if (other || voff[k] > 0xffffffffull) {
                if (Shader::g_pt_vertex == 0) {
                    // q4: 2nd buffer / offset; q53: 3rd buffer
                    ++owner->sub[third ? 48 : (other && !v2_on) ? 45 : 46];
                    return refuse(RefuseColourVertex);
                }
                pt_loc = true; // q18: pulled through the page table from its full guest address
            }
            if (!vb->IsInBounds(vaddr_vi[k], attribs[k].size)) {
                return refuse(RefuseColourStream); // NativeRingPrepare fail 7 (the run's span)
            }
            if (pt_loc) {
                // q18: the page-table fetch reads the same bytes as a direct bind only if the stream's last byte
                // maps to the same cached buffer and that buffer starts on a page (entry = BDA + page distance).
                const u64 last = vaddr_vi[k] + std::max<u64>(attribs[k].size, 1) - 1;
                if (ctx.buffer_cache->A2LookupBuffer(last) != vb ||
                    (vb->CpuAddr() & (VideoCore::BufferCache::CACHING_PAGESIZE - 1)) != 0) {
                    ++owner->sub[45];
                    return refuse(RefuseColourVertex);
                }
                vptmask |= 1u << loc;
            } else if (v2_loc) {
                vbuf2 = vb_handle[k];
                v2mask |= 1u << loc;
            } else {
                vbuf = vb_handle[k];
            }
            vmask |= 1u << loc;
        }
    }
    if (colour && vbuf == 0) {
        ++owner->sub[47]; // q4: every stream null (or every stream page-table pulled: the vertex SSBO needs one)
        return refuse(RefuseColourVertex);
    }
    if (vptmask != 0) {
        // q18: the shader reads the mask from buf_offsets[0..1] (native shaders never read buf_offsets).
        push.buf_offsets[0] = static_cast<u8>(vptmask & 0xff);
        push.buf_offsets[1] = static_cast<u8>(vptmask >> 8);
    }
    if (v2mask != 0) {
        // q53: the shader reads the second-SSBO mask from buf_offsets[2..3] (bits 16-31 of the push word).
        push.buf_offsets[2] = static_cast<u8>(v2mask & 0xff);
        push.buf_offsets[3] = static_cast<u8>(v2mask >> 8);
    }

    // Dynamic state of this draw on top of the tracked state (a copy: a refusal below must not
    // change what the open sub-batch recorded). The first draw of a stretch starts from an
    // invalidated state, so its sub-batch records everything.
    DynamicState cand = dyn;
    if (!stretch) {
        cand.Invalidate();
    }
    if (!ComputeDynamicState(cand, *ctx.instance, regs, native)) {
        return refuse(RefuseRestartIndex);
    }

    // Colour: the draw's material in the stretch's list (dedup by base pipeline + key words), or
    // a new one if the list has room. The list itself changes only when the draw is taken.
    s32 mat = -1;
    if (colour) {
        const u32 nm = stretch ? static_cast<u32>(stretch->materials.size()) : 0;
        for (u32 m = 0; m < nm; ++m) {
            if (mat_base[m] == base && mat_keys[m].n == key.n &&
                std::memcmp(mat_keys[m].w.data(), key.w.data(), sizeof(u32) * key.n) == 0) {
                mat = static_cast<s32>(m);
                break;
            }
        }
        if (mat < 0 && nm >= MaxMaterials) {
            return refuse(RefuseMaterials);
        }
    }

    // Can the draw continue the open sub-batch? Keys: the variant, the next ring entry of the
    // same job slot and run, the inherited set and the inherited registers' values, the same
    // index / vertex VkBuffers and strides, the same push constants, vertex description and
    // dynamic state, every Guest V# inside the buffers the opener's table bases name, the
    // vertex rebase quotient rule (below), fewer than MaxDraws draws. Colour: the same vertex
    // VkBuffer and the same set of non-null locations (NativeJoin's `had` rule); the material
    // may differ (each draw's record gets its own slot words).
    const u32 head = wrec ? 0u : (run_k == 0 ? idx : head_idx); // r9: no runs
    bool cont = !ind && !b_ind && batch_open && b_native == native && b_colour == colour &&
                b_slot == slot &&
                idx == b_next_idx && cmds->batches.back().n < MaxDraws && union_idx == b_union &&
                head == b_head && push_inh == b_push_inh && inh_vs == b_inh_vs &&
                inh_ps == b_inh_ps && ib_handle == b_ib && index_size == b_index_size &&
                vb_n == b_vb_n && draw.vi->desc_id == last_desc_id && PushKeySame(push, b_push) &&
                guest_mask == b_guest_mask && !AnyDirty(cand) &&
                (!colour || (vbuf == b_vbuf && vmask == b_vmask && vptmask == b_vptmask &&
                             vbuf2 == b_vbuf2 && v2mask == b_v2mask)) &&
                inh_slots == b_inh_slots &&
                clip_on == b_clip && (!clip_on || planes == b_planes) && fmt_mask == b_fmt_mask;
    for (u64 m = fmt_mask; cont && m != 0; m &= m - 1) {
        // The back's member rule: a formatted slot binds the opener's own V#.
        const u32 b = static_cast<u32>(std::countr_zero(m));
        cont = o_addr[b] == b_iaddr[b] && o_vsize[b] == b_isize[b];
    }
    for (u64 m = inh_slots; cont && m != 0; m &= m - 1) {
        const u32 b = static_cast<u32>(std::countr_zero(m));
        cont = o_addr[b] == b_iaddr[b] && o_vsize[b] == b_isize[b];
    }
    for (u32 m = inh_vs; cont && m != 0; m &= m - 1) {
        const u32 i = static_cast<u32>(std::countr_zero(m));
        cont = i < 16 && regs.vs_program.user_data[i] == b_ud_vs[i];
    }
    for (u32 m = inh_ps; cont && m != 0; m &= m - 1) {
        const u32 i = static_cast<u32>(std::countr_zero(m));
        cont = i < 16 && regs.ps_program.user_data[i] == b_ud_ps[i];
    }
    for (u32 k = 0; cont && k < vb_n; ++k) {
        cont = vb_handle[k] == b_vb[k] && attribs[k].desc.stride == b_stride[k] &&
               u8(attribs[k].desc.instanced) == b_inst[k];
    }
    for (u64 m = guest_mask; cont && m != 0; m &= m - 1) {
        const u32 b = static_cast<u32>(std::countr_zero(m));
        cont = o_addr[b] >= b_glo[b] && o_addr[b] < b_ghi[b] &&
               (!(b_fb_mask & (u64{1} << b)) || o_addr[b] + o_vsize[b] <= b_ghi[b]) &&
               (!(b_big_mask & (u64{1} << b)) ||
                (o_addr[b] == b_iaddr[b] && o_vsize[b] == b_isize[b]));
    }
    // The member's vertexOffset against the opener's rebased bindings (NativeRingPrepare's
    // d / q loop): one common quotient over the per-vertex streams; instance-rate and
    // stride-0 streams may not move. A failure ends the sub-batch, not the stretch. Colour: no
    // rebase (the record carries each location's offset), vertexOffset as is.
    s64 vo = 0;
    if (cont && colour) {
        vo = s64(s32(vertex_offset));
        cont = vo >= 0 && vo <= INT32_MAX;
    } else if (cont) {
        s64 q = 0;
        bool q_set = false;
        bool ok = true;
        for (u32 k = 0; ok && k < vb_n; ++k) {
            if (vb_handle[k] == 0) {
                continue; // null here and (same handles checked) null at the opener
            }
            const s64 d = s64(voff[k]) - s64(b_vb_off[k]);
            if (attribs[k].desc.instanced || attribs[k].desc.stride == 0) {
                ok = d == 0;
                continue;
            }
            const s64 stride = s64(attribs[k].desc.stride);
            if (d % stride != 0) {
                ok = false;
                break;
            }
            const s64 qa = d / stride;
            if (q_set && qa != q) {
                ok = false;
                break;
            }
            q = qa;
            q_set = true;
        }
        vo = s64(s32(vertex_offset)) + q;
        cont = ok && vo >= 0 && vo <= INT32_MAX;
    }

    // Open a sub-batch (and the stretch if none is open).
    if (!cont) {
        // Rebase (NativeOpen's !colour block): the per-vertex strided streams to the lowest
        // offset of their stride residue if they share one quotient, else the full offsets.
        // Colour: no rebase (NativeOpen binds the opener's offsets as they are).
        std::array<u64, 16> bo = voff;
        u64 q0 = 0;
        if (!colour) {
            bool ok = true, any = false;
            u64 qq = 0;
            for (u32 k = 0; k < vb_n; ++k) {
                if (vb_handle[k] == 0 || attribs[k].desc.instanced || attribs[k].desc.stride == 0) {
                    continue;
                }
                const u64 qk = voff[k] / attribs[k].desc.stride;
                if (any && qk != qq) {
                    ok = false;
                }
                qq = qk;
                any = true;
            }
            if (ok && any && qq <= 0x7fffffffull) {
                for (u32 k = 0; k < vb_n; ++k) {
                    if (vb_handle[k] == 0 || attribs[k].desc.instanced ||
                        attribs[k].desc.stride == 0) {
                        continue;
                    }
                    bo[k] = voff[k] % attribs[k].desc.stride;
                }
                q0 = qq;
            }
        }
        vo = s64(s32(vertex_offset)) + s64(q0);
        if (vo < 0 || vo > INT32_MAX) {
            return refuse(RefuseVertexOffset);
        }
        // The run union (NativeRingPrepare): per Guest slot the buffer holding the run's range
        // (its CPU address low 32 bits is the table base), or the opener's own V# when every
        // draw of the run binds the same one. Seqlock as the back reads it.
        // r9 (worker records): no run union - every Guest slot takes the opener's own V# inside one
        // cached buffer (the fallback below); members must fit those buffers (the continuation check).
        u64 u_guest = guest_mask, u_eq = 0;
        std::array<u64, Shader::NUM_BUFFERS> lo{}, hi{}, first{};
        if (!wrec) {
            const NR::RunUnion& u = unions[union_idx & NR::UnionMask];
            const u32 un = u.n.load(std::memory_order_acquire);
            if (u.stamp.load(std::memory_order_relaxed) != head + 1 || un <= run_k) {
                return refuse(RefuseUnion);
            }
            u_guest = u.guest_mask;
            u_eq = u.eq_mask;
            for (u64 m = guest_mask; m != 0; m &= m - 1) {
                const u32 b = static_cast<u32>(std::countr_zero(m));
                lo[b] = u.lo[b];
                hi[b] = u.hi[b];
                first[b] = u.first_addr[b];
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            if (u.stamp.load(std::memory_order_relaxed) != head + 1) {
                return refuse(RefuseUnion);
            }
        }
        std::array<u32, Shader::NUM_BUFFERS> base{};
        std::array<VAddr, Shader::NUM_BUFFERS> glo{}, ghi{};
        std::array<u32, Shader::NUM_BUFFERS> i_off{}, i_size{};
        u64 fb_mask = 0;
        u64 big_mask = 0;
        for (u64 m = guest_mask; m != 0; m &= m - 1) {
            const u32 b = static_cast<u32>(std::countr_zero(m));
            if (inh_slots & (u64{1} << b)) {
                // The opener's own V# (no union): one cached buffer must hold all of it.
                const u64 vsz = o_vsize[b];
                if (vsz == 0 || vsz >= (u64{1} << 30)) {
                    ++owner->sub[16];
                    return refuse(RefuseInheritedSlot);
                }
                const VideoCore::Buffer* ibuf = ctx.buffer_cache->A2LookupBuffer(o_addr[b]);
                if (!ibuf || !ibuf->IsInBounds(o_addr[b], vsz)) {
                    ++owner->sub[17];
                    return refuse(RefuseInheritedSlot);
                }
                const u64 ioff = o_addr[b] - ibuf->CpuAddr();
                if (ioff > 0xffffffffull) {
                    ++owner->sub[18];
                    return refuse(RefuseInheritedSlot);
                }
                i_off[b] = static_cast<u32>(ioff);
                i_size[b] = static_cast<u32>(vsz);
                glo[b] = ibuf->CpuAddr();
                ghi[b] = ibuf->CpuAddr() + ibuf->SizeBytes();
                continue;
            }
            if (!(u_guest & (u64{1} << b))) {
                return refuse(RefuseUnion);
            }
            const VideoCore::Buffer* buf = nullptr;
            if ((u_eq & (u64{1} << b)) && first[b] == o_addr[b]) {
                buf = ctx.buffer_cache->A2LookupBuffer(o_addr[b]);
            } else if (lo[b] != 0 && hi[b] > lo[b] && hi[b] - lo[b] <= MaxSpan) {
                buf = ctx.buffer_cache->A2LookupBuffer(lo[b]);
                if (buf && !buf->IsInBounds(lo[b], hi[b] - lo[b])) {
                    buf = nullptr;
                }
            }
            static const bool big_on = [] {
                const char* e = std::getenv("SHADPS4_NR_WORKER_BIG");
                return !(e && e[0] == '0'); // =0: r9's behaviour (screen arm)
            }();
            if (!buf && wrec && big_on && o_vsize[b] > MaxSpan) {
                // r12 (mode 1): a V# larger than the union bound - the game binds its whole mesh
                // memory (~2 GB at 0x271258000) and picks each draw's part by offsets. Mode 0 took
                // these through the union's equal-V# path (the cached buffer holding the address,
                // no size check); mode 1 has no union, so the same rule per run: the opener binds
                // the cached buffer holding its address, members must bind the identical V#
                // (b_big_mask, the continuation check).
                buf = ctx.buffer_cache->A2LookupBuffer(o_addr[b]);
                if (buf && o_addr[b] >= buf->CpuAddr() &&
                    o_addr[b] - buf->CpuAddr() < buf->SizeBytes()) {
                    big_mask |= u64{1} << b;
                    ++nrw[18];
                } else {
                    buf = nullptr;
                }
            }
            if (!buf || o_addr[b] < buf->CpuAddr() ||
                o_addr[b] - buf->CpuAddr() >= buf->SizeBytes()) {
                // Fallback: the opener's own V# inside one cached buffer (members must then fit
                // that buffer whole: the continuation check below).
                const VideoCore::Buffer* fb =
                    o_vsize[b] != 0 && o_vsize[b] <= MaxSpan
                        ? ctx.buffer_cache->A2LookupBuffer(o_addr[b])
                        : nullptr;
                if (fb && fb->IsInBounds(o_addr[b], o_vsize[b])) {
                    buf = fb;
                    fb_mask |= u64{1} << b;
                    ++owner->sub[23];
                } else {
                    ++owner->sub[!buf ? 10 : 11];
                    if (wrec) {
                        // r10 census: why the mode-1 fallback failed.
                        if (o_vsize[b] == 0) {
                            ++nrw[9];
                        } else if (o_vsize[b] > MaxSpan) {
                            ++nrw[10];
                        } else if (!fb) {
                            ++nrw[11];
                        } else if (o_addr[b] < fb->CpuAddr()) {
                            ++nrw[12];
                        } else {
                            const u64 over = o_addr[b] + o_vsize[b] - (fb->CpuAddr() + fb->SizeBytes());
                            ++nrw[over <= 4096 ? 13 : over <= 65536 ? 14 : 15];
                        }
                    }
                    return refuse(RefuseGuestLookup);
                }
            }
            if (wm == 2 && !(fb_mask & (u64{1} << b))) {
                // r10 census (mode 2 runs the job-record path): Guest binds through the run union,
                // and how many of them have an own V# that runs past the cached buffer's end.
                ++nrw[17];
                if (!buf->IsInBounds(o_addr[b], o_vsize[b])) {
                    ++nrw[16];
                }
            }
            base[b] = static_cast<u32>(buf->CpuAddr());
            glo[b] = buf->CpuAddr();
            ghi[b] = buf->CpuAddr() + buf->SizeBytes();
        }
        // The arena region: table, commands, count. The last check that can refuse.
        u64 region_start = 0;
        u64 region_off = 0;
        u8* region = nullptr;
        if (!AllocRegion(region_bytes, region_start, region_off, region)) {
            return refuse(RefuseArena);
        }
        const bool first_batch = !stretch;
        if (first_batch) {
            stretch = std::make_unique<Stretch>();
            stretch->first_seq = draw.seq;
            stretch->rt_pipe = native; // p2: the pass formats of the secondary's inheritance
            stretch->tick = std::make_shared<std::atomic<u64>>(0);
            rec_stretch_start = rec_head; // r9
            stretch->buf_gen = ctx.buffer_cache->StructGeneration();
            stretch->probe_gen = VideoCore::g_lprobe_gen.load(std::memory_order_acquire); // q9b
            cmds = cmds_pool ? TakePooledCmds() : std::make_shared<StretchCmds>();
            cmds->batches.reserve(perdraw ? batch_hint : 8);
            stretch->cmds = cmds;
            stretch_start = region_start;
            last_desc_id = 0;
            b_native = nullptr;
            batch_open = false;
        }
        // The table (NativeRingPrepare + NativeOpen's ring-mode words; SegMat 0; InhSlots / Size /
        // PtLo / PtHi only for inherited slots, as NativeRingPrepare writes them).
        auto* table = reinterpret_cast<u32*>(region);
        std::memset(table, 0, TableBytes);
        table[Shader::NativeTableFirst] = idx & mask;
        table[Shader::NativeTableMask] = mask;
        table[Shader::NativeTablePtFallback] = static_cast<u32>(pt_fallback);
        table[Shader::NativeTablePtFallback + 1] = static_cast<u32>(pt_fallback >> 32);
        table[Shader::NativeTableFaultAddr] = static_cast<u32>(fault_addr);
        table[Shader::NativeTableFaultAddr + 1] = static_cast<u32>(fault_addr >> 32);
        table[Shader::NativeTablePushInh] = push_inh;
        for (u32 pm = push_inh; pm != 0; pm &= pm - 1) {
            const u32 pi = static_cast<u32>(std::countr_zero(pm));
            table[Shader::NativeTableUd + pi] = push.ud_regs[pi];
        }
        for (u64 m = guest_mask; m != 0; m &= m - 1) {
            const u32 b = static_cast<u32>(std::countr_zero(m));
            if (inh_slots & (u64{1} << b)) {
                table[Shader::NativeTableBase + b] = 0u - i_off[b];
                table[Shader::NativeTableSize + b] = i_size[b];
                table[Shader::NativeTableInhSlots + b / 32] |= 1u << (b % 32);
                table[Shader::NativeTablePtLo + b] = static_cast<u32>(o_addr[b]);
                table[Shader::NativeTablePtHi + b / 4] |=
                    static_cast<u32>((o_addr[b] >> 32) & 0xffu) << ((b % 4) * 8);
                continue;
            }
            table[Shader::NativeTableBase + b] = base[b];
        }
        if (colour) {
            // NativeRingPrepare's per-location bases: the vertex VkBuffer's CPU address (the GPU
            // record holds each stream's guest address; record - base = the offset). SegMat
            // stays 0: the slots come from the record words.
            for (u32 k = 0; k < vb_n; ++k) {
                if (vb_handle[k] != 0) {
                    table[Shader::NativeTableVbase + attribs[k].desc.location] =
                        static_cast<u32>(vcpu[k]);
                }
            }
        }
        const u64 cmd_off = region_off + TableBytes;
        const u64 count_off = cmd_off + u64(MaxDraws) * CmdBytes;
        cmd_ptr = region + TableBytes;
        count_ptr = reinterpret_cast<u32*>(region + TableBytes + u64(MaxDraws) * CmdBytes);
        *count_ptr = 0;

        // The sub-batch record: what Replay emits before its indirect draw.
        if (perdraw && cmds->batches.size() == cmds->batches.capacity()) {
            cmds->batches.reserve(cmds->batches.capacity() * 2); // one move per doubling
        }
        SubBatch& sb = cmds->batches.emplace_back();
        const bool ib_changed =
            first_batch || ib_handle != b_ib || index_size != b_index_size || ind || b_ind;
        bool vb_changed = first_batch || vb_n != b_vb_n;
        for (u32 k = 0; !vb_changed && k < vb_n; ++k) {
            vb_changed = vb_handle[k] != b_vb[k] || bo[k] != b_vb_off[k];
        }
        sb.emit_pipeline = first_batch || native != b_native;
        sb.emit_push = first_batch || !PushKeySame(push, b_push) || vptmask != b_vptmask || // q18
                       v2mask != b_v2mask;                                                    // q53
        sb.emit_descriptors = true; // the table is a new region every sub-batch
        sb.emit_vertex_input = first_batch || draw.vi->desc_id != last_desc_id;
        sb.emit_index = ib_changed;
        sb.emit_vertex_buffers = vb_changed;
        sb.bindless = colour;
        sb.pipeline = native->Handle();
        sb.layout = native->GetLayout();
        sb.set_owner = native->UsesPushDescriptors() ? nullptr : native; // l4g
        sb.push = push;
        writes[num_writes++] = {native->native_records_binding,
                                vk::Buffer{std::bit_cast<VkBuffer>(ring_handle)}, 0, VK_WHOLE_SIZE};
        if (colour) {
            // The whole vertex VkBuffer (NativeOpen's colour write, before the table).
            writes[num_writes++] = {native->native_vertex_binding,
                                    vk::Buffer{std::bit_cast<VkBuffer>(vbuf)}, 0, VK_WHOLE_SIZE};
        }
        writes[num_writes++] = {native->native_table_binding, ctx.arena->buffer, region_off,
                                TableBytes};
        if (v2_on) {
            // q53: the second vertex VkBuffer (the first one when the draw has none), the last binding.
            writes[num_writes++] = {native->native_vertex2_binding,
                                    vk::Buffer{std::bit_cast<VkBuffer>(vbuf2 != 0 ? vbuf2 : vbuf)}, 0,
                                    VK_WHOLE_SIZE};
        }
        if (clip_on) {
            std::memcpy(region + clip_off_in_region, planes.data(), ClipBytes);
            writes[clip_wi].buffer = ctx.arena->buffer;
            writes[clip_wi].offset = region_off + clip_off_in_region;
            writes[clip_wi].range = ClipBytes;
        }
        sb.num_writes = num_writes;
        sb.writes = writes;
        BuildVertexInputArrays(*draw.vi, sb);
        sb.ib = vk::Buffer{std::bit_cast<VkBuffer>(ib_handle)};
        if (ind) {
            sb.ib_off = ib_off; // q2: firstIndex in the args is relative to INDEX_BASE
            sb.ind_buf = vk::Buffer{std::bit_cast<VkBuffer>(ind_handle)};
            sb.ind_off = ind_off;
        }
        sb.ib_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
        sb.vb_n = vb_n;
        for (u32 k = 0; k < vb_n; ++k) {
            sb.vb[k] = vk::Buffer{std::bit_cast<VkBuffer>(vb_handle[k])};
            sb.vb_off[k] = vb_handle[k] != 0 ? bo[k] : 0;
        }
        sb.dyn = cand; // dirty bits = what changed (everything for the first sub-batch)
        {
            // p2b (review #1): Commit skips a value while its enable is off but its dirty bit is
            // cleared below anyway; when an enable changes, record its dependent values too (a
            // secondary inherits nothing).
            auto& d = sb.dyn.dirty_state;
            if (d.depth_test_enabled) {
                d.depth_compare_op = true;
            }
            if (d.stencil_test_enabled) {
                d.stencil_front_ops = d.stencil_front_reference = true;
                d.stencil_front_write_mask = d.stencil_front_compare_mask = true;
                d.stencil_back_ops = d.stencil_back_reference = true;
                d.stencil_back_write_mask = d.stencil_back_compare_mask = true;
            }
            if (d.depth_bias_enabled) {
                d.depth_bias = true;
            }
            if (d.depth_bounds_test_enabled) {
                d.depth_bounds = true;
            }
        }
        sb.arena = ctx.arena->buffer;
        sb.cmd_off = cmd_off;
        sb.count_off = count_off;
        sb.n = 0;
        ClearDirty(cand);

        // The keys of the now-open sub-batch.
        batch_open = true;
        b_native = native;
        b_colour = colour;
        b_vbuf = vbuf;
        b_vmask = vmask;
        b_vptmask = vptmask;
        b_vbuf2 = vbuf2;
        b_v2mask = v2mask;
        if (vptmask != 0) [[unlikely]] {
            static std::atomic<u64> pt_batches{0};
            const u64 n = pt_batches.fetch_add(1, std::memory_order_relaxed) + 1;
            if ((n & (n - 1)) == 0) {
                LOG_INFO(Render_Vulkan, "q18: {} colour sub-batches with page-table vertex streams", n);
            }
        }
        b_slot = slot;
        b_union = union_idx;
        b_head = head;
        b_push_inh = push_inh;
        b_inh_vs = inh_vs;
        b_inh_ps = inh_ps;
        b_ud_vs = regs.vs_program.user_data;
        b_ud_ps = regs.ps_program.user_data;
        b_ib = ib_handle;
        b_index_size = index_size;
        b_vb_n = vb_n;
        b_vb = vb_handle;
        b_vcpu = vcpu;
        b_vb_off = bo;
        for (u32 k = 0; k < 16; ++k) {
            b_stride[k] = k < vb_n ? attribs[k].desc.stride : 0;
            b_inst[k] = k < vb_n ? u8(attribs[k].desc.instanced) : 0;
        }
        b_guest_mask = guest_mask;
        b_glo = glo;
        b_ghi = ghi;
        b_push = push;
        b_inh_slots = inh_slots;
        b_iaddr = o_addr;
        b_isize = o_vsize;
        b_ioff = i_off;
        b_isz = i_size;
        b_clip = clip_on;
        b_planes = planes;
        b_fmt_mask = fmt_mask;
        b_tbase = base;
        b_fb_mask = fb_mask;
        b_big_mask = big_mask;
        b_ind = ind;
        last_desc_id = draw.vi->desc_id;
        ++owner->sub_batches;
    }

    // The draw's indirect command (NativeRingPrepare's member command: the GPU reads its record
    // at ring entry first + n).
    SubBatch& sb = cmds->batches.back();
    const u32 n = sb.n;
    if (!ind) {
        const u32 cmdw[5] = {index_count, n_inst, static_cast<u32>(ib_off / index_size),
                             static_cast<u32>(s32(vo)), 0u};
        std::memcpy(cmd_ptr + u64(n) * CmdBytes, cmdw, CmdBytes);
    }
    sb.n = n + 1;
    *count_ptr = n + 1;
    if (colour) {
        // The draw's material (a new one is appended now: nothing above can refuse any more)
        // and its slot write (Replay fills record (tag, idx) from mat_slots[mat]).
        if (mat < 0) {
            mat = static_cast<s32>(stretch->materials.size());
            u64 mh1 = 0, mh2 = 0;
            NR::ImageKeyHash(key.w.data(), key.n, mh1, mh2);
            stretch->materials.push_back({base, draw.rec_pos, mh1, mh2, key.n});
            mat_keys[mat] = key;
            mat_base[mat] = base;
            namespace TT = VideoCore::TexTable;
            if (TT::Mode() != 0) {
                // q8: the slots from the table (the back's BindlessResolve keys: T# words + desc
                // bits, S# words after its anisotropy rule); all must hit at one generation.
                TT::g_c[0].fetch_add(1, std::memory_order_relaxed);
                const u64 ks = TT::g_kill_seq.load(std::memory_order_acquire); // q10: first
                const u64 tg = TT::Gen();
                const size_t imgs0 = stretch->tab_imgs.size();
                const u32 set = g_bindless_cur.load(std::memory_order_acquire);
                bool all = (stretch->mat_pre == 0 || (stretch->tab_gen == tg &&
                                                      stretch->tab_set == set));
                auto& ms = cmds->mat_slots[mat];
                for (u32 i = 0; all && i < tl_ni; ++i) {
                    const auto& d = *tl_img[i].first;
                    const auto ts = d.GetSharp(*tl_img[i].second);
                    std::array<u32, 8> w;
                    static_assert(sizeof(ts) == sizeof(w));
                    std::memcpy(w.data(), &ts, sizeof(w));
                    const u32 dk = u32(d.is_depth) | u32(d.is_array) << 1 | u32(d.is_r128) << 2 |
                                   (1u << 8);
                    u32 slot = 0, iid = 0, iep = 0;
                    all = TT::LookupImg(dk, w, tg, slot, &iid, &iep);
                    ms[i] = slot;
                    if (all && TT::V2()) {
                        stretch->tab_imgs.emplace_back(iid, iep);
                    }
                }
                for (u32 j = 0; all && j < tl_ns; ++j) {
                    const auto& sd = *tl_smp[j].first;
                    const auto& sb = *tl_smp[j].second;
                    auto ss = sd.GetSharp(sb);
                    if (sd.disable_aniso) {
                        const auto& at = sb.info->images[sd.associated_image].GetSharp(sb);
                        if (at.base_level == 0 && at.last_level == 0) {
                            ss.max_aniso.Assign(AmdGpu::AnisoRatio::One);
                        }
                    }
                    std::array<u32, 4> w;
                    static_assert(sizeof(ss) == sizeof(w));
                    std::memcpy(w.data(), &ss, sizeof(w));
                    u32 slot = 0;
                    all = TT::LookupSmp(w, tg, slot);
                    ms[Shader::g_bl_img_max + j] = slot;
                }
                if (!all) {
                    stretch->tab_imgs.resize(imgs0); // q10: only taken materials' images
                }
                if (all) {
                    TT::g_c[1].fetch_add(1, std::memory_order_relaxed);
                    if (stretch->mat_pre == 0) {
                        stretch->tab_kseq = ks;
                    }
                    stretch->mat_pre |= 1u << mat;
                    stretch->tab_gen = tg;
                    stretch->tab_set = set;
                    stretch->pre_n[mat] = {static_cast<u8>(tl_ni), static_cast<u8>(tl_ns)};
                }
            }
        }
        cmds->slot_writes.push_back({draw.job_tag, idx, static_cast<u32>(mat), wrec ? rec_mapped : nullptr});
    }
    if (inh_slots != 0) {
        ++owner->sub[19];
        if (Mode() == 2) {
            for (u64 m = inh_slots; m != 0; m &= m - 1) {
                const u32 b = static_cast<u32>(std::countr_zero(m));
                cmds->inh_checks.push_back({draw.seq, b_iaddr[b], b_isize[b], b_ioff[b], b_isz[b]});
            }
        }
    }
    if (v2mask != 0) {
        ++owner->sub[49]; // q53: taken with a second vertex VkBuffer
    }
    if (fmt_mask != 0) {
        ++owner->sub[22];
        for (u64 m = fmt_mask; m != 0; m &= m - 1) {
            const u32 b = static_cast<u32>(std::countr_zero(m));
            bool known = false;
            for (const auto& [ta, ts] : stretch->texel) {
                known = known || (ta == o_addr[b] && ts == o_vsize[b]);
            }
            if (!known) {
                stretch->texel.emplace_back(o_addr[b], o_vsize[b]);
            }
            if (Mode() == 2) {
                cmds->fmt_checks.push_back({draw.seq, o_addr[b], b_tbase[b]});
            }
        }
    }
    if (b_fb_mask != 0 && Mode() == 2) {
        // Guest-lookup fallback slots: their table base is checked like a formatted slot's.
        for (u64 m = b_fb_mask; m != 0; m &= m - 1) {
            const u32 b = static_cast<u32>(std::countr_zero(m));
            cmds->fmt_checks.push_back({draw.seq, o_addr[b], b_tbase[b]});
        }
    }
    if (clip_on) {
        ++owner->sub[20];
        if (Mode() == 2) {
            cmds->clip_checks.push_back(
                {draw.seq, regs.clipper_control.user_clip_plane_enable, b_planes});
        }
    }
    if (wm != 0) {
        // r9: this draw's GPU record from the translator's own state.
        std::array<u32, NR::GpuRecDwords> g;
        BuildWorkerRecord(g, push, guest_mask, o_addr, o_vsize, vb_n, attribs, vaddr_vi);
        if (wrec) {
            u32* const rec = rec_mapped + size_t(idx & WorkerRecMask) * NR::GpuRecDwords;
            std::memcpy(rec, g.data(), sizeof(u32) * NR::GpuRecJobDwords);
            std::memcpy(rec + NR::GpuRecHiOffset, g.data() + NR::GpuRecHiOffset,
                        sizeof(u32) * (NR::GpuRecDwords - NR::GpuRecHiOffset));
            ++rec_head;
            ++nrw[0];
        } else if (push_inh == 0 && inh_slots == 0 && inh_vs == 0 && inh_ps == 0) {
            // =2 verify: field compare with the job thread's record of the same draw.
            const u32* jr = NR::g_gpu_recs[slot].load(std::memory_order_acquire);
            if (jr) {
                jr += size_t(idx & mask) * NR::GpuRecDwords;
                ++nrw[1];
                const auto diff = [&](u32 lo_dw, u32 n) {
                    return std::memcmp(g.data() + lo_dw, jr + lo_dw, sizeof(u32) * n) != 0;
                };
                const bool d_ud = diff(Shader::NativeRecordUdOffset, Shader::NUM_USER_DATA_REGS);
                const bool d_off = diff(Shader::NativeRecordOffOffset, Shader::NUM_BUFFERS);
                const bool d_size = diff(Shader::NativeRecordSizeOffset, Shader::NUM_BUFFERS);
                const bool d_voff = diff(Shader::NativeRecordVoffOffset, Shader::NativeMaxPulledLocations);
                const bool d_vsize = diff(Shader::NativeRecordVsizeOffset, Shader::NativeMaxPulledLocations);
                const bool d_hi = diff(NR::GpuRecHiOffset, NR::GpuRecDwords - NR::GpuRecHiOffset);
                nrw[2] += d_ud;
                nrw[3] += d_off;
                nrw[4] += d_size;
                nrw[5] += d_voff;
                nrw[6] += d_vsize;
                nrw[7] += d_hi;
            }
        }
        if (((nrw[0] + nrw[1]) & 0xffff) == 0 && nrw[0] + nrw[1] != 0) {
            NrwLog();
        }
    }
    b_next_idx = idx + 1;
    dyn = cand;
    stretch->last_seq = draw.seq;
    ++stretch->draws;
    ++owner->taken;
    if (VideoCore::g_lane_probe &&
        VideoCore::g_lprobe_n.load(std::memory_order_acquire) != 0) [[unlikely]] {
        // q9 census: this draw's reads over a lane pool probe (guest slots, vertex streams, the
        // index range, the indirect args).
        auto& pr = stretch->probe_reads;
        const auto add = [&](VAddr lo, u64 size) {
            if (lo == 0 || size == 0 || !VideoCore::LaneProbeAny(lo, lo + size)) {
                return;
            }
            if (pr.size() >= 256) {
                // q32: SHADPS4_A2_PROBE_COALESCE=1 - merge ranges closer than one probe window, then go on.
                static const bool coalesce = [] {
                    const char* e = std::getenv("SHADPS4_A2_PROBE_COALESCE");
                    return e && e[0] == '1';
                }();
                if (coalesce) {
                    std::sort(pr.begin(), pr.end());
                    size_t w = 0;
                    for (size_t r = 1; r < pr.size(); ++r) {
                        if (pr[r].first <= pr[w].second + (VAddr{1} << 18)) {
                            pr[w].second = std::max(pr[w].second, pr[r].second);
                        } else {
                            pr[++w] = pr[r];
                        }
                    }
                    pr.resize(w + 1);
                }
                if (pr.size() >= 256) {
                    stretch->probe_over = true; // q9b
                    return;
                }
            }
            for (const auto& [a, b] : pr) {
                if (a == lo && b == lo + size) {
                    return;
                }
            }
            pr.emplace_back(lo, lo + size);
        };
        for (u64 m = guest_mask; m != 0; m &= m - 1) {
            const u32 b = static_cast<u32>(std::countr_zero(m));
            add(o_addr[b], o_vsize[b]);
        }
        for (u32 k = 0; k < vb_n; ++k) {
            add(vaddr_vi[k], attribs[k].size);
        }
        add(index_addr, u64(ind ? regs.num_indices : index_count) * index_size);
        if (ind) {
            add(draw.arg_address, 20);
        }
    }
    return true;
}

vk::CommandBuffer StretchTranslator::Impl::AcquireSecondary(
    const std::shared_ptr<std::atomic<u64>>& tick) {
    SecPool* p = &sec_pools[sec_cur];
    if (p->used >= SecPerPool) {
        // The next pool: reusable once every stretch recorded from it is past its batch.
        const u32 next = (sec_cur + 1) % SecPools;
        SecPool& np = sec_pools[next];
        if (!np.ticks.empty()) {
            const u64 known = ctx.scheduler->GetMasterSemaphore()->KnownGpuTick();
            for (const auto& t : np.ticks) {
                const u64 v = t->load(std::memory_order_acquire);
                if (v == 0 || v > known) {
                    return {}; // busy: the closure path (never wait)
                }
            }
            ctx.instance->GetDevice().resetCommandPool(np.pool);
            np.ticks.clear();
            ++g_sec_c[7];
        }
        np.used = 0;
        sec_cur = next;
        p = &np;
    }
    if (p->used == p->cbs.size()) {
        auto [r, cbs] = ctx.instance->GetDevice().allocateCommandBuffers(
            vk::CommandBufferAllocateInfo{
                .commandPool = p->pool,
                .level = vk::CommandBufferLevel::eSecondary,
                .commandBufferCount = 1,
            });
        if (r != vk::Result::eSuccess || cbs.empty()) {
            return {};
        }
        p->cbs.push_back(cbs[0]);
    }
    p->ticks.push_back(tick);
    return p->cbs[p->used++];
}

void StretchTranslator::Impl::RecordSecondary() {
    if (!sec_on || !stretch || !cmds || !stretch->rt_pipe || cmds->batches.empty()) {
        return;
    }
    for (const auto& b : cmds->batches) {
        if (b.set_owner) {
            ++g_sec_c[2]; // l4g: the back commits that set at publish
            return;
        }
    }
    const vk::CommandBuffer cb = AcquireSecondary(stretch->tick);
    if (!cb) {
        ++g_sec_c[1];
        return;
    }
    // The pass's formats from the pipeline key (as GraphicsPipeline's PipelineRenderingCreateInfo).
    const Instance& inst = *ctx.instance;
    const auto& key = stretch->rt_pipe->GetGraphicsKey();
    Stretch::Inh inh{};
    inh.n_color = std::min<u32>(key.num_color_attachments, 8);
    for (u32 i = 0; i < inh.n_color; ++i) {
        const auto& col = key.color_buffers[i];
        inh.color[i] = inst.GetSupportedFormat(
            LiverpoolToVK::SurfaceFormat(col.data_format, col.num_format),
            vk::FormatFeatureFlagBits2::eColorAttachment);
    }
    const auto depth_format =
        inst.GetSupportedFormat(LiverpoolToVK::DepthFormat(key.z_format, key.stencil_format),
                                vk::FormatFeatureFlagBits2::eDepthStencilAttachment);
    inh.depth = key.z_format != AmdGpu::DepthBuffer::ZFormat::Invalid ? depth_format
                                                                       : vk::Format::eUndefined;
    inh.stencil = key.stencil_format != AmdGpu::DepthBuffer::StencilFormat::Invalid
                      ? depth_format
                      : vk::Format::eUndefined;
    std::array<vk::SampleCountFlagBits, AmdGpu::NUM_COLOR_BUFFERS> color_samples;
    std::ranges::transform(key.color_samples, color_samples.begin(), [&inst](u8 n) {
        return n ? LiverpoolToVK::NumSamples(n, inst.GetColorSampleCounts())
                 : vk::SampleCountFlagBits::e1;
    });
    const vk::AttachmentSampleCountInfoAMD mixed = {
        .colorAttachmentCount = inh.n_color,
        .pColorAttachmentSamples = color_samples.data(),
        .depthStencilAttachmentSamples =
            LiverpoolToVK::NumSamples(key.depth_samples, inst.GetDepthSampleCounts()),
    };
    const vk::CommandBufferInheritanceRenderingInfo rinfo = {
        .pNext = inst.IsMixedDepthSamplesSupported() ? &mixed : nullptr,
        // p2b (review #2, VUID-vkCmdExecuteCommands-flags-06026): the pass's flags minus
        // SECONDARY_COMMAND_BUFFERS.
        .flags = static_cast<vk::RenderingFlags>(VK_RENDERING_CONTENTS_INLINE_BIT_KHR),
        .colorAttachmentCount = inh.n_color,
        .pColorAttachmentFormats = inh.color.data(),
        .depthAttachmentFormat = inh.depth,
        .stencilAttachmentFormat = inh.stencil,
        .rasterizationSamples = LiverpoolToVK::NumSamples(
            key.num_samples, inst.GetColorSampleCounts() & inst.GetDepthSampleCounts()),
    };
    const vk::CommandBufferInheritanceInfo info = {.pNext = &rinfo};
    const u32 cur = g_bindless_cur.load(std::memory_order_acquire) % Bindless::NumSets;
    if (cb.begin(vk::CommandBufferBeginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit |
                     vk::CommandBufferUsageFlagBits::eRenderPassContinue,
            .pInheritanceInfo = &info,
        }) != vk::Result::eSuccess) {
        ++g_sec_c[1];
        return; // p2b (review #11): no secondary (the pool keeps it until its tick passes)
    }
    RecordBatches(*cmds, Bindless::G().sets[cur], cb, inst);
    if (cb.end() != vk::Result::eSuccess) {
        ++g_sec_c[1];
        return;
    }
    stretch->secondary = cb;
    stretch->bl_idx = cur;
    stretch->inh = inh;
    ++g_sec_c[0];
}

std::unique_ptr<Stretch> StretchTranslator::Impl::End() {
    if (!stretch) {
        return nullptr;
    }
    RecordSecondary();
    stretch->arena_end = ctx.arena->head;
    ctx.arena->fences.push_back({stretch_start, ctx.arena->head, stretch->tick});
    if (rec_mapped) {
        _mm_sfence(); // r9: the records' write-combined stores before the stretch is published
        rec_fences.push_back({rec_stretch_start, rec_head, stretch->tick});
    }
    batch_open = false;
    b_native = nullptr;
    cmd_ptr = nullptr;
    count_ptr = nullptr;
    if (perdraw && cmds) {
        // The next reserve: this stretch's sub-batch count, or the previous hint minus 1/8, in
        // [8, 16] (bounds the reserved memory per stretch at 2x the old reserve(8)).
        const u32 n = static_cast<u32>(cmds->batches.size());
        batch_hint = std::clamp<u32>(std::max(n, batch_hint - batch_hint / 8), 8, 16);
    }
    cmds.reset();
    return std::move(stretch);
}

StretchTranslator::StretchTranslator(const WorkerContext& ctx)
    : impl{std::make_unique<Impl>(this, ctx)} {}

StretchTranslator::~StretchTranslator() = default;

bool StretchTranslator::TakeDraw(const AmdGpu::Regs& regs, const DrawIn& draw) {
    return impl->Take(regs, draw);
}

std::unique_ptr<Stretch> StretchTranslator::EndStretch() {
    return impl->End();
}

} // namespace Vulkan::A2
