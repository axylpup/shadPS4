// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <cstdlib>
#include <ranges>

#include "shader_recompiler/spirv_variants.h"
#include "common/fault_monitor.h"
#include "common/hash.h"
#include "common/lean.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/native_emit.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/lean_front_api.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/cache_storage.h"
#include "video_core/native_census.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_bindless.h"
#include "video_core/renderer_vulkan/vk_dlss.h"
#include "video_core/renderer_vulkan/vk_dlss_upscale.h"
#include "video_core/renderer_vulkan/vk_pipeline_compile_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

using Shader::LogicalStage;
using Shader::Output;
using Shader::Stage;

constexpr static auto SpirvVersion1_6 = 0x00010600U;

constexpr static std::array DescriptorHeapSizes = {
    vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 512},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1024},
};

static u32 MapOutputs(std::span<Shader::OutputMap, 3> outputs, const AmdGpu::VsOutputControl& ctl) {
    u32 num_outputs = 0;

    if (ctl.vs_out_misc_enable) {
        auto& misc_vec = outputs[num_outputs++];
        misc_vec[0] = ctl.use_vtx_point_size ? Output::PointSize : Output::None;
        misc_vec[1] = ctl.use_vtx_edge_flag
                          ? Output::EdgeFlag
                          : (ctl.use_vtx_gs_cut_flag ? Output::GsCutFlag : Output::None);
        misc_vec[2] =
            ctl.use_vtx_kill_flag
                ? Output::KillFlag
                : (ctl.use_vtx_render_target_idx ? Output::RenderTargetIndex : Output::None);
        misc_vec[3] = ctl.use_vtx_viewport_idx ? Output::ViewportIndex : Output::None;
    }

    if (ctl.vs_out_ccdist0_enable) {
        auto& ccdist0 = outputs[num_outputs++];
        ccdist0[0] = ctl.IsClipDistEnabled(0)
                         ? Output::ClipDist0
                         : (ctl.IsCullDistEnabled(0) ? Output::CullDist0 : Output::None);
        ccdist0[1] = ctl.IsClipDistEnabled(1)
                         ? Output::ClipDist1
                         : (ctl.IsCullDistEnabled(1) ? Output::CullDist1 : Output::None);
        ccdist0[2] = ctl.IsClipDistEnabled(2)
                         ? Output::ClipDist2
                         : (ctl.IsCullDistEnabled(2) ? Output::CullDist2 : Output::None);
        ccdist0[3] = ctl.IsClipDistEnabled(3)
                         ? Output::ClipDist3
                         : (ctl.IsCullDistEnabled(3) ? Output::CullDist3 : Output::None);
    }

    if (ctl.vs_out_ccdist1_enable) {
        auto& ccdist1 = outputs[num_outputs++];
        ccdist1[0] = ctl.IsClipDistEnabled(4)
                         ? Output::ClipDist4
                         : (ctl.IsCullDistEnabled(4) ? Output::CullDist4 : Output::None);
        ccdist1[1] = ctl.IsClipDistEnabled(5)
                         ? Output::ClipDist5
                         : (ctl.IsCullDistEnabled(5) ? Output::CullDist5 : Output::None);
        ccdist1[2] = ctl.IsClipDistEnabled(6)
                         ? Output::ClipDist6
                         : (ctl.IsCullDistEnabled(6) ? Output::CullDist6 : Output::None);
        ccdist1[3] = ctl.IsClipDistEnabled(7)
                         ? Output::ClipDist7
                         : (ctl.IsCullDistEnabled(7) ? Output::CullDist7 : Output::None);
    }

    return num_outputs;
}

// The graphics stages of BuildRuntimeInfo over an explicit register file: a pure function of
// `regs`, `key.color_buffers` (first key pass) and, for the LS stage, the hull Info's per-draw
// state. Shared by the front path (member scratch) and the read-only path (caller's scratch).
void PipelineCache::FillStageRuntimeInfo(Shader::RuntimeInfo& info, const AmdGpu::Regs& regs,
                                         Stage stage, LogicalStage l_stage,
                                         const Shader::Info* hull_info,
                                         const GraphicsPipelineKey& key, bool dump) const {
    const auto BuildCommon = [&](const auto& program) {
        info.num_user_data = program.settings.num_user_regs;
        info.num_input_vgprs = program.settings.vgpr_comp_cnt;
        info.num_allocated_vgprs = program.NumVgprs();
        info.fp_denorm_mode32 = program.settings.fp_denorm_mode32;
        info.fp_denorm_mode16_64 = program.settings.fp_denorm_mode64;
        info.fp_round_mode32 = program.settings.fp_round_mode32;
        info.fp_round_mode16_64 = program.settings.fp_round_mode64;
    };
    switch (stage) {
    case Stage::Local: {
        BuildCommon(regs.ls_program);
        Shader::TessellationDataConstantBuffer tess_constants{};
        hull_info->ReadTessConstantBuffer(tess_constants);
        info.ls_info.ls_stride = tess_constants.ls_stride;
        break;
    }
    case Stage::Hull: {
        BuildCommon(regs.hs_program);
        info.hs_info.num_input_control_points = regs.ls_hs_config.hs_input_control_points;
        info.hs_info.num_threads = regs.ls_hs_config.hs_output_control_points;
        info.hs_info.tess_type = regs.tess_config.type;
        info.hs_info.offchip_lds_enable = regs.hs_program.settings.oc_lds_en;

        // We need to initialize most hs_info fields after finding the V# with tess constants
        break;
    }
    case Stage::Export: {
        BuildCommon(regs.es_program);
        info.es_info.vertex_data_size = regs.vgt_esgs_ring_itemsize;
        if (l_stage == LogicalStage::TessellationEval) {
            info.es_vs_info.tess_type = regs.tess_config.type;
            info.es_vs_info.tess_topology = regs.tess_config.topology;
            info.es_vs_info.tess_partitioning = regs.tess_config.partitioning;
        }
        break;
    }
    case Stage::Vertex: {
        BuildCommon(regs.vs_program);
        info.vs_info.user_clip_plane_mask = regs.clipper_control.user_clip_plane_enable;
        info.vs_info.step_rate_0 = regs.vgt_instance_step_rate_0;
        info.vs_info.step_rate_1 = regs.vgt_instance_step_rate_1;
        info.vs_info.num_outputs = MapOutputs(info.vs_info.outputs, regs.vs_output_control);
        info.vs_info.emulate_depth_negative_one_to_one =
            !instance.IsDepthClipControlSupported() &&
            regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW;
        info.vs_info.tess_emulated_primitive =
            regs.primitive_type == AmdGpu::PrimitiveType::RectList ||
            regs.primitive_type == AmdGpu::PrimitiveType::QuadList;
        info.vs_info.clip_disable = regs.IsClipDisabled();
        if (l_stage == LogicalStage::TessellationEval) {
            info.es_vs_info.tess_type = regs.tess_config.type;
            info.es_vs_info.tess_topology = regs.tess_config.topology;
            info.es_vs_info.tess_partitioning = regs.tess_config.partitioning;
        }
        break;
    }
    case Stage::Geometry: {
        BuildCommon(regs.gs_program);
        auto& gs_info = info.gs_info;
        gs_info.num_outputs = MapOutputs(gs_info.outputs, regs.vs_output_control);
        gs_info.output_vertices = regs.vgt_gs_max_vert_out;
        gs_info.num_invocations =
            regs.vgt_gs_instance_cnt.IsEnabled() ? regs.vgt_gs_instance_cnt.count : 1;
        if (regs.stage_enable.raw == AmdGpu::ShaderStageEnable::LsHsEsGs) {
            gs_info.in_primitive = [&]() {
                switch (regs.tess_config.topology) {
                case AmdGpu::TessellationTopology::Point:
                    return AmdGpu::PrimitiveType::PointList;
                case AmdGpu::TessellationTopology::Line:
                    return AmdGpu::PrimitiveType::LineList;
                case AmdGpu::TessellationTopology::TriangleCw:
                case AmdGpu::TessellationTopology::TriangleCcw:
                    return AmdGpu::PrimitiveType::TriangleList;
                default:
                    UNREACHABLE();
                }
            }();
        } else {
            gs_info.in_primitive = regs.primitive_type;
        }
        for (u32 stream_id = 0; stream_id < Shader::GsMaxOutputStreams; ++stream_id) {
            gs_info.out_primitive[stream_id] =
                regs.vgt_gs_out_prim_type.GetPrimitiveType(stream_id);
        }
        gs_info.in_vertex_data_size = regs.vgt_esgs_ring_itemsize;
        gs_info.out_vertex_data_size = regs.vgt_gs_vert_itemsize[0];
        gs_info.mode = regs.vgt_gs_mode.mode;
        const auto params_vc = AmdGpu::GetParams(regs.vs_program);
        gs_info.vs_copy = params_vc.code;
        gs_info.vs_copy_hash = params_vc.hash;
        if (dump) {
            DumpShader(gs_info.vs_copy, gs_info.vs_copy_hash, Shader::Stage::Vertex, 0,
                       "copy.bin");
        }
        break;
    }
    case Stage::Fragment: {
        BuildCommon(regs.ps_program);
        info.fs_info.en_flags = regs.ps_input_ena;
        info.fs_info.addr_flags = regs.ps_input_addr;
        info.fs_info.num_inputs = regs.num_interp;
        info.fs_info.z_export_format = regs.z_export_format;
        u8 stencil_ref_export_enable = regs.depth_shader_control.stencil_op_val_export_enable |
                                       regs.depth_shader_control.stencil_test_val_export_enable;
        info.fs_info.mrtz_mask = regs.depth_shader_control.z_export_enable |
                                 (stencil_ref_export_enable << 1) |
                                 (regs.depth_shader_control.mask_export_enable << 2) |
                                 (regs.depth_shader_control.coverage_to_mask_enable << 3);
        const auto& cb0_blend = regs.blend_control[0];
        if (cb0_blend.enable) {
            info.fs_info.dual_source_blending =
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_dst_factor) ||
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_src_factor);
            if (cb0_blend.separate_alpha_blend) {
                info.fs_info.dual_source_blending |=
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_dst_factor) ||
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_src_factor);
            }
        } else {
            info.fs_info.dual_source_blending = false;
        }
        const auto& ps_inputs = regs.ps_inputs;
        for (u32 i = 0; i < regs.num_interp; i++) {
            info.fs_info.inputs[i] = {
                .param_index = u8(ps_inputs[i].input_offset),
                .is_default = bool(ps_inputs[i].use_default),
                .is_flat = bool(ps_inputs[i].flat_shade),
                .default_value = u8(ps_inputs[i].default_value),
            };
        }
        for (u32 i = 0; i < Shader::MaxColorBuffers; i++) {
            info.fs_info.color_buffers[i] = key.color_buffers[i];
        }
        // Lowered user clip planes ride the same emulation path as guest-exported distances, so
        // the fragment side arms whenever the hardware vertex stage lowers them, keeping its input
        // locations in sync with the shifted vertex outputs.
        const bool lowers_user_clip_planes =
            regs.clipper_control.user_clip_plane_enable &&
            !regs.stage_enable.IsStageEnabled(static_cast<u32>(Stage::Geometry));
        info.fs_info.clip_distance_emulation =
            ((regs.vs_output_control.clip_distance_enable &&
              !regs.stage_enable.IsStageEnabled(static_cast<u32>(Stage::Local))) ||
             lowers_user_clip_planes) &&
            profile.needs_clip_distance_emulation;
        break;
    }
    default:
        break;
    }
}

const Shader::RuntimeInfo& PipelineCache::BuildRuntimeInfo(const AmdGpu::Regs& regs, Stage stage,
                                                           LogicalStage l_stage) {
    auto& info = runtime_infos[u32(l_stage)];
    info.Initialize(stage);
    if (stage != Stage::Compute) {
        FillStageRuntimeInfo(info, regs, stage, l_stage,
                             infos[u32(Shader::LogicalStage::TessellationControl)], graphics_key,
                             true);
        return info;
    }
    {
        const auto& cs_pgm = liverpool->GetCsRegs();
        info.num_user_data = cs_pgm.settings.num_user_regs;
        info.num_allocated_vgprs = cs_pgm.settings.num_vgprs * 4;
        info.cs_info.workgroup_size = {cs_pgm.num_thread_x.full, cs_pgm.num_thread_y.full,
                                       cs_pgm.num_thread_z.full};
        info.cs_info.tgid_enable = {cs_pgm.IsTgidEnabled(0), cs_pgm.IsTgidEnabled(1),
                                    cs_pgm.IsTgidEnabled(2)};
        info.cs_info.shared_memory_size = cs_pgm.SharedMemSize();
        // SHADPS4_CS_FLOAT_MODE_LOG=1 (diagnostic, default off): the FLOAT_MODE bits (12-19) of
        // COMPUTE_PGM_RSRC1, once per program address and mode. Compute shaders are translated
        // with mode 0 whatever the guest sets (upstream #5126 changes that).
        static const bool log_cs_float_mode = std::getenv("SHADPS4_CS_FLOAT_MODE_LOG") != nullptr;
        if (log_cs_float_mode) {
            u64 raw{};
            std::memcpy(&raw, &cs_pgm.settings, sizeof(raw));
            const u32 mode = u32(raw >> 12) & 0xff;
            static std::mutex seen_mutex;
            static std::unordered_set<u64> seen;
            std::scoped_lock lk{seen_mutex};
            const u64 pgm = u64(cs_pgm.address) << 8;
            if (seen.insert((pgm << 8) | mode).second) {
                LOG_INFO(Render_Vulkan,
                         "CsFloatMode: pgm={:#x} mode={:#04x} round32={} round64={} denorm32={} "
                         "denorm64={} uniques={}",
                         pgm, mode, mode & 3, (mode >> 2) & 3, (mode >> 4) & 3, (mode >> 6) & 3,
                         seen.size());
            }
        }
    }
    return info;
}

namespace {
// Wall time of one shader translation or pipeline construction; read only around those, never
// per draw (Common::FaultMonitor::Now is a no-op unless SHADPS4_FAULT_TIMERS).
using CompileClock = std::chrono::steady_clock;
double MsSince(CompileClock::time_point start) {
    return std::chrono::duration<double, std::milli>(CompileClock::now() - start).count();
}
} // namespace

// SHADPS4_ASYNC_PIPELINES: one graphics pipeline construction, owned end to end. The compiler
// thread reads only copies, plus two kinds of handle to objects the cache never destroys while a
// job can exist: `infos` point into program_cache's Programs (unique_ptr, never destroyed:
// RetirePrecompiled erases compute Programs from the map but parks them in retired_programs; the
// constructor dereferences them only for the immutable stage/pgm_hash of its debug name, every
// per-draw field was read by DeriveStageInputs on the lookup thread) and `modules`, which only
// ReplaceShader destroys, after WaitIdle.
struct PipelineCache::AsyncJob {
    GraphicsPipelineKey key{};
    u64 hash{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    std::shared_ptr<const Shader::Gcn::FetchShaderData> fetch_shader;
    GraphicsPipeline::SerializationSupport sdata{};
    GraphicsPipeline::DescSetBindings desc_bindings;
    double translate_ms{};
    u32 queued_ahead{};
    // Compiler thread output, handed back under the queue mutex.
    double create_ms{};
    std::unique_ptr<GraphicsPipeline> pipeline;
};

// FIFO worker. vk::PipelineCache was created without the externally-synchronized flag, so the
// driver serializes its use between this thread and the synchronous compute path.
struct PipelineCache::AsyncCompiler {
    AsyncCompiler(const Instance& instance_, Scheduler& scheduler_, DescriptorHeap& desc_heap_,
                  const Shader::Profile& profile_, vk::PipelineCache pipeline_cache_)
        : instance{instance_}, scheduler{scheduler_}, desc_heap{desc_heap_}, profile{profile_},
          pipeline_cache{pipeline_cache_}, thread{std::bind_front(&AsyncCompiler::Run, this)} {}

    // Queued jobs are dropped; a construction in progress completes first (one driver compile).
    // Finished, unpublished pipelines are destroyed with `finished`, after the thread.
    ~AsyncCompiler() {
        thread.request_stop();
        thread.join();
    }

    void Push(std::unique_ptr<AsyncJob> job) {
        {
            std::scoped_lock lk{mutex};
            job->queued_ahead = static_cast<u32>(queue.size()) + (busy ? 1u : 0u);
            queue.push_back(std::move(job));
        }
        cv.notify_one();
    }

    bool HasFinished() const {
        return num_finished.load(std::memory_order_acquire) != 0;
    }

    std::vector<std::unique_ptr<AsyncJob>> TakeFinished() {
        std::scoped_lock lk{mutex};
        num_finished.store(0, std::memory_order_relaxed);
        return std::exchange(finished, {});
    }

    /// Returns once nothing is queued or in construction.
    void WaitIdle() {
        std::unique_lock lk{mutex};
        idle_cv.wait(lk, [&] { return queue.empty() && !busy; });
    }

private:
    void Run(std::stop_token stoken) {
        Common::SetCurrentThreadName("shadPS4:PipelineCompiler");
        while (true) {
            std::unique_ptr<AsyncJob> job;
            {
                std::unique_lock lk{mutex};
                Common::CondvarWait(cv, lk, stoken, [&] { return !queue.empty(); });
                if (stoken.stop_requested()) {
                    return;
                }
                job = std::move(queue.front());
                queue.pop_front();
                busy = true;
            }
            const auto t_create = CompileClock::now();
            job->pipeline = std::make_unique<GraphicsPipeline>(
                instance, scheduler, desc_heap, profile, job->key, pipeline_cache, job->infos,
                job->runtime_infos, job->fetch_shader, job->modules, job->sdata, false,
                &job->desc_bindings);
            job->create_ms = MsSince(t_create);
            {
                std::scoped_lock lk{mutex};
                finished.push_back(std::move(job));
                busy = false;
                num_finished.fetch_add(1, std::memory_order_release);
            }
            idle_cv.notify_all();
        }
    }

    const Instance& instance;
    Scheduler& scheduler;
    DescriptorHeap& desc_heap;
    const Shader::Profile& profile;
    vk::PipelineCache pipeline_cache;
    std::mutex mutex;
    std::condition_variable_any cv;
    std::condition_variable_any idle_cv;
    std::deque<std::unique_ptr<AsyncJob>> queue;
    std::vector<std::unique_ptr<AsyncJob>> finished;
    bool busy{};
    std::atomic<u32> num_finished{};
    std::jthread thread; // last: joined before anything above is destroyed
};

// SHADPS4_NATIVE_COLOUR (notes/colour-pass-2026-10-09.md sec. 3): one colour variant's driver
// compile, built on the front and run on the native compiler thread. Like AsyncJob: the thread
// reads only copies plus Infos / modules the cache never destroys while it runs.
struct PipelineCache::NativeColourJob {
    const GraphicsPipeline* base{};
    u32 slot{};
    GraphicsPipelineKey key{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    std::shared_ptr<const Shader::Gcn::FetchShaderData> fetch_shader;
    GraphicsPipeline::SerializationSupport sdata{};
    GraphicsPipeline::DescSetBindings bindings;
    u32 records_binding{};
    u32 vertex_binding{};
    u32 table_binding{GraphicsPipeline::NoNativeRecords};
    u32 pt_debug_binding{GraphicsPipeline::NoNativeRecords}; // SHADPS4_NATIVE_PT=2
    u32 vertex2_binding{GraphicsPipeline::NoNativeRecords};  // q53: SHADPS4_A2_VBUF2
    bool bindless{};
    const char* bindless_why{}; // q1b census
    u64 dead_mask{};            // q3: bindings the sparse layout dropped
    bool bindless_vs{};         // q3
    std::string counts;         // q1b census: descriptor counts per stage
};

struct PipelineCache::NativeColourCompiler {
    NativeColourCompiler(const Instance& instance_, Scheduler& scheduler_,
                         DescriptorHeap& desc_heap_, const Shader::Profile& profile_,
                         vk::PipelineCache pipeline_cache_)
        : instance{instance_}, scheduler{scheduler_}, desc_heap{desc_heap_}, profile{profile_},
          pipeline_cache{pipeline_cache_},
          thread{std::bind_front(&NativeColourCompiler::Run, this)} {}

    ~NativeColourCompiler() {
        thread.request_stop();
        thread.join();
    }

    void Push(std::unique_ptr<NativeColourJob> job) {
        {
            std::scoped_lock lk{mutex};
            queue.push_back(std::move(job));
        }
        cv.notify_one();
    }

    /// Returns once nothing is queued or in construction (shader replacement erases bases).
    void WaitIdle() {
        std::unique_lock lk{mutex};
        idle_cv.wait(lk, [&] { return queue.empty() && !busy; });
    }

private:
    void Run(std::stop_token stoken) {
        Common::SetCurrentThreadName("shadPS4:NativeColourCompiler");
        while (true) {
            std::unique_ptr<NativeColourJob> job;
            {
                std::unique_lock lk{mutex};
                Common::CondvarWait(cv, lk, stoken, [&] { return !queue.empty(); });
                if (stoken.stop_requested()) {
                    return;
                }
                job = std::move(queue.front());
                queue.pop_front();
                busy = true;
            }
            // 1.1.1 (GitHub #25, #31): named before the driver sees it, so a driver crash or
            // refusal in a player's log says which variant (vs / ps program hashes).
            const u64 key_hash = std::hash<GraphicsPipelineKey>{}(job->key);
            const auto pgm = [&](size_t i) { return job->infos[i] ? job->infos[i]->pgm_hash : 0; };
            LOG_INFO(Render_Vulkan,
                     "NativeColour: compiling variant of {:#x} slot {} (vs {:#x} ps {:#x}, "
                     "bindless {}, {} bindings)",
                     key_hash, job->slot, pgm(size_t(Shader::LogicalStage::Vertex)),
                     pgm(size_t(Shader::LogicalStage::Fragment)), job->bindless,
                     job->bindings.size());
            const auto t0 = CompileClock::now();
            auto variant = std::make_unique<GraphicsPipeline>(
                instance, scheduler, desc_heap, profile, job->key, pipeline_cache, job->infos,
                job->runtime_infos, job->fetch_shader, job->modules, job->sdata, false,
                &job->bindings, job->records_binding, job->bindless, /*refusable=*/true);
            const double ms = MsSince(t0);
            variant->sparse_dead_mask = job->dead_mask; // q3 (before the slot publishes it)
            variant->native_bindless_vs = job->bindless_vs;
            auto& slot = job->base->native_colour[job->slot];
            if (!variant->IsCreated()) {
                // The driver refused it (logged by the constructor): not eligible, the normal
                // pipeline draws this description - as for no push descriptors below.
                LOG_WARNING(Render_Vulkan,
                            "NativeColour: variant of {:#x} refused by the driver - not eligible, "
                            "the normal path draws it",
                            key_hash);
                job->base->native_colour_why.store("driver refused", std::memory_order_relaxed);
                slot.state.store(2, std::memory_order_release);
            } else if (!variant->UsesPushDescriptors()) {
                // Too many descriptors for push descriptors (NVIDIA: 32): the slot stays claimed
                // with no variant, so the back keeps the generic path for this description.
                LOG_INFO(Render_Vulkan,
                         "NativeColour: variant of {:#x} has no push descriptors (q1b: bindless {} "
                         "why-not '{}' bindings {} | {})",
                         std::hash<GraphicsPipelineKey>{}(job->key), job->bindless,
                         job->bindless_why ? job->bindless_why : "-", job->bindings.size(),
                         job->counts);
                job->base->native_colour_why.store("no push descriptors", std::memory_order_relaxed);
                slot.state.store(2, std::memory_order_release);
                PipelineCache::g_nc_claims[6].fetch_add(1, std::memory_order_relaxed);
                static const bool set_variants = [] {
                    const char* e = std::getenv("SHADPS4_NATIVE_SET_VARIANTS");
                    return e && e[0] == '1';
                }();
                if (set_variants) {
                    // l4g: kept for the A2 translator only (descriptor set per sub-batch).
                    variant->native_vertex_binding = job->vertex_binding;
                    variant->native_table_binding = job->table_binding;
                    variant->native_pt_debug_binding = job->pt_debug_binding;
                    variant->native_vertex2_binding = job->vertex2_binding;
                    slot.set_variant.store(variant.get(), std::memory_order_release);
                    PipelineCache::g_nc_claims[7].fetch_add(1, std::memory_order_relaxed);
                }
            } else {
                variant->native_vertex_binding = job->vertex_binding;
                variant->native_table_binding = job->table_binding;
                variant->native_pt_debug_binding = job->pt_debug_binding;
                variant->native_vertex2_binding = job->vertex2_binding;
                slot.variant.store(variant.get(), std::memory_order_release);
                PipelineCache::g_nc_claims[5].fetch_add(1, std::memory_order_relaxed);
            }
            if (++built <= 64 || (built & 63) == 0) {
                LOG_INFO(Render_Vulkan, "NativeColour: variant #{} of {:#x} slot {} built in {:.1f} ms",
                         built, std::hash<GraphicsPipelineKey>{}(job->key), job->slot, ms);
            }
            {
                std::scoped_lock lk{mutex};
                owned.push_back(std::move(variant));
                busy = false;
            }
            idle_cv.notify_all();
        }
    }

    const Instance& instance;
    Scheduler& scheduler;
    DescriptorHeap& desc_heap;
    const Shader::Profile& profile;
    vk::PipelineCache pipeline_cache;
    std::mutex mutex;
    std::condition_variable_any cv;
    std::condition_variable_any idle_cv;
    bool busy{};
    std::deque<std::unique_ptr<NativeColourJob>> queue;
    std::vector<std::unique_ptr<GraphicsPipeline>> owned;
    u32 built{};
    std::jthread thread; // last: started after every member above exists
};

PipelineCache::PipelineCache(const Instance& instance_, Scheduler& scheduler_,
                             AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      desc_heap{instance, scheduler.GetMasterSemaphore(), DescriptorHeapSizes} {
    const auto& vk12_props = instance.GetVk12Properties();
    profile = Shader::Profile{
        // When binding a UBO, we calculate its size considering the offset in the larger buffer
        // cache underlying resource. In some cases, it may produce sizes exceeding the system
        // maximum allowed UBO range, so we need to reduce the threshold to prevent issues.
        .max_ubo_size = instance.UniformMaxSize() - instance.UniformMinAlignment(),
        .max_viewport_width = instance.GetMaxViewportWidth(),
        .max_viewport_height = instance.GetMaxViewportHeight(),
        .max_shared_memory_size = instance.MaxComputeSharedMemorySize(),
        .supported_spirv = SpirvVersion1_6,
        .subgroup_size = instance.SubgroupSize(),
        .support_int8 = instance.IsShaderInt8Supported(),
        .support_int16 = instance.IsShaderInt16Supported(),
        .support_int64 = instance.IsShaderInt64Supported(),
        .support_float16 = instance.IsShaderFloat16Supported(),
        .support_float64 = instance.IsShaderFloat64Supported(),
        .supports_denorm_behavior_independence =
            vk12_props.denormBehaviorIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .supports_rounding_mode_independence =
            vk12_props.roundingModeIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .support_fp16_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat16),
        .support_fp16_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat16),
        .support_fp16_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat16),
        .support_fp32_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat32),
        .support_fp32_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat32),
        .support_fp32_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat32),
        .support_fp64_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat64),
        .support_fp64_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat64),
        .support_fp64_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat64),
        .support_fp16_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat16),
        .support_fp32_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat32),
        .support_fp64_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat64),
        .supports_image_load_store_lod = instance_.IsImageLoadStoreLodSupported(),
        .supports_native_cube_calc = instance_.IsAmdGcnShaderSupported(),
        .supports_trinary_minmax = instance_.IsAmdShaderTrinaryMinMaxSupported(),
        .supports_buffer_fp32_atomic_min_max =
            instance_.IsShaderAtomicFloatBuffer32MinMaxSupported(),
        .supports_image_fp32_atomic_min_max = instance_.IsShaderAtomicFloatImage32MinMaxSupported(),
        .supports_buffer_int64_atomics = instance_.IsBufferInt64AtomicsSupported(),
        .supports_shared_int64_atomics = instance_.IsSharedInt64AtomicsSupported(),
        .supports_workgroup_explicit_memory_layout =
            instance_.IsWorkgroupMemoryExplicitLayoutSupported(),
        .supports_amd_shader_explicit_vertex_parameter =
            instance_.IsAmdShaderExplicitVertexParameterSupported(),
        .supports_fragment_shader_barycentric = instance_.IsFragmentShaderBarycentricSupported(),
        .needs_manual_interpolation = instance.IsFragmentShaderBarycentricSupported() &&
                                      instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .needs_lds_barriers = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary ||
                              instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_buffer_offsets = instance.StorageMinAlignment() > 4,
        .needs_unorm_fixup = instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_clip_distance_emulation = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .supports_shader_stencil_export = instance_.IsShaderStencilExportSupported(),
    };
    WarmUp();

    auto [cache_result, cache] = instance.GetDevice().createPipelineCacheUnique({});
    ASSERT_MSG(cache_result == vk::Result::eSuccess, "Failed to create pipeline cache: {}",
               vk::to_string(cache_result));
    pipeline_cache = std::move(cache);
}

PipelineCache::~PipelineCache() {
    // SHADPS4_ASYNC_PIPELINES: the compiler references members of this object; stop it first.
    // Queued jobs are dropped, finished unpublished pipelines are destroyed with it.
    async_compiler.reset();
    // SHADPS4_ASYNC_COMPUTE_PIPELINES: no worker may write into a pipeline destroyed below.
    PipelineCompilePool::Shutdown();
    scheduler.DrainRecording();
}

namespace {
// Per-draw / per-stage environment flags, read once at startup. A function-local `static const`
// costs a guard check (a TLS epoch load chain under the MSVC ABI) on every call.
bool EnvExactly1(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] == '1' && value[1] == '\0';
}
bool EnvIs1(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] == '1';
}
const bool g_prepared_pipelines = EnvExactly1("SHADPS4_PREPARED_PIPELINES");
const bool g_prepared_tags = EnvIs1("SHADPS4_PREPARED_TAGS");
const bool g_prepared_pipelines_verify = EnvExactly1("SHADPS4_PREPARED_PIPELINES_VERIFY");
const size_t g_prepared_ways_env = [] {
    const char* value = std::getenv("SHADPS4_PREPARED_WAYS");
    return value ? static_cast<size_t>(std::strtoull(value, nullptr, 10)) : size_t{4};
}();
const bool g_prepared_lru = EnvIs1("SHADPS4_PREPARED_LRU");
const bool g_log_mixed_samples = EnvIs1("SHADPS4_MIXED_SAMPLES_LOG");
const bool g_shared_fetch_verify = [] {
    const char* value = std::getenv("SHADPS4_SHARED_FETCH_VERIFY");
    return value && std::string_view{value} == "1";
}();
const bool g_prepared_programs = EnvExactly1("SHADPS4_PREPARED_PROGRAMS");
const bool g_prepared_programs_verify = EnvExactly1("SHADPS4_PREPARED_PROGRAMS_VERIFY");
const bool g_prepared_programs_two_entry = EnvExactly1("SHADPS4_PREPARED_PROGRAMS_TWO_ENTRY");
const bool g_async_pipelines = EnvExactly1("SHADPS4_ASYNC_PIPELINES");
} // namespace

static bool PreparedPipelinesEnabled() {
    return g_prepared_pipelines;
}

static bool PreparedTags() {
    return g_prepared_tags;
}

static bool PreparedPipelinesVerify() {
    return g_prepared_pipelines_verify;
}

namespace {
// Command-processor thread only, like every GetGraphicsPipeline caller.
struct PreparedPipelineStats {
    u64 attempts, ineligible, candidates, stale, guard_fallbacks, hits, seeds, seed_rejects,
        verified, input_changes, evictions;
};
PreparedPipelineStats prepared_stats{};

// Verification only: what the fast path's own refresh left in each Info, to compare with
// the original path's refresh of the same draw.
struct PreparedRefreshRecord {
    VAddr pgm_base{};
    const u32* user_data{};
    std::vector<u32> flat;

    void Save(const Shader::Info& info) {
        pgm_base = info.pgm_base;
        user_data = info.user_data.data();
        flat = info.flattened_ud_buf;
    }

    bool Same(const Shader::Info& info) const {
        return pgm_base == info.pgm_base && user_data == info.user_data.data() &&
               flat == info.flattened_ud_buf;
    }
};
std::array<PreparedRefreshRecord, 2> prepared_refresh_records;

// Every initialized field; the raw representation is not compared here.
bool SameRuntimeInfo(const Shader::RuntimeInfo& a, const Shader::RuntimeInfo& b) {
    return a.stage == b.stage && a.num_user_data == b.num_user_data &&
           a.num_input_vgprs == b.num_input_vgprs &&
           a.num_allocated_vgprs == b.num_allocated_vgprs &&
           a.fp_denorm_mode32 == b.fp_denorm_mode32 &&
           a.fp_denorm_mode16_64 == b.fp_denorm_mode16_64 &&
           a.fp_round_mode32 == b.fp_round_mode32 && a.fp_round_mode16_64 == b.fp_round_mode16_64 &&
           a == b;
}
} // namespace

void PipelineCache::NoteStageSelection(LogicalStage l_stage, Program* program, size_t permutation,
                                       bool compiled) {
    if (PreparedPipelinesEnabled()) {
        stage_selections[u32(l_stage)] = {program, permutation, compiled};
    }
}

// Read once: EmulatorSettings takes a mutex and copies a shared_ptr per access, and this ran per
// draw (~0.7% of CP wall time, 2026-09-24). Shader collection is a launch-time setting.
static bool ShaderCollectCached() {
    static const bool collect = EmulatorSettings.IsShaderCollect();
    return collect;
}

// SHADPS4_ASYNC_PIPELINES, off under shader collection: that path records module/pipeline
// relations at creation and erases map entries on hot reload, both from the creating thread.
static bool AsyncPipelinesEnabled() {
    return g_async_pipelines && !ShaderCollectCached();
}

size_t PipelineCache::PreparedPipelineWays() {
    return std::clamp<size_t>(g_prepared_ways_env, 1, PreparedPipelineMaxWays);
}

PipelineCache::PreparedPipelineEntry* PipelineCache::TryPreparedPipeline(
    const AmdGpu::Regs& regs, PreparedGraphicsState& state, bool& eligible) {
    constexpr auto fs_idx = u32(LogicalStage::Fragment);
    constexpr auto vs_idx = u32(LogicalStage::Vertex);
    auto& stats = prepared_stats;
    if ((++stats.attempts & 0x3ffff) == 0 && !::Common::Lean()) {
        LOG_INFO(Render_Vulkan,
                 "PreparedPipelines: attempts={} ineligible={} candidates={} stale={} "
                 "guard_fallbacks={} hits={} seeds={} seed_rejects={} verified={} input_changes={} "
                 "evictions={} ways={}",
                 stats.attempts, stats.ineligible, stats.candidates, stats.stale,
                 stats.guard_fallbacks, stats.hits, stats.seeds, stats.seed_rejects,
                 stats.verified, stats.input_changes, stats.evictions, PreparedPipelineWays());
    }

    eligible = false;
    // Same stage/program presence tests as RefreshGraphicsStages for the plain VS+PS case.
    // Shader collection records module/pipeline relations on the original path only.
    if (regs.stage_enable.raw != AmdGpu::ShaderStageEnable::VgtStages::Vs ||
        !regs.ps_program.Address<u32*>() || !regs.vs_program.Address<u32*>() ||
        ShaderCollectCached()) {
        ++stats.ineligible;
        return nullptr;
    }
    const auto fs_params = AmdGpu::GetParams(regs.ps_program);
    const auto vs_params = AmdGpu::GetParams(regs.vs_program);
    if (!state.Capture(regs, fs_params.hash, vs_params.hash)) {
        ++stats.ineligible;
        return nullptr;
    }
    eligible = true;

    if (prepared_pipelines.empty()) {
        prepared_pipelines.resize(PreparedPipelineSets * PreparedPipelineWays());
        prepared_tags.assign(PreparedPipelineSets * PreparedPipelineWays(), 0);
    }
    const u64 state_hash = state.Hash();
    const size_t set_base = (state_hash & (PreparedPipelineSets - 1)) * PreparedPipelineWays();
    auto* set = &prepared_pipelines[set_base];
    const u64* tags = &prepared_tags[set_base];
    const bool use_tags = PreparedTags();
    const bool check_formats = !instance.IsVertexInputDynamicState();
    bool refreshed = false;
    const size_t num_ways = PreparedPipelineWays();
    for (size_t way = 0; way < num_ways; ++way) {
        if (use_tags && tags[way] != state_hash) {
            continue;
        }
        auto& entry = set[way];
        if (!entry.valid || entry.state != state) {
            continue;
        }
        if (entry.epoch != prepared_pipeline_epoch ||
            entry.fs_program->generation != entry.fs_generation ||
            entry.vs_program->generation != entry.vs_generation) {
            entry.valid = false;
            ++stats.stale;
            continue;
        }
        // Equal hashes name the same never-erased Program objects as at seeding. Several
        // ways may share a snapshot and differ only in their guards, so refresh once.
        auto& fs_info = entry.fs_program->info;
        auto& vs_info = entry.vs_program->info;
        if (!refreshed) {
            ++stats.candidates;
            // Live per-draw guest state, in the original FS-then-VS order. The walker
            // rereads memory behind extended-user-data pointers on every attempt.
            fs_info.pgm_base = fs_params.Base();
            fs_info.user_data = fs_params.user_data;
            fs_info.RefreshFlatBuf();
            vs_info.pgm_base = vs_params.Base();
            vs_info.user_data = vs_params.user_data;
            vs_info.RefreshFlatBuf();
            refreshed = true;
        }
        if (!entry.fs_guard.MatchesInputs(fs_info) || !entry.vs_guard.MatchesInputs(vs_info)) {
            continue;
        }
        if (check_formats && entry.fetch_shader) {
            // Without dynamic vertex input the key specializes on the current formats.
            const auto& formats = entry.pipeline->GetGraphicsKey().vertex_buffer_formats;
            const auto& attributes = entry.fetch_shader->attributes;
            bool same = attributes.size() <= MaxVertexBufferCount;
            for (size_t i = 0; same && i < attributes.size(); ++i) {
                const auto buffer = attributes[i].GetSharp(vs_info);
                same = formats[i] == LiverpoolToVK::SurfaceFormat(buffer.GetDataFmt(),
                                                                  buffer.GetNumberFmt());
            }
            if (!same) {
                continue;
            }
        }

        ++stats.hits;
        entry.last_use = ++prepared_clock;
        if (PreparedPipelinesVerify()) {
            prepared_refresh_records[0].Save(fs_info);
            prepared_refresh_records[1].Save(vs_info);
        }
        // Everything the original selection leaves in the cache, from owned storage.
        infos.fill(nullptr);
        modules.fill(nullptr);
        infos[fs_idx] = &fs_info;
        infos[vs_idx] = &vs_info;
        modules[fs_idx] = entry.fs_module;
        modules[vs_idx] = entry.vs_module;
        std::memcpy(&runtime_infos[fs_idx], entry.fs_guard.runtime.data(),
                    sizeof(Shader::RuntimeInfo));
        std::memcpy(&runtime_infos[vs_idx], entry.vs_guard.runtime.data(),
                    sizeof(Shader::RuntimeInfo));
        graphics_key = entry.pipeline->GetGraphicsKey();
        if (fetch_shader != entry.fetch_shader) {
            fetch_shader = entry.fetch_shader;
        }
        return &entry;
    }
    if (refreshed) {
        ++stats.guard_fallbacks;
    }
    return nullptr;
}

void PipelineCache::SeedPreparedPipeline(const PreparedGraphicsState& state,
                                         const GraphicsPipeline* pipeline) {
    constexpr auto fs_idx = u32(LogicalStage::Fragment);
    constexpr auto vs_idx = u32(LogicalStage::Vertex);
    using Guard = Shader::PreparedSpecialization;
    const auto& fs_sel = stage_selections[fs_idx];
    const auto& vs_sel = stage_selections[vs_idx];
    bool ok = pipeline && fs_sel.program && vs_sel.program && fs_sel.program != vs_sel.program &&
              !fs_sel.compiled && !vs_sel.compiled;
    for (u32 i = 0; ok && i < MaxShaderStages; ++i) {
        ok = i == fs_idx   ? infos[i] == &fs_sel.program->info
             : i == vs_idx ? infos[i] == &vs_sel.program->info
                           : infos[i] == nullptr;
    }
    if (ok) {
        const auto& fs_info = fs_sel.program->info;
        const auto& vs_info = vs_sel.program->info;
        const auto& fs_modules = fs_sel.program->modules;
        const auto& vs_modules = vs_sel.program->modules;
        ok = fs_info.pgm_hash == state.fs_hash && vs_info.pgm_hash == state.vs_hash &&
             Guard::Eligible(fs_info, Stage::Fragment, LogicalStage::Fragment) &&
             Guard::Eligible(vs_info, Stage::Vertex, LogicalStage::Vertex) &&
             fs_sel.permutation < fs_modules.size() && vs_sel.permutation < vs_modules.size() &&
             fs_modules[fs_sel.permutation].module == modules[fs_idx] &&
             vs_modules[vs_sel.permutation].module == modules[vs_idx] &&
             vs_modules[vs_sel.permutation].fetch_shader == fetch_shader;
    }
    if (!ok) {
        ++prepared_stats.seed_rejects;
        return;
    }

    // Capture the guards belonging to the actual original selection. Re-parsing fetch code
    // here could pair a newly changed guest fetch program with the previously selected module.
    // If the retained per-program cache is disabled or could not capture, fall back normally.
    const auto& fs_info = fs_sel.program->info;
    const auto& vs_info = vs_sel.program->info;
    const auto* fs_guard = fs_sel.program->prepared.get();
    const auto* vs_guard = vs_sel.program->prepared.get();
    Shader::Backend::Bindings binding{};
    if (!fs_guard || !vs_guard || fs_guard->permutation != fs_sel.permutation ||
        vs_guard->permutation != vs_sel.permutation ||
        !fs_guard->Matches(fs_info, runtime_infos[fs_idx], binding)) {
        ++prepared_stats.seed_rejects;
        return;
    }
    fs_info.AddBindings(binding);
    if (!vs_guard->Matches(vs_info, runtime_infos[vs_idx], binding)) {
        ++prepared_stats.seed_rejects;
        return;
    }

    const u64 state_hash = state.Hash();
    const size_t set_base = (state_hash & (PreparedPipelineSets - 1)) * PreparedPipelineWays();
    auto* set = &prepared_pipelines[set_base];
    PreparedPipelineEntry* entry = nullptr;
    const size_t num_ways = PreparedPipelineWays();
    for (size_t way = 0; way < num_ways && !entry; ++way) {
        if (!set[way].valid || set[way].epoch != prepared_pipeline_epoch) {
            entry = &set[way];
        }
    }
    if (!entry) {
        // SHADPS4_PREPARED_LRU=1: evict the least recently hit way instead of round-robin.
        const bool lru = g_prepared_lru;
        if (lru) {
            entry = &set[0];
            for (size_t way = 1; way < num_ways; ++way) {
                if (set[way].last_use < entry->last_use) {
                    entry = &set[way];
                }
            }
        } else {
            entry = &set[prepared_pipeline_victim++ % num_ways];
        }
        ++prepared_stats.evictions;
    }

    entry->valid = false;
    entry->fs_guard = *fs_guard;
    entry->vs_guard = *vs_guard;
    entry->state = state;
    entry->epoch = prepared_pipeline_epoch;
    entry->fs_program = fs_sel.program;
    entry->vs_program = vs_sel.program;
    entry->fs_generation = fs_sel.program->generation;
    entry->vs_generation = vs_sel.program->generation;
    entry->pipeline = pipeline;
    entry->fs_module = modules[fs_idx];
    entry->vs_module = modules[vs_idx];
    entry->fetch_shader = fetch_shader;
    entry->last_use = ++prepared_clock;
    entry->valid = true;
    prepared_tags[entry - prepared_pipelines.data()] = state_hash;
    ++prepared_stats.seeds;
}

void PipelineCache::VerifyPreparedPipeline(PreparedPipelineEntry& entry,
                                           const GraphicsPipeline* pipeline, bool is_new) {
    constexpr auto fs_idx = u32(LogicalStage::Fragment);
    constexpr auto vs_idx = u32(LogicalStage::Vertex);
    const auto& fs_sel = stage_selections[fs_idx];
    const auto& vs_sel = stage_selections[vs_idx];
    // The oracle deliberately performs the original live refresh. Guest descriptors or
    // fetch code can change between the two observations. Such a draw is unverified, not
    // a cache mismatch; return the original result and discard this entry. This check
    // precedes selection assertions, so legitimate changed-input compilations are allowed.
    const bool changed_program =
        (fs_sel.program && fs_sel.program->info.pgm_hash != entry.state.fs_hash) ||
        (vs_sel.program && vs_sel.program->info.pgm_hash != entry.state.vs_hash);
    if (changed_program || !prepared_refresh_records[0].Same(entry.fs_program->info) ||
        !prepared_refresh_records[1].Same(entry.vs_program->info) ||
        !entry.fs_guard.MatchesInputs(entry.fs_program->info) ||
        !entry.vs_guard.MatchesInputs(entry.vs_program->info)) {
        ++prepared_stats.input_changes;
        entry.valid = false;
        return;
    }
    ASSERT_MSG(!is_new && pipeline == entry.pipeline,
               "PreparedPipelines selected a different pipeline: fs={:#x} vs={:#x}",
               entry.state.fs_hash, entry.state.vs_hash);
    ASSERT_MSG(graphics_key == entry.pipeline->GetGraphicsKey(), "PreparedPipelines key mismatch");
    for (u32 i = 0; i < MaxShaderStages; ++i) {
        const Shader::Info* info = i == fs_idx   ? &entry.fs_program->info
                                   : i == vs_idx ? &entry.vs_program->info
                                                 : nullptr;
        const vk::ShaderModule module = i == fs_idx   ? entry.fs_module
                                        : i == vs_idx ? entry.vs_module
                                                      : vk::ShaderModule{};
        ASSERT_MSG(infos[i] == info && modules[i] == module,
                   "PreparedPipelines stage {} info/module mismatch", i);
    }
    ASSERT_MSG(fs_sel.program == entry.fs_program && vs_sel.program == entry.vs_program &&
                   !fs_sel.compiled && !vs_sel.compiled &&
                   fs_sel.permutation == entry.fs_guard.permutation &&
                   vs_sel.permutation == entry.vs_guard.permutation,
               "PreparedPipelines program/permutation mismatch");
    ASSERT_MSG(fetch_shader == entry.fetch_shader, "PreparedPipelines fetch metadata mismatch");
    Shader::RuntimeInfo fs_runtime{}, vs_runtime{};
    std::memcpy(&fs_runtime, entry.fs_guard.runtime.data(), sizeof(fs_runtime));
    std::memcpy(&vs_runtime, entry.vs_guard.runtime.data(), sizeof(vs_runtime));
    ASSERT_MSG(SameRuntimeInfo(runtime_infos[fs_idx], fs_runtime) &&
                   SameRuntimeInfo(runtime_infos[vs_idx], vs_runtime),
               "PreparedPipelines runtime info mismatch");
    ++prepared_stats.verified;
}

const GraphicsPipeline* PipelineCache::GetGraphicsPipeline() {
    auto lk = LockExclusiveServicing();
    return GraphicsLookupLocked(liverpool->regs);
}

std::unique_lock<std::shared_mutex> PipelineCache::LockExclusiveServicing() const {
    std::unique_lock lk{graphics_mutex, std::defer_lock};
    if (!AmdGpu::t_lock_service) [[likely]] {
        lk.lock();
        return lk;
    }
    bool counted = false;
    for (u32 spins = 0; !lk.try_lock(); ++spins) {
        if (!counted) {
            AmdGpu::g_lean_l2.lock_service_waits.fetch_add(1, std::memory_order_relaxed);
            counted = true;
        }
        AmdGpu::t_lock_service(AmdGpu::t_lock_service_ctx);
        spins < 64 ? _mm_pause() : std::this_thread::yield();
    }
    return lk;
}

void PipelineCache::PrepareNativeVariantRefreshed(const GraphicsPipeline* base,
                                                  const AmdGpu::Regs& regs) {
    std::unique_lock lk{graphics_mutex};
    (void)GraphicsLookupLocked(regs); // the shared Infos now hold this draw's per-draw fields
    PrepareNativeVariantLocked(base, regs);
}

void PipelineCache::PrepareNativeColourRefreshed(const GraphicsPipeline* base,
                                                 const AmdGpu::CpSplit::VertexInputsDerived& vi,
                                                 const AmdGpu::Regs& regs) {
    std::unique_lock lk{graphics_mutex};
    (void)GraphicsLookupLocked(regs);
    PrepareNativeColourLocked(base, vi, regs);
}

const GraphicsPipeline* PipelineCache::GetGraphicsPipelineLocked() {
    return GraphicsLookupLocked(liverpool->regs);
}

const GraphicsPipeline* PipelineCache::GraphicsLookupLocked(const AmdGpu::Regs& regs) {
    // MEASUREMENT (perf-forest): GetGraphicsPipeline is 0.99 us/draw = 6.3 ms/frame and never
    // split. Before designing a memo, learn (a) how the cost divides between RefreshGraphicsKey
    // (memset + wide reg read + stage resolve) and the robin_map lookup, and (b) how often the
    // freshly built key equals the previous draw's - the hit rate a memo would ride on. Windowed so
    // a stable-scene line is not diluted by boot. ~3 clock reads + one key compare per draw, the
    // memo's own cost.
    PreparedGraphicsState prepared_state;
    PreparedPipelineEntry* prepared_hit = nullptr;
    bool prepared_eligible = false;
    if (PreparedPipelinesEnabled()) {
        prepared_hit = TryPreparedPipeline(regs, prepared_state, prepared_eligible);
        if (prepared_hit && !PreparedPipelinesVerify()) {
            return prepared_hit->pipeline;
        }
        if (prepared_eligible) {
            stage_selections.fill({});
        }
    }
    if (async_pending != 0 && async_compiler->HasFinished()) {
        PublishAsyncPipelines();
    }
    lookup_translate_ms = 0;
    const auto t_probe0 = Common::FaultMonitor::Now();
    if (!RefreshGraphicsKey(regs)) {
        ASSERT_MSG(!prepared_hit, "PreparedPipelines hit a draw the original path rejects");
        return nullptr;
    }
    const auto t_probe1 = Common::FaultMonitor::Now();
    const auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    const auto t_probe2 = Common::FaultMonitor::Now();
    if constexpr (Common::FaultMonitor::DetailCountersEnabled) {
        static GraphicsPipelineKey probe_last_key{};
        static u64 probe_last_gen = 0;
        static bool probe_have_last = false;
        static u64 probe_calls = 0, probe_same = 0, probe_gen_unchanged = 0, probe_gen_mismatch = 0,
                   probe_refresh_ns = 0, probe_lookup_ns = 0;
        const u64 gen = liverpool->regs_generation_ctx; // context/uconfig/config/clear only, no SH
        const bool key_same = probe_have_last && graphics_key == probe_last_key;
        const bool gen_same = probe_have_last && gen == probe_last_gen;
        if (key_same) {
            ++probe_same;
        }
        if (gen_same) {
            // gen_unchanged is the memo's real hit rate; a mismatch (gen unchanged yet key differs)
            // means the generation signal misses a register the key reads - it MUST stay 0 before
            // any build is allowed to actually skip RefreshGraphicsKey on this signal.
            ++probe_gen_unchanged;
            if (!key_same) {
                ++probe_gen_mismatch;
            }
        }
        probe_last_key = graphics_key;
        probe_last_gen = gen;
        probe_have_last = true;
        ++probe_calls;
        probe_refresh_ns += Common::FaultMonitor::DiffNs(t_probe0, t_probe1);
        probe_lookup_ns += Common::FaultMonitor::DiffNs(t_probe1, t_probe2);
        if (probe_calls >= 200000) {
            LOG_INFO(Render_Vulkan,
                     "PipelineProbe: calls={} same_key={:.1f}% vctxgen_unchanged={:.1f}% "
                     "vctxgen_mismatch={} refresh={:.3f}us/call lookup={:.3f}us/call",
                     probe_calls, 100.0 * static_cast<double>(probe_same) / probe_calls,
                     100.0 * static_cast<double>(probe_gen_unchanged) / probe_calls,
                     probe_gen_mismatch, static_cast<double>(probe_refresh_ns) / 1e3 / probe_calls,
                     static_cast<double>(probe_lookup_ns) / 1e3 / probe_calls);
            probe_calls = probe_same = probe_gen_unchanged = probe_gen_mismatch = probe_refresh_ns =
                probe_lookup_ns = 0;
        }
    }
    if (is_new) {
        const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(graphics_key);
        LOG_INFO(Render_Vulkan, "Compiling graphics pipeline {:#x}", pipeline_hash);

        // A draw that carries a register-driven depth or stencil clear is not skipped: the clear
        // is that draw's own load op (Rasterizer depth attachment), so skipping it would leave
        // the frame's depth/stencil uncleared (Opus review 2026-10-01). It compiles in line.
        const auto& drc = regs.depth_render_control;
        if (AsyncPipelinesEnabled() && !drc.depth_clear_enable && !drc.stencil_clear_enable) {
            // The entry stays null until a later lookup publishes the compiler's result; this
            // draw and every draw of the key until then are skipped. Nothing below (memo seed,
            // collection) sees a null pipeline.
            QueueAsyncPipeline(pipeline_hash);
            fetch_shader.reset();
            ++compile_stats.skipped_draws;
            return nullptr;
        }

        GraphicsPipeline::SerializationSupport sdata{};
        const auto t_create = CompileClock::now();
        it.value() = std::make_unique<GraphicsPipeline>(
            instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
            runtime_infos, fetch_shader, modules, sdata, false);
        const double create_ms = MsSince(t_create);

        RegisterPipelineData(graphics_key, pipeline_hash, sdata);
        ++num_new_pipelines;

        if (EmulatorSettings.IsShaderCollect()) {
            for (auto stage = 0; stage < MaxShaderStages; ++stage) {
                if (infos[stage]) {
                    auto& m = modules[stage];
                    module_related_pipelines[m].emplace_back(graphics_key);
                }
            }
        }
        fetch_shader.reset();
        NoteCompiledPipeline(pipeline_hash, lookup_translate_ms, create_ms, 0, false);
    } else if (!it->second) {
        // Pending in the compiler: entries are null only while SHADPS4_ASYNC_PIPELINES is on.
        // Not memoised: a prepared entry is seeded from a non-null pipeline only.
        ++compile_stats.skipped_draws;
        return nullptr;
    }
    if (prepared_hit) {
        VerifyPreparedPipeline(*prepared_hit, it->second.get(), is_new);
    } else if (prepared_eligible && !is_new) {
        // Only a complete original selection of an existing pipeline is ever captured.
        SeedPreparedPipeline(prepared_state, it->second.get());
    }
    return it->second.get();
}

void PipelineCache::PrepareNativeVariant(const GraphicsPipeline* base) {
    std::unique_lock lk{graphics_mutex};
    PrepareNativeVariantLocked(base, liverpool->regs);
}

void PipelineCache::PrepareNativeVariant(const GraphicsPipeline* base, const AmdGpu::Regs& regs) {
    std::unique_lock lk{graphics_mutex};
    PrepareNativeVariantLocked(base, regs);
}

void PipelineCache::PrepareNativeVariantLocked(const GraphicsPipeline* base,
                                               const AmdGpu::Regs& regs) {
    // Exclusive lock held; registers = the draw that just looked `base` up. Decided once per
    // pipeline.
    base->native_checked = true;
    const auto why_not = [&]() -> const char* {
        if (regs.stage_enable.raw != AmdGpu::ShaderStageEnable::VgtStages::Vs) {
            return "stages";
        }
        const auto stages = base->GetStages();
        const auto* vs = stages[u32(LogicalStage::Vertex)];
        const auto* fs = stages[u32(LogicalStage::Fragment)];
        if (!vs) {
            return "no vs";
        }
        for (u32 i = 0; i < stages.size(); ++i) {
            if (stages[i] && i != u32(LogicalStage::Vertex) && i != u32(LogicalStage::Fragment)) {
                return "extra stage";
            }
        }
        if (fs && (!fs->buffers.empty() || !fs->images.empty() || !fs->samplers.empty() ||
                   fs->ud_mask.mask != 0 || fs->uses_dma)) {
            return "fs resources";
        }
        if (!vs->images.empty() || !vs->samplers.empty() || vs->uses_dma) {
            return "vs images";
        }
        if (vs->loads.Get(Shader::IR::Attribute::VertexId)) {
            // Opus review #3: a rebased vertexOffset shifts VertexId.
            return "vs reads VertexId";
        }
        if (vs->buffers.empty() || vs->buffers.size() >= Shader::NUM_BUFFERS) {
            return "vs buffer count";
        }
        for (const auto& desc : vs->buffers) {
            if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                continue;
            }
            if (desc.buffer_type != Shader::BufferType::Guest || desc.is_written ||
                desc.is_formatted || !desc.IsStorage(desc.GetSharp(*vs))) {
                return "vs buffer kind";
            }
            // Native loads apply the guest bound for 32-bit loads only (Opus review #1).
            if (True(desc.used_types &
                     (Shader::IR::Type::U8 | Shader::IR::Type::U16 | Shader::IR::Type::U64))) {
                return "vs buffer load width";
            }
        }
        return nullptr;
    }();
    static u32 logged = 0;
    if (why_not) {
        if (logged++ < 64) {
            LOG_INFO(Render_Vulkan, "NativePass: pipeline {:#x} not eligible ({})",
                     std::hash<GraphicsPipelineKey>{}(base->GetGraphicsKey()), why_not);
        }
        return;
    }
    // Rebuild the stage selection for the current registers (members infos / modules /
    // runtime_infos / fetch_shader) and make sure it is `base`.
    if (!RefreshGraphicsKey(regs) || !(graphics_key == base->GetGraphicsKey())) {
        if (logged++ < 64) {
            LOG_WARNING(Render_Vulkan, "NativePass: pipeline {:#x} key did not rebuild",
                        std::hash<GraphicsPipelineKey>{}(base->GetGraphicsKey()));
        }
        return;
    }
    // Binding numbering as GraphicsPipeline::CollectDescSetBindings / the stage loop: every
    // stage in LogicalStage order advances the counters by its shared Info.
    Shader::Backend::Bindings total{};
    std::array<Shader::Backend::Bindings, MaxShaderStages> starts{};
    // RefreshGraphicsStages binds the fragment stage first, then the vertex stage.
    const std::array<LogicalStage, 2> order{LogicalStage::Fragment, LogicalStage::Vertex};
    for (const auto l : order) {
        starts[u32(l)] = total;
        if (infos[u32(l)]) {
            infos[u32(l)]->AddBindings(total);
        }
    }
    const u32 records_binding = total.unified;
    auto native_modules = modules;
    {
        const u32 vs_idx = u32(LogicalStage::Vertex);
        const auto* pgm = regs.ProgramForStage(u32(Shader::Stage::Vertex));
        if (!pgm || !pgm->Address<u32*>()) {
            return;
        }
        const auto params = AmdGpu::GetParams(*pgm);
        auto new_info = Shader::Info(Shader::Stage::Vertex, LogicalStage::Vertex, params);
        new_info.pinned_uses_dma = infos[vs_idx]->uses_dma;
        auto runtime_info = runtime_infos[vs_idx];
        auto binding = starts[vs_idx];
        struct NativeEmitScope {
            NativeEmitScope(u32 rb) {
                Shader::t_native_emit = {true, rb};
                Shader::t_native_emit.ring = Shader::g_native_ring;
                Shader::t_native_emit.table_binding = rb + 1;
                // SHADPS4_NATIVE_PT: the page table is read by pointer (no binding); =2 the
                // compare SSBO after the table, as the GraphicsPipeline constructor appends it.
                Shader::t_native_emit.pt = Shader::g_native_pt != 0 && Shader::g_native_ring;
                Shader::t_native_emit.pt_compare = Shader::g_native_pt == 2 && Shader::g_native_ring;
                Shader::t_native_emit.pt_debug_binding = rb + 2;
                Shader::t_native_emit.seg = Shader::g_native_seg != 0 && Shader::t_native_emit.pt;
                Shader::t_native_emit.seg_compare =
                    Shader::g_native_seg_compare && Shader::t_native_emit.pt;
            }
            ~NativeEmitScope() {
                Shader::t_native_emit = {};
            }
        };
        std::vector<u32> spv;
        {
            NativeEmitScope scope{records_binding};
            const auto ir_program = Shader::TranslateProgram(params.code, pools, new_info,
                                                             runtime_info, profile);
            spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, binding);
        }
        if (!SameBindingLayout(new_info, *infos[vs_idx])) {
            LOG_ERROR(Render_Vulkan, "NativePass: vs_{:#x} native layout differs; not used",
                      params.hash);
            return;
        }
        DumpShader(spv, params.hash, Shader::Stage::Vertex, 0x4e41, "native.spv");
        native_modules[vs_idx] = CompileSPV(spv, instance.GetDevice());
        Vulkan::SetObjectName(instance.GetDevice(), native_modules[vs_idx], "vs_{:#x}_native",
                              params.hash);
    }
    GraphicsPipeline::SerializationSupport sdata{};
    LOG_INFO(Render_Vulkan, "NativePass: compiling variant of {:#x} (vs {:#x})",
             std::hash<GraphicsPipelineKey>{}(graphics_key),
             infos[u32(LogicalStage::Vertex)] ? infos[u32(LogicalStage::Vertex)]->pgm_hash : 0);
    auto variant = std::make_unique<GraphicsPipeline>(
        instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
        runtime_infos, fetch_shader, native_modules, sdata, false, nullptr, records_binding,
        false, /*refusable=*/true);
    if (!variant->IsCreated()) {
        // 1.1.1 (GitHub #25, #31): a variant the driver refuses is left out; the base draws.
        LOG_WARNING(Render_Vulkan, "NativePass: variant of {:#x} refused by the driver - not used",
                    std::hash<GraphicsPipelineKey>{}(graphics_key));
        native_variants.push_back(std::move(variant));
        return;
    }
    if (!variant->UsesPushDescriptors()) {
        LOG_WARNING(Render_Vulkan, "NativePass: variant of {:#x} has no push descriptors",
                    std::hash<GraphicsPipelineKey>{}(graphics_key));
        native_variants.push_back(std::move(variant));
        return;
    }
    base->native_variant.store(variant.get(), std::memory_order_release);
    if (logged++ < 64) {
        LOG_INFO(Render_Vulkan, "NativePass: variant of pipeline {:#x} built (records binding {})",
                 std::hash<GraphicsPipelineKey>{}(graphics_key), records_binding);
    }
    native_variants.push_back(std::move(variant));
}

namespace {
/// SHADPS4_NATIVE_COLOUR: how the native VS decodes a vertex input format (Vulkan's own
/// conversion); kind None = not pullable.
Shader::NativeVtxAttr NativeVtxFromFormat(vk::Format f, u32 stride) {
    using K = Shader::NativeVtxKind;
    const auto a = [&](K kind, u8 bits, u8 count) {
        return Shader::NativeVtxAttr{kind, bits, count, 1, stride};
    };
    switch (f) {
    case vk::Format::eR32Sfloat: return a(K::Float, 32, 1);
    case vk::Format::eR32G32Sfloat: return a(K::Float, 32, 2);
    case vk::Format::eR32G32B32Sfloat: return a(K::Float, 32, 3);
    case vk::Format::eR32G32B32A32Sfloat: return a(K::Float, 32, 4);
    case vk::Format::eR32Uint: return a(K::Uint, 32, 1);
    case vk::Format::eR32G32Uint: return a(K::Uint, 32, 2);
    case vk::Format::eR32G32B32Uint: return a(K::Uint, 32, 3);
    case vk::Format::eR32G32B32A32Uint: return a(K::Uint, 32, 4);
    case vk::Format::eR32Sint: return a(K::Sint, 32, 1);
    case vk::Format::eR32G32Sint: return a(K::Sint, 32, 2);
    case vk::Format::eR32G32B32Sint: return a(K::Sint, 32, 3);
    case vk::Format::eR32G32B32A32Sint: return a(K::Sint, 32, 4);
    case vk::Format::eA2B10G10R10UnormPack32: return a(K::Unorm, 0, 4);
    case vk::Format::eA2B10G10R10SnormPack32: return a(K::Snorm, 0, 4);
    case vk::Format::eA2B10G10R10UintPack32: return a(K::Uint16, 0, 4);
    case vk::Format::eA2B10G10R10SintPack32: return a(K::Sint16, 0, 4);
    case vk::Format::eA2B10G10R10UscaledPack32: return a(K::Uscaled, 0, 4);
    case vk::Format::eA2B10G10R10SscaledPack32: return a(K::Sscaled, 0, 4);
    case vk::Format::eR16Sfloat: return a(K::Half, 16, 1);
    case vk::Format::eR16Unorm: return a(K::Unorm, 16, 1);
    case vk::Format::eR16Snorm: return a(K::Snorm, 16, 1);
    case vk::Format::eR16Uint: return a(K::Uint16, 16, 1);
    case vk::Format::eR16Sint: return a(K::Sint16, 16, 1);
    case vk::Format::eR16G16B16Sfloat: return a(K::Half, 16, 3);
    case vk::Format::eR16G16B16Unorm: return a(K::Unorm, 16, 3);
    case vk::Format::eR16G16B16Snorm: return a(K::Snorm, 16, 3);
    case vk::Format::eR8Unorm: return a(K::Unorm, 8, 1);
    case vk::Format::eR8Snorm: return a(K::Snorm, 8, 1);
    case vk::Format::eR8Uint: return a(K::Uint16, 8, 1);
    case vk::Format::eR8Sint: return a(K::Sint16, 8, 1);
    case vk::Format::eR8G8Unorm: return a(K::Unorm, 8, 2);
    case vk::Format::eR8G8Snorm: return a(K::Snorm, 8, 2);
    case vk::Format::eR8G8Uint: return a(K::Uint16, 8, 2);
    case vk::Format::eR8G8Sint: return a(K::Sint16, 8, 2);
    case vk::Format::eR16G16Sfloat: return a(K::Half, 16, 2);
    case vk::Format::eR16G16B16A16Sfloat: return a(K::Half, 16, 4);
    case vk::Format::eR16G16Unorm: return a(K::Unorm, 16, 2);
    case vk::Format::eR16G16B16A16Unorm: return a(K::Unorm, 16, 4);
    case vk::Format::eR16G16Snorm: return a(K::Snorm, 16, 2);
    case vk::Format::eR16G16B16A16Snorm: return a(K::Snorm, 16, 4);
    case vk::Format::eR16G16Uint: return a(K::Uint16, 16, 2);
    case vk::Format::eR16G16B16A16Uint: return a(K::Uint16, 16, 4);
    case vk::Format::eR16G16Sint: return a(K::Sint16, 16, 2);
    case vk::Format::eR16G16B16A16Sint: return a(K::Sint16, 16, 4);
    case vk::Format::eR16G16Uscaled: return a(K::Uscaled, 16, 2);
    case vk::Format::eR16G16B16A16Uscaled: return a(K::Uscaled, 16, 4);
    case vk::Format::eR16G16Sscaled: return a(K::Sscaled, 16, 2);
    case vk::Format::eR16G16B16A16Sscaled: return a(K::Sscaled, 16, 4);
    case vk::Format::eR8G8B8A8Unorm: return a(K::Unorm, 8, 4);
    case vk::Format::eR8G8B8A8Snorm: return a(K::Snorm, 8, 4);
    case vk::Format::eR8G8B8A8Uint: return a(K::Uint16, 8, 4);
    case vk::Format::eR8G8B8A8Sint: return a(K::Sint16, 8, 4);
    case vk::Format::eR8G8B8A8Uscaled: return a(K::Uscaled, 8, 4);
    case vk::Format::eR8G8B8A8Sscaled: return a(K::Sscaled, 8, 4);
    default: return {};
    }
}

/// The number class the native VS will hand out for a kind (must match the input variable's
/// class, which comes from the V# number format).
AmdGpu::NumberClass NativeVtxClass(Shader::NativeVtxKind k) {
    using K = Shader::NativeVtxKind;
    switch (k) {
    case K::Uint:
    case K::Uint16:
        return AmdGpu::NumberClass::Uint;
    case K::Sint:
    case K::Sint16:
        return AmdGpu::NumberClass::Sint;
    default:
        return AmdGpu::NumberClass::Float;
    }
}

const char* NativeBufferWhyNot(const Shader::Info& info, u32 narrow_diag_bit = 4) {
    if (info.uses_dma) {
        return "dma";
    }
    if (info.buffers.size() >= Shader::NUM_BUFFERS) {
        return "buffer count";
    }
    for (const auto& desc : info.buffers) {
        if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
            continue;
        }
        if (desc.buffer_type != Shader::BufferType::Guest) {
            return "buffer type";
        }
        if (desc.is_written) {
            return "buffer written";
        }
        if (desc.is_formatted && (VideoCore::g_native_colour_diag & 8)) {
            return "buffer formatted (diag)";
        }
        if (True(desc.used_types & (Shader::IR::Type::U8 | Shader::IR::Type::U16)) &&
            (VideoCore::g_native_colour_diag & (4 | narrow_diag_bit))) {
            return "buffer narrow loads (diag)";
        }
        // Formatted (typed) reads are lowered to raw loads in the shader; the join passes
        // is_texel to ObtainBuffer like BindBuffers (an image->buffer sync records a copy,
        // which refuses the join).
        if (!desc.IsStorage(desc.GetSharp(info))) {
            return "buffer not storage";
        }
        // 8 / 16 / 32-bit loads apply the guest bound (EmitLoadBufferU8 / U16 / B32xN).
        if (True(desc.used_types & Shader::IR::Type::U64)) {
            return "buffer load width";
        }
    }
    for (const auto& desc : info.images) {
        if (desc.is_written) {
            return "image written";
        }
    }
    return nullptr;
}
} // namespace

void PipelineCache::PrepareNativeColour(const GraphicsPipeline* base,
                                        const AmdGpu::CpSplit::VertexInputsDerived& vi) {
    std::unique_lock lk{graphics_mutex};
    PrepareNativeColourLocked(base, vi, liverpool->regs);
}

void PipelineCache::PrepareNativeColour(const GraphicsPipeline* base,
                                        const AmdGpu::CpSplit::VertexInputsDerived& vi,
                                        const AmdGpu::Regs& regs) {
    std::unique_lock lk{graphics_mutex};
    PrepareNativeColourLocked(base, vi, regs);
}

void PipelineCache::PrepareNativeColourLocked(const GraphicsPipeline* base,
                                              const AmdGpu::CpSplit::VertexInputsDerived& vi,
                                              const AmdGpu::Regs& regs) {
    // Exclusive lock held; right after FRONT_DERIVE: registers and `vi` = the draw that looked
    // `base` up. Static eligibility once per pipeline, then one variant per vertex description.
    if (base->native_colour_static == 2) {
        return;
    }
    const auto stages = base->GetStages();
    const auto* vs = stages[u32(LogicalStage::Vertex)];
    const auto* fs = stages[u32(LogicalStage::Fragment)];
    GraphicsPipeline::NativeColourSlot* claimed = nullptr; // SHADPS4_NATIVE_SEGB census
    const auto log_not = [&](const char* why) {
        base->native_colour_why.store(why, std::memory_order_relaxed);
        if (claimed) {
            claimed->state.store(2, std::memory_order_release);
            g_nc_claims[1].fetch_add(1, std::memory_order_relaxed);
        }
        if (native_colour_logged++ < 2000) {
            LOG_INFO(Render_Vulkan, "NativeColour: pipeline {:#x} not eligible ({})",
                     std::hash<GraphicsPipelineKey>{}(base->GetGraphicsKey()), why);
        }
    };
    if (base->native_colour_static == 0) {
        const char* why = [&]() -> const char* {
            if (regs.stage_enable.raw != AmdGpu::ShaderStageEnable::VgtStages::Vs) {
                return "stages";
            }
            if (!vs || !fs) {
                return "no vs / ps";
            }
            // SHADPS4_PATCH_SHADERS: a stage with a patch installed keeps its plain (patched)
            // module; the native colour variant is compiled from the recompiler and would
            // bypass the patch (2026-10-14, the W4 water patch).
            if ((EmulatorSettings.IsPatchShaders() || Shader::PatchShadersEnv()) &&
                (HasAnyShaderPatch(fs->pgm_hash, Shader::Stage::Fragment) ||
                 HasAnyShaderPatch(vs->pgm_hash, Shader::Stage::Vertex))) {
                return "shader patch";
            }
            // The SSR dither swaps fs_204f45fe's plain module by its exact SPIR-V; a native
            // colour variant is emitted differently and would keep the stripes (one fullscreen
            // draw per frame on the plain path).
            if (fs->pgm_hash == 0x204f45fe && SsrDitherEnabled()) {
                return "ssr dither";
            }
            for (u32 i = 0; i < stages.size(); ++i) {
                if (stages[i] && i != u32(LogicalStage::Vertex) &&
                    i != u32(LogicalStage::Fragment)) {
                    return "extra stage";
                }
            }
            if (fs->images.empty()) {
                // SHADPS4_NATIVE_COLOUR_UNTEXTURED (l4b): an fs with buffers / samplers / user
                // data / dma but no image is what NativePass refuses ("fs resources"): without
                // this it has no variant at all. Plain untextured stays the shadow kind.
                static const bool untextured = [] {
                    const char* e = std::getenv("SHADPS4_NATIVE_COLOUR_UNTEXTURED");
                    return e && e[0] == '1';
                }();
                const bool fs_resources = !fs->buffers.empty() || !fs->samplers.empty() ||
                                          fs->ud_mask.mask != 0 || fs->uses_dma;
                if (!untextured || !fs_resources) {
                    base->native_colour_untextured = 1;
                    return "untextured"; // the shadow kind (SHADPS4_NATIVE_PASS) or nothing
                }
            }
            // Diag 32 / 64: narrow loads ineligible in the VS / PS only.
            if (const char* w = NativeBufferWhyNot(*vs, 32)) {
                return w;
            }
            if (const char* w = NativeBufferWhyNot(*fs, 64)) {
                return w;
            }
            // + the records and vertex SSBOs in buffer_infos (static_vector of NUM_BUFFERS).
            const u32 extra = ((VideoCore::g_native_colour_diag & 4096) ? 3u : 2u) +
                              (Shader::g_native_ring ? 1u : 0u) + // review #5; + the table
                              (Shader::g_native_ring ? u32(Shader::g_native_pt) : 0u) + // + PT
                              (Shader::g_native_ring && Shader::g_vbuf2 != 0 ? 1u : 0u); // q53: + vbuf2
            if (vs->buffers.size() + fs->buffers.size() + extra > Shader::NUM_BUFFERS) {
                return "buffer count";
            }
            return nullptr;
        }();
        // SHADPS4_NATIVE_COLOUR_BISECT=<mask>:<value> (diagnostic): pipelines whose VS has 8 /
        // 16-bit loads are eligible only if (key hash & mask) == value; every other pipeline
        // stays ineligible. Logs the eligible ones.
        static const auto bisect = []() -> std::pair<u64, u64> {
            const char* e = std::getenv("SHADPS4_NATIVE_COLOUR_BISECT");
            if (!e) {
                return {0, ~0ull};
            }
            char* end = nullptr;
            const u64 mask = std::strtoull(e, &end, 0);
            const u64 value = end && *end == ':' ? std::strtoull(end + 1, nullptr, 0) : 0;
            return {mask, value};
        }();
        if (!why && bisect.second != ~0ull) {
            bool vs_narrow = false;
            for (const auto& desc : vs->buffers) {
                vs_narrow |= True(desc.used_types & (Shader::IR::Type::U8 | Shader::IR::Type::U16));
            }
            const u64 h = std::hash<GraphicsPipelineKey>{}(base->GetGraphicsKey());
            if (!vs_narrow || (h & bisect.first) != bisect.second) {
                why = "bisect";
            } else {
                LOG_INFO(Render_Vulkan, "NativeColour: bisect keeps pipeline {:#x} (vs {:#x} ps {:#x})",
                         h, vs->pgm_hash, fs->pgm_hash);
            }
        }
        base->native_colour_static = why ? 2 : 1;
        if (why) {
            base->native_colour_why.store(why, std::memory_order_relaxed);
        }
        if (why) {
            if (std::strcmp(why, "bisect") != 0) {
                log_not(why);
            }
            return;
        }
    }
    const u64 key = VideoCore::NativeVertexKey(vi.attribs.data(), vi.header.count);
    u32 free_slot = GraphicsPipeline::NativeColourSlots;
    for (u32 i = 0; i < GraphicsPipeline::NativeColourSlots; ++i) {
        const u64 k = base->native_colour[i].key.load(std::memory_order_relaxed);
        if (k == key) {
            return; // claimed: built, building or failed
        }
        if (k == 0 && free_slot == GraphicsPipeline::NativeColourSlots) {
            free_slot = i;
        }
    }
    if (free_slot == GraphicsPipeline::NativeColourSlots) {
        return;
    }
    // Translating two stages costs the front ~1 ms (guess); at most one per 2 ms.
    const auto now = std::chrono::steady_clock::now();
    if (now - native_colour_last < std::chrono::milliseconds(2)) {
        return;
    }
    native_colour_last = now;
    // From here on the slot is claimed for this key whatever happens (a failure is final).
    base->native_colour[free_slot].key.store(key, std::memory_order_release);
    claimed = &base->native_colour[free_slot];
    g_nc_claims[0].fetch_add(1, std::memory_order_relaxed);
    // Per-description eligibility: per-vertex attributes of pullable formats, locations < 16,
    // number class matching the shader's input type.
    Shader::NativeEmit emit{};
    for (u32 k = 0; k < vi.header.count; ++k) {
        const auto& d = vi.attribs[k].desc;
        if (d.instanced || d.location >= Shader::NativeMaxPulledLocations) {
            log_not("instanced / location");
            return;
        }
        auto attr = NativeVtxFromFormat(static_cast<vk::Format>(d.vk_format), d.stride);
        if (attr.bits == 0 && (VideoCore::g_native_colour_diag & 2)) {
            attr.valid = 0;
        }
        if (!attr.valid) {
            if (native_colour_logged++ < 96) {
                LOG_INFO(Render_Vulkan, "NativeColour: pipeline {:#x} vertex format {} not pulled",
                         std::hash<GraphicsPipelineKey>{}(base->GetGraphicsKey()),
                         vk::to_string(static_cast<vk::Format>(d.vk_format)));
            }
            g_nc_claims[2].fetch_add(1, std::memory_order_relaxed);
            return;
        }
        emit.attrs[d.location] = attr;
        const u32 dg = VideoCore::g_native_colour_diag;
        if ((dg & 4096) || (attr.bits == 0 && (dg & 128)) || (dg & 256) ||
            (attr.bits == 8 && (dg & 512)) ||
            (attr.bits == 16 && (dg & 1024)) || (attr.bits == 32 && (dg & 2048))) {
            emit.keep_inputs |= 1u << d.location;
        }
    }
    if (const auto* fetch = base->GetFetchShader()) {
        for (const auto& attrib : fetch->attributes) {
            if (attrib.semantic >= emit.attrs.size() || !emit.attrs[attrib.semantic].valid ||
                AmdGpu::GetNumberClass(attrib.GetSharp(*vs).GetNumberFmt()) !=
                    NativeVtxClass(emit.attrs[attrib.semantic].kind)) {
                log_not("fetch attribute vs description");
                return;
            }
        }
    }
    if (!RefreshGraphicsKey(regs) || !(graphics_key == base->GetGraphicsKey())) {
        log_not("key did not rebuild");
        return;
    }
    // s324 validation: a tessellated pipeline slipped through (its PS followed a TES); recheck
    // the stage selection the refresh just made.
    if (regs.stage_enable.raw != AmdGpu::ShaderStageEnable::VgtStages::Vs) {
        log_not("stages at refresh");
        return;
    }
    for (u32 i = 0; i < infos.size(); ++i) {
        if (infos[i] && i != u32(LogicalStage::Vertex) && i != u32(LogicalStage::Fragment)) {
            log_not("extra stage at refresh");
            return;
        }
    }
    if (runtime_infos[u32(LogicalStage::Vertex)].vs_info.tess_emulated_primitive) {
        // Quad / rect lists: a generated TCS / TES sits between VS and PS and would drop the
        // draw-index varying (s325 validation).
        log_not("emulated primitive (generated tessellation)");
        return;
    }
    {
        // The draw-index varying goes just above every VS output and PS input (+1 for the
        // clip-distance emulation shift).
        s32 hi = -1;
        for (u32 p = 0; p < Shader::IR::NumParams; ++p) {
            if (vs->stores.GetAny(Shader::IR::Attribute::Param0 + p)) {
                hi = std::max<s32>(hi, s32(p));
            }
        }
        const auto& fsi = runtime_infos[u32(LogicalStage::Fragment)].fs_info;
        for (u32 i = 0; i < fsi.num_inputs; ++i) {
            hi = std::max<s32>(hi, s32(fsi.inputs[i].param_index));
        }
        emit.draw_location = static_cast<u32>(hi + 2);
        if (emit.draw_location > Shader::NativeDrawIndexMaxLocation) {
            log_not("no free varying location");
            return;
        }
    }
    Shader::Backend::Bindings total{};
    std::array<Shader::Backend::Bindings, MaxShaderStages> starts{};
    const std::array<LogicalStage, 2> order{LogicalStage::Fragment, LogicalStage::Vertex};
    for (const auto l : order) {
        starts[u32(l)] = total;
        if (infos[u32(l)]) {
            infos[u32(l)]->AddBindings(total);
        }
    }
    emit.active = true;
    emit.records_binding = total.unified;
    emit.colour = true;
    emit.pull = true;
    emit.vertex_binding = total.unified + 1;
    emit.compare = (VideoCore::g_native_colour_diag & 4096) != 0;
    emit.debug_binding = total.unified + 2;
    emit.vs_hash = vs ? static_cast<u32>(vs->pgm_hash) : 0u;
    emit.ring = Shader::g_native_ring;
    emit.table_binding = total.unified + (emit.compare ? 3u : 2u);
    // SHADPS4_NATIVE_PT (sec. 18): the page table is read by pointer (no binding); =2 the
    // compare SSBO after the table.
    emit.pt = Shader::g_native_pt != 0 && emit.ring;
    emit.pt_compare = Shader::g_native_pt == 2 && emit.ring;
    emit.pt_debug_binding = emit.table_binding + 1;
    // SHADPS4_NATIVE_SEG (sec. 20 brick A): records by pointer through the batch's list.
    emit.seg = Shader::g_native_seg != 0 && emit.pt;
    emit.seg_compare = Shader::g_native_seg_compare && emit.pt;
    // q53 (SHADPS4_A2_VBUF2): the second vertex SSBO after every other set-0 binding (ring mode only: the A2
    // translator's layout check keeps records / vertex / table at unified + 0 / 1 / 2).
    emit.vbuf2 = Shader::g_vbuf2 != 0 && emit.ring;
    emit.vertex2_binding = (emit.pt_compare ? emit.pt_debug_binding : emit.table_binding) + 1;
    // SHADPS4_NATIVE_BINDLESS (sec. 12): the PS samples the global arrays; the VS has no image
    // or sampler, the PS at most 16 of each, all sampled (no storage, no mip fallback).
    const char* bindless_why = nullptr; // q1b census
    emit.bindless = [&] {
        if (!instance.IsBindlessSupported() || !Bindless::Init(instance.GetDevice()) || !vs ||
            !fs) {
            bindless_why = "support / stages";
            return false;
        }
        if ((!vs->images.empty() || !vs->samplers.empty()) && !Shader::g_native_bindless_vs) {
            bindless_why = "vs images / samplers";
            return false;
        }
        // q3: the VS's after the FS's, <= 16 of each in all.
        if (fs->images.size() + vs->images.size() > Shader::g_bl_img_max ||
            fs->samplers.size() + vs->samplers.size() > Shader::g_bl_smp_max) {
            bindless_why = "fs > image / sampler slots"; // q54: 16 / 16, or 22 / 10 with SHADPS4_BINDLESS_SPLIT=1
            return false;
        }
        for (const auto& desc : vs->images) {
            if (desc.is_written || desc.is_atomic ||
                desc.mip_fallback_mode != Shader::MipStorageFallbackMode::None) {
                bindless_why = "vs image written / atomic / mip fallback";
                return false;
            }
        }
        for (const auto& desc : fs->images) {
            if (desc.is_written || desc.is_atomic) {
                bindless_why = "fs image written / atomic";
                return false;
            }
            if (desc.mip_fallback_mode != Shader::MipStorageFallbackMode::None) {
                bindless_why = "fs mip fallback";
                return false;
            }
        }
        return true;
    }();
    if (emit.bindless && vs && (!vs->images.empty() || !vs->samplers.empty())) {
        emit.bindless_vs = true; // q3
        emit.vs_img_base = static_cast<u32>(fs->images.size());
        emit.vs_smp_base = static_cast<u32>(fs->samplers.size());
    }
    auto native_modules = modules;
    struct NativeEmitScope {
        NativeEmitScope(const Shader::NativeEmit& e) {
            Shader::t_native_emit = e;
        }
        ~NativeEmitScope() {
            Shader::t_native_emit = {};
        }
    };
    for (const auto l : order) {
        const u32 idx = u32(l);
        const auto hw = l == LogicalStage::Vertex ? Shader::Stage::Vertex : Shader::Stage::Fragment;
        const auto* pgm = regs.ProgramForStage(u32(hw));
        if (!pgm || !pgm->Address<u32*>() || !infos[idx]) {
            log_not("program");
            return;
        }
        const auto params = AmdGpu::GetParams(*pgm);
        auto new_info = Shader::Info(hw, l, params);
        new_info.pinned_uses_dma = infos[idx]->uses_dma;
        auto runtime_info = runtime_infos[idx];
        auto binding = starts[idx];
        std::vector<u32> spv;
        {
            NativeEmitScope scope{emit};
            const auto ir_program =
                Shader::TranslateProgram(params.code, pools, new_info, runtime_info, profile);
            spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, binding);
        }
        if (!SameBindingLayout(new_info, *infos[idx])) {
            log_not("native layout differs");
            return;
        }
        // SHADPS4_NATIVE_COLOUR_DUMP=<dir> (diagnostic): the native and the plain SPIR-V of the
        // first 12 stages whose shader has 8 / 16-bit buffer loads, side by side.
        static const char* dump_dir = std::getenv("SHADPS4_NATIVE_COLOUR_DUMP");
        static u32 dumped = 0;
        bool narrow = false;
        for (const auto& desc : new_info.buffers) {
            narrow |= True(desc.used_types & (Shader::IR::Type::U8 | Shader::IR::Type::U16));
        }
        const bool pulls = l == LogicalStage::Vertex && base->GetFetchShader() &&
                           !base->GetFetchShader()->attributes.empty();
        if (dump_dir && narrow && pulls && dumped < 12) {
            ++dumped;
            auto plain_info = Shader::Info(hw, l, params);
            plain_info.pinned_uses_dma = infos[idx]->uses_dma;
            auto plain_rt = runtime_infos[idx];
            auto plain_binding = starts[idx];
            const auto plain_ir =
                Shader::TranslateProgram(params.code, pools, plain_info, plain_rt, profile);
            const auto plain =
                Shader::Backend::SPIRV::EmitSPIRV(profile, plain_rt, plain_ir, plain_binding);
            const auto write = [&](const std::vector<u32>& code, const char* kind) {
                const auto path = fmt::format("{}/ncol_{:016x}_{}_{}.spv", dump_dir, params.hash,
                                              l == LogicalStage::Vertex ? "vs" : "ps", kind);
                Common::FS::IOFile file{path, Common::FS::FileAccessMode::Create};
                file.WriteSpan(std::span<const u32>{code});
            };
            write(spv, "native");
            write(plain, "plain");
        }
        DumpShader(spv, params.hash, hw, 0x4e43, "native.spv");
        native_modules[idx] = CompileSPV(spv, instance.GetDevice());
        Vulkan::SetObjectName(instance.GetDevice(), native_modules[idx], "{}_{:#x}_native_colour",
                              l == LogicalStage::Vertex ? "vs" : "ps", params.hash);
    }
    auto job = std::make_unique<NativeColourJob>();
    job->base = base;
    job->slot = free_slot;
    job->key = graphics_key;
    job->infos = infos;
    job->runtime_infos = runtime_infos;
    job->modules = native_modules;
    job->fetch_shader = fetch_shader;
    job->bindings = GraphicsPipeline::DeriveStageInputs(instance, infos, runtime_infos,
                                                        fetch_shader.get(), job->sdata);
    const u32 next = job->bindings.empty()
                         ? 0u
                         : job->bindings.back().binding + job->bindings.back().descriptorCount;
    if (next != emit.records_binding) {
        if (native_colour_logged++ < 96) {
            LOG_WARNING(Render_Vulkan, "NativeColour: records binding {} vs {}", next,
                        emit.records_binding);
        }
        g_nc_claims[3].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    job->bindings.push_back({
        .binding = emit.records_binding,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
    });
    job->bindings.push_back({
        .binding = emit.vertex_binding,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eVertex,
    });
    if (emit.compare) {
        job->bindings.push_back({
            .binding = emit.debug_binding,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex,
        });
    }
    if (emit.ring) {
        job->bindings.push_back({
            .binding = emit.table_binding,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        });
        job->table_binding = emit.table_binding;
        if (emit.pt_compare) {
            job->bindings.push_back({
                .binding = emit.pt_debug_binding,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            });
            job->pt_debug_binding = emit.pt_debug_binding;
        }
        if (emit.vbuf2) {
            job->bindings.push_back({
                .binding = emit.vertex2_binding,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eVertex,
            });
            job->vertex2_binding = emit.vertex2_binding;
        }
    }
    // SHADPS4_NATIVE_SPARSE_LAYOUT (l4c): the bindless PS samples set 1; set 0's image / sampler
    // bindings are dead (numbers kept: the bindings after them keep their indices).
    static const bool sparse = [] {
        const char* e = std::getenv("SHADPS4_NATIVE_SPARSE_LAYOUT");
        return e && e[0] == '1';
    }();
    if (sparse && emit.bindless) {
        auto& jb = job->bindings; // a static_vector: no std::erase_if
        const auto dead = [](const vk::DescriptorSetLayoutBinding& b) {
            return b.descriptorType == vk::DescriptorType::eSampledImage ||
                   b.descriptorType == vk::DescriptorType::eSampler;
        };
        // q3: the dropped bindings' mask (BindResources filters writes to them); none above 63.
        u64 mask = 0;
        bool fits = true;
        for (const auto& b : jb) {
            if (dead(b)) {
                for (u32 k = 0; k < b.descriptorCount; ++k) {
                    fits = fits && b.binding + k < 64;
                    if (b.binding + k < 64) {
                        mask |= u64{1} << (b.binding + k);
                    }
                }
            }
        }
        if (fits) {
            jb.erase(std::remove_if(jb.begin(), jb.end(), dead), jb.end());
            job->dead_mask = mask;
        }
    }
    job->bindless_vs = emit.bindless_vs;
    job->records_binding = emit.records_binding;
    job->vertex_binding = emit.vertex_binding;
    job->bindless = emit.bindless;
    job->bindless_why = bindless_why;
    job->counts = fmt::format("vs img {} smp {} buf {} | fs img {} smp {} buf {}",
                              vs ? vs->images.size() : 0, vs ? vs->samplers.size() : 0,
                              vs ? vs->buffers.size() : 0, fs ? fs->images.size() : 0,
                              fs ? fs->samplers.size() : 0, fs ? fs->buffers.size() : 0);
    if (!native_colour_compiler) {
        native_colour_compiler = std::make_unique<NativeColourCompiler>(
            instance, scheduler, desc_heap, profile, *pipeline_cache);
    }
    g_nc_claims[4].fetch_add(1, std::memory_order_relaxed);
    native_colour_compiler->Push(std::move(job));
}

void PipelineCache::QueueAsyncPipeline(u64 pipeline_hash) {
    if (!async_compiler) {
        async_compiler = std::make_unique<AsyncCompiler>(instance, scheduler, desc_heap, profile,
                                                         *pipeline_cache);
        LOG_INFO(Render_Vulkan, "AsyncPipelines: compiler thread started; draws of a graphics "
                                "pipeline are skipped while it compiles");
    }
    auto job = std::make_unique<AsyncJob>();
    job->key = graphics_key;
    job->hash = pipeline_hash;
    job->infos = infos;
    job->runtime_infos = runtime_infos;
    job->modules = modules;
    job->fetch_shader = fetch_shader;
    job->translate_ms = lookup_translate_ms;
    // The only reads of the Infos' per-draw state, here on the lookup thread.
    job->desc_bindings = GraphicsPipeline::DeriveStageInputs(instance, infos, runtime_infos,
                                                             fetch_shader.get(), job->sdata);
    async_compiler->Push(std::move(job));
    ++async_pending;
}

void PipelineCache::PublishAsyncPipelines() {
    for (auto& job : async_compiler->TakeFinished()) {
        // Inserted null by the lookup that queued the job; nothing erases entries while async
        // is on (hot reload needs shader collection).
        const auto it = graphics_pipelines.find(job->key);
        ASSERT_MSG(it != graphics_pipelines.end() && !it->second,
                   "Async pipeline {:#x} has no pending entry", job->hash);
        it.value() = std::move(job->pipeline);
        RegisterPipelineData(job->key, job->hash, job->sdata);
        ++num_new_pipelines;
        --async_pending;
        NoteCompiledPipeline(job->hash, job->translate_ms, job->create_ms, job->queued_ahead,
                             true);
    }
}

void PipelineCache::NoteCompiledPipeline(u64 hash, double translate_ms, double create_ms,
                                         u32 queued_ahead, bool async) {
    auto& stats = compile_stats;
    ++stats.n;
    stats.translate_ms_total += translate_ms;
    stats.create_ms_total += create_ms;
    stats.max_create_ms = std::max(stats.max_create_ms, create_ms);
    // translate: new programs/modules this lookup created (0 when all were cached); create: the
    // GraphicsPipeline constructor, which holds the driver compile; queued: jobs ahead of this
    // one in the compiler when it was queued.
    LOG_INFO(Render_Vulkan,
             "PipelineCompile: hash={:#x} translate_ms={:.2f} create_ms={:.2f} queued={} async={}",
             hash, translate_ms, create_ms, queued_ahead, async);
    if (stats.n % 64 == 0) {
        LOG_INFO(Render_Vulkan,
                 "PipelineCompileSummary: n={} translate_ms_total={:.1f} create_ms_total={:.1f} "
                 "max_create_ms={:.2f} skipped_draws={}",
                 stats.n, stats.translate_ms_total, stats.create_ms_total, stats.max_create_ms,
                 stats.skipped_draws);
    }
}

const ComputePipeline* PipelineCache::GetComputePipeline() {
    // Exclusive: GetProgram may insert into program_cache (a rehash under a concurrent
    // LookupGraphics would be a race) and compute_pipelines is mutated below.
    auto lk = LockExclusiveServicing();
    if (!RefreshComputeKey()) {
        return nullptr;
    }
    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    if (is_new) {
        const auto pipeline_hash = std::hash<ComputePipelineKey>{}(compute_key);
        LOG_INFO(Render_Vulkan, "Compiling compute pipeline {:#x}", pipeline_hash);

        ComputePipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile,
                                                       *pipeline_cache, compute_key, *infos[0],
                                                       modules[0], sdata, false);
        RegisterPipelineData(compute_key, sdata);
        ++num_new_pipelines;

        if (EmulatorSettings.IsShaderCollect()) {
            auto& m = modules[0];
            module_related_pipelines[m].emplace_back(compute_key);
        }
    } else {
        // SHADPS4_ASYNC_COMPUTE_PIPELINES: a preloaded pipeline still waiting for its compile.
        it->second->PromoteDeferred();
    }
    return it->second.get();
}

void PipelineCache::GraphicsKeyFirstPass(const AmdGpu::Regs& regs, GraphicsPipelineKey& key,
                                         bool& db_enabled, bool& skip_cb_binding) {
    std::memset(&key, 0, sizeof(GraphicsPipelineKey));

    db_enabled = regs.depth_buffer.DepthValid() || regs.depth_buffer.StencilValid();

    key.z_format = regs.depth_buffer.DepthValid() ? regs.depth_buffer.z_info.format
                                                  : AmdGpu::DepthBuffer::ZFormat::Invalid;
    key.stencil_format = regs.depth_buffer.StencilValid()
                             ? regs.depth_buffer.stencil_info.format
                             : AmdGpu::DepthBuffer::StencilFormat::Invalid;
    if (LiverpoolToVK::PromoteDepthToStencil(key.z_format, key.stencil_format)) {
        // Matches the promoted D32S8 attachment; stencil testing stays off (not StencilValid).
        key.stencil_format = AmdGpu::DepthBuffer::StencilFormat::Stencil8;
    }
    key.depth_clamp_enable = !regs.depth_render_override.disable_viewport_clamp;
    key.depth_clip_enable = regs.clipper_control.ZclipEnable();
    key.clip_space = regs.clipper_control.clip_space;
    key.provoking_vtx_last = regs.polygon_control.provoking_vtx_last;
    key.prim_type = regs.primitive_type;
    key.polygon_mode = regs.polygon_control.PolyMode();
    key.patch_control_points =
        regs.stage_enable.hs_en ? regs.ls_hs_config.hs_input_control_points : 0;
    key.logic_op = regs.color_control.rop3;
    key.depth_samples = db_enabled ? regs.depth_buffer.NumSamples() : 1;
    key.num_samples = key.depth_samples;
    key.cb_shader_mask = regs.color_shader_mask;

    skip_cb_binding = regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;

    // First pass to fill render target information needed by shader recompiler
    for (s32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        if (!col_buf || !regs.color_target_mask.GetMask(cb)) {
            // No attachment bound or writing to it is disabled.
            continue;
        }

        // Fill color target information
        auto& color_buffer = key.color_buffers[cb];
        color_buffer.data_format = col_buf.GetDataFmt();
        color_buffer.num_format = col_buf.GetNumberFmt();
        color_buffer.num_conversion = col_buf.GetNumberConversion();
        color_buffer.export_format = regs.color_export_format.GetFormat(cb);
        color_buffer.swizzle = col_buf.Swizzle();
    }
}

bool PipelineCache::RefreshGraphicsKey(const AmdGpu::Regs& regs) {
    // MEASUREMENT (perf-forest): split the 0.955 us into the SKIPPABLE key assembly (pre+post:
    // memset
    // + register reads + color-buffer loops) vs the MANDATORY stage resolution
    // (RefreshGraphicsStages
    // -> GetProgram, which refreshes per-draw user_data that BindResources consumes and cannot be
    // skipped). Only pre+post is reclaimable by a memo; this sizes it before any is written.
    const auto rgk_t0 = Common::FaultMonitor::Now();
    bool db_enabled = false;
    bool skip_cb_binding = false;
    GraphicsKeyFirstPass(regs, graphics_key, db_enabled, skip_cb_binding);

    // Compile and bind shader stages
    const auto rgk_t1 = Common::FaultMonitor::Now();
    if (!RefreshGraphicsStages(regs)) {
        return false;
    }
    const auto rgk_t2 = Common::FaultMonitor::Now();

    GraphicsKeySecondPass(regs, graphics_key, db_enabled, skip_cb_binding, true);

    if constexpr (Common::FaultMonitor::DetailCountersEnabled || SHADPS4_FAULT_TIMERS) {
        static u64 rgk_calls = 0, rgk_pre = 0, rgk_stages = 0, rgk_post = 0;
        const auto rgk_t3 = Common::FaultMonitor::Now();
        rgk_pre += Common::FaultMonitor::DiffNs(rgk_t0, rgk_t1);
        rgk_stages += Common::FaultMonitor::DiffNs(rgk_t1, rgk_t2);
        rgk_post += Common::FaultMonitor::DiffNs(rgk_t2, rgk_t3);
        if (++rgk_calls >= 200000) {
            LOG_INFO(
                Render_Vulkan,
                "RgkSplit: pre={:.3f}us stages={:.3f}us post={:.3f}us skippable(pre+post)={:.3f}us",
                static_cast<double>(rgk_pre) / 1e3 / rgk_calls,
                static_cast<double>(rgk_stages) / 1e3 / rgk_calls,
                static_cast<double>(rgk_post) / 1e3 / rgk_calls,
                static_cast<double>(rgk_pre + rgk_post) / 1e3 / rgk_calls);
            rgk_calls = rgk_pre = rgk_stages = rgk_post = 0;
        }
    }
    return true;
}

void PipelineCache::GraphicsKeySecondPass(const AmdGpu::Regs& regs, GraphicsPipelineKey& key,
                                          bool db_enabled, bool skip_cb_binding,
                                          bool allow_log) const {
    // Second pass to mask out render targets not written by shader and fill remaining info
    u8 color_samples = 0;
    bool all_color_samples_same = true;
    for (s32 cb = 0; cb < key.num_color_attachments && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (!col_buf || !target_mask) {
            continue;
        }
        if ((key.mrt_mask & (1u << cb)) == 0) {
            std::memset(&key.color_buffers[cb], 0, sizeof(Shader::PsColorBuffer));
            continue;
        }

        // Fill color blending information
        if (regs.blend_control[cb].enable && !col_buf.info.blend_bypass) {
            key.blend_controls[cb] = regs.blend_control[cb];
        }

        // Apply swizzle to target mask
        key.write_masks[cb] =
            vk::ColorComponentFlags{key.color_buffers[cb].swizzle.ApplyMask(target_mask)};

        // SHADPS4_DLSS_UPSCALE (2026-10-18, user: the dark panel behind the karma-choice text
        // vanished with DLSS): HUD draws into the HUD layer also write alpha (ApplyHudLayerMask),
        // but their alpha blend was whatever the game set for a channel it never wrote. The layer's
        // alpha must be its coverage, consistent with the colour equation: the scene below keeps
        // the colour dst factor's share. ONE_MINUS_SRC_ALPHA -> A = As + Ad (1 - As); ONE (additive)
        // -> A unchanged; ZERO -> A = As. Draws that read the destination colour (multiply) stay as
        // they were. Static per mode (UpscaleWanted), so lookup memos see one key; without the HUD
        // layer the alpha channel is masked off and the change has no effect.
        if (cb == 0 && key.blend_controls[0].enable && Dlss::UpscaleWanted() &&
            col_buf.Address() == Dlss::HudLayerAddr() &&
            !(key.write_masks[0] & vk::ColorComponentFlagBits::eA)) {
            using Factor = AmdGpu::BlendControl::BlendFactor;
            auto& bc = key.blend_controls[0];
            const Factor src = bc.color_src_factor;
            const Factor dst = bc.color_dst_factor;
            // The karma-choice panel (RenderDoc 10-18, karmachoiceLaggy): DST_COLOR / ONE_MINUS_
            // SRC_ALPHA, i.e. C = Cd (Cs + 1 - As), a dark brush darkening what is below. The layer
            // colour gets the same factor from the game's own blend; for the scene below it is "over"
            // with coverage As (exact for a black brush, slightly darker for a lighter one).
            const bool darken_over = src == Factor::DstColor && dst == Factor::OneMinusSrcAlpha;
            const bool reads_dst = !darken_over && (src == Factor::DstColor ||
                                                    src == Factor::OneMinusDstColor ||
                                                    src == Factor::DstAlpha ||
                                                    src == Factor::OneMinusDstAlpha);
            if (!reads_dst && (dst == Factor::OneMinusSrcAlpha || dst == Factor::One ||
                               dst == Factor::Zero)) {
                bc.separate_alpha_blend = 1;
                bc.alpha_src_factor = dst == Factor::One ? Factor::Zero : Factor::One;
                bc.alpha_dst_factor = dst == Factor::Zero ? Factor::Zero : dst == Factor::One
                                                                            ? Factor::One
                                                                            : Factor::OneMinusSrcAlpha;
                bc.alpha_func = AmdGpu::BlendControl::BlendFunc::Add;
            }
        }

        // Fill color samples
        const u8 prev_color_samples = std::exchange(color_samples, col_buf.NumSamples());
        all_color_samples_same &= color_samples == prev_color_samples || prev_color_samples == 0;
        key.color_samples[cb] = color_samples;
        key.num_samples = std::max(key.num_samples, color_samples);
    }

    // Force all color samples to match depth samples to avoid unsupported MSAA configuration
    if (color_samples != 0) {
        const bool depth_mismatch = db_enabled && color_samples != key.depth_samples;
        // SHADPS4_MIXED_SAMPLES_LOG=1 (diagnostic): each distinct shader pair / sample-count
        // combination that draws with color samples != depth samples, logged once. These are
        // the pipelines VK_NV_framebuffer_mixed_samples keeps mixed (Pro skin edge highlights).
        // Front path only (`allow_log`): the `seen` set below is unsynchronized.
        const bool log_mixed = g_log_mixed_samples && allow_log;
        if (log_mixed && depth_mismatch && regs.ps_program.Address<u32*>() &&
            regs.vs_program.Address<u32*>()) {
            static std::unordered_set<u64> seen;
            const u64 fs_hash = AmdGpu::GetParams(regs.ps_program).hash;
            const u64 vs_hash = AmdGpu::GetParams(regs.vs_program).hash;
            const u64 id = fs_hash ^ (vs_hash * 31) ^ (u64(color_samples) << 56) ^
                           (u64(key.depth_samples) << 60) ^ (u64(std::bit_cast<u32>(regs.blend_control[0])) << 8);
            if (seen.insert(id).second) {
                LOG_INFO(Render_Vulkan,
                         "MIXED_SAMPLES: fs {:#018x} vs {:#018x} color_samples {} depth_samples {} "
                         "all_color_same {} blend0 {:#010x} mrt_mask {:#x} depth_test {} "
                         "depth_write {} extension {}",
                         fs_hash, vs_hash, color_samples, key.depth_samples,
                         all_color_samples_same, std::bit_cast<u32>(regs.blend_control[0]), key.mrt_mask,
                         bool(regs.depth_control.depth_enable),
                         bool(regs.depth_control.depth_write_enable),
                         instance.IsMixedDepthSamplesSupported());
            }
        }
        if (!all_color_samples_same && !instance.IsMixedAnySamplesSupported() ||
            all_color_samples_same && depth_mismatch && !instance.IsMixedDepthSamplesSupported()) {
            key.color_samples.fill(key.depth_samples);
            key.num_samples = key.depth_samples;
        }
    }
}

bool PipelineCache::RefreshGraphicsStages(const AmdGpu::Regs& regs) {
    auto& key = graphics_key;
    fetch_shader.reset();

    Shader::Backend::Bindings binding{};
    const auto bind_stage = [&](Shader::Stage stage_in, Shader::LogicalStage stage_out) -> bool {
        const auto stage_in_idx = static_cast<u32>(stage_in);
        const auto stage_out_idx = static_cast<u32>(stage_out);
        if (!regs.stage_enable.IsStageEnabled(stage_in_idx)) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto* pgm = regs.ProgramForStage(stage_in_idx);
        if (!pgm || !pgm->Address<u32*>()) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto params = AmdGpu::GetParams(*pgm);
        std::shared_ptr<const Shader::Gcn::FetchShaderData> fetch_shader_;
        std::tie(infos[stage_out_idx], modules[stage_out_idx], fetch_shader_,
                 key.stage_hashes[stage_out_idx]) =
            GetProgram(regs, stage_in, stage_out, params, binding);
        if (fetch_shader_) {
            fetch_shader = std::move(fetch_shader_);
        }
        return true;
    };

    infos.fill(nullptr);
    modules.fill(nullptr);

    bind_stage(Stage::Fragment, LogicalStage::Fragment);

    const auto* fs_info = infos[static_cast<u32>(LogicalStage::Fragment)];
    key.mrt_mask = fs_info ? fs_info->mrt_mask : 0u;
    key.num_color_attachments = std::bit_width(key.mrt_mask);

    switch (regs.stage_enable.raw) {
    case AmdGpu::ShaderStageEnable::VgtStages::EsGs:
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (!bind_stage(Stage::Export, LogicalStage::Vertex)) {
            return false;
        }
        if (!bind_stage(Stage::Geometry, LogicalStage::Geometry)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (!bind_stage(Stage::Hull, LogicalStage::TessellationControl)) {
            return false;
        }
        if (!bind_stage(Stage::Vertex, LogicalStage::TessellationEval)) {
            return false;
        }
        if (!bind_stage(Stage::Local, LogicalStage::Vertex)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHsEsGs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (!bind_stage(Stage::Hull, LogicalStage::TessellationControl)) {
            return false;
        }
        if (!bind_stage(Stage::Export, LogicalStage::TessellationEval)) {
            return false;
        }
        if (!bind_stage(Stage::Local, LogicalStage::Vertex)) {
            return false;
        }
        if (!bind_stage(Stage::Geometry, LogicalStage::Geometry)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::Vs:
        bind_stage(Stage::Vertex, LogicalStage::Vertex);
        break;
    default:
        UNREACHABLE_MSG("unhandled stage_en: {}", (u32)regs.stage_enable.raw);
    }

    const auto* vs_info = infos[static_cast<u32>(Shader::LogicalStage::Vertex)];
    if (vs_info && fetch_shader && !instance.IsVertexInputDynamicState()) {
        // Without vertex input dynamic state, the pipeline needs to specialize on format.
        // Stride will still be handled outside the pipeline using dynamic state.
        u32 vertex_binding = 0;
        for (const auto& attrib : fetch_shader->attributes) {
            const auto& buffer = attrib.GetSharp(*vs_info);
            ASSERT_MSG(vertex_binding < MaxVertexBufferCount,
                       "Vertex attribute binding count exceeded limit: {} >= {}", vertex_binding,
                       MaxVertexBufferCount);
            key.vertex_buffer_formats[vertex_binding++] =
                Vulkan::LiverpoolToVK::SurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
        }
    }

    return true;
}

bool PipelineCache::RefreshComputeKey() {
    Shader::Backend::Bindings binding{};
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto cs_params = AmdGpu::GetParams(cs_pgm);
    std::tie(infos[0], modules[0], fetch_shader, compute_key.value) =
        GetProgram(Shader::Stage::Compute, LogicalStage::Compute, cs_params, binding);
    return true;
}

vk::ShaderModule PipelineCache::CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                              const std::span<const u32>& code, size_t perm_idx,
                                              Shader::Backend::Bindings& binding) {
    LOG_INFO(Render_Vulkan, "Compiling {} shader {:#x} {}", info.stage, info.pgm_hash,
             perm_idx != 0 ? "(permutation)" : "");
    DumpShader(code, info.pgm_hash, info.stage, perm_idx, "bin");

    const auto ir_program = Shader::TranslateProgram(code, pools, info, runtime_info, profile);
    auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, binding);
    DumpShader(spv, info.pgm_hash, info.stage, perm_idx, "spv");
    DumpOneShader(spv, info.pgm_hash, info.stage, perm_idx, "compiled");

    vk::ShaderModule module;

    auto patch = GetShaderPatch(info.pgm_hash, info.stage, perm_idx, "spv");
    if (!patch && (EmulatorSettings.IsPatchShaders() || Shader::PatchShadersEnv())) {
        // 2026-10-14: a patch keyed on the SPIR-V it replaces, <name>_src<fnv1a64>.spv - the
        // permutation index depends on the order shaders are first used (W4 water patch).
        patch = GetShaderPatchBySource(info.pgm_hash, info.stage, spv);
    }
    const bool is_patched = patch && (EmulatorSettings.IsPatchShaders() || Shader::PatchShadersEnv());
    if (is_patched) {
        LOG_INFO(Loader, "Loaded patch for {} shader {:#x}", info.stage, info.pgm_hash);
        module = CompileSPV(*patch, instance.GetDevice());
    } else {
        module = CompileSPV(spv, instance.GetDevice());
    }

    RegisterShaderBinary(std::move(spv), info.pgm_hash, perm_idx);

    const auto name = GetShaderName(info.stage, info.pgm_hash, perm_idx);
    Vulkan::SetObjectName(instance.GetDevice(), module, name);
    if (EmulatorSettings.IsShaderCollect()) {
        DebugState.CollectShader(name, info.l_stage, module, spv, code,
                                 patch ? *patch : std::span<const u32>{}, is_patched);
    }
    return module;
}

static std::shared_ptr<const Shader::Gcn::FetchShaderData> SharedFetchMetadata(
    const Program::Module& module) {
    const bool verify = g_shared_fetch_verify;
    if (verify) {
        const auto& expected = module.spec.fetch_shader_data;
        const auto& actual = module.fetch_shader;
        ASSERT_MSG(bool(expected) == bool(actual), "SharedFetch presence mismatch");
        if (expected) {
            ASSERT_MSG(expected->size == actual->size &&
                           expected->vertex_offset_sgpr == actual->vertex_offset_sgpr &&
                           expected->instance_offset_sgpr == actual->instance_offset_sgpr &&
                           expected->attributes.size() == actual->attributes.size(),
                       "SharedFetch header mismatch");
            for (size_t i = 0; i < expected->attributes.size(); ++i) {
                const auto& a = expected->attributes[i];
                const auto& b = actual->attributes[i];
                ASSERT_MSG(a.semantic == b.semantic && a.dest_vgpr == b.dest_vgpr &&
                               a.num_elements == b.num_elements && a.sgpr_base == b.sgpr_base &&
                               a.dword_offset == b.dword_offset &&
                               a.instance_data == b.instance_data && a.inst_offset == b.inst_offset &&
                               a.data_format == b.data_format && a.num_format == b.num_format,
                           "SharedFetch attribute mismatch");
            }
        }
        static u64 calls = 0, nonempty = 0;
        nonempty += bool(actual);
        if ((++calls & 0x3ffff) == 0) {
            LOG_INFO(Render_Vulkan, "SharedFetch: verified={} nonempty={} empty={}", calls,
                     nonempty, calls - nonempty);
        }
    }
    return module.fetch_shader;
}

void PipelineCache::RetirePrecompiled(u64 hash) {
    const auto it = program_cache.find(hash);
    if (it == program_cache.end()) {
        return;
    }
    size_t pipelines = 0, cancelled = 0;
    const size_t perms = it.value()->modules.size();
    for (size_t perm = 0; perm < perms; ++perm) {
        // A compute pipeline's key is its permutation hash (RefreshComputeKey).
        const auto pit = compute_pipelines.find(ComputePipelineKey{HashCombine(hash, perm)});
        if (pit != compute_pipelines.end()) {
            // Fix B: never used again, so a still-queued pool compile of it is dropped.
            cancelled += pit.value()->CancelDeferred() ? 1 : 0;
            retired_pipelines.push_back(std::move(pit.value()));
            compute_pipelines.erase(pit);
            ++pipelines;
        }
    }
    retired_programs.push_back(std::move(it.value()));
    program_cache.erase(hash);
    // Fix A: later launches skip this program's guessed metas (the live rebuild overwrites only
    // the permutation indices it reaches).
    precompile_retired.insert(hash);
    AppendPrecompileHash(PrecompileRetiredFile, hash);
    LOG_INFO(Render_Vulkan,
             "PrecompiledUse: cs_{:#x} retired after a live miss ({} permutations, {} pipelines, "
             "{} queued compiles dropped; retired so far {})",
             hash, perms, pipelines, cancelled, retired_programs.size());
}

PipelineCache::Result PipelineCache::GetProgram(Stage stage, LogicalStage l_stage,
                                                const Shader::ShaderParams& params,
                                                Shader::Backend::Bindings& binding) {
    return GetProgram(liverpool->regs, stage, l_stage, params, binding);
}

PipelineCache::Result PipelineCache::GetProgram(const AmdGpu::Regs& regs, Stage stage,
                                                LogicalStage l_stage,
                                                const Shader::ShaderParams& params,
                                                Shader::Backend::Bindings& binding) {
    Dlss::NoteProgram(stage == Stage::Fragment, params.hash, params.Base());
    auto runtime_info = BuildRuntimeInfo(regs, stage, l_stage);
    auto [it_pgm, new_program] = program_cache.try_emplace(params.hash);
    if (new_program) {
        const auto t_translate = CompileClock::now();
        it_pgm.value() = std::make_unique<Program>(stage, l_stage, params);
        auto& program = it_pgm.value();
        auto start = binding;
        const auto module = CompileModule(program->info, runtime_info, params.code, 0, binding);
        auto spec = Shader::StageSpecialization(program->info, runtime_info, profile, start);
        const auto perm_hash = HashCombine(params.hash, 0);

        RegisterShaderMeta(program->info, spec.fetch_shader_data, spec, perm_hash, 0);
        program->AddPermut(module, std::move(spec));
        NoteStageSelection(l_stage, program.get(), 0, true);
        Dlss::NoteWeightsInfo(program->info, params.Base());
        Dlss::NoteShaderInfo(program->info, params.Base());
        lookup_translate_ms += MsSince(t_translate);
        return std::make_tuple(&program->info, module, SharedFetchMetadata(program->modules[0]),
                               perm_hash);
    }

    auto& program = it_pgm.value();
    auto& info = program->info;
    Dlss::NoteWeightsInfo(info, params.Base());
    Dlss::NoteShaderInfo(info, params.Base());
    info.pgm_base = params.Base(); // Needs to be actualized for inline cbuffer address fixup
    info.user_data = params.user_data;
    info.RefreshFlatBuf();
    const bool prepared_enabled = g_prepared_programs;
    const bool prepared_verify = g_prepared_programs_verify;
    const bool prepared_two_entry_enabled = g_prepared_programs_two_entry;
    const bool eligible =
        prepared_enabled && Shader::PreparedSpecialization::Eligible(info, stage, l_stage);
    const auto start = binding;
    if (eligible) {
        if (!program->prepared) {
            program->prepared = std::make_unique<Shader::PreparedSpecialization>();
        }
        bool hit = program->prepared->permutation < program->modules.size() &&
                   program->prepared->Matches(info, runtime_info, binding);
        static std::array<u64, 3> calls{}, hits{}, secondary_hits{}, verified{};
        const size_t counter = stage == Stage::Vertex ? 1 : stage == Stage::Compute ? 2 : 0;
        ++calls[counter];
        if (!hit && prepared_two_entry_enabled && program->prepared_secondary &&
            program->prepared_secondary->permutation < program->modules.size() &&
            program->prepared_secondary->Matches(info, runtime_info, binding)) {
            std::swap(program->prepared, program->prepared_secondary);
            hit = true;
            ++secondary_hits[counter];
        }
        if (hit) {
            ++hits[counter];
            const auto& prepared = *program->prepared;
            if (prepared_verify) {
                const auto fresh =
                    Shader::StageSpecialization(info, runtime_info, profile, binding);
                const auto found =
                    std::ranges::find(program->modules, fresh, &Program::Module::spec);
                ASSERT_MSG(found != program->modules.end() &&
                               size_t(std::distance(program->modules.begin(), found)) ==
                                   prepared.permutation,
                           "PreparedPrograms selected a different permutation: stage={} hash={:#x}",
                           stage, params.hash);
                ++verified[counter];
            }
        }
        if ((calls[counter] & 0x3ffff) == 0 && !::Common::Lean()) {
            LOG_INFO(Render_Vulkan,
                     "PreparedPrograms: stage={} calls={} hit={} secondary={} verified={}", stage,
                     calls[counter], hits[counter], secondary_hits[counter], verified[counter]);
        }
        if (hit) {
            info.AddBindings(binding);
            const auto& prepared = *program->prepared;
            const auto& selected = program->modules[prepared.permutation];
            NoteStageSelection(l_stage, program.get(), prepared.permutation, false);
            return std::make_tuple(&info, selected.module, SharedFetchMetadata(selected),
                                   HashCombine(params.hash, prepared.permutation));
        }
    }
    auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding);

    size_t perm_idx = program->modules.size();
    u64 perm_hash = HashCombine(params.hash, perm_idx);

    vk::ShaderModule module{};

    const auto it = std::ranges::find(program->modules, spec, &Program::Module::spec);
    const bool compiled = it == program->modules.end();
    if (program->precompiled && !program->precompiled_seen) [[unlikely]] {
        program->precompiled_seen = true;
        NotePrecompiledUse(*program, spec, !compiled);
        if (compiled && stage == Stage::Compute) {
            // Fable finding 1: a live MISS. The shared Info (ud_mask, binding layout) came from a
            // guessed context, and every permutation is pushed and bound through it, so a module
            // compiled now for the live context could be fed by the guess's layout. Retire the
            // precompiled program and its pipelines; this dispatch then builds the program from
            // scratch (the new-program path), exactly as for a program the cache never held.
            RetirePrecompiled(params.hash);
            binding = start;
            return GetProgram(regs, stage, l_stage, params, binding);
        }
    }
    if (compiled) {
        const auto t_translate = CompileClock::now();
        auto new_info = Shader::Info(stage, l_stage, params);
        // The pipeline layout and descriptor binding of every permutation come from the
        // program's shared Info (returned below), so the new module must have the same
        // bindings. The DMA buffers (bda_pagetable, fault_buffer) sit between the guest buffers
        // and the images: a program preloaded from an older cache generation without DMA plus
        // a fresh DMA permutation shifted every image binding by 2 (city lighting void,
        // 2026-10-04).
        new_info.pinned_uses_dma = info.uses_dma;
        const auto bind_start = binding;
        module = CompileModule(new_info, runtime_info, params.code, perm_idx, binding);
        if (!SameBindingLayout(new_info, info)) {
            LOG_ERROR(Render_Vulkan,
                      "Permutation {} of {}_{:#x} has a different binding layout than its "
                      "program (buffers {} vs {}, images {} vs {}, samplers {} vs {}, dma {} vs "
                      "{}); its bindings will not match the pipeline layout",
                      perm_idx, stage, params.hash, new_info.buffers.size(), info.buffers.size(),
                      new_info.images.size(), info.images.size(), new_info.samplers.size(),
                      info.samplers.size(), new_info.uses_dma, info.uses_dma);
        }
        // Advance the caller's binding counters by the shared Info, which is what the
        // pipeline layout and the descriptor writes use.
        binding = bind_start;
        info.AddBindings(binding);

        // SHADPS4_PRECOMPILE (Opus review 2026-10-07 finding 3): the meta stores the program's
        // shared Info; while that Info is a guess (a precompiled program that matched its first
        // dispatch), the new permutation is saved marked too, so a later launch neither creates
        // the program from it as a live one nor learns from the guessed Info.
        RegisterShaderMeta(info, spec.fetch_shader_data, spec, perm_hash, perm_idx,
                           program->precompiled);
        program->AddPermut(module, std::move(spec));
        lookup_translate_ms += MsSince(t_translate);
    } else {
        info.AddBindings(binding);
        module = it->module;
        perm_idx = std::distance(program->modules.begin(), it);
        perm_hash = HashCombine(params.hash, perm_idx);
        if (eligible) {
            // Seed only a successful ordinary first-match selection. New or
            // inserted permutations invalidate the guard before the next use.
            if (prepared_two_entry_enabled) {
                if (!program->prepared_secondary) {
                    program->prepared_secondary =
                        std::make_unique<Shader::PreparedSpecialization>();
                }
                std::swap(program->prepared, program->prepared_secondary);
            }
            program->prepared->Capture(info, runtime_info, start, spec, perm_idx);
        }
    }
    NoteStageSelection(l_stage, program.get(), perm_idx, compiled);
    return std::make_tuple(&program->info, module,
                           SharedFetchMetadata(program->modules[perm_idx]), perm_hash);
}

std::optional<vk::ShaderModule> PipelineCache::ReplaceShader(vk::ShaderModule module,
                                                             std::span<const u32> spv_code) {
    std::optional<vk::ShaderModule> new_module{};
    // Devtools calls this from the UI thread. Serialize replacement with guest
    // draws, then finish all uses of the old pipelines before destroying them.
    liverpool->SendCommand<true>([&] {
        scheduler.Finish();
        // SHADPS4_ASYNC_PIPELINES: a queued job copied module handles; let the compiler drain
        // before any handle below is destroyed. Its finished jobs are published by the next
        // lookup, not here (this may be the back thread while the front is mid-lookup).
        if (native_colour_compiler) {
            native_colour_compiler->WaitIdle();
        }
        if (async_compiler) {
            async_compiler->WaitIdle();
        }
        // SHADPS4_ASYNC_COMPUTE_PIPELINES: queued compute compiles hold module handles too.
        PipelineCompilePool::WaitIdle();
        // Architecture A: the mutation below (modules, prepared guards, pipeline erasure) is
        // exclusive against worker lookups. Taken after the waits above so a worker blocked on
        // the lock cannot be what those waits are waiting for.
        std::unique_lock lk{graphics_mutex};
        // Entries hold pipeline pointers and module handles; drop them all before either
        // can be destroyed below. Runs on the command-processor thread like the draws.
        ++prepared_pipeline_epoch;
        for (const auto& [_, program] : program_cache) {
            for (auto& m : program->modules) {
                if (m.module == module) {
                    program->InvalidatePrepared();
                    const auto& d = instance.GetDevice();
                    d.destroyShaderModule(m.module);
                    m.module = CompileSPV(spv_code, d);
                    new_module = m.module;
                }
            }
        }
        if (module_related_pipelines.contains(module)) {
            auto& pipeline_keys = module_related_pipelines[module];
            for (auto& key : pipeline_keys) {
                if (std::holds_alternative<GraphicsPipelineKey>(key)) {
                    auto& graphics_key = std::get<GraphicsPipelineKey>(key);
                    graphics_pipelines.erase(graphics_key);
                } else if (std::holds_alternative<ComputePipelineKey>(key)) {
                    auto& compute_key = std::get<ComputePipelineKey>(key);
                    compute_pipelines.erase(compute_key);
                }
            }
        }
    });
    return new_module;
}

std::string PipelineCache::GetShaderName(Shader::Stage stage, u64 hash,
                                         std::optional<size_t> perm) {
    if (perm) {
        return fmt::format("{}_{:#018x}_{}", stage, hash, *perm);
    }
    return fmt::format("{}_{:#018x}", stage, hash);
}

void PipelineCache::DumpShader(std::span<const u32> code, u64 hash, Shader::Stage stage,
                               size_t perm_idx, std::string_view ext) const {
    if (!EmulatorSettings.IsDumpShaders()) {
        return;
    }

    using namespace Common::FS;
    const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
    if (!std::filesystem::exists(dump_dir)) {
        std::filesystem::create_directories(dump_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto file = IOFile{dump_dir / filename, FileAccessMode::Create};
    file.WriteSpan(code);
}

void PipelineCache::DumpOneShader(std::span<const u32> code, u64 hash, Shader::Stage stage,
                                  size_t perm_idx, const char* origin) {
    static const u64 want = [] {
        const char* v = std::getenv("SHADPS4_DUMP_SHADER_HASH");
        return v ? std::strtoull(v, nullptr, 16) : 0ull;
    }();
    if (want == 0 || hash != want || code.empty()) {
        return;
    }
    static std::mutex m;
    static std::unordered_set<std::string> done;
    const auto name = fmt::format("one_{}_{}", GetShaderName(stage, hash, perm_idx), origin);
    std::scoped_lock lk{m};
    if (!done.insert(name).second) {
        return;
    }
    using namespace Common::FS;
    const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
    std::error_code ec;
    std::filesystem::create_directories(dump_dir, ec);
    const auto file = IOFile{dump_dir / (name + ".spv"), FileAccessMode::Create};
    file.WriteSpan(code);
    LOG_INFO(Render_Vulkan, "DumpOneShader: wrote {}.spv ({} words)", name, code.size());
}

std::optional<std::vector<u32>> PipelineCache::GetShaderPatchBySource(
    u64 hash, Shader::Stage stage, std::span<const u32> source) {
    u64 h = 0xcbf29ce484222325ull; // FNV-1a 64 over the source module's bytes
    for (const u32 w : source) {
        for (u32 b = 0; b < 4; ++b) {
            h ^= (w >> (8 * b)) & 0xffu;
            h *= 0x100000001b3ull;
        }
    }
    using namespace Common::FS;
    const auto patch_dir = GetUserPath(PathType::ShaderDir) / "patch";
    const auto filepath =
        patch_dir / fmt::format("{}_src{:016x}.spv", GetShaderName(stage, hash), h);
    if (!std::filesystem::exists(filepath)) {
        return {};
    }
    const auto file = IOFile{filepath, FileAccessMode::Read};
    std::vector<u32> code(file.GetSize() / sizeof(u32));
    file.Read(code);
    LOG_INFO(Loader, "Shader patch by source: {}", filepath.filename().string());
    return code;
}

bool PipelineCache::HasAnyShaderPatch(u64 hash, Shader::Stage stage) {
    using namespace Common::FS;
    const auto patch_dir = GetUserPath(PathType::ShaderDir) / "patch";
    const auto prefix = GetShaderName(stage, hash);
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(patch_dir, ec)) {
        if (e.path().filename().string().starts_with(prefix)) {
            return true;
        }
    }
    return false;
}

std::optional<std::vector<u32>> PipelineCache::GetShaderPatch(u64 hash, Shader::Stage stage,
                                                              size_t perm_idx,
                                                              std::string_view ext) {

    using namespace Common::FS;
    const auto patch_dir = GetUserPath(PathType::ShaderDir) / "patch";
    if (!std::filesystem::exists(patch_dir)) {
        std::filesystem::create_directories(patch_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto filepath = patch_dir / filename;
    if (!std::filesystem::exists(filepath)) {
        return {};
    }
    const auto file = IOFile{patch_dir / filename, FileAccessMode::Read};
    std::vector<u32> code(file.GetSize() / sizeof(u32));
    file.Read(code);
    return code;
}

// ---- Architecture A (notes/NATIVE-RENDERER.md sec. 9): the lookup from any thread -----------
//
// THREAD SAFETY. Every non-local object LookupGraphics (shared lock) reads, and why that is safe:
//  - `regs` (the caller's private register file) and guest memory reached through it (shader
//    code for AmdGpu::GetParams / SearchBinaryInfo, whose scan memo is thread_local and whose
//    prefetch hints are atomics; user-data tables through the SRT walker and the fetch shader
//    code behind user_data[fetch_shader_sgpr_base]): the caller owns `regs`; guest memory is the
//    same racy input the front reads today.
//  - program_cache (robin_map) and Program::modules (small_vector): read-only under the shared
//    lock; every mutation (GetProgram's try_emplace / AddPermut, RetirePrecompiled's erase,
//    ReplaceShader's module swap) runs under the exclusive lock. Programs are never destroyed
//    (unique_ptr in the map or parked in retired_programs), so Program / Info pointers stay valid
//    for the run.
//  - The IMMUTABLE half of a shared Shader::Info (buffers, images, samplers, fmasks, ud_mask,
//    srt_info, fetch_shader_sgpr_base, has_fetch_shader, stage, l_stage, pgm_hash, mrt_mask,
//    uses_dma, loads / stores, ...): written once by CompileModule / Info::Deserialize before the
//    Program is published into program_cache (later permutations translate into a fresh
//    `new_info`), never afterwards. The read-only path copies it into the scratch's InfoView and
//    never reads the Info's per-draw half (pgm_base, user_data, flattened_ud_buf), which the
//    front path rewrites under the exclusive lock.
//  - srt_info.walker_func: JIT-generated code called with the scratch's own buffers; assumed
//    reentrant (it takes its input and output pointers as arguments - inferred from its
//    signature, not verified against the emitter).
//  - graphics_pipelines (robin_map): read-only under the shared lock; inserted / erased only
//    under the exclusive lock (GraphicsLookupLocked, PublishAsyncPipelines, ReplaceShader;
//    LoadGraphicsPipeline / WarmUp run before the game, single-threaded). A GraphicsPipeline is
//    never destroyed while the game runs except by ReplaceShader (devtools, after WaitIdle).
//  - `instance` (const feature queries), `profile` (set in the constructor), the g_* launch
//    flags (const, read once), Shader::g_synthetic_flat (set only during WarmUp).
//  - Shader::StageSpecialization's constructor, Gcn::ParseFetchShader and the LiverpoolToVK
//    helpers: pure over their arguments (function-local `static const` flags only).
// NOT reproduced read-only (routed to LookupOrBuildGraphics' exclusive path, which runs the
// front's own code and is therefore bit-identical to it): any stage configuration other than
// stage_enable == Vs (tessellation / geometry read the hull Info's per-draw state and dump the
// copy shader); a program hash missing from program_cache; a precompiled program's first live use
// (writes precompiled_seen, may retire the program); a specialization with no module (compiles);
// a key with no pipeline (compiles); the SHADPS4_PREPARED_PROGRAMS / _PIPELINES memos (front-only
// guards whose selections the verify modes assert equal to the plain first match, which this
// path always makes); the MIXED_SAMPLES diagnostic log; SharedFetchMetadata's verify counters.

static_assert(std::is_copy_assignable_v<Shader::Info>,
              "GraphicsLookupScratch::InfoView copies the shared Shader::Info");

Shader::Info& GraphicsLookupScratch::View(const Shader::Info& shared) {
    InfoView* victim = &views[0];
    for (auto& view : views) {
        if (view.source == &shared) {
            view.last_use = ++view_clock;
            return view.info;
        }
        if (view.last_use < victim->last_use) {
            victim = &view;
        }
    }
    // The copy takes the immutable half (never rewritten after publication) and whatever the
    // per-draw half holds right now; the latter is overwritten before every use.
    victim->info = shared;
    victim->source = &shared;
    victim->last_use = ++view_clock;
    return victim->info;
}

void GraphicsLookupScratch::Reset() {
    pipeline = nullptr;
    fetch_shader = nullptr;
    miss = Miss::None;
    key = GraphicsPipelineKey{};
    for (auto& stage : stages) {
        stage.info = nullptr;
        stage.pgm_base = 0;
        stage.user_data.clear();
        stage.flattened_ud_buf.clear();
        stage.module = vk::ShaderModule{};
    }
}

std::unique_lock<std::shared_mutex> PipelineCache::AcquireExclusive() const {
    return std::unique_lock{graphics_mutex};
}

const Shader::Info* PipelineCache::ProgramInfoFor(u64 hash) const {
    std::shared_lock lk{graphics_mutex};
    const auto it = program_cache.find(hash);
    if (it == program_cache.end()) {
        return nullptr;
    }
    const Program& program = *it->second;
    if (program.precompiled && !program.precompiled_seen) {
        return nullptr;
    }
    return &program.info;
}

const GraphicsPipeline* PipelineCache::LookupGraphics(const AmdGpu::Regs& regs,
                                                      GraphicsLookupScratch& out) const {
    out.Reset();
    std::shared_lock lk{graphics_mutex};
    return LookupGraphicsShared(regs, out);
}

PipelineCache::Resolve PipelineCache::ResolveStageReadOnly(
    const AmdGpu::Regs& regs, Stage stage_in, LogicalStage stage_out,
    Shader::Backend::Bindings& binding, GraphicsLookupScratch& out,
    const Shader::Info*& view_out, const Program::Module*& module_out) const {
    // RefreshGraphicsStages' bind_stage + GetProgram's cached path, without the writes: the
    // same presence tests, the same per-draw refresh (into the scratch's copy), the same
    // RuntimeInfo, the same StageSpecialization and the same first-match permutation search.
    using Miss = GraphicsLookupScratch::Miss;
    const auto in_idx = static_cast<u32>(stage_in);
    const auto out_idx = static_cast<u32>(stage_out);
    auto& slot = out.stages[out_idx];
    slot.info = nullptr;
    out.key.stage_hashes[out_idx] = 0;
    view_out = nullptr;
    module_out = nullptr;
    if (!regs.stage_enable.IsStageEnabled(in_idx)) {
        return Resolve::Unbound;
    }
    const auto* pgm = regs.ProgramForStage(in_idx);
    if (!pgm || !pgm->Address<u32*>()) {
        return Resolve::Unbound;
    }
    const bool pre = out.prewalked && out.pre_params[out_idx].has_value();
    if (AmdGpu::t_lean_worker && !pre) [[unlikely]] {
        out.miss = Miss::NoProgram; // a lean worker never reads guest memory under the lock
        return Resolve::Miss;
    }
    const auto params = pre ? *out.pre_params[out_idx] : AmdGpu::GetParams(*pgm);
    Dlss::NoteProgram(stage_in == Stage::Fragment, params.hash, params.Base());
    const auto it = program_cache.find(params.hash);
    if (it == program_cache.end() || (pre && out.pre_missing[out_idx])) {
        out.miss = Miss::NoProgram; // the front would translate and compile it (or: published
                                    // after the worker's prewalk, which built no flat buffer)
        return Resolve::Miss;
    }
    const Program& program = *it->second;
    Dlss::NoteWeightsInfo(program.info, params.Base());
    Dlss::NoteShaderInfo(program.info, params.Base());
    if (program.precompiled && !program.precompiled_seen) [[unlikely]] {
        out.miss = Miss::PrecompiledFirstUse; // the front records the use, may retire
        return Resolve::Miss;
    }
    // GetProgram's per-draw refresh, on the private copy.
    Shader::Info& view = out.View(program.info);
    view.pgm_base = params.Base();
    view.user_data = params.user_data;
    if (pre) {
        view.flattened_ud_buf = out.pre_flat[out_idx]; // the worker's checked walk, outside the lock
    } else {
        view.RefreshFlatBuf();
    }
    auto& runtime_info = out.runtime_infos[out_idx];
    runtime_info.Initialize(stage_in);
    FillStageRuntimeInfo(runtime_info, regs, stage_in, stage_out, nullptr, out.key, false);
    // The plain first-match selection (the prepared-programs guard, when enabled, is asserted
    // equal to it by its verify mode and is skipped here).
    const Shader::StageSpecialization spec(view, runtime_info, profile, binding);
    const auto found = std::ranges::find(program.modules, spec, &Program::Module::spec);
    if (found == program.modules.end()) {
        out.miss = Miss::NoPermutation; // the front would compile a permutation
        return Resolve::Miss;
    }
    const size_t perm_idx = static_cast<size_t>(std::distance(program.modules.begin(), found));
    if (out.capture_guards) {
        // The front's prepared-programs guard for this exact selection (incoming bindings).
        out.guards[out_idx].Capture(view, runtime_info, binding, spec, perm_idx);
        out.guard_views[out_idx] = &view;
    }
    view.AddBindings(binding); // identical to the shared Info's AddBindings (immutable half)
    out.key.stage_hashes[out_idx] = HashCombine(params.hash, perm_idx);
    slot.info = &program.info;
    slot.pgm_base = view.pgm_base;
    slot.user_data.assign(view.user_data.begin(), view.user_data.end());
    slot.flattened_ud_buf = view.flattened_ud_buf;
    slot.module = found->module;
    view_out = &view;
    module_out = &*found;
    return Resolve::Bound;
}

const GraphicsPipeline* PipelineCache::LookupGraphicsShared(const AmdGpu::Regs& regs,
                                                            GraphicsLookupScratch& out) const {
    using Miss = GraphicsLookupScratch::Miss;
    if (regs.stage_enable.raw != AmdGpu::ShaderStageEnable::VgtStages::Vs) {
        out.miss = Miss::UnsupportedStages;
        return nullptr;
    }
    auto& key = out.key;
    bool db_enabled = false;
    bool skip_cb_binding = false;
    GraphicsKeyFirstPass(regs, key, db_enabled, skip_cb_binding);

    // RefreshGraphicsStages for the Vs case: fragment first, then vertex, one binding chain;
    // neither stage's absence is an error (bind_stage's results are ignored there).
    Shader::Backend::Bindings binding{};
    const Shader::Info* fs_view = nullptr;
    const Shader::Info* vs_view = nullptr;
    const Program::Module* fs_module = nullptr;
    const Program::Module* vs_module = nullptr;
    if (ResolveStageReadOnly(regs, Stage::Fragment, LogicalStage::Fragment, binding, out,
                             fs_view, fs_module) == Resolve::Miss) {
        return nullptr;
    }
    const auto* fs_info = out.stages[static_cast<u32>(LogicalStage::Fragment)].info;
    key.mrt_mask = fs_info ? fs_info->mrt_mask : 0u;
    key.num_color_attachments = std::bit_width(key.mrt_mask);
    if (ResolveStageReadOnly(regs, Stage::Vertex, LogicalStage::Vertex, binding, out, vs_view,
                             vs_module) == Resolve::Miss) {
        return nullptr;
    }
    // RefreshGraphicsStages' member fetch_shader = the selected VS permutation's (the fragment
    // stage never carries one).
    const Shader::Gcn::FetchShaderData* fetch =
        vs_module ? vs_module->fetch_shader.get() : nullptr;
    if (vs_view && fetch && !instance.IsVertexInputDynamicState()) {
        // Without vertex input dynamic state, the pipeline needs to specialize on format.
        u32 vertex_binding = 0;
        for (const auto& attrib : fetch->attributes) {
            const auto& buffer = attrib.GetSharp(*vs_view);
            ASSERT_MSG(vertex_binding < MaxVertexBufferCount,
                       "Vertex attribute binding count exceeded limit: {} >= {}", vertex_binding,
                       MaxVertexBufferCount);
            key.vertex_buffer_formats[vertex_binding++] =
                Vulkan::LiverpoolToVK::SurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
        }
    }

    GraphicsKeySecondPass(regs, key, db_enabled, skip_cb_binding, false);

    const auto it = graphics_pipelines.find(key);
    if (it == graphics_pipelines.end()) {
        out.miss = Miss::NoPipeline; // the front would construct it (or queue it, async)
        return nullptr;
    }
    if (!it->second) {
        out.miss = Miss::PipelinePending; // SHADPS4_ASYNC_PIPELINES: the front returns null too
        return nullptr;
    }
    out.pipeline = it->second.get();
    out.fetch_shader = out.pipeline->GetFetchShader();
    return out.pipeline;
}

const GraphicsPipeline* PipelineCache::LookupOrBuildGraphics(const AmdGpu::Regs& regs,
                                                             GraphicsLookupScratch& out) {
    if (const auto* pipeline = LookupGraphics(regs, out)) {
        return pipeline;
    }
    // The front's own lookup over `regs` (re-checks the tables, then compiles exactly as
    // GetGraphicsPipeline would; a pending async key still yields null). It writes the member
    // scratch and the bound stages' shared Infos, from which the result is captured.
    std::unique_lock lk{graphics_mutex};
    const GraphicsPipeline* pipeline = GraphicsLookupLocked(regs);
    CaptureFrontLocked(out, pipeline);
    return pipeline;
}

void PipelineCache::CaptureFront(GraphicsLookupScratch& out,
                                 const GraphicsPipeline* pipeline) const {
    CaptureFrontLocked(out, pipeline);
}

void PipelineCache::CaptureFrontLocked(GraphicsLookupScratch& out,
                                       const GraphicsPipeline* pipeline) const {
    // What liverpool.cpp copies after the front's lookup: pipeline->GetStages() and, per bound
    // stage, the shared Info's per-draw fields (WriteStage), plus the member selection.
    out.Reset();
    out.pipeline = pipeline;
    out.key = graphics_key;
    out.runtime_infos = runtime_infos;
    if (!pipeline) {
        return;
    }
    out.fetch_shader = pipeline->GetFetchShader();
    const auto stages = pipeline->GetStages();
    for (u32 i = 0; i < stages.size() && i < out.stages.size(); ++i) {
        const Shader::Info* info = stages[i];
        if (!info) {
            continue;
        }
        auto& slot = out.stages[i];
        slot.info = info;
        slot.pgm_base = info->pgm_base;
        slot.user_data.assign(info->user_data.begin(), info->user_data.end());
        slot.flattened_ud_buf = info->flattened_ud_buf;
        slot.module = modules[i];
    }
}

bool PipelineCache::LookupMatchesFront(const GraphicsLookupScratch& s,
                                       const GraphicsPipeline* front_result,
                                       const char** why) const {
    // Field by field, never struct bytes (rules-appendix: a verify mode compares FIELDS).
    const auto fail = [&](const char* what) {
        if (why) {
            *why = what;
        }
        return false;
    };
    if (s.pipeline != front_result) {
        return fail("pipeline");
    }
    if (!front_result) {
        if (why) {
            *why = nullptr;
        }
        return true; // both null: the miss reason may differ by design (see out.miss)
    }
    if (!(s.key == graphics_key)) {
        return fail("key");
    }
    if (s.fetch_shader != front_result->GetFetchShader()) {
        return fail("fetch shader");
    }
    const auto stages = front_result->GetStages();
    for (u32 i = 0; i < s.stages.size(); ++i) {
        const Shader::Info* info = i < stages.size() ? stages[i] : nullptr;
        const auto& slot = s.stages[i];
        if (slot.info != info) {
            return fail("stage info pointer");
        }
        if (!info) {
            if (!slot.user_data.empty() || !slot.flattened_ud_buf.empty()) {
                return fail("unbound stage carries data");
            }
            continue;
        }
        if (slot.pgm_base != info->pgm_base) {
            return fail("pgm_base");
        }
        if (!std::ranges::equal(slot.user_data, info->user_data)) {
            return fail("user data words");
        }
        if (!std::ranges::equal(slot.flattened_ud_buf, info->flattened_ud_buf)) {
            return fail("flattened words");
        }
        if (slot.module != modules[i]) {
            return fail("module");
        }
        if (!SameRuntimeInfo(s.runtime_infos[i], runtime_infos[i])) {
            return fail("runtime info");
        }
    }
    if (why) {
        *why = nullptr;
    }
    return true;
}

} // namespace Vulkan
