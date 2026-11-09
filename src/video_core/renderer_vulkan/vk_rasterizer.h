// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>

#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "common/unique_function.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/cp_split.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/amdgpu/regs_color.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_bindless.h"
#include "video_core/renderer_vulkan/vk_native_bindings.h"
#include "video_core/renderer_vulkan/vk_occlusion.h"
#include "shader_recompiler/native_emit.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Common {
class DrawSampleProbe;
}

namespace Vulkan {

class Scheduler;
class RenderState;
class GraphicsPipeline;
struct TexMemoRecord;
struct DrawPacketHeader;

class Rasterizer {
public:
    explicit Rasterizer(const Instance& instance, Scheduler& scheduler,
                        AmdGpu::Liverpool* liverpool);
    /// Multi-threaded back P1a: a worker instance on `primary`'s caches with its own per-draw
    /// scratch and memos (not used yet; notes/parrec-build-2026-10-08.md sec. 6).
    struct WorkerTag {};
    Rasterizer(Rasterizer& primary, WorkerTag);
    ~Rasterizer();

    /// The instance that translates split draw / dispatch records: SHADPS4_MTB_P1's worker
    /// instance (P1d, notes/parrec-build-2026-10-08.md sec. 14), else this one. Packets, Calls,
    /// faults and OnSubmit stay on the primary.
    [[nodiscard]] Rasterizer& Translating() noexcept {
        return mtb_worker ? *mtb_worker : *this;
    }
    [[nodiscard]] bool HasMtbWorker() const noexcept {
        return mtb_worker != nullptr;
    }

    [[nodiscard]] Scheduler& GetScheduler() noexcept {
        return scheduler;
    }

    [[nodiscard]] VideoCore::BufferCache& GetBufferCache() noexcept {
        return buffer_cache;
    }

    [[nodiscard]] VideoCore::TextureCache& GetTextureCache() noexcept {
        return texture_cache;
    }

    /// What a draw packet does, decided from registers alone (CP front/back split, step 3).
    /// Everything except Draw replaces the draw with a back-side action or a marker.
    enum class DrawClass : u8 {
        Draw,               ///< An ordinary draw.
        EliminateFastClear, ///< CB mode EliminateFastClear: EliminateFastClear() instead.
        FmaskDecompress,    ///< CB mode FmaskDecompress: skipped (marker only).
        Resolve,            ///< CB mode Resolve: Resolve() instead.
        PrimitiveNone,      ///< Primitive type None: skipped (marker only).
        DepthCopy,          ///< CB disabled + forced depth dirty/valid: DepthStencilCopy(1, 0).
        StencilCopy,        ///< Same for stencil: DepthStencilCopy(0, 1).
        DepthStencilCopy,   ///< Both: DepthStencilCopy(1, 1).
        DlssResolve,        ///< SHADPS4_DLSS: the game's T2x resolve -> DlssResolve() (vk_dlss.h).
    };
    /// Pure: reads only `regs`, touches no cache, scheduler or Rasterizer member, so the CP front
    /// can call it.
    static DrawClass ClassifyDraw(const AmdGpu::Regs& regs);
    /// The back-side action for a class other than Draw (texture cache work or a marker).
    void ExecuteFilteredDraw(DrawClass draw_class);

    /// CP front/back split (SHADPS4_CP_SPLIT, video_core/amdgpu/cp_split.h): one draw as the
    /// front hands it to the back. Trivially copyable; it travels inside a ring record.
    struct SplitDraw {
        AmdGpu::CpSplit::DrawArgs args{};
        DrawClass draw_class{DrawClass::Draw};
        bool skip{};                         ///< SHADPS4_EMPTY_DRAWS=2 dropped it on the front.
        u8 dlss_phase{};                     ///< DlssResolve: the frame's jitter (vk_dlss.h).
        float dlss_jitter[2]{};              ///< phase 4: the explicit offset (Halton)
        const GraphicsPipeline* pipeline{};  ///< Null: the lookup failed or was not reached.
    };
    /// Front thread: classification, hazard note, empty-draw skip and pipeline lookup, reading
    /// the front's registers. Touches no cache, the scheduler or any per-draw member.
    void PrepareDrawFront(SplitDraw& draw);
    /// SHADPS4_LEAN=2: the per-draw stage fields the front's last PrepareDrawFront captured under
    /// the cache's exclusive lock (null in other modes / for draws without a lookup).
    const GraphicsLookupScratch* FrontCapture() const {
        return front_capture_valid ? &front_capture : nullptr;
    }
    /// SHADPS4_NATIVE_REC=2, front: PrepareDrawFront for a job-thread run member, whose
    /// pipeline is the run head's (no lookup). False: not a plain draw (take the normal path).
    bool PrepareMemberFront(SplitDraw& draw, const GraphicsPipeline* pipeline);
    /// SHADPS4_NATIVE_COLOUR: front thread, after FRONT_DERIVE (notes/colour-pass-2026-10-09.md).
    void PrepareNativeColourFront(const GraphicsPipeline* pipeline,
                                  const AmdGpu::CpSplit::VertexInputsDerived& vi) {
        if (pipeline) {
            pipeline_cache.PrepareNativeColour(pipeline, vi);
        }
    }
    // ---- Architecture A worker parser (video_core/amdgpu/lean_parse.*): pure, regs-taking
    // mirrors of the front's per-draw decisions. Any thread; no member, counter or cache touched.
    /// SkipEmptyDraw's decision for `regs` (SHADPS4_EMPTY_DRAWS=2 and zero indices / instances)
    /// without the front's EMPTY_DRAWS counters.
    static bool SkipEmptyDrawRegs(const AmdGpu::Regs& regs);
    /// q40 (diagnostic): SHADPS4_NO_PARTICLES=1 and this is the particle card pipeline (vs_bbd2d927 + fs_24dad4be).
    static bool NoParticlesSkip(const GraphicsPipeline* pipeline);
    /// The lean memo is exact only with dynamic vertex input (else the key specializes on V#
    /// formats, which the front's prepared path checks separately).
    bool LeanVertexInputDynamic() const {
        return instance.IsVertexInputDynamicState();
    }
    /// Read-only: would PipelineCache::PrepareNativeColour(pipeline, vi) return without changing
    /// anything? False when the front would decide the pipeline's eligibility or claim a slot for
    /// this vertex description (the worker stops and leaves the draw to the front). Reads the
    /// front-written eligibility byte and the slot keys only.
    static bool NativeColourVariantReady(const GraphicsPipeline* pipeline,
                                         const AmdGpu::CpSplit::VertexInputsDerived& vi);

    /// Front thread: compute pipeline lookup against the current queue's cs registers.
    const ComputePipeline* PrepareDispatchFront();
    /// CP split run-ahead VERIFY, back thread, only while the front is parked on a synchronous
    /// draw record: the graphics pipeline lookup of PrepareDrawFront alone (no hazard note, no
    /// empty-draw counters), against the front's registers and pipeline cache.
    const GraphicsPipeline* RelookupDrawFront();
    /// Back thread: runs a draw the front prepared, with the per-stage user data it copied
    /// (indexed by LogicalStage) and, with FRONT_DERIVE, the vertex inputs it derived (null
    /// otherwise). Registers come from Liverpool::BackRegs().
    void ExecuteDraw(const SplitDraw& draw,
                     const std::array<Shader::BoundStage, Shader::MaxStageTypes>& stages,
                     const AmdGpu::CpSplit::VertexInputsPayload* front_vi);
    /// SHADPS4_DRAW_TAG: the game job thread's (slot, sub-run) tag of the draw ExecuteDraw runs
    /// next (0 = untagged), set by Liverpool::ExecuteDrawRecord.
    u32 draw_tag{};
    /// SHADPS4_NATIVE_REC_RUN: the job-thread run this draw heads (tag 0: none), set by
    /// Liverpool::ExecuteDrawRecord.
    u32 nr_head_tag{};
    u32 nr_head_idx{};
    /// SHADPS4_NATIVE_REC >= 2: the job record of the draw ExecuteDraw runs next (tag 0: none),
    /// its packet and whether the front's member path posted it; set by ExecuteDrawRecord.
    u32 draw_job_tag{};
    u32 draw_job_idx{};
    const void* draw_job_cmd{};
    bool draw_job_member{};
    /// SHADPS4_NATIVE_SEGB: NativeOpen opened a live batch while this draw record ran (reset by
    /// Liverpool::ExecuteDrawRecord; with rec_outcome it tells an opener from a generic draw).
    bool segb_opened{};
    /// SHADPS4_NATIVE_SEGX (sec. 23): the read-only checks of a Segment record (variant, job
    /// ring lifetime, each union inside one cached buffer); `why` = the reason code on false.
    bool NativeSegEvaluate(const AmdGpu::CpSplit::SplitSegmentRecord& seg,
                           const GraphicsPipeline* base,
                           const AmdGpu::CpSplit::VertexInputsPayload* vi, const u32* list,
                           u32& why);
    /// SHADPS4_NATIVE_SEGX=1: the Segment record whose member 0 ExecuteDraw runs now (null:
    /// none), its ops / commands / list, and the outcome (0 open not reached, 1 executed, else
    /// a reason code), set by Liverpool::ExecuteSegmentRecord.
    const AmdGpu::CpSplit::SplitSegmentRecord* segx_pending{};
    const AmdGpu::CpSplit::SegOp* segx_ops{};
    const u32* segx_cmds{};
    const u32* segx_list{};
    u32 segx_result{};
    /// Brick C v3 item 1: draws of the Segment record being executed that still run after the
    /// current one (member 0 of a segment of n: n - 1; a replayed member j: n - 1 - j). While it
    /// is > 0 the single-draw peek is skipped: the next POSTED record is not a member.
    u32 segx_left{};
    /// Brick C v3 item 4: the pending segment's distinct materials (0: none) and, per further
    /// material, its representative member's stages (spans into the Segment record: valid
    /// while it runs) and the slots the open resolved for it (row 0 = the opener's bl_slots).
    u32 segx_n_mat{};
    std::array<std::array<Shader::BoundStage, Shader::MaxStageTypes>, Shader::NativeSegMaxMaterials>
        segx_mat_stages{};
    std::array<std::array<u32, Shader::NativeSegMatWords>, Shader::NativeSegMaxMaterials>
        segx_mat_slots{};
    /// SHADPS4_NATIVE_REC_PREP (notes/native-path-2026-10-09.md sec. 9): the open batch's run
    /// members prepared from the job records (records + indirect commands in the batch slots,
    /// count not advanced); each is confirmed at its own draw record (NativePrepConfirm).
    struct NativePrep {
        bool active{};
        bool verify{};      ///< =2: prepared into `side`, never confirmed; NativeJoin compares
        u32 tag{};          ///< job tag of the run
        u32 first_idx{};    ///< job ring index of the first prepared member
        u32 next_idx{};     ///< the next member to confirm
        u32 end_idx{};      ///< one past the last prepared member
        u32 slot0{};        ///< batch slot of first_idx
        u64 epoch{}, gen{}; ///< dirty epoch / buffer structure generation at the prepare
        std::array<const void*, 64> cmd{}; ///< each member's DRAW_INDEX_2 packet
        std::array<u32, 64> icount{}, inst{}; ///< ring mode: each member's index / instance count
        std::vector<u32> side;              ///< verify: (record + 5 command words) per member
        u32 n{};                            ///< members prepared
        /// Ring mode (sec. 10-11): each member's index address; the opener's inherited VS / PS
        /// user-data registers and their values (every member must see the same).
        bool ring_src{};
        std::array<u64, 64> iaddr{};
        u32 inh_vs{}, inh_ps{};
        std::array<u32, 16> inh_val_vs{}, inh_val_ps{};
    } native_prep;
    /// Back thread, Liverpool::ExecuteDrawRecord: true = the draw was a prepared member and went
    /// into the open batch; false = run the record normally (the prepare is dropped on refusal).
    bool NativePrepConfirm(const SplitDraw& draw, bool plain,
                           const AmdGpu::CpSplit::VertexInputsPayload* vi);
    /// SHADPS4_NATIVE_TIMING: what the back did with the draw record it runs (Liverpool's record
    /// timer): 0 other / filtered, 1 shadow join, 2 colour join, 3 DrawTail, 4 shadow confirm,
    /// 5 colour confirm.
    u32 rec_outcome{};
    void NoteRecordCycles(u64 dt);
    /// SHADPS4_A1 census (Liverpool, back thread): native batch opens so far, whether the last
    /// open was a colour batch, and the generic-path reason of the last draw (reset per draw).
    [[nodiscard]] u64 A1NativeOpens() const noexcept {
        return native_stats.opens;
    }
    [[nodiscard]] bool A1LastOpenColour() const noexcept {
        return native_stats.last_open_colour;
    }
    [[nodiscard]] const char* A1TakeWhy() noexcept {
        return std::exchange(native_stats.last_why, nullptr);
    }
    /// SHADPS4_A2 (back thread): a worker stretch starts at this draw - closes any native batch,
    /// resolves the render targets for `pipeline` from the back's mirror and begins rendering.
    void A2BeginStretch(const GraphicsPipeline* pipeline);
    /// After the stretch's commands: every tracked command-buffer state is unknown.
    void A2EndStretch();
    /// Ends rendering and records one global memory barrier (writes -> any access).
    void A2GlobalBarrier();
    /// SHADPS4_A2 formatted slots, back thread, before the stretch's pass: the typed lookup of
    /// [addr, addr + V# size) as NativeDirectBind does it (ObtainBuffer is_texel: an image ->
    /// buffer sync of overlapping image data) and its shader-read barrier. False: no buffer
    /// starts at addr (the stretch runs on the old path).
    bool A2SyncTexel(VAddr addr, u64 vsize);
    [[nodiscard]] bool A2OcclusionOpen() const;
    /// SHADPS4_LEAN worker pre-check (lock-free read-protection test).
    [[nodiscard]] const VideoCore::PageManager& LeanPageManager() const noexcept {
        return page_manager;
    }
    /// SHADPS4_A2 colour stretches (architecture-a sec. 10), back thread, before the stretch's
    /// pass, once per distinct material: its 32 slot words through BindlessResolve with the
    /// representative draw's fragment stage; its images are appended to `ids`. False: the
    /// stretch runs on the old path. May record sampled-layout barriers (outside any pass).
    bool A2ResolveMaterial(const GraphicsPipeline* base, const Shader::BoundStage& fs, u32* slots,
                           u32& n_img, u32& n_smp, std::vector<VideoCore::ImageId>& ids,
                           u64 h1 = 0, u64 h2 = 0, u32 kn = 0,
                           const Shader::BoundStage* vs = nullptr);
    /// SHADPS4_A2_MAT_MEMO (l4d): the memo's result for this material key if every condition of
    /// an all-hit BindlessResolve holds (then the same LRU touches); false: resolve it.
    bool A2MaterialMemo(const GraphicsPipeline* base, u64 h1, u64 h2, u32 kn, u32* slots,
                        u32& n_img, u32& n_smp, std::vector<VideoCore::ImageId>& ids);
    /// NativeColourGates (config + no pending stencil write-backs), once per colour stretch.
    [[nodiscard]] bool A2ColourGates() const {
        return NativeColourGates();
    }
    [[nodiscard]] u32 A2BindlessCur() const noexcept {
        return bindless.cur;
    }
    /// Marks the current bindless set used by the batch being recorded; returns it.
    vk::DescriptorSet A2BindlessCommit();
    /// After A2BeginStretch: none of `ids` is a bound render target or forced to general.
    [[nodiscard]] bool A2ImagesSafe(const std::vector<VideoCore::ImageId>& ids) const;
    /// SHADPS4_A2=2: the slot words BindlessRingWrite wrote last (the colour verify).
    struct A2SlotProbe {
        u32 tag{}, idx{};
        u64 n{};
        std::array<u32, Shader::NativeBindlessSlots> slots{};
    } a2_probe;
    /// SHADPS4_A2=2: what each colour draw of a verified stretch must get (key tag << 32 | idx);
    /// BindlessRingWrite compares every write against it and erases the entry.
    struct A2Expect {
        std::array<u32, Shader::NativeBindlessSlots> slots{};
        u32 n_img{}, n_smp{};
    };
    std::unordered_map<u64, A2Expect> a2_expect;
    u64 a2_slot_cmp{}, a2_slot_bad{};
    [[nodiscard]] const Instance& A2Instance() const noexcept {
        return instance;
    }
    void A2TexTableTouch(); // q8: LRU touches of the table's images / samplers (rate-limited)
    /// q8b (A): the sampler generation and meta-clear epoch the bindless caches hold still match
    /// the texture cache's (a sampler GC / meta clear since: no table hit may be used).
    bool A2TexTableStateOk() const noexcept {
        return texture_cache.SamplerGeneration() == bindless.sampler_gen &&
               texture_cache.MetaClearEpoch() == bindless.meta_epoch;
    }
    void MaybeEarlyCut(bool run_end = false);                  // q7 / q7b
    static int EarlyCutMode();                                 // q7b
    [[nodiscard]] VideoCore::BufferCache& A2BufferCache() noexcept {
        return buffer_cache;
    }
    /// Back thread: runs a dispatch the front looked up. The caller points
    /// Liverpool::BackCsRegs() at the record's ComputeProgram snapshot.
    void ExecuteDispatch(const ComputePipeline* pipeline,
                         const std::array<Shader::BoundStage, Shader::MaxStageTypes>& stages,
                         bool is_indirect, VAddr address, u32 offset, u32 size);
    /// SHADPS4_GUEST_DISPATCH_ORDER: the guest run id of the dispatch about to execute (0 = none).
    void SetDispatchRun(u64 run) {
        dispatch_run = run;
    }
    /// SHADPS4_ASC_STREAM: guest queue of the dispatch record being executed (0 = graphics).
    void SetDispatchQueue(u32 queue) {
        dispatch_queue = queue;
    }

    void Draw(bool is_indexed, u32 index_offset = 0);
    void ScanRegistersForValue();
    /// SHADPS4_NEO_IDBUF_DERIVE=1: fill the Pro ID buffer from the last 2-sample scene depth.
    void DeriveNeoIdBuffer(VideoCore::Image& id_image);
    /// SHADPS4_DLSS: the back-side action that replaces the T2x resolve draw.
    void DlssResolve();
    u64 dlss_calls{};
    u64 dlss_missing{};
    u8 dlss_phase_in{};                   ///< the phase of the action being executed
    float dlss_jitter_in[2]{};            ///< phase 4: its explicit offset (render px)
    bool dlss_last_explicit{};            ///< the last resolve used an explicit offset
    std::array<u64, 4> dlss_phases{};     ///< by phase: unknown / 1 / 2 / T2x off
    u64 dlss_evaluated{};
    u32 dlss_last_combo{};
    bool dlss_reset_pending{true};
    struct NeoIdDerive;
    std::shared_ptr<NeoIdDerive> neo_id_derive;
    VAddr neo_depth_addr{};
    VideoCore::Extent3D neo_depth_size{};
    u32 neo_depth_x0y0{};
    void DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 size, u32 max_count,
                      VAddr count_address);

    void DispatchDirect();
    void DispatchIndirect(VAddr address, u32 offset, u32 size);

    void ScopeMarkerBegin(const std::string_view& str, bool from_guest = false);
    void ScopeMarkerEnd(bool from_guest = false);

    /// SHADPS4_OCCLUSION_QUERIES: the identity of a query's draw (pipeline and targets).
    u64 OcclusionDrawSig(const Pipeline* pipeline) const;
    /// SHADPS4_OCCLUSION_QUERIES: a ZPASS counter dump of the guest (command processor thread, in
    /// order with the draws). Returns false when the feature is off: the caller writes the fake.
    bool OcclusionDump(u64* results, s32 pairs) {
        if (!occlusion) {
            return false;
        }
        occlusion->Dump(results, pairs);
        return true;
    }
    void ScopedMarkerInsert(const std::string_view& str, bool from_guest = false);
    void ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                 bool from_guest = false);

    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);
    u32 ReadDataFromGds(u32 gsd_offset);
    bool InvalidateMemory(VAddr addr, u64 size,
                          VideoCore::InvalidateOrigin origin = VideoCore::InvalidateOrigin::Explicit);
    bool ReadMemory(VAddr addr, u64 size);
    void ProcessDownloadImages();
    bool IsMapped(VAddr addr, u64 size);
    /// SHADPS4_LEAN_PARSE workers: the GPU-mapped interval [lo, hi) containing addr (false if
    /// unmapped). Cached by the caller together with MappedGen(), bumped on every map / unmap.
    bool MappedInterval(VAddr addr, VAddr& lo, VAddr& hi);
    static u64 MappedGen() {
        return mapped_gen.load(std::memory_order_acquire);
    }
    void MapMemory(VAddr addr, u64 size);
    void UnmapMemory(VAddr addr, u64 size);
    /// MemoryManager's own protection changes (see PageManager::ExternalProtect).
    void ExternalProtect(VAddr addr, u64 size, Core::MemoryPermission perms) {
        page_manager.ExternalProtect(addr, size, perms);
    }

    void CpSync();
    /// SHADPS4_CP_SPLIT_WAIT_BARRIER: a skipped guest wait on an end-of-pipe label - every earlier
    /// command's memory writes are visible to every later command.
    void WaitBarrier();
    u64 Flush();
    void Finish();
    /// SHADPS4_FENCE_AT_GPU: run `func` on the scheduler's priority thread once the host GPU has
    /// finished everything recorded so far; submits the current batch so that happens.
    void DeferToGpu(Common::UniqueFunction<void>&& func, bool flush);
    /// Mode 2 (lazy): submits the batch holding deferred fences if it is still being recorded.
    void FlushLazyFences();
    u64 lazy_fence_tick{};
    void OnSubmit();

    PipelineCache& GetPipelineCache() {
        return pipeline_cache;
    }

    template <typename Func>
    void ForEachMappedRangeInRange(VAddr addr, u64 size, Func&& func) {
        const auto range = boost::icl::interval_set<VAddr>::interval_type::right_open(addr, addr + size);
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (const auto& mapped_range : (mapped_ranges & range)) {
            func(mapped_range);
        }
    }

private:
    void PrepareRenderState(const GraphicsPipeline* pipeline);
    RenderState BeginRendering(const GraphicsPipeline* pipeline);
    void Resolve();
    void DepthStencilCopy(bool is_depth, bool is_stencil);
    void EliminateFastClear();

    /// `commit`: end with Scheduler::CommitDynamicState (false only while a draw packet is
    /// pending; the caller commits after recording the packet if a commit would record).
    void UpdateDynamicState(const GraphicsPipeline* pipeline, bool is_indexed,
                            bool commit = true) const;
    void UpdateViewportScissorState() const;
    mutable u64 viewport_epoch{0};
    mutable u64 viewport_memo_skips{0};
    mutable u64 viewport_memo_mismatches{0};
    void UpdateDepthStencilState() const;
    void UpdatePrimitiveState(bool is_indexed) const;
    void UpdateRasterizationState() const;
    void UpdateColorBlendingState(const GraphicsPipeline* pipeline) const;

    bool FilterDraw();
    /// SHADPS4_EMPTY_DRAWS counting and (=2) skipping; registers and statics only.
    static bool SkipEmptyDraw(const AmdGpu::Regs& regs);

    // Everything after the pipeline lookup, shared by the inline path and the CP split's back.
    // bound_stages must already hold the draw's stage data. `front_vi`: the split record's
    // FRONT_DERIVE block (null on the inline path and without the flag).
    void DrawTail(const GraphicsPipeline* pipeline, bool is_indexed, u32 index_offset,
                  Common::DrawSampleProbe& sample, std::chrono::steady_clock::time_point t_entry,
                  std::chrono::steady_clock::time_point t_pipeline, u64 pipeline_ns,
                  const AmdGpu::CpSplit::VertexInputsPayload* front_vi = nullptr);
    void DrawIndirectTail(const GraphicsPipeline* pipeline, bool is_indexed, VAddr arg_address,
                          u32 offset, u32 stride, u32 max_count, VAddr count_address);
    void DispatchDirectTail(const ComputePipeline* pipeline);
    void DispatchIndirectTail(const ComputePipeline* pipeline, VAddr address, u32 offset,
                              u32 size);

    void BindBuffers(const Shader::BoundStage& bound, Shader::Backend::Bindings& binding,
                     Shader::PushData& push_data);
    void BindTextures(const Shader::BoundStage& bound, Shader::Backend::Bindings& binding);
    bool BindResources(const Pipeline* pipeline);
    /// SHADPS4_NATIVE_DIRECT (notes/native-path-2026-10-09.md sec. 15): the native run start.
    /// Candidate = a bindless colour variant (ring mode) for a draw with a job-thread record;
    /// the bind = set 0 from the draw's V#s (validity cache, no per-binding lookups, no texture
    /// lookups: images are bindless slots), vertex / index from the front, then NativeOpen.
    /// False: nothing to undo, the generic BindResources follows.
    const GraphicsPipeline* NativeDirectCandidate(const GraphicsPipeline* pipeline, bool is_indexed,
                                                  const AmdGpu::CpSplit::VertexInputsPayload* vi);
    bool NativeDirectBind(const GraphicsPipeline* base, const GraphicsPipeline* native,
                          u32 index_offset, const AmdGpu::CpSplit::VertexInputsPayload* front_vi,
                          bool open);
    const VideoCore::Buffer* NativeSpanSync(VAddr lo, VAddr hi, VideoCore::NativeRunAccess access);

    /// SHADPS4_BIND_MEMO_STATS=1 (diagnostic, default off; notes/bind-memo-stats-2026-09-25.md):
    /// evaluates a per-stage binding memo next to the unchanged full path and logs its hit rate,
    /// miss reasons, output mismatches and the time a hit would save. Never skips anything.
    /// Reads user data and the flattened buffer only through `bound`.
    /// Returns bit 0 = both key halves hit, bit 1 = both halves would hit.
    u8 BindStageMemoStats(const Pipeline* pipeline, const Shader::BoundStage& bound,
                          Shader::Backend::Bindings& binding);
    void BindMemoEndDraw(bool is_compute, u8 draw_bits);
    bool bind_memo_stats{};  // read once in the constructor
    bool bind_memo_active{}; // inside a stats stage: BindBuffers records uploads per binding
    bool bind_memo_timing{}; // inside a timed stats stage: BindBuffers times always-full bindings
    u64 bind_memo_fixed_ticks{};
    u64 bind_memo_redirected{}; // image bindings BindTextures redirected to another image
    std::array<u8, Shader::NUM_BUFFERS> bind_memo_uploaded{};

    /// SHADPS4_TEX_BIND_MEMO (default off; notes/bind-memo-stats-2026-09-25.md): the texture
    /// half of the binding memo. =1: when a stage's T#/S#/border key and every validity check
    /// match the record kept on the pipeline, replays the recorded views and samplers instead of
    /// BindTextures, keeping what a hit must still do (binding flags, LRU touches, bound_images,
    /// a layout transition when the state differs, descriptor writes). =2 verify: runs the hit,
    /// rolls its outputs back, runs BindTextures and compares (log + counter, never a crash).
    void BindTexturesMemo(const Pipeline* pipeline, const Shader::BoundStage& bound,
                          Shader::Backend::Bindings& binding);
    /// A hit's outputs; returns the number of Transit calls it made.
    u32 EmitTexMemoHit(const TexMemoRecord& rec, Shader::Backend::Bindings& binding);
    void RecordTexMemo(TexMemoRecord& rec, const Shader::Info& stage, u64 tex_gen, u64 smp_gen,
                       u64 border, size_t img_begin, u32 sw_begin);
    u8 tex_bind_memo{};        // 0 off, 1 memo, 2 verify; read once in the constructor
    // SHADPS4_GUEST_DISPATCH_ORDER state (back thread): the guest run of the next dispatch, the
    // effective run passed to buffer barriers while a dispatch binds (0 = draws / no run), and the
    // previous dispatch's guest run, effective run and recording sequence at its end.
    u64 dispatch_run{};
    u64 cur_dispatch_run{};
    u64 prev_dispatch_run{};
    u64 prev_effective_run{};
    u64 prev_dispatch_end_seq{~u64{0}};
    u64 prev_dispatch_end_tick{};
    // SHADPS4_ASC_STREAM: per command stream copies of the run state above, the queue of the
    // dispatch being executed and the stream it is headed for (decided after buffer pass 1).
    u32 dispatch_queue{};
    u32 dispatch_target{};
    bool binding_compute{};
    bool dispatch_on_asc{};
    std::array<u64, 2> s_prev_run{};
    std::array<u64, 2> s_prev_effective{};
    std::array<u64, 2> s_prev_end_seq{~u64{0}, ~u64{0}};
    std::array<u64, 2> s_prev_end_batch{};
    u64 asc_dispatches{};
    u64 asc_demoted{};
    std::array<u64, 7> asc_demote_reason{};
    u64 asc_flushes{};
    bool AscCandidate(const ComputePipeline* pipeline) const;
    /// q9 census: AscCandidate's last refusal (1 heap descriptors, 2 no info, 3 images, 4 samplers,
    /// 5 uses_dma, 6 formatted, 7 special buffer).
    mutable u32 asc_why{};
    u64 StreamBatchId(u32 stream) const;
    u64 effective_run_counter{};
    bool bind_memo_rebound{};  // BindTextures re-found a needs_rebind image

    /// Fills bound_stages from the pipeline lookup result; called right after the lookup.
    void BindStageData(const Pipeline* pipeline) {
        const auto stages = pipeline->GetStages();
        const u32 first = pipeline->IsCompute() ? u32(Shader::LogicalStage::Compute) : 0u;
        for (u32 i = 0; i < stages.size(); ++i) {
            bound_stages[first + i] = Shader::BoundStage{stages[i]};
        }
    }
    /// The bound_stages entries of `pipeline`, in Pipeline::GetStages() order.
    std::span<const Shader::BoundStage> BoundStages(const Pipeline* pipeline) const {
        const auto all = std::span{bound_stages};
        return pipeline->IsCompute() ? all.last(1) : all.first(Shader::MaxStageTypes - 1);
    }
    const Shader::BoundStage& GetBoundStage(Shader::LogicalStage stage) const {
        return bound_stages[u32(stage)];
    }

    /// SHADPS4_STENCIL_STORAGE=1: shader storage writes into a depth buffer's stencil plane
    /// (PS4 Pro checkerboard mask) go to an R8 scratch image that is seeded from and copied back
    /// into the depth image's stencil aspect, instead of an invalid storage view of D32S8.
    VideoCore::ImageId StencilScratchFor(VideoCore::ImageId depth_id,
                                         const VideoCore::ImageInfo& info);
    bool IsStencilScratch(VideoCore::ImageId id) const;
    void CopyStencilPlane(VideoCore::ImageId src_id, VideoCore::ImageId dst_id);
    void FlushStencilWritebacks();
    struct StencilScratch {
        VideoCore::ImageId depth_id;
        u64 depth_uid;
        VideoCore::ImageId scratch_id;
    };
    std::vector<StencilScratch> stencil_scratches;
    boost::container::small_vector<std::pair<VideoCore::ImageId, VideoCore::ImageId>, 2>
        pending_stencil_writebacks; // (scratch, depth)

    /// Emits the bindings assembled by BindResources through the pipeline's binding model.
    void EmitPipelineResources(const Pipeline* pipeline);

    /// SHADPS4_DRAW_PACKET: EmitPipelineResources, or (true) the decision to send this draw's
    /// resources as one packet instead. When true nothing was recorded here: the caller must
    /// call RecordPacket before the next command it records for the draw, or fold the draw
    /// into it. `occlusion_draw`: the draw will be wrapped in an occlusion query (legacy path).
    /// Mode 0: exactly EmitPipelineResources. Mode 2: verify, then EmitPipelineResources.
    bool EmitPipelineResourcesOrPacket(const Pipeline* pipeline, bool occlusion_draw);
    /// Records the pending packet from the bindings BindResources left behind (buffer_barriers,
    /// buffer_infos, image_infos, push_data). `tail` carries only the flags and draw / dispatch
    /// arguments (none: the draw is recorded separately, as today).
    void RecordPacket(const Pipeline* pipeline, const DrawPacketHeader& tail);

    void ResetBindings() {
        for (auto& image_id : bound_images) {
            texture_cache.GetImage(image_id).binding = {};
        }
        bound_images.clear();
    }

    bool IsComputeMetaClear(const Pipeline* pipeline);
    bool IsComputeImageCopy(const Pipeline* pipeline);
    bool IsComputeImageClear(const Pipeline* pipeline);

private:
    friend class VideoCore::BufferCache;

    const Instance& instance;
    Scheduler& scheduler;
    /// What every rasterizer instance shares (multi-threaded back P1a,
    /// notes/parrec-build-2026-10-08.md sec. 6): owned by the primary rasterizer; a worker
    /// instance (Rasterizer(primary, WorkerTag)) references the primary's. The members below of
    /// the same names are references into it, so the draw path reads them as before. Member
    /// order = the old declaration order (same construction / destruction order).
    struct Shared {
        Shared(Rasterizer* owner, const Instance& instance, Scheduler& scheduler,
               AmdGpu::Liverpool* liverpool);
        // Declared before the caches: created before any buffer, view, sampler or pipeline, and
        // destroyed after them so object-owned heap slots can still be released. Null in legacy.
        std::unique_ptr<NativeBindings> native_bindings;
        VideoCore::PageManager page_manager;
        VideoCore::BufferCache buffer_cache;
        VideoCore::TextureCache texture_cache;
        boost::icl::interval_set<VAddr> mapped_ranges;
        Common::SharedFirstMutex mapped_ranges_mutex;
        PipelineCache pipeline_cache;
        std::unique_ptr<OcclusionQueries> occlusion;
        /// Set by any instance's draw path, consumed by the primary's OnSubmit (P1d audit).
        bool fault_process_pending{};
    };
    std::unique_ptr<Shared> owned_shared; ///< Primary instance only.
    GraphicsLookupScratch front_capture{}; ///< SHADPS4_LEAN=2 (FrontCapture)
    /// One GPU address space (mapped_ranges is shared by every instance): see MappedGen.
    static inline std::atomic<u64> mapped_gen{0};
    bool front_capture_valid{};
    Shared& shared;
    std::unique_ptr<NativeBindings>& native_bindings;
    VideoCore::PageManager& page_manager;
    VideoCore::BufferCache& buffer_cache;
    VideoCore::TextureCache& texture_cache;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    boost::icl::interval_set<VAddr>& mapped_ranges;
    Common::SharedFirstMutex& mapped_ranges_mutex;
    PipelineCache& pipeline_cache;
    /// SHADPS4_DRAW_REPEAT (back thread only): the previous direct draw, see DrawRepeatMatch.
    struct DrawRepeatState {
        bool valid{};    ///< the previous direct draw completed and is a base for a repeat
        bool eligible{}; ///< the draw being processed: key gathered, pipeline eligible
        const GraphicsPipeline* pipeline{};
        /// Binding inputs: words[kidx] = the previous draw's (the key), words[kidx ^ 1] = the
        /// draw being processed (the candidate). DrawRepeatStore flips kidx.
        static constexpr u32 MaxKeyWords = 1024;
        std::array<std::array<u32, MaxKeyWords>, 2> words{};
        std::array<u32, 2> sizes{};
        u32 kidx{};
        u64 dirty_epoch{}, image_epoch{}, buf_gen{}, tex_gen{}, smp_gen{}, border{}; ///< previous
        u64 c_dirty_epoch{}, c_image_epoch{}, c_buf_gen{}, c_tex_gen{}, c_smp_gen{},
            c_border{}; ///< this draw
        /// Textured stages of the draw being processed that did not hit the texture bind memo
        /// (reset by BindResources): such a draw is never the base of a repeat.
        u32 tex_nonhit_stages{};
        u64 end_seq{}; ///< Scheduler::RecordSeq after the previous draw's records
        Shader::PushData push_data{};
        /// SHADPS4_DRAW_REPEAT_HEAP (dcb note sec. 9): bindings into the per-frame heap may differ
        /// in address only; a repeat re-points them (same Buffer, new offset / push adjust).
        struct HeapSlot {
            u32 info_idx;
            u32 bind_idx;
            const VideoCore::Buffer* buffer;
            u32 alignment; ///< BindBuffers' alignment for this binding (review item 3)
        };
        struct HeapNew {
            vk::DescriptorBufferInfo info;
            u32 adjust;
        };
        std::vector<HeapSlot> heap_base, heap_pending;
        std::vector<AmdGpu::Buffer> heap_cand;
        std::vector<HeapNew> heap_new;
        u64 f_heap{}, heap_repeats{};
        /// SHADPS4_DRAW_REPEAT_HEAP: user clip planes of the base draw (enable bits + plane data);
        /// a repeat with planes enabled reuses the base's stream copy when both are equal.
        std::array<u32, 1 + 4 * AmdGpu::NUM_CLIP_PLANES> clip_base{}, clip_cur{};
        u64 f_clip{}, clip_repeats{};
        u32 clip_info_idx{~0u}; ///< BindBuffers: info index of the clip-plane copy
        u32 clip_base_idx{~0u}; ///< the base's clip-plane info index (DrawRepeatStore)
        bool clip_patch{};      ///< this repeat re-copied the planes (Opus delta review item 1)
        vk::DescriptorBufferInfo clip_new{};
        // Verify scratch.
        std::vector<vk::DescriptorBufferInfo> v_buffers;
        std::vector<vk::DescriptorImageInfo> v_images;
        u64 v_seq{};
        // Statistics (window).
        u64 draws{}, repeats{}, f_ineligible{}, f_pipeline{}, f_key{}, f_rs{}, f_seq{},
            f_epoch{}, f_image{}, f_gen{}, f_other{}, verified{}, mismatches{}, f_gated{};
    };
    DrawRepeatState draw_repeat;

    /// SHADPS4_NATIVE_PASS (notes/native-renderer-2026-10-09.md sec. 3): an open native batch =
    /// one recorded drawIndexedIndirectCount whose commands, per-draw records and count live in
    /// a stream-buffer reservation the back keeps appending to while nothing else records.
    static constexpr u32 NativeMaxDraws = 64;
    static constexpr u32 NativeRecBytes = Shader::NativeRecordDwords * 4;
    static constexpr u32 NativeCmdBytes = 20; // VkDrawIndexedIndirectCommand
    struct NativeRunSpan {
        VAddr lo{}, hi{}, cpu{};
        u64 handle{};
    };
    struct NativeBatch {
        bool open{};
        bool ring{}; ///< SHADPS4_NATIVE_GPU_DIRECT: records read from the opener's job ring
        /// Ring-join (sec. 11): the opener's job tag / ring index and inherited set; a draw joins
        /// a ring batch when its job record sits at ring_idx0 + n.
        u32 ring_tag{}, ring_idx0{};
        u32 ring_push_inh{}, ring_inh_vs{}, ring_inh_ps{};
        u64 ring_inh_slots{};
        // SHADPS4_NATIVE_REC_RUN (stage c1): spans validated once at the open for the job-thread
        // run the opener heads; valid while the dirty epoch and buffer structure stay put.
        bool run_valid{};
        u64 run_epoch{}, run_gen{};
        std::array<NativeRunSpan, Shader::NUM_BUFFERS> run_buf{};
        std::array<NativeRunSpan, 16> run_vb{};
        NativeRunSpan run_ib{};
        bool verify{};
        const GraphicsPipeline* base{};
        u64 seq{};
        u64 tick{};
        u64 ctx_epoch{};
        u64 desc_id{};
        u64 tex_gen{};
        u64 meta_epoch{};
        u64 image_epoch{};
        std::array<u32, AmdGpu::NUM_COLOR_BUFFERS + 1> extents{};
        u32 index_size{};
        u64 ib{};
        u32 vb_n{};
        std::array<u64, 16> vb{};
        std::array<u64, 16> vb_off{};
        std::array<u32, 16> vb_stride{};
        bool vb_rebased{};
        u32 vb_q0{};
        // SHADPS4_NATIVE_COLOUR: a textured batch (vertex pulling, image key).
        bool colour{};
        u64 vbuf{};   ///< the one vertex VkBuffer every stream of the batch sits in (0: none)
        u32 vmask{};  ///< locations with a non-null stream
        u64 smp_gen{};
        u64 border{};
        u32 img_n{};  ///< words of native_img_key
        /// SHADPS4_NATIVE_BINDLESS (sec. 12): the variant samples the global arrays; the opener's
        /// slots (16 images, 16 samplers) and the set copy they live in.
        bool bindless{};
        u32 bl_set{};
        std::array<u32, Shader::NativeBindlessSlots> bl_slots{};
        std::array<u64, Shader::NUM_BUFFERS> buf{};
        u64 guest_mask{};
        u64 stream_off{};
        u8* recs{};
        u8* cmds{};
        u32* count{};
        u32 n{};
        // Verify mode: the join's candidate for the draw the generic path is rendering now.
        bool cand_valid{};
        std::array<u32, Shader::NUM_USER_DATA_REGS> cand_ud{};
        std::array<u64, Shader::NUM_BUFFERS> cand_buf{};
        std::array<u64, Shader::NUM_BUFFERS> cand_off{};
        std::array<u64, Shader::NUM_BUFFERS> cand_size{};
        u64 open_tick{};
        VideoCore::NativeJoinVi cand_vi{};
        u32 cand_vertex_offset{};
        u32 cand_instance_offset{};
        /// SHADPS4_NATIVE_SEGX: the brick-A list of this batch (stream memory) and whether the
        /// batch is an executed segment (its members are in; no joins).
        u32* list{};
        bool sealed{};
    } native_batch;
    boost::container::small_vector<vk::BufferMemoryBarrier2, 16> native_carry;
    /// SHADPS4_NATIVE_COLOUR: the opener's raw T# / S# words (the draw-repeat image key) and,
    /// in verify mode, its resolved image infos.
    std::array<u32, 512> native_img_key{};
    /// SHADPS4_NATIVE_BINDLESS (sec. 12): T# -> slot and S# -> slot caches over the global
    /// arrays (vk_bindless.h). Image entries are valid while the texture cache's structure
    /// generation, the image epoch and the meta-clear epoch stay put AND the image's backing and
    /// state serial are unchanged (no transition due); sampler entries while the sampler
    /// generation stays put. Slots are never rewritten until their set copy rotates out.
    struct BindlessImgEntry {
        const void* desc{};
        std::array<u32, 8> words{};
        u64 gen{};
        u32 slot{};
        VideoCore::ImageId id{};
        u64 uid{}, tex_gen{}, overlap{};
        bool local{};
        const void* backing{};
        u64 serial{};
        vk::ImageView view{};
        vk::ImageLayout layout{};
    };
    struct BindlessSmpEntry {
        std::array<u32, 4> words{};
        u64 gen{};
        u32 slot{};
        size_t lru{};
        vk::Sampler sampler{};
    };
    struct BindlessState {
        u32 cur{};
        std::array<u64, Bindless::NumSets> last_tick{};
        u32 img_next{}, smp_next{};
        struct ViewKey {
            u64 uid;
            const void* backing;
            u64 view;
            u32 layout;
            bool operator==(const ViewKey&) const = default;
        };
        struct ViewKeyHash {
            size_t operator()(const ViewKey& k) const noexcept {
                return std::hash<u64>{}(k.uid * 0x9E3779B97F4A7C15ull ^
                                        std::bit_cast<u64>(k.backing) ^ (k.view * 31) ^ k.layout);
            }
        };
        std::unordered_map<ViewKey, u32, ViewKeyHash> view_slot;
        std::unordered_map<u64, u32> smp_slot;
        std::vector<BindlessImgEntry> img_cache;
        std::vector<BindlessSmpEntry> smp_cache;
        /// l4d: BindlessResolve copies each image's cache entry / each sampler's LRU id here
        /// when set (the material memo's fill).
        BindlessImgEntry* cap_img{};
        size_t* cap_lru{};
        BindlessSmpEntry* cap_smp{}; // q8
        u64 img_gen{1}, smp_gen{1};
        u64 tex_gen{~0ull}, img_epoch{~0ull}, meta_epoch{~0ull}, sampler_gen{~0ull};
    } bindless;
    /// SHADPS4_A2_MAT_MEMO (l4d): one resolved stretch material.
    struct MatMemoImg {
        VideoCore::ImageId id{};
        bool local{};
        u64 uid{}, tex_gen{}, overlap{};
        const void* backing{};
        u64 serial{};
    };
    struct MatMemo {
        const GraphicsPipeline* base{};
        u64 h1{}, h2{};
        u32 kn{};
        u64 img_gen{}, smp_gen{};
        u32 n_img{}, n_smp{};
        u64 touch_gc{~0ull}, touch_age{~0ull}; ///< l4e: the last touch's gc_tick / AgeTick
        u64 used{}; ///< l4e: replacement order within the set
        std::array<u32, Shader::NativeBindlessSlots> slots{};
        std::array<MatMemoImg, Shader::NativeBindlessImgCap> img{};
        std::array<size_t, Shader::NativeBindlessSmpCap> lru{};
    };
    u64 mat_memo_clock{};
    std::vector<MatMemo> mat_memo;
    bool A2MaterialMemoValid(const MatMemo& m);
    /// Back thread: the fragment stage's image / sampler slots of the draw `base` describes now
    /// (32 words: images then samplers). views / samplers (optional): what they hold.
    void Q4NoteMaterial(const VideoCore::ImageId* ids, u32 n); // q4 census
    bool BindlessResolve(const GraphicsPipeline* base, u32* slots, vk::ImageView* views,
                         vk::Sampler* samplers, u32& n_img, u32& n_smp, bool retry = true,
                         const Shader::BoundStage* fs_override = nullptr,
                         VideoCore::ImageId* ids = nullptr,
                         const Shader::BoundStage* vs_override = nullptr);
    bool BindlessRotate();
    void BindlessRingWrite(u32 tag, u32 idx, const u32* slots);
    std::unique_ptr<VideoCore::Buffer> native_dbg; ///< SHADPS4_NATIVE_COLOUR_DIAG 4096
    u32 native_dbg_logged{};
    /// SHADPS4_NATIVE_PT (notes/native-path-2026-10-09.md sec. 18): native variants read Guest
    /// bindings through the GPU page table; on = the flag with GPU_DIRECT 1 / 3; the extra
    /// bindings it pushes (page table, =2 the compare SSBO) and that SSBO's readback copy.
    bool native_pt_on{};
    u32 native_pt_extra{};
    std::unique_ptr<VideoCore::Buffer> native_pt_dbg;
    u32 native_pt_logged{};
    /// SHADPS4_NATIVE_SEG (sec. 20 brick A): records addressed through a per-batch list by
    /// pointer; on = the flag with pt on. The open writes the table's slot addresses and the
    /// list (compare entries share native_pt_dbg: dword 1, entries from dword 2048).
    bool native_seg_on{};
    u32 native_seg_logged{};
    void NativeSegOpen(u64 recs_addr, u64 list_addr, u32* list);
    /// The back's own (legacy) record in pt mode: guest address low 32 bits in the Guest offset
    /// words and the hi bytes (vertex hi bytes 0: unused by the shaders).
    void NativePtRecord(u32* rec, const std::array<u64, Shader::NUM_BUFFERS>& addr) const;
    boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> native_open_images;
    bool NativeImageKey(const GraphicsPipeline* base, std::array<u32, 512>& out, u32& n,
                        const std::array<Shader::BoundStage, Shader::MaxStageTypes>* stages =
                            nullptr) const;
    bool NativeColourGates() const {
        return tex_bind_memo == 1 && !bind_memo_stats && pending_stencil_writebacks.empty();
    }
    /// The context register block at the batch open (compared when the epoch moved).
    std::array<u32, 0x400> native_ctx{};
    struct NativeStats {
        u64 opens{}, joins{}, open_fail{}, verify_draws{}, verify_mismatch{}, epoch_same{};
        // SHADPS4_NATIVE_REC_RUN: runs validated / refused (0 ring 1 record 2 infos 3 buffers 4
        // vertex count 5-7 buffer / vertex / index span), member bindings and vertex-index sets
        // taken from the spans, verify mismatches.
        u64 run_ok{}, run_buf_fast{}, run_buf_slow{}, run_vi_fast{}, run_vi_slow{},
            run_mismatch{};
        std::array<u64, 8> run_fail{};
        std::array<u64, 16> close{};
        u64 tick_moved{};
        // SHADPS4_NATIVE_COLOUR: colour batches; textured indexed draws (any image binding) and
        // how many of them had a published colour variant at the open point.
        u64 col_opens{}, col_joins{}, col_open_fail{}, col_verify{}, col_tex_draws{},
            col_tex_variant{};
        // SHADPS4_NATIVE_TIMING=1: TSC cycles (join ok / join refused / DrawTail of a draw whose
        // pipeline has a native variant / DrawTail of any other draw) and the join's phases.
        u64 c_join_ok{}, n_join_ok{}, c_join_fail{}, n_join_fail{}, c_tail_nat{}, n_tail_nat{},
            c_tail_other{}, n_tail_other{};
        // "other" DrawTail split (2026-10-09, dcb note sec. 8): bit 0 draw-repeat hit, bit 1 the
        // pipeline samples images. Real cycles, one rdtsc pair per draw.
        std::array<u64, 4> c_tail_cls{}, n_tail_cls{};
        bool last_tail_repeat{};
        // Full textured draws by what happened in BindResources (dcb note sec. 9, s368): bit 0 a
        // texture stage took the full path, 1 commands recorded, 2 bytes uploaded, 3 a > 64 MB
        // formatted view bound, 4 clip planes copied.
        u32 last_flags{};
        bool giant_bound{};
        std::array<u64, 128> c_flags{}, n_flags{};
        std::array<u64, 4> c_phase{};
        // SHADPS4_NATIVE_TIMING, colour split: colour joins (and their phases) and the DrawTail
        // of draws that opened a colour batch.
        u64 c_cjoin_ok{}, n_cjoin_ok{}, c_tail_copen{}, n_tail_copen{};
        /// SHADPS4_NATIVE_DIRECT: candidates, direct openers, failures by reason, verify, the
        /// direct openers' DrawTail cycles.
        u64 nd_cand{}, nd_ok{}, nd_vfy_ok{}, nd_vfy_bad{}, c_tail_dopen{}, n_tail_dopen{};
        std::array<u64, 16> nd_fail{};
        std::array<u64, 4> c_nd_ph{}; ///< direct bind: buffers, vertex / index, NativeOpen
        u64 c_bind_copen{}, n_bind_copen{}, last_bind_cyc{}; ///< generic BindResources of colour openers
        std::array<u64, 6> c_no_ph{}; ///< successful NativeOpen phases (colour)
        u64 n_no{};
        bool last_open_direct{};
        /// Bindless census (sec. 11): colour openers by DrawTail event flags (texmiss 1,
        /// recorded 2, uploaded 4, giant 8, clip 16), [1] = right after an image-key close.
        bool close_key{};
        std::array<std::array<u64, 32>, 2> c_cop_fl{}, n_cop_fl{};
        std::array<u64, 4> c_cphase{};
        bool last_open_colour{};
        u64 peek_yes{}, peek_no{}, peek_unknown{};
        // Open refusals by source line (SHADPS4_BACK_CYCLES study, 2026-10-09).
        std::array<std::pair<u32, u64>, 24> fail_lines{};
        // SHADPS4_NATIVE_REC_PREP: prepares (attempted / with members), members prepared, the
        // prepare's cycles, stops by reason, confirms ok / refused by reason, verify compares.
        u64 prep_try{}, prep_runs{}, prep_members{}, c_prep{}, prep_conf{}, prep_conf_col{},
            prep_cmp_ok{}, prep_cmp_bad{}, prep_join_refused{};
        std::array<u64, 16> prep_jref{}; ///< verify: the join refused a prepared member (close why)
        std::array<u64, 8> c_prep_ph{}; ///< head, member compute, spans, final; 4-6 inside compute
        u64 prep_looked{};              ///< members that passed the compute loop
        u64 prep_spans{};               ///< NativeRunSpan calls of the prepares
        /// SHADPS4_NATIVE_GPU_DIRECT=2: 0 no ring / stale, 1 no GPU record, 2 union gone, 3 ok,
        /// 4 mismatch.
        std::array<u64, 5> gv{};
        /// SHADPS4_NATIVE_GPU_DIRECT=1: ring batches / members, refusals by reason, cycles,
        /// span syncs.
        u64 ring_runs{}, ring_members{}, c_ring{}, ring_spans{}, ring_opens_legacy{};
        std::array<u64, 16> ring_fail{};
        std::array<u64, 4> c_ring_ph{};
        u64 ring_span_hits{};
        u64 ring_span_bytes{};
        /// SHADPS4_NATIVE_SPAN_CHECK: cache hits over CPU-dirty pages, by kind: 0 the granule
        /// clock moved meanwhile (a marking in flight), 1 still dirty after a real sync (the
        /// normal path leaves them too), 2 a real sync cleaned them (the skip missed an upload).
        std::array<u64, 3> ring_span_stale{};
        u64 ring_inh_runs{};   ///< ring batches with inherited registers
        /// Ring-join: joins into ring batches, refusals (0 not the next ring entry, 1 job record
        /// gone / no GPU record, 2 another inherited set, 3 inherited values), =2 compare.
        u64 rj_ok{}, rj_cmp_ok{}, rj_cmp_bad{};
        /// SHADPS4_NATIVE_BINDLESS: batches, joins with another material, failures by reason
        /// (0 no FS, 1 null / invalid T#, 2 depth redirect / target / rebind, 3 slots full and
        /// the next copy busy, 4 sampler touch, 5 join: set rotated / recorded / tick), cache
        /// hits / misses, opener verify against the generic writes, rotations.
        u64 bl_open{}, bl_open_fail{}, bl_join{}, bl_join_fail{};
        std::array<u64, 8> bl_fail{};
        u64 bl_img_hit{}, bl_img_miss{}, bl_smp_hit{}, bl_smp_miss{};
        u64 bl_vfy_ok{}, bl_vfy_bad{}, bl_vfy_none{}, bl_rot{};
        u64 c_bl_hit{}, c_bl_miss{}; ///< cycles in image hits / misses
        u64 c_bl_smp{}, c_bl_touch{}, n_bl_res{}; ///< sampler loop / LRU touch cycles, resolves
        /// l4d material memo: hits, misses (no entry), stale entries, verify compares / BAD.
        u64 mm_hit{}, mm_miss{}, mm_stale{}, mm_vfy{}, mm_bad{}, c_mm_hit{}, mm_touch_skip{};
        /// Segment census (sec. 14): pipeline-change closes by bits 1 = fixed state equal (key
        /// minus shaders and vertex formats), 2 = vertex formats equal, 4 = next pipeline has a
        /// native variant, 8 = context registers moved; native -> non-native draw transitions.
        std::array<u64, 16> seg_c2{};
        /// Why a draw took the generic path (DrawTail decision; reason literals), with cycles.
        const char* last_why{};
        std::unordered_map<const char*, std::pair<u64, u64>> why_other;
        u64 seg_to_other{};
        bool seg_last_native{};
        /// Segment continuation (sec. 14): at each batch open vs the previous batch - could one
        /// GPU-driven segment have run across? Break causes: 1 fixed state, 2 dynamic state
        /// values, 4 render targets, 8 a generic draw in between, 16 index type.
        u64 seg_opens{}, seg_cont{};
        std::array<u64, 32> seg_break{};
        u64 seg_prev_fixed{}, seg_prev_dyn{}, seg_prev_rt{};
        u32 seg_prev_index{};
        bool seg_prev_valid{}, seg_generic_since{};
        u64 seg_native_draws{};
        /// D breaks by group: 0 viewport, 1 scissor, 2 depth test/write/compare/bounds, 3 depth
        /// bias, 4 stencil ops, 5 stencil reference, 6 stencil masks, 7 cull / front face,
        /// 8 blend constants, 9 colour write masks, 10 other (restart / discard / feedback / line).
        std::array<u64, 11> seg_dgroup{};
        std::array<u64, 11> seg_prev_g{};
        std::array<u64, 4> rj_fail{};
        std::array<u64, 4> prep_c10{};  ///< verify: close 10 of a prepared member by where
        std::array<u64, 16> prep_stop{};
        std::array<u64, 16> prep_refuse{};
        // SHADPS4_NATIVE_TIMING: whole draw records (Liverpool's timer) by rec_outcome x posted by
        // the front's member path.
        std::array<u64, 12> c_rec{}, n_rec{};
        /// SHADPS4_NATIVE_SEGX=1: executed segments, their members, joins refused by a seal.
        u64 segx_opens{}, segx_members{}, segx_join_refused{};
        u64 segx_guest_gpu{}; ///< Guest slots of segments left to the page table (no span)
    } native_stats;
    void NativePrepare(const GraphicsPipeline* base,
                       const AmdGpu::CpSplit::VertexInputAttrib* attribs,
                       const std::array<u64, Shader::NUM_BUFFERS>& o_addr,
                       const std::array<u64, Shader::NUM_BUFFERS>& o_size,
                       const std::array<u32, Shader::NUM_BUFFERS>& o_off,
                       const std::array<u32, Shader::NUM_BUFFERS>& o_rsize);
    bool NativeCtxSame();
    bool NativeRingJoinOk();
    /// SHADPS4_NATIVE_GPU_DIRECT=1 (sec. 10): ring mode for the batch being opened.
    static u32* NativeGpuRingAlloc(u32 slot);
    static void NativeGpuRingInit(Rasterizer* owner);
    bool NativeRingPrepare(const GraphicsPipeline* base,
                           const AmdGpu::CpSplit::VertexInputAttrib* attribs,
                           const std::array<u64, Shader::NUM_BUFFERS>& o_addr,
                           const std::array<u32, Shader::NUM_BUFFERS>& o_off,
                           const std::array<u32, Shader::NUM_BUFFERS>& o_size, const u32* o_ud);
    void NativeRingWrite();
    /// SHADPS4_NATIVE_SEGX=1 (sec. 23): NativeOpen of member 0 of a Segment record, in place of
    /// NativeRingPrepare: the unions, bases and spans from the record (no member is read); then
    /// the members' commands, list entries and bindless slot words into the batch; then the op
    /// list in place of the batch draw.
    bool NativeSegPrepare(const GraphicsPipeline* base,
                          const AmdGpu::CpSplit::VertexInputAttrib* attribs,
                          const std::array<u64, Shader::NUM_BUFFERS>& o_addr,
                          const std::array<u32, Shader::NUM_BUFFERS>& o_off,
                          const std::array<u32, Shader::NUM_BUFFERS>& o_size, const u32* o_ud);
    void NativeSegWrite(u8* map_ptr, u64 map_off, u64 table_bytes, u64 tables_rel, u8* mat_ptr);
    void NativeSegRecord(const GraphicsPipeline* pipeline, vk::Buffer sb, u64 cmd_off,
                         vk::PipelineLayout bl_layout, vk::DescriptorSet bl_set);
    u32 t_segx_delta{};  ///< (index union lo - the index buffer's CPU base) / index size
    u32 t_segx_ranges{}; ///< DRAW_RANGE ops of the pending segment (table copies: ranges - 1)
    std::array<u64, AmdGpu::CpSplit::SegMaxOps + 3> segx_table_off{}; ///< per range > 0
    void NativeRingVerifyWrite();
    std::array<u32, Shader::NativeTableDwords> native_ring_table{};
    struct NativeSpanEntry {
        u64 handle;
        u32 access;
        VAddr lo, hi;
        u64 epoch, gen; ///< epoch: the granule clock before the validation
        bool gpu_mod;   ///< vertex / index: the range had GPU-written pages at the validation
    };
    std::array<NativeSpanEntry, 16> native_span_cache{};
    u32 native_span_rr{};
    u64 native_ring_handle{};
    void NativeGpuVerify(const u32* ud, const std::array<u32, Shader::NUM_BUFFERS>& off,
                         const std::array<u32, Shader::NUM_BUFFERS>& size,
                         const std::array<u64, Shader::NUM_BUFFERS>& cpu, const u32* voff,
                         const AmdGpu::CpSplit::VertexInputAttrib* attribs, const u32* cmd,
                         u32 vertex_offset, u32 instance_offset);
    /// SHADPS4_NATIVE_REC_PREP: bounds of the buffers the prepares checked ranges against, by
    /// VkBuffer, valid while the buffer structure generation stays put.
    struct NativeBoundsEntry {
        u64 handle;
        VAddr lo, hi;
    };
    std::array<NativeBoundsEntry, 16> native_bcache{};
    u32 native_bcache_n{};
    u32 native_bcache_rr{};
    u64 native_bcache_gen{~u64{0}};
    void NativePrepWrite();
    bool NativeJoin(const SplitDraw& draw, const AmdGpu::CpSplit::VertexInputsPayload* front_vi);
    bool NativeOpen(const GraphicsPipeline* base, const GraphicsPipeline* native, bool is_indexed,
                    u32 index_offset, const AmdGpu::CpSplit::VertexInputsPayload* front_vi,
                    bool colour = false);
    void NativeClose(u32 why);
    void NativeVerifyCompare(const GraphicsPipeline* pipeline, bool is_indexed, u32 index_offset);
    void NativeLog();
    void NativeRunValidate(const GraphicsPipeline* base, Pipeline::BufferBarriers* barriers);
    bool NativeRunVi(const AmdGpu::CpSplit::VertexInputsPayload& vi, u32 index_offset,
                     VideoCore::NativeJoinVi& out) const;
    /// SHADPS4_DRAW_TAG + SHADPS4_NATIVE_TIMING: the back's real cycles grouped by the game's
    /// runs (consecutive draws with one job-thread tag). cls: 0 shadow join, 1 shadow opener,
    /// 2 colour join, 3 colour opener, 4 full untextured, 5 repeat untextured, 6 full textured,
    /// 7 repeat textured.
    void DrawTagNote(const GraphicsPipeline* pipeline, u32 cls, u64 dt);
    void DrawTagFlush();
    struct DrawTagKind {
        u64 runs{}, draws{}, cycles{}, pipe_changes{}, epoch_changes{}, key_changes{},
            texmiss{}, uploads{};
        std::array<u64, 8> cls{};
        std::array<u64, 8> cls_cycles{};
        std::array<u64, 8> hist{}; // draws in runs of 1, 2-3, 4-7, ..., 64-127, 128+
    };
    struct DrawTagStats {
        u32 tag{};
        u32 n{};
        u64 cycles{};
        std::array<u32, 8> cls{};
        std::array<u64, 8> cls_cycles{};
        u32 pipe_changes{}, epoch_changes{}, key_changes{}, texmiss{}, uploads{};
        const GraphicsPipeline* last_pipe{};
        u64 last_epoch{};
        std::array<DrawTagKind, 3> kind{}; // shadow, textured, untextured
        u64 untagged{}, c_untagged{}, total{};
    } draw_tag_stats;
    bool DrawRepeatMatch(const GraphicsPipeline* pipeline, bool rs_hit);
    void DrawRepeatApply(const GraphicsPipeline* pipeline, bool patch_infos = true);
    void DrawRepeatVerify(const GraphicsPipeline* pipeline);
    void DrawRepeatStore(const GraphicsPipeline* pipeline, bool completed);
    /// SHADPS4_PB_CENSUS (diagnostic): the parallel-back census, see PbCensusEnd.
    void PbCensusBegin();
    void PbCensusPoint(u32 point);
    void PbCensusDispatch(const ComputePipeline* pipeline, u32 point);
    void PbCensusEnd(const GraphicsPipeline* pipeline, bool rs_hit, bool repeat_skip,
                     const AmdGpu::CpSplit::VertexInputsPayload* front_vi, bool is_indexed,
                     u32 index_offset);
    /// Null unless SHADPS4_OCCLUSION_QUERIES=1 (vk_occlusion.h). In Shared (one slot ring).
    std::unique_ptr<OcclusionQueries>& occlusion;

    using RenderTargetInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    std::array<RenderTargetInfo, AmdGpu::NUM_COLOR_BUFFERS> cb_descs;
    std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc> db_desc;
    // Pure preparation inputs/results only. Live image identities, uploads and layouts
    // are resolved separately; FindImage may mutate the working descriptor's view range.
    struct PreparedColorTarget {
        AmdGpu::ColorBuffer buffer{};
        AmdGpu::CbDbExtent extent{};
        VideoCore::TextureCache::ImageDesc value{};
        bool valid{};
    };
    struct PreparedDepthTarget {
        AmdGpu::DepthBuffer buffer{};
        AmdGpu::DepthView view{};
        AmdGpu::DepthControl control{};
        AmdGpu::CbDbExtent extent{};
        VAddr htile_address{};
        VideoCore::TextureCache::ImageDesc value{};
        bool valid{};
    };
    std::array<PreparedColorTarget, AmdGpu::NUM_COLOR_BUFFERS> prepared_color_targets{};
    PreparedDepthTarget prepared_depth_target{};
    boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> image_infos;
    boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos;
    boost::container::static_vector<VideoCore::ImageId, Shader::NUM_IMAGES> bound_images;

    // Heap slot per unified binding, filled from the same resolved buffer_infos/image_infos
    // entries the legacy writes point at. Legacy writes are skipped unless verifying.
    std::array<u32, NativeMaxBindingsPerDraw> native_indices{};
    std::array<u32, Shader::NUM_IMAGES> native_image_indices{};
    u32 native_bound_count{};
    bool build_set_writes{true};

    u32 set_write_index{};
    /// p1c tripwire: dispatches executed, and their count when the current BindResources began.
    u64 p1c_dispatch_serial{};
    u64 p1c_bind_serial{};
    void P1cCheckWrites(const Pipeline* pipeline);
    Pipeline::DescriptorWrites set_writes;
    Pipeline::BufferBarriers buffer_barriers;
    Shader::PushData push_data;
    // Per-draw user data of each stage, indexed like Pipeline::stages (by LogicalStage). The back
    // path reads user data only through these (CP front/back split, step 2).
    std::array<Shader::BoundStage, Shader::MaxStageTypes> bound_stages{};

    using BufferBindingInfo = std::tuple<VideoCore::BufferId, AmdGpu::Buffer, u64>;
    boost::container::static_vector<BufferBindingInfo, Shader::NUM_BUFFERS> buffer_bindings;
    using ImageBindingInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    boost::container::static_vector<ImageBindingInfo, Shader::NUM_IMAGES> image_bindings;
    struct DecodedImageDesc {
        AmdGpu::Image sharp{};
        u8 flags{};
        bool valid{};
        VideoCore::TextureCache::ImageDesc value{};
    };
    std::array<std::array<DecodedImageDesc, Shader::NUM_IMAGES>, Shader::MaxStageTypes>
        decoded_image_descs{};
    bool& fault_process_pending; ///< In Shared.
    bool attachment_feedback_loop{};
    bool hud_redirected{}; ///< SHADPS4_HUD_NATIVE: this pass draws into the native HUD image

    // SHADPS4_RENDER_STATE_MEMO=1 (=2 verify), default off, direct draws (DrawTail) only:
    // PrepareRenderState + BeginRendering recompute the same render targets and RenderState for
    // almost every draw of a pass. One entry remembers the last state that the full path built
    // without a clear, a feedback loop or an overlap-resolved target; a draw with the same
    // registers, pipeline targets, texture-cache structure, metadata clear state, images and
    // backings takes the short path (notes/back-split-phase1-2026-09-30.md section 6).
    // cb_descs / db_desc stay as the remembered draw left them while the entry is valid: every
    // other PrepareRenderState call invalidates it first.
    struct RenderStateMemo {
        struct Target {
            VideoCore::ImageId id{};
            u64 uid{};
            const void* backing{};
            u8 cb{}; ///< colour buffer index; DepthSlot for the depth target
        };
        static constexpr u8 DepthSlot = 0xff;
        bool valid{};
        u64 ctx_epoch{};
        u32 mrt_mask{};
        std::array<u8, AmdGpu::NUM_COLOR_BUFFERS> color_samples{};
        u64 struct_gen{};
        u64 meta_epoch{};
        // Every register PrepareRenderState and BeginRendering read (colour buffers below
        // bit_width(mrt_mask) only), plus the extent hints (slot 8 = depth).
        std::array<AmdGpu::ColorBuffer, AmdGpu::NUM_COLOR_BUFFERS> color_buffers{};
        std::array<u32, AmdGpu::NUM_COLOR_BUFFERS + 1> extents{};
        decltype(AmdGpu::Regs::color_target_mask) color_target_mask{};
        decltype(AmdGpu::Regs::color_control) color_control{};
        decltype(AmdGpu::Regs::depth_control) depth_control{};
        decltype(AmdGpu::Regs::depth_buffer) depth_buffer{};
        decltype(AmdGpu::Regs::depth_view) depth_view{};
        decltype(AmdGpu::Regs::depth_htile_data_base) depth_htile_data_base{};
        decltype(AmdGpu::Regs::depth_render_control) depth_render_control{};
        decltype(AmdGpu::Regs::depth_clear) depth_clear{};
        decltype(AmdGpu::Regs::stencil_clear) stencil_clear{};
        boost::container::static_vector<Target, AmdGpu::NUM_COLOR_BUFFERS + 1> targets;
        RenderState state{};
        vk::ImageLayout depth_layout{};
    };
    struct RenderStateMemoStats {
        u64 draws{}, hits{}, begin_hits{}, begin_fallbacks{};
        u64 miss_invalid{}, miss_key{}, miss_regs{}, miss_refused{}, miss_struct{}, miss_meta{},
            miss_image{};
        u64 refused_clear{}, refused_feedback{}, refused_overlap{}, refused_rebind{},
            refused_gen{}, refused_other{};
        u64 epoch_moved{}, verify_prepare{}, verify_begin{}, mismatches{};
    };
    RenderStateMemo rs_memo;
    u64 rs_memo_pb_written{}; ///< SHADPS4_PB_CENSUS: census draw index of the last rs_memo store
    RenderStateMemoStats rs_stats;
    /// SHADPS4_MTB_P1: the worker instance (primary only). Declared after owned_shared, so it
    /// is destroyed first.
    std::unique_ptr<Rasterizer> mtb_worker;
    /// Depth layout BeginRendering transitioned the depth target to (valid when one is bound).
    vk::ImageLayout rs_last_depth_layout{};
    [[nodiscard]] bool RenderStateMemoRegsEqual(u32 num_cbs) const;
    /// True when the entry serves this draw. With `apply` it also does what PrepareRenderState
    /// leaves behind (target flags, bound_images, access touches).
    [[nodiscard]] bool RenderStateMemoMatch(const GraphicsPipeline* pipeline, bool apply);
    /// After BindResources: the entry can still stand in for BeginRendering.
    [[nodiscard]] bool RenderStateMemoBeginOk();
    void RenderStateMemoBegin(RenderState& state);
    void RenderStateMemoStore(const GraphicsPipeline* pipeline, const RenderState& state,
                              size_t bound_begin, u64 struct_gen_at_entry);
    void RenderStateMemoLog();

};

} // namespace Vulkan
