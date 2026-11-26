// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include "common/fault_monitor.h"
#include "common/logging/log.h"
#include "video_core/amdgpu/mtb_conflict.h"
#include "common/types.h"
#include "core/memory.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {
/// SHADPS4_ASC_STREAM: the command stream being recorded (0 main, 1 ASC); CP thread only.
/// Buffer barrier state is kept per stream (Buffer::SelectStreamState).
inline u32 g_record_stream = 0;
/// SHADPS4_ASC_I1 (CP thread): the ASC semaphore's known value, refreshed at each main flush,
/// and the newest ASC value the open main batch depends on (precise replacement for I1).
inline u64 g_asc_known = 0;
inline u64 g_main_wait_asc = 0;
inline u64 g_asc_dep_slow = 0;
/// SHADPS4_ASC_I1 completeness falsifier: the open main tick (set at each flush) and misses.
inline bool g_asc_deps_active = false;
inline u64 g_asc_open_tick = 0;
inline u64 g_asc_hook_miss = 0;
inline u64 g_asc_whole_buffers = 0;
/// Counters: raises of g_main_wait_asc by cause, and per Buffer base (CP thread only).
inline u64 g_asc_raise_whole = 0;
inline u64 g_asc_raise_entry = 0;
inline std::unordered_map<u64, u64>* g_asc_raise_by_base = nullptr;
inline void CountAscRaise(bool whole, VAddr base) {
    (whole ? g_asc_raise_whole : g_asc_raise_entry)++;
    if (!g_asc_raise_by_base) {
        g_asc_raise_by_base = new std::unordered_map<u64, u64>();
    }
    ++(*g_asc_raise_by_base)[base];
}
} // namespace VideoCore

namespace Vulkan {
class Instance;
class Scheduler;
} // namespace Vulkan

VK_DEFINE_HANDLE(VmaAllocation)
VK_DEFINE_HANDLE(VmaAllocator)

struct VmaAllocationInfo;

namespace VideoCore {

/// Hints and requirements for the backing memory type of a commit
enum class MemoryUsage {
    DeviceLocal, ///< Requests device local buffer.
    Upload,      ///< Requires a host visible memory type optimized for CPU to GPU uploads
    Download,    ///< Requires a host visible memory type optimized for GPU to CPU readbacks
    Stream,      ///< Requests device local host visible buffer, falling back host memory.
};

constexpr vk::BufferUsageFlags ReadFlags =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eUniformBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer;

constexpr vk::BufferUsageFlags AllFlags =
    ReadFlags | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;

struct UniqueBuffer {
    explicit UniqueBuffer(vk::Device device, VmaAllocator allocator);
    ~UniqueBuffer();

    UniqueBuffer(const UniqueBuffer&) = delete;
    UniqueBuffer& operator=(const UniqueBuffer&) = delete;

    UniqueBuffer(UniqueBuffer&& other) noexcept
        : device{std::exchange(other.device, VK_NULL_HANDLE)},
          allocator{std::exchange(other.allocator, VK_NULL_HANDLE)},
          allocation{std::exchange(other.allocation, VK_NULL_HANDLE)},
          buffer{std::exchange(other.buffer, VK_NULL_HANDLE)},
          bda_addr{std::exchange(other.bda_addr, 0)} {}
    UniqueBuffer& operator=(UniqueBuffer&& other) noexcept;

    void Create(const vk::BufferCreateInfo& image_ci, MemoryUsage usage,
                VmaAllocationInfo* out_alloc_info);

    operator vk::Buffer() const {
        return buffer;
    }

    vk::Device device;
    VmaAllocator allocator;
    VmaAllocation allocation;
    vk::Buffer buffer{};
    vk::DeviceAddress bda_addr = 0;
};

/// 1.1.1 (GitHub #5): allocates what a Buffer of `size` bytes with `flags` would (same create info
/// and memory request, the over-budget retry included) and frees it again at once. true = it fit;
/// `heap` = the memory heap it landed in. A refusal logs nothing: the caller steps down.
/// `require_video`: as a buffer made under RequireVideoMemory would be.
bool ProbeBufferAllocation(const Vulkan::Instance& instance, MemoryUsage usage,
                           vk::BufferUsageFlags flags, u64 size, u32& heap,
                           bool require_video = false);

/// 1.1.1 (GitHub #30: a 6 GB card's 2816 MB mesh buffer landed in system RAM): while one is alive
/// on this thread, a DeviceLocal buffer must be in video memory and skips VMA's 80%-of-heap
/// estimate (the caller checked the driver's own budget). If the driver refuses, the normal
/// allocation runs.
class RequireVideoMemory {
public:
    RequireVideoMemory();
    ~RequireVideoMemory();
    RequireVideoMemory(const RequireVideoMemory&) = delete;
    RequireVideoMemory& operator=(const RequireVideoMemory&) = delete;
};

/// The opposite of RequireVideoMemory: while one is alive on this thread, a DeviceLocal buffer is
/// allocated in host memory (VMA's AUTO_PREFER_HOST). Used for BIG_GROW's mesh buffer when the
/// driver's budget has no room for it in video memory: reads over PCIe are slow, an out-of-memory
/// assert ends the game (RTX 3050 Laptop 4 GB, 2.7 GB mesh buffer, 2026-10-27).
class PreferHostMemory {
public:
    PreferHostMemory();
    ~PreferHostMemory();
    PreferHostMemory(const PreferHostMemory&) = delete;
    PreferHostMemory& operator=(const PreferHostMemory&) = delete;
};

class Buffer {
public:
    explicit Buffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                    MemoryUsage usage, VAddr cpu_addr_, vk::BufferUsageFlags flags,
                    u64 size_bytes_);

    /// SHADPS4_HOST_MIRROR: wraps a VkBuffer bound to imported host memory
    /// (VK_EXT_external_memory_host). The memory is not VMA's and lives until process exit.
    explicit Buffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                    vk::Buffer imported, VAddr cpu_addr_, u64 size_bytes_);

    Buffer& operator=(const Buffer&) = delete;
    Buffer(const Buffer&) = delete;

    Buffer& operator=(Buffer&&) = default;
    Buffer(Buffer&&) = default;

    void IncreaseStreamScore(int score) noexcept {
        stream_score += score;
    }

    [[nodiscard]] int StreamScore() const noexcept {
        return stream_score;
    }

    [[nodiscard]] bool IsInBounds(VAddr addr, u64 size) const noexcept {
        return addr >= cpu_addr && addr + size <= cpu_addr + SizeBytes();
    }

    [[nodiscard]] VAddr CpuAddr() const noexcept {
        return cpu_addr;
    }

    [[nodiscard]] u64 Offset(VAddr other_cpu_addr) const noexcept {
        return other_cpu_addr - cpu_addr;
    }

    size_t SizeBytes() const {
        return size_bytes;
    }

    void SetLRUId(u64 id) noexcept {
        lru_id = id;
    }

    u64 LRUId() const noexcept {
        return lru_id;
    }

    vk::Buffer Handle() const noexcept {
        return buffer;
    }

    vk::DeviceAddress BufferDeviceAddress() const noexcept {
        ASSERT_MSG(buffer.bda_addr != 0, "Can't get BDA from a non BDA buffer");
        return buffer.bda_addr;
    }

    /// SHADPS4_SMART_BUFFER_BARRIERS=1: emit a barrier only for a real hazard. A read after a
    /// read needs none; a read after a write needs one only until that write has been made
    /// visible to the reading stage and access; a write waits for every earlier read and write.
    /// Measured 2026-09-24: ~490 buffer barrier calls per frame (each also ends the render pass)
    /// under the default "any change of access or stage" rule.
    /// Read once at startup (a function-local static would add a guard check per binding).
    static inline const bool smart_barriers_enabled = [] {
        const char* value = std::getenv("SHADPS4_SMART_BUFFER_BARRIERS");
        return value != nullptr && value[0] == '1';
    }();
    static bool SmartBarriers() {
        return smart_barriers_enabled;
    }

    /// SHADPS4_MTB_CONFLICT: the hazard state GetBarrier may change.
    auto HazardTuple() const {
        return std::tuple{hz_write_stages,   hz_write_access, hz_read_stages, hz_visible_stages,
                          hz_visible_access, hz_run,          hz_run_wrote,   access_mask,
                          stage};
    }

    /// `run` (SHADPS4_GUEST_DISPATCH_ORDER, smart barriers only): nonzero for a dispatch binding;
    /// equal values mean consecutive dispatches of one guest queue with no guest sync packet and
    /// nothing else recorded in between. GCN overlaps such dispatches (the game orders dependent
    /// ones with RELEASE_MEM / WAIT_REG_MEM / ACQUIRE_MEM / events), so no barrier among them.
    std::optional<vk::BufferMemoryBarrier2> GetBarrier(vk::AccessFlags2 dst_acess_mask,
                                                       vk::PipelineStageFlagBits2 dst_stage,
                                                       u32 offset = 0, u64 run = 0) {
        if (g_asc_deps_active && g_record_stream == 0 && asc_log &&
            asc_log->touch > g_asc_known && asc_checked_tick != g_asc_open_tick) [[unlikely]] {
            // A main access to a Buffer with unretired ASC work that no dependency check saw in
            // this batch (design sec. 3.5): the batch may not wait the ASC segment it needs.
            const u64 n = ++g_asc_hook_miss;
            if ((n & (n - 1)) == 0) {
                LOG_CRITICAL(Render_Vulkan, "AscHookMiss: Buffer {:#x}+{:#x} (n={})", cpu_addr,
                             size_bytes, n);
            }
        }
        SelectStreamState();
        if (AmdGpu::MtbConflict::g_enabled) [[unlikely]] {
            // SHADPS4_MTB_CONFLICT (diagnostic): a draw's first touch, and any hazard-state change.
            AmdGpu::MtbConflict::Touch(cf, AmdGpu::MtbConflict::Buffer);
            const auto before = HazardTuple();
            auto result = GetBarrierImpl(dst_acess_mask, dst_stage, offset, run);
            if (HazardTuple() != before) {
                AmdGpu::MtbConflict::Change(cf, AmdGpu::MtbConflict::Buffer);
            }
            return result;
        }
        return GetBarrierImpl(dst_acess_mask, dst_stage, offset, run);
    }

    std::optional<vk::BufferMemoryBarrier2> GetBarrierImpl(vk::AccessFlags2 dst_acess_mask,
                                                           vk::PipelineStageFlagBits2 dst_stage,
                                                           u32 offset, u64 run) {
        if (SmartBarriers()) {
            return GetHazardBarrier(dst_acess_mask, vk::PipelineStageFlags2{dst_stage}, offset,
                                    run);
        }
        if (dst_acess_mask == access_mask && stage == dst_stage) {
            return {};
        }

        DEBUG_ASSERT(offset < size_bytes);

        const auto barrier = vk::BufferMemoryBarrier2{
            .srcStageMask = stage,
            .srcAccessMask = access_mask,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_acess_mask,
            .buffer = buffer.buffer,
            .offset = offset,
            .size = size_bytes - offset,
        };
        access_mask = dst_acess_mask;
        stage = dst_stage;
        return barrier;
    }

    void Fill(u64 offset, u32 num_bytes, u32 value);

    /// SHADPS4_MTB_CONFLICT (diagnostic): batch of the last hazard-state change / draw touch.
    AmdGpu::MtbConflict::Stamp cf{};

private:
    static constexpr vk::AccessFlags2 WriteAccessBits =
        vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderStorageWrite |
        vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eMemoryWrite |
        vk::AccessFlagBits2::eHostWrite | vk::AccessFlagBits2::eColorAttachmentWrite |
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
        vk::AccessFlagBits2::eTransformFeedbackWriteEXT |
        vk::AccessFlagBits2::eTransformFeedbackCounterWriteEXT;

    std::optional<vk::BufferMemoryBarrier2> GetHazardBarrier(vk::AccessFlags2 dst_access,
                                                             vk::PipelineStageFlags2 dst_stages,
                                                             u32 offset, u64 run) {
        if (run != 0) {
            // Guest dispatch run: the run's accesses are never hazards against each other (the
            // game ordered nothing between them). Each barrier emitted in a run covers both
            // shader access kinds, so the run's later accesses of this buffer are visible too.
            const bool write = bool(dst_access & WriteAccessBits);
            if (run == hz_run) {
                if (!write) {
                    hz_read_stages |= dst_stages;
                    return {};
                }
                if (hz_run_wrote) {
                    hz_write_stages |= dst_stages;
                    hz_write_access |= dst_access & WriteAccessBits;
                    return {};
                }
                // First write of the run: order it once against reads (and unseen writes) that
                // may precede the run; reads made by the run itself are included conservatively.
            }
            hz_run = run;
            hz_run_wrote = write;
            dst_access |= vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
            if (!write) {
                // A read opening the run: the normal read rule, widened access.
                const bool covered = (hz_visible_stages & dst_stages) == dst_stages &&
                                     (hz_visible_access & dst_access) == dst_access;
                hz_read_stages |= dst_stages;
                if (!hz_write_access || covered) {
                    return {};
                }
                const auto barrier = vk::BufferMemoryBarrier2{
                    .srcStageMask = hz_write_stages,
                    .srcAccessMask = hz_write_access,
                    .dstStageMask = dst_stages,
                    .dstAccessMask = dst_access,
                    .buffer = buffer.buffer,
                    .offset = offset,
                    .size = size_bytes - offset,
                };
                hz_visible_stages |= dst_stages;
                hz_visible_access |= dst_access;
                return barrier;
            }
            // A write (opening the run or its first write): always ordered against any earlier
            // read or write (no write-repeat shortcut: a guest sync may precede this run).
            std::optional<vk::BufferMemoryBarrier2> barrier{};
            if (hz_write_access || hz_read_stages) {
                vk::PipelineStageFlags2 src_stages = hz_write_stages | hz_read_stages;
                if (!src_stages) {
                    src_stages = vk::PipelineStageFlagBits2::eAllCommands;
                }
                barrier = vk::BufferMemoryBarrier2{
                    .srcStageMask = src_stages,
                    .srcAccessMask = hz_write_access,
                    .dstStageMask = dst_stages,
                    .dstAccessMask = dst_access,
                    .buffer = buffer.buffer,
                    .offset = offset,
                    .size = size_bytes - offset,
                };
            }
            hz_run_wrote = true;
            hz_write_stages = dst_stages;
            hz_write_access = dst_access & WriteAccessBits;
            hz_read_stages = {};
            hz_visible_stages = {};
            hz_visible_access = {};
            return barrier;
        }
        hz_run = 0;
        const bool is_write = bool(dst_access & WriteAccessBits);
        if (!is_write) {
            const bool covered = (hz_visible_stages & dst_stages) == dst_stages &&
                                 (hz_visible_access & dst_access) == dst_access;
            hz_read_stages |= dst_stages;
            if (!hz_write_access || covered) {
                return {};
            }
            const auto barrier = vk::BufferMemoryBarrier2{
                .srcStageMask = hz_write_stages,
                .srcAccessMask = hz_write_access,
                .dstStageMask = dst_stages,
                .dstAccessMask = dst_access,
                .buffer = buffer.buffer,
                .offset = offset,
                .size = size_bytes - offset,
            };
            hz_visible_stages |= dst_stages;
            hz_visible_access |= dst_access;
            return barrier;
        }
        // Same as the default rule: a repeat of the same write with no read in between gets no
        // barrier (consecutive dispatches writing one buffer, as before this option).
        if (!hz_read_stages && hz_write_stages == dst_stages &&
            hz_write_access == (dst_access & WriteAccessBits)) {
            return {};
        }
        vk::PipelineStageFlags2 src_stages = hz_write_stages | hz_read_stages;
        if (!src_stages) {
            src_stages = vk::PipelineStageFlagBits2::eAllCommands;
        }
        const auto barrier = vk::BufferMemoryBarrier2{
            .srcStageMask = src_stages,
            .srcAccessMask = hz_write_access,
            .dstStageMask = dst_stages,
            .dstAccessMask = dst_access,
            .buffer = buffer.buffer,
            .offset = offset,
            .size = size_bytes - offset,
        };
        hz_write_stages = dst_stages;
        hz_write_access = dst_access & WriteAccessBits;
        hz_read_stages = {};
        hz_visible_stages = {};
        hz_visible_access = {};
        return barrier;
    }

    /// SHADPS4_ASC_STREAM: barriers order work inside one command stream only; the other stream's
    /// accesses are ordered by semaphores (I1/I2). The fields below hold the state of
    /// `state_stream`; the other stream's state is parked here and swapped in on demand.
    struct ParkedBarrierState {
        vk::PipelineStageFlags2 hz_write_stages{vk::PipelineStageFlagBits2::eAllCommands};
        vk::AccessFlags2 hz_write_access{vk::AccessFlagBits2::eMemoryWrite};
        vk::PipelineStageFlags2 hz_read_stages{};
        vk::PipelineStageFlags2 hz_visible_stages{};
        vk::AccessFlags2 hz_visible_access{};
        u64 hz_run{};
        bool hz_run_wrote{};
        vk::Flags<vk::AccessFlagBits2> access_mask{
            vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite |
            vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite};
        vk::PipelineStageFlagBits2 stage{vk::PipelineStageFlagBits2::eAllCommands};
    };
    ParkedBarrierState parked_state{};
    u32 state_stream = 0;
    void SelectStreamState() {
        if (state_stream == g_record_stream) [[likely]] {
            return;
        }
        std::swap(hz_write_stages, parked_state.hz_write_stages);
        std::swap(hz_write_access, parked_state.hz_write_access);
        std::swap(hz_read_stages, parked_state.hz_read_stages);
        std::swap(hz_visible_stages, parked_state.hz_visible_stages);
        std::swap(hz_visible_access, parked_state.hz_visible_access);
        std::swap(hz_run, parked_state.hz_run);
        std::swap(hz_run_wrote, parked_state.hz_run_wrote);
        std::swap(access_mask, parked_state.access_mask);
        std::swap(stage, parked_state.stage);
        state_stream = g_record_stream;
    }

    // Hazard state for SmartBarriers(); starts as "unknown write by anything".
    vk::PipelineStageFlags2 hz_write_stages{vk::PipelineStageFlagBits2::eAllCommands};
    vk::AccessFlags2 hz_write_access{vk::AccessFlagBits2::eMemoryWrite};
    vk::PipelineStageFlags2 hz_read_stages{};
    vk::PipelineStageFlags2 hz_visible_stages{};
    vk::AccessFlags2 hz_visible_access{};
    u64 hz_run{}; ///< guest dispatch run of the last access (0 = not a dispatch run)
    bool hz_run_wrote{}; ///< that run has written this buffer

public:

public:
    VAddr cpu_addr = 0;
    bool is_picked{};
    bool is_coherent{};
    bool is_deleted{};
    /// q22: a lane-mode record synced (bound) this buffer; CoherenceSync uploads it on the lane (LANE_UPLOADS).
    bool lane_bound{};
    /// SHADPS4_ASC_SPAN_STATS: newest main-timeline tick that accessed / wrote this buffer from
    /// a graphics-queue binding (diagnostic).
    u64 diag_main_access_tick{};
    /// SHADPS4_READ_BIND_MEMO=3|4: unique per created cache buffer (0 for special buffers), so a
    /// memo entry cannot match a new buffer that reused a slot.
    u64 memo_serial{};
    /// SHADPS4_RMEMO_CENSUS: drained dirty-log entries that touched this Buffer's pages (back only).
    u64 census_dirty_seq{};
    u64 diag_main_write_tick{};
    /// SHADPS4_ASC_STREAM: newest main-stream tick that accessed / wrote this buffer (bindings,
    /// uploads, copies, fault copies). An ASC segment binding it waits for that main batch; if it
    /// is the batch still being recorded the dispatch stays on the main stream.
    mutable u64 main_access_tick{};
    mutable u64 prev_access_tick{};
    mutable u64 prev_write_tick{};
    mutable u64 main_write_tick{};
    void NoteMainAccess(u64 tick, bool write) const {
        if (g_record_stream != 0) {
            return;
        }
        if (tick != main_access_tick) {
            // SHADPS4_ASC_RANGE: the newest main batch before `tick` that touched this Buffer,
            // the wait target of an ASC segment that has no range conflict with the open batch.
            prev_access_tick = main_access_tick;
        }
        main_access_tick = tick;
        if (write) {
            if (tick != main_write_tick) {
                prev_write_tick = main_write_tick; // ASC reads wait flushed main writes only
            }
            main_write_tick = tick;
        }
    }

    /// SHADPS4_ASC_I1 (notes/asc-i1-relax-design-2026-09-28.md sec. 3): what the ASC stream
    /// accessed, keyed by ASC value, so a later main access can wait exactly the segment it
    /// depends on (RAW / WAR / WAW) instead of I1's blanket wait. CP thread only.
    struct AscLog {
        struct Entry {
            VAddr lo;
            VAddr hi;
            bool write;
            u64 value;
        };
        std::vector<Entry> entries;
        u64 touch = 0; ///< newest ASC value that accessed any part
        u64 write = 0; ///< newest ASC value that wrote any part
        bool whole = false;
        /// Paged mode (after the entry list overflows): newest ASC read / write value per 4 KiB
        /// page, with 256 KiB block maxima (the heap is read in ~1.8 M small slices per run).
        bool paged = false;
        std::vector<u64> page_rd, page_wr, blk_rd, blk_wr;
    };
    static constexpr u64 AscPageBits = 12;
    static constexpr u64 AscBlockBits = 18;
    void AscPagedNote(AscLog& log, VAddr lo, u64 size, bool write, u64 value) const {
        const VAddr clo = std::max(lo, cpu_addr);
        const VAddr chi = std::min<VAddr>(lo + size, cpu_addr + size_bytes);
        if (size == 0 || chi <= clo) {
            return;
        }
        const u64 first = (clo - cpu_addr) >> AscPageBits;
        const u64 last = std::min<u64>((chi - 1 - cpu_addr) >> AscPageBits,
                                       log.page_rd.size() - 1);
        auto& pages = write ? log.page_wr : log.page_rd;
        auto& blocks = write ? log.blk_wr : log.blk_rd;
        for (u64 pg = first; pg <= last; ++pg) {
            pages[pg] = std::max(pages[pg], value);
            auto& b = blocks[pg >> (AscBlockBits - AscPageBits)];
            b = std::max(b, value);
        }
    }
    u64 AscPagedMax(const AscLog& log, VAddr lo, u64 size, bool include_reads) const {
        const u64 first = (std::max(lo, cpu_addr) - cpu_addr) >> AscPageBits;
        const u64 end_addr = std::min<u64>(lo + size, cpu_addr + size_bytes);
        if (end_addr <= cpu_addr + (first << AscPageBits)) {
            return 0;
        }
        const u64 last = std::min<u64>((end_addr - 1 - cpu_addr) >> AscPageBits,
                                       log.page_rd.size() - 1);
        constexpr u64 per_block = 1ULL << (AscBlockBits - AscPageBits);
        u64 need = 0;
        u64 pg = first;
        while (pg <= last) {
            const u64 blk = pg / per_block;
            const u64 blk_first = blk * per_block;
            const u64 blk_last = blk_first + per_block - 1;
            if (pg == blk_first && blk_last <= last) {
                need = std::max(need, log.blk_wr[blk]);
                if (include_reads) {
                    need = std::max(need, log.blk_rd[blk]);
                }
                pg = blk_last + 1;
                continue;
            }
            need = std::max(need, log.page_wr[pg]);
            if (include_reads) {
                need = std::max(need, log.page_rd[pg]);
            }
            ++pg;
        }
        return need;
    }
    mutable std::unique_ptr<AscLog> asc_log;
    void NoteAscAccess(VAddr lo, u64 size, bool write, u64 value) const {
        if (!asc_log) {
            asc_log = std::make_unique<AscLog>();
        }
        auto& log = *asc_log;
        if (log.touch <= g_asc_known) {
            // Everything logged has retired: start clean (a whole fallback does not stick).
            log.entries.clear();
            g_asc_whole_buffers -= log.whole;
            log.whole = false;
        }
        log.touch = std::max(log.touch, value);
        if (write) {
            log.write = std::max(log.write, value);
        }
        if (log.paged) {
            AscPagedNote(log, lo, size, write, value);
            return;
        }
        if (log.whole) {
            return;
        }
        // Drop retired entries, then merge an identical range (raise its value).
        const u64 known = g_asc_known;
        std::erase_if(log.entries, [known](const AscLog::Entry& e) { return e.value <= known; });
        const VAddr hi = lo + size;
        for (auto& e : log.entries) {
            if (e.lo == lo && e.hi == hi && e.write == write) {
                e.value = std::max(e.value, value);
                return;
            }
        }
        if (log.entries.size() >= 256) {
            // Switch to per-page values (stays paged for the Buffer's life; values <= known are
            // ignored by the check, so no pruning is needed).
            const u64 pages = (size_bytes + (1ULL << AscPageBits) - 1) >> AscPageBits;
            const u64 blocks = (pages + (1ULL << (AscBlockBits - AscPageBits)) - 1) >>
                               (AscBlockBits - AscPageBits);
            log.page_rd.assign(pages, 0);
            log.page_wr.assign(pages, 0);
            log.blk_rd.assign(blocks, 0);
            log.blk_wr.assign(blocks, 0);
            log.paged = true;
            ++g_asc_whole_buffers;
            for (const auto& e : log.entries) {
                AscPagedNote(log, e.lo, e.hi - e.lo, e.write, e.value);
            }
            log.entries.clear();
            AscPagedNote(log, lo, size, write, value);
            return;
        }
        log.entries.push_back({lo, hi, write, value});
    }
    /// Main access check: raises g_main_wait_asc to the newest unretired ASC value this access
    /// depends on (RAW: ASC wrote; WAR: main writes what ASC read; WAW).
    mutable u64 asc_checked_tick = ~0ULL;
    void CheckMainVsAsc(VAddr lo, u64 size, bool write) const {
        asc_checked_tick = g_asc_open_tick;
        if (!asc_log || asc_log->touch <= g_asc_known) {
            return;
        }
        if (!write && asc_log->write <= g_asc_known) {
            return;
        }
        ++g_asc_dep_slow;
        const auto& log = *asc_log;
        if (log.paged) {
            const u64 need = AscPagedMax(log, lo, size, write);
            if (need > g_asc_known && need > g_main_wait_asc) {
                g_main_wait_asc = need;
                CountAscRaise(false, cpu_addr);
            }
            return;
        }
        if (log.whole) {
            const u64 need = write ? log.touch : log.write;
            if (need > g_main_wait_asc) {
                g_main_wait_asc = need;
                CountAscRaise(true, cpu_addr);
            }
            return;
        }
        const VAddr hi = lo + size;
        for (const auto& e : log.entries) {
            if (e.value > g_asc_known && e.lo < hi && lo < e.hi && (e.write || write) &&
                e.value > g_main_wait_asc) {
                g_main_wait_asc = e.value;
                CountAscRaise(false, cpu_addr);
            }
        }
    }

    /// SHADPS4_ASC_V2_CENSUS=2 (shadow range tracker, notes/asc-stream-v2-design-2026-09-28.md
    /// step 2): exact guest ranges the open main batch read or wrote, kept only for Buffers an
    /// ASC candidate has bound (asc_bound). Counts only; no decision uses it.
    struct MainRangeLog {
        struct Entry {
            VAddr lo;
            VAddr hi;
            bool write;
            u8 kind;
            u64 id;
        };
        u64 tick = ~0ULL;
        bool whole = false;
        u8 whole_kind = 0;
        std::vector<Entry> entries;
    };
    mutable std::unique_ptr<MainRangeLog> range_log;
    mutable std::unique_ptr<MainRangeLog> prev_range_log;
    /// Census: does the logged batch `tick` (previous log) touch [lo, lo+size) in a conflicting
    /// way? -1 = no log for that tick, 0 = no overlap, 1 = overlap (or whole).
    int PrevLogConflict(u64 tick, VAddr lo, u64 size, bool written) const {
        const MainRangeLog* log = prev_range_log.get();
        if (!log || log->tick != tick) {
            return -1;
        }
        if (log->whole) {
            return 1;
        }
        const VAddr hi = lo + size;
        for (const auto& e : log->entries) {
            if (e.lo < hi && lo < e.hi && (e.write || written)) {
                return 1;
            }
        }
        return 0;
    }
    mutable bool asc_bound = false;
    MainRangeLog* OpenRangeLog(u64 tick) const {
        if (g_record_stream != 0 || !asc_bound) {
            return nullptr;
        }
        if (!range_log) {
            range_log = std::make_unique<MainRangeLog>();
        }
        if (range_log->tick != tick) {
            // Keep the previous batch's log (census: would a flushed-batch wait survive range
            // precision?).
            if (!prev_range_log) {
                prev_range_log = std::make_unique<MainRangeLog>();
            }
            std::swap(prev_range_log, range_log);
            range_log->tick = tick;
            range_log->whole = false;
            range_log->entries.clear();
        }
        return range_log.get();
    }
    void NoteMainRange(u64 tick, bool write, VAddr lo, u64 size, u8 kind, u64 id) const {
        if (g_record_stream == 0) {
            CheckMainVsAsc(lo, size, write);
        }
        MainRangeLog* log = OpenRangeLog(tick);
        if (!log || log->whole) {
            return;
        }
        const VAddr hi = lo + size;
        // Repeated bindings of the same range in one batch are the common case: skip exact
        // repeats among the last few entries, and extend the last entry when contiguous.
        const size_t n = log->entries.size();
        for (size_t k = n > 8 ? n - 8 : 0; k < n; ++k) {
            const auto& e = log->entries[k];
            if (e.lo <= lo && hi <= e.hi && (e.write || !write)) {
                return;
            }
        }
        if (n != 0) {
            auto& last = log->entries.back();
            if (last.write == write && last.kind == kind && last.hi == lo) {
                last.hi = hi;
                return;
            }
        }
        if (n >= 1024) {
            log->whole = true;
            log->whole_kind = 100 + kind; // overflow, not a whole-Buffer access
            return;
        }
        log->entries.push_back({lo, hi, write, kind, id});
    }
    void NoteMainWhole(u64 tick, u8 kind) const {
        if (g_record_stream == 0) {
            CheckMainVsAsc(cpu_addr, size_bytes, true);
        }
        if (MainRangeLog* log = OpenRangeLog(tick); log && !log->whole) {
            log->whole = true;
            log->whole_kind = kind;
        }
    }
    int stream_score = 0;
    size_t size_bytes = 0;
    u64 lru_id = 0;
    std::span<u8> mapped_data;
    const Vulkan::Instance* instance;
    Vulkan::Scheduler* scheduler;
    MemoryUsage usage;
    UniqueBuffer buffer;
    vk::Flags<vk::AccessFlagBits2> access_mask{
        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite |
        vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite};
    vk::PipelineStageFlagBits2 stage{vk::PipelineStageFlagBits2::eAllCommands};
};

class StreamBuffer : public Buffer {
public:
    explicit StreamBuffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                          MemoryUsage usage, u64 size_bytes_,
                          vk::BufferUsageFlags extra_flags = {});

    bool IsCoherent() const noexcept {
        return is_coherent;
    }

    /// Reserves a region of memory from the stream buffer.
    std::pair<u8*, u64> Map(u64 size, u64 alignment = 0, bool allow_wait = true);

    /// Ensures that reserved bytes of memory are available to the GPU.
    void Commit();

    /// Maps and commits a memory region with user provided data
    u64 Copy(auto src, size_t size, size_t alignment = 0) {
        // Measured 2026-08-09: 6.6 calls per draw on the ObtainBuffer early-exit path, 2.29
        // us/draw, of which 1.98 us is CopySparseMemory - i.e. ~43 MB per frame actually copied
        // out of guest memory into the stream buffer, mostly cold-miss and write-combine traffic
        // rather than lookup overhead. Cutting it needs the copy to not happen (a memo keyed on
        // address+size, invalidated by the CPU-modified tracking), not a faster copy. Timers were
        // removed after the split was recorded; they cost 0.57 us/draw on their own.
        const auto [data, offset] = Map(size, alignment);
        auto* memory = Core::Memory::Instance();
        const VAddr src_vaddr = reinterpret_cast<const VAddr>(src);
        if (memory->IsValidMapping(src_vaddr)) {
            memory->CopySparseMemory(src_vaddr, data, size);
        } else {
            std::memcpy(data, reinterpret_cast<const void*>(src), size);
        }
        Commit();
        return offset;
    }

private:
    struct Watch {
        u64 tick{};
        u64 upper_bound{};
    };

    /// Increases the amount of watches available.
    void ReserveWatches(std::vector<Watch>& watches, std::size_t grow_size);

    /// Waits pending watches until requested upper bound.
    bool WaitPendingOperations(u64 requested_upper_bound, bool allow_wait);

private:
    u64 offset{};
    u64 mapped_size{};
    std::vector<Watch> current_watches;
    std::size_t current_watch_cursor{};
    std::optional<size_t> invalidation_mark;
    std::vector<Watch> previous_watches;
    std::size_t wait_cursor{};
    u64 wait_bound{};
};

} // namespace VideoCore
