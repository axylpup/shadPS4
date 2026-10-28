// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <map>
#include <mutex>
#include <tuple>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <thread>
#include <immintrin.h>
#include <unordered_map>
#include "common/alignment.h"
#include "common/buffer_sample_probe.h"
#include "common/debug.h"
#include "common/fault_monitor.h"
#include "common/lean.h"
#include "common/readback_offload.h"
#include "common/scope_exit.h"
#include "common/thread.h"
#include "common/memory_patcher.h"
#include "common/path_util.h"
#include "core/memory.h"
#include "shader_recompiler/native_emit.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/mtb_census.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/readback_queue.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/pb_census.h"
#include "video_core/native_census.h"
#include "video_core/renderer_vulkan/vk_gpu_batch_timer.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_memory_budget.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

#include <vk_mem_alloc.h>
#include <xxhash.h>
#include "video_core/buffer_cache/protect_queue.h"

namespace VideoCore {
namespace {
void NoteReadbackWindow(VAddr addr, u64 size); // q7, defined below
}

const bool g_native_census = [] {
    const char* e = std::getenv("SHADPS4_NATIVE_CENSUS");
    return e && e[0] == '1';
}();
const int g_native_pass = [] {
    const char* e = std::getenv("SHADPS4_NATIVE_PASS");
    return e ? e[0] - '0' : 0;
}();
const u32 g_native_colour_diag = [] {
    const char* e = std::getenv("SHADPS4_NATIVE_COLOUR_DIAG");
    return e ? static_cast<u32>(std::strtoul(e, nullptr, 0)) : 0u;
}();
bool g_native_colour = [] {
    const char* e = std::getenv("SHADPS4_NATIVE_COLOUR");
    return g_native_pass != 0 && e && e[0] == '1';
}();


namespace {
// SHADPS4_DRAW_COST=1 (diagnostic): ObtainBufferForImage path and cycle split for calls over
// ~100 us (SynchronizeBuffer: dirty-page walk vs rest); first 8 lines per address.
const bool g_image_buffer_cost = [] {
    const char* value = std::getenv("SHADPS4_DRAW_COST");
    return value != nullptr && value[0] == '1';
}();
thread_local u64 g_sync_walk_cyc = 0;
thread_local u64 g_sync_rest_cyc = 0;
} // namespace

u64& DiagSyncWalkCycles() {
    return g_sync_walk_cyc;
}
u64& DiagSyncRestCycles() {
    return g_sync_rest_cyc;
}

namespace Offload = Common::ReadbackOffload;

namespace {
// Flags checked per binding, per draw or per fault, read once at startup: a function-local
// `static const` costs a guard check (a TLS epoch load chain under the MSVC ABI) on every call.
bool EnvIs1(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] == '1';
}
const char g_prebatch_mode = [] {
    const char* value = std::getenv("SHADPS4_READBACK_PREBATCH");
    return value && (value[0] == '1' || value[0] == '2') ? value[0] : '0';
}();
const bool g_precise_producer = EnvIs1("SHADPS4_PRECISE_PRODUCER");
const bool g_buffer_traffic_enabled = EnvIs1("SHADPS4_BUFFER_TRAFFIC");
const bool g_fault_write_skip_settle = EnvIs1("SHADPS4_FAULT_WRITE_SKIP_SETTLE");
const bool g_async_submit = EnvIs1("SHADPS4_ASYNC_SUBMIT");
/// SHADPS4_UPLOAD_AHEAD_STATS=1 (diagnostic, notes/transfer-serialization-2026-09-28.md): per
/// upload, how many CPU-dirty tracker pages lie in the 64 pages after it (read-ahead potential),
/// and how many transfer ops follow another transfer with nothing recorded in between (barrier
/// coalescing potential). Each transfer op records its own barrier pair today.
const bool g_upload_ahead_stats = EnvIs1("SHADPS4_UPLOAD_AHEAD_STATS");
/// SHADPS4_FAULT_VETO_RETRY=N (0-4, default 0): a guest fault job that was vetoed (a GPU write was
/// marked in its regions while it settled) is prepared again as a new asynchronous job up to N
/// times before the synchronous fallback. The fallback downloads on the command processor with a
/// flush and a GPU wait; measured 2026-09-30 in fire: 4.3% of jobs are vetoed (0.36 per frame)
/// and ResolveFaultSync is ~4% of the CP back's samples. A retry costs the faulting guest thread
/// one more job round trip instead. A vetoed job has published nothing and released its staging
/// and window (FinalizeFaultJob), so preparing again is the same as a new fault on that address.
/// Read faults only: a vetoed write job must keep its single fallback behind the unmap gate (Opus
/// review 2026-09-30). Measured the same night: falsified - retried jobs are vetoed again about two
/// times in three and the synchronous fallbacks do not fall (fire, N=2). Leave at 0.
const u32 g_fault_veto_retry = [] {
    const char* value = std::getenv("SHADPS4_FAULT_VETO_RETRY");
    const unsigned long n = value ? std::strtoul(value, nullptr, 10) : 0ul;
    return static_cast<u32>(n > 4 ? 4 : n);
}();
std::atomic<u64> g_fault_veto_retries{0};
/// SHADPS4_FAULT_GUEST_COPY=1 (default off): the guest thread that owns a fault job copies the
/// downloaded bytes from the job's staging buffer into guest memory itself, after its GPU wait
/// and before FinishFaultDownload, instead of the command processor doing it inside the
/// publication. Measured 2026-10-01 (a41, fire): TryWriteBacking under PublishFaultJob is 5.9%
/// of the CP back's wall time (1 MiB windows, ~650 jobs a second), and the back is the frame
/// rate limiter while the faulting thread is waiting anyway.
/// Why the early copy is safe: the pages stay GPU-modified and protected until the CP's
/// publication, so no guest code can see them; fault windows never overlap (a second fault on
/// a pending window gets a retry ticket) and every CP path that touches a pending window
/// settles the job first, which waits for or forestalls this copy (FaultJob::guest_copy). A
/// job vetoed after the copy leaves its bytes behind the protection: GPU bits are cleared only
/// by paths that download every GPU-modified byte of the pages they unmark, so something newer
/// overwrites them before any unprotect (Opus review 2026-10-01). Not used in verify mode.
const bool g_fault_guest_copy = EnvIs1("SHADPS4_FAULT_GUEST_COPY");
std::atomic<u64> g_guest_copies{0}, g_guest_copy_closed{0}, g_guest_copy_cp_waits{0};
/// SHADPS4_FAULT_GUEST_SUBMIT=1 (default off): for a job copied on a lane with its own queue
/// (the fast lane; the prebatch readback queue is the main queue in submission order and its
/// copies are submitted by the recording worker), the command processor records the copy into a
/// ReadbackQueue slot and the owning guest thread calls vkQueueSubmit before it waits, instead of
/// the command processor doing it inside PrepareFaultDownload. Measured 2026-10-01 (a41, fire):
/// ReadbackQueue::Submit is 2.6% of the CP back's samples, vk::Queue::submit 2.0% of them (~23 us
/// per job in the driver). The lane's timeline value is taken under the lane's queue mutex at
/// submit time, so values are submitted in order whichever thread submits; the job's tick is
/// unknown until then, and every command processor reader of it closes the submit first
/// (FaultJob::guest_submit: wait for an in-flight guest submit, or submit the slot itself).
/// Buffer deletion closes the submits of the jobs over the buffer before it reads the lane tick
/// it waits for (DeleteBuffer). Not used in verify mode.
const bool g_fault_guest_submit = EnvIs1("SHADPS4_FAULT_GUEST_SUBMIT");
std::atomic<u64> g_guest_submits{0}, g_guest_submit_takeovers{0}, g_guest_submit_cp_waits{0};
/// SHADPS4_DEFERRED_COPY_CENSUS=1 (diagnostic): with DEFERRED_PROTECT the guest -> staging copy
/// of an upload runs on the recording thread, later than the command processor recorded it, and
/// guest labels are written at CP time. Counts deferred copies whose guest bytes changed between
/// record time and copy time, and how many of those had a guest label written in between (the
/// guest was told the GPU was done and may have reused the memory). Opus review 2026-09-30.
const bool g_deferred_copy_census = EnvIs1("SHADPS4_DEFERRED_COPY_CENSUS");

struct DeferredCopyCensus {
    std::atomic<u64> copies{0}, bytes{0}, ahead{0}, unreadable{0}, bound{0}, changed{0},
        after_label{0}, changed_after_label{0}, max_label_lag{0};
};
DeferredCopyCensus g_dcc;

/// At most the first and the last 512 bytes of [data, data + len): cheap enough to leave the
/// timing alone (hashing whole copies ran the fire scene at 39 FPS and saturated the recorder).
u64 CensusSampleHash(const u8* data, u64 len) {
    constexpr u64 Edge = 512;
    if (len <= 2 * Edge) {
        return XXH3_64bits(data, len);
    }
    return XXH3_64bits(data, Edge) ^ (XXH3_64bits(data + len - Edge, Edge) * 0x9E3779B97F4A7C15ULL);
}

/// Recording thread, right after the guest -> staging copy of `dest`.
void NoteDeferredCopyCensus(VAddr addr, const u8* dest, u64 size, u64 hash, u64 label_seq,
                            u32 hash_off, u32 hash_len, u32 kind) {
    auto& c = g_dcc;
    const u64 n = c.copies.fetch_add(1, std::memory_order_relaxed) + 1;
    c.bytes.fetch_add(size, std::memory_order_relaxed);
    if (kind == 2) {
        c.ahead.fetch_add(1, std::memory_order_relaxed);
    } else if (kind == 3) {
        c.unreadable.fetch_add(1, std::memory_order_relaxed);
    } else if (kind == 1) {
        c.bound.fetch_add(1, std::memory_order_relaxed);
        const u64 now_seq = VideoCore::g_guest_label_seq.load(std::memory_order_relaxed);
        const bool after = now_seq != label_seq;
        const bool changed = CensusSampleHash(dest + hash_off, hash_len) != hash;
        c.after_label.fetch_add(after, std::memory_order_relaxed);
        c.changed.fetch_add(changed, std::memory_order_relaxed);
        if (after) {
            u64 lag = now_seq - label_seq;
            u64 prev = c.max_label_lag.load(std::memory_order_relaxed);
            while (lag > prev && !c.max_label_lag.compare_exchange_weak(prev, lag)) {
            }
        }
        if (after && changed) {
            const u64 k = c.changed_after_label.fetch_add(1, std::memory_order_relaxed) + 1;
            if (k <= 24) {
                LOG_WARNING(Render,
                            "DeferredCopyCensus: bound guest bytes changed after a label: "
                            "{:#x}+{:#x} (labels between record and copy: {})",
                            addr + hash_off, hash_len, now_seq - label_seq);
            }
        }
    }
    if ((n & 0x3ffff) == 0) {
        LOG_INFO(Render,
                 "DeferredCopyCensus: copies={} bytes={} upload_ahead={} unreadable={} | bound={} "
                 "changed={} copied_after_label={} changed_after_label={} max_labels_between={}",
                 n, c.bytes.load(), c.ahead.load(), c.unreadable.load(), c.bound.load(),
                 c.changed.load(), c.after_label.load(), c.changed_after_label.load(),
                 c.max_label_lag.load());
    }
}
/// SHADPS4_UPLOAD_AHEAD=<pages> (notes/transfer-serialization-2026-09-28.md): after a small
/// upload (<= 16 KiB), also upload the CPU-dirty pages among the next <pages> tracker pages of the
/// same Buffer, as a later binding of them would. Fire's particle dispatches each bind a fresh
/// page of an arena the CPU filled before submitting: one upload + barrier pair per dispatch
/// serialized the chain (60% of uploads had the next page dirty, 67% >= 16 of the next 64).
const u32 g_upload_ahead_pages = [] {
    const char* value = std::getenv("SHADPS4_UPLOAD_AHEAD");
    return value ? static_cast<u32>(std::clamp(std::atoi(value), 0, 256)) : 0u;
}();
thread_local bool t_upload_ahead_active = false;
std::atomic<u64> g_upload_ahead_calls{}, g_upload_ahead_bytes{};
/// SHADPS4_TRANSFER_COALESCE=1: uploads (deferred-protect path) and guest buffer copies use the
/// scheduler's transfer group (one barrier pair per run of consecutive transfers).
const bool g_transfer_coalesce = Vulkan::Scheduler::TransferCoalesceEnabled();
struct UploadAheadStats {
    std::atomic<u64> uploads{}, upload_bytes{}, copies{}, transfers{}, adjacent{};
    std::atomic<u64> ahead[8]{}; ///< dirty pages ahead: 0, 1, 2-3, 4-7, 8-15, 16-31, 32-63, 64
    std::atomic<u64> gap[4]{};   ///< first dirty page ahead at: +1, +2..4, +5..16, +17.. or none
    std::atomic<u64> ahead_same_run[2]{}; ///< [0] next page dirty, [1] next page clean
    u64 last_seq = ~0ULL;
    u32 last_stream = ~0U;
};
UploadAheadStats g_uas;
const bool g_readback_early_flush = EnvIs1("SHADPS4_READBACK_EARLY_FLUSH");
const bool g_readback_producer_stats = EnvIs1("SHADPS4_READBACK_PRODUCER_STATS");
/// SHADPS4_WRITE_FAULT_WIDEN=1: a write fault on GPU-modified memory first downloads the same
/// widened window a read fault would (one submit and wait), so neighbouring GPU-patched pages the
/// guest CPU rewrites next no longer each cost their own synchronous download. Train-dip captures
/// (2026-09-26): ~10 consecutive 4 KB record pages, patched by cs_a16fa93a copies and then
/// rewritten by the CPU, cost ~10 extra submits and GPU waits per frame.
/// =2: the window spans every buffer in it (the train records are split into 16 KB buffers), all
/// downloaded with one submit and one wait.
const char g_write_fault_widen = [] {
    const char* value = std::getenv("SHADPS4_WRITE_FAULT_WIDEN");
    return value && (value[0] == '1' || value[0] == '2') ? value[0] : '0';
}();
/// SHADPS4_WRITE_FAULT_OFFLOAD=1 (with WRITE_FAULT_WIDEN=2 and READBACK_OFFLOAD=1, default off): a
/// guest thread's write fault on GPU-modified memory becomes a readback job over the same window
/// the synchronous =2 path downloads. The command processor records the copies and moves on; the
/// faulting thread waits for the job tick, and publication writes the bytes back, clears the
/// window's GPU state and CPU-marks the exact faulting range, before the guest write proceeds.
/// Measured 2026-09-26 (a25 dip anatomy): the dip adds ~2.5 command processor GPU waits per frame.
const bool g_write_fault_offload =
    g_write_fault_widen == '2' && EnvIs1("SHADPS4_WRITE_FAULT_OFFLOAD");
const bool g_range_add_skip = EnvIs1("SHADPS4_RANGE_ADD_SKIP");
const bool g_vb_repeat_stats = EnvIs1("SHADPS4_VB_REPEAT_STATS");
/// SHADPS4_CP_SPLIT_FRONT_DERIVE (cp_split.h): 0 off, 1 use, 2 verify; and mode 1's
/// setVertexInputEXT skip switch. See BindVertexBuffers.
const int g_front_derive = AmdGpu::CpSplit::FrontDerive();
const bool g_front_derive_vi_skip = AmdGpu::CpSplit::FrontDeriveVertexInputSkip();
const bool g_written_memo = EnvIs1("SHADPS4_WRITTEN_MEMO");
/// SHADPS4_BUFFER_RANGES_LOCK (default on; =0 disables): see BufferCache::buffer_ranges_mutex.
/// Found 2026-09-28: a freeze's all-thread capture (logs/hang-night0928-1-*) had a guest job thread
/// spinning forever inside buffer_ranges' tree lookup (IsRegionRegistered from a write fault) while
/// the command processor edits that tree on buffer create/join/delete: an unsynchronized reader of a
/// red-black tree being rebalanced can loop. The game then waited forever for that job's fence.
const bool g_buffer_ranges_lock = [] {
    const char* value = std::getenv("SHADPS4_BUFFER_RANGES_LOCK");
    return !(value && value[0] == '0');
}();
/// SHADPS4_BUFFER_RANGES_LOCK=2: IsRegionRegistered reads page_table (the buffer id per 16 KB caching
/// page that ChangeRegister writes next to buffer_ranges; buffers are caching-page aligned, so "any
/// page has a buffer" equals Intersects) with no lock and no tree walk. The table's second-level
/// pages are never freed and find() never allocates. Ranges over 64 MB use the locked tree. Added
/// because a short screen (s73) ran a32 (mode 1) ~2% below a30 in fire, possibly the shared lock's
/// cache-line traffic across faulting guest threads (guess).
const bool g_buffer_ranges_table = [] {
    const char* value = std::getenv("SHADPS4_BUFFER_RANGES_LOCK");
    return value && value[0] == '2';
}();
const bool g_thread_assert = EnvIs1("SHADPS4_THREAD_ASSERT");
const bool g_asc_range_shadow = [] {
    const char* value = std::getenv("SHADPS4_ASC_V2_CENSUS");
    return value && value[0] == '2';
}();
const bool g_asc_census = EnvIs1("SHADPS4_ASC_V2_CENSUS") || g_asc_range_shadow;
/// SHADPS4_ASC_RANGE=1: see BufferCache::AscRangeEnabled. Range notes feed a decision then.
const bool g_asc_range = EnvIs1("SHADPS4_ASC_RANGE");
/// SHADPS4_ASC_I1: unset/1 = I1 (every main submission waits the last ASC segment); 2 = I1 plus
/// the precise dependency tracking in shadow (counters); 0 = precise waits only. Either turns
/// the range notes on (the main-side checks run from them; Opus review 2026-09-28).
const bool g_asc_deps = [] {
    const char* value = std::getenv("SHADPS4_ASC_I1");
    const bool on = value && (value[0] == '0' || value[0] == '2');
    VideoCore::g_asc_deps_active = on;
    return on;
}();
const bool g_range_notes = g_asc_range_shadow || g_asc_range || g_asc_deps;
/// SHADPS4_WRITTEN_RUN_MEMO: 0 off, 1 on, 2 verify. See ObtainBuffer.
const int g_written_run_memo = [] {
    const char* value = std::getenv("SHADPS4_WRITTEN_RUN_MEMO");
    const char* asc = std::getenv("SHADPS4_ASC_STREAM");
    if (!value || (value[0] != '1' && value[0] != '2') || (asc && asc[0] != '0')) {
        return 0;
    }
    return value[0] - '0';
}();
/// SHADPS4_READ_BIND_MEMO: 0 off, 1 on, 2 verify (a hit still runs the full path and checks that
/// it uploaded nothing; the returned pair is identical by construction); 3 / 4 the same with the
/// buffer's creation serial as the structure guard instead of the global StructGeneration. Off
/// under SHADPS4_EPOCH_DEFER_UPLOAD, which can leave a synchronized read range dirty on purpose.
/// A hit returns before the ASC_SPAN_STATS / BUFFER_TRAFFIC / sample-probe accounting: take those
/// censuses with the memo off (Opus review 2026-09-28).
const int g_read_bind_memo = [] {
    const char* value = std::getenv("SHADPS4_READ_BIND_MEMO");
    const char* defer = std::getenv("SHADPS4_EPOCH_DEFER_UPLOAD");
    if (!value || value[0] < '1' || value[0] > '4' || (defer && defer[0] == '1')) {
        return 0;
    }
    return value[0] - '0';
}();
const bool g_deferred_protect = EnvIs1("SHADPS4_DEFERRED_PROTECT");
} // namespace

/// SHADPS4_READBACK_PREBATCH: 1 = copies of data the recorded batch does not write go ahead of
/// it; 2 = the same, and the batch is still submitted right after, as a fault always did.
/// Measured 2026-09-24: mode 1 alone let batches grow (787 vs 650 draws) and the remaining
/// in-batch readbacks waited 2.9 ms instead of 1.1 ms.
static char PrebatchMode() {
    return g_prebatch_mode;
}

/// SHADPS4_PRECISE_PRODUCER=1 (with SHADPS4_READBACK_QUEUE=1 or 2, prebatch off): a readback
/// copy on the readback queue waits only for the newest batch that actually wrote its bytes,
/// found in a 64-batch write history, instead of the coarse 1 MiB-slot producer ticks (whose
/// floor made nearly every copy wait for the newest submitted batch). Measured 2026-09-24:
/// early copies in queue order waited ~0.7 ms behind unrelated batches.
static bool PreciseProducerEnabled() {
    return g_precise_producer;
}

static bool PrebatchEnabled() {
    return PrebatchMode() != '0';
}

/// SHADPS4_READBACK_FAST_LANE=1 (with SHADPS4_READBACK_QUEUE=1 for the extra queue and
/// SHADPS4_READBACK_PREBATCH=2): a guest readback whose newest GPU writer (64-batch write
/// history) has already completed is copied at once on a second queue instead of behind the
/// unfinished main-queue batches; every other readback keeps the prebatch path. On PS4 the CPU
/// reads finished GPU results in place; measured 2026-09-27 (ASC_YIELD on) such copies waited
/// ~1.4 ms of unrelated GPU work. Use READBACK_QUEUE=3 (compute family): a second graphics-family
/// queue time-slices the graphics engine on NVIDIA.
static bool FastLaneEnabled() {
    static const bool enabled = EnvIs1("SHADPS4_READBACK_FAST_LANE");
    return enabled;
}

/// SHADPS4_BATCH_MIX_SS=1 (diagnostic, inFAMOUS Second Son only, with SHADPS4_READBACK_LATENCY and
/// SHADPS4_GPU_BATCH_TIMING): classifies a read fault by the game's own particle globals (REFERENCE
/// 4.2): the per-frame linear arena that holds the effect control blocks (cursor 0x807d3eb20, limit
/// 0x807d3eb28; the arena's size is not known, so "arena" = the 64 MiB below the limit, checked by
/// the logged samples) and the particle pool ([0x807d3eb40], 0x20000 records of 96 bytes).
static u32 ClassifyReadbackAddr(VAddr addr) {
    using Timer = Vulkan::GpuBatchTimer;
    static const bool enabled = EnvIs1("SHADPS4_BATCH_MIX_SS");
    if (!enabled) {
        return Timer::MixOther;
    }
    // Exact when the faulting instruction is known: the game functions of REFERENCE 4.2
    // (control-block reader 0x3f3ed0, mesh snap 0x3fed00, exposure sum 0x171610; function
    // extents assumed, entry + a few KB).
    if (Offload::t_in_hle_memset) {
        return Timer::MixMemset;
    }
    if (const u64 tagged = Offload::t_read_fault_rip; tagged != 0) {
        const bool write = (tagged >> 63) != 0;
        const u64 rip = tagged & ~(1ULL << 63);
        const u64 off = rip - 0x800000000ULL;
        u32 by_rip = write ? Timer::MixWriteFault : Timer::MixOther;
        if (off >= 0x3f3ed0 && off < 0x3f4800) {
            by_rip = Timer::MixArena;
        } else if (off >= 0x3fed00 && off < 0x400000) {
            by_rip = Timer::MixPool;
        } else if (off >= 0x171610 && off < 0x171a00) {
            by_rip = Timer::MixExposure;
        }
        static std::atomic<u32> other_samples{0};
        if ((by_rip == Timer::MixOther || by_rip == Timer::MixWriteFault) &&
            other_samples.fetch_add(1, std::memory_order_relaxed) < 256) {
            LOG_INFO(Render_Vulkan, "BatchMixOtherRip: write={} rip={:#x} off={:#x} addr={:#x}",
                     write, rip, off, addr);
        }
        return by_rip;
    }
    const auto read = [](u64 a) { return *reinterpret_cast<const volatile u64*>(a); };
    const u64 cursor = read(0x807d3eb20);
    const u64 limit = read(0x807d3eb28);
    const u64 pool = read(0x807d3eb40);
    u32 cls = Timer::MixOther;
    if (pool != 0 && addr >= pool && addr < pool + 0x20000ULL * 96) {
        cls = Timer::MixPool;
    } else if (limit != 0 && addr < limit && addr + (64ULL << 20) >= limit) {
        cls = Timer::MixArena;
    }
    static std::atomic<u32> samples{0};
    if (samples.fetch_add(1, std::memory_order_relaxed) < 48) {
        LOG_INFO(Render_Vulkan,
                 "BatchMixSample: addr={:#x} cls={} cursor={:#x} limit={:#x} pool={:#x} "
                 "limit-addr={:#x} cursor-addr={}",
                 addr, cls, cursor, limit, pool, limit - addr, s64(cursor) - s64(addr));
    }
    return cls;
}

namespace {
/// SHADPS4_READBACK_LATENCY: a thread spinning on the master timeline semaphore that records
/// when each tick was first seen complete. Diagnostic only; it burns one core.
struct TickPoller {
    static constexpr u64 Size = 8192;
    std::array<std::atomic<u64>, Size> ticks{};
    std::array<std::atomic<u64>, Size> done_ns{};
    std::jthread thread;

    /// When `tick` was first observed complete, or 0 if unknown.
    u64 DoneNs(u64 tick) const {
        const u64 i = tick % Size;
        if (ticks[i].load(std::memory_order_acquire) != tick) {
            return 0;
        }
        return done_ns[i].load(std::memory_order_relaxed);
    }
};
TickPoller* g_tick_poller = nullptr;

void StartTickPoller(Vulkan::MasterSemaphore* semaphore) {
    static TickPoller poller;
    g_tick_poller = &poller;
    poller.thread = std::jthread([semaphore](std::stop_token stop) {
        Common::SetCurrentThreadName("shadPS4:TickPoller");
        u64 last = semaphore->QueryCounter();
        while (!stop.stop_requested()) {
            const u64 value = semaphore->QueryCounter();
            if (value > last) {
                const u64 now = Offload::NowNs();
                const u64 first =
                    std::max(last + 1, value > TickPoller::Size ? value - TickPoller::Size + 1 : 1);
                for (u64 t = first; t <= value; ++t) {
                    const u64 i = t % TickPoller::Size;
                    poller.done_ns[i].store(now, std::memory_order_relaxed);
                    poller.ticks[i].store(t, std::memory_order_release);
                }
                last = value;
            } else {
                for (int i = 0; i < 8; ++i) {
                    _mm_pause();
                }
            }
        }
    });
}
} // namespace

static constexpr size_t DataShareBufferSize = 64_KB;
static constexpr size_t StagingBufferSize = 512_MB;
// 1.1.1 (#36): an upload whose one host buffer is refused goes through a buffer of this size.
static constexpr size_t UploadPieceSize = 256_MB;
static constexpr size_t DownloadBufferSize = 32_MB;
static constexpr size_t UboStreamBufferSize = 64_MB;
static constexpr size_t DeviceBufferSize = 128_MB;

/**
 * Alignment a guest read fault is widened to before downloading.
 *
 * Windows faults report 8 bytes, while protection is page granular and a nonempty buffer
 * download submits and waits. Combining nearby read faults can reduce those queue drains.
 *
 * Widening stays within the existing buffer and selects only tracked GPU-dirty pages/bytes.
 * Extra bytes are speculative: the guest may not read them. Enlarged windows also preflight
 * staging capacity before dirty-state mutation. Keep the default for unmeasured workloads.
 */
static constexpr u64 DefaultReadFaultWidenSize = 64_KB;

static u64 ReadFaultWidenSize(u64 original_size) {
    static const u64 value = []() -> u64 {
        if (const char* env = std::getenv("SHADPS4_READ_FAULT_WIDEN_SIZE")) {
            char* end = nullptr;
            const u64 parsed = std::strtoull(env, &end, 0);
            if (end != env && *end == '\0' &&
                (parsed == 64_KB || parsed == 256_KB || parsed == 1_MB)) {
                LOG_INFO(Render, "BufferCache: read-fault window overridden to {} bytes", parsed);
                return parsed;
            }
            LOG_WARNING(Render, "Invalid read-fault window override; retaining 64 KiB");
        }
        return DefaultReadFaultWidenSize;
    }();
    // Only tune small read faults. Preserve baseline handling of larger explicit reads.
    return original_size > 0 && original_size <= 4_KB ? value : DefaultReadFaultWidenSize;
}

/**
 * Size ceiling for `ObtainBuffer`'s read-only stream-copy fast path.
 *
 * Upstream compares against `CACHING_PAGESIZE` (16 KB), sending small read-only bindings down a
 * path that copies guest memory into the stream buffer every draw "to reduce renderpass breaks".
 * Measured, that trade loses: the copy runs 6.6 times per draw and moves ~43 MB/frame, while the
 * tracked buffer-cache path it avoids costs about the same per call (~0.37 us vs ~0.35 us) and
 * copies nothing when the range is clean.
 *
 * **Default is 0, i.e. the fast path is off.** Same build, same scene, same draw count:
 *
 *     stream path ON   12.47 fps    ObtainBuffer ~5.2 us/draw
 *     stream path OFF  14.57 fps    ObtainBuffer  3.06 us/draw     +17%
 *
 * The objection to changing the default was memory: every small binding becomes a resident cache
 * entry. Measured head-to-head, 10 minutes idle each - growth is 6.00 MB/min with the path on and
 * 8.26 MB/min with it off, but the OFF arm DECELERATES (10.42 -> 6.00 MB/min across the run) and
 * converges on the ON arm's rate. So the extra ~22 MB is a cache filling once, not a leak. The
 * ~6 MB/min both arms share is a pre-existing leak unrelated to this.
 *
 * Set `SHADPS4_STREAM_MAX_SIZE=16384` to restore upstream behaviour. Both paths are correct - this
 * is a performance knob, not a compatibility one.
 */
static u64 StreamCopyMaxSize() {
    static const u64 value = []() -> u64 {
        if (const char* env = std::getenv("SHADPS4_STREAM_MAX_SIZE")) {
            char* end = nullptr;
            const u64 parsed = std::strtoull(env, &end, 0);
            if (end != env) {
                LOG_INFO(Render, "BufferCache: stream-copy max size overridden to {}", parsed);
                return parsed;
            }
        }
        return 0;
    }();
    return value;
}

namespace {
/// SHADPS4_BUFFER_TRAFFIC=1 (diagnostic, costs a tracker query per binding): buffer bindings
/// per 16 MB guest region and per size class - kind, bytes bound, whether the range had
/// CPU-modified pages at bind time, and bytes uploaded - logged every ~2 s. Sizes plan (b).
struct BufferTraffic {
    struct Region {
        u64 calls[4]{}; // ro, ro_texel, written, other-caller share of ro
        u64 bound{}, dirty{}, uploaded{};
    };
    std::unordered_map<u64, Region> regions;
    u64 size_calls[6]{}, size_dirty[6]{}, size_bound[6]{}, size_uploaded[6]{};
    u64 calls{}, last_frames{};
    std::chrono::steady_clock::time_point last{std::chrono::steady_clock::now()};

    static bool Enabled() {
        return g_buffer_traffic_enabled;
    }

    void Record(VAddr addr, u32 size, bool written, bool texel, bool other_caller, bool dirty,
                u64 uploaded) {
        auto& r = regions[addr >> 24];
        ++r.calls[written ? 2 : texel ? 1 : 0];
        r.calls[3] += (!written && !texel && other_caller);
        r.bound += size;
        r.dirty += dirty;
        r.uploaded += uploaded;
        const int b = size <= 256 ? 0 : size <= 4096 ? 1 : size <= 65536 ? 2
                    : size <= (1u << 20) ? 3 : size <= (16u << 20) ? 4 : 5;
        ++size_calls[b];
        size_dirty[b] += dirty;
        size_bound[b] += size;
        size_uploaded[b] += uploaded;
        if ((++calls & 1023) != 0) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - last < std::chrono::seconds(2)) {
            return;
        }
        const u64 frames_now = Common::FaultMonitor::presented_frames.load();
        const double f = static_cast<double>(std::max<u64>(frames_now - last_frames, 1));
        std::vector<std::pair<u64, Region>> sorted(regions.begin(), regions.end());
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            return a.second.calls[0] + a.second.calls[1] + a.second.calls[2] >
                   b.second.calls[0] + b.second.calls[1] + b.second.calls[2];
        });
        u64 total = 0, total_dirty = 0, total_up = 0;
        for (const auto& [base, reg] : sorted) {
            total += reg.calls[0] + reg.calls[1] + reg.calls[2];
            total_dirty += reg.dirty;
            total_up += reg.uploaded;
        }
        std::string out = fmt::format(
            "BufferTraffic: frames={:.0f} binds/frame={:.0f} dirty/frame={:.0f} "
            "upload={:.0f}kB/frame regions={} | size(<=256,4K,64K,1M,16M,>16M) binds/frame",
            f, total / f, total_dirty / f, total_up / 1024.0 / f, sorted.size());
        for (int i = 0; i < 6; ++i) {
            out += fmt::format(" {:.0f}(d{:.0f},up{:.0f}k)", size_calls[i] / f, size_dirty[i] / f,
                               size_uploaded[i] / 1024.0 / f);
        }
        out += " | top regions base:ro/texel/written(other) dirty up_kB bound_MB per frame:";
        for (size_t i = 0; i < std::min<size_t>(10, sorted.size()); ++i) {
            const auto& reg = sorted[i].second;
            out += fmt::format(" {:#x}:{:.0f}/{:.0f}/{:.0f}({:.0f}) d{:.0f} up{:.0f} b{:.1f}",
                               sorted[i].first << 24, reg.calls[0] / f, reg.calls[1] / f,
                               reg.calls[2] / f, reg.calls[3] / f, reg.dirty / f,
                               reg.uploaded / 1024.0 / f, reg.bound / 1048576.0 / f);
        }
        LOG_INFO(Render, "{}", out);
        regions.clear();
        std::fill(std::begin(size_calls), std::end(size_calls), 0);
        std::fill(std::begin(size_dirty), std::end(size_dirty), 0);
        std::fill(std::begin(size_bound), std::end(size_bound), 0);
        std::fill(std::begin(size_uploaded), std::end(size_uploaded), 0);
        last = now;
        last_frames = frames_now;
    }
};
BufferTraffic g_buffer_traffic;
} // namespace

/// Expands a read fault to the selected window, clipped to the buffer that already covers it.
/// The clip is what keeps this safe: `FindBuffer` has already run on the original range, so
/// buffer creation and merging behaviour is unchanged by the wider download.
static std::pair<VAddr, u64> WidenReadFault(const Buffer& buffer, VAddr device_addr, u64 size,
                                            u64 window) {
    if (!buffer.IsInBounds(device_addr, size)) {
        return {device_addr, size};
    }
    const VAddr begin = std::max(Common::AlignDown(device_addr, window), buffer.CpuAddr());
    const VAddr end = std::min(Common::AlignUp(device_addr + size, window),
                               buffer.CpuAddr() + buffer.SizeBytes());
    return {begin, end - begin};
}

extern BufferCache* g_discard_cache; // SHADPS4_PARTICLE_ARENA_DISCARD, defined below

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         AmdGpu::Liverpool* liverpool_, TextureCache& texture_cache_,
                         PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_}, page_manager{tracker},
      fault_manager{instance, scheduler, *this, CACHING_PAGEBITS, CACHING_NUMPAGES},
      staging_buffer{instance, scheduler, MemoryUsage::Upload, StagingBufferSize},
      // eShaderDeviceAddress: SHADPS4_NATIVE_SEG reads a batch's record list from it by pointer.
      stream_buffer{instance, scheduler, MemoryUsage::Stream, UboStreamBufferSize,
                    Shader::g_native_seg != 0 ? vk::BufferUsageFlagBits::eShaderDeviceAddress
                                              : vk::BufferUsageFlags{}},
      download_buffer{instance, scheduler, MemoryUsage::Download, DownloadBufferSize},
      device_buffer{instance, scheduler, MemoryUsage::DeviceLocal, DeviceBufferSize},
      gds_buffer{instance, scheduler, MemoryUsage::Stream, 0, AllFlags, DataShareBufferSize},
      // eShaderDeviceAddress: SHADPS4_NATIVE_PT reads a page-table miss from the table itself.
      bda_pagetable_buffer{instance, scheduler, MemoryUsage::DeviceLocal,
                           0,        AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                           BdaPageTableSize()} {
    g_discard_cache = this; // SHADPS4_PARTICLE_ARENA_DISCARD hook
    g_lane_on = scheduler.HasLane();
    g_lane_counters = &scheduler.lane_c;
    g_lane_open_src = &scheduler.lane_open_src;
    if (const char* mirror = std::getenv("SHADPS4_HOST_MIRROR"); mirror && mirror[0] == '1') {
        host_mirror = instance.IsExternalMemoryHostEnabled();
        host_mirror_max = static_cast<u32>(256_MB);
        if (const char* max = std::getenv("SHADPS4_HOST_MIRROR_MAX")) {
            host_mirror_max = static_cast<u32>(std::strtoul(max, nullptr, 0));
        }
        // "lo-hi" (hex ok): mirror every eligible binding inside this guest range, without the
        // per-bind dirty-page query (which walks the range a second time).
        if (const char* texel = std::getenv("SHADPS4_HOST_MIRROR_TEXEL")) {
            host_mirror_texel = texel[0] == '1';
        }
        if (const char* range = std::getenv("SHADPS4_HOST_MIRROR_RANGE")) {
            char* end = nullptr;
            host_mirror_lo = std::strtoull(range, &end, 0);
            if (end && *end == '-') {
                host_mirror_hi = std::strtoull(end + 1, nullptr, 0);
            }
            LOG_INFO(Render_Vulkan, "SHADPS4_HOST_MIRROR_RANGE: {:#x}-{:#x}", host_mirror_lo,
                     host_mirror_hi);
        }
        LOG_INFO(Render_Vulkan, "SHADPS4_HOST_MIRROR: {} (max binding {} bytes)",
                 host_mirror ? "on" : "unavailable (VK_EXT_external_memory_host)",
                 host_mirror_max);
    }
    Vulkan::SetObjectName(instance.GetDevice(), gds_buffer.Handle(), "GDS Buffer");
    LOG_INFO(Render_Vulkan, "GPU page table: {} MB (covers {} GB of guest addresses)",
             BdaPageTableSize() >> 20, (BdaNumPages() * CACHING_PAGESIZE) >> 30);
    Vulkan::SetObjectName(instance.GetDevice(), bda_pagetable_buffer.Handle(),
                          "BDA Page Table Buffer");

    memory_tracker = std::make_unique<MemoryTracker>(tracker);
    // Same family (READBACK_QUEUE=1) or a CONCURRENT one (3); never EXCLUSIVE across families.
    const bool lane_queue_ok =
        instance.HasReadbackQueue() &&
        (instance.GetReadbackQueueFamilyIndex() == instance.GetGraphicsQueueFamilyIndex() ||
         instance.BuffersShareReadbackFamily());
    if (FastLaneEnabled() && lane_queue_ok && PrebatchEnabled() && Offload::Enabled()) {
        readback_queue = std::make_unique<ReadbackQueue>(
            instance, instance.GetGraphicsQueue(), instance.GetGraphicsQueueFamilyIndex(),
            &Vulkan::Scheduler::submit_mutex);
        fast_lane = std::make_unique<ReadbackQueue>(instance);
        LOG_INFO(Render, "ReadbackFastLane: completed-producer copies on a second queue, the "
                         "rest prebatched");
        if (instance.HasRetryQueue()) {
            // r18 SHADPS4_ARENA_RETRY_QUEUE: the arena retry copies wait for their producer;
            // on their own queue they never hold up a fault copy queued after them.
            retry_lane = std::make_unique<ReadbackQueue>(instance, instance.GetRetryQueue(),
                                                         instance.GetReadbackQueueFamilyIndex(),
                                                         nullptr);
            LOG_INFO(Render, "ArenaRetryQueue: arena prefetch retry copies on their own queue");
        }
    } else if (instance.HasReadbackQueue()) {
        readback_queue = std::make_unique<ReadbackQueue>(instance);
    } else if (PrebatchEnabled() && Offload::Enabled()) {
        readback_queue = std::make_unique<ReadbackQueue>(
            instance, instance.GetGraphicsQueue(), instance.GetGraphicsQueueFamilyIndex(),
            &Vulkan::Scheduler::submit_mutex);
        LOG_INFO(Render, "ReadbackPrebatch: copies of data no unsubmitted draw writes go "
                         "ahead of the batch being recorded");
    }

    if (Offload::LatencyEnabled()) {
        StartTickPoller(scheduler.GetMasterSemaphore());
    }
    if (Offload::Enabled() || Offload::VerifyEnabled() || Offload::ProfileEnabled()) {
        LOG_INFO(Render, "ReadbackOffload: offload={} verify={} profile={}", Offload::Enabled(),
                 Offload::VerifyEnabled(), Offload::ProfileEnabled());
    }

    std::memset(gds_buffer.mapped_data.data(), 0, DataShareBufferSize);

    // SHADPS4_SIM_VRAM_MB (test switch, vk_instance.cpp CreateSimBallast): fill the ballast once
    // on the GPU so the driver makes it resident and counts it as used video memory.
    if (!instance.SimBallastBuffers().empty()) {
        {
            auto access = scheduler.AcquireCommandBuffer();
            for (const vk::Buffer ballast : instance.SimBallastBuffers()) {
                access.Handle().fillBuffer(ballast, 0, VK_WHOLE_SIZE, 0x5a5a5a5au);
            }
        }
        scheduler.Finish();
        LOG_WARNING(Render_Vulkan, "SHADPS4_SIM_VRAM_MB: ballast filled; {}",
                    Vulkan::DescribeDeviceLocalBudgets(instance.GetAllocator()));
    }

    // Set up garbage collection parameters
    if (!instance.CanReportMemoryUsage()) {
        trigger_gc_memory = DEFAULT_TRIGGER_GC_MEMORY;
        critical_gc_memory = DEFAULT_CRITICAL_GC_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(instance.GetTotalMemoryBudget());
    const s64 min_spacing_expected = device_local_memory - 1_GB;
    const s64 min_spacing_critical = device_local_memory - 512_MB;
    const s64 mem_threshold = std::min<s64>(device_local_memory, TARGET_GC_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    trigger_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_expected, min_spacing_expected),
                      DEFAULT_TRIGGER_GC_MEMORY));
    critical_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_critical, min_spacing_critical),
                      DEFAULT_CRITICAL_GC_MEMORY));
}

BufferCache::~BufferCache() {
    if (scheduler.LaneUploadOn()) {
        scheduler.LaneUploadDrain(); // q36d (review 7): FIFO jobs use page_manager / memory
    }
    scheduler.DrainRecording();
    // Not reached in the supported lifecycle. The presenter that owns this cache is a global
    // that is never reset (gnmdriver.cpp), and every exit once it exists is std::quick_exit
    // (emulator.cpp Run and Relaunch), which runs Emulator::Shutdown and no static destructor.
    // Guest threads therefore never outlive this object, and there is no teardown protocol
    // for them: an in-process teardown has to stop the guest and the command processor
    // first. One that did not must not continue into freed jobs with threads still waiting.
    const bool jobs_idle = std::ranges::all_of(
        fault_jobs, [](const FaultJob& job) { return job.state == FaultJobState::Free; });
    ASSERT_MSG(jobs_idle && fault_guest_waiters.load() == 0,
               "BufferCache destroyed with fault downloads in flight ({} guest waiters)",
               fault_guest_waiters.load());
}

namespace {
// q36 =3 verify: deferred lane-upload pages {page -> (hash, seq, ready)}; see q36b_lane_upload_thread.py.
struct Q36Page {
    u64 hash = 0, seq = 0;
    bool ready = false;
};
std::mutex q36_mu;
std::map<VAddr, Q36Page> q36_pages;
u64 q36_seq = 0;
std::atomic<u64> q36_compares{0}, q36_lost{0}, q36_dropped{0}, q36_unmapped{0}, q36_unbacked{0}, q36_raced{0};
void Q36Drop(VAddr lo, u64 size) {
    std::scoped_lock lk{q36_mu};
    auto it = q36_pages.lower_bound(lo & ~u64{0xfff});
    u64 n = 0;
    while (it != q36_pages.end() && it->first < lo + size) {
        it = q36_pages.erase(it);
        ++n;
    }
    q36_dropped.fetch_add(n, std::memory_order_relaxed);
}
int Q36Mode() {
    static const int m = [] {
        const char* e = std::getenv("SHADPS4_LANE_DEFER_UPLOAD");
        const int v = e && (e[0] == '2' || e[0] == '3') ? e[0] - '0' : 0;
        if (v == 3) {
            Core::g_backing_write_hook.store(&Q36Drop, std::memory_order_relaxed);
        }
        return v;
    }();
    return m;
}
} // namespace

void NoParticlesLog() {
    LOG_INFO(Render, "q40 NO_PARTICLES (diagnostic): skipped ASC dispatches {} card draws {}", g_np_disp.load(),
             g_np_draw.load());
}

void BufferCache::BeginGpuUnmap(VAddr device_addr, u64 size) {
    if (Q36Mode() == 3) [[unlikely]] {
        Q36Drop(device_addr, size); // q36 verify: the range's bytes change without a fault
    }
    backing_span_generation.fetch_add(1, std::memory_order_release);
    // SHADPS4_BIND_MEMO_STATS: an unmap can change ClampRangeSize and the buffers found there.
    // Rasterizer::UnmapMemory bumps again once the unmap is complete.
    BumpStructGeneration();
    if (!Offload::Enabled() || size == 0) {
        return;
    }
    // Linearizes against job admission, not against CPU memory operations. Release the gate
    // before waiting for CP: finalization may need the memory manager's shared lock.
    fault_mapping_gate.Block(device_addr, size);
    if (HasOverlappingFaultJob(device_addr, size)) {
        liverpool->SendCommand<true>([this, device_addr, size] {
            SettleOverlappingJobs(device_addr, size);
            ASSERT_MSG(!HasOverlappingFaultJob(device_addr, size),
                       "GPU unmap reentered an active readback publication");
        });
    }
    // Keep the range blocked through physical unmap and until a later EndGpuMap. Late faults
    // retain the synchronous contract instead of admitting a job that could outlive unmap.
}

void BufferCache::EndGpuMap(VAddr device_addr, u64 size) {
    BumpStructGeneration(); // SHADPS4_BIND_MEMO_STATS: a map can change ClampRangeSize.
    if (Offload::Enabled() && size != 0) {
        // MapMemory can hold the memory manager's writer lock. This must not call CP or wait
        // for a job; TryAdmit's callback only publishes bookkeeping under the same gate.
        fault_mapping_gate.Unblock(device_addr, size);
    }
}

// SHADPS4_PARTICLE_ARENA_DISCARD (2026-10-13, with SHADPS4_PARTICLE_LATE_READ): the game's particle
// arena is two 30 MiB halves at [0x807d3eb10] alternating by frame (parity [0x807d3eb38]); 0x3ee7d0
// reuses the half of two frames ago. At that point its GPU data is dead: the CPU read it at the
// previous frame's start (V1), the previous frame's chain (which reads it as old particle state)
// finished at that frame's end (0x3ee470's label wait), and nothing reads it again. Discarding it
// (no download) turns the game's writes into the half from readback-class faults (~0.6 ms each,
// a GPU round trip) into plain faults. SHADPS4_PARTICLE_ARENA_BLOCK=1 also gives write faults in
// the arena the heap span's block widening (SHADPS4_HEAP_WRITE_BLOCK).
BufferCache* g_discard_cache = nullptr;
std::atomic<VAddr> g_arena_lo{0}, g_arena_hi{0};
static const bool g_arena_block = [] {
    const char* v = std::getenv("SHADPS4_PARTICLE_ARENA_BLOCK");
    return v && v[0] == '1';
}();
static std::atomic<u64> g_discard_calls{0}, g_discard_bytes{0};
/// Opus review (F2-lite): hook calls skipped for an invalid arena base / parity, and 4 MB chunks
/// skipped because their GPU write sequence moved after the snapshot.
static std::atomic<u64> g_discard_skipped_hook{0}, g_discard_skipped_chunk{0};

// SHADPS4_HOT_PAGE_PREFETCH=1 (2026-10-13, "F2-lite"; see PrefetchHotPages): the pages guest
// threads took readback-class write faults on (InvalidateMemory's flush callback), with the
// presented frame they were last seen in. ~800 faults/s on ~30 pages (q43c): a mutex is fine.
static const bool g_hot_prefetch = EnvIs1("SHADPS4_HOT_PAGE_PREFETCH");

// q56b (region_manager.h): read-watch removals wait for queued deferred read protections first.
static const bool g_read_untrack_fence = EnvIs1("SHADPS4_READ_UNTRACK_FENCE");
static std::atomic<u64> g_untrack_fences{0}, g_untrack_fence_waits{0};
bool ReadUntrackFenceOn() {
    return g_read_untrack_fence && ProtectQueue::Enabled();
}
void FenceDeferredReadProtects() {
    // Opus review (q56b): every read-watch removal runs on the CP, and Fence() assumes one claimer that
    // nothing can push behind while it waits. Off the CP this ordering argument does not hold.
    if (!Common::FaultMonitor::t_is_cp_thread) [[unlikely]] {
        static std::atomic<u64> off_cp{0};
        if (off_cp.fetch_add(1, std::memory_order_relaxed) == 0) {
            LOG_WARNING(Render, "q56b: read-untrack fence called off the CP thread (review assumed CP only)");
        }
    }
    DEBUG_ASSERT(Common::FaultMonitor::t_is_cp_thread);
    const u64 n = g_untrack_fences.fetch_add(1, std::memory_order_relaxed) + 1;
    if (ProtectQueue::Instance().Pending()) {
        g_untrack_fence_waits.fetch_add(1, std::memory_order_relaxed);
    }
    ProtectQueue::Instance().Fence();
    if ((n & 0xffff) == 0) {
        LOG_INFO(Render, "q56b read-untrack fences: {} (of which the queue had pending protections {})", n,
                 g_untrack_fence_waits.load(std::memory_order_relaxed));
    }
}
struct HotPage {
    VAddr page;
    u64 frame;
};
static constexpr size_t HotPageSlots = 64; // the oldest is evicted
static std::mutex g_hot_mutex;
static std::array<HotPage, HotPageSlots> g_hot_pages{};
// q56 (SHADPS4_HOT_PREFETCH_AT_WRITE=1, with HOT_PAGE_PREFETCH): guest READ faults outside the particle arena make
// their page hot too (s909: the auto-exposure page 0x24bfde000 read-faults ~0.9 times per frame on Game:Main, its
// producer finished in 80%), and a dispatch whose write covers a hot page runs PrefetchHotPages right after it on the
// CP thread, so the download is recorded behind the producing dispatch in the same batch instead of at the arena hook
// (where the page is often not GPU-modified yet: 'page clean' ~2.8 per frame). g_hot_buckets: bit (page >> 20) & 63
// of every hot page, a cheap pre-filter for NoteBatchWrite (rebuilt when the set changes).
static const bool g_hot_at_write = g_hot_prefetch && EnvIs1("SHADPS4_HOT_PREFETCH_AT_WRITE");
static std::atomic<u64> g_hot_buckets{0};
static thread_local bool t_hot_write_pending = false; // CP thread (set in NoteBatchWrite, taken after the dispatch)
static thread_local VAddr t_hot_write_page = 0;        // q56c (review 2): the hot page the dispatch wrote
static std::atomic<u64> g_hot_trigger{0};               // q56: dispatches that wrote a hot page
/// 0 jobs, 1 of them on the fast lane, 2 hot pages covered, 3 KB copied; skipped: 4 page clean,
/// 5 pending job or arena, 6 cap / slot / staging, 7 other; waiter: 8 published, 9 vetoed,
/// 10 settled by a guest fault first (f2's H7 retry).
static std::atomic<u64> g_hot_n[11]{};

static void NoteHotPage(VAddr addr, u64 size) {
    const u64 frame = Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
    const VAddr first = Common::AlignDown(addr, 4096);
    // Opus review: pages in the particle arena are skipped by PrefetchHotPages (the discard
    // settles them), so they would only take hot-set entries from pages that can be prefetched.
    const VAddr arena_lo = g_arena_lo.load(std::memory_order_relaxed);
    const VAddr arena_hi = g_arena_hi.load(std::memory_order_relaxed);
    std::scoped_lock lk{g_hot_mutex};
    for (VAddr page = first; page < addr + size && page < first + 4 * 4096; page += 4096) {
        if (page >= arena_lo && page < arena_hi) {
            continue;
        }
        HotPage* slot = &g_hot_pages[0]; // the same page, else the oldest (empty: frame 0)
        for (HotPage& h : g_hot_pages) {
            if (h.page == page) {
                slot = &h;
                break;
            }
            slot = h.frame < slot->frame ? &h : slot;
        }
        *slot = {page, frame};
        if (g_hot_at_write) {
            // q56c (review 1): rebuilt from the live slots, so evicted / stale pages leave the filter.
            u64 b = 0;
            for (const HotPage& h : g_hot_pages) {
                if (h.page != 0 && h.frame + 16 >= frame) {
                    b |= u64{1} << ((h.page >> 20) & 63);
                }
            }
            g_hot_buckets.store(b, std::memory_order_relaxed);
        }
    }
}

// SHADPS4_ARENA_READ_PREFETCH=1 (q48, 2026-10-13; with SHADPS4_PARTICLE_LATE_READ,
// SHADPS4_PARTICLE_ARENA_DISCARD and HotPrefetch's prerequisites). With V1 the frame start reads
// the previous frame's half of the particle arena: 0x3edef0 on Game:Main (~770 control blocks) and
// the collision jobs 0x3fed00 on the job threads (records, sim state). Those pages are GPU-written,
// so each read faults and downloads (~0.45 ms per fault: two CP round trips and the copy; s852 city:
// the cave's two steps cost Game:Main 0.97 + 0.68 ms per frame). The pages read move little from
// frame to frame relative to the half (0x3ee7d0 allocates in instance order). So every guest read
// fault in the current half is noted by its offset in the half, and at the end of the next frame
// (sceGnmSubmitDone on Game:Main: the game's wait #2 at 0x29cd56 has passed the chain's CP-time
// label, so the chain is recorded) the same offsets of that frame's half get owner-less read jobs
// like HotPrefetch's. The f2 waiter copies the bytes and has them published before the next frame
// start reads them; a read that comes first takes f2's H7 retry (waits for the job, no new copy).
static const bool g_arena_read_prefetch = EnvIs1("SHADPS4_ARENA_READ_PREFETCH");
/// q51 SHADPS4_FAST_LANE_PRODUCER=1 (see PrepareFaultDownload) and its job count.
static const bool g_fast_lane_producer = EnvIs1("SHADPS4_FAST_LANE_PRODUCER");
static std::atomic<u64> g_fast_lane_producer_jobs{0};
/// r18: of those, the copies whose newest writer had already completed - they wait only for the
/// + 1 batch (a write stamped at bind time could land in the next batch).
static std::atomic<u64> g_fast_lane_plus1_only{0};
static constexpr u64 ArenaHalf = 0x1E00000ULL;
/// The half 0x3ee7d0 fills this frame (set by the discard hook before it runs).
static std::atomic<VAddr> g_arena_cur_half{0};
/// q50 (review): set by the hook, taken by the frame end - no prefetch for a frame whose hook
/// was skipped (bad base / parity), so a stale half is never prefetched.
static std::atomic<bool> g_arena_hook_ran{false};
struct ArenaReadPage {
    u32 page1; ///< page index in the half + 1 (0: empty)
    u64 frame;
};
static constexpr size_t ArenaReadSlots = 96; // the oldest is evicted
static std::mutex g_arena_rd_mutex;
static std::array<ArenaReadPage, ArenaReadSlots> g_arena_rd{};
/// 0 guest read faults in the current half; 1 prefetch calls, 2 jobs, 3 of them on the fast lane,
/// 4 KB, 5 learned pages covered; skipped: 6 page clean, 7 pending job, 8 cap / slot / staging,
/// 9 other; waiter: 10 published, 11 vetoed, 12 settled before the waiter (a guest fault, a discard or an overlap settle); 13 skipped: the producing batch open or not yet submitted (q50; with ARENA_PREFETCH_RETRY the no-fast-lane case moves to g_arena_retry_n[11]).
static std::atomic<u64> g_arena_n[14]{};
// SHADPS4_ARENA_PREFETCH_RETRY=1 (2026-10-14, notes/arena-prefetch-retry-design-2026-10-14.md;
// needs ARENA_READ_PREFETCH and the async submit timeline). A window the frame-end run skipped
// because the batch that wrote it was still open or not yet handed to vkQueueSubmit (13) arms a
// retry: the scheduler's submitted hook (recording worker or par-rec lane worker, after its
// submit) posts PrefetchArenaReads of the same half to the CP thread once that batch is
// submitted. The retry runs every eligibility rule again; at most 2 per frame, dropped once the
// half's discard was posted (discard hook generation + 2).
static const bool g_arena_prefetch_retry = EnvIs1("SHADPS4_ARENA_PREFETCH_RETRY");
/// The armed retry: the main-timeline value that must be submitted first (0: none). Written
/// only under g_arena_retry_mutex, stored last (release); the hook's fast path reads it alone.
static std::atomic<u64> g_arena_retry_need{0};
static std::mutex g_arena_retry_mutex; ///< the (half, gen) pair is taken whole with `need`
static VAddr g_arena_retry_half = 0;
static u64 g_arena_retry_gen = 0;
/// Discard hook calls (Game:Main, one per frame; counted only with the retry on).
static std::atomic<u64> g_arena_hook_gen{0};
/// 0 armed, 1 posted (12 of them run at once on the CP thread), 2 retries run, 3 retry jobs,
/// 4 their KB, 5 windows a retry skipped again (producer still open / unsubmitted), dropped:
/// 6 stale (the half's discard posted), 7 per-frame cap; waiter: 8 retry jobs published,
/// 9 vetoed, 10 settled before the waiter; 11 skipped: no fast lane (split from 13); r18:
/// 13 guest faults that took a pending retry job's ticket (H7), 14 of them on Game:Main.
static std::atomic<u64> g_arena_retry_n[15]{};
/// r20: the newest ranges written by the exposure metering dispatch (CP thread only), only small
/// ones (a whole-heap binding would make every heap read tolerant; such a range is not kept).
static std::array<std::pair<VAddr, VAddr>, 8> g_tolerant_writes{};
static std::array<u64, 8> g_tolerant_frames{}; ///< presented_frames when noted (review: expire)
static u32 g_tolerant_next = 0;
/// r20 counters: 0 guest reads released stale, 1 released jobs published fresh, 2 copies skipped at
/// that publish (a newer GPU write re-marked them), 3 tolerant ranges noted, 4 ranges refused as too
/// large, 5 H7 hits on a hot job that could not be released (not tolerant, changed or finished).
static std::atomic<u64> g_stale_n[6]{};

bool BufferCache::ExposureStaleRead() {
    static const bool on = [] {
        const bool v = EnvIs1("SHADPS4_EXPOSURE_STALE_READ");
        if (v) {
            LOG_INFO(Render, "SHADPS4_EXPOSURE_STALE_READ=1: exposure metering reads do not wait "
                             "for the GPU (r20)");
        }
        return v;
    }();
    return on;
}

template <typename Copies> // BufferCache::FaultCopy is private: deduced
static bool TolerantCopies(const Copies& copies) {
    if (copies.empty()) {
        return false;
    }
    const u64 now = Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
    for (const auto& c : copies) {
        bool inside = false;
        for (size_t k = 0; k < g_tolerant_writes.size(); ++k) {
            const auto& [lo, hi] = g_tolerant_writes[k];
            inside = inside || (hi != 0 && g_tolerant_frames[k] + 3 >= now && lo <= c.addr &&
                                c.addr + c.size <= hi);
        }
        if (!inside) {
            return false;
        }
    }
    return true;
}

static void ArmArenaRetry(VAddr half, u64 gen, u64 need) {
    std::scoped_lock lk{g_arena_retry_mutex};
    g_arena_retry_half = half;
    g_arena_retry_gen = gen;
    g_arena_retry_need.store(need, std::memory_order_release); // last
}

/// Takes the armed retry if `submitted` covers it (CAS need -> 0); false if none or not due.
static bool TakeArenaRetry(u64 submitted, VAddr& half, u64& gen) {
    std::scoped_lock lk{g_arena_retry_mutex};
    u64 need = g_arena_retry_need.load(std::memory_order_acquire);
    if (need == 0 || submitted < need ||
        !g_arena_retry_need.compare_exchange_strong(need, 0, std::memory_order_acq_rel)) {
        return false;
    }
    half = g_arena_retry_half;
    gen = g_arena_retry_gen;
    return true;
}

static void NoteArenaRead(VAddr addr) {
    const VAddr half = g_arena_cur_half.load(std::memory_order_relaxed);
    if (half == 0 || addr < half || addr >= half + ArenaHalf) {
        return;
    }
    g_arena_n[0].fetch_add(1, std::memory_order_relaxed);
    const u32 page1 = static_cast<u32>((addr - half) >> 12) + 1;
    const u64 frame = Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
    std::scoped_lock lk{g_arena_rd_mutex};
    ArenaReadPage* slot = &g_arena_rd[0]; // the same page, else the oldest (empty: frame 0)
    for (ArenaReadPage& h : g_arena_rd) {
        if (h.page1 == page1) {
            slot = &h;
            break;
        }
        slot = h.frame < slot->frame ? &h : slot;
    }
    *slot = {page1, frame};
}

void ArenaReadPrefetchFrameEnd() {
    BufferCache* bc = g_discard_cache;
    const VAddr half = g_arena_cur_half.load(std::memory_order_relaxed);
    if (!g_arena_read_prefetch || !bc || half == 0 || !g_arena_hook_ran.exchange(false)) {
        return;
    }
    bc->PostArenaReadPrefetch(half);
}

void BufferCache::PostArenaReadPrefetch(VAddr half) {
    if (g_arena_prefetch_retry) {
        // The generation of the discard hook that set `half` (this same thread, Game:Main).
        const u64 gen = g_arena_hook_gen.load(std::memory_order_relaxed);
        liverpool->SendCommand<false>([this, half, gen] { PrefetchArenaReads(half, gen, false); });
        return;
    }
    liverpool->SendCommand<false>([this, half] { PrefetchArenaReads(half); });
}

// SHADPS4_ARENA_PREFETCH_RETRY: Vulkan::Scheduler::submitted_hook, on the recording worker or a
// par-rec lane worker right after its vkQueueSubmit of `submitted`, outside submit_mutex.
static void ArenaRetryOnSubmit(u64 submitted) {
    const u64 need = g_arena_retry_need.load(std::memory_order_acquire);
    if (need == 0 || submitted < need) [[likely]] {
        return;
    }
    if (BufferCache* bc = g_discard_cache) {
        bc->PostArenaReadRetry(submitted);
    }
}

void BufferCache::PostArenaReadRetry(u64 submitted) {
    // On the CP thread SendCommand runs the retry inline, inside a submit: leave it armed for
    // the next worker submit instead.
    if (liverpool->IsGpuThread()) {
        return;
    }
    VAddr half = 0;
    u64 gen = 0;
    if (!TakeArenaRetry(submitted, half, gen)) {
        return; // another submit thread took it first
    }
    g_arena_retry_n[1].fetch_add(1, std::memory_order_relaxed);
    liverpool->SendCommand<false>([this, half, gen] { PrefetchArenaReads(half, gen, true); });
}

void PS4_SYSV_ABI ParticleArenaDiscardHook() {
    BufferCache* bc = g_discard_cache;
    const u64 eboot = MemoryPatcher::g_eboot_address;
    if (!bc || eboot == 0) {
        return;
    }
    const VAddr base = *reinterpret_cast<const u64*>(eboot + 0x7d3eb10ULL);
    const u64 parity = *reinterpret_cast<const u64*>(eboot + 0x7d3eb38ULL);
    constexpr u64 Half = 0x1E00000ULL;
    if (base < 0x100000000ULL || base >= 0x1000000000ULL || parity > 1) {
        g_discard_skipped_hook.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    g_arena_cur_half.store(base + parity * Half, std::memory_order_relaxed); // q48
    if (g_lane_frame_scope) {
        // q60: the end-of-frame label's value the frame fence has seen (>= F-2 for this chain F).
        u32 label = 0;
        if (!Core::Memory::Instance()->TryReadBacking(0x248000000ULL, reinterpret_cast<u8*>(&label), sizeof(label))) {
            label = *reinterpret_cast<const volatile u32*>(0x248000000ULL);
        }
        g_lane_frame_label.store(u64{label} + 1, std::memory_order_release);
    }
    if (g_arena_prefetch_retry) {
        g_arena_hook_gen.fetch_add(1, std::memory_order_relaxed); // before this half's discard post
    }
    g_arena_hook_ran.store(true, std::memory_order_relaxed);
    static bool logged = false;
    if (!logged) {
        logged = true;
        LOG_INFO(Render, "ArenaDiscard: arena {:#x}-{:#x}, first discard of half {:#x}-{:#x}", base,
                 base + 2 * Half, base + parity * Half, base + (parity + 1) * Half);
    }
    bc->PostArenaDiscard(base + parity * Half, Half, base, base + 2 * Half);
}

void BufferCache::PostArenaDiscard(VAddr lo, u64 size, VAddr arena_lo, VAddr arena_hi) {
    g_arena_lo.store(arena_lo, std::memory_order_relaxed);
    g_arena_hi.store(arena_hi, std::memory_order_relaxed);
    // Opus review (q43): nothing orders this command before the new chain's ASC records (early
    // ASC service), so snapshot each 4 MB chunk's GPU write sequence here, before the game
    // submits the new chain; the discard skips chunks a GPU write marked since.
    auto chunks = std::make_shared<std::vector<std::pair<VAddr, std::vector<u64>>>>();
    for (VAddr c = lo; c < lo + size;) {
        const VAddr end = std::min<VAddr>(Common::AlignDown(c, 4_MB) + 4_MB, lo + size);
        std::vector<u64> snap;
        memory_tracker->SnapshotGpuWriteSeq(c, end - c, snap);
        chunks->emplace_back(c, std::move(snap));
        c = end;
    }
    liverpool->SendCommand<false>([this, lo, size, chunks] { DiscardDeadGpuData(lo, size, *chunks); });
}

void BufferCache::DiscardDeadGpuData(
    VAddr lo, u64 size, const std::vector<std::pair<VAddr, std::vector<u64>>>& chunks) {
    // On the CP thread, like ResolveFaultSync: no fault job may still publish over the range.
    SettleOverlappingJobs(lo, size);
    u64 bytes = 0;
    for (size_t ci = 0; ci < chunks.size(); ++ci) {
        const VAddr c = chunks[ci].first;
        const VAddr end = ci + 1 < chunks.size() ? chunks[ci + 1].first : lo + size;
        if (!memory_tracker->GpuWriteSeqMatches(c, end - c, chunks[ci].second)) {
            g_discard_skipped_chunk.fetch_add(1, std::memory_order_relaxed); // a new GPU write
            continue;
        }
        memory_tracker->ForEachDownloadRange<false>(c, end - c, [&](u64 range_addr, u64 range_size) {
            gpu_modified_ranges.Subtract(range_addr, range_size);
            MemoryTracker::gpu_clear_epoch.fetch_add(1, std::memory_order_acq_rel);
            bytes += range_size;
        });
        memory_tracker->UnmarkRegionAsGpuModified(c, end - c);
    }
    g_discard_bytes.fetch_add(bytes, std::memory_order_relaxed);
    const u64 n = g_discard_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    if ((n & 1023) == 0) {
        LOG_INFO(Render,
                 "ArenaDiscard: {} discards, {:.2f} MB of GPU-modified state dropped per discard, "
                 "skipped: {} hook calls (bad arena base / parity), {} chunks (GPU write after the "
                 "snapshot)",
                 n, double(g_discard_bytes.load()) / double(n) / 1048576.0,
                 g_discard_skipped_hook.load(), g_discard_skipped_chunk.load());
    }
    if (g_hot_prefetch) {
        PrefetchHotPages(); // the same once-per-frame CP step
    }
}

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, InvalidateOrigin origin) {
    if (!IsRegionRegistered(device_addr, size)) {
        return;
    }
    Common::FaultMonitor::t_fault_class |= 1; // SHADPS4_GAME_MAIN_CENSUS
    // SHADPS4_HEAP_WRITE_BLOCK=<bytes> (2026-10-08, notes/zero-copy-build-2026-10-07.md sec. 5-6):
    // a guest write fault inside the game's per-frame heap span (SHADPS4_HEAP_WRITE_RANGE=lo-hi
    // hex, default 248000000-269e00000) unprotects and dirties its whole block, not one 4 KB
    // page. The game writes ~1,200 pages of the span per frame in ~130 64 KB blocks (s215), so
    // its next writes to the block's other pages fault no more. Extra dirty pages only cost
    // uploads of unchanged bytes. A block with GPU-modified pages keeps the single page (a write
    // fault must never start a download for bytes it did not touch).
    struct HeapBlock {
        u64 bytes = 0;
        VAddr lo = 0x248000000ULL, hi = 0x269e00000ULL;
    };
    static const HeapBlock heap_block = [] {
        HeapBlock c{};
        const char* value = std::getenv("SHADPS4_HEAP_WRITE_BLOCK");
        const u64 bytes = value ? std::strtoull(value, nullptr, 0) : 0;
        c.bytes = bytes >= 8192 && bytes <= (4ULL << 20) && std::has_single_bit(bytes) ? bytes : 0;
        if (const char* range = std::getenv("SHADPS4_HEAP_WRITE_RANGE")) {
            char* end = nullptr;
            const VAddr lo = std::strtoull(range, &end, 16);
            const VAddr hi = end && *end == '-' ? std::strtoull(end + 1, nullptr, 16) : 0;
            if (hi > lo) {
                c.lo = lo;
                c.hi = hi;
            }
        }
        if (c.bytes != 0) {
            LOG_INFO(Render, "SHADPS4_HEAP_WRITE_BLOCK: {} KB blocks over {:#x}-{:#x}",
                     c.bytes >> 10, c.lo, c.hi);
        }
        return c;
    }();
    // v2 (2026-10-08, Opus review of 469e9c20): the "no GPU-modified page in the block" check
    // is made by MemoryTracker::InvalidateRegion under the region lock that also marks the
    // block, so the back cannot mark a page GPU-modified between check and marking. Jobs are
    // settled over the whole block (as in v1): the block's other pages become writable without
    // a fault, so no job may still publish there afterwards.
    u64 block_bytes = 0;
    VAddr settle_addr = device_addr;
    u64 settle_size = size;
    if (heap_block.bytes != 0 && origin == InvalidateOrigin::GuestWriteFault &&
        ((device_addr >= heap_block.lo && device_addr + size <= heap_block.hi) ||
         (g_arena_block && device_addr >= g_arena_lo.load(std::memory_order_relaxed) &&
          device_addr + size <= g_arena_hi.load(std::memory_order_relaxed)))) {
        const VAddr block = Common::AlignDown(device_addr, heap_block.bytes);
        const bool in_heap = device_addr >= heap_block.lo && device_addr + size <= heap_block.hi;
        const VAddr span_lo = in_heap ? heap_block.lo : g_arena_lo.load(std::memory_order_relaxed);
        const VAddr span_hi = in_heap ? heap_block.hi : g_arena_hi.load(std::memory_order_relaxed);
        if (block >= span_lo && block + heap_block.bytes <= span_hi &&
            device_addr + size <= block + heap_block.bytes) {
            block_bytes = heap_block.bytes;
            settle_addr = block;
            settle_size = heap_block.bytes;
        }
    }
    // SHADPS4_FAULT_WRITE_SKIP_SETTLE=1: a write fault (guest or CP thread) needs no
    // pre-settle. It cannot unmap, and a job neither copies nor publishes a GPU-clean page,
    // while a GPU-dirty page takes the flush callback below, which settles in
    // ResolveFaultSync. A GPU write landing later bumps the region sequence and vetoes the job.
    // Measured 2026-09-23: this settle, reached from CP-thread DMA fast-path faults, blocked
    // the CP on the GPU for 3.9% of its wall time.
    const bool skip_fault_settle = g_fault_write_skip_settle;
    if (Offload::Enabled() && !(skip_fault_settle && origin != InvalidateOrigin::Explicit) &&
        HasOverlappingFaultJob(settle_addr, settle_size)) {
        // InvalidateRegion marks CPU state on this thread wherever it finds no GPU bit, and an
        // unmap then removes the pages. A job over this range still has a publication to
        // make, so it is finalized first, while the range is mapped and untouched. The flush
        // callback below cannot do this: it never runs for the GPU-clean part of the range.
        Offload::Add(Offload::State().invalidate_settle_hops);
        const auto settle_start = std::chrono::steady_clock::now();
        liverpool->SendCommand<true>(
            [this, settle_addr, settle_size] { SettleOverlappingJobs(settle_addr, settle_size); });
        Common::FaultMonitor::invalidate_settles.fetch_add(1, std::memory_order_relaxed);
        Common::FaultMonitor::invalidate_settle_ns.fetch_add(
            static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now() - settle_start)
                                 .count()),
            std::memory_order_relaxed);
    }
    const bool from_guest_fault = origin == InvalidateOrigin::GuestWriteFault;
    const bool offload_write = from_guest_fault && g_write_fault_offload;
    const bool widened = memory_tracker->InvalidateRegion(
        device_addr, size, from_guest_fault,
        [this, device_addr, size, offload_write, from_guest_fault] {
            Common::FaultMonitor::t_fault_class |= 2; // SHADPS4_GAME_MAIN_CENSUS: readback
            if (g_hot_prefetch && from_guest_fault) {
                NoteHotPage(device_addr, size);
            }
            ReadMemory(device_addr, size, true, offload_write);
        },
        block_bytes);
    if (block_bytes != 0) {
        (widened ? Common::FaultMonitor::heap_block_widened
                 : Common::FaultMonitor::heap_block_kept)
            .fetch_add(1, std::memory_order_relaxed);
    }
}

/// Span of the tracker pages a range touches. Publication and tracker clears act on whole
/// tracker pages, so this is the granularity jobs are compared at.
static std::pair<VAddr, VAddr> TrackerPageSpan(VAddr device_addr, u64 size) {
    return {Common::AlignDown(device_addr, TRACKER_BYTES_PER_PAGE),
            Common::AlignUp(device_addr + size, TRACKER_BYTES_PER_PAGE)};
}

bool BufferCache::HasOverlappingFaultJob(VAddr device_addr, u64 size) {
    // Fast path. The command processor raises the count, under the mutex and together with
    // the window, before it records the job's copy, and lowers it only once the job has been
    // published or vetoed. A zero read here therefore proves that every job admitted before
    // this load is finalized. One admitted after it is concurrent with this invalidation, as
    // a synchronous fault download would be. Where the range is GPU-dirty its bits are
    // intact, so the flush callback meets it in SettleOverlappingJobs; where the range is
    // GPU-clean the job copied nothing and writes nothing back there.
    if (pending_fault_jobs.load() == 0) {
        return false;
    }
    const auto [begin, end] = TrackerPageSpan(device_addr, size);
    std::scoped_lock lk{fault_window_mutex};
    for (const FaultWindow& window : fault_windows) {
        if (window.begin < end && begin < window.end) {
            return true;
        }
    }
    return false;
}

void BufferCache::CompleteEmptyWriteFault(VAddr device_addr, u64 size) {
    // The synchronous download marks CPU state only after a nonempty download. With jobs in
    // play an invalidation can legitimately arrive here with nothing left: InvalidateRegion
    // saw the GPU bits of a pending job, and that job was settled, publishing the same data,
    // before this callback collected anything. The caller is owed the marking
    // InvalidateRegion would have made itself had it found the range GPU-clean, which is the
    // state the range is now in. If GPU bits remain there is unpublished data the range set
    // does not describe; that is left exactly as the synchronous path leaves it.
    if (!memory_tracker->IsRegionGpuModified(device_addr, size)) {
        memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        Offload::Add(Offload::State().write_marks);
    }
}

/// Superset of any window ResolveFaultWindow can pick for this fault, computable before the
/// buffer lookup so overlapping jobs are settled while no Buffer reference is held.
static bool EarlyFlushEnabled();

static std::pair<VAddr, u64> FaultSettleSpan(VAddr device_addr, u64 size, bool is_write) {
    if (is_write) {
        return {device_addr, size};
    }
    const u64 window = std::max(ReadFaultWidenSize(size), DefaultReadFaultWidenSize);
    const VAddr begin = Common::AlignDown(device_addr, window);
    const VAddr end = Common::AlignUp(device_addr + size, window);
    return {begin, end - begin};
}

/// SHADPS4_WRITE_FAULT_WIDEN=2: the window a write fault downloads, and settles before that.
static std::pair<VAddr, u64> WriteFaultWindow(VAddr device_addr, u64 size) {
    const VAddr begin = Common::AlignDown(device_addr, DefaultReadFaultWidenSize);
    const VAddr end = Common::AlignUp(device_addr + size, DefaultReadFaultWidenSize);
    return {begin, end - begin};
}

/// Calls func(first, count) for each run of equal consecutive entries of sources[0, size): the
/// copies of one source buffer, which a fault job lists together in address order.
template <typename T, typename Func>
static void ForEachSourceRun(const T* sources, size_t size, Func&& func) {
    for (size_t first = 0; first < size;) {
        size_t last = first + 1;
        while (last < size && sources[last] == sources[first]) {
            ++last;
        }
        func(first, last - first);
        first = last;
    }
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write, bool offload_write) {
    const bool cp_self = liverpool->IsGpuThread();
    const auto origin = cp_self ? Offload::Origin::CpSelf : Offload::Origin::Guest;
    Offload::CountFault(origin);
    if (g_arena_read_prefetch && !is_write && !cp_self) {
        NoteArenaRead(device_addr); // q48
    }
    if (g_hot_at_write && !is_write && !cp_self) {
        NoteHotPage(device_addr, std::min<u64>(size, 4096)); // q56: read-fault pages (arena pages are skipped)
    }
    // Thread-wait column rf_guest_ms (always on): a guest thread's time in a read fault,
    // synchronous or offloaded (review 2026-10-02); command processor faults are not counted.
    struct GuestReadTime {
        bool on;
        u64 from;
        ~GuestReadTime() {
            if (on) {
                Common::FaultMonitor::rf_guest_ns.fetch_add(Offload::NowNs() - from,
                                                            std::memory_order_relaxed);
            }
        }
    } guest_read_time{!is_write && !cp_self, !is_write && !cp_self ? Offload::NowNs() : 0};
    if (!Offload::Enabled() || cp_self || (is_write && !offload_write)) {
        // A command processor fault cannot leave its own thread: the wait would land there
        // anyway, and a second step queued to it could never run while it is blocked here.
        // Write invalidations also stay synchronous: an offloaded write job could mark CPU
        // state over clean pages after a concurrent unmap. Read jobs never make that mark.
        // SHADPS4_WRITE_FAULT_OFFLOAD admits guest write faults: a write job CPU-marks only in
        // its publication, which an unmap settles while the range is still mapped, and a vetoed
        // write job's fallback is skipped once an unmap has blocked the range (FinishFaultDownload).
        liverpool->SendCommand<true>([this, device_addr, size, is_write, origin] {
            ResolveFaultSync(device_addr, size, is_write, origin);
        });
        Offload::MaybeLog();
        return;
    }

    // Guest-origin fault. The command processor records an owned staging copy and flushes;
    // this thread waits for that tick on the master semaphore only (Scheduler::Wait may flush
    // and is not callable here); the command processor then votes and publishes. Tracker
    // bits, gpu_modified_ranges and page protection stay untouched until that last step.
    AmdGpu::Liverpool* const cp = liverpool;
    Vulkan::MasterSemaphore* const semaphore = scheduler.GetMasterSemaphore();
    fault_guest_waiters.fetch_add(1);
    FaultTicket ticket{};
    const bool latency = Offload::LatencyEnabled();
    const u64 t_fault = latency ? Offload::NowNs() : 0;
    u64 t_cp = 0;
    u64 t_wait = 0;
    u64 t_woke = 0;
    bool free_at_wait = false;
    u32 passes = 0;
    const u64 spin_ns = Offload::SpinWaitNs();
    u32 early_hit = 0; // m1: the range overlaps an early ASC write (CP thread sets it)
    u64 tick_at_fault = 0; // m2
    for (u32 veto_retries = 0;; ++veto_retries) {
    for (;;) {
        const Common::FaultMonitor::SendCmdKind sc_kind{0}; // f2
        cp->SendCommand<true>([&] {
            if (latency && passes == 0) {
                t_cp = Offload::NowNs();
            }
            tick_at_fault = scheduler.CurrentTick();
            ticket = PrepareFaultDownload(device_addr, size, is_write);
            if (g_early_inv) {
                // m1: early ASC writes ring order has not reached (back thread: this lambda)
                for (u64 i = g_early_w_head; i != g_early_w_tail; ++i) {
                    const auto& w = g_early_writes[i % g_early_writes.size()];
                    if (w.lo < device_addr + size && device_addr < w.hi) {
                        early_hit = 1;
                        break;
                    }
                }
            }
            if (g_fault_guest_copy && ticket.id != 0 && !ticket.retry && ticket.job == nullptr &&
                !Offload::VerifyEnabled()) {
                // The job created for this fault, if the command processor has not settled it
                // already inside the prepare step. The slot is not reused before this fault's
                // FinishFaultDownload.
                for (FaultJob& candidate : fault_jobs) {
                    if (candidate.state == FaultJobState::Pending && candidate.id == ticket.id) {
                        ticket.job = &candidate;
                        break;
                    }
                }
            }
        });
        ++passes;
        if (ticket.id == 0) {
            fault_guest_waiters.fetch_sub(1);
            Offload::MaybeLog();
            return;
        }
        const u64 wait_start = Offload::ProfileNowNs();
        if (ticket.deferred_submit) {
            // SHADPS4_FAULT_GUEST_SUBMIT: the copy is recorded, not submitted. Submit it here
            // (or take the tick the command processor submitted meanwhile), then wait.
            ticket.tick = GuestSubmitFaultJob(*ticket.job);
        }
        if (latency) {
            t_wait = Offload::NowNs();
            free_at_wait = !ticket.on_readback_queue && semaphore->IsFree(ticket.tick);
        }
        const u64 m2_t0 = ticket.m2_main && Common::FaultMonitor::t_is_game_main ? __rdtsc() : 0;
        {
            // Site 8 (r18): a pending arena retry job's copy on its own queue. Site 9 (r19c): a
            // main-timeline ticket taken from a pending job (H7 / overlap retry, e.g. a hot-page
            // prefetch recorded in the main batch) - split from site 1, the fault's own main copy.
            const Common::FaultMonitor::GpuWaitTag tag{
                ticket.on_readback_queue
                    ? (ticket.on_retry_lane ? 8u : ticket.on_fast_lane ? 3u : 2u)
                    : (ticket.retry ? 9u : 1u),
                early_hit};
            if (ticket.on_retry_lane && Common::FaultMonitor::t_is_game_main) {
                g_arena_retry_n[14].fetch_add(1, std::memory_order_relaxed); // r18: on the waiter
            }
            if (ticket.on_readback_queue) {
                JobSemaphore(ticket.on_fast_lane, ticket.on_asc, ticket.on_retry_lane)
                    .Wait(ticket.tick);
            } else if (spin_ns != 0) {
                semaphore->SpinWait(ticket.tick, spin_ns);
            } else {
                semaphore->Wait(ticket.tick);
            }
        }
        if (latency) {
            t_woke = Offload::NowNs();
        }
        if (m2_t0 != 0) [[unlikely]] {
            M2NoteMainWait(ticket, __rdtsc() - m2_t0, tick_at_fault);
        }
        Offload::AddSince(Offload::State().guest_wait_ns, wait_start);
        if (!ticket.retry) {
            break;
        }
        // Another fault's job covered this window; publish it, then look again.
        const Common::FaultMonitor::SendCmdKind sc_kind_settle{1}; // f2
        cp->SendCommand<true>([&] { SettlePendingJob(ticket.id); });
    }
    if (g_fault_guest_copy && ticket.job != nullptr) {
        GuestCopyFaultJob(*ticket.job);
    }
    // No teardown check: the cache outlives every guest thread (see ~BufferCache).
    bool retry = false;
    const Common::FaultMonitor::SendCmdKind sc_kind_finish{2}; // f2
    cp->SendCommand<true>([&] {
        retry = FinishFaultDownload(ticket.id, device_addr, size, is_write,
                                    !is_write && veto_retries < g_fault_veto_retry);
    });
    if (!retry) {
        break;
    }
    // SHADPS4_FAULT_VETO_RETRY: vetoed, nothing published; prepare a new job.
    ticket = FaultTicket{};
    }
    fault_guest_waiters.fetch_sub(1);
    if (latency) {
        RecordFaultLatency(ticket, passes, t_fault, t_cp, t_wait, t_woke, free_at_wait,
                           device_addr);
    }
    Offload::MaybeLog();
}

void BufferCache::RecordFaultLatency(const FaultTicket& ticket, u32 passes, u64 t_fault,
                                     u64 t_cp, u64 t_wait, u64 t_woke, bool free_at_wait,
                                     VAddr device_addr) {
    auto& l = Offload::Latency();
    if (passes == 1 && ticket.mix_flush_ns != 0) {
        if (auto* timer = scheduler.BatchTimer()) {
            timer->RequestMix({.done_tick = ticket.done_tick,
                               .last_ahead = ticket.last_ahead,
                               .own_tick = ticket.own_tick,
                               .producer = ticket.producer,
                               .flush_ns = ticket.mix_flush_ns,
                               .cls = ClassifyReadbackAddr(device_addr),
                               .label_tick = ticket.label_tick});
        }
    }
    const u64 t_done = Offload::NowNs();
    const auto add = [](std::atomic<u64>& counter, u64 value) {
        counter.fetch_add(value, std::memory_order_relaxed);
    };
    if (passes == 1 && ticket.on_readback_queue) {
        add(l.side_jobs, 1);
        add(l.side_wait_ns, t_woke - t_wait);
        add(l.side_total_ns, t_done - t_fault);
        Offload::MaybeLogLatency();
        return;
    }
    if (passes != 1 || ticket.flush_end_ns == 0) {
        // Retried jobs have no single flush to time against.
        add(l.other_jobs, 1);
        Offload::MaybeLogLatency();
        return;
    }
    add(l.jobs, 1);
    add(l.hop_ns, t_cp - t_fault);
    add(l.prep_ns, ticket.prep_end_ns - t_cp);
    add(l.flush_ns, ticket.flush_end_ns - ticket.prep_end_ns);
    add(l.wait_ns, t_woke - t_wait);
    add(l.finish_ns, t_done - t_woke);
    add(l.total_ns, t_done - t_fault);
    add(l.batch_draws, ticket.batch_draws);
    add(l.inflight, ticket.inflight);
    add(l.free_at_wait, free_at_wait ? 1 : 0);
    if (g_tick_poller) {
        const u64 own_done = ticket.on_asc ? 0 : g_tick_poller->DoneNs(ticket.tick);
        const u64 prev_done = ticket.on_asc ? 0 : g_tick_poller->DoneNs(ticket.tick - 1);
        if (own_done != 0 && own_done <= t_woke + 1'000'000) {
            add(l.timed_jobs, 1);
            const u64 start = std::max(ticket.flush_end_ns, prev_done);
            add(l.ahead_ns, start - ticket.flush_end_ns);
            add(l.own_ns, own_done > start ? own_done - start : 0);
            add(l.wake_ns, t_woke > own_done ? t_woke - own_done : 0);
        }
    }
    Offload::MaybeLogLatency();
}

std::pair<VAddr, u64> BufferCache::ResolveFaultWindow(const Buffer& buffer, VAddr device_addr,
                                                      u64 size, bool is_write) {
    if (is_write) {
        // Invalidation path. The caller picked this exact range and is about to mark it CPU
        // modified, so it must not be widened.
        return {device_addr, size};
    }
    const u64 window = ReadFaultWidenSize(size);
    auto [wide_addr, wide_size] = WidenReadFault(buffer, device_addr, size, window);
    if (window != DefaultReadFaultWidenSize) {
        // Preflight the SAME page/byte intersections and 64-byte padding used
        // by DownloadBufferMemory, before it subtracts any GPU-dirty ranges.
        // Both passes run in this CP callback; GPU range additions are owned
        // by that thread. Never turn an oversized speculative transfer into
        // a dropped download or a null staging pointer.
        u64 padded_bytes = 0;
        bool fits = true;
        memory_tracker->ForEachDownloadRange<false>(
            wide_addr, wide_size, [&](u64 range_addr, u64 range_size) {
                gpu_modified_ranges.ForEachInRange(
                    range_addr, range_size, [&](VAddr begin, VAddr end) {
                        if (!fits) {
                            return;
                        }
                        const u64 bytes = end - begin;
                        if (bytes > DownloadBufferSize) {
                            fits = false;
                            return;
                        }
                        const u64 padded = (bytes + 63) & ~63ULL;
                        if (padded > DownloadBufferSize - padded_bytes) {
                            fits = false;
                        } else {
                            padded_bytes += padded;
                        }
                    });
            });
        if (!fits) {
            LOG_INFO(Render, "Read-fault window exceeded staging capacity; using 64 KiB");
            const auto fallback =
                WidenReadFault(buffer, device_addr, size, DefaultReadFaultWidenSize);
            wide_addr = fallback.first;
            wide_size = fallback.second;
        }
    }
    return {wide_addr, wide_size};
}

bool BufferCache::DownloadWindowMultiBuffer(VAddr window_addr, u64 window_size) {
    struct Part {
        Buffer* buffer;
        vk::BufferCopy copy;
    };
    boost::container::small_vector<Part, 32> parts;
    boost::container::small_vector<BufferId, 8> seen;
    boost::container::small_vector<std::pair<VAddr, VAddr>, 8> spans;
    u64 total_size_bytes = 0;
    ForEachBufferInRange(window_addr, window_size, [&](BufferId id, Buffer& buffer) {
        if (std::ranges::find(seen, id) != seen.end()) {
            return;
        }
        seen.push_back(id);
        const VAddr lo = std::max(window_addr, buffer.CpuAddr());
        const VAddr hi = std::min(window_addr + window_size, buffer.CpuAddr() + buffer.SizeBytes());
        if (hi <= lo) {
            return;
        }
        spans.emplace_back(lo, hi);
        memory_tracker->ForEachDownloadRange<false>(
            lo, hi - lo, [&](u64 range_addr, u64 range_size) {
                gpu_modified_ranges.ForEachInRange(
                    range_addr, range_size, [&](VAddr start, VAddr end) {
                        parts.push_back(Part{&buffer, vk::BufferCopy{
                                                          .srcOffset = start - buffer.CpuAddr(),
                                                          .dstOffset = total_size_bytes,
                                                          .size = end - start,
                                                      }});
                        total_size_bytes += (end - start + 63) & ~63ULL;
                    });
                gpu_modified_ranges.Subtract(range_addr, range_size);
                MemoryTracker::gpu_clear_epoch.fetch_add(1, std::memory_order_acq_rel);
                Common::FaultMonitor::CountRangeSubtract(
                    gpu_modified_ranges.Fragments());
            });
    });
    if (total_size_bytes == 0) {
        return false;
    }
    // Write faults reach here with 64 KiB windows; a larger explicit invalidation that exceeds the
    // staging ring would have failed in DownloadBufferMemory as well.
    ASSERT(total_size_bytes <= DownloadBufferSize);
    for (const Part& part : parts) {
        // A main read of each source (it was un-noted; Opus review 2026-09-28).
        part.buffer->NoteMainAccess(scheduler.CurrentTick(), false);
        if (g_range_notes) {
            part.buffer->NoteMainRange(scheduler.CurrentTick(), false,
                                       part.buffer->CpuAddr() + part.copy.srcOffset,
                                       part.copy.size, 14, 0);
        }
    }
    Common::FaultMonitor::CountReadbackDownload(total_size_bytes);
    const auto [download, offset] = download_buffer.Map(total_size_bytes);
    download_buffer.Commit();
    scheduler.EndRendering();
    {
        auto command_access = scheduler.AcquireCommandBuffer();
        const auto cmdbuf = command_access.Handle();
        if (EnvIs1("SHADPS4_DOWNLOAD_BARRIER")) {
            // As in DownloadBufferMemory: make earlier GPU writes visible to the transfer read.
            const vk::MemoryBarrier2 pre_barrier = {
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .srcAccessMask =
                    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .memoryBarrierCount = 1,
                .pMemoryBarriers = &pre_barrier,
            });
        }
        for (const Part& part : parts) {
            vk::BufferCopy copy = part.copy;
            copy.dstOffset += offset;
            cmdbuf.copyBuffer(part.buffer->buffer, download_buffer.Handle(), copy);
        }
    }
    {
        const Common::FaultMonitor::GpuWaitTag tag{4};
        scheduler.Finish();
    }
    auto* memory = Core::Memory::Instance();
    for (const Part& part : parts) {
        memory->TryWriteBackingDownload(
            std::bit_cast<u8*>(part.buffer->CpuAddr() + part.copy.srcOffset),
            download + part.copy.dstOffset, part.copy.size);
    }
    // Only the spans inside the downloaded buffers: the rest of the window may be GPU-modified
    // image memory that was not downloaded here.
    for (const auto& [lo, hi] : spans) {
        memory_tracker->UnmarkRegionAsGpuModified(lo, hi - lo);
    }
    return true;
}

/// SHADPS4_WRITE_FAULT_OVERLAP_STATS=1 (diagnostic, command-processor thread): for each write
/// fault that reaches a download, does the faulting 8-byte write land on GPU-written bytes, or only
/// share its 4 KiB page (or 64 KiB window) with them? If writes rarely hit GPU bytes, the download
/// is avoidable in principle (the CPU and GPU halves of the page could be merged instead).
void BufferCache::NoteWriteFaultOverlap(VAddr device_addr, u64 size) {
    static const bool enabled = EnvIs1("SHADPS4_WRITE_FAULT_OVERLAP_STATS");
    if (!enabled) {
        return;
    }
    static u64 faults = 0, on_gpu = 0, page_gpu = 0, page_gpu_bytes = 0, window_gpu_bytes = 0;
    static u64 same_page_as_last = 0, last_page = 0;
    u64 hit = 0, page_bytes = 0, window_bytes = 0;
    gpu_modified_ranges.ForEachInRange(device_addr, size,
                                       [&](VAddr start, VAddr end) { hit += end - start; });
    const VAddr page = Common::AlignDown(device_addr, 4096);
    gpu_modified_ranges.ForEachInRange(page, 4096,
                                       [&](VAddr start, VAddr end) { page_bytes += end - start; });
    const VAddr window = Common::AlignDown(device_addr, DefaultReadFaultWidenSize);
    gpu_modified_ranges.ForEachInRange(
        window, DefaultReadFaultWidenSize,
        [&](VAddr start, VAddr end) { window_bytes += end - start; });
    ++faults;
    on_gpu += hit != 0;
    page_gpu += page_bytes != 0;
    page_gpu_bytes += page_bytes;
    window_gpu_bytes += window_bytes;
    same_page_as_last += page == last_page;
    last_page = page;
    if ((faults & 0x3ff) == 0) {
        LOG_INFO(Render, "WriteFaultOverlap: faults={} write_on_gpu_bytes={} ({:.1f}%) page_has_gpu={} "
                         "({:.1f}%) avg_gpu_bytes/page={:.0f} avg_gpu_bytes/64KiB={:.0f} "
                         "same_page_as_previous={:.1f}%",
                 faults, on_gpu, 100.0 * double(on_gpu) / double(faults), page_gpu,
                 100.0 * double(page_gpu) / double(faults),
                 double(page_gpu_bytes) / double(faults), double(window_gpu_bytes) / double(faults),
                 100.0 * double(same_page_as_last) / double(faults));
    }
}

void BufferCache::ResolveFaultSync(VAddr device_addr, u64 size, bool is_write,
                                   Offload::Origin origin) {
    // SHADPS4_ASC_STREAM: a CP-self fault inside an ASC dispatch runs inline; its download must be
    // ordered after main-stream producers, so record it on the main stream.
    const u32 prev_stream = scheduler.ActiveStream();
    if (prev_stream == 1 && g_asc_deps) {
        // The dispatch's ASC accesses so far must be visible to this fault's main accesses.
        StampPendingAsc(scheduler.OpenAscValue());
    }
    const u64 lane_wait = scheduler.LaneWaitMain();
    scheduler.SwitchStream(0);
    SCOPE_EXIT {
        scheduler.SwitchStream(prev_stream);
        if (prev_stream == 1 && scheduler.HasLane()) {
            scheduler.LaneNeedMain(lane_wait, 4); // review S1: the record continues in a new segment
        }
    };
    if (g_lane_on) [[unlikely]] {
        // Review S5: this download runs on main; a lane segment that wrote the range comes first.
        const FaultCopy fc{device_addr, 0, size};
        const u64 w = NewestAscWriter({&fc, 1}, AscDoneBelow(2)); // q11e: only unfinished writers act
        if (w != 0 && !scheduler.GetAscSemaphore()->IsFree(w + 2)) {
            scheduler.LaneFallbackMain();
        }
    }
    if (is_write) {
        NoteWriteFaultOverlap(device_addr, size);
    } else {
        NoteDiagRead(device_addr);
    }
    const auto [settle_addr, settle_size] = FaultSettleSpan(device_addr, size, is_write);
    SettleOverlappingJobs(settle_addr, settle_size);
    Buffer& buffer = slot_buffers[FindBuffer(device_addr, size)];
    if (is_write && g_write_fault_widen == '2' && Offload::Enabled()) {
        const VAddr begin = Common::AlignDown(device_addr, DefaultReadFaultWidenSize);
        const VAddr end = Common::AlignUp(device_addr + size, DefaultReadFaultWidenSize);
        SettleOverlappingJobs(begin, end - begin);
        if (DownloadWindowMultiBuffer(begin, end - begin)) {
            Offload::CountSyncResolve(origin, Offload::SyncTiming{});
        }
    } else if (is_write && g_write_fault_widen == '1' && Offload::Enabled()) {
        // Download the read-fault window as a read (it contains the faulting range). The exact
        // write path below then finds no GPU data left and completes as an empty write fault,
        // marking only the faulting range CPU modified.
        const auto [wide_addr, wide_bytes] = ResolveFaultWindow(buffer, device_addr, size, false);
        if (wide_bytes > size) {
            SettleOverlappingJobs(wide_addr, wide_bytes);
            Offload::SyncTiming wide_timing{};
            const bool wide_downloaded = DownloadBufferMemory<false>(
                buffer, wide_addr, wide_bytes, false,
                Offload::ProfileEnabled() ? &wide_timing : nullptr);
            if (wide_downloaded) {
                Offload::CountSyncResolve(origin, wide_timing);
            }
        }
    }
    const auto [addr, bytes] = ResolveFaultWindow(buffer, device_addr, size, is_write);
    Offload::SyncTiming timing{};
    const bool downloaded = DownloadBufferMemory<false>(
        buffer, addr, bytes, is_write, Offload::ProfileEnabled() ? &timing : nullptr);
    Offload::CountSyncResolve(origin, timing);
    if (!downloaded && is_write && Offload::Enabled()) {
        // With the offload off nothing here differs from the path this was split out of.
        CompleteEmptyWriteFault(device_addr, size);
    }
}

u64 BufferCache::CollectFaultCopies(VAddr device_addr, u64 size, std::vector<FaultCopy>& copies,
                                    u64 total_size_bytes) {
    memory_tracker->ForEachDownloadRange<false>(
        device_addr, size, [&](u64 device_addr_out, u64 range_size) {
            gpu_modified_ranges.ForEachInRange(
                device_addr_out, range_size, [&](VAddr start, VAddr end) {
                    const u64 new_size = end - start;
                    copies.push_back(FaultCopy{
                        .addr = start,
                        .staging_offset = total_size_bytes,
                        .size = new_size,
                    });
                    // Same padding as DownloadBufferMemory.
                    total_size_bytes += (new_size + 63) & ~63ULL;
                });
        });
    return total_size_bytes;
}

std::unique_ptr<Buffer> BufferCache::AcquireFaultStaging() {
    if (!fault_staging_pool.empty()) {
        auto staging = std::move(fault_staging_pool.back());
        fault_staging_pool.pop_back();
        return staging;
    }
    // One per job plus one for the verifier. Never the shared download ring: it reclaims by
    // GPU tick alone and could hand these bytes out again before they are published.
    if (fault_staging_created >= MaxFaultJobs + 1) {
        return nullptr;
    }
    auto staging =
        std::make_unique<Buffer>(instance, scheduler, MemoryUsage::Download, 0,
                                 vk::BufferUsageFlagBits::eTransferDst, FaultStagingSize());
    if (staging->mapped_data.empty()) {
        LOG_ERROR(Render, "ReadbackOffload: staging buffer is not host mapped");
        return nullptr;
    }
    ++fault_staging_created;
    return staging;
}

void BufferCache::ReleaseFaultStaging(std::unique_ptr<Buffer> staging) {
    // Callers only release after the tick of the last copy into it has been reached.
    if (staging) {
        fault_staging_pool.push_back(std::move(staging));
    }
}

/// SHADPS4_READ_BURST_STATS=1 (census; command processor only): see
/// scripts/patch/read-burst-census-2026-10-02.py.
static void NoteReadBurst(VAddr addr, u64 bytes) {
    struct State {
        u64 frame{~0ULL};
        u64 first_pos{};
        u64 first_ns{};
        boost::container::small_vector<VAddr, 32> cur;
        boost::container::small_vector<VAddr, 32> prev;
        u64 frames{}, jobs{}, later{}, ready{}, touched{}, overrun{}, seen_prev{}, batchable{};
        u64 first_seen_prev{}, kb{}, batchable_kb{};
        std::array<u64, 6> after{}; // later jobs by time since the first: <0.25 <0.5 <1 <2 <4 >=4 ms
    };
    static State st;
    const u64 frame = Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
    const u64 now = Offload::NowNs();
    const bool first = frame != st.frame;
    if (first) {
        if ((st.frames & 2047) == 0 && st.frames != 0) {
            const double later = static_cast<double>(std::max<u64>(st.later, 1));
            LOG_INFO(Render,
                     "ReadBurst: frames={} jobs/frame={:.2f} kB/frame={:.0f} | later jobs: "
                     "ready={:.1f}% touched={:.1f}% overrun={:.1f}% seen_prev={:.1f}% "
                     "batchable={:.1f}% ({:.2f} per frame, {:.0f} kB per frame) | first job "
                     "seen_prev={:.1f}% | later jobs after the first: <0.25ms {:.1f}% <0.5 "
                     "{:.1f}% <1 {:.1f}% <2 {:.1f}% <4 {:.1f}% >=4 {:.1f}%",
                     st.frames, static_cast<double>(st.jobs) / st.frames,
                     static_cast<double>(st.kb) / st.frames, 100.0 * st.ready / later,
                     100.0 * st.touched / later, 100.0 * st.overrun / later,
                     100.0 * st.seen_prev / later, 100.0 * st.batchable / later,
                     static_cast<double>(st.batchable) / st.frames,
                     static_cast<double>(st.batchable_kb) / st.frames,
                     100.0 * st.first_seen_prev / st.frames, 100.0 * st.after[0] / later,
                     100.0 * st.after[1] / later, 100.0 * st.after[2] / later,
                     100.0 * st.after[3] / later, 100.0 * st.after[4] / later,
                     100.0 * st.after[5] / later);
            st.frames = st.jobs = st.later = st.ready = st.touched = st.overrun = 0;
            st.seen_prev = st.batchable = st.first_seen_prev = st.kb = st.batchable_kb = 0;
            st.after = {};
        }
        st.frame = frame;
        st.first_pos = GpuWriteLog::Position();
        st.first_ns = now;
        st.prev = st.cur;
        st.cur.clear();
        ++st.frames;
    }
    ++st.jobs;
    st.kb += bytes >> 10;
    const bool seen = std::ranges::find(st.prev, addr) != st.prev.end();
    st.cur.push_back(addr);
    if (first) {
        st.first_seen_prev += seen;
        return;
    }
    ++st.later;
    st.seen_prev += seen;
    const auto [begin, end] = TrackerPageSpan(addr, bytes);
    u64 hit_begin = 0;
    u64 hit_end = 0;
    switch (GpuWriteLog::Check(st.first_pos, begin, end, hit_begin, hit_end)) {
    case GpuWriteLog::Result::Clean:
        ++st.ready;
        if (seen) {
            ++st.batchable;
            st.batchable_kb += bytes >> 10;
        }
        break;
    case GpuWriteLog::Result::Touched:
        ++st.touched;
        break;
    case GpuWriteLog::Result::Overrun:
        ++st.overrun;
        break;
    }
    const u64 us = (now - st.first_ns) / 1000;
    ++st.after[us < 250 ? 0 : us < 500 ? 1 : us < 1000 ? 2 : us < 2000 ? 3 : us < 4000 ? 4 : 5];
}

BufferCache::FaultTicket BufferCache::PrepareFaultDownload(VAddr device_addr, u64 size,
                                                           bool is_write) {
    // SHADPS4_ASC_LANE: a lane segment deferred behind the open main batch is submitted (main
    // flushed first) before the copy list and the tick are taken.
    scheduler.LaneFlushIfDeferred();
    if (!is_write) {
        NoteDiagRead(device_addr);
    }
    ASSERT_MSG(!is_write || g_write_fault_offload,
               "Write faults use the readback offload only with SHADPS4_WRITE_FAULT_OFFLOAD");
    if (is_write) {
        NoteWriteFaultOverlap(device_addr, size);
    }
    // A write job covers the whole WIDEN=2 window, so the window is what must be free of jobs.
    const auto [settle_addr, settle_size] = is_write ? WriteFaultWindow(device_addr, size)
                                                     : FaultSettleSpan(device_addr, size, is_write);
    // SHADPS4_FAULT_OVERLAP_RETRY=1: a read fault inside a pending job's window no longer makes
    // the command processor wait on the GPU for that job. The faulting guest thread waits for
    // the job's tick itself, has it settled (no GPU wait left by then) and retries from the
    // start; the job's own owner still frees it. No overlapping job is ever created, which is
    // what the settle guaranteed. Measured 2026-09-23 with deferred protection: this settle
    // blocked the CP on the GPU for 13.75% of its wall time.
    static const bool overlap_retry = [] {
        const char* value = std::getenv("SHADPS4_FAULT_OVERLAP_RETRY");
        return value && value[0] == '1';
    }();
    if (pf_pending != 0) {
        // f2 (H7): never settle a prefetch on the back (a GPU wait): the faulting thread waits its
        // lane value, has it settled and retries.
        const auto [begin, end] = TrackerPageSpan(settle_addr, settle_size);
        for (FaultJob& job : fault_jobs) {
            if (job.state != FaultJobState::Pending || !job.prefetch) {
                continue;
            }
            const auto [job_begin, job_end] = TrackerPageSpan(job.addr, job.size);
            if (job_begin < end && begin < job_end) {
                if (ExposureStaleRead() && !job.released && !is_write && !job.on_readback_queue &&
                    !job.on_asc && !job.arena && !Offload::VerifyEnabled() &&
                    PrefetchMode() != 2) [[unlikely]] {
                    // r20: a main-batch hot-page job (its copy completes with the open batch,
                    // ~3 ms away). If every copied byte is exposure metering output, let the
                    // read through with the bytes in memory now (what a PS4 read sees before the
                    // GPU finished); the job's publish writes the fresh bytes later.
                    // Review BUG 1: only while the window holds nothing but the job's own
                    // (tolerant) GPU bytes - no GPU write since the snapshot, the same copy list
                    // now - and the batch is still unfinished (a finished one costs no wait).
                    bool releasable = TolerantCopies(job.copies) && FaultJobUnwritten(job);
                    if (releasable) {
                        std::vector<FaultCopy> now_copies;
                        u64 now_bytes = 0;
                        for (const auto& [lo, hi] : job.spans) {
                            now_bytes = CollectFaultCopies(lo, hi - lo, now_copies, now_bytes);
                        }
                        releasable = now_bytes == job.total_bytes && now_copies == job.copies;
                    }
                    if (releasable) {
                        auto* main_sem = scheduler.GetMasterSemaphore();
                        main_sem->Refresh();
                        releasable = !main_sem->IsFree(job.tick);
                    }
                    if (releasable) {
                        for (const auto& [lo, hi] : job.spans) {
                            memory_tracker->ForEachDownloadRange<false>(
                                lo, hi - lo, [&](u64 device_addr_out, u64 range_size) {
                                    gpu_modified_ranges.Subtract(device_addr_out, range_size);
                                    MemoryTracker::gpu_clear_epoch.fetch_add(
                                        1, std::memory_order_acq_rel);
                                    Common::FaultMonitor::CountRangeSubtract(
                                        gpu_modified_ranges.Fragments());
                                });
                            memory_tracker->UnmarkRegionAsGpuModified(lo, hi - lo);
                        }
                        job.released = true;
                        g_stale_n[0].fetch_add(1, std::memory_order_relaxed);
                        return FaultTicket{}; // resolved: the guest retries its access
                    }
                    g_stale_n[5].fetch_add(1, std::memory_order_relaxed);
                }
                ++pf_c[8]; // guest early
                if (job.arena && job.arena_retry) { // r18: guests waiting on a retry copy
                    g_arena_retry_n[13].fetch_add(1, std::memory_order_relaxed);
                }
                // The job's own timeline: the lane for f2, the fast lane or main for a hot-page
                // prefetch (SHADPS4_HOT_PAGE_PREFETCH), the retry queue for an r18 retry job.
                FaultTicket retry_ticket{job.id, job.tick, true, job.on_readback_queue};
                retry_ticket.on_fast_lane = job.on_fast_lane;
                retry_ticket.on_asc = job.on_asc;
                retry_ticket.on_retry_lane = job.on_retry_lane;
                return retry_ticket;
            }
        }
    }
    if (overlap_retry) {
        if (pending_fault_jobs.load(std::memory_order_relaxed) != 0) {
            const auto [begin, end] = TrackerPageSpan(settle_addr, settle_size);
            for (FaultJob& job : fault_jobs) {
                if (job.state != FaultJobState::Pending) {
                    continue;
                }
                const auto [job_begin, job_end] = TrackerPageSpan(job.addr, job.size);
                if (job_begin < end && begin < job_end) {
                    // The ticket carries the job's tick: a deferred submit has to exist first.
                    CloseGuestSubmit(job);
                    FaultTicket retry_ticket{job.id, job.tick, true, job.on_readback_queue};
                    retry_ticket.on_fast_lane = job.on_fast_lane;
                    retry_ticket.on_asc = job.on_asc;
                    retry_ticket.on_retry_lane = job.on_retry_lane;
                    return retry_ticket;
                }
            }
        }
    } else {
        SettleOverlappingJobs(settle_addr, settle_size);
    }
    const BufferId buffer_id = FindBuffer(device_addr, size);
    Buffer& buffer = slot_buffers[buffer_id];
    auto [addr, bytes] = is_write ? std::pair<VAddr, u64>{settle_addr, settle_size}
                                  : ResolveFaultWindow(buffer, device_addr, size, is_write);

    std::vector<FaultCopy> copies;
    std::vector<std::pair<VAddr, VAddr>> spans;
    boost::container::small_vector<BufferId, 8> sources; // Source buffer of each copy.
    u64 total_size_bytes = 0;
    if (is_write) {
        // DownloadWindowMultiBuffer's walk, without subtracting anything: every buffer of the
        // window, in address order, clipped to it.
        boost::container::small_vector<BufferId, 8> seen;
        ForEachBufferInRange(addr, bytes, [&](BufferId id, Buffer& window_buffer) {
            if (std::ranges::find(seen, id) != seen.end()) {
                return;
            }
            seen.push_back(id);
            const VAddr lo = std::max(addr, window_buffer.CpuAddr());
            const VAddr hi =
                std::min(addr + bytes, window_buffer.CpuAddr() + window_buffer.SizeBytes());
            if (hi <= lo) {
                return;
            }
            spans.emplace_back(lo, hi);
            total_size_bytes = CollectFaultCopies(lo, hi - lo, copies, total_size_bytes);
            sources.resize(copies.size(), id);
        });
        // The job's window is the hull of what its publication touches: the buffer spans and
        // the faulting range. Snapshot, admission and overlap checks all use it.
        VAddr hull_begin = device_addr;
        VAddr hull_end = device_addr + size;
        for (const auto& [lo, hi] : spans) {
            hull_begin = std::min(hull_begin, lo);
            hull_end = std::max(hull_end, hi);
        }
        addr = hull_begin;
        bytes = hull_end - hull_begin;
    } else {
        spans.emplace_back(addr, addr + bytes);
        total_size_bytes = CollectFaultCopies(addr, bytes, copies);
        sources.resize(copies.size(), buffer_id);
    }
    if (total_size_bytes == 0) {
        // DownloadBufferMemory returns here too, without touching the tracker.
        Offload::Add(Offload::State().prepare_empty);
        if (is_write) {
            CompleteEmptyWriteFault(device_addr, size);
        }
        return {};
    }

    FaultJob* job = nullptr;
    for (FaultJob& candidate : fault_jobs) {
        if (candidate.state == FaultJobState::Free &&
            !candidate.pf_hold.load(std::memory_order_acquire)) { // f2
            job = &candidate;
            break;
        }
    }
    std::unique_ptr<Buffer> staging;
    if (job && total_size_bytes <= FaultStagingSize()) {
        staging = AcquireFaultStaging();
    }
    if (!staging) {
        if (is_write) {
            // Nothing was mutated above and no job overlaps the window: the synchronous
            // WIDEN=2 download, exactly as a write fault without the offload takes it.
            Offload::Add(Offload::State().capacity_fallbacks);
            ResolveFaultSync(device_addr, size, is_write, Offload::Origin::Guest);
            return {};
        }
        // Out of job slots or staging, or too large for a job. Nothing was mutated above, and
        // the list just collected is not empty, so this download marks what is_write needs.
        Offload::SyncTiming timing{};
        DownloadBufferMemory<false>(buffer, addr, bytes, is_write,
                                    Offload::ProfileEnabled() ? &timing : nullptr);
        // A guest-origin synchronous download like any other, and additionally a capacity miss.
        Offload::CountSyncResolve(Offload::Origin::Guest, timing);
        Offload::Add(Offload::State().capacity_fallbacks);
        Offload::Add(Offload::State().capacity_wait_ns, timing.total_ns);
        return {};
    }

    boost::container::small_vector<vk::BufferCopy, 8> vk_copies;
    boost::container::small_vector<vk::Buffer, 8> vk_sources;
    for (size_t i = 0; i < copies.size(); ++i) {
        const Buffer& source = slot_buffers[sources[i]];
        vk_copies.push_back(vk::BufferCopy{
            .srcOffset = copies[i].addr - source.CpuAddr(),
            .dstOffset = copies[i].staging_offset,
            .size = copies[i].size,
        });
        vk_sources.push_back(source.Handle());
    }

    // Snapshot and copy list are taken in one uninterrupted step on this thread, which also
    // owns every GPU write marking, so the recorded copy matches them. The job is complete
    // before anything below can reenter.
    NoteReadbackProducer(addr, bytes);
    if (EarlyFlushEnabled()) {
        if (!readback_hot_ticks) {
            readback_hot_ticks = std::make_unique<u64[]>(1ULL << 20);
        }
        const u64 now = scheduler.CurrentTick();
        for (u64 slot = addr >> 20; slot <= ((addr + bytes - 1) >> 20) && slot < (1ULL << 20);
             ++slot) {
            readback_hot_ticks[slot] = now;
        }
    }
    const u64 id = next_fault_job_id++;
    const u64 tick = scheduler.CurrentTick();
    job->seqs.clear();
    memory_tracker->SnapshotGpuWriteSeq(addr, bytes, job->seqs);
    job->write_log_pos = GpuWriteLog::Position();
    const bool admitted = fault_mapping_gate.TryAdmit(addr, bytes, [&] {
        // No GPU, memory-manager call or wait inside this callback. Unmap either blocks this
        // admission or observes the published window and settles it before removing pages.
        job->id = id;
        job->guest_copy.store(0, std::memory_order_seq_cst);
        job->guest_submit.store(4, std::memory_order_seq_cst); // nothing deferred until recorded
        job->state = FaultJobState::Pending;
        job->abandoned = false;
        job->prefetch = false; // f2
        job->arena = false;    // q48
        job->is_write = is_write;
        job->addr = addr;
        job->size = bytes;
        job->tick = tick;
        if (!is_write) {
            NoteReadbackWindow(addr, bytes); // q7
            if (PrefetchMode() != 0) {
                PrefetchNoteWindow(addr, bytes); // f2
            }
        }
        job->total_bytes = total_size_bytes;
        job->copies = std::move(copies);
        job->spans = std::move(spans);
        job->mark_addr = device_addr;
        job->mark_size = size;
        job->staging = std::move(staging);
        std::scoped_lock lk{fault_window_mutex};
        const auto [window_begin, window_end] = TrackerPageSpan(addr, bytes);
        fault_windows[job - fault_jobs.data()] = {window_begin, window_end};
        pending_fault_jobs.fetch_add(1);
        fault_jobs_admitted.fetch_add(1, std::memory_order_acq_rel);
    });
    if (!admitted) {
        job->seqs.clear();
        ReleaseFaultStaging(std::move(staging)); // Never submitted or host-consumed.
        Offload::Add(Offload::State().mapping_fallbacks);
        Offload::Add(Offload::State().sync_fallbacks);
        Offload::g_regime.sync_fallbacks.fetch_add(1, std::memory_order_relaxed);
        ResolveFaultSync(device_addr, size, is_write, Offload::Origin::Guest);
        return {};
    }
    const vk::Buffer staging_handle = job->staging->Handle();
    Offload::Add(Offload::State().jobs);
    Offload::g_regime.jobs.fetch_add(1, std::memory_order_relaxed);
    if (is_write) {
        Offload::Add(Offload::State().write_jobs);
    }
    Common::FaultMonitor::CountReadbackDownload(total_size_bytes);
    static const bool read_burst_stats = [] {
        const char* value = std::getenv("SHADPS4_READ_BURST_STATS");
        return value && value[0] == '1';
    }();
    if (read_burst_stats && !is_write) [[unlikely]] {
        NoteReadBurst(job->addr, job->size);
    }

    job->on_readback_queue = false;
    job->on_fast_lane = false;
    job->on_asc = false;
    job->on_retry_lane = false;
    job->released = false;
    // SHADPS4_ASC_STREAM: the newest ASC-stream writer of the window and whether it completed.
    u64 asc_producer = 0;
    bool asc_done = true;
    if (scheduler.HasAscStream() || scheduler.HasLane()) {
        auto* asc_sem = scheduler.GetAscSemaphore();
        asc_producer = NewestAscWriter(job->copies, AscDoneBelow(scheduler.HasLane() ? 2 : 1)); // q11e
        if (asc_hist_evicted != 0 && !asc_sem->IsFree(asc_hist_evicted)) {
            // A history slot of an unfinished segment was overwritten: assume it wrote here.
            asc_producer = std::max(asc_producer, asc_hist_evicted);
        }
        if (asc_producer != 0) {
            // +1: a flush inside a dispatch's pass 2 reopens the segment, so the dispatch can land
            // one segment later than its history stamp (Opus review 2026-09-28).
            const u64 need = asc_producer + (scheduler.HasLane() ? 2 : 1); // lane: two cuts
            asc_done = asc_sem->IsFree(need);
            if (!asc_done) {
                asc_sem->Refresh();
                asc_done = asc_sem->IsFree(need);
            }
        }
    }
    if (g_asc_census && (scheduler.HasAscStream() || scheduler.HasLane())) { // q22: lane too
        CensusCtrlJob(job->copies, tick, asc_producer, asc_done);
    }
    if (!asc_done && fast_lane) {
        // The producer is still on the ASC stream: copy right behind it there, so the guest waits
        // for the particle work only, not for the graphics queued on the main queue. No unfinished
        // main batch may write the window (it would not be ordered before this copy).
        auto* master = scheduler.GetMasterSemaphore();
        master->Refresh();
        const u64 completed = master->KnownGpuTick();
        if (completed != 0 && completed < tick && !BatchesWriteAny(job->copies, completed, tick)) {
            scheduler.SwitchStream(1);
            ForEachSourceRun(sources.data(), sources.size(), [&](size_t first, size_t count) {
                RecordFaultCopiesAsync(slot_buffers[sources[first]], staging_handle,
                                       {vk_copies.data() + first, count});
            });
            scheduler.SwitchStream(0);
            const u64 asc_value = scheduler.SubmitAscSegment();
            if (g_lane_seg_census && scheduler.HasLane()) [[unlikely]] {
                scheduler.LaneCopyCensus(asc_value); // q33
            }
            if (g_asc_deps) {
                for (size_t k = 0; k < job->copies.size() && k < sources.size(); ++k) {
                    slot_buffers[sources[k]].NoteAscAccess(job->copies[k].addr,
                                                           job->copies[k].size, false, asc_value);
                }
            }
            job->tick = asc_value;
            job->on_readback_queue = true;
            job->on_asc = true;
            FaultTicket asc_ticket{id, asc_value, false, true};
            asc_ticket.on_asc = true;
            static std::atomic<u64> asc_jobs{0};
            if (((asc_jobs.fetch_add(1) + 1) & 0x3ff) == 0) {
                LOG_INFO(Render_Vulkan, "AscStream: {} readback copies on the ASC stream",
                         asc_jobs.load());
            }
            return asc_ticket;
        }
    }
    if (!asc_done && scheduler.HasLane()) {
        // The lane producer is unfinished and the copy stays on main: submit the open segment now
        // (not deferred: LaneFlushIfDeferred ran first) and let the copy's batch wait on it.
        scheduler.LaneSubmitIfOpen();
        VideoCore::g_main_wait_asc = std::max(VideoCore::g_main_wait_asc, asc_producer + 2);
    }
    if (fast_lane && asc_done) {
        // The newest batch that GPU-wrote any copied byte; batches before it completed first
        // (one queue), batches after it wrote none of these bytes. If it has completed, nothing
        // unfinished stands between the data and this copy.
        // Eligible when no batch from the newest completed one on wrote any copied byte: the
        // newest writer p then satisfies p + 1 <= completed (a write is stamped at bind time and
        // could be recorded into the next batch if a flush ran in between; Opus review
        // 2026-09-28). Only the unfinished batches are scanned (the CP back is the forest's
        // limiter; a 64-batch scan per fault measured -1.8% there, screen s64).
        auto* master = scheduler.GetMasterSemaphore();
        master->Refresh();
        const u64 completed = master->KnownGpuTick();
        const bool done = completed != 0 && completed < tick &&
                          !BatchesWriteAny(job->copies, completed, tick);
        u64 producer = completed;
        bool waits_producer = false;
        if (!done && g_fast_lane_producer) {
            // q51 (SHADPS4_FAST_LANE_PRODUCER=1; s870 fire: prebatched copies waited 1.2 -> 2.0
            // ms each behind the main queue's backlog). The newest batch that wrote a copied
            // byte, + 1 (a write is stamped at bind time and can land in the next batch; f2
            // review), is submitted but not complete: the copy waits on the main timeline for
            // that value only, on the fast lane, not for everything queued after it - like the
            // readback queue's side path (modes 1 / 2, which waits producer without + 1). Unsubmitted: the prebatch below.
            // Opus review (q51): always the per-batch write history (kept whenever the fast lane
            // is on); ProducerTick misses writers that only call NoteBatchWrite.
            const u64 p = PreciseProducer(job->copies) + 1;
            const VkSemaphore async_timeline = Vulkan::Scheduler::async_timeline.load();
            const bool submitted =
                async_timeline == VK_NULL_HANDLE ||
                p <= Vulkan::Scheduler::async_submitted_tick.load(std::memory_order_acquire);
            if (p > 1 && p > completed && p < tick && submitted) {
                producer = p;
                waits_producer = true;
                g_fast_lane_producer_jobs.fetch_add(1, std::memory_order_relaxed);
                if (p - 1 <= completed) { // r18 census: the writer itself has completed
                    g_fast_lane_plus1_only.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        if (done || waits_producer) {
            const auto record = [&](vk::CommandBuffer cmd) {
                ForEachSourceRun(vk_sources.data(), vk_sources.size(),
                                 [&](size_t first, size_t count) {
                                     RecordFaultCopiesOnReadbackQueue(
                                         cmd, vk_sources[first], staging_handle,
                                         {vk_copies.data() + first, count}, false);
                                 });
            };
            // SHADPS4_FAULT_GUEST_SUBMIT: record only; the owner submits before its wait and
            // the tick is unknown until then (0 here). Lane fields first: the release below
            // publishes them to the owner and to CloseGuestSubmit.
            const bool deferred =
                g_fault_guest_submit && !Offload::VerifyEnabled() && !fast_lane->InQueueOrder();
            u64 lane_tick = 0;
            if (deferred) {
                job->submit_slot = static_cast<u32>(fast_lane->RecordSlot(master->Handle(),
                                                                          producer, record));
            } else {
                lane_tick = fast_lane->Submit(master->Handle(), producer, record);
            }
            // No main-queue wait on the copy: a GPU write to the window marked before
            // publication changes its write sequence and vetoes the job (PublishFaultJob), and
            // one marked after it comes after the guest waited for this copy (Opus review).
            job->tick = lane_tick;
            job->on_readback_queue = true;
            job->on_fast_lane = true;
            if (deferred) {
                job->guest_submit.store(0, std::memory_order_seq_cst);
            }
            FaultTicket lane_ticket{id, lane_tick, false, true};
            lane_ticket.on_fast_lane = true;
            lane_ticket.deferred_submit = deferred;
            lane_ticket.job = deferred ? job : nullptr;
            if (Offload::LatencyEnabled()) {
                master->Refresh();
                lane_ticket.done_tick = master->KnownGpuTick();
                lane_ticket.last_ahead = tick - 1;
                lane_ticket.producer = producer;
                lane_ticket.mix_flush_ns = Vulkan::GpuBatchTimer::NowNs();
                lane_ticket.label_tick = Offload::g_last_gfx_label_tick.load(std::memory_order_relaxed);
            }
            if (PrebatchMode() == '2') {
                // Same batch cutting as a prebatched copy.
                scheduler.EndRendering();
                scheduler.Flush();
            }
            static std::atomic<u64> lane_jobs{0};
            if (((lane_jobs.fetch_add(1) + 1) & 0x3ff) == 0) {
                LOG_INFO(Render_Vulkan,
                         "ReadbackFastLane: {} jobs on the fast lane ({} waiting on their producer, "
                         "q51; {} of them only for the batch after a completed writer, r18)",
                         lane_jobs.load(), g_fast_lane_producer_jobs.load(std::memory_order_relaxed),
                         g_fast_lane_plus1_only.load(std::memory_order_relaxed));
            }
            return lane_ticket;
        }
    }
    if (readback_queue && asc_done) {
        // Every GPU write into this window was recorded no later than `producer`. If that batch
        // is already submitted, copy on the readback queue after it alone.
        // SHADPS4_READBACK_PREBATCH: the queue is the main queue, so submission order already
        // puts the copy after every submitted write; it only has to avoid bytes the batch still
        // being recorded writes, which BatchWritesAny checks exactly.
        const bool prebatch = readback_queue->InQueueOrder();
        const u64 producer = prebatch                   ? 0
                             : PreciseProducerEnabled() ? PreciseProducer(job->copies)
                                                        : ProducerTick(addr, bytes);
        // With SHADPS4_ASYNC_SUBMIT, `producer < tick` means the CP flushed that batch, not that
        // the recording worker has called vkQueueSubmit for it; only wait on signals already
        // submitted (the scheduler guards the same-queue case likewise). Review 2026-09-27.
        const VkSemaphore async_timeline = Vulkan::Scheduler::async_timeline.load();
        const bool producer_submitted =
            async_timeline == VK_NULL_HANDLE ||
            producer <= Vulkan::Scheduler::async_submitted_tick.load(std::memory_order_acquire);
        const bool side_ok = prebatch ? !BatchWritesAny(job->copies)
                                      : (producer != 0 && producer < tick && producer_submitted);
        if (side_ok) {
            static std::atomic<u64> side{0}, total{0};
            u64 side_tick = 0;
            bool deferred = false;
            if (prebatch && AsyncFaultCopies()) {
                // Asynchronous submission: batches recorded before this point may not be
                // submitted yet, so the recording worker submits the copy when it gets here.
                side_tick = readback_queue->ReserveTick();
                std::vector<vk::BufferCopy> owned(vk_copies.begin(), vk_copies.end());
                std::vector<vk::Buffer> owned_src(vk_sources.begin(), vk_sources.end());
                // SHADPS4_PAR_REC: submitted after every earlier batch (queue order).
                scheduler.RecordOrdered([this, rq = readback_queue.get(), side_tick, staging_handle,
                                  owned = std::move(owned),
                                  owned_src = std::move(owned_src)](vk::CommandBuffer) {
                    rq->SubmitReserved(side_tick, [&](vk::CommandBuffer cmd) {
                        ForEachSourceRun(owned_src.data(), owned_src.size(),
                                         [&](size_t first, size_t count) {
                                             RecordFaultCopiesOnReadbackQueue(
                                                 cmd, owned_src[first], staging_handle,
                                                 {owned.data() + first, count}, true);
                                         });
                    });
                });
                scheduler.PublishRecording();
            } else {
                const auto record = [&](vk::CommandBuffer cmd) {
                    ForEachSourceRun(vk_sources.data(), vk_sources.size(),
                                     [&](size_t first, size_t count) {
                                         RecordFaultCopiesOnReadbackQueue(
                                             cmd, vk_sources[first], staging_handle,
                                             {vk_copies.data() + first, count}, prebatch);
                                     });
                };
                // SHADPS4_FAULT_GUEST_SUBMIT, own-queue lane only (as on the fast lane). In
                // queue order the copy must precede the batch being recorded, and DeleteBuffer
                // relies on that order.
                deferred = g_fault_guest_submit && !Offload::VerifyEnabled() && !prebatch;
                if (deferred) {
                    job->submit_slot = static_cast<u32>(readback_queue->RecordSlot(
                        scheduler.GetMasterSemaphore()->Handle(), producer, record));
                } else {
                    side_tick = readback_queue->Submit(scheduler.GetMasterSemaphore()->Handle(),
                                                       producer, record);
                }
            }
            job->tick = side_tick;
            job->on_readback_queue = true;
            if (deferred) {
                job->guest_submit.store(0, std::memory_order_seq_cst);
            }
            FaultTicket side_ticket{id, side_tick, false, true};
            side_ticket.deferred_submit = deferred;
            side_ticket.job = deferred ? job : nullptr;
            if (Offload::LatencyEnabled()) {
                auto* semaphore = scheduler.GetMasterSemaphore();
                semaphore->Refresh();
                side_ticket.done_tick = semaphore->KnownGpuTick();
                side_ticket.last_ahead = tick - 1;
                side_ticket.producer = PreciseProducer(job->copies);
                side_ticket.mix_flush_ns = Vulkan::GpuBatchTimer::NowNs();
                side_ticket.label_tick = Offload::g_last_gfx_label_tick.load(std::memory_order_relaxed);
            }
            if (prebatch && PrebatchMode() == '2') {
                // After the copy's submission, so the copy does not wait for this batch.
                scheduler.EndRendering();
                scheduler.Flush();
            }
            const u64 n = total.fetch_add(1) + 1;
            side.fetch_add(1);
            if ((n & 0x3ff) == 0) {
                LOG_INFO(Render_Vulkan, "ReadbackQueue: {} of {} jobs on the readback queue",
                         side.load(), n);
            }
            return side_ticket;
        }
        static std::atomic<u64> main_jobs{0};
        const u64 m = main_jobs.fetch_add(1) + 1;
        if (prebatch && (m & 0x3ff) == 0) {
            LOG_INFO(Render_Vulkan, "ReadbackPrebatch: {} jobs stayed in the recorded batch", m);
        }
    }

    scheduler.EndRendering();
    if (AsyncFaultCopies()) {
        ForEachSourceRun(sources.data(), sources.size(), [&](size_t first, size_t count) {
            RecordFaultCopiesAsync(slot_buffers[sources[first]], staging_handle,
                                   {vk_copies.data() + first, count});
        });
    } else {
        auto command_access = scheduler.AcquireCommandBuffer();
        ForEachSourceRun(sources.data(), sources.size(), [&](size_t first, size_t count) {
            RecordFaultCopies(command_access.Handle(), slot_buffers[sources[first]],
                              staging_handle, {vk_copies.data() + first, count});
        });
    }
    // m2: the newest writer of these bytes in the write history (the batch this copy waits for).
    const auto m2_fill = [&](FaultTicket& t) {
        t.m2_main = true;
        t.m2_addr = job->copies.empty() ? 0 : job->copies.front().addr;
        for (const FaultCopy& c : job->copies) {
            t.m2_bytes += c.size;
        }
        for (u64 back = 0; back < WriteHistory && back < tick; ++back) {
            const u64 wt = tick - back;
            const u64 i = wt % WriteHistory;
            if (write_hist_tick[i] != wt) {
                continue;
            }
            const auto& hist = write_hist[i];
            for (size_t k = hist.size(); k-- > 0;) {
                bool hit = false;
                for (const FaultCopy& c : job->copies) {
                    hit = hit || (hist[k].first < c.addr + c.size && c.addr < hist[k].second);
                }
                if (hit) {
                    t.m2_writer_tick = wt;
                    if (k < write_hist_tag[i].size()) {
                        t.m2_kind = write_hist_tag[i][k].first;
                        t.m2_id = write_hist_tag[i][k].second;
                    }
                    return;
                }
            }
        }
    };
    if (Offload::LatencyEnabled()) {
        FaultTicket ticket{.id = id, .tick = tick};
        m2_fill(ticket);
        ticket.producer = PreciseProducer(job->copies);
        scheduler.GetMasterSemaphore()->Refresh();
        ticket.done_tick = scheduler.GetMasterSemaphore()->KnownGpuTick();
        ticket.last_ahead = tick - 1;
        ticket.own_tick = tick;
        ticket.prep_end_ns = Offload::NowNs();
        ticket.mix_flush_ns = Vulkan::GpuBatchTimer::NowNs();
        ticket.label_tick = Offload::g_last_gfx_label_tick.load(std::memory_order_relaxed);
        scheduler.Flush();
        ticket.flush_end_ns = Offload::NowNs();
        ticket.batch_draws = scheduler.LastBatchDraws();
        const u64 known = scheduler.GetMasterSemaphore()->KnownGpuTick();
        ticket.inflight = tick > known + 1 ? static_cast<u32>(tick - known - 1) : 0;
        return ticket;
    }
    scheduler.Flush();
    FaultTicket main_ticket{id, tick};
    m2_fill(main_ticket);
    return main_ticket;
}

void BufferCache::M2NoteMainWait(const FaultTicket& t, u64 wait_tsc, u64 tick_at_fault) {
    struct Agg {
        u64 n{}, tsc{}, behind{};
    };
    static std::mutex mu;
    static std::map<std::tuple<u64, u64, u8, u64>, Agg> agg;
    static u64 t0 = __rdtsc();
    static u64 total_n = 0, total_tsc = 0, no_writer = 0;
    std::scoped_lock lk{mu};
    auto& a = agg[{t.m2_addr, t.m2_bytes, t.m2_kind, t.m2_id}];
    ++a.n;
    a.tsc += wait_tsc;
    a.behind += t.m2_writer_tick ? tick_at_fault - std::min(tick_at_fault, t.m2_writer_tick) : 0;
    ++total_n;
    total_tsc += wait_tsc;
    no_writer += t.m2_writer_tick == 0;
    const u64 now = __rdtsc();
    if (now - t0 < 35'000'000'000ull) {
        return;
    }
    const double sec = double(now - t0) / 3.5e9;
    std::vector<std::pair<u64, std::tuple<u64, u64, u8, u64>>> v;
    for (const auto& [k, x] : agg) {
        v.push_back({x.tsc, k});
    }
    std::sort(v.rbegin(), v.rend());
    std::string s;
    for (size_t i = 0; i < std::min<size_t>(v.size(), 8); ++i) {
        const auto& [addr, bytes, kind, id] = v[i].second;
        const auto& x = agg[v[i].second];
        s += fmt::format(" [{:#x} {} B, writer kind {} id {:#x}: {:.1f}/s, {:.2f} ms each, writer {:.1f} "
                         "batches behind the fault]",
                         addr, bytes, kind, id, double(x.n) / sec, double(x.tsc) / 3.5e6 / double(x.n),
                         double(x.behind) / double(x.n));
    }
    LOG_INFO(Render_Vulkan,
             "m2 main-semaphore fault downloads on Game:Main: {:.1f}/s, {:.1f} ms/s, no writer in "
             "history {} | by window and writer (kind: 1 draw 2 gfx dispatch 3 async dispatch on main "
             "4 DMA 5 WRITE_DATA 6 join 7 image->buffer 8 fill 255 untagged):{}",
             double(total_n) / sec, double(total_tsc) / 3.5e6 / sec, no_writer, s);
    agg.clear();
    t0 = now;
    total_n = total_tsc = no_writer = 0;
}

void BufferCache::RecordFaultCopiesOnReadbackQueue(vk::CommandBuffer cmdbuf, vk::Buffer source,
                                                    vk::Buffer staging,
                                                    std::span<const vk::BufferCopy> copies,
                                                    bool in_queue_order) {
    // The producer's main-timeline value is waited on at the transfer stage, which makes its
    // writes visible here. Main-queue access tracking of `source` is left untouched.
    // In queue order (prebatch) there is no semaphore wait: a barrier's first scope covers
    // every earlier submission on the queue, and the trailing one keeps later submissions'
    // writes of `source` from overtaking this read.
    if (in_queue_order) {
        const vk::MemoryBarrier2 before{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &before,
        });
    }
    cmdbuf.copyBuffer(source, staging, copies);
    if (in_queue_order) {
        const vk::MemoryBarrier2 after{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eNone,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eNone,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &after,
        });
    }
    const vk::BufferMemoryBarrier2 host_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = staging,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &host_barrier,
    });
}

void BufferCache::RecordFaultCopies(vk::CommandBuffer cmdbuf, Buffer& source, vk::Buffer staging,
                                    std::span<const vk::BufferCopy> copies) {
    source.NoteMainAccess(scheduler.CurrentTick(), false);
    if (g_lane_probe && Common::FaultMonitor::t_is_cp_thread &&
        !(g_lane_copy_nostamp && Vulkan::Scheduler::t_lane_mode)) [[unlikely]] { // q35 H1
        for (const auto& copy : copies) {
            const VAddr lo = source.CpuAddr() + copy.srcOffset;
            LaneProbeRead(lo, lo + copy.size, scheduler.CurrentTick(), 10, 11); // q9 census
        }
    }
    if (g_range_notes) {
        for (const auto& copy : copies) {
            source.NoteMainRange(scheduler.CurrentTick(), false, source.CpuAddr() + copy.srcOffset,
                                 copy.size, 10, 0);
        }
    }
    // A nested verification can run while a binding's barriers have been collected but not
    // recorded. The tracked access may already describe that future binding, so use an
    // explicit all-writes source dependency rather than trusting it for this readback.
    const vk::BufferMemoryBarrier2 source_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = source.Handle(),
        .offset = 0,
        .size = source.SizeBytes(),
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &source_barrier,
    });
    // Keep access tracking consistent for the closing dependency and subsequent users.
    (void)source.GetBarrier(vk::AccessFlagBits2::eTransferRead,
                            vk::PipelineStageFlagBits2::eTransfer);
    cmdbuf.copyBuffer(source.Handle(), staging, copies);
    // The timeline wait establishes completion; this dependency also makes transfer writes
    // available to the host domain before mapped bytes are read (including by the verifier).
    const vk::BufferMemoryBarrier2 host_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = staging,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &host_barrier,
    });
    if (const auto barrier = source.GetBarrier(
            vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            vk::PipelineStageFlagBits2::eAllCommands)) {
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &*barrier,
        });
    }
}

/// SHADPS4_ASYNC_SUBMIT=1 also records a guest fault's copies through the recording worker
/// instead of draining it for a synchronous borrow (measured 2026-09-24: that drain was 2.8% of
/// the CP's wall time). Access tracking of the source is still updated here, in order.
bool BufferCache::AsyncFaultCopies() {
    return g_async_submit && scheduler.IsAsyncRecording();
}

void BufferCache::RecordFaultCopiesAsync(Buffer& source, vk::Buffer staging,
                                         std::span<const vk::BufferCopy> copies) {
    scheduler.NoteAscWork(2); // G0
    source.NoteMainAccess(scheduler.CurrentTick(), false);
    if (g_lane_probe && Common::FaultMonitor::t_is_cp_thread &&
        !(g_lane_copy_nostamp && Vulkan::Scheduler::t_lane_mode)) [[unlikely]] { // q35 H1
        for (const auto& copy : copies) {
            const VAddr lo = source.CpuAddr() + copy.srcOffset;
            LaneProbeRead(lo, lo + copy.size, scheduler.CurrentTick(), 10, 11); // q9 census
        }
    }
    if (g_range_notes) {
        for (const auto& copy : copies) {
            source.NoteMainRange(scheduler.CurrentTick(), false, source.CpuAddr() + copy.srcOffset,
                                 copy.size, 10, 0);
        }
    }
    const vk::Buffer src = source.Handle();
    const u64 src_size = source.SizeBytes();
    (void)source.GetBarrier(vk::AccessFlagBits2::eTransferRead,
                            vk::PipelineStageFlagBits2::eTransfer);
    const auto closing = source.GetBarrier(
        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        vk::PipelineStageFlagBits2::eAllCommands);
    const bool has_closing = closing.has_value();
    const vk::BufferMemoryBarrier2 closing_barrier = closing.value_or(vk::BufferMemoryBarrier2{});
    scheduler.RecordArrays(
        [src, src_size, staging, has_closing,
         closing_barrier](vk::CommandBuffer cmdbuf, std::span<const vk::BufferCopy> list) {
            const vk::BufferMemoryBarrier2 source_barrier{
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = src,
                .offset = 0,
                .size = src_size,
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .bufferMemoryBarrierCount = 1,
                .pBufferMemoryBarriers = &source_barrier,
            });
            cmdbuf.copyBuffer(src, staging, list);
            const vk::BufferMemoryBarrier2 host_barrier{
                .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eHost,
                .dstAccessMask = vk::AccessFlagBits2::eHostRead,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = staging,
                .offset = 0,
                .size = VK_WHOLE_SIZE,
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .bufferMemoryBarrierCount = 1,
                .pBufferMemoryBarriers = &host_barrier,
            });
            if (has_closing) {
                cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                    .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                    .bufferMemoryBarrierCount = 1,
                    .pBufferMemoryBarriers = &closing_barrier,
                });
            }
        },
        copies);
}

static bool EarlyFlushEnabled() {
    return g_readback_early_flush;
}

static bool ProducerStatsEnabled() {
    return g_readback_producer_stats;
}

namespace {
/// q7: recently read-back windows, packed (first page << 24 | pages), 0 = empty.
std::array<std::atomic<u64>, 16> g_rb_hot{};
std::array<std::atomic<bool>, 16> g_rb_armed{}; // q7c: a read-back since the last cut
std::atomic<u32> g_rb_hot_next{0};
bool EarlyCutOn() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_READBACK_EARLY_CUT");
        return e && (e[0] == '1' || e[0] == '2'); // q7b: 2 = once per dispatch run
    }();
    return on;
}
void NoteReadbackWindow(VAddr addr, u64 size) {
    if (!EarlyCutOn() || size == 0) {
        return;
    }
    const u64 first = addr >> 12;
    const u64 pages = std::min<u64>(((addr + size + 4095) >> 12) - first, (u64{1} << 24) - 1);
    const u64 packed = (first << 24) | pages;
    for (u32 i = 0; i < g_rb_hot.size(); ++i) {
        if (g_rb_hot[i].load(std::memory_order_relaxed) == packed) {
            g_rb_armed[i].store(true, std::memory_order_relaxed); // q7c: re-armed
            return;
        }
    }
    const u32 i = g_rb_hot_next.fetch_add(1, std::memory_order_relaxed) & 15;
    g_rb_hot[i].store(packed, std::memory_order_relaxed);
    g_rb_armed[i].store(true, std::memory_order_relaxed);
}
/// q7c: the armed entries the range hits, as a mask (0: none).
u32 HitsReadbackWindow(VAddr addr, u64 size) {
    const u64 lo = addr >> 12, hi = (addr + size + 4095) >> 12;
    u32 mask = 0;
    for (u32 i = 0; i < g_rb_hot.size(); ++i) {
        if (!g_rb_armed[i].load(std::memory_order_relaxed)) {
            continue;
        }
        const u64 v = g_rb_hot[i].load(std::memory_order_relaxed);
        if (v != 0) {
            const u64 first = v >> 24, end = first + (v & 0xffffff);
            if (lo < end && first < hi) {
                mask |= 1u << i;
            }
        }
    }
    return mask;
}
} // namespace

void BufferCache::DisarmEarlyCut(u32 mask) {
    for (u32 i = 0; i < g_rb_armed.size(); ++i) {
        if ((mask >> i) & 1) {
            g_rb_armed[i].store(false, std::memory_order_relaxed);
        }
    }
}

thread_local bool t_in_dispatch = false;
thread_local bool t_tolerant_writer = false;


namespace {
// q9 census (back thread): lane producer hits by writer, kept-on-main by reason, logged every ~5 s.
struct WhoCount {
    u64 n{}, open{};
};
std::unordered_map<u64, WhoCount> g_who_hits;
std::unordered_map<u64, u64> g_who_kept;
u64 g_who_log_tsc = 0;
u64 g_who_bindings = 0, g_who_written = 0;
u64 WhoKey(u32 kind, u64 id) {
    return (u64{kind & 0xff} << 56) | (id & 0x00ffffffffffffffull);
}
void LaneWhoLog() {
    std::vector<std::pair<u64, WhoCount>> h(g_who_hits.begin(), g_who_hits.end());
    std::sort(h.begin(), h.end(), [](const auto& a, const auto& b) { return a.second.n > b.second.n; });
    u64 tot = 0, tot_open = 0;
    std::string top;
    for (size_t i = 0; i < h.size(); ++i) {
        tot += h[i].second.n;
        tot_open += h[i].second.open;
        if (i < 12) {
            top += fmt::format(" [k{} {:08x} n={} open={}]", h[i].first >> 56,
                               h[i].first & 0x00ffffffffffffffull, h[i].second.n, h[i].second.open);
        }
    }
    LOG_INFO(Render_Vulkan, "LaneWho hits (~5 s): lane bindings {} written {} producer hits {} open {}:{}",
             g_who_bindings, g_who_written, tot, tot_open, top);
    std::vector<std::pair<u64, u64>> k(g_who_kept.begin(), g_who_kept.end());
    std::sort(k.begin(), k.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    std::string kt;
    u64 ktot = 0;
    for (size_t i = 0; i < k.size(); ++i) {
        ktot += k[i].second;
        if (i < 12) {
            kt += fmt::format(" [r{} {:08x} n={}]", k[i].first >> 56, k[i].first & 0x00ffffffffffffffull,
                              k[i].second);
        }
    }
    LOG_INFO(Render_Vulkan, "LaneWho kept on main (~5 s): {}:{}", ktot, kt);
    std::string pr;
    for (u32 i = 0; i < g_lprobe_n.load(std::memory_order_relaxed); ++i) {
        pr += fmt::format(" [{:#x}+{:#x}]", g_lprobe[i].lo.load(std::memory_order_relaxed),
                          g_lprobe[i].hi.load(std::memory_order_relaxed) -
                              g_lprobe[i].lo.load(std::memory_order_relaxed));
    }
    const auto& d = g_lprobe_dist;
    const auto& l = g_label_dist;
    LOG_INFO(Render_Vulkan,
             "LaneWho pool probe (~5 s): written lane bindings over a probe {} | open - newest read "
             "0:{} 1:{} 2:{} 3:{} 4+:{} done:{} never:{} | read after newest gfx label {} | label "
             "distance 0:{} 1:{} 2:{} 3:{} 4+:{} done:{} | stamps A2 {} other {} | probes{} | WAR "
             "scoped {} label {} scoped-open {} stamp-all {} (cumulative)",
             d[8], d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], l[0], l[1], l[2], l[3], l[4], l[5],
             d[9], d[10], pr, g_war_scoped.load(), g_war_label.load(), g_war_open.load(),
             g_war_dma_all.load());
    g_who_hits.clear();
    g_who_kept.clear();
    g_lprobe_dist.fill(0);
    g_label_dist.fill(0);
    g_who_bindings = g_who_written = 0;
}
} // namespace

void LaneWhoNote(u64 t, u64 open, bool written, u64 rd, bool hit) {
    ++g_who_bindings;
    if (t != 0) {
        auto& c = g_who_hits[WhoKey(g_lc_hit_wk, g_lc_hit_wid)];
        ++c.n;
        c.open += t >= open ? 1 : 0;
    }
    if (written) {
        ++g_who_written;
        if (hit) {
            ++g_lprobe_dist[8];
            const auto bucket = [&](u64 v) -> u32 {
                return v <= g_lc_done ? 5u : static_cast<u32>(std::min<u64>(open - std::min(open, v), 4));
            };
            g_lprobe_dist[rd == 0 ? 6 : bucket(rd)]++;
            if (rd > g_gfx_label_tick) {
                ++g_lprobe_dist[7];
            }
            g_label_dist[bucket(g_gfx_label_tick)]++;
        }
    }
    const u64 now = __rdtsc();
    if (now - g_who_log_tsc > 17'500'000'000ULL) {
        g_who_log_tsc = now;
        LaneWhoLog();
    }
}

void LaneWhoKept(u32 reason, u64 hash) {
    ++g_who_kept[WhoKey(reason, hash)];
}

/// q9c: the WAR wait of a lane write of [lo, hi): scoped to the newest main read of its probe + 1 (a read
/// stamped at lookup may be recorded into the next batch), deferred if that is the open batch; the newest
/// gfx label batch (capped) when no probe covers the range.
/// q60b census (SHADPS4_LANE_WAIT_CENSUS): distinct lane writes outside the arena, with the pipeline being bound.
u64 g_q60_census_id = 0;
static void LaneOutsideArenaNote(u64 lo, u64 hi, VAddr alo, VAddr ahi, const char* via) {
    static std::vector<std::array<u64, 3>> seen;
    static std::array<u64, 4> cls{}; // fully below, overlaps the arena, fully above, no arena yet
    const u32 c = alo >= ahi ? 3 : hi <= alo ? 0 : lo >= ahi ? 2 : 1;
    ++cls[c];
    const std::array<u64, 3> key{lo, hi - lo, g_q60_census_id};
    if (seen.size() < 64 && std::find(seen.begin(), seen.end(), key) == seen.end()) {
        seen.push_back(key);
        LOG_INFO(Render_Vulkan, "q60b OutsideArena: lane write {:#x}-{:#x} size {:#x} ({} MiB) pipeline {:08x} via {} | "
                 "arena {:#x}-{:#x} | so far below {} overlapping {} above {} no-arena {}", lo, hi, hi - lo,
                 (hi - lo) >> 20, g_q60_census_id, via, alo, ahi, cls[0], cls[1], cls[2], cls[3]);
    }
}

/// q60: SHADPS4_LANE_FRAME_SCOPE - a lane write inside the particle arena waits for the frame-fence frame's end only.
static bool LaneFrameScopedWar(Vulkan::Scheduler& sch, u64 lo, u64 hi, const char* via = "dma") {
    if (!g_lane_frame_scope) {
        return false;
    }
    const VAddr alo = g_arena_lo.load(std::memory_order_relaxed), ahi = g_arena_hi.load(std::memory_order_relaxed);
    if (alo >= ahi || lo < alo || hi > ahi) {
        g_lane_frame_c[2].fetch_add(1, std::memory_order_relaxed);
        if (g_lane_wait_census) [[unlikely]] {
            LaneOutsideArenaNote(lo, hi, alo, ahi, via);
        }
        return false;
    }
    u64 tick = 0;
    if (!LaneFrameTick(tick)) {
        g_lane_frame_c[g_lane_frame_label.load(std::memory_order_relaxed) == 0 ? 3 : 1].fetch_add(
            1, std::memory_order_relaxed);
        return false;
    }
    const u64 n = g_lane_frame_c[0].fetch_add(1, std::memory_order_relaxed) + 1;
    if ((n & 0x3ffff) == 0) {
        LOG_INFO(Render_Vulkan, "q60 LaneFrameScope: {} arena writes scoped to the frame-fence frame end, fell back: {} "
                 "(frame end not recorded yet), {} outside the arena, {} before the first hook",
                 n, g_lane_frame_c[1].load(), g_lane_frame_c[2].load(), g_lane_frame_c[3].load());
    }
    g_lane_war_exact = 13; // q59 census: frame scoped
    g_lane_war_tsc = 0;
    sch.LaneNeedMain(tick, 2, false, LaneSiteWar(40));
    return true;
}

void LaneWarWait(Vulkan::Scheduler& sch, u64 lo, u64 hi) {
    if (LaneFrameScopedWar(sch, lo, hi)) {
        return; // q60
    }
    const u64 open = sch.CurrentTick();
    u64 rd = 0;
    bool hit = false;
    if (g_lane_war && hi > lo && LaneProbeWrite(lo, hi, open, rd, hit)) {
        g_war_scoped.fetch_add(1, std::memory_order_relaxed);
        const u64 w = rd != 0 ? std::min(rd + (g_lane_win == 2 ? 0 : 1), open) : 0; // q26 E3-lite
        if (w >= open) {
            g_war_open.fetch_add(1, std::memory_order_relaxed);
        }
        sch.LaneNeedMain(w, 2, false, LaneSiteWar(g_lane_war_site), g_lane_war_wid);
        return;
    }
    g_war_label.fetch_add(g_lane_war ? 1 : 0, std::memory_order_relaxed);
    sch.LaneNeedMain(g_gfx_label_tick, 2, true);
}

void BufferCache::NoteBatchWrite(VAddr addr, u64 size) {
    if (size == 0) {
        return;
    }
    if (t_tolerant_writer && Common::FaultMonitor::t_is_cp_thread) [[unlikely]] {
        // r20: the exposure metering output (CP thread only: the ring has no lock).
        const bool small = size <= (u64{64} << 10);
        if (small) {
            const u32 k = g_tolerant_next++ % g_tolerant_writes.size();
            g_tolerant_writes[k] = {addr, addr + size};
            g_tolerant_frames[k] =
                Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
        }
        const u64 n = g_stale_n[small ? 3 : 4].fetch_add(1, std::memory_order_relaxed) + 1;
        if (g_stale_n[3].load(std::memory_order_relaxed) + g_stale_n[4].load(std::memory_order_relaxed) <= 4 ||
            (n & 0x3ff) == 0) {
            LOG_INFO(Render,
                     "ExposureStaleRead: metering write {:#x}+{:#x} ({}) | reads released {} | "
                     "released jobs published {} copies re-marked {} | ranges noted {} refused {} | "
                     "hot jobs not released {}",
                     addr, size, small ? "kept" : "too large, not tolerant", g_stale_n[0].load(),
                     g_stale_n[1].load(), g_stale_n[2].load(), g_stale_n[3].load(),
                     g_stale_n[4].load(), g_stale_n[5].load());
        }
    }
    if (g_hot_at_write && t_in_dispatch && !t_hot_write_pending &&
        Common::FaultMonitor::t_is_cp_thread) [[unlikely]] {
        // q56: does this dispatch write a hot page (outside the arena)? Bucket pre-filter, then the slots.
        const VAddr arena_lo = g_arena_lo.load(std::memory_order_relaxed);
        const VAddr arena_hi = g_arena_hi.load(std::memory_order_relaxed);
        const u64 buckets = g_hot_buckets.load(std::memory_order_relaxed);
        if (buckets != 0 && !(addr >= arena_lo && addr + size <= arena_hi)) {
            u64 mask = 0;
            if (size >= (u64{64} << 20)) {
                mask = ~u64{0};
            } else {
                for (VAddr b = addr >> 20; b <= (addr + size - 1) >> 20; ++b) {
                    mask |= u64{1} << (b & 63);
                }
            }
            if (mask & buckets) {
                const u64 now = Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
                std::scoped_lock lk{g_hot_mutex};
                for (const HotPage& h : g_hot_pages) {
                    if (h.page != 0 && h.frame + 16 >= now && h.page < addr + size && addr < h.page + 4096) {
                        t_hot_write_pending = true;
                        t_hot_write_page = h.page;
                        break;
                    }
                }
            }
        }
    }
    if (EarlyCutOn() && Common::FaultMonitor::t_is_cp_thread && t_in_dispatch) [[unlikely]] {
        if (const u32 m = HitsReadbackWindow(addr, size)) {
            early_cut_pending = true; // q7; q7c: dispatch writes only, armed windows only
            early_cut_mask |= m;
        }
    }
    if (g_lane_census || g_lane_on) [[unlikely]] {
        if (Common::FaultMonitor::t_is_cp_thread) {
            const bool lane_side =
                g_lane_on ? Vulkan::Scheduler::t_lane_mode : t_force_binding_sync;
            LaneNoteRange(addr, addr + size, scheduler.CurrentTick(), lane_side ? 2 : 0, census_kind,
                          census_id);
        } else {
            g_lc[13].fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (g_record_stream == 1) {
        // SHADPS4_ASC_STREAM: keyed by the open ASC segment's value, not a main tick.
        const u64 value = scheduler.OpenAscValue();
        const u64 i = value % asc_write_hist.size();
        if (asc_write_hist_value[i] != value) {
            asc_hist_evicted = std::max(asc_hist_evicted, asc_write_hist_value[i]);
            asc_write_hist[i].clear();
            asc_write_hist_value[i] = value;
            asc_write_bbox[i] = {~VAddr{0}, 0}; // q11e
        }
        // The particle chain rebinds the same written windows dispatch after dispatch: skip a
        // range the last few entries already cover (the history is scanned on every fault).
        auto& hist = asc_write_hist[i];
        const VAddr end = addr + size;
        for (size_t k = hist.size() > 8 ? hist.size() - 8 : 0; k < hist.size(); ++k) {
            if (hist[k].first <= addr && end <= hist[k].second) {
                return;
            }
        }
        hist.emplace_back(addr, end);
        asc_write_bbox[i].first = std::min(asc_write_bbox[i].first, addr); // q11e
        asc_write_bbox[i].second = std::max(asc_write_bbox[i].second, end);
        return;
    }
    if (PreciseProducerEnabled() || Offload::LatencyEnabled() || FastLaneEnabled()) {
        const u64 tick = scheduler.CurrentTick();
        const u64 i = tick % WriteHistory;
        if (write_hist_tick[i] != tick) {
            write_hist[i].clear();
            write_hist_tag[i].clear();
            write_hist_tick[i] = tick;
        }
        write_hist[i].emplace_back(addr, addr + size);
        if (g_asc_census) {
            write_hist_tag[i].emplace_back(census_kind, census_id);
        }
    }
    if (!PrebatchEnabled()) {
        return;
    }
    const u64 tick = scheduler.CurrentTick();
    if (tick != batch_writes_tick) {
        batch_writes.clear();
        batch_writes_tick = tick;
    }
    batch_writes.emplace_back(addr, addr + size);
}

Vulkan::MasterSemaphore& BufferCache::JobSemaphore(bool on_fast_lane, bool on_asc,
                                                   bool on_retry_lane) {
    if (on_retry_lane) {
        return retry_lane->Semaphore();
    }
    return on_asc ? *scheduler.GetAscSemaphore() : LaneOf(on_fast_lane).Semaphore();
}

u64 BufferCache::AscDoneBelow(u64 margin) {
    static const bool fast = [] {
        const char* env = std::getenv("SHADPS4_ASC_HIST_FAST");
        return !env || env[0] != '0';
    }();
    if (!fast || g_asc_census) {
        return 0;
    }
    const u64 known = scheduler.GetAscSemaphore()->KnownGpuTick();
    return known > margin ? known - margin : 0;
}

u64 BufferCache::NewestAscWriter(std::span<const FaultCopy> copies, u64 done_below) const {
    // Slots hold consecutive ASC values (value % size): walk from the newest value down and
    // return the first writer (was a scan of all 64 slots: 11% of the CP back with ASC_TEXEL,
    // run threads-stack-fire-ascrange-on-20260928-a).
    u64 top = 0;
    for (size_t i = 0; i < asc_write_hist.size(); ++i) {
        top = std::max(top, asc_write_hist_value[i]);
    }
    VAddr clo = ~VAddr{0}, chi = 0; // q11e: the copies' bounding box
    for (const FaultCopy& copy : copies) {
        clo = std::min(clo, copy.addr);
        chi = std::max(chi, copy.addr + copy.size);
    }
    for (u64 v = top; v != 0 && v > done_below && top - v < asc_write_hist.size(); --v) {
        const size_t i = v % asc_write_hist.size();
        if (asc_write_hist_value[i] != v) {
            continue;
        }
        if (done_below != 0 &&
            (asc_write_bbox[i].first >= chi || clo >= asc_write_bbox[i].second)) {
            continue; // q11e: no copy can overlap this slot's ranges
        }
        for (const auto& [begin, end] : asc_write_hist[i]) {
            for (const FaultCopy& copy : copies) {
                if (begin < copy.addr + copy.size && copy.addr < end) {
                    return v;
                }
            }
        }
    }
    return 0;
}

bool BufferCache::BatchesWriteAny(std::span<const FaultCopy> copies, u64 first,
                                  u64 last) const {
    // A batch missing from the history (older than WriteHistory or never recorded) counts as a
    // writer, so the answer is conservative.
    for (u64 t = first; t <= last; ++t) {
        const u64 i = t % WriteHistory;
        if (last - t >= WriteHistory) {
            return true;
        }
        if (write_hist_tick[i] != t) {
            continue; // No GPU write was noted in batch t.
        }
        for (const auto& [begin, end] : write_hist[i]) {
            for (const FaultCopy& copy : copies) {
                if (begin < copy.addr + copy.size && copy.addr < end) {
                    return true;
                }
            }
        }
    }
    return false;
}

u64 BufferCache::PreciseProducer(std::span<const FaultCopy> copies) const {
    const u64 current = scheduler.CurrentTick();
    for (u64 back = 0; back < WriteHistory && back < current; ++back) {
        const u64 t = current - back;
        const u64 i = t % WriteHistory;
        if (write_hist_tick[i] != t) {
            continue;
        }
        for (const auto& [begin, end] : write_hist[i]) {
            for (const FaultCopy& copy : copies) {
                if (begin < copy.addr + copy.size && copy.addr < end) {
                    return t;
                }
            }
        }
    }
    // Older than the history: complete long ago, or at worst ordered by an earlier wait.
    return current > WriteHistory ? current - WriteHistory : 0;
}

bool BufferCache::BatchWritesAny(std::span<const FaultCopy> copies) {
    if (batch_writes_tick != scheduler.CurrentTick()) {
        return false;
    }
    for (const FaultCopy& copy : copies) {
        const VAddr begin = copy.addr;
        const VAddr end = copy.addr + copy.size;
        for (const auto& [write_begin, write_end] : batch_writes) {
            if (write_begin < end && begin < write_end) {
                return true;
            }
        }
    }
    return false;
}

bool BufferCache::ReadProducerStatsEnabled() {
    static const bool enabled = EnvIs1("SHADPS4_READ_PRODUCER_STATS");
    return enabled;
}

void BufferCache::NoteDiagRead(VAddr addr) {
    if (!ReadProducerStatsEnabled()) {
        return;
    }
    const auto it = diag_window_writer.find(addr >> 16);
    const u64 producer = it == diag_window_writer.end() ? 0 : it->second;
    ++diag_read_hist[producer];
    // SHADPS4_READ_FAULT_RIP=1: one line per call for the offline join with the guest-side
    // "ReadFaultRip" lines (page_manager.cpp); a fault can reach here more than once.
    static const bool per_fault = EnvIs1("SHADPS4_READ_FAULT_RIP");
    if (per_fault && diag_reads < 400000) {
        // Was the last writable binding over this window already complete on the GPU? If so the
        // readback only waits for queue order, not for its data.
        const auto tick_it = diag_window_tick.find(addr >> 16);
        const u64 ptick = tick_it == diag_window_tick.end() ? 0 : tick_it->second;
        const u64 current = scheduler.CurrentTick();
        const bool done = ptick != 0 && scheduler.IsFree(ptick);
        if (producer == 0xc73c37c1ULL) { // the 2026-09-27 question; writers of its windows only
            ++diag_faulted_windows[addr >> 16];
        }
        LOG_INFO(Render,
                 "ReadProducerFault: k={} addr={:#x} producer={:#x} ptick={} cur={} gpu={} done={}",
                 diag_reads, addr, producer, ptick, current,
                 scheduler.GetMasterSemaphore()->KnownGpuTick(), done ? 1 : 0);
    }
    if ((++diag_reads & 0x3ff) != 0) {
        return;
    }
    std::vector<std::pair<u64, u64>> top(diag_read_hist.begin(), diag_read_hist.end());
    std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    std::string line;
    for (size_t i = 0; i < std::min<size_t>(top.size(), 10); ++i) {
        line += fmt::format(" {:#x}={}", top[i].first, top[i].second);
    }
    LOG_INFO(Render, "ReadProducer: {} read faults, {} producers, windows tracked {}; top:{}", diag_reads,
             top.size(), diag_window_writer.size(), line);
}

bool BufferCache::CensusEnabled() {
    return g_asc_census;
}

std::pair<u8, u64> BufferCache::NewestMainWriterTag(std::span<const FaultCopy> copies, u64 first,
                                                    u64 last) const {
    for (u64 t = last; t >= first && t != 0; --t) {
        if (last - t >= WriteHistory) {
            return {255, 0};
        }
        const u64 i = t % WriteHistory;
        if (write_hist_tick[i] != t) {
            continue;
        }
        const auto& hist = write_hist[i];
        const auto& tags = write_hist_tag[i];
        for (size_t k = hist.size(); k-- > 0;) {
            for (const FaultCopy& copy : copies) {
                if (hist[k].first < copy.addr + copy.size && copy.addr < hist[k].second) {
                    return k < tags.size() ? tags[k] : std::pair<u8, u64>{0, 0};
                }
            }
        }
    }
    return {0, 0};
}

/// Census (a): control-block fault jobs (0x270000000-0x2701fffff) by why they did (not) take the
/// ASC copy path: producer not on the ASC stream, ASC producer already done, ASC producer
/// unfinished and blocked by an unfinished main writer (by kind and pipeline), or ASC path taken.
void BufferCache::CensusCtrlJob(std::span<const FaultCopy> copies, u64 tick, u64 asc_producer,
                                bool asc_done) {
    if (copies.empty() || copies.front().addr < 0x270000000ULL ||
        copies.front().addr >= 0x270200000ULL) {
        return;
    }
    struct Census {
        u64 jobs = 0, not_asc = 0, asc_done = 0, blocked = 0, asc_path = 0;
        std::array<u64, 256> by_kind{};
        std::unordered_map<u64, u64> by_id;
    };
    static Census c;
    ++c.jobs;
    if (asc_producer == 0) {
        ++c.not_asc;
    } else if (asc_done) {
        ++c.asc_done;
    } else {
        auto* master = scheduler.GetMasterSemaphore();
        const u64 completed = master->KnownGpuTick();
        if (completed != 0 && completed < tick && BatchesWriteAny(copies, completed, tick)) {
            ++c.blocked;
            const auto [kind, id] = NewestMainWriterTag(copies, completed, tick);
            ++c.by_kind[kind];
            ++c.by_id[(u64(kind) << 56) ^ id];
        } else {
            ++c.asc_path;
        }
    }
    if ((c.jobs & 0xff) != 0) {
        return;
    }
    std::string kinds;
    for (size_t k = 0; k < c.by_kind.size(); ++k) {
        if (c.by_kind[k] != 0) {
            kinds += fmt::format(" k{}={}", k, c.by_kind[k]);
        }
    }
    std::vector<std::pair<u64, u64>> top(c.by_id.begin(), c.by_id.end());
    std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    std::string ids;
    for (size_t k = 0; k < std::min<size_t>(top.size(), 6); ++k) {
        ids += fmt::format(" k{}:{:08x}={}", top[k].first >> 56, top[k].first & 0x00ffffffffffffffULL,
                           top[k].second);
    }
    LOG_INFO(Render, "AscCensusCtrl: jobs={} producer_not_asc={} asc_done={} blocked_by_main={} "
                     "asc_path={} blockers:{} top:{}",
             c.jobs, c.not_asc, c.asc_done, c.blocked, c.asc_path, kinds, ids);
}

void BufferCache::CensusAscBinding(BufferId buffer_id, VAddr addr, u64 size, bool written) {
    struct Entry {
        VAddr base = 0;
        u64 buffer_size = 0, n = 0, written = 0, max_size = 0;
    };
    static std::unordered_map<u32, Entry> by_buffer;
    static u64 bindings = 0, in_texel_span = 0, written_in_texel_span = 0;
    ++bindings;
    constexpr VAddr TexelLo = 0x248002000ULL;
    constexpr VAddr TexelHi = 0x248002000ULL + 0x21dfe000ULL;
    if (addr < TexelHi && TexelLo < addr + size) {
        ++in_texel_span;
        written_in_texel_span += written;
    }
    if (buffer_id) {
        const Buffer& buffer = slot_buffers[buffer_id];
        MarkAscBound(buffer_id);
        auto& e = by_buffer[buffer_id.index];
        e.base = buffer.CpuAddr();
        e.buffer_size = buffer.SizeBytes();
        ++e.n;
        e.written += written;
        e.max_size = std::max(e.max_size, size);
    }
    if ((bindings & 0xffff) != 0) {
        return;
    }
    std::vector<Entry> top;
    for (const auto& [id, e] : by_buffer) {
        top.push_back(e);
    }
    std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.n > b.n; });
    std::string line;
    for (size_t k = 0; k < std::min<size_t>(top.size(), 8); ++k) {
        line += fmt::format(" [{:#x}+{:#x} n={} w={} max={:#x}]", top[k].base, top[k].buffer_size,
                            top[k].n, top[k].written, top[k].max_size);
    }
    LOG_INFO(Render, "AscCensusBind: bindings={} buffers={} in_texel_span={} written_in_span={} top:{}",
             bindings, by_buffer.size(), in_texel_span, written_in_texel_span, line);
}

bool BufferCache::RangeShadowEnabled() {
    return g_asc_range_shadow;
}

bool BufferCache::AscRangeEnabled() {
    return g_asc_range;
}

bool BufferCache::AscDepsEnabled() {
    return g_asc_deps;
}

void BufferCache::MarkAscBound(BufferId buffer_id) const {
    const Buffer& buffer = slot_buffers[buffer_id];
    if (buffer.asc_bound) {
        return;
    }
    buffer.asc_bound = true;
    const u64 now = scheduler.CurrentTick();
    if (buffer.main_access_tick >= now) {
        buffer.NoteMainWhole(now, 13);
    }
}

u8 BufferCache::RangeConflict(BufferId buffer_id, VAddr addr, u64 size, bool written, u64 now,
                              u64& id) const {
    id = 0;
    const Buffer& buffer = slot_buffers[buffer_id];
    const auto* log = buffer.range_log.get();
    if (!log || log->tick != now) {
        // No range was noted in the open batch (the Buffer may have been bound by an ASC
        // candidate only after its main accesses, which then count as whole-Buffer).
        return buffer.main_access_tick >= now ? 254 : 0;
    }
    if (log->whole) {
        id = log->whole_kind; // 6 JoinOverlap, 100+kind overflow
        return 255;
    }
    const VAddr end = addr + size;
    // Pass 2 uploads whole tracker pages: the upload-WAR test uses the page-rounded range.
    const VAddr page_lo = Common::AlignDown(addr, TRACKER_BYTES_PER_PAGE);
    const VAddr page_hi = Common::AlignUp(end, TRACKER_BYTES_PER_PAGE);
    bool page_overlap = false;
    for (const auto& e : log->entries) {
        if (e.lo < end && addr < e.hi && (e.write || written)) {
            id = e.id;
            return e.kind == 0 ? 253 : e.kind;
        }
        page_overlap |= e.lo < page_hi && page_lo < e.hi;
    }
    // Upload WAR (design step 4): if the pages are CPU-dirty, the ASC dispatch's pass 2 uploads
    // them on the ASC stream, which runs before the open batch's accesses to those pages.
    if (page_overlap && memory_tracker->IsRegionCpuModified(page_lo, page_hi - page_lo)) {
        return 12;
    }
    return 0;
}

void BufferCache::NoteGpuWrite(VAddr addr, u64 size) {
    NoteBatchWrite(addr, size);
    if (ReadProducerStatsEnabled() && size != 0) [[unlikely]] {
        const u64 first = addr >> 16;
        const u64 last = std::min((addr + size - 1) >> 16, first + 1023);
        for (u64 w = first; w <= last; ++w) {
            diag_window_writer[w] = diag_producer;
        }
        static const bool per_fault = EnvIs1("SHADPS4_READ_FAULT_RIP");
        if (per_fault) {
            const u64 tick = scheduler.CurrentTick();
            bool covers_faulted = false;
            for (u64 w = first; w <= last; ++w) {
                diag_window_tick[w] = tick;
                covers_faulted |= diag_faulted_windows.contains(w);
            }
            if (covers_faulted) {
                const u64 seen = diag_fault_window_writers[{diag_producer, addr, size}]++;
                if ((seen & (seen + 1)) == 0) { // seen = 0, 1, 3, 7, ...
                    LOG_INFO(Render,
                             "ReadFaultWindowWriter: producer={:#x} addr={:#x} size={:#x} "
                             "seen={} tick={}",
                             diag_producer, addr, size, seen, tick);
                }
            }
            static std::unordered_map<u64, u64> writes_per_producer; // CP back thread only
            const u64 count = writes_per_producer[diag_producer]++;
            if (count < 64 || (count & 0xfff) == 0) {
                LOG_INFO(Render, "ReadProducerWrite: producer={:#x} n={} addr={:#x} size={:#x}",
                         diag_producer, count, addr, size);
            }
        }
    }
    if ((!ProducerStatsEnabled() && !readback_queue && !EarlyFlushEnabled()) || size == 0 ||
        PreciseProducerEnabled()) {
        return;
    }
    if (!producer_ticks) {
        producer_ticks = std::make_unique<u64[]>(1ULL << 20);
    }
    const u64 tick = scheduler.CurrentTick();
    const u64 first = addr >> 20;
    const u64 last = (addr + size - 1) >> 20;
    if (EarlyFlushEnabled() && readback_hot_ticks && !readback_flush_requested) {
        // A slot read back within the last ~64 submissions is likely read back again; submit
        // this write soon so the readback finds it complete (and can use the readback queue).
        for (u64 slot = first; slot <= std::min(last, first + 64) && slot < (1ULL << 20);
             ++slot) {
            if (readback_hot_ticks[slot] != 0 && readback_hot_ticks[slot] + 64 > tick) {
                readback_flush_requested = true;
                break;
            }
        }
    }
    if (last - first > 64 || last >= (1ULL << 20)) {
        // Too wide to record per slot: every readback until the next submission sees it.
        producer_floor = tick;
        return;
    }
    for (u64 slot = first; slot <= last; ++slot) {
        producer_ticks[slot] = tick;
    }
}

u64 BufferCache::ProducerTick(VAddr addr, u64 size) const {
    u64 producer = producer_floor;
    if (!producer_ticks || size == 0) {
        return producer;
    }
    for (u64 slot = addr >> 20; slot <= ((addr + size - 1) >> 20) && slot < (1ULL << 20);
         ++slot) {
        producer = std::max(producer, producer_ticks[slot]);
    }
    return producer;
}

void BufferCache::WaitFaultJob(FaultJob& job) {
    if (job.on_readback_queue) {
        CloseGuestSubmit(job); // SHADPS4_FAULT_GUEST_SUBMIT: the tick exists only after this
        JobSemaphore(job.on_fast_lane, job.on_asc, job.on_retry_lane).Wait(job.tick);
        return;
    }
    // Main only: a fault copy recorded on main waits its producers explicitly; a dual wait would
    // tie Game:Main to all ASC work before the batch (asc-i1-relax design, R0 table).
    if (!scheduler.IsFreeMain(job.tick)) {
        scheduler.WaitMain(job.tick);
    }
}

void BufferCache::NoteReadbackProducer(VAddr addr, u64 size) {
    if (!ProducerStatsEnabled() || !producer_ticks || size == 0) {
        return;
    }
    const u64 producer = ProducerTick(addr, size);
    static u64 jobs = 0, done = 0, in_current = 0, unknown = 0, age_sum = 0;
    const u64 current = scheduler.CurrentTick();
    ++jobs;
    if (producer == 0) {
        ++unknown;
    } else if (producer >= current) {
        ++in_current;
    } else {
        age_sum += current - producer;
        done += scheduler.IsFree(producer);
    }
    if ((jobs & 0x3ff) == 0) {
        LOG_INFO(Render_Vulkan,
                 "ReadbackProducer: jobs={} producer_done_at_fault={} ({:.1f}%) producer_in_"
                 "unsubmitted_batch={} unknown={} mean_age_ticks={:.1f}",
                 jobs, done, 100.0 * done / jobs, in_current, unknown,
                 jobs - in_current - unknown ? double(age_sum) / (jobs - in_current - unknown)
                                             : 0.0);
    }
}

void BufferCache::SettlePendingJob(u64 id) {
    for (FaultJob& job : fault_jobs) {
        if (job.state != FaultJobState::Pending || job.id != id) {
            continue;
        }
        job.state = FaultJobState::Settling;
        if (g_fault_guest_copy) {
            // A Settling job is skipped by nested settles: close the owner's copy now, so
            // nothing the wait below could reenter can be followed by a late guest copy.
            CloseGuestCopy(job);
        }
        Offload::Add(Offload::State().cp_settles);
        const u64 wait_start = Offload::ProfileNowNs();
        WaitFaultJob(job);
        Offload::AddSince(Offload::State().cp_settle_wait_ns, wait_start);
        FinalizeFaultJob(job);
        return;
    }
}

bool BufferCache::FinishFaultDownload(u64 id, VAddr device_addr, u64 size, bool is_write,
                                      bool allow_retry) {
    FaultJob* job = nullptr;
    for (FaultJob& candidate : fault_jobs) {
        if (candidate.state != FaultJobState::Free && candidate.id == id) {
            job = &candidate;
            break;
        }
    }
    bool vetoed = true;
    bool clean_veto = false; // the job itself ended Vetoed (not missing, not abandoned)
    if (!job) {
        // Only teardown frees a job its owner has not consumed. Tracking is intact by
        // construction, so the synchronous path below is still correct.
        LOG_ERROR(Render, "ReadbackOffload: job {} missing at finish", id);
    } else {
        if (job->state == FaultJobState::Pending) {
            job->state = FaultJobState::Settling;
            // The owner already waited for this tick, so this is not expected to block.
            const u64 wait_start = Offload::ProfileNowNs();
            WaitFaultJob(*job);
            Offload::AddSince(Offload::State().cp_settle_wait_ns, wait_start);
            FinalizeFaultJob(*job);
        }
        if (job->state == FaultJobState::Settling) {
            // Only reachable nested inside a settle of this job, which the command queue
            // does not do. Let that settle free the slot and resolve synchronously here.
            LOG_ERROR(Render, "ReadbackOffload: job {} finished while settling", id);
            job->abandoned = true;
        } else {
            // A job the command processor settled earlier stays in Success or Vetoed until
            // this point, so removal after a veto is never mistaken for success.
            vetoed = job->state == FaultJobState::Vetoed;
            clean_veto = vetoed;
            job->state = FaultJobState::Free;
        }
    }
    if (clean_veto && allow_retry) {
        // The job is Free again, its staging and window were released when it was finalized and
        // the tracker is untouched: the caller's next PrepareFaultDownload starts from scratch.
        const u64 n = g_fault_veto_retries.fetch_add(1, std::memory_order_relaxed) + 1;
        if ((n & 0x3ff) == 0) {
            LOG_INFO(Render, "FaultVetoRetry: {} vetoed jobs retried asynchronously", n);
        }
        return true;
    }
    if (vetoed) {
        if (is_write && !fault_mapping_gate.TryAdmit(device_addr, size, [] {})) {
            // A write job was admitted with this range unblocked, so an unmap has begun over it
            // since. That unmap's own invalidation flushes and CPU-marks the whole range while
            // it is still mapped; a fallback here could CPU-mark it after the unmap. If the
            // guest's write finds the page still protected it faults again, and that fault is
            // not admitted as a job and resolves synchronously.
            Offload::Add(Offload::State().write_map_skips);
            return false;
        }
        // Exactly one synchronous fallback per vetoed job (after SHADPS4_FAULT_VETO_RETRY's
        // asynchronous attempts, if any).
        Offload::Add(Offload::State().sync_fallbacks);
        Offload::g_regime.sync_fallbacks.fetch_add(1, std::memory_order_relaxed);
        ResolveFaultSync(device_addr, size, is_write, Offload::Origin::Guest);
    }
    return false;
}

void BufferCache::SettleOverlappingJobs(VAddr device_addr, u64 size) {
    if (pending_fault_jobs.load(std::memory_order_relaxed) == 0) {
        return;
    }
    // Downloads and tracker clears act on whole tracker pages, so compare page spans.
    const auto [begin, end] = TrackerPageSpan(device_addr, size);
    for (;;) {
        FaultJob* found = nullptr;
        for (FaultJob& job : fault_jobs) {
            if (job.state != FaultJobState::Pending) {
                continue;
            }
            const auto [job_begin, job_end] = TrackerPageSpan(job.addr, job.size);
            if (job_begin < end && begin < job_end) {
                found = &job;
                break;
            }
        }
        if (!found) {
            return;
        }
        // Leave Pending before anything that can reenter, then rescan from the start. The
        // slot array never moves and a Settling slot is not reused, so the pointer holds.
        // This runs on the command processor and needs no callback from it, which is what a
        // command processor fault during a guest job requires.
        found->state = FaultJobState::Settling;
        if (g_fault_guest_copy) {
            CloseGuestCopy(*found); // as in SettlePendingJob
        }
        Offload::Add(Offload::State().cp_settles);
        const u64 wait_start = Offload::ProfileNowNs();
        WaitFaultJob(*found);
        Offload::AddSince(Offload::State().cp_settle_wait_ns, wait_start);
        FinalizeFaultJob(*found);
    }
}

void BufferCache::FinalizeFaultJob(FaultJob& job) {
    // Precondition: state is Settling and the GPU has reached job.tick.
    // Before anything reads or releases the staging: wait for or forestall the owner's copy.
    const bool guest_copied = g_fault_guest_copy && CloseGuestCopy(job);
    bool ok = true;
    if (job.prefetch && PrefetchMode() == 2 && !Offload::VerifyEnabled()) {
        // f2 =2: a prefetch publishes only if a fresh readback equals its bytes (the waiter did
        // not copy; PublishFaultJob copies after the compare).
        const auto r = VerifyFaultJob(job);
        ++pf_c[r == VerifyResult::Matched ? 9 : r == VerifyResult::Mismatch ? 10 : 11];
        ok = r == VerifyResult::Matched;
    }
    if (Offload::VerifyEnabled()) {
        // Only a completed comparison lets a job through. Anything else is a veto, which
        // leaves the dirty state for the synchronous fallback, so a verified run contains no
        // publication that was not checked.
        ok = VerifyFaultJob(job) == VerifyResult::Matched;
    }
    if (ok) {
        ok = PublishFaultJob(job, guest_copied);
    }
    ReleaseFaultStaging(std::move(job.staging));
    job.copies.clear();
    job.spans.clear();
    job.seqs.clear();
    {
        // After the publication or veto, never before: an invalidation that no longer sees
        // this window must find the tracker already in its final state for this job.
        std::scoped_lock lk{fault_window_mutex};
        fault_windows[&job - fault_jobs.data()] = {};
        pending_fault_jobs.fetch_sub(1);
    }
    if (ok) {
        Offload::Add(Offload::State().successes);
    }
    if (job.prefetch) {
        pf_last_ok = ok; // f2
        --pf_pending;
        if (!job.on_asc && !job.arena) { // q48: an arena job keeps the hold until its publish
            // SHADPS4_HOT_PAGE_PREFETCH (Opus review): the waiter never touches a hot job's slot
            // (no guest copy off the ASC lane; PublishPrefetch checks the job id), so the slot is
            // free for guest faults as soon as the job is finalized. f2 jobs keep the hold until
            // PublishPrefetch: the waiter may still guest-copy through the slot.
            job.pf_hold.store(false, std::memory_order_release);
        }
    }
    job.released = false; // r20
    if (job.abandoned) {
        job.state = FaultJobState::Free;
    } else {
        job.state = ok ? FaultJobState::Success : FaultJobState::Vetoed;
    }
}

void BufferCache::GuestCopyFaultJob(FaultJob& job) {
    u32 expected = 0;
    if (!job.guest_copy.compare_exchange_strong(expected, 1, std::memory_order_seq_cst)) {
        // The command processor settled this job first (it needed the window): it copied
        // itself or vetoed, and the staging may already be released.
        g_guest_copy_closed.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    // While the state is 1 the command processor cannot finalize the job: the copy list and
    // the staging buffer stay as PrepareFaultDownload left them.
    if (!job.staging->is_coherent) {
        vmaInvalidateAllocation(instance.GetAllocator(), job.staging->buffer.allocation, 0,
                                job.total_bytes);
    }
    const u8* const download = job.staging->mapped_data.data();
    auto* memory = Core::Memory::Instance();
    for (const FaultCopy& copy : job.copies) {
        memory->TryWriteBackingDownload(std::bit_cast<u8*>(copy.addr),
                                        download + copy.staging_offset, copy.size);
    }
    job.guest_copy.store(2, std::memory_order_seq_cst);
    const u64 n = g_guest_copies.fetch_add(1, std::memory_order_relaxed) + 1;
    if ((n & 0x3fff) == 0) {
        LOG_INFO(Render, "FaultGuestCopy: guest copies {} closed first {} cp waits {}", n,
                 g_guest_copy_closed.load(std::memory_order_relaxed),
                 g_guest_copy_cp_waits.load(std::memory_order_relaxed));
    }
}

bool BufferCache::CloseGuestCopy(FaultJob& job) {
    bool waited = false;
    u32 spins = 0;
    for (;;) {
        u32 state = job.guest_copy.load(std::memory_order_seq_cst);
        if (state == 2) {
            return true;
        }
        if (state == 3) {
            return false;
        }
        if (state == 0) {
            if (job.guest_copy.compare_exchange_strong(state, 3, std::memory_order_seq_cst)) {
                return false;
            }
            continue;
        }
        // 1: the owner is copying (at most one window, well under a millisecond).
        if (!waited) {
            waited = true;
            g_guest_copy_cp_waits.fetch_add(1, std::memory_order_relaxed);
        }
        // The owner may be descheduled or suspended (debugger pause, stack sampler): do not
        // burn the core on it for long.
        if (++spins < 4096) {
            _mm_pause();
        } else {
            std::this_thread::yield();
        }
    }
}

bool BufferCache::FaultJobUnwritten(const FaultJob& job) {
    if (memory_tracker->GpuWriteSeqMatches(job.addr, job.size, job.seqs)) {
        return true;
    }
    const int mode = GpuWriteLog::Mode();
    if (mode == 0) {
        return false;
    }
    // A region sequence moved: some GPU write marking landed in one of the job's 4 MiB
    // regions since the snapshot. Every such marking logged its page range; the job is
    // stale only if one of them intersects its window. The copy-list comparison in
    // PublishFaultJob still follows and catches a window whose GPU-modified set changed.
    const auto [window_begin, window_end] = TrackerPageSpan(job.addr, job.size);
    u64 hit_begin = 0;
    u64 hit_end = 0;
    auto& regime = Offload::g_regime;
    switch (GpuWriteLog::Check(job.write_log_pos, window_begin, window_end, hit_begin, hit_end)) {
    case GpuWriteLog::Result::Clean:
        regime.exact_clean.fetch_add(1, std::memory_order_relaxed);
        return mode == 1;
    case GpuWriteLog::Result::Overrun:
        regime.exact_overrun.fetch_add(1, std::memory_order_relaxed);
        return false;
    case GpuWriteLog::Result::Touched:
        break;
    }
    const u64 touched = regime.exact_touched.fetch_add(1, std::memory_order_relaxed);
    if (touched < 48 || (touched & 1023) == 0) {
        LOG_INFO(Render,
                 "FaultVetoTouched: #{} job {:#x}:{:#x} write={} log_age={} by {:#x}:{:#x}",
                 touched, job.addr, job.size, job.is_write,
                 GpuWriteLog::Position() - job.write_log_pos, hit_begin, hit_end - hit_begin);
    }
    return false;
}

u64 BufferCache::GuestSubmitFaultJob(FaultJob& job) {
    u32 expected = 0;
    if (job.guest_submit.compare_exchange_strong(expected, 1, std::memory_order_seq_cst)) {
        // While the state is 1 the command processor waits in CloseGuestSubmit; nothing in
        // SubmitSlot needs the command processor, so that wait is bounded by the submit.
        const u64 tick = LaneOf(job.on_fast_lane).SubmitSlot(job.submit_slot);
        job.tick = tick;
        job.guest_submit.store(2, std::memory_order_seq_cst);
        const u64 n = g_guest_submits.fetch_add(1, std::memory_order_relaxed) + 1;
        if ((n & 0x3fff) == 0) {
            LOG_INFO(Render,
                     "FaultGuestSubmit: guest submits={} cp takeovers={} cp waits={} jobs={}", n,
                     g_guest_submit_takeovers.load(std::memory_order_relaxed),
                     g_guest_submit_cp_waits.load(std::memory_order_relaxed),
                     Offload::g_regime.jobs.load(std::memory_order_relaxed));
        }
        return tick;
    }
    // The command processor needed this job first (a settle, a retry ticket or a buffer
    // deletion): it is submitting (3) or has submitted (4). Its tick is valid at 4 only.
    u32 spins = 0;
    while (job.guest_submit.load(std::memory_order_seq_cst) != 4) {
        if (++spins < 4096) {
            _mm_pause();
        } else {
            std::this_thread::yield();
        }
    }
    return job.tick;
}

void BufferCache::CloseGuestSubmit(FaultJob& job) {
    bool waited = false;
    u32 spins = 0;
    for (;;) {
        u32 state = job.guest_submit.load(std::memory_order_seq_cst);
        if (state == 2 || state == 4) {
            return;
        }
        if (state == 0) {
            if (job.guest_submit.compare_exchange_strong(state, 3, std::memory_order_seq_cst)) {
                // Forestalled: the owner finds 3 and waits for 4 (GuestSubmitFaultJob).
                job.tick = LaneOf(job.on_fast_lane).SubmitSlot(job.submit_slot);
                job.guest_submit.store(4, std::memory_order_seq_cst);
                g_guest_submit_takeovers.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            continue;
        }
        // 1: the owner is inside vkQueueSubmit (tens of microseconds, unless descheduled). 3 is
        // not reached here: one command processor thread closes, and SubmitSlot never reenters.
        if (!waited) {
            waited = true;
            g_guest_submit_cp_waits.fetch_add(1, std::memory_order_relaxed);
        }
        if (++spins < 4096) {
            _mm_pause();
        } else {
            std::this_thread::yield();
        }
    }
}

void BufferCache::CloseGuestSubmitsOver(VAddr device_addr, u64 size) {
    if (!g_fault_guest_submit || pending_fault_jobs.load(std::memory_order_relaxed) == 0) {
        return;
    }
    for (FaultJob& job : fault_jobs) {
        // Only an unfinalized job can hold a recorded, unsubmitted slot. Its copies lie inside
        // its window, and a buffer that does not overlap the window is not one of its sources.
        if (job.state == FaultJobState::Free || job.state == FaultJobState::Success ||
            job.state == FaultJobState::Vetoed) {
            continue;
        }
        if (job.addr < device_addr + size && device_addr < job.addr + job.size) {
            CloseGuestSubmit(job);
        }
    }
}

int BufferCache::PrefetchMode() {
    static const int mode = [] {
        const char* e = std::getenv("SHADPS4_LANE_PREFETCH");
        const char* lane = std::getenv("SHADPS4_ASC_LANE");
        const int m = e ? std::atoi(e) : 0;
        if (m != 0 && !(lane && lane[0] == '1')) {
            LOG_WARNING(Render, "f2: SHADPS4_LANE_PREFETCH needs SHADPS4_ASC_LANE=1; off");
            return 0;
        }
        if (m != 0) {
            LOG_INFO(Render, "f2: lane prefetch of read-back windows on (mode {})", m);
        }
        return m;
    }();
    return mode;
}

void BufferCache::PrefetchNoteWindow(VAddr addr, u64 size) {
    // H3 / H3b (back, at a guest read job's admission): the window and the last ASC label recorded
    // before the fault (the label the guest polls).
    if (size == 0) {
        return;
    }
    const u64 first = addr >> 12;
    const u64 pages = std::min<u64>(((addr + size + 4095) >> 12) - first, (u64{1} << 24) - 1);
    const u64 packed = (first << 24) | pages;
    for (auto& w : pf_win) {
        if (w.packed == packed) {
            w.label = g_pf_label_addr;
            return;
        }
    }
    auto& w = pf_win[pf_win_next++ & 15];
    w = PrefetchWindow{packed, g_pf_label_addr, 0};
    ++pf_c[12]; // windows learned
}

u64 BufferCache::PrefetchAfterLabel(VAddr label) {
    // H4 (back; lane stream active, the chain's dispatches of this label are in the open segment).
    if (PrefetchMode() == 0 || label == 0 || !scheduler.HasLane() || scheduler.ActiveStream() != 1 ||
        !(PreciseProducerEnabled() || Offload::LatencyEnabled() || FastLaneEnabled())) {
        return 0;
    }
    u64 recorded = 0;
    for (auto& w : pf_win) {
        if (w.packed == 0 || w.label != label) {
            continue;
        }
        ++pf_c[0]; // triggers
        const VAddr lo = (w.packed >> 24) << 12;
        const u64 bytes = (w.packed & 0xffffff) << 12;
        const auto age = [&] {
            if (++w.idle >= 8) {
                w = PrefetchWindow{};
                ++pf_c[13]; // windows aged out
            }
        };
        std::vector<FaultCopy> copies;
        const u64 total = CollectFaultCopies(lo, bytes, copies);
        if (total == 0) {
            ++pf_c[2]; // nothing GPU-modified
            age();
            continue;
        }
        const u64 asc = NewestAscWriter(copies, AscDoneBelow(2));
        if (asc == 0) {
            ++pf_c[3]; // the lane is not writing it
            age();
            continue;
        }
        w.idle = 0;
        // One source buffer, looked up without creating or joining anything.
        const BufferId buffer_id = page_table[lo >> CACHING_PAGEBITS].buffer_id;
        if (IsBufferInvalid(buffer_id) || !slot_buffers[buffer_id].IsInBounds(lo, bytes)) {
            ++pf_c[4]; // not one buffer
            continue;
        }
        if (pf_pending >= 2 || total > FaultStagingSize()) {
            ++pf_c[5]; // cap / size
            continue;
        }
        bool overlap = false;
        {
            const auto [begin, end] = TrackerPageSpan(lo, bytes);
            std::scoped_lock lk{fault_window_mutex};
            for (const auto& fw : fault_windows) {
                overlap = overlap || (fw.begin < end && begin < fw.end);
            }
        }
        if (overlap) {
            ++pf_c[6]; // a pending job overlaps
            continue;
        }
        FaultJob* job = nullptr;
        for (FaultJob& candidate : fault_jobs) {
            if (candidate.state == FaultJobState::Free &&
                !candidate.pf_hold.load(std::memory_order_acquire)) {
                job = &candidate;
                break;
            }
        }
        std::unique_ptr<Buffer> staging = job ? AcquireFaultStaging() : nullptr;
        if (!staging) {
            ++pf_c[5];
            continue;
        }
        // Main writers still unfinished: the segment waits on them (rule A; an open-batch writer
        // defers it and the caller's LaneCutNow flushes main first).
        auto* master = scheduler.GetMasterSemaphore();
        // Review: a write is stamped at bind time and can land in the next batch: wait pm + 1.
        const u64 pm = PreciseProducer(copies) + 1;
        if (pm > master->KnownGpuTick()) {
            master->Refresh();
            if (pm > master->KnownGpuTick()) {
                scheduler.LaneNeedMain(pm, 0);
                ++pf_c[7]; // main waits added
            }
        }
        const u64 id = next_fault_job_id++;
        job->seqs.clear();
        memory_tracker->SnapshotGpuWriteSeq(lo, bytes, job->seqs);
        job->write_log_pos = GpuWriteLog::Position();
        const u64 value = scheduler.OpenAscValue();
        const bool admitted = fault_mapping_gate.TryAdmit(lo, bytes, [&] {
            job->id = id;
            job->guest_copy.store(0, std::memory_order_seq_cst);
            job->guest_submit.store(4, std::memory_order_seq_cst);
            job->state = FaultJobState::Pending;
            job->abandoned = true; // no owner: finalization frees the slot
            job->prefetch = true;
            job->arena = false;
            job->pf_hold.store(true, std::memory_order_release);
            job->is_write = false;
            job->addr = lo;
            job->size = bytes;
            job->tick = value;
            job->on_readback_queue = true;
            job->on_fast_lane = false;
            job->on_asc = true;
            job->on_retry_lane = false;
            job->released = false;
            job->total_bytes = total;
            job->copies = std::move(copies);
            job->spans.clear();
            job->spans.emplace_back(lo, lo + bytes);
            job->mark_addr = lo;
            job->mark_size = bytes;
            job->staging = std::move(staging);
            std::scoped_lock lk{fault_window_mutex};
            const auto [window_begin, window_end] = TrackerPageSpan(lo, bytes);
            fault_windows[job - fault_jobs.data()] = {window_begin, window_end};
            pending_fault_jobs.fetch_add(1);
            fault_jobs_admitted.fetch_add(1, std::memory_order_acq_rel);
        });
        if (!admitted) {
            job->seqs.clear();
            ReleaseFaultStaging(std::move(staging));
            ++pf_c[6];
            continue;
        }
        ++pf_pending;
        boost::container::small_vector<vk::BufferCopy, 8> vk_copies;
        Buffer& source = slot_buffers[buffer_id];
        for (const FaultCopy& c : job->copies) {
            vk_copies.push_back(vk::BufferCopy{
                .srcOffset = c.addr - source.CpuAddr(),
                .dstOffset = c.staging_offset,
                .size = c.size,
            });
        }
        // Recorded on the lane, after the chain's dispatches (inline lane / queued recorder).
        RecordFaultCopiesAsync(source, job->staging->Handle(), {vk_copies.data(), vk_copies.size()});
        ++pf_c[1]; // jobs recorded
        pf_c[14] += total;
        {
            std::scoped_lock lk{pf_mutex};
            pf_queue.push_back({id, static_cast<u32>(job - fault_jobs.data()), value});
        }
        pf_cv.notify_one();
        ++recorded;
    }
    if (recorded != 0 && !pf_thread.joinable()) {
        pf_thread = std::jthread([this](std::stop_token st) { PrefetchWaiterLoop(st); });
    }
    PrefetchMaybeLog();
    return recorded;
}

void BufferCache::PrefetchWaiterLoop(std::stop_token stop, bool own) {
    // H5: wait the lane value, copy staging -> guest memory (FAULT_GUEST_COPY CAS 0 -> 1 -> 2; fails
    // when the back already closed it), then a non-blocking publish on the back.
    // `own` (r18): the arena retry items' waiter, on pf_retry_queue.
    Common::SetCurrentThreadName(own ? "shadPS4:ArenaRetryWait" : "shadPS4:LanePrefetch");
    std::mutex& mutex = own ? pf_retry_mutex : pf_mutex;
    std::condition_variable_any& cv = own ? pf_retry_cv : pf_cv;
    std::vector<PrefetchItem>& queue = own ? pf_retry_queue : pf_queue;
    for (;;) {
        PrefetchItem it{};
        {
            std::unique_lock lk{mutex};
            if (!cv.wait(lk, stop, [&queue] { return !queue.empty(); })) {
                return;
            }
            it = queue.front();
            queue.erase(queue.begin());
        }
        // lane 0: the ASC lane (f2); 1 the fast lane, 2 main (SHADPS4_HOT_PAGE_PREFETCH), 3 the
        // arena retry queue (r18).
        (it.lane == 2 ? *scheduler.GetMasterSemaphore()
                      : JobSemaphore(it.lane == 1, it.lane == 0, it.lane == 3))
            .Wait(it.value);
        // f2 jobs only: a hot-page job's slot is released when it is finalized (FinalizeFaultJob),
        // so it may already hold another job; its bytes are copied by PublishFaultJob on the back.
        // q48: arena jobs too - their slot is held until PublishPrefetch, which this posts below.
        if (((it.lane == 0 && PrefetchMode() == 1) || it.arena) && g_fault_guest_copy &&
            !Offload::VerifyEnabled()) {
            GuestCopyFaultJob(fault_jobs[it.slot]); // slot held: never another job's
        }
        liverpool->SendCommand<false>(
            [this, id = it.id, slot = it.slot, lane = it.lane] { PublishPrefetch(id, slot, lane); });
    }
}

void BufferCache::PublishPrefetch(u64 id, u32 slot, u8 lane) {
    // H6 (back). A guest's retry or an overlapping settle may have finalized the job already; a
    // hot-page job's slot (lane 1 / 2) may then even hold a newer job (its hold ends at finalize).
    FaultJob& job = fault_jobs[slot];
    const bool same_job = job.id == id; // ids are never reused
    const bool pending = same_job && job.state == FaultJobState::Pending && job.prefetch;
    if (pending) {
        job.state = FaultJobState::Settling;
        FinalizeFaultJob(job); // the lane value is complete; ProcessCommands runs this on main
        ++pf_c[pf_last_ok ? 15 : 16]; // published / vetoed
    } else {
        ++pf_c[17]; // settled before the waiter (guest early or an overlap settle)
    }
    if (same_job && job.arena && job.arena_retry) { // SHADPS4_ARENA_PREFETCH_RETRY: own counters
        g_arena_retry_n[!pending ? 10 : pf_last_ok ? 8 : 9].fetch_add(1, std::memory_order_relaxed);
    } else if (same_job && job.arena) { // q48
        g_arena_n[!pending ? 12 : pf_last_ok ? 10 : 11].fetch_add(1, std::memory_order_relaxed);
    } else if (lane != 0) { // SHADPS4_HOT_PAGE_PREFETCH (f2 jobs are on the ASC lane, lane 0)
        g_hot_n[!pending ? 10 : pf_last_ok ? 8 : 9].fetch_add(1, std::memory_order_relaxed);
    }
    if (same_job) { // never another job's hold (Opus review)
        job.pf_hold.store(false, std::memory_order_release);
    }
    if (lane == 0) {
        PrefetchMaybeLog();
    }
}

void BufferCache::PrefetchMaybeLog() {
    const auto now = std::chrono::steady_clock::now();
    if (now - pf_log_t < std::chrono::seconds(5)) {
        return;
    }
    const double s = pf_log_t.time_since_epoch().count() == 0
                         ? 5.0
                         : std::chrono::duration<double>(now - pf_log_t).count();
    pf_log_t = now;
    const auto per = [&](u32 i) { return double(pf_c[i]) / s; };
    LOG_INFO(Render,
             "f2 LanePrefetch (per s): triggers {:.0f} recorded {:.0f} ({:.0f} KB) | skipped: clean {:.0f}, "
             "no lane writer {:.0f}, not one buffer {:.0f}, cap/staging {:.0f}, overlap {:.0f} | main waits "
             "added {:.0f} | published by the waiter {:.0f} vetoed {:.0f} settled before the waiter {:.0f} | "
             "guest early {:.0f} | verify matched {:.0f} MISMATCH {:.0f} unverified {:.0f} | windows learned "
             "{:.0f} aged {:.0f}",
             per(0), per(1), per(14) / 1024.0, per(2), per(3), per(4), per(5), per(6), per(7), per(15),
             per(16), per(17), per(8), per(9), per(10), per(11), per(12), per(13));
    pf_c.fill(0);
}

// SHADPS4_HOT_PAGE_PREFETCH=1 ("F2-lite", 2026-10-13). CP thread, once per frame after the arena
// discard. Guest threads take ~7.8 readback-class write faults per frame (~0.85 ms each: a wait for
// the GPU writer, then a download) on ~30 pages, the same every frame or two (q43c). Each such page
// seen in the last 16 frames and still GPU-modified gets the read job a guest read fault there
// would start (its read-fault window), owner-less like an f2 prefetch: on the fast lane when no
// unfinished batch wrote it, else in the open main batch. The f2 waiter publishes it when the copy
// is done, so the guest's write meets a GPU-clean page (a plain fault). A GPU write after the
// snapshot vetoes the publication; a guest fault in the window before it takes f2's H7 retry.
// Nothing here waits on the GPU: what would (no slot or staging, a pending job there) is skipped.
void BufferCache::HotPrefetchAfterWrite() {
    // q56: right after a dispatch that wrote a hot page (CP thread): its download goes into the open batch behind it.
    if (!t_hot_write_pending) {
        return;
    }
    t_hot_write_pending = false;
    // At most 8 per presented frame (a hot page written by many dispatches must not run this per dispatch).
    static u64 frame = 0, in_frame = 0;
    const u64 now = Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
    if (now != frame) {
        frame = now;
        in_frame = 0;
    }
    if (++in_frame > 8) {
        return;
    }
    g_hot_trigger.fetch_add(1, std::memory_order_relaxed);
    PrefetchHotPages(t_hot_write_page); // q56c: only the page this dispatch wrote
}

void BufferCache::PrefetchHotPages(VAddr only_page) {
    // Guest write faults must reach H7 with no CP settle first (offloaded, no pre-settle); an ASC
    // stream or lane would need PrepareFaultDownload's ASC producer ordering.
    static const bool usable = [this] {
        const bool ok = Offload::Enabled() && g_write_fault_offload && g_fault_write_skip_settle &&
                        !scheduler.HasAscStream() && !scheduler.HasLane();
        LOG_INFO(Render, "HotPrefetch: {}", ok ? "on" : "off (needs SHADPS4_READBACK_OFFLOAD, "
                 "WRITE_FAULT_OFFLOAD, FAULT_WRITE_SKIP_SETTLE and no ASC stream or lane)");
        return ok;
    }();
    if (!usable) {
        return;
    }
    constexpr u32 MaxJobs = 4; // prefetch jobs held at once; guest faults keep the other slots
    // Opus review: a main-batch job holds its slot until that batch completes (~a frame), so at
    // most this many per call (per frame) go into the main batch; fast-lane jobs are not capped.
    constexpr u32 MaxMainJobs = 2;
    u32 main_jobs = 0;
    const u64 now = Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
    std::array<VAddr, HotPageSlots> pages{};
    size_t n = 0;
    {
        std::scoped_lock lk{g_hot_mutex};
        for (const HotPage& h : g_hot_pages) {
            if (h.page != 0 && h.frame + 16 >= now && (only_page == 0 || h.page == only_page)) {
                pages[n++] = h.page;
            }
        }
    }
    std::sort(pages.begin(), pages.begin() + n);
    auto* master = scheduler.GetMasterSemaphore();
    master->Refresh();
    const VAddr arena_lo = g_arena_lo.load(std::memory_order_relaxed);
    const VAddr arena_hi = g_arena_hi.load(std::memory_order_relaxed);
    const auto count = [](u32 k, u64 v = 1) { g_hot_n[k].fetch_add(v, std::memory_order_relaxed); };
    boost::container::small_vector<PrefetchItem, MaxJobs> items;
    bool main_copies = false;
    for (size_t i = 0; i < n;) {
        const VAddr page = pages[i];
        const BufferId buffer_id = page_table[page >> CACHING_PAGEBITS].buffer_id;
        if (IsBufferInvalid(buffer_id) || !slot_buffers[buffer_id].IsInBounds(page, 4096) ||
            !memory_tracker->IsRegionGpuModified(page, 4096)) {
            ++i;
            count(4);
            continue;
        }
        Buffer& source = slot_buffers[buffer_id];
        const auto window = ResolveFaultWindow(source, page, 4096, false);
        const VAddr lo = window.first;
        const u64 bytes = window.second;
        u64 covered = 0;
        do {
            ++covered; // the hot pages of this window (sorted; none below `page`)
        } while (++i < n && pages[i] < lo + bytes);
        // The next arena discard settles its range; a pending job there is published anyway.
        if ((lo < arena_hi && arena_lo < lo + bytes) || HasOverlappingFaultJob(lo, bytes)) {
            count(5);
            continue;
        }
        FaultJob* job = nullptr;
        u32 held = 0; // unfinalized hot jobs (their hold ends in FinalizeFaultJob)
        u32 free_slots = 0;
        for (FaultJob& candidate : fault_jobs) {
            const bool hold = candidate.pf_hold.load(std::memory_order_acquire);
            held += hold && !candidate.arena; // q50 (review): arena holds have their own cap
            if (candidate.state == FaultJobState::Free && !hold) {
                ++free_slots;
                job = job ? job : &candidate;
            }
        }
        if (g_hot_at_write && free_slots <= 3) { // q56c (review 4): 3 slots kept for guest faults
            count(6);
            continue;
        }
        std::vector<FaultCopy> copies;
        const u64 total = held < MaxJobs ? CollectFaultCopies(lo, bytes, copies) : 0;
        const bool fits = total != 0 && total <= FaultStagingSize();
        std::unique_ptr<Buffer> staging = job && fits ? AcquireFaultStaging() : nullptr;
        if (!staging) {
            count(held >= MaxJobs || !job || fits ? 6 : 7);
            continue;
        }
        // Fast lane as in PrepareFaultDownload: no batch from the newest completed one on wrote
        // a copied byte, so the copy waits for nothing unfinished.
        const u64 tick = scheduler.CurrentTick();
        const u64 completed = master->KnownGpuTick();
        const bool lane = fast_lane && completed != 0 && completed < tick &&
                          !BatchesWriteAny(copies, completed, tick);
        if (!lane && main_jobs >= MaxMainJobs) {
            ReleaseFaultStaging(std::move(staging)); // nothing else was taken or mutated
            count(6);
            continue;
        }
        const u64 id = next_fault_job_id++;
        job->seqs.clear();
        memory_tracker->SnapshotGpuWriteSeq(lo, bytes, job->seqs);
        job->write_log_pos = GpuWriteLog::Position();
        const bool admitted = fault_mapping_gate.TryAdmit(lo, bytes, [&] {
            job->id = id;
            job->guest_copy.store(0, std::memory_order_seq_cst);
            job->guest_submit.store(4, std::memory_order_seq_cst);
            job->state = FaultJobState::Pending;
            job->abandoned = true; // no owner: finalization frees the slot
            job->prefetch = true;  // published by the f2 waiter; guest faults take H7
            job->arena = false;
            // Held until finalized (FinalizeFaultJob clears it for non-ASC jobs).
            job->pf_hold.store(true, std::memory_order_release);
            job->is_write = false;
            job->addr = lo;
            job->size = bytes;
            job->tick = tick; // replaced below, in this same CP step
            job->on_readback_queue = lane;
            job->on_fast_lane = lane;
            job->on_asc = false;
            job->on_retry_lane = false;
            job->released = false;
            job->total_bytes = total;
            job->copies = std::move(copies);
            job->spans.clear();
            job->spans.emplace_back(lo, lo + bytes);
            job->mark_addr = lo;
            job->mark_size = bytes;
            job->staging = std::move(staging);
            std::scoped_lock lk{fault_window_mutex};
            const auto [window_begin, window_end] = TrackerPageSpan(lo, bytes);
            fault_windows[job - fault_jobs.data()] = {window_begin, window_end};
            pending_fault_jobs.fetch_add(1);
            fault_jobs_admitted.fetch_add(1, std::memory_order_acq_rel);
        });
        if (!admitted) {
            job->seqs.clear();
            ReleaseFaultStaging(std::move(staging));
            count(7);
            continue;
        }
        ++pf_pending;
        boost::container::small_vector<vk::BufferCopy, 8> vk_copies;
        for (const FaultCopy& c : job->copies) {
            vk_copies.push_back(vk::BufferCopy{
                .srcOffset = c.addr - source.CpuAddr(),
                .dstOffset = c.staging_offset,
                .size = c.size,
            });
        }
        const std::span<const vk::BufferCopy> list{vk_copies.data(), vk_copies.size()};
        const vk::Buffer staging_handle = job->staging->Handle();
        if (lane) {
            job->tick = fast_lane->Submit(master->Handle(), completed, [&](vk::CommandBuffer cmd) {
                RecordFaultCopiesOnReadbackQueue(cmd, source.Handle(), staging_handle, list, false);
            });
        } else {
            scheduler.EndRendering();
            if (AsyncFaultCopies()) {
                RecordFaultCopiesAsync(source, staging_handle, list);
            } else {
                auto command_access = scheduler.AcquireCommandBuffer();
                RecordFaultCopies(command_access.Handle(), source, staging_handle, list);
            }
            job->tick = scheduler.CurrentTick(); // after recording: the batch holding the copy
            main_copies = true;
            ++main_jobs;
        }
        items.push_back({id, static_cast<u32>(job - fault_jobs.data()), job->tick,
                         static_cast<u8>(lane ? 1 : 2)});
        count(0);
        count(1, lane);
        count(2, covered);
        count(3, total >> 10);
    }
    if (main_copies) {
        scheduler.Flush(); // the waiter and a guest's H7 retry wait for this batch's tick
    }
    if (!items.empty()) {
        {
            // The waiter is FIFO: fast-lane copies (~0.1 ms) go ahead of main-tick waits.
            std::scoped_lock lk{pf_mutex};
            for (const PrefetchItem& it : items) {
                pf_queue.insert(it.lane == 1 ? pf_queue.begin() : pf_queue.end(), it);
            }
        }
        pf_cv.notify_one();
        if (!pf_thread.joinable()) {
            pf_thread = std::jthread([this](std::stop_token st) { PrefetchWaiterLoop(st); });
        }
    }
    static u64 calls = 0;
    if ((++calls & 1023) == 0) {
        const auto v = [](u32 k) { return g_hot_n[k].load(std::memory_order_relaxed); };
        LOG_INFO(Render,
                 "HotPrefetch: {} calls ({} after a hot write, q56), {} hot pages now | {} jobs ({} on the fast lane) "
                 "covering {} hot pages, {} KB | skipped: page clean {}, pending job / arena {}, "
                 "cap / slot / staging {}, other {} | published {} vetoed {} settled by a guest "
                 "fault first {}",
                 calls, g_hot_trigger.load(std::memory_order_relaxed), n, v(0), v(1), v(2), v(3), v(4), v(5),
                 v(6), v(7), v(8), v(9), v(10));
    }
}

// SHADPS4_ARENA_READ_PREFETCH=1 (q48, see NoteArenaRead). CP thread, once per frame end. As
// PrefetchHotPages, over the learned arena offsets mapped into `half`, with these differences: the
// arena is the target (no arena skip); up to 6 jobs per call, most in the open main batch (the
// chain that wrote the half was just recorded); 4 slots always stay free for guest faults; the
// jobs keep their slot until PublishPrefetch so the waiter copies their bytes (up to ~1 MB each).
// SHADPS4_ARENA_PREFETCH_RETRY: `gen` is the discard hook generation of `half`; `retry` marks a
// run posted by the submitted hook (its own counters; the main line counts frame-end runs only).
void BufferCache::PrefetchArenaReads(VAddr half, u64 gen, bool retry) {
    static const bool usable = [this] {
        const bool ok = Offload::Enabled() && g_write_fault_offload && g_fault_write_skip_settle &&
                        !scheduler.HasAscStream() && !scheduler.HasLane();
        LOG_INFO(Render, "ArenaReadPrefetch: {}", ok ? "on" : "off (needs SHADPS4_READBACK_OFFLOAD, "
                 "WRITE_FAULT_OFFLOAD, FAULT_WRITE_SKIP_SETTLE and no ASC stream or lane)");
        if (ok && g_arena_prefetch_retry) {
            Vulkan::Scheduler::submitted_hook.store(&ArenaRetryOnSubmit, std::memory_order_release);
            LOG_INFO(Render, "ArenaReadPrefetch: retry at batch submit on (SHADPS4_ARENA_PREFETCH_RETRY)");
        }
        return ok;
    }();
    if (!usable) {
        return;
    }
    constexpr u32 MaxJobs = 3;  // q51: 8 slots again (s867 asserted with 16; cause not found)
    constexpr u32 KeepFree = 3; // slots left for guest faults
    const auto count = [retry](u32 k, u64 v = 1) {
        if (!retry) {
            g_arena_n[k].fetch_add(v, std::memory_order_relaxed);
        }
    };
    const auto rcount = [](u32 k, u64 v = 1) {
        g_arena_retry_n[k].fetch_add(v, std::memory_order_relaxed);
    };
    // Retry runs of the current frame (CP thread only); the frame is the discard hook generation.
    static u64 retry_frame = 0;
    static u32 retry_runs = 0;
    constexpr u32 MaxRetries = 2;
    if (retry) {
        const u64 gen_now = g_arena_hook_gen.load(std::memory_order_relaxed);
        if (gen_now - gen >= 2) {
            rcount(6); // the half's discard is posted: it settles the half's jobs
            return;
        }
        if (retry_frame != gen_now) {
            retry_frame = gen_now;
            retry_runs = 0;
        }
        if (retry_runs >= MaxRetries) {
            rcount(7);
            return;
        }
        ++retry_runs;
        rcount(2);
    }
    u64 need = 0; // max producer of the windows skipped as open / unsubmitted
    count(1);
    // r18 SHADPS4_ARENA_RETRY_QUEUE: a retry run copies on its own queue, only the pages faulted
    // in the last frame, and submits its copies in ascending wait order after the scan.
    const bool own_queue = retry && retry_lane != nullptr;
    const u64 recent_frames = own_queue ? 1 : 4;
    struct OwnSubmit {
        FaultJob* job;
        u64 id;
        u64 wait_value;
        vk::Buffer source;
        vk::Buffer staging;
        boost::container::small_vector<vk::BufferCopy, 8> copies;
    };
    boost::container::small_vector<OwnSubmit, MaxJobs> own_submits;
    const u64 now = Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
    std::array<VAddr, ArenaReadSlots> pages{};
    size_t n = 0;
    {
        std::scoped_lock lk{g_arena_rd_mutex};
        for (const ArenaReadPage& h : g_arena_rd) {
            if (h.page1 != 0 && h.frame + recent_frames >= now) {
                pages[n++] = half + (u64(h.page1 - 1) << 12);
            }
        }
    }
    std::sort(pages.begin(), pages.begin() + n);
    auto* master = scheduler.GetMasterSemaphore();
    master->Refresh();
    boost::container::small_vector<PrefetchItem, MaxJobs> items;
    for (size_t i = 0; i < n && items.size() + own_submits.size() < MaxJobs;) { // review nit: stop at the cap
        const VAddr page = pages[i];
        const BufferId buffer_id = page_table[page >> CACHING_PAGEBITS].buffer_id;
        if (IsBufferInvalid(buffer_id) || !slot_buffers[buffer_id].IsInBounds(page, 4096) ||
            !memory_tracker->IsRegionGpuModified(page, 4096)) {
            ++i;
            count(6);
            continue;
        }
        Buffer& source = slot_buffers[buffer_id];
        const auto window = ResolveFaultWindow(source, page, 4096, false);
        const VAddr lo = window.first;
        const u64 bytes = window.second;
        u64 covered = 0;
        do {
            ++covered; // the learned pages of this window (sorted; none below `page`)
        } while (++i < n && pages[i] < lo + bytes);
        if (HasOverlappingFaultJob(lo, bytes)) {
            count(7);
            continue;
        }
        FaultJob* job = nullptr;
        u32 free_slots = 0;
        for (FaultJob& candidate : fault_jobs) {
            if (candidate.state == FaultJobState::Free &&
                !candidate.pf_hold.load(std::memory_order_acquire)) {
                ++free_slots;
                job = job ? job : &candidate;
            }
        }
        if (items.size() + own_submits.size() >= MaxJobs || free_slots <= KeepFree) { // r18 review: own-queue jobs count
            count(8);
            continue;
        }
        std::vector<FaultCopy> copies;
        const u64 total = CollectFaultCopies(lo, bytes, copies);
        const bool fits = total != 0 && total <= FaultStagingSize();
        std::unique_ptr<Buffer> staging = fits ? AcquireFaultStaging() : nullptr;
        if (!staging) {
            count(fits ? 8 : 9);
            continue;
        }
        // q50 (s866: main-batch jobs ride the frame-end batch, which holds the whole frame's
        // graphics; a frame-start read that met one waited ~1 ms): fast lane only. The copy
        // waits on the main timeline for the newest batch that wrote a copied byte (the
        // chain's), not for later graphics. That batch must be submitted (as the readback
        // queue's side path requires, review 2026-09-27); otherwise no job - the guest's own
        // read fault downloads as before.
        const u64 tick = scheduler.CurrentTick();
        const u64 completed = master->KnownGpuTick();
        // + 1: a write is stamped at bind time and can land in the next batch (f2 review).
        // Opus review (q51): the per-batch write history, never ProducerTick.
        const u64 producer = PreciseProducer(copies) + 1;
        const VkSemaphore async_timeline = Vulkan::Scheduler::async_timeline.load();
        const bool producer_submitted =
            async_timeline == VK_NULL_HANDLE ||
            producer <= Vulkan::Scheduler::async_submitted_tick.load(std::memory_order_acquire);
        const bool done = completed != 0 && completed < tick &&
                          !BatchesWriteAny(copies, completed, tick);
        const u64 wait_value = done ? completed : producer;
        if (!fast_lane || (!done && (producer <= 1 || producer >= tick || !producer_submitted))) {
            ReleaseFaultStaging(std::move(staging));
            if (!g_arena_prefetch_retry) {
                count(13);
            } else if (!fast_lane || producer <= 1) {
                rcount(11); // no fast lane (or no known producer): a retry cannot help
            } else {
                // The producer batch is open or not yet submitted: retry once it is submitted.
                count(13);
                if (retry) {
                    rcount(5);
                }
                need = std::max(need, producer);
            }
            continue;
        }
        const bool lane = true;
        const u64 id = next_fault_job_id++;
        job->seqs.clear();
        memory_tracker->SnapshotGpuWriteSeq(lo, bytes, job->seqs);
        job->write_log_pos = GpuWriteLog::Position();
        const bool admitted = fault_mapping_gate.TryAdmit(lo, bytes, [&] {
            job->id = id;
            job->guest_copy.store(0, std::memory_order_seq_cst);
            job->guest_submit.store(4, std::memory_order_seq_cst);
            job->state = FaultJobState::Pending;
            job->abandoned = true; // no owner: finalization frees the slot
            job->prefetch = true;  // published by the f2 waiter; guest faults take H7
            job->arena = true;     // held until PublishPrefetch; the waiter copies
            job->arena_retry = retry; // read only with `arena` (PublishPrefetch's counters)
            job->pf_hold.store(true, std::memory_order_release);
            job->is_write = false;
            job->addr = lo;
            job->size = bytes;
            job->tick = tick; // replaced below, in this same CP step
            job->on_readback_queue = lane;
            job->on_fast_lane = lane && !own_queue;
            job->on_asc = false;
            job->on_retry_lane = own_queue;
            job->released = false;
            job->total_bytes = total;
            job->copies = std::move(copies);
            job->spans.clear();
            job->spans.emplace_back(lo, lo + bytes);
            job->mark_addr = lo;
            job->mark_size = bytes;
            job->staging = std::move(staging);
            std::scoped_lock lk{fault_window_mutex};
            const auto [window_begin, window_end] = TrackerPageSpan(lo, bytes);
            fault_windows[job - fault_jobs.data()] = {window_begin, window_end};
            pending_fault_jobs.fetch_add(1);
            fault_jobs_admitted.fetch_add(1, std::memory_order_acq_rel);
        });
        if (!admitted) {
            job->seqs.clear();
            ReleaseFaultStaging(std::move(staging));
            count(9);
            continue;
        }
        ++pf_pending;
        boost::container::small_vector<vk::BufferCopy, 8> vk_copies;
        for (const FaultCopy& c : job->copies) {
            vk_copies.push_back(vk::BufferCopy{
                .srcOffset = c.addr - source.CpuAddr(),
                .dstOffset = c.staging_offset,
                .size = c.size,
            });
        }
        const vk::Buffer staging_handle = job->staging->Handle();
        if (own_queue) {
            // Submitted after the scan, in ascending wait order. Safe while every reader of a
            // Pending job's tick runs on the CP thread and nothing in this scan settles a job
            // (r18 review): the placeholder main tick is never waited on the retry timeline.
            own_submits.push_back({job, id, wait_value, source.Handle(), staging_handle,
                                   std::move(vk_copies)});
        } else {
            const std::span<const vk::BufferCopy> list{vk_copies.data(), vk_copies.size()};
            job->tick = fast_lane->Submit(master->Handle(), wait_value, [&](vk::CommandBuffer cmd) {
                RecordFaultCopiesOnReadbackQueue(cmd, source.Handle(), staging_handle, list, false);
            });
            items.push_back({id, static_cast<u32>(job - fault_jobs.data()), job->tick, 1, true});
        }
        count(2);
        count(3, lane);
        count(4, total >> 10);
        count(5, covered);
        if (retry) {
            rcount(3);
            rcount(4, total >> 10);
        }
    }
    if (!own_submits.empty()) {
        // r18: one in-order queue - an early producer's copy must not wait behind a later one.
        std::sort(own_submits.begin(), own_submits.end(),
                  [](const OwnSubmit& a, const OwnSubmit& b) { return a.wait_value < b.wait_value; });
        for (OwnSubmit& o : own_submits) {
            const std::span<const vk::BufferCopy> list{o.copies.data(), o.copies.size()};
            o.job->tick = retry_lane->Submit(master->Handle(), o.wait_value, [&](vk::CommandBuffer cmd) {
                RecordFaultCopiesOnReadbackQueue(cmd, o.source, o.staging, list, false);
            });
            const PrefetchItem it{o.id, static_cast<u32>(o.job - fault_jobs.data()), o.job->tick, 3,
                                  true};
            std::scoped_lock lk{pf_retry_mutex};
            pf_retry_queue.push_back(it);
        }
        pf_retry_cv.notify_one();
        if (!pf_retry_thread.joinable()) {
            pf_retry_thread =
                std::jthread([this](std::stop_token st) { PrefetchWaiterLoop(st, true); });
        }
    }
    if (!items.empty()) {
        {
            std::scoped_lock lk{pf_mutex};
            for (const PrefetchItem& it : items) {
                pf_queue.insert(it.lane == 1 ? pf_queue.begin() : pf_queue.end(), it);
            }
        }
        pf_cv.notify_one();
        if (!pf_thread.joinable()) {
            pf_thread = std::jthread([this](std::stop_token st) { PrefetchWaiterLoop(st); });
        }
    }
    // SHADPS4_ARENA_PREFETCH_RETRY: arm after this run's jobs are queued. Without the async submit
    // timeline no worker submit calls the hook. A retry run past the cap does not arm (nothing
    // would take it this frame; the next frame end arms afresh).
    bool retry_now = false;
    VAddr retry_half = 0;
    u64 retry_gen = 0;
    if (g_arena_prefetch_retry && need != 0 &&
        Vulkan::Scheduler::async_timeline.load() != VK_NULL_HANDLE &&
        (!retry || retry_runs < MaxRetries)) {
        ArmArenaRetry(half, gen, need);
        rcount(0);
        // A worker submit between the skip and the store above found nothing armed.
        retry_now = TakeArenaRetry(
            Vulkan::Scheduler::async_submitted_tick.load(std::memory_order_acquire), retry_half,
            retry_gen);
    }
    static u64 calls = 0;
    if (!retry && (++calls & 1023) == 0) {
        const auto v = [](u32 k) { return g_arena_n[k].load(std::memory_order_relaxed); };
        if (!g_arena_prefetch_retry) {
            LOG_INFO(Render,
                     "ArenaReadPrefetch: {} calls | guest read faults in the current half {} | {} jobs "
                     "({} on the fast lane), {} KB, covering {} learned pages | skipped: page clean {}, "
                     "pending job {}, cap / slot / staging {}, producer batch open / unsubmitted {}, "
                     "other {} | published {} vetoed {} settled before the waiter {}",
                     v(1), v(0), v(2), v(3), v(4), v(5), v(6), v(7), v(8), v(13), v(9), v(10), v(11), v(12));
        } else {
            const auto r = [](u32 k) { return g_arena_retry_n[k].load(std::memory_order_relaxed); };
            LOG_INFO(Render,
                     "ArenaReadPrefetch: {} calls | guest read faults in the current half {} | {} jobs "
                     "({} on the fast lane), {} KB, covering {} learned pages | skipped: page clean {}, "
                     "pending job {}, cap / slot / staging {}, producer batch open / unsubmitted {}, "
                     "no fast lane {}, other {} | published {} vetoed {} settled before the waiter {} "
                     "| retry: armed {}, posted {} ({} at once on the CP thread), run {}, {} jobs, "
                     "{} KB, skipped again {}, dropped stale {} / cap {} | retry jobs published {} "
                     "vetoed {} settled before the waiter {} | guest faults on a pending retry "
                     "job {} (Game:Main {}), own queue {}",
                     v(1), v(0), v(2), v(3), v(4), v(5), v(6), v(7), v(8), v(13), r(11), v(9), v(10),
                     v(11), v(12), r(0), r(1), r(12), r(2), r(3), r(4), r(5), r(6), r(7), r(8), r(9),
                     r(10), r(13), r(14), retry_lane != nullptr);
        }
    }
    if (retry_now) {
        // Already on the CP thread, where SendCommand would run it inline: run it here, last.
        // Bounded: each retry run counts against MaxRetries before it can arm again.
        rcount(1);
        rcount(12);
        PrefetchArenaReads(retry_half, retry_gen, true);
    }
}

bool BufferCache::PublishFaultJob(FaultJob& job, bool guest_copied) {
    if (job.released) [[unlikely]] {
        // r20: the tracker was made CPU-valid when the read was let through; only the fresh
        // bytes are left to write - always (review BUG 2: a vote failing on a write elsewhere
        // in the window would freeze the exposure).
        if (!job.staging->is_coherent) {
            vmaInvalidateAllocation(instance.GetAllocator(), job.staging->buffer.allocation, 0,
                                    job.total_bytes);
        }
        const u8* const fresh = job.staging->mapped_data.data();
        auto* mem = Core::Memory::Instance();
        if (!guest_copied) {
            // Every copy (re-check: the tracker's page bits are coarser than the 2.5 KB buffers,
            // so a skip by page could drop these bytes for good). Safe: this job is settled and
            // published before any later download of its window, which then writes newer bytes.
            for (const FaultCopy& copy : job.copies) {
                mem->TryWriteBackingDownload(std::bit_cast<u8*>(copy.addr),
                                             fresh + copy.staging_offset, copy.size);
            }
        }
        const u64 n = g_stale_n[1].fetch_add(1, std::memory_order_relaxed) + 1;
        if ((n & 0x3ff) == 0) {
            LOG_INFO(Render,
                     "ExposureStaleRead: reads released {} | released jobs published {} copies "
                     "re-marked {} | tolerant ranges noted {} refused (too large) {} | hot jobs "
                     "not released {}",
                     g_stale_n[0].load(), n, g_stale_n[2].load(), g_stale_n[3].load(),
                     g_stale_n[4].load(), g_stale_n[5].load());
        }
        return true;
    }
    // One uninterrupted command processor step: vote, compare, then mutate. Nothing here can
    // fault or submit, so no GPU write marking lands between the vote and the clears.
    if (!FaultJobUnwritten(job)) {
        Offload::Add(Offload::State().seq_vetoes);
        Offload::g_regime.seq_vetoes.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    // Compare BEFORE subtracting: on a mismatch the sync fallback must find the ranges.
    std::vector<FaultCopy> current;
    u64 current_bytes = 0;
    for (const auto& [lo, hi] : job.spans) {
        current_bytes = CollectFaultCopies(lo, hi - lo, current, current_bytes);
    }
    if (current_bytes != job.total_bytes || current != job.copies) {
        // A nested synchronous download took part of the window while this job settled.
        LOG_ERROR(Render, "ReadbackOffload: copy list changed for {:#x}:{:#x}; falling back",
                  job.addr, job.size);
        Offload::Add(Offload::State().list_vetoes);
        return false;
    }
    for (const auto& [lo, hi] : job.spans) {
        memory_tracker->ForEachDownloadRange<false>(
            lo, hi - lo, [&](u64 device_addr_out, u64 range_size) {
                gpu_modified_ranges.Subtract(device_addr_out, range_size); MemoryTracker::gpu_clear_epoch.fetch_add(1, std::memory_order_acq_rel);
                Common::FaultMonitor::CountRangeSubtract(
                    gpu_modified_ranges.Fragments());
            });
    }
    if (!job.staging->is_coherent) {
        vmaInvalidateAllocation(instance.GetAllocator(), job.staging->buffer.allocation, 0,
                                job.total_bytes);
    }
    const u8* const download = job.staging->mapped_data.data();
    auto* memory = Core::Memory::Instance();
    if (!guest_copied) {
        for (const FaultCopy& copy : job.copies) {
            memory->TryWriteBackingDownload(std::bit_cast<u8*>(copy.addr),
                                            download + copy.staging_offset, copy.size);
        }
    }
    // Only the buffer spans: for a write job the rest of the window may be GPU-modified image
    // memory that was not downloaded (as in DownloadWindowMultiBuffer).
    for (const auto& [lo, hi] : job.spans) {
        memory_tracker->UnmarkRegionAsGpuModified(lo, hi - lo);
    }
    if (job.is_write &&
        !memory_tracker->IsRegionGpuModified(job.mark_addr, job.mark_size)) {
        // What the synchronous WIDEN=2 path ends in: its exact download finds nothing left and
        // CompleteEmptyWriteFault CPU-marks the faulting range, never the window.
        memory_tracker->MarkRegionAsCpuModified(job.mark_addr, job.mark_size);
    }
    return true;
}

BufferCache::VerifyResult BufferCache::VerifyFaultJob(FaultJob& job) {
    // Oracle for GPU writes that bypass the region sequence. It does not prove the absence
    // of races; it only compares two readbacks the sequence claims are identical.
    if (!FaultJobUnwritten(job)) {
        // Differing bytes are expected, so nothing is compared. The veto PublishFaultJob
        // would have cast is counted here, since it is no longer reached.
        Offload::Add(Offload::State().seq_vetoes);
        Offload::g_regime.seq_vetoes.fetch_add(1, std::memory_order_relaxed);
        return VerifyResult::Unverified;
    }
    // Look the buffers up without creating or merging anything. The source of each copy has to
    // be the cache's current buffer for those bytes, found independently of the job. A write
    // job's copies can come from several buffers.
    std::unique_ptr<Buffer> second;
    boost::container::small_vector<vk::BufferCopy, 8> vk_copies;
    boost::container::small_vector<BufferId, 8> sources;
    bool in_bounds = true;
    for (const FaultCopy& copy : job.copies) {
        const BufferId buffer_id = page_table[copy.addr >> CACHING_PAGEBITS].buffer_id;
        if (IsBufferInvalid(buffer_id) ||
            !slot_buffers[buffer_id].IsInBounds(copy.addr, copy.size)) {
            in_bounds = false;
            break;
        }
        sources.push_back(buffer_id);
        vk_copies.push_back(vk::BufferCopy{
            .srcOffset = copy.addr - slot_buffers[buffer_id].CpuAddr(),
            .dstOffset = copy.staging_offset,
            .size = copy.size,
        });
    }
    if (in_bounds && !job.copies.empty()) {
        second = AcquireFaultStaging();
    }
    if (!second) {
        // No source buffer spans the list any more, or no second staging is left. The job is
        // vetoed rather than published unchecked.
        LOG_WARNING(Render, "ReadbackOffload: no independent readback for {:#x}:{:#x}; vetoed",
                    job.addr, job.size);
        Offload::Add(Offload::State().verify_skips);
        return VerifyResult::Unverified;
    }
    // Same ordering rules as the first copy: end rendering, drain, record in stream order.
    // The reference is taken here and dropped before Finish, which can reenter the cache.
    scheduler.EndRendering();
    {
        auto command_access = scheduler.AcquireCommandBuffer();
        ForEachSourceRun(sources.data(), sources.size(), [&](size_t first, size_t count) {
            RecordFaultCopies(command_access.Handle(), slot_buffers[sources[first]],
                              second->Handle(), {vk_copies.data() + first, count});
        });
    }
    {
        const Common::FaultMonitor::GpuWaitTag tag{5};
        scheduler.Finish();
    }
    VerifyResult result = VerifyResult::Unverified;
    // Finish can reenter. A marking since then makes differing bytes legitimate; the job is
    // then vetoed by the same vote in PublishFaultJob's place.
    if (!FaultJobUnwritten(job)) {
        Offload::Add(Offload::State().seq_vetoes);
        Offload::g_regime.seq_vetoes.fetch_add(1, std::memory_order_relaxed);
    } else {
        for (const Buffer* staging : {second.get(), job.staging.get()}) {
            if (!staging->is_coherent) {
                vmaInvalidateAllocation(instance.GetAllocator(), staging->buffer.allocation, 0,
                                        job.total_bytes);
            }
        }
        Offload::Add(Offload::State().verify_checks);
        result = VerifyResult::Matched;
        const u8* const first_data = job.staging->mapped_data.data();
        const u8* const second_data = second->mapped_data.data();
        for (const FaultCopy& copy : job.copies) {
            if (std::memcmp(first_data + copy.staging_offset, second_data + copy.staging_offset,
                            copy.size) != 0) {
                LOG_CRITICAL(Render,
                             "ReadbackOffload: verify mismatch at {:#x}:{:#x} in job window "
                             "{:#x}:{:#x}; GPU write without a sequence bump",
                             copy.addr, copy.size, job.addr, job.size);
                Offload::Add(Offload::State().verify_mismatches);
                result = VerifyResult::Mismatch;
                break;
            }
        }
    }
    ReleaseFaultStaging(std::move(second));
    return result;
}

namespace {
// SHADPS4_RANGE_ADD_SKIP=1: skip gpu_modified_ranges.Add when the range is already recorded
// (one tree lookup instead of insert + join; the set stays identical). A direct-mapped memo of
// ranges added since the last Subtract (gpu_clear_epoch) answers repeats without the lookup.
bool RangeAddSkip() {
    return g_range_add_skip;
}
struct RangeAddMemo {
    VAddr addr;
    u64 size;
    u64 epoch;
};
std::array<RangeAddMemo, 256> range_add_memo{};
u64 range_add_calls = 0, range_add_skipped = 0;

template <typename Set>
void AddGpuModified(Set& set, VAddr addr, u64 size) {
    if (!RangeAddSkip()) {
        set.Add(addr, size);
        return;
    }
    const u64 epoch = MemoryTracker::gpu_clear_epoch.load(std::memory_order_acquire);
    auto& memo = range_add_memo[(addr >> 6) & (range_add_memo.size() - 1)];
    bool contained = memo.addr == addr && memo.size >= size && memo.epoch == epoch;
    if (!contained) {
        contained = set.Contains(addr, size);
        if (!contained) {
            set.Add(addr, size);
        }
        memo = {addr, size, epoch};
    }
    ++range_add_calls;
    range_add_skipped += contained;
    if ((range_add_calls & 0xfffff) == 0) {
        LOG_INFO(Render_Vulkan, "RangeAddSkip: adds={} contained={} ({:.1f}%)", range_add_calls,
                 range_add_skipped, 100.0 * range_add_skipped / range_add_calls);
    }
}
} // namespace

template <bool async>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, VAddr device_addr, u64 size, bool is_write,
                                       Offload::SyncTiming* timing) {
    boost::container::small_vector<vk::BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    memory_tracker->ForEachDownloadRange<false>(
        device_addr, size, [&](u64 device_addr_out, u64 range_size) {
            const VAddr buffer_addr = buffer.CpuAddr();
            const auto add_download = [&](VAddr start, VAddr end) {
                const u64 new_offset = start - buffer_addr;
                const u64 new_size = end - start;
                copies.push_back(vk::BufferCopy{
                    .srcOffset = new_offset,
                    .dstOffset = total_size_bytes,
                    .size = new_size,
                });
                // Align up to avoid cache conflicts
                constexpr u64 align = 64ULL;
                constexpr u64 mask = ~(align - 1ULL);
                total_size_bytes += (new_size + align - 1) & mask;
            };
            gpu_modified_ranges.ForEachInRange(device_addr_out, range_size, add_download);
            gpu_modified_ranges.Subtract(device_addr_out, range_size); MemoryTracker::gpu_clear_epoch.fetch_add(1, std::memory_order_acq_rel);
            Common::FaultMonitor::CountRangeSubtract(
                gpu_modified_ranges.Fragments());
        });
    if (total_size_bytes == 0) {
        return false;
    }
    Common::FaultMonitor::CountReadbackDownload(total_size_bytes);
    const auto [download, offset] = download_buffer.Map(total_size_bytes);
    buffer.NoteMainAccess(scheduler.CurrentTick(), false);
    if (g_lane_probe && Common::FaultMonitor::t_is_cp_thread) [[unlikely]] {
        for (const auto& copy : copies) {
            const VAddr lo = buffer.CpuAddr() + copy.srcOffset;
            LaneProbeRead(lo, lo + copy.size, scheduler.CurrentTick(), 10, 11); // q9 census
        }
    }
    if (g_range_notes) {
        for (const auto& copy : copies) {
            buffer.NoteMainRange(scheduler.CurrentTick(), false, buffer.CpuAddr() + copy.srcOffset,
                                 copy.size, 14, 0);
        }
    }
    for (auto& copy : copies) {
        // Modify copies to have the staging offset in mind
        copy.dstOffset += offset;
    }
    download_buffer.Commit();
    scheduler.EndRendering();
    auto command_access = scheduler.AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();
    // SHADPS4_DOWNLOAD_BARRIER=1 (port of upstream c1d5b311 / #4889): make earlier GPU writes of
    // the buffer visible to this transfer read. Without it the copy may read data from before
    // the dispatch or draw that produced it (upstream: all-zero shadow readbacks).
    static const bool download_barrier = [] {
        const char* value = std::getenv("SHADPS4_DOWNLOAD_BARRIER");
        return value && value[0] == '1';
    }();
    if (download_barrier) {
        const vk::BufferMemoryBarrier2 pre_barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .buffer = buffer.buffer,
            .offset = 0,
            .size = buffer.SizeBytes(),
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &pre_barrier,
        });
    }
    cmdbuf.copyBuffer(buffer.buffer, download_buffer.Handle(), copies);
    const auto write_data = [&]() {
        auto* memory = Core::Memory::Instance();
        for (const auto& copy : copies) {
            const VAddr copy_device_addr = buffer.CpuAddr() + copy.srcOffset;
            const u64 dst_offset = copy.dstOffset - offset;
            memory->TryWriteBackingDownload(std::bit_cast<u8*>(copy_device_addr),
                                            download + dst_offset, copy.size);
        }
        memory_tracker->UnmarkRegionAsGpuModified(device_addr, size);
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
    };
    if constexpr (async) {
        (void)timing;
        scheduler.DeferOperation(write_data);
    } else {
        if (timing) {
            // Scheduler::Finish, statement for statement (vk_scheduler.cpp): take the tick,
            // submit with an empty SubmitInfo, which is all Flush() does, then Wait on the
            // tick. The submit has advanced the current tick past it, so Wait does not flush
            // again and its time is the GPU wait alone.
            const u64 start = Offload::NowNs();
            const u64 presubmit_tick = scheduler.CurrentTick();
            scheduler.Flush();
            const u64 submitted = Offload::NowNs();
            const Common::FaultMonitor::GpuWaitTag tag{6};
            scheduler.Wait(presubmit_tick);
            const u64 done = Offload::NowNs();
            timing->submit_ns += submitted - start;
            timing->gpu_wait_ns += done - submitted;
            timing->total_ns += done - start;
        } else {
            const Common::FaultMonitor::GpuWaitTag tag{6};
            scheduler.Finish();
        }
        write_data();
    }
    return true;
}

namespace {
/// SHADPS4_CP_SPLIT_FRONT_DERIVE, the thread that executes draws (the CP back with the split):
/// the description id of the setVertexInputEXT recorded last (0: unknown, a legacy record or
/// none yet), counters, and a log line every ~10 s (the clock is checked every 4096 blocks).
struct FrontDeriveState {
    u64 recorded_desc_id{};
    /// Mode 2: the arrays of the last setVertexInputEXT record (the skip oracle).
    std::vector<vk::VertexInputBindingDescription2EXT> last_bindings;
    std::vector<vk::VertexInputAttributeDescription2EXT> last_attributes;
    u64 last_desc_id{};
    u64 draws{};         ///< Draw records carrying a block.
    u64 shipped{};       ///< ... whose description id differs from the previous block's.
    u64 reused{};        ///< setVertexInputEXT skipped (mode 1), or skippable (mode 2).
    u64 verify_checks{};
    u64 mismatches{};
    u64 mismatches_total{};
    u64 block_bytes{};
    u64 window_start_ns{};
    u32 calls{};
    AmdGpu::CpSplit::VertexInputsDerived scratch{}; ///< Mode 2: this thread's own derivation.

    void Note(const AmdGpu::CpSplit::VertexInputsPayload& vi) {
        ++draws;
        block_bytes += AmdGpu::CpSplit::VertexInputsBytes(vi.count, vi.range_count);
        shipped += vi.desc_id != last_desc_id;
        last_desc_id = vi.desc_id;
        if ((++calls & 4095) == 0) {
            MaybeLog();
        }
    }
    void Mismatch(const char* what, u32 index, const AmdGpu::CpSplit::VertexInputsPayload& vi) {
        ++mismatches;
        if (++mismatches_total <= 16) {
            LOG_WARNING(Render_Vulkan,
                        "FrontDerive mismatch: {} {} (front block: {} attributes, {} ranges, "
                        "desc_id {}; back derivation: {} attributes, {} ranges)",
                        what, index, vi.count, vi.range_count, vi.desc_id, scratch.header.count,
                        scratch.header.range_count);
        }
    }
    void MaybeLog() {
        const u64 now = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             std::chrono::steady_clock::now().time_since_epoch())
                                             .count());
        if (window_start_ns == 0) {
            window_start_ns = now;
            return;
        }
        if (now - window_start_ns < 10'000'000'000ull) {
            return;
        }
        LOG_INFO(Render_Vulkan,
                 "FrontDerive: mode={} draws={} shipped={} reused={} verify_checks={} "
                 "mismatches={} mismatches_total={} block_bytes_per_draw={:.0f} window_s={:.1f}",
                 g_front_derive, draws, shipped, reused, verify_checks, mismatches,
                 mismatches_total, double(block_bytes) / double(std::max<u64>(draws, 1)),
                 double(now - window_start_ns) / 1e9);
        draws = shipped = reused = verify_checks = mismatches = block_bytes = 0;
        window_start_ns = now;
    }
};
FrontDeriveState g_fd;

/// The 2EXT arrays of a block: what GetVertexInputs builds for setVertexInputEXT.
void BuildVertexInputArrays(
    const AmdGpu::CpSplit::VertexInputsPayload& vi,
    Vulkan::VertexInputs<vk::VertexInputAttributeDescription2EXT>& attributes,
    Vulkan::VertexInputs<vk::VertexInputBindingDescription2EXT>& bindings) {
    const auto* attribs = AmdGpu::CpSplit::VertexInputAttribs(vi);
    for (u32 i = 0; i < vi.count; ++i) {
        const auto& desc = attribs[i].desc;
        attributes.push_back(vk::VertexInputAttributeDescription2EXT{
            .location = desc.location,
            .binding = desc.location,
            .format = static_cast<vk::Format>(desc.vk_format),
            .offset = 0,
        });
        bindings.push_back(vk::VertexInputBindingDescription2EXT{
            .binding = desc.location,
            .stride = desc.stride,
            .inputRate = desc.instanced ? vk::VertexInputRate::eInstance
                                        : vk::VertexInputRate::eVertex,
            .divisor = desc.divisor,
        });
    }
}

/// FRONT_DERIVE=2: derives the block again here (the same function; this thread's registers
/// and the V# table as it reads now) and compares it with the front's; then compares the arrays
/// mode 1 would record with the legacy GetVertexInputs output. False on a mismatch.
bool FrontDeriveVerify(
    const Vulkan::Instance& instance, const AmdGpu::Regs& regs,
    const Vulkan::GraphicsPipeline& pipeline, const Shader::BoundStage& vs,
    const AmdGpu::CpSplit::VertexInputsPayload& vi,
    const Vulkan::VertexInputs<vk::VertexInputAttributeDescription2EXT>& attributes,
    const Vulkan::VertexInputs<vk::VertexInputBindingDescription2EXT>& bindings) {
    using namespace AmdGpu::CpSplit;
    auto& fd = g_fd;
    ++fd.verify_checks;
    pipeline.DeriveVertexInputs(fd.scratch, regs.vgt_instance_step_rate_0,
                                regs.vgt_instance_step_rate_1, vs);
    const auto* attribs = VertexInputAttribs(vi);
    const auto* ranges = VertexInputRanges(vi);
    const char* what = nullptr;
    u32 index = 0;
    if (fd.scratch.header.count != vi.count) {
        what = "attribute count";
    } else if (fd.scratch.header.range_count != vi.range_count) {
        what = "range count";
    }
    for (u32 i = 0; !what && i < vi.count; ++i) {
        if (std::memcmp(&fd.scratch.attribs[i], &attribs[i], sizeof(VertexInputAttrib)) != 0) {
            what = std::memcmp(&fd.scratch.attribs[i].desc, &attribs[i].desc,
                               sizeof(VertexInputDesc)) != 0
                       ? "attribute description"
                       : "attribute buffer";
            index = i;
        }
    }
    for (u32 i = 0; !what && i < vi.range_count; ++i) {
        if (std::memcmp(&fd.scratch.ranges[i], &ranges[i], sizeof(VertexInputRange)) != 0) {
            what = "range";
            index = i;
        }
    }
    if (!what && instance.IsVertexInputDynamicState()) {
        Vulkan::VertexInputs<vk::VertexInputAttributeDescription2EXT> block_attributes;
        Vulkan::VertexInputs<vk::VertexInputBindingDescription2EXT> block_bindings;
        BuildVertexInputArrays(vi, block_attributes, block_bindings);
        bool same = block_attributes.size() == attributes.size() &&
                    block_bindings.size() == bindings.size();
        for (size_t i = 0; same && i < attributes.size(); ++i) {
            const auto& a = block_attributes[i];
            const auto& b = block_bindings[i];
            same = a.location == attributes[i].location && a.binding == attributes[i].binding &&
                   a.format == attributes[i].format && a.offset == attributes[i].offset &&
                   b.binding == bindings[i].binding && b.stride == bindings[i].stride &&
                   b.inputRate == bindings[i].inputRate && b.divisor == bindings[i].divisor;
            index = static_cast<u32>(i);
        }
        if (!same) {
            what = "vulkan arrays";
        }
    }
    if (what) {
        fd.Mismatch(what, index, vi);
    }
    return what == nullptr;
}
} // namespace

void BufferCache::BindVertexBuffers(
    const Vulkan::GraphicsPipeline& pipeline, const Shader::BoundStage& vs,
    boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers,
    const AmdGpu::CpSplit::VertexInputsPayload* front_vi) {
    // SHADPS4_CP_SPLIT_FRONT_DERIVE: a split draw record may carry everything below up to the
    // ObtainBuffer loop, derived on the front (cp_split.h VertexInputsPayload). Mode 1 consumes
    // it (BindVertexBuffersDerived); mode 2 verifies it against this thread's own derivation and
    // the arrays built here, then uses its own result. Every other caller (no split, indirect
    // draws) runs the path below unchanged.
    const int fd_mode = front_vi ? g_front_derive : 0;
    if (fd_mode != 0) {
        g_fd.Note(*front_vi);
    }
    if (fd_mode == 1) {
        BindVertexBuffersDerived(*front_vi, barriers);
        return;
    }
    const auto& regs = liverpool->BackRegs();
    Vulkan::VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    Vulkan::VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    Vulkan::VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    Vulkan::VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline.GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                             regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1, vs);
    const bool fd_verified =
        fd_mode == 2 &&
        FrontDeriveVerify(instance, regs, pipeline, vs, *front_vi, attributes, bindings);

    if (instance.IsVertexInputDynamicState()) {
        auto& dynamic_state = scheduler.GetDynamicState();
        if (fd_mode == 2) {
            // Mode 1 would skip this record (the command buffer holds this description).
            const bool would_skip = !dynamic_state.vertex_input_unknown &&
                                    front_vi->desc_id == g_fd.recorded_desc_id;
            g_fd.reused += would_skip;
            // Skip oracle: the arrays recorded last must be the arrays recorded now.
            // Field by field: the Vulkan structs have padding after sType (a byte compare of
            // them reported ~100 false mismatches per second, 2026-10-02).
            bool same_arrays = g_fd.last_bindings.size() == bindings.size() &&
                               g_fd.last_attributes.size() == attributes.size();
            for (size_t i = 0; same_arrays && i < bindings.size(); ++i) {
                const auto& a = g_fd.last_bindings[i];
                const auto& b = bindings[i];
                same_arrays = a.binding == b.binding && a.stride == b.stride &&
                              a.inputRate == b.inputRate && a.divisor == b.divisor;
            }
            for (size_t i = 0; same_arrays && i < attributes.size(); ++i) {
                const auto& a = g_fd.last_attributes[i];
                const auto& b = attributes[i];
                same_arrays = a.location == b.location && a.binding == b.binding &&
                              a.format == b.format && a.offset == b.offset;
            }
            if (would_skip && !same_arrays) {
                g_fd.Mismatch("skip arrays", 0, *front_vi);
            }
            g_fd.last_bindings.assign(bindings.begin(), bindings.end());
            g_fd.last_attributes.assign(attributes.begin(), attributes.end());
        }
        // Update current vertex inputs.
        scheduler.RecordArrays(
            [](vk::CommandBuffer cmdbuf, auto bindings, auto attributes) {
                cmdbuf.setVertexInputEXT(bindings, attributes);
            },
            std::span<const vk::VertexInputBindingDescription2EXT>{bindings},
            std::span<const vk::VertexInputAttributeDescription2EXT>{attributes});
        if (g_front_derive != 0) {
            // Also for a draw without a block (indirect): a later derived draw must not skip
            // against an id this record did not establish.
            dynamic_state.vertex_input_unknown = false;
            g_fd.recorded_desc_id = fd_verified ? front_vi->desc_id : 0;
        }
    }

    if (bindings.empty()) {
        // If there are no bindings, there is nothing further to do.
        return;
    }

    struct BufferRange {
        VAddr base_address;
        VAddr end_address;
        vk::Buffer vk_buffer;
        u64 offset;

        [[nodiscard]] size_t GetSize() const {
            return end_address - base_address;
        }
    };

    // Build list of ranges covering the requested buffers
    Vulkan::VertexInputs<BufferRange> ranges{};
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            ranges.emplace_back(buffer.base_address, buffer.base_address + buffer.GetSize());
        }
    }

    // Merge connecting ranges together
    Vulkan::VertexInputs<BufferRange> ranges_merged{};
    if (!ranges.empty()) {
        std::ranges::sort(ranges, [](const BufferRange& lhv, const BufferRange& rhv) {
            return lhv.base_address < rhv.base_address;
        });
        ranges_merged.emplace_back(ranges[0]);
        for (auto range : ranges) {
            auto& prev_range = ranges_merged.back();
            if (prev_range.end_address < range.base_address) {
                ranges_merged.emplace_back(range);
            } else {
                prev_range.end_address = std::max(prev_range.end_address, range.end_address);
            }
        }
    }
    if (fd_mode == 2) {
        // The block's ranges against this merge (the derivation's merge vs the original one).
        const auto* vi_ranges = AmdGpu::CpSplit::VertexInputRanges(*front_vi);
        bool same = ranges_merged.size() == front_vi->range_count;
        u32 index = 0;
        for (size_t i = 0; same && i < ranges_merged.size(); ++i) {
            same = ranges_merged[i].base_address == vi_ranges[i].base_address &&
                   ranges_merged[i].end_address == vi_ranges[i].end_address;
            index = static_cast<u32>(i);
        }
        if (!same) {
            g_fd.Mismatch("merged ranges", index, *front_vi);
        }
    }

    // SHADPS4_VB_REPEAT_STATS=1 (diagnostic): how often a draw binds exactly the merged vertex
    // ranges of the previous draw, i.e. the ceiling for reusing its ObtainBuffer results.
    const bool repeat_stats = g_vb_repeat_stats;
    if (repeat_stats) {
        static std::vector<std::pair<VAddr, VAddr>> previous;
        static u64 draws = 0, repeats = 0, with_ranges = 0, range_total = 0;
        bool same = previous.size() == ranges_merged.size();
        for (size_t i = 0; same && i < ranges_merged.size(); ++i) {
            same = previous[i].first == ranges_merged[i].base_address &&
                   previous[i].second == ranges_merged[i].end_address;
        }
        previous.clear();
        for (const auto& range : ranges_merged) {
            previous.emplace_back(range.base_address, range.end_address);
        }
        ++draws;
        with_ranges += !ranges_merged.empty();
        repeats += same && !ranges_merged.empty();
        range_total += ranges_merged.size();
        if ((draws & ((1u << 18) - 1)) == 0) {
            LOG_INFO(Render_Vulkan,
                     "VB_REPEAT_STATS: draws {} with_ranges {} same_as_previous {} ({:.1f}%) "
                     "ranges/draw {:.2f}",
                     draws, with_ranges, repeats, 100.0 * repeats / std::max<u64>(with_ranges, 1),
                     double(range_total) / draws);
        }
    }

    // Map buffers for merged ranges
    for (auto& range : ranges_merged) {
        const u64 size = memory->ClampRangeSize(range.base_address, range.GetSize());
        const auto [buffer, offset] = ObtainBuffer(range.base_address, size, false);
        range.vk_buffer = buffer->buffer;
        range.offset = offset;
        if (IsRegionGpuModified(range.base_address, size)) {
            if (auto barrier =
                    buffer->GetBarrier(vk::AccessFlagBits2::eVertexAttributeRead,
                                       vk::PipelineStageFlagBits2::eVertexAttributeInput)) {
                barriers.emplace_back(*barrier);
            }
        }
    }

    // Bind vertex buffers
    Vulkan::VertexInputs<vk::Buffer> host_buffers;
    Vulkan::VertexInputs<vk::DeviceSize> host_offsets;
    Vulkan::VertexInputs<vk::DeviceSize> host_sizes;
    Vulkan::VertexInputs<vk::DeviceSize> host_strides;
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            const auto host_buffer_info =
                std::ranges::find_if(ranges_merged, [&](const BufferRange& range) {
                    return buffer.base_address >= range.base_address &&
                           buffer.base_address < range.end_address;
                });
            ASSERT(host_buffer_info != ranges_merged.cend());
            host_buffers.emplace_back(host_buffer_info->vk_buffer);
            host_offsets.push_back(host_buffer_info->offset + buffer.base_address -
                                   host_buffer_info->base_address);
        } else {
            host_buffers.emplace_back(VK_NULL_HANDLE);
            host_offsets.push_back(0);
        }
        host_sizes.push_back(buffer.GetSize());
        host_strides.push_back(buffer.GetStride());
    }
    if (g_native_census || g_native_pass != 0) [[unlikely]] {
        auto& nv = t_native_vi;
        nv.n = static_cast<u32>(std::min<size_t>(host_buffers.size(), nv.buf.size()));
        for (u32 k = 0; k < nv.n; ++k) {
            nv.buf[k] = std::bit_cast<u64>(static_cast<VkBuffer>(host_buffers[k]));
            nv.off[k] = host_offsets[k];
            nv.stride[k] = static_cast<u32>(host_strides[k]);
        }
    }

    scheduler.RecordArrays(
        [dynamic = instance.IsVertexInputDynamicState()](vk::CommandBuffer cmdbuf, auto buffers,
                                                         auto offsets, auto sizes, auto strides) {
            if (dynamic) {
                cmdbuf.bindVertexBuffers(0, u32(buffers.size()), buffers.data(), offsets.data());
            } else {
                cmdbuf.bindVertexBuffers2(0, u32(buffers.size()), buffers.data(), offsets.data(),
                                          sizes.data(), strides.data());
            }
        },
        std::span<const vk::Buffer>{host_buffers}, std::span<const vk::DeviceSize>{host_offsets},
        std::span<const vk::DeviceSize>{host_sizes}, std::span<const vk::DeviceSize>{host_strides});
}

void BufferCache::BindVertexBuffersDerived(
    const AmdGpu::CpSplit::VertexInputsPayload& vi,
    boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers) {
    // BindVertexBuffers with the front's block: the attributes, bindings and merged ranges are
    // read from the record (no V# reads, no sort); the address-space and cache queries and the
    // recording stay here.
    using namespace AmdGpu::CpSplit;
    const auto* attribs = VertexInputAttribs(vi);
    const auto* vi_ranges = VertexInputRanges(vi);
    const bool dynamic = instance.IsVertexInputDynamicState();
    if (dynamic) {
        auto& dynamic_state = scheduler.GetDynamicState();
        if (g_front_derive_vi_skip && !dynamic_state.vertex_input_unknown &&
            vi.desc_id == g_fd.recorded_desc_id) {
            // The command buffer holds this description already: recorded for an earlier draw
            // and nothing invalidated it since (DynamicState::Invalidate).
            ++g_fd.reused;
        } else {
            Vulkan::VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
            Vulkan::VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
            BuildVertexInputArrays(vi, attributes, bindings);
            scheduler.RecordArrays(
                [](vk::CommandBuffer cmdbuf, auto bindings, auto attributes) {
                    cmdbuf.setVertexInputEXT(bindings, attributes);
                },
                std::span<const vk::VertexInputBindingDescription2EXT>{bindings},
                std::span<const vk::VertexInputAttributeDescription2EXT>{attributes});
            dynamic_state.vertex_input_unknown = false;
            g_fd.recorded_desc_id = vi.desc_id;
        }
    }

    if (vi.count == 0) {
        // No bindings: nothing further to do.
        return;
    }

    // Map buffers for the merged ranges.
    struct Mapped {
        vk::Buffer vk_buffer;
        u64 offset;
    };
    std::array<Mapped, MaxVertexInputs> mapped;
    const u64 pb_t0 = g_pb_census != 0 && t_pb_sampled ? __rdtsc() : 0;
    for (u32 i = 0; i < vi.range_count; ++i) {
        const auto& range = vi_ranges[i];
        const u64 size =
            memory->ClampRangeSize(range.base_address, range.end_address - range.base_address);
        const auto [buffer, offset] = ObtainBuffer(range.base_address, size, false);
        mapped[i] = {buffer->buffer, offset};
        if (g_pb_census != 0) [[unlikely]] {
            t_pb_vi.bufs[i] = std::bit_cast<u64>(static_cast<VkBuffer>(mapped[i].vk_buffer));
        }
        if (IsRegionGpuModified(range.base_address, size)) {
            t_native_vi.gpu_mod = true; // SHADPS4_NATIVE_COLOUR: the open adds a storage barrier
            if (g_pb_census != 0) [[unlikely]] {
                ++t_pb_vi.gpu_mod;
            }
            if (auto barrier =
                    buffer->GetBarrier(vk::AccessFlagBits2::eVertexAttributeRead,
                                       vk::PipelineStageFlagBits2::eVertexAttributeInput)) {
                barriers.emplace_back(*barrier);
            }
        }
    }

    if (g_pb_census != 0) [[unlikely]] {
        t_pb_vi.nranges = vi.range_count;
        if (pb_t0 != 0) {
            t_pb_vi.obtain_cyc += __rdtsc() - pb_t0;
        }
    }
    // Bind vertex buffers. With dynamic vertex input the sizes and strides are not used
    // (bindVertexBuffers), so they are not cloned into the recorder.
    Vulkan::VertexInputs<vk::Buffer> host_buffers;
    Vulkan::VertexInputs<vk::DeviceSize> host_offsets;
    Vulkan::VertexInputs<vk::DeviceSize> host_sizes;
    Vulkan::VertexInputs<vk::DeviceSize> host_strides;
    for (u32 i = 0; i < vi.count; ++i) {
        const auto& attrib = attribs[i];
        if (attrib.range != VertexInputNoRange) {
            ASSERT(attrib.range < vi.range_count);
            host_buffers.emplace_back(mapped[attrib.range].vk_buffer);
            host_offsets.push_back(mapped[attrib.range].offset + attrib.base_address -
                                   vi_ranges[attrib.range].base_address);
        } else {
            host_buffers.emplace_back(VK_NULL_HANDLE);
            host_offsets.push_back(0);
        }
        if (!dynamic) {
            host_sizes.push_back(attrib.size);
            host_strides.push_back(attrib.desc.stride);
        }
    }
    if (g_native_census || g_native_pass != 0) [[unlikely]] {
        auto& nv = t_native_vi;
        nv.n = std::min<u32>(vi.count, static_cast<u32>(nv.buf.size()));
        for (u32 k = 0; k < nv.n; ++k) {
            nv.buf[k] = std::bit_cast<u64>(static_cast<VkBuffer>(host_buffers[k]));
            nv.off[k] = host_offsets[k];
            nv.stride[k] = attribs[k].desc.stride;
        }
    }

    scheduler.RecordArrays(
        [dynamic](vk::CommandBuffer cmdbuf, auto buffers, auto offsets, auto sizes, auto strides) {
            if (dynamic) {
                cmdbuf.bindVertexBuffers(0, u32(buffers.size()), buffers.data(), offsets.data());
            } else {
                cmdbuf.bindVertexBuffers2(0, u32(buffers.size()), buffers.data(), offsets.data(),
                                          sizes.data(), strides.data());
            }
        },
        std::span<const vk::Buffer>{host_buffers}, std::span<const vk::DeviceSize>{host_offsets},
        std::span<const vk::DeviceSize>{host_sizes}, std::span<const vk::DeviceSize>{host_strides});
}

const Buffer* BufferCache::NativeBufferAt(VAddr addr, u64 size, u32 site) {
    if (addr == 0) {
        return nullptr;
    }
    const BufferId id = page_table[addr >> CACHING_PAGEBITS].buffer_id;
    if (!id) {
        return nullptr;
    }
    const Buffer& found = slot_buffers[id];
    if (g_lane_probe && !found.is_deleted && Common::FaultMonitor::t_is_cp_thread &&
        !Vulkan::Scheduler::t_lane_mode) [[unlikely]] {
        // q9b: the native back paths read through the page table (segment members); the whole
        // cached buffer counts as read (conservative). q26 E2: the caller's range when it has one.
        // q37 H2: SHADPS4_LANE_LOOKUP_NOSTAMP=1 - handle lookups with no GPU read of their own (31-33) do not stamp.
        static const bool lookup_nostamp = [] {
            const char* e = std::getenv("SHADPS4_LANE_LOOKUP_NOSTAMP");
            return e && e[0] == '1';
        }();
        if (lookup_nostamp && site >= 31 && site <= 33) {
            // no stamp
        } else if (g_lane_native_exact && size != 0) {
            LaneProbeRead(addr, addr + size, scheduler.CurrentTick(), 10, 12);
        } else if (site != 23) {
            LaneProbeRead(found.CpuAddr(), found.CpuAddr() + found.SizeBytes(), scheduler.CurrentTick(), 10, site);
        } else {
            LaneProbeRead(found.CpuAddr(), found.CpuAddr() + found.SizeBytes(), scheduler.CurrentTick(), 10, 23);
        }
    }
    return found.is_deleted ? nullptr : &found;
}

const Buffer* BufferCache::NativeRunSpan(
    VAddr lo, VAddr hi, NativeRunAccess access,
    boost::container::small_vector<vk::BufferMemoryBarrier2, 16>* barriers, u64 dispatch_run,
    bool sync, int gpu_mod_hint, bool* gpu_mod_out) {
    if (lo == 0 || hi <= lo || hi - lo > (u64{64} << 20)) {
        return nullptr;
    }
    if (g_lane_probe && Common::FaultMonitor::t_is_cp_thread && !Vulkan::Scheduler::t_lane_mode)
        [[unlikely]] {
        LaneProbeRead(lo, hi, scheduler.CurrentTick(), 10, 13); // q9 census
    }
    const BufferId id = page_table[lo >> CACHING_PAGEBITS].buffer_id;
    if (!id) {
        return nullptr;
    }
    const Buffer& found = slot_buffers[id];
    if (found.is_deleted || !found.IsInBounds(lo, hi - lo)) {
        return nullptr;
    }
    const u64 size = memory->ClampRangeSize(lo, hi - lo);
    // SHADPS4_NATIVE_REC_PREP (s402: ~430 cycles per span): the lean read path's clean answer
    // first (memo probe + READ_SYNC_FAST's clean check, no upload due); the full ObtainBuffer
    // only when it says no.
    Buffer* buffer = nullptr;
    if (!sync) {
        // SHADPS4_NATIVE_GPU_DIRECT: the caller knows the range is already synced (a superset
        // validated since the last CPU write); only the access barrier below.
        buffer = &slot_buffers[id];
        buffer->NoteMainAccess(scheduler.CurrentTick(), false); // as ObtainBuffer / the lean path
    }
    if (!buffer && LeanReadOk()) {
        buffer = ObtainReadLean(lo, static_cast<u32>(size), id).first;
    }
    if (!buffer) {
        buffer = ObtainBuffer(lo, static_cast<u32>(size), false).first;
    }
    if (buffer != &found) {
        return nullptr;
    }
    switch (access) {
    case NativeRunAccess::Storage:
        if (auto barrier = buffer->GetBarrier(vk::AccessFlagBits2::eShaderRead,
                                              vk::PipelineStageFlagBits2::eAllCommands, 0,
                                              dispatch_run)) {
            barriers->emplace_back(*barrier);
        }
        break;
    case NativeRunAccess::Vertex: {
        // gpu_mod_hint (SHADPS4_NATIVE_GPU_DIRECT): the caller's cached answer for a superset
        // that saw no GPU write since (1 may only add a barrier the walk would not).
        const bool gm = gpu_mod_hint >= 0 ? gpu_mod_hint != 0 : IsRegionGpuModified(lo, size);
        if (gpu_mod_out) {
            *gpu_mod_out = gm;
        }
        if (gm) {
            if (auto barrier =
                    buffer->GetBarrier(vk::AccessFlagBits2::eVertexAttributeRead,
                                       vk::PipelineStageFlagBits2::eVertexAttributeInput)) {
                barriers->emplace_back(*barrier);
            }
        }
        break;
    }
    case NativeRunAccess::Index:
        if (gpu_mod_hint >= 0 ? gpu_mod_hint != 0 : IsRegionGpuModified(lo, size)) {
            if (gpu_mod_out) {
                *gpu_mod_out = true;
            }
            if (auto barrier = buffer->GetBarrier(vk::AccessFlagBits2::eIndexRead,
                                                  vk::PipelineStageFlagBits2::eIndexInput)) {
                barriers->emplace_back(*barrier);
            }
        }
        break;
    }
    return buffer;
}

void BufferCache::NativeJoinVertexIndex(
    const AmdGpu::CpSplit::VertexInputsPayload& vi, u32 index_offset, NativeJoinVi& out,
    boost::container::small_vector<vk::BufferMemoryBarrier2, 16>* barriers) {
    using namespace AmdGpu::CpSplit;
    const auto* attribs = VertexInputAttribs(vi);
    const auto* vi_ranges = VertexInputRanges(vi);
    struct Mapped {
        vk::Buffer vk_buffer;
        u64 offset;
    };
    std::array<Mapped, MaxVertexInputs> mapped{};
    for (u32 i = 0; i < vi.range_count; ++i) {
        const auto& range = vi_ranges[i];
        const u64 size =
            memory->ClampRangeSize(range.base_address, range.end_address - range.base_address);
        const auto [buffer, offset] = ObtainBuffer(range.base_address, size, false);
        mapped[i] = {buffer->buffer, offset};
        if (barriers && IsRegionGpuModified(range.base_address, size)) {
            if (auto barrier =
                    buffer->GetBarrier(vk::AccessFlagBits2::eVertexAttributeRead,
                                       vk::PipelineStageFlagBits2::eVertexAttributeInput)) {
                barriers->emplace_back(*barrier);
            }
        }
    }
    out.n = std::min<u32>(vi.count, static_cast<u32>(out.buf.size()));
    for (u32 k = 0; k < out.n; ++k) {
        const auto& attrib = attribs[k];
        if (attrib.range != VertexInputNoRange) {
            out.buf[k] = std::bit_cast<u64>(static_cast<VkBuffer>(mapped[attrib.range].vk_buffer));
            out.off[k] = mapped[attrib.range].offset + attrib.base_address -
                         vi_ranges[attrib.range].base_address;
        } else {
            out.buf[k] = 0;
            out.off[k] = 0;
        }
        out.stride[k] = attrib.desc.stride;
    }
    const auto& regs = liverpool->BackRegs();
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    const VAddr index_address =
        regs.index_base_address.Address<VAddr>() + index_offset * index_size;
    const u32 index_buffer_size = regs.num_indices * index_size;
    const auto [vk_buffer, offset] = ObtainBuffer(index_address, index_buffer_size, false);
    if (barriers && IsRegionGpuModified(index_address, index_buffer_size)) {
        if (auto barrier = vk_buffer->GetBarrier(vk::AccessFlagBits2::eIndexRead,
                                                 vk::PipelineStageFlagBits2::eIndexInput)) {
            barriers->emplace_back(*barrier);
        }
    }
    out.ib_buf = std::bit_cast<u64>(static_cast<VkBuffer>(vk_buffer->Handle()));
    out.ib_off = offset;
}

u32 IndirectIndexCapacity(const AmdGpu::Regs& regs, u32 path) {
    if (g_ib_indirect_cap == 0) {
        return 0;
    }
    const u32 num = regs.num_indices;
    const u32 cap = std::min<u32>(std::max(num, regs.max_index_size), 1u << 24);
    // Census: indexed indirect draws by path, of which the old rule bound fewer indices than the
    // capacity (and the largest shortfall), logged every 2^12.
    static std::array<std::atomic<u64>, 2> total{}, short_n{};
    static std::atomic<u32> worst{0};
    const u64 t = total[path & 1].fetch_add(1, std::memory_order_relaxed) + 1;
    if (cap > num) {
        short_n[path & 1].fetch_add(1, std::memory_order_relaxed);
        u32 w = worst.load(std::memory_order_relaxed);
        while (cap - num > w && !worst.compare_exchange_weak(w, cap - num)) {
        }
    }
    if ((t & 0xfff) == 0) {
        LOG_INFO(Render_Vulkan,
                 "IbIndirectCap (mode {}): indexed indirect draws generic {} (old range short {}) "
                 "A2 {} (short {}), worst shortfall {} indices",
                 g_ib_indirect_cap, total[0].load(), short_n[0].load(), total[1].load(),
                 short_n[1].load(), worst.load());
    }
    return g_ib_indirect_cap == 1 ? cap : 0;
}

void BufferCache::BindIndexBuffer(
    u32 index_offset, boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers,
    u32 num_indices) {
    const auto& regs = liverpool->BackRegs();

    // Figure out index type and size.
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const vk::IndexType index_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    const VAddr index_address =
        regs.index_base_address.Address<VAddr>() + index_offset * index_size;

    // Bind index buffer. r4: an override (indexed indirect capacity) is clamped to the guest memory
    // mapped contiguously from the index base (TryClampRangeSize clamps only from 1 GB up: ask for
    // at least that much); unmapped -> the old range.
    u32 index_buffer_size = regs.num_indices * index_size;
    if (num_indices != 0 && num_indices > regs.num_indices) {
        const u64 want = u64(num_indices) * index_size;
        const u64 mapped = Core::Memory::Instance()->TryClampRangeSize(
            index_address, std::max<u64>(want, u64{1} << 30));
        const u64 bytes = std::min(want, mapped) / index_size * index_size;
        if (bytes > index_buffer_size) {
            index_buffer_size = static_cast<u32>(bytes);
        }
    }
    const u64 pb_t0 = g_pb_census != 0 && t_pb_sampled ? __rdtsc() : 0;
    const auto [vk_buffer, offset] = ObtainBuffer(index_address, index_buffer_size, false);
    if (g_pb_census != 0) [[unlikely]] {
        t_pb_vi.ib_buf = std::bit_cast<u64>(static_cast<VkBuffer>(vk_buffer->Handle()));
        t_pb_vi.ib_type = is_index16 ? 1u : 2u;
        if (pb_t0 != 0) {
            t_pb_vi.obtain_cyc += __rdtsc() - pb_t0;
        }
    }
    if (IsRegionGpuModified(index_address, index_buffer_size)) {
        if (auto barrier = vk_buffer->GetBarrier(vk::AccessFlagBits2::eIndexRead,
                                                 vk::PipelineStageFlagBits2::eIndexInput)) {
            barriers.emplace_back(*barrier);
        }
    }
    if (g_native_census || g_native_pass != 0) [[unlikely]] {
        t_native_vi.ib_buf = std::bit_cast<u64>(static_cast<VkBuffer>(vk_buffer->Handle()));
        t_native_vi.ib_off = offset;
        t_native_vi.ib_type = index_size;
    }
    scheduler.Record([buffer = vk_buffer->Handle(), offset, index_type](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindIndexBuffer(buffer, offset, index_type);
    });
}

void BufferCache::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    WriterTagScope census_tag{*this, 8};
    ASSERT_MSG(address % 4 == 0, "GDS offset must be dword aligned");
    if (!is_gds) {
        texture_cache.ClearMeta(address);
        if (!IsRegionGpuModified(address, num_bytes)) {
            u32* buffer = std::bit_cast<u32*>(address);
            std::fill(buffer, buffer + num_bytes / sizeof(u32), value);
            memory_tracker->ClearRecent(address, num_bytes);
            // SHADPS4_NATIVE_COHERENT: the CP wrote guest memory mid-submission; later reads in
            // this submission skip validity, so the bytes are due now.
            CoherentCpWrite(address, num_bytes);
            return;
        }
    }
    if (is_gds && g_lane_on && Vulkan::Scheduler::t_lane_mode) [[unlikely]] {
        scheduler.LaneNeedMain(gds_buffer.main_access_tick, 3, true); // review S2
        scheduler.LaneNeedMain(g_gfx_label_tick, 2, true);
    }
    Buffer* buffer = [&] {
        if (is_gds) {
            return &gds_buffer;
        }
        const auto [buffer, offset] = ObtainBuffer(address, num_bytes, true);
        return buffer;
    }();
    buffer->NoteMainAccess(scheduler.CurrentTick(), true);
    if (is_gds && g_record_stream == 0) {
        buffer->CheckMainVsAsc(buffer->CpuAddr(), buffer->SizeBytes(), true);
    }
    if (g_range_notes && !is_gds) {
        buffer->NoteMainRange(scheduler.CurrentTick(), true, address, num_bytes, 8, 0);
    }
    buffer->Fill(buffer->Offset(address), num_bytes, value);
}

void BufferCache::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    WriterTagScope census_tag{*this, 4};
    if (!dst_gds && !IsRegionGpuModified(dst, num_bytes)) {
        if (!src_gds && !IsRegionGpuModified(src, num_bytes) &&
            !texture_cache.FindImageFromRange(src, num_bytes)) {
            // Both buffers were not transferred to GPU yet. Can safely copy in host memory.
            memcpy(std::bit_cast<void*>(dst), std::bit_cast<void*>(src), num_bytes);
            memory_tracker->ClearRecent(dst, num_bytes);
            CoherentCpWrite(dst, num_bytes); // SHADPS4_NATIVE_COHERENT: as FillBuffer's CPU path.
            return;
        }
        // Without a readback there's nothing we can do with this
        // Fallback to creating dst buffer on GPU to at least have this data there
    }
    texture_cache.InvalidateMemoryFromGPU(dst, num_bytes);
    if (g_lane_on && Vulkan::Scheduler::t_lane_mode) [[unlikely]] {
        if (!src_gds && texture_cache.FindImageFromRange(src, num_bytes)) {
            scheduler.SwitchStream(0); // the packet finishes on main, after the lane so far
            scheduler.LaneFallbackMain();
        } else {
            g_lc_done = scheduler.GetMasterSemaphore()->KnownGpuTick();
            if (!src_gds) {
                const u64 t = LaneMainProducer(src, src + num_bytes);
                scheduler.LaneNeedMain(t, g_lc_src);
            }
            if (!dst_gds) {
                const u64 t = LaneMainProducer(dst, dst + num_bytes);
                scheduler.LaneNeedMain(t, g_lc_src);
            }
            if (src_gds || dst_gds) {
                scheduler.LaneNeedMain(gds_buffer.main_access_tick, 3, true);
            }
            if (dst_gds) {
                scheduler.LaneNeedMain(g_gfx_label_tick, 2, true); // WAR: the copy writes
            } else {
                LaneWarWait(scheduler, dst, dst + num_bytes); // q9c
            }
        }
    }
    auto& src_buffer = [&] -> const Buffer& {
        if (src_gds) {
            return gds_buffer;
        }
        const auto buffer_id = FindBuffer(src, num_bytes);
        auto& buffer = slot_buffers[buffer_id];
        SynchronizeBuffer(buffer, src, num_bytes, false, true, nullptr);
        return buffer;
    }();
    auto& dst_buffer = [&] -> const Buffer& {
        if (dst_gds) {
            return gds_buffer;
        }
        const auto buffer_id = FindBuffer(dst, num_bytes);
        auto& buffer = slot_buffers[buffer_id];
        SynchronizeBuffer(buffer, dst, num_bytes, true, true, nullptr);
        NoteGpuWrite(dst, num_bytes);
        AddGpuModified(gpu_modified_ranges, dst, num_bytes);
        Common::FaultMonitor::CountRangeAdd(gpu_modified_ranges.Fragments());
        return buffer;
    }();
    src_buffer.NoteMainAccess(scheduler.CurrentTick(), false);
    dst_buffer.NoteMainAccess(scheduler.CurrentTick(), true);
    if (g_lane_probe && !src_gds && Common::FaultMonitor::t_is_cp_thread &&
        !Vulkan::Scheduler::t_lane_mode) [[unlikely]] {
        LaneProbeRead(src, src + num_bytes, scheduler.CurrentTick(), 10, 14); // q9 census
    }
    if (g_range_notes) {
        if (!src_gds) {
            src_buffer.NoteMainRange(scheduler.CurrentTick(), false, src, num_bytes, 4, 0);
        }
        if (!dst_gds) {
            dst_buffer.NoteMainRange(scheduler.CurrentTick(), true, dst, num_bytes, 4, 0);
        }
    }
    if (g_record_stream == 0) {
        if (src_gds) {
            src_buffer.CheckMainVsAsc(src_buffer.CpuAddr(), src_buffer.SizeBytes(), false);
        }
        if (dst_gds) {
            dst_buffer.CheckMainVsAsc(dst_buffer.CpuAddr(), dst_buffer.SizeBytes(), true);
        }
    }
    const vk::BufferCopy region = {
        .srcOffset = src_buffer.Offset(src),
        .dstOffset = dst_buffer.Offset(dst),
        .size = num_bytes,
    };
    const vk::BufferMemoryBarrier2 buf_barriers_before[2] = {
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .buffer = dst_buffer.Handle(),
            .offset = dst_buffer.Offset(dst),
            .size = num_bytes,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .buffer = src_buffer.Handle(),
            .offset = src_buffer.Offset(src),
            .size = num_bytes,
        },
    };
    scheduler.EndRendering();
    const vk::BufferMemoryBarrier2 buf_barriers_after[2] = {
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .buffer = dst_buffer.Handle(),
            .offset = dst_buffer.Offset(dst),
            .size = num_bytes,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .buffer = src_buffer.Handle(),
            .offset = src_buffer.Offset(src),
            .size = num_bytes,
        },
    };
    if (g_upload_ahead_stats) {
        ++g_uas.copies;
        NoteTransferBegin();
    }
    SCOPE_EXIT {
        if (g_upload_ahead_stats) {
            NoteTransferEnd();
        }
    };
    if (g_transfer_coalesce) {
        const Vulkan::Scheduler::TransferRange read{static_cast<VkBuffer>(src_buffer.Handle()),
                                                    region.srcOffset,
                                                    region.srcOffset + region.size};
        const Vulkan::Scheduler::TransferRange write{static_cast<VkBuffer>(dst_buffer.Handle()),
                                                     region.dstOffset,
                                                     region.dstOffset + region.size};
        scheduler.Record<true>([source = src_buffer.Handle(), destination = dst_buffer.Handle(),
                                region, read, write,
                                sched = &scheduler](vk::CommandBuffer cmdbuf) {
            sched->BeginTransferOnWorker(cmdbuf, read, write);
            cmdbuf.copyBuffer(source, destination, region);
        });
        return;
    }
    scheduler.Record([source = src_buffer.Handle(), destination = dst_buffer.Handle(), region,
                      before = std::to_array(buf_barriers_before),
                      after = std::to_array(buf_barriers_after)](vk::CommandBuffer cmdbuf) {
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 2,
            .pBufferMemoryBarriers = before.data(),
        });
        cmdbuf.copyBuffer(source, destination, region);
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 2,
            .pBufferMemoryBarriers = after.data(),
        });
    });
}

u64 BufferCache::RmemoCensusStamp(const Buffer& buffer) {
    auto& c = rmemo_census;
    if (!c.started) [[unlikely]] {
        c.started = true;
        c.gen = 1; // a stamp is never 0, the value of a never-stamped entry
        c.dirty_head = DirtyLog::Start();
        c.tsc0 = __rdtsc();
        c.t0 = std::chrono::steady_clock::now();
        LOG_INFO(Render, "RmemoCensus: on (back_clean={}, pb_census={}; needs 1 / 0)", g_back_clean,
                 g_pb_census);
    }
    // A second consumer of the dirty-page log: every page range marked CPU-dirty since the last
    // drain bumps the counter of each cache Buffer its 16 KiB pages map to now.
    const bool ok = DirtyLog::Drain(c.dirty_head, [&](u64 first_page, u64 count) {
        ++c.drained;
        const u64 lo = (first_page << TRACKER_PAGE_BITS) >> CACHING_PAGEBITS;
        const u64 hi = (((first_page + count) << TRACKER_PAGE_BITS) - 1) >> CACHING_PAGEBITS;
        BufferId last{};
        for (u64 page = lo; page <= hi; ++page) {
            const PageData* entry = page_table.find(page);
            if (!entry || IsBufferInvalid(entry->buffer_id) || entry->buffer_id == last ||
                !slot_buffers.is_allocated(entry->buffer_id)) {
                continue;
            }
            last = entry->buffer_id;
            ++slot_buffers[last].census_dirty_seq;
            ++c.bumps;
        }
    });
    if (!ok) [[unlikely]] {
        ++c.gen;
        ++c.resets;
    }
    return (c.gen << 48) | (buffer.census_dirty_seq & ((1ULL << 48) - 1));
}

void BufferCache::RmemoCensusCount(int cls, const Buffer& buffer, bool epoch_same, u64 cyc,
                                   bool timed) {
    auto& c = rmemo_census;
    ++c.calls[cls];
    const u64 mib = buffer.SizeBytes() >> 20;
    const int size_class = mib < 1      ? 0
                           : mib < 4    ? 1
                           : mib < 16   ? 2
                           : mib < 64   ? 3
                           : mib < 256  ? 4
                           : mib < 1024 ? 5
                                        : 6;
    ++c.size_class[cls][size_class];
    int path = 0;
    if (cls != 0) {
        const u8 bc = memory_tracker->census_bc;
        path = last_sync_upload_bytes != 0 ? 2 : bc == 1 ? 0 : bc == 2 ? 1 : 3;
        ++c.path[cls][path];
        if (cls == 2 && epoch_same) {
            ++c.epoch_same_path[path];
        }
    }
    if (timed) {
        ++c.timed[cls];
        c.timed_cyc[cls] += cyc;
        if (cls != 0 && path == 0) {
            ++c.clean_timed[cls];
            c.clean_cyc[cls] += cyc;
            c.clean_check_cyc[cls] += memory_tracker->census_check_cyc;
        }
    }
    const u64 total = c.calls[0] + c.calls[1] + c.calls[2] + c.calls[3];
    if (total < c.next_log) [[likely]] {
        return;
    }
    c.next_log = total + (1ULL << 22);
    const double us = std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - c.t0)
                          .count();
    const double tsc_per_ns = us > 0 ? double(__rdtsc() - c.tsc0) / (us * 1000.0) : 1.0;
    const auto ns = [&](u64 cycles, u64 n) { return n ? double(cycles) / n / tsc_per_ns : 0.0; };
    const auto& p = c.path;
    const auto& s = c.size_class;
    LOG_INFO(Render,
             "RmemoCensus: calls hit/key/epoch/other={}/{}/{}/{} | path clean/walk0/upload/nonbc "
             "key={}/{}/{}/{} epoch={}/{}/{}/{} other={}/{}/{}/{} | epoch own-buffer-clean by path "
             "{}/{}/{}/{} | size <1/4/16/64/256/1024/more MiB hit={}/{}/{}/{}/{}/{}/{} key={}/{}/{}/{}/"
             "{}/{}/{} epoch={}/{}/{}/{}/{}/{}/{} other={}/{}/{}/{}/{}/{}/{} | ns hit={:.1f} (n={}) "
             "miss key={:.1f} epoch={:.1f} other={:.1f} | clean miss ns key={:.1f} check={:.1f} "
             "epoch={:.1f} check={:.1f} other={:.1f} check={:.1f} (n={}/{}/{}) | drained={} bumps={} "
             "resets={} tsc/ns={:.3f}",
             c.calls[0], c.calls[1], c.calls[2], c.calls[3], p[1][0], p[1][1], p[1][2], p[1][3],
             p[2][0], p[2][1], p[2][2], p[2][3], p[3][0], p[3][1], p[3][2], p[3][3],
             c.epoch_same_path[0], c.epoch_same_path[1], c.epoch_same_path[2],
             c.epoch_same_path[3], s[0][0], s[0][1], s[0][2], s[0][3], s[0][4], s[0][5], s[0][6],
             s[1][0], s[1][1], s[1][2], s[1][3], s[1][4], s[1][5], s[1][6], s[2][0], s[2][1],
             s[2][2], s[2][3], s[2][4], s[2][5], s[2][6], s[3][0], s[3][1], s[3][2], s[3][3],
             s[3][4], s[3][5], s[3][6], ns(c.timed_cyc[0], c.timed[0]), c.timed[0],
             ns(c.timed_cyc[1], c.timed[1]), ns(c.timed_cyc[2], c.timed[2]),
             ns(c.timed_cyc[3], c.timed[3]), ns(c.clean_cyc[1], c.clean_timed[1]),
             ns(c.clean_check_cyc[1], c.clean_timed[1]), ns(c.clean_cyc[2], c.clean_timed[2]),
             ns(c.clean_check_cyc[2], c.clean_timed[2]), ns(c.clean_cyc[3], c.clean_timed[3]),
             ns(c.clean_check_cyc[3], c.clean_timed[3]), c.clean_timed[1], c.clean_timed[2],
             c.clean_timed[3], c.drained, c.bumps, c.resets, tsc_per_ns);
}

std::pair<Buffer*, u32> BufferCache::ObtainBuffer(VAddr device_addr, u32 size, bool is_written,
                                                  bool is_texel_buffer, BufferId buffer_id,
                                                  Common::BufferSampleCaller sample_caller) {
    SCOPE_EXIT {
        g_q31_pre = false; // q31c: the pre-stamp capture is valid only inside this call
    };
    // This function is 4.93 us/draw, 61% of all per-draw CPU cost, and it has two halves that
    // share no code: the early exit below (a tracker query plus a guest-memory memcpy) and the
    // SynchronizeBuffer fall-through. Time them separately - the split decides which is worth
    // cutting, and it has never actually been measured.
    if (g_lane_on && Vulkan::Scheduler::t_lane_mode) [[unlikely]] {
        LbScope lb_scope{6}; // q11a
        g_lc_done = scheduler.GetMasterSemaphore()->KnownGpuTick();
        const u64 t = LaneMainProducer(device_addr, device_addr + size);
        if (g_lc_memo_mode != 0 && ((g_lc_memo_c[0] + g_lc_memo_c[1]) & 0xfffff) == 0) [[unlikely]] {
            LOG_INFO(Render_Vulkan, "q25 LaneProducerMemo: hits {} misses {} verified {} mismatches {} (mode {})",
                     g_lc_memo_c[0], g_lc_memo_c[1], g_lc_memo_c[2], g_lc_memo_c[3], g_lc_memo_mode);
        }
        const u64 open = scheduler.CurrentTick();
        u64 war_rd = 0;
        bool war_hit = false, war_cov = false;
        if (g_lane_probe && is_written) [[unlikely]] {
            war_cov = LaneProbeWrite(device_addr, device_addr + size, open, war_rd, war_hit);
            if (g_lane_seg_census && war_rd > g_lc_done + 1 && Q35Fs(g_lane_war_wid & 0xffffffffULL)) [[unlikely]] {
                static u32 n36 = 0; // q36c: lane writes held by a named draw's stamp (unfinished reader batch)
                if (n36++ < 96) {
                    LOG_INFO(Render_Vulkan, "q36c Held: lane write {:#x} size {:#x} (pipeline {:08x}) by fs {:08x} "
                             "window {:#x} stamp args {:#x} meta {:#x} rd {} open {} known {}", device_addr, size,
                             census_id, g_lane_war_wid & 0xffffffffULL, g_lane_war_win, g_lane_war_args,
                             g_lane_war_wid >> 56, war_rd, open, g_lc_done);
                }
            }
        }
        if (g_lane_who) [[unlikely]] {
            LaneWhoNote(t, open, is_written, war_rd, war_hit);
        }
        scheduler.LaneNeedMain(t, g_lc_src, false, LaneSiteRaw(g_lc_hit_wk));
        g_q60_census_id = census_id; // q60b census
        if (is_written && LaneFrameScopedWar(scheduler, device_addr, device_addr + size, "bind")) {
            // q60: the frame-fence frame's end (inside the particle arena).
        } else if (is_written) {
            if (g_lane_war && war_cov) {
                // q9b: WAR scoped to the newest main batch that read this pool range (deferred
                // if it is the open batch: a read recorded there is a real hazard); q9c: + 1.
                g_war_scoped.fetch_add(1, std::memory_order_relaxed);
                const u64 w = war_rd != 0 ? std::min(war_rd + (g_lane_win == 2 ? 0 : 1), open) : 0; // q26 E3-lite
                if (w >= open) {
                    g_war_open.fetch_add(1, std::memory_order_relaxed);
                }
                scheduler.LaneNeedMain(w, 2, false, LaneSiteWar(g_lane_war_site), g_lane_war_wid);
            } else {
                g_war_label.fetch_add(g_lane_war ? 1 : 0, std::memory_order_relaxed);
                scheduler.LaneNeedMain(g_gfx_label_tick, 2, true); // WAR, never the open batch (p1d)
            }
        }
    } else if (g_lane_probe && !is_written && Common::FaultMonitor::t_is_cp_thread) [[unlikely]] {
        if (g_lane_bind_uploads) {
            // q31b: quiet-ness before this draw's own stamp (its read is ordered after a lane upload by the pin).
            g_q31_pre_lo = device_addr;
            g_q31_pre_hi = device_addr + size;
            g_q31_pre_tick = scheduler.CurrentTick();
            g_q31_pre = LaneProbeQuiet(device_addr, device_addr + size, g_q31_pre_tick);
        }
        g_lane_stamp_id = census_id; // q34
        g_lane_stamp_args = g_census_args; // q35 H5
        if (g_lane_seg_census && Q35Fs(census_id) && LaneProbeAny(device_addr, device_addr + size)) [[unlikely]] {
            static std::vector<std::array<u64, 3>> seen36; // q36c: the named draws' stamps over probes
            const std::array<u64, 3> key{census_id, device_addr, size};
            if (seen36.size() < 96 && std::find(seen36.begin(), seen36.end(), key) == seen36.end()) {
                seen36.push_back(key);
                LOG_INFO(Render_Vulkan, "q36c Stamp: fs {:08x} vs {:08x} {:#x} size {:#x} texel {} args {:#x} flip {}",
                         census_id, g_census_vs, device_addr, size, is_texel_buffer, g_census_args,
                         g_back_flip_seq);
            }
        }
        if (g_lane_seg_census && LaneProbeAny(device_addr, device_addr + size)) [[unlikely]] {
            // q35: the read bindings behind the copy-holding stamps (big windows first, then those draws' small ones).
            static std::vector<std::array<u64, 5>> seen;
            static std::vector<u64> big_ids;
            static u32 small_n = 0;
            const bool big = size >= (u64{64} << 20);
            const std::array<u64, 5> key{census_id, g_census_vs, device_addr, size, g_census_args};
            const bool known_id = std::find(big_ids.begin(), big_ids.end(), census_id) != big_ids.end();
            if ((big ? seen.size() < 48 : (known_id && small_n < 64)) &&
                std::find(seen.begin(), seen.end(), key) == seen.end()) {
                seen.push_back(key);
                small_n += big ? 0 : 1;
                if (big && !known_id) {
                    big_ids.push_back(census_id);
                }
                std::string ps;
                const u32 pn = g_lprobe_n.load(std::memory_order_relaxed);
                for (u32 i = 0; i < pn; ++i) {
                    const u64 plo = g_lprobe[i].lo.load(std::memory_order_relaxed);
                    const u64 phi = g_lprobe[i].hi.load(std::memory_order_relaxed);
                    if (plo < device_addr + size && device_addr < phi) {
                        ps += fmt::format(" [{:#x}-{:#x}]", plo, phi);
                    }
                }
                LOG_INFO(Render_Vulkan, "q35 ReadBind: fs {:08x} vs {:08x} {} {:#x} size {:#x} ({} MiB) texel {} args {:#x} "
                         "flip {} | probes:{}", census_id, g_census_vs, big ? "WINDOW" : "small", device_addr, size,
                         size >> 20, is_texel_buffer, g_census_args, g_back_flip_seq, ps);
            }
        }
        LaneProbeRead(device_addr, device_addr + size, scheduler.CurrentTick(), 10, 15); // q9 census
        g_lane_stamp_id = 0;
        g_lane_stamp_args = 0;
    }
    if (g_lane_census && Common::FaultMonitor::t_is_cp_thread) [[unlikely]] {
        const u64 lo = device_addr, hi = device_addr + size;
        if (t_force_binding_sync) {
            const LaneNote* m = LaneFind(g_lc_main, lo, hi);
            const LaneNote* e = is_written ? nullptr : LaneFind(g_lc_early, lo, hi);
            g_lc[is_written ? 5 : 0].fetch_add(1, std::memory_order_relaxed);
            if (m && (!e || m->seq > e->seq)) {
                const bool open = m->tick >= scheduler.CurrentTick();
                const u32 idx = is_written ? (open ? 6 : 7) : (m->kind == 1 ? (open ? 3 : 4)
                                                                            : (open ? 1 : 2));
                g_lc[idx].fetch_add(1, std::memory_order_relaxed);
                t_lc_rec_flags |= open ? 1u : 2u;
            } else if (e) {
                g_lc[15].fetch_add(1, std::memory_order_relaxed);
            }
        } else if (!is_written) {
            g_lc[8].fetch_add(1, std::memory_order_relaxed);
            if (LaneFind(g_lc_early, lo, hi)) {
                const bool skip = g_native_coherent != 0 && coherent_live && !is_texel_buffer;
                g_lc[skip ? 9 : 10].fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    if (g_early_inv && Common::FaultMonitor::t_is_cp_thread) [[unlikely]] {
        if (t_force_binding_sync) {
            if (is_written) {
                if (g_early_w_tail - g_early_w_head >= g_early_writes.size()) {
                    ++g_early_w_head; // oldest dropped (coverage loss, not counted as inverted)
                }
                g_early_writes[g_early_w_tail++ % g_early_writes.size()] = {
                    device_addr, device_addr + size, t_early_pos};
                g_early_inv_writes.fetch_add(1, std::memory_order_relaxed);
            }
        } else if (!is_written) {
            g_early_inv_checks.fetch_add(1, std::memory_order_relaxed);
            for (u64 i = g_early_w_head; i != g_early_w_tail; ++i) {
                const auto& w = g_early_writes[i % g_early_writes.size()];
                if (w.pos > t_inorder_pos && w.lo < device_addr + size && device_addr < w.hi) {
                    g_early_inv_reads.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
            }
        }
    }
    // SHADPS4_NATIVE_COHERENT: memory was made coherent at the submission's start (CoherenceSync;
    // a buffer created since was synced whole in CreateBuffer). A read is the lookup only.
    if (CoherentSkip(is_written, is_texel_buffer)) {
        if (IsBufferInvalid(buffer_id)) {
            buffer_id = FindBuffer(device_addr, size);
        }
        Buffer& buffer = slot_buffers[buffer_id];
        buffer.NoteMainAccess(scheduler.CurrentTick(), false);
        ++coherent_stats.skips;
        if (g_native_coherent == 2) [[unlikely]] {
            CoherentVerify(buffer, device_addr, size);
        }
        return {&buffer, buffer.Offset(device_addr)};
    }
    // SHADPS4_LEAN: no probe object at all. An inactive probe still costs a ~100-byte construction,
    // the sampling-flag guard and a destructor check per call (a19 profile: 9 of 1279 CP samples
    // under BufferSampleProbe frames).
    using ProbeClock = Common::BufferSampleProbe::Clock;
    std::optional<Common::BufferSampleProbe> sample_storage;
    if (!::Common::Lean()) {
        sample_storage.emplace(Common::BufferSampleProbe::ShouldSample(), is_written,
                               is_texel_buffer, size, sample_caller);
    }
    Common::BufferSampleProbe* const sample = sample_storage ? &*sample_storage : nullptr;
    AmdGpu::MtbCensus::NoteUse(device_addr, size);
    const auto t_entry = Common::FaultMonitor::Now();
    u64 query_ns = 0;
    // SHADPS4_RMEMO_CENSUS: read syncs only, 1 in 64 timed with rdtsc.
    int census_cls = -1;
    bool census_same = false;
    u64 census_stamp = 0;
    const bool census_timed =
        g_rmemo_census && !is_written && !is_texel_buffer && ((++rmemo_census.seq & 63) == 0);
    const u64 census_c0 = census_timed ? __rdtsc() : 0;
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    //
    // That trade is unmeasured. The fast path costs a guest->staging memcpy per binding per draw
    // (6.6 calls/draw, ~43 MB/frame, 2.3 us/draw measured) and cannot be memoised, because it
    // never clears the range's CPU-modified bits and so never write-protects the pages - there is
    // no invalidation signal to hang a cache on. The fall-through is the fully tracked path, so
    // lowering this threshold can only cost performance, never correctness. Overridable to A/B it
    // without a rebuild; 0 disables the fast path entirely.
    if (!is_written && size <= StreamCopyMaxSize()) {
        const bool gpu_modified = IsRegionGpuModified(device_addr, size);
        const auto t_query = Common::FaultMonitor::Now();
        query_ns = Common::FaultMonitor::DiffNs(t_entry, t_query);
        if (!gpu_modified) {
            const u64 offset =
                stream_buffer.Copy(device_addr, size, instance.UniformMinAlignment());
            if (sample) {
                sample->RecordStream();
            }
            Common::FaultMonitor::CountObtainStream(query_ns,
                                                    Common::FaultMonitor::SinceNs(t_query), size);
            return {&stream_buffer, offset};
        }
    }
    if (host_mirror && !is_written && (!is_texel_buffer || host_mirror_texel) &&
        sample_caller != Common::BufferSampleCaller::Other) {
        if (const auto mirrored = TryHostMirror(device_addr, size, is_texel_buffer);
            mirrored.first) {
            return mirrored;
        }
    }
    const bool buffer_id_fallback = IsBufferInvalid(buffer_id);
    if (sample) {
        sample->RecordBufferIdFallback(buffer_id_fallback);
    }
    if (buffer_id_fallback) {
        const auto t_find = sample ? sample->Now() : ProbeClock::time_point{};
        buffer_id = FindBuffer(device_addr, size);
        if (sample) {
            sample->AddFindNs(sample->SinceNs(t_find));
        }
    }
    Buffer& buffer = slot_buffers[buffer_id];
    buffer.NoteMainAccess(scheduler.CurrentTick(), is_written);
    if (g_range_notes) {
        buffer.NoteMainRange(scheduler.CurrentTick(), is_written, device_addr, size, census_kind,
                             census_id);
    }
    if (g_asc_deps && g_record_stream == 1) {
        pending_asc.push_back({buffer_id, nullptr, device_addr, size, is_written});
    }
    // SHADPS4_READ_BIND_MEMO: a read-only range is synchronized when no page of it is CPU-dirty,
    // and g_dirty_epoch is bumped after every CPU-dirty publication (the invariant the tracker's
    // own clean memo, MemoSaysClean, already relies on). So if this exact (buffer, range) ran in
    // full at the current epoch and no buffer was created, joined, deleted, mapped or unmapped
    // since (StructGeneration), SynchronizeBuffer would upload nothing and change no state: skip
    // it and everything around it. GPU writes need no upload for a GPU read (the caller's
    // GetBarrier orders them). Texel reads also sync from images and are excluded; the stream
    // and host-mirror paths returned above. 58.9% of read bindings in fire repeat at an unchanged
    // epoch within a submission (DRAW_COST census, run diag-fire-a30-drawcost-20260928-t).
    ReadMemoEntry* read_memo = nullptr;
    u64 read_memo_epoch = 0;
    u64 read_memo_gen = 0;
    bool read_memo_verify = false;
    u64 read_memo_bytes_before = 0;
    // Shader bindings only: vertex/index/image callers (BufferSampleCaller::Other) hit 18% and the
    // probe cost them ~30 ns/draw (same-EXE DRAW_COST pair, runs diag-fire-rbmemo2-dc-*-20260928).
    if (g_read_bind_memo != 0 && !is_written && !is_texel_buffer &&
        sample_caller != Common::BufferSampleCaller::Other) {
        read_memo_epoch = g_dirty_epoch.load(std::memory_order_acquire);
        const bool serial_guard = g_read_bind_memo >= 3;
        read_memo_gen = serial_guard ? buffer.memo_serial : StructGeneration();
        if (g_rmemo_census) [[unlikely]] {
            census_stamp = RmemoCensusStamp(buffer);
        }
        const u64 h = (device_addr * 0x9E3779B97F4A7C15ULL) ^ (u64(size) * 0xC2B2AE3D27D4EB4FULL) ^
                      u64(buffer_id.index);
        read_memo = &read_memo_table[(h ^ (h >> 31)) & (read_memo_table.size() - 1)];
        auto& st = read_memo_stats[sample_caller == Common::BufferSampleCaller::Other ? 1 : 0];
        ++st[0];
        const bool key = read_memo->addr == device_addr && read_memo->size == size &&
                         read_memo->buffer_index == buffer_id.index;
        const bool epoch_ok = read_memo->epoch == read_memo_epoch;
        const bool guard_ok = read_memo->gen == read_memo_gen && (!serial_guard || read_memo_gen);
        if (!key) {
            ++st[2];
            if (g_rmemo_census) [[unlikely]] {
                census_cls = 1;
            }
        } else if (!epoch_ok) {
            ++st[3];
            if (g_rmemo_census) [[unlikely]] {
                census_cls = 2;
                census_same = read_memo->pb_written == census_stamp;
            }
        } else if (!guard_ok) {
            ++st[4];
        } else {
            ++st[1];
            if (g_pb_census != 0) [[unlikely]] {
                PbFoldMemo(read_memo->pb_written);
            }
            if (g_read_bind_memo == 1 || g_read_bind_memo == 3) {
                if (g_rmemo_census) [[unlikely]] {
                    RmemoCensusCount(0, buffer, false, census_timed ? __rdtsc() - census_c0 : 0,
                                     census_timed);
                }
                return {&buffer, buffer.Offset(device_addr)};
            }
            read_memo_verify = true;
            read_memo_bytes_before = sync_upload_bytes_total;
        }
        if ((++read_memo_calls & 0x3fffff) == 0) {
            const auto& b = read_memo_stats[0];
            const auto& o = read_memo_stats[1];
            LOG_INFO(Render,
                     "ReadBindMemo: bind calls={} hits={} miss key/epoch/guard={}/{}/{} | other "
                     "calls={} hits={} miss key/epoch/guard={}/{}/{} | verify_bad={}",
                     b[0], b[1], b[2], b[3], b[4], o[0], o[1], o[2], o[3], o[4],
                     read_memo_verify_bad);
        }
    }
    if (g_rmemo_census && !is_written && !is_texel_buffer &&
        sample_caller == Common::BufferSampleCaller::Other) [[unlikely]] {
        census_cls = 3;
    }
    const u64 census_c1 = census_timed && census_cls >= 1 ? __rdtsc() : 0;
    if (census_cls >= 1) [[unlikely]] {
        memory_tracker->census_bc = 0;
    }
    if (Offload::AscSpanStatsEnabled()) [[unlikely]] {
        auto& span = Offload::AscSpan();
        const u64 now = scheduler.CurrentTick();
        if (span.in_asc_dispatch) {
            span.w_access = std::max(span.w_access, buffer.diag_main_access_tick);
            span.w_write = std::max(span.w_write, buffer.diag_main_write_tick);
            ++span.binds;
            span.max_buffer_mb = std::max<u64>(span.max_buffer_mb, buffer.SizeBytes() >> 20);
        } else {
            buffer.diag_main_access_tick = now;
            if (is_written) {
                buffer.diag_main_write_tick = now;
            }
        }
    }
    // SHADPS4_WRITTEN_RUN_MEMO: the particle ring binds the same written ranges dispatch after
    // dispatch (fire: ~2.9 written bindings per dispatch at ~390 ns, ~1,000 dispatches/frame on
    // the path Game:Main waits for). A full written run leaves its range CPU-clean (uploaded),
    // GPU-marked in the tracker and gpu_modified_ranges, bumps gpu_write_seq in its regions and
    // records the range in this batch's write history. Repeating it in the same batch (tick)
    // while no page anywhere became CPU-dirty (g_dirty_epoch), no GPU bit or range was cleared
    // (gpu_clear_epoch, bumped by every Subtract / UnmarkRegionAsGpuModified) and no fault job
    // was admitted (a job admitted before the first run was already vetoed by that run: its seq
    // bump, and under SHADPS4_FAULT_VETO_EXACT its logged range, which is this same range - a job
    // the first run's range does not touch is not touched by the repeat either) only
    // bumps gpu_write_seq again, which nothing can observe: skip it. Unlike WRITTEN_MEMO (0 hits:
    // it demanded whole 4 MB regions CPU-clean) this needs only no dirtying since the first run.
    const int written_run = g_written_run_memo;
    WrittenRunEntry* run_entry = nullptr;
    WrittenRunEntry run_now{};
    bool run_verify = false;
    u64 run_bytes_before = 0;
    size_t run_ranges_before = 0;
    if (written_run != 0 && is_written) {
        run_now = {device_addr,
                   size,
                   buffer.memo_serial,
                   scheduler.CurrentTick(),
                   g_dirty_epoch.load(std::memory_order_acquire),
                   MemoryTracker::gpu_clear_epoch.load(std::memory_order_acquire),
                   fault_jobs_admitted.load(std::memory_order_acquire)};
        const u64 h = (device_addr * 0x9E3779B97F4A7C15ULL) ^ (u64(size) * 0xC2B2AE3D27D4EB4FULL);
        run_entry = &written_run_table[(h ^ (h >> 29)) & (written_run_table.size() - 1)];
        auto& st = written_run_stats;
        ++st[0];
        if ((st[0] & 0x7ffff) == 0) {
            LOG_INFO(Render,
                     "WrittenRunMemo: calls={} hits={} miss key/tick/dirty/clear/jobs="
                     "{}/{}/{}/{}/{} verify_bad={}",
                     st[0], st[1], st[2], st[3], st[4], st[5], st[6], st[7]);
        }
        const auto& e = *run_entry;
        if (e.addr != device_addr || e.size != size || e.serial != run_now.serial ||
            run_now.serial == 0) {
            ++st[2];
        } else if (e.tick != run_now.tick) {
            ++st[3];
        } else if (e.dirty_epoch != run_now.dirty_epoch) {
            ++st[4];
        } else if (e.clear_epoch != run_now.clear_epoch) {
            ++st[5];
        } else if (e.jobs != run_now.jobs) {
            ++st[6];
        } else {
            ++st[1];
            if (written_run == 1) {
                return {&buffer, buffer.Offset(device_addr)};
            }
            run_verify = true;
            run_bytes_before = sync_upload_bytes_total;
            run_ranges_before = gpu_modified_ranges.Fragments();
        }
    }
    // SHADPS4_WRITTEN_MEMO=1: a GPU-written binding of exactly this range was fully synchronized
    // and marked GPU-modified before, and since then no GPU-modified bit was cleared anywhere
    // (download/publication bump the epoch), no page in its regions became CPU-dirty, and no
    // readback job is pending (their write-sequence vote needs the marking). Then the walk
    // would upload nothing and change no bit: skip it, and the range add it would repeat.
    // Measured 2026-09-24: written bindings walk and lock every region with no fast path.
    const bool written_memo = g_written_memo;
    u64 memo_epoch = 0;
    WrittenMemoEntry* memo_entry = nullptr;
    if (written_memo && is_written) {
        static u64 calls = 0, hits = 0, miss_key = 0, miss_epoch = 0, miss_jobs = 0, miss_cpu = 0;
        ++calls;
        memo_epoch = MemoryTracker::gpu_clear_epoch.load(std::memory_order_acquire);
        const u64 h = (device_addr * 0x9E3779B97F4A7C15ULL) ^ (u64(size) * 0xC2B2AE3D27D4EB4FULL);
        memo_entry = &written_memo_table[(h ^ (h >> 29)) % written_memo_table.size()];
        bool hit = false;
        if (memo_entry->addr != device_addr || memo_entry->size != size) {
            ++miss_key;
        } else if (memo_entry->epoch != memo_epoch) {
            ++miss_epoch;
        } else if (pending_fault_jobs.load(std::memory_order_acquire) != 0) {
            ++miss_jobs;
        } else if (!memory_tracker->RangeCpuClean(device_addr, size)) {
            ++miss_cpu;
        } else {
            hit = true;
            ++hits;
        }
        if (!::Common::Lean() && (calls & 0xffff) == 0) {
            LOG_INFO(Render, "WrittenMemo: calls={} hits={} ({:.1f}%) miss_key={} miss_epoch={} "
                             "miss_jobs={} miss_cpu={}",
                     calls, hits, 100.0 * double(hits) / double(calls), miss_key, miss_epoch,
                     miss_jobs, miss_cpu);
        }
        if (hit) {
            // The write history (prebatch / fast lane) must still see this GPU write.
            NoteBatchWrite(device_addr, size);
            return {&buffer, buffer.Offset(device_addr)};
        }
    }
    // SHADPS4_READ_SYNC_FAST (notes/buffer-sync-validity-2026-10-07.md): 99.9% of the read syncs
    // that get here (read-memo misses, vertex / index bindings) find no CPU-dirty page, and the
    // back's private BackClean bits answer that in about a third of the full path's time
    // (census run city-s188). With no unverified page SynchronizeBuffer records nothing (no copy,
    // no protect change, no upload-ahead; texel syncs are excluded): return now. The memo entry
    // is written as the full path writes it (epoch and guard read before the check). =2 runs the
    // full path after every clean answer and counts uploads.
    const auto write_read_memo = [&] {
        // Epoch and generation read before the walk / check: a change during it misses next time.
        *read_memo = {device_addr, size, read_memo_epoch, read_memo_gen, buffer_id.index};
        if (g_pb_census != 0) [[unlikely]] {
            read_memo->pb_written = PbDraw();
        } else if (g_rmemo_census) [[unlikely]] {
            // Stamp read before the walk, like the epoch.
            read_memo->pb_written = census_stamp;
        }
    };
    bool fast_verify = false;
    u64 fast_tail = 0;
    if (g_read_sync_fast != 0 && !is_written && !is_texel_buffer && !read_memo_verify) {
        auto& fs = read_sync_fast_stats;
        if ((++fs[0] & 0xffffff) == 0) {
            LOG_INFO(Render,
                     "ReadSyncFast: mode={} checks={} clean={} ({:.2f}%) verify uploads raced={} "
                     "bad={}",
                     g_read_sync_fast, fs[0], fs[1], 100.0 * double(fs[1]) / double(fs[0]), fs[2],
                     fs[3]);
        }
        if (memory_tracker->ReadSyncClean(device_addr, size)) {
            ++fs[1];
            if (g_read_sync_fast == 1) {
                last_sync_upload_bytes = 0;
                if (read_memo) {
                    write_read_memo();
                }
                return {&buffer, buffer.Offset(device_addr)};
            }
            fast_verify = true;
            // The drain point, not the tail now: an entry appended after the drain is a race.
            fast_tail = memory_tracker->BackCleanDrainedTo();
            memory_tracker->census_bc = 0;
        }
    }
    const auto t_sync = Common::FaultMonitor::Now();
    const auto t_sample_sync = sample ? sample->Now() : ProbeClock::time_point{};
    const bool traffic = BufferTraffic::Enabled();
    const bool traffic_dirty = traffic && memory_tracker->IsRegionCpuModified(device_addr, size);
    last_sync_upload_bytes = 0;
    SynchronizeBuffer(buffer, device_addr, size, is_written, is_texel_buffer,
                      sample && sample->Active() ? sample : nullptr);
    if (traffic) {
        g_buffer_traffic.Record(device_addr, size, is_written, is_texel_buffer,
                                sample_caller == Common::BufferSampleCaller::Other, traffic_dirty,
                                last_sync_upload_bytes);
    }
    if (fast_verify && (last_sync_upload_bytes != 0 || memory_tracker->census_bc == 2))
        [[unlikely]] {
        // The full path walked (found an unverified page) or uploaded after a clean answer. A
        // dirty-log entry appended after the check's drain point that overlaps the range is a guest
        // write that landed in between (the full path is right); none means the answer was wrong.
        // With SHADPS4_BACK_CLEAN=2 the full path also checks the private bits against the
        // tracker's own (its gap check), so a stale verified bit shows up here too.
        const u64 lo = device_addr >> TRACKER_PAGE_BITS;
        const u64 hi = (device_addr + size + TRACKER_BYTES_PER_PAGE - 1) >> TRACKER_PAGE_BITS;
        bool raced = false;
        u64 head = fast_tail;
        raced |= !DirtyLog::Drain(head, [&](u64 first_page, u64 count) {
            raced |= first_page < hi && lo < first_page + count;
        });
        auto& fs = read_sync_fast_stats;
        if (++fs[raced ? 2 : 3] <= 20 && !raced) {
            LOG_ERROR(Render,
                      "ReadSyncFast verify: clean answer, full path walked (bc={}) and uploaded {} "
                      "bytes at {:#x}+{:#x}",
                      memory_tracker->census_bc, last_sync_upload_bytes, device_addr, size);
        }
    }
    if (memo_entry) {
        // Epoch read before the walk: a clear during it makes the record miss next time.
        *memo_entry = {device_addr, size, memo_epoch};
    }
    if (read_memo) {
        // SHADPS4_READ_BIND_MEMO=2: a hit must have been a no-op.
        if (read_memo_verify && sync_upload_bytes_total != read_memo_bytes_before) {
            if (++read_memo_verify_bad <= 20) {
                LOG_ERROR(Render, "ReadBindMemo verify: hit uploaded {} bytes at {:#x}+{:#x}",
                          sync_upload_bytes_total - read_memo_bytes_before, device_addr, size);
            }
        }
        write_read_memo();
    }
    if (census_cls >= 1) [[unlikely]] {
        RmemoCensusCount(census_cls, buffer, census_same,
                         census_timed ? __rdtsc() - census_c1 : 0, census_timed);
    }
    if (sample) {
        sample->AddSyncNs(sample->SinceNs(t_sample_sync));
    }
    Common::FaultMonitor::CountObtainSync(query_ns, Common::FaultMonitor::SinceNs(t_sync));
    if (is_written) {
        const auto t_range_add = sample ? sample->Now() : ProbeClock::time_point{};
        NoteGpuWrite(device_addr, size);
        AddGpuModified(gpu_modified_ranges, device_addr, size);
        if (sample) {
            sample->AddRangeAddNs(sample->SinceNs(t_range_add));
        }
        Common::FaultMonitor::CountRangeAdd(gpu_modified_ranges.Fragments());
    }
    if (run_entry) {
        // SHADPS4_WRITTEN_RUN_MEMO=2: a hit must have uploaded nothing and added no new interval.
        if (run_verify &&
            (sync_upload_bytes_total != run_bytes_before ||
             gpu_modified_ranges.Fragments() != run_ranges_before)) {
            if (++written_run_stats[7] <= 20) {
                LOG_ERROR(Render,
                          "WrittenRunMemo verify: hit at {:#x}+{:#x} uploaded {} bytes, ranges "
                          "{} -> {}",
                          device_addr, size, sync_upload_bytes_total - run_bytes_before,
                          run_ranges_before, gpu_modified_ranges.Fragments());
            }
        }
        // Values read before the full run: a change during it misses next time.
        *run_entry = run_now;
    }
    return {&buffer, buffer.Offset(device_addr)};
}

std::pair<Buffer*, u32> BufferCache::TryHostMirror(VAddr device_addr, u32 size, bool is_texel) {
    ++hm_calls;
    std::pair<Buffer*, u32> result{};
    u64 backing = 0;
    const bool range_mode = host_mirror_hi > host_mirror_lo;
    const bool overlaps_range =
        range_mode && device_addr < host_mirror_hi && device_addr + size > host_mirror_lo;
    if (size > host_mirror_max) {
        ++hm_toobig;
        hm_max_seen = std::max<u64>(hm_max_seen, size);
        if (overlaps_range && hm_samples < 12) {
            ++hm_samples;
            LOG_INFO(Render_Vulkan, "HostMirror: too big {:#x}+{:#x}", device_addr, size);
        }
    } else if (range_mode && !overlaps_range) {
        // Range mode: any binding that overlaps the range (the per-frame heap) is mirrored.
        ++hm_outside;
    } else if (is_texel && texture_cache.FindImageFromRange(device_addr, size)) {
        // A cached image aliases this formatted buffer; its newest bytes may be in the image.
        ++hm_image;
    } else if (!range_mode && !memory_tracker->IsRegionCpuModified(device_addr, size)) {
        // Clean: the tracked copy is current and in VRAM; keep using it.
        ++hm_clean;
    } else if (IsRegionGpuModified(device_addr, size)) {
        // Newest bytes are in a cache buffer, not in guest memory.
        ++hm_gpu;
        if (is_texel && (hm_gpu & 0x3fff) == 1 && hm_samples < 16) {
            // Which GPU-written ranges keep this formatted binding off the mirror.
            ++hm_samples;
            u64 count = 0, bytes = 0;
            std::string first;
            gpu_modified_ranges.ForEachInRange(device_addr, size, [&](VAddr lo, VAddr hi) {
                ++count;
                bytes += hi - lo;
                if (count <= 4) {
                    first += fmt::format(" {:#x}+{:#x}", lo, hi - lo);
                }
            });
            LOG_INFO(Render_Vulkan, "HostMirror: texel {:#x}+{:#x} blocked by {} GPU ranges, {} KB:{}",
                     device_addr, size, count, bytes / 1024, first);
        }
    } else if (!MirrorBackingOffset(device_addr, size, backing)) {
        ++hm_nobacking;
    } else if ((backing >> HostMirrorChunkBits) != ((backing + size - 1) >> HostMirrorChunkBits)) {
        ++hm_straddle;
    } else if (Buffer* chunk = HostMirrorChunk(backing >> HostMirrorChunkBits)) {
        ++hm_hits;
        hm_bytes += size;
        result = {chunk, static_cast<u32>(backing & ((u64{1} << HostMirrorChunkBits) - 1))};
    }
    if ((hm_calls & 0xfffff) == 0) {
        LOG_INFO(Render_Vulkan,
                 "HostMirror: calls={} hits={} ({:.1f}%) clean={} gpu={} nobacking={} "
                 "straddle={} toobig={} (max {:#x}) outside={} image={} chunks={} "
                 "avg_hit_bytes={:.0f}",
                 hm_calls, hm_hits, 100.0 * double(hm_hits) / double(hm_calls), hm_clean, hm_gpu,
                 hm_nobacking, hm_straddle, hm_toobig, hm_max_seen, hm_outside, hm_image,
                 host_mirror_chunks.size(), hm_hits ? double(hm_bytes) / double(hm_hits) : 0.0);
    }
    return result;
}

bool BufferCache::MirrorBackingOffset(VAddr device_addr, u32 size, u64& backing) {
    // Spans are stable until an unmap (BeginGpuUnmap bumps the generation); four recent ones
    // cover the per-frame heap without taking the memory manager's lock per binding.
    const u64 generation = backing_span_generation.load(std::memory_order_acquire);
    for (auto& span : backing_spans) {
        if (span.generation == generation && device_addr >= span.start &&
            device_addr + size <= span.end) {
            backing = span.backing + (device_addr - span.start);
            return true;
        }
    }
    BackingSpanEntry entry{};
    if (!memory->BackingSpan(device_addr, entry.start, entry.end, entry.backing) ||
        device_addr + size > entry.end) {
        return false;
    }
    entry.generation = generation;
    backing_spans[backing_span_next++ & 3] = entry;
    backing = entry.backing + (device_addr - entry.start);
    return true;
}

Buffer* BufferCache::HostMirrorChunk(u64 index) {
    if (index >= host_mirror_chunks.size()) {
        host_mirror_chunks.resize(index + 1);
    }
    auto& slot = host_mirror_chunks[index];
    if (slot.buffer || slot.failed) {
        return slot.buffer.get();
    }
    slot.failed = true;
    auto& space = memory->GetAddressSpace();
    const u64 chunk_offset = index << HostMirrorChunkBits;
    const u64 chunk_size = std::min<u64>(u64{1} << HostMirrorChunkBits,
                                         space.BackingSizeBytes() - chunk_offset);
    void* pointer = space.BackingBase() + chunk_offset;
    const auto device = static_cast<VkDevice>(instance.GetDevice());
    const auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
    constexpr auto handle_type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkMemoryHostPointerPropertiesEXT props{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    VkResult r = d.vkGetMemoryHostPointerPropertiesEXT(device, handle_type, pointer, &props);
    if (r != VK_SUCCESS || props.memoryTypeBits == 0) {
        LOG_ERROR(Render_Vulkan, "HostMirror: chunk {} pointer properties failed ({}), bits={:#x}",
                  index, static_cast<int>(r), props.memoryTypeBits);
        return nullptr;
    }
    const auto& mem_props = instance.GetMemoryProperties();
    u32 type_index = ~0u;
    for (u32 i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((props.memoryTypeBits & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible)) {
            type_index = i;
            break;
        }
    }
    if (type_index == ~0u) {
        LOG_ERROR(Render_Vulkan, "HostMirror: no host-visible type in {:#x}", props.memoryTypeBits);
        return nullptr;
    }
    const VkExternalMemoryBufferCreateInfo external_ci{
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = handle_type,
    };
    const VkBufferCreateInfo buffer_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &external_ci,
        .size = chunk_size,
        .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkBuffer handle{};
    r = d.vkCreateBuffer(device, &buffer_ci, nullptr, &handle);
    if (r != VK_SUCCESS) {
        LOG_ERROR(Render_Vulkan, "HostMirror: vkCreateBuffer failed ({})", static_cast<int>(r));
        return nullptr;
    }
    const VkImportMemoryHostPointerInfoEXT import_info{
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
        .handleType = handle_type,
        .pHostPointer = pointer,
    };
    const VkMemoryAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &import_info,
        .allocationSize = chunk_size,
        .memoryTypeIndex = type_index,
    };
    VkDeviceMemory device_memory{};
    r = d.vkAllocateMemory(device, &alloc_info, nullptr, &device_memory);
    if (r == VK_SUCCESS) {
        r = d.vkBindBufferMemory(device, handle, device_memory, 0);
    }
    if (r != VK_SUCCESS) {
        LOG_ERROR(Render_Vulkan, "HostMirror: import of chunk {} failed ({})", index,
                  static_cast<int>(r));
        d.vkDestroyBuffer(device, handle, nullptr);
        return nullptr;
    }
    slot.buffer = std::make_unique<Buffer>(instance, scheduler, vk::Buffer{handle}, 0, chunk_size);
    slot.failed = false;
    LOG_INFO(Render_Vulkan, "HostMirror: imported backing chunk {} ({:#x}+{:#x}) as memory type {}",
             index, chunk_offset, chunk_size, type_index);
    return slot.buffer.get();
}

std::pair<Buffer*, u32> BufferCache::ObtainBufferForImage(VAddr gpu_addr, u32 size) {
    const u64 ibc_t0 = g_image_buffer_cost ? __rdtsc() : 0;
    const auto note = [&](const char* path, const Buffer* buf) {
        if (!g_image_buffer_cost) {
            return;
        }
        const u64 cyc = __rdtsc() - ibc_t0;
        if (cyc < 350000) {
            return;
        }
        static thread_local std::unordered_map<VAddr, u32> logged;
        if (logged[gpu_addr]++ < 8) {
            LOG_INFO(Render_Vulkan,
                     "ImageBufferCost: {:#x} size={} path={} total={} sync_walk={} sync_rest={} "
                     "cyc buffer={:#x} buffer_size={}",
                     gpu_addr, size, path, cyc, g_sync_walk_cyc, g_sync_rest_cyc,
                     buf ? buf->CpuAddr() : 0, buf ? buf->SizeBytes() : 0);
        }
    };
    g_sync_walk_cyc = 0;
    g_sync_rest_cyc = 0;
    // Check if any buffer contains the full requested range.
    const BufferId buffer_id = page_table[gpu_addr >> CACHING_PAGEBITS].buffer_id;
    if (buffer_id) {
        if (Buffer& buffer = slot_buffers[buffer_id]; buffer.IsInBounds(gpu_addr, size)) {
            // The image upload reads this range on main (and SynchronizeBuffer may upload into it):
            // an ASC dispatch on the same Buffer must see the access (Opus review 2026-09-28).
            buffer.NoteMainAccess(scheduler.CurrentTick(), true);
            if (g_range_notes) {
                buffer.NoteMainRange(scheduler.CurrentTick(), true, gpu_addr, size, 11, 0);
            }
            if (g_lane_probe && Common::FaultMonitor::t_is_cp_thread &&
                !Vulkan::Scheduler::t_lane_mode) [[unlikely]] {
                LaneProbeRead(gpu_addr, gpu_addr + size, scheduler.CurrentTick(), 10, 16); // q9c
            }
            SynchronizeBuffer(buffer, gpu_addr, size, false, false, nullptr);
            note("cached", &buffer);
            return {&buffer, buffer.Offset(gpu_addr)};
        }
    }
    // If some buffer within was GPU modified create a full buffer to avoid losing GPU data.
    if (IsRegionGpuModified(gpu_addr, size)) {
        auto result = ObtainBuffer(gpu_addr, size, false, false);
        note("gpu_modified", result.first);
        return result;
    }
    SCOPE_EXIT {
        note("staging", nullptr);
    };
    // In all other cases, just do a CPU copy to the staging buffer.
    const auto [data, offset] = staging_buffer.Map(size, 16);
    memory->CopySparseMemory(gpu_addr, data, size);
    staging_buffer.Commit();
    return {&staging_buffer, offset};
}

bool BufferCache::IsRegionRegistered(VAddr addr, size_t size) {
    // Check if we are missing some edge case here
    // Bounded: find() does no range check (page_table covers 2^40 bytes); larger or wider
    // queries take the locked tree (Opus review).
    if (g_buffer_ranges_table && size != 0 && size <= (u64{64} << 20) && addr < (u64{1} << 40) &&
        size <= (u64{1} << 40) - addr) {
        const u64 first = addr >> CACHING_PAGEBITS;
        const u64 last = (addr + size - 1) >> CACHING_PAGEBITS;
        const auto& table = page_table;
        for (u64 page = first; page <= last; ++page) {
            const PageData* entry = table.find_acquire(page);
            // Written by the command processor in ChangeRegister; an empty page is INVALID_INDEX.
            if (entry && std::atomic_ref<u32>(const_cast<u32&>(entry->buffer_id.index))
                                 .load(std::memory_order_acquire) != BufferId::INVALID_INDEX) {
                return true;
            }
        }
        return false;
    }
    if (g_buffer_ranges_lock) {
        std::shared_lock lk{buffer_ranges_mutex};
        return buffer_ranges.Intersects(addr, size);
    }
    return buffer_ranges.Intersects(addr, size);
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionGpuModified(addr, size);
}

bool BufferCache::IsRegionCpuDirtyRelaxed(VAddr addr, u64 size) const {
    return memory_tracker->CpuDirtyRelaxed(addr, size);
}

BufferId BufferCache::FindBuffer(VAddr device_addr, u32 size) {
    ASSERT(device_addr != 0);
    const u64 page = device_addr >> CACHING_PAGEBITS;
    const BufferId buffer_id = page_table[page].buffer_id;
    if (!buffer_id) {
        return CreateBuffer(device_addr, size);
    }
    const Buffer& buffer = slot_buffers[buffer_id];
    if (buffer.IsInBounds(device_addr, size)) {
        return buffer_id;
    }
    return CreateBuffer(device_addr, size);
}

BufferId BufferCache::FindBufferHinted(VAddr device_addr, u32 size) {
    const u64 gen = StructGeneration();
    if (gen == buffer_hint_gen) {
        const VAddr end = device_addr + size;
        for (const BufferHint& hint : buffer_hints) {
            // An empty hint (hi == 0) never contains a range: end > 0.
            if (device_addr >= hint.lo && end <= hint.hi) {
                ++lean_stats[0];
                if (g_lean_bind == 2) [[unlikely]] {
                    const BufferId found = FindBuffer(device_addr, size);
                    if (found != hint.id) {
                        if (++lean_stats[2] <= 20) {
                            LOG_ERROR(Render, "LeanBind verify: hint {} vs FindBuffer {} at {:#x}+{:#x}",
                                      hint.id.index, found.index, device_addr, size);
                        }
                        buffer_hint_gen = ~0ULL;
                        return found;
                    }
                }
                return hint.id;
            }
        }
    }
    ++lean_stats[1];
    const BufferId found = FindBuffer(device_addr, size);
    // FindBuffer may have created or joined buffers (ChangeRegister bumped the generation). Page
    // tables change only on this thread, so the bounds read now hold until the next bump.
    const u64 gen_after = StructGeneration();
    if (gen_after != buffer_hint_gen) {
        buffer_hints = {};
        buffer_hint_gen = gen_after;
    }
    const Buffer& buffer = slot_buffers[found];
    buffer_hints[buffer_hint_next++ & 3] = {buffer.CpuAddr(), buffer.CpuAddr() + buffer.SizeBytes(),
                                            found};
    return found;
}

bool BufferCache::LeanReadOk() {
    if (lean_read_state < 0) [[unlikely]] {
        // Every per-call option ObtainBuffer tests on a read binding before or around the memo
        // and the clean check must be off; then its answer is the memo probe plus the check.
        const bool ok = g_lean_bind != 0 && ::Common::Lean() && StreamCopyMaxSize() == 0 &&
                        !host_mirror && !g_range_notes && !g_asc_deps && !g_rmemo_census &&
                        g_pb_census == 0 && !Offload::AscSpanStatsEnabled() &&
                        !BufferTraffic::Enabled() &&
                        (g_read_bind_memo == 1 || g_read_bind_memo == 3) && g_read_sync_fast == 1;
        lean_read_state = ok ? 1 : 0;
        LOG_INFO(Render, "LeanBind: mode={} lean read path {}", g_lean_bind, ok ? "on" : "off");
    }
    return lean_read_state == 1;
}

bool BufferCache::ReadRangeClean(VAddr addr, u64 size) {
    return memory_tracker->ReadSyncClean(addr, size);
}

std::pair<Buffer*, u32> BufferCache::ObtainReadLean(VAddr device_addr, u32 size,
                                                    BufferId buffer_id) {
    AmdGpu::MtbCensus::NoteUse(device_addr, size);
    if (IsBufferInvalid(buffer_id)) [[unlikely]] {
        ++lean_stats[4];
        return {};
    }
    Buffer& buffer = slot_buffers[buffer_id];
    if (g_lane_on && Vulkan::Scheduler::t_lane_mode) [[unlikely]] {
        // q11c: as ObtainBuffer's lane block for a read (rule A only).
        LbScope lb_scope{6};
        g_lc_done = scheduler.GetMasterSemaphore()->KnownGpuTick();
        const u64 t = LaneMainProducer(device_addr, device_addr + size);
        if (g_lane_who) [[unlikely]] {
            LaneWhoNote(t, scheduler.CurrentTick(), false, 0, false);
        }
        scheduler.LaneNeedMain(t, g_lc_src);
    } else if (g_lane_probe && Common::FaultMonitor::t_is_cp_thread) [[unlikely]] {
        LaneProbeRead(device_addr, device_addr + size, scheduler.CurrentTick(), 10, 17); // q9 census
    }
    if (g_lane_census && Common::FaultMonitor::t_is_cp_thread && !t_force_binding_sync)
        [[unlikely]] {
        g_lc[8].fetch_add(1, std::memory_order_relaxed);
        if (LaneFind(g_lc_early, device_addr, device_addr + size)) {
            g_lc[11].fetch_add(1, std::memory_order_relaxed);
        }
    }
    // As ObtainBuffer: NoteMainAccess, then the memo probe with the epoch and guard read first.
    buffer.NoteMainAccess(scheduler.CurrentTick(), false);
    if (CoherentSkip(false, false)) {
        // SHADPS4_NATIVE_COHERENT: no memo probe, no clean check (see ObtainBuffer).
        ++coherent_stats.skips;
        if (g_native_coherent == 2) [[unlikely]] {
            CoherentVerify(buffer, device_addr, size);
        }
        return {&buffer, buffer.Offset(device_addr)};
    }
    const u64 epoch = g_dirty_epoch.load(std::memory_order_acquire);
    const bool serial_guard = g_read_bind_memo >= 3;
    const u64 gen = serial_guard ? buffer.memo_serial : StructGeneration();
    const u64 h = (device_addr * 0x9E3779B97F4A7C15ULL) ^ (u64(size) * 0xC2B2AE3D27D4EB4FULL) ^
                  u64(buffer_id.index);
    ReadMemoEntry& entry = read_memo_table[(h ^ (h >> 31)) & (read_memo_table.size() - 1)];
    std::pair<Buffer*, u32> result{};
    bool memo_hit = false; // q11d
    if (entry.addr == device_addr && entry.size == size && entry.buffer_index == buffer_id.index &&
        entry.epoch == epoch && entry.gen == gen && (!serial_guard || gen != 0)) {
        result = {&buffer, buffer.Offset(device_addr)};
        memo_hit = true;
    } else if (memory_tracker->ReadSyncClean(device_addr, size)) {
        // READ_SYNC_FAST=1's clean answer: record nothing, write the memo as the full path does.
        last_sync_upload_bytes = 0;
        entry = {device_addr, size, epoch, gen, buffer_id.index};
        result = {&buffer, buffer.Offset(device_addr)};
    } else {
        ++lean_stats[4];
        return {};
    }
    if ((++lean_stats[3] & 0x1ffffff) == 0) [[unlikely]] {
        LogLeanStats();
    }
    if (g_lean_bind == 2) [[unlikely]] {
        const u64 uploads = sync_upload_bytes_total;
        const auto full = ObtainBuffer(device_addr, size, false, false, buffer_id,
                                       Common::BufferSampleCaller::Uniform);
        static u64 v_n = 0, v_bad = 0, v_bad_still = 0; // q11d (back thread)
        ++v_n;
        if (full != result || sync_upload_bytes_total != uploads) {
            const bool moved = g_dirty_epoch.load(std::memory_order_acquire) != epoch;
            ++v_bad;
            v_bad_still += moved ? 0 : 1;
            if (++lean_stats[5] <= 40) {
                LOG_ERROR(Render,
                          "LeanBind verify: lean read {:#x}+{:#x} -> {}+{:#x}, full -> {}+{:#x}, "
                          "uploaded {} | q11d: {} epoch {} | lane {} early {} coherent {}",
                          device_addr, size, fmt::ptr(result.first), result.second,
                          fmt::ptr(full.first), full.second, sync_upload_bytes_total - uploads,
                          memo_hit ? "memo hit" : "clean check",
                          moved ? "MOVED (a CPU write in between: race)" : "UNCHANGED (a miss)",
                          Vulkan::Scheduler::t_lane_mode, t_force_binding_sync, coherent_live);
            }
        }
        if ((v_n & 0xfffff) == 0) {
            LOG_INFO(Render, "LeanBind verify (q11d): compares {} mismatches {} with the epoch unchanged {}",
                     v_n, v_bad, v_bad_still);
        }
        return full;
    }
    return result;
}

void BufferCache::LogLeanStats() {
    const auto& s = lean_stats;
    LOG_INFO(Render,
             "LeanBind: mode={} hint hits={} misses={} ({:.2f}% hit) verify_bad={} | lean reads={} "
             "fallbacks={} verify_bad={}",
             g_lean_bind, s[0], s[1], 100.0 * double(s[0]) / double(std::max<u64>(s[0] + s[1], 1)),
             s[2], s[3], s[4], s[5]);
}

// 1.1.1 (GitHub #30): whether `size` more bytes fit the driver's device-local budget with a 256 MB
// margin (BIG_GROW's mesh buffer); nullopt when the driver gives no budget (VK_EXT_memory_budget).
// `freed`: the joined buffers' bytes, released right after the copy - credited up to a quarter of
// the new size (i30: 524 MB of pieces at the first join; a whole-buffer regrow gets no free pass,
// its old copy would stay over budget for the whole copy).
static std::optional<bool> MeshFitsVideo(const Vulkan::Instance& instance, u64 size, u64 freed) {
    const auto free = Vulkan::DriverFreeDeviceLocal(instance.GetAllocator());
    if (!free) {
        return std::nullopt;
    }
    const u64 credit = std::min(freed, size / 4);
    return size + (256ull << 20) <= *free + credit;
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(VAddr device_addr, u32 wanted_size) {
    static constexpr int STREAM_LEAP_THRESHOLD = 16;
    boost::container::small_vector<BufferId, 16> overlap_ids;
    VAddr begin = device_addr;
    VAddr end = device_addr + wanted_size;
    int stream_score = 0;
    bool has_stream_leap = false;
    bool mesh = false; // a BIG_GROW mesh-sized join (>= 1 GB)
    const auto expand_begin = [&](VAddr add_value) {
        static constexpr VAddr min_page = CACHING_PAGESIZE + DEVICE_PAGESIZE;
        if (add_value > begin - min_page) {
            begin = min_page;
            device_addr = DEVICE_PAGESIZE;
            return;
        }
        begin -= add_value;
        device_addr = begin - CACHING_PAGESIZE;
    };
    const auto expand_end = [&](VAddr add_value) {
        static constexpr VAddr max_page = 1ULL << MemoryTracker::MAX_CPU_PAGE_BITS;
        if (add_value > max_page - end) {
            end = max_page;
            return;
        }
        end += add_value;
    };
    if (begin == 0) {
        return OverlapResult{
            .ids = std::move(overlap_ids),
            .begin = begin,
            .end = end,
            .has_stream_leap = has_stream_leap,
        };
    }
    for (; device_addr >> CACHING_PAGEBITS < Common::DivCeil(end, CACHING_PAGESIZE);
         device_addr += CACHING_PAGESIZE) {
        const BufferId overlap_id = page_table[device_addr >> CACHING_PAGEBITS].buffer_id;
        if (!overlap_id) {
            continue;
        }
        Buffer& overlap = slot_buffers[overlap_id];
        if (overlap.is_picked) {
            continue;
        }
        overlap_ids.push_back(overlap_id);
        overlap.is_picked = true;
        const VAddr overlap_device_addr = overlap.CpuAddr();
        const bool expands_left = overlap_device_addr < begin;
        if (expands_left) {
            begin = overlap_device_addr;
        }
        const VAddr overlap_end = overlap_device_addr + overlap.SizeBytes();
        const bool expands_right = overlap_end > end;
        if (overlap_end > end) {
            end = overlap_end;
        }
        stream_score += overlap.StreamScore();
        if (stream_score > STREAM_LEAP_THRESHOLD && !has_stream_leap) {
            // When this memory region has been joined a bunch of times, we assume it's being used
            // as a stream buffer. Increase the size to skip constantly recreating buffers.
            has_stream_leap = true;
            if (expands_right) {
                expand_end(CACHING_PAGESIZE * 128);
            }
            if (expands_left) {
                expand_begin(CACHING_PAGESIZE * 128);
            }
        }
    }
    // The one mesh buffer BIG_GROW creates at the first big join; its size is chosen once from
    // the video memory size (0 = BIG_GROW off).
    static const u64 big_grow_bytes = [this]() -> u64 {
        const char* e = std::getenv("SHADPS4_BIG_GROW");
        if (!e || e[0] != '1') {
            return 0;
        }
        // Release check (2026-10-15, study-release-readiness sec. 1.3): the 3.75 GB buffer must fit.
        // BIG_GROW stays on only when the largest DEVICE_LOCAL heap is >= 11 GiB (this PC, 16 GB:
        // whole-GPU peak 8.5-9.2 GB with it, desktop included; so 12 GB cards pass, 8 / 10 GB cards
        // do not) and the device allows a 3.75 GB allocation (maxMemoryAllocationSize). Otherwise the
        // old growth path (stream-leap expansion only: the mesh memory regrows as blocks stream in).
        // SHADPS4_BIG_GROW_MIN_VRAM_MB overrides the 11264 MiB threshold (a huge value forces the
        // fallback for testing).
        //
        // v1.0.2 (2026-10-19; an 8 GB RTX 4060 Laptop ended with ErrorOutOfDeviceMemory loading a
        // save): below the threshold the old growth path re-created the ~2.5 GB mesh buffer at
        // each growth, the old one alive until the GPU finished the copy (2491 -> 2492 -> 2496 MB
        // within 17 submissions, ~5 GB of mesh buffers at once against VMA's 6365 MiB budget).
        // Smaller cards now get one buffer sized once too, by tier: >= the threshold 3.75 GB
        // (unchanged), >= 7 GiB (8 GB cards) 3 GB, below 2.75 GB. The game's mesh memory reached
        // 2491-2515 MB in every measured load (r14 s1041 / s1042, the player's log), so 2.75 GB
        // still covers it; a growth past the size takes the old copy path (the allocation then
        // may go over budget, vk_memory_budget.h). SHADPS4_BIG_GROW_TIERS=0 restores the v1.0.1
        // rule (old growth path below the threshold); SHADPS4_SIM_VRAM_MB stands in for the heap.
        u64 heap = 0;
        const auto& mp = instance.GetMemoryProperties();
        for (u32 i = 0; i < mp.memoryHeapCount; ++i) {
            if (mp.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
                heap = std::max<u64>(heap, mp.memoryHeaps[i].size);
            }
        }
        const bool simulated = instance.SimVramBytes() != 0;
        if (simulated) {
            heap = std::min<u64>(heap, instance.SimVramBytes());
        }
        u64 min_mb = 11264;
        if (const char* m = std::getenv("SHADPS4_BIG_GROW_MIN_VRAM_MB")) {
            min_mb = std::strtoull(m, nullptr, 10);
        }
        const char* t = std::getenv("SHADPS4_BIG_GROW_TIERS");
        const bool tiers = !(t && t[0] == '0');
        // 1.1 (PERF-PLAN 11): with < 24 GB of RAM the 3.75 GB tier is skipped (3 GB / 2.75 GB by VRAM). Every byte of video
        // memory also commits a byte of system memory (s1436: a 2 GB device-local buffer added
        // 2 GB of private RAM), and the game's mesh memory measured 2491-2515 MB: the 3.75 GB
        // buffer's spare memory costs a 16 GB PC RAM it needs (s1438, 16 GB simulated: VRAM 6.89 ->
        // 6.12 GB, private RAM 14.3 -> 13.5 GB).
        const u64 ram = Common::PhysicalRamBytes();
        const bool low_ram = ram != 0 && ram < (24ull << 30);
        u64 size = 0;
        if (heap >= (min_mb << 20) && !(tiers && low_ram)) {
            size = 0xF0000000ull; // 3.75 GB
        } else if (tiers && heap >= (7168ull << 20)) {
            size = 0xC0000000ull; // 3 GB
        } else if (tiers) {
            size = 0xB0000000ull; // 2.75 GB
        }
        u64 max_alloc = instance.MaxMemoryAllocationSize();
        // SHADPS4_SIM_MAX_ALLOC_MB (test): a smaller reported limit, e.g. 2048 like the RX 6600.
        if (const char* m = std::getenv("SHADPS4_SIM_MAX_ALLOC_MB")) {
            max_alloc = std::min<u64>(max_alloc, std::strtoull(m, nullptr, 10) << 20);
        }
        if (size > max_alloc) {
            // 1.1 (GitHub #15, RX 6600 8 GB, reported max allocation 2048 MiB): turning BIG_GROW
            // off below 2.75 GB did not avoid a big allocation - the old growth path made the
            // mesh buffer 2471 MB on that card anyway, then a 2473 MB copy beside it at the next
            // growth (6212 / 6336 MB -> out of video memory). The mesh memory needs one ~2.5 GB
            // buffer on either path, so a reported limit below 2.75 GB keeps the card's tier, at
            // most 3 GB (the city save's mesh memory reached 2790 MB in s1558: 2.75 GB is too
            // tight). A limit of 2.75 GB or more is respected as before.
            size = !tiers                        ? 0
                   : max_alloc >= 0xB0000000ull ? 0xB0000000ull
                                                 : std::min<u64>(size, 0xC0000000ull);
        }
        LOG_INFO(Render_Vulkan,
                 "BIG_GROW: largest device-local heap {} MiB{} (3.75 GB needs >= {} MiB and >= 24 GB "
                 "RAM, RAM {} MB, tiers {}), max allocation {} MiB -> {} MB{}",
                 heap >> 20, simulated ? " (SIMULATED, SHADPS4_SIM_VRAM_MB)" : "", min_mb, ram >> 20,
                 tiers ? "on" : "off", max_alloc >> 20, size >> 20,
                 size != 0 ? " - one mesh buffer sized once"
                           : " - OFF: fallback to the old growth path");
        return size;
    }();
    const bool big_grow = big_grow_bytes != 0;
    // r15e: only joins of >= 1 GB (the mesh memory) - growing the 542 MB heap buffer over the
    // particle arena put CPU-written pages under tracking: write faults 35 k -> 57-60 k/s (s1051-s1056).
    if (big_grow && !overlap_ids.empty() && end - begin >= (1ull << 30)) {
        // r15 SHADPS4_BIG_GROW=1 (2026-10-14): a join that yields a buffer of >= 256 MB is sized
        // ahead by max(768 MB, 50%) (clamped to mapped guest memory, under 4 GB). The game's mesh
        // memory (~2-2.5 GB at 0x271258000) grows as blocks stream in; each growth re-created it
        // and copied it whole on the GPU (r14 BigJoin: 2.0-2.5 GB copies, an 85 ms frame on the
        // rooftop at the last one - the play hitches). Buffers inside the extension are joined
        // like any overlap, so no page keeps two buffers.
        const u64 cur = end - begin;
        // 1.1.1 (GitHub #30): every mesh-sized join prefers video memory (CreateBuffer checks the
        // driver's budget for its real size) - not only the extended one: the city save's first
        // join was already 2766 MB, above a 2560 MB step (s1584), and later regrows are joins too.
        mesh = true;
        // r15d: straight to the cap (3.75 GB) at the first big join - normally the load - so the
        // region never grows again in the scene (r15b: an in-scene growth's coherent upload stalled ~1 s).
        // 1.1.1 (GitHub #5: an RTX 5050 8 GB on 16 GB of RAM - the 3 GB buffer fitted neither in
        // video nor in system memory at the first big join, on 1.0.2 and 1.1, and the assert closed
        // the game): the first big join tries the tier first (allocated and freed again), then the
        // smaller tiers down to 2.75 GB, then no extension (the old growth path). The pick holds
        // for the session. SHADPS4_BIG_GROW_PROBE_FAIL_ABOVE_MB (test): a try above it asks for
        // 64 GB instead, which the driver really refuses.
        u64 pieces = 0; // the joined buffers' bytes (MeshFitsVideo's credit)
        for (const BufferId id : overlap_ids) {
            pieces += slot_buffers[id].SizeBytes();
        }
        static const u64 want = [&]() -> u64 {
            u64 fail_above = ~0ull;
            if (const char* f = std::getenv("SHADPS4_BIG_GROW_PROBE_FAIL_ABOVE_MB")) {
                fail_above = std::strtoull(f, nullptr, 10) << 20;
            }
            const VmaAllocator allocator = instance.GetAllocator();
            const vk::BufferUsageFlags flags = AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress;
            // The tier, then the smaller tiers (3 GB, 2.75 GB) as steps down; below 7 GiB of video
            // memory (the 2.75 GB tier) also 2.5 GB: the city's mesh memory reached 2790 MB (s1558),
            // so that one regrows once in the scene, still better than system RAM (GitHub #30).
            boost::container::small_vector<u64, 4> sizes{big_grow_bytes};
            for (const u64 step : {u64{0xC0000000ull}, u64{0xB0000000ull}}) {
                if (step < big_grow_bytes) {
                    sizes.push_back(step);
                }
            }
            if (big_grow_bytes <= 0xB0000000ull) {
                sizes.push_back(0xA0000000ull);
            }
            const auto& mp = instance.GetMemoryProperties();
            for (const u64 size : sizes) {
                const u64 ask = size > fail_above ? (64ull << 30) : size;
                // 1.1.1 (GitHub #30, RX 5600 XT 6 GB): the probe used to count any heap as a fit -
                // VMA's budget is its own 80%-of-heap estimate (4697 of 5872 MiB), so the 2816 MB
                // buffer went to system RAM (heap 1) although the card had room (2370 MB in use).
                // Now: video memory only, sized against the driver's budget when it gives one.
                // Without a driver budget: the old probe, but only a device-local heap counts.
                const std::optional<bool> in_budget = MeshFitsVideo(instance, ask, pieces);
                u32 heap = 0;
                bool fits = false;
                bool system_heap = false;
                if (in_budget.value_or(true)) {
                    fits = ProbeBufferAllocation(instance, MemoryUsage::DeviceLocal, flags, ask, heap,
                                                 /*require_video=*/in_budget.has_value());
                    system_heap =
                        fits && !(mp.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal);
                    fits = fits && !system_heap;
                }
                if (fits) {
                    LOG_INFO(Render_Vulkan, "BIG_GROW: the {} MB mesh buffer fits (heap {}){}; {}",
                             size >> 20, heap, size < big_grow_bytes ? " - stepped down" : "",
                             Vulkan::DescribeDriverBudgets(allocator));
                    return size;
                }
                LOG_WARNING(Render_Vulkan,
                            "BIG_GROW: a {} MB mesh buffer does not fit in video memory{}{}; {}; {}; {}",
                            ask >> 20,
                            in_budget == false ? " (over the driver's budget with a 256 MB margin)"
                            : system_heap      ? " (it would land in system memory)"
                                               : "",
                            ask != size ? " (test: SHADPS4_BIG_GROW_PROBE_FAIL_ABOVE_MB)" : "",
                            Vulkan::DescribeDeviceLocalBudgets(allocator),
                            Vulkan::DescribeDriverBudgets(allocator),
                            Common::DescribeSystemMemory());
            }
            LOG_WARNING(Render_Vulkan, "BIG_GROW: no mesh buffer size fits - the old growth path "
                                       "(the mesh memory regrows as blocks stream in)");
            return 0;
        }();
        const u64 mapped = memory->TryClampRangeSize(begin, want);
        VAddr grow_end = Common::AlignDown(begin + mapped, CACHING_PAGESIZE);
        if (mapped > cur && grow_end > end) {
            // r15b: the extension stops at the next existing buffer (s1043: the 542 MB heap buffer
            // grew into the mesh buffer and merged with it - a 3 GB copy and one 3.6 GB buffer
            // holding the CPU-written heap too, rooftop -2.7%).
            // r15c: small buffers (< 64 MB) inside the extension are absorbed (joined - cheap
            // copies; s1047 / s1050: stopping at them pushed the growth into the scene, where the
            // coherent upload of the extension stalled 1.0-1.2 s); a big one ends the extension.
            boost::container::small_vector<BufferId, 16> extra;
            for (VAddr a = Common::AlignUp(end, CACHING_PAGESIZE); a < grow_end;
                 a += CACHING_PAGESIZE) {
                const BufferId id = page_table[a >> CACHING_PAGEBITS].buffer_id;
                if (!id || slot_buffers[id].is_picked) {
                    continue;
                }
                Buffer& o = slot_buffers[id];
                if (o.SizeBytes() >= (64ull << 20) || o.CpuAddr() < end) {
                    grow_end = Common::AlignDown(o.CpuAddr(), CACHING_PAGESIZE);
                    break;
                }
                o.is_picked = true;
                extra.push_back(id);
                grow_end = std::max<VAddr>(grow_end, o.CpuAddr() + o.SizeBytes());
            }
            if (grow_end > end && grow_end - begin < 0xFFFF0000ull) {
                overlap_ids.insert(overlap_ids.end(), extra.begin(), extra.end());
                end = grow_end;
                ++big_grow_count;
            } else {
                for (const BufferId id : extra) {
                    slot_buffers[id].is_picked = false;
                }
            }
        }
    }
    return OverlapResult{
        .ids = std::move(overlap_ids),
        .begin = begin,
        .end = end,
        .has_stream_leap = has_stream_leap,
        .mesh = mesh,
    };
}

void BufferCache::JoinOverlap(BufferId new_buffer_id, BufferId overlap_id,
                              bool accumulate_stream_score) {
    WriterTagScope census_tag{*this, 6};
    if (g_lane_on) [[unlikely]] {
        scheduler.LaneFallbackMain(); // the join's copies read what lane segments may have written
    }
    Buffer& new_buffer = slot_buffers[new_buffer_id];
    Buffer& overlap = slot_buffers[overlap_id];
    NoteBatchWrite(overlap.CpuAddr(), overlap.SizeBytes());
    new_buffer.NoteMainAccess(scheduler.CurrentTick(), true);
    overlap.NoteMainAccess(scheduler.CurrentTick(), false);
    if (g_range_notes) {
        new_buffer.NoteMainWhole(scheduler.CurrentTick(), 6);
        overlap.NoteMainWhole(scheduler.CurrentTick(), 6);
    }
    if (accumulate_stream_score) {
        new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
    }
    const size_t dst_base_offset = overlap.CpuAddr() - new_buffer.CpuAddr();
    const vk::BufferCopy copy = {
        .srcOffset = 0,
        .dstOffset = dst_base_offset,
        .size = overlap.SizeBytes(),
    };
    scheduler.EndRendering();
    auto command_access = scheduler.AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();

    boost::container::static_vector<vk::BufferMemoryBarrier2, 2> pre_barriers{};
    if (auto src_barrier = overlap.GetBarrier(vk::AccessFlagBits2::eTransferRead,
                                              vk::PipelineStageFlagBits2::eTransfer)) {
        pre_barriers.push_back(*src_barrier);
    }
    if (auto dst_barrier =
            new_buffer.GetBarrier(vk::AccessFlagBits2::eTransferWrite,
                                  vk::PipelineStageFlagBits2::eTransfer, dst_base_offset)) {
        pre_barriers.push_back(*dst_barrier);
    }
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = static_cast<u32>(pre_barriers.size()),
        .pBufferMemoryBarriers = pre_barriers.data(),
    });

    cmdbuf.copyBuffer(overlap.Handle(), new_buffer.Handle(), copy);

    boost::container::static_vector<vk::BufferMemoryBarrier2, 2> post_barriers{};
    if (auto src_barrier =
            overlap.GetBarrier(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                               vk::PipelineStageFlagBits2::eAllCommands)) {
        post_barriers.push_back(*src_barrier);
    }
    if (auto dst_barrier = new_buffer.GetBarrier(
            vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            vk::PipelineStageFlagBits2::eAllCommands, dst_base_offset)) {
        post_barriers.push_back(*dst_barrier);
    }
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = static_cast<u32>(post_barriers.size()),
        .pBufferMemoryBarriers = post_barriers.data(),
    });
    DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(VAddr device_addr, u32 wanted_size) {
    AmdGpu::MtbCensus::Note(AmdGpu::MtbCensus::BufferCreate);
    const VAddr device_addr_end = Common::AlignUp(device_addr + wanted_size, CACHING_PAGESIZE);
    device_addr = Common::AlignDown(device_addr, CACHING_PAGESIZE);
    wanted_size = static_cast<u32>(device_addr_end - device_addr);
    const OverlapResult overlap = ResolveOverlaps(device_addr, wanted_size);
    const u32 size = static_cast<u32>(overlap.end - overlap.begin);
    // r14 (2026-10-14, always on, rare): every buffer of >= 256 MB created by a join - the game's
    // ~2 GB mesh memory is one buffer-cache buffer re-created (and copied whole on the GPU) each
    // time a new block's range touches it (study, s1021 / s1023); suspect for the city-streaming
    // hitches. Old sizes, copy bytes, CPU ms of the create + joins.
    u64 big_copy = 0, big_old = 0;
    const bool big = size >= (256u << 20) && !overlap.ids.empty();
    const auto big_t0 = big ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (big) {
        for (const BufferId overlap_id : overlap.ids) {
            const u64 s = slot_buffers[overlap_id].SizeBytes();
            big_copy += s;
            big_old = std::max(big_old, s);
        }
    }
    // 1.1.1 (GitHub #30): BIG_GROW's mesh buffer goes to video memory whenever the driver's budget
    // has room for it now (the joined buffers still alive), not where VMA's estimate sends it.
    std::optional<RequireVideoMemory> video;
    if (overlap.mesh && MeshFitsVideo(instance, size, big_copy).value_or(false)) {
        video.emplace();
    }
    const BufferId new_buffer_id =
        slot_buffers.insert(instance, scheduler, MemoryUsage::DeviceLocal, overlap.begin,
                            AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, size);
    video.reset();
    auto& new_buffer = slot_buffers[new_buffer_id];
    new_buffer.memo_serial = ++buffer_memo_serial;
    for (const BufferId overlap_id : overlap.ids) {
        JoinOverlap(new_buffer_id, overlap_id, !overlap.has_stream_leap);
    }
    if (big) {
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - big_t0)
                              .count();
        LOG_INFO(Render_Vulkan,
                 "BigJoin: new buffer [{:#x}, +{:#x}) ({} MB) from {} joined (largest {} MB, copy {} "
                 "MB on the GPU), wanted [{:#x}, +{:#x}), stream leap {}, CPU {:.2f} ms, tick {}, "
                 "grown ahead so far {}; VMA {}",
                 overlap.begin, size, size >> 20, overlap.ids.size(), big_old >> 20,
                 big_copy >> 20, device_addr, wanted_size, overlap.has_stream_leap, ms,
                 scheduler.CurrentTick(), big_grow_count,
                 Vulkan::DescribeDeviceLocalBudgets(instance.GetAllocator()));
    }
    Register(new_buffer_id);
    if (g_native_coherent != 0 && coherent_live &&
        (Common::FaultMonitor::t_is_cp_thread || Common::FaultMonitor::t_acting_cp)) {
        // SHADPS4_NATIVE_COHERENT: reads of this buffer skip validity from now on, so every page
        // the joined buffers did not carry is uploaded now (the tracker starts them dirty), not
        // at a binding.
        ++coherent_stats.created;
        SynchronizeBuffer(new_buffer, new_buffer.CpuAddr(), size, false, false, nullptr);
    }
    return new_buffer_id;
}

void BufferCache::ProcessFaultBuffer() {
    fault_manager.ProcessFaultBuffer();
}

void BufferCache::Register(BufferId buffer_id) {
    ChangeRegister<true>(buffer_id);
}

void BufferCache::Unregister(BufferId buffer_id) {
    ChangeRegister<false>(buffer_id);
}

void BufferCache::CheckCpThread(int site) const {
    if (!g_thread_assert || !liverpool || !liverpool->GpuThreadStarted() ||
        liverpool->IsGpuThread()) {
        return;
    }
    static std::atomic<u32> reported{0};
    static std::atomic<u64> violations[2]{};
    const u64 n = violations[site].fetch_add(1, std::memory_order_relaxed) + 1;
    if ((reported.fetch_or(1u << site, std::memory_order_relaxed) & (1u << site)) == 0) {
        LOG_CRITICAL(Render, "ThreadAssert: buffer_ranges {} off the command processor, thread '{}'",
                     site == 0 ? "edited (ChangeRegister)" : "walked (ForEachBufferInRange)",
                     Common::GetCurrentThreadName());
    } else if ((n & (n - 1)) == 0) {
        LOG_WARNING(Render, "ThreadAssert: site {} violations={}", site, n);
    }
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId buffer_id) {
    CheckCpThread(0);
    // SHADPS4_BIND_MEMO_STATS: every CreateBuffer, JoinOverlap and DeleteBuffer passes here.
    BumpStructGeneration();
    Buffer& buffer = slot_buffers[buffer_id];
    const auto size = buffer.SizeBytes();
    const VAddr device_addr_begin = buffer.CpuAddr();
    const VAddr device_addr_end = device_addr_begin + size;
    const u64 page_begin = device_addr_begin / CACHING_PAGESIZE;
    const u64 page_end = Common::DivCeil(device_addr_end, CACHING_PAGESIZE);
    const u64 size_pages = page_end - page_begin;
    for (u64 page = page_begin; page != page_end; ++page) {
        // Atomic: IsRegionRegistered (SHADPS4_BUFFER_RANGES_LOCK=2) reads it on guest threads.
        std::atomic_ref<u32>(page_table[page].buffer_id.index)
            .store(insert ? buffer_id.index : BufferId::INVALID_INDEX, std::memory_order_relaxed);
    }
    if constexpr (insert) {
        total_used_memory += Common::AlignUp(size, CACHING_PAGESIZE);
        buffer.SetLRUId(lru_cache.Insert(buffer_id, gc_tick));
        boost::container::small_vector<vk::DeviceAddress, 128> bda_addrs;
        bda_addrs.reserve(size_pages);
        for (u64 i = 0; i < size_pages; ++i) {
            vk::DeviceAddress addr = buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS);
            bda_addrs.push_back(addr);
        }
        // SHADPS4_PT_BITS: entries above the table are not written (that range reads as a miss).
        const u64 table_pages = BdaNumPages();
        const u64 n = page_begin < table_pages
                          ? std::min<u64>(bda_addrs.size(), table_pages - page_begin)
                          : 0;
        if (n != 0) {
            WriteDataBuffer(bda_pagetable_buffer, page_begin * sizeof(vk::DeviceAddress),
                            bda_addrs.data(), n * sizeof(vk::DeviceAddress));
        }
        if (n < bda_addrs.size()) { // wholly or partly above the table (review 10-20, item 3)
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true)) {
                LOG_WARNING(Render_Vulkan, "GPU page table: a buffer at {:#x} is above the table "
                                           "(SHADPS4_PT_BITS) - GPU pointer reads there see a miss",
                            device_addr_begin);
            }
        }
        if (g_buffer_ranges_lock) {
            std::unique_lock lk{buffer_ranges_mutex};
            buffer_ranges.Add(buffer.CpuAddr(), buffer.SizeBytes(), buffer_id);
        } else {
            buffer_ranges.Add(buffer.CpuAddr(), buffer.SizeBytes(), buffer_id);
        }
    } else {
        total_used_memory -= Common::AlignUp(size, CACHING_PAGESIZE);
        lru_cache.Free(buffer.LRUId());
        if (const u64 table_pages = BdaNumPages(); page_begin < table_pages) {
            const u64 offset =
                bda_pagetable_buffer.Offset(page_begin * sizeof(vk::DeviceAddress));
            bda_pagetable_buffer.Fill(
                offset, std::min<u64>(size_pages, table_pages - page_begin) * sizeof(vk::DeviceAddress),
                0);
        }
        if (g_buffer_ranges_lock) {
            std::unique_lock lk{buffer_ranges_mutex};
            buffer_ranges.Subtract(buffer.CpuAddr(), buffer.SizeBytes());
        } else {
            buffer_ranges.Subtract(buffer.CpuAddr(), buffer.SizeBytes());
        }
    }
}

void BufferCache::UploadAhead(Buffer& buffer, VAddr device_addr, u32 size) {
    constexpr u64 page = TRACKER_BYTES_PER_PAGE;
    const VAddr lo = Common::AlignUp(device_addr + size, page);
    const VAddr hi = std::min<VAddr>(lo + u64{g_upload_ahead_pages} * page,
                                     buffer.CpuAddr() + buffer.SizeBytes());
    if (hi <= lo) {
        return;
    }
    t_upload_ahead_active = true;
    const u64 saved_last = last_sync_upload_bytes;
    // A plain read sync of the range: uploads its CPU-dirty pages, re-protects them and records
    // the copy right after the triggering one (same transfer group under TRANSFER_COALESCE).
    SynchronizeBuffer(buffer, lo, static_cast<u32>(hi - lo), false, false, nullptr);
    const u64 bytes = last_sync_upload_bytes;
    last_sync_upload_bytes = saved_last;
    t_upload_ahead_active = false;
    g_upload_ahead_bytes.fetch_add(bytes, std::memory_order_relaxed);
    const u64 n = g_upload_ahead_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    if ((n & 0xffff) == 0) {
        LOG_INFO(Render_Vulkan, "UploadAheadRun: calls={} bytes={}", n,
                 g_upload_ahead_bytes.load(std::memory_order_relaxed));
    }
}

void BufferCache::NoteTransferBegin() {
    const u32 stream = scheduler.ActiveStream();
    ++g_uas.transfers;
    if (stream == g_uas.last_stream && scheduler.RecordSeqOf(stream) == g_uas.last_seq) {
        ++g_uas.adjacent;
    }
}

void BufferCache::NoteTransferEnd() {
    g_uas.last_stream = scheduler.ActiveStream();
    g_uas.last_seq = scheduler.RecordSeqOf(g_uas.last_stream);
}

void BufferCache::NoteUploadAhead(const Buffer& buffer, VAddr device_addr, u32 size, u64 bytes) {
    constexpr u64 page = TRACKER_BYTES_PER_PAGE;
    const VAddr start = Common::AlignUp(device_addr + size, page);
    const VAddr buffer_end = buffer.CpuAddr() + buffer.SizeBytes();
    u32 dirty = 0;
    u32 first = 0;
    for (u32 i = 0; i < 64; ++i) {
        const VAddr p = start + i * page;
        if (p + page > buffer_end) {
            break;
        }
        if (memory_tracker->IsRegionCpuModified(p, page)) {
            ++dirty;
            if (first == 0) {
                first = i + 1;
            }
        }
    }
    const u32 bin = dirty == 0 ? 0 : dirty == 64 ? 7 : std::min<u32>(std::bit_width(dirty), 6);
    ++g_uas.ahead[bin];
    ++g_uas.gap[first == 1 ? 0 : first != 0 && first <= 4 ? 1 : first != 0 && first <= 16 ? 2 : 3];
    ++g_uas.ahead_same_run[first == 1 ? 0 : 1];
    g_uas.upload_bytes += bytes;
    const u64 n = ++g_uas.uploads;
    if ((n & 0x3fff) == 0) {
        auto v = [](const std::atomic<u64>& a) { return a.load(std::memory_order_relaxed); };
        LOG_INFO(Render_Vulkan,
                 "UploadAhead: uploads={} bytes={} copies={} transfers={} adjacent={} "
                 "ahead[0,1,2-3,4-7,8-15,16-31,32-63,64]={},{},{},{},{},{},{},{} "
                 "first_gap[1,2-4,5-16,none/far]={},{},{},{}",
                 n, v(g_uas.upload_bytes), v(g_uas.copies), v(g_uas.transfers), v(g_uas.adjacent),
                 v(g_uas.ahead[0]), v(g_uas.ahead[1]), v(g_uas.ahead[2]), v(g_uas.ahead[3]),
                 v(g_uas.ahead[4]), v(g_uas.ahead[5]), v(g_uas.ahead[6]), v(g_uas.ahead[7]),
                 v(g_uas.gap[0]), v(g_uas.gap[1]), v(g_uas.gap[2]), v(g_uas.gap[3]));
    }
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, VAddr device_addr, u32 size, bool is_written,
                                    bool is_texel_buffer,
                                    Common::BufferSampleProbe* sample_probe) {
    // q31 (Study A E8): SHADPS4_LANE_BIND_UPLOADS=1 - a main-stream upload into a quiet probe range goes to the lane.
    if (g_lane_bind_uploads && g_lane_on && !Vulkan::Scheduler::t_lane_mode &&
        Common::FaultMonitor::t_is_cp_thread && scheduler.HasLane() && scheduler.ActiveStream() == 0 &&
        size != 0 && !is_texel_buffer && scheduler.LaneWaitMain() < scheduler.CurrentTick()) [[unlikely]] {
        // q31c: no texel binds; no routing while the open segment is deferred behind the open batch.
        const u64 open = scheduler.CurrentTick();
        // q31b: the quiet answer captured before the bind's own stamp, for exactly this range.
        const bool pre = g_q31_pre && g_q31_pre_tick == open && g_q31_pre_lo == device_addr &&
                         g_q31_pre_hi == device_addr + size;
        g_q31_pre = false;
        u64 prod = 0;
        bool route = memory_tracker->IsRegionCpuModified(device_addr, size) &&
                     (pre || LaneProbeQuiet(device_addr, device_addr + size, open)) &&
                     !memory_tracker->IsRegionGpuModified(device_addr, size);
        if (route) {
            // q31b (review blocker, WAW): no main write / upload of the range in the open batch; an older unfinished
            // one is waited for by the lane segment.
            g_lc_done = scheduler.GetMasterSemaphore()->KnownGpuTick();
            prod = LaneMainProducer(device_addr, device_addr + size);
            route = prod < open;
        }
        if (route) {
            static u64 q31_routed = 0;
            const u64 b0 = sync_upload_bytes_total;
            scheduler.SwitchStream(1);
            scheduler.LaneNeedMain(prod, 0, false, LaneSiteRaw(g_lc_hit_wk));
            const bool r = SynchronizeBuffer(buffer, device_addr, size, is_written, is_texel_buffer, sample_probe);
            scheduler.SwitchStream(0);
            if (sync_upload_bytes_total != b0) {
                scheduler.LaneFallbackMain(); // the open main batch (the draw that bound it) reads after the upload
                if ((++q31_routed & 0x3fff) == 0) {
                    LOG_INFO(Render, "q31: {} bind-time uploads routed to the lane", q31_routed);
                }
            }
            return r;
        }
    }
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes = 0;
    VAddr buffer_start = buffer.CpuAddr();
    vk::Buffer src_buffer = VK_NULL_HANDLE;
    if (g_lane_on && Vulkan::Scheduler::t_lane_mode) [[unlikely]] {
        buffer.lane_bound = true; // q22
    }
    const auto t_upload = sample_probe ? sample_probe->Now()
                                       : Common::BufferSampleProbe::Clock::time_point{};
    // SHADPS4_DEFERRED_PROTECT=1: the re-protect syscalls and the guest-to-staging copy of this
    // upload move to the recording thread, in the same order (protect, then copy). The command
    // processor still clears the dirty bits and updates watcher counts now. A guest write that
    // lands before the deferred protect is applied is captured by the copy that follows it; one
    // after it faults as usual. The copy reads the physical backing, so pages read-protected in
    // the meantime (readbacks) cannot fault on the recording thread.
    // Measured 2026-09-23: VirtualProtect was 10.2% and this copy ~2% of CP wall time.
    const bool deferred_protect = g_deferred_protect;
    // q27: SHADPS4_LANE_DEFER_UPLOAD=1 - lane-mode uploads defer to the lane recorder thread (q12) the same way.
    static const bool lane_defer = [] {
        const char* e = std::getenv("SHADPS4_LANE_DEFER_UPLOAD");
        const bool on = e && e[0] == '1';
        if (on) {
            LOG_INFO(Render, "q27: SHADPS4_LANE_DEFER_UPLOAD=1 (lane uploads: protect + copy on the lane recorder)");
        }
        return on;
    }();
    const bool lane_deferred = lane_defer && deferred_protect && scheduler.LaneRecordsQueued();
    // q36 (D2 R2): SHADPS4_LANE_DEFER_UPLOAD=2/3 - without the recorder: the CPU half on the LaneUpload FIFO.
    const bool lane_deferred2 = !lane_deferred && deferred_protect && g_lane_on && Vulkan::Scheduler::t_lane_mode &&
                                Common::FaultMonitor::t_is_cp_thread && scheduler.LaneUploadOn();
    const bool defer = deferred_protect && (scheduler.IsAsyncRecording() || lane_deferred || lane_deferred2);
    if (Q36Mode() == 3 && Common::FaultMonitor::t_is_cp_thread) [[unlikely]] {
        // q36 verify: deferred pages in this range that are clean now must still hold the bytes uploaded.
        std::scoped_lock lk{q36_mu};
        auto it = q36_pages.lower_bound(device_addr & ~u64{0xfff});
        while (it != q36_pages.end() && it->first < device_addr + size) {
            if (!it->second.ready || memory_tracker->IsRegionCpuModified(it->first, 0x1000)) {
                ++it; // not hashed yet / re-dirtied (its next upload drops it)
                continue;
            }
            if (memory_tracker->IsRegionGpuModified(it->first, 0x1000)) {
                it = q36_pages.erase(it);
                continue;
            }
            std::array<u8, 0x1000> page;
            if (memory->TryReadBacking(it->first, page.data(), page.size())) {
                q36_compares.fetch_add(1, std::memory_order_relaxed);
                if (XXH3_64bits(page.data(), page.size()) != it->second.hash &&
                    (memory_tracker->IsRegionCpuModified(it->first, 0x1000) ||
                     memory_tracker->IsRegionGpuModified(it->first, 0x1000))) {
                    q36_raced.fetch_add(1, std::memory_order_relaxed); // review 5a: a fault / GPU mark during the read
                } else if (XXH3_64bits(page.data(), page.size()) != it->second.hash) {
                    const u64 n = q36_lost.fetch_add(1, std::memory_order_relaxed) + 1;
                    if ((n & (n - 1)) == 0) {
                        LOG_CRITICAL(Render, "q36 verify: LOST CPU WRITE? clean page {:#x} differs from its deferred "
                                     "lane upload (n={})", it->first, n);
                    }
                }
            }
            it = q36_pages.erase(it);
        }
    }
    if (g_deferred_copy_census) [[unlikely]] {
        census_lo = device_addr;
        census_hi = device_addr + size;
    }
    if (defer) {
        deferred_protects.clear();
        deferred_copies.clear();
        PageManager::SetDeferredProtectSink(&deferred_protects);
    }
    const u64 ibc_t0 = g_image_buffer_cost ? __rdtsc() : 0;
    SCOPE_EXIT {
        if (g_image_buffer_cost) {
            g_sync_rest_cyc = __rdtsc() - ibc_t0 - g_sync_walk_cyc;
        }
    };
    const auto upload_path = memory_tracker->ForEachUploadRange(
        device_addr, size, is_written,
        [&](u64 device_addr_out, u64 range_size) {
            copies.emplace_back(total_size_bytes, device_addr_out - buffer_start, range_size);
            total_size_bytes += range_size;
        },
        [&] {
            src_buffer = defer ? ReserveDeferredCopies(buffer, copies, total_size_bytes)
                               : UploadCopies(buffer, copies, total_size_bytes);
        });
    if (g_image_buffer_cost) {
        g_sync_walk_cyc = __rdtsc() - ibc_t0;
    }
    last_sync_upload_bytes = total_size_bytes;
    sync_upload_bytes_total += total_size_bytes;
    if (Q36Mode() == 3 && total_size_bytes != 0) [[unlikely]] {
        for (const auto& copy : copies) {
            Q36Drop(buffer_start + copy.dstOffset, copy.size); // q36 verify: re-uploaded, older entries void
        }
    }
    if (total_size_bytes != 0) {
        // Upload volume, every ~5 s, only from calls that upload (NATIVE_COHERENT pair check).
        ++sync_upload_calls;
        Common::FaultMonitor::sync_upload_bytes.fetch_add(total_size_bytes, std::memory_order_relaxed);
        Common::FaultMonitor::sync_upload_calls.fetch_add(1, std::memory_order_relaxed);
        const u64 now = __rdtsc();
        if (now - sync_upload_log_tsc > 17'500'000'000ULL) {
            sync_upload_log_tsc = now;
            LOG_INFO(Render, "SyncUploads: total={} MB uploads={} coherent={}",
                     sync_upload_bytes_total >> 20, sync_upload_calls, g_native_coherent);
            if (Q36Mode() == 3) {
                size_t tracked = 0;
                {
                    std::scoped_lock lk{q36_mu};
                    tracked = q36_pages.size();
                }
                LOG_INFO(Render, "q36 verify: compares {} lost_write {} raced {} dropped {} unbacked_at_fifo {} "
                         "unbacked_inline {} tracked {} (cap 8192: at the cap verification stops)",
                         q36_compares.load(), q36_lost.load(), q36_raced.load(), q36_dropped.load(),
                         q36_unmapped.load(), q36_unbacked.load(), tracked);
            }
            if (const volatile u32* g = g_sort_guard_words) {
                // q36: the sort replacement's invalid-width exits (0 = never; the hang guard did not fire).
                LOG_INFO(Render, "q36 SortGuard: invalid-width exits {} (last first {} last {}), overflow exits {}",
                         g[0], g[1], g[2], g[3]);
                if (g[4] != 0) { // SHADPS4_SORT_VERIFY=1
                    LOG_INFO(Render,
                             "SortVerify: dispatches {} | input out of order: lanes {} groups {} (worst "
                             "first width {}, count {}) | own output out of order: lanes {}",
                             g[4], g[5], g[6], g[8], g[9], g[7]);
                }
            }
        }
    }
    if (total_size_bytes != 0) {
        // Whole tracker pages: a later binding of the same page would have found it dirty too.
        const u64 page_lo = device_addr & ~u64{0xfff};
        const u64 page_hi = (device_addr + size + 0xfff) & ~u64{0xfff};
        AmdGpu::MtbCensus::NoteEventRange(AmdGpu::MtbCensus::BufferUpload, page_lo, page_hi - page_lo);
    }
    if (g_upload_ahead_stats && src_buffer) [[unlikely]] {
        NoteUploadAhead(buffer, device_addr, size, total_size_bytes);
    }
    if ((g_lane_census || g_lane_on) && total_size_bytes != 0) [[unlikely]] {
        if (Common::FaultMonitor::t_is_cp_thread) {
            const bool lane_side =
                g_lane_on ? Vulkan::Scheduler::t_lane_mode : t_force_binding_sync;
            if (g_lane_on && lane_side) {
                for (const auto& copy : copies) { // q9c: the upload's own ranges
                    const VAddr lo = buffer_start + copy.dstOffset;
                    LaneWarWait(scheduler, lo, lo + copy.size);
                }
            }
            for (const auto& copy : copies) {
                const VAddr lo = buffer_start + copy.dstOffset;
                LaneNoteRange(lo, lo + copy.size, scheduler.CurrentTick(), lane_side ? 3 : 1,
                              100 + census_kind, census_id);
            }
        } else {
            g_lc[13].fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (total_size_bytes != 0) {
        // An upload is a main write of the Buffer (SynchronizeBuffersInRange reaches here without
        // any other note; Opus review 2026-09-28). Skipped on the ASC stream by NoteMainAccess.
        buffer.NoteMainAccess(scheduler.CurrentTick(), true);
        if (g_asc_deps && g_record_stream == 1) {
            for (const auto& copy : copies) {
                pending_asc.push_back({page_table[buffer_start >> CACHING_PAGEBITS].buffer_id,
                                       nullptr, buffer_start + copy.dstOffset, copy.size, true});
            }
        }
    }
    if (g_range_notes && total_size_bytes != 0) {
        for (const auto& copy : copies) {
            buffer.NoteMainRange(scheduler.CurrentTick(), true, buffer_start + copy.dstOffset,
                                 copy.size, 9, 0);
        }
        if (g_record_stream == 1 && buffer.range_log &&
            buffer.range_log->tick == scheduler.CurrentTick()) [[unlikely]] {
            // SHADPS4_ASC_RANGE falsifier: an ASC-stream upload over a range the open main batch
            // accessed (the pre-check raced a guest write, or a note is missing).
            for (const auto& copy : copies) {
                const VAddr lo = buffer_start + copy.dstOffset;
                const VAddr hi = lo + copy.size;
                bool hit = buffer.range_log->whole;
                for (const auto& e : buffer.range_log->entries) {
                    hit |= e.lo < hi && lo < e.hi;
                }
                if (hit) {
                    static std::atomic<u64> late{0};
                    const u64 n = late.fetch_add(1) + 1;
                    if ((n & (n - 1)) == 0) {
                        LOG_CRITICAL(Render, "AscRange: late upload WAR at {:#x}+{:#x} (n={})", lo,
                                     copy.size, n);
                    }
                    break;
                }
            }
        }
    }
    if (defer) {
        PageManager::SetDeferredProtectSink(nullptr);
        if (!src_buffer && !deferred_protects.empty()) {
            // Nothing to copy after all; apply the protection here.
            page_manager.ReconcileProtection(deferred_protects);
            deferred_protects.clear();
        }
    }
    if (sample_probe) {
        sample_probe->AddUploadNs(sample_probe->SinceNs(t_upload));
        sample_probe->RecordUpload(upload_path, total_size_bytes, copies.size());
    }

    if (src_buffer) {
        scheduler.EndRendering();
        const vk::BufferMemoryBarrier2 pre_barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite |
                             vk::AccessFlagBits2::eTransferRead |
                             vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .buffer = buffer.Handle(),
            .offset = 0,
            .size = buffer.SizeBytes(),
        };
        const vk::BufferMemoryBarrier2 post_barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .buffer = buffer.Handle(),
            .offset = 0,
            .size = buffer.SizeBytes(),
        };
        if (g_upload_ahead_stats) {
            NoteTransferBegin();
        }
        SCOPE_EXIT {
            if (g_upload_ahead_stats) {
                NoteTransferEnd();
            }
        };
        bool q36_backed = lane_deferred2;
        if (lane_deferred2) {
            // q36d (review 1): a range without physical backing gets its CPU half inline (the main path's order).
            for (const DeferredCopy& c : deferred_copies) {
                q36_backed = q36_backed && memory->FullyBacked(c.addr, c.size);
            }
            if (!q36_backed) {
                q36_unbacked.fetch_add(1, std::memory_order_relaxed);
                page_manager.ReconcileProtection(deferred_protects);
                for (const DeferredCopy& c : deferred_copies) {
                    if (!memory->TryReadBacking(c.addr, c.dest, c.size)) {
                        memory->CopySparseMemory(c.addr, c.dest, c.size);
                    }
                }
            }
        }
        if (lane_deferred2) {
            // q36: the CPU half (protect, then copy - the main path's order) on the LaneUpload FIFO; the lane
            // segment's submit is queued behind it (AscSubmitOnWorker). The GPU half is recorded now, as inline.
            // (=3: this upload's older page entries were dropped above, with every upload's.)
            std::vector<std::tuple<VAddr, u64, u8*>> verify_pages;
            if (Q36Mode() == 3 && q36_backed) [[unlikely]] {
                std::scoped_lock lk{q36_mu};
                for (const DeferredCopy& c : deferred_copies) {
                    for (VAddr pg = (c.addr + 0xfff) & ~u64{0xfff}; pg + 0x1000 <= c.addr + c.size; pg += 0x1000) {
                        if (q36_pages.size() < 8192) {
                            q36_pages[pg] = {0, ++q36_seq, false};
                            verify_pages.emplace_back(pg, q36_seq, c.dest + (pg - c.addr));
                        }
                    }
                }
            }
            if (q36_backed && !(deferred_copies.empty() && deferred_protects.empty())) // review 8: no empty jobs
            scheduler.LaneUploadDefer(
                [pm = &page_manager, memory = memory,
                 protects = std::vector<PageManager::ProtectRange>(deferred_protects.begin(), deferred_protects.end()),
                 cpu = std::vector<DeferredCopy>(deferred_copies.begin(), deferred_copies.end()),
                 verify_pages = std::move(verify_pages)]() {
                    pm->ReconcileProtection(protects);
                    bool all_ok = true;
                    for (const DeferredCopy& copy : cpu) {
                        if (!memory->TryReadGuest(copy.addr, copy.dest, copy.size)) {
                            q36_unmapped.fetch_add(1, std::memory_order_relaxed); // I3: unbacked since the push
                            all_ok = false;
                        }
                    }
                    if (!verify_pages.empty() && all_ok) { // review 5b: never arm a page filled with zeros
                        std::scoped_lock lk{q36_mu};
                        for (const auto& [pg, seq, src] : verify_pages) {
                            if (auto it = q36_pages.find(pg); it != q36_pages.end() && it->second.seq == seq) {
                                it->second.hash = XXH3_64bits(src, 0x1000);
                                it->second.ready = true;
                            }
                        }
                    }
                });
            scheduler.NoteAscWork(1); // G0: an upload into the open lane segment
            scheduler.RecordArrays(
                [src_buffer, destination = buffer.Handle(), pre_barrier,
                 post_barrier](vk::CommandBuffer cmdbuf, auto owned_copies) {
                    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                        .bufferMemoryBarrierCount = 1,
                        .pBufferMemoryBarriers = &pre_barrier,
                    });
                    cmdbuf.copyBuffer(src_buffer, destination, owned_copies);
                    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                        .bufferMemoryBarrierCount = 1,
                        .pBufferMemoryBarriers = &post_barrier,
                    });
                },
                std::span<const vk::BufferCopy>{copies});
            TouchBuffer(buffer);
        } else if (defer && g_transfer_coalesce && !lane_deferred) {
            // SHADPS4_TRANSFER_COALESCE: the barriers come from the scheduler's transfer group.
            Vulkan::Scheduler::TransferRange write{static_cast<VkBuffer>(buffer.Handle()), ~0ULL, 0};
            for (const auto& copy : copies) {
                write.lo = std::min<u64>(write.lo, copy.dstOffset);
                write.hi = std::max<u64>(write.hi, copy.dstOffset + copy.size);
            }
            // SHADPS4_PAR_REC: protect + guest copy in the single recorder's order.
            scheduler.NoteAscWork(1); // G0: an upload into the open lane segment
            scheduler.RecordArraysOrdered<true>(
                [src_buffer, destination = buffer.Handle(), write, sched = &scheduler,
                 memory = memory, pm = &page_manager](vk::CommandBuffer cmdbuf, auto owned_copies,
                                                      auto cpu_copies, auto protects) {
                    // Protect first, then copy: see SynchronizeBuffer.
                    pm->ReconcileProtection(protects);
                    for (const DeferredCopy& copy : cpu_copies) {
                        if (!memory->TryReadBacking(copy.addr, copy.dest, copy.size)) {
                            memory->CopySparseMemory(copy.addr, copy.dest, copy.size);
                        }
                        if (g_deferred_copy_census) [[unlikely]] {
                            NoteDeferredCopyCensus(copy.addr, copy.dest, copy.size, copy.hash,
                                                   copy.label_seq, copy.hash_off, copy.hash_len,
                                                   copy.census_kind);
                        }
                    }
                    sched->BeginTransferOnWorker(cmdbuf, {}, write);
                    cmdbuf.copyBuffer(src_buffer, destination, owned_copies);
                },
                std::span<const vk::BufferCopy>{copies},
                std::span<const DeferredCopy>{deferred_copies},
                std::span<const PageManager::ProtectRange>{deferred_protects});
            TouchBuffer(buffer);
        } else if (defer) {
            scheduler.NoteAscWork(1); // G0: an upload into the open lane segment
            scheduler.RecordArraysOrdered(
                [src_buffer, destination = buffer.Handle(), pre_barrier, post_barrier,
                 memory = memory, pm = &page_manager](vk::CommandBuffer cmdbuf, auto owned_copies,
                                                      auto cpu_copies, auto protects) {
                    // Protect first, then copy: see SynchronizeBuffer.
                    pm->ReconcileProtection(protects);
                    for (const DeferredCopy& copy : cpu_copies) {
                        if (!memory->TryReadBacking(copy.addr, copy.dest, copy.size)) {
                            memory->CopySparseMemory(copy.addr, copy.dest, copy.size);
                        }
                        if (g_deferred_copy_census) [[unlikely]] {
                            NoteDeferredCopyCensus(copy.addr, copy.dest, copy.size, copy.hash,
                                                   copy.label_seq, copy.hash_off, copy.hash_len,
                                                   copy.census_kind);
                        }
                    }
                    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                        .bufferMemoryBarrierCount = 1,
                        .pBufferMemoryBarriers = &pre_barrier,
                    });
                    cmdbuf.copyBuffer(src_buffer, destination, owned_copies);
                    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                        .bufferMemoryBarrierCount = 1,
                        .pBufferMemoryBarriers = &post_barrier,
                    });
                },
                std::span<const vk::BufferCopy>{copies},
                std::span<const DeferredCopy>{deferred_copies},
                std::span<const PageManager::ProtectRange>{deferred_protects});
            TouchBuffer(buffer);
        } else {
        scheduler.NoteAscWork(1); // G0: an upload into the open lane segment
        scheduler.RecordArrays(
            [src_buffer, destination = buffer.Handle(), pre_barrier,
             post_barrier](vk::CommandBuffer cmdbuf, auto owned_copies) {
                cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                    .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                    .bufferMemoryBarrierCount = 1,
                    .pBufferMemoryBarriers = &pre_barrier,
                });
                cmdbuf.copyBuffer(src_buffer, destination, owned_copies);
                cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                    .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                    .bufferMemoryBarrierCount = 1,
                    .pBufferMemoryBarriers = &post_barrier,
                });
            },
            std::span<const vk::BufferCopy>{copies});
        TouchBuffer(buffer);
        }
    }
    // Main stream only: on the ASC stream the eligibility check (RangeConflict) saw the binding's
    // range, not the pages ahead (Opus review 2026-09-28).
    if (g_upload_ahead_pages != 0 && src_buffer && total_size_bytes <= 16_KB &&
        !t_upload_ahead_active && g_record_stream == 0) [[unlikely]] {
        UploadAhead(buffer, device_addr, size);
    }
    if (is_texel_buffer && !is_written) {
        const auto t_texel = sample_probe ? sample_probe->Now()
                                          : Common::BufferSampleProbe::Clock::time_point{};
        const bool result = SynchronizeBufferFromImage(buffer, device_addr, size);
        if (sample_probe) {
            sample_probe->AddTexelNs(sample_probe->SinceNs(t_texel));
        }
        return result;
    }
    return false;
}

vk::Buffer BufferCache::ReserveDeferredCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                              size_t total_size_bytes) {
    if (copies.empty()) {
        return VK_NULL_HANDLE;
    }
    // Commit flushes non-coherent memory immediately, before a deferred copy would have run.
    const auto [staging, offset] = staging_buffer.is_coherent
                                       ? staging_buffer.Map(total_size_bytes)
                                       : std::pair<u8*, u64>{nullptr, 0};
    if (!staging) {
        // Non-coherent staging or a large one-time transfer: synchronous, protection first.
        page_manager.ReconcileProtection(deferred_protects);
        deferred_protects.clear();
        return UploadCopies(buffer, copies, total_size_bytes);
    }
    for (auto& copy : copies) {
        deferred_copies.push_back(
            {buffer.CpuAddr() + copy.dstOffset, staging + copy.srcOffset, copy.size});
        if (g_deferred_copy_census) [[unlikely]] {
            // The part of this copy inside the binding being served, as the command processor
            // sees it now (through the backing, like the deferred copy itself). The rest of the
            // copied pages belongs to other allocations the guest may still be filling.
            auto& census = deferred_copies.back();
            const VAddr lo = std::max(census.addr, census_lo);
            const VAddr hi = std::min<VAddr>(census.addr + census.size, census_hi);
            if (t_upload_ahead_active) {
                census.census_kind = 2;
            } else if (lo < hi) {
                constexpr u64 Edge = 512;
                const u64 len = hi - lo;
                std::array<u8, 2 * Edge> sample;
                bool ok;
                u64 hash;
                if (len <= 2 * Edge) {
                    ok = memory->TryReadBacking(lo, sample.data(), len);
                    hash = ok ? XXH3_64bits(sample.data(), len) : 0;
                } else {
                    ok = memory->TryReadBacking(lo, sample.data(), Edge) &&
                         memory->TryReadBacking(hi - Edge, sample.data() + Edge, Edge);
                    hash = ok ? XXH3_64bits(sample.data(), Edge) ^
                                    (XXH3_64bits(sample.data() + Edge, Edge) *
                                     0x9E3779B97F4A7C15ULL)
                              : 0;
                }
                census.census_kind = ok ? 1 : 3;
                census.hash = hash;
                census.hash_off = static_cast<u32>(lo - census.addr);
                census.hash_len = static_cast<u32>(len);
                census.label_seq = VideoCore::g_guest_label_seq.load(std::memory_order_relaxed);
            }
        }
        copy.srcOffset += offset;
    }
    staging_buffer.Commit();
    return staging_buffer.Handle();
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     size_t total_size_bytes) {
    if (copies.empty()) {
        return VK_NULL_HANDLE;
    }
    const auto [staging, offset] = staging_buffer.Map(total_size_bytes);
    if (staging) {
        for (auto& copy : copies) {
            u8* const src_pointer = staging + copy.srcOffset;
            const VAddr device_addr = buffer.CpuAddr() + copy.dstOffset;
            memory->CopySparseMemory(device_addr, src_pointer, copy.size);
            // Apply the staging offset
            copy.srcOffset += offset;
        }
        staging_buffer.Commit();
        return staging_buffer.Handle();
    } else {
        // 1.1.1 (GitHub #36, RTX 5070 12 GB, 32 GB RAM, driver 621.10): the 3.75 GB mesh buffer's
        // first upload asked for one 3822 MB host buffer and the driver refused it (error -2, with
        // 12.7 GB of the system heap's budget free) - the game closed at the loading screen. Probed
        // first; refused -> UploadCopiesInPieces (256 MB at a time). PCs where it fits are unchanged.
        // SHADPS4_UPLOAD_PIECES_TEST=1 (test): every upload above 256 MB takes the pieces path.
        static const bool pieces_test = [] {
            const char* v = std::getenv("SHADPS4_UPLOAD_PIECES_TEST");
            return v && v[0] == '1';
        }();
        u32 heap = 0;
        if (total_size_bytes > UploadPieceSize &&
            (pieces_test ||
             !ProbeBufferAllocation(instance, MemoryUsage::Upload,
                                    vk::BufferUsageFlagBits::eTransferSrc, total_size_bytes, heap))) {
            UploadCopiesInPieces(buffer, copies, total_size_bytes, pieces_test);
            return VK_NULL_HANDLE;
        }
        // For large one time transfers use a temporary host buffer.
        auto temp_buffer =
            std::make_unique<Buffer>(instance, scheduler, MemoryUsage::Upload, 0,
                                     vk::BufferUsageFlagBits::eTransferSrc, total_size_bytes);
        const vk::Buffer src_buffer = temp_buffer->Handle();
        u8* const staging = temp_buffer->mapped_data.data();
        for (const auto& copy : copies) {
            u8* const src_pointer = staging + copy.srcOffset;
            const VAddr device_addr = buffer.CpuAddr() + copy.dstOffset;
            memory->CopySparseMemory(device_addr, src_pointer, copy.size);
        }
        scheduler.DeferOperation([buffer = std::move(temp_buffer)]() mutable { buffer.reset(); });
        return src_buffer;
    }
}

void BufferCache::UploadCopiesInPieces(Buffer& buffer, std::span<const vk::BufferCopy> copies,
                                       size_t total_size_bytes, bool test) {
    // One host buffer of UploadPieceSize, refilled per piece; the GPU finishes each piece's copies
    // before the next fill (once per mesh buffer at a loading screen, so the waits do not matter).
    auto piece = std::make_unique<Buffer>(instance, scheduler, MemoryUsage::Upload, 0,
                                          vk::BufferUsageFlagBits::eTransferSrc, UploadPieceSize);
    u8* const staging = piece->mapped_data.data();
    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .buffer = buffer.Handle(),
        .offset = 0,
        .size = buffer.SizeBytes(),
    };
    const vk::BufferMemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .buffer = buffer.Handle(),
        .offset = 0,
        .size = buffer.SizeBytes(),
    };
    boost::container::small_vector<vk::BufferCopy, 16> batch;
    u64 used = 0;
    u32 pieces = 0;
    const auto submit = [&] {
        if (batch.empty()) {
            return;
        }
        scheduler.EndRendering();
        {
            auto access = scheduler.AcquireCommandBuffer();
            const auto cmdbuf = access.Handle();
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                .bufferMemoryBarrierCount = 1,
                .pBufferMemoryBarriers = &pre_barrier,
            });
            cmdbuf.copyBuffer(piece->Handle(), buffer.Handle(), batch);
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                .bufferMemoryBarrierCount = 1,
                .pBufferMemoryBarriers = &post_barrier,
            });
        }
        scheduler.Finish(); // the piece buffer is refilled next
        batch.clear();
        used = 0;
        ++pieces;
    };
    for (const vk::BufferCopy& copy : copies) {
        u64 done = 0;
        while (done < copy.size) {
            if (used == UploadPieceSize) {
                submit();
            }
            const u64 n = std::min<u64>(copy.size - done, UploadPieceSize - used);
            const VAddr device_addr = buffer.CpuAddr() + copy.dstOffset + done;
            memory->CopySparseMemory(device_addr, staging + used, n);
            batch.push_back({used, copy.dstOffset + done, n});
            used += n;
            done += n;
        }
    }
    submit();
    LOG_WARNING(Render_Vulkan,
                "Large upload of {} MB in {} pieces of up to {} MB ({}); {}", total_size_bytes >> 20,
                pieces, UploadPieceSize >> 20,
                test ? "SHADPS4_UPLOAD_PIECES_TEST" : "one host buffer of that size was refused",
                Vulkan::DescribeDeviceLocalBudgets(instance.GetAllocator()));
    scheduler.DeferOperation([piece = std::move(piece)]() mutable { piece.reset(); });
    TouchBuffer(buffer);
}

bool BufferCache::SynchronizeBufferFromImage(Buffer& buffer, VAddr device_addr, u32 size) {
    WriterTagScope census_tag{*this, 7};
    buffer.NoteMainAccess(scheduler.CurrentTick(), true);
    if (g_range_notes) {
        buffer.NoteMainRange(scheduler.CurrentTick(), true, device_addr, size, 7, 0);
    }
    // TIER-2 SPLIT PROBE: the ob_sync gap (ob_sync - scan/copy/mark = ~7.7 ms/frame) is untimed and
    // was GUESSED to be lock/framing. This path (read-only texel buffers) is the prime suspect.
    // Split it: FindImageFromRange (texture-cache lookup under TextureCache::mutex) vs TileImage
    // (the real work, only when an image actually aliases). Strip before ship.
    static u64 sfi_calls = 0, sfi_found = 0, sfi_find_ns = 0, sfi_tile_ns = 0;
    const auto sfi_t0 = Common::FaultMonitor::Now();
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    sfi_find_ns += Common::FaultMonitor::SinceNs(sfi_t0);
    // SHADPS4_LEAN: sfi_calls stays 0, so the SfiSplit line is never formatted or written.
    const bool sfi_stats = !::Common::Lean();
    sfi_calls += sfi_stats;
    const auto sfi_report = [&] {
        if (!sfi_stats || sfi_calls < 300000) {
            return;
        }
        LOG_INFO(Render, "SfiSplit: calls={} found={:.1f}% find={:.3f}us/call tile={:.3f}us/found",
                 sfi_calls, 100.0 * static_cast<double>(sfi_found) / sfi_calls,
                 static_cast<double>(sfi_find_ns) / 1e3 / sfi_calls,
                 sfi_found ? static_cast<double>(sfi_tile_ns) / 1e3 / sfi_found : 0.0);
        sfi_calls = sfi_found = sfi_find_ns = sfi_tile_ns = 0;
    };
    if (!image_id) {
        sfi_report();
        return false;
    }
    if (g_record_stream != 0) [[unlikely]] {
        // SHADPS4_ASC_TEXEL falsifier: an image->buffer copy must never be recorded on the ASC
        // stream (BindBuffers keeps such dispatches on main).
        static std::atomic<u64> on_asc{0};
        const u64 n = on_asc.fetch_add(1) + 1;
        if ((n & (n - 1)) == 0) {
            LOG_CRITICAL(Render, "AscTexel: image->buffer copy on the ASC stream at {:#x} (n={})",
                         device_addr, n);
        }
    }
    ++sfi_found;
    if (g_range_notes) {
        buffer.NoteMainWhole(scheduler.CurrentTick(), 7); // TileImage writes the image extent
    }
    Image& image = texture_cache.GetImage(image_id);
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    const u32 buf_offset = buffer.Offset(image.info.guest_address);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    u32 copy_size = 0;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (buf_offset + mip_info.offset + mip_info.size > buffer.SizeBytes()) {
            break;
        }
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
        copy_size += mip_info.size;
    }
    if (copy_size == 0) {
        sfi_report();
        return false;
    }
    if (PrebatchEnabled() || PreciseProducerEnabled() || g_lane_on) { // q31c: the lane's producer lookup too
        NoteBatchWrite(image.info.guest_address,
                       std::min<u64>(std::max<u64>(copy_size, image.info.guest_size),
                                     buffer.SizeBytes() - buf_offset));
    }
    if (pending_fault_jobs.load(std::memory_order_relaxed) != 0) {
        // TileImage writes this cache buffer with no tracker marking, so the region sequence
        // would not move and a pending fault job over these bytes would still pass its vote.
        // Bump it by hand, before the write is recorded: every publication from here on is
        // vetoed and resolved by a synchronous download recorded after the tiling. A settle
        // would also do, but it waits and can reenter the cache while the caller holds
        // `buffer`. A tiled image binds its whole guest size, so cover that too.
        const u64 written = std::min<u64>(std::max<u64>(copy_size, image.info.guest_size),
                                          buffer.SizeBytes() - buf_offset);
        memory_tracker->InvalidateGpuWriteSeq(image.info.guest_address, written);
        Offload::Add(Offload::State().alias_bumps);
    }
    auto& tile_manager = texture_cache.GetTileManager();
    const auto sfi_t1 = Common::FaultMonitor::Now();
    tile_manager.TileImage(image, buffer_copies, buffer.Handle(), buf_offset, copy_size);
    sfi_tile_ns += Common::FaultMonitor::SinceNs(sfi_t1);
    sfi_report();
    return true;
}

void BufferCache::SynchronizeBuffersInRange(VAddr device_addr, u64 size) {
    const VAddr device_addr_end = device_addr + size;
    ForEachBufferInRange(device_addr, size, [&](BufferId buffer_id, Buffer& buffer) {
        RENDERER_TRACE;
        VAddr start = std::max(buffer.CpuAddr(), device_addr);
        VAddr end = std::min(buffer.CpuAddr() + buffer.SizeBytes(), device_addr_end);
        u32 size = static_cast<u32>(end - start);
        SynchronizeBuffer(buffer, start, size, false, false, nullptr);
    });
}

void BufferCache::CoherenceSync(bool bounded, u32 bound_lo) {
    if (g_native_coherent == 0) {
        return;
    }
    if (g_back_clean == 0 || g_epoch_defer_upload || g_range_notes || g_asc_deps) {
        // BACK_CLEAN: the dirty log. EPOCH_DEFER_UPLOAD leaves drained pages dirty and unprotected
        // (never logged again: stale for good; Opus review A). RANGE_NOTES / ASC deps need the
        // per-binding notes the skip does not make (review E).
        if (!coherent_warned) {
            coherent_warned = true;
            LOG_CRITICAL(Render, "NativeCoherent: needs SHADPS4_BACK_CLEAN=1 and no EPOCH_DEFER_UPLOAD "
                                 "/ range notes / ASC deps; off");
        }
        return;
    }
    auto& cs = coherent_stats;
    WriterTagScope census_tag{*this, 9}; // q22: its uploads are not the last draw's
    // q22 (design-game-main D2 blocker 1): SHADPS4_LANE_UPLOADS=1 - dirty runs of lane-bound buffers upload on the lane.
    static const bool lane_uploads = [] {
        const char* e = std::getenv("SHADPS4_LANE_UPLOADS");
        const bool on = e && e[0] == '1';
        if (on) {
            LOG_INFO(Render, "q22: SHADPS4_LANE_UPLOADS=1 (CoherenceSync uploads lane-bound buffers on the lane)");
        }
        return on;
    }();
    const bool lane_up = lane_uploads && g_lane_on && g_lane_war && scheduler.HasLane();
    const u32 stream0 = scheduler.ActiveStream();
    if (lane_up && stream0 != 0) {
        scheduler.SwitchStream(0); // review q22 #2: the sweep and non-routed runs upload on main
    }
    const u64 c0 = __rdtsc();
    const u64 tail = DirtyLog::Start();
    u64 end = tail;
    if (bounded) {
        // The front read the tail when it began the submission; it is at most one u32 lap back.
        end = tail - static_cast<u32>(static_cast<u32>(tail) - bound_lo);
        ++cs.bounded;
    }
    coherent_runs.clear();
    const bool ok = DirtyLog::DrainTo(coherent_head, end, [this](u64 first, u64 count, bool) {
        coherent_runs.emplace_back(first, count);
    });
    bool peek_ok = true;
    if (bounded && ok && end != tail) {
        // The bound is the front's view; the front runs ahead of the back, so the CP's own guest
        // writes (bit 10; before this point in stream order) may sit after it (s451: 75 missed
        // pages at load). Peek past the bound for them; they are drained again later (clean then).
        u64 peek = coherent_head;
        peek_ok = DirtyLog::DrainTo(peek, tail, [this, &cs](u64 first, u64 count, bool by_cp) {
            if (by_cp) {
                ++cs.cp_peeked;
                coherent_runs.emplace_back(first, count);
            }
        });
    }
    const u64 bytes0 = sync_upload_bytes_total;
    if (!ok || !peek_ok || !coherent_live) {
        // First point, or log entries were lost: every cached buffer, whole (rare: loading).
        ++cs.sweeps;
        ForEachBufferInRange(0, u64{1} << 40, [this](BufferId, Buffer& buffer) {
            SynchronizeBuffer(buffer, buffer.CpuAddr(), static_cast<u32>(buffer.SizeBytes()),
                              false, false, nullptr);
        });
    } else if (!coherent_runs.empty()) {
        std::sort(coherent_runs.begin(), coherent_runs.end());
        u64 lo = coherent_runs[0].first;
        u64 hi = lo + coherent_runs[0].second;
        const auto flush = [&] {
            ++cs.runs;
            if (!lane_up) {
                SynchronizeBuffersInRange(lo << TRACKER_PAGE_BITS, (hi - lo) << TRACKER_PAGE_BITS);
                return;
            }
            // q22: SynchronizeBuffersInRange per buffer; a lane-bound buffer goes to the lane pass only where the
            // lane's WAR wait is exact (one pool probe contains the range: LaneWarWait scopes to the probe's newest
            // main read, deferred behind the open batch if that read is in it), main has not touched the buffer in
            // the open batch, and something is dirty (review q22 #1, #4). Everything else uploads here, on main.
            const VAddr a = lo << TRACKER_PAGE_BITS;
            const VAddr a_end = hi << TRACKER_PAGE_BITS;
            ForEachBufferInRange(a, a_end - a, [&](BufferId id, Buffer& buffer) {
                const VAddr start = std::max(buffer.CpuAddr(), a);
                const VAddr end = std::min(buffer.CpuAddr() + buffer.SizeBytes(), a_end);
                if (buffer.lane_bound) {
                    bool probe = false;
                    const u32 np = g_lprobe_n.load(std::memory_order_acquire);
                    for (u32 k = 0; k < np && !probe; ++k) {
                        probe = g_lprobe[k].lo.load(std::memory_order_relaxed) <= start &&
                                end <= g_lprobe[k].hi.load(std::memory_order_relaxed);
                    }
                    if (probe && buffer.main_access_tick < scheduler.CurrentTick() &&
                        memory_tracker->IsRegionCpuModified(start, end - start)) {
                        coherent_lane.emplace_back(id, start, end);
                        return;
                    }
                    ++cs.lane_kept;
                }
                SynchronizeBuffer(buffer, start, static_cast<u32>(end - start), false, false, nullptr);
            });
        };
        for (size_t i = 1; i < coherent_runs.size(); ++i) {
            const auto [first, count] = coherent_runs[i];
            if (first <= hi) {
                hi = std::max(hi, first + count);
                continue;
            }
            flush();
            lo = first;
            hi = first + count;
        }
        flush();
    }
    if (lane_up) {
        if (!coherent_lane.empty()) {
            // q22: one lane pass (one switch) after the main uploads.
            const u64 b0 = sync_upload_bytes_total;
            scheduler.SwitchStream(1);
            for (const auto& [id, start, end] : coherent_lane) {
                if (IsBufferInvalid(id)) {
                    scheduler.SwitchStream(0);
                    SynchronizeBuffersInRange(start, end - start);
                    scheduler.SwitchStream(1);
                    continue;
                }
                // q31b (review, WAW): an older unfinished main write / upload of the range comes first.
                g_lc_done = scheduler.GetMasterSemaphore()->KnownGpuTick();
                scheduler.LaneNeedMain(LaneMainProducer(start, end), 0, false, LaneSiteRaw(g_lc_hit_wk));
                SynchronizeBuffer(slot_buffers[id], start, static_cast<u32>(end - start), false, false,
                                  nullptr);
                ++cs.lane_bufs;
            }
            scheduler.SwitchStream(0);
            coherent_lane.clear();
            if (sync_upload_bytes_total != b0) {
                // q22: graphics reading the uploaded bytes runs after them (the main batch waits the segment).
                cs.lane_bytes += sync_upload_bytes_total - b0;
                ++cs.lane_points;
                scheduler.LaneFallbackMain();
            }
        }
        if (scheduler.ActiveStream() != stream0) {
            scheduler.SwitchStream(stream0);
        }
    }
    memory_tracker->BackCleanSync();
    coherent_live = true;
    ++cs.points;
    cs.bytes += sync_upload_bytes_total - bytes0;
    const u64 dt = __rdtsc() - c0;
    cs.cyc += dt;
    cs.max_cyc = std::max(cs.max_cyc, dt);
    if ((cs.points & 0x7fff) == 0) {
        LOG_INFO(Render,
                 "NativeCoherent: mode={} points={} (bounded {}) sweeps={} runs={} uploaded={} KB "
                 "cycles/point={:.0f} (max {}) dma_syncs={} created={} cp_peeked={} | skips={} span_skips={} | "
                 "verify: checked={} late={} cp={} BAD={} raced={} bytes={} KB | q22 lane: points={} buffers={} "
                 "uploaded={} KB kept on main={}",
                 g_native_coherent, cs.points, cs.bounded, cs.sweeps, cs.runs, cs.bytes >> 10,
                 double(cs.cyc) / double(cs.points), cs.max_cyc, cs.dma_syncs, cs.created,
                 cs.cp_peeked, cs.skips, cs.span_skips, cs.verified, cs.v_late, cs.v_cp, cs.v_bad, cs.v_raced,
                 cs.v_bytes >> 10, cs.lane_points, cs.lane_bufs, cs.lane_bytes >> 10, cs.lane_kept);
        cs.max_cyc = 0;
    }
}

void BufferCache::CoherentVerify(Buffer& buffer, VAddr addr, u32 size) {
    // The per-binding sync the skip replaced. An upload means a page of the range was dirty:
    // after the point's drained position by a guest thread = written after the submission began
    // (legal to miss, submit_epoch.h); by the CP thread = the CP's own write mid-submission that
    // nothing synced (a bug); no log entry at all = a bug.
    auto& cs = coherent_stats;
    ++cs.verified;
    const u64 before = sync_upload_bytes_total;
    SynchronizeBuffer(buffer, addr, size, false, false, nullptr);
    const u64 uploaded = sync_upload_bytes_total - before;
    if (uploaded == 0) {
        return;
    }
    cs.v_bytes += uploaded;
    const u64 lo = addr >> TRACKER_PAGE_BITS;
    const u64 hi = (addr + size + TRACKER_BYTES_PER_PAGE - 1) >> TRACKER_PAGE_BITS;
    bool late = false;
    bool cp = false;
    u64 cp_page = 0;
    u64 head = coherent_head;
    const bool ok = DirtyLog::DrainTo(head, ~u64{0}, [&](u64 first, u64 count, bool by_cp) {
        if (first < hi && lo < first + count) {
            (by_cp ? cp : late) = true;
            if (by_cp) {
                cp_page = first;
            }
        }
    });
    if (!ok) {
        ++cs.v_raced;
    } else if (cp) {
        if (++cs.v_cp <= 20) {
            LOG_ERROR(Render,
                      "NativeCoherent verify: CP-written page not synced, {:#x}+{:#x} ({} B), CP "
                      "entry page {:#x}",
                      addr, size, uploaded, cp_page << TRACKER_PAGE_BITS);
        }
    } else if (late) {
        ++cs.v_late;
    } else if (++cs.v_bad <= 20) {
        LOG_ERROR(Render, "NativeCoherent verify: BAD, {:#x}+{:#x} uploaded {} B with no log entry",
                  addr, size, uploaded);
    }
}

void BufferCache::WriteDataBuffer(Buffer& buffer, VAddr address, const void* value, u32 num_bytes) {
    WriterTagScope census_tag{*this, 5};
    NoteBatchWrite(address, num_bytes);
    buffer.NoteMainAccess(scheduler.CurrentTick(), true);
    if (g_range_notes) {
        buffer.NoteMainRange(scheduler.CurrentTick(), true, address, num_bytes, 5, 0);
    }
    vk::BufferCopy copy = {
        .srcOffset = 0,
        .dstOffset = buffer.Offset(address),
        .size = num_bytes,
    };
    vk::Buffer src_buffer = staging_buffer.Handle();
    if (num_bytes < StagingBufferSize) {
        const auto [staging, offset] = staging_buffer.Map(num_bytes);
        std::memcpy(staging, value, num_bytes);
        copy.srcOffset = offset;
        staging_buffer.Commit();
    } else {
        // For large one time transfers use a temporary host buffer.
        // RenderDoc can lag quite a bit if the stream buffer is too large.
        Buffer temp_buffer{
            instance, scheduler, MemoryUsage::Upload, 0, vk::BufferUsageFlagBits::eTransferSrc,
            num_bytes};
        src_buffer = temp_buffer.Handle();
        u8* const staging = temp_buffer.mapped_data.data();
        std::memcpy(staging, value, num_bytes);
        scheduler.DeferOperation([buffer = std::move(temp_buffer)]() mutable {});
    }
    scheduler.EndRendering();
    auto command_access = scheduler.AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();
    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .buffer = buffer.Handle(),
        .offset = buffer.Offset(address),
        .size = num_bytes,
    };
    const vk::BufferMemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .buffer = buffer.Handle(),
        .offset = buffer.Offset(address),
        .size = num_bytes,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
    });
    cmdbuf.copyBuffer(src_buffer, buffer.Handle(), copy);
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &post_barrier,
    });
}

void BufferCache::RunGarbageCollector() {
    SCOPE_EXIT {
        ++gc_tick;
    };
    if (instance.CanReportMemoryUsage()) {
        total_used_memory = instance.GetDeviceMemoryUsage();
    }
    {
        // v1.0.2: video memory in players' logs - VMA's device-local usage (the number its budget
        // check compares), its peak, and the driver's process usage, every 60 s.
        VmaBudget budgets[VK_MAX_MEMORY_HEAPS]{};
        vmaGetHeapBudgets(instance.GetAllocator(), budgets);
        const auto& mp = instance.GetMemoryProperties();
        u64 vma_local = 0;
        for (u32 i = 0; i < mp.memoryHeapCount; ++i) {
            if (mp.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
                vma_local += budgets[i].usage;
            }
        }
        vram_peak_vma = std::max(vram_peak_vma, vma_local);
        vram_peak_driver = std::max(vram_peak_driver, total_used_memory);
        const auto now = std::chrono::steady_clock::now();
        // SHADPS4_MEM_CENSUS=1: every 10 s (the census below rides on this line).
        static const bool census_fast = [] {
            const char* v = std::getenv("SHADPS4_MEM_CENSUS");
            return v && (v[0] == '1' || v[0] == '2');
        }();
        if (now - vram_last_log >= std::chrono::seconds{census_fast ? 10 : 60}) {
            vram_last_log = now;
            LOG_INFO(Render_Vulkan,
                     "VideoMemory: VMA video {} MB (peak {} MB); driver process usage {} MB (peak "
                     "{} MB), GC trigger {} MB; {}",
                     vma_local >> 20, vram_peak_vma >> 20, total_used_memory >> 20,
                     vram_peak_driver >> 20, trigger_gc_memory >> 20,
                     Vulkan::DescribeDeviceLocalBudgets(instance.GetAllocator()));
            // SHADPS4_MEM_CENSUS=1 (PERF-PLAN 11, diagnostic): where the memory goes. Buffer and
            // image sizes are guest sizes (host allocations add alignment / tiling: inferred).
            static const bool census = [] {
                const char* v = std::getenv("SHADPS4_MEM_CENSUS");
                return v && (v[0] == '1' || v[0] == '2');
            }();
            if (census) {
                u64 buf_bytes = 0, buf_largest = 0;
                u32 bufs = 0;
                for (const auto& buffer : slot_buffers) {
                    buf_bytes += buffer.SizeBytes();
                    buf_largest = std::max<u64>(buf_largest, buffer.SizeBytes());
                    ++bufs;
                }
                const auto img = texture_cache.MemoryCensus();
                u64 priv = 0, ws = 0;
                Common::ProcessMemory(priv, ws);
                // SHADPS4_MEM_PROBE=1 (once): does freed host-visible upload memory return to
                // Windows? A 2 GB Upload buffer as UploadCopies' large branch makes, filled, freed.
                static bool probed = false;
                if (const char* pv = std::getenv("SHADPS4_MEM_PROBE"); pv && pv[0] == '1' &&
                                                                        !probed) {
                    probed = true;
                    u64 p0 = 0, p1 = 0, p2 = 0, w = 0;
                    Common::ProcessMemory(p0, w);
                    {
                        Buffer probe{instance, scheduler, MemoryUsage::Upload, 0,
                                     vk::BufferUsageFlagBits::eTransferSrc, 2_GB};
                        if (!probe.mapped_data.empty()) {
                            std::memset(probe.mapped_data.data(), 1, probe.mapped_data.size());
                        }
                        Common::ProcessMemory(p1, w);
                    }
                    Common::ProcessMemory(p2, w);
                    LOG_INFO(Render_Vulkan,
                             "MemProbe: private before {} MB, with a filled 2 GB upload buffer {} "
                             "MB, after freeing it {} MB",
                             p0 >> 20, p1 >> 20, p2 >> 20);
                    // (A 2 GB device-local probe, 10-20 s1436: private RAM +2 GB while it
                    // existed - the driver commits system memory for video memory. Removed:
                    // its GPU wait in this context lost the device.)
                }
                const auto cc = Common::WalkCommitted();
                std::string top;
                for (int i = 0; i < 8 && cc.top_bytes[i] != 0; ++i) {
                    top += fmt::format(" {:#x}:{}MB", cc.top_base[i], cc.top_bytes[i] >> 20);
                }
                LOG_INFO(Render_Vulkan,
                         "MemCensus: committed - private in the guest range {} MB, private "
                         "outside {} MB in {} allocations (largest:{}), mapped in the guest range "
                         "{} MB, mapped outside {} MB, modules {} MB | heaps: process heap {} MB, "
                         "{} other heaps {} MB",
                         cc.private_guest >> 20, cc.private_host >> 20, cc.host_blocks, top,
                         cc.mapped_guest >> 20, cc.mapped_host >> 20, cc.image >> 20,
                         cc.default_heap >> 20, cc.heaps - 1, cc.other_heaps >> 20);
                LOG_INFO(Render_Vulkan,
                         "MemCensus: buffers {} MB in {} (largest {} MB) | render targets / depth {} "
                         "MB in {} | textures {} MB in {} (largest image {} MB) | process RAM "
                         "private {} MB, working set {} MB | game quality level {}",
                         buf_bytes >> 20, bufs, buf_largest >> 20, img.targets_bytes >> 20,
                         img.targets, img.textures_bytes >> 20, img.textures,
                         img.largest_bytes >> 20, priv >> 20, ws >> 20,
                         MemoryPatcher::GameQualityLevel());
                // What VMA itself holds in video memory (blocks / live allocations) vs the
                // driver's count for the process (budget usage): the rest is driver-internal
                // (NGX, shader code, swapchain, command memory: inferred).
                u64 vma_blocks = 0, vma_allocs = 0;
                u32 vma_count = 0;
                for (u32 i = 0; i < mp.memoryHeapCount; ++i) {
                    if (mp.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
                        vma_blocks += budgets[i].statistics.blockBytes;
                        vma_allocs += budgets[i].statistics.allocationBytes;
                        vma_count += budgets[i].statistics.allocationCount;
                    }
                }
                LOG_INFO(Render_Vulkan,
                         "MemCensus: video memory - VMA blocks {} MB, VMA allocations {} MB in {}, "
                         "process usage {} MB (outside VMA {} MB)",
                         vma_blocks >> 20, vma_allocs >> 20, vma_count, vma_local >> 20,
                         (vma_local > vma_blocks ? vma_local - vma_blocks : 0) >> 20);
                // SHADPS4_MEM_CENSUS=2: once, at the 6th census (~60 s), every VMA allocation (size,
                // type, usage flags) to vma-stats.json in the log folder.
                static u32 census_n = 0;
                static const bool dump = [] {
                    const char* v = std::getenv("SHADPS4_MEM_CENSUS");
                    return v && v[0] == '2';
                }();
                if (dump && ++census_n == 6) {
                    char* stats = nullptr;
                    vmaBuildStatsString(instance.GetAllocator(), &stats, VK_TRUE);
                    if (stats) {
                        const auto path =
                            Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "vma-stats.json";
                        if (std::FILE* f = std::fopen(path.string().c_str(), "wb")) {
                            std::fwrite(stats, 1, std::strlen(stats), f);
                            std::fclose(f);
                        }
                        vmaFreeStatsString(instance.GetAllocator(), stats);
                        LOG_INFO(Render_Vulkan, "MemCensus: VMA map written to {}", path.string());
                    }
                }
            }
        }
    }
    if (total_used_memory < trigger_gc_memory) {
        return;
    }
    const bool aggressive = total_used_memory >= critical_gc_memory;
    const u64 ticks_to_destroy = std::min<u64>(aggressive ? 80 : 160, gc_tick);
    int max_deletions = aggressive ? 64 : 32;
    const auto clean_up = [&](BufferId buffer_id) {
        if (max_deletions == 0) {
            return;
        }
        --max_deletions;
        // Settle before taking the reference: no in-flight fault job may span this download.
        SettleOverlappingJobs(slot_buffers[buffer_id].CpuAddr(),
                              slot_buffers[buffer_id].SizeBytes());
        Buffer& buffer = slot_buffers[buffer_id];
        // InvalidateMemory(buffer.CpuAddr(), buffer.SizeBytes());
        DownloadBufferMemory<true>(buffer, buffer.CpuAddr(), buffer.SizeBytes(), true);
        DeleteBuffer(buffer_id);
    };
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
    lru_cache.Touch(buffer.LRUId(), gc_tick);
}

void BufferCache::DeleteBuffer(BufferId buffer_id) {
    AmdGpu::MtbCensus::Note(AmdGpu::MtbCensus::BufferDelete);
    Buffer& buffer = slot_buffers[buffer_id];
    Unregister(buffer_id);
    // SHADPS4_READBACK_QUEUE=1/2: a copy on the separate readback queue waits only for its
    // producer tick, not for this deletion tick, so it can still be queued when the main queue
    // passes the deletion and the buffer is freed (review 2026-09-27, use-after-free). Before
    // freeing, wait for every readback-queue copy submitted up to now; each waits on a producer
    // below the deletion tick, which has completed when this deferred operation runs.
    // SHADPS4_FAULT_GUEST_SUBMIT: a copy of this buffer recorded for a guest to submit has no
    // value yet; submit those here first, so the ticks read below cover them.
    CloseGuestSubmitsOver(buffer.CpuAddr(), buffer.SizeBytes());
    const u64 rq_tick = readback_queue && !readback_queue->InQueueOrder()
                            ? readback_queue->Semaphore().CurrentTick() - 1
                            : 0;
    // The fast lane likewise waits only for completed producers, or (q51) for a submitted main
    // value below the current tick - below this deletion's wait, so no cycle.
    const u64 lane_tick = fast_lane ? fast_lane->Semaphore().CurrentTick() - 1 : 0;
    // r18: the arena retry queue's copies wait for submitted producers too.
    const u64 retry_tick = retry_lane ? retry_lane->Semaphore().CurrentTick() - 1 : 0;
    Common::UniqueFunction<void> destroy = [this, buffer_id, rq_tick, lane_tick, retry_tick] {
        if (rq_tick != 0) {
            readback_queue->Semaphore().Wait(rq_tick);
        }
        if (lane_tick != 0) {
            fast_lane->Semaphore().Wait(lane_tick);
        }
        if (retry_tick != 0) {
            retry_lane->Semaphore().Wait(retry_tick);
        }
        slot_buffers.erase(buffer_id);
    };
    if (DeleteAtNextDraw()) {
        // r13 (2026-10-14, the rooftop load crash): a draw can capture this buffer's handle, then
        // join it away here, then flush mid-draw (a stream-buffer wrap, a download) and queue its
        // push into the NEXT batch; deferring to today's tick freed the buffer while that push
        // still waited for the recorder (GPU-AV: invalid VkBuffer in vkCmdPushDescriptorSetKHR;
        // the binding report: the fault lies in the just-unbound mesh buffer). Park the destroy
        // until the next draw / dispatch entry (CommitDeletes), when every packet of this draw
        // is queued, and defer it to the tick current then.
        std::scoped_lock lk{pending_deletes_mutex};
        pending_deletes.push_back(std::move(destroy));
    } else {
        scheduler.DeferOperation(std::move(destroy));
    }
    buffer.is_deleted = true;
}

bool BufferCache::DeleteAtNextDraw() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_DELETE_AT_NEXT_DRAW");
        return !(e && e[0] == '0');
    }();
    return on;
}

void BufferCache::CommitDeletes() {
    std::vector<Common::UniqueFunction<void>> todo;
    {
        std::scoped_lock lk{pending_deletes_mutex};
        if (pending_deletes.empty()) {
            return;
        }
        todo.swap(pending_deletes);
    }
    for (auto& f : todo) {
        scheduler.DeferOperation(std::move(f));
    }
}

} // namespace VideoCore
