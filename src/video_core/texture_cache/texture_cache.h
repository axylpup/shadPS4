// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include "common/switch_mutex.h"
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <thread>
#include <unordered_set>
#include <boost/container/small_vector.hpp>
#include <queue>
#include <tsl/robin_map.h>

#include "common/lru_cache.h"
#include "common/slot_vector.h"
#include "shader_recompiler/resource.h"
#include "video_core/multi_level_page_table.h"
#include "video_core/texture_cache/blit_helper.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/sampler.h"
#include "video_core/texture_cache/tile_manager.h"

namespace AmdGpu {
struct Liverpool;
}

namespace VideoCore {

class BufferCache;
class PageManager;

class TextureCache {
    // Default values for garbage collection
    static constexpr s64 DEFAULT_PRESSURE_GC_MEMORY = 1_GB + 512_MB;
    static constexpr s64 DEFAULT_CRITICAL_GC_MEMORY = 3_GB;
    static constexpr s64 TARGET_GC_THRESHOLD = 8_GB;
    // Cards up to this size keep a fixed headroom free for buffers/staging (see RefreshGcLimits).
    static constexpr s64 SMALL_CARD_MEMORY = 12_GB;

    using ImageIds = boost::container::small_vector<ImageId, 16>;

    struct Traits {
        using Entry = ImageIds;
        static constexpr size_t AddressSpaceBits = 40;
        static constexpr size_t FirstLevelBits = 10;
        static constexpr size_t PageBits = 20;
    };
    using PageTable = MultiLevelPageTable<Traits>;

public:
    enum class BindingType : u32 {
        Texture,
        Storage,
        RenderTarget,
        DepthTarget,
        VideoOut,
    };

    struct ImageDesc {
        ImageInfo info;
        ImageViewInfo view_info;
        BindingType type{BindingType::Texture};

        ImageDesc() = default;
        ImageDesc(const AmdGpu::Image& image, const Shader::ImageResource& desc)
            : info{image, desc}, view_info{image, desc},
              type{desc.is_written ? BindingType::Storage : BindingType::Texture} {}
        ImageDesc(const AmdGpu::ColorBuffer& buffer, AmdGpu::CbDbExtent hint)
            : info{buffer, hint}, view_info{buffer}, type{BindingType::RenderTarget} {}
        ImageDesc(const AmdGpu::DepthBuffer& buffer, AmdGpu::DepthView view,
                  AmdGpu::DepthControl ctl, VAddr htile_address, AmdGpu::CbDbExtent hint,
                  bool write_buffer = false)
            : info{buffer, view.NumSlices(), htile_address, hint, write_buffer},
              view_info{buffer, view, ctl}, type{BindingType::DepthTarget} {}
        ImageDesc(const Libraries::VideoOut::BufferAttributeGroup& group, VAddr cpu_address)
            : info{group, cpu_address}, type{BindingType::VideoOut} {}
    };

public:
    TextureCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                 AmdGpu::Liverpool* liverpool, BufferCache& buffer_cache, PageManager& tracker);
    ~TextureCache();

    TileManager& GetTileManager() noexcept {
        return tile_manager;
    }

    /// Invalidates any image in the logical page range.
    void InvalidateMemory(VAddr addr, size_t size);

    /// Marks an image as dirty if it exists at the provided address.
    void InvalidateMemoryFromGPU(VAddr address, size_t max_size);

    /// SHADPS4_BUFFER_IMAGE_ALIAS_DIAG=1 (diagnostic): a shader-written buffer range overlapping
    /// cached images. "handled" is the formatted same-base case InvalidateMemoryFromGPU marks
    /// GpuDirty; anything else ("missed") leaves the cached image as it was.
    static bool BufferImageAliasDiag();
    void DiagBufferWriteOverImages(VAddr address, size_t size, bool formatted);

    /// Evicts any images that overlap the unmapped range.
    void UnmapMemory(VAddr cpu_addr, size_t size);

    /// Schedules a copy of pending images for download back to CPU memory.
    void ProcessDownloadImages();

    /// Retrieves the image handle of the image with the provided attributes.
    [[nodiscard]] ImageId FindImage(ImageDesc& desc, bool exact_fmt = false);

    /// Retrieves image whose address matches provided
    [[nodiscard]] ImageId FindImageFromRange(VAddr address, size_t size, bool ensure_valid = true);

    /// SHADPS4_DLSS: the image at exactly `address` with this size, the most recently used one
    /// when several alias it (0x26b200000 holds the 1920x1080 scene depth and, after the AA, the
    /// minimap's 768x768 depth: FindImageFromRange's byte-size test matches neither).
    [[nodiscard]] ImageId FindImageAtSize(VAddr address, u32 width, u32 height);

    /// SHADPS4_DRAW_REPEAT_HEAP: any image whose guest address is exactly `address` (the
    /// candidates FindImageFromRange considers, without its size test or log).
    [[nodiscard]] bool AnyImageAt(VAddr address);

    /// Retrieves an image view with the properties of the specified image id.
    [[nodiscard]] ImageView& FindTexture(ImageId image_id, const ImageDesc& desc);

    /// Retrieves the render target with specified properties
    [[nodiscard]] ImageView& FindRenderTarget(ImageId image_id, const ImageDesc& desc);

    /// Retrieves the depth target with specified properties
    [[nodiscard]] ImageView& FindDepthTarget(ImageId image_id, const ImageDesc& desc);

    /// Updates image contents if it was modified by CPU.
    void UpdateImage(ImageId image_id);

    /// Resolves overlap between existing cache image and pending merged image
    [[nodiscard]] std::tuple<ImageId, int, int> ResolveOverlap(const ImageInfo& info,
                                                               BindingType binding,
                                                               ImageId cache_img_id,
                                                               ImageId merged_image_id);

    /// Resolves depth overlap and either re-creates the image or returns existing one
    /// SHADPS4_STENCIL_SHADOW=1: when overlap resolution replaces a depth-stencil image with
    /// one without stencil (depth read or written as R32, depth-only target), the stencil plane
    /// is saved to an R8 shadow keyed by guest address and restored into the next depth-stencil
    /// image recreated there. Without it every hop drops the stencil (PS4 Pro checkerboard mask).
    void SaveStencilShadow(ImageId depth_id);
    void RestoreStencilShadow(ImageId depth_id);
    void CopyStencilAspect(ImageId src_id, ImageId dst_id);
    std::unordered_map<VAddr, ImageId> stencil_shadows;
    // SHADPS4_STENCIL_PLANE_MERGE=1: guest stencil addresses of bound depth targets. A storage
    // write to the R8 placeholder at one of them (Pro depth+stencil resolve compute) is merged
    // into the depth's stencil aspect when that depth is next bound as a depth target.
    std::unordered_set<VAddr> stencil_plane_addrs;

public:
    static bool StencilPlaneMerge() {
        static const bool enabled = [] {
            const char* value = std::getenv("SHADPS4_STENCIL_PLANE_MERGE");
            return value && value[0] == '1';
        }();
        return enabled;
    }
    /// Called by the storage-image bind path for written images with no live depth association.
    void NoteStorageWrite(Image& image) {
        if (StencilPlaneMerge() && !image.info.props.is_depth &&
            stencil_plane_addrs.contains(image.info.guest_address)) {
            image.stencil_plane_written = true;
        }
    }

public:

    [[nodiscard]] ImageId ResolveDepthOverlap(const ImageInfo& requested_info, BindingType binding,
                                              ImageId cache_img_id);

    /// Creates a new image with provided image info and copies subresources from image_id
    [[nodiscard]] ImageId ExpandImage(const ImageInfo& info, ImageId image_id);

    /// Reuploads image contents.
    void RefreshImage(Image& image);

    /// SHADPS4_TEX_REPLACE: the user's pictures over parts of a just-uploaded texture.
    void ApplyTexReplace(Image& image);

    /// Retrieves the sampler that matches the provided S# descriptor. When native_index is
    /// provided, it also receives the sampler's descriptor heap slot; the sampler object may
    /// move in its map, so the slot is resolved here under the same lock.
    [[nodiscard]] vk::Sampler GetSampler(const AmdGpu::Sampler& sampler,
                                         AmdGpu::BorderColorBuffer border_color_base,
                                         u32* native_index = nullptr);

    /// Retrieves the image with the specified id.
    [[nodiscard]] Image& GetImage(ImageId id) {
        auto& image = slot_images[id];
        TouchImage(image);
        return image;
    }

    /// SHADPS4_BIND_MEMO_STATS: the image without the LRU touch GetImage does.
    /// q8: the table's bulk touch checks an id before PeekImage.
    [[nodiscard]] bool IsImageAllocated(ImageId id) noexcept {
        return id && slot_images.is_allocated(id);
    }
    [[nodiscard]] Image& PeekImage(ImageId id) noexcept {
        return slot_images[id];
    }

    /// SHADPS4_NATIVE_BINDLESS (back thread): a T# -> image cache entry still names the image
    /// FindImage would return - PreparedEntryValid's rule (allocated, same uid, registered, not
    /// picked; same structure generation, or an exact-range match whose overlap epoch is
    /// unchanged) - and the image needs no re-validation on a bind (clean, no depth association).
    [[nodiscard]] bool BindlessEntryValid(ImageId id, u64 uid, u64 generation, bool local,
                                          u64 overlap_epoch) {
        if (!id || !uid || !slot_images.is_allocated(id)) {
            return false;
        }
        const Image& image = slot_images[id];
        const auto flags = std::atomic_ref<ImageFlagBits>{const_cast<ImageFlagBits&>(image.flags)}
                               .load(std::memory_order_relaxed);
        if (image.image_uid != uid || False(flags & ImageFlagBits::Registered) ||
            True(flags & (ImageFlagBits::Picked | ImageFlagBits::Dirty)) || image.depth_id) {
            return false;
        }
        if (struct_generation.load(std::memory_order_acquire) == generation) {
            return true;
        }
        return local && std::atomic_ref<u64>{const_cast<u64&>(image.overlap_epoch)}.load(
                            std::memory_order_relaxed) == overlap_epoch;
    }

    /// SHADPS4_BIND_MEMO_STATS: the UpdateImageLockFree predicate (registered, clean, tracked
    /// over its whole guest range) with relaxed loads, no LRU touch and no counters. CP thread.
    [[nodiscard]] bool ImageUpToDateRelaxed(ImageId id) noexcept {
        Image& image = slot_images[id];
        const auto flags =
            std::atomic_ref<ImageFlagBits>{image.flags}.load(std::memory_order_relaxed);
        const VAddr begin = image.info.guest_address;
        const VAddr end = begin + image.info.guest_size;
        return True(flags & ImageFlagBits::Registered) && False(flags & ImageFlagBits::Dirty) &&
               std::atomic_ref<VAddr>{image.track_addr}.load(std::memory_order_relaxed) == begin &&
               std::atomic_ref<VAddr>{image.track_addr_end}.load(std::memory_order_relaxed) == end;
    }

    /// SHADPS4_BIND_MEMO_STATS: bumped by every InvalidatePreparedImages call (image register,
    /// unregister, delete, deferred erase), whether or not the prepared-image memo is enabled.
    /// l4e: the GC tick and the age tick the LRU touches write (material memo touch skip).
    [[nodiscard]] u64 GcTickNow() const noexcept {
        return gc_tick;
    }
    [[nodiscard]] u64 AgeTickNow() const {
        return AgeTick();
    }
    [[nodiscard]] u64 StructGeneration() const noexcept {
        return struct_generation.load(std::memory_order_acquire);
    }

    /// SHADPS4_BIND_MEMO_STATS: bumped for every sampler the sampler GC destroys.
    [[nodiscard]] u64 SamplerGeneration() const noexcept {
        return sampler_generation.load(std::memory_order_acquire);
    }

    /// SHADPS4_TEX_BIND_MEMO: protects a binding-memo hit's validation and LRU touches the way
    /// the FindImage/UpdateImage fast paths are protected: the lock-free handshake when
    /// SHADPS4_TEX_LOCKFREE_HIT is on (Refused while a foreign CacheLock holder is active; the
    /// caller then runs the full path), otherwise the cache mutex. Nothing inside the section may
    /// take the cache lock.
    /// SHADPS4_TEX_MEMO_LOCAL_GEN: the slot still holds an image (not erased). Inside a memo section.
    [[nodiscard]] bool ImageAllocated(ImageId id) const {
        return slot_images.is_allocated(id);
    }
    enum class MemoSection : u8 { Refused, LockFree, Locked };
    [[nodiscard]] MemoSection EnterMemoSection();
    void LeaveMemoSection(MemoSection section);

    /// SHADPS4_TEX_BIND_MEMO: FindImage's perfect-match predicate (non-exact format) for `id`
    /// against the requested info, including its "enough resources" check. Bindings resolved by
    /// overlap resolution fail it and are never memoized.
    [[nodiscard]] bool IsPerfectMatch(const ImageInfo& requested, ImageId id);

    /// SHADPS4_TEX_BIND_MEMO: what FindImage/UpdateImage do to a reused image on their fast
    /// paths (access tick and LRU touch). Call inside a memo section.
    void TouchImageAccess(Image& image);

    /// SHADPS4_TEX_BIND_MEMO: LRU id of the cached sampler for this S#, or ~0 when absent.
    [[nodiscard]] size_t SamplerLruId(const AmdGpu::Sampler& sampler);

    /// SHADPS4_TEX_BIND_MEMO: the LRU touch GetSampler does on a hit, for every id, if no
    /// sampler has been destroyed since `generation` (else returns false and touches nothing).
    [[nodiscard]] bool TouchSamplers(std::span<const size_t> lru_ids, u64 generation);

    /// Retrieves the image view with the specified id.
    [[nodiscard]] ImageView& GetImageView(ImageId id) {
        return slot_image_views[id];
    }

    /// Diagnostic (SHADPS4_DUMP_IMAGES): writes mip 0 / layer 0 of every registered image whose
    /// guest address is listed to `<log dir>/imgdump_<tag>_<addr>_<w>x<h>_<format>_<uid>.raw`
    /// (tightly packed texels, first aspect). Blocks on the GPU; CP thread only.
    void DebugDumpImages(const std::vector<VAddr>& addresses, u32 tag);

    /// SHADPS4_STENCIL_STORAGE: the registered depth image at `address` whose extent matches
    /// `size`. One address can hold several depth surfaces (PS4 Pro: 1600x1800 scene depth and a
    /// 3200x1800 reconstruction depth-stencil share a base address).
    ImageId FindDepthImageBySize(VAddr address, const Extent3D& size) {
        std::scoped_lock lock{mutex};
        ImageId found{};
        if (const auto it = page_table.find(address >> Traits::PageBits); it != nullptr) {
            for (const ImageId id : *it) {
                const Image& image = slot_images[id];
                if (image.info.guest_address == address && image.info.props.is_depth &&
                    image.info.size == size && True(image.flags & ImageFlagBits::Registered)) {
                    found = id;
                }
            }
        }
        return found;
    }

    /// SHADPS4_STENCIL_STORAGE: an image outside the cache's page table, LRU and page tracking,
    /// never refreshed from guest memory. The caller owns it for the rest of the run.
    ImageId InsertScratchImage(const ImageInfo& info) {
        CacheLock lock{*this};
        const ImageId id =
            slot_images.insert(instance, scheduler, blit_helper, slot_image_views, info);
        slot_images[id].flags = ImageFlagBits::Empty;
        return id;
    }

    /// Get the associated depth stencil image if it is still valid.
    ImageId GetAssociatedDepth(Image& image) {
        if (!image.depth_id) {
            return {};
        }
        if (slot_images.is_allocated(image.depth_id)) {
            auto& depth_image = slot_images[image.depth_id];
            if (depth_image.image_uid == image.depth_uid &&
                depth_image.flags & ImageFlagBits::Registered) {
                return image.depth_id;
            }
        }
        // The linked depth image is no longer valid, disassociate it.
        image.DisassociateDepth();
        return {};
    }

    /// Returns true if the specified address is a metadata surface.
    bool IsMeta(VAddr address) const {
        return surface_metas.contains(address);
    }

    /// Returns true if a slice of the specified metadata surface has been cleared.
    bool IsMetaCleared(VAddr address, u32 slice) const {
        const auto& it = surface_metas.find(address);
        if (it != surface_metas.end()) {
            if (AmdGpu::MtbConflict::g_enabled) [[unlikely]] {
                AmdGpu::MtbConflict::Touch(it.value().cf, AmdGpu::MtbConflict::Meta);
            }
            return it.value().clear_mask & (1u << slice);
        }
        return false;
    }

    /// Clears all slices of the specified metadata surface.
    bool ClearMeta(VAddr address) {
        auto it = surface_metas.find(address);
        if (it != surface_metas.end()) {
            it.value().clear_mask = u32(-1);
            ++meta_clear_epoch;
            if (AmdGpu::MtbConflict::g_enabled) [[unlikely]] {
                AmdGpu::MtbConflict::Change(it.value().cf, AmdGpu::MtbConflict::Meta);
            }
            return true;
        }
        return false;
    }

    /// SHADPS4_RENDER_STATE_MEMO: bumped whenever a clear bit of a metadata surface is set
    /// (ClearMeta, TouchMeta with is_clear). While it stands still, a slice that read "not
    /// cleared" still reads so. CP thread only, like surface_metas.
    [[nodiscard]] u64 MetaClearEpoch() const noexcept {
        return meta_clear_epoch;
    }

    /// SHADPS4_RENDER_STATE_MEMO: what FindRenderTarget / FindDepthTarget do to a target that
    /// was already resolved by them with the same descriptor and whose image, backing and
    /// registered metadata are unchanged since: the GPU-modified flag, the usage bit, the
    /// linear-readback registration and UpdateImage. No meta registration, stencil link or view
    /// lookup (their results are unchanged by construction; see the memo's validity checks).
    void NoteTargetUse(ImageId image_id, bool is_depth);

    /// Updates the state of a slice of the specified metadata surface.
    bool TouchMeta(VAddr address, u32 slice, bool is_clear) {
        auto it = surface_metas.find(address);
        if (it != surface_metas.end()) {
            if (AmdGpu::MtbConflict::g_enabled) [[unlikely]] {
                AmdGpu::MtbConflict::Touch(it.value().cf, AmdGpu::MtbConflict::Meta);
            }
            const s32 old_mask = it.value().clear_mask;
            if (is_clear) {
                it.value().clear_mask |= 1u << slice;
                ++meta_clear_epoch;
            } else {
                it.value().clear_mask &= ~(1u << slice);
            }
            if (AmdGpu::MtbConflict::g_enabled && it.value().clear_mask != old_mask) [[unlikely]] {
                AmdGpu::MtbConflict::Change(it.value().cf, AmdGpu::MtbConflict::Meta);
            }
            return true;
        }
        return false;
    }

    /// Runs the garbage collector.
    void RunGarbageCollector();

    /// Recomputes the collector limits from the driver's current budget (VK_EXT_memory_budget).
    void RefreshGcLimits();
    /// True when the texture cache, not the whole device, is measured against the budget.
    bool ImageBudgetMode() const;

    /// SHADPS4_MEM_CENSUS (PERF-PLAN 11): guest bytes of the cached images by kind - render
    /// targets / depth (attachment usage) and the rest (textures). Takes the cache mutex.
    struct ImageCensus {
        u64 targets_bytes{}, textures_bytes{}, largest_bytes{};
        u32 targets{}, textures{};
    };
    ImageCensus MemoryCensus() {
        std::scoped_lock lock{mutex};
        ImageCensus c{};
        for (const auto& image : slot_images) {
            const u64 bytes = image.info.guest_size;
            const bool target =
                bool(image.usage_flags & (vk::ImageUsageFlagBits::eColorAttachment |
                                          vk::ImageUsageFlagBits::eDepthStencilAttachment));
            (target ? c.targets_bytes : c.textures_bytes) += bytes;
            ++(target ? c.targets : c.textures);
            c.largest_bytes = std::max(c.largest_bytes, bytes);
        }
        return c;
    }

    template <typename Func>
    void ForEachImageInRegion(VAddr cpu_addr, size_t size, Func&& func) {
        using FuncReturn = typename std::invoke_result<Func, ImageId, Image&>::type;
        static constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
        ImageIds images;
        ForEachPage(cpu_addr, size, [this, &images, cpu_addr, size, func](u64 page) {
            const auto it = page_table.find(page);
            if (it == nullptr) {
                if constexpr (BOOL_BREAK) {
                    return false;
                } else {
                    return;
                }
            }
            for (const ImageId image_id : *it) {
                Image& image = slot_images[image_id];
                if (image.flags & ImageFlagBits::Picked) {
                    continue;
                }
                if (!image.Overlaps(cpu_addr, size)) {
                    continue;
                }
                image.flags |= ImageFlagBits::Picked;
                images.push_back(image_id);
                if constexpr (BOOL_BREAK) {
                    if (func(image_id, image)) {
                        return true;
                    }
                } else {
                    func(image_id, image);
                }
            }
            if constexpr (BOOL_BREAK) {
                return false;
            }
        });
        for (const ImageId image_id : images) {
            slot_images[image_id].flags &= ~ImageFlagBits::Picked;
        }
    }

private:
    /// Iterate over all page indices in a range
    template <typename Func>
    static void ForEachPage(PAddr addr, size_t size, Func&& func) {
        static constexpr bool RETURNS_BOOL = std::is_same_v<std::invoke_result<Func, u64>, bool>;
        const u64 page_end = (addr + size - 1) >> Traits::PageBits;
        for (u64 page = addr >> Traits::PageBits; page <= page_end; ++page) {
            if constexpr (RETURNS_BOOL) {
                if (func(page)) {
                    break;
                }
            } else {
                func(page);
            }
        }
    }

    /// Copies image memory back to CPU.
    void DownloadImageMemory(ImageId image_id, bool sync = false);


    /// Thread function for copying downloaded images out to CPU memory.
    void DownloadedImagesThread(const std::stop_token& token);

    /// Create an image from the given parameters
    [[nodiscard]] ImageId InsertImage(const ImageInfo& info, VAddr cpu_addr);

    /// Register image in the page table
    void RegisterImage(ImageId image);

    /// Unregister image from the page table
    void UnregisterImage(ImageId image);

    /// Track CPU reads and writes for image
    void TrackImage(ImageId image_id);
    void TrackImageHead(ImageId image_id);
    void TrackImageTail(ImageId image_id);

    /// Stop tracking CPU reads and writes for image
    void UntrackImage(ImageId image_id);
    void UntrackImageHead(ImageId image_id);
    void UntrackImageTail(ImageId image_id);

    void MarkAsMaybeDirty(ImageId image_id, Image& image);

    /// Removes the image and any views/surface metas that reference it.
    void DeleteImage(ImageId image_id);

    /// Clock of Image::tick_accessed_last and of ResolveOverlap's "safe to delete" age: the
    /// scheduler tick minus the cadence-only submissions (Scheduler::NoteCadenceFlush), so the
    /// eviction age does not shrink when batches are submitted more often.
    /// SHADPS4_TEX_AGE_RAW=1 restores the plain scheduler tick.
    u64 AgeTick() const;

    /// True if another registered image over [guest_address, +guest_size) uses `meta_addr` as
    /// one of its metadata surfaces (images that alias the same guest memory share them).
    bool MetaUsedByOtherImage(VAddr meta_addr, VAddr guest_address, u64 guest_size);

    /// Touch the image in the LRU cache.
    void TouchImage(const Image& image);

    void FreeImage(ImageId image_id) {
        UntrackImage(image_id);
        UnregisterImage(image_id);
        DeleteImage(image_id);
    }

    void GarbageCollectImages();
    /// The image was touched at GC tick `tick` or later (its own gc_touch).
    bool UsedSince(const Image& image, u64 tick) const;
    /// Live screen percentage: drops the old size's targets the frame did not use.
    void PurgeOldRenderSize(u32 old_w, u32 old_h, u32 new_w, u32 new_h);
    void GarbageCollectSamplers();

    // Only the surviving perfect-match path is memoized. These are all its selection inputs;
    // view_info is left live (that path never changes its range). Binding type is conservative.
    struct PreparedImageKey {
        VAddr address{};
        u32 guest_size{};
        Extent3D size{};
        AmdGpu::ImageType type{};
        vk::Format format{};
        SubresourceExtent resources{};
        BindingType binding{};
        bool exact_fmt{};

        bool operator==(const PreparedImageKey&) const = default;
    };
    struct PreparedImage {
        PreparedImageKey key{};
        ImageId id{};
        u64 uid{};
        u64 generation{};
        /// SHADPS4_PREPARED_LOCAL: the image's overlap_epoch when recorded. A perfect match covers
        /// exactly the lookup range, and every register / unregister of another image on one of
        /// its pages bumps that image's epoch, so an unchanged epoch keeps the entry exact even
        /// after the global generation moved (texture streaming moves it constantly).
        u64 overlap_epoch{};
        bool local{};
    };
    struct PreparedImages {
        static constexpr size_t Capacity = 4096;
        std::array<PreparedImage, Capacity> entries{};
        // Atomic because deferred slot reclamation does not take the lookup mutex. This does
        // not replace the existing page-table/slot synchronization or authorize concurrent use.
        std::atomic<u64> generation{1};
        bool verify{};
        bool local{}; ///< SHADPS4_PREPARED_LOCAL=1
        u64 calls{}, hits{}, fallbacks{}, verified{}; // FindImage mutex protects entries/counters.
        u64 local_hits{}; ///< hits the global generation test would have missed
    };

    /// True when `memo` (key already equal) still names a registered, unpicked image with the
    /// recorded uid, and either the global generation or (SHADPS4_PREPARED_LOCAL) the image's
    /// overlap epoch is unchanged. `local_hit` reports a hit only the epoch test allowed.
    [[nodiscard]] bool PreparedEntryValid(const PreparedImage& memo, u64 generation,
                                          bool& local_hit);

    /// Memo slot for a lookup, or nullptr when the lookup is not eligible for memoization.
    PreparedImage* PreparedSlot(const ImageInfo& info, BindingType type, bool exact_fmt,
                                PreparedImageKey& key);

    /**
     * SHADPS4_TEX_LOCKFREE_HIT=1: the CP thread answers FindImage memo hits and no-op UpdateImage
     * calls without the cache mutex (~30k lock pairs a frame, ~9% of CP time on a16).
     *
     * Who else touches the cache: guest write-fault handlers (InvalidateMemory) only change an
     * image's flags, tracked range and hash under the mutex - the fast paths read those with
     * relaxed loads, and a fault landing just after the read is the same as one landing just
     * after the locked version. Structural changes (slot vector, page table, LRU, memo) are made
     * by the CP itself, except for off-CP callers such as video-out buffer registration and guest
     * unmaps. Those take CacheLock, which first raises `lf_foreign`, forces a serializing
     * instruction on every core (FlushProcessWriteBuffers) and waits until the CP is outside its
     * fast path; the CP marks `lf_cp_in_fast` with a plain store and then checks `lf_foreign`,
     * falling back to the locked path when set (asymmetric Dekker handshake).
     */
    class CacheLock {
    public:
        explicit CacheLock(TextureCache& cache);
        ~CacheLock();
        CacheLock(const CacheLock&) = delete;
        CacheLock& operator=(const CacheLock&) = delete;

    private:
        TextureCache& cache;
        bool foreign;
    };
    ImageId FindImageLockFree(const ImageDesc& desc, bool exact_fmt);
    bool UpdateImageLockFree(ImageId image_id);
    void CountLockFree(bool find, bool hit);
    bool lockfree_hit{};
    alignas(64) std::atomic<u32> lf_cp_in_fast{0};
    alignas(64) std::atomic<u32> lf_foreign{0};
    u64 lf_find_calls{}, lf_find_fast{}, lf_update_calls{}, lf_update_fast{};

    // Any future in-place change of guest range, size, type, format or resources must invalidate
    // here under the same synchronization as the mapping change. Currently those are immutable:
    // expansion/depth replacement creates and registers a new Image instead.
    void InvalidatePreparedImages();

private:
    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    BufferCache& buffer_cache;
    PageManager& tracker;
    BlitHelper blit_helper;
    TileManager tile_manager;
    Common::SlotVector<Image, Common::StableSlotCapacity> slot_images;
    Common::SlotVector<ImageView, Common::StableSlotCapacity> slot_image_views;
    tsl::robin_map<u64, Sampler> samplers;
    std::unordered_set<ImageId> download_images;
    u64 total_used_memory = 0;
    u64 trigger_gc_memory = 0;
    u64 pressure_gc_memory = 0;
    u64 critical_gc_memory = 0;
    /// Bytes of the registered images (total_used_memory is the whole device's usage).
    u64 image_bytes = 0;
    /// Limits computed at startup, before RefreshGcLimits applies the budget and the headroom.
    u64 base_trigger_gc_memory = 0;
    u64 base_pressure_gc_memory = 0;
    u64 base_critical_gc_memory = 0;
    /// The card's budget at startup; the fallback when the driver cannot report the current one.
    u64 card_budget_memory = 0;
    /// SHADPS4_VRAM_BUDGET_MB: fixed texture budget, 0 when unset.
    u64 forced_gc_budget = 0;
    /// SHADPS4_VRAM_AUTO=1: the texture budget follows the device (RefreshGcLimits).
    bool auto_gc_budget = false;
    /// The texture budget the automatic mode computed last (0 outside it).
    u64 auto_budget_memory = 0;
    /// Device usage above which the collector runs critical whatever the image bytes are.
    u64 device_guard_memory = 0;
    u64 total_used_samplers = 0;
    u64 trigger_gc_samplers = 0;
    u64 pressure_gc_samplers = 0;
    u64 critical_gc_samplers = 0;
    u64 gc_tick = 0;
    Common::LeastRecentlyUsedCache<ImageId, u64> lru_cache;
    Common::LeastRecentlyUsedCache<u64, u64> sampler_lru_cache;
    bool readback_linear_images;
    PageTable page_table;
    /// Registered images per page_table page, maintained under `mutex`, read without it by the
    /// guest write-fault pre-check in InvalidateMemory (SHADPS4_TEX_INVALIDATE_PRECHECK=1).
    std::unique_ptr<std::atomic<u32>[]> page_image_counts;
    // SHADPS4_FAST_TEX_MUTEX=1: inline futex locks (measured 2026-09-24: lock/unlock of these
    // two was ~9% of the CP's wall time at ~30k acquisitions per frame).
    Common::SwitchMutex<Common::FastTexMutexFlag> mutex;
    Common::SwitchMutex<Common::FastTexMutexFlag> samplers_mutex;
    std::mutex download_images_mutex;
    std::unique_ptr<PreparedImages> prepared_images; // Allocated only when explicitly enabled.
    // SHADPS4_BIND_MEMO_STATS: see StructGeneration and SamplerGeneration.
    std::atomic<u64> struct_generation{0};
    std::atomic<u64> sampler_generation{0};
    struct MetaDataInfo {
        enum class Type {
            CMask,
            FMask,
            HTile,
        };
        Type type;
        s32 clear_mask = -1;
        /// SHADPS4_MTB_CONFLICT (diagnostic): batch of the last clear-bit change / draw touch.
        mutable AmdGpu::MtbConflict::Stamp cf{};
    };
    tsl::robin_map<VAddr, MetaDataInfo> surface_metas;
    u64 meta_clear_epoch{}; // SHADPS4_RENDER_STATE_MEMO, see MetaClearEpoch
};

} // namespace VideoCore
