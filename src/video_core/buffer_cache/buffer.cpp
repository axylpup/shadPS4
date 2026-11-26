// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/alignment.h"
#include "common/assert.h"
#include "common/release_mode.h"
#include <immintrin.h>
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_memory_budget.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

#include <vk_mem_alloc.h>

namespace VideoCore {

std::string_view BufferTypeName(MemoryUsage type) {
    switch (type) {
    case MemoryUsage::Upload:
        return "Upload";
    case MemoryUsage::Download:
        return "Download";
    case MemoryUsage::Stream:
        return "Stream";
    case MemoryUsage::DeviceLocal:
        return "DeviceLocal";
    default:
        return "Invalid";
    }
}

[[nodiscard]] VkMemoryPropertyFlags MemoryUsagePreferredVmaFlags(MemoryUsage usage) {
    return usage != MemoryUsage::DeviceLocal ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                             : VkMemoryPropertyFlagBits{};
}

[[nodiscard]] VmaAllocationCreateFlags MemoryUsageVmaFlags(MemoryUsage usage) {
    switch (usage) {
    case MemoryUsage::Upload:
    case MemoryUsage::Stream:
        return VMA_ALLOCATION_CREATE_MAPPED_BIT |
               VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    case MemoryUsage::Download:
        return VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    case MemoryUsage::DeviceLocal:
        return {};
    }
    return {};
}

[[nodiscard]] VmaMemoryUsage MemoryUsageVma(MemoryUsage usage) {
    switch (usage) {
    case MemoryUsage::DeviceLocal:
    case MemoryUsage::Stream:
        return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    case MemoryUsage::Upload:
    case MemoryUsage::Download:
        return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    }
    return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
}

static thread_local bool t_require_video = false;

RequireVideoMemory::RequireVideoMemory() {
    t_require_video = true;
}

RequireVideoMemory::~RequireVideoMemory() {
    t_require_video = false;
}

static thread_local bool t_prefer_host = false;

PreferHostMemory::PreferHostMemory() {
    t_prefer_host = true;
}

PreferHostMemory::~PreferHostMemory() {
    t_prefer_host = false;
}

/// RequireVideoMemory: device-local memory types only, VMA's budget estimate skipped.
static void RequireVideo(VmaAllocationCreateInfo& ci) {
    ci.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
    ci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
}

UniqueBuffer::UniqueBuffer(vk::Device device_, VmaAllocator allocator_)
    : device{device_}, allocator{allocator_} {}

UniqueBuffer::~UniqueBuffer() {
    if (buffer) {
        vmaDestroyBuffer(allocator, buffer, allocation);
    }
}

UniqueBuffer& UniqueBuffer::operator=(UniqueBuffer&& other) noexcept {
    if (this != &other) {
        if (buffer) {
            vmaDestroyBuffer(allocator, buffer, allocation);
        }
        device = std::exchange(other.device, VK_NULL_HANDLE);
        allocator = std::exchange(other.allocator, VK_NULL_HANDLE);
        allocation = std::exchange(other.allocation, VK_NULL_HANDLE);
        buffer = std::exchange(other.buffer, VK_NULL_HANDLE);
        bda_addr = std::exchange(other.bda_addr, 0);
    }
    return *this;
}

void UniqueBuffer::Create(const vk::BufferCreateInfo& buffer_ci, MemoryUsage usage,
                          VmaAllocationInfo* out_alloc_info) {
    const bool with_bda = bool(buffer_ci.usage & vk::BufferUsageFlagBits::eShaderDeviceAddress);
    const VmaAllocationCreateFlags bda_flag =
        with_bda ? VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT : 0;
    VmaAllocationCreateInfo alloc_ci = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | bda_flag | MemoryUsageVmaFlags(usage),
        .usage = MemoryUsageVma(usage),
        .requiredFlags = 0,
        .preferredFlags = MemoryUsagePreferredVmaFlags(usage),
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    const VkBufferCreateInfo buffer_ci_unsafe = static_cast<VkBufferCreateInfo>(buffer_ci);
    VkBuffer unsafe_buffer{};
    VkResult result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    if (t_require_video && usage == MemoryUsage::DeviceLocal) {
        VmaAllocationCreateInfo video_ci = alloc_ci;
        RequireVideo(video_ci);
        result = vmaCreateBuffer(allocator, &buffer_ci_unsafe, &video_ci, &unsafe_buffer,
                                 &allocation, out_alloc_info);
        if (result != VK_SUCCESS) {
            LOG_WARNING(Render_Vulkan,
                        "A {} MB buffer did not fit in video memory ({}); the normal allocation "
                        "decides; {}",
                        buffer_ci.size >> 20, vk::to_string(vk::Result{result}),
                        Vulkan::DescribeDeviceLocalBudgets(allocator));
        }
    }
    if (result != VK_SUCCESS && t_prefer_host && usage == MemoryUsage::DeviceLocal) {
        // No room in video memory (BIG_GROW's mesh buffer on a small card): host memory instead
        // of an out-of-memory assert. The buffer stays device-addressable; reads over PCIe are
        // slow, so this is a fallback, not a preference.
        alloc_ci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        alloc_ci.preferredFlags |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        LOG_WARNING(Render_Vulkan,
                    "A {} MB buffer does not fit in video memory: trying system memory (slower); "
                    "{}",
                    buffer_ci.size >> 20, Vulkan::DescribeDeviceLocalBudgets(allocator));
    }
    if (result != VK_SUCCESS) {
        // v1.0.2: over VMA's budget, retry without it (vk_memory_budget.h) instead of asserting.
        result = Vulkan::AllocateWithBudgetRetry(
            allocator, alloc_ci, "buffer", buffer_ci.size,
            [&] { return std::string{BufferTypeName(usage)}; },
            [&](const VmaAllocationCreateInfo& ci) {
                return vmaCreateBuffer(allocator, &buffer_ci_unsafe, &ci, &unsafe_buffer,
                                       &allocation, out_alloc_info);
            });
    }
    if (result != VK_SUCCESS && buffer_ci.size >= (256ull << 20)) {
        Common::ReleaseMode::ShowOutOfMemoryNotice(buffer_ci.size); // 1.1.1 (GitHub #5)
    }
    ASSERT_MSG(result == VK_SUCCESS, "Failed allocating buffer of {} MB ({}) with error {}",
               buffer_ci.size >> 20, BufferTypeName(usage), vk::to_string(vk::Result{result}));
    buffer = vk::Buffer{unsafe_buffer};

    if (with_bda) {
        vk::BufferDeviceAddressInfo bda_info{
            .buffer = buffer,
        };
        auto bda_result = device.getBufferAddress(bda_info);
        ASSERT_MSG(bda_result != 0, "Failed to get buffer device address");
        bda_addr = bda_result;
    }
}

bool ProbeBufferAllocation(const Vulkan::Instance& instance, MemoryUsage usage,
                           vk::BufferUsageFlags flags, u64 size, u32& heap, bool require_video) {
    // The same create info and memory request as Buffer::Buffer + UniqueBuffer::Create.
    if (instance.IsNativeBindingsActive() &&
        (flags & (vk::BufferUsageFlagBits::eUniformBuffer |
                  vk::BufferUsageFlagBits::eStorageBuffer))) {
        flags |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
    }
    const std::array<u32, 2> families = {instance.GetGraphicsQueueFamilyIndex(),
                                         instance.GetReadbackQueueFamilyIndex()};
    const bool concurrent = instance.BuffersShareReadbackFamily();
    const VkBufferCreateInfo buffer_ci = static_cast<VkBufferCreateInfo>(vk::BufferCreateInfo{
        .size = size,
        .usage = flags,
        .sharingMode = concurrent ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = concurrent ? 2u : 0u,
        .pQueueFamilyIndices = concurrent ? families.data() : nullptr,
    });
    const bool with_bda = bool(flags & vk::BufferUsageFlagBits::eShaderDeviceAddress);
    VmaAllocationCreateInfo alloc_ci = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT |
                 (with_bda ? VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT : 0) |
                 MemoryUsageVmaFlags(usage),
        .usage = MemoryUsageVma(usage),
        .requiredFlags = 0,
        .preferredFlags = MemoryUsagePreferredVmaFlags(usage),
    };
    if (require_video && usage == MemoryUsage::DeviceLocal) {
        RequireVideo(alloc_ci);
    }
    VmaAllocator allocator = instance.GetAllocator();
    VkBuffer probe{};
    VmaAllocation allocation{};
    VmaAllocationInfo info{};
    VkResult result = vmaCreateBuffer(allocator, &buffer_ci, &alloc_ci, &probe, &allocation, &info);
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY && Vulkan::AllowOverBudget() &&
        (alloc_ci.flags & VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT) != 0) {
        alloc_ci.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        result = vmaCreateBuffer(allocator, &buffer_ci, &alloc_ci, &probe, &allocation, &info);
    }
    if (result != VK_SUCCESS) {
        return false;
    }
    const VkPhysicalDeviceMemoryProperties* props{};
    vmaGetMemoryProperties(allocator, &props);
    heap = props->memoryTypes[info.memoryType].heapIndex;
    vmaDestroyBuffer(allocator, probe, allocation);
    return true;
}

Buffer::Buffer(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
               vk::Buffer imported, VAddr cpu_addr_, u64 size_bytes_)
    : cpu_addr{cpu_addr_}, size_bytes{size_bytes_}, instance{&instance_}, scheduler{&scheduler_},
      usage{MemoryUsage::Upload}, buffer{instance->GetDevice(), instance->GetAllocator()} {
    // vmaDestroyBuffer with a null allocation only destroys the buffer.
    buffer.buffer = imported;
    is_coherent = true;
}

Buffer::Buffer(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_, MemoryUsage usage_,
               VAddr cpu_addr_, vk::BufferUsageFlags flags, u64 size_bytes_)
    : cpu_addr{cpu_addr_}, size_bytes{size_bytes_}, instance{&instance_}, scheduler{&scheduler_},
      usage{usage_}, buffer{instance->GetDevice(), instance->GetAllocator()} {
    // Descriptor heap buffer descriptors are encoded from device addresses, so every buffer a
    // guest shader can bind (stream, GDS, fault and page-table buffers included) needs one.
    // Opt-in mode only; the legacy allocation flags are otherwise untouched.
    if (instance->IsNativeBindingsActive() &&
        (flags & (vk::BufferUsageFlagBits::eUniformBuffer |
                  vk::BufferUsageFlagBits::eStorageBuffer))) {
        flags |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
    }
    // Create buffer object.
    const std::array<u32, 2> families = {instance->GetGraphicsQueueFamilyIndex(),
                                         instance->GetReadbackQueueFamilyIndex()};
    const bool concurrent = instance->BuffersShareReadbackFamily();
    const vk::BufferCreateInfo buffer_ci = {
        .size = size_bytes,
        .usage = flags,
        .sharingMode = concurrent ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = concurrent ? 2u : 0u,
        .pQueueFamilyIndices = concurrent ? families.data() : nullptr,
    };
    VmaAllocationInfo alloc_info{};
    buffer.Create(buffer_ci, usage, &alloc_info);

    const auto device = instance->GetDevice();
    Vulkan::SetObjectName(device, Handle(), "Buffer {:#x}:{:#x}", cpu_addr, size_bytes);

    // Map it if it is host visible.
    VkMemoryPropertyFlags property_flags{};
    vmaGetAllocationMemoryProperties(instance->GetAllocator(), buffer.allocation, &property_flags);
    if (alloc_info.pMappedData) {
        mapped_data = std::span<u8>{std::bit_cast<u8*>(alloc_info.pMappedData), size_bytes};
    }
    is_coherent = property_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
}

void Buffer::Fill(u64 offset, u32 num_bytes, u32 value) {
    scheduler->EndRendering();
    ASSERT_MSG(offset % 4 == 0 && num_bytes % 4 == 0,
               "FillBuffer size must be a multiple of 4 bytes");
    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .buffer = buffer,
        .offset = offset,
        .size = num_bytes,
    };
    const vk::BufferMemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .buffer = buffer,
        .offset = offset,
        .size = num_bytes,
    };
    scheduler->Record([buffer = vk::Buffer{buffer}, offset, num_bytes, value, pre_barrier,
                       post_barrier](vk::CommandBuffer cmdbuf) {
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &pre_barrier,
        });
        cmdbuf.fillBuffer(buffer, offset, num_bytes, value);
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &post_barrier,
        });
    });
}

constexpr u64 WATCHES_INITIAL_RESERVE = 0x4000;
constexpr u64 WATCHES_RESERVE_CHUNK = 0x1000;

StreamBuffer::StreamBuffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                           MemoryUsage usage, u64 size_bytes, vk::BufferUsageFlags extra_flags)
    : Buffer{instance, scheduler, usage, 0, AllFlags | extra_flags, size_bytes} {
    ReserveWatches(current_watches, WATCHES_INITIAL_RESERVE);
    ReserveWatches(previous_watches, WATCHES_INITIAL_RESERVE);
    const auto device = instance.GetDevice();
    Vulkan::SetObjectName(device, Handle(), "StreamBuffer({}):{:#x}", BufferTypeName(usage),
                          size_bytes);
}

std::pair<u8*, u64> StreamBuffer::Map(u64 size, u64 alignment, bool allow_wait) {
    if (!is_coherent && usage == MemoryUsage::Stream) {
        size = Common::AlignUp(size, instance->NonCoherentAtomSize());
    }

    if (size > this->size_bytes) {
        return {nullptr, 0};
    }

    mapped_size = size;

    if (alignment > 0) {
        offset = Common::AlignUp(offset, alignment);
    }

    if (offset + size > this->size_bytes) {
        // The buffer would overflow, save the amount of used watches and reset the state.
        if (Vulkan::BackCyclesEnabled()) [[unlikely]] {
            ++Vulkan::t_back_cycles.stream_wraps;
        }
        invalidation_mark = current_watch_cursor;
        current_watch_cursor = 0;
        offset = 0;

        // Swap watches and reset waiting cursors.
        std::swap(previous_watches, current_watches);
        wait_cursor = 0;
        wait_bound = 0;
    }

    const u64 mapped_upper_bound = offset + size;
    if (!WaitPendingOperations(mapped_upper_bound, allow_wait)) {
        return {nullptr, 0};
    }

    return {mapped_data.data() + offset, offset};
}

void StreamBuffer::Commit() {
    if (!is_coherent) {
        if (usage == MemoryUsage::Download) {
            vmaInvalidateAllocation(instance->GetAllocator(), buffer.allocation, offset,
                                    mapped_size);
        } else {
            vmaFlushAllocation(instance->GetAllocator(), buffer.allocation, offset, mapped_size);
        }
    }

    offset += mapped_size;
    // The cursor points to the next free slot; extend the last committed watch.
    if (current_watch_cursor != 0 &&
        current_watches[current_watch_cursor - 1].tick == scheduler->CurrentTick()) {
        current_watches[current_watch_cursor - 1].upper_bound = offset;
        return;
    }

    if (current_watch_cursor + 1 >= current_watches.size()) {
        // Ensure that there are enough watches.
        ReserveWatches(current_watches, WATCHES_RESERVE_CHUNK);
    }

    auto& watch = current_watches[current_watch_cursor++];
    watch.upper_bound = offset;
    watch.tick = scheduler->CurrentTick();
}

void StreamBuffer::ReserveWatches(std::vector<Watch>& watches, std::size_t grow_size) {
    watches.resize(watches.size() + grow_size);
}

bool StreamBuffer::WaitPendingOperations(u64 requested_upper_bound, bool allow_wait) {
    if (!invalidation_mark) {
        return true;
    }
    while (requested_upper_bound > wait_bound && wait_cursor < *invalidation_mark) {
        auto& watch = previous_watches[wait_cursor];
        if (!scheduler->IsFree(watch.tick) && !allow_wait) {
            return false;
        }
        if (Vulkan::BackCyclesEnabled() && !scheduler->IsFree(watch.tick)) [[unlikely]] {
            const u64 c0 = __rdtsc();
            scheduler->Wait(watch.tick);
            ++Vulkan::t_back_cycles.stream_waits;
            Vulkan::t_back_cycles.stream_wait_cyc += __rdtsc() - c0;
        } else {
            scheduler->Wait(watch.tick);
        }
        wait_bound = watch.upper_bound;
        ++wait_cursor;
    }
    return true;
}

} // namespace VideoCore
