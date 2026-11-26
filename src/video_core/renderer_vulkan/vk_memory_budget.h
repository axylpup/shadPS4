// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <optional>
#include <string>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "common/thread.h"
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

#include <vk_mem_alloc.h>

namespace Vulkan {

// v1.0.2 (2026-10-19; an RTX 4060 Laptop, 8 GB, ended with "Failed allocating buffer with error
// ErrorOutOfDeviceMemory" when loading a save). Every allocation asks VMA for
// VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT, and the allocator is created without
// VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT, so VMA's "budget" is 80% of the heap size counted
// against VMA's own allocations only (vk_mem_alloc.h GetHeapBudgets: 7956 MiB * 0.8 = 6365 MiB
// on that card). Past it VMA refuses before the driver is asked, and the ASSERT ends the game.
// Now a refused allocation is retried once without the budget check: the driver may then page
// video memory out to system memory (slower, no crash), and VMA itself moves on to the host
// memory types if the device-local heap really is full. SHADPS4_VRAM_OVER_BUDGET=0 restores the
// v1.0.1 behaviour (assert at the budget).
inline bool AllowOverBudget() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_VRAM_OVER_BUDGET");
        return !(e && e[0] == '0');
    }();
    return on;
}

/// SHADPS4_MESH_HOST_FALLBACK=0: BIG_GROW's mesh buffer refuses system memory and asserts as it
/// did before (for measuring the fallback's cost).
inline bool AllowMeshHostFallback() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_MESH_HOST_FALLBACK");
        return !(e && e[0] == '0');
    }();
    return on;
}

/// VMA's view of every device-local heap: "heap i: usage / budget MB (n allocations)".
inline std::string DescribeDeviceLocalBudgets(VmaAllocator allocator) {
    const VkPhysicalDeviceMemoryProperties* props{};
    vmaGetMemoryProperties(allocator, &props);
    VmaBudget budgets[VK_MAX_MEMORY_HEAPS]{};
    vmaGetHeapBudgets(allocator, budgets);
    std::string out;
    for (u32 i = 0; i < props->memoryHeapCount; ++i) {
        const bool local = (props->memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
        out += fmt::format("{}heap {}{}: {} / {} MB ({} allocations)", out.empty() ? "" : ", ", i,
                           local ? " (video)" : " (system)", budgets[i].usage >> 20,
                           budgets[i].budget >> 20, budgets[i].statistics.allocationCount);
    }
    return out;
}

/// 1.1.1 (GitHub #5): VK_EXT_memory_budget is on (Instance sets it), so the driver's own numbers
/// can be read - VMA's above are its 80%-of-heap estimate against its own blocks only.
inline std::atomic<bool> g_driver_budget{false};

/// The driver's view of every heap: "driver: heap i used / budget MB". Empty without the extension.
inline std::string DescribeDriverBudgets(VmaAllocator allocator) {
    if (!g_driver_budget.load(std::memory_order_relaxed)) {
        return {};
    }
    VmaAllocatorInfo info{};
    vmaGetAllocatorInfo(allocator, &info);
    vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    vk::PhysicalDeviceMemoryProperties2 props = {
        .pNext = &budget,
    };
    vk::PhysicalDevice{info.physicalDevice}.getMemoryProperties2(&props);
    std::string out = "driver:";
    for (u32 i = 0; i < props.memoryProperties.memoryHeapCount; ++i) {
        out += fmt::format("{} heap {} {} / {} MB", i == 0 ? "" : ",", i,
                           budget.heapUsage[i] >> 20, budget.heapBudget[i] >> 20);
    }
    return out;
}

/// 1.1.1 (GitHub #30): the driver's free bytes (budget - usage) on its largest device-local heap,
/// or nullopt without VK_EXT_memory_budget. SHADPS4_SIM_DRIVER_FREE_MB (test) caps the answer.
inline std::optional<u64> DriverFreeDeviceLocal(VmaAllocator allocator) {
    if (!g_driver_budget.load(std::memory_order_relaxed)) {
        return std::nullopt;
    }
    VmaAllocatorInfo info{};
    vmaGetAllocatorInfo(allocator, &info);
    vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    vk::PhysicalDeviceMemoryProperties2 props = {
        .pNext = &budget,
    };
    vk::PhysicalDevice{info.physicalDevice}.getMemoryProperties2(&props);
    u64 largest = 0;
    std::optional<u64> free;
    for (u32 i = 0; i < props.memoryProperties.memoryHeapCount; ++i) {
        const auto& heap = props.memoryProperties.memoryHeaps[i];
        if ((heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) && heap.size > largest) {
            largest = heap.size;
            free = budget.heapBudget[i] > budget.heapUsage[i]
                       ? budget.heapBudget[i] - budget.heapUsage[i]
                       : 0;
        }
    }
    static const u64 sim = [] {
        const char* e = std::getenv("SHADPS4_SIM_DRIVER_FREE_MB");
        return e ? std::strtoull(e, nullptr, 10) << 20 : ~0ull;
    }();
    if (free && *free > sim) {
        free = sim;
    }
    return free;
}

/// Runs `alloc(create_info)`; on VK_ERROR_OUT_OF_DEVICE_MEMORY with the budget bit set, retries
/// once without it. Logs the first retries of each kind and every final failure (with the size,
/// `detail()` and VMA's budgets, so a player's log says what did not fit).
template <typename D, typename F>
VkResult AllocateWithBudgetRetry(VmaAllocator allocator, VmaAllocationCreateInfo& create_info,
                                 const char* what, u64 bytes, D&& detail, F&& alloc) {
    VkResult result = alloc(create_info);
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY &&
        (create_info.flags & VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT) != 0 && AllowOverBudget()) {
        create_info.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        result = alloc(create_info);
        static std::atomic<u64> retries{0};
        const u64 n = ++retries;
        if (n <= 8 || (n & (n - 1)) == 0) {
            LOG_WARNING(Render_Vulkan,
                        "Video memory over budget: {} of {} MB {} allocated without the budget "
                        "check ({}) - retry #{}; {}",
                        what, bytes >> 20, detail(),
                        result == VK_SUCCESS ? "ok" : "still failed", n,
                        DescribeDeviceLocalBudgets(allocator));
        }
    }
    if (result != VK_SUCCESS) {
        LOG_CRITICAL(Render_Vulkan, "Allocation failed: {} of {} bytes ({} MB) {}: error {}; {}; {}; {}",
                     what, bytes, bytes >> 20, detail(), static_cast<int>(result),
                     DescribeDeviceLocalBudgets(allocator), DescribeDriverBudgets(allocator),
                     Common::DescribeSystemMemory());
    }
    return result;
}

} // namespace Vulkan
