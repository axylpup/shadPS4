// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdlib>
#include <string_view>
#include <boost/container/static_vector.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/debug.h"
#include "common/types.h"
#include "core/address_space.h"
#include "imgui/renderer/imgui_core.h"
#include "sdl_window.h"
#include "video_core/buffer_cache/pt_bits.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_dlss.h"
#include "video_core/renderer_vulkan/vk_dlss5.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_memory_budget.h"
#include "video_core/renderer_vulkan/vk_platform.h"

#include <vk_mem_alloc.h>

namespace Vulkan {

namespace {

std::vector<vk::PhysicalDevice> EnumeratePhysicalDevices(vk::UniqueInstance& instance) {
    auto [devices_result, devices] = instance->enumeratePhysicalDevices();
    ASSERT_MSG(devices_result == vk::Result::eSuccess, "Failed to enumerate physical devices: {}",
               vk::to_string(devices_result));
    return std::move(devices);
}

std::vector<std::string> GetSupportedExtensions(vk::PhysicalDevice physical) {
    const auto [extensions_result, extensions] = physical.enumerateDeviceExtensionProperties();
    if (extensions_result != vk::Result::eSuccess) {
        LOG_ERROR(Render_Vulkan, "Could not query supported extensions: {}",
                  vk::to_string(extensions_result));
        return {};
    }
    std::vector<std::string> supported_extensions;
    supported_extensions.reserve(extensions.size());
    for (const auto& extension : extensions) {
        supported_extensions.emplace_back(extension.extensionName.data());
    }
    return supported_extensions;
}

vk::FormatProperties3 GetFormatProperties(vk::PhysicalDevice physical, vk::Format format) {
    vk::FormatProperties3 properties3{};
    vk::FormatProperties2 properties2 = {
        .pNext = &properties3,
    };
    physical.getFormatProperties2(format, &properties2);
    return properties3;
}

std::unordered_map<vk::Format, vk::FormatProperties3> GetFormatProperties(
    vk::PhysicalDevice physical) {
    std::unordered_map<vk::Format, vk::FormatProperties3> format_properties;
    for (const auto& format_info : LiverpoolToVK::SurfaceFormats()) {
        const auto format = format_info.vk_format;
        if (!format_properties.contains(format)) {
            format_properties.emplace(format, GetFormatProperties(physical, format));
        }
    }
    for (const auto& format_info : LiverpoolToVK::DepthFormats()) {
        const auto format = format_info.vk_format;
        if (!format_properties.contains(format)) {
            format_properties.emplace(format, GetFormatProperties(physical, format));
        }
    }
    // Other miscellaneous formats, e.g. for color buffers, swizzles, or compatibility
    static constexpr std::array misc_formats = {
        vk::Format::eA2R10G10B10UnormPack32,
        vk::Format::eB8G8R8A8Unorm,
        vk::Format::eB8G8R8A8Srgb,
        vk::Format::eD24UnormS8Uint,
    };
    for (const auto& format : misc_formats) {
        if (!format_properties.contains(format)) {
            format_properties.emplace(format, GetFormatProperties(physical, format));
        }
    }
    return format_properties;
}

std::string GetReadableVersion(u32 version) {
    return fmt::format("{}.{}.{}", VK_VERSION_MAJOR(version), VK_VERSION_MINOR(version),
                       VK_VERSION_PATCH(version));
}

} // Anonymous namespace

Instance::Instance(bool enable_validation, bool enable_crash_diagnostic)
    : instance{CreateInstance(Frontend::WindowSystemType::Headless, enable_validation,
                              enable_crash_diagnostic)},
      physical_devices{EnumeratePhysicalDevices(instance)} {}

Instance::Instance(Frontend::WindowSDL& window, s32 physical_device_index,
                   bool enable_validation /*= false*/, bool enable_crash_diagnostic /*= false*/)
    : instance{CreateInstance(window.GetWindowInfo().type, enable_validation,
                              enable_crash_diagnostic)},
      physical_devices{EnumeratePhysicalDevices(instance)} {
    if (enable_validation) {
        debug_callback = CreateDebugCallback(*instance);
    }
    if (AddrBindReportOn()) {
        addr_bind_callback = CreateAddressBindingCallback(*instance);
    }
    const std::size_t num_physical_devices = static_cast<u16>(physical_devices.size());
    ASSERT_MSG(num_physical_devices > 0, "No physical devices found");
    LOG_INFO(Render_Vulkan, "Found {} physical devices", num_physical_devices);

    if (physical_device_index < 0) {
        std::vector<
            std::tuple<size_t, vk::PhysicalDeviceProperties2, vk::PhysicalDeviceMemoryProperties>>
            properties2{};
        for (auto const& physical : physical_devices) {
            properties2.emplace_back(properties2.size(), physical.getProperties2(),
                                     physical.getMemoryProperties());
        }
        std::sort(properties2.begin(), properties2.end(), [](const auto& left, const auto& right) {
            const vk::PhysicalDeviceProperties& left_prop = std::get<1>(left).properties;
            const vk::PhysicalDeviceProperties& right_prop = std::get<1>(right).properties;
            const bool left_supports_api = left_prop.apiVersion >= TargetVulkanApiVersion;
            const bool right_supports_api = right_prop.apiVersion >= TargetVulkanApiVersion;
            if (left_supports_api != right_supports_api) {
                return left_supports_api;
            }

            const bool left_is_discrete =
                left_prop.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
            const bool right_is_discrete =
                right_prop.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
            if (left_is_discrete != right_is_discrete) {
                return left_is_discrete;
            }

            constexpr auto get_mem = [](const vk::PhysicalDeviceMemoryProperties& mem) -> size_t {
                size_t max = 0;
                for (u32 i = 0; i < mem.memoryHeapCount; i++) {
                    const vk::MemoryHeap& heap = mem.memoryHeaps[i];
                    if (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal && heap.size > max) {
                        max = heap.size;
                    }
                }
                return max;
            };
            size_t left_mem_size = get_mem(std::get<2>(left));
            size_t right_mem_size = get_mem(std::get<2>(right));
            return left_mem_size > right_mem_size;
        });
        physical_device = physical_devices[std::get<0>(properties2[0])];
    } else {
        ASSERT_MSG(physical_device_index < num_physical_devices,
                   "Invalid physical device index {} provided when only {} devices exist",
                   physical_device_index, num_physical_devices);

        physical_device = physical_devices[physical_device_index];
    }

    available_extensions = GetSupportedExtensions(physical_device);
    format_properties = GetFormatProperties(physical_device);
    properties = physical_device.getProperties();
    memory_properties = physical_device.getMemoryProperties();
    CollectDeviceParameters();
    ASSERT_MSG(properties.apiVersion >= TargetVulkanApiVersion,
               "Vulkan {}.{} is required, but only {}.{} is supported by device!",
               VK_VERSION_MAJOR(TargetVulkanApiVersion), VK_VERSION_MINOR(TargetVulkanApiVersion),
               VK_VERSION_MAJOR(properties.apiVersion), VK_VERSION_MINOR(properties.apiVersion));

    CreateDevice();
    CollectPhysicalMemoryInfo();
    CreateSimBallast();
    {
        // The GPU page table's default coverage (pt_bits.h): the 64 MB table where video memory
        // is short (the game maps GPU memory no higher than 63.5 GB, s1457 / s1460).
        u64 heap = 0;
        for (u32 i = 0; i < memory_properties.memoryHeapCount; ++i) {
            if (memory_properties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
                heap = std::max<u64>(heap, memory_properties.memoryHeaps[i].size);
            }
        }
        if (sim_vram_bytes != 0) {
            heap = std::min<u64>(heap, sim_vram_bytes);
        }
        VideoCore::g_pt_auto_bits.store(heap < (10ull << 30) ? 37u : 40u);
    }
    CollectImageFormatInfo();
    CollectToolingInfo();
}

Instance::~Instance() {
    if (Dlss::Mode() == 1 && device) {
        // Streamline goes before the device (its rule); the GPU must be done with its work.
        (void)device->waitIdle();
        Dlss5::Shutdown();
        Dlss::Shutdown();
    }
    ImGui::Core::Shutdown(GetDevice());
    for (size_t i = 0; i < sim_ballast_buffers.size(); ++i) {
        vmaDestroyBuffer(allocator, sim_ballast_buffers[i], sim_ballast_allocations[i]);
    }
    vmaDestroyAllocator(allocator);
}

std::string Instance::GetDriverVersionName() {
    // Extracted from
    // https://github.com/SaschaWillems/vulkan.gpuinfo.org/blob/5dddea46ea1120b0df14eef8f15ff8e318e35462/functions.php#L308-L314
    const u32 version = properties.driverVersion;
    if (driver_id == vk::DriverId::eNvidiaProprietary) {
        const u32 major = (version >> 22) & 0x3ff;
        const u32 minor = (version >> 14) & 0x0ff;
        const u32 secondary = (version >> 6) & 0x0ff;
        const u32 tertiary = version & 0x003f;
        return fmt::format("{}.{}.{}.{}", major, minor, secondary, tertiary);
    }
    if (driver_id == vk::DriverId::eIntelProprietaryWindows) {
        const u32 major = version >> 14;
        const u32 minor = version & 0x3fff;
        return fmt::format("{}.{}", major, minor);
    }
    return GetReadableVersion(version);
}

bool Instance::CreateDevice() {
    const vk::StructureChain feature_chain =
        physical_device
            .getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
                          vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features,
                          vk::PhysicalDeviceRobustness2FeaturesEXT,
                          vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT,
                          vk::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT,
                          vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT,
                          vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR,
                          vk::PhysicalDeviceImage2DViewOf3DFeaturesEXT>();
    features = feature_chain.get().features;

    const vk::StructureChain properties_chain = physical_device.getProperties2<
        vk::PhysicalDeviceProperties2, vk::PhysicalDeviceVulkan11Properties,
        vk::PhysicalDeviceVulkan12Properties, vk::PhysicalDeviceVulkan13Properties,
        vk::PhysicalDevicePushDescriptorPropertiesKHR>();
    vk11_props = properties_chain.get<vk::PhysicalDeviceVulkan11Properties>();
    vk12_props = properties_chain.get<vk::PhysicalDeviceVulkan12Properties>();
    vk13_props = properties_chain.get<vk::PhysicalDeviceVulkan13Properties>();
    push_descriptor_props = properties_chain.get<vk::PhysicalDevicePushDescriptorPropertiesKHR>();
    LOG_INFO(Render_Vulkan, "Physical device subgroup size {}", vk11_props.subgroupSize);

    if (available_extensions.empty()) {
        LOG_CRITICAL(Render_Vulkan, "No extensions supported by device.");
        return false;
    }

    boost::container::static_vector<const char*, 48> enabled_extensions;
    // SHADPS4_SKIP_DEVICE_EXT=a,b (2026-10-17, diagnosis): treat these extensions as unavailable.
    static const std::string skip_device_ext = [] {
        const char* v = std::getenv("SHADPS4_SKIP_DEVICE_EXT");
        return std::string{v ? v : ""};
    }();
    const auto add_extension = [&](std::string_view extension) -> bool {
        if (!skip_device_ext.empty() && skip_device_ext.find(extension) != std::string::npos) {
            LOG_WARNING(Render_Vulkan, "Extension {} skipped (SHADPS4_SKIP_DEVICE_EXT)", extension);
            return false;
        }
        const auto result =
            std::find_if(available_extensions.begin(), available_extensions.end(),
                         [&](const std::string& name) { return name == extension; });

        if (result != available_extensions.end()) {
            LOG_INFO(Render_Vulkan, "Enabling extension: {}", extension);
            enabled_extensions.push_back(extension.data());
            return true;
        }

        LOG_WARNING(Render_Vulkan, "Extension {} unavailable.", extension);
        return false;
    };

    // Required
    ASSERT_MSG(add_extension(VK_KHR_SWAPCHAIN_EXTENSION_NAME),
               "Required Vulkan extension unavailable: {}", VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    ASSERT_MSG(add_extension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME),
               "Required Vulkan extension unavailable: {}", VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    ASSERT_MSG(add_extension(VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME),
               "Required Vulkan extension unavailable: {}",
               VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME);
    ASSERT_MSG(add_extension(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME),
               "Required Vulkan extension unavailable: {}", VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);

    const auto robustness2_features = feature_chain.get<vk::PhysicalDeviceRobustness2FeaturesEXT>();
    ASSERT_MSG(robustness2_features.robustBufferAccess2,
               "Required Vulkan feature unavailable: robustBufferAccess2");
    ASSERT_MSG(robustness2_features.robustImageAccess2,
               "Required Vulkan feature unavailable: robustImageAccess2");
    ASSERT_MSG(robustness2_features.nullDescriptor,
               "Required Vulkan feature unavailable: nullDescriptor");

    // Optional
    maintenance_8 = add_extension(VK_KHR_MAINTENANCE_8_EXTENSION_NAME);
    // p2 (sec. 20): INLINE_BIT_KHR - one pass mixes inline draws and worker secondaries.
    maintenance_7 = add_extension(VK_KHR_MAINTENANCE_7_EXTENSION_NAME);
    attachment_feedback_loop = add_extension(VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME);
    if (attachment_feedback_loop) {
        attachment_feedback_loop =
            add_extension(VK_EXT_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_EXTENSION_NAME);
        if (!attachment_feedback_loop) {
            // We want both extensions so remove the first if the second isn't available
            enabled_extensions.pop_back();
        }
    }
    depth_range_unrestricted = add_extension(VK_EXT_DEPTH_RANGE_UNRESTRICTED_EXTENSION_NAME);
    dynamic_state_3 = add_extension(VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME);
    if (dynamic_state_3) {
        dynamic_state_3_features =
            feature_chain.get<vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT>();
        LOG_INFO(Render_Vulkan, "- extendedDynamicState3ColorWriteMask: {}",
                 dynamic_state_3_features.extendedDynamicState3ColorWriteMask);
    }
    custom_border_color = add_extension(VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME);
    depth_clip_control = add_extension(VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME);
    depth_clip_enable = add_extension(VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME);
    vertex_input_dynamic_state = add_extension(VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME);
    list_restart = add_extension(VK_EXT_PRIMITIVE_TOPOLOGY_LIST_RESTART_EXTENSION_NAME);
    if (list_restart) {
        list_restart_features =
            feature_chain.get<vk::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT>();
        LOG_INFO(Render_Vulkan, "- primitiveTopologyListRestart: {}",
                 list_restart_features.primitiveTopologyListRestart);
        LOG_INFO(Render_Vulkan, "- primitiveTopologyPatchListRestart: {}",
                 list_restart_features.primitiveTopologyPatchListRestart);
    }
    amd_shader_explicit_vertex_parameter =
        add_extension(VK_AMD_SHADER_EXPLICIT_VERTEX_PARAMETER_EXTENSION_NAME);
    if (!amd_shader_explicit_vertex_parameter) {
        fragment_shader_barycentric =
            add_extension(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
    }
    provoking_vertex = add_extension(VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME);
    shader_stencil_export = add_extension(VK_EXT_SHADER_STENCIL_EXPORT_EXTENSION_NAME);
    image_load_store_lod = add_extension(VK_AMD_SHADER_IMAGE_LOAD_STORE_LOD_EXTENSION_NAME);
    amd_gcn_shader = add_extension(VK_AMD_GCN_SHADER_EXTENSION_NAME);
    amd_shader_trinary_minmax = add_extension(VK_AMD_SHADER_TRINARY_MINMAX_EXTENSION_NAME);
    // SHADPS4_NO_MIXED_SAMPLES=1: ignore VK_NV_framebuffer_mixed_samples, so pipelines force color
    // and depth sample counts equal (what happens under RenderDoc, which hides the extension). Test
    // for the PS4 Pro triangle-edge pattern on skin (2026-09-24), seen only with the extension.
    const bool no_mixed_samples = [] {
        const char* value = std::getenv("SHADPS4_NO_MIXED_SAMPLES");
        return value && value[0] == '1';
    }();
    nv_framebuffer_mixed_samples =
        !no_mixed_samples && add_extension(VK_NV_FRAMEBUFFER_MIXED_SAMPLES_EXTENSION_NAME);
    amd_mixed_attachment_samples = add_extension(VK_AMD_MIXED_ATTACHMENT_SAMPLES_EXTENSION_NAME);
    shader_atomic_float = add_extension(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    shader_atomic_float2 = add_extension(VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME);
    if (shader_atomic_float2) {
        shader_atomic_float2_features =
            feature_chain.get<vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT>();
        LOG_INFO(Render_Vulkan, "- shaderBufferFloat32AtomicMinMax: {}",
                 shader_atomic_float2_features.shaderBufferFloat32AtomicMinMax);
        LOG_INFO(Render_Vulkan, "- shaderImageFloat32AtomicMinMax: {}",
                 shader_atomic_float2_features.shaderImageFloat32AtomicMinMax);
    }
    workgroup_memory_explicit_layout =
        add_extension(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME);
    if (workgroup_memory_explicit_layout) {
        workgroup_memory_explicit_layout_features =
            feature_chain.get<vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR>();
        LOG_INFO(Render_Vulkan, "- workgroupMemoryExplicitLayout: {}",
                 workgroup_memory_explicit_layout_features.workgroupMemoryExplicitLayout);
        LOG_INFO(Render_Vulkan, "- workgroupMemoryExplicitLayoutScalarBlockLayout: {}",
                 workgroup_memory_explicit_layout_features
                     .workgroupMemoryExplicitLayoutScalarBlockLayout);
        LOG_INFO(
            Render_Vulkan, "- workgroupMemoryExplicitLayout16BitAccess: {}",
            workgroup_memory_explicit_layout_features.workgroupMemoryExplicitLayout16BitAccess);
    }
    image_2d_view_of_3d = add_extension(VK_EXT_IMAGE_2D_VIEW_OF_3D_EXTENSION_NAME);
    if (image_2d_view_of_3d) {
        image_2d_view_of_3d_features =
            feature_chain.get<vk::PhysicalDeviceImage2DViewOf3DFeaturesEXT>();
        LOG_INFO(Render_Vulkan, "- image2DViewOf3D: {}",
                 image_2d_view_of_3d_features.image2DViewOf3D);
        LOG_INFO(Render_Vulkan, "- sampler2DViewOf3D: {}",
                 image_2d_view_of_3d_features.sampler2DViewOf3D);
    }
    image_view_min_lod = add_extension(VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME);
    supports_memory_budget = add_extension(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    g_driver_budget.store(supports_memory_budget, std::memory_order_relaxed); // vk_memory_budget.h
    if (const char* fault = std::getenv("SHADPS4_DEVICE_FAULT"); fault && fault[0] == '1') {
        device_fault = add_extension(VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
        LOG_INFO(Render_Vulkan, "DeviceFault: VK_EXT_device_fault {}",
                 device_fault ? "enabled" : "not offered by the driver");
    }
    if (AddrBindReportOn()) {
        addr_bind_report = add_extension(VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME);
        LOG_INFO(Render_Vulkan, "AddrBind: VK_EXT_device_address_binding_report {}",
                 addr_bind_report ? "enabled" : "not offered by the driver");
    }
    if (const char* mirror = std::getenv("SHADPS4_HOST_MIRROR"); mirror && mirror[0] == '1') {
        external_memory_host = add_extension(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    }
    const bool calibrated_timestamps =
        TRACY_GPU_ENABLED ? add_extension(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME) : false;

    // SHADPS4_DLSS=1 (vk_dlss.h): the device extensions Streamline asked for, when DLSS runs on
    // this GPU. VK_EXT_buffer_device_address is left out: the core bufferDeviceAddress feature
    // (enabled below) must not be combined with that name.
    bool dlss_device = Dlss::DeviceSupported(physical_device);
    if (dlss_device) {
        for (const char* ext : Dlss::DeviceExtensions()) {
            const bool present = std::ranges::any_of(
                enabled_extensions, [ext](const char* e) { return std::strcmp(e, ext) == 0; });
            // SHADPS4_DLSS_SKIP_EXT=a,b: leave these out (start-up diagnosis, 2026-10-17).
            const char* skip = std::getenv("SHADPS4_DLSS_SKIP_EXT");
            if (present || std::strcmp(ext, "VK_EXT_buffer_device_address") == 0 ||
                (skip && std::strstr(skip, ext))) {
                LOG_INFO(Render_Vulkan, "DLSS: device extension {} {}", ext,
                         present ? "already enabled" : "left out");
                continue;
            }
            if (!add_extension(ext)) {
                dlss_device = false;
            }
        }
        LOG_INFO(Render_Vulkan, "DLSS: device extensions {}", dlss_device ? "enabled" : "missing");
        // DLSS 5 (vk_dlss5.h): the model's two vendor extensions; Streamline's DLSS list has
        // them already on NVIDIA drivers, so this normally only logs.
        if (dlss_device && Dlss5::Wanted()) {
            for (const char* ext : Dlss5::DeviceExtensions()) {
                if (std::ranges::any_of(enabled_extensions, [ext](const char* e) {
                        return std::strcmp(e, ext) == 0;
                    })) {
                    LOG_INFO(Render_Vulkan, "DLSS 5: device extension {} enabled", ext);
                } else if (!add_extension(ext)) {
                    Dlss5::Disable("the GPU driver lacks a Vulkan extension the model needs");
                }
            }
        }
    }

    // Opt-in descriptor heap experiment. Nothing below is touched unless
    // SHADPS4_NATIVE_BINDINGS=1 and the device passes every capability check.
    bool native_bindings = QueryNativeBindings();
    if (native_bindings && enabled_extensions.size() + 2 > enabled_extensions.capacity()) {
        LOG_WARNING(Render_Vulkan, "NativeBindings: no room in the extension list, using legacy");
        native_bindings = false;
    }
    if (native_bindings) {
        native_bindings = add_extension(NativeApi::Maintenance5ExtensionName) &&
                          add_extension(NativeApi::DescriptorHeapExtensionName);
    }

    const auto family_properties = physical_device.getQueueFamilyProperties();
    if (family_properties.empty()) {
        LOG_CRITICAL(Render_Vulkan, "Physical device reported no queues.");
        return false;
    }

    bool graphics_queue_found = false;
    for (std::size_t i = 0; i < family_properties.size(); i++) {
        const u32 index = static_cast<u32>(i);
        if (family_properties[i].queueFlags & vk::QueueFlagBits::eGraphics) {
            queue_family_index = index;
            graphics_queue_found = true;
        }
    }

    if (!graphics_queue_found) {
        LOG_CRITICAL(Render_Vulkan, "Unable to find graphics and/or present queues.");
        return false;
    }

    // SHADPS4_READBACK_QUEUE for guest readback copies (see buffer_cache/readback_queue.h):
    // 1 = a second queue of the graphics family (measured 2026-09-23: no gain, likely shares
    // the graphics engine); 2 = a transfer-only family, i.e. a dedicated copy engine.
    static constexpr std::array queue_priorities = {1.0f, 1.0f, 1.0f};
    const char readback_mode = [] {
        const char* value = std::getenv("SHADPS4_READBACK_QUEUE");
        return value ? value[0] : '0';
    }();
    readback_queue_family = queue_family_index;
    if (readback_mode == '2') {
        for (u32 i = 0; i < family_properties.size(); ++i) {
            const auto flags = family_properties[i].queueFlags;
            if ((flags & vk::QueueFlagBits::eTransfer) && !(flags & vk::QueueFlagBits::eGraphics) &&
                !(flags & vk::QueueFlagBits::eCompute)) {
                readback_queue_family = i;
                has_readback_queue = true;
                break;
            }
        }
    } else if (readback_mode == '1') {
        has_readback_queue = family_properties[queue_family_index].queueCount >= 2;
    } else if (readback_mode == '3') {
        // 3 = a compute-only family (the async compute engine on NVIDIA; a second queue of the
        // graphics family time-slices the graphics engine: 2026-09-27 fast-lane copies still
        // waited ~0.7 ms there). Buffers are made CONCURRENT between the two families, so no
        // ownership transfer is needed (mode 2 relied on undefined EXCLUSIVE behaviour).
        for (u32 i = 0; i < family_properties.size(); ++i) {
            const auto flags = family_properties[i].queueFlags;
            if ((flags & vk::QueueFlagBits::eCompute) && !(flags & vk::QueueFlagBits::eGraphics)) {
                readback_queue_family = i;
                has_readback_queue = true;
                readback_family_concurrent = true;
                break;
            }
        }
    }
    const bool separate_family = has_readback_queue && readback_queue_family != queue_family_index;
    // SHADPS4_ASC_STREAM=1 (with READBACK_QUEUE=3): queue 1 of the compute-only family records the
    // guest's async compute (see Scheduler::SwitchStream).
    has_asc_queue = [] {
        const char* value = std::getenv("SHADPS4_ASC_STREAM");
        return value && (value[0] == '1' || value[0] == '2');
    }() && readback_family_concurrent && family_properties[readback_queue_family].queueCount >= 2;
    // SHADPS4_ASC_LANE=1 (sec. 21): the same queue for the back's compute lane (no ASC_STREAM routing).
    has_asc_lane = !has_asc_queue && [] {
        const char* value = std::getenv("SHADPS4_ASC_LANE");
        return value && value[0] == '1';
    }() && readback_family_concurrent && family_properties[readback_queue_family].queueCount >= 2;
    // SHADPS4_ARENA_RETRY_QUEUE=1 (r18): the next free queue of the compute-only family for the
    // arena prefetch retry copies (CONCURRENT buffers already cover the family).
    const u32 retry_index = has_asc_queue || has_asc_lane ? 2u : 1u;
    has_retry_queue = [] {
        const char* value = std::getenv("SHADPS4_ARENA_RETRY_QUEUE");
        return value && value[0] == '1';
    }() && readback_family_concurrent &&
                      family_properties[readback_queue_family].queueCount > retry_index;
    const std::array queue_infos = {
        vk::DeviceQueueCreateInfo{
            .queueFamilyIndex = queue_family_index,
            .queueCount = has_readback_queue && !separate_family ? 2u : 1u,
            .pQueuePriorities = queue_priorities.data(),
        },
        vk::DeviceQueueCreateInfo{
            .queueFamilyIndex = readback_queue_family,
            .queueCount = has_retry_queue ? retry_index + 1 : (has_asc_queue || has_asc_lane ? 2u : 1u),
            .pQueuePriorities = queue_priorities.data(),
        },
    };

    const auto vk11_features = feature_chain.get<vk::PhysicalDeviceVulkan11Features>();
    vk12_features = feature_chain.get<vk::PhysicalDeviceVulkan12Features>();
    {
        const char* e = std::getenv("SHADPS4_NATIVE_BINDLESS");
        bindless_supported = e && (e[0] == '1' || e[0] == '2') && vk12_features.descriptorIndexing &&
                             vk12_features.runtimeDescriptorArray &&
                             vk12_features.shaderSampledImageArrayNonUniformIndexing &&
                             vk12_features.descriptorBindingSampledImageUpdateAfterBind &&
                             vk12_features.descriptorBindingUpdateUnusedWhilePending &&
                             vk12_features.descriptorBindingPartiallyBound;
        LOG_INFO(Render_Vulkan, "NativeBindless: {}",
                 bindless_supported ? "descriptor indexing enabled" : "off");
    }
    vk13_features = feature_chain.get<vk::PhysicalDeviceVulkan13Features>();
    vk::StructureChain device_chain = {
        vk::DeviceCreateInfo{
            .queueCreateInfoCount = separate_family ? 2u : 1u,
            .pQueueCreateInfos = queue_infos.data(),
            .enabledExtensionCount = static_cast<u32>(enabled_extensions.size()),
            .ppEnabledExtensionNames = enabled_extensions.data(),
        },
        vk::PhysicalDeviceFeatures2{
            .features{
                .robustBufferAccess = features.robustBufferAccess,
                .imageCubeArray = features.imageCubeArray,
                .independentBlend = features.independentBlend,
                .geometryShader = features.geometryShader,
                .tessellationShader = features.tessellationShader,
                .sampleRateShading = features.sampleRateShading,
                .dualSrcBlend = features.dualSrcBlend,
                .logicOp = features.logicOp,
                .multiDrawIndirect = features.multiDrawIndirect,
                .depthClamp = features.depthClamp,
                .depthBiasClamp = features.depthBiasClamp,
                .fillModeNonSolid = features.fillModeNonSolid,
                .depthBounds = features.depthBounds,
                .wideLines = features.wideLines,
                .multiViewport = features.multiViewport,
                .samplerAnisotropy = features.samplerAnisotropy,
                .occlusionQueryPrecise = features.occlusionQueryPrecise,
                .vertexPipelineStoresAndAtomics = features.vertexPipelineStoresAndAtomics,
                .fragmentStoresAndAtomics = features.fragmentStoresAndAtomics,
                .shaderImageGatherExtended = features.shaderImageGatherExtended,
                .shaderStorageImageExtendedFormats = features.shaderStorageImageExtendedFormats,
                .shaderStorageImageMultisample = features.shaderStorageImageMultisample,
                .shaderClipDistance = features.shaderClipDistance,
                .shaderFloat64 = features.shaderFloat64,
                .shaderInt64 = features.shaderInt64,
                .shaderInt16 = features.shaderInt16,
            },
        },
        vk::PhysicalDeviceVulkan11Features{
            .storageBuffer16BitAccess = vk11_features.storageBuffer16BitAccess,
            .uniformAndStorageBuffer16BitAccess = vk11_features.uniformAndStorageBuffer16BitAccess,
            .shaderDrawParameters = vk11_features.shaderDrawParameters,
        },
        vk::PhysicalDeviceVulkan12Features{
            .samplerMirrorClampToEdge = vk12_features.samplerMirrorClampToEdge,
            .drawIndirectCount = vk12_features.drawIndirectCount,
            .storageBuffer8BitAccess = vk12_features.storageBuffer8BitAccess,
            .uniformAndStorageBuffer8BitAccess = vk12_features.uniformAndStorageBuffer8BitAccess,
            .shaderBufferInt64Atomics = vk12_features.shaderBufferInt64Atomics,
            .shaderSharedInt64Atomics = vk12_features.shaderSharedInt64Atomics,
            .shaderFloat16 = vk12_features.shaderFloat16,
            .shaderInt8 = vk12_features.shaderInt8,
            .descriptorIndexing =
                bindless_supported || (dlss_device && vk12_features.descriptorIndexing),
            .shaderSampledImageArrayNonUniformIndexing = bindless_supported,
            .descriptorBindingSampledImageUpdateAfterBind = bindless_supported,
            .descriptorBindingUpdateUnusedWhilePending = bindless_supported,
            .descriptorBindingPartiallyBound = bindless_supported,
            .runtimeDescriptorArray = bindless_supported,
            .scalarBlockLayout = vk12_features.scalarBlockLayout,
            .uniformBufferStandardLayout = vk12_features.uniformBufferStandardLayout,
            .separateDepthStencilLayouts = vk12_features.separateDepthStencilLayouts,
            .hostQueryReset = vk12_features.hostQueryReset,
            .timelineSemaphore = vk12_features.timelineSemaphore,
            .bufferDeviceAddress = vk12_features.bufferDeviceAddress,
            .shaderOutputLayer = vk12_features.shaderOutputLayer,
        },
        vk::PhysicalDeviceVulkan13Features{
            .robustImageAccess = vk13_features.robustImageAccess,
            .shaderDemoteToHelperInvocation = vk13_features.shaderDemoteToHelperInvocation,
            .subgroupSizeControl = vk13_features.subgroupSizeControl,
            .synchronization2 = vk13_features.synchronization2,
            .dynamicRendering = vk13_features.dynamicRendering,
            .maintenance4 = vk13_features.maintenance4,
        },
        // Extensions
        vk::PhysicalDeviceCustomBorderColorFeaturesEXT{
            .customBorderColors = true,
            .customBorderColorWithoutFormat = true,
        },
        vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT{
            .extendedDynamicState3ColorWriteMask =
                dynamic_state_3_features.extendedDynamicState3ColorWriteMask,
        },
        vk::PhysicalDeviceDepthClipControlFeaturesEXT{
            .depthClipControl = true,
        },
        vk::PhysicalDeviceDepthClipEnableFeaturesEXT{
            .depthClipEnable = true,
        },
        vk::PhysicalDeviceRobustness2FeaturesEXT{
            .robustBufferAccess2 = true,
            .robustImageAccess2 = true,
            .nullDescriptor = true,
        },
        vk::PhysicalDeviceVertexInputDynamicStateFeaturesEXT{
            .vertexInputDynamicState = true,
        },
        vk::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT{
            .primitiveTopologyListRestart = list_restart_features.primitiveTopologyListRestart,
            .primitiveTopologyPatchListRestart =
                list_restart_features.primitiveTopologyPatchListRestart,
        },
        vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR{
            .fragmentShaderBarycentric = true,
        },
        vk::PhysicalDeviceProvokingVertexFeaturesEXT{
            .provokingVertexLast = true,
        },
        vk::PhysicalDeviceVertexAttributeDivisorFeatures{
            .vertexAttributeInstanceRateDivisor = true,
        },
        vk::PhysicalDeviceMaintenance8FeaturesKHR{
            .maintenance8 = true,
        },
        vk::PhysicalDeviceMaintenance7FeaturesKHR{
            .maintenance7 = true,
        },
        vk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT{
            .attachmentFeedbackLoopLayout = true,
        },
        vk::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT{
            .attachmentFeedbackLoopDynamicState = true,
        },
        vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT{
            .shaderBufferFloat32AtomicMinMax =
                shader_atomic_float2_features.shaderBufferFloat32AtomicMinMax,
            .shaderImageFloat32AtomicMinMax =
                shader_atomic_float2_features.shaderImageFloat32AtomicMinMax,
        },
        vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR{
            .workgroupMemoryExplicitLayout =
                workgroup_memory_explicit_layout_features.workgroupMemoryExplicitLayout,
            .workgroupMemoryExplicitLayoutScalarBlockLayout =
                workgroup_memory_explicit_layout_features
                    .workgroupMemoryExplicitLayoutScalarBlockLayout,
            .workgroupMemoryExplicitLayout16BitAccess =
                workgroup_memory_explicit_layout_features.workgroupMemoryExplicitLayout16BitAccess,
        },
        vk::PhysicalDeviceImage2DViewOf3DFeaturesEXT{
            .image2DViewOf3D = image_2d_view_of_3d_features.image2DViewOf3D,
            .sampler2DViewOf3D = image_2d_view_of_3d_features.sampler2DViewOf3D,
        },
        vk::PhysicalDeviceImageViewMinLodFeaturesEXT{
            .minLod = true,
        },
    };

    if (!custom_border_color) {
        device_chain.unlink<vk::PhysicalDeviceCustomBorderColorFeaturesEXT>();
    }
    if (!dynamic_state_3) {
        device_chain.unlink<vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT>();
    }
    if (!depth_clip_control) {
        device_chain.unlink<vk::PhysicalDeviceDepthClipControlFeaturesEXT>();
    }
    if (!depth_clip_enable) {
        device_chain.unlink<vk::PhysicalDeviceDepthClipEnableFeaturesEXT>();
    }
    if (!vertex_input_dynamic_state) {
        device_chain.unlink<vk::PhysicalDeviceVertexInputDynamicStateFeaturesEXT>();
    }
    if (!list_restart) {
        device_chain.unlink<vk::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT>();
    }
    if (!fragment_shader_barycentric) {
        device_chain.unlink<vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR>();
    }
    if (!provoking_vertex) {
        device_chain.unlink<vk::PhysicalDeviceProvokingVertexFeaturesEXT>();
    }
    if (!maintenance_8) {
        device_chain.unlink<vk::PhysicalDeviceMaintenance8FeaturesKHR>();
    }
    if (!maintenance_7) {
        device_chain.unlink<vk::PhysicalDeviceMaintenance7FeaturesKHR>();
    }
    if (!attachment_feedback_loop) {
        device_chain.unlink<vk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT>();
        device_chain.unlink<vk::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT>();
    }
    if (!shader_atomic_float2) {
        device_chain.unlink<vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT>();
    }
    if (!workgroup_memory_explicit_layout) {
        device_chain.unlink<vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR>();
    }
    if (!image_2d_view_of_3d) {
        device_chain.unlink<vk::PhysicalDeviceImage2DViewOf3DFeaturesEXT>();
    }
    if (!image_view_min_lod) {
        device_chain.unlink<vk::PhysicalDeviceImageViewMinLodFeaturesEXT>();
    }

    // The heap feature struct is unknown to the project's Vulkan-Hpp chain traits, so it is
    // spliced into the raw pNext chain by hand, and only in the opt-in mode.
    NativeApi::PhysicalDeviceDescriptorHeapFeatures heap_features = {
        .sType = NativeApi::StructureTypePhysicalDeviceDescriptorHeapFeatures,
        .descriptorHeap = VK_TRUE,
    };
    vk::PhysicalDeviceMaintenance5FeaturesKHR maintenance5_features = {
        .maintenance5 = true,
    };
    if (native_bindings) {
        auto& device_ci = device_chain.get<vk::DeviceCreateInfo>();
        maintenance5_features.pNext = const_cast<void*>(device_ci.pNext);
        heap_features.pNext = &maintenance5_features;
        device_ci.pNext = &heap_features;
    }

    // SHADPS4_DEVICE_FAULT: spliced by hand like the heap features above.
    vk::PhysicalDeviceFaultFeaturesEXT fault_features = {
        .deviceFault = true,
    };
    if (device_fault) {
        auto& device_ci = device_chain.get<vk::DeviceCreateInfo>();
        fault_features.pNext = const_cast<void*>(device_ci.pNext);
        device_ci.pNext = &fault_features;
    }
    // SHADPS4_ADDR_BIND_REPORT: the same splice.
    vk::PhysicalDeviceAddressBindingReportFeaturesEXT bind_report_features = {
        .reportAddressBinding = true,
    };
    if (addr_bind_report) {
        auto& device_ci = device_chain.get<vk::DeviceCreateInfo>();
        bind_report_features.pNext = const_cast<void*>(device_ci.pNext);
        device_ci.pNext = &bind_report_features;
    }

    if (dlss_device) {
        // SHADPS4_DLSS: try the device once with Streamline's extensions (a raw call: no result
        // assert). A driver that refuses them gets the device without them - the game's own
        // anti-aliasing - never a failed start. Measured 10-17 (driver 617.14): VK_NVX_binary_import
        // made vkCreateDevice crash here while the guest held all address space below 1 TB.
        LOG_INFO(Render_Vulkan, "DLSS: creating the device with {} extensions", enabled_extensions.size());
        std::fprintf(stderr, "DLSS-MARK probe start (%u extensions)\n",
                     static_cast<unsigned>(enabled_extensions.size()));
        std::fflush(stderr);
        // VK_NVX_binary_import needs ~56 GB of free address space below 1 TB (the driver crashes
        // without it); the address space has held it out of the guest's map until now.
        Core::ReleaseDriverVaHole();
        VkDevice probe = VK_NULL_HANDLE;
        const VkResult probe_result = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateDevice(
            static_cast<VkPhysicalDevice>(physical_device),
            reinterpret_cast<const VkDeviceCreateInfo*>(&device_chain.get<vk::DeviceCreateInfo>()),
            nullptr, &probe);
        std::fprintf(stderr, "DLSS-MARK probe vkCreateDevice -> %d\n", static_cast<int>(probe_result));
        std::fflush(stderr);
        LOG_INFO(Render_Vulkan, "DLSS: device probe {}", vk::to_string(vk::Result(probe_result)));
        if (probe_result == VK_SUCCESS) {
            VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyDevice(probe, nullptr);
        } else {
            for (const char* ext : Dlss::DeviceExtensions()) {
                if (std::strcmp(ext, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME) == 0) {
                    continue; // the emulator's own
                }
                enabled_extensions.erase(
                    std::remove_if(enabled_extensions.begin(), enabled_extensions.end(),
                                   [ext](const char* e) { return std::strcmp(e, ext) == 0; }),
                    enabled_extensions.end());
            }
            auto& dci = device_chain.get<vk::DeviceCreateInfo>();
            dci.enabledExtensionCount = static_cast<u32>(enabled_extensions.size());
            dci.ppEnabledExtensionNames = enabled_extensions.data();
            for (const char* ext : Dlss5::DeviceExtensions()) {
                enabled_extensions.erase(
                    std::remove_if(enabled_extensions.begin(), enabled_extensions.end(),
                                   [ext](const char* e) { return std::strcmp(e, ext) == 0; }),
                    enabled_extensions.end());
            }
            dci.enabledExtensionCount = static_cast<u32>(enabled_extensions.size());
            dci.ppEnabledExtensionNames = enabled_extensions.data();
            dlss_device = false;
            Dlss5::Disable("the driver refused the DLSS device extensions");
            LOG_WARNING(Render_Vulkan, "DLSS: the driver refused the device with the DLSS extensions "
                                       "- DLSS off, the game's own anti-aliasing stays");
        }
    }
    auto [device_result, dev] = physical_device.createDeviceUnique(device_chain.get());
    if (dlss_device) {
        LOG_INFO(Render_Vulkan, "DLSS: device created ({})", vk::to_string(device_result));
    }
    if (device_result != vk::Result::eSuccess) {
        LOG_CRITICAL(Render_Vulkan, "Failed to create device: {}", vk::to_string(device_result));
        return false;
    }
    device = std::move(dev);

    VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);

    if (native_bindings) {
        const auto load = [&](const char* name) {
            return VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr(static_cast<VkDevice>(*device),
                                                                     name);
        };
        native_api.write_sampler_descriptors =
            reinterpret_cast<NativeApi::PfnWriteSamplerDescriptors>(
                load("vkWriteSamplerDescriptorsEXT"));
        native_api.write_resource_descriptors =
            reinterpret_cast<NativeApi::PfnWriteResourceDescriptors>(
                load("vkWriteResourceDescriptorsEXT"));
        native_api.cmd_bind_sampler_heap =
            reinterpret_cast<NativeApi::PfnCmdBindHeap>(load("vkCmdBindSamplerHeapEXT"));
        native_api.cmd_bind_resource_heap =
            reinterpret_cast<NativeApi::PfnCmdBindHeap>(load("vkCmdBindResourceHeapEXT"));
        native_api.cmd_push_data =
            reinterpret_cast<NativeApi::PfnCmdPushData>(load("vkCmdPushDataEXT"));
        native_api.register_custom_border_color =
            reinterpret_cast<NativeApi::PfnRegisterCustomBorderColor>(
                load("vkRegisterCustomBorderColorEXT"));
        native_api.unregister_custom_border_color =
            reinterpret_cast<NativeApi::PfnUnregisterCustomBorderColor>(
                load("vkUnregisterCustomBorderColorEXT"));
        native_bindings_active =
            native_api.IsComplete() &&
            (!IsCustomBorderColorSupported() ||
             (native_api.register_custom_border_color && native_api.unregister_custom_border_color));
        if (native_bindings_active) {
            LOG_INFO(Render_Vulkan, "NativeBindings: descriptor heap guest bindings enabled");
        } else {
            // The extension and feature stay enabled on the device but are never used.
            LOG_WARNING(Render_Vulkan,
                        "NativeBindings: descriptor heap entry points missing, using legacy");
        }
    }

    graphics_queue = device->getQueue(queue_family_index, 0);
    present_queue = device->getQueue(queue_family_index, 0);
    if (dlss_device || Dlss::Mode() == 2) {
        Dlss::DeviceCreated(*instance, physical_device, *device, queue_family_index, 0);
    }
    if (dlss_device && Dlss::g_ready.load(std::memory_order_relaxed)) {
        Dlss5::DeviceCreated(*instance, physical_device, *device);
    } else if (Dlss::Mode() == 1) {
        Dlss5::Disable("DLSS 4.5 did not start");
    }
    if (has_readback_queue) {
        const u32 index = readback_queue_family == queue_family_index ? 1u : 0u;
        readback_queue = device->getQueue(readback_queue_family, index);
        if (has_asc_queue || has_asc_lane) {
            asc_queue = device->getQueue(readback_queue_family, 1);
            LOG_INFO(Render_Vulkan, "AscStream: guest async compute on queue 1 of family {}",
                     readback_queue_family);
        }
        if (has_retry_queue) {
            retry_queue = device->getQueue(readback_queue_family, retry_index);
            LOG_INFO(Render_Vulkan, "ArenaRetryQueue: retry copies on queue {} of family {}",
                     retry_index, readback_queue_family);
        }
        LOG_INFO(Render_Vulkan, "ReadbackQueue: using queue {} of family {}", index,
                 readback_queue_family);
    }

    if (calibrated_timestamps) {
        const auto [time_domains_result, time_domains] =
            physical_device.getCalibrateableTimeDomainsEXT();
        if (time_domains_result == vk::Result::eSuccess) {
#if _WIN64
            const bool has_host_time_domain =
                std::find(time_domains.cbegin(), time_domains.cend(),
                          vk::TimeDomainEXT::eQueryPerformanceCounter) != time_domains.cend();
#elif __linux__
            const bool has_host_time_domain =
                std::find(time_domains.cbegin(), time_domains.cend(),
                          vk::TimeDomainEXT::eClockMonotonicRaw) != time_domains.cend();
#else
            // Tracy limitation means only Windows and Linux can use host time domain.
            // https://github.com/shadps4-emu/tracy/blob/c6d779d78508514102fbe1b8eb28bda10d95bb2a/public/tracy/TracyVulkan.hpp#L384-L389
            const bool has_host_time_domain = false;
#endif
            if (has_host_time_domain) {
                static constexpr std::string_view context_name{"vk_rasterizer"};
                profiler_context = TracyVkContextHostCalibrated(
                    *instance, physical_device, *device,
                    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
                    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr);
                TracyVkContextName(profiler_context, context_name.data(), context_name.size());
            }
        } else {
            LOG_WARNING(Render_Vulkan, "Could not query calibrated time domains for profiling: {}",
                        vk::to_string(time_domains_result));
        }
    }

    CreateAllocator();
    return true;
}

bool Instance::QueryNativeBindings() {
    const char* requested = std::getenv("SHADPS4_NATIVE_BINDINGS");
    if (!requested || std::string_view{requested} != "1") {
        return false;
    }
    const auto has_extension = [&](std::string_view name) {
        return std::ranges::any_of(available_extensions,
                                   [&](const std::string& ext) { return ext == name; });
    };
    for (const char* name :
         {NativeApi::DescriptorHeapExtensionName, NativeApi::Maintenance5ExtensionName,
          NativeApi::ShaderUntypedPointersExtensionName}) {
        if (!has_extension(name)) {
            LOG_WARNING(Render_Vulkan, "NativeBindings: {} unavailable, using legacy", name);
            return false;
        }
    }

    NativeApi::PhysicalDeviceDescriptorHeapFeatures heap_features = {
        .sType = NativeApi::StructureTypePhysicalDeviceDescriptorHeapFeatures,
    };
    vk::PhysicalDeviceMaintenance5FeaturesKHR maintenance5_features{};
    maintenance5_features.pNext = &heap_features;
    vk::PhysicalDeviceVulkan12Features vk12{};
    vk12.pNext = &maintenance5_features;
    vk::PhysicalDeviceFeatures2 features2{};
    features2.pNext = &vk12;
    physical_device.getFeatures2(&features2);
    if (!heap_features.descriptorHeap || !maintenance5_features.maintenance5 ||
        !vk12.bufferDeviceAddress) {
        LOG_WARNING(Render_Vulkan,
                    "NativeBindings: descriptorHeap={} maintenance5={} bufferDeviceAddress={}, "
                    "using legacy",
                    heap_features.descriptorHeap, maintenance5_features.maintenance5,
                    vk12.bufferDeviceAddress);
        return false;
    }

    NativeApi::PhysicalDeviceDescriptorHeapProperties props = {
        .sType = NativeApi::StructureTypePhysicalDeviceDescriptorHeapProperties,
    };
    vk::PhysicalDeviceProperties2 properties2{};
    properties2.pNext = &props;
    physical_device.getProperties2(&properties2);
    LOG_INFO(Render_Vulkan,
             "NativeBindings: resourceHeap max={:#x} align={} reserved={:#x}, samplerHeap "
             "max={:#x} align={} reserved={:#x}, descriptor size/align sampler={}/{} image={}/{} "
             "buffer={}/{}, maxPushDataSize={}",
             props.maxResourceHeapSize, props.resourceHeapAlignment,
             props.minResourceHeapReservedRange, props.maxSamplerHeapSize,
             props.samplerHeapAlignment, props.minSamplerHeapReservedRange,
             props.samplerDescriptorSize, props.samplerDescriptorAlignment,
             props.imageDescriptorSize, props.imageDescriptorAlignment, props.bufferDescriptorSize,
             props.bufferDescriptorAlignment, props.maxPushDataSize);

    if (props.maxPushDataSize < NativePushBlockSize) {
        LOG_WARNING(Render_Vulkan, "NativeBindings: maxPushDataSize {} < {}, using legacy",
                    props.maxPushDataSize, NativePushBlockSize);
        return false;
    }
    if (!props.samplerDescriptorSize || !props.imageDescriptorSize || !props.bufferDescriptorSize ||
        !props.samplerDescriptorAlignment || !props.imageDescriptorAlignment ||
        !props.bufferDescriptorAlignment || !props.resourceHeapAlignment || !props.samplerHeapAlignment) {
        LOG_WARNING(Render_Vulkan, "NativeBindings: zero descriptor size/alignment, using legacy");
        return false;
    }

    const auto stride = [](u64 size, u64 alignment) {
        return Common::AlignUp(size, std::max<u64>(alignment, 1));
    };
    auto& layout = native_binding_layout;
    layout = {};
    layout.max_push_data_size = props.maxPushDataSize;

    // Resource heap: [reserved][persistent image slots][static + ring buffer slots].
    constexpr u64 MaxResourceHeap = 32ULL << 20;
    constexpr u64 MaxImageSlots = 1ULL << 18;
    constexpr u64 MinImageSlots = 4096;
    const u64 region_alignment =
        std::max<u64>({props.imageDescriptorAlignment, props.bufferDescriptorAlignment, 64});
    const u64 heap_alignment = std::max<u64>(props.resourceHeapAlignment, 1);
    const u64 heap_budget =
        Common::AlignDown(std::min(props.maxResourceHeapSize, MaxResourceHeap), heap_alignment);
    const u64 image_stride = stride(props.imageDescriptorSize, props.imageDescriptorAlignment);
    const u64 buffer_stride = stride(props.bufferDescriptorSize, props.bufferDescriptorAlignment);
    const u64 buffer_slots = NativeBufferStaticSlots + u64(NativeBufferBlocks) * NativeBufferBlockSlots;
    const u64 buffer_bytes = Common::AlignUp(buffer_slots * buffer_stride, region_alignment);
    const u64 image_base = Common::AlignUp(props.minResourceHeapReservedRange, region_alignment);
    if (image_base + buffer_bytes + region_alignment >= heap_budget) {
        LOG_WARNING(Render_Vulkan, "NativeBindings: resource heap too small, using legacy");
        return false;
    }
    const u64 image_slots = std::min(
        (heap_budget - image_base - buffer_bytes - region_alignment) / image_stride, MaxImageSlots);
    if (image_slots < MinImageSlots) {
        LOG_WARNING(Render_Vulkan, "NativeBindings: only {} image slots fit, using legacy",
                    image_slots);
        return false;
    }
    const u64 buffer_base =
        Common::AlignUp(image_base + image_slots * image_stride, region_alignment);
    layout.resource_heap_size = Common::AlignUp(buffer_base + buffer_bytes, heap_alignment);
    ASSERT(layout.resource_heap_size <= heap_budget);
    layout.resource_reserved_size = props.minResourceHeapReservedRange;
    layout.resource_heap_alignment = heap_alignment;
    layout.image_base = u32(image_base);
    layout.image_stride = u32(image_stride);
    layout.image_descriptor_size = u32(props.imageDescriptorSize);
    layout.image_slots = u32(image_slots);
    layout.buffer_base = u32(buffer_base);
    layout.buffer_stride = u32(buffer_stride);
    layout.buffer_descriptor_size = u32(props.bufferDescriptorSize);
    layout.buffer_slots = u32(buffer_slots);

    // Sampler heap: [reserved][persistent sampler slots]. No embedded samplers are used.
    constexpr u64 MaxSamplerHeap = 1ULL << 20;
    constexpr u64 MinSamplerSlots = 256;
    const u64 sampler_alignment = std::max<u64>(props.samplerHeapAlignment, 1);
    const u64 sampler_budget =
        Common::AlignDown(std::min(props.maxSamplerHeapSize, MaxSamplerHeap), sampler_alignment);
    const u64 sampler_stride =
        stride(props.samplerDescriptorSize, props.samplerDescriptorAlignment);
    const u64 sampler_base = Common::AlignUp(props.minSamplerHeapReservedRange, sampler_stride);
    const u64 sampler_slots =
        sampler_budget > sampler_base ? (sampler_budget - sampler_base) / sampler_stride : 0;
    if (sampler_slots < MinSamplerSlots) {
        LOG_WARNING(Render_Vulkan, "NativeBindings: only {} sampler slots fit, using legacy",
                    sampler_slots);
        return false;
    }
    layout.sampler_heap_size = sampler_budget;
    layout.sampler_reserved_size = props.minSamplerHeapReservedRange;
    layout.sampler_heap_alignment = sampler_alignment;
    layout.sampler_base = u32(sampler_base);
    layout.sampler_stride = u32(sampler_stride);
    layout.sampler_descriptor_size = u32(props.samplerDescriptorSize);
    layout.sampler_slots = u32(sampler_slots);
    if (layout.sampler_slots < GetMaxSamplerAllocationCount()) {
        // Correct but slow: exhaustion is handled by a full rollover.
        LOG_WARNING(Render_Vulkan,
                    "NativeBindings: {} sampler slots is below the sampler cache limit {}",
                    layout.sampler_slots, GetMaxSamplerAllocationCount());
    }

    LOG_INFO(Render_Vulkan,
             "NativeBindings: resource heap {:#x} bytes (images {} x {} at {:#x}, buffers {} x {} "
             "at {:#x}), sampler heap {:#x} bytes ({} x {} at {:#x})",
             layout.resource_heap_size, layout.image_slots, layout.image_stride, layout.image_base,
             layout.buffer_slots, layout.buffer_stride, layout.buffer_base,
             layout.sampler_heap_size, layout.sampler_slots, layout.sampler_stride,
             layout.sampler_base);
    return true;
}

void Instance::LogDeviceFault() const {
    if (!device_fault) {
        LOG_CRITICAL(Render_Vulkan, "DeviceFault: no report (set SHADPS4_DEVICE_FAULT=1)");
        return;
    }
    vk::DeviceFaultCountsEXT counts{};
    const auto counted = device->getFaultInfoEXT(&counts, nullptr);
    std::vector<vk::DeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
    std::vector<vk::DeviceFaultVendorInfoEXT> vendors(counts.vendorInfoCount);
    counts.vendorBinarySize = 0;
    vk::DeviceFaultInfoEXT info{};
    info.pAddressInfos = addresses.empty() ? nullptr : addresses.data();
    info.pVendorInfos = vendors.empty() ? nullptr : vendors.data();
    info.pVendorBinaryData = nullptr;
    const auto result = device->getFaultInfoEXT(&counts, &info);
    LOG_CRITICAL(Render_Vulkan,
                 "DeviceFault: count result {} info result {} description '{}' addresses {} "
                 "vendor infos {}",
                 vk::to_string(counted), vk::to_string(result),
                 std::string_view{info.description.data()}, counts.addressInfoCount,
                 counts.vendorInfoCount);
    for (u32 i = 0; i < counts.addressInfoCount && i < addresses.size(); ++i) {
        LOG_CRITICAL(Render_Vulkan, "DeviceFault: address {} type {} reported {:#x} precision {:#x}",
                     i, vk::to_string(addresses[i].addressType), addresses[i].reportedAddress,
                     addresses[i].addressPrecision);
        if (addr_bind_report) {
            DumpAddressBindings(addresses[i].reportedAddress);
        }
    }
    for (u32 i = 0; i < counts.vendorInfoCount && i < vendors.size(); ++i) {
        LOG_CRITICAL(Render_Vulkan, "DeviceFault: vendor {} '{}' code {:#x} data {:#x}", i,
                     std::string_view{vendors[i].description.data()}, vendors[i].vendorFaultCode,
                     vendors[i].vendorFaultData);
    }
}

void Instance::CreateAllocator() {
    const VmaVulkanFunctions functions = {
        .vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr,
    };

    // SHADPS4_VMA_BLOCK_MB (PERF-PLAN 11, 2026-10-20): VMA's block size for large heaps (default
    // 256 MB). City 1080p (s1456): 620 MB of the 6.9 GB VMA holds was empty space inside blocks
    // (video 340 MB of 736, the host-visible video type 282 MB of 480).
    u64 block_bytes = 0;
    if (const char* v = std::getenv("SHADPS4_VMA_BLOCK_MB"); v && std::atoi(v) >= 16) {
        block_bytes = u64(std::min(std::atoi(v), 1024)) << 20;
    }
    const VmaAllocatorCreateInfo allocator_info = {
        .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = physical_device,
        .device = *device,
        .preferredLargeHeapBlockSize = block_bytes,
        .pVulkanFunctions = &functions,
        .instance = *instance,
        .vulkanApiVersion = TargetVulkanApiVersion,
    };

    const VkResult result = vmaCreateAllocator(&allocator_info, &allocator);
    if (result != VK_SUCCESS) {
        UNREACHABLE_MSG("Failed to initialize VMA with error {}",
                        vk::to_string(vk::Result{result}));
    }
}

void Instance::CollectDeviceParameters() {
    const vk::StructureChain property_chain =
        physical_device
            .getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceDriverProperties>();
    const vk::PhysicalDeviceDriverProperties driver =
        property_chain.get<vk::PhysicalDeviceDriverProperties>();

    driver_id = driver.driverID;
    vendor_name = driver.driverName.data();

    const std::string model_name{GetModelName()};
    const std::string driver_version = GetDriverVersionName();
    const std::string driver_name = fmt::format("{} {}", vendor_name, driver_version);
    const std::string api_version = GetReadableVersion(properties.apiVersion);
    const std::string extensions = fmt::format("{}", fmt::join(available_extensions, ", "));

    LOG_INFO(Render_Vulkan, "GPU_Vendor: {}", vendor_name);
    LOG_INFO(Render_Vulkan, "GPU_Model: {}", model_name);
    LOG_INFO(Render_Vulkan, "GPU_Integrated: {}", IsIntegrated() ? "Yes" : "No");
    LOG_INFO(Render_Vulkan, "GPU_Vulkan_Driver: {}", driver_name);
    LOG_INFO(Render_Vulkan, "GPU_Vulkan_Version: {}", api_version);
    LOG_INFO(Render_Vulkan, "GPU_Vulkan_Extensions: {}", extensions);
}

void Instance::CollectPhysicalMemoryInfo() {
    vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    vk::PhysicalDeviceMemoryProperties2 props = {
        .pNext = supports_memory_budget ? &budget : nullptr,
    };
    physical_device.getMemoryProperties2(&props);
    const auto& memory_props = props.memoryProperties;
    const size_t num_props = memory_props.memoryHeapCount;
    total_memory_budget = 0;
    u64 device_initial_usage = 0;
    u64 local_memory = 0;
    for (size_t i = 0; i < num_props; ++i) {
        const bool is_device_local =
            (memory_props.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) !=
            vk::MemoryHeapFlags{};
        if (!IsIntegrated() && !is_device_local) {
            // Ignore non-device local memory on discrete GPUs.
            continue;
        }
        valid_heaps.push_back(i);
        if (is_device_local) {
            local_memory += memory_props.memoryHeaps[i].size;
        }
        if (supports_memory_budget) {
            device_initial_usage += budget.heapUsage[i];
            total_memory_budget += budget.heapBudget[i];
            continue;
        }
        // If memory budget is not supported, use the size of the heap as the budget.
        total_memory_budget += memory_props.memoryHeaps[i].size;
    }
    if (!IsIntegrated()) {
        // We reserve some memory for the system.
        const u64 system_memory = std::min<u64>(total_memory_budget / 8, 1_GB);
        total_memory_budget -= system_memory;
        return;
    }
    // Leave at least 8 GB for the system on integrated GPUs.
    const s64 available_memory = static_cast<s64>(total_memory_budget - device_initial_usage);
    total_memory_budget =
        static_cast<u64>(std::max<s64>(available_memory - 8_GB, static_cast<s64>(local_memory)));
}

void Instance::CreateSimBallast() {
    // SHADPS4_SIM_VRAM_MB=N (test switch, v1.0.2): run as if the card had N MiB of video memory.
    // VMA's budget here is 80% of the heap size against VMA's own allocations (vk_memory_budget.h),
    // so a ballast of 80% of (real heap - N), allocated through VMA, leaves the game exactly the
    // VMA budget an N MiB card has (7956 MiB -> 6365 MiB, the 8 GB player's ceiling). The buffer
    // cache fills the ballast once on the GPU, so it also takes real video memory (the driver's
    // heap usage counts it, which the texture / buffer cache GC thresholds read).
    const char* e = std::getenv("SHADPS4_SIM_VRAM_MB");
    if (!e || !*e) {
        return;
    }
    const u64 sim_mb = std::strtoull(e, nullptr, 10);
    u64 heap = 0;
    for (u32 i = 0; i < memory_properties.memoryHeapCount; ++i) {
        if (memory_properties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
            heap = std::max<u64>(heap, memory_properties.memoryHeaps[i].size);
        }
    }
    if (sim_mb == 0 || (sim_mb << 20) >= heap) {
        LOG_WARNING(Render_Vulkan, "SHADPS4_SIM_VRAM_MB={}: not below the real {} MiB, ignored",
                    sim_mb, heap >> 20);
        return;
    }
    sim_vram_bytes = sim_mb << 20;
    const u64 ballast = (heap - sim_vram_bytes) * 8 / 10;
    constexpr u64 Chunk = 512ull << 20;
    u64 made = 0;
    while (made < ballast) {
        const u64 size = std::min(Chunk, ballast - made);
        const VkBufferCreateInfo ci = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        const VmaAllocationCreateInfo aci = {
            .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        };
        VkBuffer buffer{};
        VmaAllocation allocation{};
        if (vmaCreateBuffer(allocator, &ci, &aci, &buffer, &allocation, nullptr) != VK_SUCCESS) {
            break;
        }
        sim_ballast_buffers.push_back(vk::Buffer{buffer});
        sim_ballast_allocations.push_back(allocation);
        made += size;
    }
    LOG_WARNING(Render_Vulkan,
                "SHADPS4_SIM_VRAM_MB={}: TEST MODE - real heap {} MiB, ballast {} MiB of {} wanted "
                "({} buffers) through VMA; VMA budget left for the game {} MiB",
                sim_mb, heap >> 20, made >> 20, ballast >> 20, sim_ballast_buffers.size(),
                ((heap * 8 / 10) - made) >> 20);
}

void Instance::CollectImageFormatInfo() {
    // Check for block texel view support using basic image format info.
    const vk::PhysicalDeviceImageFormatInfo2 block_texel_view_info{
        .format = vk::Format::eBc1RgbaUnormBlock,
        .type = vk::ImageType::e2D,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eSampled,
        .flags = vk::ImageCreateFlagBits::eBlockTexelViewCompatible,
    };
    const auto block_texel_view_props =
        physical_device.getImageFormatProperties2(block_texel_view_info);
    supports_block_texel_view = block_texel_view_props.result == vk::Result::eSuccess;
    LOG_INFO(Render_Vulkan, "Block Texel View support: {}",
             supports_block_texel_view ? "Yes" : "No");

    // Check and log format support details.
    for (const auto& format : LiverpoolToVK::SurfaceFormats()) {
        if (!IsFormatSupported(format.vk_format, format.flags)) {
            LOG_WARNING(Render_Vulkan,
                        "Surface format data_format={}, number_format={} is not fully supported "
                        "(vk_format={}, missing features={})",
                        static_cast<u32>(format.data_format),
                        static_cast<u32>(format.number_format), vk::to_string(format.vk_format),
                        vk::to_string(format.flags & ~GetFormatFeatureFlags(format.vk_format)));
        }
    }
    for (const auto& format : LiverpoolToVK::DepthFormats()) {
        if (!IsFormatSupported(format.vk_format, format.flags)) {
            LOG_WARNING(Render_Vulkan,
                        "Depth format z_format={}, stencil_format={} is not fully supported "
                        "(vk_format={}, missing features={})",
                        static_cast<u32>(format.z_format), static_cast<u32>(format.stencil_format),
                        vk::to_string(format.vk_format),
                        vk::to_string(format.flags & ~GetFormatFeatureFlags(format.vk_format)));
        }
    }
}

void Instance::CollectToolingInfo() const {
    if (driver_id == vk::DriverId::eAmdProprietary ||
        driver_id == vk::DriverId::eIntelProprietaryWindows) {
        // AMD: Causes issues with Reshade.
        // Intel: Causes crash on start.
        return;
    }
    const auto [tools_result, tools] = physical_device.getToolProperties();
    if (tools_result != vk::Result::eSuccess) {
        LOG_ERROR(Render_Vulkan, "Could not get Vulkan tool properties: {}",
                  vk::to_string(tools_result));
        return;
    }
    for (const vk::PhysicalDeviceToolProperties& tool : tools) {
        const std::string_view name = tool.name;
        LOG_INFO(Render_Vulkan, "Attached debugging tool: {}", name);
    }
}

u64 Instance::GetDeviceMemoryUsage() const {
    vk::PhysicalDeviceMemoryBudgetPropertiesEXT memory_budget_props{};
    vk::PhysicalDeviceMemoryProperties2 props = {
        .pNext = &memory_budget_props,
    };
    physical_device.getMemoryProperties2(&props);

    u64 total_usage = 0;
    for (const size_t heap : valid_heaps) {
        total_usage += memory_budget_props.heapUsage[heap];
    }
    return total_usage;
}

u64 Instance::GetDeviceMemoryBudgetNow() const {
    if (!supports_memory_budget) {
        return 0;
    }
    vk::PhysicalDeviceMemoryBudgetPropertiesEXT memory_budget_props{};
    vk::PhysicalDeviceMemoryProperties2 props = {
        .pNext = &memory_budget_props,
    };
    physical_device.getMemoryProperties2(&props);

    u64 total_budget = 0;
    for (const size_t heap : valid_heaps) {
        total_budget += memory_budget_props.heapBudget[heap];
    }
    return total_budget;
}

vk::FormatFeatureFlags2 Instance::GetFormatFeatureFlags(vk::Format format) const {
    const auto it = format_properties.find(format);
    if (it == format_properties.end()) {
        UNIMPLEMENTED_MSG("Properties of format {} have not been queried.", vk::to_string(format));
    }

    return it->second.optimalTilingFeatures | it->second.bufferFeatures;
}

bool Instance::IsFormatSupported(const vk::Format format,
                                 const vk::FormatFeatureFlags2 flags) const {
    if (format == vk::Format::eUndefined) [[unlikely]] {
        return true;
    }
    return (GetFormatFeatureFlags(format) & flags) == flags;
}

vk::Format Instance::GetSupportedFormat(const vk::Format format,
                                        const vk::FormatFeatureFlags2 flags) const {
    if (!IsFormatSupported(format, flags)) [[unlikely]] {
        switch (format) {
        case vk::Format::eD16UnormS8Uint:
            if (IsFormatSupported(vk::Format::eD24UnormS8Uint, flags)) {
                return vk::Format::eD24UnormS8Uint;
            }
            if (IsFormatSupported(vk::Format::eD32SfloatS8Uint, flags)) {
                return vk::Format::eD32SfloatS8Uint;
            }
            break;
        case vk::Format::eR8Srgb:
            if (IsFormatSupported(vk::Format::eR8Unorm, flags)) {
                return vk::Format::eR8Unorm;
            }
            break;
        default:
            break;
        }
    }
    return format;
}

} // namespace Vulkan
