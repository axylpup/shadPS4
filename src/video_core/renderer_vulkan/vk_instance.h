// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>
#include <unordered_map>

#include "video_core/renderer_vulkan/vk_native_bindings_api.h"
#include "video_core/renderer_vulkan/vk_platform.h"

#define TRACY_VK_USE_SYMBOL_TABLE
#include <tracy/TracyVulkan.hpp>

namespace Frontend {
class WindowSDL;
}

VK_DEFINE_HANDLE(VmaAllocator)
VK_DEFINE_HANDLE(VmaAllocation)

namespace Vulkan {

class Instance {
public:
    explicit Instance(bool validation = false, bool crash_diagnostic = false);
    explicit Instance(Frontend::WindowSDL& window, s32 physical_device_index,
                      bool enable_validation = false, bool enable_crash_diagnostic = false);
    ~Instance();

    /// Returns a formatted string for the driver version
    std::string GetDriverVersionName();

    /// Gets a compatibility format if the format is not supported.
    [[nodiscard]] vk::Format GetSupportedFormat(vk::Format format,
                                                vk::FormatFeatureFlags2 flags) const;

    /// Returns the Vulkan instance
    vk::Instance GetInstance() const {
        return *instance;
    }

    /// Returns the current physical device
    vk::PhysicalDevice GetPhysicalDevice() const {
        return physical_device;
    }

    /// Returns the Vulkan device
    vk::Device GetDevice() const {
        return *device;
    }

    /// Returns the VMA allocator handle
    VmaAllocator GetAllocator() const {
        return allocator;
    }

    /// Returns a list of the available physical devices
    std::span<const vk::PhysicalDevice> GetPhysicalDevices() const {
        return physical_devices;
    }

    /// Retrieve queue information
    u32 GetGraphicsQueueFamilyIndex() const {
        return queue_family_index;
    }

    u32 GetPresentQueueFamilyIndex() const {
        return queue_family_index;
    }

    vk::Queue GetGraphicsQueue() const {
        return graphics_queue;
    }

    vk::Queue GetPresentQueue() const {
        return present_queue;
    }

    /// Second queue of the graphics family, when SHADPS4_READBACK_QUEUE=1 and available.
    bool HasReadbackQueue() const {
        return has_readback_queue;
    }

    vk::Queue GetReadbackQueue() const {
        return readback_queue;
    }

    /// SHADPS4_READBACK_QUEUE=3: the readback queue is in another family, so buffers are
    /// created VK_SHARING_MODE_CONCURRENT between the graphics and readback families.
    /// SHADPS4_ASC_STREAM: second queue of the compute-only family for guest async compute.
    bool HasAscQueue() const {
        return has_asc_queue;
    }
    /// SHADPS4_ASC_LANE: queue 1 of the compute-only family for the back's compute lane.
    bool HasAscLaneQueue() const {
        return has_asc_lane;
    }
    vk::Queue GetAscQueue() const {
        return asc_queue;
    }
    /// SHADPS4_ARENA_RETRY_QUEUE=1 (r18): one more queue of the compute-only family for the
    /// arena prefetch retry copies, so a copy waiting for its producer never blocks fault copies.
    bool HasRetryQueue() const {
        return has_retry_queue;
    }
    vk::Queue GetRetryQueue() const {
        return retry_queue;
    }
    bool BuffersShareReadbackFamily() const {
        return readback_family_concurrent;
    }
    u32 GetReadbackQueueFamilyIndex() const {
        return readback_queue_family;
    }

    TracyVkCtx GetProfilerContext() const {
        return profiler_context;
    }

    /// Returns true if anisotropic filtering is supported
    bool IsAnisotropicFilteringSupported() const {
        return features.samplerAnisotropy;
    }

    /// SHADPS4_DEVICE_FAULT=1: logs the driver's VK_EXT_device_fault report (call on a device
    /// loss, before asserting). Says so when the extension is not enabled.
    void LogDeviceFault() const;

    /// Returns true if depth bounds testing is supported
    bool IsDepthBoundsSupported() const {
        return features.depthBounds;
    }

    /// Occlusion queries that return the exact sample count (SHADPS4_OCCLUSION_QUERIES).
    bool IsOcclusionQueryPreciseSupported() const {
        return features.occlusionQueryPrecise;
    }

    /// vkResetQueryPool from the host (SHADPS4_OCCLUSION_QUERIES).
    bool IsHostQueryResetSupported() const {
        return vk12_features.hostQueryReset;
    }

    /// Returns true if 16-bit floats are supported in shaders
    bool IsShaderFloat16Supported() const {
        return vk12_features.shaderFloat16;
    }

    /// Returns true if 64-bit floats are supported in shaders
    bool IsShaderFloat64Supported() const {
        return features.shaderFloat64;
    }

    /// Returns true if 64-bit ints are supported in shaders
    bool IsShaderInt64Supported() const {
        return features.shaderInt64;
    }

    /// Returns true if 16-bit ints are supported in shaders
    bool IsShaderInt16Supported() const {
        return features.shaderInt16;
    }

    /// Returns true if 8-bit ints are supported in shaders
    bool IsShaderInt8Supported() const {
        return vk12_features.shaderInt8;
    }

    /// Returns true if VK_KHR_maintenance8 is supported
    bool IsMaintenance8Supported() const {
        return maintenance_8;
    }
    bool IsMaintenance7Supported() const {
        return maintenance_7;
    }

    /// Returns true if VK_EXT_attachment_feedback_loop_layout is supported
    bool IsAttachmentFeedbackLoopLayoutSupported() const {
        return attachment_feedback_loop;
    }

    /// Returns true when VK_EXT_custom_border_color is supported
    bool IsCustomBorderColorSupported() const {
        return custom_border_color;
    }

    /// Returns true when VK_EXT_shader_stencil_export is supported
    bool IsShaderStencilExportSupported() const {
        return shader_stencil_export;
    }

    /// Returns true when VK_EXT_depth_clip_control is supported
    bool IsDepthClipControlSupported() const {
        return depth_clip_control;
    }

    /// Returns true when VK_EXT_depth_clip_enable is supported
    bool IsDepthClipEnableSupported() const {
        return depth_clip_enable;
    }

    /// Returns true when VK_EXT_depth_range_unrestricted is supported
    bool IsDepthRangeUnrestrictedSupported() const {
        return depth_range_unrestricted;
    }

    /// Returns true when VK_EXT_extended_dynamic_state3 is supported
    bool IsExtendedDynamicState3Supported() const {
        return dynamic_state_3;
    }

    /// Returns true when the extendedDynamicState3ColorWriteMask feature of
    /// VK_EXT_extended_dynamic_state3 is supported.
    bool IsDynamicColorWriteMaskSupported() const {
        return dynamic_state_3 && dynamic_state_3_features.extendedDynamicState3ColorWriteMask;
    }

    /// Returns true when VK_EXT_vertex_input_dynamic_state is supported.
    bool IsVertexInputDynamicState() const {
        return vertex_input_dynamic_state;
    }

    /// Returns true when VK_KHR_fragment_shader_barycentric is supported.
    bool IsFragmentShaderBarycentricSupported() const {
        return fragment_shader_barycentric;
    }

    /// Returns true when VK_AMD_shader_explicit_vertex_parameter is supported.
    bool IsAmdShaderExplicitVertexParameterSupported() const {
        return amd_shader_explicit_vertex_parameter;
    }

    /// Returns true when VK_EXT_primitive_topology_list_restart is supported for regular lists.
    bool IsListRestartSupported() const {
        return list_restart && list_restart_features.primitiveTopologyListRestart;
    }

    /// Returns true when VK_EXT_primitive_topology_list_restart is supported for patch lists.
    bool IsPatchListRestartSupported() const {
        return list_restart && list_restart_features.primitiveTopologyPatchListRestart;
    }

    /// Returns true when VK_EXT_provoking_vertex is supported.
    bool IsProvokingVertexSupported() const {
        return provoking_vertex;
    }

    /// Returns true when VK_AMD_shader_image_load_store_lod is supported.
    bool IsImageLoadStoreLodSupported() const {
        return image_load_store_lod;
    }

    /// Returns true when VK_AMD_gcn_shader is supported.
    bool IsAmdGcnShaderSupported() const {
        return amd_gcn_shader;
    }

    /// Returns true when VK_AMD_shader_trinary_minmax is supported.
    bool IsAmdShaderTrinaryMinMaxSupported() const {
        return amd_shader_trinary_minmax;
    }

    /// Returns true when the shaderBufferFloat32AtomicMinMax feature of
    /// VK_EXT_shader_atomic_float2 is supported.
    bool IsShaderAtomicFloatBuffer32MinMaxSupported() const {
        return shader_atomic_float2 &&
               shader_atomic_float2_features.shaderBufferFloat32AtomicMinMax;
    }

    /// Returns true when the shaderImageFloat32AtomicMinMax feature of
    /// VK_EXT_shader_atomic_float2 is supported.
    bool IsShaderAtomicFloatImage32MinMaxSupported() const {
        return shader_atomic_float2 && shader_atomic_float2_features.shaderImageFloat32AtomicMinMax;
    }

    /// Returns true if 64-bit integer atomic operations can be used on buffers
    bool IsBufferInt64AtomicsSupported() const {
        return vk12_features.shaderBufferInt64Atomics;
    }

    /// Returns true if 64-bit integer atomic operations can be used on shared memory
    bool IsSharedInt64AtomicsSupported() const {
        return vk12_features.shaderSharedInt64Atomics;
    }

    /// Returns true if the subgroup size can be set to match guest subgroup size
    bool IsSubgroupSize64Supported() const {
        return vk13_features.subgroupSizeControl && vk13_props.maxSubgroupSize >= 64;
    }

    /// Returns true when VK_KHR_workgroup_memory_explicit_layout is supported.
    bool IsWorkgroupMemoryExplicitLayoutSupported() const {
        return workgroup_memory_explicit_layout &&
               workgroup_memory_explicit_layout_features.workgroupMemoryExplicitLayout16BitAccess;
    }

    /// Returns true if VK_NV_framebuffer_mixed_samples or
    /// VK_AMD_mixed_attachment_samples is supported
    bool IsMixedDepthSamplesSupported() const {
        return nv_framebuffer_mixed_samples || amd_mixed_attachment_samples;
    }

    /// Returns true if VK_AMD_mixed_attachment_samples is supported
    bool IsMixedAnySamplesSupported() const {
        return amd_mixed_attachment_samples;
    }

    /// Returns true when geometry shaders are supported by the device
    bool IsGeometryStageSupported() const {
        return features.geometryShader;
    }

    /// Returns true when tessellation is supported by the device
    bool IsTessellationSupported() const {
        return features.tessellationShader;
    }

    /// Returns the vendor ID of the physical device
    u32 GetVendorID() const {
        return properties.vendorID;
    }

    /// Returns the device ID of the physical device
    u32 GetDeviceID() const {
        return properties.deviceID;
    }

    /// Returns the driver ID.
    vk::DriverId GetDriverID() const {
        return driver_id;
    }

    /// Returns the current driver version provided in Vulkan-formatted version numbers.
    u32 GetDriverVersion() const {
        return properties.driverVersion;
    }

    /// Returns the current Vulkan API version provided in Vulkan-formatted version numbers.
    u32 ApiVersion() const {
        return properties.apiVersion;
    }

    /// Returns the vendor name reported from Vulkan.
    std::string_view GetVendorName() const {
        return vendor_name;
    }

    /// Returns the list of available extensions.
    std::span<const std::string> GetAvailableExtensions() const {
        return available_extensions;
    }

    /// Returns the device name.
    std::string_view GetModelName() const {
        return properties.deviceName;
    }

    /// Returns if the device is an integrated GPU.
    bool IsIntegrated() const {
        return properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;
    }

    /// Returns the pipeline cache unique identifier
    const auto GetPipelineCacheUUID() const {
        return properties.pipelineCacheUUID;
    }

    /// Returns the minimum required alignment for uniforms
    vk::DeviceSize UniformMinAlignment() const {
        return properties.limits.minUniformBufferOffsetAlignment;
    }

    ///  Returns the maximum size of uniform buffers.
    vk::DeviceSize UniformMaxSize() const {
        return properties.limits.maxUniformBufferRange;
    }

    /// Returns the minimum required alignment for storage buffers
    vk::DeviceSize StorageMinAlignment() const {
        return properties.limits.minStorageBufferOffsetAlignment;
    }

    /// Returns the minimum alignemt required for accessing host-mapped device memory
    vk::DeviceSize NonCoherentAtomSize() const {
        return properties.limits.nonCoherentAtomSize;
    }

    /// Returns the subgroup size of the selected physical device.
    u32 SubgroupSize() const {
        return vk11_props.subgroupSize;
    }

    /// Returns the largest single memory allocation the device reports (Vulkan 1.1).
    vk::DeviceSize MaxMemoryAllocationSize() const {
        return vk11_props.maxMemoryAllocationSize;
    }

    /// Returns the maximum size of compute shared memory.
    u32 MaxComputeSharedMemorySize() const {
        return properties.limits.maxComputeSharedMemorySize;
    }

    /// Returns the maximum sampler LOD bias.
    float MaxSamplerLodBias() const {
        return properties.limits.maxSamplerLodBias;
    }

    /// Returns the maximum sampler anisotropy.
    float MaxSamplerAnisotropy() const {
        return properties.limits.maxSamplerAnisotropy;
    }

    /// Returns the maximum number of push descriptors.
    u32 MaxPushDescriptors() const {
        return push_descriptor_props.maxPushDescriptors;
    }

    /// Returns the vulkan 1.2 physical device properties.
    const vk::PhysicalDeviceVulkan12Properties& GetVk12Properties() const noexcept {
        return vk12_props;
    }

    /// Returns the memory properties of the physical device.
    /// SHADPS4_HOST_MIRROR: VK_EXT_external_memory_host enabled on the device.
    bool IsExternalMemoryHostEnabled() const noexcept {
        return external_memory_host;
    }

    const vk::PhysicalDeviceMemoryProperties& GetMemoryProperties() const noexcept {
        return memory_properties;
    }

    /// Returns true if shaders can declare the ClipDistance attribute
    bool IsShaderClipDistanceSupported() const {
        return features.shaderClipDistance;
    }

    /// Returns the maximim viewport width.
    u32 GetMaxViewportWidth() const {
        return properties.limits.maxViewportDimensions[0];
    }

    /// Returns the maximum viewport height.
    u32 GetMaxViewportHeight() const {
        return properties.limits.maxViewportDimensions[1];
    }

    /// Returns the maximum render area width.
    u32 GetMaxFramebufferWidth() const {
        return properties.limits.maxFramebufferWidth;
    }

    /// Returns the maximum render area height.
    u32 GetMaxFramebufferHeight() const {
        return properties.limits.maxFramebufferHeight;
    }

    /// Returns the maximum number of samplers that can be allocated at once.
    u32 GetMaxSamplerAllocationCount() const {
        if (driver_id == vk::DriverId::eMesaKosmickrisp) {
            // FIXME: KosmicKrisp has an internal 1024 unique sampler limit before
            // vkCreateSampler starts returning VK_ERROR_OUT_OF_HOST_MEMORY. Work
            // around this for now by reducing the value to 1024.
            return 1024;
        }
        return properties.limits.maxSamplerAllocationCount;
    }

    /// Returns the sample count flags supported by color buffers.
    vk::SampleCountFlags GetColorSampleCounts() const {
        return properties.limits.framebufferColorSampleCounts;
    }

    /// Returns the sample count flags supported by depth buffer.
    vk::SampleCountFlags GetDepthSampleCounts() const {
        return properties.limits.framebufferDepthSampleCounts &
               properties.limits.framebufferStencilSampleCounts;
    }

    /// Returns true if logic ops are supported by the device.
    bool IsLogicOpSupported() const {
        return features.logicOp;
    }

    /// Returns whether VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT is supported on compressed
    /// images.
    bool IsBlockTexelViewSupported() const {
        return supports_block_texel_view;
    }

    /// Returns whether VK_IMAGE_CREATE_2D_VIEW_COMPATIBLE_BIT_EXT is supported on 3D images
    bool Is2dViewOf3dSupported() const {
        return image_2d_view_of_3d && image_2d_view_of_3d_features.image2DViewOf3D &&
               image_2d_view_of_3d_features.sampler2DViewOf3D;
    }

    /// Returns whether VK_EXT_image_view_min_lod is supported.
    bool IsImageViewMinLodSupported() const {
        return image_view_min_lod;
    }

    /// Returns whether the device can report memory usage.
    bool CanReportMemoryUsage() const {
        return supports_memory_budget;
    }

    /// SHADPS4_NATIVE_BINDLESS=1 and the device has the descriptor-indexing features the global
    /// sampled-image / sampler arrays need (enabled at device creation; sec. 12).
    bool IsBindlessSupported() const {
        return bindless_supported;
    }

    /// Returns true when the opt-in VK_EXT_descriptor_heap guest binding model is in use.
    /// Fixed before any guest pipeline, buffer, image view or sampler is created.
    bool IsNativeBindingsActive() const {
        return native_bindings_active;
    }

    /// Startup fallback to the legacy model when the heaps themselves cannot be set up.
    void DeactivateNativeBindings() const {
        native_bindings_active = false;
    }

    const NativeBindingLayout& GetNativeBindingLayout() const {
        return native_binding_layout;
    }

    const NativeApi::Functions& GetNativeApi() const {
        return native_api;
    }

    /// Returns the amount of memory used.
    [[nodiscard]] u64 GetDeviceMemoryUsage() const;

    /// The driver's current budget (VK_EXT_memory_budget): what this process may use now. Lower
    /// than at startup while other programs hold video memory. 0 when unsupported.
    [[nodiscard]] u64 GetDeviceMemoryBudgetNow() const;

    /// Returns the total memory budget available to the device.
    [[nodiscard]] u64 GetTotalMemoryBudget() const {
        return total_memory_budget;
    }

    /// SHADPS4_SIM_VRAM_MB (test switch, v1.0.2): the video memory size this run pretends to have
    /// (0 = off). The BIG_GROW size choice reads it in place of the real heap size.
    [[nodiscard]] u64 SimVramBytes() const {
        return sim_vram_bytes;
    }

    /// The ballast buffers of SHADPS4_SIM_VRAM_MB; the buffer cache fills them once on the GPU so
    /// the driver makes them resident.
    [[nodiscard]] const std::vector<vk::Buffer>& SimBallastBuffers() const {
        return sim_ballast_buffers;
    }

    /// Determines if a format is supported for a set of feature flags.
    [[nodiscard]] bool IsFormatSupported(vk::Format format, vk::FormatFeatureFlags2 flags) const;

private:
    /// Creates the logical device opportunistically enabling extensions
    bool CreateDevice();

    /// Creates the VMA allocator handle
    void CreateAllocator();

    /// Checks SHADPS4_NATIVE_BINDINGS and the device's descriptor heap capabilities, and
    /// derives the heap layout. Returns false, with the reason logged, to stay legacy.
    bool QueryNativeBindings();

    /// Collects various information from the device.
    void CollectDeviceParameters();
    void CollectPhysicalMemoryInfo();
    void CreateSimBallast();
    void CollectImageFormatInfo();
    void CollectToolingInfo() const;

    /// Gets the supported feature flags for a format.
    [[nodiscard]] vk::FormatFeatureFlags2 GetFormatFeatureFlags(vk::Format format) const;

private:
    vk::UniqueInstance instance;
    vk::PhysicalDevice physical_device;
    vk::UniqueDevice device;
    vk::PhysicalDeviceProperties properties;
    vk::PhysicalDeviceMemoryProperties memory_properties;
    bool external_memory_host{};
    vk::PhysicalDeviceVulkan11Properties vk11_props;
    vk::PhysicalDeviceVulkan12Properties vk12_props;
    vk::PhysicalDeviceVulkan13Properties vk13_props;
    vk::PhysicalDevicePushDescriptorPropertiesKHR push_descriptor_props;
    vk::PhysicalDeviceFeatures features;
    vk::PhysicalDeviceVulkan12Features vk12_features;
    bool bindless_supported{};
    vk::PhysicalDeviceVulkan13Features vk13_features;
    vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT dynamic_state_3_features;
    vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT shader_atomic_float2_features;
    vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR
        workgroup_memory_explicit_layout_features;
    vk::PhysicalDeviceImage2DViewOf3DFeaturesEXT image_2d_view_of_3d_features;
    vk::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT list_restart_features;
    vk::DriverIdKHR driver_id;
    vk::UniqueDebugUtilsMessengerEXT debug_callback{};
    vk::UniqueDebugUtilsMessengerEXT addr_bind_callback{}; ///< SHADPS4_ADDR_BIND_REPORT=1
    std::string vendor_name;
    VmaAllocator allocator{};
    vk::Queue present_queue;
    vk::Queue graphics_queue;
    vk::Queue readback_queue;
    bool has_readback_queue{};
    bool readback_family_concurrent{};
    bool has_asc_queue{};
    bool has_asc_lane{};
    vk::Queue asc_queue{};
    bool has_retry_queue{};
    vk::Queue retry_queue{};
    u32 readback_queue_family{};
    std::vector<vk::PhysicalDevice> physical_devices;
    std::vector<std::string> available_extensions;
    std::unordered_map<vk::Format, vk::FormatProperties3> format_properties;
    TracyVkCtx profiler_context{};
    u32 queue_family_index{0};
    bool custom_border_color{};
    bool fragment_shader_barycentric{};
    bool amd_shader_explicit_vertex_parameter{};
    bool depth_clip_control{};
    bool depth_clip_enable{};
    bool dynamic_state_3{};
    bool depth_range_unrestricted{};
    bool vertex_input_dynamic_state{};
    bool list_restart{};
    bool provoking_vertex{};
    bool shader_stencil_export{};
    bool image_load_store_lod{};
    bool amd_gcn_shader{};
    bool amd_shader_trinary_minmax{};
    bool nv_framebuffer_mixed_samples{};
    bool amd_mixed_attachment_samples{};
    bool shader_atomic_float{};
    bool shader_atomic_float2{};
    bool workgroup_memory_explicit_layout{};
    bool maintenance_8{};
    bool maintenance_7{};
    bool attachment_feedback_loop{};
    bool image_2d_view_of_3d{};
    bool image_view_min_lod{};
    bool supports_memory_budget{};
    bool device_fault{}; ///< SHADPS4_DEVICE_FAULT=1 and VK_EXT_device_fault enabled
    bool addr_bind_report{}; ///< SHADPS4_ADDR_BIND_REPORT=1 and the extension enabled
    bool supports_block_texel_view{};
    mutable bool native_bindings_active{};
    NativeBindingLayout native_binding_layout{};
    NativeApi::Functions native_api{};
    u64 total_memory_budget{};
    std::vector<size_t> valid_heaps;
    u64 sim_vram_bytes{};
    std::vector<vk::Buffer> sim_ballast_buffers;
    std::vector<VmaAllocation> sim_ballast_allocations;
};

} // namespace Vulkan
