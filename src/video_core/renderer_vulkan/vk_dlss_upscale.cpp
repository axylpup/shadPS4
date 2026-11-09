// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/host_shaders/dlss_composite_comp.h"
#include "video_core/renderer_vulkan/vk_dlss_upscale.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan::Dlss {

namespace {
constexpr vk::ImageUsageFlags SceneUsage =
    vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
    vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc;
constexpr vk::ImageSubresourceRange ColorRange{
    .aspectMask = vk::ImageAspectFlagBits::eColor,
    .levelCount = 1,
    .layerCount = 1,
};
} // namespace

bool UpscaleWanted() {
    static const bool wanted = [] {
        const char* v = std::getenv("SHADPS4_DLSS_UPSCALE");
        return Mode() == 1 && v != nullptr && v[0] == '1';
    }();
    return wanted;
}

u32 UpscaleTestBits() {
    static const u32 bits = [] {
        const char* v = std::getenv("SHADPS4_DLSS_UPSCALE_TEST");
        return v ? static_cast<u32>(std::atoi(v)) : 0u;
    }();
    return bits;
}

u32 HudNativeSetting() {
    static const u32 setting = [] {
        const char* v = std::getenv("SHADPS4_HUD_NATIVE");
        const int n = v ? std::atoi(v) : 0;
        const u32 s = n <= 0 ? 0u : static_cast<u32>(std::min(n, 4));
        if (s != 0) {
            LOG_INFO(Render_Vulkan, "SHADPS4_HUD_NATIVE={}: the HUD layer is drawn at {}", v,
                     s == 1 ? "the output size (integer scale)" : "a fixed scale");
        }
        return s;
    }();
    return setting;
}

u32 HudNativeScale(vk::Extent2D game, vk::Extent2D output) {
    const u32 setting = HudNativeSetting();
    if (setting == 0 || game.width == 0 || game.height == 0) {
        return 1;
    }
    if (setting >= 2) {
        return setting;
    }
    // The smallest integer scale whose layer is at least the output (2% slack: 2560x1440 over
    // 1712x968 is 1.495 / 1.488 = 2; 1920x1080 over 1920x1080 stays 1).
    const double r =
        std::max(double(output.width) / game.width, double(output.height) / game.height);
    const u32 s = static_cast<u32>(std::ceil(r - 0.02));
    return std::clamp<u32>(s, 1, 4);
}

Upscaler::Upscaler(const Instance& instance_) : instance{instance_}, device{instance_.GetDevice()} {
    // SHADPS4_DLSS_OUTPUT_SCALE=1.5: start with the supersampled output (Insert toggles it).
    if (const char* v = std::getenv("SHADPS4_DLSS_OUTPUT_SCALE")) {
        const float scale = static_cast<float>(std::atof(v));
        if (scale >= 1.0f && scale <= 2.0f) {
            g_output_scale.store(scale, std::memory_order_relaxed);
            if (scale > 1.0f) {
                g_output_pick.store(scale, std::memory_order_relaxed); // what Insert returns to
            }
        }
    }
    sampler = Check<"create dlss composite sampler">(device.createSamplerUnique(vk::SamplerCreateInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .maxAnisotropy = 1.0f,
        .minLod = 0.0f,
        .maxLod = 0.0f,
    }));
    const std::array<vk::DescriptorSetLayoutBinding, 4> bindings{{
        {.binding = 0,
         .descriptorType = vk::DescriptorType::eSampledImage,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 1,
         .descriptorType = vk::DescriptorType::eSampledImage,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 2,
         .descriptorType = vk::DescriptorType::eStorageImage,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 3,
         .descriptorType = vk::DescriptorType::eSampler,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute,
         .pImmutableSamplers = &sampler.get()},
    }};
    set_layout = Check<"create dlss composite set layout">(device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    }));
    pipeline_layout = Check<"create dlss composite layout">(device.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts = &set_layout.get(),
    }));
    const auto module =
        Compile(HostShaders::DLSS_COMPOSITE_COMP, vk::ShaderStageFlagBits::eCompute, device);
    ASSERT(module);
    pipeline = Check<"create dlss composite pipeline">(device.createComputePipelineUnique(
        {}, vk::ComputePipelineCreateInfo{
                .stage{
                    .stage = vk::ShaderStageFlagBits::eCompute,
                    .module = module,
                    .pName = "main",
                },
                .layout = pipeline_layout.get(),
            }));
    device.destroyShaderModule(module);
}

Upscaler::~Upscaler() = default;

void Upscaler::SetOutputSize(u32 width, u32 height, bool game_pq) {
    ++flips;
    while (!retired.empty() && retired.front().first + 16 <= flips) {
        retired.pop_front();
    }
    // SHADPS4_DLSS_UPSCALE_FORCE=1 (verification): the whole HUD-layer path at 1920x1080 (DLAA),
    // to compare with the plain DLAA path on the same scene.
    static const bool force = [] {
        const char* v = std::getenv("SHADPS4_DLSS_UPSCALE_FORCE");
        return v != nullptr && v[0] == '1';
    }();
    // Insert: 1.5x the window (supersampling), scaled down to the window by the post pass.
    const float scale = force ? 1.0f : g_output_scale.load(std::memory_order_relaxed);
    const u32 out_w = std::min<u32>(static_cast<u32>(width * scale + 0.5f), 7680);
    const u32 out_h = std::min<u32>(static_cast<u32>(height * scale + 0.5f), 4320);
    // The game's render size (1920x1080, or what SHADPS4_ASPECT / SHADPS4_RENDER_HEIGHT / a
    // resolution patch made it; learned from the resolve, vk_dlss.h).
    const u32 game_w = g_game_width.load(std::memory_order_relaxed);
    const u32 game_h = g_game_height.load(std::memory_order_relaxed);
    // At most 2x the game's size (DLSS Performance's floor; s1452: Ultra Performance takes only
    // one exact size, 853x478 for 2560x1435, not SHADPS4_RENDER_HEIGHT=480's 856x480): the post
    // pass scales the rest to the window, as it scales the 1.5x output down.
    double fit = 1.0;
    if (game_w > 0 && game_h > 0 && (out_w > 2 * game_w || out_h > 2 * game_h)) {
        fit = std::min(2.0 * game_w / out_w, 2.0 * game_h / out_h);
    }
    const u32 fit_w = std::max<u32>(game_w, static_cast<u32>(out_w * fit));
    const u32 fit_h = std::max<u32>(game_h, static_cast<u32>(out_h * fit));
    output_size = force ? vk::Extent2D{game_w, game_h} : vk::Extent2D{fit_w, fit_h};
    // DLSS needs an output larger than its input; smaller windows keep DLAA in place.
    // 1.1.1 (GitHub #13, until 1.2's HDR work): off while the game sends HDR10 (PQ) frames. In
    // this mode the game's own display buffer holds only the HUD, and the composite (DLSS's scene
    // + that layer) is not PQ (inferred cause of #13's black city: that player's log shows DLSS
    // upscale running in the game's HDR mode). The game then draws its whole picture itself (at
    // the chosen render size, scaled by the post pass).
    const bool on = (force || (width > game_w && height > game_h)) && g_ready.load() &&
                    instance.IsDynamicColorWriteMaskSupported() && !game_pq;
    if (on != g_hud_layer.load(std::memory_order_relaxed)) {
        LOG_INFO(Render_Vulkan, "DLSS upscale: {} (output {}x{}{}{})", on ? "on" : "off",
                 output_size.width, output_size.height, force ? ", forced" : "",
                 game_pq ? ", the game sends HDR10 (PQ) frames" : "");
        g_hud_layer.store(on, std::memory_order_relaxed);
    }
}

void Upscaler::Resize(vk::Extent2D size) {
    if (scene) {
        retired.emplace_back(flips, std::move(scene));
    }
    if (out) {
        retired.emplace_back(flips, std::move(out));
    }
    const auto make = [&](vk::Format format, vk::ImageUsageFlags usage, const char* name) {
        auto t = std::make_unique<Target>();
        t->image = VideoCore::UniqueImage(device, instance.GetAllocator());
        t->image.Create(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent{size.width, size.height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .usage = usage,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        SetObjectName(device, static_cast<vk::Image>(t->image), "{}", name);
        t->view = Check<"create dlss upscale view">(device.createImageViewUnique({
            .image = t->image,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange = ColorRange,
        }));
        return t;
    };
    // Transfer usage too: NGX clears its output with vkCmdClearColorImage (validation run s1359,
    // VUID-vkCmdClearColorImage-image-00002 on the storage + sampled image).
    scene = make(vk::Format::eR8G8B8A8Unorm, SceneUsage, "DLSS output");
    out = make(vk::Format::eR16G16B16A16Sfloat, SceneUsage, "DLSS composite");
    scene_layout = vk::ImageLayout::eUndefined;
    image_size = size;
    LOG_INFO(Render_Vulkan, "DLSS upscale: output images {}x{}", size.width, size.height);
}

EvalImage Upscaler::SceneTarget(vk::CommandBuffer cmdbuf) {
    if (!scene || image_size != output_size) {
        Resize(output_size);
    }
    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eUndefined, // fully overwritten
        .newLayout = vk::ImageLayout::eGeneral,
        .image = scene->image,
        .subresourceRange = ColorRange,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    });
    scene_layout = vk::ImageLayout::eGeneral;
    ready = false;
    return EvalImage{
        .image = scene->image,
        .view = scene->view.get(),
        .format = vk::Format::eR8G8B8A8Unorm,
        .layout = vk::ImageLayout::eGeneral,
        .usage = SceneUsage,
        .width = image_size.width,
        .height = image_size.height,
        .depth = false,
    };
}

void Upscaler::SceneWritten(vk::CommandBuffer cmdbuf) {
    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .image = scene->image,
        .subresourceRange = ColorRange,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    });
    scene_layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    ready = true;
}

CompositeJob Upscaler::TakeJob(vk::ImageView hud_view) {
    ready = false;
    return CompositeJob{
        .pipeline = pipeline.get(),
        .layout = pipeline_layout.get(),
        .hud_view = hud_view,
        .scene_view = scene->view.get(),
        .out_image = out->image,
        .out_view = out->view.get(),
        .size = image_size,
    };
}

void Upscaler::BeginNativeHud(Scheduler& scheduler, vk::Extent2D game, vk::Format format,
                              u32 scale) {
    const vk::Extent2D size{game.width * scale, game.height * scale};
    if (!hud || hud_info.format != format || hud_info.game != game || hud_info.scale != scale) {
        if (hud) {
            retired.emplace_back(flips, std::move(hud));
        }
        hud = std::make_unique<Target>();
        hud->image = VideoCore::UniqueImage(device, instance.GetAllocator());
        hud->image.Create(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent{size.width, size.height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
                     vk::ImageUsageFlagBits::eTransferDst,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        SetObjectName(device, static_cast<vk::Image>(hud->image), "Native HUD {}x{}", size.width,
                      size.height);
        hud->view = Check<"create native hud view">(device.createImageViewUnique({
            .image = hud->image,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange = ColorRange,
        }));
        hud_info = NativeHud{
            .image = hud->image,
            .view = hud->view.get(),
            .format = format,
            .game = game,
            .scale = scale,
        };
        LOG_INFO(Render_Vulkan, "SHADPS4_HUD_NATIVE: HUD drawn at {}x{} ({}x the {}x{} layer)",
                 size.width, size.height, scale, game.width, game.height);
    }
    // Recorded in stream order with the draws (the async recorder runs them later).
    scheduler.EndRendering();
    scheduler.Record([image = hud_info.image](vk::CommandBuffer cmdbuf) {
    // Last frame's composite read it (an earlier submission); the contents are discarded.
    const vk::ImageMemoryBarrier2 to_clear{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eNone,
        .dstStageMask = vk::PipelineStageFlagBits2::eClear,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eTransferDstOptimal,
        .image = image,
        .subresourceRange = ColorRange,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &to_clear,
    });
    cmdbuf.clearColorImage(image, vk::ImageLayout::eTransferDstOptimal,
                           vk::ClearColorValue{std::array{0.0f, 0.0f, 0.0f, 0.0f}}, ColorRange);
    const vk::ImageMemoryBarrier2 to_draw{
        .srcStageMask = vk::PipelineStageFlagBits2::eClear,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentRead |
                         vk::AccessFlagBits2::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .image = image,
        .subresourceRange = ColorRange,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &to_draw,
    });
    });
    hud_on = true;
    hud_passes = 0;
    hud_depth_ready = false;
}

vk::ImageView Upscaler::NativeHudDepth(Scheduler& scheduler, vk::Image src, vk::Format format,
                                       vk::ImageAspectFlags aspect) {
    const vk::Extent2D size{hud_info.game.width * hud_info.scale,
                            hud_info.game.height * hud_info.scale};
    if (!hud_depth || hud_depth_format != format || hud_depth_size != size) {
        const auto props = instance.GetPhysicalDevice().getFormatProperties(format);
        const auto need = vk::FormatFeatureFlagBits::eBlitSrc |
                          vk::FormatFeatureFlagBits::eBlitDst |
                          vk::FormatFeatureFlagBits::eDepthStencilAttachment;
        if ((props.optimalTilingFeatures & need) != need) {
            static bool logged = false;
            if (!logged) {
                logged = true;
                LOG_WARNING(Render_Vulkan, "SHADPS4_HUD_NATIVE: depth format {} cannot be blitted; "
                                           "HUD passes with depth stay at the render size",
                            vk::to_string(format));
            }
            return {};
        }
        if (hud_depth) {
            retired.emplace_back(flips, std::move(hud_depth));
        }
        hud_depth = std::make_unique<Target>();
        hud_depth->image = VideoCore::UniqueImage(device, instance.GetAllocator());
        hud_depth->image.Create(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent{size.width, size.height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment |
                     vk::ImageUsageFlagBits::eTransferDst,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        SetObjectName(device, static_cast<vk::Image>(hud_depth->image), "Native HUD depth {}x{}",
                      size.width, size.height);
        hud_depth->view = Check<"create native hud depth view">(device.createImageViewUnique({
            .image = hud_depth->image,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange{.aspectMask = aspect, .levelCount = 1, .layerCount = 1},
        }));
        hud_depth_format = format;
        hud_depth_size = size;
        hud_depth_ready = false;
        LOG_INFO(Render_Vulkan, "SHADPS4_HUD_NATIVE: HUD depth {}x{} {}", size.width, size.height,
                 vk::to_string(format));
    }
    if (!hud_depth_ready) {
        hud_depth_ready = true;
        scheduler.EndRendering();
        scheduler.Record([src, dst = static_cast<vk::Image>(hud_depth->image), aspect,
                          game = hud_info.game, size](vk::CommandBuffer cmdbuf) {
            const vk::ImageSubresourceRange range{
                .aspectMask = aspect, .levelCount = 1, .layerCount = 1};
            const vk::ImageMemoryBarrier2 to_copy{
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .srcAccessMask = vk::AccessFlagBits2::eNone,
                .dstStageMask = vk::PipelineStageFlagBits2::eBlit,
                .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eTransferDstOptimal,
                .image = dst,
                .subresourceRange = range,
            };
            cmdbuf.pipelineBarrier2(
                vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_copy});
            const vk::ImageSubresourceLayers layers{.aspectMask = aspect, .layerCount = 1};
            const vk::ImageBlit blit{
                .srcSubresource = layers,
                .srcOffsets = std::array{vk::Offset3D{0, 0, 0},
                                         vk::Offset3D{s32(game.width), s32(game.height), 1}},
                .dstSubresource = layers,
                .dstOffsets = std::array{vk::Offset3D{0, 0, 0},
                                         vk::Offset3D{s32(size.width), s32(size.height), 1}},
            };
            cmdbuf.blitImage(src, vk::ImageLayout::eTransferSrcOptimal, dst,
                             vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eNearest);
            const vk::ImageMemoryBarrier2 to_draw{
                .srcStageMask = vk::PipelineStageFlagBits2::eBlit,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                                vk::PipelineStageFlagBits2::eLateFragmentTests,
                .dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
                                 vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                .newLayout = vk::ImageLayout::eGeneral,
                .image = dst,
                .subresourceRange = range,
            };
            cmdbuf.pipelineBarrier2(
                vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_draw});
        });
    }
    return hud_depth->view.get();
}

vk::ImageView Upscaler::EndNativeHud(Scheduler& scheduler) {
    if (!hud_on) {
        return {};
    }
    hud_on = false;
    ++hud_frames;
    hud_passes_total += hud_passes;
    hud_frames_empty += hud_passes == 0;
    if (hud_frames % 600 == 1) {
        LOG_INFO(Render_Vulkan, "SHADPS4_HUD_NATIVE: {} frames, {:.1f} HUD passes per frame, {} "
                                "frames without one",
                 hud_frames, double(hud_passes_total) / double(hud_frames), hud_frames_empty);
    }
    scheduler.EndRendering();
    scheduler.Record([image = hud_info.image](vk::CommandBuffer cmdbuf) {
    const vk::ImageMemoryBarrier2 to_read{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .image = image,
        .subresourceRange = ColorRange,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &to_read,
    });
    });
    return hud_info.view;
}

void RecordComposite(vk::CommandBuffer cmdbuf, const CompositeJob& job) {
    const vk::ImageMemoryBarrier2 to_write{
        .srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .oldLayout = vk::ImageLayout::eUndefined, // fully overwritten
        .newLayout = vk::ImageLayout::eGeneral,
        .image = job.out_image,
        .subresourceRange = ColorRange,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &to_write,
    });
    const vk::DescriptorImageInfo hud_info{.imageView = job.hud_view,
                                           .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::DescriptorImageInfo scene_info{.imageView = job.scene_view,
                                             .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::DescriptorImageInfo out_info{.imageView = job.out_view,
                                           .imageLayout = vk::ImageLayout::eGeneral};
    const std::array<vk::WriteDescriptorSet, 3> writes{{
        {.dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eSampledImage,
         .pImageInfo = &hud_info},
        {.dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eSampledImage,
         .pImageInfo = &scene_info},
        {.dstBinding = 2,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageImage,
         .pImageInfo = &out_info},
    }};
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, job.pipeline);
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, job.layout, 0, writes);
    cmdbuf.dispatch((job.size.width + 7) / 8, (job.size.height + 7) / 8, 1);
    const vk::ImageMemoryBarrier2 to_read{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .image = job.out_image,
        .subresourceRange = ColorRange,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &to_read,
    });
}

Upscaler* GetUpscaler(const Instance& instance) {
    if (!UpscaleWanted()) {
        return nullptr;
    }
    // Never destroyed: a static destructor would run after the device is gone.
    static Upscaler* const upscaler = new Upscaler(instance);
    return upscaler;
}

} // namespace Vulkan::Dlss
