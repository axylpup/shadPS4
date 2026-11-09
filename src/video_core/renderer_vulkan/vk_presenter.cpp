// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#ifdef _MSC_VER
#include <optional>
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "common/scope_exit.h"
#include <cstdlib>
#include "common/debug.h"
#include "common/elf_info.h"
#include "common/fault_monitor.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "common/singleton.h"
#include "common/update_notice.h"
#include "core/debug_state.h"
#include "core/devtools/layer.h"
#include "core/emulator_settings.h"
#include "core/libraries/system/systemservice.h"
#include "imgui/friends_layer.h"
#include "imgui/invitation_prompt_layer.h"
#include "imgui/notifications_layer.h"
#include "imgui/renderer/imgui_core.h"
#include "imgui/renderer/texture_manager.h"
#include "imgui/renderer/imgui_impl_vulkan.h"
#include "imgui/shadnet_notifications_layer.h"
#include "sdl_window.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_dlss5.h"
#include "video_core/renderer_vulkan/vk_dlss_upscale.h"
#include "video_core/renderer_vulkan/vk_memory_budget.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_presenter.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/texture_cache/image.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <thread>
#include <SDL3/SDL_events.h>
#include <csetjmp>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <system_error>
#include <vector>
#include <imgui.h>
#include <png.h>
#include <vk_mem_alloc.h>

namespace Vulkan {

bool CanBlitToSwapchain(const vk::PhysicalDevice physical_device, vk::Format format) {
    const vk::FormatProperties props{physical_device.getFormatProperties(format)};
    return static_cast<bool>(props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eBlitDst);
}

[[nodiscard]] vk::ImageSubresourceLayers MakeImageSubresourceLayers() {
    return vk::ImageSubresourceLayers{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
}

[[nodiscard]] vk::ImageBlit MakeImageBlit(s32 frame_width, s32 frame_height, s32 dst_width,
                                          s32 dst_height, s32 offset_x, s32 offset_y) {
    return vk::ImageBlit{
        .srcSubresource = MakeImageSubresourceLayers(),
        .srcOffsets =
            std::array{
                vk::Offset3D{
                    .x = 0,
                    .y = 0,
                    .z = 0,
                },
                vk::Offset3D{
                    .x = frame_width,
                    .y = frame_height,
                    .z = 1,
                },
            },
        .dstSubresource = MakeImageSubresourceLayers(),
        .dstOffsets =
            std::array{
                vk::Offset3D{
                    .x = offset_x,
                    .y = offset_y,
                    .z = 0,
                },
                vk::Offset3D{
                    .x = offset_x + dst_width,
                    .y = offset_y + dst_height,
                    .z = 1,
                },
            },
    };
}

[[nodiscard]] vk::ImageBlit MakeImageBlitStretch(s32 frame_width, s32 frame_height,
                                                 s32 swapchain_width, s32 swapchain_height) {
    return MakeImageBlit(frame_width, frame_height, swapchain_width, swapchain_height, 0, 0);
}

static vk::Rect2D FitImage(s32 frame_width, s32 frame_height, s32 swapchain_width,
                           s32 swapchain_height) {
    float frame_aspect = static_cast<float>(frame_width) / frame_height;
    float swapchain_aspect = static_cast<float>(swapchain_width) / swapchain_height;

    u32 dst_width = swapchain_width;
    u32 dst_height = swapchain_height;

    // Within 1% of the window's shape: fill it (user 10-21: render sizes are multiples of 8, so
    // at some screen percentages the picture was a few pixels off the window's shape and thin
    // black bars appeared, changing with the percentage; a 1% stretch is not visible).
    if (std::abs(frame_aspect / swapchain_aspect - 1.0f) < 0.01f) {
        return vk::Rect2D{{0, 0}, {dst_width, dst_height}};
    }
    if (frame_aspect > swapchain_aspect) {
        dst_height = static_cast<s32>(swapchain_width / frame_aspect);
    } else {
        dst_width = static_cast<s32>(swapchain_height * frame_aspect);
    }

    const s32 offset_x = (swapchain_width - dst_width) / 2;
    const s32 offset_y = (swapchain_height - dst_height) / 2;

    return vk::Rect2D{{offset_x, offset_y}, {dst_width, dst_height}};
}

[[nodiscard]] vk::ImageBlit MakeImageBlitFit(s32 frame_width, s32 frame_height, s32 swapchain_width,
                                             s32 swapchain_height) {
    const auto& dst_rect = FitImage(frame_width, frame_height, swapchain_width, swapchain_height);

    return MakeImageBlit(frame_width, frame_height, dst_rect.extent.width, dst_rect.extent.height,
                         dst_rect.offset.x, dst_rect.offset.y);
}

enum class ScreenshotKind : u8 {
    GameOnly,
    WithOverlays,
};

struct ScreenshotReadback {
    ScreenshotKind kind{};
    std::vector<std::filesystem::path> paths{};
    VideoCore::Buffer buffer;
    u32 width{};
    u32 height{};
    vk::Format format{};
    bool hdr_encoded{};

    ScreenshotReadback(const Instance& instance, Scheduler& scheduler, ScreenshotKind kind_,
                       std::vector<std::filesystem::path> paths_, const u32 width_,
                       const u32 height_, const vk::Format format_, const bool hdr_encoded_)
        : kind{kind_}, paths{std::move(paths_)},
          buffer{instance,
                 scheduler,
                 VideoCore::MemoryUsage::Download,
                 0,
                 vk::BufferUsageFlagBits::eTransferDst,
                 static_cast<u64>(width_) * static_cast<u64>(height_) * 4},
          width{width_}, height{height_}, format{format_}, hdr_encoded{hdr_encoded_} {}
};

static std::string SanitizeFilenameComponent(std::string value) {
    for (char& c : value) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (!std::isalnum(uc) && c != '_' && c != '-') {
            c = '_';
        }
    }
    if (value.empty()) {
        return "UNKNOWN";
    }
    return value;
}

static std::vector<std::filesystem::path> BuildScreenshotPaths(const ScreenshotKind kind,
                                                               const u32 count) {
    static std::atomic<u64> screenshot_sequence{0};
    std::vector<std::filesystem::path> paths{};
    if (count == 0) {
        return paths;
    }

    const auto& screenshots_dir = Common::FS::GetUserPath(Common::FS::PathType::ScreenshotsDir);
    std::filesystem::create_directories(screenshots_dir);

    const auto game_id =
        SanitizeFilenameComponent(std::string(Common::ElfInfo::Instance().GameSerial()));
    const auto now = std::chrono::system_clock::now();
    const auto now_time = std::chrono::system_clock::to_time_t(now);
    const auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() %
        1000;

    std::tm local_tm{};
#ifdef _WIN32
    localtime_s(&local_tm, &now_time);
#else
    localtime_r(&now_time, &local_tm);
#endif

    std::ostringstream stamp;
    stamp << std::put_time(&local_tm, "%Y%m%d_%H%M%S") << '_' << std::setw(3) << std::setfill('0')
          << ms;

    const char* suffix = kind == ScreenshotKind::GameOnly ? "game" : "hud";
    const auto first_sequence = screenshot_sequence.fetch_add(count, std::memory_order_relaxed);

    paths.reserve(count);
    const auto stamp_str = stamp.str();
    for (u32 i = 0; i < count; ++i) {
        paths.emplace_back(screenshots_dir / fmt::format("{}_{}_{}_{:06}.png", game_id, stamp_str,
                                                         suffix, first_sequence + i));
    }

    return paths;
}

static float PqToNits(const float encoded) {
    // ST.2084 inverse EOTF
    constexpr float m1 = 2610.0f / 16384.0f;
    constexpr float m2 = 2523.0f / 32.0f;
    constexpr float c1 = 3424.0f / 4096.0f;
    constexpr float c2 = 2413.0f / 128.0f;
    constexpr float c3 = 2392.0f / 128.0f;

    const float v = std::clamp(encoded, 0.0f, 1.0f);
    const float vp = std::pow(v, 1.0f / m2);
    const float num = std::max(vp - c1, 0.0f);
    const float den = std::max(c2 - c3 * vp, 1e-6f);
    return 10000.0f * std::pow(num / den, 1.0f / m1);
}

static float ToneMapToSdrLinear(const float nits) {
    // Map absolute HDR luminance into SDR [0,1], preserving 100-nit white.
    constexpr float sdr_white_nits = 100.0f;
    const float x = std::max(nits, 0.0f) / sdr_white_nits;
    const float mapped = (2.0f * x) / (1.0f + x);
    return std::clamp(mapped, 0.0f, 1.0f);
}

static float LinearToSrgb(const float linear) {
    const float x = std::clamp(linear, 0.0f, 1.0f);
    if (x <= 0.0031308f) {
        return 12.92f * x;
    }
    return 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
}

static const std::array<float, 1024>& GetPqDecodeNitsLut() {
    static const std::array<float, 1024> lut = [] {
        std::array<float, 1024> values{};
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = PqToNits(static_cast<float>(i) / 1023.0f);
        }
        return values;
    }();
    return lut;
}

static const std::array<u8, 1024>& GetUnorm10ToU8Lut() {
    static const std::array<u8, 1024> lut = [] {
        std::array<u8, 1024> values{};
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<u8>((i * 255u + 511u) / 1023u);
        }
        return values;
    }();
    return lut;
}

static void CopyImageToReadback(const vk::CommandBuffer& cmdbuf, const vk::Image image,
                                const vk::ImageLayout layout, ScreenshotReadback& readback) {
    const vk::BufferImageCopy copy_region = {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource =
            {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {readback.width, readback.height, 1},
    };
    cmdbuf.copyImageToBuffer(image, layout, readback.buffer.Handle(), copy_region);
}

static bool ConvertReadbackToRgba8(const ScreenshotReadback& readback, std::vector<u8>& out_rgba) {
    const u64 pixel_count = static_cast<u64>(readback.width) * static_cast<u64>(readback.height);
    const u64 byte_size = pixel_count * 4;
    if (readback.buffer.mapped_data.size() < byte_size) {
        LOG_ERROR(Render_Vulkan, "Screenshot readback buffer size mismatch (have {}, need {})",
                  readback.buffer.mapped_data.size(), byte_size);
        return false;
    }

    const auto src =
        std::span<const u8>{readback.buffer.mapped_data.data(), static_cast<size_t>(byte_size)};
    out_rgba.resize(static_cast<size_t>(byte_size));

    switch (readback.format) {
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eR8G8B8A8Srgb:
        std::memcpy(out_rgba.data(), src.data(), out_rgba.size());
        for (u64 i = 0; i < pixel_count; ++i) {
            out_rgba[static_cast<size_t>(i) * 4 + 3] = 255;
        }
        return true;
    case vk::Format::eB8G8R8A8Unorm:
    case vk::Format::eB8G8R8A8Srgb:
        for (u64 i = 0; i < pixel_count; ++i) {
            const size_t o = static_cast<size_t>(i) * 4;
            out_rgba[o + 0] = src[o + 2];
            out_rgba[o + 1] = src[o + 1];
            out_rgba[o + 2] = src[o + 0];
            out_rgba[o + 3] = 255;
        }
        return true;
    case vk::Format::eA2R10G10B10UnormPack32: {
        const auto& pq_decode_lut = GetPqDecodeNitsLut();
        const auto& unorm10_to_u8 = GetUnorm10ToU8Lut();

        for (u64 i = 0; i < pixel_count; ++i) {
            const size_t o = static_cast<size_t>(i) * 4;
            const u32 packed = static_cast<u32>(src[o + 0]) | (static_cast<u32>(src[o + 1]) << 8) |
                               (static_cast<u32>(src[o + 2]) << 16) |
                               (static_cast<u32>(src[o + 3]) << 24);
            const u32 b = (packed >> 0) & 0x3FF;
            const u32 g = (packed >> 10) & 0x3FF;
            const u32 r = (packed >> 20) & 0x3FF;

            if (readback.hdr_encoded) {
                // Rec.2020 + PQ. Convert to SDR Rec.709 for PNG output.
                const float r2020 = pq_decode_lut[r];
                const float g2020 = pq_decode_lut[g];
                const float b2020 = pq_decode_lut[b];

                const float r709_nits = 1.6605f * r2020 - 0.5876f * g2020 - 0.0728f * b2020;
                const float g709_nits = -0.1246f * r2020 + 1.1329f * g2020 - 0.0083f * b2020;
                const float b709_nits = -0.0182f * r2020 - 0.1006f * g2020 + 1.1187f * b2020;

                const float r_srgb = LinearToSrgb(ToneMapToSdrLinear(r709_nits));
                const float g_srgb = LinearToSrgb(ToneMapToSdrLinear(g709_nits));
                const float b_srgb = LinearToSrgb(ToneMapToSdrLinear(b709_nits));

                out_rgba[o + 0] = static_cast<u8>(std::clamp(r_srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
                out_rgba[o + 1] = static_cast<u8>(std::clamp(g_srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
                out_rgba[o + 2] = static_cast<u8>(std::clamp(b_srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
            } else {
                out_rgba[o + 0] = unorm10_to_u8[r];
                out_rgba[o + 1] = unorm10_to_u8[g];
                out_rgba[o + 2] = unorm10_to_u8[b];
            }
            out_rgba[o + 3] = 255;
        }
        return true;
    }
    case vk::Format::eA2B10G10R10UnormPack32: {
        const auto& pq_decode_lut = GetPqDecodeNitsLut();
        const auto& unorm10_to_u8 = GetUnorm10ToU8Lut();

        for (u64 i = 0; i < pixel_count; ++i) {
            const size_t o = static_cast<size_t>(i) * 4;
            const u32 packed = static_cast<u32>(src[o + 0]) | (static_cast<u32>(src[o + 1]) << 8) |
                               (static_cast<u32>(src[o + 2]) << 16) |
                               (static_cast<u32>(src[o + 3]) << 24);
            const u32 r = (packed >> 0) & 0x3FF;
            const u32 g = (packed >> 10) & 0x3FF;
            const u32 b = (packed >> 20) & 0x3FF;

            if (readback.hdr_encoded) {
                // HDR swapchain path is Rec.2020 + PQ. Convert to SDR Rec.709 for PNG output.
                const float r2020 = pq_decode_lut[r];
                const float g2020 = pq_decode_lut[g];
                const float b2020 = pq_decode_lut[b];

                const float r709_nits = 1.6605f * r2020 - 0.5876f * g2020 - 0.0728f * b2020;
                const float g709_nits = -0.1246f * r2020 + 1.1329f * g2020 - 0.0083f * b2020;
                const float b709_nits = -0.0182f * r2020 - 0.1006f * g2020 + 1.1187f * b2020;

                const float r_srgb = LinearToSrgb(ToneMapToSdrLinear(r709_nits));
                const float g_srgb = LinearToSrgb(ToneMapToSdrLinear(g709_nits));
                const float b_srgb = LinearToSrgb(ToneMapToSdrLinear(b709_nits));

                out_rgba[o + 0] = static_cast<u8>(std::clamp(r_srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
                out_rgba[o + 1] = static_cast<u8>(std::clamp(g_srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
                out_rgba[o + 2] = static_cast<u8>(std::clamp(b_srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
            } else {
                out_rgba[o + 0] = unorm10_to_u8[r];
                out_rgba[o + 1] = unorm10_to_u8[g];
                out_rgba[o + 2] = unorm10_to_u8[b];
            }
            out_rgba[o + 3] = 255;
        }
        return true;
    }
    default:
        LOG_WARNING(Render_Vulkan, "Unsupported screenshot format: {}",
                    vk::to_string(readback.format));
        return false;
    }
}

static bool WritePng(const std::filesystem::path& path, const std::span<const u8> rgba,
                     const u32 width, const u32 height) {
    Common::FS::IOFile file(path, Common::FS::FileAccessMode::Create);
    if (!file.IsOpen()) {
        return false;
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (png_ptr == nullptr) {
        return false;
    }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == nullptr) {
        png_destroy_write_struct(&png_ptr, nullptr);
        return false;
    }

    if (setjmp(png_jmpbuf(png_ptr)) != 0) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return false;
    }

    png_init_io(png_ptr, file.file);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);

    thread_local std::vector<png_bytep> rows;
    rows.resize(height);
    for (u32 y = 0; y < height; ++y) {
        rows[y] = const_cast<png_bytep>(rgba.data() + static_cast<size_t>(y) * width * 4);
    }

    png_write_image(png_ptr, rows.data());
    png_write_end(png_ptr, info_ptr);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    return true;
}

// 1.2 HDR checks (2026-10-22): an HDR screenshot's own 10-bit values (Rec.2020 PQ code values,
// scaled to 16 bits, RGB) beside the tone-mapped PNG - for measuring nits and making HDR10 clips.
// A viewer shows it washed out; it is data, not a picture.
static bool WritePq16Png(const std::filesystem::path& path, const ScreenshotReadback& readback) {
    const bool bgr_low = readback.format == vk::Format::eA2R10G10B10UnormPack32; // B in bits 0-9
    if (!readback.hdr_encoded || (!bgr_low && readback.format != vk::Format::eA2B10G10R10UnormPack32)) {
        return false;
    }
    const u64 pixel_count = u64(readback.width) * readback.height;
    if (readback.buffer.mapped_data.size() < pixel_count * 4) {
        return false;
    }
    const u8* src = readback.buffer.mapped_data.data();
    std::vector<u16> rgb(static_cast<size_t>(pixel_count) * 3);
    const auto widen = [](u32 v) { return static_cast<u16>((v << 6) | (v >> 4)); };
    for (u64 i = 0; i < pixel_count; ++i) {
        u32 packed;
        std::memcpy(&packed, src + i * 4, 4);
        const u32 lo = packed & 0x3FF, mid = (packed >> 10) & 0x3FF, hi = (packed >> 20) & 0x3FF;
        rgb[i * 3 + 0] = widen(bgr_low ? hi : lo);
        rgb[i * 3 + 1] = widen(mid);
        rgb[i * 3 + 2] = widen(bgr_low ? lo : hi);
    }
    std::vector<png_bytep> rows(readback.height); // before setjmp: a longjmp skips destructors
    for (u32 y = 0; y < readback.height; ++y) {
        rows[y] = reinterpret_cast<png_bytep>(rgb.data() + size_t(y) * readback.width * 3);
    }
    Common::FS::IOFile file(path, Common::FS::FileAccessMode::Create);
    if (!file.IsOpen()) {
        return false;
    }
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (png_ptr == nullptr) {
        return false;
    }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == nullptr || setjmp(png_jmpbuf(png_ptr)) != 0) {
        png_destroy_write_struct(&png_ptr, info_ptr ? &info_ptr : nullptr);
        return false;
    }
    png_init_io(png_ptr, file.file);
    png_set_IHDR(png_ptr, info_ptr, readback.width, readback.height, 16, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);
    png_set_swap(png_ptr); // 16-bit samples from this little-endian buffer
    png_write_image(png_ptr, rows.data());
    png_write_end(png_ptr, info_ptr);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    return true;
}

static void SavePendingScreenshots(const std::vector<ScreenshotReadback>& readbacks) {
    for (const auto& readback : readbacks) {
        if (readback.paths.empty()) {
            continue;
        }

        std::vector<u8> rgba;
        if (!ConvertReadbackToRgba8(readback, rgba)) {
            continue;
        }

        const auto& primary_path = readback.paths.front();
        if (!WritePng(primary_path, rgba, readback.width, readback.height)) {
            LOG_ERROR(Render_Vulkan, "Failed saving screenshot to {}", primary_path.string());
            continue;
        }

        LOG_INFO(Render_Vulkan, "Saved screenshot: {}", primary_path.string());
        if (readback.hdr_encoded) {
            auto raw = primary_path;
            raw.replace_filename(primary_path.stem().string() + "-pq2020-16bit.png");
            LOG_INFO(Render_Vulkan, "Saved the HDR10 values: {} ({})", raw.string(),
                     WritePq16Png(raw, readback) ? "ok" : "failed");
        }

        std::ifstream file(primary_path, std::ios::binary);
        std::vector<u8> imgdata;
        if (file) {
            imgdata = std::vector<u8>(std::istreambuf_iterator<char>(file),
                                      std::istreambuf_iterator<char>());
        }
        shadNotifications::QueueNotification("Saved screenshot:\n" + primary_path.string(), 3.0f,
                                             shadNotifications::position::BottomRight, imgdata);

        for (size_t i = 1; i < readback.paths.size(); ++i) {
            const auto& path = readback.paths[i];
            std::error_code ec{};
            std::filesystem::copy_file(primary_path, path, std::filesystem::copy_options::none, ec);
            if (ec) {
                // Fallback for platforms/filesystems where copy_file can fail for transient
                // reasons.
                if (!WritePng(path, rgba, readback.width, readback.height)) {
                    LOG_ERROR(Render_Vulkan, "Failed saving screenshot to {}", path.string());
                    continue;
                }
            }

            LOG_INFO(Render_Vulkan, "Saved screenshot: {}", path.string());
            std::ifstream file(path, std::ios::binary);
            std::vector<u8> imgdata;
            if (file) {
                imgdata = std::vector<u8>(std::istreambuf_iterator<char>(file),
                                          std::istreambuf_iterator<char>());
            }
            shadNotifications::QueueNotification("Saved screenshot:\n" + path.string(), 3.0f,
                                                 shadNotifications::position::BottomRight, imgdata);
        }
    }
}

namespace {
// SHADPS4_BOOT_SCREEN (2026-10-17, user: the first launch's preparation is too easy to miss): the
// presenter whose rasterizer is being built - WarmUp runs inside that, on the main thread.
Presenter* g_booting_presenter = nullptr;
// 1.1 (user 10-21): when the first boot frame was shown (HoldBootScreen's 3 s minimum).
std::chrono::steady_clock::time_point g_boot_first_frame{};

bool BootScreenEnabled() {
    static const bool on = [] {
        const char* v = std::getenv("SHADPS4_BOOT_SCREEN");
        return v && v[0] == '1';
    }();
    return on;
}
} // namespace

void PresentBootScreen(std::string_view headline, std::string_view detail, float fraction,
                       bool one_time) {
    if (g_booting_presenter && BootScreenEnabled()) {
        if (g_boot_first_frame == std::chrono::steady_clock::time_point{}) {
            g_boot_first_frame = std::chrono::steady_clock::now();
        }
        g_booting_presenter->DrawBootScreen(headline, detail, fraction, one_time);
    }
}

void HoldBootScreen(std::string_view headline, std::string_view detail, bool one_time,
                    std::chrono::milliseconds min_shown) {
    if (!g_booting_presenter || !BootScreenEnabled() ||
        g_boot_first_frame == std::chrono::steady_clock::time_point{}) {
        return;
    }
    // A warm cache can load in well under a second; the version line must still be readable.
    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - g_boot_first_frame < min_shown) {
        g_booting_presenter->DrawBootScreen(headline, detail, 1.0f, one_time);
        SDL_PumpEvents();
        std::this_thread::sleep_for(std::chrono::milliseconds{50});
    }
    LOG_INFO(Render_Vulkan, "Boot screen: shown {} ms, held {} ms for the version line",
             std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - g_boot_first_frame)
                 .count(),
             std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - start)
                 .count());
}

void Presenter::DrawBootScreen(std::string_view headline, std::string_view detail,
                               float fraction, bool one_time) {
    if (window.GetWidth() != swapchain.GetWidth() || window.GetHeight() != swapchain.GetHeight()) {
        swapchain.Recreate(window.GetWidth(), window.GetHeight());
    }
    if (!swapchain.AcquireNextImage()) {
        swapchain.Recreate(window.GetWidth(), window.GetHeight());
        if (!swapchain.AcquireNextImage()) {
            return;
        }
    }

    ImGui::Core::NewFrame(false);
    {
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        const float width = std::min(vp->WorkSize.x * 0.7f, 1100.0f * ImGui::GetIO().FontGlobalScale);
        ImGui::SetNextWindowPos(vp->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(width, 0.0f), ImGuiCond_Always);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.95f, 0.45f, 0.10f, 1.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        if (ImGui::Begin("##boot_screen", nullptr,
                         ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoNav |
                             ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoMove)) {
            ImGui::SetWindowFontScale(2.0f);
            ImGui::TextUnformatted(headline.data(), headline.data() + headline.size());
            ImGui::SetWindowFontScale(1.2f);
            ImGui::Spacing();
            if (one_time) {
                ImGui::TextWrapped("This happens only on the first launch and takes a few minutes "
                                   "(about 2-4, depending on CPU and disk). Closing the game now is "
                                   "safe: the next launch continues where this one stopped.");
                ImGui::Spacing();
            }
            const float f = std::clamp(fraction, 0.0f, 1.0f);
            const auto percent = fmt::format("{:.0f} %", f * 100.0f);
            ImGui::ProgressBar(f, ImVec2(-1.0f, 0.0f), percent.c_str());
            ImGui::Spacing();
            ImGui::TextDisabled("%.*s", static_cast<int>(detail.size()), detail.data());
            // 1.1 (user 10-21): every launch shows the latest release, so players learn about an
            // update without a post in every thread (common/update_notice.cpp).
            const Common::UpdateLine line = Common::GetUpdateLine();
            if (!line.text.empty()) {
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();
                if (line.newer) {
                    ImGui::SetWindowFontScale(1.6f);
                    ImGui::TextColored(ImVec4(0.40f, 0.95f, 0.40f, 1.0f), "%s", line.text.c_str());
                    ImGui::SetWindowFontScale(1.2f);
                    ImGui::TextUnformatted(line.detail.c_str());
                } else {
                    ImGui::TextUnformatted(line.text.c_str());
                }
            }
        }
        ImGui::End();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(2);
    }

    auto& scheduler = present_scheduler;
    auto command_access = scheduler.AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();
    const vk::ImageSubresourceRange range{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };
    const vk::ImageMemoryBarrier to_attachment{
        .srcAccessMask = vk::AccessFlagBits::eNone,
        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = swapchain.Image(),
        .subresourceRange = range,
    };
    cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                           vk::PipelineStageFlagBits::eColorAttachmentOutput,
                           vk::DependencyFlagBits::eByRegion, {}, {}, to_attachment);
    ImGui::Core::Render(cmdbuf, swapchain.ImageView(), swapchain.GetExtent());
    const vk::ImageMemoryBarrier to_present{
        .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
        .dstAccessMask = vk::AccessFlagBits::eNone,
        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .newLayout = vk::ImageLayout::ePresentSrcKHR,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = swapchain.Image(),
        .subresourceRange = range,
    };
    cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                           vk::PipelineStageFlagBits::eBottomOfPipe,
                           vk::DependencyFlagBits::eByRegion, {}, {}, to_present);

    SubmitInfo info{};
    info.AddWait(swapchain.GetImageAcquiredSemaphore());
    info.AddSignal(swapchain.GetPresentReadySemaphore());
    scheduler.Flush(info);
    ImGui::Core::TextureManager::NotePresented();
    bool recreate = false;
    {
        std::scoped_lock submit_lock{Scheduler::submit_mutex};
        recreate = !swapchain.Present();
    }
    if (recreate) {
        swapchain.Recreate(window.GetWidth(), window.GetHeight());
    }
    // A few frames a second: wait, so the next one (or the game's first) starts from idle.
    scheduler.Finish();
}

Presenter::Presenter(Frontend::WindowSDL& window_, AmdGpu::Liverpool* liverpool_)
    : window{window_}, liverpool{liverpool_},
      instance{window, EffectiveGpuId(), EmulatorSettings.IsVkValidationEnabled(),
               EmulatorSettings.IsVkCrashDiagnosticEnabled()},
      draw_scheduler{instance, true}, present_scheduler{instance}, flip_scheduler{instance},
      swapchain{instance, window},
      // SHADPS4_BOOT_SCREEN: WarmUp (inside the Rasterizer) may draw boot frames through `this`.
      rasterizer{(g_booting_presenter = this,
                  std::make_unique<Rasterizer>(instance, draw_scheduler, liverpool))},
      texture_cache{rasterizer->GetTextureCache()} {
    g_booting_presenter = nullptr;
    // q5 (SHADPS4_PRESENT_FRAMES=N): more presentation frames than swapchain images, so the
    // back's flip (PrepareFrame -> GetRenderFrame) does not wait for an older frame's GPU work.
    const u32 num_images = [&] {
        const char* e = std::getenv("SHADPS4_PRESENT_FRAMES");
        const u32 want = e ? static_cast<u32>(std::atoi(e)) : 0u;
        return std::clamp(want, swapchain.GetImageCount(), 8u);
    }();
    LOG_INFO(Render_Vulkan, "Presenter: {} presentation frames (swapchain images {})", num_images,
             swapchain.GetImageCount());
    const vk::Device device = instance.GetDevice();

    // Create presentation frames.
    present_frames.resize(num_images);
    for (u32 i = 0; i < num_images; i++) {
        Frame& frame = present_frames[i];
        frame.id = i;
        auto fence = Check<"create present done fence">(
            device.createFence({.flags = vk::FenceCreateFlagBits::eSignaled}));
        frame.present_done = fence;
        free_queue.push(&frame);
    }

    fsr_settings.enable = EmulatorSettings.IsFsrEnabled();
    fsr_settings.use_rcas = EmulatorSettings.IsRcasEnabled();
    fsr_settings.rcas_attenuation =
        static_cast<float>(EmulatorSettings.GetRcasAttenuation() / 1000.f);

    fsr_pass.Create(device, instance.GetAllocator(), num_images);
    pp_pass.Create(device, swapchain.GetSurfaceFormat().format);

    ImGui::Layer::AddLayer(Common::Singleton<Core::Devtools::Layer>::Instance());
    ImGui::Friends::Register();
    ImGui::ShadNetNotify::Register();
    ImGui::InvitationPrompt::Register();
}

Presenter::~Presenter() {
    ImGui::InvitationPrompt::Unregister();
    ImGui::ShadNetNotify::Unregister();
    ImGui::Friends::Unregister();
    ImGui::Layer::RemoveLayer(Common::Singleton<Core::Devtools::Layer>::Instance());

    draw_scheduler.Finish();
    present_scheduler.Finish();
    flip_scheduler.Finish();
    Check(draw_scheduler.AcquireCommandBuffer().Handle().reset());
    Check(present_scheduler.AcquireCommandBuffer().Handle().reset());
    Check(flip_scheduler.AcquireCommandBuffer().Handle().reset());

    const vk::Device device = instance.GetDevice();
    for (auto& frame : present_frames) {
        vmaDestroyImage(instance.GetAllocator(), frame.image, frame.allocation);
        device.destroyImageView(frame.image_view);
        device.destroyFence(frame.present_done);
    }
}

bool Presenter::IsVideoOutSurface(const AmdGpu::ColorBuffer& color_buffer) const {
    return std::ranges::find(vo_buffers_addr, color_buffer.Address()) != vo_buffers_addr.cend();
}

void Presenter::RecreateFrame(Frame* frame, u32 width, u32 height) {
    const vk::Device device = instance.GetDevice();
    if (frame->imgui_texture) {
        ImGui::Vulkan::RemoveTexture(frame->imgui_texture);
    }
    if (frame->image_view) {
        device.destroyImageView(frame->image_view);
    }
    if (frame->image) {
        vmaDestroyImage(instance.GetAllocator(), frame->image, frame->allocation);
    }

    const vk::Format format = swapchain.GetSurfaceFormat().format;
    const vk::ImageCreateInfo image_info = {
        .flags = vk::ImageCreateFlagBits::eMutableFormat,
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst |
                 vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eSampled,
    };

    VmaAllocationCreateInfo alloc_info = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    VkImage unsafe_image{};
    VkImageCreateInfo unsafe_image_info = static_cast<VkImageCreateInfo>(image_info);

    // v1.0.2: over VMA's budget, retry without it (vk_memory_budget.h) instead of asserting.
    VkResult result = AllocateWithBudgetRetry(
        instance.GetAllocator(), alloc_info, "frame image", u64{width} * height * 4,
        [&] { return fmt::format("{}x{}", width, height); },
        [&](const VmaAllocationCreateInfo& ci) {
            return vmaCreateImage(instance.GetAllocator(), &unsafe_image_info, &ci, &unsafe_image,
                                  &frame->allocation, nullptr);
        });
    if (result != VK_SUCCESS) [[unlikely]] {
        LOG_CRITICAL(Render_Vulkan, "Failed allocating texture with error {}",
                     vk::to_string(vk::Result{result}));
        UNREACHABLE();
    }
    frame->image = vk::Image{unsafe_image};
    SetObjectName(device, frame->image, "Frame image #{}", frame->id);

    const vk::ImageViewCreateInfo view_info = {
        .image = frame->image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    auto view = Check<"create frame image view">(device.createImageView(view_info));
    frame->image_view = view;
    frame->width = width;
    frame->height = height;

    frame->imgui_texture = ImGui::Vulkan::AddTexture(view, vk::ImageLayout::eShaderReadOnlyOptimal);
    frame->is_hdr = swapchain.GetHDR();
}

Frame* Presenter::PrepareLastFrame() {
    if (last_submit_frame == nullptr) {
        return nullptr;
    }

    Frame* frame = last_submit_frame;

    while (true) {
        vk::Result result = instance.GetDevice().waitForFences(frame->present_done, false,
                                                               std::numeric_limits<u64>::max());
        if (result == vk::Result::eSuccess) {
            break;
        }
        if (result == vk::Result::eTimeout) {
            continue;
        }
        ASSERT_MSG(result != vk::Result::eErrorDeviceLost,
                   "Device lost during waiting for a frame");
    }

    auto& scheduler = flip_scheduler;
    scheduler.EndRendering();
    auto command_access = scheduler.AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();

    const auto frame_subresources = vk::ImageSubresourceRange{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };

    const auto pre_barrier =
        vk::ImageMemoryBarrier2{.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentRead,
                                .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                                .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                                .newLayout = vk::ImageLayout::eGeneral,
                                .image = frame->image,
                                .subresourceRange{frame_subresources}};

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &pre_barrier,
    });

    // Flush frame creation commands.
    frame->ready_semaphore = scheduler.GetMasterSemaphore()->Handle();
    frame->ready_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return frame;
}

static vk::Format GetFrameViewFormat(const Libraries::VideoOut::PixelFormat format) {
    switch (format) {
    case Libraries::VideoOut::PixelFormat::A8B8G8R8Srgb:
        return vk::Format::eR8G8B8A8Srgb;
    case Libraries::VideoOut::PixelFormat::A8R8G8B8Srgb:
        return vk::Format::eB8G8R8A8Srgb;
    case Libraries::VideoOut::PixelFormat::A2R10G10B10:
    case Libraries::VideoOut::PixelFormat::A2R10G10B10Srgb:
    case Libraries::VideoOut::PixelFormat::A2R10G10B10Bt2020Pq:
        return vk::Format::eA2R10G10B10UnormPack32;
    default:
        break;
    }
    UNREACHABLE_MSG("Unknown format={}", static_cast<u32>(format));
    return {};
}

Frame* Presenter::PrepareFrame(const Libraries::VideoOut::BufferAttributeGroup& attribute,
                               VAddr cpu_address) {
    // q5: phase timers (find / update the display image, GetRenderFrame, the rest).
    static std::array<u64, 5> q5{}; // frames, find, getframe, rest, max getframe
    const u64 q5_t0 = __rdtsc();
    auto desc = VideoCore::TextureCache::ImageDesc{attribute, cpu_address};
    const auto image_id = texture_cache.FindImage(desc);
    texture_cache.UpdateImage(image_id);
    const u64 q5_t1 = __rdtsc();

    Frame* frame = GetRenderFrame();
    const u64 q5_t2 = __rdtsc();
    SCOPE_EXIT {
        const u64 t3 = __rdtsc();
        ++q5[0];
        q5[1] += q5_t1 - q5_t0;
        q5[2] += q5_t2 - q5_t1;
        q5[3] += t3 - q5_t2;
        q5[4] = std::max(q5[4], q5_t2 - q5_t1);
        if ((q5[0] & 1023) == 0) {
            const double n = 1024.0 * 3.5e3; // cycles -> us at ~3.5 GHz, per frame
            LOG_INFO(Render_Vulkan,
                     "q5 PrepareFrame per flip (us, last 1024): find/update image {:.0f} "
                     "GetRenderFrame {:.0f} (max {:.0f}) rest {:.0f}",
                     double(q5[1]) / n, double(q5[2]) / n, double(q5[4]) / 3.5e3,
                     double(q5[3]) / n);
            q5[1] = q5[2] = q5[3] = q5[4] = 0;
        }
    };

    const auto frame_subresources = vk::ImageSubresourceRange{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };

    const auto pre_barrier = vk::ImageMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .image = frame->image,
        .subresourceRange{frame_subresources},
    };

    draw_scheduler.EndRendering();

    VideoCore::ImageViewInfo view_info{};
    view_info.format = GetFrameViewFormat(attribute.attrib.pixel_format);
    // Exclude alpha from output frame to avoid blending with UI.
    view_info.mapping.a = vk::ComponentSwizzle::eOne;

    auto& image = texture_cache.GetImage(image_id);
    auto image_view = *image.FindView(view_info).image_view;
    const vk::Extent2D image_size = {image.info.size.width, image.info.size.height};
    expected_ratio = static_cast<float>(image_size.width) / static_cast<float>(image_size.height);

    // SHADPS4_DLSS_UPSCALE (vk_dlss_upscale.h): this frame's DLSS output at the window size with
    // the game's HUD layer over it replaces the display buffer as the post pass's input. The
    // layer's image is found and moved here (stream order: after this frame's HUD draws, before
    // the next frame's resolve clears it).
    // 1.1.1 (GitHub #13): the game's own HDR10 output (PQ display buffers; the game turns it on by
    // itself when an HDR TV is reported, s1571). On an SDR screen the post pass decodes PQ (mode 2).
    // The DLSS composite below is kept off for PQ frames: it builds its picture from DLSS's scene
    // image, not from the game's PQ encode. Measured s1578: in the game's HDR mode the DLSS taps
    // never fire (0 resolves), so DLSS is off there anyway - this only guards a future hook.
    // SHADPS4_PQ_DLSS_COMPOSITE=1 (test) keeps the composite for PQ frames too (v1.1).
    const bool game_pq =
        attribute.attrib.pixel_format == Libraries::VideoOut::PixelFormat::A2R10G10B10Bt2020Pq;
    static const bool pq_composite = [] {
        const char* v = std::getenv("SHADPS4_PQ_DLSS_COMPOSITE");
        return v && v[0] == '1';
    }();
    std::optional<Dlss::CompositeJob> dlss_job;
    if (Dlss::Upscaler* upscaler = Dlss::GetUpscaler(instance)) {
        if (upscaler->Ready() && !(Dlss::UpscaleTestBits() & 1) && (!game_pq || pq_composite)) {
            const auto hud_id = texture_cache.FindImageAtSize(
                Dlss::HudLayerAddr(), Dlss::g_game_width.load(std::memory_order_relaxed),
                Dlss::g_game_height.load(std::memory_order_relaxed));
            if (hud_id) {
                auto& hud = texture_cache.GetImage(hud_id);
                hud.Transit(vk::ImageLayout::eShaderReadOnlyOptimal,
                            vk::AccessFlagBits2::eShaderRead, {});
                VideoCore::ImageViewInfo hud_view_info{};
                hud_view_info.type = AmdGpu::ImageType::Color2D;
                hud_view_info.format = hud.info.pixel_format;
                dlss_job = upscaler->TakeJob(*hud.FindView(hud_view_info).image_view);
            }
        }
        // SHADPS4_HUD_NATIVE (vk_dlss_upscale.h): this frame's HUD went to the upscaler's own
        // image; the redirect ends at every flip (the layer's memory is an ordinary target
        // earlier in the next frame).
        if (upscaler->NativeHudTarget()) {
            const vk::ImageView native_hud = upscaler->EndNativeHud(draw_scheduler);
            if (dlss_job) {
                dlss_job->hud_view = native_hud;
            }
        }
        upscaler->SetOutputSize(frame->width, frame->height, game_pq && !pq_composite);
    }

    // DLSS 5 on video cutscenes (vk_dlss5.h): a frame the game did not render through the T2x
    // pass while its video decoder runs goes through the model here, before the post pass.
    static u64 dlss5_resolves = 0;
    const bool dlss5_video =
        !dlss_job && !frame->is_hdr && !game_pq && Dlss5::VideoFrame(dlss5_resolves);
    dlss5_resolves = Dlss5::g_resolves.load(std::memory_order_acquire);
    // 1.1.1 (GitHub #13): a PQ frame on an SDR screen is decoded to SDR by the post pass (mode 2);
    // an SDR frame (or the DLSS composite, which is SDR) on an HDR screen is encoded to PQ (mode 3,
    // SDR white at SHADPS4_HDR_SDR_WHITE nits, default 203 = BT.2408's reference white).
    HostPasses::PostProcessingPass::Settings frame_pp = pp_settings;
    if (game_pq && !frame->is_hdr && !dlss_job) {
        frame_pp.hdr = 2;
    } else if (frame->is_hdr && (!game_pq || dlss_job)) {
        static const float sdr_white = [] {
            const char* v = std::getenv("SHADPS4_HDR_SDR_WHITE");
            const float nits = v ? float(std::atof(v)) : 0.0f;
            return nits >= 40.0f && nits <= 1000.0f ? nits : 203.0f;
        }();
        frame_pp.hdr = 3;
        frame_pp.sdr_white = sdr_white;
    }
    {
        // Log only: 1 PQ as is, 2 PQ decoded, 3 SDR encoded to PQ, 4 PQ through the composite.
        static std::atomic<int> last_mode{-1};
        const int mode = game_pq ? (frame->is_hdr ? (dlss_job ? 4 : 1) : dlss_job ? 4 : 2)
                                 : (frame->is_hdr ? 3 : 0);
        if (last_mode.exchange(mode) != mode && mode != 0) {
            LOG_INFO(Render_Vulkan, "Presenter: {}",
                     mode == 1   ? "the game sends HDR10 (PQ) frames - shown on the HDR screen as they are"
                     : mode == 2 ? "the game sends HDR10 (PQ) frames - decoded to SDR for this screen "
                                   "(DLSS composite skipped)"
                     : mode == 3 ? "SDR frames on the HDR screen - encoded to PQ"
                                 : "the game sends HDR10 (PQ) frames - through the DLSS composite "
                                   "(SHADPS4_PQ_DLSS_COMPOSITE test)");
        }
    }
    // FSR's HDR handling follows the frame's content, not the screen's mode.
    const bool fsr_hdr = frame->is_hdr && game_pq;
    // The post pass treats what it samples as linear for every SDR frame (an sRGB view decodes,
    // a UNORM one is taken as is), so DLSS 5 does the same (review 2).
    const bool frame_linear = true;

    const u32 capture_game_only_count = VideoCore::ConsumeGameOnlyScreenshotRequests();
    // q6 (SHADPS4_PRESENT_ASYNC=1): no synchronous borrow - AcquireCommandBuffer drains the
    // recording lanes (s620: ~1.5 ms per flip on the back). The display image's Transit goes
    // through the scheduler (its state on this thread), the passes are one closure recorded in
    // stream order; the flush below hands it on as before.
    static const bool present_async = [] {
        const char* e = std::getenv("SHADPS4_PRESENT_ASYNC");
        return e && e[0] == '1';
    }();
    if (present_async && capture_game_only_count == 0 &&
        GuestLayout(vk::ImageLayout::eShaderReadOnlyOptimal) ==
            vk::ImageLayout::eShaderReadOnlyOptimal) { // q6b (3)
        image.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead,
                      {});
        draw_scheduler.EndRendering();
        // q6b (2): ordered (FsrPass keeps state across flips); the Frame by value.
        draw_scheduler.RecordOrdered([this, pre_barrier, image_view, image_size, fc = *frame,
                                      fsr = fsr_settings, pp = frame_pp, dlss_job, dlss5_video,
                                      frame_linear, fsr_hdr](vk::CommandBuffer cb) mutable {
            cb.pipelineBarrier2(vk::DependencyInfo{
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &pre_barrier,
            });
            if (dlss_job) {
                Dlss::RecordComposite(cb, *dlss_job);
                pp_pass.Render(cb, dlss_job->out_view, dlss_job->size, fc, pp);
                return;
            }
            vk::ImageView source = image_view;
            if (dlss5_video) {
                if (const auto v = Dlss5::ApplyVideo(instance, cb, image_view, image_size.width,
                                                     image_size.height, frame_linear)) {
                    source = *v;
                }
            }
            const auto view =
                fsr_pass.Render(cb, source, image_size, {fc.width, fc.height}, fsr, fsr_hdr);
            pp_pass.Render(cb, view, image_size, fc, pp);
        });
        DebugState.game_resolution = {image_size.width, image_size.height};
        DebugState.output_resolution = {frame->width, frame->height};
        frame->ready_semaphore = draw_scheduler.GetMasterSemaphore()->Handle();
        frame->ready_tick = draw_scheduler.CurrentTick();
        SubmitInfo info{};
        draw_scheduler.Flush(info);
        return frame;
    }
    auto command_access = draw_scheduler.AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &pre_barrier,
    });
    std::vector<ScreenshotReadback> pending_screenshots;
    if (capture_game_only_count > 0) {
        pending_screenshots.reserve(1);
        const bool hdr_encoded =
            attribute.attrib.pixel_format == Libraries::VideoOut::PixelFormat::A2R10G10B10Bt2020Pq;
        pending_screenshots.emplace_back(
            instance, draw_scheduler, ScreenshotKind::GameOnly,
            BuildScreenshotPaths(ScreenshotKind::GameOnly, capture_game_only_count),
            image_size.width, image_size.height, view_info.format, hdr_encoded);
        auto& readback = pending_screenshots.back();

        // Capture the guest output before any host-side scaling (FSR/PP) is applied.
        image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
                      cmdbuf);
        CopyImageToReadback(cmdbuf, image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                            readback);
    }

    // Continue with host-side passes that draw the displayed (scaled) frame.
    image.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead, {},
                  cmdbuf);

    if (dlss_job) {
        Dlss::RecordComposite(cmdbuf, *dlss_job);
        pp_pass.Render(cmdbuf, dlss_job->out_view, dlss_job->size, *frame, frame_pp);
    } else {
        if (dlss5_video) {
            if (const auto v = Dlss5::ApplyVideo(instance, cmdbuf, image_view, image_size.width,
                                                 image_size.height, frame_linear)) {
                image_view = *v;
            }
        }
        image_view = fsr_pass.Render(cmdbuf, image_view, image_size,
                                     {frame->width, frame->height}, fsr_settings, fsr_hdr);
        pp_pass.Render(cmdbuf, image_view, image_size, *frame, frame_pp);
    }

    DebugState.game_resolution = {image_size.width, image_size.height};
    DebugState.output_resolution = {frame->width, frame->height};

    std::shared_ptr<std::vector<ScreenshotReadback>> deferred_screenshots{};
    if (!pending_screenshots.empty()) {
        deferred_screenshots =
            std::make_shared<std::vector<ScreenshotReadback>>(std::move(pending_screenshots));
        draw_scheduler.DeferPriorityOperation(
            [deferred_screenshots]() { SavePendingScreenshots(*deferred_screenshots); });
    }

    // Flush frame creation commands.
    frame->ready_semaphore = draw_scheduler.GetMasterSemaphore()->Handle();
    frame->ready_tick = draw_scheduler.CurrentTick();
    SubmitInfo info{};
    draw_scheduler.Flush(info);
    return frame;
}

Frame* Presenter::PrepareBlankFrame(bool present_thread) {
    // Request a free presentation frame.
    Frame* frame = GetRenderFrame();

    auto& scheduler = present_thread ? present_scheduler : draw_scheduler;
    scheduler.EndRendering();

    auto command_access = scheduler.AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();

    constexpr vk::ImageSubresourceRange simple_subresource = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .levelCount = 1,
        .layerCount = 1,
    };
    const auto pre_barrier = vk::ImageMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .image = frame->image,
        .subresourceRange = simple_subresource,
    };

    const auto post_barrier = vk::ImageMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = frame->image,
        .subresourceRange = simple_subresource,
    };

    const vk::RenderingAttachmentInfo attachment = {
        .imageView = frame->image_view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    const vk::RenderingInfo rendering_info = {
        .renderArea =
            {
                .extent = {frame->width, frame->height},
            },
        .layerCount = 1,
        .colorAttachmentCount = 1u,
        .pColorAttachments = &attachment,
    };

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &pre_barrier,
    });

    cmdbuf.beginRendering(rendering_info);
    cmdbuf.endRendering();

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &post_barrier,
    });

    // Flush frame creation commands.
    frame->ready_semaphore = scheduler.GetMasterSemaphore()->Handle();
    frame->ready_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return frame;
}

void Presenter::ReleaseFrame(Frame* frame) {
    std::scoped_lock fl{free_mutex};
    free_queue.push(frame);
    free_cv.notify_one();
}

void Presenter::Present(Frame* frame, bool is_reusing_frame, bool hold_frame) {
    // Free the frame for reuse
    const auto free_frame = [&] {
        if (!is_reusing_frame) {
            last_submit_frame = frame;
            if (!hold_frame) {
                ReleaseFrame(frame);
            }
        }
    };

    // Recreate the swapchain if the window was resized or F3's Present Mode changed (live).
    if (window.GetWidth() != swapchain.GetWidth() || window.GetHeight() != swapchain.GetHeight() ||
        swapchain.PresentModeChanged()) {
        swapchain.Recreate(window.GetWidth(), window.GetHeight());
    }

    const auto acquire_start = Common::FaultMonitor::Now();
    if (!swapchain.AcquireNextImage()) {
        swapchain.Recreate(window.GetWidth(), window.GetHeight());
        if (!swapchain.AcquireNextImage()) {
            // User resizes the window too fast and GPU can't keep up. Skip this frame.
            LOG_WARNING(Render_Vulkan, "Skipping frame!");
            free_frame();
            return;
        }
    }

    const u64 acquire_ns = Common::FaultMonitor::SinceNs(acquire_start);

    // Reset fence for queue submission. Do it here instead of GetRenderFrame() because we may
    // skip frame because of slow swapchain recreation. If a frame skip occurs, we skip signal
    // the frame's present fence and future GetRenderFrame() call will hang waiting for this frame.
    const auto reset_result = instance.GetDevice().resetFences(frame->present_done);
    ASSERT_MSG(reset_result == vk::Result::eSuccess,
               "Unexpected error resetting present done fence: {}", vk::to_string(reset_result));

    ImGuiID dockId = ImGui::Core::NewFrame(is_reusing_frame);

    const vk::Image swapchain_image = swapchain.Image();
    const vk::ImageView swapchain_image_view = swapchain.ImageView();

    auto& scheduler = present_scheduler;
    auto command_access = scheduler.AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();
    const u32 capture_with_overlays_count = VideoCore::ConsumeWithOverlaysScreenshotRequests();
    std::vector<ScreenshotReadback> pending_screenshots;
    if (capture_with_overlays_count > 0) {
        pending_screenshots.reserve(1);
    }

    if (EmulatorSettings.IsVkHostMarkersEnabled()) {
        cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
            .pLabelName = "Present",
        });
    }

    {
        auto* profiler_ctx = instance.GetProfilerContext();
        TracyVkNamedZoneC(profiler_ctx, renderer_gpu_zone, cmdbuf, "Host frame",
                          MarkersPalette::GpuMarkerColor, profiler_ctx != nullptr);

        const vk::Extent2D extent = swapchain.GetExtent();
        const std::array pre_barriers{
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eNone,
                .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = swapchain_image,
                .subresourceRange{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            },
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                .dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = frame->image,
                .subresourceRange{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            },
        };

        bool swapchain_copied_for_screenshot = false;

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                               vk::PipelineStageFlagBits::eColorAttachmentOutput,
                               vk::DependencyFlagBits::eByRegion, {}, {}, pre_barriers);

        { // Draw the game
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{0.0f});
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
            ImGui::SetNextWindowDockID(dockId, ImGuiCond_Once);
            if (ImGui::Begin("Display##game_display", nullptr, ImGuiWindowFlags_NoNav)) {
                auto game_texture = frame->imgui_texture;
                auto game_width = frame->width;
                auto game_height = frame->height;

                if (Libraries::SystemService::IsSplashVisible()) { // draw splash
                    if (!splash_img.has_value()) {
                        splash_img.emplace();
                        const auto& splash_data = Common::ElfInfo::Instance().GetSplashData();
                        if (!splash_data.empty()) {
                            splash_img = ImGui::RefCountedTexture::DecodePngTexture(splash_data);
                        }
                    }
                    if (auto& splash_image = this->splash_img.value()) {
                        auto [im_id, width, height] = splash_image.GetTexture();
                        game_texture = im_id;
                        game_width = width;
                        game_height = height;
                    }
                }

                ImVec2 contentArea = ImGui::GetContentRegionAvail();
                SetExpectedGameSize((s32)contentArea.x, (s32)contentArea.y);

                const auto imgRect =
                    FitImage(game_width, game_height, (s32)contentArea.x, (s32)contentArea.y);
                ImVec2 offset{
                    static_cast<float>(imgRect.offset.x),
                    static_cast<float>(imgRect.offset.y),
                };
                ImVec2 size{
                    static_cast<float>(imgRect.extent.width),
                    static_cast<float>(imgRect.extent.height),
                };

                ImGui::SetCursorPos(ImGui::GetCursorStartPos() + offset);
                ImGui::Image(game_texture, size);

                if (EmulatorSettings.IsNullGPU()) {
                    Core::Devtools::Layer::DrawNullGpuNotice();
                }
            }
            ImGui::End();
            ImGui::PopStyleVar(3);
            ImGui::PopStyleColor();
        }
        ImGui::Core::Render(cmdbuf, swapchain_image_view, swapchain.GetExtent());

        if (capture_with_overlays_count > 0) {
            pending_screenshots.emplace_back(
                instance, scheduler, ScreenshotKind::WithOverlays,
                BuildScreenshotPaths(ScreenshotKind::WithOverlays, capture_with_overlays_count),
                extent.width, extent.height,
                swapchain.GetHDR() ? vk::Format::eA2B10G10R10UnormPack32
                                   : swapchain.GetSurfaceFormat().format,
                swapchain.GetHDR());
            auto& readback = pending_screenshots.back();

            const vk::ImageMemoryBarrier to_transfer{
                .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = swapchain_image,
                .subresourceRange{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            };

            cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                   vk::PipelineStageFlagBits::eTransfer,
                                   vk::DependencyFlagBits::eByRegion, {}, {}, to_transfer);
            CopyImageToReadback(cmdbuf, swapchain_image, vk::ImageLayout::eTransferSrcOptimal,
                                readback);
            swapchain_copied_for_screenshot = true;
        }

        const vk::AccessFlags post_src_access_mask =
            swapchain_copied_for_screenshot ? vk::AccessFlagBits::eTransferRead
                                            : vk::AccessFlagBits::eColorAttachmentWrite;
        const vk::ImageLayout post_old_layout = swapchain_copied_for_screenshot
                                                    ? vk::ImageLayout::eTransferSrcOptimal
                                                    : vk::ImageLayout::eColorAttachmentOptimal;
        const vk::ImageMemoryBarrier post_barrier{
            .srcAccessMask = post_src_access_mask,
            .dstAccessMask = vk::AccessFlagBits::eNone,
            .oldLayout = post_old_layout,
            .newLayout = vk::ImageLayout::ePresentSrcKHR,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = swapchain_image,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        };
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                               vk::PipelineStageFlagBits::eAllCommands,
                               vk::DependencyFlagBits::eByRegion, {}, {}, post_barrier);

        if (profiler_ctx) {
            TracyVkCollect(profiler_ctx, cmdbuf);
        }
    }
    if (EmulatorSettings.IsVkHostMarkersEnabled()) {
        cmdbuf.endDebugUtilsLabelEXT();
    }

    // Flush vulkan commands.
    std::shared_ptr<std::vector<ScreenshotReadback>> deferred_screenshots{};
    if (!pending_screenshots.empty()) {
        deferred_screenshots =
            std::make_shared<std::vector<ScreenshotReadback>>(std::move(pending_screenshots));
        scheduler.DeferPriorityOperation(
            [deferred_screenshots]() { SavePendingScreenshots(*deferred_screenshots); });
    }

    SubmitInfo info{};
    info.AddWait(swapchain.GetImageAcquiredSemaphore());
    info.AddWait(frame->ready_semaphore, frame->ready_tick);
    info.AddSignal(swapchain.GetPresentReadySemaphore());
    info.AddSignal(frame->present_done);
    const auto flush_start = Common::FaultMonitor::Now();
    scheduler.Flush(info);
    // This frame's ImGui draws are submitted (texture destroys wait for two of these).
    ImGui::Core::TextureManager::NotePresented();
    const u64 flush_ns = Common::FaultMonitor::SinceNs(flush_start);

    // Present to swapchain.
    u64 lock_ns = 0;
    u64 khr_ns = 0;
    bool recreate_after_present = false;
    {
        // submit_mutex is static and every one of the ~208 submits/frame the readback path makes
        // takes it too, so time the acquisition separately from the present itself.
        const auto lock_start = Common::FaultMonitor::Now();
        std::scoped_lock submit_lock{Scheduler::submit_mutex};
        lock_ns = Common::FaultMonitor::SinceNs(lock_start);

        const auto khr_start = Common::FaultMonitor::Now();
        if (!swapchain.Present()) {
            if (SwapchainSafeWait()) {
                recreate_after_present = true; // its wait takes submit_mutex: after the lock
            } else {
                swapchain.Recreate(window.GetWidth(), window.GetHeight());
            }
        }
        khr_ns = Common::FaultMonitor::SinceNs(khr_start);
    }
    if (recreate_after_present) {
        swapchain.Recreate(window.GetWidth(), window.GetHeight());
    }
    Common::FaultMonitor::CountPresentPhases(acquire_ns, flush_ns, lock_ns, khr_ns);

    free_frame();
    if (!is_reusing_frame) {
        // Drive the RenderDoc capture state machine on real flips only, so a capture starts just
        // after one present and ends at the next: exactly one displayed frame.
        VideoCore::OnFrameBoundary();
        Common::FaultMonitor::CountPresentedFrame();
        DebugState.IncFlipFrameNum();
    }
}

Frame* Presenter::GetRenderFrame() {
    // Wait for free presentation frames
    Frame* frame;
    const auto queue_start = Common::FaultMonitor::Now();
    {
        std::unique_lock lock{free_mutex};
        free_cv.wait(lock, [this] { return !free_queue.empty(); });
        LOG_DEBUG(Render_Vulkan, "Got render frame, remaining {}", free_queue.size() - 1);

        // Take the frame from the queue
        frame = free_queue.front();
        free_queue.pop();
    }
    const u64 queue_ns = Common::FaultMonitor::SinceNs(queue_start);

    const vk::Device device = instance.GetDevice();
    vk::Result result{};

    const auto wait = [&]() {
        result = device.waitForFences(frame->present_done, false, std::numeric_limits<u64>::max());
        return result;
    };

    // Wait for the presentation to be finished so all frame resources are free
    const auto fence_start = Common::FaultMonitor::Now();
    while (wait() != vk::Result::eSuccess) {
        ASSERT_MSG(result != vk::Result::eErrorDeviceLost,
                   "Device lost during waiting for a frame");
        // Retry if the waiting times out
        if (result == vk::Result::eTimeout) {
            continue;
        }
    }
    Common::FaultMonitor::CountRenderFrame(queue_ns, Common::FaultMonitor::SinceNs(fence_start));

    if (frame->width != expected_frame_width || frame->height != expected_frame_height ||
        frame->is_hdr != swapchain.GetHDR()) {
        RecreateFrame(frame, expected_frame_width, expected_frame_height);
    }

    return frame;
}

void Presenter::SetExpectedGameSize(s32 width, s32 height) {
    const float ratio = (float)width / (float)height;

    expected_frame_height = height;
    expected_frame_width = width;
    if (ratio > expected_ratio) {
        expected_frame_width = static_cast<s32>(height * expected_ratio);
    } else {
        expected_frame_height = static_cast<s32>(width / expected_ratio);
    }
}

} // namespace Vulkan
