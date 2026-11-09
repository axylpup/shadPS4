// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <deque>
#include <memory>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_dlss.h"
#include "video_core/texture_cache/image.h"

namespace Vulkan {
class Instance;
class Scheduler;
}

// SHADPS4_DLSS_UPSCALE=1 (with SHADPS4_DLSS=1; plan notes/study-dlss-plan-2026-10-14.md sec. 4,
// U1): the game renders 1920x1080 as always; DLSS outputs the window size (e.g. 2560x1440, the
// Quality mode range) into a host image instead of the T2x resolve's target. That target
// (0x26bd70000) is cleared to 0 and the HUD draws into it also write alpha, so it ends as a
// premultiplied HUD layer (they blend ONE / ONE_MINUS_SRC_ALPHA, measured). At the flip the
// presenter composites HUD over DLSS at the window size and its post pass draws that 1:1; the
// game's own final copy still runs (its display buffer, HUD over black, is not shown).
// Off when the window is not larger than 1920x1080 (then DLSS = DLAA in place) or the driver has
// no dynamic colour write mask.
namespace Vulkan::Dlss {

/// The resolve's target (0x26bd70000 at 1080p; learned per frame, vk_dlss.h "Any render size").
inline VAddr HudLayerAddr() {
    return g_hud_addr.load(std::memory_order_relaxed);
}

bool UpscaleWanted();

/// SHADPS4_DLSS_UPSCALE_TEST bits (crash bisect, 2026-10-17): 1 = no composite at the flip (the
/// display buffer is shown), 2 = no alpha on HUD draws, 4 = no clear of the resolve target.
u32 UpscaleTestBits();

/// SHADPS4_HUD_NATIVE (off by default): with the HUD-layer mode on, the game's HUD draws after
/// the resolve (the passes whose colour target is the HUD layer) render into the upscaler's own
/// image at an integer multiple of the render size instead of the game's layer - sharp text and
/// menus at render sizes below the window (e.g. 67% of 1440p). The game's images are untouched
/// (its layer stays cleared). =1: the smallest integer scale that reaches the DLSS output size;
/// =2..4: that scale. The composite supersamples the larger layer down to the output.
u32 HudNativeSetting();

/// The native HUD scale for a HUD layer of `game` size under an output of `output` size (1 = none).
u32 HudNativeScale(vk::Extent2D game, vk::Extent2D output);

/// The HUD-layer mode is on for this frame's draws (the presenter decides it each flip).
inline std::atomic<bool> g_hud_layer{false};

inline bool HudLayerActive() {
    return g_hud_layer.load(std::memory_order_relaxed) &&
           g_user_on.load(std::memory_order_relaxed);
}

/// Draw paths: a draw into the HUD layer that writes RGB writes RGBA (coverage in alpha).
template <typename Masks>
inline void ApplyHudLayerMask(Masks& masks, VAddr cb0_address) {
    constexpr vk::ColorComponentFlags rgb =
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB;
    if (cb0_address == HudLayerAddr() && masks[0] == rgb && HudLayerActive() &&
        !(UpscaleTestBits() & 2)) {
        masks[0] = rgb | vk::ColorComponentFlagBits::eA;
    }
}

/// What the presenter's recorded closure needs (plain handles; decided on the GPU thread).
struct CompositeJob {
    vk::Pipeline pipeline;
    vk::PipelineLayout layout;
    vk::ImageView hud_view;  ///< the HUD layer (game resolution), shader-read layout
    vk::ImageView scene_view; ///< DLSS output, shader-read layout
    vk::Image out_image;
    vk::ImageView out_view;
    vk::Extent2D size;
};
/// Records the composite (barriers + dispatch) into `cmdbuf`; any thread.
void RecordComposite(vk::CommandBuffer cmdbuf, const CompositeJob& job);

class Upscaler {
public:
    explicit Upscaler(const Instance& instance);
    ~Upscaler();

    /// Presenter, each flip (GPU thread): the output size; turns the HUD-layer mode on / off.
    /// `game_pq`: this flip is the game's HDR10 (PQ) output - the HUD-layer mode stays off.
    void SetOutputSize(u32 width, u32 height, bool game_pq = false);
    vk::Extent2D OutputSize() const {
        return output_size;
    }

    /// Resolve (GPU thread): the DLSS output image at the output size, moved to GENERAL for the
    /// evaluate with a barrier after the previous composite's read.
    EvalImage SceneTarget(vk::CommandBuffer cmdbuf);
    /// Resolve, after the evaluate: write -> read barrier; the next flip may composite.
    void SceneWritten(vk::CommandBuffer cmdbuf);

    /// Flip (GPU thread): this frame's DLSS output is waiting.
    bool Ready() const {
        return ready;
    }
    /// Flip: the job for this frame (and ready cleared). hud_view = the HUD layer.
    CompositeJob TakeJob(vk::ImageView hud_view);

    /// SHADPS4_HUD_NATIVE, resolve (GPU thread): (re)creates the native HUD image at game x
    /// scale, clears it to 0 and leaves it as a colour attachment; the redirect is on until the
    /// flip.
    void BeginNativeHud(Scheduler& scheduler, vk::Extent2D game, vk::Format format, u32 scale);
    struct NativeHud {
        vk::Image image;
        vk::ImageView view;
        vk::Format format;
        vk::Extent2D game; ///< the HUD layer's guest size
        u32 scale;
    };
    /// Draw paths (GPU thread): the redirect target while it is on, else nullptr.
    const NativeHud* NativeHudTarget() const {
        return hud_on ? &hud_info : nullptr;
    }
    /// Flip (GPU thread): ends the redirect; with it on, records the write -> read barrier and
    /// returns the native image's view for the composite (else a null view).
    vk::ImageView EndNativeHud(Scheduler& scheduler);
    /// Draw paths (GPU thread): the HUD passes' depth at the native scale (most HUD passes bind
    /// the scene depth). The first call of a frame records a nearest-filter blit of `src` (the
    /// game's depth, already in TRANSFER_SRC) into it; layout GENERAL afterwards. Returns a null
    /// view when the format cannot be blitted.
    vk::ImageView NativeHudDepth(Scheduler& scheduler, vk::Image src, vk::Format format,
                                 vk::ImageAspectFlags aspect);
    /// This frame's depth copy was recorded (the caller then skips the source transition).
    bool NativeHudDepthCopied() const {
        return hud_depth_ready;
    }
    /// Draw paths (GPU thread): a pass was sent to the native HUD image (statistics).
    void NoteNativeHudPass() const {
        ++hud_passes;
    }

private:
    struct Target {
        VideoCore::UniqueImage image;
        vk::UniqueImageView view;
    };
    void Resize(vk::Extent2D size);

    const Instance& instance;
    vk::Device device;
    vk::UniqueSampler sampler;
    vk::UniqueDescriptorSetLayout set_layout;
    vk::UniquePipelineLayout pipeline_layout;
    vk::UniquePipeline pipeline;
    std::unique_ptr<Target> scene; ///< DLSS output, RGBA8 UNORM (storage)
    std::unique_ptr<Target> out;   ///< composite, RGBA16F linear
    std::unique_ptr<Target> hud;   ///< SHADPS4_HUD_NATIVE: the HUD at game x scale
    NativeHud hud_info{};
    bool hud_on{};
    mutable u64 hud_passes{};  ///< passes redirected this frame
    std::unique_ptr<Target> hud_depth; ///< SHADPS4_HUD_NATIVE: scene depth at game x scale
    vk::Format hud_depth_format{};
    vk::Extent2D hud_depth_size{};
    bool hud_depth_ready{}; ///< copied this frame
    u64 hud_frames{}, hud_frames_empty{}, hud_passes_total{};
    vk::ImageLayout scene_layout{vk::ImageLayout::eUndefined};
    vk::Extent2D output_size{};
    vk::Extent2D image_size{};
    bool ready{};
    u64 flips{};
    std::deque<std::pair<u64, std::unique_ptr<Target>>> retired; ///< freed some flips later
};

/// The one upscaler (created on first use; nullptr when not wanted).
Upscaler* GetUpscaler(const Instance& instance);

} // namespace Vulkan::Dlss
