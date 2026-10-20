// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_hints.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_properties.h>
#include <SDL3/SDL_timer.h>
#include <SDL3/SDL_video.h>
#include <cmrc/cmrc.hpp>
#include <stb_image.h>

#include "common/assert.h"
#include "common/elf_info.h"
#include "common/io_file.h"
#include "common/logging/formatter.h"
#include "common/scope_exit.h"
#include "core/debug_state.h"
#include "core/devtools/layer.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/time.h"
#include "core/libraries/pad/pad.h"
#include "core/libraries/system/userservice.h"
#include "core/mouse_aim.h"
#include "imgui/big_picture/settings_dialog_layer.h"
#include "imgui/dlss5_panel.h"
#include "core/user_settings.h"
#include "imgui/friends_layer.h"
#include "imgui/renderer/imgui_core.h"
#include "input/controller.h"
#include "input/input_handler.h"
#include "input/input_mouse.h"
#include "input/pad_gestures.h"
#include "sdl_window.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_dlss.h"
#include "video_core/renderer_vulkan/vk_dlss5.h"
#include "video_core/texture_cache/tex_replace.h"

#ifdef __APPLE__
#include <SDL3/SDL_metal.h>
#endif
#include <core/emulator_settings.h>
#include "core/libraries/mouse/sdl_mouse.h"

CMRC_DECLARE(res);

namespace Frontend {

using namespace Libraries::Pad;

static OrbisPadButtonDataOffset SDLGamepadToOrbisButton(u8 button) {
    using OPBDO = OrbisPadButtonDataOffset;

    switch (button) {
    case SDL_GAMEPAD_BUTTON_DPAD_DOWN:
        return OPBDO::Down;
    case SDL_GAMEPAD_BUTTON_DPAD_UP:
        return OPBDO::Up;
    case SDL_GAMEPAD_BUTTON_DPAD_LEFT:
        return OPBDO::Left;
    case SDL_GAMEPAD_BUTTON_DPAD_RIGHT:
        return OPBDO::Right;
    case SDL_GAMEPAD_BUTTON_SOUTH:
        return OPBDO::Cross;
    case SDL_GAMEPAD_BUTTON_NORTH:
        return OPBDO::Triangle;
    case SDL_GAMEPAD_BUTTON_WEST:
        return OPBDO::Square;
    case SDL_GAMEPAD_BUTTON_EAST:
        return OPBDO::Circle;
    case SDL_GAMEPAD_BUTTON_START:
        return OPBDO::Options;
    case SDL_GAMEPAD_BUTTON_TOUCHPAD:
        return OPBDO::TouchPad;
    case SDL_GAMEPAD_BUTTON_BACK:
        return OPBDO::TouchPad;
    case SDL_GAMEPAD_BUTTON_LEFT_SHOULDER:
        return OPBDO::L1;
    case SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER:
        return OPBDO::R1;
    case SDL_GAMEPAD_BUTTON_LEFT_STICK:
        return OPBDO::L3;
    case SDL_GAMEPAD_BUTTON_RIGHT_STICK:
        return OPBDO::R3;
    default:
        return OPBDO::None;
    }
}

static Uint32 SDLCALL PollController(void* userdata, SDL_TimerID timer_id, Uint32 interval) {
    auto* controller = reinterpret_cast<Input::GameController*>(userdata);
    controller->PollState();
    return interval;
}

static Uint32 SDLCALL PollControllerLightColour(void* userdata, SDL_TimerID timer_id,
                                                Uint32 interval) {
    auto* controller = reinterpret_cast<Input::GameController*>(userdata);
    controller->PollLightColour();
    return interval;
}

WindowSDL::WindowSDL(s32 width_, s32 height_, Input::GameControllers* controllers_,
                     std::string_view window_title)
    : width{width_}, height{height_}, controllers{*controllers_} {
    if (!SDL_SetHint(SDL_HINT_APP_NAME, "shadPS4")) {
        UNREACHABLE_MSG("Failed to set SDL window hint: {}", SDL_GetError());
    }
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        UNREACHABLE_MSG("Failed to initialize SDL video subsystem: {}", SDL_GetError());
    }
    // On macOS, the future Intel compatibility environment does not include camera frameworks.
    // Just skip initializing it entirely, no point in splitting old vs new OS versions here.
#ifndef __APPLE__
    if (!SDL_Init(SDL_INIT_CAMERA)) {
        LOG_ERROR(Input, "Failed to initialize SDL camera subsystem: {}", SDL_GetError());
    }
#endif
    SDL_InitSubSystem(SDL_INIT_AUDIO);

    SDL_PropertiesID props = SDL_CreateProperties();
    SDL_SetStringProperty(props, SDL_PROP_WINDOW_CREATE_TITLE_STRING,
                          std::string(window_title).c_str());
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_X_NUMBER, SDL_WINDOWPOS_CENTERED);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_Y_NUMBER, SDL_WINDOWPOS_CENTERED);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER, width);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER, height);
    SDL_SetNumberProperty(props, "flags", SDL_WINDOW_VULKAN);
    SDL_SetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_RESIZABLE_BOOLEAN, true);
    // Creating the window directly in fullscreen avoids a visible windowed -> fullscreen
    // transition on startup. SDL sizes the window to the display and keeps the requested
    // width/height as the windowed size to restore when leaving fullscreen.
    SDL_SetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_FULLSCREEN_BOOLEAN,
                           EmulatorSettings.IsFullScreen());
    window = SDL_CreateWindowWithProperties(props);
    SDL_DestroyProperties(props);
    if (window == nullptr) {
        UNREACHABLE_MSG("Failed to create window handle: {}", SDL_GetError());
    }

    SDL_SetWindowMinimumSize(window, 640, 360);

    bool error = false;
    const SDL_DisplayID displayIndex = SDL_GetDisplayForWindow(window);
    if (displayIndex == 0) {
        LOG_ERROR(Frontend, "Error getting display index: {}", SDL_GetError());
        error = true;
    }
    const SDL_DisplayMode* displayMode;
    if ((displayMode = SDL_GetCurrentDisplayMode(displayIndex)) == 0) {
        LOG_ERROR(Frontend, "Error getting display mode: {}", SDL_GetError());
        error = true;
    }
    if (!error) {
        SDL_SetWindowFullscreenMode(
            window, EmulatorSettings.GetFullScreenMode() == "Fullscreen" ? displayMode : NULL);
    }
    SDL_SetWindowFullscreen(window, EmulatorSettings.IsFullScreen());
    SDL_SyncWindow(window);
    // The window geometry is only final once the fullscreen transition has settled; refresh
    // the cached size so the first swapchain and the splashscreen use the real drawable size.
    SDL_GetWindowSizeInPixels(window, &width, &height);

    SDL_InitSubSystem(SDL_INIT_GAMEPAD);

#if defined(SDL_PLATFORM_WIN32)
    window_info.type = WindowSystemType::Windows;
    window_info.render_surface = SDL_GetPointerProperty(SDL_GetWindowProperties(window),
                                                        SDL_PROP_WINDOW_WIN32_HWND_POINTER, NULL);
#elif defined(SDL_PLATFORM_LINUX) || defined(__FreeBSD__)
    // SDL doesn't have a platform define for FreeBSD AAAAAAAAAA
    if (SDL_strcmp(SDL_GetCurrentVideoDriver(), "x11") == 0) {
        window_info.type = WindowSystemType::X11;
        window_info.display_connection = SDL_GetPointerProperty(
            SDL_GetWindowProperties(window), SDL_PROP_WINDOW_X11_DISPLAY_POINTER, NULL);
        window_info.render_surface = (void*)SDL_GetNumberProperty(
            SDL_GetWindowProperties(window), SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0);
    } else if (SDL_strcmp(SDL_GetCurrentVideoDriver(), "wayland") == 0) {
        window_info.type = WindowSystemType::Wayland;
        window_info.display_connection = SDL_GetPointerProperty(
            SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER, NULL);
        window_info.render_surface = SDL_GetPointerProperty(
            SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, NULL);
    }
#elif defined(SDL_PLATFORM_MACOS)
    window_info.type = WindowSystemType::Metal;
    window_info.render_surface = SDL_Metal_GetLayer(SDL_Metal_CreateView(window));
#endif
    // input handler init-s
    Input::ControllerOutput::LinkJoystickAxes();
    Input::ParseInputConfig(std::string(Common::ElfInfo::Instance().GameSerial()));

    if (EmulatorSettings.IsBackgroundControllerInput()) {
        SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");
    }
}

WindowSDL::~WindowSDL() = default;

void WindowSDL::SetIcon(std::span<const u8> png_data) {
    if (png_data.empty()) {
        LOG_WARNING(Core, "No window icon data available, using default icon.");
        SetDefaultWindowIcon(window);
        return;
    }
    SetWindowIcon(window, std::vector<u8>(png_data.begin(), png_data.end()));
}

void WindowSDL::WaitEvent() {
    // Called on main thread
    SDL_Event event;

    if (!SDL_WaitEvent(&event)) {
        return;
    }

    // The emulator's settings menu (F3) needs a free cursor: any mouse mode (aim, touchpad, motion)
    // is switched off while it is open and comes back when it closes (user 2026-10-17: mouse aim
    // kept the cursor captured, the menu could not be left without Alt+F4). The DLSS 5 panel
    // (Page Up) too.
    {
        static bool saved = false;
        static Input::MouseMode before = Input::MouseMode::Off;
        const bool open = ImGuiEmuSettings::IsInGameSettingsDialogOpen() || Dlss5Panel::IsOpen();
        if (open && !saved) {
            before = Input::GetMouseMode();
            if (before != Input::MouseMode::Off) {
                Input::ToggleMouseModeTo(before); // the same mode again = off
            }
            SDL_SetWindowRelativeMouseMode(this->GetSDLWindow(), false);
            saved = true;
        } else if (!open && saved) {
            if (before != Input::MouseMode::Off) {
                Input::ToggleMouseModeTo(before);
            }
            SDL_SetWindowRelativeMouseMode(this->GetSDLWindow(),
                                           before == Input::MouseMode::Joystick ||
                                               before == Input::MouseMode::Gyro);
            saved = false;
        }
    }

    // SHADPS4_TEX_REPLACE: the button icons follow the device used last (tex_replace.h).
    if (VideoCore::TexReplace::AutoSwitch()) [[unlikely]] {
        switch (event.type) {
        case SDL_EVENT_KEY_DOWN:
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
            VideoCore::TexReplace::NoteInputDevice(false);
            break;
        case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
            VideoCore::TexReplace::NoteInputDevice(true);
            break;
        case SDL_EVENT_GAMEPAD_AXIS_MOTION:
            // Firm pushes only (Opus review 10-18): a drifting stick must not flip the icons back
            // on every key press.
            if (event.gaxis.value > 26000 || event.gaxis.value < -26000) {
                VideoCore::TexReplace::NoteInputDevice(true);
            }
            break;
        default:
            break;
        }
    }

    if (Libraries::Mouse::PushSDLEvent(event)) {
        return;
    }

    // SHADPS4_TOUCHPAD_BLOCK_FIX=1: while the mouse drives the pad, overlay windows (the FPS
    // counter in the top right) never see its presses / motion, so a touch can't start a window
    // drag that swallows the keyboard. Button-up still goes to ImGui (ends a drag begun earlier).
    const bool mouse_to_pad =
        Input::MouseEventsBelongToPad() &&
        (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN || event.type == SDL_EVENT_MOUSE_MOTION ||
         event.type == SDL_EVENT_MOUSE_WHEEL);
    // DLSS 5 (vk_dlss5.h): End = on / off, Page Up = its panel (2026-10-18; Page Up was the
    // jitter cycle, whose pick D is fixed now - Page Down still cycles). Handled before ImGui and
    // consumed: once the panel has the keyboard ImGui swallowed Page Up, so the key could not
    // close it (user 2026-10-18). Not with Ctrl: Ctrl+Shift+End is the quit hotkey.
    if (Vulkan::Dlss::Enabled() && event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat &&
        (event.key.key == SDLK_END || event.key.key == SDLK_PAGEUP) &&
        !(event.key.mod & SDL_KMOD_CTRL)) {
        if (event.key.key == SDLK_PAGEUP) {
            Dlss5Panel::Toggle();
        } else if (Vulkan::Dlss5::GetStatus().state != Vulkan::Dlss5::State::Ready) {
            const auto status = Vulkan::Dlss5::GetStatus();
            Overlay::ShowNotice(status.state == Vulkan::Dlss5::State::NotFound
                                    ? "DLSS 5 not found (dlss5 folder)"
                                    : "DLSS 5 not available (Page Up shows why)");
        } else {
            const bool on = Vulkan::Dlss5::ToggleEnabled();
            LOG_INFO(Render_Vulkan, "DLSS 5: {} (End)", on ? "on" : "off");
            Overlay::ShowNotice(on ? "DLSS 5: ON" : "DLSS 5: OFF (DLSS 4.5 only)");
        }
        return;
    }

    if (!mouse_to_pad && ImGui::Core::ProcessEvent(&event)) {
        return;
    }

    // SHADPS4_SYNTH_MOTION: the mouse motion source follows the mouse mode (controller 0, the
    // one Input::MousePolling drives).
    const auto SyncSyntheticMotion = [this] {
        if (!Input::g_synth_motion) {
            return;
        }
        const bool on = Input::IsMouseGyroMode();
        controllers[0]->SetSyntheticMotion(on);
        Overlay::ShowNotice(on ? "Mouse = motion sensor (hold Z tilt, tap Z rattle)"
                               : "Mouse motion sensor off");
    };

    // Touchpad / motion modes hand the mouse back to the camera (mouse aim) when they are toggled
    // off, if it had it before (user 2026-10-17: the finger scanner, the DUP core, the spray can).
    const auto ToggleSpecialMouseMode = [this](Input::MouseMode m) {
        static Input::MouseMode before = Input::MouseMode::Off;
        const Input::MouseMode prev = Input::GetMouseMode();
        if (Input::ToggleMouseModeTo(m)) {
            if (prev == Input::MouseMode::Joystick) {
                before = prev;
            }
        } else if (before == Input::MouseMode::Joystick) {
            Input::ToggleMouseModeTo(Input::MouseMode::Joystick);
            before = Input::MouseMode::Off;
        }
        const Input::MouseMode now = Input::GetMouseMode();
        SDL_SetWindowRelativeMouseMode(this->GetSDLWindow(), now == Input::MouseMode::Joystick ||
                                                                 now == Input::MouseMode::Gyro);
    };

    switch (event.type) {
    case SDL_EVENT_WINDOW_RESIZED:
    case SDL_EVENT_WINDOW_MAXIMIZED:
    case SDL_EVENT_WINDOW_RESTORED:
    case SDL_EVENT_WINDOW_ENTER_FULLSCREEN:
    case SDL_EVENT_WINDOW_LEAVE_FULLSCREEN:
        OnResize();
        break;
    case SDL_EVENT_WINDOW_MINIMIZED:
    case SDL_EVENT_WINDOW_EXPOSED:
        is_shown = event.type == SDL_EVENT_WINDOW_EXPOSED;
        OnResize();
        break;
    case SDL_EVENT_MOUSE_MOTION:
        if (Input::MouseAimActive()) {
            // SHADPS4_MOUSE_AIM: every pixel goes straight to the camera's pan / tilt.
            Core::MouseAim::OnMouseMotion(event.motion.xrel, event.motion.yrel);
        }
        if (Input::g_synth_motion && Input::IsMouseGyroMode()) {
            // SHADPS4_SYNTH_MOTION: every pixel counts, consumed by the 4 ms pushes.
            Input::g_synth_dx.fetch_add(event.motion.xrel, std::memory_order_relaxed);
            Input::g_synth_dy.fetch_add(event.motion.yrel, std::memory_order_relaxed);
        }
        break;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP:
        if (Input::g_synth_motion &&
            (event.button.button == SDL_BUTTON_X1 || event.button.button == SDL_BUTTON_X2)) {
            // Mouse back = rattle, forward = tilt (SHADPS4_SYNTH_MOTION).
            const bool down = event.type == SDL_EVENT_MOUSE_BUTTON_DOWN;
            (event.button.button == SDL_BUTTON_X1 ? Input::g_synth_rattle_btn
                                                  : Input::g_synth_tilt_btn)
                .store(down, std::memory_order_relaxed);
        }
        OnKeyboardMouseInput(&event);
        break;
    case SDL_EVENT_MOUSE_WHEEL:
    case SDL_EVENT_MOUSE_WHEEL_OFF:
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP:
        if (Input::g_synth_motion &&
            (event.type == SDL_EVENT_KEY_DOWN || event.type == SDL_EVENT_KEY_UP) &&
            !event.key.repeat &&
            event.key.key == SDLK_Z) {
            // SHADPS4_SYNTH_MOTION (user 2026-10-18): hold Z = tilt the pad sideways; a quick tap of Z
            // (released within 250 ms) = a 600 ms rattle burst. X only toggles the mouse motion mode
            // (global.ini); it used to rattle while held as well.
            static u64 z_down_ms = 0;
            const bool down = event.type == SDL_EVENT_KEY_DOWN;
            Input::g_synth_tilt.store(down, std::memory_order_relaxed);
            if (down) {
                z_down_ms = SDL_GetTicks();
            } else if (SDL_GetTicks() - z_down_ms < 250) {
                Input::g_synth_rattle_until.store(SDL_GetTicks() + 600, std::memory_order_relaxed);
            }
        }
        if (Vulkan::Dlss::Enabled() && event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat &&
            (event.key.key == SDLK_HOME || event.key.key == SDLK_PAGEDOWN ||
             event.key.key == SDLK_INSERT)) {
            // SHADPS4_DLSS (vk_dlss.h): Home = DLSS on / off, Page Down = jitter sign combination
            // (D is the pick), Insert = output at the window size / 1.5x (upscale mode).
            if (!Vulkan::Dlss::g_ready.load(std::memory_order_relaxed)) {
                Overlay::ShowNotice("DLSS not available (see the log)");
            } else if (event.key.key == SDLK_HOME) {
                const bool on = !Vulkan::Dlss::g_user_on.load(std::memory_order_relaxed);
                Vulkan::Dlss::g_user_on.store(on, std::memory_order_relaxed);
                LOG_INFO(Render_Vulkan, "DLSS: {} (Home)", on ? "on" : "off - the game's own AA");
                Overlay::ShowNotice(on ? "DLSS: ON" : "DLSS: OFF (game's own anti-aliasing)");
            } else if (event.key.key == SDLK_INSERT) {
                const float scale =
                    Vulkan::Dlss::g_output_scale.load(std::memory_order_relaxed) > 1.0f ? 1.0f
                                                                                       : 1.5f;
                Vulkan::Dlss::g_output_scale.store(scale, std::memory_order_relaxed);
                LOG_INFO(Render_Vulkan, "DLSS: output scale {} (Insert)", scale);
                Overlay::ShowNotice(scale > 1.0f ? "DLSS output: 1.5x the screen, scaled down"
                                                 : "DLSS output: the screen size");
            } else {
                const u32 step = event.key.key == SDLK_PAGEDOWN ? 3 : 1;
                const u32 combo =
                    (Vulkan::Dlss::g_jitter_combo.load(std::memory_order_relaxed) + step) & 3;
                Vulkan::Dlss::g_jitter_combo.store(combo, std::memory_order_relaxed);
                LOG_INFO(Render_Vulkan, "DLSS: jitter signs {} (Page Up / Down)", combo);
                static const char* const names[] = {"DLSS jitter: A (+x +y)", "DLSS jitter: B (-x +y)",
                                                    "DLSS jitter: C (+x -y)", "DLSS jitter: D (-x -y)"};
                Overlay::ShowNotice(names[combo]);
            }
        }
        if (Input::MouseAimActive() && event.type == SDL_EVENT_KEY_DOWN &&
            (event.key.key == SDLK_MINUS || event.key.key == SDLK_EQUALS)) {
            // SHADPS4_MOUSE_AIM: - / = = mouse sensitivity down / up (keys no shipped config binds;
            // the brackets are the default touchpad halves).
            const float s =
                Core::MouseAim::ScaleSensitivity(event.key.key == SDLK_MINUS ? 0.8f : 1.25f);
            Overlay::ShowNotice(fmt::format("Mouse aim: {:.1f} degrees per 100 px", s).c_str());
        }
        if (Input::g_stick_smoothing_toggle && event.type == SDL_EVENT_KEY_DOWN &&
            event.key.key == SDLK_F1 && !event.key.repeat) {
            // SHADPS4_STICK_SMOOTHING_TOGGLE: F1 flips upstream #5340 live (input/controller.h).
            const bool unsmoothed =
                !Input::g_real_axis_unsmoothed.load(std::memory_order_relaxed);
            Input::g_real_axis_unsmoothed.store(unsmoothed, std::memory_order_relaxed);
            LOG_INFO(Input, "StickSmoothing: real pad sticks / triggers {}",
                     unsmoothed ? "UNSMOOTHED (upstream #5340)" : "smoothed over 33 ms (old)");
            Overlay::ShowNotice(unsmoothed ? "Pad sticks: direct (fix #5340 ON)"
                                           : "Pad sticks: 33 ms smoothing (old)");
        }
        OnKeyboardMouseInput(&event);
        break;
    case SDL_EVENT_GAMEPAD_ADDED:
    case SDL_EVENT_GAMEPAD_REMOVED:
        controllers.TryOpenSDLControllers();
        break;
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
    case SDL_EVENT_GAMEPAD_BUTTON_UP:
    case SDL_EVENT_GAMEPAD_AXIS_MOTION:
    case SDL_EVENT_GAMEPAD_TOUCHPAD_DOWN:
    case SDL_EVENT_GAMEPAD_TOUCHPAD_UP:
    case SDL_EVENT_GAMEPAD_TOUCHPAD_MOTION:
    case SDL_EVENT_GAMEPAD_SENSOR_UPDATE:
        OnGamepadEvent(&event);
        break;
    case SDL_EVENT_QUIT:
        is_open = false;
        break;
    case SDL_EVENT_QUIT_DIALOG:
        Overlay::ToggleQuitWindow();
        break;
    case SDL_EVENT_TOGGLE_FULLSCREEN: {
        if (SDL_GetWindowFlags(window) & SDL_WINDOW_FULLSCREEN) {
            SDL_SetWindowFullscreen(window, 0);
        } else {
            SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN);
        }
        break;
    }
    case SDL_EVENT_TOGGLE_PAUSE:
        if (DebugState.IsGuestThreadsPaused()) {
            LOG_INFO(Frontend, "Game Resumed");
            DebugState.ResumeGuestThreads();
        } else {
            LOG_INFO(Frontend, "Game Paused");
            DebugState.PauseGuestThreads();
        }
        break;
    case SDL_EVENT_CHANGE_CONTROLLER:
        UNREACHABLE_MSG("todo");
        break;
    case SDL_EVENT_TOGGLE_SIMPLE_FPS:
        Overlay::ToggleSimpleFps();
        break;
    case SDL_EVENT_TOGGLE_FRIENDS:
        ImGui::Friends::Toggle();
        break;
    case SDL_EVENT_RELOAD_INPUTS:
        Input::ParseInputConfig(std::string(Common::ElfInfo::Instance().GameSerial()));
        break;
    case SDL_EVENT_MOUSE_TO_JOYSTICK:
        SDL_SetWindowRelativeMouseMode(this->GetSDLWindow(),
                                       Input::ToggleMouseModeTo(Input::MouseMode::Joystick));
        SyncSyntheticMotion();
        if (Core::MouseAim::Installed()) {
            Overlay::ShowNotice(Input::MouseAimActive()
                                    ? "Mouse aim ON (- / = sensitivity)"
                                    : "Mouse aim off");
        }
        break;
    case SDL_EVENT_MOUSE_TO_GYRO:
        ToggleSpecialMouseMode(Input::MouseMode::Gyro);
        SyncSyntheticMotion();
        break;
    case SDL_EVENT_MOUSE_TO_TOUCHPAD:
        ToggleSpecialMouseMode(Input::MouseMode::Touchpad);
        SyncSyntheticMotion();
        if (Core::MouseAim::Installed() && Input::MouseAimActive()) {
            Overlay::ShowNotice("Mouse aim ON (- / = sensitivity)");
        }
        break;
    case SDL_EVENT_ADD_VIRTUAL_USER:
        for (int i = 0; i < 4; i++) {
            if (controllers[i]->user_id == -1) {
                auto u = UserManagement.GetUserByPlayerIndex(i + 1);
                if (!u) {
                    break;
                }
                controllers[i]->user_id = u->user_id;
                controllers[i]->ConnectController(controllers[i]->m_sdl_gamepad);
                UserManagement.LoginUser(u, i + 1);
                break;
            }
        }
        break;
    case SDL_EVENT_REMOVE_VIRTUAL_USER:
        LOG_INFO(Input, "Remove user");
        for (int i = 3; i >= 0; i--) {
            if (controllers[i]->user_id != -1) {
                UserManagement.LogoutUser(UserManagement.GetUserByID(controllers[i]->user_id));
                controllers[i]->DisconnectController();
                controllers[i]->user_id = -1;
                break;
            }
        }
        break;
    case SDL_EVENT_RDOC_CAPTURE:
        if (VideoCore::IsRenderDocLoaded()) {
            VideoCore::TriggerCapture();
        } else {
            VideoCore::RequestScreenshot(VideoCore::ScreenshotRequest::GameOnly);
        }
        break;
    case SDL_EVENT_SCREENSHOT_WITH_OVERLAYS:
        VideoCore::RequestScreenshot(VideoCore::ScreenshotRequest::WithOverlays);
        break;
    default:
        break;
    }
}

void WindowSDL::InitTimers() {
    for (int i = 0; i < 4; ++i) {
        SDL_AddTimer(4, &PollController, controllers[i]);
    }
    SDL_AddTimer(33, Input::MousePolling, (void*)controllers[0]);
    Input::PadGestures::Attach(controllers[0]);
    SDL_AddTimer(4, &Input::PadGestures::Tick, (void*)controllers[0]);
    if (Core::MouseAim::Enabled()) {
        // SHADPS4_MOUSE_AIM: the mouse turns the camera from the start (the hotkey still toggles).
        Input::ToggleMouseModeTo(Input::MouseMode::Joystick);
        SDL_SetWindowRelativeMouseMode(window, true);
    }
}

void WindowSDL::RequestKeyboard() {
    if (keyboard_grab == 0) {
        SDL_RunOnMainThread(
            [](void* userdata) { SDL_StartTextInput(static_cast<SDL_Window*>(userdata)); }, window,
            true);
    }
    keyboard_grab++;
}

void WindowSDL::ReleaseKeyboard() {
    ASSERT(keyboard_grab > 0);
    keyboard_grab--;
    if (keyboard_grab == 0) {
        SDL_RunOnMainThread(
            [](void* userdata) { SDL_StopTextInput(static_cast<SDL_Window*>(userdata)); }, window,
            true);
    }
}

void WindowSDL::OnResize() {
    SDL_GetWindowSizeInPixels(window, &width, &height);
    ImGui::Core::OnResize();
}

Uint32 wheelOffCallback(void* og_event, Uint32 timer_id, Uint32 interval) {
    SDL_Event off_event = *(SDL_Event*)og_event;
    off_event.type = SDL_EVENT_MOUSE_WHEEL_OFF;
    SDL_PushEvent(&off_event);
    delete (SDL_Event*)og_event;
    return 0;
}

void WindowSDL::OnKeyboardMouseInput(const SDL_Event* event) {
    using Libraries::Pad::OrbisPadButtonDataOffset;

    // get the event's id, if it's keyup or keydown
    const bool input_down = event->type == SDL_EVENT_KEY_DOWN ||
                            event->type == SDL_EVENT_MOUSE_BUTTON_DOWN ||
                            event->type == SDL_EVENT_MOUSE_WHEEL;
    Input::InputEvent input_event = Input::InputBinding::GetInputEventFromSDLEvent(*event);

    // if it's a wheel event, make a timer that turns it off after a set time
    if (event->type == SDL_EVENT_MOUSE_WHEEL) {
        const SDL_Event* copy = new SDL_Event(*event);
        SDL_AddTimer(33, wheelOffCallback, (void*)copy);
    }

    // add/remove it from the list
    bool inputs_changed = Input::UpdatePressedKeys(input_event);

    // update bindings
    if (inputs_changed) {
        Input::ActivateOutputsFromInputs();
    }
}

void WindowSDL::OnGamepadEvent(const SDL_Event* event) {
    // 2026-10-21: a pad without a motion sensor or a touch surface (input/pad_gestures) plays
    // L3+R3 and View here, before the bindings see them.
    SDL_Event filtered = *event;
    if (const u8 pad = controllers.GetGamepadIndexFromJoystickId(filtered.gbutton.which);
        pad < 5 && Input::PadGestures::FilterEvent(filtered, controllers[pad])) {
        return;
    }
    event = &filtered;

    bool input_down = event->type == SDL_EVENT_GAMEPAD_AXIS_MOTION ||
                      event->type == SDL_EVENT_GAMEPAD_BUTTON_DOWN;
    Input::InputEvent input_event = Input::InputBinding::GetInputEventFromSDLEvent(*event);

    // the touchpad button shouldn't be rebound to anything else,
    // as it would break the entire touchpad handling
    // You can still bind other things to it though
    if (event->gbutton.button == SDL_GAMEPAD_BUTTON_TOUCHPAD) {
        controllers[controllers.GetGamepadIndexFromJoystickId(event->gbutton.which)]->Button(
            OrbisPadButtonDataOffset::TouchPad, input_down);
        return;
    }

    u8 gamepad;

    switch (event->type) {
    case SDL_EVENT_GAMEPAD_SENSOR_UPDATE:
        switch ((SDL_SensorType)event->gsensor.sensor) {
        case SDL_SENSOR_GYRO:
            gamepad = controllers.GetGamepadIndexFromJoystickId(event->gsensor.which);
            if (gamepad < 5) {
                controllers[gamepad]->UpdateGyro(event->gsensor.data);
            }
            break;
        case SDL_SENSOR_ACCEL:
            gamepad = controllers.GetGamepadIndexFromJoystickId(event->gsensor.which);
            if (gamepad < 5) {
                controllers[gamepad]->UpdateAcceleration(event->gsensor.data);
            }
            break;
        default:
            break;
        }
        return;
    case SDL_EVENT_GAMEPAD_TOUCHPAD_DOWN:
    case SDL_EVENT_GAMEPAD_TOUCHPAD_UP:
    case SDL_EVENT_GAMEPAD_TOUCHPAD_MOTION:
        controllers[controllers.GetGamepadIndexFromJoystickId(event->gtouchpad.which)]
            ->SetTouchpadState(event->gtouchpad.finger,
                               event->type != SDL_EVENT_GAMEPAD_TOUCHPAD_UP, event->gtouchpad.x,
                               event->gtouchpad.y);
        return;
    default:
        break;
    }

    // add/remove it from the list
    bool inputs_changed = Input::UpdatePressedKeys(input_event);

    if (inputs_changed) {
        // update bindings
        Input::ActivateOutputsFromInputs();
    }
}

#ifndef __APPLE__
void SetWindowIcon(SDL_Window* window, const std::vector<u8>& png) {
    int imageWidth = 0;
    int imageHeight = 0;
    constexpr int numChannels = 4;
    unsigned char* imageData = stbi_load_from_memory(png.data(), png.size(), &imageWidth,
                                                     &imageHeight, nullptr, numChannels);
    if (imageData == nullptr) {
        LOG_ERROR(Core, "Failed to load window icon image: {}", stbi_failure_reason());
        return;
    }
    SCOPE_EXIT {
        stbi_image_free(imageData);
    };

    SDL_Surface* surface = SDL_CreateSurfaceFrom(imageWidth, imageHeight, SDL_PIXELFORMAT_RGBA32,
                                                 imageData, imageWidth * numChannels);
    if (surface == nullptr) {
        LOG_ERROR(Core, "Failed to create SDL surface for window icon: {}", SDL_GetError());
    }
    if (!SDL_SetWindowIcon(window, surface)) {
        LOG_ERROR(Core, "Failed to set SDL window icon: {}", SDL_GetError());
    }
    SDL_DestroySurface(surface);
}
#endif

void SetDefaultWindowIcon(SDL_Window* window) {
    const auto resource = cmrc::res::get_filesystem();
    const auto file = resource.open("src/resources/shadps4.png");
    const std::vector<u8> texData = std::vector<u8>(file.begin(), file.end());
    SetWindowIcon(window, texData);
}

} // namespace Frontend
