// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "input/pad_gestures.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <mutex>

#include <SDL3/SDL_gamepad.h>
#include <SDL3/SDL_timer.h>

#include "common/logging/log.h"
#include "core/devtools/layer.h"
#include "input/controller.h"
#include "input/input_mouse.h"

namespace Input::PadGestures {
namespace {

using Libraries::Pad::OrbisPadButtonDataOffset;

/// A click without a touch surface is played like a finger, with iss_pc's timings: God of War
/// III opened photo mode on the release of a left click while the finger was still down, and
/// ignored a touch that came and went with the button. inFAMOUS wants the same kind of touch.
constexpr u64 kLeadUs = 50000;
constexpr u64 kHoldUs = 50000;
constexpr u64 kTailUs = 100000;

/// The touch surface's own coordinates, as fractions of it (1920 x 942 in the pad's report).
constexpr float kClickX = 0.25f; ///< the left half, where iss_pc's View click lands
constexpr float kCenterY = 0.5f;
constexpr float kTwoFingerLeftX = 0.0625f;
constexpr float kTwoFingerRightX = 0.9379f;

/// Past this push of the right stick (0.3) View becomes a swipe instead of a click.
constexpr float kSwipeStartSq = 0.09f;
/// Radians of yaw / pitch a second at a full push of an aiming stick (iss_pc's 1.2).
constexpr float kAimSpeed = 1.2f;

bool DisabledByEnv() {
    static const bool disabled = [] {
        const char* v = std::getenv("SHADPS4_PAD_GESTURES");
        return v && v[0] == '0';
    }();
    return disabled;
}

u64 NowUs() {
    return static_cast<u64>(SDL_GetTicksNS() / 1000);
}

/// The window thread feeds events, the SDL timer thread ticks: one lock for both.
std::mutex g_mutex;

/// The pad as it is, from its own events (the bindings only see what a gesture leaves them).
struct RawPad {
    bool present{}; ///< the capabilities below are for a pad that is connected
    bool has_touchpad{};
    bool has_gyro{};
    bool back{}, r1{}, face{}, l3{}, r3{};
    float left_x{}, left_y{}, right_x{}, right_y{};
    u64 checked{};
};

RawPad g_raw;

bool g_sensor_mode{};
bool g_chord{}; ///< L3 and R3 went down together: both stay hidden until released

struct Click {
    u64 pressed{};
    u64 released{};
};

/// pressed == 0 means no click (or none lingering): the touch may still be out for a moment.
Click g_click;

struct TwoFingers {
    bool active{};
    u64 pressed{};
    u64 released{};
};
TwoFingers g_two;

bool g_swipe{};
u64 g_last_tick{};
bool g_last_tick_set{};
/// The controller the 4 ms timer drives (the first one, as the mouse modes are): the gestures are
/// only produced for the pad the game reads.
GameController* g_target{};

/// What the game was last told about the touch surface, so only changes are pushed.
struct Reported {
    bool touch[2]{};
    float x[2]{};
    float y[2]{};
    bool button{};
};
Reported g_reported;

/// A pad's capabilities, re-read twice a second: plugging a DualSense in mid-game switches the
/// gestures off on their own.
void RefreshPads(GameController* controller, u64 now) {
    if (g_raw.present && now - g_raw.checked < 250000) {
        return;
    }
    g_raw.checked = now;
    SDL_Gamepad* pad = controller->m_sdl_gamepad;
    g_raw.present = pad != nullptr;
    if (pad == nullptr) {
        g_raw.has_touchpad = false;
        g_raw.has_gyro = false;
        return;
    }
    g_raw.has_touchpad = SDL_GetNumGamepadTouchpads(pad) > 0;
    g_raw.has_gyro = SDL_GamepadHasSensor(pad, SDL_SENSOR_GYRO);
}

/// A stick's push (-1..1) as an aiming speed (-1..1): nothing inside the dead zone, then the
/// square of the rest, so a small push traces a stencil and a full one moves briskly.
float AimCurve(float v) {
    constexpr float dead = 0.15f;
    const float a = std::abs(v);
    if (a <= dead) {
        return 0.0f;
    }
    const float t = std::min((a - dead) / (1.0f - dead), 1.0f);
    return v < 0.0f ? -t * t : t * t;
}

void StartClick(u64 now) {
    g_click.pressed = now;
    g_click.released = 0;
    g_two = {};
    g_swipe = false;
}

void StartTwoFingers(u64 now) {
    g_two.active = true;
    g_two.pressed = now;
    g_two.released = 0;
    g_click = {}; // two fingers, not a click
    g_swipe = false;
}

void ToggleSensorMode(GameController* controller) {
    g_sensor_mode = !g_sensor_mode;
    controller->SetPadSyntheticMotion(g_sensor_mode);
    // The sticks and the face button are the can's while the mode is on: a press that was still
    // down when the mode changed must not stay stuck in the game.
    controller->Button(OrbisPadButtonDataOffset::Cross, false);
    controller->Button(OrbisPadButtonDataOffset::L3, false);
    controller->Button(OrbisPadButtonDataOffset::R3, false);
    LOG_INFO(Input, "PadGestures: pad sensor mode {}", g_sensor_mode ? "on" : "off");
    Overlay::ShowNotice(g_sensor_mode
                            ? "Pad sensor mode ON: stick aims, A shakes, R2 sprays (L3+R3 off)"
                            : "Pad sensor mode off");
}

void Push(GameController* controller, const Reported& want) {
    if (Input::GetMouseMode() == Input::MouseMode::Touchpad) {
        return; // the mouse owns the touchpad (Delete): don't fight it
    }
    for (int i = 0; i < 2; ++i) {
        const bool moved = want.touch[i] &&
                           (want.x[i] != g_reported.x[i] || want.y[i] != g_reported.y[i]);
        if (want.touch[i] == g_reported.touch[i] && !moved) {
            continue;
        }
        controller->SetTouchpadState(i, want.touch[i], want.x[i], want.y[i]);
        g_reported.touch[i] = want.touch[i];
        g_reported.x[i] = want.x[i];
        g_reported.y[i] = want.y[i];
    }
    if (want.button != g_reported.button) {
        controller->Button(OrbisPadButtonDataOffset::TouchPad, want.button);
        g_reported.button = want.button;
    }
}

void ReleaseTouch(GameController* controller) {
    Push(controller, Reported{});
}

/// Nothing is held any more: the pad left, or the game lost the window.
void ReleaseAll(GameController* controller) {
    g_click = {};
    g_two = {};
    g_swipe = false;
    g_chord = false;
    g_raw = {};
    if (g_sensor_mode) {
        g_sensor_mode = false;
        controller->SetPadSyntheticMotion(false);
        g_synth_pad_rattle.store(false, std::memory_order_relaxed);
    }
    ReleaseTouch(controller);
}

} // namespace

void Attach(GameController* controller) {
    std::lock_guard lock{g_mutex};
    g_target = controller;
}

bool FilterEvent(SDL_Event& event, GameController* controller) {
    if (DisabledByEnv() || controller == nullptr || controller != g_target) {
        return false;
    }
    std::lock_guard lock{g_mutex};
    const u64 now = NowUs();
    RefreshPads(controller, now);
    if (!g_raw.present) {
        return false;
    }

    switch (event.type) {
    case SDL_EVENT_GAMEPAD_AXIS_MOTION: {
        const float v = static_cast<float>(event.gaxis.value) / 32767.0f;
        bool hidden = g_sensor_mode && !g_raw.has_gyro;
        switch (event.gaxis.axis) {
        case SDL_GAMEPAD_AXIS_LEFTX:
            g_raw.left_x = v;
            break;
        case SDL_GAMEPAD_AXIS_LEFTY:
            g_raw.left_y = v;
            break;
        case SDL_GAMEPAD_AXIS_RIGHTX:
            g_raw.right_x = v;
            hidden = hidden || g_raw.back;
            break;
        case SDL_GAMEPAD_AXIS_RIGHTY:
            g_raw.right_y = v;
            hidden = hidden || g_raw.back;
            break;
        default:
            return false;
        }
        if (hidden && !g_raw.has_touchpad) {
            event.gaxis.value = 0; // the gesture's stick, not the camera's
        }
        return false;
    }
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
    case SDL_EVENT_GAMEPAD_BUTTON_UP: {
        const bool down = event.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN;
        switch (event.gbutton.button) {
        case SDL_GAMEPAD_BUTTON_BACK:
            g_raw.back = down;
            if (g_raw.has_touchpad) {
                return false; // the pad's own surface: the click is the surface being pressed
            }
            if (down) {
                if (g_raw.r1) {
                    StartTwoFingers(now);
                } else {
                    StartClick(now);
                }
            } else {
                if (g_two.active) {
                    g_two.released = now;
                }
                if (g_click.pressed != 0) {
                    g_click.released = now;
                }
            }
            return true; // played here, with a finger
        case SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER:
            g_raw.r1 = down;
            if (!g_raw.back || g_raw.has_touchpad) {
                return false;
            }
            if (down) {
                StartTwoFingers(now);
            } else {
                g_two.released = now;
            }
            return true; // the second finger of the car-lifting gesture
        case SDL_GAMEPAD_BUTTON_SOUTH:
            g_raw.face = down;
            return g_sensor_mode; // shakes the can
        case SDL_GAMEPAD_BUTTON_LEFT_STICK:
        case SDL_GAMEPAD_BUTTON_RIGHT_STICK: {
            const bool was = g_raw.l3 && g_raw.r3;
            if (event.gbutton.button == SDL_GAMEPAD_BUTTON_LEFT_STICK) {
                g_raw.l3 = down;
            } else {
                g_raw.r3 = down;
            }
            const bool both = g_raw.l3 && g_raw.r3;
            if (g_raw.has_gyro) {
                return false; // a pad with its own sensor keeps L3 and R3
            }
            if (both && !was) {
                g_chord = true;
                ToggleSensorMode(controller);
            } else if (!both && was) {
                g_chord = false;
            }
            return g_chord;
        }
        default:
            return false;
        }
    }
    default:
        return false;
    }
}

Uint32 Tick(void* param, Uint32 id, Uint32 interval) {
    auto* controller = static_cast<GameController*>(param);
    if (DisabledByEnv() || controller == nullptr) {
        return interval;
    }
    std::lock_guard lock{g_mutex};
    const u64 now = NowUs();
    float dt = g_last_tick_set ? static_cast<float>(now - g_last_tick) / 1e6f : 0.0f;
    g_last_tick = now;
    g_last_tick_set = true;
    dt = std::clamp(dt, 0.0f, 0.1f);

    RefreshPads(controller, now);
    if (!g_raw.present) {
        ReleaseAll(controller);
        return interval;
    }

    if (g_sensor_mode) {
        // Either stick points the can (the one pushed further sets each axis), as far as ~80
        // degrees each way; the aim is a rate, so the cursor stays where it is left.
        float x = g_raw.left_x;
        float y = g_raw.left_y;
        if (std::abs(g_raw.right_x) > std::abs(x)) {
            x = g_raw.right_x;
        }
        if (std::abs(g_raw.right_y) > std::abs(y)) {
            y = g_raw.right_y;
        }
        g_synth_pad_yaw.fetch_add(-AimCurve(x) * kAimSpeed * dt, std::memory_order_relaxed);
        g_synth_pad_pitch.fetch_add(-AimCurve(y) * kAimSpeed * dt, std::memory_order_relaxed);
    }
    g_synth_pad_rattle.store(g_sensor_mode && g_raw.face, std::memory_order_relaxed);

    if (g_raw.has_touchpad) {
        ReleaseTouch(controller); // the pad's own surface is the game's
        return interval;
    }

    Reported want;
    if (g_two.active) {
        if (g_two.released != 0 && now - g_two.released >= kTailUs) {
            g_two = {};
        } else {
            // Two fingers on both ends of the pad; the button follows the lead time, the fingers
            // stay out for the tail.
            want.touch[0] = true;
            want.x[0] = kTwoFingerLeftX;
            want.y[0] = kCenterY;
            want.touch[1] = true;
            want.x[1] = kTwoFingerRightX;
            want.y[1] = kCenterY;
            want.button = g_two.released == 0 && now >= g_two.pressed + kLeadUs;
        }
    } else {
        // A swipe takes over from a click that was starting: the finger lands on the pad and
        // moves with the stick, from the middle toward the side pushed.
        if (g_raw.back &&
            g_raw.right_x * g_raw.right_x + g_raw.right_y * g_raw.right_y > kSwipeStartSq) {
            g_swipe = true;
            g_click = {};
        } else if (!g_raw.back) {
            g_swipe = false;
        }
        if (g_swipe) {
            want.touch[0] = true;
            want.x[0] = std::clamp(0.5f + g_raw.right_x * 0.47f, 0.03f, 0.97f);
            want.y[0] = std::clamp(0.5f + g_raw.right_y * 0.47f, 0.03f, 0.97f);
        } else if (g_click.pressed != 0) {
            const u64 begin = g_click.pressed + kLeadUs;
            const u64 end = std::max(begin + kHoldUs, g_click.released);
            if (g_click.released != 0 && now >= end + kTailUs) {
                g_click = {};
            } else {
                want.touch[0] = true;
                want.x[0] = kClickX;
                want.y[0] = kCenterY;
                want.button = now >= begin && (g_click.released == 0 || now < end);
            }
        }
    }
    Push(controller, want);
    return interval;
}

} // namespace Input::PadGestures
