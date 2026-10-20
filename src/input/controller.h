// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <utility>

#include <SDL3/SDL_gamepad.h>
#include "SDL3/SDL_joystick.h"
#include "common/assert.h"
#include "common/ring_buffer_queue.h"
#include "core/libraries/pad/pad.h"
#include "core/libraries/system/userservice.h"

namespace Input {

/// SHADPS4_STICK_SMOOTHING_TOGGLE=1 (2026-10-14, upstream #5340 behind a live toggle): F1 flips
/// g_real_axis_unsmoothed. While it is set, stick / trigger values that come from a real pad's
/// analog axis reach the game unsmoothed (upstream #5340: the 33 ms ramp meant for keyboard
/// keys bound to a stick was applied to real sticks and triggers too); keyboard-driven axis
/// values keep the ramp. SHADPS4_STICK_UNSMOOTHED=1 starts with it set. Off = today's behaviour.
inline const bool g_stick_smoothing_toggle = [] {
    const char* v = std::getenv("SHADPS4_STICK_SMOOTHING_TOGGLE");
    return v && v[0] == '1';
}();
inline std::atomic<bool> g_real_axis_unsmoothed{[] {
    const char* v = std::getenv("SHADPS4_STICK_UNSMOOTHED");
    return v && v[0] == '1';
}()};

/// SHADPS4_SYNTH_MOTION=1 (2026-10-14, notes/study-input-kbm-2026-10-14.md sec. 3): while the
/// mouse is the motion sensor (F6), controller 0's orientation is SET from the accumulated mouse
/// (Second Son aims the spray can from the orientation quaternion: +-0.35 rad yaw / +-0.25 rad
/// pitch = the whole cursor), the accelerometer reads the gravity that matches the pose (so the
/// game's re-levelling never fights it), the gyro its derivative; R (or mouse back button)
/// = rattle (pad rolled 90 degrees + a 5 Hz 2.5 g vertical shake: the game's SprayShake test is
/// any axis > 2.5 g while rolled > 30 degrees), Z (or mouse forward button) = tilt (roll only).
/// A real pad's sensor events are ignored meanwhile. Mouse deltas come from SDL motion events
/// on the main thread (g_synth_dx / dy), consumed by every 4 ms state push.
inline const bool g_synth_motion = [] {
    const char* v = std::getenv("SHADPS4_SYNTH_MOTION");
    return v && v[0] == '1';
}();
/// Radians of pad yaw / pitch per mouse pixel (SHADPS4_SYNTH_MOTION_DEG_PER_100PX, default 2.5 (user 2026-10-14: 5 was too fast):
/// 800 px from the centre to the cursor's side edge, 573 px to the top).
inline const float g_synth_rad_per_px = [] {
    const char* v = std::getenv("SHADPS4_SYNTH_MOTION_DEG_PER_100PX");
    const float deg = v ? static_cast<float>(std::atof(v)) : 2.5f;
    return (deg > 0.0f ? deg : 2.5f) * 3.14159265f / 180.0f / 100.0f;
}();
inline std::atomic<float> g_synth_dx{0.0f};
inline std::atomic<float> g_synth_dy{0.0f};
inline std::atomic<bool> g_synth_tilt{false};       ///< Z held
inline std::atomic<bool> g_synth_rattle{false};     ///< R held
/// 2026-10-18 (user): a quick tap of Z rattles until this SDL_GetTicks() time (ms).
inline std::atomic<u64> g_synth_rattle_until{0};
inline std::atomic<bool> g_synth_tilt_btn{false};   ///< mouse forward button held
inline std::atomic<bool> g_synth_rattle_btn{false}; ///< mouse back button held
/// 2026-10-21: a pad without a gyro reports its motion from its own sticks (input/pad_gestures):
/// the aim rates are radians accumulated since the last 4 ms push (Second Son's graffiti can),
/// the rattle is the face button held, and SetPadSyntheticMotion rolls the pose 90 degrees.
inline std::atomic<float> g_synth_pad_yaw{0.0f};
inline std::atomic<float> g_synth_pad_pitch{0.0f};
inline std::atomic<bool> g_synth_pad_rattle{false};

enum class Axis {
    LeftX = 0,
    LeftY = 1,
    RightX = 2,
    RightY = 3,
    TriggerLeft = 4,
    TriggerRight = 5,

    AxisMax
};

struct TouchpadEntry {
    u8 ID = 0;
    bool state{};
    u16 x{};
    u16 y{};
};

struct Colour {
    u8 r, g, b;
};
static constexpr Input::Colour g_user_colours[4]{
    {0, 0, 255},   // blue
    {255, 0, 0},   // red
    {0, 255, 0},   // green
    {255, 0, 255}, // pink
};

struct State {
private:
    template <typename T>
    using AxisArray = std::array<T, std::to_underlying(Axis::AxisMax)>;
    static constexpr AxisArray<s32> axis_defaults{128, 128, 128, 128, 0, 0};
    static constexpr u64 axis_smoothing_time{33000};
    AxisArray<bool> axis_smoothing_flags{true};
    AxisArray<u64> axis_smoothing_start_times{0};
    AxisArray<int> axis_smoothing_start_values{axis_defaults};
    AxisArray<int> axis_smoothing_end_values{axis_defaults};

public:
    void OnButton(Libraries::Pad::OrbisPadButtonDataOffset, bool);
    void OnAxis(Axis, int, u64 timestamp, bool smooth = true);
    void OnTouchpad(int touchIndex, bool isDown, float x, float y);
    void OnGyro(const float[3]);
    void OnAccel(const float[3]);
    void UpdateAxisSmoothing(u64 timestamp);

    Libraries::Pad::OrbisPadButtonDataOffset buttonsState{};
    u64 time = 0;
    AxisArray<s32> axes{axis_defaults};
    TouchpadEntry touchpad[2]{};
    Libraries::Pad::OrbisFVector3 acceleration = {0.0f, -9.81f, 0.0f};
    Libraries::Pad::OrbisFVector3 angularVelocity = {0.0f, 0.0f, 0.0f};
    Libraries::Pad::OrbisFQuaternion orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    u64 touch_time_since_held_down{};
    bool connected{};
    u8 connected_count{};
};

inline int GetAxis(int min, int max, int value) {
    int v = (255 * (value - min)) / (max - min);
    return (v < 0 ? 0 : (v > 255 ? 255 : v));
}

class GameController {
    friend class GameControllers;

public:
    GameController();
    virtual ~GameController() = default;
    void ConnectController(SDL_Gamepad* pad);
    void DisconnectController();

    State ReadState();
    int ReadStates(State* states, int states_num);

    void Button(Libraries::Pad::OrbisPadButtonDataOffset button, bool isPressed);
    void Axis(Input::Axis axis, int value, bool smooth = true);
    void UpdateGyro(const float gyro[3]);
    void UpdateAcceleration(const float acceleration[3]);
    void PollState();
    void ResetOrientation();
    /// SHADPS4_SYNTH_MOTION: turns the mouse motion source on / off (F6 Gyro mode).
    void SetSyntheticMotion(bool active);
    /// 2026-10-21: the same pose from a pad's sticks (input/pad_gestures, L3+R3 on a pad without
    /// a gyro). The two sources are independent: the pose is synthetic while either is on.
    void SetPadSyntheticMotion(bool active);
    void SetLightBarRGB(u8 const r, u8 const g, u8 const b);
    void SetLightBarRGB(Colour const c);
    Colour GetLightBarRGB();
    void PollLightColour();
    bool SetVibration(u8 smallMotor, u8 largeMotor);
    void SetTouchpadState(int touchIndex, bool touchDown, float x, float y);

    float gyro_buf[3] = {0.0f, 0.0f, 0.0f}, accel_buf[3] = {0.0f, 9.81f, 0.0f};
    s32 user_id = Libraries::UserService::ORBIS_USER_SERVICE_USER_ID_INVALID;
    SDL_Gamepad* m_sdl_gamepad = nullptr;

private:
    /// SHADPS4_SYNTH_MOTION state (m_state_mutex).
    struct SyntheticMotion {
        bool mouse{}; ///< F6 / SHADPS4_SYNTH_MOTION: the mouse is the sensor
        bool pad{};   ///< the pad's sensor mode: its sticks aim, always rolled on its side
        float yaw{}, pitch{}, roll{}, phi{};
        u64 t0{}, last_t{};
        Libraries::Pad::OrbisFQuaternion last_q{0.0f, 0.0f, 0.0f, 1.0f};
    };
    SyntheticMotion m_synth{};
    bool SyntheticActiveLocked() const {
        return m_synth.mouse || m_synth.pad;
    }
    void SetSyntheticSourceLocked(bool pad, bool active);
    void SyntheticMotionLocked(u64 timestamp);

    // m_state_mutex must be held by the caller.
    void PushStateLocked(u64 timestamp = 0);
    void UpdateOrientationLocked(u64 timestamp);

    u8 m_next_touch_id{1};
    u64 m_touch_down_timestamp{};
    u64 m_last_orientation_update{};
    Colour colour;
    std::optional<Colour> override_colour{};

    State m_state;

    std::mutex m_state_mutex;
    RingBufferQueue<State> m_states_queue;
};

class GameControllers {
    std::array<GameController*, 5> controllers;

public:
    GameControllers()
        : controllers({new GameController(), new GameController(), new GameController(),
                       new GameController(), new GameController()}) {};
    virtual ~GameControllers() = default;
    GameController* operator[](const size_t& i) const {
        if (i > 4) {
            UNREACHABLE_MSG("Index {} is out of bounds for GameControllers!", i);
        }
        return controllers[i];
    }
    void TryOpenSDLControllers();
    u8 GetGamepadIndexFromJoystickId(SDL_JoystickID id);
    static std::optional<u8> GetControllerIndexFromUserID(s32 user_id);
    static std::optional<u8> GetControllerIndexFromControllerID(s32 controller_id);

    void SetControllerCustomColor(s32 i, u8 r, u8 g, u8 b) {
        // reset to ensure the next function always runs, even if there already was a preexisting
        // override colour before
        controllers[i]->override_colour = std::nullopt;
        controllers[i]->SetLightBarRGB(r, g, b);
        controllers[i]->override_colour = {r, g, b};
    }
    void ResetLightbarColors();
};

} // namespace Input
