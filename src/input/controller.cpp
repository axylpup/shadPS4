// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <unordered_set>
#include <utility>

#include <SDL3/SDL.h>

#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/time.h"
#include "core/libraries/system/userservice.h"
#include "core/user_settings.h"
#include "input/controller.h"
#include "input/input_mouse.h"

namespace Input {

using Libraries::Pad::OrbisPadButtonDataOffset;

namespace {

void CalculateOrientation(const Libraries::Pad::OrbisFVector3& angular_velocity, float delta_time,
                          const Libraries::Pad::OrbisFQuaternion& last_orientation,
                          Libraries::Pad::OrbisFQuaternion& orientation) {
    if (delta_time > 1.0f) {
        orientation = last_orientation;
        return;
    }
    Libraries::Pad::OrbisFQuaternion q = last_orientation;
    const Libraries::Pad::OrbisFQuaternion omega = {angular_velocity.x, angular_velocity.y,
                                                    angular_velocity.z, 0.0f};

    const Libraries::Pad::OrbisFQuaternion q_omega = {
        q.w * omega.x + q.x * omega.w + q.y * omega.z - q.z * omega.y,
        q.w * omega.y + q.y * omega.w + q.z * omega.x - q.x * omega.z,
        q.w * omega.z + q.z * omega.w + q.x * omega.y - q.y * omega.x,
        q.w * omega.w - q.x * omega.x - q.y * omega.y - q.z * omega.z};

    const Libraries::Pad::OrbisFQuaternion q_dot = {0.5f * q_omega.x, 0.5f * q_omega.y,
                                                    0.5f * q_omega.z, 0.5f * q_omega.w};

    q.x += q_dot.x * delta_time;
    q.y += q_dot.y * delta_time;
    q.z += q_dot.z * delta_time;
    q.w += q_dot.w * delta_time;

    const float norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    q.x /= norm;
    q.y /= norm;
    q.z /= norm;
    q.w /= norm;
    orientation = q;
}

} // namespace

GameController::GameController() : m_states_queue(64) {}

State GameController::ReadState() {
    std::lock_guard lock{m_state_mutex};
    return m_state;
}

int GameController::ReadStates(State* states, int states_num) {
    std::lock_guard lock{m_state_mutex};
    if (states_num <= 0) {
        return 0;
    }

    if (!m_state.connected) {
        states[0] = m_state;
        return 1;
    }

    if (states_num == 1) {
        // Retained history can make a later multi-sample read return up to 64 stale reports, so
        // mixed single- and multi-sample reads require dedicated tests.
        states[0] = m_state;
        return 1;
    }

    int read_count = 0;
    while (read_count < states_num) {
        auto state = m_states_queue.Pop();
        if (!state) {
            break;
        }
        states[read_count++] = std::move(*state);
    }
    return read_count;
}

void GameController::Button(OrbisPadButtonDataOffset button, bool is_pressed) {
    std::lock_guard lock{m_state_mutex};
    m_state.OnButton(button, is_pressed);
    PushStateLocked();
}

void GameController::Axis(Input::Axis axis, int value, bool smooth) {
    std::lock_guard lock{m_state_mutex};
    const u64 timestamp = Libraries::Kernel::sceKernelGetProcessTime();
    m_state.OnAxis(axis, value, timestamp, smooth);
    PushStateLocked(timestamp);
}

void GameController::UpdateGyro(const float gyro[3]) {
    std::lock_guard lock{m_state_mutex};
    if (SyntheticActiveLocked()) {
        return; // synthetic motion: the pose is generated; a real pad's events are ignored
    }
    std::memcpy(gyro_buf, gyro, sizeof(gyro_buf));
}

void GameController::UpdateAcceleration(const float acceleration[3]) {
    std::lock_guard lock{m_state_mutex};
    if (SyntheticActiveLocked()) {
        return;
    }
    std::memcpy(accel_buf, acceleration, sizeof(accel_buf));
}

void GameController::SetSyntheticMotion(bool active) {
    std::lock_guard lock{m_state_mutex};
    SetSyntheticSourceLocked(false, active);
}

void GameController::SetPadSyntheticMotion(bool active) {
    std::lock_guard lock{m_state_mutex};
    SetSyntheticSourceLocked(true, active);
}

void GameController::SetSyntheticSourceLocked(bool pad, bool active) {
    bool& flag = pad ? m_synth.pad : m_synth.mouse;
    if (flag == active) {
        return;
    }
    flag = active;
    const bool keep_pad = m_synth.pad;
    const bool keep_mouse = m_synth.mouse;
    const bool any = keep_pad || keep_mouse;
    // Toggling a source recentres the pose: aiming starts from the middle of the screen again.
    m_synth = {};
    m_synth.pad = keep_pad;
    m_synth.mouse = keep_mouse;
    g_synth_dx.store(0.0f, std::memory_order_relaxed);
    g_synth_dy.store(0.0f, std::memory_order_relaxed);
    g_synth_pad_yaw.store(0.0f, std::memory_order_relaxed);
    g_synth_pad_pitch.store(0.0f, std::memory_order_relaxed);
    g_synth_pad_rattle.store(false, std::memory_order_relaxed);
    if (!any) {
        m_state.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
        std::fill(gyro_buf, gyro_buf + 3, 0.0f);
        std::fill(accel_buf, accel_buf + 3, 0.0f);
        accel_buf[1] = 9.81f;
    }
    m_last_orientation_update = 0;
    LOG_INFO(Input, "SynthMotion: {} {}", pad ? "pad sensor mode" : "mouse motion sensor",
             active ? "ON" : "off");
    PushStateLocked();
}

namespace {
using Quat = Libraries::Pad::OrbisFQuaternion;
Quat QMul(const Quat& a, const Quat& b) { // Hamilton product, (x, y, z, w)
    return {a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
            a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x,
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}
Quat QAxis(float ax, float ay, float az, float angle) {
    const float s = std::sin(angle * 0.5f);
    return {ax * s, ay * s, az * s, std::cos(angle * 0.5f)};
}
} // namespace

void GameController::SyntheticMotionLocked(u64 timestamp) {
    constexpr float pi = 3.14159265f;
    const float dx = g_synth_dx.exchange(0.0f, std::memory_order_relaxed);
    const float dy = g_synth_dy.exchange(0.0f, std::memory_order_relaxed);
    const float k = g_synth_rad_per_px;
    auto& s = m_synth;
    // Mouse right -> cursor right (yaw down), mouse up -> cursor up (pitch up); the study's
    // pose table (scratch/study-input-20261014/padmath.py).
    if (IsMouseGyroRollMode()) {
        s.roll -= k * dx;
    } else {
        s.yaw -= k * dx;
    }
    s.pitch = std::clamp(s.pitch - k * dy, -80.0f * pi / 180.0f, 80.0f * pi / 180.0f);
    if (s.pad) {
        // 2026-10-21 (input/pad_gestures): a pad without a gyro aims the stick as a rate, the
        // sticks' radians arriving from the 4 ms tick. The game's own limit: +-80 degrees of yaw
        // and +-69 of pitch keep the graffiti cursor on the screen (iss_pc's aim_pitch clamp).
        s.yaw = std::clamp(s.yaw + g_synth_pad_yaw.exchange(0.0f, std::memory_order_relaxed),
                           -1.5f, 1.5f);
        s.pitch = std::clamp(s.pitch + g_synth_pad_pitch.exchange(0.0f, std::memory_order_relaxed),
                             -1.2f, 1.2f);
    }
    if (s.yaw > pi) {
        s.yaw -= 2.0f * pi;
    } else if (s.yaw < -pi) {
        s.yaw += 2.0f * pi;
    }
    if (s.t0 == 0) {
        s.t0 = timestamp;
    }
    const float dt = s.last_t != 0 && timestamp > s.last_t
                         ? std::min(static_cast<float>(timestamp - s.last_t) / 1e6f, 0.1f)
                         : 0.0f;
    const bool rattle = g_synth_rattle.load(std::memory_order_relaxed) ||
                        g_synth_rattle_btn.load(std::memory_order_relaxed) ||
                        g_synth_pad_rattle.load(std::memory_order_relaxed) ||
                        SDL_GetTicks() < g_synth_rattle_until.load(std::memory_order_relaxed);
    // The pad's sensor mode holds the pad on its side, as the graffiti scene asks ("Hold Wireless
    // Controller Sideways"): the roll is part of the mode, not a button.
    const bool tilt = rattle || s.pad || g_synth_tilt.load(std::memory_order_relaxed) ||
                      g_synth_tilt_btn.load(std::memory_order_relaxed);
    // Roll toward 90 degrees while tilted (eased over ~80 ms: a turn, not a jump).
    const float target = tilt ? pi * 0.5f : 0.0f;
    s.phi += (target - s.phi) * (1.0f - std::exp(-dt / 0.08f));
    const Quat q = QMul(QMul(QAxis(0, 1, 0, s.yaw), QAxis(1, 0, 0, s.pitch)),
                        QAxis(0, 0, 1, s.phi + s.roll));
    // Gravity reaction in the pad frame = world up projected on the pad axes (rows of q, the
    // game's own formula): (r0.y, r1.y, r2.y).
    const float gx = 2 * q.x * q.y + 2 * q.w * q.z;
    const float gy = 1 - 2 * q.x * q.x - 2 * q.z * q.z;
    const float gz = 2 * q.y * q.z - 2 * q.w * q.x;
    float shake = 0.0f;
    if (rattle) {
        const float t = static_cast<float>(timestamp - s.t0) / 1e6f;
        shake = 2.5f * std::sin(2.0f * pi * 5.0f * t);
    }
    const float g = 9.81f * (1.0f + shake);
    m_state.acceleration = {gx * g, gy * g, gz * g};
    // Body rate from the pose change (small-angle form), so "still" reads as still.
    Libraries::Pad::OrbisFVector3 omega{0.0f, 0.0f, 0.0f};
    if (dt > 0.0f) {
        const Quat conj{-s.last_q.x, -s.last_q.y, -s.last_q.z, s.last_q.w};
        Quat dq = QMul(conj, q);
        if (dq.w < 0) {
            dq = {-dq.x, -dq.y, -dq.z, -dq.w};
        }
        omega = {2.0f * dq.x / dt, 2.0f * dq.y / dt, 2.0f * dq.z / dt};
    }
    m_state.angularVelocity = omega;
    m_state.orientation = q;
    s.last_q = q;
    s.last_t = timestamp;
    m_last_orientation_update = timestamp;
}

void GameController::PollState() {
    std::lock_guard lock{m_state_mutex};
    PushStateLocked();
}

void GameController::ResetOrientation() {
    std::lock_guard lock{m_state_mutex};
    m_state.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    m_last_orientation_update = 0;
    PushStateLocked();
}

void GameController::SetTouchpadState(int touch_index, bool touch_down, float x, float y) {
    if (touch_index < 0 || touch_index >= 2) {
        return;
    }

    std::lock_guard lock{m_state_mutex};
    const u64 timestamp = Libraries::Kernel::sceKernelGetProcessTime();
    const bool was_pressed = m_state.touchpad[0].state || m_state.touchpad[1].state;
    auto& touch = m_state.touchpad[touch_index];
    if (touch_down && !touch.state) {
        touch.ID = m_next_touch_id;
        m_next_touch_id = m_next_touch_id == 127 ? 1 : m_next_touch_id + 1;
    }
    m_state.OnTouchpad(touch_index, touch_down, x, y);
    const bool is_pressed = m_state.touchpad[0].state || m_state.touchpad[1].state;
    if (!was_pressed && is_pressed) {
        m_touch_down_timestamp = timestamp;
    } else if (was_pressed && !is_pressed) {
        m_touch_down_timestamp = 0;
    }
    PushStateLocked(timestamp);
}

void GameController::ConnectController(SDL_Gamepad* pad) {
    std::lock_guard lock{m_state_mutex};
    m_sdl_gamepad = pad;
    m_states_queue.Clear();
    if (!m_state.connected) {
        ++m_state.connected_count;
        if (m_state.connected_count == 0) {
            m_state.connected_count = 1;
        }
    }
    m_state.connected = true;
    m_last_orientation_update = 0;
    PushStateLocked();
}

void GameController::DisconnectController() {
    std::lock_guard lock{m_state_mutex};
    m_states_queue.Clear();
    m_sdl_gamepad = nullptr;

    const u8 connected_count = m_state.connected_count;
    m_state = {};
    m_state.connected_count = connected_count;
    std::fill(gyro_buf, gyro_buf + 3, 0.0f);
    std::fill(accel_buf, accel_buf + 3, 0.0f);
    accel_buf[1] = 9.81f;
    m_next_touch_id = 1;
    m_touch_down_timestamp = 0;
    m_state.connected = false;
    m_last_orientation_update = 0;
    PushStateLocked();
}

void GameController::UpdateOrientationLocked(u64 timestamp) {
    if (m_last_orientation_update == 0 || timestamp <= m_last_orientation_update) {
        m_last_orientation_update = timestamp;
        return;
    }
    const float delta_time =
        static_cast<float>(timestamp - m_last_orientation_update) / 1'000'000.f;
    Libraries::Pad::OrbisFQuaternion orientation{};
    CalculateOrientation(m_state.angularVelocity, delta_time, m_state.orientation, orientation);
    m_state.orientation = orientation;
    m_last_orientation_update = timestamp;
}

void GameController::PushStateLocked(u64 timestamp) {
    if (timestamp == 0) {
        timestamp = Libraries::Kernel::sceKernelGetProcessTime();
    }
    m_state.UpdateAxisSmoothing(timestamp);
    if (SyntheticActiveLocked()) [[unlikely]] {
        SyntheticMotionLocked(timestamp);
    } else {
        m_state.OnGyro(gyro_buf);
        m_state.OnAccel(accel_buf);
        UpdateOrientationLocked(timestamp);
    }
    m_state.time = timestamp;
    m_state.touch_time_since_held_down =
        m_touch_down_timestamp == 0 ? 0 : timestamp - m_touch_down_timestamp;
    m_states_queue.Push(m_state);
}

void GameController::SetLightBarRGB(u8 const r, u8 const g, u8 const b) {
    if (override_colour.has_value()) {
        return;
    }
    colour = {r, g, b};
    if (m_sdl_gamepad != nullptr) {
        SDL_SetGamepadLED(m_sdl_gamepad, r, g, b);
    }
}

void GameController::SetLightBarRGB(Colour const c) {
    SetLightBarRGB(c.r, c.g, c.b);
}

Colour GameController::GetLightBarRGB() {
    return colour;
}

void GameController::PollLightColour() {
    if (m_sdl_gamepad != nullptr) {
        SDL_SetGamepadLED(m_sdl_gamepad, colour.r, colour.g, colour.b);
    }
}

void GameControllers::ResetLightbarColors() {
    for (auto& c : controllers) {
        auto const* u = UserManagement.GetUserByID(c->user_id);
        if (!u || !c->m_sdl_gamepad) {
            continue;
        }
        auto const i = u->user_color - 1;
        if (i < 0 || i > 3) {
            continue;
        }
        auto const& col = g_user_colours[i];
        c->override_colour = std::nullopt;
        c->SetLightBarRGB(col);
    }
}

bool GameController::SetVibration(u8 smallMotor, u8 largeMotor) {
    if (m_sdl_gamepad != nullptr) {
        return SDL_RumbleGamepad(m_sdl_gamepad, (smallMotor / 255.0f) * 0xFFFF,
                                 (largeMotor / 255.0f) * 0xFFFF, -1);
    }
    return true;
}

static bool is_first_check = true;

void GameControllers::TryOpenSDLControllers() {
    using namespace Libraries::UserService;
    int controller_count;
    SDL_JoystickID* new_joysticks = SDL_GetGamepads(&controller_count);
    LOG_INFO(Input, "{} controllers are currently connected", controller_count);

    std::unordered_set<SDL_JoystickID> assigned_ids;
    std::array<bool, 4> slot_taken{false, false, false, false};

    for (int i = 0; i < 4; i++) {
        SDL_Gamepad* pad = controllers[i]->m_sdl_gamepad;
        if (pad) {
            SDL_JoystickID id = SDL_GetGamepadID(pad);
            bool still_connected = false;
            for (int j = 0; j < controller_count; j++) {
                if (new_joysticks[j] == id) {
                    still_connected = true;
                    assigned_ids.insert(id);
                    slot_taken[i] = true;
                    break;
                }
            }
            if (!still_connected) {
                SDL_CloseGamepad(pad);
                controllers[i]->DisconnectController();
                controllers[i]->user_id = -1;
                slot_taken[i] = false;
            }
        }
    }

    for (int j = 0; j < controller_count; j++) {
        SDL_JoystickID id = new_joysticks[j];
        if (assigned_ids.contains(id))
            continue;

        SDL_Gamepad* pad = SDL_OpenGamepad(id);
        if (!pad) {
            continue;
        }

        for (int i = 0; i < 4; i++) {
            if (!slot_taken[i]) {
                auto u = UserManagement.GetUserByPlayerIndex(i + 1);
                if (!u) {
                    LOG_INFO(Input, "User {} not found", i + 1);
                    continue; // for now, if you don't specify who Player N is in the config,
                              // Player N won't be registered at all
                }
                auto* c = controllers[i];
                LOG_INFO(Input, "Gamepad registered for slot {}! Handle: {}", i,
                         SDL_GetGamepadID(pad));
                slot_taken[i] = true;
                c->user_id = u->user_id;
                UserManagement.LoginUser(u, i + 1);
                c->ConnectController(pad);
                if (EmulatorSettings.IsMotionControlsEnabled()) {
                    if (SDL_SetGamepadSensorEnabled(c->m_sdl_gamepad, SDL_SENSOR_GYRO, true)) {
                        const float poll_rate =
                            SDL_GetGamepadSensorDataRate(c->m_sdl_gamepad, SDL_SENSOR_GYRO);
                        LOG_INFO(Input, "Gyro initialized, poll rate: {}", poll_rate);
                    } else {
                        LOG_ERROR(Input, "Failed to initialize gyro controls for gamepad {}",
                                  c->user_id);
                    }
                    if (SDL_SetGamepadSensorEnabled(c->m_sdl_gamepad, SDL_SENSOR_ACCEL, true)) {
                        const float poll_rate =
                            SDL_GetGamepadSensorDataRate(c->m_sdl_gamepad, SDL_SENSOR_ACCEL);
                        LOG_INFO(Input, "Accel initialized, poll rate: {}", poll_rate);
                    } else {
                        LOG_ERROR(Input, "Failed to initialize accel controls for gamepad {}",
                                  c->user_id);
                    }
                }
                break;
            }
        }
    }
    if (is_first_check) [[unlikely]] {
        is_first_check = false;
        if (controller_count == 0) {
            auto u = UserManagement.GetUserByPlayerIndex(1);
            controllers[0]->user_id = u->user_id;
            controllers[0]->ConnectController(nullptr);
            UserManagement.LoginUser(u, 1);
        }
    }
    // SHADPS4_PAD_KEYBOARD_FALLBACK=1 (2026-10-18, user report): when the last controller is
    // unplugged, player 1's pad goes back to the keyboard, as at a boot without controllers.
    // Without it the pad stays disconnected and the game's "reconnect the controller, press X"
    // screen cannot be answered from the keyboard (only a restart or the controller helped).
    static const bool keyboard_fallback = [] {
        const char* v = std::getenv("SHADPS4_PAD_KEYBOARD_FALLBACK");
        return v && v[0] == '1';
    }();
    if (keyboard_fallback && controller_count == 0 && controllers[0]->m_sdl_gamepad == nullptr &&
        controllers[0]->user_id == -1) {
        if (auto u = UserManagement.GetUserByPlayerIndex(1)) {
            controllers[0]->user_id = u->user_id;
            controllers[0]->ConnectController(nullptr);
            UserManagement.LoginUser(u, 1);
            LOG_INFO(Input, "Last controller removed: player 1's pad is the keyboard again");
        }
    }
    SDL_free(new_joysticks);
}
u8 GameControllers::GetGamepadIndexFromJoystickId(SDL_JoystickID id) {
    auto g = SDL_GetGamepadFromID(id);
    ASSERT(g != nullptr);
    for (int i = 0; i < 5; i++) {
        if (controllers[i]->m_sdl_gamepad == g) {
            return i;
        }
    }
    // LOG_TRACE(Input, "Gamepad index: {}", index);
    return -1;
}

std::optional<u8> GameControllers::GetControllerIndexFromUserID(s32 user_id) {
    auto const u = UserManagement.GetUserByID(user_id);
    if (!u) {
        return std::nullopt;
    }
    return u->player_index - 1;
}

std::optional<u8> GameControllers::GetControllerIndexFromControllerID(s32 controller_id) {
    if (controller_id < 1 || controller_id > 5) {
        return std::nullopt;
    }
    return controller_id - 1;
}

} // namespace Input
