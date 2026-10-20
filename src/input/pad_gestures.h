// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <SDL3/SDL_events.h>

namespace Input {

class GameController;

/// Second Son asks for a motion sensor and a touch surface. A pad without them (an Xbox pad)
/// gets both from the buttons it has, as the iss_pc port does (port/src/runtime_pad.c):
///  - L3 + R3 taps the pad's sensor mode: the pad counts as held on its side, the left (or the
///    right) stick aims the graffiti can, the face button shakes it and R2 still sprays. The
///    sticks and the face button are hidden from the game meanwhile. Off on the next tap.
///  - View (SDL's Back, the pad's TouchPad button) is a touchpad click, played like a finger:
///    the touch lands 50 ms before the button and lingers 100 ms after the release. View plus
///    the right stick swipes a finger across the pad (open door, hand scanner); View plus R1
///    puts two fingers on both ends (the car off Reggie).
/// The pad's own capabilities decide: a DualShock keeps its sensors and its touch surface.
/// SHADPS4_PAD_GESTURES=0 turns all of it off.
namespace PadGestures {

/// The controller the pad's own gestures belong to (the first one, as the mouse modes do). Called
/// once from WindowSDL::InitTimers, before the events and the timer run.
void Attach(GameController* controller);

/// A gamepad event on its way to the bindings. Returns true when the event was used up here; an
/// axis a gesture holds (the sticks of the sensor mode, a swipe's right stick) is set to its
/// resting value, so the game reads a centred stick instead of one stuck at its last value.
bool FilterEvent(SDL_Event& event, GameController* controller);

/// 4 ms timer (WindowSDL::InitTimers): plays the clicks and swipes and aims the sensor mode.
Uint32 Tick(void* param, Uint32 id, Uint32 interval);

} // namespace PadGestures
} // namespace Input
