// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// The cheats panel (F6; common/cheats.h): lists the GoldHEN cheat files in user/cheats with a
// checkbox or a button per cheat. Cheats that do not match the running game are shown greyed out
// with the reason.
namespace CheatsPanel {

/// F6: opens the panel, or closes it when open (window thread).
void Toggle();
/// The window thread frees the mouse while it is open, as for the F3 menu.
bool IsOpen();

} // namespace CheatsPanel
