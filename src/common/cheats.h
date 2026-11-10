// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "common/types.h"

// Cheats (optional; GitHub #40, #58): GoldHEN cheat files (.json, the format of
// github.com/GoldHEN/GoldHEN_Cheat_Repository) dropped into user/cheats are listed in the cheats
// panel (F6, imgui/cheats_panel.h) and switched on and off while the game runs. Nothing happens
// without a file there.
//
// Every memory entry of a cheat is checked against the loaded game before it can be used: the
// bytes at its offset must be the file's "off" (or "on") bytes. A file written for another
// version or region therefore shows its cheats as "does not match" instead of corrupting the
// game, and a file for another region with the same code simply works. Offsets are tried as
// relative to the eboot and as absolute PS4 addresses (minus 0x400000): both are in use.
namespace Cheats {

enum class ModState : u8 {
    Ready,    ///< every entry found: can be switched
    Mismatch, ///< the game's bytes are not the file's (another version)
    Outside,  ///< an entry or its jump target is outside the game's loaded code / data
};

struct Entry {
    u64 offset = 0; ///< resolved, relative to the eboot
    std::vector<u8> on;
    std::vector<u8> off;
};

struct Mod {
    std::string name;
    bool is_button = false; ///< "button": written once; "checkbox": on / off
    ModState state = ModState::Mismatch;
    bool on = false;
    std::vector<Entry> entries;
};

struct File {
    std::string file_name;
    std::string title;
    std::string id;
    std::string version;
    std::string credits;
    std::vector<Mod> mods;
};

/// user/cheats.
std::filesystem::path Folder();

/// After the eboot is loaded (before it runs): reads the folder and switches the remembered
/// checkboxes on.
void OnGameLoaded();

/// Reads the folder again (the panel's Reload button). Cheats that are on stay on.
void Reload();

/// A copy for the panel.
std::vector<File> Files();

/// Switches a checkbox, or presses a button (`on` = true). False when it cannot be applied.
bool Set(size_t file, size_t mod, bool on);

} // namespace Cheats
