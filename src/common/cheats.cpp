// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <fstream>
#include <mutex>
#include <set>
#include <utility>
#include <nlohmann/json.hpp>

#include "common/cheats.h"
#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "common/path_util.h"

namespace Cheats {

namespace {

constexpr u64 Ps4ImageBase = 0x400000; ///< some files give absolute addresses

std::mutex g_mutex;
std::vector<File> g_files;
bool g_game_loaded = false;

using Remembered = std::set<std::pair<std::string, std::string>>; ///< file name, cheat name

std::filesystem::path RememberedPath() {
    return Folder() / "enabled.txt";
}

Remembered ReadRemembered() {
    Remembered out;
    std::ifstream in{RememberedPath()};
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const auto tab = line.find('\t');
        if (tab != std::string::npos) {
            out.emplace(line.substr(0, tab), line.substr(tab + 1));
        }
    }
    return out;
}

void WriteRemembered() {
    std::ofstream out{RememberedPath(), std::ios::trunc};
    for (const File& file : g_files) {
        for (const Mod& mod : file.mods) {
            if (mod.on && !mod.is_button) {
                out << file.file_name << '\t' << mod.name << '\n';
            }
        }
    }
}

bool ParseHex(const std::string& text, std::vector<u8>& out) {
    const auto digit = [](char c) -> int {
        if (c >= '0' && c <= '9') {
            return c - '0';
        }
        if (c >= 'a' && c <= 'f') {
            return c - 'a' + 10;
        }
        if (c >= 'A' && c <= 'F') {
            return c - 'A' + 10;
        }
        return -1;
    };
    out.clear();
    if (text.empty() || text.size() % 2 != 0) {
        return false;
    }
    for (size_t i = 0; i < text.size(); i += 2) {
        const int hi = digit(text[i]);
        const int lo = digit(text[i + 1]);
        if (hi < 0 || lo < 0) {
            return false;
        }
        out.push_back(u8(hi << 4 | lo));
    }
    return true;
}

const u8* At(u64 offset) {
    return reinterpret_cast<const u8*>(MemoryPatcher::g_eboot_address + offset);
}

/// 0 = the game holds `off` there, 1 = `on`, -1 = neither (or not loaded).
int Holds(const Entry& e, u64 offset) {
    if (!MemoryPatcher::EbootRangeLoaded(offset, e.on.size())) {
        return -1;
    }
    if (std::memcmp(At(offset), e.off.data(), e.off.size()) == 0) {
        return 0;
    }
    if (std::memcmp(At(offset), e.on.data(), e.on.size()) == 0) {
        return 1;
    }
    return -1;
}

/// An entry that cannot be told from the game's bytes: a value write (on == off).
bool Blind(const Entry& e) {
    return e.on == e.off;
}

/// Finds where every entry of the file lives in the loaded game and what state it is in.
void Resolve(File& file) {
    // Pass 1: the entries the game's bytes identify. They also tell which kind of offset the
    // file uses, for the entries that cannot be identified (value writes).
    struct Found {
        bool known = false;
        bool absolute = false;
        int holds = -1;
    };
    std::vector<std::vector<Found>> found(file.mods.size());
    int relative_votes = 0, absolute_votes = 0;
    for (size_t m = 0; m < file.mods.size(); ++m) {
        Mod& mod = file.mods[m];
        found[m].resize(mod.entries.size());
        for (size_t i = 0; i < mod.entries.size(); ++i) {
            const Entry& e = mod.entries[i];
            if (Blind(e)) {
                continue;
            }
            Found& f = found[m][i];
            if (const int h = Holds(e, e.offset); h >= 0) {
                f = {true, false, h};
                ++relative_votes;
            } else if (e.offset >= Ps4ImageBase) {
                if (const int h2 = Holds(e, e.offset - Ps4ImageBase); h2 >= 0) {
                    f = {true, true, h2};
                    ++absolute_votes;
                }
            }
        }
    }
    const bool file_absolute = absolute_votes > relative_votes;

    for (size_t m = 0; m < file.mods.size(); ++m) {
        Mod& mod = file.mods[m];
        mod.state = ModState::Ready;
        bool all_on = !mod.entries.empty();
        bool mod_absolute = file_absolute;
        for (const Found& f : found[m]) {
            if (f.known) {
                mod_absolute = f.absolute;
                break;
            }
        }
        for (size_t i = 0; i < mod.entries.size() && mod.state == ModState::Ready; ++i) {
            Entry& e = mod.entries[i];
            const Found& f = found[m][i];
            if (f.known) {
                if (f.absolute) {
                    e.offset -= Ps4ImageBase;
                }
                all_on &= f.holds == 1;
            } else if (Blind(e)) {
                if (mod_absolute && e.offset >= Ps4ImageBase) {
                    e.offset -= Ps4ImageBase;
                }
                if (!MemoryPatcher::EbootRangeLoaded(e.offset, e.on.size())) {
                    mod.state = ModState::Outside;
                }
                all_on = false;
            } else {
                // Not the file's bytes at either place: outside the image (a code cave at an
                // address only a console has) or another version's code.
                const bool loaded =
                    MemoryPatcher::EbootRangeLoaded(e.offset, e.on.size()) ||
                    (e.offset >= Ps4ImageBase &&
                     MemoryPatcher::EbootRangeLoaded(e.offset - Ps4ImageBase, e.on.size()));
                mod.state = loaded ? ModState::Mismatch : ModState::Outside;
            }
        }
        // A jump the cheat writes into the game's code must land in the game's code as well:
        // files made on a console often jump to a cave far outside the image.
        for (size_t i = 0; i < mod.entries.size() && mod.state == ModState::Ready; ++i) {
            const Entry& e = mod.entries[i];
            if (e.on.size() >= 5 && e.on[0] == 0xe9 &&
                MemoryPatcher::EbootRangeIsCode(e.offset, e.on.size())) {
                s32 rel;
                std::memcpy(&rel, e.on.data() + 1, 4);
                const s64 target = s64(e.offset) + 5 + rel;
                if (target < 0 || !MemoryPatcher::EbootRangeIsCode(u64(target), 1)) {
                    mod.state = ModState::Outside;
                }
            }
        }
        mod.on = mod.state == ModState::Ready && !mod.is_button && all_on;
    }
}

bool ReadFile(const std::filesystem::path& path, File& file) {
    std::ifstream in{path};
    nlohmann::json json = nlohmann::json::parse(in, nullptr, false, true);
    if (json.is_discarded() || !json.is_object() || !json.contains("mods") ||
        !json["mods"].is_array()) {
        return false;
    }
    const auto text = [&json](const char* key) {
        return json.contains(key) && json[key].is_string() ? json[key].get<std::string>()
                                                             : std::string{};
    };
    file.file_name = path.filename().string();
    file.title = text("name");
    file.id = text("id");
    file.version = text("version");
    if (json.contains("credits") && json["credits"].is_array()) {
        for (const auto& c : json["credits"]) {
            if (c.is_string()) {
                file.credits += (file.credits.empty() ? "" : ", ") + c.get<std::string>();
            }
        }
    }
    for (const auto& jm : json["mods"]) {
        if (!jm.is_object() || !jm.contains("memory") || !jm["memory"].is_array()) {
            continue;
        }
        Mod mod;
        mod.name = jm.value("name", std::string{"(no name)"});
        mod.is_button = jm.contains("type") && jm["type"].is_string() &&
                        jm["type"].get<std::string>() == "button";
        bool ok = !jm["memory"].empty();
        for (const auto& je : jm["memory"]) {
            Entry e;
            const std::string offset = je.is_object() ? je.value("offset", std::string{}) : "";
            char* end = nullptr;
            e.offset = std::strtoull(offset.c_str(), &end, 16);
            ok &= !offset.empty() && end && *end == '\0' && je.contains("on") &&
                  je.contains("off") && je["on"].is_string() && je["off"].is_string() &&
                  ParseHex(je["on"].get<std::string>(), e.on) &&
                  ParseHex(je["off"].get<std::string>(), e.off) && e.on.size() == e.off.size();
            if (!ok) {
                break;
            }
            mod.entries.push_back(std::move(e));
        }
        if (ok) {
            file.mods.push_back(std::move(mod));
        } else {
            LOG_WARNING(Loader, "Cheats: {}: \"{}\" is not understood, left out",
                        file.file_name, mod.name);
        }
    }
    return true;
}

void Write(const Mod& mod, bool on) {
    for (const Entry& e : mod.entries) {
        const auto& bytes = on ? e.on : e.off;
        std::memcpy(reinterpret_cast<void*>(MemoryPatcher::g_eboot_address + e.offset),
                    bytes.data(), bytes.size());
    }
}

/// Reads the folder; cheats in `keep_on` (and resolvable) are switched on.
void LoadLocked(const Remembered& keep_on) {
    g_files.clear();
    std::error_code ec;
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(Folder(), ec)) {
        if (entry.is_regular_file(ec) && entry.path().extension() == ".json") {
            paths.push_back(entry.path());
        }
    }
    std::ranges::sort(paths);
    for (const auto& path : paths) {
        File file;
        if (!ReadFile(path, file)) {
            LOG_WARNING(Loader, "Cheats: {} is not a GoldHEN cheat file (.json)",
                        path.filename().string());
            continue;
        }
        Resolve(file);
        u32 ready = 0;
        for (Mod& mod : file.mods) {
            ready += mod.state == ModState::Ready;
            if (mod.state == ModState::Ready && !mod.is_button && !mod.on &&
                keep_on.contains({file.file_name, mod.name})) {
                Write(mod, true);
                mod.on = true;
                LOG_INFO(Loader, "Cheats: \"{}\" on ({})", mod.name, file.file_name);
            }
        }
        LOG_INFO(Loader, "Cheats: {} ({} {}): {} of {} cheats fit this game", file.file_name,
                 file.id, file.version, ready, file.mods.size());
        g_files.push_back(std::move(file));
    }
}

} // namespace

std::filesystem::path Folder() {
    return Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "cheats";
}

void OnGameLoaded() {
    std::scoped_lock lk{g_mutex};
    g_game_loaded = true;
    std::error_code ec;
    if (!std::filesystem::exists(Folder(), ec)) {
        return; // the feature is not in use
    }
    LoadLocked(ReadRemembered());
}

void Reload() {
    std::scoped_lock lk{g_mutex};
    if (!g_game_loaded) {
        return;
    }
    Remembered keep_on = ReadRemembered();
    for (const File& file : g_files) {
        for (const Mod& mod : file.mods) {
            if (mod.on) {
                keep_on.emplace(file.file_name, mod.name);
            }
        }
    }
    LoadLocked(keep_on);
}

std::vector<File> Files() {
    std::scoped_lock lk{g_mutex};
    return g_files;
}

bool Set(size_t file, size_t mod_index, bool on) {
    std::scoped_lock lk{g_mutex};
    if (file >= g_files.size() || mod_index >= g_files[file].mods.size()) {
        return false;
    }
    Mod& mod = g_files[file].mods[mod_index];
    if (mod.state != ModState::Ready) {
        return false;
    }
    if (mod.is_button) {
        Write(mod, true);
    } else {
        Write(mod, on);
        mod.on = on;
        WriteRemembered();
    }
    LOG_INFO(Loader, "Cheats: \"{}\" {} ({})", mod.name,
             mod.is_button ? "applied" : on ? "on" : "off", g_files[file].file_name);
    return true;
}

} // namespace Cheats
