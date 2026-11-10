// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <tuple>
#include <utility>
#include <nlohmann/json.hpp>

#include "common/cheats.h"
#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "common/path_util.h"

namespace Cheats {

namespace {

constexpr u64 Ps4ImageBase = 0x400000; ///< some files give absolute addresses
/// Room for moved code caves: the tail of the trampoline area behind the eboot's image
/// (read-write-execute, within +-2 GB of the code), just below MemoryPatcher::g_eboot_scratch.
constexpr u64 CaveAreaSize = 16 * 1024;

std::mutex g_mutex;
std::vector<File> g_files;
bool g_game_loaded = false;
/// A cave keeps its place for the whole run (file name, cheat name, entry index), so a cheat
/// that is on survives Reload.
std::map<std::tuple<std::string, std::string, size_t>, u64> g_caves;
u64 g_cave_used = 0;

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

u8* At(u64 offset) {
    return reinterpret_cast<u8*>(MemoryPatcher::g_eboot_address + offset);
}

/// The cave area, as an offset from the eboot (0 = none on this build).
u64 CaveAreaBegin() {
    if (MemoryPatcher::g_eboot_scratch == 0) {
        return 0;
    }
    return MemoryPatcher::g_eboot_scratch - CaveAreaSize - MemoryPatcher::g_eboot_address;
}

bool InCaveArea(u64 offset, u64 size) {
    const u64 begin = CaveAreaBegin();
    return begin != 0 && offset >= begin && offset + size <= begin + CaveAreaSize;
}

/// Readable and writable for a cheat: the game's image or the cave area.
bool Usable(u64 offset, u64 size) {
    return MemoryPatcher::EbootRangeLoaded(offset, size) || InCaveArea(offset, size);
}

/// 0 = the game holds `off` there, 1 = `on`, -1 = neither (or not loaded).
int Holds(const Entry& e, u64 offset) {
    if (!Usable(offset, e.on.size())) {
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

/// A relative branch at `i` of `code`: its length (5 or 6), or 0.
size_t BranchAt(const std::vector<u8>& code, size_t i) {
    if ((code[i] == 0xe9 || code[i] == 0xe8) && i + 5 <= code.size()) {
        return 5;
    }
    if (code[i] == 0x0f && i + 6 <= code.size() && (code[i + 1] & 0xf0) == 0x80) {
        return 6;
    }
    return 0;
}

struct OldCave {
    u64 old_offset;
    u64 new_offset;
    u64 size;
};

/// Moves the entries of `mod` that lie outside the game into the cave area and rewrites the
/// branches for it. False when a cave cannot be placed.
bool MoveCaves(const File& file, Mod& mod, const std::vector<bool>& outside) {
    std::vector<OldCave> caves;
    for (size_t i = 0; i < mod.entries.size(); ++i) {
        if (!outside[i]) {
            continue;
        }
        Entry& e = mod.entries[i];
        const u64 area = CaveAreaBegin();
        if (area == 0) {
            return false;
        }
        const auto key = std::make_tuple(file.file_name, mod.name, i);
        auto it = g_caves.find(key);
        if (it == g_caves.end()) {
            const u64 size = (e.on.size() + 15) & ~u64(15);
            if (g_cave_used + size > CaveAreaSize) {
                return false;
            }
            it = g_caves.emplace(key, area + g_cave_used).first;
            g_cave_used += size;
        }
        caves.push_back({e.offset, it->second, e.on.size()});
        e.offset = it->second;
        e.cave = true;
        // A console's cave holds whatever was there; here it is ours and starts empty.
        e.off.assign(e.on.size(), 0);
    }
    const auto moved_target = [&caves](u64 old_target, u64& new_target) {
        for (const OldCave& c : caves) {
            if (old_target >= c.old_offset && old_target < c.old_offset + c.size) {
                new_target = c.new_offset + (old_target - c.old_offset);
                return true;
            }
        }
        return false;
    };
    size_t cave_index = 0;
    for (Entry& e : mod.entries) {
        const u64 old_base = e.cave ? caves[cave_index++].old_offset : e.offset;
        for (size_t i = 0; i < e.on.size();) {
            const size_t len = BranchAt(e.on, i);
            if (len == 0) {
                ++i;
                continue;
            }
            s32 rel;
            std::memcpy(&rel, e.on.data() + i + len - 4, 4);
            const u64 old_target = old_base + i + len + s64(rel);
            u64 new_target = old_target;
            // Into a moved cave from anywhere; out of a moved cave only back into the game's
            // code (anything else found in a cave's bytes is not a branch).
            const bool fix = moved_target(old_target, new_target) ||
                             (e.cave && MemoryPatcher::EbootRangeIsCode(old_target, 1));
            if (!fix) {
                ++i;
                continue;
            }
            const s64 new_rel = s64(new_target) - s64(e.offset + i + len);
            if (new_rel < INT32_MIN || new_rel > INT32_MAX) {
                return false;
            }
            const s32 rel32 = s32(new_rel);
            std::memcpy(e.on.data() + i + len - 4, &rel32, 4);
            i += len;
        }
    }
    mod.moved = !caves.empty();
    return true;
}

/// Finds where every entry of the file lives in the loaded game.
void Resolve(File& file) {
    // Pass 1: the entries the game's bytes identify. They also tell which kind of offset the
    // file uses, for the entries that cannot be identified (value writes).
    struct Found {
        bool known = false;
        bool absolute = false;
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
            if (Holds(e, e.offset) >= 0) {
                found[m][i] = {true, false};
                ++relative_votes;
            } else if (e.offset >= Ps4ImageBase && Holds(e, e.offset - Ps4ImageBase) >= 0) {
                found[m][i] = {true, true};
                ++absolute_votes;
            }
        }
    }
    const bool file_absolute = absolute_votes > relative_votes;

    for (size_t m = 0; m < file.mods.size(); ++m) {
        Mod& mod = file.mods[m];
        mod.state = ModState::Ready;
        bool mod_absolute = file_absolute;
        for (const Found& f : found[m]) {
            if (f.known) {
                mod_absolute = f.absolute;
                break;
            }
        }
        std::vector<bool> outside(mod.entries.size(), false);
        bool any_outside = false;
        for (size_t i = 0; i < mod.entries.size(); ++i) {
            Entry& e = mod.entries[i];
            const Found& f = found[m][i];
            const bool can_be_absolute = e.offset >= Ps4ImageBase;
            if (f.known) {
                if (f.absolute) {
                    e.offset -= Ps4ImageBase;
                }
                continue;
            }
            const bool loaded_relative = MemoryPatcher::EbootRangeLoaded(e.offset, e.on.size());
            const bool loaded_absolute =
                can_be_absolute &&
                MemoryPatcher::EbootRangeLoaded(e.offset - Ps4ImageBase, e.on.size());
            if (Blind(e)) {
                if (mod_absolute && can_be_absolute) {
                    e.offset -= Ps4ImageBase;
                }
                if (!MemoryPatcher::EbootRangeLoaded(e.offset, e.on.size())) {
                    mod.state = ModState::Outside; // a value outside the game: nothing to move
                }
            } else if (!loaded_relative && !(mod_absolute && loaded_absolute)) {
                // Outside the game's image: a code cave at an address only a console has.
                if (mod_absolute && can_be_absolute) {
                    e.offset -= Ps4ImageBase;
                }
                outside[i] = true;
                any_outside = true;
            } else {
                mod.state = ModState::Mismatch; // inside, but not the file's bytes
            }
        }
        if (mod.state == ModState::Ready && any_outside && !MoveCaves(file, mod, outside)) {
            mod.state = ModState::Outside;
        }
        // A jump the cheat writes into the game's code must land in the game's code or in a
        // cave of ours.
        for (size_t i = 0; i < mod.entries.size() && mod.state == ModState::Ready; ++i) {
            const Entry& e = mod.entries[i];
            if (e.on.size() >= 5 && e.on[0] == 0xe9 && !e.cave &&
                MemoryPatcher::EbootRangeIsCode(e.offset, e.on.size())) {
                s32 rel;
                std::memcpy(&rel, e.on.data() + 1, 4);
                const s64 target = s64(e.offset) + 5 + rel;
                if (target < 0 || !(MemoryPatcher::EbootRangeIsCode(u64(target), 1) ||
                                    InCaveArea(u64(target), 1))) {
                    mod.state = ModState::Outside;
                }
            }
        }
    }
}

/// Whether the game's memory holds the cheat's "on" bytes (checkboxes).
bool IsOn(const Mod& mod) {
    if (mod.state != ModState::Ready || mod.is_button) {
        return false;
    }
    bool any = false;
    for (const Entry& e : mod.entries) {
        if (Blind(e) || e.cave) {
            continue;
        }
        if (Holds(e, e.offset) != 1) {
            return false;
        }
        any = true;
    }
    return any;
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
        mod.name = jm.contains("name") && jm["name"].is_string() ? jm["name"].get<std::string>()
                                                                   : std::string{"(no name)"};
        mod.is_button = jm.contains("type") && jm["type"].is_string() &&
                        jm["type"].get<std::string>() == "button";
        bool ok = !jm["memory"].empty();
        for (const auto& je : jm["memory"]) {
            Entry e;
            const auto field = [&je](const char* key) {
                return je.is_object() && je.contains(key) && je[key].is_string()
                           ? je[key].get<std::string>()
                           : std::string{};
            };
            const std::string offset = field("offset");
            char* end = nullptr;
            e.offset = std::strtoull(offset.c_str(), &end, 16);
            ok &= !offset.empty() && end && *end == '\0' && ParseHex(field("on"), e.on) &&
                  ParseHex(field("off"), e.off) && e.on.size() == e.off.size();
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

/// On: the caves first, then the game's code that jumps to them. Off: the other way round.
void Write(const Mod& mod, bool on) {
    for (const bool caves : {on, !on}) {
        for (const Entry& e : mod.entries) {
            if (e.cave == caves) {
                const auto& bytes = on ? e.on : e.off;
                std::memcpy(At(e.offset), bytes.data(), bytes.size());
            }
        }
    }
}

/// Empty when the game's code still is what the cheat expects to replace (`on`) or what it
/// wrote (off); otherwise the name of the cheat that changed it meanwhile, or "?".
std::string Conflict(const Mod& mod, bool on, size_t file_index, size_t mod_index) {
    for (const Entry& e : mod.entries) {
        if (e.cave || Blind(e) || Holds(e, e.offset) == (on ? 0 : 1)) {
            continue;
        }
        if (on && Holds(e, e.offset) == 1) {
            continue; // already there (the same cheat from another file)
        }
        for (size_t f = 0; f < g_files.size(); ++f) {
            for (size_t m = 0; m < g_files[f].mods.size(); ++m) {
                const Mod& other = g_files[f].mods[m];
                if ((f == file_index && m == mod_index) || !IsOn(other)) {
                    continue;
                }
                for (const Entry& o : other.entries) {
                    if (!o.cave && o.offset < e.offset + e.on.size() &&
                        e.offset < o.offset + o.on.size()) {
                        return other.name;
                    }
                }
            }
        }
        return "?";
    }
    return {};
}

void WriteRemembered() {
    std::ofstream out{RememberedPath(), std::ios::trunc};
    for (const File& file : g_files) {
        for (const Mod& mod : file.mods) {
            if (IsOn(mod)) {
                out << file.file_name << '\t' << mod.name << '\n';
            }
        }
    }
}

/// Reads the folder; cheats in `keep_on` (and usable) are switched on.
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
        u32 ready = 0, moved = 0;
        for (const Mod& mod : file.mods) {
            ready += mod.state == ModState::Ready;
            moved += mod.state == ModState::Ready && mod.moved;
        }
        LOG_INFO(Loader, "Cheats: {} ({} {}): {} of {} cheats fit this game ({} with a moved "
                         "code cave)",
                 file.file_name, file.id, file.version, ready, file.mods.size(), moved);
        g_files.push_back(std::move(file));
    }
    for (size_t f = 0; f < g_files.size(); ++f) {
        for (size_t m = 0; m < g_files[f].mods.size(); ++m) {
            const Mod& mod = g_files[f].mods[m];
            if (mod.state != ModState::Ready || mod.is_button || IsOn(mod) ||
                !keep_on.contains({g_files[f].file_name, mod.name})) {
                continue;
            }
            if (const std::string by = Conflict(mod, true, f, m); !by.empty()) {
                LOG_WARNING(Loader, "Cheats: \"{}\" stays off: \"{}\" changes the same code",
                            mod.name, by);
                continue;
            }
            Write(mod, true);
            LOG_INFO(Loader, "Cheats: \"{}\" on ({})", mod.name, g_files[f].file_name);
        }
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
            if (IsOn(mod)) {
                keep_on.emplace(file.file_name, mod.name);
            }
        }
    }
    LoadLocked(keep_on);
}

std::vector<File> Files() {
    std::scoped_lock lk{g_mutex};
    for (File& file : g_files) {
        for (Mod& mod : file.mods) {
            mod.on = IsOn(mod);
        }
    }
    return g_files;
}

std::string Set(size_t file, size_t mod_index, bool on) {
    std::scoped_lock lk{g_mutex};
    if (file >= g_files.size() || mod_index >= g_files[file].mods.size()) {
        return "No such cheat.";
    }
    const Mod& mod = g_files[file].mods[mod_index];
    if (mod.state != ModState::Ready) {
        return "This cheat does not fit the game.";
    }
    if (mod.is_button) {
        on = true;
    } else if (IsOn(mod) == on) {
        return {};
    }
    if (const std::string by = Conflict(mod, on, file, mod_index); !by.empty()) {
        if (on) {
            return by == "?" ? "\"" + mod.name + "\": the game's code there has been changed."
                             : "\"" + mod.name + "\" changes the same code as \"" + by +
                                   "\": switch that one off first.";
        }
        // Off, but another cheat wrote over it since: leave the game's code alone.
        return "\"" + mod.name + "\" was already replaced by \"" + by + "\".";
    }
    Write(mod, on);
    if (!mod.is_button) {
        WriteRemembered();
    }
    LOG_INFO(Loader, "Cheats: \"{}\" {} ({})", mod.name,
             mod.is_button ? "applied" : on ? "on" : "off", g_files[file].file_name);
    return {};
}

} // namespace Cheats
