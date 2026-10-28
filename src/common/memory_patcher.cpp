// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <codecvt>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <pugixml.hpp>
#include "common/elf_info.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "video_core/renderer_vulkan/vk_dlss.h"
#include "core/emulator_state.h"
#include "core/file_format/psf.h"
#include "memory_patcher.h"

namespace VideoCore {
void PS4_SYSV_ABI ParticleArenaDiscardHook(); // buffer_cache.cpp
}

namespace MemoryPatcher {

EXPORT uintptr_t g_eboot_address;
uint64_t g_eboot_image_size;
uintptr_t g_eboot_scratch = 0;
std::string g_game_serial;
std::string patch_file;
bool patches_applied = false;
std::vector<patchInfo> pending_patches;
static bool g_particle_late_read_applied = false; // V1 written (ApplyParticleLightOneSubmit needs it)

// GitHub #22 (2026-10-22): every built-in patch and game-memory reader in this file uses eboot
// 01.07 offsets. A US 01.00 eboot (CUSA00223, code hash 0xc8fba35195996bbd) loads its code at
// +0x400000, so +0x3ee160 was not mapped and the first byte compare crashed at boot. g_layout
// is decided once in OnGameLoaded (the offsets inside loaded segments + anchor bytes); everything
// below checks it first. Not the code hash: a 01.07 copy with another hash (i24) runs the patches.
static std::vector<std::pair<uintptr_t, u64>> g_eboot_segments;

// inFAMOUS First Light (2026-10-26): the standalone CUSA00575 / CUSA00897 (01.04, code hash
// 0x9b94fdbae3a6f151) is the same engine built later - its code and globals moved, the renderer
// object and the camera kept their layout (the mode toggle's field offsets are byte-identical).
// The sites below were found from 01.07's by masked instruction signatures (rip-relative and branch
// displacements masked; each one unique) and are byte-compared before any write, as on 01.07.
// First Light's main loop, particle code and streamer differ: the particle, house-split and
// performance patches stay Second Son only (second_son).
struct EbootLayout {
    const char* name;
    bool second_son;
    u64 light_list;    // imul esi, r15d, 0x320000 (imm at +3); mov r8d, 0x320000 at +0x19
    u64 two_load;      // the projection builder's vmovss xmm3, [rip + 2.0]
    u32 two_disp;      // its disp32
    u64 pad;           // 4 never-executed padding bytes before that function
    u64 size_imms[8];  // render w / h, output w / h, 2D screen space w / h, mode toggle w / h
    u64 movie_k;       // the movie draw's 16.0, 9.0
    u64 movie_rect;    // its quad (-1, -1, 1, 1)
    u64 renderer;      // the renderer object (mode bytes and size table, see Renderer*)
    u64 hud_floor[3];  // the safe-area floor loads (size setup and two refreshes)
    u64 hud_ymul[3];   // their vmulss H * r
    u64 jitter_a;      // the y numerator load (x numerator at +8, the parity jz at +0x22)
    u64 jitter_b;      // the first SMAA subsample-index load (the second at +0xa)
    u32 jitter_disp[4]; // their original disp32: y, x numerator, index 1, index 2
    u64 camera_ptr;    // the main camera object pointer
    u64 metering_size[4];  // the exposure metering buffers' mov esi, 0x9c4 (0 = not patched)
    u64 metering_count[4]; // their mov ecx, 0x271 (the V#'s 625 records)
};

constexpr EbootLayout SecondSon107{
    "Second Son 01.07", true, 0x16be04, 0x581d10, 0x562798, 0x581c84,
    {0x1eff4a, 0x1eff51, 0x1eff57, 0x1eff5e, 0x1f0403, 0x1f03fb, 0x1f0d3e, 0x1f0d4d},
    0xaadbe0, 0xaadd60, 0xf1bcd0, {0x1f042f, 0x1f2aae, 0x1f2bae}, {0x1f0471, 0x1f2afb, 0x1f2bfb},
    0x472976, 0x47223c, {0x0063c6aa, 0x0063c6a6, 0x0063cd6c, 0x0063cd52}, 0x7f2d2b8,
};
constexpr EbootLayout FirstLight104{
    "First Light 01.04", false, 0x190694, 0x65c4f0, 0x576a70, 0x65c464,
    {0x2369da, 0x2369e1, 0x2369e7, 0x2369ee, 0x236e93, 0x236e8b, 0x2377ce, 0x2377dd},
    0xb9c0c0, 0xb9c240, 0x11362d0, {0x236ebf, 0x23954e, 0x23964e}, {0x236f01, 0x23959b, 0x23969b},
    0x51c9ad, 0x51c26c, {0x00680b93, 0x00680b8f, 0x0068125c, 0x00681242}, 0x81a3f98,
    {0x19767a, 0x1976c4, 0x198680, 0x1986c7}, {0x19769e, 0x1976e8, 0x1986a3, 0x1986ea},
};
// Renderer fields (the same in both): the stored / wanted colour option bytes, the size table, the
// quality level.
constexpr u64 RendererStoredOption = 0xdf0f06;
constexpr u64 RendererWantedOption = 0xdf0f11;
constexpr u64 RendererSizeTable = 0xdf0f14;
constexpr u64 RendererQualityLevel = 0x1dca8;

// The layout of the loaded eboot, decided once in OnGameLoaded; nullptr = the built-in patches off.
static const EbootLayout* g_layout = nullptr;

static bool SecondSon107Loaded() {
    return g_layout == &SecondSon107;
}

void AddEbootSegment(uintptr_t address, uint64_t size) {
    g_eboot_segments.emplace_back(address, size);
}

static bool EbootRangeLoaded(u64 offset, u64 size) {
    const uintptr_t begin = g_eboot_address + offset;
    return std::ranges::any_of(g_eboot_segments, [&](const auto& s) {
        return begin >= s.first && begin + size <= s.first + s.second;
    });
}

static bool LayoutMatches(const EbootLayout& l) {
    // The lowest and highest code-segment offsets the patches touch (the light list ... the movie
    // quad) and the highest data offset (the main camera pointer).
    if (!EbootRangeLoaded(l.light_list, l.movie_rect + 16 - l.light_list) ||
        !EbootRangeLoaded(l.camera_ptr, 8)) {
        return false;
    }
    // Anchors: the light list's imul esi, r15d, 0x320000 and the projection's 2.0 load.
    static constexpr u8 imul[7] = {0x41, 0x69, 0xf7, 0x00, 0x00, 0x32, 0x00};
    u8 two_load[8] = {0xc5, 0xfa, 0x10, 0x1d};
    std::memcpy(two_load + 4, &l.two_disp, 4);
    return std::memcmp(reinterpret_cast<const void*>(g_eboot_address + l.light_list), imul, 7) ==
               0 &&
           std::memcmp(reinterpret_cast<const void*>(g_eboot_address + l.two_load), two_load, 8) ==
               0;
}

static const EbootLayout* DetectLayout() {
    for (const EbootLayout* l : {&SecondSon107, &FirstLight104}) {
        if (LayoutMatches(*l)) {
            return l;
        }
    }
    return nullptr;
}

std::string toHex(u64 value, size_t byteSize) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(byteSize * 2) << value;
    return ss.str();
}

static bool isHexSym(const std::string& s) {
    return (s.size() >= 2 && (s[0] == '$' || s[0] == '#'));
}

static bool isHex0x(const std::string& s) {
    return (s.size() >= 3 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'));
}

// possible prefixes == (s == `0x` || s == `#` || s == `$`) will be interpet as hex value, else is
// decimal
static int convertNumBase(const std::string& s) {
    return (isHex0x(s) || isHexSym(s)) ? 16 : 10;
}

std::string convertValueToHex(const std::string type, const std::string valueStr) {
    std::string result;

    if (type == "byte") {
        const u32 value = std::stoul(valueStr, nullptr, convertNumBase(valueStr));
        result = toHex(value, 1);
    } else if (type == "bytes16") {
        const u32 value = std::stoul(valueStr, nullptr, convertNumBase(valueStr));
        result = toHex(value, 2);
    } else if (type == "bytes32") {
        const u32 value = std::stoul(valueStr, nullptr, convertNumBase(valueStr));
        result = toHex(value, 4);
    } else if (type == "bytes64") {
        const u64 value = std::stoull(valueStr, nullptr, convertNumBase(valueStr));
        result = toHex(value, 8);
    } else if (type == "float32") {
        union {
            float f;
            uint32_t i;
        } floatUnion;
        floatUnion.f = std::stof(valueStr);
        result = toHex(std::byteswap(floatUnion.i), sizeof(floatUnion.i));
    } else if (type == "float64") {
        union {
            double d;
            uint64_t i;
        } doubleUnion;
        doubleUnion.d = std::stod(valueStr);
        result = toHex(std::byteswap(doubleUnion.i), sizeof(doubleUnion.i));
    } else if (type == "utf8") {
        std::vector<unsigned char> byteArray =
            std::vector<unsigned char>(valueStr.begin(), valueStr.end());
        byteArray.push_back('\0');
        std::stringstream ss;
        for (unsigned char c : byteArray) {
            ss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(c);
        }
        result = ss.str();
    } else if (type == "utf16") {
        std::wstring wide_str(valueStr.size(), L'\0');
        std::mbstowcs(&wide_str[0], valueStr.c_str(), valueStr.size());
        wide_str.resize(std::wcslen(wide_str.c_str()));

        std::u16string valueStringU16;

        for (wchar_t wc : wide_str) {
            if (wc <= 0xFFFF) {
                valueStringU16.push_back(static_cast<char16_t>(wc));
            } else {
                wc -= 0x10000;
                valueStringU16.push_back(static_cast<char16_t>(0xD800 | (wc >> 10)));
                valueStringU16.push_back(static_cast<char16_t>(0xDC00 | (wc & 0x3FF)));
            }
        }

        std::vector<unsigned char> byteArray;
        // convert to little endian
        for (char16_t ch : valueStringU16) {
            unsigned char low_byte = static_cast<unsigned char>(ch & 0x00FF);
            unsigned char high_byte = static_cast<unsigned char>((ch >> 8) & 0x00FF);

            byteArray.push_back(low_byte);
            byteArray.push_back(high_byte);
        }
        byteArray.push_back('\0');
        byteArray.push_back('\0');
        std::stringstream ss;

        for (unsigned char ch : byteArray) {
            ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(ch);
        }
        result = ss.str();
    } else if (type == "bytes") {
        result = valueStr;
    } else if (type == "mask" || type == "mask_jump32") {
        result = valueStr;
    } else {
        LOG_INFO(Loader, "Error applying Patch, unknown type: {}", type);
    }
    return result;
}

void ApplyPendingPatches();

void ApplyPatchesFromXML(std::filesystem::path path) {
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(path.c_str());

    auto* param_sfo = Common::Singleton<PSF>::Instance();
    auto app_version = param_sfo->GetString("APP_VER").value_or("Unknown version");

    if (result) {
        auto patchXML = doc.child("Patch");
        for (pugi::xml_node_iterator it = patchXML.children().begin();
             it != patchXML.children().end(); ++it) {

            if (std::string(it->name()) == "Metadata") {
                if (std::string(it->attribute("isEnabled").value()) == "true") {
                    std::string currentPatchName = it->attribute("Name").value();
                    std::string metadataAppVer = it->attribute("AppVer").value();
                    bool versionMatches = metadataAppVer == app_version;

                    auto patchList = it->first_child();
                    for (pugi::xml_node_iterator patchLineIt = patchList.children().begin();
                         patchLineIt != patchList.children().end(); ++patchLineIt) {

                        std::string type = patchLineIt->attribute("Type").value();
                        if (!versionMatches && type != "mask" && type != "mask_jump32")
                            continue;

                        std::string address = patchLineIt->attribute("Address").value();
                        std::string patchValue = patchLineIt->attribute("Value").value();
                        std::string maskOffsetStr = patchLineIt->attribute("Offset").value();
                        std::string targetStr = "";
                        std::string sizeStr = "";
                        if (type == "mask_jump32") {
                            targetStr = patchLineIt->attribute("Target").value();
                            sizeStr = patchLineIt->attribute("Size").value();
                        } else {
                            try {
                                patchValue = convertValueToHex(type, patchValue);
                            } catch (std::exception& e) {
                                ASSERT_MSG(false,
                                           "Failed to parse patch value \"{}\" for \"{}\" in "
                                           "patch \"{}\", error: \"{}\"\n"
                                           "If the patch was working on earlier versions, then it "
                                           "was using a format that shadPS4 handled incorrectly, "
                                           "and the patch should instead be fixed.",
                                           patchValue, address, currentPatchName, e.what());
                            }
                        }

                        bool littleEndian = false;
                        if (type == "bytes16" || type == "bytes32" || type == "bytes64") {
                            littleEndian = true;
                        }

                        MemoryPatcher::PatchMask patchMask = MemoryPatcher::PatchMask::None;
                        int maskOffsetValue = 0;

                        if (type == "mask")
                            patchMask = MemoryPatcher::PatchMask::Mask;

                        if (type == "mask_jump32")
                            patchMask = MemoryPatcher::PatchMask::Mask_Jump32;

                        if ((type == "mask" || type == "mask_jump32") && !maskOffsetStr.empty()) {
                            maskOffsetValue = std::stoi(maskOffsetStr, 0, 10);
                        }

                        const patchInfo patch = {
                            .gameSerial = "*",
                            .modNameStr = currentPatchName,
                            .offsetStr = address,
                            .valueStr = patchValue,
                            .targetStr = targetStr,
                            .sizeStr = sizeStr,
                            .littleEndian = littleEndian,
                            .patchMask = patchMask,
                            .maskOffset = maskOffsetValue,
                        };
                        MemoryPatcher::PatchMemory(patch);
                    }
                }
            }
        }
    } else {
        LOG_ERROR(Loader, "Could not parse patch XML: {}", result.description());
    }
}

// SHADPS4_PARTICLE_LATE_READ=1 (2026-10-12, notes/one-frame-late-particles-2026-10-12.md sec. 3.1, V1):
// inFAMOUS Second Son (CUSA00004 / 01.07) reads its particle control blocks (survivor counts, light
// counts, CPU collision) at the start of the next frame, just before 0x3ee7d0 consumes them, instead
// of right after kicking this frame's particle chain. Game:Main then no longer waits for that chain's
// GPU work mid-frame. Three writes (eboot offsets): a cave in the unreferenced 0x3ee160 that calls
// 0x3edef0 (readback) and 0x3ee470 (collision join + label wait), then sets up 0x3ee7d0's arguments
// and jumps back; the main loop jumps to it at 0x29cc89; the old 0x3edef0 call at 0x29cd1e becomes a
// 5-byte NOP. The end-of-frame 0x3ee470 call at 0x29cd56 stays. Refused unless every byte matches.
static void ApplyParticleLateRead() {
    const char* env = std::getenv("SHADPS4_PARTICLE_LATE_READ");
    if (!env || env[0] != '1') {
        return;
    }
    // No serial / APP_VER check (2026-10-19): a DLC param.sfo in a player's game folder read as
    // "Unknown version" and switched this off; the byte compare below is the gate.
    struct Bytes {
        u64 offset;
        std::vector<u8> bytes;
    };
    // Original bytes: the patched ranges plus the instructions around them the cave relies on.
    const std::vector<Bytes> original = {
        // FUN_8003ee160 (no references): 43 code bytes + 5 pad
        {0x3ee160, {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x8b, 0x5f, 0x58, 0xeb, 0x13,
                    0x0f, 0x1f, 0x40, 0x00, 0x48, 0x89, 0xdf, 0xe8, 0x58, 0x5d, 0x00, 0x00,
                    0x48, 0x8b, 0x9b, 0xb0, 0x00, 0x00, 0x00, 0x48, 0x85, 0xdb, 0x75, 0xec,
                    0x48, 0x83, 0xc4, 0x08, 0x5b, 0x5d, 0xc3, 0x90, 0x90, 0x90, 0x90, 0x90}},
        // call 0x41b8d0; lea rax,[0x800f1bcd0]; lea rdi,[0x807d3eb10]; mov rsi,[rax+0x1dca0];
        // call 0x3ee7d0
        {0x29cc84, {0xe8, 0x47, 0xec, 0x17, 0x00, 0x48, 0x8d, 0x05, 0x40, 0xf0, 0xc7,
                    0x00, 0x48, 0x8d, 0x3d, 0x79, 0x1e, 0xaa, 0x07, 0x48, 0x8b, 0xb0,
                    0xa0, 0xdc, 0x01, 0x00, 0xe8, 0x2d, 0x1b, 0x15, 0x00}},
        // lea rbx,[0x807d3eb10]; mov rdi,rbx; call 0x3edef0
        {0x29cd14, {0x48, 0x8d, 0x1d, 0xf5, 0x1d, 0xaa, 0x07, 0x48, 0x89, 0xdf, 0xe8, 0xcd, 0x11,
                    0x15, 0x00}},
        // mov rdi,rbx; call 0x3ee470 (kept)
        {0x29cd53, {0x48, 0x89, 0xdf, 0xe8, 0x15, 0x17, 0x15, 0x00}},
    };
    const std::vector<Bytes> patched = {
        // lea rbx,[0x807d3eb10]; mov rdi,rbx; call 0x3edef0; mov rdi,rbx; call 0x3ee470;
        // lea rax,[0x800f1bcd0]; mov rdi,rbx; mov rsi,[rax+0x1dca0]; jmp 0x29cc9e
        {0x3ee160, {0x48, 0x8d, 0x1d, 0xa9, 0x09, 0x95, 0x07, 0x48, 0x89, 0xdf, 0xe8, 0x81,
                    0xfd, 0xff, 0xff, 0x48, 0x89, 0xdf, 0xe8, 0xf9, 0x02, 0x00, 0x00, 0x48,
                    0x8d, 0x05, 0x52, 0xdb, 0xb2, 0x00, 0x48, 0x89, 0xdf, 0x48, 0x8b, 0xb0,
                    0xa0, 0xdc, 0x01, 0x00, 0xe9, 0x11, 0xeb, 0xea, 0xff}},
        // jmp 0x3ee160 + 16 NOPs up to the call 0x3ee7d0 at 0x29cc9e
        {0x29cc89, {0xe9, 0xd2, 0x14, 0x15, 0x00, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90,
                    0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90}},
        // nopl 0x0(rax,rax,1) in place of call 0x3edef0
        {0x29cd1e, {0x0f, 0x1f, 0x44, 0x00, 0x00}},
    };
    for (const auto& o : original) {
        const auto* live = reinterpret_cast<const u8*>(g_eboot_address + o.offset);
        if (std::memcmp(live, o.bytes.data(), o.bytes.size()) != 0) {
            std::string found;
            for (size_t i = 0; i < o.bytes.size(); ++i) {
                found += fmt::format("{:02x}", live[i]);
            }
            LOG_ERROR(Loader, "SHADPS4_PARTICLE_LATE_READ: refused, eboot+{:#x} holds {}", o.offset,
                      found);
            return;
        }
    }
    // SHADPS4_PARTICLE_ARENA_DISCARD=1 (2026-10-13): the main loop first calls a host hook that
    // drops the GPU-modified state of the arena half 0x3ee7d0 is about to reuse (dead data), then
    // jumps to the cave: movabs rax, hook; call rax; jmp 0x3ee160; 4 NOPs (21 bytes at 0x29cc89).
    std::vector<Bytes> writes = patched;
    const char* discard = std::getenv("SHADPS4_PARTICLE_ARENA_DISCARD");
    if (discard && discard[0] == '1') {
        const u64 hook = reinterpret_cast<u64>(&VideoCore::ParticleArenaDiscardHook);
        std::vector<u8> b = {0x48, 0xb8};
        for (int i = 0; i < 8; ++i) {
            b.push_back(static_cast<u8>(hook >> (8 * i)));
        }
        // Opus review (q43): the hook runs AFTER the cave (the cave may still read a stale
        // control block in the half being discarded): 0x29cc89 jmp cave (as V1); the cave's
        // last jmp returns to 0x29cc8e = push rdi; push rsi; movabs rax, hook; call rax;
        // pop rsi; pop rdi (16 bytes, ends at 0x29cc9e = call 0x3ee7d0).
        std::vector<u8> tail = {0x57, 0x56};
        tail.insert(tail.end(), b.begin(), b.end());
        tail.insert(tail.end(), {0xff, 0xd0, 0x5e, 0x5f});
        std::vector<u8> jump = {0xe9, 0xd2, 0x14, 0x15, 0x00};
        jump.insert(jump.end(), tail.begin(), tail.end());
        writes[1] = {0x29cc89, jump};
        writes[0].bytes[41] = 0x01; // cave: jmp 0x29cc8e (E9 01 EB EA FF) instead of 0x29cc9e
        if (writes[1].bytes.size() != 21 || writes[0].bytes[40] != 0xe9) {
            LOG_ERROR(Loader, "SHADPS4_PARTICLE_ARENA_DISCARD: bad patch layout, not applied");
            writes = patched;
        } else {
            LOG_INFO(Loader, "SHADPS4_PARTICLE_ARENA_DISCARD: frame-start hook {:#x} after the cave",
                     hook);
        }
    }
    for (const auto& p : writes) {
        std::memcpy(reinterpret_cast<void*>(g_eboot_address + p.offset), p.bytes.data(),
                    p.bytes.size());
    }
    g_particle_late_read_applied = true;
    LOG_INFO(Loader,
             "SHADPS4_PARTICLE_LATE_READ: applied V1 (particle readback one frame later): cave "
             "{:#x}, jump {:#x}, nop {:#x}",
             g_eboot_address + 0x3ee160, g_eboot_address + 0x29cc89, g_eboot_address + 0x29cd1e);
}

// SHADPS4_PARTICLE_LIGHT_ONE_SUBMIT=1 (2026-10-16, notes/study-fight-dips-2026-10-15.md sec. 5, G1):
// 0x3f3ed0 records two DMA_DATAs per light-emitting particle effect and then submits the ring
// (call 0x1691e0 at eboot 0x3f4036) - one ring submit per light, ~25 per typical fight frame, ~74 in
// the worst. With V1 (PARTICLE_LATE_READ) the copies stay in the command buffer 0x807d3cf20 and go out
// with 0x3ee7d0's first submit of the same frame, so the call becomes a 5-byte NOP. Only with V1
// applied: in the vanilla order the copies would wait for the next frame (lights one frame late).
static void ApplyParticleLightOneSubmit() {
    const char* env = std::getenv("SHADPS4_PARTICLE_LIGHT_ONE_SUBMIT");
    if (!env || env[0] != '1') {
        return;
    }
    if (!g_particle_late_read_applied) {
        LOG_ERROR(Loader, "SHADPS4_PARTICLE_LIGHT_ONE_SUBMIT: refused, needs SHADPS4_PARTICLE_LATE_READ "
                          "applied (V1)");
        return;
    }
    constexpr u64 offset = 0x3f4036;
    static constexpr u8 original[] = {0xe8, 0xa5, 0x51, 0xd7, 0xff}; // call 0x1691e0
    static constexpr u8 nop5[] = {0x0f, 0x1f, 0x44, 0x00, 0x00};
    auto* live = reinterpret_cast<u8*>(g_eboot_address + offset);
    if (std::memcmp(live, original, sizeof(original)) != 0) {
        LOG_ERROR(Loader, "SHADPS4_PARTICLE_LIGHT_ONE_SUBMIT: refused, eboot+{:#x} holds "
                          "{:02x} {:02x} {:02x} {:02x} {:02x}",
                  offset, live[0], live[1], live[2], live[3], live[4]);
        return;
    }
    std::memcpy(live, nop5, sizeof(nop5));
    LOG_INFO(Loader, "SHADPS4_PARTICLE_LIGHT_ONE_SUBMIT: per-light ring submit at {:#x} is a NOP",
             g_eboot_address + offset);
}

// SHADPS4_HOUSE_SPLIT=1 (2026-10-14, notes/study-house-draws-2026-10-14.md sec. 3): the game's
// command-buffer splitter FUN_80043cef0 cuts a pass's item list into jobs of <= 1000 items only where
// key & mask changes; the mask comes from FUN_800197130 (default 0xffff000000000000 = class runs).
// The rooftop's ~3,300 distant houses are one class run, so ONE job records them all. Writing
// 0xffffffffffff0000 (everything above bit 16: a superset of today's cut points) lets the splitter
// cut that run into ~5 jobs recorded in parallel. SHADPS4_HOUSE_CHUNK=N also sets the 1000-item chunk
// (FUN_800197050's imm32). Refused unless every original byte matches. The game's context pool has
// no bounds check (at most 254 per frame): the census in gnmdriver logs the count per frame.
static int g_screen_w = 0, g_screen_h = 0;
static int g_display_w = 0, g_display_h = 0;    // the monitor the window opened on, in pixels
static u32 g_render_w = 1920, g_render_h = 1080; // the game's render size after ApplyAspectRatio
static float g_hud_ratio = 1.0f;
static double g_aspect = 16.0 / 9.0; // the shape the game renders, decided at boot
static u32 g_h0 = 1080;              // its height at 16:9 (the size setup's basis)
static float g_percent_now = 0.0f;   // the screen percentage in effect (0 = the game's 1080 lines)
static bool g_height_from_flags = false;
// Live screen percentage (1.1, user 10-20 ~21:00; notes/study-live-resolution-2026-10-20.md route
// A): the game boots at the largest size M (its buffers get memory for M once) and draws at the
// live size S <= M inside it; a change pokes the game's own Pro / HDR mode switch FUN_8001f0be0.
static RenderSize g_max{1920, 1080};       // M: the boot size (the allocation)
static bool g_live_ok = false;             // live changes possible this run
static std::atomic<u64> g_live_pending{0}; // S to apply at the next sceGnmSubmitDone (w << 32 | h)
static std::atomic<u64> g_now_size{(u64(1920) << 32) | 1080}; // the size drawn (w << 32 | h)
static std::atomic<s64> g_live_ms{0};      // steady-clock ms of the last applied change
// A change was taken: the back drops the old size's targets (old w << 32 | h, new; 0 = none).
static std::atomic<u64> g_purge_old{0};
static std::atomic<u64> g_purge_new{0};
static std::atomic<int> g_purge_tries{0};
static u64 g_purge_last_old = 0; // the back: the old size it is trying to drop
static u64 g_live_check = 0; // Game:Main: the size last poked, until the game took it
static int g_live_tries = 0; // Game:Main: pokes of g_live_check so far

static s64 NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
constexpr const char* RenderSizeFile = "render-size.ini";
constexpr u32 MinLines = 96, MaxLines = 4320;
// At most 4K's pixels in any shape (user 10-21: native 4K in 1.1). 3840x2160 lost the device on
// 10-20 (s1491 / s1493): the game's tile light list held only 25,600 tiles - ApplyLightListCapacity
// now sizes it from the boot size. Was 3200x1800 (s1492 ran clean, 97 FPS); 10% (256x144) ran too.
constexpr u64 MaxPixels = u64(3840) * 2160;
// The live headroom (the size every run boots at for live changes) stays at 3200x1800's pixels:
// tested on 10-20 / 10-21, and smaller cards do not draw the loading frames at 4K. A pick above it
// boots at the pick.
constexpr u64 HeadroomPixels = u64(3200) * 1800;

float HudSafeAreaRatio() {
    return g_hud_ratio;
}

void SetScreenSize(int width, int height) {
    g_screen_w = width;
    g_screen_h = height;
}

void SetDisplaySize(int width, int height) {
    g_display_w = width;
    g_display_h = height;
}

namespace {

u32 Round8(double v) {
    return u32(std::lround(v / 8.0) * 8);
}

/// The render size at 16:9-basis height h0 in the shape `aspect`: wide shapes keep the height
/// (at most 32:9), narrow ones the 16:9 width (at most 1:1).
RenderSize SizeAt(u32 h0, double aspect) {
    const bool narrow = aspect < 16.0 / 9.0 - 1e-6;
    const u32 w0 = Round8(h0 * 16.0 / 9.0);
    return narrow ? RenderSize{w0, std::min(w0, Round8(w0 / aspect))}
                  : RenderSize{std::min(Round8(h0 * 32.0 / 9.0), Round8(h0 * aspect)), h0};
}

/// Lowers h0 until the size has at most `max_pixels` (the wide and tall shapes).
u32 CapPixels(u32 h0, double aspect, u64 max_pixels = MaxPixels) {
    while (h0 > MinLines) {
        const RenderSize s = SizeAt(h0, aspect);
        if (u64(s.width) * s.height <= max_pixels) {
            break;
        }
        h0 -= 8;
    }
    return h0;
}

/// 100% of the screen: the largest box of the render shape on the monitor (the boot window is
/// often 1280x720 before F11, so the monitor is the reference; the window without one).
RenderSize Reference(double aspect) {
    double dw = g_display_w > 0 ? g_display_w : g_screen_w;
    double dh = g_display_w > 0 ? g_display_h : g_screen_h;
    if (dw <= 0.0 || dh <= 0.0) {
        dw = 1920.0;
        dh = 1080.0;
    }
    return aspect >= dw / dh ? RenderSize{u32(dw), u32(std::lround(dw / aspect))}
                             : RenderSize{u32(std::lround(dh * aspect)), u32(dh)};
}

/// The 16:9-basis height of 100%.
double FullBaseHeight(double aspect) {
    const RenderSize ref = Reference(aspect);
    return std::min<double>(ref.height, ref.width * 9.0 / 16.0);
}

/// The 16:9-basis height for `percent` of the screen, 96 lines .. 8K.
u32 BaseHeightForPercent(float percent, double aspect) {
    // 33 and 67 = DLSS's exact 1/3 and 2/3 (Ultra Performance takes one exact size: 4K -> 720
    // lines, 1080p -> 640x360; 0.33 would miss it by 8 lines).
    const int whole = int(std::lround(percent));
    const double p = whole == 33   ? 1.0 / 3.0
                     : whole == 67 ? 2.0 / 3.0
                                   : std::clamp(percent, 10.0f, 200.0f) / 100.0;
    return CapPixels(std::clamp(Round8(FullBaseHeight(aspect) * p), MinLines, MaxLines), aspect);
}

std::filesystem::path RenderSizePath() {
    return Common::FS::GetUserPath(Common::FS::PathType::UserDir) / RenderSizeFile;
}

// Native 4K (user 10-21; notes/study-4k-crash-2026-10-20.md, measured 10-21 in the user's capture):
// the deferred lighting cs_ce1675e0 writes up to 128 light indices per 16x16-pixel tile into one
// u16 buffer and skips the tiles past its size; ~40 forward-lit pixel shaders walk a skipped
// tile's list to an end marker that was never written, and the GPU hangs (device lost). The game
// makes it 0x320000 entries (25,600 tiles: 3200x1800 needs 22,600, 3840x2160 32,400) in
// FUN_80016bd10: the allocation (element bytes x 0x320000) and the descriptor's record count
// (0x320000) are two immediates. Both get the boot size M's tile count x 128 (M: every target is
// allocated at it, live sizes stay below).
void ApplyLightListCapacity(RenderSize max) {
    constexpr u32 Original = 0x320000;
    const u64 tiles = u64((max.width + 15) / 16) * ((max.height + 15) / 16);
    const u64 need = tiles * 128;
    if (need <= Original) {
        return;
    }
    const u32 entries = u32((need + 0xffff) & ~u64(0xffff));
    const auto at = [](u64 offset) { return reinterpret_cast<u8*>(g_eboot_address + offset); };
    const u64 imul_at = g_layout->light_list; // 0x16be04 on 01.07
    const u64 mov_at = imul_at + 0x19;
    static constexpr u8 imul_head[3] = {0x41, 0x69, 0xf7}; // imul esi, r15d, imm32
    static constexpr u8 mov_head[2] = {0x41, 0xb8};        // mov r8d, imm32
    u32 imul_imm = 0, mov_imm = 0;
    std::memcpy(&imul_imm, at(imul_at + 3), 4);
    std::memcpy(&mov_imm, at(mov_at + 2), 4);
    if (std::memcmp(at(imul_at), imul_head, 3) != 0 || std::memcmp(at(mov_at), mov_head, 2) != 0 ||
        imul_imm != Original || mov_imm != Original) {
        LOG_ERROR(Loader, "Light list: refused, FUN_80016bd10 differs ({}) - sizes above "
                          "3200x1800 may hang the GPU",
                  g_layout->name);
        return;
    }
    std::memcpy(at(imul_at + 3), &entries, 4);
    std::memcpy(at(mov_at + 2), &entries, 4);
    LOG_INFO(Loader, "Light list: {} entries for {}x{} ({} tiles; the game's own fits 25,600)",
             entries, max.width, max.height, tiles);
}

// First Light above 3200x1800 (2026-10-26, user: dark after the pause from 84% of a 4K screen,
// fine at 83%): the exposure metering cs_a8b68717 reduces 128x72 pixels per workgroup to one float
// in a buffer of 625 (0x9c4 bytes: 25 x 25 groups = 3200x1800, the Pro size). 3232x1816 needs
// 26 x 26 = 676 and the extra results ran over into the next buffer. Its four buffers (two
// setups, one per frame parity each) get the boot size M's group count: the allocation and the
// V#'s record count are two immediates per buffer.
void ApplyMeteringCapacity(RenderSize max) {
    constexpr u32 Original = 625;
    const u32 groups = ((max.width + 127) / 128) * ((max.height + 71) / 72);
    if (groups <= Original || g_layout->metering_size[0] == 0) {
        return;
    }
    const auto at = [](u64 offset) { return reinterpret_cast<u8*>(g_eboot_address + offset); };
    static constexpr u8 size_insn[5] = {0xbe, 0xc4, 0x09, 0x00, 0x00};  // mov esi, 0x9c4
    static constexpr u8 count_insn[5] = {0xb9, 0x71, 0x02, 0x00, 0x00}; // mov ecx, 0x271
    for (size_t i = 0; i < 4; ++i) {
        if (std::memcmp(at(g_layout->metering_size[i]), size_insn, 5) != 0 ||
            std::memcmp(at(g_layout->metering_count[i]), count_insn, 5) != 0) {
            LOG_ERROR(Loader, "Exposure metering: refused, buffer {} setup differs ({}) - the "
                              "picture may darken after the pause above 3200x1800",
                      i, g_layout->name);
            return;
        }
    }
    const u32 bytes = groups * 4;
    for (size_t i = 0; i < 4; ++i) {
        std::memcpy(at(g_layout->metering_size[i] + 1), &bytes, 4);
        std::memcpy(at(g_layout->metering_count[i] + 1), &groups, 4);
    }
    LOG_INFO(Loader, "Exposure metering: {} results for {}x{} (the game's own fit 625)", groups,
             max.width, max.height);
}

} // namespace

float SavedScreenPercent() {
    std::ifstream in{RenderSizePath()};
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("screen_percent=", 0) == 0) {
            const float v = float(std::atof(line.c_str() + 15));
            return v > 0.0f ? std::clamp(v, 10.0f, 200.0f) : 0.0f;
        }
    }
    return 0.0f;
}

void SaveScreenPercent(float percent) {
    const int value = percent > 0.0f ? int(std::lround(std::clamp(percent, 10.0f, 200.0f))) : 0;
    std::ofstream out{RenderSizePath(), std::ios::trunc};
    out << "# Written by the DLSS panel (Page Up): the game's render size in percent of the\n"
           "# screen (10-200; 0 = the game's own 1080 lines), read at start. SHADPS4_RENDER_HEIGHT\n"
           "# or SHADPS4_SCREEN_PERCENT in shadps4-flags-user.txt win over this file.\n"
        << "screen_percent=" << value << "\n";
    LOG_INFO(Loader, "Screen percentage {} saved (applies at the next start)", value);
}

RenderSize CurrentRenderSize() {
    const u64 v = g_now_size.load(std::memory_order_acquire);
    return {u32(v >> 32), u32(v)};
}

RenderSize PercentReference() {
    return Reference(g_aspect);
}

float CurrentScreenPercent() {
    // From the size drawn (a live change moves it): its 16:9-basis height over 100%'s.
    const RenderSize s = CurrentRenderSize();
    const double h0 = g_aspect < 16.0 / 9.0 - 1e-6 ? s.width * 9.0 / 16.0 : double(s.height);
    return float(100.0 * h0 / FullBaseHeight(g_aspect));
}

RenderSize RenderSizeForPercent(float percent) {
    return SizeAt(percent > 0.0f ? BaseHeightForPercent(percent, g_aspect) : 1080, g_aspect);
}

bool RenderHeightFromFlags() {
    return g_height_from_flags;
}

// Movies fit inside the screen (1.1.1, a player's DM 2026-10-22: the 21:9 intro was cut off at the
// top and bottom). The full-screen movie draw FUN_8004575b0 computes f = (ccc4 * 16) / (ccc0 * 9)
// and samples the movie at v = (1 - f) / 2 .. (1 + f) / 2 on a quad at (-1, -1, 1, 1): on the PS4
// f = 1; on 21:9 f = 0.762, so the movie filled the width and lost 11.9% at the top and bottom
// (measured s1597: v 0.119 - 0.874; on 4:3 f = 1.333 would sample outside the movie). Its
// constants are read by that function only: 16 / 9 become the screen space's width / height (f = 1:
// the whole movie) and the quad takes the movie's shape inside the screen (bars at the sides on
// wider screens, above and below on narrower ones; never stretched).
static float g_movie_fit_x = 0.0f, g_movie_fit_y = 0.0f; // 0 = not fitted

bool MovieFitRect(float& x, float& y) {
    x = g_movie_fit_x;
    y = g_movie_fit_y;
    return x > 0.0f;
}

static void ApplyMovieFit(u32 sw, u32 sh) {
    const double f = double(sh) * 16.0 / (double(sw) * 9.0);
    if (std::abs(f - 1.0) < 0.01) {
        return; // 16:9: the game's own draw
    }
    static constexpr float k_orig[2] = {16.0f, 9.0f};
    static constexpr float rect_orig[4] = {-1.0f, -1.0f, 1.0f, 1.0f};
    auto* k = reinterpret_cast<float*>(g_eboot_address + g_layout->movie_k);
    auto* rect = reinterpret_cast<float*>(g_eboot_address + g_layout->movie_rect);
    if (std::memcmp(k, k_orig, sizeof(k_orig)) != 0 ||
        std::memcmp(rect, rect_orig, sizeof(rect_orig)) != 0) {
        LOG_ERROR(Loader, "Movie fit: refused, the movie draw's constants differ ({})",
                  g_layout->name);
        return;
    }
    const float x = f < 1.0 ? float(f) : 1.0f;
    const float y = f < 1.0 ? 1.0f : float(1.0 / f);
    const float k_new[2] = {float(sw), float(sh)};
    const float rect_new[4] = {-x, -y, x, y};
    std::memcpy(k, k_new, sizeof(k_new));
    std::memcpy(rect, rect_new, sizeof(rect_new));
    g_movie_fit_x = x;
    g_movie_fit_y = y;
    LOG_INFO(Loader, "Movie fit: movies drawn {:.0f}% wide x {:.0f}% high, centred (screen space "
             "{}x{})", x * 100.0f, y * 100.0f, sw, sh);
}

// SHADPS4_ASPECT (1.1, user 2026-10-20; EXPERIMENTS "v1.1 C"): the game fills the screen's shape.
// The renderer's size setup FUN_8001eff00 stores the render size (+0xc / +0x10), the output size
// ([0] / [4]) and the 2D screen space (globals 0x801d0ccc0 / ccc4: HUD layout, and the camera's
// aspect = ccc0 / ccc4 at camera creation, 0x80058030c); the mode toggle FUN_8001f0be0 re-applies
// render / output from esi / edi. All eight 1920 / 1080 immediates get the target size: the
// buffers, the video-out buffers, the HUD (anchored to the edges) and the camera follow (Hor+: the
// vertical view stays, wider screens see more at the sides). Measured on 21:9 (s1426): round
// shapes, more picture left and right, HUD at the new edges, 0 Critical.
//   auto (default): wider screens see more at the sides, narrower ones (16:10, 4:3, 1:1) more above
//        and below (the 16:9 width is kept); crop: narrow screens cut the sides instead; W:H (e.g.
//        21:9, 32:9, 4:3): that shape; off.
// Decided once at boot (the size setup runs once): a new window shape needs a restart. DLSS finds
// its inputs at any size (vk_dlss.h "Any render size").
// SHADPS4_RENDER_HEIGHT (1.1, user 2026-10-20: "720p or 480p internal" for weaker cards): the
// height the game renders at instead of 1080 (360 - 2160; the width follows the shape). Every
// buffer, the HUD and the camera follow as for a shape; the presenter (FSR) or DLSS scales the
// picture to the window. E.g. 720 = 1280x720 (0.44x the pixels), 480 = 856x480 (0.2x), 1440 =
// 2560x1440 (supersampling). 96 - 4320 lines (8K; at most 8K's pixel count in any shape).
// Screen percentage (1.1, user 10-20 ~20:15; the DLSS panel's slider, saved in
// user/render-size.ini, or SHADPS4_SCREEN_PERCENT=NN): the render size as 10-200% of the
// screen in the render shape (the monitor's largest box of that shape), e.g. 67% of 2560x1440 =
// 1712x968 (DLSS Quality), 150% = 3840x2160 (DLAA at 4K, the presenter scales it down).
// SHADPS4_RENDER_HEIGHT wins over both; with neither the game keeps its 1080 lines.
static void ApplyAspectRatio() {
    const char* env = std::getenv("SHADPS4_ASPECT");
    const std::string mode = env ? env : "auto";
    constexpr double base = 16.0 / 9.0;
    double aspect = 0.0;
    if (mode == "off" || mode == "0") {
        aspect = base;
    } else if (const auto colon = mode.find(':'); colon != std::string::npos) {
        const double w = std::atof(mode.substr(0, colon).c_str());
        const double h = std::atof(mode.substr(colon + 1).c_str());
        aspect = h > 0.0 ? w / h : 0.0;
    } else if (g_screen_w > 0 && g_screen_h > 0) {
        aspect = double(g_screen_w) / double(g_screen_h);
    }
    // 3%: a "1280x720" window reports a client area like 1279x724 (measured, s1427).
    if (aspect <= 0.0 || std::abs(aspect - base) < 0.03 * base) {
        aspect = base; // 16:9 or unknown
    }
    u32 h0 = 1080;
    if (const char* rh = std::getenv("SHADPS4_RENDER_HEIGHT"); rh && std::atoi(rh) > 0) {
        h0 = CapPixels(std::clamp(Round8(std::atof(rh)), MinLines, MaxLines), aspect);
        g_height_from_flags = true;
    } else {
        const char* pe = std::getenv("SHADPS4_SCREEN_PERCENT");
        const float percent = pe && std::atof(pe) > 0.0
                                  ? std::clamp(float(std::atof(pe)), 10.0f, 200.0f)
                                  : SavedScreenPercent();
        if (percent > 0.0f) {
            h0 = BaseHeightForPercent(percent, aspect);
            g_percent_now = percent;
        }
    }
    g_aspect = aspect;
    g_h0 = h0;
    // The boot size M (live changes, a percentage chosen, SHADPS4_LIVE_RES not 0, normal mode -
    // the Pro toggle loads other immediates): at least 100% of the screen, so the slider is live
    // from its lowest value up to the screen's size (above: after a restart).
    const char* live_env = std::getenv("SHADPS4_LIVE_RES");
    // User 10-20 ~23:40: live from the first start, no restart for any size - the game boots at
    // the largest size (200% of the screen within the 3200x1800-pixel headroom, or the pick when
    // that is larger, up to 4K's pixels) and the first frames
    // switch it to the pick, or to its own 1080 lines without one. Only the loading frames draw
    // at the big size; the purge then frees those targets (lower-end PCs keep nothing big).
    // SHADPS4_LIVE_RES=0: boot at the pick (no live changes), SHADPS4_RENDER_HEIGHT too.
    const bool live = !g_height_from_flags && !(live_env && live_env[0] == '0') &&
                      !EmulatorSettings.IsNeo();
    const u32 headroom = CapPixels(BaseHeightForPercent(200.0f, aspect), aspect, HeadroomPixels);
    const u32 hm = live ? std::max(h0, headroom) : h0;
    if (aspect == base && h0 == 1080 && hm == 1080) {
        return; // 16:9 at 1080: nothing to do
    }
    const bool narrow = aspect < base - 1e-6;
    // Narrow screens (not crop): the camera keeps the 16:9 horizontal view and gains picture above
    // and below. The projection builder 0x800581c90 computes P00 = 2 / (r - l) / tanX and
    // P11 = 2 / (t - b) / tanY from ONE load of 2.0 (0x581d10, used by those two divisions only);
    // it is pointed at 2 / k (k = (16/9) / aspect) placed in the 12 padding bytes before the
    // function (0x581c84, after a ret: never executed). With the camera aspect = the screen's
    // (screen space below), tanX x k = the 16:9 tanX, tanY x k = the taller view.
    const bool add_vertical = narrow && mode != "crop";
    u8 two_load[8] = {0xc5, 0xfa, 0x10, 0x1d};
    std::memcpy(two_load + 4, &g_layout->two_disp, 4);
    static constexpr u8 pad[4] = {0x90, 0x90, 0x90, 0x90};
    if (add_vertical &&
        (std::memcmp(reinterpret_cast<const void*>(g_eboot_address + g_layout->two_load), two_load,
                     8) != 0 ||
         std::memcmp(reinterpret_cast<const void*>(g_eboot_address + g_layout->pad), pad, 4) != 0)) {
        LOG_ERROR(Loader, "SHADPS4_ASPECT: refused, the projection builder differs ({})",
                  g_layout->name);
        return;
    }
    // Multiples of 8; wide: the height h0 (1080), narrow: its 16:9 width (1920); at most 32:9
    // (3840 x 1080) and 1:1 (1920 x 1920).
    const RenderSize size = SizeAt(h0, aspect); // S: drawn
    const RenderSize max = SizeAt(hm, aspect);  // M: allocated (= S without live changes)
    const u32 w = size.width;
    const u32 h = size.height;
    // The 2D screen space (HUD layout; its ratio = the camera aspect) stays at the 1080-line scale
    // when the render height differs (s1440 rh720: at 1280x720 screen space the minimap drew ~1.2x
    // too large and the skyline strip above it broke - some HUD parts are sized in 1080p pixels).
    // SHADPS4_RENDER_HUD=native keeps the screen space at the render size (that look).
    const char* hud_env = std::getenv("SHADPS4_RENDER_HUD");
    const bool hud_native = hud_env && std::string{hud_env} == "native";
    double hud_scale = hud_native ? 1.0 : 1080.0 / h0;
    // SHADPS4_HUD_SPACE_SCALE (test, 10-20): scales the 2D space at the same shape (the camera
    // aspect = its ratio stays) - how the HUD's size follows the space on narrow screens.
    if (const char* v = std::getenv("SHADPS4_HUD_SPACE_SCALE"); v && std::atof(v) > 0.2) {
        hud_scale *= std::atof(v);
    }
    const u32 sw = u32(std::lround(w * hud_scale));
    const u32 sh = u32(std::lround(h * hud_scale));
    // FUN_8001eff00 (render, output, 2D screen space) and FUN_8001f0be0 (the mode toggle), in
    // g_layout->size_imms' order; byte-compared before any write.
    const auto& offsets = g_layout->size_imms;
    for (size_t i = 0; i < std::size(offsets); ++i) {
        const u32 original = i % 2 == 0 ? 0x780 : 0x438; // width 1920, height 1080
        u32 live;
        std::memcpy(&live, reinterpret_cast<const void*>(g_eboot_address + offsets[i]), 4);
        if (live != original) {
            LOG_ERROR(Loader, "SHADPS4_ASPECT: refused, eboot+{:#x} holds {:#x} (not {:#x})",
                      offsets[i], live, original);
            return;
        }
    }
    // The size setup (render, output) allocates at M; the mode toggle loads S.
    for (size_t i = 0; i < std::size(offsets); ++i) {
        const bool width = i % 2 == 0;
        const bool screen_space = i == 4 || i == 5;
        const bool toggle = i >= 6;
        const u32 value = screen_space ? (width ? sw : sh)
                          : toggle     ? (width ? w : h)
                                       : (width ? max.width : max.height);
        std::memcpy(reinterpret_cast<void*>(g_eboot_address + offsets[i]), &value, 4);
    }
    g_render_w = w;
    g_render_h = h;
    // Until the switch is taken the game draws at M (the purge then drops M's targets).
    g_now_size.store((u64(max.width) << 32) | max.height, std::memory_order_release);
    g_max = max;
    g_live_ok = live;
    ApplyLightListCapacity(max);
    ApplyMeteringCapacity(max);
    if (max.width != w || max.height != h) {
        // The renderer's init zeroes the mode bytes after these patches (study d2): the first
        // sceGnmSubmitDone switches the game from M to S.
        g_live_pending.store((u64(w) << 32) | h, std::memory_order_release);
    }
    if (add_vertical) {
        const float two_over_k = float(2.0 / (base / aspect));
        std::memcpy(reinterpret_cast<void*>(g_eboot_address + g_layout->pad), &two_over_k, 4);
        // rip-relative from the next instruction
        const s32 disp = s32(s64(g_layout->pad) - s64(g_layout->two_load + 8));
        std::memcpy(reinterpret_cast<void*>(g_eboot_address + g_layout->two_load + 4), &disp, 4);
    }
    ApplyMovieFit(sw, sh);
    LOG_INFO(Loader, "SHADPS4_ASPECT: the game renders {}x{} (aspect {:.3f}, screen {}x{}, "
             "monitor {}x{}, {}, render height {}{}, HUD space {}x{}{}{})", w, h, aspect,
             g_screen_w, g_screen_h, g_display_w, g_display_h, mode, h0,
             g_height_from_flags    ? std::string{" from SHADPS4_RENDER_HEIGHT"}
             : g_percent_now > 0.0f ? fmt::format(" = {:.0f}% of the screen", g_percent_now)
                                    : std::string{},
             sw, sh,
             add_vertical ? ", 16:9 width kept, more picture above / below" : "",
             live ? fmt::format(", live changes up to {}x{}", max.width, max.height)
                  : std::string{});
}

bool LiveResizeAvailable() {
    return g_live_ok;
}

RenderSize LiveMaxSize() {
    return g_max;
}

bool RequestScreenPercent(float percent) {
    if (!g_live_ok || percent <= 0.0f) {
        return false;
    }
    const RenderSize s = SizeAt(BaseHeightForPercent(percent, g_aspect), g_aspect);
    if (s.width > g_max.width || s.height > g_max.height) {
        return false; // more than the boot allocation: after a restart
    }
    g_live_pending.store((u64(s.width) << 32) | s.height, std::memory_order_release);
    return true;
}

float ToggleNativePercent() {
    if (!g_live_ok) {
        return 0.0f;
    }
    // No pick saved: the game's own 1080 lines as a percentage.
    const float saved = SavedScreenPercent();
    const float pick = saved > 0.0f ? saved : float(100.0 * 1080.0 / FullBaseHeight(g_aspect));
    const bool at_native = std::abs(CurrentScreenPercent() - 100.0f) < 0.5f;
    const float to = at_native ? pick : 100.0f;
    return RequestScreenPercent(to) ? to : 0.0f;
}

bool TakeLivePurge(u64 sizes[2]) {
    const u64 old_size = g_purge_old.exchange(0, std::memory_order_acq_rel);
    if (old_size == 0) {
        return false;
    }
    sizes[0] = old_size;
    sizes[1] = g_purge_new.load(std::memory_order_relaxed);
    g_purge_last_old = old_size;
    return true;
}

void RequeueLivePurge() {
    // The boot switch lands during the loading screens, where the new size's targets appear
    // only when the world draws: up to 3000 frames. A newer change replaces it anyway.
    if (g_purge_tries.fetch_add(1, std::memory_order_relaxed) < 3000) {
        u64 expected = 0;
        g_purge_old.compare_exchange_strong(expected, g_purge_last_old,
                                            std::memory_order_acq_rel);
    }
}

bool InLiveResizeWindow() {
    const s64 at = g_live_ms.load(std::memory_order_relaxed);
    return at != 0 && NowMs() - at < 3000;
}

bool IsRenderShaped(u32 width, u32 height) {
    if (height == 0) {
        return false;
    }
    return std::abs(double(width) / height - g_aspect) < 0.03 * g_aspect;
}

s64 MsSinceLiveResize() {
    const s64 at = g_live_ms.load(std::memory_order_relaxed);
    return at == 0 ? INT64_MAX : NowMs() - at;
}

namespace {

/// Writes S into the mode toggle's two imm32 (01.07: 0x1f0d3e esi = width, 0x1f0d4d edi = height).
bool WriteToggleSize(u32 w, u32 h) {
    const u64 toggle_w = g_layout->size_imms[6];
    const u64 toggle_h = g_layout->size_imms[7];
    u8* const imm_w = reinterpret_cast<u8*>(g_eboot_address + toggle_w);
    u8* const imm_h = reinterpret_cast<u8*>(g_eboot_address + toggle_h);
    const size_t span = toggle_h + 4 - toggle_w; // within a page on both layouts
#ifdef _WIN32
    DWORD old = 0;
    if (!VirtualProtect(imm_w, span, PAGE_EXECUTE_READWRITE, &old)) {
        LOG_ERROR(Loader, "Live render size: VirtualProtect failed ({})", GetLastError());
        return false;
    }
#endif
    std::memcpy(imm_w, &w, 4);
    std::memcpy(imm_h, &h, 4);
#ifdef _WIN32
    VirtualProtect(imm_w, span, old, &old);
    FlushInstructionCache(GetCurrentProcess(), imm_w, span);
#endif
    return true;
}

// The trigger: the stored colour option (renderer +0xdf0f06, low byte) made to differ from the
// wanted one (+0xdf0f11): the toggle's compare at 0x1f0d0a fails, its change block re-sizes
// everything to S and stores the right byte back (study sec. 3, the quietest of the three).
volatile u8* StoredOption() {
    return reinterpret_cast<volatile u8*>(g_eboot_address + g_layout->renderer +
                                          RendererStoredOption);
}
u8 WantedOption() {
    return *reinterpret_cast<const volatile u8*>(g_eboot_address + g_layout->renderer +
                                                 RendererWantedOption);
}

} // namespace

// DLSS jitter (1.1, user 10-21; notes/study-jitter-2026-10-20.md): the game's TAA moves the
// camera between 2 sub-pixel positions (FUN_800472810: +-0.5 / the render size in NDC = +-0.25
// render px, by frame parity). DLSS wants many: with DLSS on, five guarded edits point the
// game's two numerator loads at a slot the emulator writes each frame (a Halton(2,3) point),
// take the parity-0 path always, and point the SMAA subsample-index loads (which land in the
// blend-weights CB at +48, where DlssResolve reads the phase) at the slot's (jx, jy, tag, 0) -
// DLSS gets the exact offset of the frame it resolves. DLSS off: the original bytes come back
// (the game's own T2x needs its 2-phase pattern and matching SMAA index). Pro mode has no jitter.
namespace {

struct CodeEdit {
    u64 offset;    // eboot offset of the bytes changed
    u8 size;       // 4 = a rip-relative disp32, 2 = the jz
    u64 next_insn; // for disp32: the eboot offset of the next instruction
    u32 slot_off;  // for disp32: the slot byte the load reads
    u32 original;  // the original disp32 / 2 bytes (little endian)
};
// Measured bytes (study sec. 3): A1 y numerator, A2 x numerator, A3 jz -> nop, B1 / B2 SMAA index
// (01.07: A1 0x47297a, A2 0x472982, A3 0x472998, B1 0x472240, B2 0x47224a).
std::array<CodeEdit, 5> JitterEdits() {
    const u64 a = g_layout->jitter_a;
    const u64 b = g_layout->jitter_b;
    const auto& d = g_layout->jitter_disp;
    return {{
        {a + 0x4, 4, a + 0x8, 4, d[0]},
        {a + 0xc, 4, a + 0x10, 0, d[1]},
        {a + 0x22, 2, 0, 0, 0x0674},
        {b + 0x4, 4, b + 0x8, 8, d[2]},
        {b + 0xe, 4, b + 0x12, 8, d[3]},
    }};
}
// Whole-instruction guards (the opcode bytes before each disp32).
constexpr u8 GuardA1[] = {0xc5, 0xfa, 0x10, 0x05};
constexpr u8 GuardA2[] = {0xc5, 0xfa, 0x10, 0x15};
constexpr u8 GuardB[] = {0xc5, 0xfa, 0x6f, 0x05};

bool g_jitter_patched = false;
bool g_jitter_refused = false;
u32 g_jitter_k = 0;

bool WriteCodeBytes(u64 offset, const void* data, size_t size) {
    u8* const p = reinterpret_cast<u8*>(g_eboot_address + offset);
#ifdef _WIN32
    DWORD old = 0;
    if (!VirtualProtect(p, size, PAGE_EXECUTE_READWRITE, &old)) {
        return false;
    }
#endif
    std::memcpy(p, data, size);
#ifdef _WIN32
    VirtualProtect(p, size, old, &old);
    FlushInstructionCache(GetCurrentProcess(), p, size);
#endif
    return true;
}

bool JitterBytesOriginal() {
    const auto at = [](u64 off) { return reinterpret_cast<const u8*>(g_eboot_address + off); };
    const u64 a = g_layout->jitter_a;
    const u64 b = g_layout->jitter_b;
    if (std::memcmp(at(a), GuardA1, 4) != 0 || std::memcmp(at(a + 0x8), GuardA2, 4) != 0 ||
        std::memcmp(at(b), GuardB, 4) != 0 || std::memcmp(at(b + 0xa), GuardB, 4) != 0) {
        return false;
    }
    for (const auto& e : JitterEdits()) {
        u32 live = 0;
        std::memcpy(&live, at(e.offset), e.size);
        if (live != e.original) {
            return false;
        }
    }
    return true;
}

void SetJitterPatch(bool on) {
    if (on == g_jitter_patched) {
        return;
    }
    if (on && (!g_layout || g_eboot_scratch == 0 || !JitterBytesOriginal())) {
        if (!g_jitter_refused) {
            g_jitter_refused = true;
            LOG_ERROR(Loader, "DLSS jitter: refused (eboot {} bytes differ or no slot)",
                      g_layout ? g_layout->name : "of no known layout");
        }
        return;
    }
    for (const auto& e : JitterEdits()) {
        u32 value = e.original;
        if (on) {
            value = e.size == 2 ? 0x9066 // 66 90: a 2-byte nop
                                : u32(s64(g_eboot_scratch + e.slot_off) -
                                      s64(g_eboot_address + e.next_insn));
        }
        if (!WriteCodeBytes(e.offset, &value, e.size)) {
            LOG_ERROR(Loader, "DLSS jitter: VirtualProtect failed at {:#x}", e.offset);
            return;
        }
    }
    g_jitter_patched = on;
    LOG_INFO(Loader, "DLSS jitter: {} (slot {:#x})",
             on ? "Halton sequence, DLSS sets the sub-pixel position" : "the game's own 2 phases",
             g_eboot_scratch);
}

float Halton(u32 index, u32 base) {
    float f = 1.0f, r = 0.0f;
    while (index > 0) {
        f /= float(base);
        r += f * float(index % base);
        index /= base;
    }
    return r;
}

/// SHADPS4_HALTON_PROBE=1 (diagnostic, 10-21; study-delsin-mv-ssr sec. 1.6): with the Halton jitter
/// running, 60 s after its first frame, for 4 frames: every view-projection-like 4x4 (16-byte
/// aligned rows) in the main camera object ([0x807f2d2b8], 0x2000 bytes) and in the ranges of
/// SHADPS4_HALTON_PROBE_SCAN=<hex addr>:<hex size>,... with its jitter (A, B) = the projection's
/// row-2 terms read back from the matrix, next to this frame's step k and steps k-1 / k-2, so the
/// field that keeps a 2-frame-old projection shows up.
void HaltonProbe(RenderSize now) {
    static const bool on = [] {
        const char* v = std::getenv("SHADPS4_HALTON_PROBE");
        return v && v[0] == '1';
    }();
    if (!on || !g_layout || now.width == 0 || now.height == 0) {
        return;
    }
    static const s64 start = NowMs();
    static int done = 0;
    if (done >= 4 || NowMs() - start < 60000) {
        return;
    }
    ++done;
    const auto step_ab = [&](u32 k) {
        const float jx = Halton(k + 1, 2) - 0.5f;
        const float jy = Halton(k + 1, 3) - 0.5f;
        return std::pair<double, double>{2.0 * jx / now.width, -2.0 * jy / now.height};
    };
    // The phase count as JitterFrame derives it (k wraps there).
    const RenderSize ref = Reference(g_aspect);
    const double ratio = std::max(1.0, double(ref.width) / now.width);
    const u32 phases = std::clamp(u32(std::lround(8.0 * ratio * ratio)), 8u, 64u);
    const auto [a0, b0] = step_ab(g_jitter_k);
    const auto [a1, b1] = step_ab((g_jitter_k + phases - 1) % phases);
    const auto [a2, b2] = step_ab((g_jitter_k + phases - 2) % phases);
    LOG_INFO(Loader,
             "HaltonProbe frame {}: k={} of {} | step k A,B=({:+.4e},{:+.4e}) k-1 ({:+.4e},{:+.4e}) "
             "k-2 ({:+.4e},{:+.4e}) (the next frame draws step k)",
             done, g_jitter_k, phases, a0, b0, a1, b1, a2, b2);
    const auto scan = [&](VAddr base, u64 size, const char* what) {
        auto* const memory = Core::Memory::Instance();
        if (!memory || !memory->IsMappedRange(base, size)) {
            LOG_INFO(Loader, "HaltonProbe   {} ({:#x}, {:#x} bytes): not mapped", what, base, size);
            return;
        }
        for (u64 off = 0; off + 64 <= size; off += 16) {
            float m[16];
            std::memcpy(m, reinterpret_cast<const void*>(base + off), sizeof(m));
            const double c3[3] = {m[3], m[7], m[11]};
            const double n3 = c3[0] * c3[0] + c3[1] * c3[1] + c3[2] * c3[2];
            if (!(n3 > 0.8 && n3 < 1.2)) {
                continue;
            }
            const double a = (m[0] * c3[0] + m[4] * c3[1] + m[8] * c3[2]) / n3;
            const double b = (m[1] * c3[0] + m[5] * c3[1] + m[9] * c3[2]) / n3;
            const double p00 = std::sqrt(std::pow(m[0] - a * c3[0], 2) +
                                         std::pow(m[4] - a * c3[1], 2) +
                                         std::pow(m[8] - a * c3[2], 2));
            if (!(std::abs(a) < 0.01 && std::abs(b) < 0.01 && p00 > 0.3 && p00 < 6.0)) {
                continue;
            }
            LOG_INFO(Loader, "HaltonProbe   {}+{:#x} ({:#x}): A,B=({:+.4e},{:+.4e}) P00 {:.4f}", what,
                     off, base + off, a, b, p00);
        }
    };
    const VAddr cam = *reinterpret_cast<const u64*>(g_eboot_address + g_layout->camera_ptr);
    if (cam != 0) {
        scan(cam, 0x2000, "camera");
    }
    if (const char* v = std::getenv("SHADPS4_HALTON_PROBE_SCAN")) {
        std::stringstream ss{v};
        std::string item;
        while (std::getline(ss, item, ',')) {
            const auto colon = item.find(':');
            if (colon == std::string::npos) {
                continue;
            }
            const VAddr base = std::strtoull(item.substr(0, colon).c_str(), nullptr, 16);
            const u64 size = std::strtoull(item.substr(colon + 1).c_str(), nullptr, 16);
            scan(base, size, item.c_str());
        }
    }
}

/// The Halton wobble fix (10-21; study-delsin-mv-ssr sec. 1.6-1.7, probe s1539): the main camera
/// ([0x807f2d2b8]) keeps its jittered view-projection history as (VP, unjittered VP) pairs:
/// +0x200 this frame, +0x380 one frame back, +0x300 two back; each frame start shifts +0x380 ->
/// +0x300 and +0x200 -> +0x380 (measured). The moving objects' velocity pass reads +0x300 as the
/// previous VP: with the game's 2 jitter phases its jitter always equals the current one and
/// cancels; with Halton every mover's motion vector was off by j(N) - j(N-2) (Delsin's wobble).
/// Here, between frames, the entry about to become +0x300 gets the NEXT frame's jitter: for
/// M = V x P (row vectors) column 0 = P00 x V.col0 + A x V.col2 and column 3 = V.col2, so
/// col0 += dA x col3 (all 4 rows) moves exactly the jitter term A; col1 likewise for B. The view
/// (camera motion) is untouched. SHADPS4_HALTON_REBASE=0 off.
void RebaseHistoryJitter(RenderSize now, float jx, float jy) {
    static const bool on = [] {
        const char* v = std::getenv("SHADPS4_HALTON_REBASE");
        return !(v && v[0] == '0');
    }();
    if (!on || !g_layout || now.width == 0 || now.height == 0) {
        return;
    }
    const VAddr cam = *reinterpret_cast<const u64*>(g_eboot_address + g_layout->camera_ptr);
    // A stale or not-yet-set camera pointer (loads, menus) must not fault Game:Main.
    auto* const memory = Core::Memory::Instance();
    if (cam == 0 || !memory || !memory->IsMappedRange(cam + 0x380, 0x80)) {
        return;
    }
    // The jitter terms of a VP read back from the matrix (V's rotation is orthonormal).
    const auto terms = [](const float* m, double& a, double& b) {
        const double n3 = double(m[3]) * m[3] + double(m[7]) * m[7] + double(m[11]) * m[11];
        if (!(n3 > 0.8 && n3 < 1.2)) {
            return false;
        }
        a = (double(m[0]) * m[3] + double(m[4]) * m[7] + double(m[8]) * m[11]) / n3;
        b = (double(m[1]) * m[3] + double(m[5]) * m[7] + double(m[9]) * m[11]) / n3;
        return std::abs(a) < 0.01 && std::abs(b) < 0.01;
    };
    float* const vp = reinterpret_cast<float*>(cam + 0x380);
    const float* const plain = reinterpret_cast<const float*>(cam + 0x3c0);
    double a = 0.0, b = 0.0, pa = 0.0, pb = 0.0;
    // Layout check every frame: a jittered VP at +0x380 and its unjittered twin at +0x3c0 (another
    // camera class or a cutscene camera fails it and is left alone).
    if (!terms(vp, a, b) || !terms(plain, pa, pb) || std::abs(pa) > 1e-6 || std::abs(pb) > 1e-6) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            LOG_WARNING(Loader, "Halton rebase: the camera history layout differs, left alone");
        }
        return;
    }
    static bool applied_logged = false;
    if (!applied_logged) {
        applied_logged = true;
        LOG_INFO(Loader, "Halton rebase: camera history at {:#x} ({})", cam, g_layout->name);
    }
    // The game's window offset is (X / W, Y / H) with X = -2 jx, Y = 2 jy: A = 2 jx / W, B = -2 jy / H.
    const double da = 2.0 * jx / now.width - a;
    const double db = -2.0 * jy / now.height - b;
    for (int r = 0; r < 4; ++r) {
        vp[r * 4 + 0] = float(vp[r * 4 + 0] + da * vp[r * 4 + 3]);
        vp[r * 4 + 1] = float(vp[r * 4 + 1] + db * vp[r * 4 + 3]);
    }
}

/// Game:Main, inside sceGnmSubmitDone (the render jobs of this frame joined; the next frame's
/// jitter function and SMAA setup both run before the next call - study sec. 3).
void JitterFrame() {
    // On by default again (10-21: the wobble - movers' motion vectors kept a 2-frame-old jitter -
    // is fixed by RebaseHistoryJitter); SHADPS4_DLSS_HALTON=0 = the game's own 2 phases.
    // SHADPS4_DLSS_HALTON=force (test): on without DLSS too - RenderDoc turns DLSS off, and the
    // motion-vector checks need a capture with the Halton jitter running.
    static const int mode = [] {
        const char* v = std::getenv("SHADPS4_DLSS_HALTON");
        return !v ? 1 : v[0] == '0' ? 0 : v[0] == 'f' ? 2 : 1;
    }();
    const bool dlss_on = Vulkan::Dlss::Enabled() &&
                         Vulkan::Dlss::g_ready.load(std::memory_order_relaxed) &&
                         Vulkan::Dlss::g_user_on.load(std::memory_order_relaxed);
    const bool want = mode != 0 && !EmulatorSettings.IsNeo() && (dlss_on || mode == 2);
    SetJitterPatch(want);
    if (!g_jitter_patched) {
        return;
    }
    // DLSS's guide: 8 x (output / render)^2 phases (8 at DLAA, 72 at 33%), here 8..64.
    const RenderSize now = CurrentRenderSize();
    const RenderSize ref = Reference(g_aspect);
    const double ratio = now.width > 0 ? std::max(1.0, double(ref.width) / now.width) : 1.0;
    const u32 phases = std::clamp(u32(std::lround(8.0 * ratio * ratio)), 8u, 64u);
    g_jitter_k = (g_jitter_k + 1) % phases;
    const float jx = Halton(g_jitter_k + 1, 2) - 0.5f; // render px, (-0.5, 0.5)
    const float jy = Halton(g_jitter_k + 1, 3) - 0.5f;
    // The image moves by (-X/2, +Y/2) px for numerators (X, Y) (study sec. 2).
    struct Slot {
        float x, y;
        float cb[4]; // (jx, jy, tag, 0): the blend-weights CB floats 12-15
    } slot{-2.0f * jx, 2.0f * jy, {jx, jy, 100.0f + float(g_jitter_k), 0.0f}};
    std::memcpy(reinterpret_cast<void*>(g_eboot_scratch), &slot.x, 8);
    std::memcpy(reinterpret_cast<void*>(g_eboot_scratch + 8), slot.cb, 16);
    HaltonProbe(now);
    RebaseHistoryJitter(now, jx, jy);
}

} // namespace

/// SHADPS4_LIVE_RES_TEST=50,67,100,... (test): from 40 s after the first frame, every 8 s
/// (SHADPS4_LIVE_RES_TEST_MS) the next percentage of the list is requested, as the panel's slider
/// would; the list repeats.
static void LiveResTestStep() {
    static const std::vector<float> list = [] {
        std::vector<float> out;
        if (const char* v = std::getenv("SHADPS4_LIVE_RES_TEST")) {
            std::stringstream ss{v};
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (std::atof(item.c_str()) > 0.0) {
                    out.push_back(float(std::atof(item.c_str())));
                }
            }
        }
        return out;
    }();
    if (list.empty()) {
        return;
    }
    static const s64 start = NowMs();
    static const s64 step = [] {
        const char* v = std::getenv("SHADPS4_LIVE_RES_TEST_MS");
        return v && std::atoi(v) > 0 ? s64(std::atoi(v)) : s64(8000);
    }();
    static const s64 delay = [] {
        const char* v = std::getenv("SHADPS4_LIVE_RES_TEST_DELAY_MS");
        return v && std::atoi(v) > 0 ? s64(std::atoi(v)) : s64(40000);
    }();
    static size_t next = 0;
    const s64 due = start + delay + s64(next) * step;
    if (NowMs() >= due) {
        const float p = list[next % list.size()];
        LOG_INFO(Loader, "SHADPS4_LIVE_RES_TEST: {}% -> {}", p,
                 RequestScreenPercent(p) ? "queued" : "refused");
        ++next;
    }
}

void OnSubmitDone() {
    JitterFrame();
    LiveResTestStep();
    // Game:Main is inside this HLE call, so its frame begin (FUN_8001f0be0, the toggle) cannot run
    // meanwhile; job threads never run it (study sec. 3).
    if (g_live_check != 0) {
        // The last poke: taken when the game's size table holds S (slot 0 = the output size at
        // renderer +0xdf0f14 = 0x801d0cbe4, study sec. 1; its change block wrote it).
        u32 table[2];
        std::memcpy(table,
                    reinterpret_cast<const void*>(g_eboot_address + g_layout->renderer +
                                                  RendererSizeTable),
                    8);
        if (table[0] == u32(g_live_check >> 32) && table[1] == u32(g_live_check)) {
            const u64 old_size = g_now_size.exchange(g_live_check, std::memory_order_acq_rel);
            g_live_ms.store(NowMs(), std::memory_order_relaxed);
            // This frame drew at S from its start (the change block runs at frame begin): at the
            // end of the back's first frame at S it drops the old size's unused targets.
            if (old_size != g_live_check) {
                g_purge_new.store(g_live_check, std::memory_order_relaxed);
                g_purge_tries.store(0, std::memory_order_relaxed);
                g_purge_old.store(old_size, std::memory_order_release);
            }
            LOG_INFO(Loader, "Live render size: the game draws {}x{} ({:.0f}% of the screen; boot "
                             "allocation {}x{}) after {} poke(s)",
                     u32(g_live_check >> 32), u32(g_live_check), CurrentScreenPercent(),
                     g_max.width, g_max.height, g_live_tries);
            g_live_check = 0;
        } else if (g_live_tries >= 2400) {
            // s1494: the boot request (M -> S) is not taken through the logos and loading
            // screens (the mode check runs in game); 2400 frames is well past them.
            LOG_ERROR(Loader, "Live render size: the game did not take {}x{} after {} frames "
                              "(its table: {}x{})",
                      u32(g_live_check >> 32), u32(g_live_check), g_live_tries, table[0], table[1]);
            g_live_check = 0;
        } else {
            ++g_live_tries;
            *StoredOption() = WantedOption() != 0 ? 0 : 1; // the renderer was not up yet
            return;
        }
    }
    const u64 pending = g_live_pending.exchange(0, std::memory_order_acq_rel);
    if (pending == 0) {
        return;
    }
    if (!WriteToggleSize(u32(pending >> 32), u32(pending))) {
        g_live_pending.store(pending, std::memory_order_release); // next frame
        return;
    }
    // Window for the texture cache and DLSS from the poke on (the back may still be recording
    // this frame: stragglers at the old size perfect-match their old images).
    g_live_ms.store(NowMs(), std::memory_order_relaxed);
    *StoredOption() = WantedOption() != 0 ? 0 : 1;
    g_live_check = pending;
    g_live_tries = 1;
}

// SHADPS4_HUD (1.1, user 2026-10-20): where the HUD is drawn, to taste - e.g. a 16:9 HUD on a 21:9
// picture. The game places its HUD inside the console's "display safe area": FUN_8001eff00 asks
// sceSystemServiceGetDisplaySafeAreaInfo for a ratio r (clamped to 0.85 - 1.0) and sets the HUD
// margins 0x801d0ccc8 / cccc = (W - W r) / 2 and (H - H r) / 2 (at least 16 px). Here r = the HUD
// width / the render width, the 0.85 floor is lifted and the vertical margin keeps r = 1, in all
// three copies of that code: the HUD narrows sideways only.
//   default 16:9 on a widened picture, else full; full = the whole width; 16:9 / 21:9 / any W:H = that shape, centred; NN% = NN percent
//   of the width. Decided at boot, as SHADPS4_ASPECT.
static void ApplyHudWidth() {
    // Default: 16:9 on a widened picture (the HUD art is made for 16:9; at full width the minimap
    // ends partly off the right edge, s1427), the full width otherwise.
    const char* env = std::getenv("SHADPS4_HUD");
    const std::string v = env ? env : (double(g_render_w) * 9.0 > double(g_render_h) * 16.0 * 1.03 ? "16:9" : "full");
    if (v == "full") {
        return;
    }
    double width = 0.0;
    if (const auto colon = v.find(':'); colon != std::string::npos) {
        const double a = std::atof(v.substr(0, colon).c_str());
        const double b = std::atof(v.substr(colon + 1).c_str());
        width = b > 0.0 ? double(g_render_h) * a / b : 0.0;
    } else if (!v.empty() && v.back() == '%') {
        width = double(g_render_w) * std::atof(v.c_str()) / 100.0;
    }
    const float ratio = float(std::clamp(width / double(g_render_w), 0.2, 1.0));
    if (width <= 0.0 || ratio >= 0.999f) {
        LOG_INFO(Loader, "SHADPS4_HUD={}: the full width ({}x{})", v, g_render_w, g_render_h);
        return;
    }
    // The margin code exists three times (the size setup and two refreshes): per copy the floor
    // load vmovss disp(%rip), %xmm2 -> vxorps xmm2, xmm2, xmm2 + 4-byte nop (followed by
    // vminss / vcmpltss), and vmulss %xmm0, %xmm2, %xmm0 (H * r) -> vmovaps %xmm2, %xmm0 (H).
    const auto& floors = g_layout->hud_floor; // 01.07: 0x1f042f, 0x1f2aae, 0x1f2bae
    const auto& ymuls = g_layout->hud_ymul;   // 01.07: 0x1f0471, 0x1f2afb, 0x1f2bfb
    static constexpr u8 floor_head[4] = {0xc5, 0xfa, 0x10, 0x15};
    static constexpr u8 floor_tail[9] = {0xc5, 0xf2, 0x5d, 0xc8, 0xc5, 0xfa, 0xc2, 0xc2, 0x01};
    static constexpr u8 floor_new[8] = {0xc5, 0xe8, 0x57, 0xd2, 0x0f, 0x1f, 0x40, 0x00};
    static constexpr u8 ymul_old[4] = {0xc5, 0xea, 0x59, 0xc0};
    static constexpr u8 ymul_new[4] = {0xc5, 0xf8, 0x28, 0xc2};
    const auto at = [](u64 offset) { return reinterpret_cast<u8*>(g_eboot_address + offset); };
    for (int i = 0; i < 3; ++i) {
        if (std::memcmp(at(floors[i]), floor_head, 4) != 0 ||
            std::memcmp(at(floors[i] + 8), floor_tail, 9) != 0 ||
            std::memcmp(at(ymuls[i]), ymul_old, 4) != 0) {
            LOG_ERROR(Loader, "SHADPS4_HUD: refused, the safe-area code differs at copy {} ({})",
                      i, g_layout->name);
            return;
        }
    }
    for (int i = 0; i < 3; ++i) {
        std::memcpy(at(floors[i]), floor_new, 8);
        std::memcpy(at(ymuls[i]), ymul_new, 4);
    }
    g_hud_ratio = ratio;
    LOG_INFO(Loader, "SHADPS4_HUD={}: HUD {} px wide of {} (safe-area ratio {:.3f}, sides only)", v,
             u32(width + 0.5), g_render_w, ratio);
}

// SHADPS4_PERF_MODE (1.1 prototype, user 2026-10-20: "cut back on particles, drop the render or
// LOD distance" with the game's own assets; study notes/study-performance-mode-2026-10-20.md).
// The game ships a 21-row quality ladder at 0x800da08c0 (20 bytes per row: LOD scale for its 7 LOD
// distance bands, sun-shadow cascade offset, a distance with no known reader, a screen-size cull,
// random thinning) that it climbs only when a frame overflows its GPU command memory; level 0 =
// row 0 in normal play. A preset is written INTO row 0, so the game's own path runs it (the
// harsher rows stay for overflow). Plus: the engine's low-res particle pass forced (it steps down
// only on a GPU timer that means nothing under emulation), the streamer's high-res texture slots
// (4 nearest blocks) and, in "office", the ambient pedestrian ratio.
//   low:    LOD 0.7, one sun-shadow cascade less (= the game's row 3), particles at half
//           resolution, 2 high-res texture blocks.
//   office: LOD 0.5, one cascade less (row 5's values, its unknown distance column kept at row
//           0's), particles at quarter resolution, no high-res texture blocks, half the pedestrians.
// Decided at boot; every site byte-compared first (eboot 01.07), all or nothing.
static void ApplyPerfMode() {
    const char* env = std::getenv("SHADPS4_PERF_MODE");
    const std::string mode = env ? env : "off";
    const bool low = mode == "low" || mode == "1";
    const bool office = mode == "office" || mode == "2";
    if (!low && !office) {
        return;
    }
    struct Site {
        u64 offset;
        std::vector<u8> original;
        std::vector<u8> value;
        const char* what;
    };
    const std::vector<u8> row0 = {0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24,
                                  0x74, 0x49, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    std::vector<u8> row = row0;
    const float lod = low ? 0.7f : 0.5f;
    std::memcpy(row.data(), &lod, 4);
    const s32 cascades = -1;
    std::memcpy(row.data() + 4, &cascades, 4);
    const u8 particle_level = low ? 1 : 2; // resolution 1.0 / 0.5 / 0.25 (table 0x800aaa918)
    const u8 hires_slots = low ? 2 : 0;
    std::vector<Site> sites = {
        {0xda08c0, row0, row, "quality ladder row 0"},
        // FUN_8003ed6a0: je over the "set level" store -> 2 nops, and the stored level.
        {0x3ed6b0, {0x74, 0x0f}, {0x90, 0x90}, "particle level: always set"},
        {0x3ed6b8, {0x00, 0x00, 0x00, 0x00}, {particle_level, 0x00, 0x00, 0x00}, "particle level"},
        // FUN_8000fd890: mov r12d, 4 / cmp eax, 4 (the 4 nearest blocks get .hires.xpps).
        {0xfe9a2, {0x04}, {hires_slots}, "high-res texture slots (mov)"},
        {0xfe9b2, {0x04}, {hires_slots}, "high-res texture slots (cmp)"},
    };
    if (office) {
        // Pedestrian count per class = level count x ratio[world state] (0x40ee42): 1.0 -> 0.5.
        sites.push_back({0x40ca01, {0x00, 0x00, 0x80, 0x3f}, {0x00, 0x00, 0x00, 0x3f},
                         "pedestrian ratio 0"});
        sites.push_back({0x40c83f, {0x00, 0x00, 0x80, 0x3f}, {0x00, 0x00, 0x00, 0x3f},
                         "pedestrian ratio 1"});
    }
    for (const auto& s : sites) {
        if (std::memcmp(reinterpret_cast<const void*>(g_eboot_address + s.offset),
                        s.original.data(), s.original.size()) != 0) {
            LOG_ERROR(Loader, "SHADPS4_PERF_MODE: refused, {} at eboot+{:#x} differs (eboot 01.07 "
                              "only)", s.what, s.offset);
            return;
        }
    }
    for (const auto& s : sites) {
        std::memcpy(reinterpret_cast<void*>(g_eboot_address + s.offset), s.value.data(),
                    s.value.size());
    }
    LOG_INFO(Loader, "SHADPS4_PERF_MODE={}: LOD scale {:.1f}, one sun-shadow cascade less, "
                     "particles at {} resolution, {} high-res texture blocks{}",
             mode, lod, low ? "half" : "quarter", hires_slots,
             office ? ", half the pedestrians" : "");
}

int GameQualityLevel() {
    if (g_eboot_address == 0 || !g_layout) {
        return -1;
    }
    s32 level;
    std::memcpy(&level,
                reinterpret_cast<const void*>(g_eboot_address + g_layout->renderer +
                                              RendererQualityLevel),
                4);
    return level;
}

static void ApplyHouseSplit() {
    const char* env = std::getenv("SHADPS4_HOUSE_SPLIT");
    const char* chunk_env = std::getenv("SHADPS4_HOUSE_CHUNK");
    // =2 (diagnostic, study 11:35): FUN_800197130 -> mov rax, -1; ret - every run > 1000 items is
    // cut at its chunk (200 in passes 0x13 / 0x17); tells apart "houses in another pass / equal keys"
    // (contexts up, span down) from "the pass is walked serially" (contexts up, span flat).
    const bool all_ones = env && env[0] == '2';
    const bool mask_on = env && (env[0] == '1' || all_ones);
    const u32 chunk = chunk_env ? static_cast<u32>(std::strtoul(chunk_env, nullptr, 10)) : 0;
    if (!mask_on && chunk == 0) {
        return;
    }
    // No serial / APP_VER check (2026-10-19, as PARTICLE_LATE_READ): the byte compare is the gate.
    // FUN_800197130 (split mask by pass) and FUN_800197050 (chunk by pass), whole functions.
    static constexpr u8 mask_fn[58] = {
        0xba, 0x01, 0x00, 0x00, 0x00, 0x40, 0x88, 0xf9, 0xb8, 0x00, 0x00, 0xff, 0x7f, 0x48, 0xbe,
        0x00, 0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00, 0x48, 0xd3, 0xe2, 0x48, 0x0f, 0xa3, 0xfe,
        0x72, 0x19, 0xf7, 0xc2, 0x00, 0x44, 0x88, 0x00, 0xb9, 0x00, 0x00, 0xff, 0xff, 0x48, 0xb8,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x48, 0x0f, 0x45, 0xc1, 0xc3};
    static constexpr u8 chunk_fn[19] = {0x83, 0xcf, 0x04, 0xb8, 0xc8, 0x00, 0x00, 0x00, 0x83, 0xff,
                                        0x17, 0x74, 0x05, 0xb8, 0xe8, 0x03, 0x00, 0x00, 0xc3};
    const auto matches = [](u64 offset, const u8* bytes, size_t n) {
        const auto* live = reinterpret_cast<const u8*>(g_eboot_address + offset);
        if (std::memcmp(live, bytes, n) == 0) {
            return true;
        }
        std::string found;
        for (size_t i = 0; i < n; ++i) {
            found += fmt::format("{:02x}", live[i]);
        }
        LOG_ERROR(Loader, "SHADPS4_HOUSE_SPLIT: refused, eboot+{:#x} holds {}", offset, found);
        return false;
    };
    if (!matches(0x197130, mask_fn, sizeof(mask_fn)) ||
        !matches(0x197050, chunk_fn, sizeof(chunk_fn))) {
        return;
    }
    if (all_ones) {
        // mov rax, -1; ret at the function start (48 c7 c0 ff ff ff ff c3)
        static constexpr u8 ret_all[8] = {0x48, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff, 0xc3};
        std::memcpy(reinterpret_cast<void*>(g_eboot_address + 0x197130), ret_all, sizeof(ret_all));
    } else if (mask_on) {
        // imm64 of mov rax, 0xffff000000000000 at 0x19715d -> 0xffffffffffff0000
        static constexpr u8 mask[8] = {0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
        std::memcpy(reinterpret_cast<void*>(g_eboot_address + 0x19715d), mask, sizeof(mask));
    }
    if (chunk != 0) {
        // imm32 of mov eax, 1000 at 0x19705e
        std::memcpy(reinterpret_cast<void*>(g_eboot_address + 0x19705e), &chunk, sizeof(chunk));
    }
    LOG_INFO(Loader, "SHADPS4_HOUSE_SPLIT: split mask {} | chunk {}",
             all_ones  ? "all ones in every pass (diagnostic)"
             : mask_on ? "default 0xffffffffffff0000"
                       : "unchanged",
             chunk != 0 ? chunk : 1000);
}

void OnGameLoaded() {
    g_layout = DetectLayout();
    if (g_layout) {
        LOG_INFO(Loader, "Built-in game patches: {} layout", g_layout->name);
        if (SecondSon107Loaded()) {
            ApplyParticleLateRead();
            ApplyParticleLightOneSubmit();
            ApplyHouseSplit();
        }
        ApplyAspectRatio();
        ApplyHudWidth();
        if (SecondSon107Loaded()) {
            ApplyPerfMode();
        }
    } else {
        LOG_WARNING(Loader,
                    "This game's code is not Second Son update 01.07 or First Light 01.04 (code "
                    "hash {:#018x}, {} segments): the built-in game patches are off (screen shape "
                    "and HUD width, render size, particle and performance patches, DLSS jitter). "
                    "Install update 01.07 for them.",
                    Common::ElfInfo::Instance().EbootCodeHash(), g_eboot_segments.size());
    }
    std::filesystem::path patch_dir = Common::FS::GetUserPath(Common::FS::PathType::PatchesDir);
    if (!patch_file.empty()) {

        auto file_path = (patch_dir / patch_file).native();
        if (std::filesystem::exists(patch_file)) {
            ApplyPatchesFromXML(patch_file);
        } else {
            ApplyPatchesFromXML(file_path);
        }
    } else if (EmulatorState::GetInstance()->IsAutoPatchesLoadEnabled()) {
        for (auto const& repo : std::filesystem::directory_iterator(patch_dir)) {
            if (!repo.is_directory()) {
                continue;
            }
            std::ifstream json_file{repo.path() / "files.json"};
            nlohmann::json available_patches = nlohmann::json::parse(json_file);
            std::filesystem::path game_patch_file;
            for (auto const& [filename, serials] : available_patches.items()) {
                if (std::find(serials.begin(), serials.end(), g_game_serial) != serials.end()) {
                    game_patch_file = repo.path() / filename;
                    break;
                }
            }
            if (std::filesystem::exists(game_patch_file)) {
                ApplyPatchesFromXML(game_patch_file);
            }
        }
    }
    ApplyPendingPatches();
}

void AddPatchToQueue(const patchInfo& patchToAdd) {
    if (patches_applied) {
        PatchMemory(patchToAdd);
        return;
    }
    pending_patches.push_back(patchToAdd);
}

void ApplyPendingPatches() {
    patches_applied = true;
    for (size_t i = 0; i < pending_patches.size(); ++i) {
        const patchInfo& currentPatch = pending_patches[i];

        if (currentPatch.gameSerial != "*" && currentPatch.gameSerial != g_game_serial)
            continue;

        PatchMemory(currentPatch);
    }

    pending_patches.clear();
}

void PatchMemory(const patchInfo& patch) {
    // Send a request to modify the process memory.
    void* cheatAddress = nullptr;

    if (patch.patchMask == PatchMask::None) {
        if (patch.isOffset) {
            cheatAddress =
                reinterpret_cast<void*>(g_eboot_address + std::stoi(patch.offsetStr, 0, 16));
        } else {
            cheatAddress = reinterpret_cast<void*>(g_eboot_address +
                                                   (std::stoi(patch.offsetStr, 0, 16) - 0x400000));
        }
    }

    if (patch.patchMask == PatchMask::Mask) {
        cheatAddress = reinterpret_cast<void*>(PatternScan(patch.offsetStr) + patch.maskOffset);
    }

    if (patch.patchMask == PatchMask::Mask_Jump32) {
        int jumpSize = std::stoi(patch.sizeStr);

        constexpr int MAX_PATTERN_LENGTH = 256;
        if (jumpSize < 5) {
            LOG_ERROR(Loader, "Jump size must be at least 5 bytes");
            return;
        }
        if (jumpSize > MAX_PATTERN_LENGTH) {
            LOG_ERROR(Loader, "Jump size must be no more than {} bytes.", MAX_PATTERN_LENGTH);
            return;
        }

        // Find the base address using "Address"
        uintptr_t baseAddress = PatternScan(patch.offsetStr);
        if (baseAddress == 0) {
            LOG_ERROR(Loader, "PatternScan failed for mask_jump32 with pattern: {}",
                      patch.offsetStr);
            return;
        }
        uintptr_t patchAddress = baseAddress + patch.maskOffset;

        // Fills the original region (jumpSize bytes) with NOPs
        std::vector<u8> nopBytes(jumpSize, 0x90);
        std::memcpy(reinterpret_cast<void*>(patchAddress), nopBytes.data(), nopBytes.size());

        // Use "Target" to locate the start of the code cave
        uintptr_t jump_target = PatternScan(patch.targetStr);
        if (jump_target == 0) {
            LOG_ERROR(Loader, "PatternScan failed to Target with pattern: {}", patch.targetStr);
            return;
        }

        // Converts the Value attribute to a byte array (payload)
        std::vector<u8> payload;
        for (size_t i = 0; i < patch.valueStr.length(); i += 2) {
            std::string tempStr = patch.valueStr.substr(i, 2);
            const char* byteStr = tempStr.c_str();
            char* endPtr;
            unsigned int byteVal = std::strtoul(byteStr, &endPtr, 16);

            if (endPtr != byteStr + 2) {
                LOG_ERROR(Loader, "Invalid byte in Value: {}", patch.valueStr.substr(i, 2));
                return;
            }
            payload.push_back(static_cast<u8>(byteVal));
        }

        // Calculates the end of the code cave (where the return jump will be inserted)
        uintptr_t code_cave_end = jump_target + payload.size();

        // Write the payload to the code cave, from jump_target
        std::memcpy(reinterpret_cast<void*>(jump_target), payload.data(), payload.size());

        // Inserts the initial jump in the original region to divert to the code cave
        u8 jumpInstruction[5];
        jumpInstruction[0] = 0xE9;
        s32 relJump = static_cast<s32>(jump_target - patchAddress - 5);
        std::memcpy(&jumpInstruction[1], &relJump, sizeof(relJump));
        std::memcpy(reinterpret_cast<void*>(patchAddress), jumpInstruction,
                    sizeof(jumpInstruction));

        // Inserts jump back at the end of the code cave to resume execution after patching
        u8 jumpBack[5];
        jumpBack[0] = 0xE9;
        // Calculates the relative offset to return to the instruction immediately following the
        // overwritten region
        s32 target_return = static_cast<s32>((patchAddress + jumpSize) - (code_cave_end + 5));
        std::memcpy(&jumpBack[1], &target_return, sizeof(target_return));
        std::memcpy(reinterpret_cast<void*>(code_cave_end), jumpBack, sizeof(jumpBack));

        LOG_INFO(Loader,
                 "Applied Patch mask_jump32: {}, PatchAddress: {:#x}, JumpTarget: {:#x}, "
                 "CodeCaveEnd: {:#x}, JumpSize: {}",
                 patch.modNameStr, patchAddress, jump_target, code_cave_end, jumpSize);
        return;
    }

    if (cheatAddress == nullptr) {
        LOG_ERROR(Loader, "Failed to get address for patch {}", patch.modNameStr);
        return;
    }

    std::vector<unsigned char> bytePatch;

    for (size_t i = 0; i < patch.valueStr.length(); i += 2) {
        unsigned char byte = static_cast<unsigned char>(
            std::strtol(patch.valueStr.substr(i, 2).c_str(), nullptr, 16));

        bytePatch.push_back(byte);
    }

    if (patch.littleEndian) {
        std::reverse(bytePatch.begin(), bytePatch.end());
    }

    std::memcpy(cheatAddress, bytePatch.data(), bytePatch.size());

    LOG_INFO(Loader, "Applied patch: {}, Offset: {:#x}, Value: {}", patch.modNameStr,
             (uintptr_t)cheatAddress, patch.valueStr);
}

static std::vector<int32_t> PatternToByte(const std::string& pattern) {
    std::vector<int32_t> bytes;
    const char* start = pattern.data();
    const char* end = start + pattern.size();

    for (const char* current = start; current < end; ++current) {
        if (*current == '?') {
            ++current;
            if (*current == '?')
                ++current;
            bytes.push_back(-1);
        } else {
            bytes.push_back(strtoul(current, const_cast<char**>(&current), 16));
        }
    }

    return bytes;
}

uintptr_t PatternScan(const std::string& signature) {
    std::vector<int32_t> patternBytes = PatternToByte(signature);
    const auto scanBytes = static_cast<uint8_t*>((void*)g_eboot_address);

    const int32_t* sigPtr = patternBytes.data();
    const size_t sigSize = patternBytes.size();

    uint32_t foundResults = 0;
    for (uint32_t i = 0; i < g_eboot_image_size - sigSize; ++i) {
        bool found = true;
        for (uint32_t j = 0; j < sigSize; ++j) {
            if (scanBytes[i + j] != sigPtr[j] && sigPtr[j] != -1) {
                found = false;
                break;
            }
        }

        if (found) {
            foundResults++;
            return reinterpret_cast<uintptr_t>(&scanBytes[i]);
        }
    }

    return 0;
}

} // namespace MemoryPatcher
