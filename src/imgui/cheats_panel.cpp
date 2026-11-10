// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <string>
#include <vector>
#include <imgui.h>

#include "common/cheats.h"
#include "common/memory_patcher.h"
#include "imgui/cheats_panel.h"
#include "imgui/imgui_layer.h"

namespace CheatsPanel {

namespace {

std::atomic<bool> g_open{false};

const ImVec4 Yellow{0.95f, 0.80f, 0.35f, 1.0f};
const ImVec4 Grey{0.65f, 0.65f, 0.68f, 1.0f};

class Panel final : public ImGui::Layer {
public:
    void Draw() override;
    std::atomic<bool> closing{false}; ///< set by either thread; the layer removal is queued
    bool stale = true;                ///< the list is read again on the next frame

private:
    void DrawFile(size_t index, const Cheats::File& file);

    std::vector<Cheats::File> files;
    std::string message; ///< why the last click was not done
};

Panel g_panel; // one object for the whole run: adds and removes are queued in order

void Panel::DrawFile(size_t index, const Cheats::File& file) {
    std::string header = file.title.empty() ? file.file_name : file.title;
    header += "  (" + file.file_name + ")";
    if (!ImGui::CollapsingHeader(header.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }
    if (!file.id.empty() && file.id != MemoryPatcher::g_game_serial) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(Grey,
                           "Made for %s %s, this game is %s: only the cheats whose code is the "
                           "same here can be used.",
                           file.id.c_str(), file.version.c_str(),
                           MemoryPatcher::g_game_serial.c_str());
        ImGui::PopTextWrapPos();
    }
    for (size_t m = 0; m < file.mods.size(); ++m) {
        const Cheats::Mod& mod = file.mods[m];
        const bool ready = mod.state == Cheats::ModState::Ready;
        ImGui::PushID(int(m));
        ImGui::BeginDisabled(!ready);
        if (mod.is_button) {
            if (ImGui::Button(mod.name.c_str())) {
                message = Cheats::Set(index, m, true);
            }
        } else {
            bool on = mod.on;
            if (ImGui::Checkbox(mod.name.c_str(), &on)) {
                message = Cheats::Set(index, m, on);
                stale = true;
            }
        }
        ImGui::EndDisabled();
        if (!ready) {
            ImGui::SameLine();
            ImGui::TextColored(Yellow, "%s",
                               mod.state == Cheats::ModState::Mismatch
                                   ? "- does not match this game's code"
                                   : "- needs memory outside the game");
        }
        ImGui::PopID();
    }
    if (!file.credits.empty()) {
        ImGui::TextColored(Grey, "By %s", file.credits.c_str());
    }
    ImGui::Spacing();
}

void Panel::Draw() {
    if (closing) {
        return;
    }
    if (stale) {
        files = Cheats::Files();
        stale = false;
    }

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + 24.0f, vp->WorkPos.y + 90.0f),
                            ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(560.0f, 0.0f),
                                        ImVec2(560.0f, std::max(200.0f, vp->WorkSize.y - 110.0f)));
    ImGui::SetNextWindowBgAlpha(0.88f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(16.0f, 14.0f));
    bool keep_open = true;
    if (ImGui::Begin("Cheats###cheatspanel", &keep_open,
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize |
                         ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking)) {
        ImGui::SetWindowFontScale(1.15f);
        if (files.empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextUnformatted("No cheat files yet.");
            ImGui::Spacing();
            ImGui::TextUnformatted(
                "Put GoldHEN cheat files (.json) for this game into the folder below and press "
                "Reload. They are in the \"json\" folder of "
                "github.com/GoldHEN/GoldHEN_Cheat_Repository, named after the game and its "
                "version, for example CUSA00004_01.07.json.");
            ImGui::Spacing();
            ImGui::TextColored(Grey, "%s", Cheats::Folder().string().c_str());
            ImGui::PopTextWrapPos();
        }
        for (size_t i = 0; i < files.size(); ++i) {
            ImGui::PushID(int(i));
            DrawFile(i, files[i]);
            ImGui::PopID();
        }
        ImGui::Separator();
        if (!message.empty()) {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(Yellow, "%s", message.c_str());
            ImGui::PopTextWrapPos();
        }
        if (ImGui::Button("Reload")) {
            Cheats::Reload();
            message.clear();
            stale = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Close  (F6)")) {
            keep_open = false;
        }
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(Grey, "Checkboxes are remembered and switched on at the next start. "
                                 "Buttons write a value once.");
        ImGui::PopTextWrapPos();
    }
    ImGui::End();
    ImGui::PopStyleVar(3);

    if (!keep_open && !closing.exchange(true)) {
        ImGui::Layer::RemoveLayer(this); // before IsOpen() turns false (key race)
        g_open.store(false, std::memory_order_release);
    }
}

} // namespace

void Toggle() {
    if (g_open.exchange(false, std::memory_order_acq_rel)) {
        if (!g_panel.closing.exchange(true)) {
            ImGui::Layer::RemoveLayer(&g_panel);
        }
        return;
    }
    g_panel.closing.store(false);
    g_panel.stale = true;
    ImGui::Layer::AddLayer(&g_panel);
    g_open.store(true, std::memory_order_release);
}

bool IsOpen() {
    return g_open.load(std::memory_order_acquire);
}

} // namespace CheatsPanel
