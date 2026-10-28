<!--
SPDX-FileCopyrightText: 2026 axylpup
SPDX-License-Identifier: GPL-2.0-or-later

The front page of github.com/axylpup/shadPS4 (shown from .github/README.md on the default branch, second-son).
GitHub shows .github/README.md before the root README.md, so upstream's README.md stays untouched.
-->

<div align="center">

# inFAMOUS Second Son on PC

**A build of the [shadPS4](https://github.com/shadps4-emu/shadPS4) PlayStation 4 emulator, tuned for one game.**<br>
High frame rates, DLSS, native 4K and ultrawide, real mouse aiming and a full keyboard layout, on Windows.

[![Latest build](https://img.shields.io/github/v/release/axylpup/shadPS4?label=latest%20build&style=for-the-badge)](https://github.com/axylpup/shadPS4/releases/latest)
[![Downloads](https://img.shields.io/github/downloads/axylpup/shadPS4/total?style=for-the-badge)](https://github.com/axylpup/shadPS4/releases)

### [Download the latest build](https://github.com/axylpup/shadPS4/releases/latest)

Windows 10 / 11, one zip of about 74 MB. **You need your own copy of the game** - no game files are included.

**[Features](#features) · [Getting started](#getting-started) · [FAQ](#faq) · [Controls](#controls) · [DLSS](#dlss-optional) · [Known issues](#known-issues)**

![Delsin on a rooftop in Seattle, 131 FPS on the test PC](media/city-rooftop.jpg)

<sub>Seattle rooftop on the test PC (i5-13600K + RTX 4070 Ti SUPER, 1080p). The counter shows that moment.</sub>

</div>

> [!NOTE]
> Unofficial and experimental. Not affiliated with Sony Interactive Entertainment, Sucker Punch Productions or the
> shadPS4 team. Tested with the EU version (CUSA00004, update 1.07); the US version (CUSA00223) gets the same
> settings - reports welcome.

> [!TIP]
> **1.1.1 (hotfix):** video cutscenes now fit ultrawide screens with black bars instead of being cut off; fixes for
> crashes at start (game version 01.00 - install update 1.07), on the loading screen (12 GB cards), on some AMD cards
> and under Linux / Proton. **HDR is experimental - keep it off for now** (F3 -> Graphics -> Enable HDR); proper HDR
> with DLSS comes in 1.2. [All changes](https://github.com/axylpup/shadPS4/releases/latest).
>
> **New in 1.1:** any screen shape (ultrawide 21:9 / 32:9, 16:10, 4:3), the render size as a percentage of your
> screen changed live, native 4K, sharper DLSS without the wobble, far fewer stripes in street reflections, even motion
> on screens without G-Sync / FreeSync, a Quick settings tab, and the loading screen tells you when a new version is
> out. [All changes](https://github.com/axylpup/shadPS4/releases/latest).

## Features

Everything below is in the download, set up for this game - no tweaking needed.

### ⚡ Speed

| Feature | What you get |
|---|---|
| **High frame rates** | A multithreaded renderer and faster memory handling, frame rate unlocked. On the test PC the open-world city runs at about 140 FPS (1% lows about 134) and the heaviest rooftop view at about 110 FPS (measured with 1.0; 1.1 runs the same within 2%). One PC only - see the release notes for rough numbers on smaller CPUs. |
| **Render size, live** | 10-200% of your screen with a slider or a number (<kbd>Page Up</kbd>), DLSS presets 33 / 50 / 58 / 67 / 100%. Lower = faster on weaker cards, higher = sharper. No restart up to about 3200x1800 (4K after one restart). Works on every graphics card. |
| **Tunes itself to your PC** | CPUs with fewer threads get a better thread setup; cards under 10 GB and PCs under 24 GB of RAM use less memory, by themselves. |
| **Shader cache included** | Shaders are prepared once on the first start, with a progress screen - far less first-time stutter. |
| **Performance mode** | Optional: the game's own lower settings for weak PCs (+13% / +19% FPS in our heaviest view). |

### 🖼️ Picture

| Feature | What you get |
|---|---|
| **DLSS 4.5 (NVIDIA RTX)** | Upscaling or DLAA with a sharp HUD, 8-64 sub-pixel positions for smoother edges, with the game's motion data corrected so characters do not wobble. Bring your own NVIDIA files ([2 minutes](#dlss-optional)). |
| **Any screen shape** | Ultrawide 21:9 and 32:9, 16:10, 4:3, even square - wider screens see more, nothing is stretched; the HUD stays in a 16:9 area. |
| **Native 4K** | Renders 3840x2160 (the game's light lists are enlarged for it; before, the game crashed above 3200x1800). |
| **Graphics fixes** | Lighting, smoke / foam / leaf flicker and hair fixed; far fewer stripes in street reflections at low render sizes (NVIDIA for now); TVs in the city play their video; sharper water reflections. |
| **Even motion without VRR** | V-Sync "half refresh": every picture shown for exactly two refreshes (72 FPS on 144 Hz, 120 on 240 Hz) - smooth on screens without G-Sync / FreeSync. Frame-rate cap included. |

### 🎮 Input

| Feature | What you get |
|---|---|
| **Real 1:1 mouse aiming** | The mouse turns the camera directly, like a PC game; left click shoots, right click aims. Speed with <kbd>-</kbd> / <kbd>=</kbd>, remembered. |
| **Full keyboard + mouse layout** | Every move, including the spray-can (motion sensor) and touchpad moments ([picture](#keyboard-and-mouse)). |
| **Keyboard / mouse icons in the game** | Button prompts show your keys; pick up a controller and they switch back to PlayStation buttons. |
| **Any controller** | DualSense / DualShock 4 as on the console; Xbox and other pads get a sensor mode (<kbd>L3</kbd> + <kbd>R3</kbd>) and touchpad buttons ([picture](#controllers)). Stick deadzone setting. |
| **Every key can be changed** | Including the emulator's own keys (DLSS, panels, mouse speed) - handy on laptops. |

### 🛋️ Comfort

| Feature | What you get |
|---|---|
| **Quick settings tab (<kbd>F3</kbd>)** | Display mode, V-Sync, frame-rate cap, volume, deadzones and language in one place, applied right away; the menu fits any window. |
| **Remembers your setup** | Fullscreen, mouse speed, render size and the game's own settings stay after a restart; the Load menu's "No" works. |
| **Update check** | The loading screen shows the newest version every start and says so in green when an update is out. Nothing is downloaded. |
| **Easy updates** | Unzip the new version over the old folder - saves, settings, keys and DLSS files stay. |
| **Fixes for more PCs** | 8 GB cards (NVIDIA and AMD) load without running out of memory, AMD RX 5000 shader crash fixed, laptops start on the real graphics card, folders with non-English letters work, clear crash reports (`fatal.txt`). |

![The forest fire, 119 FPS on the test PC](media/forest-fire.jpg)

<p align="right"><a href="#infamous-second-son-on-pc">Back to top ↑</a></p>

## Getting started

1. Download the zip from the [latest release](https://github.com/axylpup/shadPS4/releases/latest) and unzip it
   anywhere.
2. Run `shadPS4.exe` and pick your game folder (the one with `eboot.bin`). It remembers it.
3. The first start prepares shaders (one time, with a progress screen). Then play.

A DualSense / DualShock 4 works as on the console. All keys, the special sections (<kbd>T</kbd> = touchpad,
<kbd>X</kbd> = motion sensor), changing settings and more questions are in the README inside the zip and in the
[FAQ](#faq) below; `shadps4-flags-user-EXAMPLES.txt` lists the settings players usually want.

> [!TIP]
> **Updating:** unzip the new version over your old folder and replace the files. Saves, settings, keys, DLSS files
> and your own `shadps4-flags-user.txt` stay. The first start prepares shaders once more. The loading screen tells you
> when a new version is out.

## FAQ

Click a question to open it. Lines like `SHADPS4_...=...` go into `shadps4-flags-user.txt` next to `shadPS4.exe`
(create it if it is not there; restart the game after editing). The README inside the zip has more.

<details>
<summary><b>How do I bring my saves over?</b></summary>

1. Close the game.
2. Find your old saves: other shadPS4 installs keep them in `%APPDATA%\shadPS4\home\1000\savedata\CUSA00004\` (paste
   that into the Explorer address bar), older shadPS4 versions in `user\savedata\1\CUSA00004\`. Each save is a folder
   like `infamousssAuto0`.
3. Copy those folders into `user\home\1000\savedata\CUSA00004\` next to this build's `shadPS4.exe` (start the game
   once so the folders exist, or create them).
4. US version: `CUSA00223` instead of `CUSA00004`.

Updating this build keeps your saves - nothing to copy. Saves from a PS4 itself are encrypted; only saves decrypted
with homebrew save tools can be copied the same way (untested).
</details>

<details>
<summary><b>How do I change the language?</b></summary>

<kbd>F3</kbd> -> **Quick** -> **Language**, Save, restart the game. Text and voices change together. The game has
only the languages of your copy; if it stays in English, try the other variant (for example Spanish (Spain) / Spanish
(Latin America)).
</details>

<details>
<summary><b>Motion stutters although the FPS is high (no G-Sync / FreeSync)</b></summary>

A normal screen shows a new picture only at each refresh, so at, say, 100 FPS on 144 Hz some pictures stay one
refresh and some two - motion jitters. <kbd>F3</kbd> -> **Quick** -> **V-Sync** -> **Fifo (half refresh)**, Save:
every picture stays exactly two refreshes (72 FPS on 144 Hz, 60 on 120 Hz, 120 on 240 Hz). If your PC holds your full
refresh rate, plain **Fifo** is smoother still. Applies right away. With G-Sync / FreeSync none of this is needed.
</details>

<details>
<summary><b>How do I make it run faster?</b></summary>

Lower the render size: <kbd>Page Up</kbd> -> **Render size**, for example 67% or 50% (DLSS on RTX cards, or the
normal scaler on any card, brings it back up to your screen). It changes live. Weaker PCs can also try
`SHADPS4_PERF_MODE=low` (the game's own lower settings).
</details>

<details>
<summary><b>How do I play fullscreen? Ultrawide?</b></summary>

<kbd>F11</kbd> (or <kbd>F3</kbd> -> **Quick** -> **Display Mode**) - remembered for the next start. Ultrawide and
other screen shapes are filled by themselves; restart the game after resizing the window. The HUD stays in a 16:9 area
in the middle (`SHADPS4_HUD=full` moves it to the edges); video cutscenes keep black bars.
</details>

<details>
<summary><b>The frame rate stops at my monitor's refresh rate</b></summary>

Your graphics driver has no "Mailbox" mode (AMD on Windows, some Intel and laptop drivers), so the game uses V-Sync.
For an uncapped frame rate: <kbd>F3</kbd> -> **Quick** -> **V-Sync** -> **Immediate** (can tear without FreeSync /
G-Sync).
</details>

<details>
<summary><b>Mouse too fast / stuck to the camera? Changing keys?</b></summary>

<kbd>-</kbd> / <kbd>=</kbd> change the mouse speed (kept). <kbd>Y</kbd> turns mouse aiming off, <kbd>F3</kbd> frees
the mouse while open. Keys: `user\input_config\default.ini` (game) and `global.ini` (emulator keys), then
<kbd>F8</kbd> in game to reload.
</details>

<details>
<summary><b>Xbox or other controller without a touchpad / motion sensor?</b></summary>

Press <kbd>L3</kbd> + <kbd>R3</kbd> (both sticks) for the spray-can and touchpad moments: the right stick aims the
can, hold <kbd>A</kbd> to tilt, tap <kbd>A</kbd> to shake; the left stick is your finger on the touchpad.
<kbd>D-pad up</kbd> / <kbd>View</kbd> are the touchpad halves. All of it as a picture: [Controllers](#controllers).
Sticks drifting? <kbd>F3</kbd> -> **Quick** -> **Stick Deadzone**.
</details>

<details>
<summary><b>DLSS does not turn on</b></summary>

Check the five files (see [DLSS](#dlss-optional)): exactly Streamline **2.14.16**, from `bin\x64` (not
`development`), and an NVIDIA RTX card. `user\log\shad_log.txt` says why - search it for `DLSS:`.
</details>

<details>
<summary><b>The first start takes long / it asks for the game folder again</b></summary>

The first start prepares shaders once (a few minutes, with a progress screen) - leave it running. To pick another
game folder, delete `shadps4-game.txt` next to `shadPS4.exe` and choose the folder that contains `eboot.bin`.
</details>

<details>
<summary><b>Crashes, wrong textures or out of video memory (AMD, Intel, older cards)</b></summary>

Put `SHADPS4_A2=0` into `shadps4-flags-user.txt` (the multithreaded renderer off - slower, but closest to standard
shadPS4), then [open an issue](https://github.com/axylpup/shadPS4/issues/new/choose) with your logs.
</details>

<details>
<summary><b>Does it go online?</b></summary>

Only for the update check: one small request to GitHub per start, shown on the loading screen. Nothing is downloaded.
`SHADPS4_UPDATE_CHECK=0` turns it off.
</details>

<p align="right"><a href="#infamous-second-son-on-pc">Back to top ↑</a></p>

## Controls

### Keyboard and mouse

![Keyboard and mouse layout](media/keyboard-mouse.png)

### Controllers

**DualSense / DualShock 4:** works as on the console - touchpad and motion sensor included, nothing to set up.

**Xbox and other controllers** (no touchpad or motion sensor) get these extras:

![Xbox and other controllers: D-pad up and View are the touchpad halves, L3 + R3 turns on sensor mode](media/controller.png)

<details>
<summary>The same as a list</summary>

| The game asks for | On your controller |
|---|---|
| Touchpad, left half (for example draining power sources) | <kbd>D-pad up</kbd> (it still works as up in menus) |
| Touchpad, right half | <kbd>View</kbd> (Back / Select) |
| Both touchpad halves | <kbd>D-pad up</kbd> + <kbd>View</kbd> together |
| Spray can or touchpad moments | <kbd>L3</kbd> + <kbd>R3</kbd> (press both sticks) turns on **sensor mode**; a notice shows the controls. <kbd>L3</kbd> + <kbd>R3</kbd> again turns it off |
| In sensor mode: aim the spray can | **Right stick** |
| In sensor mode: hold the can sideways / rattle it | Hold <kbd>A</kbd> / tap <kbd>A</kbd> (spray with the trigger as usual) |
| In sensor mode: your finger on the touchpad (fingerprint scanners, the D.U.P. core lifts) | **Left stick** - push it to touch, its direction is where the finger goes |

</details>

Sensor mode always gives you the stick aiming and <kbd>A</kbd> for the spray can, also on controllers with their own gyro
(Switch Pro, ROG Ally) - their real gyro works whenever sensor mode is off. The left-stick finger only comes on controllers
without a touchpad. Sticks drifting? <kbd>F3</kbd> -> **Quick** -> **Left / Right Stick Deadzone**.

<p align="right"><a href="#infamous-second-son-on-pc">Back to top ↑</a></p>

## DLSS (optional)

NVIDIA's files are not included (licence). Adding them takes about 2 minutes:

1. Download **Streamline SDK 2.14.16** from <https://github.com/NVIDIA-RTX/Streamline/releases> - exactly that
   version.
2. In its zip, open the `bin\x64` folder (not `development`).
3. Copy these five files into the `dlss` folder next to `shadPS4.exe`: `sl.interposer.dll`, `sl.common.dll`,
   `sl.dlss.dll`, `sl.pcl.dll`, `nvngx_dlss.dll`.
4. Start the game: DLSS turns on by itself. <kbd>Home</kbd> switches it on / off to compare, <kbd>Page Up</kbd>
   opens the DLSS panel (render size, network, status).

> [!NOTE]
> Without the files (or without an NVIDIA RTX card) the game simply uses its own anti-aliasing - everything else works
> the same.

## Requirements

- Windows 10 / 11 64-bit, a Vulkan 1.3 graphics card (12 GB video memory recommended; 8 GB runs since 1.0.2, smaller
  cards can lower the render size), a fast 6+ core CPU.
- DLSS: an NVIDIA RTX card (20 series or newer) and a recent driver.
- Other hardware should work (every new feature has a fallback) but only the test PC below is tested - reports
  welcome.

**The test PC** - every number on this page comes from it:

| Part | Model |
|---|---|
| CPU | Intel Core i5-13600K (6 performance + 8 efficiency cores, 20 threads) |
| Graphics card | NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB, driver 617.14 |
| RAM | 32 GB DDR5 (2 x 16 GB, running at 4800 MT/s) |
| Storage | WD_BLACK SN850X 4 TB NVMe (game files); Windows on a WD_BLACK SN770 1 TB NVMe |
| Screen | MSI MPG 271QRX, 27" QD-OLED, 2560x1440, set to 240 Hz, G-Sync |
| Windows | Windows 11 Home |

## Known issues

Real bugs that are not fixed yet (stutter, language, saves and other questions with a quick fix: [FAQ](#faq)).

- HDR (<kbd>F3</kbd> -> Graphics -> Enable HDR) is experimental: too dark and contrasty, and no DLSS while it is on -
  keep it off until 1.2.
- Touchpad prompts and the controller picture in the Options screen still show the PlayStation controller.
- A few particle effects (smoke, fire) can flicker in places.
- On graphics cards other than NVIDIA, street reflections still show stripes at low render sizes.
- With DLSS on, menus look a little less smooth around the edges; the <kbd>F12</kbd> screenshot shows the HUD on
  black (use <kbd>Alt</kbd> + <kbd>F12</kbd>).
- Some heavy views (for example the first karma choice) sometimes run at about half their usual frame rate;
  reloading usually brings it back.

What changed in each version is on the [Releases](https://github.com/axylpup/shadPS4/releases) page.

## Found a problem?

[Open an issue](https://github.com/axylpup/shadPS4/issues/new/choose) and attach `user\log\shad_log.txt` (and
`user\log\fatal.txt` after a crash). The form asks for your graphics card, CPU and what you were doing. This page is
for this build only - problems with other games or the official shadPS4 builds belong on
[shadPS4's own page](https://github.com/shadps4-emu/shadPS4).

## This repository

<details>
<summary>For developers: branches, tags and building</summary>

- **`second-son`** (this branch) - the source of the builds: shadPS4 0.17 (upstream `main` of 2026-08-08) plus our
  commits. Each release has a tag (`second-son-v1.1`, ...) pointing at the exact source of its zip.
- **`main`** - upstream shadPS4, unchanged.
- Building: follow shadPS4's
  [Windows build guide](https://github.com/axylpup/shadPS4/blob/second-son/documents/building-windows.md). Release
  builds use the profile-guided optimisation data in [`pgo/`](https://github.com/axylpup/shadPS4/tree/second-son/pgo).

</details>

## Support

This build is free and always will be - no paid builds, no early access. If you'd like to chip in,
[GitHub Sponsors](https://github.com/sponsors/axylpup) helps cover the AI subscription that writes the code (about
$100 a month; GitHub takes no fee). Anything beyond that buys test hardware like players actually own - first an
entry-level AMD graphics card, since AMD-only problems are hard to fix without one. You still need your own copy of
the game; no game files are included.

## How this was made

This build was made with AI. I'm a 3D artist, not a programmer: I directed the project - set the goals, chose what to
work on, played and tested every build, judged what looked and felt right, and made the release calls. The code, the
tests and measurements and most of the documentation were written by AI models: most of it by Anthropic's Claude Opus
5.5, earlier parts by other Claude models and by OpenAI's GPT models. Announcement images were made with ChatGPT
(1.0 - 1.0.2) and Claude (1.1). All of it stands on shadPS4, written by its team and contributors over years.

## Credits and licence

Directed and tested by Axylpup - code by AI (mostly Claude Opus 5.5, Anthropic) - built on
[shadPS4](https://github.com/shadps4-emu/shadPS4), thanks to its team and contributors - you can support them on
[Ko-fi](https://ko-fi.com/shadps4). GPL-2.0-or-later.
NVIDIA Streamline headers are MIT; the DLSS runtime files are NVIDIA's and come from your own download. inFAMOUS
Second Son (c) Sony Interactive Entertainment / Sucker Punch Productions.
