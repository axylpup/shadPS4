// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <cstdlib>
#include <memory>
#include <vector>
#include <glslang/Include/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/ssr_dither_spv.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

namespace {
constexpr TBuiltInResource DefaultTBuiltInResource = {
    .maxLights = 32,
    .maxClipPlanes = 6,
    .maxTextureUnits = 32,
    .maxTextureCoords = 32,
    .maxVertexAttribs = 64,
    .maxVertexUniformComponents = 4096,
    .maxVaryingFloats = 64,
    .maxVertexTextureImageUnits = 32,
    .maxCombinedTextureImageUnits = 80,
    .maxTextureImageUnits = 32,
    .maxFragmentUniformComponents = 4096,
    .maxDrawBuffers = 32,
    .maxVertexUniformVectors = 128,
    .maxVaryingVectors = 8,
    .maxFragmentUniformVectors = 16,
    .maxVertexOutputVectors = 16,
    .maxFragmentInputVectors = 15,
    .minProgramTexelOffset = -8,
    .maxProgramTexelOffset = 7,
    .maxClipDistances = 8,
    .maxComputeWorkGroupCountX = 65535,
    .maxComputeWorkGroupCountY = 65535,
    .maxComputeWorkGroupCountZ = 65535,
    .maxComputeWorkGroupSizeX = 1024,
    .maxComputeWorkGroupSizeY = 1024,
    .maxComputeWorkGroupSizeZ = 64,
    .maxComputeUniformComponents = 1024,
    .maxComputeTextureImageUnits = 16,
    .maxComputeImageUniforms = 8,
    .maxComputeAtomicCounters = 8,
    .maxComputeAtomicCounterBuffers = 1,
    .maxVaryingComponents = 60,
    .maxVertexOutputComponents = 64,
    .maxGeometryInputComponents = 64,
    .maxGeometryOutputComponents = 128,
    .maxFragmentInputComponents = 128,
    .maxImageUnits = 8,
    .maxCombinedImageUnitsAndFragmentOutputs = 8,
    .maxCombinedShaderOutputResources = 8,
    .maxImageSamples = 0,
    .maxVertexImageUniforms = 0,
    .maxTessControlImageUniforms = 0,
    .maxTessEvaluationImageUniforms = 0,
    .maxGeometryImageUniforms = 0,
    .maxFragmentImageUniforms = 8,
    .maxCombinedImageUniforms = 8,
    .maxGeometryTextureImageUnits = 16,
    .maxGeometryOutputVertices = 256,
    .maxGeometryTotalOutputComponents = 1024,
    .maxGeometryUniformComponents = 1024,
    .maxGeometryVaryingComponents = 64,
    .maxTessControlInputComponents = 128,
    .maxTessControlOutputComponents = 128,
    .maxTessControlTextureImageUnits = 16,
    .maxTessControlUniformComponents = 1024,
    .maxTessControlTotalOutputComponents = 4096,
    .maxTessEvaluationInputComponents = 128,
    .maxTessEvaluationOutputComponents = 128,
    .maxTessEvaluationTextureImageUnits = 16,
    .maxTessEvaluationUniformComponents = 1024,
    .maxTessPatchComponents = 120,
    .maxPatchVertices = 32,
    .maxTessGenLevel = 64,
    .maxViewports = 16,
    .maxVertexAtomicCounters = 0,
    .maxTessControlAtomicCounters = 0,
    .maxTessEvaluationAtomicCounters = 0,
    .maxGeometryAtomicCounters = 0,
    .maxFragmentAtomicCounters = 8,
    .maxCombinedAtomicCounters = 8,
    .maxAtomicCounterBindings = 1,
    .maxVertexAtomicCounterBuffers = 0,
    .maxTessControlAtomicCounterBuffers = 0,
    .maxTessEvaluationAtomicCounterBuffers = 0,
    .maxGeometryAtomicCounterBuffers = 0,
    .maxFragmentAtomicCounterBuffers = 1,
    .maxCombinedAtomicCounterBuffers = 1,
    .maxAtomicCounterBufferSize = 16384,
    .maxTransformFeedbackBuffers = 4,
    .maxTransformFeedbackInterleavedComponents = 64,
    .maxCullDistances = 8,
    .maxCombinedClipAndCullDistances = 8,
    .maxSamples = 4,
    .maxMeshOutputVerticesNV = 256,
    .maxMeshOutputPrimitivesNV = 512,
    .maxMeshWorkGroupSizeX_NV = 32,
    .maxMeshWorkGroupSizeY_NV = 1,
    .maxMeshWorkGroupSizeZ_NV = 1,
    .maxTaskWorkGroupSizeX_NV = 32,
    .maxTaskWorkGroupSizeY_NV = 1,
    .maxTaskWorkGroupSizeZ_NV = 1,
    .maxMeshViewCountNV = 4,
    .maxDualSourceDrawBuffersEXT = 1,
    .limits =
        TLimits{
            .nonInductiveForLoops = 1,
            .whileLoops = 1,
            .doWhileLoops = 1,
            .generalUniformIndexing = 1,
            .generalAttributeMatrixVectorIndexing = 1,
            .generalVaryingIndexing = 1,
            .generalSamplerIndexing = 1,
            .generalVariableIndexing = 1,
            .generalConstantMatrixVectorIndexing = 1,
        },
};

EShLanguage ToEshShaderStage(vk::ShaderStageFlagBits stage) {
    switch (stage) {
    case vk::ShaderStageFlagBits::eVertex:
        return EShLanguage::EShLangVertex;
    case vk::ShaderStageFlagBits::eGeometry:
        return EShLanguage::EShLangGeometry;
    case vk::ShaderStageFlagBits::eTessellationControl:
        return EShLanguage::EShLangTessControl;
    case vk::ShaderStageFlagBits::eTessellationEvaluation:
        return EShLanguage::EShLangTessEvaluation;
    case vk::ShaderStageFlagBits::eFragment:
        return EShLanguage::EShLangFragment;
    case vk::ShaderStageFlagBits::eCompute:
        return EShLanguage::EShLangCompute;
    default:
        UNREACHABLE_MSG("Unkown shader stage {}", vk::to_string(stage));
    }
    return EShLanguage::EShLangVertex;
}

bool InitializeCompiler() {
    static bool glslang_initialized = false;

    if (glslang_initialized) {
        return true;
    }

    if (!glslang::InitializeProcess()) {
        LOG_CRITICAL(Render_Vulkan, "Failed to initialize glslang shader compiler");
        return false;
    }

    std::atexit([]() { glslang::FinalizeProcess(); });

    glslang_initialized = true;
    return true;
}
} // Anonymous namespace

vk::ShaderModule Compile(std::string_view code, vk::ShaderStageFlagBits stage, vk::Device device,
                         std::vector<std::string> defines) {
    if (!InitializeCompiler()) {
        return {};
    }

    EProfile profile = ECoreProfile;
    EShMessages messages =
        static_cast<EShMessages>(EShMsgDefault | EShMsgSpvRules | EShMsgVulkanRules);
    EShLanguage lang = ToEshShaderStage(stage);

    const int default_version = 450;
    const char* pass_source_code = code.data();
    int pass_source_code_length = static_cast<int>(code.size());

    auto shader = std::make_unique<glslang::TShader>(lang);
    shader->setEnvTarget(glslang::EShTargetSpv,
                         glslang::EShTargetLanguageVersion::EShTargetSpv_1_3);
    shader->setStringsWithLengths(&pass_source_code, &pass_source_code_length, 1);

    std::string preambleString;
    std::vector<std::string> processes;

    for (auto& def : defines) {
        processes.push_back("define-macro ");
        processes.back().append(def);

        preambleString.append("#define ");
        if (const size_t equal = def.find_first_of("="); equal != def.npos) {
            def[equal] = ' ';
        }
        preambleString.append(def);
        preambleString.append("\n");
    }

    shader->setPreamble(preambleString.c_str());
    shader->addProcesses(processes);

    glslang::TShader::ForbidIncluder includer;

    std::string preprocessedStr;
    if (!shader->preprocess(&DefaultTBuiltInResource, default_version, profile, false, true,
                            messages, &preprocessedStr, includer)) [[unlikely]] {
        LOG_ERROR(Render_Vulkan,
                  "Shader preprocess error\n"
                  "Shader Info Log:\n"
                  "{}\n{}",
                  shader->getInfoLog(), shader->getInfoDebugLog());
        LOG_ERROR(Render_Vulkan, "Shader Source:\n{}", code);
        return {};
    }

    if (!shader->parse(&DefaultTBuiltInResource, default_version, profile, false, true, messages,
                       includer)) [[unlikely]] {
        LOG_ERROR(Render_Vulkan,
                  "Shader parse error\n"
                  "Shader Info Log:\n"
                  "{}\n{}",
                  shader->getInfoLog(), shader->getInfoDebugLog());
        LOG_ERROR(Render_Vulkan, "Shader Source:\n{}", code);
        return {};
    }

    // Even though there's only a single shader, we still need to link it to generate SPV
    auto program = std::make_unique<glslang::TProgram>();
    program->addShader(shader.get());
    if (!program->link(messages)) {
        LOG_ERROR(Render_Vulkan,
                  "Shader link error\n"
                  "Program Info Log:\n"
                  "{}\n{}",
                  program->getInfoLog(), program->getInfoDebugLog());
        return {};
    }

    glslang::TIntermediate* intermediate = program->getIntermediate(lang);
    std::vector<u32> out_code;
    spv::SpvBuildLogger logger;
    glslang::SpvOptions options;

    // Enable optimizations on the generated SPIR-V code.
    options.disableOptimizer = false;
    options.validate = false;
    options.optimizeSize = true;

    glslang::GlslangToSpv(*intermediate, out_code, &logger, &options);

    const std::string spv_messages = logger.getAllMessages();
    if (!spv_messages.empty()) {
        LOG_INFO(Render_Vulkan, "SPIR-V conversion messages: {}", spv_messages);
    }

    return CompileSPV(out_code, device);
}

/// SHADPS4_SSR_DITHER: 0 = the game's march, 1 = a fixed per-pixel start offset, 2 (default,
/// user 10-21) = the offset also moves every frame (the march's own history blend and DLSS
/// average it; a fixed pattern stays glued to the screen while the street moves).
int SsrDitherMode() {
    static const int mode = [] {
        const char* value = std::getenv("SHADPS4_SSR_DITHER");
        return !value ? 2 : value[0] == '0' ? 0 : value[0] == '1' ? 1 : 2;
    }();
    return mode;
}

bool SsrDitherEnabled() {
    return SsrDitherMode() != 0;
}

namespace {

/// SHADPS4_PROBE_WALK_NAN_EXIT=1 (default off; GitHub #27, fights and the chain): the game hides
/// dead debris and particle instances by writing NaN into their translation. Their vertex shaders
/// (vs_3ab1ca58: debris into the G-buffer, vs_bbd2d927: particles) look up the light-probe
/// tetrahedron of every vertex with a walk from a start cell to the neighbour across the most
/// negative barycentric coordinate, bounded only by the number of cells (15167 in the first
/// district). A NaN position is inside no cell, so those vertices walk to the bound: about
/// 15000 dependent buffer loads each, 2 - 3 ms of GPU time per draw whatever the resolution,
/// for primitives the clipper then throws away. The module's words are edited so that a
/// non-finite probe-space position leaves the walk after one cell. Finite vertices are untouched
/// (their walks take 20 - 70 cells). Matched by size and FNV-1a like the SSR dither below, so
/// another GPU's SPIR-V keeps the game's.
struct ProbeWalkPatch {
    size_t words;           ///< the original module's size
    u64 fnv;                ///< FNV-1a 64 over its bytes
    u32 insert_at;          ///< word index of the OpBranch into the loop header
    u32 limit_at;           ///< word index of the walk's step limit (operand of OpULessThan)
    u32 limit_id;           ///< the id found there
    u32 type_bool, type_u32, type_f32;
    u32 const_one;          ///< OpConstant %u32 1
    std::array<u32, 3> pos; ///< ids of the probe-space position
};

constexpr std::array<ProbeWalkPatch, 2> ProbeWalkPatches{{
    {10280, 0xb0005c73dfb93d89ull, 3606, 3706, 184, 3, 10, 8, 29, {403, 405, 407}},
    {13734, 0x3e6652c353f4b2b9ull, 5919, 6019, 760, 3, 10, 8, 29, {785, 788, 791}},
}};

u64 Fnv1a64(std::span<const u32> code) {
    u64 h = 0xcbf29ce484222325ull;
    for (const u32 w : code) {
        for (u32 b = 0; b < 4; ++b) {
            h ^= (w >> (8 * b)) & 0xffu;
            h *= 0x100000001b3ull;
        }
    }
    return h;
}

/// Returns true when `code` is one of the two probe-walk shaders and `out` holds the edit.
bool PatchProbeWalk(std::span<const u32> code, std::vector<u32>& out) {
    for (const auto& p : ProbeWalkPatches) {
        if (code.size() != p.words || code[p.limit_at] != p.limit_id || Fnv1a64(code) != p.fnv) {
            continue;
        }
        const u32 n = code[3]; // the id bound: six new ids
        constexpr u32 OpFAdd = 129, OpIsNan = 156, OpIsInf = 157, OpLogicalOr = 166,
                      OpSelect = 169;
        const std::array<u32, 29> walk_limit{
            // sum = x + y + z: non-finite when any of them is
            (5u << 16) | OpFAdd, p.type_f32, n, p.pos[0], p.pos[1],
            (5u << 16) | OpFAdd, p.type_f32, n + 1, n, p.pos[2],
            (4u << 16) | OpIsNan, p.type_bool, n + 2, n + 1,
            (4u << 16) | OpIsInf, p.type_bool, n + 3, n + 1,
            (5u << 16) | OpLogicalOr, p.type_bool, n + 4, n + 2, n + 3,
            // limit = non-finite ? 1 : the game's
            (6u << 16) | OpSelect, p.type_u32, n + 5, n + 4, p.const_one, p.limit_id,
        };
        out.clear();
        out.reserve(code.size() + walk_limit.size());
        out.insert(out.end(), code.begin(), code.begin() + p.insert_at);
        out.insert(out.end(), walk_limit.begin(), walk_limit.end());
        out.insert(out.end(), code.begin() + p.insert_at, code.end());
        out[3] = n + 6;
        out[p.limit_at + walk_limit.size()] = n + 5;
        return true;
    }
    return false;
}

} // namespace

bool ProbeWalkNanExitEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_PROBE_WALK_NAN_EXIT");
        return value && value[0] == '1';
    }();
    return enabled;
}

vk::ShaderModule CompileSPV(std::span<const u32> code, vk::Device device) {
    // SHADPS4_PROBE_WALK_NAN_EXIT (above): every path compiles these two modules here.
    std::vector<u32> probe_walk;
    if (ProbeWalkNanExitEnabled() && code.size() > 6019 && PatchProbeWalk(code, probe_walk))
        [[unlikely]] {
        static std::atomic<u32> logged{0}; // compile-pool threads call this too
        if (logged.fetch_add(1) < 2) {
            LOG_INFO(Render_Vulkan, "ProbeWalk: a non-finite position leaves the light-probe walk "
                                    "({} words)",
                     code.size());
        }
        code = probe_walk;
    }
    // SHADPS4_SPIRV_SALT=<n> (diagnostic, 2026-10-07): writes n into the SPIR-V header's generator
    // word (informational only), so the driver's own shader cache misses and a run measures
    // first-time driver compiles without touching that cache.
    static const u32 salt = [] {
        const char* value = std::getenv("SHADPS4_SPIRV_SALT");
        return value ? static_cast<u32>(std::strtoul(value, nullptr, 0)) : 0u;
    }();
    std::vector<u32> salted;
    if (salt != 0 && code.size() > 2) [[unlikely]] {
        salted.assign(code.begin(), code.end());
        salted[2] = salt;
        code = salted;
    }
    // The 33% street streaks (1.1, user 10-21; study-delsin-mv-ssr sec. 2): the game's reflection
    // ray march fs_204f45fe takes 100 fixed steps with no per-pixel offset, so neighbouring rows
    // stop at the same step and draw stripes. Its exact module (every path: fresh, cached,
    // prebuilt) is swapped for one that starts each pixel at a dithered offset. Another GPU's
    // SPIR-V differs and keeps the game's. SHADPS4_SSR_DITHER=0 off.
    if (code.size() == SsrDither::OriginalWords) [[unlikely]] {
        const bool on = SsrDitherEnabled();
        u64 h = 0xcbf29ce484222325ull; // FNV-1a 64 over the bytes
        for (const u32 w : code) {
            for (u32 b = 0; b < 4; ++b) {
                h ^= (w >> (8 * b)) & 0xffu;
                h *= 0x100000001b3ull;
            }
        }
        if (on && h == SsrDither::OriginalFnv) {
            const bool temporal = SsrDitherMode() == 2;
            static std::atomic<bool> logged{false}; // compile-pool threads call this too
            if (!logged.exchange(true)) {
                LOG_INFO(Render_Vulkan, "SSR: the reflection ray march gets a dithered start{}",
                         temporal ? " (moving every frame)" : "");
            }
            if (temporal) {
                code = SsrDither::PatchedTemporal;
            } else {
                code = SsrDither::Patched;
            }
        }
    }
    const vk::ShaderModuleCreateInfo shader_info = {
        .codeSize = code.size() * sizeof(u32),
        .pCode = code.data(),
    };

    auto [module_result, module] = device.createShaderModule(shader_info);
    ASSERT_MSG(module_result == vk::Result::eSuccess, "Failed to compile SPIR-V shader: {}",
               vk::to_string(module_result));
    return module;
}

} // namespace Vulkan
