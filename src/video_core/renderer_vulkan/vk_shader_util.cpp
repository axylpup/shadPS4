// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
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
/// negative barycentric coordinate. The walk goes on while "not (-0.01 <= the smallest
/// coordinate)", bounded only by the number of cells (15167 in the first district). With a NaN
/// position every coordinate is NaN, the test never passes, and those vertices walk to the
/// bound: about 15000 dependent buffer loads each, 2 - 3 ms of GPU time per draw whatever the
/// resolution, for primitives the clipper then throws away.
/// The compare is made unordered, so a NaN leaves the walk at the first closed cell. Finite
/// vertices run the game's code (their walks take 20 - 70 cells). Found by its shape, not by a
/// hash, so every GPU's variant of the two modules gets it:
///   vertex stage, a loop, %c = OpFOrdLessThanEqual %bool <constant -0.01> %x
///                         %n = OpLogicalNot %bool %c
/// Returns how many compares were changed (`out` then holds the module).
u32 PatchProbeWalk(std::span<const u32> code, std::vector<u32>& out) {
    constexpr u32 OpEntryPoint = 15, OpConstant = 43, OpLogicalNot = 168,
                  OpFOrdLessThanEqual = 188, OpFUnordLessThanEqual = 189, OpLoopMerge = 246;
    constexpr u32 MinusOneHundredth = 0xbc23d70a; // -0.01f
    if (code.size() < 16) {
        return 0;
    }
    std::array<u32, 4> constants{};
    u32 num_constants = 0;
    std::array<u32, 4> compares{};
    u32 num_compares = 0;
    bool has_loop = false;
    for (size_t i = 5; i < code.size();) {
        const u32 op = code[i] & 0xffffu;
        const u32 count = code[i] >> 16;
        if (count == 0 || i + count > code.size()) {
            return 0;
        }
        switch (op) {
        case OpEntryPoint:
            if (code[i + 1] != 0) { // spv::ExecutionModel::Vertex
                return 0;
            }
            break;
        case OpConstant:
            if (count == 4 && code[i + 3] == MinusOneHundredth &&
                num_constants < constants.size()) {
                constants[num_constants++] = code[i + 2];
            }
            break;
        case OpLoopMerge:
            has_loop = true;
            break;
        case OpFOrdLessThanEqual: {
            const size_t next = i + count;
            const bool is_limit =
                count == 5 && std::find(constants.begin(), constants.begin() + num_constants,
                                        code[i + 3]) != constants.begin() + num_constants;
            if (is_limit && next + 3 < code.size() &&
                code[next] == ((4u << 16) | OpLogicalNot) && code[next + 3] == code[i + 2] &&
                num_compares < compares.size()) {
                compares[num_compares++] = u32(i);
            }
            break;
        }
        default:
            break;
        }
        i += count;
    }
    if (!has_loop || num_compares == 0) {
        return 0;
    }
    out.assign(code.begin(), code.end());
    for (u32 c = 0; c < num_compares; ++c) {
        out[compares[c]] = (5u << 16) | OpFUnordLessThanEqual;
    }
    return num_compares;
}

bool ProbeWalkNanExitEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_PROBE_WALK_NAN_EXIT");
        return value && value[0] == '1';
    }();
    return enabled;
}

} // namespace

vk::ShaderModule CompileSPV(std::span<const u32> code, vk::Device device) {
    // SHADPS4_PROBE_WALK_NAN_EXIT (above): every path compiles the modules here.
    std::vector<u32> probe_walk;
    if (ProbeWalkNanExitEnabled()) [[unlikely]] {
        if (const u32 changed = PatchProbeWalk(code, probe_walk)) {
            static std::atomic<u32> logged{0}; // compile-pool threads call this too
            if (logged.fetch_add(1) < 8) {
                LOG_INFO(Render_Vulkan,
                         "ProbeWalk: a NaN position leaves the light-probe walk ({} compare(s) "
                         "in a vertex module of {} words)",
                         changed, code.size());
            }
            code = probe_walk;
        }
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
