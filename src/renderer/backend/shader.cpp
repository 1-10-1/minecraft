#include <mc/asserts.hpp>
#include <mc/renderer/backend/descriptor.hpp>
#include <mc/renderer/backend/shader.hpp>
#include <mc/renderer/backend/vk_checker.hpp>
#include <mc/utils.hpp>

#include <filesystem>
#include <iosfwd>

#include <shaderc/env.h>
#include <shaderc/shaderc.h>
#include <shaderc/shaderc.hpp>

#include <spirv_cross/spirv.hpp>
#include <spirv_cross/spirv_common.hpp>
#include <spirv_cross/spirv_glsl.hpp>

namespace renderer::backend
{
    namespace fs = std::filesystem;

    // TODO(aether) Parallelize this if shader compilation takes a lot of time
    void ShaderManager::build()
    {
        for (auto& [path, entrypoint, shaderKind] : m_shaderDescriptions)
        {
            fs::path relativePath = path.filename();

            std::vector<uint32_t> spirv = compileShader(relativePath.string(),
                                                        shaderKind.value_or(getShaderKindFromFile(path)),
                                                        utils::readFileIntoString(path));

            vk::raii::ShaderModule& module = m_shaderModules.emplace_back(
                m_device->get().createShaderModule(vk::ShaderModuleCreateInfo().setCode(spirv)) >>
                ResultChecker());

            m_shaderStageInfos.push_back(vk::PipelineShaderStageCreateInfo()
                                             .setStage(getShaderStageFromFile(path))
                                             .setPName(entrypoint.c_str())
                                             .setModule(module));
        }

        m_dirty = false;
    };

    auto ShaderManager::compileShader(std::string const& source_name,
                                      shaderc_shader_kind kind,
                                      std::string const& source,
                                      std::string_view entrypoint) -> std::vector<uint32_t>
    {
        shaderc::Compiler compiler;
        shaderc::CompileOptions options;
        options.SetOptimizationLevel(shaderc_optimization_level_performance);
        options.SetTargetSpirv(shaderc_spirv_version_1_6);
        options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
        options.SetSourceLanguage(shaderc_source_language_glsl);

        if constexpr (kDebug)
        {
            options.SetGenerateDebugInfo();
        }

        options.SetIncluder(std::make_unique<Includer>());

        shaderc::SpvCompilationResult spirvBinaryResult =
            compiler.CompileGlslToSpv(source, kind, source_name.c_str(), entrypoint.data(), options);

        MC_ASSERT_MSG(spirvBinaryResult.GetCompilationStatus() == shaderc_compilation_status_success,
                      "{}",
                      spirvBinaryResult.GetErrorMessage());

        std::vector<uint32_t> spirvBinary = { spirvBinaryResult.cbegin(), spirvBinaryResult.cend() };

        if constexpr (kDebug)
        {
            shaderc::AssemblyCompilationResult spirvAssemblyResult =
                compiler.CompileGlslToSpvAssembly(source, kind, source_name.c_str(), options);

            if (spirvAssemblyResult.GetCompilationStatus() != shaderc_compilation_status_success)
            {
                logger::warn("Failed to generate spirv assembly for shader {}: {}",
                             source_name.c_str(),
                             spirvAssemblyResult.GetErrorMessage());

                return spirvBinary;
            }

            size_t assemblyStringSize =
                std::distance(spirvAssemblyResult.cbegin(), spirvAssemblyResult.cend());

            fs::path asmFilePath = fs::current_path() / fmt::format("shaderAssemblies/{}.asm", source_name);

            if (fs::path parent = asmFilePath.parent_path(); !fs::exists(parent))
            {
                MC_ASSERT(fs::create_directory(parent));
            }

            std::ofstream stream(asmFilePath, std::ios::trunc);

            MC_ASSERT(stream.is_open());

            stream.write(spirvAssemblyResult.cbegin(), assemblyStringSize);

            stream.close();
        }

        return spirvBinary;
    }
}  // namespace renderer::backend
