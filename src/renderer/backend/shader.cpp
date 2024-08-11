#include <filesystem>
#include <mc/asserts.hpp>
#include <mc/renderer/backend/shader.hpp>
#include <mc/utils.hpp>

#include <shaderc/env.h>
#include <shaderc/shaderc.h>
#include <shaderc/shaderc.hpp>

namespace renderer::backend
{
    namespace fs = std::filesystem;

    ShaderCode::ShaderCode(fs::path path,
                           std::string_view entrypoint,
                           std::optional<shaderc_shader_kind> stage)
    {
        fs::path relativePath = path.filename();

        std::ifstream file(path, std::ios::ate | std::ios::binary);

        MC_ASSERT_MSG(file.is_open(), "Failed to read file '{}'", relativePath.string());

        auto fileSize = file.tellg();
        file.seekg(0);

        std::string buffer;
        buffer.resize(static_cast<std::size_t>(fileSize));

        file.read(reinterpret_cast<char*>(buffer.data()), fileSize);

        file.close();

        m_spirv = compileShader(relativePath.string(), stage.value_or(getShaderKindFromFile(path)), buffer);
    };

    auto ShaderCode::compileShader(std::string const& source_name,
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

        // Like -DMY_DEFINE=1
        options.AddMacroDefinition("MY_DEFINE", "1");

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
