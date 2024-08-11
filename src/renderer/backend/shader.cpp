#include "magic_enum.hpp"
#include "mc/renderer/backend/descriptor.hpp"
#include <algorithm>
#include <mc/asserts.hpp>
#include <mc/renderer/backend/shader.hpp>
#include <mc/utils.hpp>

#include <filesystem>
#include <iosfwd>
#include <ranges>

#include <shaderc/env.h>
#include <shaderc/shaderc.h>
#include <shaderc/shaderc.hpp>
#include <spirv_cross/spirv.hpp>
#include <spirv_cross/spirv_glsl.hpp>

namespace rn = std::ranges;
namespace vi = std::ranges::views;

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

        spirv_cross::CompilerGLSL reflection(m_spirv);

        spirv_cross::ShaderResources resources = reflection.get_shader_resources();

        std::array resArray =
            std::to_array<std::pair<std::reference_wrapper<spirv_cross::SmallVector<spirv_cross::Resource>>,
                                    std::string_view>>({
                { resources.uniform_buffers,         "uniform_buffers"         },
                { resources.storage_buffers,         "storage_buffers"         },
                { resources.stage_inputs,            "stage_inputs"            },
                { resources.stage_outputs,           "stage_outputs"           },
                { resources.subpass_inputs,          "subpass_inputs"          },
                { resources.storage_images,          "storage_images"          },
                { resources.sampled_images,          "sampled_images"          },
                { resources.atomic_counters,         "atomic_counters"         },
                { resources.acceleration_structures, "acceleration_structures" },
                { resources.gl_plain_uniforms,       "gl_plain_uniforms"       },
                { resources.push_constant_buffers,   "push_constant_buffers"   },
                { resources.shader_record_buffers,   "shader_record_buffers"   },
                { resources.separate_images,         "separate_images"         },
                { resources.separate_samplers,       "separate_samplers"       },
        });

        std::array unpreparedResources { "subpass_inputs",
                                         "atomic_counters",
                                         "gl_plain_uniforms",
                                         "shader_record_buffers",
                                         "acceleration_structures" };

        for (auto& [resource, name] : resArray)
        {
            if (auto it = rn::find_if(unpreparedResources,
                                      [&](std::string_view const& n)
                                      {
                                          return n == name;
                                      });
                it != unpreparedResources.end())
            {
                MC_ASSERT_MSG(resource.get().size() == 0, "Not prepared to deal with {}", *it);
            }
        }

        // uniform_buffers
        // push_constant_buffers
        // storage_buffers
        // storage_images
        // sampled_images
        // separate_images
        // separate_samplers

        std::array<DescriptorLayoutBuilder, 4> descriptorLayouts {};

        for (auto& [resourceVector, name] :
             std::array<std::pair<std::reference_wrapper<spirv_cross::SmallVector<spirv_cross::Resource>>,
                                  std::string_view>,
                        6> {
                 {
                  { resources.uniform_buffers, "uniform_buffers" },
                  { resources.storage_buffers, "storage_buffers" },
                  { resources.storage_images, "storage_images" },
                  { resources.sampled_images, "sampled_images" },
                  { resources.separate_images, "separate_images" },
                  { resources.separate_samplers, "separate_samplers " },
                  }
        })
        {
            for (auto& resource : resourceVector.get())
            {
                uint32_t binding = reflection.get_decoration(resource.id, spv::DecorationBinding);
                uint32_t set     = reflection.get_decoration(resource.id, spv::DecorationDescriptorSet);

                MC_ASSERT_MSG(set < 4, "Do not use more than 4 descriptor sets");

                // TODO(aether) what about dynamic uniform buffers?
                descriptorLayouts[set].addBinding(
                    binding,
                    std::unordered_map<std::string_view, vk::DescriptorType> {
                        { "uniform_buffers",   vk::DescriptorType::eUniformBuffer        },
                        { "storage_buffers",   vk::DescriptorType::eStorageBuffer        },
                        { "storage_images",    vk::DescriptorType::eStorageImage         },
                        { "sampled_images",    vk::DescriptorType::eCombinedImageSampler },
                        { "separate_images",   vk::DescriptorType::eSampledImage         },
                        { "separate_samplers", vk::DescriptorType::eSampler              },
                }[name]);
            }
        }

        std::ostringstream printStream;

        for (auto [i, layout] : vi::enumerate(descriptorLayouts))
        {
            if (layout.bindings.empty())
            {
                continue;
            }

            printStream << "Descriptor set " << i << ":\n";

            for (auto [j, binding] : vi::enumerate(layout.bindings))
            {
                printStream << binding.binding << ": " << binding.descriptorCount << " "
                            << magic_enum::enum_name(binding.descriptorType) << "s on stage "
                            << static_cast<uint32_t>(binding.stageFlags) << '\n';
            }
        }

        logger::debug("[DESCRIPTORS: {}]\n{}", relativePath.string(), printStream.view());
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
