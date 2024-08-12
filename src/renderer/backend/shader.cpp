#include <mc/asserts.hpp>
#include <mc/renderer/backend/descriptor.hpp>
#include <mc/renderer/backend/shader.hpp>
#include <mc/renderer/backend/vk_checker.hpp>
#include <mc/utils.hpp>

#include <algorithm>
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

    void ShaderManager::build()
    {
        for (auto& [path, entrypoint, shaderKind] : m_shaderDescriptions)
        {
            auto shaderStage = getShaderStageFromFile(path);

            fs::path relativePath = path.filename();

            // TODO(aether) any reason to keep it alive till the end?
            std::vector<uint32_t> spirv = compileShader(relativePath.string(),
                                                        shaderKind.value_or(getShaderKindFromFile(path)),
                                                        utils::readFileIntoString(path));

            m_shaderModules.push_back(
                m_device->get().createShaderModule(vk::ShaderModuleCreateInfo().setCode(spirv)) >>
                ResultChecker());

            spirv_cross::CompilerGLSL reflection(spirv);

            spirv_cross::ShaderResources resources = reflection.get_shader_resources();

            std::array resArray = std::to_array<
                std::pair<std::reference_wrapper<spirv_cross::SmallVector<spirv_cross::Resource>>,
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

            for (auto [i, pair] : vi::enumerate(
                     std::array<
                         std::pair<std::reference_wrapper<spirv_cross::SmallVector<spirv_cross::Resource>>,
                                   vk::DescriptorType>,
                         6> {
                         {
                          { resources.uniform_buffers, vk::DescriptorType::eUniformBuffer },
                          { resources.storage_buffers, vk::DescriptorType::eStorageBuffer },
                          { resources.storage_images, vk::DescriptorType::eStorageImage },
                          { resources.sampled_images, vk::DescriptorType::eCombinedImageSampler },
                          { resources.separate_images, vk::DescriptorType::eSampledImage },
                          { resources.separate_samplers, vk::DescriptorType::eSampler },
                          }
            }))
            {
                auto& [resourceVector, descriptorType] = pair;

                for (auto& resource : resourceVector.get())
                {
                    // FIXME(aether) ASAP These names are hella confusing

                    uint32_t bindingNumber = reflection.get_decoration(resource.id, spv::DecorationBinding);
                    uint32_t set = reflection.get_decoration(resource.id, spv::DecorationDescriptorSet);

                    DescriptorSetBindings* setInfo = nullptr;

                    if (auto it = m_descriptorSets.find(reflection.get_name(resource.id));
                        it != m_descriptorSets.end())
                    {
                        MC_ASSERT_MSG(
                            it->second.set == set,
                            "Different shaders have different identifiers for the same descriptor set ({})",
                            relativePath.string(),
                            set);

                        setInfo = &it->second;
                    }
                    else
                    {
                        setInfo      = &m_descriptorSets[reflection.get_name(resource.id)];
                        setInfo->set = set;
                    }

                    if (auto vkStruct = rn::find_if(setInfo->bindings,
                                                    [bindingNumber](vk::DescriptorSetLayoutBinding const& bi)
                                                    {
                                                        return bi.binding == bindingNumber;
                                                    });
                        vkStruct != setInfo->bindings.end())
                    {
                        MC_ASSERT_MSG(vkStruct->descriptorType == descriptorType,
                                      "Descriptor set {} binding {} uses different types across different "
                                      "shaders within the same pipeline",
                                      set,
                                      bindingNumber);

                        vkStruct->stageFlags |= shaderStage;
                    }
                    else
                    {
                        setInfo->bindings.push_back(vk::DescriptorSetLayoutBinding {
                            .binding         = bindingNumber,
                            .descriptorType  = descriptorType,
                            .descriptorCount = 1,
                            .stageFlags      = shaderStage,
                        });
                    };

                    // TODO(aether) what about dynamic uniform buffers?
                }
            }

            logger::warn("[SHADER] {}", relativePath.string());

            // CATCH UP:
            // There's a bug in the m_descriptorSets map
            // the string attached to each set is not unique to that set, it changes with each binding
            // for example the first set containing scene and light information will have different names
            // but they're both in the same set, being #0
            //
            // We're trying to set up information about descriptor set layouts but to what end? I'm not sure.
            // See what others do, check the vulkan discord server replies, see what the book does in later chapters
            for (auto& set : m_descriptorSets)
            {
                logger::info("[SET] {} ({})", set.second.set, set.first);

                for (auto& binding : set.second.bindings)
                {
                    logger::info(
                        "[BINDING] {} {}", binding.binding, magic_enum::enum_name(binding.descriptorType));
                }
            }
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
