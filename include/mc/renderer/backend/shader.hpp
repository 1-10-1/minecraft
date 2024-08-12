#pragma once

#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <vector>

#include "device.hpp"
#include "mc/asserts.hpp"

#define SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS
#include <shaderc/shaderc.h>
#include <shaderc/shaderc.hpp>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

namespace renderer::backend
{
    inline shaderc_shader_kind getShaderKindFromFile(std::filesystem::path const& path)
    {
        return std::unordered_map<std::string_view, shaderc_shader_kind> {
            { ".vert",  shaderc_shader_kind::shaderc_vertex_shader          },
            { ".tesc",  shaderc_shader_kind::shaderc_tess_control_shader    },
            { ".tese",  shaderc_shader_kind::shaderc_tess_evaluation_shader },
            { ".geom",  shaderc_shader_kind::shaderc_geometry_shader        },
            { ".frag",  shaderc_shader_kind::shaderc_fragment_shader        },
            { ".comp",  shaderc_shader_kind::shaderc_compute_shader         },
            { ".rgen",  shaderc_shader_kind::shaderc_raygen_shader          },
            { ".rint",  shaderc_shader_kind::shaderc_intersection_shader    },
            { ".rahit", shaderc_shader_kind::shaderc_anyhit_shader          },
            { ".rchit", shaderc_shader_kind::shaderc_closesthit_shader      },
            { ".rmiss", shaderc_shader_kind::shaderc_miss_shader            },
            { ".rcall", shaderc_shader_kind::shaderc_callable_shader        },
            { ".mesh",  shaderc_shader_kind::shaderc_mesh_shader            },
            { ".task",  shaderc_shader_kind::shaderc_task_shader            },
        }[path.extension().string()];
    };

    inline vk::ShaderStageFlagBits getShaderStageFromFile(std::filesystem::path const& path)
    {
        return std::unordered_map<std::string_view, vk::ShaderStageFlagBits> {
            { ".vert",  vk::ShaderStageFlagBits::eVertex                 },
            { ".tesc",  vk::ShaderStageFlagBits::eTessellationControl    },
            { ".tese",  vk::ShaderStageFlagBits::eTessellationEvaluation },
            { ".geom",  vk::ShaderStageFlagBits::eGeometry               },
            { ".frag",  vk::ShaderStageFlagBits::eFragment               },
            { ".comp",  vk::ShaderStageFlagBits::eCompute                },
            { ".rgen",  vk::ShaderStageFlagBits::eRaygenKHR              },
            { ".rint",  vk::ShaderStageFlagBits::eIntersectionKHR        },
            { ".rahit", vk::ShaderStageFlagBits::eAnyHitKHR              },
            { ".rchit", vk::ShaderStageFlagBits::eClosestHitKHR          },
            { ".rmiss", vk::ShaderStageFlagBits::eMissKHR                },
            { ".rcall", vk::ShaderStageFlagBits::eCallableKHR            },
            { ".mesh",  vk::ShaderStageFlagBits::eMeshEXT                },
            { ".task",  vk::ShaderStageFlagBits::eTaskEXT                },
        }[path.extension().string()];
    };

    class Includer final : public shaderc::CompileOptions::IncluderInterface
    {
    public:
        shaderc_include_result* GetInclude(char const* requested_source,
                                           shaderc_include_type type,
                                           char const* requesting_source,
                                           size_t include_depth) override
        {
            // TODO(aether) do something about this hardcoding
            auto path = std::filesystem::path("../../shaders/") / requested_source;

            // FIXME(aether)
            MC_ASSERT_MSG(include_depth == 1 && type == shaderc_include_type_relative,
                          "Not prepared to handle this :/");

            auto& result = m_includeResults[requested_source];

            std::ifstream file(path, std::ios::ate | std::ios::binary);

            MC_ASSERT_MSG(file.is_open(),
                          "{} can't be opened (included by shader {})",
                          requested_source,
                          requesting_source);

            auto fileSize = file.tellg();
            file.seekg(0);

            std::string& includeContent = m_includeContents.emplace_back();
            includeContent.resize(static_cast<std::size_t>(fileSize));

            file.read(reinterpret_cast<char*>(includeContent.data()), fileSize);

            file.close();

            result.content            = includeContent.data();
            result.source_name        = requested_source;
            result.content_length     = includeContent.size();
            result.source_name_length = std::strlen(requested_source);

            return &result;
        };

        void ReleaseInclude(shaderc_include_result* data) override
        {
            m_includeResults.erase(data->source_name);
        };

        virtual ~Includer() = default;

    private:
        // TODO(aether) these aren't released by ReleaseInclude(data)
        // use shaderc_include_result::userdata for this
        // Suggestion: cache includes by path as ambient side effects are not a thing here
        std::vector<std::string> m_includeContents;

        std::unordered_map<std::string, shaderc_include_result> m_includeResults;
    };

    struct ShaderDescription
    {
        std::filesystem::path path;
        std::string entrypoint {};
        std::optional<shaderc_shader_kind> shaderKind {};
    };

    struct DescriptorSetBindings
    {
        uint32_t set;
        std::vector<vk::DescriptorSetLayoutBinding> bindings;
    };

    class ShaderManager
    {
    public:
        ShaderManager()  = default;
        ~ShaderManager() = default;

        ShaderManager(Device& device) : m_device { &device } {};

        ShaderManager(ShaderManager&&)                    = default;
        auto operator=(ShaderManager&&) -> ShaderManager& = default;

        ShaderManager(ShaderManager const&)                    = delete;
        auto operator=(ShaderManager const&) -> ShaderManager& = delete;

        ShaderManager& addShader(std::string path,
                                 std::string_view entrypoint                   = {},
                                 std::optional<shaderc_shader_kind> shaderKind = {})
        {
            m_dirty = true;

            m_shaderDescriptions.push_back({ std::filesystem::path("../../shaders/") / path.c_str(),
                                             entrypoint.empty() ? "main" : entrypoint.data(),
                                             std::move(shaderKind) });

            return *this;
        };

        void build();

        std::vector<vk::PipelineShaderStageCreateInfo> const& getShaderStages() const
        {
            MC_ASSERT(!m_dirty);

            return m_shaderStageInfos;
        };

    private:
        auto compileShader(std::string const& source_name,
                           shaderc_shader_kind kind,
                           std::string const& source,
                           std::string_view entrypoint = "main") -> std::vector<uint32_t>;

        Device* m_device { nullptr };

        // This boolean is checked in the getter functions to ensure
        // we dont ask for things before ensuring everything is up-to-date
        bool m_dirty = true;

        std::vector<ShaderDescription> m_shaderDescriptions;

        std::vector<vk::raii::ShaderModule> m_shaderModules {};
        std::vector<vk::PipelineShaderStageCreateInfo> m_shaderStageInfos {};
    };
}  // namespace renderer::backend
