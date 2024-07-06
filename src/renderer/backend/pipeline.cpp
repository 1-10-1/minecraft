#include <mc/asserts.hpp>
#include <mc/exceptions.hpp>
#include <mc/renderer/backend/pipeline.hpp>
#include <mc/renderer/backend/utils.hpp>
#include <mc/renderer/backend/vertex.hpp>
#include <mc/renderer/backend/vk_checker.hpp>
#include <mc/utils.hpp>

#include <algorithm>

#include <ranges>
#include <vulkan/vulkan_core.h>

namespace rn = std::ranges;
namespace vi = std::ranges::views;

namespace renderer::backend
{
    auto PipelineLayoutConfig::setPushConstantSettings(uint32_t size, vk::ShaderStageFlags shaderStage)
        -> PipelineLayoutConfig&
    {
        pushConstants = {
            .stageFlags = shaderStage,
            .offset     = 0,
            .size       = size,
        };

        return *this;
    }

    auto PipelineLayoutConfig::setDescriptorSetLayouts(std::vector<vk::DescriptorSetLayout> const& layout)
        -> PipelineLayoutConfig&
    {
        descriptorSetLayouts = layout;

        return *this;
    }

    auto GraphicsPipelineConfig::addShader(std::filesystem::path const& path,
                                           vk::ShaderStageFlagBits stage,
                                           std::string const& entryPoint) -> GraphicsPipelineConfig&
    {
        shaders.push_back({ path, stage, entryPoint });

        return *this;
    };

    auto GraphicsPipelineConfig::enableBlending(bool enable) -> GraphicsPipelineConfig&
    {
        blendingEnable = enable;

        return *this;
    }

    auto GraphicsPipelineConfig::blendingSetAlphaBlend() -> GraphicsPipelineConfig&
    {
        srcColorBlendFactor = vk::BlendFactor::eOneMinusDstAlpha;

        return *this;
    }

    auto GraphicsPipelineConfig::blendingSetAdditiveBlend() -> GraphicsPipelineConfig&
    {
        srcColorBlendFactor = vk::BlendFactor::eOne;

        return *this;
    }

    auto
    GraphicsPipelineConfig::setBlendingWriteMask(vk::ColorComponentFlagBits mask) -> GraphicsPipelineConfig&
    {
        blendingColorWriteMask = mask;

        return *this;
    }

    auto GraphicsPipelineConfig::setDepthStencilSettings(bool enable,
                                                         vk::CompareOp compareOp,
                                                         bool stencilEnable,
                                                         bool enableBoundsTest,
                                                         bool enableWrite) -> GraphicsPipelineConfig&
    {
        depthTestEnable     = enable;
        depthCompareOp      = compareOp;
        depthWriteEnable    = enableWrite;
        depthBoundsTest     = enableBoundsTest;
        this->stencilEnable = stencilEnable;

        return *this;
    };

    auto GraphicsPipelineConfig::setPrimitiveSettings(
        bool primitiveRestart, vk::PrimitiveTopology primitiveTopology) -> GraphicsPipelineConfig&
    {
        this->primitiveRestart  = primitiveRestart;
        this->primitiveTopology = primitiveTopology;

        return *this;
    };

    auto GraphicsPipelineConfig::enableRasterizerDiscard(bool enable) -> GraphicsPipelineConfig&
    {
        rasterizerDiscard = enable;

        return *this;
    };

    auto GraphicsPipelineConfig::enableDepthClamp(bool enable) -> GraphicsPipelineConfig&
    {
        depthClampEnabled = enable;

        return *this;
    };

    auto GraphicsPipelineConfig::setLineWidth(float width) -> GraphicsPipelineConfig&
    {
        lineWidth = width;

        return *this;
    };

    auto GraphicsPipelineConfig::setPolygonMode(vk::PolygonMode mode) -> GraphicsPipelineConfig&
    {
        polygonMode = mode;

        return *this;
    };

    auto GraphicsPipelineConfig::setCullingSettings(vk::CullModeFlags cullMode,
                                                    vk::FrontFace frontFace) -> GraphicsPipelineConfig&
    {
        this->cullMode  = cullMode;
        this->frontFace = frontFace;

        return *this;
    };

    auto GraphicsPipelineConfig::setViewportScissorCount(uint32_t viewportCount,
                                                         uint32_t scissorCount) -> GraphicsPipelineConfig&
    {
        this->viewportCount = viewportCount;
        this->scissorCount  = scissorCount;

        return *this;
    };

    auto GraphicsPipelineConfig::setSampleShadingSettings(bool enable,
                                                          float minSampleShading) -> GraphicsPipelineConfig&
    {
        sampleShadingEnable    = enable;
        this->minSampleShading = minSampleShading;

        return *this;
    };

    auto GraphicsPipelineConfig::enableAlphaToOne(bool enable) -> GraphicsPipelineConfig&
    {
        alphaToOneEnable = enable;

        return *this;
    };

    auto GraphicsPipelineConfig::enableAlphaToCoverage(bool enable) -> GraphicsPipelineConfig&
    {
        alphaToCoverageEnable = enable;

        return *this;
    };

    auto GraphicsPipelineConfig::setSampleMask(vk::SampleMask mask) -> GraphicsPipelineConfig&
    {
        sampleMask = mask;

        return *this;
    };

    auto GraphicsPipelineConfig::setSampleCount(vk::SampleCountFlagBits count) -> GraphicsPipelineConfig&
    {
        rasterizationSamples = count;

        return *this;
    };

    auto GraphicsPipelineConfig::setDepthBiasSettings(bool enable,
                                                      float constantFactor,
                                                      float slopeFactor,
                                                      float clamp) -> GraphicsPipelineConfig&
    {
        depthBiasEnabled        = enable;
        depthBiasConstantFactor = constantFactor;
        depthBiasSlopeFactor    = slopeFactor;
        depthBiasClamp          = clamp;

        return *this;
    };

    auto GraphicsPipelineConfig::setColorAttachmentFormat(vk::Format format) -> GraphicsPipelineConfig&
    {
        colorAttachmentFormat = format;

        return *this;
    };

    auto GraphicsPipelineConfig::setDepthAttachmentFormat(vk::Format format) -> GraphicsPipelineConfig&
    {
        depthAttachmentFormat = format;

        return *this;
    };

    GraphicsPipeline::GraphicsPipeline(Device const& device,
                                       PipelineLayout const& layout,
                                       GraphicsPipelineConfig const& config)
    {
        [[maybe_unused]] auto checkShaderStagePresent =
            [&shaders = config.shaders](vk::ShaderStageFlagBits stage)
        {
            return rn::find_if(shaders,
                               [stage](ShaderInfo const& info)
                               {
                                   return info.stage == stage;
                               }) != shaders.end();
        };

        // clang-format off
        MC_ASSERT_MSG(checkShaderStagePresent(vk::ShaderStageFlagBits::eVertex)
                   && checkShaderStagePresent(vk::ShaderStageFlagBits::eFragment)
                   && config.shaders.size() >= 2
                   && config.colorAttachmentFormat.has_value()
                   && config.depthAttachmentFormat.has_value(),
                      "Graphics pipeline builder was not correctly configured");
        // clang-format on

        std::array dynamicStates { vk::DynamicState::eViewport, vk::DynamicState::eScissor };

        auto dynamicState = vk::PipelineDynamicStateCreateInfo().setDynamicStates(dynamicStates);

        vk::PipelineColorBlendAttachmentState colorBlendAttachment {
            .blendEnable         = config.blendingEnable,
            .srcColorBlendFactor = config.srcColorBlendFactor,
            .dstColorBlendFactor = vk::BlendFactor::eDstAlpha,
            .colorBlendOp        = vk::BlendOp::eAdd,
            .srcAlphaBlendFactor = vk::BlendFactor::eOne,
            .dstAlphaBlendFactor = vk::BlendFactor::eZero,
            .alphaBlendOp        = vk::BlendOp::eAdd,
            .colorWriteMask      = config.blendingColorWriteMask,
        };

        vk::PipelineColorBlendStateCreateInfo colorBlending = {
            .logicOpEnable = false,
            .logicOp       = vk::LogicOp::eCopy,
        };

        colorBlending.setAttachments(colorBlendAttachment);

        vk::PipelineVertexInputStateCreateInfo vertexInput {};

        vk::PipelineDepthStencilStateCreateInfo depthStencil {
            .depthTestEnable       = config.depthTestEnable,
            .depthWriteEnable      = config.depthWriteEnable,
            .depthCompareOp        = config.depthCompareOp,
            .depthBoundsTestEnable = config.depthBoundsTest,
            .stencilTestEnable     = config.stencilEnable,
        };

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly {
            .topology               = config.primitiveTopology,
            .primitiveRestartEnable = config.primitiveRestart,
        };

        vk::PipelineViewportStateCreateInfo viewportState {
            .viewportCount = config.viewportCount,
            .scissorCount  = config.scissorCount,
        };

        vk::PipelineRasterizationStateCreateInfo rasterizer {
            .depthClampEnable        = config.depthClampEnabled,
            .rasterizerDiscardEnable = config.rasterizerDiscard,
            .polygonMode             = config.polygonMode,
            .cullMode                = config.cullMode,
            .frontFace               = config.frontFace,
            .depthBiasEnable         = config.depthBiasEnabled,
            .depthBiasConstantFactor = config.depthBiasConstantFactor,
            .depthBiasClamp          = config.depthBiasClamp,
            .depthBiasSlopeFactor    = config.depthBiasSlopeFactor,
            .lineWidth               = config.lineWidth,
        };

        vk::PipelineMultisampleStateCreateInfo multisampling {
            .rasterizationSamples  = config.rasterizationSamples,
            .sampleShadingEnable   = config.sampleShadingEnable,
            .minSampleShading      = config.minSampleShading,
            .pSampleMask           = config.sampleMask.has_value() ? &config.sampleMask.value() : nullptr,
            .alphaToCoverageEnable = config.alphaToCoverageEnable,
            .alphaToOneEnable      = config.alphaToOneEnable,
        };

        vk::PipelineRenderingCreateInfo renderInfo {
            .colorAttachmentCount    = 1,
            .pColorAttachmentFormats = &config.colorAttachmentFormat.value(),
            .depthAttachmentFormat   = config.depthAttachmentFormat.value(),
        };

        std::vector<vk::PipelineShaderStageCreateInfo> shaderStages;
        shaderStages.reserve(config.shaders.size());

        std::vector<vk::raii::ShaderModule> shaderModules;
        shaderModules.reserve(config.shaders.size());

        for (uint32_t i : vi::iota(0u, config.shaders.size()))
        {
            ShaderInfo const& info = config.shaders[i];

            shaderModules.push_back(createShaderModule(device.get(), info.path));

            shaderStages.push_back(
                { .stage = info.stage, .module = shaderModules[i], .pName = info.entryPoint.data() });
        }

        vk::GraphicsPipelineCreateInfo pipelineInfo {
            .pNext               = &renderInfo,
            .stageCount          = utils::size(shaderStages),
            .pStages             = shaderStages.data(),
            .pVertexInputState   = &vertexInput,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState      = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState   = &multisampling,
            .pDepthStencilState  = &depthStencil,
            .pColorBlendState    = &colorBlending,
            .pDynamicState       = &dynamicState,
            .layout              = layout,
            .basePipelineHandle  = nullptr,
            .basePipelineIndex   = -1,
        };

        m_pipeline = device->createGraphicsPipeline({ nullptr }, pipelineInfo) >> ResultChecker();
    };

    ComputePipeline::ComputePipeline(Device const& device,
                                     PipelineLayout const& layout,
                                     std::filesystem::path const& path,
                                     std::string_view entryPoint)
    {
        vk::raii::ShaderModule shaderModule = createShaderModule(device.get(), path);

        vk::ComputePipelineCreateInfo pipelineCreateInfo {
            .stage  = { .stage  = vk::ShaderStageFlagBits::eCompute,
                       .module = shaderModule,
                       .pName  = entryPoint.data() },
            .layout = layout,
        };

        m_pipeline = device->createComputePipeline({ nullptr }, pipelineCreateInfo) >> ResultChecker();
    }
}  // namespace renderer::backend
