#pragma once

#include "device.hpp"
#include "mc/renderer/backend/vk_checker.hpp"

#include <filesystem>
#include <optional>
#include <vector>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

namespace renderer::backend
{
    struct ShaderInfo
    {
        std::filesystem::path path;
        vk::ShaderStageFlagBits stage;
        std::string entryPoint;
    };

    class PipelineLayoutConfig
    {
    public:
        auto setPushConstantSettings(uint32_t size,
                                     vk::ShaderStageFlags shaderStage) -> PipelineLayoutConfig&;

        auto
        setDescriptorSetLayouts(std::vector<vk::DescriptorSetLayout> const& layout) -> PipelineLayoutConfig&;

    private:
        std::optional<vk::PushConstantRange> pushConstants {};
        std::optional<std::vector<vk::DescriptorSetLayout>> descriptorSetLayouts {};

        friend class PipelineLayout;
    };

    class GraphicsPipelineConfig
    {
    public:
        auto addShader(std::filesystem::path const& path,
                       vk::ShaderStageFlagBits stage,
                       std::string const& entryPoint) -> GraphicsPipelineConfig&;

        auto enableBlending(bool enable = true) -> GraphicsPipelineConfig&;
        auto blendingSetAlphaBlend() -> GraphicsPipelineConfig&;
        auto blendingSetAdditiveBlend() -> GraphicsPipelineConfig&;
        auto setBlendingWriteMask(vk::ColorComponentFlagBits mask) -> GraphicsPipelineConfig&;

        auto setDepthStencilSettings(bool enable,
                                     vk::CompareOp compareOp,
                                     bool stencilEnable    = false,
                                     bool enableBoundsTest = false,
                                     bool enableWrite      = true) -> GraphicsPipelineConfig&;

        auto setPrimitiveSettings(bool primitiveRestart,
                                  vk::PrimitiveTopology primitiveTopology) -> GraphicsPipelineConfig&;

        auto enableRasterizerDiscard(bool enable = true) -> GraphicsPipelineConfig&;

        auto enableDepthClamp(bool enable = true) -> GraphicsPipelineConfig&;

        auto setLineWidth(float width) -> GraphicsPipelineConfig&;

        auto setPolygonMode(vk::PolygonMode mode) -> GraphicsPipelineConfig&;

        auto setCullingSettings(vk::CullModeFlags cullMode,
                                vk::FrontFace frontFace) -> GraphicsPipelineConfig&;

        auto setViewportScissorCount(uint32_t viewportCount,
                                     uint32_t scissorCount) -> GraphicsPipelineConfig&;

        auto setSampleShadingSettings(bool enable, float minSampleShading = 0.2f) -> GraphicsPipelineConfig&;

        auto enableAlphaToOne(bool enable = true) -> GraphicsPipelineConfig&;

        auto enableAlphaToCoverage(bool enable = true) -> GraphicsPipelineConfig&;

        auto setSampleMask(vk::SampleMask mask) -> GraphicsPipelineConfig&;

        auto setSampleCount(vk::SampleCountFlagBits count) -> GraphicsPipelineConfig&;

        auto setDepthBiasSettings(bool enable          = true,
                                  float constantFactor = 0.f,
                                  float slopeFactor    = 0.f,
                                  float clamp          = 0.f) -> GraphicsPipelineConfig&;

        auto setColorAttachmentFormat(vk::Format format) -> GraphicsPipelineConfig&;

        auto setDepthAttachmentFormat(vk::Format format) -> GraphicsPipelineConfig&;

    private:
        std::vector<ShaderInfo> shaders {};

        // Defaults
        // ********
        // Depth testing
        bool depthTestEnable         = false;
        bool depthWriteEnable        = true;
        bool depthBoundsTest         = false;
        bool stencilEnable           = false;
        vk::CompareOp depthCompareOp = vk::CompareOp::eLess;

        // IA
        bool primitiveRestart                   = false;
        vk::PrimitiveTopology primitiveTopology = vk::PrimitiveTopology::eTriangleList;

        // Rasterizer
        bool depthClampEnabled      = false;
        bool rasterizerDiscard      = false;
        bool depthBiasEnabled       = false;
        float lineWidth             = 1.f;
        vk::PolygonMode polygonMode = vk::PolygonMode::eFill;
        vk::CullModeFlags cullMode  = vk::CullModeFlagBits::eNone;
        vk::FrontFace frontFace     = vk::FrontFace::eClockwise;

        uint32_t viewportCount { 1 }, scissorCount { 1 };
        float depthBiasConstantFactor { 0.f }, depthBiasClamp { 0.f }, depthBiasSlopeFactor { 0.f };

        // multisampling
        bool sampleShadingEnable   = false;
        bool alphaToCoverageEnable = false;
        bool alphaToOneEnable      = false;

        vk::SampleCountFlagBits rasterizationSamples = vk::SampleCountFlagBits::e1;
        float minSampleShading                       = 0.3f;
        std::optional<vk::SampleMask> sampleMask;

        // blending
        bool blendingEnable = false;
        vk::ColorComponentFlags blendingColorWriteMask =
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
            vk::ColorComponentFlagBits::eA;
        vk::BlendFactor srcColorBlendFactor = vk::BlendFactor::eOne;  // Additive blending by default

        // Dynamic rendering
        std::optional<vk::Format> colorAttachmentFormat;
        std::optional<vk::Format> depthAttachmentFormat;

        friend class GraphicsPipeline;
    };

    class PipelineLayout
    {
    public:
        PipelineLayout()  = default;
        ~PipelineLayout() = default;

        explicit PipelineLayout(Device const& device, PipelineLayoutConfig const& config)
        {
            auto pipelineLayoutInfo = vk::PipelineLayoutCreateInfo();

            if (config.descriptorSetLayouts.has_value())
            {
                pipelineLayoutInfo.setSetLayouts(*config.descriptorSetLayouts);
            }

            if (config.pushConstants.has_value())
            {
                pipelineLayoutInfo.setPushConstantRanges(*config.pushConstants);
            }

            m_layout = device->createPipelineLayout(pipelineLayoutInfo) >> ResultChecker();
        }

        PipelineLayout(PipelineLayout const&)                    = delete;
        auto operator=(PipelineLayout const&) -> PipelineLayout& = delete;

        PipelineLayout(PipelineLayout&&)                    = default;
        auto operator=(PipelineLayout&&) -> PipelineLayout& = default;

        [[nodiscard]] operator vk::PipelineLayout() const { return m_layout; }

        [[nodiscard]] auto get() const -> vk::PipelineLayout { return m_layout; }

    private:
        vk::raii::PipelineLayout m_layout { nullptr };
    };

    class GraphicsPipeline
    {
    public:
        GraphicsPipeline()  = default;
        ~GraphicsPipeline() = default;

        GraphicsPipeline(Device const& device,
                         PipelineLayout const& layout,
                         GraphicsPipelineConfig const& config);

        GraphicsPipeline(GraphicsPipeline const& other)              = delete;
        auto operator=(GraphicsPipeline const&) -> GraphicsPipeline& = delete;

        GraphicsPipeline(GraphicsPipeline&&)                    = default;
        auto operator=(GraphicsPipeline&&) -> GraphicsPipeline& = default;

        [[nodiscard]] operator vk::Pipeline() const { return m_pipeline; }

        [[nodiscard]] auto get() const -> vk::Pipeline { return m_pipeline; }

    private:
        vk::raii::Pipeline m_pipeline { nullptr };
    };

    class ComputePipeline
    {
    public:
        ComputePipeline()  = default;
        ~ComputePipeline() = default;

        explicit ComputePipeline(Device const& device,
                                 PipelineLayout const& layout,
                                 std::filesystem::path const& path,
                                 std::string_view entryPoint);

        ComputePipeline(ComputePipeline const&)                    = delete;
        auto operator=(ComputePipeline const&) -> ComputePipeline& = delete;

        ComputePipeline(ComputePipeline&&)                    = default;
        auto operator=(ComputePipeline&&) -> ComputePipeline& = default;

        [[nodiscard]] auto setShader(std::filesystem::path const& path,
                                     std::string_view entryPoint) -> ComputePipeline&;

        [[nodiscard]] operator vk::Pipeline() const { return m_pipeline; }

        [[nodiscard]] auto get() const -> vk::Pipeline { return m_pipeline; }

    private:
        vk::raii::Pipeline m_pipeline { nullptr };
    };
}  // namespace renderer::backend
