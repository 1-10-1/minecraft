#pragma once

#include "allocator.hpp"
#include "buffer.hpp"
#include "command.hpp"
#include "constants.hpp"
#include "descriptor.hpp"
#include "device.hpp"
#include "image.hpp"
#include "instance.hpp"
#include "mc/renderer/backend/gltfloader.hpp"
#include "pipeline.hpp"
#include "surface.hpp"
#include "swapchain.hpp"

#include "vk_mem_alloc.h"
#include <GLFW/glfw3.h>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/vector_uint2.hpp>
#include <glm/mat4x4.hpp>
#include <tracy/TracyVulkan.hpp>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

namespace renderer::backend
{
    struct GPUDrawPushConstants
    {
        glm::mat4 model { glm::identity<glm::mat4>() };

        vk::DeviceAddress vertexBuffer {};
        vk::DeviceAddress materialBuffer {};

        uint32_t materialIndex { 0 };
    };

    struct alignas(16) GPUSceneData
    {
        glm::mat4 view;
        glm::mat4 proj;
        glm::mat4 viewproj;
        glm::vec4 ambientColor;
        glm::vec3 cameraPos;
        float pad;
        glm::vec3 sunlightDirection;
    };

    struct FrameResources
    {
        vk::raii::Semaphore imageAvailableSemaphore { nullptr };
        vk::raii::Semaphore renderFinishedSemaphore { nullptr };
        vk::raii::Fence inFlightFence { nullptr };

#if PROFILED
        TracyVkCtx tracyContext { nullptr };
#endif
    };

    struct alignas(16) Light
    {
        glm::vec3 position;
        float pad1;

        glm::vec3 color;
        float pad2;

        struct AttenuationFactors
        {
            float quadratic, linear, constant;
        } attenuation;
    };

    class RendererBackend
    {
    public:
        explicit RendererBackend(window::Window& window);

        RendererBackend(RendererBackend const&)                    = delete;
        RendererBackend(RendererBackend&&)                         = delete;
        auto operator=(RendererBackend const&) -> RendererBackend& = delete;
        auto operator=(RendererBackend&&) -> RendererBackend&      = delete;

        ~RendererBackend();

        void render();
        void update(glm::vec3 cameraPos, glm::mat4 view, glm::mat4 projection);
        void recordCommandBuffer(uint32_t imageIndex);

        void scheduleSwapchainUpdate();

        [[nodiscard]] auto getFramebufferSize() const -> glm::uvec2
        {
            vk::Extent2D extent = m_swapchain.getImageExtent();
            return { extent.width, extent.height };
        }

        void toggleVsync()
        {
            m_surface.scheduleVsyncChange(!m_surface.getVsync());
            scheduleSwapchainUpdate();
        }

        void toggleLightRevolution() { m_timer.isPaused() ? m_timer.unpause() : m_timer.pause(); }

    private:
        void initImgui(GLFWwindow* window);
        void renderImgui(vk::CommandBuffer cmdBuf, vk::ImageView targetImage);

        void drawGeometry(vk::CommandBuffer cmdBuf);

        void initDescriptors();

        void processGltf();

        void loadImages(tinygltf::Model& input);

        void loadTextures(tinygltf::Model& input);

        void loadMaterials(tinygltf::Model& input);

        void loadNode(tinygltf::Node const& inputNode,
                      tinygltf::Model const& input,
                      GltfNode* parent,
                      std::vector<uint32_t>& indexBuffer,
                      std::vector<Vertex>& vertexBuffer);

        void drawNode(vk::CommandBuffer commandBuffer, vk::PipelineLayout pipelineLayout, GltfNode* node);

        void drawGltf(vk::CommandBuffer commandBuffer, vk::PipelineLayout pipelineLayout);

        void handleSurfaceResize();
        void createSyncObjects();
        void destroySyncObjects();
        void updateDescriptors(glm::vec3 cameraPos, glm::mat4 model, glm::mat4 view, glm::mat4 projection);

        Instance m_instance;
        Surface m_surface;
        Device m_device;
        Swapchain m_swapchain;
        Allocator m_allocator;
        DescriptorAllocator m_descriptorAllocator;
        CommandManager m_commandManager;

        Image m_drawImage, m_drawImageResolve, m_depthImage;
        vk::DescriptorSet m_sceneDataDescriptors { nullptr };
        vk::raii::DescriptorSetLayout m_sceneDataDescriptorLayout { nullptr },
            m_materialDescriptorLayout { nullptr };

        vk::raii::DescriptorPool m_imGuiPool { nullptr };

        PipelineLayout m_texturedPipelineLayout, m_texturelessPipelineLayout;
        GraphicsPipeline m_texturedPipeline, m_texturelessPipeline;

        GPUSceneData m_sceneData {};
        GPUBuffer m_gpuSceneDataBuffer, m_lightDataBuffer;

        SceneResources m_sceneResources {};

        std::array<FrameResources, kNumFramesInFlight> m_frameResources {};

        vk::raii::Sampler m_dummySampler { nullptr };
        Texture m_dummyTexture {};

        Timer m_timer;

        Light m_light {};

        struct EngineStats
        {
            uint64_t triangle_count;
            uint64_t drawcall_count;
        } m_stats {};

        uint32_t m_currentFrame { 0 };

        bool m_windowResized { false };

        uint64_t m_frameCount {};
    };
}  // namespace renderer::backend
