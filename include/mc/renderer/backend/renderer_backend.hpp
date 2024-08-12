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

        uint32_t materialIndex { 0 };
    };
    enum class PBRWorkflows
    {
        MetallicRoughness  = 0,
        SpecularGlossiness = 1
    };

    struct alignas(16) GPUSceneData
    {
        glm::mat4 view {};
        glm::mat4 proj {};
        glm::mat4 viewproj {};

        glm::vec4 ambientColor {};

        glm::vec3 cameraPos {};
        float screenWeight {};
        glm::vec3 sunlightDirection {};
        float screenHeight {};

        vk::DeviceAddress vertexBuffer {};
        vk::DeviceAddress materialBuffer {};
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

    private:
        void initImgui(GLFWwindow* window);
        void renderImgui(vk::CommandBuffer cmdBuf, vk::ImageView targetImage);

        void drawGeometry(vk::CommandBuffer cmdBuf);

        void initDescriptors();

        void handleSurfaceResize();
        void createSyncObjects();
        void destroySyncObjects();
        void updateDescriptors(glm::vec3 cameraPos, glm::mat4 model, glm::mat4 view, glm::mat4 projection);

        void loadGltfScene();
        void renderNode(vk::CommandBuffer cmdBuf, Node* node);

        Instance m_instance;
        Surface m_surface;
        Device m_device;
        Swapchain m_swapchain;
        Allocator m_allocator;
        DescriptorAllocator m_descriptorAllocator;
        CommandManager m_commandManager;

        BasicImage m_drawImage, m_drawImageResolve, m_depthImage;
        vk::DescriptorSet m_sceneDataDescriptors { nullptr };
        vk::raii::DescriptorSetLayout m_sceneDataDescriptorLayout { nullptr },
            m_textureArrayDescriptorLayout { nullptr };

        vk::raii::DescriptorPool m_imGuiPool { nullptr };

        PipelineLayout m_pipelineLayout;
        GraphicsPipeline m_pipeline;

        GPUBuffer m_gpuSceneDataBuffer;

        Model m_scene {};

        std::array<FrameResources, kNumFramesInFlight> m_frameResources {};

        vk::raii::Sampler m_dummySampler { nullptr };
        Image m_dummyTexture {};

        Timer m_timer;

        struct EngineStats
        {
            uint64_t triangleCount;
            uint64_t drawCount;
        } m_stats {};

        int32_t m_animationIndex = 0;
        float m_animationTimer   = 0.0f;
        bool m_animate           = true;

        uint32_t m_currentFrame { 0 };

        uint64_t m_frameCount {};

        bool m_windowResized { false };
    };
}  // namespace renderer::backend
