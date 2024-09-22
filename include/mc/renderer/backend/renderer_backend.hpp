#pragma once

#include "allocator.hpp"
#include "buffer.hpp"
#include "command.hpp"
#include "constants.hpp"
#include "descriptor.hpp"
#include "device.hpp"
#include "gltf/loader.hpp"
#include "image.hpp"
#include "instance.hpp"
#include "pipeline.hpp"
#include "surface.hpp"
#include "swapchain.hpp"
#include "texture.hpp"

#include <GLFW/glfw3.h>
#include <TaskScheduler.h>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/vector_uint2.hpp>
#include <glm/mat4x4.hpp>
#include <tracy/TracyVulkan.hpp>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

namespace renderer::backend
{
    struct GPUDrawPushConstants
    {
        vk::DeviceAddress vertexBuffer {};
        vk::DeviceAddress materialBuffer {};
        vk::DeviceAddress primitiveBuffer {};
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

        RendererBackend(RendererBackend&&)                    = delete;
        auto operator=(RendererBackend&&) -> RendererBackend& = delete;

        RendererBackend(RendererBackend const&)                    = delete;
        auto operator=(RendererBackend const&) -> RendererBackend& = delete;

        ~RendererBackend();

        void render();
        void update(glm::vec3 cameraPos, glm::mat4 view, glm::mat4 projection);

        void queueTextureUpdate(ResourceHandle const& texture);

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

        uint32_t getCurrentFrameIndex() const { return m_currentFrame; }

    private:
        void initImgui(GLFWwindow* window);
        void renderImgui(vk::CommandBuffer cmdBuf, vk::ImageView targetImage);
        void recordCommandBuffer(uint32_t imageIndex);

        void drawGeometry(vk::CommandBuffer cmdBuf);

        void initDescriptors();

        void handleSurfaceResize();
        void createSyncObjects();
        void destroySyncObjects();
        void updateDescriptors(glm::vec3 cameraPos, glm::mat4 model, glm::mat4 view, glm::mat4 projection);

        void loadGltfScene();
        void renderNode(vk::CommandBuffer cmdBuf, Node* node);

        enki::TaskScheduler m_scheduler;

        Instance m_instance;
        Surface m_surface;
        Device m_device;
        Swapchain m_swapchain;
        Allocator m_allocator;
        DescriptorAllocator m_descriptorAllocator;
        CommandManager m_commandManager;

        ResourceManager<GPUBuffer> m_buffers;
        ResourceManager<Image> m_images;
        ResourceManager<Texture> m_textures;

        ResourceAccessor<Image> m_drawImage {}, m_drawImageResolve {}, m_depthImage {};
        vk::DescriptorSet m_sceneDataDescriptors { nullptr };
        vk::raii::DescriptorSetLayout m_sceneDataDescriptorLayout { nullptr },
            m_textureArrayDescriptorLayout { nullptr };

        vk::raii::DescriptorPool m_imGuiPool { nullptr };

        PipelineLayout m_pipelineLayout;
        GraphicsPipeline m_pipeline;

        ResourceAccessor<GPUBuffer> m_gpuSceneDataBuffer {};

        Model m_scene {};

        std::array<FrameResources, kNumFramesInFlight> m_frameResources {};

        std::vector<ResourceHandle> m_texturesToUpdate;
        std::mutex m_texturesUpdateMutex;
        uint32_t m_numTexturesToUpdate;

        vk::raii::Sampler m_dummySampler { nullptr };
        ResourceAccessor<Texture> m_dummyTexture {};

        Timer m_timer;

        struct EngineStats
        {
            uint64_t triangleCount;
            uint64_t drawCount;
        } m_stats {};

        uint32_t m_currentFrame  = 0;
        int32_t m_animationIndex = 0;

        uint64_t m_frameCount {};

        float m_animationTimer = 0.0f;

        bool m_animate = true;

        bool m_windowResized = false;
    };
}  // namespace renderer::backend
