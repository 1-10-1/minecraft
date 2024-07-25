#include "mc/renderer/backend/allocator.hpp"
#include <chrono>
#include <mc/asserts.hpp>
#include <mc/exceptions.hpp>
#include <mc/logger.hpp>
#include <mc/renderer/backend/command.hpp>
#include <mc/renderer/backend/constants.hpp>
#include <mc/renderer/backend/descriptor.hpp>
#include <mc/renderer/backend/image.hpp>
#include <mc/renderer/backend/info_structs.hpp>
#include <mc/renderer/backend/pipeline.hpp>
#include <mc/renderer/backend/renderer_backend.hpp>
#include <mc/renderer/backend/utils.hpp>
#include <mc/renderer/backend/vk_checker.hpp>
#include <mc/renderer/backend/vk_result_messages.hpp>
#include <mc/timer.hpp>
#include <mc/utils.hpp>

#include <filesystem>
#include <print>

#include <glm/ext.hpp>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <tracy/Tracy.hpp>
#include <tracy/TracyVulkan.hpp>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_structs.hpp>

namespace
{
    using namespace renderer::backend;

    [[maybe_unused]] void imguiCheckerFn(vk::Result result,
                                         std::source_location location = std::source_location::current())
    {
        MC_ASSERT_LOC(result == vk::Result::eSuccess, location);
    }
}  // namespace

namespace renderer::backend
{
    RendererBackend::RendererBackend(window::Window& window)
        // clang_format off
        : m_surface { window, m_instance },

          m_device { m_instance, m_surface },

          m_swapchain { m_device, m_surface },

          m_allocator { m_instance, m_device },

          m_commandManager { m_device },

          m_drawImage { m_device,
                        m_allocator,
                        m_surface.getFramebufferExtent(),
                        vk::Format::eR16G16B16A16Sfloat,
                        m_device.getMaxUsableSampleCount(),
                        vk::ImageUsageFlagBits::eTransferSrc |
                            vk::ImageUsageFlagBits::eTransferDst |  // maybe remove?
                            vk::ImageUsageFlagBits::eColorAttachment,
                        vk::ImageAspectFlagBits::eColor },

          m_drawImageResolve { m_device,
                               m_allocator,
                               m_drawImage.getDimensions(),
                               m_drawImage.getFormat(),
                               vk::SampleCountFlagBits::e1,
                               vk::ImageUsageFlagBits::eColorAttachment |
                                   vk::ImageUsageFlagBits::eTransferSrc |
                                   vk::ImageUsageFlagBits::eTransferDst,
                               vk::ImageAspectFlagBits::eColor },

          m_depthImage { m_device,
                         m_allocator,
                         m_drawImage.getDimensions(),
                         kDepthStencilFormat,
                         m_device.getMaxUsableSampleCount(),
                         vk::ImageUsageFlagBits::eDepthStencilAttachment,
                         vk::ImageAspectFlagBits::eDepth }
    // clang_format on
    {
        initImgui(window.getHandle());

        m_dummySampler = m_device->createSampler({
                             .magFilter               = vk::Filter::eNearest,
                             .minFilter               = vk::Filter::eNearest,
                             .mipmapMode              = vk::SamplerMipmapMode::eLinear,
                             .addressModeU            = vk::SamplerAddressMode::eRepeat,
                             .addressModeV            = vk::SamplerAddressMode::eRepeat,
                             .addressModeW            = vk::SamplerAddressMode::eRepeat,
                             .mipLodBias              = 0.0f,
                             .anisotropyEnable        = false,
                             .maxAnisotropy           = 0,
                             .compareEnable           = false,
                             .minLod                  = 0.0f,
                             .maxLod                  = 1,
                             .borderColor             = vk::BorderColor::eIntOpaqueBlack,
                             .unnormalizedCoordinates = false,
                         }) >>
                         ResultChecker();

        {
            uint32_t dark  = 0xFF111111;
            uint32_t light = 0xFF777777;

            std::vector<uint32_t> pixels(32 * 32);

            for (int x = 0; x < 32; x++)
            {
                for (int y = 0; y < 32; y++)
                {
                    pixels[y * 32 + x] = ((x % 2) ^ (y % 2)) ? light : dark;
                }
            }

            m_dummyTexture = Image(m_device,
                                   m_allocator,
                                   m_commandManager,
                                   vk::Extent2D { 32, 32 },
                                   pixels.data(),
                                   sizeof(float) * pixels.size());
        }

        m_gpuSceneDataBuffer = GPUBuffer(m_allocator,
                                         sizeof(GPUSceneData),
                                         vk::BufferUsageFlagBits::eUniformBuffer,
                                         VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                         VMA_ALLOCATION_CREATE_MAPPED_BIT |
                                             VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

        m_lightDataBuffer = GPUBuffer(m_allocator,
                                      sizeof(Light),
                                      vk::BufferUsageFlagBits::eUniformBuffer,
                                      VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      VMA_ALLOCATION_CREATE_MAPPED_BIT |
                                          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

        initDescriptors();

        auto pipelineLayoutConfig =
            PipelineLayoutConfig()
                .setDescriptorSetLayouts({ m_sceneDataDescriptorLayout, m_materialDescriptorLayout })
                .setPushConstantSettings(sizeof(GPUDrawPushConstants),
                                         vk::ShaderStageFlagBits::eVertex |
                                             vk::ShaderStageFlagBits::eFragment);

        m_pipelineLayout = PipelineLayout(m_device, pipelineLayoutConfig);

        {
            auto pipelineConfig =
                GraphicsPipelineConfig()
                    .addShader("shaders/fs.frag.spv", vk::ShaderStageFlagBits::eFragment, "main")
                    .addShader("shaders/vs.vert.spv", vk::ShaderStageFlagBits::eVertex, "main")
                    .setColorAttachmentFormat(m_drawImage.getFormat())
                    .setDepthAttachmentFormat(kDepthStencilFormat)
                    .setDepthStencilSettings(true, vk::CompareOp::eGreaterOrEqual)
                    // .setCullingSettings(vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise)
                    .setSampleCount(m_device.getMaxUsableSampleCount())
                    .setSampleShadingSettings(true, 0.1f);

            m_pipeline = GraphicsPipeline(m_device, m_pipelineLayout, pipelineConfig);
        }

        loadGltfScene();

        m_light = {
            .position    = { 0.67f,                 0.42f,             0.6f             },
            .color       = { 1.f,                   1.f,               1.f              },
            .attenuation = { .quadratic = 0.00007f, .linear = 0.0014f, .constant = 1.0f },
        };

#if PROFILED
        for (size_t i : vi::iota(0u, utils::size(m_frameResources)))
        {
            std::string ctxName = fmt::format("Frame {}/{}", i + 1, kNumFramesInFlight);

            auto& ctx = m_frameResources[i].tracyContext;

            ctx = TracyVkContextCalibrated(
                *m_device.getPhysical(),
                *m_device.get(),
                *m_device.getGraphicsQueue(),
                *m_commandManager.getGraphicsCmdBuffer(i),
                reinterpret_cast<PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT>(vkGetInstanceProcAddr(
                    *m_instance.get(), "vkGetPhysicalDeviceCalibratableTimeDomainsEXT")),
                reinterpret_cast<PFN_vkGetCalibratedTimestampsEXT>(vkGetInstanceProcAddr(
                    *m_instance.get(), "vkGetPhysicalDeviceCalibratableTimeDomainsEXT")));

            TracyVkContextName(ctx, ctxName.data(), ctxName.size());
        }
#endif

        createSyncObjects();
    }

    RendererBackend::~RendererBackend()
    {
        if (!*m_instance.get())
        {
            return;
        }

        m_device->waitIdle();

        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

#if PROFILED
        for (auto& resource : m_frameResources)
        {
            TracyVkDestroy(resource.tracyContext);
        }
#endif
    }

    void RendererBackend::loadGltfScene()
    {
        m_scene = Model(m_device,
                        m_allocator,
                        m_commandManager,
                        m_materialDescriptorLayout,
                        m_dummyTexture.getImageView(),
                        m_dummySampler);

        auto glTFFile = std::filesystem::path(
            std::format("../../gltfSampleAssets/Models/{0}/glTF/{0}.gltf", "ABeautifulGame"));

        logger::info("Loading scene from {}..", glTFFile.c_str());

        m_animationIndex = 0;
        m_animationTimer = 0.0f;

        auto timerStart = std::chrono::high_resolution_clock::now();

        m_scene.loadFromFile(glTFFile);

        auto timeTaken = std::chrono::duration<double, std::ratio<1, 1>>(
                             std::chrono::high_resolution_clock::now() - timerStart)
                             .count();

        logger::info("Took {:.2f}s", timeTaken);

        // Check and list unsupported extensions
        std::stringstream unsupportedExts;

        for (auto [i, ext] : vi::enumerate(m_scene.extensions))
        {
            if (std::find(m_scene.supportedExtensions.begin(), m_scene.supportedExtensions.end(), ext) ==
                m_scene.supportedExtensions.end())
            {
                unsupportedExts << ext;

                // Last iteration
                if (i == m_scene.extensions.size() - 1)
                {
                    logger::warn(
                        "Unsupported extension(s) detected: {}\nScene may not work or display as intended.",
                        unsupportedExts.str());
                }
                else
                {
                    unsupportedExts << ", ";
                }
            }
        }
    }

    void RendererBackend::initDescriptors()
    {
        {
            std::vector<DescriptorAllocator::PoolSizeRatio> sizes = {
                { vk::DescriptorType::eUniformBuffer, 2 },
            };

            m_descriptorAllocator = DescriptorAllocator(m_device, 1, sizes);
        }

        {
            m_sceneDataDescriptorLayout =
                DescriptorLayoutBuilder()
                    // The scene data buffer
                    .addBinding(0, vk::DescriptorType::eUniformBuffer)
                    // The light data buffer
                    .addBinding(1, vk::DescriptorType::eUniformBuffer)
                    .build(m_device, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);
        }

        {
            m_materialDescriptorLayout =
                DescriptorLayoutBuilder()
                    .addBinding(0, vk::DescriptorType::eCombinedImageSampler)  // diffuse
                    .addBinding(1, vk::DescriptorType::eCombinedImageSampler)  // roughness
                    .addBinding(2, vk::DescriptorType::eCombinedImageSampler)  // occlusion
                    .addBinding(3, vk::DescriptorType::eCombinedImageSampler)  // emissive
                    .addBinding(4, vk::DescriptorType::eCombinedImageSampler)  // normal
                    .build(m_device, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);
        }

        m_sceneDataDescriptors = m_descriptorAllocator.allocate(m_device, m_sceneDataDescriptorLayout);

        {
            // Global scene data descriptor set
            DescriptorWriter writer;

            writer.write_buffer(
                0, m_gpuSceneDataBuffer, sizeof(GPUSceneData), 0, vk::DescriptorType::eUniformBuffer);
            writer.write_buffer(1, m_lightDataBuffer, sizeof(Light), 0, vk::DescriptorType::eUniformBuffer);
            writer.update_set(m_device, m_sceneDataDescriptors);
        }
    }

    void RendererBackend::initImgui(GLFWwindow* window)
    {
        std::array poolSizes {
            vk::DescriptorPoolSize()
                .setType(vk::DescriptorType::eCombinedImageSampler)
                .setDescriptorCount(kNumFramesInFlight),
        };

        vk::DescriptorPoolCreateInfo poolInfo {
            .flags   = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = kNumFramesInFlight,
        };

        poolInfo.setPoolSizes(poolSizes);

        m_imGuiPool = m_device->createDescriptorPool(poolInfo) >> ResultChecker();

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        (void)io;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;      // Enable Docking
        io.ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;

        io.IniFilename = nullptr;

        ImGui::StyleColorsDark();

        ImGuiStyle& style = ImGui::GetStyle();

        style.WindowRounding = 8.0f;

        ImGui_ImplGlfw_InitForVulkan(window, true);

        ImGui::SetColorEditOptions(ImGuiColorEditFlags_Float | ImGuiColorEditFlags_DisplayRGB |
                                   ImGuiColorEditFlags_PickerHueBar);

        ImGui_ImplVulkan_InitInfo initInfo {
            .Instance                    = *m_instance.get(),
            .PhysicalDevice              = *m_device.getPhysical(),
            .Device                      = *m_device.get(),
            .QueueFamily                 = m_device.getQueueFamilyIndices().graphicsFamily,
            .Queue                       = *m_device.getGraphicsQueue(),
            .DescriptorPool              = *m_imGuiPool,
            .MinImageCount               = kNumFramesInFlight,
            .ImageCount                  = utils::size(m_swapchain.getImageViews()),
            .MSAASamples                 = VK_SAMPLE_COUNT_1_BIT,
            .UseDynamicRendering         = true,
            .PipelineRenderingCreateInfo = vk::PipelineRenderingCreateInfo()
                                               .setColorAttachmentFormats(m_surface.getDetails().format)
                                               .setDepthAttachmentFormat(m_depthImage.getFormat()),
            .CheckVkResultFn = kDebug ? reinterpret_cast<void (*)(VkResult)>(&imguiCheckerFn) : nullptr,
        };

        ImGui_ImplVulkan_Init(&initInfo);

        [[maybe_unused]] ImFont* font = io.Fonts->AddFontFromFileTTF(
            "./res/fonts/JetBrainsMonoNerdFont-Bold.ttf", 20.0f, nullptr, io.Fonts->GetGlyphRangesDefault());

        MC_ASSERT(font != nullptr);

        ImGui_ImplVulkan_CreateFontsTexture();
    }

    void RendererBackend::update(glm::vec3 cameraPos, glm::mat4 view, glm::mat4 projection)
    {
        ZoneScopedN("Backend update");

        m_timer.tick();

        // float radius = 5.0f;

        // m_light.position = {
        //     radius * glm::fastCos(glm::radians(
        //                  static_cast<float>(m_timer.getTotalTime<Timer::Seconds>().count()) * 90.f)),
        //     0,
        //     radius * glm::fastSin(glm::radians(
        //                  static_cast<float>(m_timer.getTotalTime<Timer::Seconds>().count()) * 90.f)),
        // };

        // for (RenderItem& item : m_renderItems |
        //                             rn::views::filter(
        //                                 [](auto const& pair)
        //                                 {
        //                                     return pair.first == "light";
        //                                 }) |
        //                             rn::views::values)
        // {
        //     item.model = glm::scale(glm::identity<glm::mat4>(), { 0.25f, 0.25f, 0.25f }) *
        //                  glm::translate(glm::identity<glm::mat4>(), m_light.position);
        // }

        updateDescriptors(cameraPos, glm::identity<glm::mat4>(), view, projection);
    }

    void RendererBackend::createSyncObjects()
    {
        for (FrameResources& frame : m_frameResources)
        {
            frame.imageAvailableSemaphore = m_device->createSemaphore({}) >> ResultChecker();
            frame.renderFinishedSemaphore = m_device->createSemaphore({}) >> ResultChecker();
            frame.inFlightFence =
                m_device->createFence(vk::FenceCreateInfo().setFlags(vk::FenceCreateFlagBits::eSignaled)) >>
                ResultChecker();
        }
    }

    void RendererBackend::scheduleSwapchainUpdate()
    {
        m_windowResized = true;
    }

    void RendererBackend::handleSurfaceResize()
    {
        m_device->waitIdle();

        m_swapchain = Swapchain(m_device, m_surface);

        m_drawImage.resize(m_surface.getFramebufferExtent());
        m_drawImageResolve.resize(m_surface.getFramebufferExtent());
        m_depthImage.resize(m_surface.getFramebufferExtent());
    }

    void RendererBackend::updateDescriptors(glm::vec3 cameraPos,
                                            glm::mat4 model,
                                            glm::mat4 view,
                                            glm::mat4 projection)
    {
        auto& sceneUniformData = *static_cast<GPUSceneData*>(m_gpuSceneDataBuffer.getMappedData());
        auto& lightUniformData = *static_cast<Light*>(m_lightDataBuffer.getMappedData());

        sceneUniformData = GPUSceneData {
            .view              = view,
            .proj              = projection,
            .viewproj          = projection * view,
            .ambientColor      = glm::vec4(.1f),
            .cameraPos         = cameraPos,
            .screenWeight      = static_cast<float>(m_drawImage.getDimensions().width),
            .sunlightDirection = glm::vec3 { -0.2f, -1.0f, -0.3f },
            .screenHeight      = static_cast<float>(m_drawImage.getDimensions().height),
            .vertexBuffer      = m_scene.vertexBufferAddress,
            .materialBuffer    = m_scene.materialBufferAddress,
        };

        lightUniformData = m_light;
    }
}  // namespace renderer::backend
