#include <mc/asserts.hpp>
#include <mc/exceptions.hpp>
#include <mc/logger.hpp>
#include <mc/renderer/backend/allocator.hpp>
#include <mc/renderer/backend/command.hpp>
#include <mc/renderer/backend/constants.hpp>
#include <mc/renderer/backend/descriptor.hpp>
#include <mc/renderer/backend/image.hpp>
#include <mc/renderer/backend/info_structs.hpp>
#include <mc/renderer/backend/pipeline.hpp>
#include <mc/renderer/backend/renderer_backend.hpp>
#include <mc/renderer/backend/shader.hpp>
#include <mc/renderer/backend/utils.hpp>
#include <mc/renderer/backend/vk_checker.hpp>
#include <mc/renderer/backend/vk_result_messages.hpp>
#include <mc/timer.hpp>
#include <mc/utils.hpp>

#include <chrono>
#include <filesystem>
#include <print>

#include <glm/ext.hpp>
#include <glslang/Public/ShaderLang.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <stb_image.h>
#include <tracy/Tracy.hpp>
#include <tracy/TracyVulkan.hpp>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
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
        : m_surface { window, m_instance },

          m_device { m_instance, m_surface },

          m_swapchain { m_device, m_surface },

          m_allocator { m_instance, m_device },

          m_commandManager { m_device },

          m_buffers { m_device, m_allocator },

          m_images { m_device, m_allocator },

          m_textures { m_device, m_commandManager, m_images, m_buffers }
    {
        m_drawImage = m_images.create("draw image",
                                      m_surface.getFramebufferExtent(),
                                      vk::Format::eR16G16B16A16Sfloat,
                                      m_device.getMaxUsableSampleCount(),
                                      vk::ImageUsageFlagBits::eTransferSrc |
                                          vk::ImageUsageFlagBits::eTransferDst |  // maybe remove?
                                          vk::ImageUsageFlagBits::eColorAttachment,
                                      vk::ImageAspectFlagBits::eColor);

        m_drawImageResolve =
            m_images.create("draw image resolve",
                            m_drawImage.getDimensions(),
                            m_drawImage.getFormat(),
                            vk::SampleCountFlagBits::e1,
                            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc |
                                vk::ImageUsageFlagBits::eTransferDst,
                            vk::ImageAspectFlagBits::eColor);

        m_depthImage = m_images.create("depth image",
                                       m_drawImage.getDimensions(),
                                       kDepthStencilFormat,
                                       m_device.getMaxUsableSampleCount(),
                                       vk::ImageUsageFlagBits::eDepthStencilAttachment,
                                       vk::ImageAspectFlagBits::eDepth);

        m_scheduler.Initialize({ .numTaskThreadsToCreate = 4 });

        m_asyncLoader = AsynchronousLoader(&m_scheduler, *this, m_device, m_buffers, m_textures);

        glslang::InitializeProcess();

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

            m_dummyTexture = m_textures.create(
                "dummy texture", vk::Extent2D { 32, 32 }, pixels.data(), sizeof(float) * pixels.size());
        }

        m_gpuSceneDataBuffer = m_buffers.create("GPU Scene Data",
                                                sizeof(GPUSceneData),
                                                vk::BufferUsageFlagBits::eUniformBuffer,
                                                VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                                VMA_ALLOCATION_CREATE_MAPPED_BIT |
                                                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

        ShaderManager shaders(m_device);
        shaders.addShader("fs.frag").addShader("vs.vert");

        auto timerStart = std::chrono::high_resolution_clock::now();

        shaders.build();

        auto timeTaken = std::chrono::duration<double, std::ratio<1, 1>>(
                             std::chrono::high_resolution_clock::now() - timerStart)
                             .count();

        logger::debug("Shader compilation took {:.2f}s", timeTaken);

        initDescriptors();

        auto pipelineLayoutConfig =
            PipelineLayoutConfig()
                .setDescriptorSetLayouts({ m_sceneDataDescriptorLayout, m_textureArrayDescriptorLayout })
                .setPushConstantSettings(sizeof(GPUDrawPushConstants),
                                         vk::ShaderStageFlagBits::eVertex |
                                             vk::ShaderStageFlagBits::eFragment);

        m_pipelineLayout = PipelineLayout(m_device, pipelineLayoutConfig);

        {
            auto pipelineConfig =
                GraphicsPipelineConfig()
                    .setShaderManager(shaders)
                    .setColorAttachmentFormat(m_drawImage.getFormat())
                    .setDepthAttachmentFormat(kDepthStencilFormat)
                    .setDepthStencilSettings(true, vk::CompareOp::eGreaterOrEqual)
                    // .setCullingSettings(vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise)
                    .setSampleCount(m_device.getMaxUsableSampleCount())
                    .setSampleShadingSettings(true, 0.1f);

            m_pipeline = GraphicsPipeline(m_device, "main_pipeline", m_pipelineLayout, pipelineConfig);
        }

        loadGltfScene();

        m_runPinnedTask.threadNum      = m_scheduler.GetNumTaskThreads() - 1;
        m_runPinnedTask.task_scheduler = &m_scheduler;
        m_scheduler.AddPinnedTask(&m_runPinnedTask);

        // Send async load task to external thread FILE_IO
        m_asyncLoadTask.threadNum      = m_runPinnedTask.threadNum;
        m_asyncLoadTask.task_scheduler = &m_scheduler;
        m_asyncLoadTask.async_loader   = &m_asyncLoader;
        m_scheduler.AddPinnedTask(&m_asyncLoadTask);

#if PROFILED
        for (size_t i : vi::iota(0u, utils::size(m_frameResources)))
        {
            std::string ctxName = fmt::format("Frame {}/{}", i + 1, kNumFramesInFlight);

            auto& ctx = m_frameResources[i].tracyContext;

            ctx = TracyVkContextCalibrated(
                *m_device.getPhysical(),
                *m_device.get(),
                *m_device.getMainQueue(),
                *m_commandManager.getMainCmdBuffer(i),
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

        m_runPinnedTask.execute = false;
        m_asyncLoadTask.execute = false;

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

        glslang::FinalizeProcess();
    }

    void RendererBackend::loadGltfScene()
    {
        m_scene = Model(m_device,
                        m_commandManager,
                        m_images,
                        m_buffers,
                        m_textureArrayDescriptorLayout,
                        m_dummyTexture.getImage().getImageView(),
                        m_dummySampler,
                        m_asyncLoader);

        auto glTFFile =
            std::filesystem::path(std::format("../../gltfSampleAssets/Models/{0}/glTF/{0}.gltf", "Sponza"));

        m_animationIndex = 0;
        m_animationTimer = 0.0f;

        auto timerStart = std::chrono::high_resolution_clock::now();

        m_scene.loadFromFile(glTFFile);

        auto timeTaken = std::chrono::duration<double, std::ratio<1, 1>>(
                             std::chrono::high_resolution_clock::now() - timerStart)
                             .count();

        logger::debug("{} took {:.2f}s to load", glTFFile.string(), timeTaken);

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
        std::vector<DescriptorAllocator::PoolSizeRatio> sizes = {
            { vk::DescriptorType::eUniformBuffer, 1 },
        };

        m_descriptorAllocator = DescriptorAllocator(m_device, 1, sizes);

        m_sceneDataDescriptorLayout =
            DescriptorLayoutBuilder()
                // The scene data buffer
                .setBinding(0,
                            vk::DescriptorType::eUniformBuffer,
                            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment)
                .build(m_device);

        m_sceneDataDescriptors = m_descriptorAllocator.allocate(m_device, m_sceneDataDescriptorLayout);

        DescriptorWriter writer;

        writer.writeBuffer(
            0, m_gpuSceneDataBuffer, sizeof(GPUSceneData), 0, vk::DescriptorType::eUniformBuffer);
        writer.updateSet(m_device, m_sceneDataDescriptors);

        m_textureArrayDescriptorLayout =
            DescriptorLayoutBuilder()
                .setBinding(0,
                            vk::DescriptorType::eCombinedImageSampler,
                            vk::ShaderStageFlagBits::eFragment,
                            kMaxBindlessResources)
                .build(m_device, vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool);
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
            .QueueFamily                 = m_device.getQueueFamilyIndices().mainFamily,
            .Queue                       = *m_device.getMainQueue(),
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

        sceneUniformData = GPUSceneData {
            .view              = view,
            .proj              = projection,
            .viewproj          = projection * view,
            .ambientColor      = glm::vec4(.1f),
            .cameraPos         = cameraPos,
            .screenWeight      = static_cast<float>(m_drawImage.getDimensions().width),
            .sunlightDirection = glm::vec3 { -0.2f, -1.0f, -0.3f },
            .screenHeight      = static_cast<float>(m_drawImage.getDimensions().height),
        };
    }

    void RendererBackend::queueTextureUpdate(ResourceHandle const& texture)
    {
        std::lock_guard<std::mutex> guard(m_texturesUpdateMutex);

        m_texturesToUpdate[m_numTexturesToUpdate++] = texture;
    };

    AsynchronousLoader::AsynchronousLoader(enki::TaskScheduler* taskScheduler,
                                           RendererBackend& renderer,
                                           Device& device,
                                           ResourceManager<GPUBuffer>& bufferManager,
                                           ResourceManager<Texture>& textureManager)
        : task_scheduler { taskScheduler },
          renderer { &renderer },
          device { &device },
          textureManager { &textureManager },
          bufferManager { &bufferManager }
    {
        fileLoadRequests.reserve(16);
        uploadRequests.reserve(16);

        stagingBuffer = bufferManager.create("Async loader staging buffer",
                                             64 * 1024 * 1024,
                                             vk::BufferUsageFlagBits::eTransferSrc,
                                             VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                             VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                                 VMA_ALLOCATION_CREATE_MAPPED_BIT);

        for (uint32_t i : vi::iota(0u, kNumFramesInFlight))
        {
            commandPools[i] = device->createCommandPool(
                                  vk::CommandPoolCreateInfo()
                                      .setQueueFamilyIndex(device.getQueueFamilyIndices().transferFamily)
                                      .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)) >>
                              ResultChecker();

            commandBuffers[i] =
                std::move((device->allocateCommandBuffers(vk::CommandBufferAllocateInfo()
                                                              .setLevel(vk::CommandBufferLevel::ePrimary)
                                                              .setCommandBufferCount(1)
                                                              .setCommandPool(commandPools[i])) >>
                           ResultChecker())[0]);
        }

        transferCompleteSemaphore = device->createSemaphore(vk::SemaphoreCreateInfo()) >> ResultChecker();

        transferFence =
            device->createFence(vk::FenceCreateInfo().setFlags(vk::FenceCreateFlagBits::eSignaled)) >>
            ResultChecker();
    }

    // TODO(aether) temporary: move this elsewhere
    void upload_texture_data(vk::CommandBuffer cmdBuf,
                             ResourceAccessor<Texture> texture,
                             void* textureData,
                             ResourceAccessor<GPUBuffer> stagingBuffer,
                             size_t stagingBufferOffset)
    {
        vk::Extent2D dimensions = texture.getImage().getDimensions();
        uint32_t image_size     = dimensions.width * dimensions.height * 4;

        // Copy buffer_data to staging buffer
        memcpy(reinterpret_cast<uint8_t*>(stagingBuffer.getMappedData()) + stagingBufferOffset,
               textureData,
               static_cast<size_t>(image_size));

        vk::BufferImageCopy region = {};
        region.bufferOffset        = stagingBufferOffset;
        region.bufferRowLength     = 0;
        region.bufferImageHeight   = 0;

        region.imageSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
        region.imageSubresource.mipLevel       = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount     = 1;

        region.imageExtent = vk::Extent3D { dimensions.width, dimensions.height, 1 };

        cmdBuf.pipelineBarrier(determinePipelineStageFlags(vk::AccessFlagBits::eNone),
                               determinePipelineStageFlags(vk::AccessFlagBits::eTransferWrite),
                               {},
                               {},
                               {},
                               vk::ImageMemoryBarrier()
                                   .setImage(texture.getImage())
                                   .setSrcQueueFamilyIndex(vk::QueueFamilyIgnored)
                                   .setDstQueueFamilyIndex(vk::QueueFamilyIgnored)
                                   .setSubresourceRange(vk::ImageSubresourceRange()
                                                            .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                                            .setBaseArrayLayer(0)
                                                            .setLayerCount(1)
                                                            .setLevelCount(0)
                                                            .setBaseMipLevel(0))
                                   .setOldLayout(vk::ImageLayout::eUndefined)
                                   .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
                                   .setSrcAccessMask(vk::AccessFlagBits::eNone)
                                   .setDstAccessMask(vk::AccessFlagBits::eTransferWrite));

        cmdBuf.copyBufferToImage(
            stagingBuffer, texture.getImage(), vk::ImageLayout::eTransferDstOptimal, region);

        cmdBuf.pipelineBarrier(determinePipelineStageFlags(vk::AccessFlagBits::eTransferWrite),
                               determinePipelineStageFlags(vk::AccessFlagBits::eTransferRead),
                               {},
                               {},
                               {},
                               vk::ImageMemoryBarrier()
                                   .setImage(texture.getImage())
                                   .setSrcQueueFamilyIndex(vk::QueueFamilyIgnored)
                                   .setDstQueueFamilyIndex(vk::QueueFamilyIgnored)
                                   .setSubresourceRange(vk::ImageSubresourceRange()
                                                            .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                                            .setBaseArrayLayer(0)
                                                            .setLayerCount(1)
                                                            .setLevelCount(0)
                                                            .setBaseMipLevel(0))
                                   .setOldLayout(vk::ImageLayout::eTransferDstOptimal)
                                   .setNewLayout(vk::ImageLayout::eTransferSrcOptimal)
                                   .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                                   .setDstAccessMask(vk::AccessFlagBits::eTransferRead));
    }

    void AsynchronousLoader::update()
    {
        // If a texture was processed in the previous commands, signal the renderer
        if (textureReady)
        {
            // i think i get what this class does
            // below conditional branches will check what type of updates need to be made, and make them
            // this method basically just gets repeatedly called from the async loader thread

            // i think this works due to the pinned thread
            // here's my hypothesis, but do go and check if its true or not
            // the async loader thread is a pinned task, which I'm guessing means that any calls
            // to AsyncLoader::updateThisOrCopyThatOrWhatever() will immediately pause the calling thread
            // because the scheduler hops to the async thread right away

            // TODO(aether) what to do here :(
            renderer->queueTextureUpdate(textureReady);
        }

        if (cpuBufferReady && gpuBufferReady)
        {
            MC_ASSERT(completed != nullptr);
            (*completed)++;

            // TODO(marco): free cpu buffer

            gpuBufferReady = {};
            cpuBufferReady = {};
            completed      = nullptr;
        }

        textureReady = {};

        // Process upload requests
        if (!uploadRequests.empty())
        {
            ZoneScoped;

            // Wait for transfer fence to be finished
            if (transferFence.getStatus() != vk::Result::eSuccess)
            {
                return;
            }

            device->get().resetFences({ transferFence });

            // Get last request
            UploadRequest request = uploadRequests.back();
            uploadRequests.pop_back();

            auto& cb = commandBuffers[renderer->getCurrentFrameIndex()];
            cb.begin(vk::CommandBufferBeginInfo().setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

            if (request.texture)
            {
                auto texture            = textureManager->access(request.texture);
                vk::Extent2D dimensions = texture.getImage().getDimensions();

                uint32_t const textureChannels  = 4;
                uint32_t const textureAlignment = 4;
                uint64_t const alignmentMask    = textureAlignment - 1;

                uint64_t const alignedImageSize =
                    (dimensions.width * dimensions.height * textureChannels + alignmentMask) & ~alignmentMask;

                // Request place in buffer
                uint64_t const currentOffset = std::atomic_fetch_add(&stagingBufferOffset, alignedImageSize);

                upload_texture_data(cb, texture, request.data, stagingBuffer, currentOffset);

                free(request.data);
            }
            else if (request.cpuBuffer && request.gpuBuffer)
            {
                auto src = bufferManager->access(request.cpuBuffer);
                auto dst = bufferManager->access(request.gpuBuffer);

                cb.copyBuffer(src, dst, vk::BufferCopy().setSize(dst.getSize()));
            }
            else if (request.cpuBuffer)
            {
                auto buffer = bufferManager->access(request.cpuBuffer);

                // TODO: proper alignment
                uint64_t const alignment_mask     = 63 - 1;
                uint64_t const aligned_image_size = (buffer.getSize() + alignment_mask) & ~alignment_mask;

                uint64_t const current_offset =
                    std::atomic_fetch_add(&stagingBufferOffset, aligned_image_size);

                std::memcpy(reinterpret_cast<uint8_t*>(stagingBuffer.getMappedData()) + current_offset,
                            request.data,
                            buffer.getSize());

                free(request.data);
            }

            cb.end();

            vk::PipelineStageFlags waitFlag { vk::PipelineStageFlagBits::eTransfer };

            device->getTransferQueue().submit(vk::SubmitInfo()
                                                  .setCommandBuffers(*cb)
                                                  .setWaitSemaphores(*transferCompleteSemaphore)
                                                  .setWaitDstStageMask(waitFlag),
                                              transferFence);

            // TODO(marco): better management for state machine. We need to account for file -> buffer,
            // buffer -> texture and buffer -> buffer. One the CPU buffer has been used it should be freed.
            if (request.texture)
            {
                MC_ASSERT(!textureReady);

                textureReady = request.texture;
            }
            else if (request.cpuBuffer && request.gpuBuffer)
            {
                MC_ASSERT(!cpuBufferReady);
                MC_ASSERT(!gpuBufferReady);
                MC_ASSERT(completed == nullptr);

                cpuBufferReady = request.cpuBuffer;
                gpuBufferReady = request.gpuBuffer;
                completed      = request.completed;
            }
            else if (request.cpuBuffer)
            {
                MC_ASSERT(!cpuBufferReady);

                cpuBufferReady = request.cpuBuffer;
            }
        }

        // Process a file request
        if (!fileLoadRequests.empty())
        {
            FileLoadRequest loadRequest = fileLoadRequests.back();
            fileLoadRequests.pop_back();

            // Process request
            int x, y, comp;

            auto timerStart = std::chrono::high_resolution_clock::now();

            uint8_t* texture_data = stbi_load(loadRequest.path.c_str(), &x, &y, &comp, 4);

            double timeTaken = std::chrono::duration<double, std::ratio<1, 1>>(
                                   std::chrono::high_resolution_clock::now() - timerStart)
                                   .count();

            MC_ASSERT(texture_data);

            logger::info("File {} read in {}", loadRequest.path, timeTaken);

            UploadRequest& upload_request = uploadRequests.emplace_back();
            upload_request.data           = texture_data;
            upload_request.texture        = loadRequest.texture;
        }

        stagingBufferOffset = 0;
    };

    void AsynchronousLoader::requestTextureData(std::string filename, ResourceHandle const& texture)
    {
        fileLoadRequests.push_back(FileLoadRequest { .path = std::move(filename), .texture = texture });
    }

    void AsynchronousLoader::requestBufferUpload(void* data, ResourceHandle const& handle)
    {
        uploadRequests.push_back(UploadRequest { .data = data, .cpuBuffer = handle });
    }

    void AsynchronousLoader::requestBufferCopy(ResourceHandle const& src,
                                               ResourceHandle const& dst,
                                               uint32_t* completed)
    {
        uploadRequests.push_back(
            UploadRequest { .completed = completed, .cpuBuffer = src, .gpuBuffer = dst });
    }
}  // namespace renderer::backend
