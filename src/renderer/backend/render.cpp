#include "mc/renderer/backend/constants.hpp"
#include "mc/renderer/backend/renderer_backend.hpp"
#include "mc/utils.hpp"
#include <mc/renderer/backend/image.hpp>
#include <mc/renderer/backend/info_structs.hpp>
#include <mc/renderer/backend/vk_checker.hpp>

#include <glm/glm.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <imgui_internal.h>
#include <ranges>
#include <tracy/Tracy.hpp>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

namespace renderer::backend
{
    void RendererBackend::render()
    {
        ZoneScopedN("Backend render");

        [[maybe_unused]] std::string zoneName = std::format("Render frame {}", m_currentFrame + 1);

        ZoneText(zoneName.c_str(), zoneName.size());

        FrameResources& frame = m_frameResources[m_currentFrame];

        m_device->waitForFences({ frame.inFlightFence }, true, std::numeric_limits<uint64_t>::max()) >>
            ResultChecker();
        m_device->resetFences({ frame.inFlightFence });

        uint32_t imageIndex {};

        {
            auto [result, index] = m_swapchain->acquireNextImage(
                std::numeric_limits<uint64_t>::max(), { frame.imageAvailableSemaphore }, {});

            imageIndex = index;

            if (result == vk::Result::eErrorOutOfDateKHR)
            {
                handleSurfaceResize();

                return;
            }

            if (result != vk::Result::eSuboptimalKHR)
            {
                result >> ResultChecker();
            }
        }

        vk::CommandBuffer cmdBuf = m_commandManager.getCommandBuffer(m_currentFrame, 0, false);

        m_commandManager.resetPools(m_currentFrame);

        recordCommandBuffer(imageIndex);

        auto cmdinfo = vk::CommandBufferSubmitInfo().setCommandBuffer(cmdBuf);

        auto waitInfo = vk::SemaphoreSubmitInfo()
                            .setValue(1)
                            .setStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
                            .setSemaphore(frame.imageAvailableSemaphore);

        auto signalInfo = vk::SemaphoreSubmitInfo()
                              .setValue(1)
                              .setStageMask(vk::PipelineStageFlagBits2::eAllGraphics)
                              .setSemaphore(frame.renderFinishedSemaphore);

        auto submit = vk::SubmitInfo2()
                          .setCommandBufferInfos(cmdinfo)
                          .setWaitSemaphoreInfos(waitInfo)
                          .setSignalSemaphoreInfos(signalInfo);

        {
            ZoneNamedN(tracy_queue_submit_zone, "Queue Submit", true);
            m_device.getMainQueue().submit2(submit, frame.inFlightFence);
        }

        auto presentInfo = vk::PresentInfoKHR()
                               .setWaitSemaphores(*frame.renderFinishedSemaphore)
                               .setSwapchains(*m_swapchain.get())
                               .setImageIndices(imageIndex);

        {
            ZoneNamedN(tracy_queue_present_zone, "Queue presentation", true);
            vk::Result result = m_device.getPresentQueue().presentKHR(presentInfo);

            if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR ||
                m_windowResized)
            {
                handleSurfaceResize();
                m_windowResized = false;
            }
            else
            {
                result >> ResultChecker();
            }
        }

        m_currentFrame = (m_currentFrame + 1) % kNumFramesInFlight;
        ++m_frameCount;
    }

    void RendererBackend::drawGeometry(vk::CommandBuffer primaryBuf)
    {
        vk::Extent2D imageExtent = m_drawImage.getDimensions();

        auto colorAttachment = vk::RenderingAttachmentInfo()
                                   .setImageView(m_drawImage.getImageView())
                                   .setImageLayout(vk::ImageLayout::eGeneral)
                                   .setLoadOp(vk::AttachmentLoadOp::eClear)
                                   .setClearValue(vk::ClearValue(vk::ClearColorValue(
                                       std::array { 107.f / 255.f, 102.f / 255.f, 198.f / 255.f, 1.f })))
                                   .setStoreOp(vk::AttachmentStoreOp::eStore)
                                   .setResolveImageView(m_drawImageResolve.getImageView())
                                   .setResolveImageLayout(vk::ImageLayout::eGeneral)
                                   .setResolveMode(vk::ResolveModeFlagBits::eAverage);

        auto depthAttachment = vk::RenderingAttachmentInfo()
                                   .setImageView(m_depthImage.getImageView())
                                   .setImageLayout(vk::ImageLayout::eDepthAttachmentOptimal)
                                   .setLoadOp(vk::AttachmentLoadOp::eClear)
                                   .setStoreOp(vk::AttachmentStoreOp::eStore)
                                   .setClearValue({ .depthStencil = { .depth = 0.f } });

        auto renderInfo = vk::RenderingInfo()
                              .setRenderArea({ .extent = imageExtent })
                              .setColorAttachments(colorAttachment)
                              .setPDepthAttachment(&depthAttachment)
                              .setLayerCount(1)
                              .setFlags(vk::RenderingFlagBits::eContentsSecondaryCommandBuffers);

        primaryBuf.beginRendering(renderInfo);

        auto colorFormat = m_drawImage.getFormat();
        auto inheritance = vk::StructureChain(vk::CommandBufferInheritanceInfo(),
                                              vk::CommandBufferInheritanceRenderingInfo()
                                                  .setColorAttachmentFormats(colorFormat)
                                                  .setDepthAttachmentFormat(kDepthStencilFormat)
                                                  .setRasterizationSamples(kMaxSamples));

        auto beginInfo = vk::CommandBufferBeginInfo()
                             .setFlags(vk::CommandBufferUsageFlagBits::eRenderPassContinue)
                             .setPInheritanceInfo(&inheritance.get<vk::CommandBufferInheritanceInfo>());

        auto scb = m_commandManager.getSecondaryCommandBuffer(m_currentFrame, 0);

        scb.begin(beginInfo) >> ResultChecker();

        vk::Viewport viewport = {
            .x        = 0,
            .y        = 0,
            .width    = static_cast<float>(imageExtent.width),
            .height   = static_cast<float>(imageExtent.height),
            .minDepth = 0.f,
            .maxDepth = 1.f,
        };

        scb.setViewport(0, viewport);

        auto scissor = vk::Rect2D().setExtent(imageExtent).setOffset({ 0, 0 });

        scb.setScissor(0, scissor);

        m_stats.drawCount     = 0;
        m_stats.triangleCount = 0;

        if (m_scene.indices)
        {
            scb.bindIndexBuffer(m_scene.indices, 0, vk::IndexType::eUint32);
        }

        scb.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline);

        scb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                               m_pipelineLayout,
                               0,
                               {
                                   m_sceneDataDescriptors,
                                   m_scene.bindlessMaterialDescriptorSet,
                               },
                               {});

        GPUDrawPushConstants pushConstants {
            .vertexBuffer    = m_scene.vertexBufferAddress,
            .materialBuffer  = m_scene.materialBufferAddress,
            .primitiveBuffer = m_scene.primitiveDataBufferAddress,
        };

        scb.pushConstants(m_pipelineLayout,
                          vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                          0,
                          sizeof(GPUDrawPushConstants),
                          &pushConstants);

        {
            uint64_t numDraws = m_scene.drawIndirectCommands.size();

            TracyVkZone(m_frameResources[m_currentFrame].tracyContext, primaryBuf, "Indirect draw call");

            scb.drawIndexedIndirect(m_scene.drawIndirectBuffer,
                                    0,
                                    numDraws,
                                    sizeof(decltype(m_scene.drawIndirectCommands)::value_type));
        }

        scb.end() >> ResultChecker();

        primaryBuf.executeCommands(scb);

        primaryBuf.endRendering();
    }

    void RendererBackend::recordCommandBuffer(uint32_t imageIndex)
    {
#if PROFILED
        TracyVkCtx tracyCtx = m_frameResources[m_currentFrame].tracyContext;
#endif

        vk::CommandBuffer primaryBuf = m_commandManager.getCommandBuffer(m_currentFrame, 0, true);

        {
            TracyVkZone(tracyCtx, primaryBuf, "Command buffer recording");

            vk::Image swapchainImage = m_swapchain.getImages()[imageIndex];
            vk::Extent2D imageExtent = m_swapchain.getImageExtent();

            Image::transition(primaryBuf,
                              m_depthImage,
                              vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eDepthAttachmentOptimal);

            Image::transition(primaryBuf,
                              m_drawImage,
                              vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eColorAttachmentOptimal);

            {
                TracyVkZone(tracyCtx, primaryBuf, "Geometry render");

                drawGeometry(primaryBuf);
            }

            {
                TracyVkZone(tracyCtx, primaryBuf, "Draw image copy");

                Image::transition(primaryBuf,
                                  m_drawImageResolve,
                                  vk::ImageLayout::eUndefined,
                                  vk::ImageLayout::eTransferSrcOptimal);

                Image::transition(primaryBuf,
                                  swapchainImage,
                                  vk::ImageLayout::eUndefined,
                                  vk::ImageLayout::eTransferDstOptimal);

                m_drawImageResolve.copyTo(
                    primaryBuf, swapchainImage, imageExtent, m_drawImage.getDimensions());
            }

            {
                TracyVkZone(tracyCtx, primaryBuf, "ImGui render");

                renderImgui(primaryBuf, *m_swapchain.getImageViews()[imageIndex]);
            }

            Image::transition(primaryBuf,
                              swapchainImage,
                              vk::ImageLayout::eTransferDstOptimal,
                              vk::ImageLayout::ePresentSrcKHR);
        }

        TracyVkCollect(tracyCtx, primaryBuf);

        primaryBuf.end() >> ResultChecker();
    }

    void RendererBackend::renderImgui(vk::CommandBuffer cmdBuf, vk::ImageView targetImage)
    {
        ImGuiIO& io = ImGui::GetIO();

        static double frametime                             = 0.0;
        static Timer::Clock::time_point lastFrametimeUpdate = Timer::Clock::now();

        if (auto now = Timer::Clock::now();
            std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(now - lastFrametimeUpdate)
                .count() > 333.333f)
        {
            frametime           = 1000.f / io.Framerate;
            lastFrametimeUpdate = now;
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        auto colorAttachment = vk::RenderingAttachmentInfo()
                                   .setImageView(targetImage)
                                   .setImageLayout(vk::ImageLayout::eGeneral)
                                   .setLoadOp(vk::AttachmentLoadOp::eLoad)
                                   .setStoreOp(vk::AttachmentStoreOp::eStore);

        auto renderInfo = vk::RenderingInfo()
                              .setRenderArea({ .extent = m_swapchain.getImageExtent() })
                              .setColorAttachments(colorAttachment)
                              .setLayerCount(1);

        cmdBuf.beginRendering(renderInfo);

        float windowPadding = 10.0f;

        ImVec2 prevWindowPos, prevWindowSize;

        {
            ImGui::SetNextWindowPos(
                ImVec2(windowPadding, windowPadding), ImGuiCond_Always, ImVec2(0.0f, 0.0f));
            ImGui::SetNextWindowSize({});

            ImGui::Begin("Statistics",
                         nullptr,
                         ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus |
                             ImGuiWindowFlags_NoTitleBar);

            // Get the current window's position and size
            prevWindowPos  = ImGui::GetWindowPos();
            prevWindowSize = ImGui::GetWindowSize();

            ImGui::TextColored(ImVec4(77.5f / 255, 255.f / 255, 125.f / 255, 1.f), "%.2f mspf", frametime);
            ImGui::SameLine();
            ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical, 2.5f);
            ImGui::SameLine();
            ImGui::TextColored(
                ImVec4(255.f / 255, 163.f / 255, 77.f / 255, 1.f), "%.0f fps", 1000 / frametime);
            ImGui::SameLine();
            ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical, 2.5f);
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(255.f / 255, 215.f / 255, 100.f / 255, 1.f),
                               "Vsync: %s",
                               m_surface.getVsync() ? "on" : "off");

            std::string humanReadableTriCount = utils::largeNumToHumanReadable(m_scene.triangleCount);
            ImGui::TextColored(ImVec4(147.f / 255.f, 210.f / 255.f, 2.f / 255.f, 1.f),
                               "%s triangles",
                               humanReadableTriCount.data());
            ImGui::TextColored(ImVec4(147.f / 255.f, 210.f / 255.f, 2.f / 255.f, 1.f),
                               "%lu draws",
                               m_scene.drawIndirectCommands.size());

            ImGui::TextColored(ImVec4(147.f / 255.f, 210.f / 255.f, 2.f / 255.f, 1.f),
                               "%lu images (+ %lu inactive)",
                               m_images.getNumActiveResources(),
                               m_images.getNumResources() - m_images.getNumActiveResources());

            ImGui::TextColored(ImVec4(147.f / 255.f, 210.f / 255.f, 2.f / 255.f, 1.f),
                               "%lu textures (+ %lu inactive)",
                               m_textures.getNumActiveResources(),
                               m_textures.getNumResources() - m_textures.getNumActiveResources());

            ImGui::End();
        }

        {
            ImGui::SetNextWindowPos(
                ImVec2(prevWindowPos.x, prevWindowPos.y + prevWindowSize.y + windowPadding));

            ImGui::Begin("Buffers",
                         nullptr,
                         ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoCollapse);

            // Get the current window's position and size
            prevWindowPos  = ImGui::GetWindowPos();
            prevWindowSize = ImGui::GetWindowSize();

            ImGui::TextColored(ImVec4(0.f, 220.f / 255.f, 190.f / 255.f, 1.f),
                               "%lu buffers (+ %lu inactive)",
                               m_buffers.getNumActiveResources(),
                               m_buffers.getNumResources() - m_buffers.getNumActiveResources());

            for (auto const& [name, size] : m_buffers.getAllActiveBuffersInfo())
            {
                std::string sizeHumanReadable = utils::largeSizeToHumanReadable(size);

                ImGui::TextColored(ImVec4(0.f, 170.f / 255.f, 220.f / 255.f, 1.f),
                                   "%s (%s)",
                                   name.data(),
                                   sizeHumanReadable.data());
            };

            ImGui::End();
        }

        ImGui::Render();
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmdBuf);

        cmdBuf.endRendering();
    }
}  // namespace renderer::backend
