#include "mc/renderer/backend/renderer_backend.hpp"
#include "mc/utils.hpp"
#include <mc/renderer/backend/image.hpp>
#include <mc/renderer/backend/info_structs.hpp>
#include <mc/renderer/backend/render.hpp>
#include <mc/renderer/backend/vk_checker.hpp>

#include <glm/glm.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <imgui_internal.h>
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

        vk::CommandBuffer cmdBuf = m_commandManager.getMainCmdBuffer(m_currentFrame);

        cmdBuf.reset();

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

    void RendererBackend::drawGeometry(vk::CommandBuffer cmdBuf)
    {
        auto drawImage        = m_images.access(m_drawImage);
        auto drawImageResolve = m_images.access(m_drawImageResolve);
        auto depthImage       = m_images.access(m_depthImage);

        vk::Extent2D imageExtent = drawImage.getDimensions();

        auto colorAttachment = vk::RenderingAttachmentInfo()
                                   .setImageView(drawImage.getImageView())
                                   .setImageLayout(vk::ImageLayout::eGeneral)
                                   .setLoadOp(vk::AttachmentLoadOp::eClear)
                                   .setClearValue(vk::ClearValue(vk::ClearColorValue(
                                       std::array { 107.f / 255.f, 102.f / 255.f, 198.f / 255.f, 1.f })))
                                   .setStoreOp(vk::AttachmentStoreOp::eStore)
                                   .setResolveImageView(drawImageResolve.getImageView())
                                   .setResolveImageLayout(vk::ImageLayout::eGeneral)
                                   .setResolveMode(vk::ResolveModeFlagBits::eAverage);

        auto depthAttachment = vk::RenderingAttachmentInfo()
                                   .setImageView(depthImage.getImageView())
                                   .setImageLayout(vk::ImageLayout::eDepthAttachmentOptimal)
                                   .setLoadOp(vk::AttachmentLoadOp::eClear)
                                   .setStoreOp(vk::AttachmentStoreOp::eStore)
                                   .setClearValue({ .depthStencil = { .depth = 0.f } });

        auto renderInfo = vk::RenderingInfo()
                              .setRenderArea({ .extent = imageExtent })
                              .setColorAttachments(colorAttachment)
                              .setPDepthAttachment(&depthAttachment)
                              .setLayerCount(1);

        cmdBuf.beginRendering(renderInfo);

        vk::Viewport viewport = {
            .x        = 0,
            .y        = 0,
            .width    = static_cast<float>(imageExtent.width),
            .height   = static_cast<float>(imageExtent.height),
            .minDepth = 0.f,
            .maxDepth = 1.f,
        };

        cmdBuf.setViewport(0, viewport);

        auto scissor = vk::Rect2D().setExtent(imageExtent).setOffset({ 0, 0 });

        cmdBuf.setScissor(0, scissor);

        m_stats.drawCount     = 0;
        m_stats.triangleCount = 0;

        if (m_scene.indices)
        {
            cmdBuf.bindIndexBuffer(m_buffers.access(m_scene.indices), 0, vk::IndexType::eUint32);
        }

        cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline);

        cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
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

        cmdBuf.pushConstants(m_pipelineLayout,
                             vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                             0,
                             sizeof(GPUDrawPushConstants),
                             &pushConstants);

        {
            uint64_t numDraws = m_scene.drawIndirectCommands.size();

            TracyVkZone(m_frameResources[m_currentFrame].tracyContext, cmdBuf, "Indirect draw call");

            cmdBuf.drawIndexedIndirect(m_buffers.access(m_scene.drawIndirectBuffer),
                                       0,
                                       numDraws,
                                       sizeof(decltype(m_scene.drawIndirectCommands)::value_type));
        }

        cmdBuf.endRendering();
    }

    void RendererBackend::recordCommandBuffer(uint32_t imageIndex)
    {
#if PROFILED
        TracyVkCtx tracyCtx = m_frameResources[m_currentFrame].tracyContext;
#endif

        vk::CommandBuffer cmdBuf = m_commandManager.getMainCmdBuffer(m_currentFrame);

        auto beginInfo =
            vk::CommandBufferBeginInfo().setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

        cmdBuf.begin(beginInfo) >> ResultChecker();

        auto drawImage = m_images.access(m_drawImage), drawImageResolve = m_images.access(m_drawImageResolve),
             depthImage = m_images.access(m_depthImage);

        {
            TracyVkZone(tracyCtx, cmdBuf, "Command buffer recording");

            vk::Image swapchainImage = m_swapchain.getImages()[imageIndex];
            vk::Extent2D imageExtent = m_swapchain.getImageExtent();

            Image::transition(
                cmdBuf, depthImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal);

            Image::transition(
                cmdBuf, drawImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);

            {
                TracyVkZone(tracyCtx, cmdBuf, "Geometry render");

                drawGeometry(cmdBuf);
            }

            {
                TracyVkZone(tracyCtx, cmdBuf, "Draw image copy");

                Image::transition(cmdBuf,
                                  drawImageResolve,
                                  vk::ImageLayout::eUndefined,
                                  vk::ImageLayout::eTransferSrcOptimal);

                Image::transition(cmdBuf,
                                  swapchainImage,
                                  vk::ImageLayout::eUndefined,
                                  vk::ImageLayout::eTransferDstOptimal);

                drawImageResolve.copyTo(cmdBuf, swapchainImage, imageExtent, drawImage.getDimensions());
            }

            {
                TracyVkZone(tracyCtx, cmdBuf, "ImGui render");

                renderImgui(cmdBuf, *m_swapchain.getImageViews()[imageIndex]);
            }

            Image::transition(cmdBuf,
                              swapchainImage,
                              vk::ImageLayout::eTransferDstOptimal,
                              vk::ImageLayout::ePresentSrcKHR);
        }

        TracyVkCollect(tracyCtx, cmdBuf);

        cmdBuf.end() >> ResultChecker();
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

        ImGuiWindowFlags window_flags = 0;
        window_flags |= ImGuiWindowFlags_NoScrollbar;
        window_flags |= ImGuiWindowFlags_NoMove;
        window_flags |= ImGuiWindowFlags_NoResize;
        window_flags |= ImGuiWindowFlags_NoCollapse;
        window_flags |= ImGuiWindowFlags_NoNav;
        window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus;
        window_flags |= ImGuiWindowFlags_NoTitleBar;

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

        {
            ImGui::SetNextWindowPos(
                ImVec2(windowPadding, windowPadding), ImGuiCond_Always, ImVec2(0.0f, 0.0f));
            ImGui::SetNextWindowSize({});

            ImGui::Begin("Statistics", nullptr, window_flags);

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
            ImGui::Text("%s triangles %lu", humanReadableTriCount.data(), m_scene.triangleCount);
            ImGui::Text("%lu draws", m_scene.drawIndirectCommands.size());

            ImGui::Text(
                "%lu buffers (%lu active)", m_buffers.getNumResources(), m_buffers.getNumActiveResources());

            ImGui::Text(
                "%lu images (%lu active)", m_images.getNumResources(), m_images.getNumActiveResources());

            ImGui::Text("%lu textures (%lu active)",
                        m_textures.getNumResources(),
                        m_textures.getNumActiveResources());

            ImGui::End();
        }

        ImGui::Render();
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmdBuf);

        cmdBuf.endRendering();
    }
}  // namespace renderer::backend
