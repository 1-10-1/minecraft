#include <mc/camera.hpp>
#include <mc/events.hpp>
#include <mc/logger.hpp>
#include <mc/renderer/backend/renderer_backend.hpp>
#include <mc/renderer/renderer.hpp>

#include <glm/fwd.hpp>
#include <glm/trigonometric.hpp>
#include <tracy/Tracy.hpp>

namespace renderer
{
    Renderer::Renderer(EventManager& eventManager, window::Window& window, Camera& camera)
        : m_camera { camera }, m_backend { backend::RendererBackend(window) }

    {
        eventManager.subscribe(this,
                               &Renderer::onRender,
                               &Renderer::onUpdate,
                               &Renderer::onFramebufferResize,
                               &Renderer::onKeyPress);

        camera.setLens(glm::radians(45.0f), m_backend.getFramebufferSize(), 1000.f, 0.1f);
    }

    void Renderer::onRender(AppRenderEvent const& /* unused */)
    {
        ZoneScopedN("Frontend render");

        m_backend.render();
    }

    void Renderer::onUpdate(AppUpdateEvent const& event)
    {
        ZoneScopedN("Frontend update");

        m_backend.update(m_camera.getPosition(), m_camera.getView(), m_camera.getProj());
    }

    void Renderer::onKeyPress(KeyPressEvent const& event)
    {
        if (event.repeated)
        {
            return;
        }

        switch (event.key)
        {
            case Key::V:
                {
                    m_backend.toggleVsync();
                    break;
                }
            case Key::R:
                {
                    m_backend.toggleLightRevolution();
                    break;
                }
        }
    }

    void Renderer::onFramebufferResize(WindowFramebufferResizeEvent const& /* event */)
    {
        m_backend.scheduleSwapchainUpdate();
    }
}  // namespace renderer
