#include <mc/events.hpp>
#include <mc/game/game.hpp>
#include <mc/logger.hpp>
#include <mc/window.hpp>

#include <glm/vec3.hpp>

namespace game
{
    Game::Game(EventManager& eventManager, window::Window& window, Camera& camera)
        : m_window { window }, m_eventManager { eventManager }, m_camera { camera }
    {
        // For the sponza scene:
        // m_camera.lookAt(glm::vec3 { 1.25f, 1.4f, -1.25f }, { 0.f, 0.f, 0.f }, { 0.f, 1.f, 0.f });
        // m_camera.pitch(40.f);
        // m_camera.yaw(-50.0f);

        // For the ABeautifulGame scene:
        m_camera.lookAt(glm::vec3 { 0.67f, 0.42f, 0.6f }, { 0.f, 0.f, 0.f }, { 0.f, 1.f, 0.f });
        m_camera.yaw(-2.0f);
        m_camera.pitch(18.4f);

        m_eventManager.subscribe(this, &Game::onUpdate, &Game::onMouseButton, &Game::onKeyHold);
    };

    void Game::onUpdate(AppUpdateEvent const& event)
    {
        m_lastDelta = event.globalTimer.getDeltaTime().count();
    };

    void Game::onKeyHold(KeyHoldEvent const& event)
    {
        if (event.getInputManager()->isDown(Key::LeftControl))
        {
            return;
        }

        double cameraSpeed = 0.001;

        auto positive = static_cast<float>(+cameraSpeed * m_lastDelta);
        auto negative = static_cast<float>(-cameraSpeed * m_lastDelta);

        switch (event.key)
        {
            case Key::W:
                m_camera.moveZ(positive);
                break;
            case Key::S:
                m_camera.moveZ(negative);
                break;
            case Key::A:
                m_camera.moveX(negative);
                break;
            case Key::D:
                m_camera.moveX(positive);
                break;
            case Key::E:
                m_camera.moveY(positive);
                break;
            case Key::Q:
                m_camera.moveY(negative);
                break;
        }
    };

    void Game::onCursorMove(CursorMoveEvent const& event)
    {
        float dx = 0.07f * static_cast<float>(event.position.x - m_lastCursorPos.x);
        float dy = 0.07f * static_cast<float>(m_lastCursorPos.y - event.position.y);

        m_camera.yaw(dx);
        m_camera.pitch(dy);

        m_lastCursorPos = event.position;
    }

    void Game::onMouseButton(MouseButtonEvent const& event)
    {
        if (!(event.action == MouseButtonEvent::Action::Pressed))
        {
            return;
        }

        if (event.button == MouseButton::MouseButton5)
        {
            if (m_inputFocused)
            {
                m_window.enableCursor();
                m_eventManager.unsubscribe(this, &Game::onCursorMove);
                m_inputFocused = false;
            }
            else
            {
                m_window.disableCursor();
                m_lastCursorPos = event.getInputManager()->getCurrentCursorPosition();
                m_eventManager.subscribe(this, &Game::onCursorMove);
                m_inputFocused = true;
            }
        }
    }
}  // namespace game
