#pragma once

#include "events.hpp"
#include "mc/asserts.hpp"

#include <any>
#include <functional>
#include <ranges>
#include <utility>

#include <glm/ext/vector_float2.hpp>
#include <magic_enum.hpp>
#include <tracy/Tracy.hpp>

namespace rn = std::ranges;
namespace vi = std::ranges::views;

constexpr void dormantSubscriber(std::any const& /*unused*/) {};

class EventManager
{
public:
    EventManager() = default;

    EventManager(EventManager const&)                    = delete;
    EventManager(EventManager&&)                         = delete;
    auto operator=(EventManager const&) -> EventManager& = delete;
    auto operator=(EventManager&&) -> EventManager&      = delete;

    ~EventManager() = default;

    template<EventSpec Event>
    void subscribe(void (*listener)(Event const&))
    {
        size_t eventType = std::to_underlying(Event::eventType);
        size_t listenerHash { typeid(listener).hash_code() };

        std::function<void(std::any const&)> callback =
            [listener = std::move(listener)](std::any const& event)
        {
            listener(std::any_cast<Event const&>(event));
        };

        if (m_dormantListenerIndices.empty())
        {
            m_eventListeners[eventType].push_back(std::move(callback));

            m_eventListenerHashes[eventType].push_back(listenerHash);
        }
        else
        {
            size_t index = m_dormantListenerIndices.back();
            m_dormantListenerIndices.pop_back();

            m_eventListeners[eventType][index]      = std::move(callback);
            m_eventListenerHashes[eventType][index] = listenerHash;
        }
    }

    template<EventSpec Event, typename Class>
    void subscribe(Class* instance, void (Class::*listener)(Event const&))
    {
        size_t eventType = std::to_underlying(Event::eventType);
        size_t listenerHash { typeid(listener).hash_code() };

        std::function<void(std::any const&)> callback =
            [listener = std::move(listener), instance](std::any const& event)
        {
            (instance->*listener)(std::any_cast<Event const&>(event));
        };

        if (m_dormantListenerIndices.empty())
        {
            m_eventListeners[eventType].push_back(std::move(callback));

            m_eventListenerHashes[eventType].push_back(listenerHash);
        }
        else
        {
            size_t index = m_dormantListenerIndices.back();
            m_dormantListenerIndices.pop_back();

            m_eventListeners[eventType][index]      = std::move(callback);
            m_eventListenerHashes[eventType][index] = listenerHash;
        }
    };

    template<typename Class, EventSpec... Events>
    void subscribe(Class* instance, void (Class::*... funcs)(Events const&))

    {
        (subscribe(instance, funcs), ...);
    }

    template<EventSpec... Events>
    void subscribe(void(... funcs)(Events const&))
    {
        (subscribe(funcs), ...);
    }

    template<typename Class, EventSpec Event>
    void unsubscribe(Class* instance, void (Class::*listener)(Event const&))
    {
        size_t callbackHash = typeid(listener).hash_code();

        for (auto [index, hash] : vi::enumerate(m_eventListenerHashes[std::to_underlying(Event::eventType)]))
        {
            if (hash == callbackHash)  // TODO(aether) try [[likely/unlikely]]
            {
                m_eventListeners[std::to_underlying(Event::eventType)][index] =
                    std::function<void(std::any const&)>(dormantSubscriber);

                m_eventListenerHashes[std::to_underlying(Event::eventType)][index] =
                    typeid(dormantSubscriber).hash_code();

                m_dormantListenerIndices.push_back(index);

                return;
            }
        }

        MC_ASSERT_MSG(false, "Attempted to unregister an already-unregistered callback");
    }

    template<EventSpec Event>
    void dispatchEvent(Event const& event)
    {
        ZoneScopedN("Event Dispatch");

        [[maybe_unused]] std::string_view eventName = magic_enum::enum_name(Event::eventType);

        ZoneText(eventName.data(), eventName.size());

        auto event_any = std::make_any<Event const&>(event);

        for (auto const& listener : m_eventListeners[static_cast<size_t>(Event::eventType)])
        {
            listener(event_any);
        }
    }

private:
    std::array<std::vector<std::function<void(std::any const&)>>,
               static_cast<size_t>(EventType::EVENT_TYPE_MAX)>
        m_eventListeners {};

    std::array<std::vector<size_t>, static_cast<size_t>(EventType::EVENT_TYPE_MAX)> m_eventListenerHashes {};

    std::vector<size_t> m_dormantListenerIndices {};
};
