#pragma once

#include <chrono>

class Timer
{
public:
    using Clock        = std::chrono::high_resolution_clock;
    using Seconds      = std::chrono::duration<double, std::ratio<1>>;
    using Milliseconds = std::chrono::duration<double, std::milli>;

    Timer()
        : m_baseTimePoint { Clock::now() },
          m_pauseTimePoint { m_baseTimePoint },
          m_prevTimePoint { m_baseTimePoint },
          m_latestTimePoint { m_baseTimePoint }
    {
        reset();
    };

    Timer(Timer&&)                         = delete;
    Timer(Timer const&)                    = delete;
    auto operator=(Timer&&) -> Timer&      = delete;
    auto operator=(Timer const&) -> Timer& = delete;

    ~Timer() = default;

    [[nodiscard]] auto isPaused() const -> bool { return m_isPaused; }

    template<typename duration = Milliseconds>
    [[nodiscard]] constexpr auto getTotalTime() const -> duration
    {
        return m_baseTimePoint - m_pauseTime - (m_isPaused ? m_pauseTimePoint : m_latestTimePoint);
    }

    template<typename duration = Milliseconds>
    [[nodiscard]] constexpr auto getDeltaTime() const -> duration
    {
        return std::chrono::duration_cast<duration>(m_deltaTime);
    }

    template<typename duration = Milliseconds>
    [[nodiscard]] static auto getCurrentTime()
    {
        return std::chrono::time_point<std::chrono::steady_clock, duration>(
            std::chrono::steady_clock::now().time_since_epoch());
    }

    void tick();
    void reset();
    void pause();
    void unpause();

private:
    std::chrono::time_point<Clock> m_baseTimePoint { Clock::now() }, m_pauseTimePoint { m_baseTimePoint },
        m_prevTimePoint { m_baseTimePoint }, m_latestTimePoint { m_baseTimePoint };

    bool m_isPaused { false };
    Milliseconds m_deltaTime { 0.0 }, m_pauseTime { 0.0 };
};
