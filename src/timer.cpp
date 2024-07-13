#include <mc/timer.hpp>

using namespace std::chrono_literals;

void Timer::pause()
{
    if (m_isPaused)
    {
        return;
    }

    m_pauseTimePoint = Clock::now();
    m_isPaused       = true;
}

void Timer::unpause()
{
    if (!m_isPaused)
    {
        return;
    }

    auto now = Clock::now();

    m_pauseTime += (now - m_pauseTimePoint);

    m_prevTimePoint  = now;
    m_pauseTimePoint = {};
    m_isPaused       = false;
}

void Timer::tick()
{
    if (m_isPaused)
    {
        m_deltaTime = 0.0ms;
        return;
    }

    m_latestTimePoint = Clock::now();
    m_deltaTime       = m_latestTimePoint - m_prevTimePoint;
    m_prevTimePoint   = m_latestTimePoint;
}

void Timer::reset()
{
    m_baseTimePoint = m_prevTimePoint = m_pauseTimePoint = m_latestTimePoint = Clock::now();
    m_deltaTime                                                              = 0.0ms;
    m_pauseTime                                                              = 0.0ms;
    m_isPaused                                                               = false;
}
