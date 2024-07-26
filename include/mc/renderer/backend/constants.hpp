#pragma once

#include <cstdint>

#include <vulkan/vulkan_raii.hpp>

namespace renderer::backend
{
    constexpr uint32_t kNumFramesInFlight         = 2;
    constexpr uint32_t kMaxBindlessResources      = 1024;
    constexpr vk::Format kDepthStencilFormat      = vk::Format::eD32Sfloat;
    constexpr vk::SampleCountFlagBits kMaxSamples = vk::SampleCountFlagBits::e4;
}  // namespace renderer::backend
