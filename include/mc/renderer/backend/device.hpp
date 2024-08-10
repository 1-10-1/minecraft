#pragma once

#include "instance.hpp"

#include <cstdint>

#include <vulkan/vulkan_raii.hpp>

namespace renderer::backend
{
    class Surface;

    struct QueueFamilyIndices
    {
        uint32_t graphicsFamily = std::numeric_limits<uint32_t>::max();
        uint32_t presentFamily  = std::numeric_limits<uint32_t>::max();
        uint32_t transferFamily = std::numeric_limits<uint32_t>::max();
    };

    class Device
    {
    public:
        Device()  = default;
        ~Device() = default;

        explicit Device(Instance& instance, Surface& surface);

        Device(Device const&)                    = delete;
        auto operator=(Device const&) -> Device& = delete;

        Device(Device&&)                    = default;
        auto operator=(Device&&) -> Device& = default;

        [[nodiscard]] operator vk::Device() const { return m_logicalHandle; }

        [[nodiscard]] operator vk::PhysicalDevice() const { return m_physicalHandle; }

        [[nodiscard]] operator vk::raii::Device const&() const { return m_logicalHandle; }

        [[nodiscard]] operator vk::raii::PhysicalDevice const&() const { return m_physicalHandle; }

        [[nodiscard]] vk::raii::Device const* operator->() const { return &m_logicalHandle; }

        [[nodiscard]] vk::raii::PhysicalDevice const& getPhysical() const { return m_physicalHandle; }

        [[nodiscard]] vk::raii::Device const& get() const { return m_logicalHandle; }

        [[nodiscard]] vk::raii::Instance const& getInstance() const { return m_instance->get(); }

        [[nodiscard]] auto getQueueFamilyIndices() const -> QueueFamilyIndices const&
        {
            return m_queueFamilyIndices;
        }

        [[nodiscard]] auto getGraphicsQueue() const -> vk::raii::Queue const& { return m_graphicsQueue; }

        [[nodiscard]] auto getTransferQueue() const -> vk::raii::Queue const& { return m_transferQueue; }

        [[nodiscard]] auto getPresentQueue() const -> vk::raii::Queue const& { return m_presentQueue; }

        [[nodiscard]] auto getDeviceProperties() const -> vk::PhysicalDeviceProperties
        {
            return m_physicalHandle.getProperties();
        }

        [[nodiscard]] auto getDeviceFeatures() const -> vk::PhysicalDeviceFeatures
        {
            return m_physicalHandle.getFeatures();
        }

        [[nodiscard]] auto getFormatProperties(vk::Format format) const -> vk::FormatProperties
        {
            return m_physicalHandle.getFormatProperties(format);
        }

        [[nodiscard]] auto getMaxUsableSampleCount() const -> vk::SampleCountFlagBits
        {
            return m_sampleCount;
        };

    private:
        void selectPhysicalDevice(Instance& instance, Surface& surface);
        void selectLogicalDevice();

        Instance* m_instance { nullptr };

        vk::raii::Device m_logicalHandle { nullptr };
        vk::raii::PhysicalDevice m_physicalHandle { nullptr };

        vk::SampleCountFlagBits m_sampleCount { vk::SampleCountFlagBits::e1 };

        QueueFamilyIndices m_queueFamilyIndices {};

        vk::raii::Queue m_graphicsQueue { nullptr };
        vk::raii::Queue m_presentQueue { nullptr };
        vk::raii::Queue m_transferQueue { nullptr };
    };
}  // namespace renderer::backend
