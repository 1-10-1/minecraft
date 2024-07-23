#pragma once

#include "allocator.hpp"
#include "vk_checker.hpp"

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_raii.hpp>

namespace renderer::backend
{
    class GPUBuffer
    {
    public:
        GPUBuffer() = default;

        GPUBuffer(Allocator& allocator,
                  size_t allocSize,
                  vk::BufferUsageFlags bufferUsage,
                  VmaMemoryUsage memoryUsage,
                  VmaAllocationCreateFlags allocFlags = 0);

        // Use this constructor to set a name
        GPUBuffer(Device& device,
                  Allocator& allocator,
                  std::string_view name,
                  size_t allocSize,
                  vk::BufferUsageFlags bufferUsage,
                  VmaMemoryUsage memoryUsage,
                  VmaAllocationCreateFlags allocFlags = 0);

        ~GPUBuffer();

        GPUBuffer(GPUBuffer const&)                    = delete;
        auto operator=(GPUBuffer const&) -> GPUBuffer& = delete;

        auto operator=(GPUBuffer&& other) noexcept -> GPUBuffer&
        {
            if (*this == other || !other.m_allocation)
            {
                return *this;
            }

            m_buffer     = other.m_buffer;
            m_allocator  = other.m_allocator;
            m_allocation = other.m_allocation;
            m_allocInfo  = other.m_allocInfo;

            other.m_buffer     = VK_NULL_HANDLE;
            other.m_allocation = nullptr;
            other.m_allocInfo  = {};

            return *this;
        }

        GPUBuffer(GPUBuffer&& other) noexcept
            : m_allocator { other.m_allocator },
              m_buffer { other.m_buffer },
              m_allocation { other.m_allocation },
              m_allocInfo { other.m_allocInfo }
        {
            other.m_buffer     = VK_NULL_HANDLE;
            other.m_allocation = nullptr;
            other.m_allocInfo  = {};
        };

        [[nodiscard]] operator bool() const { return m_buffer; }

        [[nodiscard]] bool operator==(std::nullptr_t) const { return !m_buffer; }

        [[nodiscard]] operator vk::Buffer() const { return m_buffer; }

        [[nodiscard]] auto operator->() const -> vk::Buffer { return m_buffer; }

        [[nodiscard]] auto get() const -> vk::Buffer { return m_buffer; }

        [[nodiscard]] auto getName() const -> std::string_view
        {
            VmaAllocationInfo allocInfo;
            vmaGetAllocationInfo(*m_allocator, m_allocation, &allocInfo);

            return allocInfo.pName;
        }

        void setName(vk::Device device, std::string_view name)
        {
            if constexpr (kDebug)
            {
                return;
            }

            vmaSetAllocationName(*m_allocator, m_allocation, name.data());

            device.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT()
                                                  .setObjectHandle(reinterpret_cast<uint64_t>(m_buffer))
                                                  .setObjectType(vk::ObjectType::eImage)
                                                  .setPObjectName(name.data())) >>
                ResultChecker();
        }

        [[nodiscard]] auto getMappedData() const -> void* { return m_allocInfo.pMappedData; }

        [[nodiscard]] auto getSize() const -> size_t { return m_allocInfo.size; }

    private:
        Allocator* m_allocator { nullptr };

        VkBuffer m_buffer { VK_NULL_HANDLE };
        VmaAllocation m_allocation { nullptr };
        VmaAllocationInfo m_allocInfo {};
    };
}  // namespace renderer::backend
