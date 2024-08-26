#pragma once

#include "allocator.hpp"
#include "resource.hpp"

#if DEBUG
#    include "vk_checker.hpp"
#endif

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_raii.hpp>

namespace renderer::backend
{
    // FIXME(aether) This name is misleading, GPUBuffer can be a cpu-only staging buffer
    class GPUBuffer : public ResourceBase
    {
        friend class ResourceAccessor<GPUBuffer>;
        friend class ResourceManagerBase<GPUBuffer>;

        GPUBuffer() : ResourceBase(ResourceHandle(0, ResourceHandle::invalidCreationNumber)) {}

        GPUBuffer(ResourceHandle handle,
                  std::string const& name,
                  Device& device,
                  Allocator& allocator,

                  size_t allocSize,
                  vk::BufferUsageFlags bufferUsage,
                  VmaMemoryUsage memoryUsage,
                  VmaAllocationCreateFlags allocFlags = 0);

    public:
        ~GPUBuffer();

        friend void swap(GPUBuffer& first, GPUBuffer& second) noexcept
        {
            using std::swap;

            swap(first.vulkanHandle, second.vulkanHandle);
            swap(first.allocator, second.allocator);
            swap(first.allocInfo, second.allocInfo);
            swap(first.allocation, second.allocation);
        }

        GPUBuffer(GPUBuffer&& other) noexcept : ResourceBase(std::move(other)) { swap(*this, other); };

        GPUBuffer& operator=(GPUBuffer other) noexcept
        {
            swap(*this, other);

            ResourceBase::operator=(std::move(*this));

            return *this;
        }

        void setName(std::string_view name)
        {
#if DEBUG
            vmaSetAllocationName(*allocator, allocation, name.data());

            auto func = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
                device->getInstance().getProcAddr("vkSetDebugUtilsObjectNameEXT"));

            VkDebugUtilsObjectNameInfoEXT info {
                .sType        = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                .objectType   = VK_OBJECT_TYPE_BUFFER,
                .objectHandle = reinterpret_cast<uint64_t>(vulkanHandle),
                .pObjectName  = name.data(),
            };

            func(*device->get(), &info) >> ResultChecker();
#endif
        }

        Device* device { nullptr };
        Allocator* allocator { nullptr };

        VkBuffer vulkanHandle { VK_NULL_HANDLE };
        VmaAllocation allocation { nullptr };
        VmaAllocationInfo allocInfo {};
    };

    template<>
    class ResourceAccessor<GPUBuffer> : public ResourceAccessorBase<GPUBuffer>
    {
    public:
        ResourceAccessor() = default;

        ResourceAccessor(ResourceManager<GPUBuffer>& manager, ResourceHandle handle)
            : ResourceAccessorBase<GPUBuffer> { manager, handle } {};

        virtual ~ResourceAccessor() = default;

        ResourceAccessor(ResourceAccessor&&)            = default;
        ResourceAccessor& operator=(ResourceAccessor&&) = default;

        ResourceAccessor(ResourceAccessor const&)            = delete;
        ResourceAccessor& operator=(ResourceAccessor const&) = delete;

        [[nodiscard]] operator bool() const { return get().vulkanHandle; }

        [[nodiscard]] bool operator==(std::nullptr_t) const { return !get().vulkanHandle; }

        [[nodiscard]] operator vk::Buffer() const { return get().vulkanHandle; }

        [[nodiscard]] auto operator->() const -> vk::Buffer { return get().vulkanHandle; }

        [[nodiscard]] auto getVulkanHandle() const -> vk::Buffer { return get().vulkanHandle; }

        [[nodiscard]] auto getName() const -> std::string_view
        {
#if DEBUG
            VmaAllocationInfo allocInfo;
            vmaGetAllocationInfo(*get().allocator, get().allocation, &allocInfo);

            return allocInfo.pName;
#else
            return "";
#endif
        }

        void setName(std::string_view name)
        {
#if DEBUG
            get().setName(name);
#endif
        }

        [[nodiscard]] auto getMappedData() const -> void* { return get().allocInfo.pMappedData; }

        [[nodiscard]] auto getSize() const -> size_t { return get().allocInfo.size; }
    };

    template<>
    class ResourceManager<GPUBuffer> final : public ResourceManagerBase<GPUBuffer>
    {
        friend class ResourceManagerBase<GPUBuffer>;

        std::tuple<Device&, Allocator&> m_extraConstructionParams;

    public:
        ResourceManager(Device& device, Allocator& allocator)
            : m_extraConstructionParams { std::tie(device, allocator) } {};

        ResourceManager(ResourceManager&&)            = default;
        ResourceManager& operator=(ResourceManager&&) = delete;

        ResourceManager(ResourceManager const&)            = delete;
        ResourceManager& operator=(ResourceManager const&) = delete;
    };
}  // namespace renderer::backend
