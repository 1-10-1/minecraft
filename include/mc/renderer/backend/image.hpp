#pragma once

#include "allocator.hpp"
#include "device.hpp"
#include "mc/asserts.hpp"
#include "resource.hpp"

#include <string_view>

#include <glm/ext/vector_uint2.hpp>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_structs.hpp>

// FIXME(aether) the operator= overload does not function as expected with an empty rhs object

namespace renderer::backend
{
    class Image : public ResourceBase
    {
        // What if I put this in ResourceBase with a template
        // what does resourcemanager even want from Image
        friend class ResourceAccessor<Image>;
        friend class ResourceManagerBase<Image>;

        Image() : ResourceBase(ResourceHandle(0, ResourceHandle::invalidCreationNumber)) {}

        Image(ResourceHandle handle,
              std::string const& name,
              Device const& device,
              Allocator const& allocator,
              vk::Extent2D dimensions,
              vk::Format format,
              vk::SampleCountFlagBits sampleCount,
              vk::ImageUsageFlags usageFlags,
              vk::ImageAspectFlags aspectFlags,
              uint32_t mipLevels = 1);

    public:
        ~Image();

        friend void swap(Image& first, Image& second) noexcept
        {
            using std::swap;

            // FIXME(aether) why is this my responsibility?
            swap(first.m_handle, second.m_handle);

            swap(first.device, second.device);
            swap(first.allocator, second.allocator);
            swap(first.imageHandle, second.imageHandle);
            swap(first.allocation, second.allocation);
            swap(first.format, second.format);
            swap(first.sampleCount, second.sampleCount);
            swap(first.usageFlags, second.usageFlags);
            swap(first.aspectFlags, second.aspectFlags);
            swap(first.mipLevels, second.mipLevels);
            swap(first.dimensions, second.dimensions);
            swap(first.imageView, second.imageView);
        }

        Image(Image&& other) noexcept : Image() { swap(*this, other); };

        Image& operator=(Image other) noexcept
        {
            swap(*this, other);

            return *this;
        }

        static void transition(vk::CommandBuffer cmdBuf,
                               vk::Image image,
                               vk::ImageLayout currentLayout,
                               vk::ImageLayout newLayout);

        void createImage(vk::Format format,
                         vk::ImageTiling tiling,
                         vk::ImageUsageFlags usage,
                         vk::MemoryPropertyFlags properties,
                         uint32_t mipLevels,
                         vk::SampleCountFlagBits numSamples);

        void createImageView(vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels);

        void create();
        void destroy();

        void setName(std::string const& name);

        Device const* device { nullptr };
        Allocator const* allocator { nullptr };

        VkImage imageHandle { nullptr };
        vk::raii::ImageView imageView { nullptr };
        VmaAllocation allocation { nullptr };

        vk::Format format;
        vk::SampleCountFlagBits sampleCount;
        vk::ImageUsageFlags usageFlags;
        vk::ImageAspectFlags aspectFlags;

        uint32_t mipLevels;

        vk::Extent2D dimensions;
    };

    template<>
    class ResourceAccessor<Image> final : ResourceAccessorBase<Image>
    {
    public:
        ResourceAccessor(ResourceManager<Image>& manager, ResourceHandle handle)
            : ResourceAccessorBase<Image> { manager, handle } {};

        [[nodiscard]] operator bool() const { return get().imageHandle; }

        [[nodiscard]] bool operator==(std::nullptr_t) const { return !get().imageHandle; }

        [[nodiscard]] operator vk::Image() const { return get().imageHandle; }

        [[nodiscard]] auto getVulkanHandle() const -> vk::Image { return get().imageHandle; }

        [[nodiscard]] auto getName() const -> std::string_view;

        void setName(std::string const& name) { get().setName(name); }

        [[nodiscard]] auto getImageView() const -> vk::raii::ImageView const&
        {
            MC_ASSERT_MSG(*get().imageView,
                          "Image view is not present, probably because the image is "
                          "being used for transfer only.");

            return get().imageView;
        }

        [[nodiscard]] auto getDimensions() const -> vk::Extent2D { return get().dimensions; }

        [[nodiscard]] auto getMipLevels() const -> uint32_t { return get().mipLevels; }

        [[nodiscard]] auto getFormat() const -> vk::Format { return get().format; }

        void copyTo(vk::CommandBuffer cmdBuf, vk::Image dst, vk::Extent2D dstSize, vk::Extent2D offset);

        static void transition(vk::CommandBuffer cmdBuf,
                               vk::Image image,
                               vk::ImageLayout currentLayout,
                               vk::ImageLayout newLayout)
        {
            Image::transition(cmdBuf, image, currentLayout, newLayout);
        };

        void resize(VkExtent2D dimensions)
        {
            get().dimensions = dimensions;

            get().destroy();
            get().create();
        }
    };
}  // namespace renderer::backend
