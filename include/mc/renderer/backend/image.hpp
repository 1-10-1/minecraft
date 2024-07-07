#pragma once

#include "allocator.hpp"
#include "command.hpp"
#include "device.hpp"
#include "mc/asserts.hpp"

#include <string>
#include <string_view>

#include <glm/ext/vector_uint2.hpp>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

namespace renderer::backend
{
    class StbiImage
    {
    public:
        StbiImage(std::string_view const& path);
        ~StbiImage();

        StbiImage(StbiImage const&)                    = delete;
        auto operator=(StbiImage const&) -> StbiImage& = delete;

        StbiImage(StbiImage&& other) noexcept
            : m_dimensions { other.m_dimensions }, m_size { other.m_size }, m_data { other.m_data }
        {
            other.m_dimensions = vk::Extent2D { 0, 0 };
            other.m_size       = 0;
            other.m_data       = nullptr;
        };

        auto operator=(StbiImage&& other) noexcept -> StbiImage&
        {
            if (this == &other)
            {
                return *this;
            }

            m_dimensions = other.m_dimensions;
            m_size       = other.m_size;
            m_data       = other.m_data;

            other.m_dimensions = vk::Extent2D { 0, 0 };
            other.m_size       = 0;
            other.m_data       = nullptr;

            return *this;
        };

        [[nodiscard]] auto getDimensions() const -> vk::Extent2D { return m_dimensions; }

        [[nodiscard]] auto getData() const -> unsigned char const* { return m_data; }

        [[nodiscard]] auto getDataSize() const -> size_t { return m_size; }

    private:
        vk::Extent2D m_dimensions {};
        size_t m_size {};
        unsigned char* m_data { nullptr };
    };

    class BasicImage
    {
    public:
        BasicImage() = default;

        BasicImage(Device const& device,
                   Allocator const& allocator,
                   vk::Extent2D dimensions,
                   vk::Format format,
                   vk::SampleCountFlagBits sampleCount,
                   vk::ImageUsageFlags usageFlags,
                   vk::ImageAspectFlags aspectFlags,
                   uint32_t mipLevels = 1);

        ~BasicImage();

        auto operator=(BasicImage const&) -> BasicImage& = delete;
        BasicImage(BasicImage const&)                    = delete;

        BasicImage(BasicImage&& other) noexcept
        {
            std::swap(m_device, other.m_device);
            std::swap(m_allocator, other.m_allocator);
            std::swap(m_handle, other.m_handle);
            std::swap(m_allocation, other.m_allocation);
            std::swap(m_format, other.m_format);
            std::swap(m_sampleCount, other.m_sampleCount);
            std::swap(m_usageFlags, other.m_usageFlags);
            std::swap(m_aspectFlags, other.m_aspectFlags);
            std::swap(m_mipLevels, other.m_mipLevels);
            std::swap(m_dimensions, other.m_dimensions);

            m_imageView = std::move(other.m_imageView);
        };

        auto operator=(BasicImage&& other) noexcept -> BasicImage&
        {
            if (this == &other)
            {
                return *this;
            }

            m_device     = std::exchange(other.m_device, { nullptr });
            m_allocator  = std::exchange(other.m_allocator, { nullptr });
            m_handle     = std::exchange(other.m_handle, { nullptr });
            m_allocation = std::exchange(other.m_allocation, { nullptr });

            m_format      = std::exchange(other.m_format, {});
            m_sampleCount = std::exchange(other.m_sampleCount, {});
            m_usageFlags  = std::exchange(other.m_usageFlags, {});
            m_aspectFlags = std::exchange(other.m_aspectFlags, {});
            m_mipLevels   = std::exchange(other.m_mipLevels, {});
            m_dimensions  = std::exchange(other.m_dimensions, {});

            m_imageView = std::move(other.m_imageView);

            other.m_imageView = nullptr;

            return *this;
        };

        [[nodiscard]] operator bool() const { return m_handle; }

        [[nodiscard]] bool operator==(std::nullptr_t) const { return !m_handle; }

        [[nodiscard]] operator vk::Image() const { return m_handle; }

        [[nodiscard]] auto get() const -> vk::Image { return m_handle; }

        [[nodiscard]] auto getImageView() const -> vk::raii::ImageView const&
        {
            MC_ASSERT_MSG(*m_imageView,
                          "Image view is not present, probably because the image is "
                          "being used for transfer only.");

            return m_imageView;
        }

        [[nodiscard]] auto getDimensions() const -> vk::Extent2D { return m_dimensions; }

        [[nodiscard]] auto getMipLevels() const -> uint32_t { return m_mipLevels; }

        [[nodiscard]] auto getFormat() const -> vk::Format { return m_format; }

        void copyTo(vk::CommandBuffer cmdBuf, vk::Image dst, vk::Extent2D dstSize, vk::Extent2D offset);
        void resolveTo(vk::CommandBuffer cmdBuf, vk::Image dst, vk::Extent2D dstSize, vk::Extent2D offset);

        static void transition(vk::CommandBuffer cmdBuf,
                               vk::Image image,
                               vk::ImageLayout currentLayout,
                               vk::ImageLayout newLayout);

        void resize(VkExtent2D dimensions)
        {
            m_dimensions = dimensions;

            destroy();
            create();
        }

    private:
        void createImage(vk::Format format,
                         vk::ImageTiling tiling,
                         vk::ImageUsageFlags usage,
                         vk::MemoryPropertyFlags properties,
                         uint32_t mipLevels,
                         vk::SampleCountFlagBits numSamples);

        void createImageView(vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels);

        void create();
        void destroy();

        Device const* m_device { nullptr };
        Allocator const* m_allocator { nullptr };

        VkImage m_handle { nullptr };
        vk::raii::ImageView m_imageView { nullptr };
        VmaAllocation m_allocation { nullptr };

        vk::Format m_format;
        vk::SampleCountFlagBits m_sampleCount;
        vk::ImageUsageFlags m_usageFlags;
        vk::ImageAspectFlags m_aspectFlags;

        uint32_t m_mipLevels;

        vk::Extent2D m_dimensions;
    };

    class Image
    {
    public:
        Image()  = default;
        ~Image() = default;

        Image(Device& device,
              Allocator& allocator,
              CommandManager& commandManager,
              StbiImage const& stbiImage);

        Image(Device& device,
              Allocator& allocator,
              CommandManager& commandManager,
              vk::Extent2D dimensions,
              void* data,
              size_t dataSize);

        Image(Image const&)                    = delete;
        auto operator=(Image const&) -> Image& = delete;

        Image(Image&&)                    = default;
        auto operator=(Image&&) -> Image& = default;

        [[nodiscard]] operator bool() const { return m_image; }

        [[nodiscard]] bool operator==(std::nullptr_t) const { return !m_image; }

        [[nodiscard]] auto getPath() const -> std::string const& { return m_path; }

        [[nodiscard]] auto getImageView() const -> vk::ImageView { return m_image.getImageView(); }

        [[nodiscard]] auto getImage() const -> BasicImage const& { return m_image; }

    private:
        Device* m_device { nullptr };
        Allocator* m_allocator { nullptr };
        CommandManager* m_commandManager { nullptr };

        void generateMipmaps(ScopedCommandBuffer& commandBuffer,
                             vk::Image image,
                             vk::Extent2D dimensions,
                             vk::Format imageFormat,
                             uint32_t mipLevels);

        std::string m_path;

        BasicImage m_image;
    };

}  // namespace renderer::backend
