#pragma once

#include "allocator.hpp"
#include "command.hpp"
#include "device.hpp"
#include "mc/asserts.hpp"
#include "resource.hpp"
#include "vk_checker.hpp"

#include <string>
#include <string_view>

#include <glm/ext/vector_uint2.hpp>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_structs.hpp>

// FIXME(aether) the operator= overload does not function as expected with an empty rhs object

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
            if (m_data == nullptr)
            {
                return;
            }

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

    struct ImageCreation
    {
        Device const& device;
        Allocator const& allocator;

        vk::Extent2D dimensions;
        vk::Format format;
        vk::SampleCountFlagBits sampleCount;
        vk::ImageUsageFlags usageFlags;
        vk::ImageAspectFlags aspectFlags;

        uint32_t mipLevels    = 1;
        std::string_view name = {};
    };

    class BasicImage : public ResourceBase
    {
        // What if I put this in ResourceBase with a template
        friend class ResourceManager<BasicImage, ImageCreation>;

        BasicImage() = delete;

        BasicImage(uint32_t index, uint64_t creationNumber, ImageCreation creation);

    public:
        ~BasicImage();

        auto operator=(BasicImage const&) -> BasicImage& = delete;
        BasicImage(BasicImage const&)                    = delete;

        // TODO(aether) there is something seriously wrong with this and the GPUBuffer class when
        // you try to assign an object to {} (someImage = {})
        // It creates a leak, try it
        BasicImage(BasicImage&& other) noexcept : ResourceBase { other.m_handle }
        {
            std::swap(m_device, other.m_device);
            std::swap(m_allocator, other.m_allocator);
            std::swap(m_imageHandle, other.m_imageHandle);
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

            m_device      = std::exchange(other.m_device, { nullptr });
            m_allocator   = std::exchange(other.m_allocator, { nullptr });
            m_imageHandle = std::exchange(other.m_imageHandle, { nullptr });
            m_imageView   = std::exchange(other.m_imageView, { nullptr });
            m_allocation  = std::exchange(other.m_allocation, { nullptr });
            m_handle      = std::exchange(other.m_handle, {});
            m_format      = std::exchange(other.m_format, {});
            m_sampleCount = std::exchange(other.m_sampleCount, {});
            m_usageFlags  = std::exchange(other.m_usageFlags, {});
            m_aspectFlags = std::exchange(other.m_aspectFlags, {});
            m_mipLevels   = std::exchange(other.m_mipLevels, {});
            m_dimensions  = std::exchange(other.m_dimensions, {});

            return *this;
        };

        [[nodiscard]] operator bool() const { return m_imageHandle; }

        [[nodiscard]] bool operator==(std::nullptr_t) const { return !m_imageHandle; }

        [[nodiscard]] operator vk::Image() const { return m_imageHandle; }

        [[nodiscard]] auto get() const -> vk::Image { return m_imageHandle; }

        [[nodiscard]] auto getName() const -> std::string_view
        {
#if DEBUG
            VmaAllocationInfo allocInfo;
            vmaGetAllocationInfo(*m_allocator, m_allocation, &allocInfo);

            return allocInfo.pName;
#else
            return "";
#endif
        }

        void setName(std::string_view name)
        {
#if DEBUG
            vmaSetAllocationName(*m_allocator, m_allocation, name.data());

            auto func = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
                m_device->getInstance().getProcAddr("vkSetDebugUtilsObjectNameEXT"));

            VkDebugUtilsObjectNameInfoEXT info {
                .sType        = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                .objectType   = VK_OBJECT_TYPE_IMAGE,
                .objectHandle = reinterpret_cast<uint64_t>(m_imageHandle),
                .pObjectName  = name.data(),
            };

            func(*m_device->get(), &info) >> ResultChecker();
#endif
        }

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

        VkImage m_imageHandle { nullptr };
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

        [[nodiscard]] auto getMipLevels() const -> uint32_t { return m_mipLevels; }

        // FIXME(aether) handle mipmapping in this class itself
        void setMipLevels(uint32_t levels) { m_mipLevels = levels; }

    private:
        Device* m_device { nullptr };
        Allocator* m_allocator { nullptr };
        CommandManager* m_commandManager { nullptr };

        std::string m_path = "<buffer>";

        uint32_t m_mipLevels { 0 };

        BasicImage m_image;
    };

}  // namespace renderer::backend
