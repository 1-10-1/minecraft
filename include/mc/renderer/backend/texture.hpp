#pragma once

#include "command.hpp"
#include "image.hpp"
#include "resource.hpp"

namespace renderer::backend
{
    class StbiWrapper
    {
        StbiWrapper() = default;

    public:
        StbiWrapper(std::string_view const& path);
        ~StbiWrapper();

        friend void swap(StbiWrapper& first, StbiWrapper& second) noexcept
        {
            using std::swap;

            swap(first.m_size, second.m_size);
            swap(first.m_data, second.m_data);
            swap(first.m_dimensions, second.m_dimensions);
        }

        StbiWrapper(StbiWrapper&& other) noexcept : StbiWrapper() { swap(*this, other); };

        StbiWrapper& operator=(StbiWrapper other) noexcept
        {
            swap(*this, other);

            return *this;
        }

        [[nodiscard]] auto getDimensions() const -> vk::Extent2D { return m_dimensions; }

        [[nodiscard]] auto getData() const -> unsigned char const* { return m_data; }

        [[nodiscard]] auto getDataSize() const -> size_t { return m_size; }

    private:
        vk::Extent2D m_dimensions { 0, 0 };
        size_t m_size { 0 };
        unsigned char* m_data { nullptr };
    };

    class Texture final : public ResourceBase
    {
        friend class ResourceManagerBase<Texture>;

        Texture() : ResourceBase(ResourceHandle(0, ResourceHandle::invalidCreationNumber)) {}

        Texture(ResourceHandle handle,
                std::string const& name,
                ResourceManager<Image>& imageManager,
                Device& device,
                Allocator& allocator,
                CommandManager& commandManager,
                StbiWrapper const& stbiImage);

        Texture(ResourceHandle handle,
                std::string const& name,
                ResourceManager<Image>& imageManager,
                Device& device,
                Allocator& allocator,
                CommandManager& commandManager,
                vk::Extent2D dimensions,
                void* data,
                size_t dataSize);

    public:
        ~Texture() = default;

        friend void swap(Texture& first, Texture& second) noexcept
        {
            using std::swap;

            swap(first.m_handle, second.m_handle);

            swap(first.mipLevels, second.mipLevels);
            swap(first.image, second.image);
            swap(first.path, second.path);
        }

        Texture(Texture&& other) noexcept : Texture() { swap(*this, other); };

        Texture& operator=(Texture other) noexcept
        {
            swap(*this, other);

            return *this;
        }

        std::string path = "<buffer>";

        uint32_t mipLevels { 0 };

        ResourceHandle image;
    };

    template<>
    class ResourceAccessor<Texture> final : ResourceAccessorBase<Texture>
    {
    public:
        ResourceAccessor(ResourceManager<Texture>& manager, ResourceHandle handle)
            : ResourceAccessorBase<Texture> { manager, handle } {};

        [[nodiscard]] auto getPath() const -> std::string const& { return get().path; }

        [[nodiscard]] auto getImage() const -> ResourceHandle const& { return get().image; }

        [[nodiscard]] auto getMipLevels() const -> uint32_t { return get().mipLevels; }

        [[nodiscard]] operator bool() const { return get().image; }

        [[nodiscard]] bool operator==(std::nullptr_t) const { return !get().image; }
    };

    template<>
    class ResourceManager<Texture> final : public ResourceManagerBase<Texture>
    {
    public:
        ResourceManager(ResourceManager<Image>& imageManager) : m_imageManager { imageManager } {};

        auto getExtraConstructionParams() { return std::tie(m_imageManager); };

    private:
        ResourceManager<Image>& m_imageManager;
    };
}  // namespace renderer::backend
