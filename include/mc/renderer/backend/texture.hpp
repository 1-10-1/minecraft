#pragma once

#include "buffer.hpp"
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
                Device& device,
                CommandManager& commandManager,
                ResourceManager<Image>& imageManager,
                ResourceManager<GPUBuffer>& bufferManager,
                StbiWrapper const& stbiImage);

        Texture(ResourceHandle handle,
                std::string const& name,
                Device& device,
                CommandManager& commandManager,
                ResourceManager<Image>& imageManager,
                ResourceManager<GPUBuffer>& bufferManager,
                vk::Extent2D dimensions,
                void* data,
                size_t dataSize);

    public:
        ~Texture() = default;

        Texture(Texture&&)            = default;
        Texture& operator=(Texture&&) = default;

        Texture(Texture const&)            = delete;
        Texture& operator=(Texture const&) = delete;

        std::string path = "<buffer>";

        uint32_t mipLevels { 0 };

        ResourceAccessor<Image> image;
    };

    template<>
    class ResourceAccessor<Texture> : public ResourceAccessorBase<Texture>
    {
    public:
        ResourceAccessor() = default;

        ResourceAccessor(ResourceManager<Texture>& manager, ResourceHandle handle)
            : ResourceAccessorBase<Texture> { manager, handle } {};

        virtual ~ResourceAccessor() = default;

        ResourceAccessor(ResourceAccessor&&)            = default;
        ResourceAccessor& operator=(ResourceAccessor&&) = default;

        ResourceAccessor(ResourceAccessor const&)            = delete;
        ResourceAccessor& operator=(ResourceAccessor const&) = delete;

        [[nodiscard]] auto getPath() const -> std::string const& { return get().path; }

        [[nodiscard]] auto getImage() const -> ResourceAccessor<Image> const& { return get().image; }

        [[nodiscard]] auto getMipLevels() const -> uint32_t { return get().mipLevels; }

        [[nodiscard]] operator bool() const { return get().image; }

        [[nodiscard]] bool operator==(std::nullptr_t) const { return !get().image; }
    };

    template<>
    class ResourceManager<Texture> : public ResourceManagerBase<Texture>
    {
        friend class ResourceManagerBase<Texture>;

        std::tuple<std::reference_wrapper<Device>,
                   std::reference_wrapper<CommandManager>,
                   std::reference_wrapper<ResourceManager<Image>>,
                   std::reference_wrapper<ResourceManager<GPUBuffer>>>
            m_extraConstructionParams;

    public:
        ResourceManager(Device& device,
                        CommandManager& commandManager,
                        ResourceManager<Image>& imageManager,
                        ResourceManager<GPUBuffer>& bufferManager)
            : m_extraConstructionParams { std::tie(device, commandManager, imageManager, bufferManager) } {};

        ResourceManager(ResourceManager&&)            = default;
        ResourceManager& operator=(ResourceManager&&) = default;

        ResourceManager(ResourceManager const&)            = delete;
        ResourceManager& operator=(ResourceManager const&) = delete;
    };
}  // namespace renderer::backend
