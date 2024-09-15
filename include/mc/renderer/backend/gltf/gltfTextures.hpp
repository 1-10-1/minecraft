#pragma once

#include "../buffer.hpp"
#include "../command.hpp"
#include "../device.hpp"
#include "../image.hpp"
#include "../resource.hpp"

#include <filesystem>

#include <tiny_gltf.h>
#include <vulkan/vulkan.hpp>

namespace renderer::backend
{
    struct TextureSampler
    {
        vk::Filter magFilter;
        vk::Filter minFilter;
        vk::SamplerAddressMode addressModeU;
        vk::SamplerAddressMode addressModeV;
        vk::SamplerAddressMode addressModeW;
    };

    struct GlTFTexture
    {
        GlTFTexture() = default;

        ~GlTFTexture() = default;

        GlTFTexture(Device& device,
                    CommandManager& cmdManager,
                    ResourceManager<GPUBuffer>& bufferManager,
                    ResourceManager<Image>& imgManager,
                    tinygltf::Image& gltfimage,
                    std::filesystem::path path,
                    TextureSampler textureSampler);

        GlTFTexture(GlTFTexture const&)            = delete;
        GlTFTexture& operator=(GlTFTexture const&) = delete;

        GlTFTexture(GlTFTexture&&)            = default;
        GlTFTexture& operator=(GlTFTexture&&) = default;

        // TODO(aether) currently, this class handles everything from uploading to compressing
        // differ that to the Texture class instead
        ResourceAccessor<Image> texture {};

        vk::ImageLayout layout {};

        vk::raii::Sampler sampler { nullptr };

    private:
        Device* m_device { nullptr };
        Allocator* m_allocator { nullptr };
        CommandManager* m_commandManager { nullptr };
    };

    // We use a custom image loading function with tinyglTF, so we can do custom stuff loading ktx textures
    bool loadImageDataFunc(tinygltf::Image* image,
                           int const imageIndex,
                           std::string* error,
                           std::string* warning,
                           int req_width,
                           int req_height,
                           unsigned char const* bytes,
                           int size,
                           void* userData);
}  // namespace renderer::backend
