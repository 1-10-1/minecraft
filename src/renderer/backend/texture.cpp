#include <mc/renderer/backend/buffer.hpp>
#include <mc/renderer/backend/texture.hpp>

#include <stb_image.h>

using namespace renderer::backend;

Texture::Texture(ResourceHandle handle,
                 std::string const& name,
                 ResourceManager<Image>& imageManager,
                 Device& device,
                 Allocator& allocator,
                 CommandManager& commandManager,
                 StbiWrapper const& stbiImage)
    : ResourceBase { handle },
      image { imageManager.create(name,
                                  device,
                                  allocator,
                                  stbiImage.getDimensions(),
                                  vk::Format::eR8G8B8A8Unorm,
                                  vk::SampleCountFlagBits::e1,
                                  vk::ImageUsageFlagBits::eTransferSrc |
                                      vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
                                  vk::ImageAspectFlagBits::eColor,
                                  static_cast<uint32_t>(std::floor(std::log2(std::max(
                                      stbiImage.getDimensions().width, stbiImage.getDimensions().height)))) +
                                      1) }
{
    vk::Extent2D dimensions = stbiImage.getDimensions();

    ResourceAccessor<Image> img = imageManager.access(image);

    GPUBuffer uploadBuffer(allocator,
                           stbiImage.getDataSize(),
                           vk::BufferUsageFlagBits::eTransferSrc,
                           VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                           VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                               VMA_ALLOCATION_CREATE_MAPPED_BIT);

    std::memcpy(uploadBuffer.getMappedData(), stbiImage.getData(), stbiImage.getDataSize());

    {
        ScopedCommandBuffer commandBuffer(device, commandManager.getMainCmdPool(), device.getMainQueue());

        Image::transition(commandBuffer,
                          img.getVulkanHandle(),
                          vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eTransferDstOptimal);

        vk::BufferImageCopy region {
                .bufferOffset      = 0,
                .bufferRowLength   = 0,
                .bufferImageHeight = 0,
                .imageSubresource  = {
                    .aspectMask     = vk::ImageAspectFlagBits::eColor,
                    .mipLevel       = 0,
                    .baseArrayLayer = 0,
                    .layerCount     = 1,
                },
                .imageOffset = { 0, 0, 0 },
                .imageExtent = { dimensions.width, dimensions.height, 1 },
            };

        commandBuffer->copyBufferToImage(
            uploadBuffer, img.getVulkanHandle(), vk::ImageLayout::eTransferDstOptimal, { region });

        //transitioned to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL here
        // generateMipmaps(commandBuffer,
        //                 m_image,
        //                 { dimensions.width, dimensions.height },
        //                 vk::Format::eR8G8B8A8Unorm,
        //                 mipLevels);

        commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                           vk::PipelineStageFlagBits::eFragmentShader,
                                           vk::DependencyFlags { 0 },
                                           {},
                                           {},
                                           {
                {
                    .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                    .dstAccessMask = vk::AccessFlagBits::eShaderRead,
                    .oldLayout     = vk::ImageLayout::eTransferDstOptimal,
                    .newLayout     = vk::ImageLayout::eShaderReadOnlyOptimal,
                    .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
                    .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
                    .image = img.getVulkanHandle(),
                    .subresourceRange = {
                        .aspectMask     = vk::ImageAspectFlagBits::eColor,
                        .levelCount     = 1,
                        .baseArrayLayer = 0,
                        .layerCount     = 1,
                    },
                }
            });
    }
}

Texture::Texture(ResourceHandle handle,
                 std::string const& name,
                 ResourceManager<Image>& imageManager,
                 Device& device,
                 Allocator& allocator,
                 CommandManager& commandManager,
                 vk::Extent2D dimensions,
                 void* data,
                 size_t dataSize)
    : ResourceBase { handle },
      image { imageManager.create(
          name,
          device,
          allocator,
          dimensions,
          vk::Format::eR8G8B8A8Unorm,
          vk::SampleCountFlagBits::e1,
          vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
              vk::ImageUsageFlagBits::eSampled,
          vk::ImageAspectFlagBits::eColor,
          static_cast<uint32_t>(std::floor(std::log2(std::max(dimensions.width, dimensions.height)))) + 1) }
{
    ResourceAccessor<Image> img = imageManager.access(image);

    GPUBuffer uploadBuffer(allocator,
                           dataSize,
                           vk::BufferUsageFlagBits::eTransferSrc,
                           VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                           VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                               VMA_ALLOCATION_CREATE_MAPPED_BIT);

    std::memcpy(uploadBuffer.getMappedData(), data, dataSize);

    {
        // TODO(aether) graphics or transfer?
        ScopedCommandBuffer commandBuffer(device, commandManager.getMainCmdPool(), device.getMainQueue());

        Image::transition(commandBuffer,
                          img.getVulkanHandle(),
                          vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eTransferDstOptimal);

        vk::BufferImageCopy region {
                .bufferOffset      = 0,
                .bufferRowLength   = 0,
                .bufferImageHeight = 0,
                .imageSubresource  = {
                    .aspectMask     = vk::ImageAspectFlagBits::eColor,
                    .mipLevel       = 0,
                    .baseArrayLayer = 0,
                    .layerCount     = 1,
                },
                .imageOffset = { 0, 0, 0 },
                .imageExtent = { dimensions.width, dimensions.height, 1 },
            };

        commandBuffer->copyBufferToImage(
            uploadBuffer, img.getVulkanHandle(), vk::ImageLayout::eTransferDstOptimal, { region });

        // transitioned to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL here
        // generateMipmaps(commandBuffer,
        //                 m_image,
        //                 { dimensions.width, dimensions.height },
        //                 vk::Format::eR8G8B8A8Unorm,
        //                 mipLevels);

        commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                           vk::PipelineStageFlagBits::eFragmentShader,
                                           vk::DependencyFlags { 0 },
                                           {},
                                           {},
                                           {
                {
                    .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                    .dstAccessMask = vk::AccessFlagBits::eShaderRead,
                    .oldLayout     = vk::ImageLayout::eTransferDstOptimal,
                    .newLayout     = vk::ImageLayout::eShaderReadOnlyOptimal,
                    .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
                    .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
                    .image = img.getVulkanHandle(),
                    .subresourceRange = {
                        .aspectMask     = vk::ImageAspectFlagBits::eColor,
                        .levelCount     = 1,
                        .baseArrayLayer = 0,
                        .layerCount     = 1,
                    },
                }
            });
    }
}

StbiWrapper::StbiWrapper(std::string_view const& path)
{
    int texWidth { 0 };
    int texHeight { 0 };
    int texChannels { 0 };

    // TODO(aether) instead of memcpying to the mapped gpu buffer region, make this
    // directly load it into the mapped region.
    m_data = stbi_load(path.data(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

    m_dimensions = vk::Extent2D { static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight) };

    m_size = static_cast<size_t>(texWidth) * texHeight * 4;  // 4 channels, RGBA

    MC_ASSERT_MSG(m_data, "Failed to load texture");
}

StbiWrapper::~StbiWrapper()
{
    stbi_image_free(m_data);
}
