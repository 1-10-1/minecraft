#include <mc/renderer/backend/utils.hpp>

#include <print>

namespace renderer::backend
{
    // auto createGPUOnlyBuffer(Device& device,
    //                          Allocator& allocator,
    //                          CommandManager const& cmdManager,
    //                          vk::BufferUsageFlags usage,
    //                          size_t size,
    //                          void* data) -> GPUBuffer
    // {
    //     GPUBuffer buffer(
    //         allocator, size, vk::BufferUsageFlagBits::eTransferDst | usage, VMA_MEMORY_USAGE_GPU_ONLY);
    //
    //     {
    //         GPUBuffer staging(allocator,
    //                           // TODO(aether) notice how the staging buffer concatenates these
    //                           // vertexBufferSize + indexBufferSize,
    //                           size,
    //                           vk::BufferUsageFlagBits::eTransferSrc,
    //                           VMA_MEMORY_USAGE_CPU_ONLY);
    //
    //         void* mapped = staging.getMappedData();
    //
    //         std::memcpy(data, mapped, size);
    //
    //         {
    //             ScopedCommandBuffer cmdBuf {
    //                 device, cmdManager.getTransferCmdPool(), device.getTransferQueue(), true
    //             };
    //
    //             cmdBuf->copyBuffer(staging, buffer, vk::BufferCopy().setSize(size));
    //         };
    //     }
    //
    //     return buffer;
    // };
}  // namespace renderer::backend
