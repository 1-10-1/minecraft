#include <mc/renderer/backend/utils.hpp>

#include <print>

namespace renderer::backend
{
    vk::PipelineStageFlags determinePipelineStageFlags(vk::AccessFlags accessFlags)
    {
        vk::PipelineStageFlags flags {};

        if ((accessFlags & (vk::AccessFlagBits::eIndexRead | vk::AccessFlagBits::eVertexAttributeRead)))
            flags |= vk::PipelineStageFlagBits::eVertexInput;

        if ((accessFlags & (vk::AccessFlagBits::eUniformRead | vk::AccessFlagBits::eShaderRead |
                            vk::AccessFlagBits::eShaderWrite)))
        {
            flags |= vk::PipelineStageFlagBits::eVertexShader;
            flags |= vk::PipelineStageFlagBits::eFragmentShader;

            // if (pRenderer->pActiveGpuSettings->mGeometryShaderSupported)
            // {
            //     flags |= VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT;
            // }
            //
            // if (pRenderer->pActiveGpuSettings->mTessellationSupported)
            // {
            //     flags |= VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT;
            //     flags |= VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT;
            // }

            flags |= vk::PipelineStageFlagBits::eComputeShader;
        }

        if ((accessFlags & vk::AccessFlagBits::eInputAttachmentRead))
            flags |= vk::PipelineStageFlagBits::eFragmentShader;

        if ((accessFlags &
             (vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite)))
            flags |= vk::PipelineStageFlagBits::eColorAttachmentOutput;

        if ((accessFlags & (vk::AccessFlagBits::eDepthStencilAttachmentRead |
                            vk::AccessFlagBits::eDepthStencilAttachmentWrite)))
            flags |= vk::PipelineStageFlagBits::eEarlyFragmentTests |
                     vk::PipelineStageFlagBits::eLateFragmentTests;

        // Compatible with both compute and graphics queues
        if ((accessFlags & vk::AccessFlagBits::eIndirectCommandRead))
            flags |= vk::PipelineStageFlagBits::eDrawIndirect;

        if ((accessFlags & (vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite)))
            flags |= vk::PipelineStageFlagBits::eTransfer;

        if ((accessFlags & (vk::AccessFlagBits::eHostRead | vk::AccessFlagBits::eHostWrite)))
            flags |= vk::PipelineStageFlagBits::eHost;

        if (!flags)
            flags = vk::PipelineStageFlagBits::eTopOfPipe;

        return flags;
    }

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
