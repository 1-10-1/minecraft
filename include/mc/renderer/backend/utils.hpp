#pragma once

// TODO(aether) CHECK OUT THIS FUNCTION
// https://github.com/1-10-1/minecraft_raytraced/blob/a515c4704c213b11e47e036fb393ebc61bfde527/src/renderer/backend/mesh_upload.cpp

#include "mc/renderer/backend/allocator.hpp"
#include "mc/renderer/backend/buffer.hpp"
#include "mc/renderer/backend/command.hpp"
#include "mc/renderer/backend/vk_checker.hpp"
#include "mc/utils.hpp"

#include <filesystem>
#include <vector>

#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>

namespace renderer::backend
{
    // Determines pipeline stages involved for given accesses
    vk::PipelineStageFlags determinePipelineStageFlags(vk::AccessFlags accessFlags);

    // TODO(aether) VERTEX BUFFER SHOULD USE STORAGEBUFFERBIT AND SHADERDEVICEADDRESSBIT
    auto createGPUOnlyBuffer(Device& device,
                             Allocator& allocator,
                             CommandManager const& cmdManager,
                             vk::BufferUsageFlags usage,
                             size_t size,
                             void* data) -> GPUBuffer;

    inline auto createShaderModule(vk::raii::Device const& device,
                                   std::filesystem::path const& shaderPath) -> vk::raii::ShaderModule
    {
        std::vector<char> shaderCode = utils::readBytes(shaderPath);

        return device.createShaderModule({
                   .codeSize = shaderCode.size(),
                   .pCode    = reinterpret_cast<uint32_t const*>(shaderCode.data()),
               }) >>
               ResultChecker();
    };
}  // namespace renderer::backend
