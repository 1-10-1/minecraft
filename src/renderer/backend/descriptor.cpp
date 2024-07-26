#include "mc/logger.hpp"
#include <cmath>
#include <mc/renderer/backend/descriptor.hpp>
#include <mc/renderer/backend/vk_checker.hpp>
#include <mc/utils.hpp>

#include <ranges>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

namespace rn = std::ranges;
namespace vi = std::ranges::views;

namespace renderer::backend
{
    auto
    DescriptorLayoutBuilder::build(vk::raii::Device const& device,
                                   vk::ShaderStageFlags shaderStages,
                                   vk::DescriptorSetLayoutCreateFlags flags) -> vk::raii::DescriptorSetLayout
    {
        for (auto& binding : bindings)
        {
            binding.stageFlags |= shaderStages;
        }

        vk::DescriptorSetLayoutCreateInfo info = {
            .bindingCount = utils::size(bindings),
            .pBindings    = bindings.data(),
            .flags        = flags,
        };

        return device.createDescriptorSetLayout(info) >> ResultChecker();
    }

    void DescriptorWriter::write_buffer(
        int binding, vk::Buffer buffer, size_t size, size_t offset, vk::DescriptorType type)
    {
        writes[binding] = vk::WriteDescriptorSet()
                              .setDstBinding(binding)
                              .setDescriptorCount(1)
                              .setDescriptorType(type)
                              .setPBufferInfo(&bufferInfos.emplace_back(vk::DescriptorBufferInfo {
                                  .buffer = buffer,
                                  .offset = offset,
                                  .range  = size,
                              }));
    }

    void DescriptorWriter::write_image(int binding,
                                       vk::ImageView image,
                                       vk::Sampler sampler,
                                       vk::ImageLayout layout,
                                       vk::DescriptorType type)
    {
        writes[binding] = vk::WriteDescriptorSet()
                              .setDstBinding(binding)
                              .setDescriptorCount(1)
                              .setDescriptorType(type)
                              .setPImageInfo(&imageInfos.emplace_back(vk::DescriptorImageInfo {
                                  .sampler     = sampler,
                                  .imageView   = image,
                                  .imageLayout = layout,
                              }));
    }

    void DescriptorWriter::clear()
    {
        imageInfos.clear();
        writes.clear();
        bufferInfos.clear();
    }

    void DescriptorWriter::update_set(vk::raii::Device const& device, vk::DescriptorSet set)
    {
        device.updateDescriptorSets(writes | vi::values |
                                        vi::transform(
                                            [set](vk::WriteDescriptorSet& write)
                                            {
                                                write.dstSet = set;
                                                return write;
                                            }) |
                                        rn::to<std::vector>(),
                                    {});
    }

    auto DescriptorAllocatorGrowable::get_pool(vk::raii::Device const& device) -> vk::raii::DescriptorPool
    {
        vk::raii::DescriptorPool newPool { nullptr };

        if (!readyPools.empty())
        {
            newPool = std::move(readyPools.back());
            readyPools.pop_back();
        }
        else
        {
            newPool = create_pool(device, setsPerPool, ratios);

            setsPerPool = std::ceil(static_cast<double>(setsPerPool) * 1.5);

            if (setsPerPool > 4092)
            {
                setsPerPool = 4092;
                logger::warn("Descriptor set limit reached by descriptor pool");
            }
        }

        return newPool;
    }

    auto
    DescriptorAllocatorGrowable::create_pool(vk::raii::Device const& device,
                                             uint32_t setCount,
                                             std::span<PoolSizeRatio> poolRatios) -> vk::raii::DescriptorPool
    {
        std::vector<vk::DescriptorPoolSize> poolSizes(poolRatios.size());

        for (size_t i : vi::iota(0u, poolRatios.size()))
        {
            poolSizes[i] = {
                .type            = poolRatios[i].type,
                .descriptorCount = static_cast<uint32_t>(static_cast<double>(poolRatios[i].ratio) * setCount),
            };
        }

        vk::DescriptorPoolCreateInfo pool_info = {
            .maxSets       = setCount,
            .poolSizeCount = utils::size(poolSizes),
            .pPoolSizes    = poolSizes.data(),
        };

        return device.createDescriptorPool(pool_info) >> ResultChecker();
    }

    void DescriptorAllocatorGrowable::init(vk::raii::Device const& device,
                                           uint32_t initialSets,
                                           std::span<PoolSizeRatio> poolRatios)
    {
        ratios.clear();

        for (auto r : poolRatios)
        {
            ratios.push_back(r);
        }

        readyPools.push_back(create_pool(device, initialSets, poolRatios));

        setsPerPool =
            static_cast<uint32_t>(static_cast<float>(initialSets) * 1.5f);  // grow it next allocation
    }

    void DescriptorAllocatorGrowable::clear_pools(vk::raii::Device const& device)
    {
        for (auto& p : readyPools)
        {
            p.reset();
        }

        fullPools.clear();
    }

    void DescriptorAllocatorGrowable::destroy_pools(vk::raii::Device const& device)
    {
        readyPools.clear();
        fullPools.clear();
    }

    auto
    DescriptorAllocatorGrowable::allocate(vk::raii::Device const& device,
                                          vk::raii::DescriptorSetLayout const& layout) -> vk::DescriptorSet
    {
        // get or create a pool to allocate from
        vk::raii::DescriptorPool poolToUse = get_pool(device);

        vk::DescriptorSetAllocateInfo allocInfo = {
            .descriptorPool     = poolToUse,
            .descriptorSetCount = 1,
            .pSetLayouts        = &*layout,
        };

        auto ds = (*device).allocateDescriptorSets(allocInfo);

        auto result = ds.result;

        // Allocation failed. Try again
        if (result == vk::Result::eErrorOutOfPoolMemory || result == vk::Result::eErrorFragmentedPool)
        {
            fullPools.push_back(std::move(poolToUse));

            poolToUse                = get_pool(device);
            allocInfo.descriptorPool = poolToUse;

            ds = (*device).allocateDescriptorSets(allocInfo);
        }
        else
        {
            result >> ResultChecker();
        }

        readyPools.push_back(std::move(poolToUse));

        return ds.value[0];
    }

    DescriptorAllocator::DescriptorAllocator(vk::raii::Device const& device,
                                             uint32_t maxSets,
                                             std::span<PoolSizeRatio> poolRatios,
                                             vk::DescriptorPoolCreateFlags flags)
    {
        std::vector<vk::DescriptorPoolSize> poolSizes(poolRatios.size());

        for (auto [index, ratio] : vi::enumerate(poolRatios))
        {
            poolSizes[index] = {
                .type            = ratio.type,
                .descriptorCount = static_cast<uint32_t>(ratio.ratio) * maxSets,
            };
        }

        vk::DescriptorPoolCreateInfo pool_info = {
            .flags         = flags,
            .maxSets       = maxSets,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes    = poolSizes.data(),
        };

        m_pool = device.createDescriptorPool(pool_info) >> ResultChecker();
    }

    auto DescriptorAllocator::allocate(vk::Device device, vk::DescriptorSetLayout layout) -> vk::DescriptorSet
    {
        return (device.allocateDescriptorSets({
                    .descriptorPool     = m_pool,
                    .descriptorSetCount = 1,
                    .pSetLayouts        = &layout,
                }) >>
                ResultChecker())[0];
    }
}  // namespace renderer::backend
