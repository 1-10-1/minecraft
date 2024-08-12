#pragma once

#include <mc/asserts.hpp>
#include <mc/logger.hpp>

#include <deque>
#include <map>
#include <span>
#include <unordered_map>
#include <vector>

#include <magic_enum.hpp>
#include <vulkan/vulkan_raii.hpp>

namespace renderer::backend
{
    class DescriptorLayoutBuilder
    {
    public:
        auto setBinding(uint32_t binding,
                        vk::DescriptorType type,
                        vk::ShaderStageFlags stages,
                        uint32_t count = 1) -> DescriptorLayoutBuilder&
        {
            // If we get the same binding again, ensure that only the stage flags differ
            if (m_bindings.find(binding) == m_bindings.end())
            {
                m_bindings[binding] = {
                    .binding = binding, .descriptorType = type, .descriptorCount = count, .stageFlags = stages
                };
            }
            else
            {
                MC_ASSERT(m_bindings[binding].descriptorType == type);
                MC_ASSERT(m_bindings[binding].descriptorCount == count);

                m_bindings[binding].stageFlags |= stages;
            }

            return *this;
        };

        void clear() { m_bindings.clear(); };

        auto build(vk::raii::Device const& device,
                   vk::DescriptorSetLayoutCreateFlags flags =
                       static_cast<vk::DescriptorSetLayoutCreateFlags>(0)) -> vk::raii::DescriptorSetLayout;

    private:
        std::unordered_map<uint32_t, vk::DescriptorSetLayoutBinding> m_bindings;
    };

    struct DescriptorWriter
    {
        std::deque<vk::DescriptorImageInfo> imageInfos {};
        std::deque<vk::DescriptorBufferInfo> bufferInfos {};
        std::map<int, vk::WriteDescriptorSet> writes {};

        DescriptorWriter& writeImage(int binding,
                                     vk::ImageView image,
                                     vk::Sampler sampler,
                                     vk::ImageLayout layout,
                                     vk::DescriptorType type);

        DescriptorWriter& writeImages(int binding,
                                      vk::ImageLayout layout,
                                      vk::DescriptorType type,
                                      std::span<vk::DescriptorImageInfo> images);

        DescriptorWriter&
        writeBuffer(int binding, vk::Buffer buffer, size_t size, size_t offset, vk::DescriptorType type);

        void clear();
        void updateSet(vk::raii::Device const& device, vk::DescriptorSet set);
    };

    struct DescriptorAllocatorGrowable
    {
    public:
        struct PoolSizeRatio
        {
            vk::DescriptorType type;
            float ratio;
        };

        void init(vk::raii::Device const& device, uint32_t initialSets, std::span<PoolSizeRatio> poolRatios);
        void clearPools(vk::raii::Device const& device);
        void destroyPools(vk::raii::Device const& device);

        [[nodiscard]] auto allocate(vk::raii::Device const& device,
                                    vk::raii::DescriptorSetLayout const& layout) -> vk::DescriptorSet;

    private:
        auto getPool(vk::raii::Device const& device) -> vk::raii::DescriptorPool;
        static auto createPool(vk::raii::Device const& device,
                               uint32_t setCount,
                               std::span<PoolSizeRatio> poolRatios) -> vk::raii::DescriptorPool;

        std::vector<PoolSizeRatio> ratios;
        std::vector<vk::raii::DescriptorPool> fullPools;
        std::vector<vk::raii::DescriptorPool> readyPools;
        uint32_t setsPerPool;
    };

    class DescriptorAllocator
    {
    public:
        struct PoolSizeRatio
        {
            vk::DescriptorType type;
            float ratio;
        };

        DescriptorAllocator()  = default;
        ~DescriptorAllocator() = default;

        DescriptorAllocator(
            vk::raii::Device const& device,
            uint32_t maxSets,
            std::span<PoolSizeRatio> poolRatios,
            vk::DescriptorPoolCreateFlags flags = static_cast<vk::DescriptorPoolCreateFlagBits>(0));

        DescriptorAllocator(DescriptorAllocator&&)                    = default;
        auto operator=(DescriptorAllocator&&) -> DescriptorAllocator& = default;

        [[nodiscard]] auto allocate(vk::Device device, vk::DescriptorSetLayout layout) -> vk::DescriptorSet;

        void clearDescriptors(vk::raii::Device const& device) { m_pool.reset(); }

        [[nodiscard]] auto getPool() const -> vk::raii::DescriptorPool const& { return m_pool; }

    private:
        vk::raii::DescriptorPool m_pool { nullptr };
    };
}  // namespace renderer::backend
