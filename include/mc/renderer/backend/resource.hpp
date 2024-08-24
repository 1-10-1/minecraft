#pragma once

#include "assert.h"

#include <cstdint>
#include <limits>
#include <vector>

namespace renderer::backend
{
    class ResourceHandle
    {
    public:
        ResourceHandle() = default;
        ResourceHandle(uint64_t index, uint64_t creationNumber)
            : m_index { index }, m_creationNumber { creationNumber } {};

        bool operator==(ResourceHandle const& rhs) const { return m_creationNumber == rhs.m_creationNumber; }

        bool isValid() const { return m_creationNumber != std::numeric_limits<uint64_t>::max(); };

        uint64_t getIndex() const
        {
            assert(this->isValid());

            return m_index;
        }

        operator uint64_t() const { return this->getIndex(); }

        operator bool() const { return this->isValid(); }

        static constexpr uint64_t invalidCreationNumber = std::numeric_limits<uint64_t>::max();

    private:
        uint64_t m_index          = 0;
        uint64_t m_creationNumber = ResourceHandle::invalidCreationNumber;
    };

    class ResourceBase
    {
    public:
        ResourceBase() = delete;

        ResourceBase(ResourceHandle handle) : m_handle { handle } {};

        auto getHandle() const -> ResourceHandle { return m_handle; }

        auto operator==(ResourceBase& rhs) -> bool { return m_handle == rhs.m_handle; }

    protected:
        ResourceHandle m_handle;
    };

    template<typename Resource>
    class ResourceAccessor;

    template<typename Resource, typename CreationParameters>
    class ResourceManager
    {
        static_assert(std::is_base_of_v<ResourceBase, Resource> && std::movable<Resource>);

        static_assert(
            requires(ResourceManager<Resource, CreationParameters> const& u) { ResourceAccessor { u }; },
            "An accessor for this resource is not present");

    public:
        ResourceManager()  = default;
        ~ResourceManager() = default;

        ResourceManager(ResourceManager const&)            = delete;
        ResourceManager& operator=(ResourceManager const&) = delete;

        ResourceManager(ResourceManager&&)            = default;
        ResourceManager& operator=(ResourceManager&&) = default;

        ResourceHandle create(CreationParameters params)
        {
            Resource resource = Resource(m_resources.size(), m_creationCounter++, std::move(params));

            if (m_dormantResources.empty())
            {
                return m_resources.emplace_back(std::move(resource)).handle;
            }
            else
            {
                Resource& dormantResource = m_resources[m_dormantResources.back()] = std::move(resource);

                m_dormantResources.pop_back();

                return dormantResource.handle;
            }
        };

        void destroy(ResourceHandle handle)
        {
            assert(isValid(handle));

            m_resources[handle.getIndex()] = Resource();
            m_dormantResources.push_back(handle.getIndex());
        };

        bool isValid(ResourceHandle handle) const
        {
            return handle.isValid() && handle.getIndex() <= m_resources.size() &&
                   m_resources[handle.getIndex()].handle == handle;
        };

        ResourceAccessor<Resource> access(ResourceHandle handle) const { return getResource(handle); };

    private:
        Resource const& getResource(ResourceHandle handle) const
        {
            assert(isValid(handle));

            return m_resources[handle.getIndex()];
        }

        std::vector<Resource> m_resources;
        std::vector<uint32_t> m_dormantResources;

        uint64_t m_creationCounter { 0 };
    };

    template<typename Resource>
    class ResourceAccessor
    {
        ResourceAccessor() = delete;
    };
}  // namespace renderer::backend
