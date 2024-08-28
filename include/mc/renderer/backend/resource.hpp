#pragma once

#include "mc/asserts.hpp"
#include "mc/logger.hpp"

#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <vector>

// TODO(aether) implement reference counting and copying ResourceAccessors
// TODO(aether) maybe remove ResourceCreationResult entirely? Seems to be of no use
// TODO(aether) Minimize copying ResourceHandles. They're not entirely cheap.

namespace renderer::backend
{
    class ResourceHandle
    {
        template<typename T>
        friend class ResourceManagerBase;

        ResourceHandle(uint64_t index, uint64_t creationNumber, [[maybe_unused]] std::string const& name = {})
            : m_index { index },
              m_creationNumber { creationNumber }
#if DEBUG
              ,
              m_name { name }
#endif
              {};

    public:
        ResourceHandle() = default;

        ResourceHandle(ResourceHandle&&)            = default;
        ResourceHandle& operator=(ResourceHandle&&) = default;

        ResourceHandle(ResourceHandle const&)            = default;
        ResourceHandle& operator=(ResourceHandle const&) = default;

        bool operator==(ResourceHandle const& rhs) const { return m_creationNumber == rhs.m_creationNumber; }

        bool hasInitialized() const { return m_creationNumber != invalidCreationNumber; };

        operator uint64_t() const { return this->getIndex(); }

        operator bool() const { return this->hasInitialized(); }

        std::string_view getName() const
        {
#if DEBUG
            return m_name;
#else
            return {};
#endif
        }

        static constexpr uint64_t invalidCreationNumber = std::numeric_limits<uint64_t>::max();

    private:
        uint64_t getIndex() const
        {
            MC_ASSERT_MSG(this->hasInitialized(), "Attempted to access an uninitialized handle");

            return m_index;
        }

        uint64_t m_index          = 0;
        uint64_t m_creationNumber = invalidCreationNumber;

#if DEBUG
        std::string m_name;
#endif
    };

    template<typename Resource>
    class ResourceAccessor;

    template<typename Resource>
    class ResourceManagerBase;

    template<typename Resource>
    class ResourceManager;

    template<typename Resource>
    class ResourceAccessorBase
    {
        friend class ResourceManagerBase<Resource>;

    public:
        virtual ~ResourceAccessorBase()
        {
            if (m_manager)
            {
                m_manager->decrementRefCount(m_handle);
            }
        }

        ResourceHandle const& getHandle() const { return m_handle; }

    protected:
        ResourceAccessorBase() = default;

        ResourceAccessorBase(ResourceManager<Resource>& manager, ResourceHandle handle)
            : m_manager { &manager }, m_handle { handle }
        {
            m_manager->incrementRefCount(m_handle);
        };

        ResourceAccessorBase(ResourceAccessorBase const& rhs) noexcept
            : m_manager { rhs.m_manager }, m_handle { rhs.m_handle }
        {
            if (m_manager)
            {
                m_manager->incrementRefCount(m_handle);
            }
        }

        ResourceAccessorBase(ResourceAccessorBase&& rhs) noexcept
            : m_manager { std::exchange(rhs.m_manager, nullptr) },
              m_handle { std::exchange(rhs.m_handle, {}) }
        {
            if (m_manager)
            {
                m_manager->incrementRefCount(m_handle);
            }
        }

        ResourceAccessorBase& operator=(ResourceAccessorBase const& rhs) noexcept
        {
            if (this == &rhs)
            {
                return *this;
            }

            m_manager = rhs.m_manager;
            m_handle  = rhs.m_handle;

            if (m_manager)
            {
                m_manager->incrementRefCount(m_handle);
            }

            return *this;
        }

        ResourceAccessorBase& operator=(ResourceAccessorBase&& rhs) noexcept
        {
            if (this == &rhs)
            {
                return *this;
            }

            m_manager = std::exchange(rhs.m_manager, nullptr);
            m_handle  = std::exchange(rhs.m_handle, {});

            if (m_manager)
            {
                m_manager->incrementRefCount(m_handle);
            }

            return *this;
        }

        auto get() -> Resource& { return m_manager->getResource(m_handle); }

        auto get() const -> Resource const& { return m_manager->getResource(m_handle); }

        ResourceManager<Resource>* m_manager { nullptr };
        ResourceHandle m_handle {};
    };

    class ResourceBase
    {
        template<typename Resource>
        friend class ResourceManagerBase;

    protected:
        ResourceBase() = default;

    public:
        ResourceBase(ResourceBase&&)            = default;
        ResourceBase& operator=(ResourceBase&&) = default;

        ResourceBase(ResourceBase const&)            = delete;
        ResourceBase& operator=(ResourceBase const&) = delete;

        virtual ~ResourceBase() = default;

        ResourceBase(ResourceHandle handle) : m_handle { handle } {};

        auto getHandle() const -> ResourceHandle { return m_handle; }

        auto operator==(ResourceBase& rhs) -> bool { return m_handle == rhs.m_handle; }

    protected:
        ResourceHandle m_handle {};
    };

    template<typename Resource>
    class ResourceManagerBase
    {
        friend class ResourceAccessorBase<Resource>;
        friend class ResourceManager<Resource>;

        static_assert(std::is_base_of_v<ResourceBase, Resource>);

        static_assert(
            requires(ResourceManager<Resource>& manager, ResourceHandle handle) {
                ResourceAccessor<Resource>(manager, handle);
            }, "An accessor for this resource is not present");
        ResourceManagerBase() = default;

    public:
        virtual ~ResourceManagerBase() = default;

        ResourceManagerBase(ResourceManagerBase const&)            = delete;
        ResourceManagerBase& operator=(ResourceManagerBase const&) = delete;

        ResourceManagerBase(ResourceManagerBase&&)            = default;
        ResourceManagerBase& operator=(ResourceManagerBase&&) = default;

        template<typename Self, typename... Args>
        auto create(this Self&& self, std::string const& name, Args&&... args) -> ResourceAccessor<Resource>
        {
            if (size_t dormResources = self.m_dormantIndices.size(); dormResources > 100)
            {
                logger::warn("Resource manager has an unexpected amount of inactive resources: {}",
                             dormResources);
            }

            auto createResource = [](auto&&... args)
            {
                return Resource(std::forward<decltype(args)>(args)...);
            };

            if (self.m_dormantIndices.empty())
            {
                RefCountedResource& res = self.m_resources.emplace_back(RefCountedResource {
                    .resource = std::apply(
                        createResource,
                        std::tuple_cat(std::make_tuple(ResourceHandle(
                                           self.m_resources.size(), self.m_creationCounter++, name)),
                                       std::tie(name),
                                       self.m_extraConstructionParams,
                                       std::forward_as_tuple(std::forward<Args>(args)...))),
                });

                return ResourceAccessor<Resource>(*dynamic_cast<ResourceManager<Resource>*>(&self),
                                                  res.resource.getHandle());
            }
            else
            {
                RefCountedResource& dormantResource = self.m_resources[self.m_dormantIndices.back()] = {
                    .resource = std::apply(
                        createResource,
                        std::tuple_cat(std::make_tuple(ResourceHandle(
                                           self.m_dormantIndices.back(), self.m_creationCounter++, name)),
                                       std::tie(name),
                                       self.m_extraConstructionParams,
                                       std::forward_as_tuple(std::forward<Args>(args)...))),
                };

                self.m_dormantIndices.pop_back();

                return ResourceAccessor<Resource>(*dynamic_cast<ResourceManager<Resource>*>(&self),
                                                  dormantResource.resource.getHandle());
            }
        };

        void destroy(ResourceHandle const& handle)

        {
            MC_ASSERT(isValid(handle));

            m_resources[handle.getIndex()] = {};
            m_dormantIndices.push_back(handle.getIndex());
        };

        auto access(ResourceHandle const& handle) -> ResourceAccessor<Resource>

        {
            MC_ASSERT(isValid(handle));

            return ResourceAccessor<Resource>(*dynamic_cast<ResourceManager<Resource>*>(this), handle);
        };

        bool isValid(ResourceHandle const& handle) const
        {
            return handle.hasInitialized() && handle.getIndex() <= m_resources.size() &&
                   m_resources[handle.getIndex()].resource.m_handle == handle;
        };

        size_t getNumResources() { return m_resources.size(); };

        size_t getNumActiveResources() { return m_resources.size() - m_dormantIndices.size(); };

    private:
        struct RefCountedResource
        {
            Resource resource;
            uint32_t refCount;

            operator Resource&() { return resource; }
        };

        Resource& getResource(ResourceHandle const& handle)
        {
            MC_ASSERT_MSG(isValid(handle),
                          "Attempted to access {}",
                          handle.hasInitialized()
                              ? std::format("a deleted handle (previously named '{}')", handle.getName())
                              : "an uninitialized handle");

            return m_resources[handle.getIndex()];
        }

        RefCountedResource& getRefCoutedResource(ResourceHandle const& handle)
        {
            MC_ASSERT_MSG(isValid(handle),
                          "Attempted to access {}",
                          handle.hasInitialized()
                              ? std::format("a deleted handle (previously named '{}')", handle.getName())
                              : "an uninitialized handle");

            return m_resources[handle.getIndex()];
        }

        auto getExtraConstructionParams() { return std::make_tuple(); };

        void incrementRefCount(ResourceHandle const& handle) { getRefCoutedResource(handle).refCount++; }

        void decrementRefCount(ResourceHandle const& handle)
        {
            if (--getRefCoutedResource(handle).refCount == 0)
            {
                destroy(handle);
            };
        };

        std::vector<RefCountedResource> m_resources;

        std::vector<uint32_t> m_dormantIndices;

        uint64_t m_creationCounter { 0 };
    };

    // To allow specific resources to extend managers via partial specialization and inheritence
    template<typename Resource>
    class ResourceManager final : public ResourceManagerBase<Resource>
    {
        friend class ResourceManagerBase<Resource>;

        std::tuple<> m_extraConstructionParams {};

    public:
        ResourceManager(ResourceManager&&)            = default;
        ResourceManager& operator=(ResourceManager&&) = default;

        ResourceManager(ResourceManager const&)            = delete;
        ResourceManager& operator=(ResourceManager const&) = delete;
    };
}  // namespace renderer::backend
