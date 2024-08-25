#pragma once

#include "mc/asserts.hpp"

#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

namespace renderer::backend
{
    class ResourceHandle
    {
    public:
        ResourceHandle() = default;
        ResourceHandle(uint64_t index, uint64_t creationNumber, [[maybe_unused]] std::string const& name = {})
            : m_index { index },
              m_creationNumber { creationNumber }
#if DEBUG
              ,
              m_name { name }
#endif
              {};

        ResourceHandle(ResourceHandle&&)                 = default;
        ResourceHandle(ResourceHandle const&)            = default;
        ResourceHandle& operator=(ResourceHandle&&)      = default;
        ResourceHandle& operator=(ResourceHandle const&) = default;

        bool operator==(ResourceHandle const& rhs) const { return m_creationNumber == rhs.m_creationNumber; }

        bool hasInitialized() const { return m_creationNumber != invalidCreationNumber; };

        uint64_t getIndex() const
        {
            MC_ASSERT_MSG(this->hasInitialized(), "Attempted to access an uninitialized handle");

            return m_index;
        }

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
        uint64_t m_index          = 0;
        uint64_t m_creationNumber = invalidCreationNumber;

#if DEBUG
        std::string m_name;
#endif
    };

    template<typename Resource>
    class ResourceAccessorBase;

    template<typename Resource>
    class ResourceAccessor;

    template<typename Resource>
    class ResourceManagerBase;

    class ResourceBase
    {
        template<typename Resource>
        friend class ResourceManagerBase;

    public:
        ResourceBase(ResourceBase&&)            = default;
        ResourceBase& operator=(ResourceBase&&) = default;

        ResourceBase(ResourceBase const&)            = delete;
        ResourceBase& operator=(ResourceBase const&) = delete;

        ResourceBase() = delete;

        ResourceBase(ResourceHandle handle) : m_handle { handle } {};

        auto getHandle() const -> ResourceHandle { return m_handle; }

        auto operator==(ResourceBase& rhs) -> bool { return m_handle == rhs.m_handle; }

    protected:
        ResourceHandle m_handle;
    };

    // To allow specific resources to extend managers via partial specialization and inheritence
    template<typename Resource>
    class ResourceManager final : public ResourceManagerBase<Resource>
    {
    };

    template<typename Resource>
    class ResourceManagerBase
    {
        friend class ResourceAccessorBase<Resource>;
        friend class ResourceManager<Resource>;

        static_assert(std::is_base_of_v<ResourceBase, Resource> && std::movable<Resource>);

        static_assert(
            requires(ResourceManager<Resource>& manager, ResourceHandle handle) {
                ResourceAccessor<Resource>(manager, handle);
            }, "An accessor for this resource is not present");

    public:
        ResourceManagerBase()  = default;
        ~ResourceManagerBase() = default;

        ResourceManagerBase(ResourceManagerBase const&)            = delete;
        ResourceManagerBase& operator=(ResourceManagerBase const&) = delete;

        ResourceManagerBase(ResourceManagerBase&&)            = default;
        ResourceManagerBase& operator=(ResourceManagerBase&&) = default;

        auto getExtraConstructionParams() { return std::make_tuple(); };

        // Using explicit object parameter here so we can use the derived class's getExtraParams() (if present)
        template<typename Self, typename... Args>
        constexpr ResourceHandle create(this Self&& self, std::string const& name, Args&&... args)
        {
            auto createResource = [](auto&&... args)
            {
                return Resource(std::forward<decltype(args)>(args)...);
            };

            Resource resource = std::apply(
                createResource,
                std::tuple_cat(
                    std::make_tuple(ResourceHandle(self.m_resources.size(), self.m_creationCounter++, name)),
                    std::tie(name),
                    self.getExtraConstructionParams(),
                    std::forward_as_tuple(std::forward<Args>(args)...)));

            if (self.m_dormantIndices.empty())
            {
                Resource& res = self.m_resources.emplace_back(std::move(resource));

                return res.getHandle();
            }
            else
            {
                Resource& dormantResource = self.m_resources[self.m_dormantIndices.back()] =
                    std::move(resource);

                self.m_dormantIndices.pop_back();

                return dormantResource.getHandle();
            }
        };

        void destroy(ResourceHandle handle)
        {
            assert(isValid(handle));

            m_resources[handle.getIndex()] = Resource();
            m_dormantIndices.push_back(handle.getIndex());
        };

        bool isValid(ResourceHandle handle) const
        {
            return handle.hasInitialized() && handle.getIndex() <= m_resources.size() &&
                   m_resources[handle.getIndex()].m_handle == handle;
        };

        ResourceAccessor<Resource> access(ResourceHandle handle)
        {
            return ResourceAccessor<Resource>(static_cast<ResourceManager<Resource>&>(*this), handle);
        };

    private:
        Resource& getResource(ResourceHandle handle)
        {
            MC_ASSERT_MSG(isValid(handle),
                          "Attempted to access {}",
                          handle.hasInitialized()
                              ? std::format("a deleted handle (previously named '{}')", handle.getName())
                              : "an uninitialized handle");

            return m_resources[handle.getIndex()];
        }

        std::vector<Resource> m_resources;
        std::vector<uint32_t> m_dormantIndices;

        uint64_t m_creationCounter { 0 };
    };

    template<typename T>
    class ResourceAccessorBase
    {
    protected:
        ResourceAccessorBase() = delete;

        ResourceAccessorBase(ResourceManager<T>& manager, ResourceHandle handle)
            : m_manager { manager }, m_handle { handle } {};

        auto get() -> T& { return m_manager.getResource(m_handle); }

        auto get() const -> T const& { return m_manager.getResource(m_handle); }

        ResourceManager<T>& m_manager;
        ResourceHandle m_handle {};
    };

    template<typename Resource>
    class ResourceAccessor
    {
        ResourceAccessor() = delete;
    };
}  // namespace renderer::backend
