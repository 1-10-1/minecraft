#pragma once

#include "device.hpp"
#include "instance.hpp"
#include "vma.hpp"

namespace renderer::backend
{
    class Allocator
    {
    public:
        Allocator() = default;
        ~Allocator();

        Allocator(Instance const& instance, Device const& device);

        friend void swap(Allocator& first, Allocator& second) noexcept
        {
            std::swap(first.m_allocator, second.m_allocator);
        }

        Allocator(Allocator&& other) noexcept : Allocator() { swap(*this, other); };

        Allocator& operator=(Allocator other) noexcept
        {
            swap(*this, other);

            return *this;
        }

        [[nodiscard]] operator VmaAllocator() const { return m_allocator; }

        [[nodiscard]] auto get() const -> VmaAllocator { return m_allocator; }

    private:
        VmaAllocator m_allocator {};
    };
}  // namespace renderer::backend
