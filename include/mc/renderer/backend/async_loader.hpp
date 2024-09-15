#pragma once

#include "buffer.hpp"
#include "constants.hpp"
#include "device.hpp"
#include "texture.hpp"

#include <TaskScheduler.h>

namespace renderer::backend
{
    class RendererBackend;

    struct FileLoadRequest
    {
        std::string path;

        ResourceHandle buffer, texture;
    };

    struct UploadRequest
    {
        void* data          = nullptr;
        uint32_t* completed = nullptr;

        ResourceHandle texture, cpuBuffer, gpuBuffer;
    };

    struct AsynchronousLoader
    {
        AsynchronousLoader()  = default;
        ~AsynchronousLoader() = default;

        AsynchronousLoader(enki::TaskScheduler* taskScheduler,
                           RendererBackend& renderer,
                           Device& device,
                           ResourceManager<GPUBuffer>& bufferManager,
                           ResourceManager<Texture>& textureManager);

        AsynchronousLoader& operator=(AsynchronousLoader&& rhs)
        {
            if (this == &rhs)
            {
                return *this;
            }

            task_scheduler            = rhs.task_scheduler;
            renderer                  = rhs.renderer;
            device                    = rhs.device;
            textureManager            = rhs.textureManager;
            bufferManager             = rhs.bufferManager;
            completed                 = rhs.completed;
            textureReady              = rhs.textureReady;
            cpuBufferReady            = rhs.cpuBufferReady;
            gpuBufferReady            = rhs.gpuBufferReady;
            stagingBufferOffset       = rhs.stagingBufferOffset.load();
            fileLoadRequests          = std::move(rhs.fileLoadRequests);
            uploadRequests            = std::move(rhs.uploadRequests);
            stagingBuffer             = std::move(rhs.stagingBuffer);
            commandBuffers            = std::move(rhs.commandBuffers);
            commandPools              = std::move(rhs.commandPools);
            transferCompleteSemaphore = std::move(rhs.transferCompleteSemaphore);
            transferFence             = std::move(rhs.transferFence);

            return *this;
        };

        void update();

        void requestTextureData(std::string filename, ResourceHandle const& texture);
        void requestBufferUpload(void* data, ResourceHandle const& handle);
        void requestBufferCopy(ResourceHandle const& src, ResourceHandle const& dst, uint32_t* completed);

        enki::TaskScheduler* task_scheduler       = nullptr;
        RendererBackend* renderer                 = nullptr;
        Device* device                            = nullptr;
        ResourceManager<Texture>* textureManager  = nullptr;
        ResourceManager<GPUBuffer>* bufferManager = nullptr;

        std::vector<FileLoadRequest> fileLoadRequests;
        std::vector<UploadRequest> uploadRequests;

        ResourceAccessor<GPUBuffer> stagingBuffer {};

        uint32_t* completed = nullptr;
        ResourceHandle textureReady {}, cpuBufferReady {}, gpuBufferReady {};
        std::atomic_size_t stagingBufferOffset = 0;

        // TODO(aether) come up with a method to initialise arrays containing class objects
        // that have a deleted default constructor
        // might as well have hardcoded "2" as the 2nd template parameter here
        std::array<vk::raii::CommandPool, kNumFramesInFlight> commandPools { nullptr, nullptr };
        std::array<vk::raii::CommandBuffer, kNumFramesInFlight> commandBuffers { nullptr, nullptr };

        vk::raii::Semaphore transferCompleteSemaphore = nullptr;
        vk::raii::Fence transferFence                 = nullptr;
    };
}  // namespace renderer::backend
