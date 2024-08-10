#include <mc/exceptions.hpp>
#include <mc/logger.hpp>
#include <mc/renderer/backend/constants.hpp>
#include <mc/renderer/backend/device.hpp>
#include <mc/renderer/backend/instance.hpp>
#include <mc/renderer/backend/renderer_backend.hpp>
#include <mc/renderer/backend/vk_checker.hpp>
#include <mc/utils.hpp>

#include <unordered_set>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

namespace
{
    using namespace renderer::backend;

    // clang-format off
    constexpr std::array requiredExtensions
    {
        vk::KHRSwapchainExtensionName,
#if PROFILED
        vk::KHRCalibratedTimestampsExtensionName
#endif
    };

    // clang-format on

    bool areAllQueueFamiliesPresent(QueueFamilyIndices const& indices)
    {
        auto max = std::numeric_limits<uint32_t>::max();

        return indices.presentFamily != max && indices.graphicsFamily != max && indices.transferFamily != max;
    };

    auto checkDeviceExtensionSupport(vk::PhysicalDevice device) -> bool
    {
        std::vector<vk::ExtensionProperties> availableExtensions =
            device.enumerateDeviceExtensionProperties() >> ResultChecker();

        std::unordered_set<std::string> requiredExtensionsSet(requiredExtensions.begin(),
                                                              requiredExtensions.end());

        for (auto const& extension : availableExtensions)
        {
            requiredExtensionsSet.erase(static_cast<char const*>(extension.extensionName));
        }

        return requiredExtensionsSet.empty();
    }

    auto findQueueFamilies(vk::PhysicalDevice device, vk::SurfaceKHR surface) -> QueueFamilyIndices
    {
        QueueFamilyIndices indices;

        std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();

        for (uint32_t i = 0; auto const& queueFamily : queueFamilies)
        {
            if (areAllQueueFamiliesPresent(indices))
            {
                break;
            }

            if (queueFamily.queueFlags & vk::QueueFlagBits::eTransfer &&
                queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
            {
                logger::debug("Found a dedicated transfer queue!");
                indices.transferFamily = i;
            }

            if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
            {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = 0u;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

            if (device.getSurfaceSupportKHR(i, surface) >> ResultChecker())
            {
                indices.presentFamily = i;
            }

            ++i;
        }

        uint32_t invalidIndex = std::numeric_limits<uint32_t>::max();

        if (indices.transferFamily == invalidIndex && indices.graphicsFamily != invalidIndex)
        {
            logger::debug("Could not find a dedicated transfer queue. Using the graphics "
                          "queue for this purpose.");

            indices.transferFamily = indices.graphicsFamily;
        }

        return indices;
    }
}  // namespace

namespace renderer::backend
{
    struct DeviceCandidate
    {
        vk::raii::PhysicalDevice device { nullptr };
        vk::PhysicalDeviceProperties properties {};
        vk::PhysicalDeviceFeatures features {};
        QueueFamilyIndices queueFamilyIndices {};
        int score { 0 };
    };

    Device::Device(Instance& instance, Surface& surface) : m_instance { &instance }
    {
        selectPhysicalDevice(instance, surface);
        selectLogicalDevice();
    }

    void Device::selectPhysicalDevice(Instance& instance, Surface& surface)
    {
        std::vector<vk::raii::PhysicalDevice> devices =
            instance->enumeratePhysicalDevices() >> ResultChecker();

        DeviceCandidate bestCandidate { .score = 0 };

        for (auto& device : devices)
        {
            QueueFamilyIndices queueFamilyIndices { findQueueFamilies(device, surface) };

            vk::PhysicalDeviceProperties deviceProperties = device.getProperties();
            vk::PhysicalDeviceFeatures deviceFeatures     = device.getFeatures();

            // clang-format off
            std::array necessaryConditions { std::to_array<std::pair<std::string_view, bool>>({
                { "Geometry shader availability",
                  static_cast<bool>(deviceFeatures.geometryShader)    },

                { "Anisotropy availability",
                  static_cast<bool>(deviceFeatures.samplerAnisotropy) },

                { "Necessary queues present",
                  areAllQueueFamiliesPresent(queueFamilyIndices)      },

                { "Necessary extensions supported",
                  checkDeviceExtensionSupport(device)                 }
            })};
            // clang-format on

            bool necessaryConditionsMet = true;

            for (auto [conditionName, condition] : necessaryConditions)
            {
                if (!condition)
                {
                    necessaryConditionsMet = false;

                    logger::trace("Graphics card {} rejected as it was unable to satisfy the "
                                  "following condition: {}",
                                  std::string_view { deviceProperties.deviceName },
                                  conditionName);

                    break;
                }
            }

            if (!necessaryConditionsMet)
            {
                continue;
            }

            int score = 0;

            if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
            {
                score += 1000;
            }

            score += static_cast<int>(deviceProperties.limits.maxImageDimension2D);

            if (score > bestCandidate.score)
            {
                bestCandidate = { .device             = device,
                                  .properties         = deviceProperties,
                                  .features           = deviceFeatures,
                                  .queueFamilyIndices = queueFamilyIndices,
                                  .score              = score };
            }
        }

        if (bestCandidate.score == 0)
        {
            MC_THROW Error(GraphicsError, "Could not find a suitable graphics card");
        }

        m_physicalHandle     = bestCandidate.device;
        m_queueFamilyIndices = bestCandidate.queueFamilyIndices;

        vk::SampleCountFlags sampleCounts = bestCandidate.properties.limits.framebufferColorSampleCounts &
                                            bestCandidate.properties.limits.framebufferDepthSampleCounts;

        std::array possibleSampleCounts {
            vk::SampleCountFlagBits::e2,  vk::SampleCountFlagBits::e4,  vk::SampleCountFlagBits::e8,
            vk::SampleCountFlagBits::e16, vk::SampleCountFlagBits::e32, vk::SampleCountFlagBits::e64,
        };

        m_sampleCount = vk::SampleCountFlagBits::e1;

        for (auto count : possibleSampleCounts)
        {
            if (count & sampleCounts)
            {
                m_sampleCount = count > kMaxSamples ? kMaxSamples : count;
            }
        }

        logger::info("Sample count set to {}", std::to_underlying(m_sampleCount));

        std::string_view deviceType;

        surface.refresh(m_physicalHandle);

        switch (bestCandidate.properties.deviceType)
        {
            case (vk::PhysicalDeviceType::eIntegratedGpu):
                deviceType = "integrated gpu";
                break;
            case (vk::PhysicalDeviceType::eDiscreteGpu):
                deviceType = "discrete GPU";
                break;
            case (vk::PhysicalDeviceType::eVirtualGpu):
                deviceType = "virtual GPU";
                break;
            case (vk::PhysicalDeviceType::eCpu):
                deviceType = "CPU";
                break;
        }

        logger::info("Using {} {} for rendering",
                     deviceType,
                     std::string_view { bestCandidate.properties.deviceName });
    }

    void Device::selectLogicalDevice()
    {
        std::unordered_set<uint32_t> queueFamilies = { m_queueFamilyIndices.graphicsFamily,
                                                       m_queueFamilyIndices.presentFamily,
                                                       m_queueFamilyIndices.transferFamily };

        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;

        float queuePriority = 1.0f;

        queueCreateInfos.reserve(queueFamilies.size());

        for (uint32_t queueFamily : queueFamilies)
        {
            queueCreateInfos.push_back(vk::DeviceQueueCreateInfo()
                                           .setQueueFamilyIndex(queueFamily)
                                           .setQueueCount(1)
                                           .setPQueuePriorities(&queuePriority));
        }

        vk::StructureChain<vk::PhysicalDeviceFeatures2,
                           vk::PhysicalDeviceVulkan12Features,
                           vk::PhysicalDeviceVulkan13Features> chain {
            {
                .features = { .sampleRateShading             = true,
                              .fillModeNonSolid              = true,
                              .samplerAnisotropy             = true,
                              .shaderStorageImageMultisample = true, },
            },
            {
                .descriptorIndexing  = true,
                .shaderSampledImageArrayNonUniformIndexing = true,
                .descriptorBindingPartiallyBound = true,
                .runtimeDescriptorArray = true,
                .bufferDeviceAddress = true,
            },

            {
                .synchronization2 = true,
                .dynamicRendering = true,
            },
        };

        m_logicalHandle =
            m_physicalHandle.createDevice(vk::DeviceCreateInfo()
                                              .setPNext(&chain.get<vk::PhysicalDeviceFeatures2>())
                                              .setQueueCreateInfos(queueCreateInfos)
                                              .setPEnabledExtensionNames(requiredExtensions)) >>
            ResultChecker();

        // Already checked that these families exist, no error handling needed here
        m_graphicsQueue = m_logicalHandle.getQueue(m_queueFamilyIndices.graphicsFamily, 0) >> ResultChecker();
        m_presentQueue  = m_logicalHandle.getQueue(m_queueFamilyIndices.presentFamily, 0) >> ResultChecker();
        m_transferQueue = m_logicalHandle.getQueue(m_queueFamilyIndices.transferFamily, 0) >> ResultChecker();
    }
}  // namespace renderer::backend
