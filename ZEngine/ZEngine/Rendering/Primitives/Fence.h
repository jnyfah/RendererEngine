#pragma once
#include <Helpers/IntrusivePtr.h>
#include <vulkan/vulkan.h>

namespace ZEngine::Hardwares
{
    struct VulkanDevice;
}

namespace ZEngine::Rendering::Primitives
{
    enum class FenceState
    {
        Idle,
        Submitted,
        Undefined,
    };

    struct Fence : public Helpers::RefCounted
    {
        Fence(Hardwares::VulkanDevice* const device, bool as_signaled = false);
        ~Fence();

        Hardwares::VulkanDevice* Device = nullptr;
        bool                     IsSignaled();

        bool                     Wait(uint64_t timeout = 1000000000);
        void                     Reset();

        void                     SetState(FenceState state);
        FenceState               GetState() const;

        VkFence                  GetHandle() const;

    private:
        FenceState m_fence_state{FenceState::Idle};
        VkFence    m_handle{VK_NULL_HANDLE};
    };
} // namespace ZEngine::Rendering::Primitives