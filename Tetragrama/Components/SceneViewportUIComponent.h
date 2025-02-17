#pragma once
#include <Components/Events/SceneViewportFocusedEvent.h>
#include <Components/Events/SceneViewportResizedEvent.h>
#include <Components/Events/SceneViewportUnfocusedEvent.h>
#include <Messengers/Message.h>
#include <UIComponent.h>
#include <imgui.h>
#include <vulkan/vulkan.h>

namespace Tetragrama::Components
{
    class SceneViewportUIComponent : public UIComponent
    {
    public:
        SceneViewportUIComponent(Layers::ImguiLayer* parent = nullptr, std::string_view name = "Scene", bool visibility = true);
        virtual ~SceneViewportUIComponent();

        void         Update(ZEngine::Core::TimeStep dt) override;
        virtual void Render(ZEngine::Rendering::Renderers::GraphicRenderer* const renderer, ZEngine::Hardwares::CommandBuffer* const command_buffer) override;

    public:
        // std::future<void> SceneViewportClickedMessageHandlerAsync(Messengers::ArrayValueMessage<int, 2>&);
        // std::future<void> SceneViewportFocusedMessageHandlerAsync(Messengers::GenericMessage<bool>&);
        // std::future<void> SceneViewportUnfocusedMessageHandlerAsync(Messengers::GenericMessage<bool>&);

    private:
        bool                                        m_is_window_focused{false};
        bool                                        m_is_window_hovered{false};
        bool                                        m_is_window_clicked{false};
        bool                                        m_refresh_texture_handle{false};
        bool                                        m_request_renderer_resize{false};
        bool                                        m_is_resizing{false};
        int                                         m_idle_frame_count     = 0;
        int                                         m_idle_frame_threshold = 9; // SwapchainImageCount * 3
        ImVec2                                      m_viewport_size{0.f, 0.f};
        ImVec2                                      m_content_region_available_size{0.f, 0.f};
        std::array<ImVec2, 2>                       m_viewport_bounds;
        ZEngine::Rendering::Textures::TextureHandle m_scene_texture{};
    };
} // namespace Tetragrama::Components
