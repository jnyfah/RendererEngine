#include <pch.h>
#include <UILayer.h>
#include <Messenger.h>
#include <MessageToken.h>

using namespace Tetragrama::Messengers;

namespace Tetragrama::Layers
{
    UILayer::~UILayer() {}
    void UILayer::Initialize()
    {
        ImguiLayer::Initialize();

        m_dockspace_component  = std::make_shared<Components::DockspaceUIComponent>();
        m_scene_component      = std::make_shared<Components::SceneViewportUIComponent>();
        m_editor_log_component = std::make_shared<Components::LogUIComponent>();
        m_demo_component       = std::make_shared<Components::DemoUIComponent>();

        m_project_view_component   = std::make_shared<Components::ProjectViewUIComponent>();
        m_inspector_view_component = std::make_shared<Components::InspectorViewUIComponent>();
        m_hierarchy_view_component = std::make_shared<Components::HierarchyViewUIComponent>();

        this->AddUIComponent(m_dockspace_component);
        this->AddUIComponent(m_demo_component);
        this->AddUIComponent(m_scene_component);
        this->AddUIComponent(m_editor_log_component);
        this->AddUIComponent(m_project_view_component);
        this->AddUIComponent(m_inspector_view_component);
        this->AddUIComponent(m_hierarchy_view_component);

        // Register components

        IMessenger::Register<ZEngine::Components::UI::UIComponent, GenericMessage<ZEngine::Event::WindowClosedEvent>>(m_dockspace_component,
            EDITOR_COMPONENT_DOCKSPACE_REQUEST_EXIT,
            std::bind(&Components::DockspaceUIComponent::RequestExitMessageHandler, reinterpret_cast<Components::DockspaceUIComponent*>(m_dockspace_component.get()),
                std::placeholders::_1));

        IMessenger::Register<ZEngine::Components::UI::UIComponent, GenericMessage<ZEngine::Rendering::Renderers::Contracts::FramebufferViewLayout>>(m_scene_component, EDITOR_COMPONENT_SCENEVIEWPORT_TEXTURE_AVAILABLE,
            std::bind(&Components::SceneViewportUIComponent::SceneTextureAvailableMessageHandler, reinterpret_cast<Components::SceneViewportUIComponent*>(m_scene_component.get()),
                std::placeholders::_1));

        IMessenger::Register<ZEngine::Components::UI::UIComponent, GenericMessage<bool>>(m_scene_component, EDITOR_COMPONENT_SCENEVIEWPORT_FOCUSED,
            std::bind(&Components::SceneViewportUIComponent::SceneViewportFocusedMessageHandler, reinterpret_cast<Components::SceneViewportUIComponent*>(m_scene_component.get()),
                std::placeholders::_1));

        IMessenger::Register<ZEngine::Components::UI::UIComponent, GenericMessage<bool>>(m_scene_component, EDITOR_COMPONENT_SCENEVIEWPORT_UNFOCUSED,
            std::bind(&Components::SceneViewportUIComponent::SceneViewportUnfocusedMessageHandler, reinterpret_cast<Components::SceneViewportUIComponent*>(m_scene_component.get()),
                std::placeholders::_1));

        IMessenger::Register<ZEngine::Components::UI::UIComponent, GenericMessage<std::pair<float, float>>>(m_scene_component, EDITOR_COMPONENT_SCENEVIEWPORT_RESIZED,
            std::bind(&Components::SceneViewportUIComponent::SceneViewportResizedMessageHandler, reinterpret_cast<Components::SceneViewportUIComponent*>(m_scene_component.get()),
                std::placeholders::_1));

        IMessenger::Register<ZEngine::Components::UI::UIComponent, GenericMessage<std::pair<int, int>>>(m_scene_component, EDITOR_COMPONENT_SCENEVIEWPORT_CLICKED,
            std::bind(&Components::SceneViewportUIComponent::SceneViewportClickedMessageHandler, reinterpret_cast<Components::SceneViewportUIComponent*>(m_scene_component.get()),
                std::placeholders::_1));

        IMessenger::Register<ZEngine::Components::UI::UIComponent, EmptyMessage>(m_scene_component, EDITOR_COMPONENT_SCENEVIEWPORT_REQUEST_RECOMPUTATION,
            std::bind(&Components::SceneViewportUIComponent::SceneViewportRequestRecomputationMessageHandler,
                reinterpret_cast<Components::SceneViewportUIComponent*>(m_scene_component.get()), std::placeholders::_1));

        IMessenger::Register<ZEngine::Components::UI::UIComponent, GenericMessage<ZEngine::Ref<ZEngine::Rendering::Scenes::GraphicScene>>>(m_hierarchy_view_component,
            EDITOR_RENDER_LAYER_SCENE_AVAILABLE,
            std::bind(&Components::HierarchyViewUIComponent::SceneAvailableMessageHandler,
                reinterpret_cast<Components::HierarchyViewUIComponent*>(m_hierarchy_view_component.get()), std::placeholders::_1));

        IMessenger::Register<ZEngine::Components::UI::UIComponent, GenericMessage<ZEngine::Ref<EditorCameraController>>>(m_hierarchy_view_component,
            EDITOR_RENDER_LAYER_CAMERA_CONTROLLER_AVAILABLE,
            std::bind(&Components::HierarchyViewUIComponent::EditorCameraAvailableMessageHandler,
                reinterpret_cast<Components::HierarchyViewUIComponent*>(m_hierarchy_view_component.get()), std::placeholders::_1));

        IMessenger::Register<ZEngine::Components::UI::UIComponent, GenericMessage<bool>>(m_hierarchy_view_component, EDITOR_COMPONENT_HIERARCHYVIEW_REQUEST_RESUME_OR_PAUSE_RENDER,
            std::bind(&Components::HierarchyViewUIComponent::RequestStartOrPauseRenderMessageHandler,
                reinterpret_cast<Components::HierarchyViewUIComponent*>(m_hierarchy_view_component.get()), std::placeholders::_1));

        IMessenger::Register<ZEngine::Components::UI::UIComponent, GenericMessage<bool>>(m_inspector_view_component, EDITOR_COMPONENT_INSPECTORVIEW_REQUEST_RESUME_OR_PAUSE_RENDER,
            std::bind(&Components::InspectorViewUIComponent::RequestStartOrPauseRenderMessageHandler,
                reinterpret_cast<Components::InspectorViewUIComponent*>(m_inspector_view_component.get()), std::placeholders::_1));

        IMessenger::Register<ZEngine::Components::UI::UIComponent, PointerValueMessage<ZEngine::Rendering::Entities::GraphicSceneEntity>>(m_inspector_view_component,
            EDITOR_COMPONENT_HIERARCHYVIEW_NODE_SELECTED,
            std::bind(&Components::InspectorViewUIComponent::SceneEntitySelectedMessageHandler,
                reinterpret_cast<Components::InspectorViewUIComponent*>(m_inspector_view_component.get()), std::placeholders::_1));

        IMessenger::Register<ZEngine::Components::UI::UIComponent, EmptyMessage>(m_inspector_view_component, EDITOR_COMPONENT_HIERARCHYVIEW_NODE_UNSELECTED,
            std::bind(&Components::InspectorViewUIComponent::SceneEntityUnSelectedMessageHandler,
                reinterpret_cast<Components::InspectorViewUIComponent*>(m_inspector_view_component.get()), std::placeholders::_1));

        IMessenger::Register<ZEngine::Components::UI::UIComponent, EmptyMessage>(m_inspector_view_component, EDITOR_COMPONENT_HIERARCHYVIEW_NODE_DELETED,
            std::bind(&Components::InspectorViewUIComponent::SceneEntityDeletedMessageHandler,
                reinterpret_cast<Components::InspectorViewUIComponent*>(m_inspector_view_component.get()), std::placeholders::_1));

        IMessenger::Register<ZEngine::Components::UI::UIComponent, GenericMessage<ZEngine::Ref<ZEngine::Rendering::Scenes::GraphicScene>>>(m_inspector_view_component,
            EDITOR_RENDER_LAYER_SCENE_AVAILABLE,
            std::bind(&Components::InspectorViewUIComponent::SceneAvailableMessageHandler,
                reinterpret_cast<Components::InspectorViewUIComponent*>(m_inspector_view_component.get()), std::placeholders::_1));
    }
} // namespace Tetragrama::Layers
