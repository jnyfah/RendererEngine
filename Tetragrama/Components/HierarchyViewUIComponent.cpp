#include <pch.h>
#include <Editor.h>
#include <HierarchyViewUIComponent.h>
#include <ImGuizmo/ImGuizmo.h>
#include <Inputs/Keyboard.h>
#include <Inputs/Mouse.h>
#include <MessageToken.h>
#include <Messenger.h>
#include <ZEngine/Engine.h>
#include <ZEngine/Rendering/Scenes/GraphicScene.h>
#include <ZEngine/Windows/Inputs/IDevice.h>
#include <ZEngine/Windows/Inputs/KeyCodeDefinition.h>
#include <glm/glm.hpp>
#include <gtc/type_ptr.hpp>
#include <imgui.h>

using namespace ZEngine;
using namespace ZEngine::Helpers;
using namespace ZEngine::Windows::Inputs;
using namespace ZEngine::Rendering::Scenes;
using namespace Tetragrama::Inputs;
using namespace Tetragrama::Controllers;

namespace Tetragrama::Components
{
    HierarchyViewUIComponent::HierarchyViewUIComponent(Layers::ImguiLayer* parent, std::string_view name, bool visibility) : UIComponent(parent, name, visibility, false)
    {
        m_node_flag = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth /* | ImGuiTreeNodeFlags_DefaultOpen*/;
    }

    HierarchyViewUIComponent::~HierarchyViewUIComponent() {}

    void HierarchyViewUIComponent::Update(ZEngine::Core::TimeStep dt)
    {

        if (ParentLayer)
        {
            auto window = ParentLayer->GetAttachedWindow();
            if (IDevice::As<Keyboard>()->IsKeyPressed(ZENGINE_KEY_T, window))
            {
                m_gizmo_operation = ImGuizmo::OPERATION::TRANSLATE;
            }

            if (IDevice::As<Keyboard>()->IsKeyPressed(ZENGINE_KEY_R, window))
            {
                m_gizmo_operation = ImGuizmo::OPERATION::ROTATE;
            }

            if (IDevice::As<Keyboard>()->IsKeyPressed(ZENGINE_KEY_S, window))
            {
                m_gizmo_operation = ImGuizmo::OPERATION::SCALE;
            }
        }
    }

    void HierarchyViewUIComponent::Render(ZEngine::Rendering::Renderers::GraphicRenderer* const renderer, ZEngine::Hardwares::CommandBuffer* const command_buffer)
    {
        auto ctx          = reinterpret_cast<EditorContext*>(ParentLayer->ParentContext);
        auto render_scene = ctx->CurrentScenePtr->RenderScene;

        ImGui::Begin(Name.c_str(), (CanBeClosed ? &CanBeClosed : NULL), ImGuiWindowFlags_NoCollapse);
        if (ImGui::BeginPopupContextWindow(Name.c_str()))
        {
            if (ImGui::MenuItem("Create Empty"))
            {
                render_scene->CreateEntityAsync();
            }
            ImGui::EndPopup();
        }

        RenderTreeNodes();

        // 0 means left buttom
        if (ImGui::IsMouseDown(0) && ImGui::IsWindowHovered())
        {
            m_selected_node_identifier = -1;
            Messengers::IMessenger::SendAsync<Components::UIComponent, Messengers::EmptyMessage>(EDITOR_COMPONENT_HIERARCHYVIEW_NODE_UNSELECTED, Messengers::EmptyMessage{});
        }

        RenderGuizmo();

        ImGui::End();
    }

    void HierarchyViewUIComponent::RenderTreeNodes()
    {
        auto ctx          = reinterpret_cast<EditorContext*>(ParentLayer->ParentContext);
        auto render_scene = ctx->CurrentScenePtr->RenderScene;

        auto root_nodes   = render_scene->GetRootSceneNodes();

        for (int node : root_nodes)
        {
            RenderSceneNodeTree(node);
        }
    }

    void HierarchyViewUIComponent::RenderGuizmo()
    {
        if (m_selected_node_identifier <= -1)
        {
            return;
        }

        // auto entity_wrapper = GraphicScene::GetSceneNodeEntityWrapper(m_selected_node_identifier);
        auto ctx          = reinterpret_cast<EditorContext*>(ParentLayer->ParentContext);
        auto render_scene = ctx->CurrentScenePtr->RenderScene;

        if (auto active_editor_camera = ctx->CameraControllerPtr)
        {
            auto       camera             = active_editor_camera->GetCamera();
            const auto camera_projection  = camera->GetPerspectiveMatrix();
            const auto camera_view_matrix = camera->GetViewMatrix();

            auto&      global_transform   = render_scene->GetSceneNodeGlobalTransform(m_selected_node_identifier);
            auto       initial_transform  = global_transform;
            auto&      local_transform    = render_scene->GetSceneNodeLocalTransform(m_selected_node_identifier);

            if (camera && IDevice::As<Keyboard>()->IsKeyPressed(ZENGINE_KEY_F, Engine::GetWindow()))
            {
                active_editor_camera->SetTarget(glm::vec3(global_transform[0][3], global_transform[1][3], global_transform[2][3]));
            }

            // snapping
            float snap_value        = 0.5f;
            bool  is_snap_operation = IDevice::As<Keyboard>()->IsKeyPressed(ZENGINE_KEY_LEFT_CONTROL, Engine::GetWindow());
            if (is_snap_operation && static_cast<ImGuizmo::OPERATION>(m_gizmo_operation) == ImGuizmo::ROTATE)
            {
                snap_value = 45.0f;
            }
            float snap_array[3] = {snap_value, snap_value, snap_value};

            if (m_gizmo_operation > 0)
            {
                ImGuizmo::Manipulate(glm::value_ptr(camera_view_matrix), glm::value_ptr(camera_projection), (ImGuizmo::OPERATION) m_gizmo_operation, ImGuizmo::MODE::WORLD, glm::value_ptr(global_transform), nullptr, is_snap_operation ? snap_array : nullptr);
            }

            auto delta_transform = glm::inverse(initial_transform) * global_transform;
            local_transform      = local_transform * delta_transform;
            render_scene->MarkSceneNodeAsChanged(m_selected_node_identifier);

            if (ImGuizmo::IsUsing())
            {
                // ZEngine::Maths::Vector3 translation, rotation, scale;
                // ZEngine::Maths::DecomposeTransformComponent(transform, translation, rotation, scale);

                // entity_transform_component.SetPosition(translation);
                // entity_transform_component.SetScaleSize(scale);
                // entity_transform_component.SetRotation(rotation);
            }
        }
    }

    void HierarchyViewUIComponent::RenderSceneNodeTree(int node_identifier)
    {
        if (node_identifier < 0)
        {
            return;
        }

        auto        ctx                     = reinterpret_cast<EditorContext*>(ParentLayer->ParentContext);
        auto        render_scene            = ctx->CurrentScenePtr->RenderScene;

        const auto& node_hierarchy          = render_scene->GetSceneNodeHierarchy(node_identifier);
        auto        node_name               = render_scene->GetSceneNodeName(node_identifier);
        auto        node_identifier_string  = fmt::format("SceneNode_{0}", node_identifier);
        auto        flags                   = (node_hierarchy.FirstChild < 0) ? (ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_Bullet | m_node_flag) : m_node_flag;
        flags                              |= (m_selected_node_identifier == node_identifier) ? ImGuiTreeNodeFlags_Selected : 0;
        auto label                          = (!node_name.empty()) ? std::string(node_name) : fmt::format("Node_{0}", node_identifier);
        bool is_node_opened                 = ImGui::TreeNodeEx(node_identifier_string.c_str(), flags, "%s", label.c_str());

        if (ImGui::IsItemClicked())
        {
            m_selected_node_identifier = node_identifier;

            auto entity                = render_scene->GetSceneNodeEntityWrapper(m_selected_node_identifier);
            Messengers::IMessenger::SendAsync<Components::UIComponent, Messengers::GenericMessage<SceneEntity>>(EDITOR_COMPONENT_HIERARCHYVIEW_NODE_SELECTED, Messengers::GenericMessage<SceneEntity>{std::move(entity)});
        }

        if (is_node_opened)
        {
            /*
             * Popup features
             */
            bool request_entity_removal = false;
            if (ImGui::BeginPopupContextItem(node_identifier_string.c_str()))
            {
                if (ImGui::MenuItem("Create Empty child"))
                {
                    render_scene->CreateEntityAsync("Empty Entity", m_selected_node_identifier, node_hierarchy.DepthLevel + 1);
                }

                if (ImGui::MenuItem("Rename"))
                {
                }
                if (ImGui::MenuItem("Delete"))
                {
                    Messengers::IMessenger::SendAsync<Components::UIComponent, Messengers::EmptyMessage>(EDITOR_COMPONENT_HIERARCHYVIEW_NODE_DELETED, Messengers::EmptyMessage{});
                    request_entity_removal = true;
                }
                ImGui::EndPopup();
            }

            if (request_entity_removal)
            {
                render_scene->RemoveNodeAsync(node_identifier);
            }

            if (node_hierarchy.FirstChild > -1)
            {
                auto sibling_scene_node_collection = render_scene->GetSceneNodeSiblingCollection(node_hierarchy.FirstChild);
                // We consider first child as sibling node
                sibling_scene_node_collection.emplace(std::begin(sibling_scene_node_collection), node_hierarchy.FirstChild);
                for (auto sibling_identifier : sibling_scene_node_collection)
                {
                    RenderSceneNodeTree(sibling_identifier);
                }
            }
            ImGui::TreePop();
        }
    }
} // namespace Tetragrama::Components
