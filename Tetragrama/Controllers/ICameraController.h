#pragma once
#include <Controllers/CameraControllerTypeEnums.h>
#include <Controllers/IController.h>
#include <ZEngine/Rendering/Cameras/Camera.h>
#include <ZEngine/Windows/CoreWindow.h>

namespace Tetragrama::Controllers
{

    struct ICameraController : public IController
    {
        ICameraController() {}
        ICameraController(const ZEngine::Helpers::Ref<ZEngine::Windows::CoreWindow>& window) : m_window(window) {}
        ICameraController(const ZEngine::Helpers::Ref<ZEngine::Windows::CoreWindow>& window, float aspect_ratio, bool can_rotate = false) : m_aspect_ratio(aspect_ratio), m_can_rotate(can_rotate), m_window(window) {}

        virtual ~ICameraController()                                                                                    = default;

        virtual glm::vec3                                                        GetPosition() const                    = 0;
        virtual void                                                             SetPosition(const glm::vec3& position) = 0;
        virtual const ZEngine::Helpers::Ref<ZEngine::Rendering::Cameras::Camera> GetCamera() const                      = 0;
        virtual void                                                             UpdateProjectionMatrix()               = 0;

        float                                                                    GetRotationAngle() const
        {
            return m_rotation_angle;
        }

        float GetZoomFactor() const
        {
            return m_zoom_factor;
        }

        float GetMoveSpeed() const
        {
            return m_move_speed;
        }

        float GetRotationSpeed() const
        {
            return m_rotation_speed;
        }

        float GetAspectRatio() const
        {
            return m_aspect_ratio;
        }

        CameraControllerType GetControllerType() const
        {
            return m_controller_type;
        }

        void SetRotationAngle(float angle)
        {
            m_rotation_angle = angle;
        }

        void SetZoomFactor(float factor)
        {
            m_zoom_factor = factor;
        }

        void SetMoveSpeed(float speed)
        {
            m_move_speed = speed;
        }

        void SetRotationSpeed(float speed)
        {
            m_rotation_speed = speed;
        }

        void SetAspectRatio(float ar)
        {
            m_aspect_ratio = ar;
            UpdateProjectionMatrix();
        }

    protected:
        glm::vec3                                               m_position{0.0f, 0.0f, 10.0f};
        float                                                   m_rotation_angle{0.0f};
        float                                                   m_zoom_factor{1.0f};
        float                                                   m_move_speed{0.05f};
        float                                                   m_rotation_speed{0.05f};
        float                                                   m_aspect_ratio{0.0f};
        bool                                                    m_can_rotate{false};
        CameraControllerType                                    m_controller_type{CameraControllerType::UNDEFINED};
        ZEngine::Helpers::WeakRef<ZEngine::Windows::CoreWindow> m_window;
    };
} // namespace Tetragrama::Controllers
