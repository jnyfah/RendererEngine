#pragma once
#include <Controllers/ICameraController.h>
#include <ZEngine/Rendering/Cameras/PerspectiveCamera.h>
#include <ZEngine/Windows/Inputs/IInputEventCallback.h>
#include <mutex>

namespace Tetragrama::Controllers
{

    class PerspectiveCameraController : public ICameraController, public ZEngine::Windows::Inputs::IMouseEventCallback
    {
    public:
        explicit PerspectiveCameraController(const ZEngine::Helpers::Ref<ZEngine::Windows::CoreWindow>& window) : ICameraController(window), m_perspective_camera(new ZEngine::Rendering::Cameras::PerspectiveCamera(m_camera_fov, m_aspect_ratio, m_camera_near, m_camera_far))
        {
            m_position        = {0.0f, 0.0f, 1.5f};
            m_controller_type = CameraControllerType::PERSPECTIVE_CONTROLLER;
            m_process_event   = true;
        }

        explicit PerspectiveCameraController(const ZEngine::Helpers::Ref<ZEngine::Windows::CoreWindow>& window, ZEngine::Rendering::Cameras::PerspectiveCamera* const camera) : ICameraController(window), m_perspective_camera(camera)
        {
            m_position        = {0.0f, 0.0f, 1.5f};
            m_controller_type = CameraControllerType::PERSPECTIVE_CONTROLLER;
            m_process_event   = true;
        }

        explicit PerspectiveCameraController(const ZEngine::Helpers::Ref<ZEngine::Windows::CoreWindow>& window, float aspect_ratio, ZEngine::Rendering::Cameras::PerspectiveCamera* const camera) : ICameraController(window, aspect_ratio), m_perspective_camera(camera)
        {
            m_position        = {0.0f, 0.0f, 1.5f};
            m_controller_type = CameraControllerType::PERSPECTIVE_CONTROLLER;
            m_process_event   = true;
        }

        explicit PerspectiveCameraController(const ZEngine::Helpers::Ref<ZEngine::Windows::CoreWindow>& window, float aspect_ratio) : ICameraController(window, aspect_ratio), m_perspective_camera(new ZEngine::Rendering::Cameras::PerspectiveCamera(m_camera_fov, m_aspect_ratio, m_camera_near, m_camera_far))
        {
            m_position        = {0.0f, 0.0f, 1.5f};
            m_controller_type = CameraControllerType::PERSPECTIVE_CONTROLLER;
            m_process_event   = true;
        }

        virtual ~PerspectiveCameraController() = default;

        void                                                             Initialize() override;
        void                                                             Update(ZEngine::Core::TimeStep) override;
        bool                                                             OnEvent(ZEngine::Core::CoreEvent&) override;

        const ZEngine::Helpers::Ref<ZEngine::Rendering::Cameras::Camera> GetCamera() const override;

        void                                                             UpdateProjectionMatrix() override;

        virtual glm::vec3                                                GetPosition() const override;
        virtual void                                                     SetPosition(const glm::vec3& position) override;

        virtual float                                                    GetFieldOfView() const;
        virtual void                                                     SetFieldOfView(float rad_fov);

        virtual float                                                    GetNear() const;
        virtual void                                                     SetNear(float value);

        virtual float                                                    GetFar() const;
        virtual void                                                     SetFar(float value);

        void                                                             SetViewport(float width, float height);
        void                                                             SetTarget(const glm::vec3& target);

        virtual void                                                     ResumeEventProcessing();
        virtual void                                                     PauseEventProcessing();

    public:
        bool OnMouseButtonPressed(ZEngine::Windows::Events::MouseButtonPressedEvent&) override
        {
            return false;
        }

        bool OnMouseButtonReleased(ZEngine::Windows::Events::MouseButtonReleasedEvent&) override
        {
            return false;
        }

        bool OnMouseButtonMoved(ZEngine::Windows::Events::MouseButtonMovedEvent&) override
        {
            return false;
        }

        bool OnMouseButtonWheelMoved(ZEngine::Windows::Events::MouseButtonWheelEvent&) override;

    protected:
        float                                                                 m_camera_fov{90.0f};
        float                                                                 m_camera_near{1.f};
        float                                                                 m_camera_far{1000.0f};
        std::recursive_mutex                                                  m_event_mutex;
        bool                                                                  m_process_event{true};
        glm::vec3                                                             m_camera_target{0.0f, 0.0f, 0.0f};
        ZEngine::Helpers::Ref<ZEngine::Rendering::Cameras::PerspectiveCamera> m_perspective_camera;
    };
} // namespace Tetragrama::Controllers
