#include <EditorCameraController.h>

using namespace ZEngine::Rendering::Cameras;
using namespace ZEngine::Helpers;

namespace Tetragrama::Controllers
{
    EditorCameraController::EditorCameraController(const ZEngine::Helpers::Ref<ZEngine::Windows::CoreWindow>& window, double distance, float yaw_angle_degree, float pitch_angle_degree) : PerspectiveCameraController(window, 0.0f)
    {
        m_process_event      = true;
        m_controller_type    = Controllers::CameraControllerType::PERSPECTIVE_CONTROLLER;
        m_perspective_camera = CreateRef<PerspectiveCamera>(m_camera_fov, m_aspect_ratio, m_camera_near, m_camera_far, glm::radians(yaw_angle_degree), glm::radians(pitch_angle_degree));
        m_perspective_camera->SetDistance(distance);
    }
} // namespace Tetragrama::Controllers
