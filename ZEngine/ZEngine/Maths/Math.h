#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>

namespace ZEngine
{
    namespace Maths = glm;
}

// GLM extensions
namespace glm
{
    using Vector4    = vec4;
    using Vector3    = vec3;
    using Vector2    = vec2;
    using Vector1    = vec1;

    using Matrix4    = mat4;
    using Matrix3    = mat3;
    using Matrix2    = mat2;

    using Quaternion = quat;

    // From TheCherno : https://github.com/TheCherno/Hazel/blob/master/Hazel/src/Hazel/Math/Math.h
    bool DecomposeTransformComponent(const Matrix4& transform, Vector3& translation, Vector3& rotation, Vector3& scale);
} // namespace glm
