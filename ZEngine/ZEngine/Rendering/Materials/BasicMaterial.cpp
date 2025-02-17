#include <pch.h>
#include <Rendering/Materials/BasicMaterial.h>

using namespace ZEngine::Helpers;

namespace ZEngine::Rendering::Materials
{

    BasicMaterial::BasicMaterial() : ShaderMaterial(Shaders::ShaderBuiltInType::BASIC)
    {
        m_material_name = typeid(*this).name();
        m_texture.reset(Textures::CreateTexture(1, 1));
    }

    void BasicMaterial::SetTexture(const Ref<Textures::Texture>& texture)
    {
        m_texture = texture;
    }

    Ref<Textures::Texture> BasicMaterial::GetTexture() const
    {
        return m_texture;
    }

    void BasicMaterial::Apply(const Ref<Shaders::Shader>& shader)
    {
        // ShaderMaterial::Apply(shader);
        // m_texture->Bind();
    }
} // namespace ZEngine::Rendering::Materials
