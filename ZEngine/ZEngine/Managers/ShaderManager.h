#pragma once
#include <Managers/IAssetManager.h>
#include <Rendering/Shaders/Shader.h>

namespace ZEngine::Hardwares
{
    struct VulkanDevice;
}

namespace ZEngine::Managers
{

    class ShaderManager : public IAssetManager<Rendering::Shaders::Shader>
    {
    public:
        ShaderManager();
        virtual ~ShaderManager() = default;

        /**
         * Add a shader to the Shader manager store
         *
         * @param name Name of the shader. This name must be unique in the entire store
         * @param filename Path to find the shader source file in the system
         * @return A shader instance
         */
        Helpers::Ref<Rendering::Shaders::Shader>        Add(const char* name, const char* filename) override;

        /**
         * Add a shader to the Shader manager store
         *
         * @param filename Path to find the shader source file in the system
         * @return A shader instance
         */
        Helpers::Ref<Rendering::Shaders::Shader>        Load(const char* filename) override;

        static const std::string                        GetFragmentFilename(std::string_view key);
        static const std::string                        GetVertexFilename(std::string_view key);

        static Helpers::Ref<Rendering::Shaders::Shader> Get(Hardwares::VulkanDevice* device, ZEngine::Rendering::Specifications::ShaderSpecification& spec);

    private:
        static const std::string_view                                                     s_base_dir;
        static const std::unordered_map<std::string, std::pair<std::string, std::string>> s_shader_path;
        static std::unordered_map<std::string, Helpers::Ref<Rendering::Shaders::Shader>>  s_shader_mappings;
    };
} // namespace ZEngine::Managers
