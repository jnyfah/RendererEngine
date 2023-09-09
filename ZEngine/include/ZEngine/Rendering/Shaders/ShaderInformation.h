#pragma once
#include <string>
#include <Rendering/Shaders/ShaderEnums.h>
#include <vulkan/vulkan.h>

namespace ZEngine::Rendering::Shaders
{

    struct ShaderInformation
    {
        VkPipelineShaderStageCreateInfo ShaderStageCreateInfo;
        VkShaderModule                  ShaderModule;
        std::vector<uint32_t>           BinarySource;
        /**
         * Shader identifier
         */
        unsigned int ShaderId;
        /**
         * Shader program identifier
         */
        unsigned int ProgramId;
        /**
         * Enumeration of shader
         * @see https://docs.gl/gl4/glCreateShader
         */
        unsigned int InternalType;
        /**
         * Wether the shader has been compiled once
         */
        bool CompiledOnce;
        /**
         * Graphic Shader type
         * @see ShaderType
         */
        ShaderType Type;
        /**
         * Name of shader
         */
        std::string Name;
        /**
         * Source of shader
         */
        std::string Source;
    };
} // namespace ZEngine::Rendering::Shaders
