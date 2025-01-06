#pragma once
#include <Rendering/Buffers/CommandBuffer.h>
#include <Rendering/Renderers/GraphicRenderer.h>

namespace ZEngine::Core
{

    struct IRenderable
    {
        IRenderable()                                                                                                                                           = default;
        virtual ~IRenderable()                                                                                                                                  = default;

        virtual void Render(Rendering::Renderers::GraphicRenderer* const renderer = nullptr, Rendering::Buffers::CommandBuffer* const command_buffer = nullptr) = 0;
    };
} // namespace ZEngine::Core
