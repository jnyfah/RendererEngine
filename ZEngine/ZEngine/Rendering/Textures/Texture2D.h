#pragma once
#include <Rendering/Buffers/Image2DBuffer.h>
#include <Rendering/Textures/Texture.h>
#include <future>

namespace ZEngine::Rendering::Textures
{

    class Texture2D : public Texture
    {
    public:
        Texture2D() = default;
        virtual ~Texture2D();

        static Helpers::Ref<Texture2D>              Create(const Specifications::TextureSpecification& spec);
        static Helpers::Ref<Texture2D>              Create(uint32_t width = 1, uint32_t height = 1);
        static Helpers::Ref<Texture2D>              Create(uint32_t width, uint32_t height, float r, float g, float b, float a);
        static Helpers::Ref<Texture2D>              Read(std::string_view filename);
        static Helpers::Ref<Texture2D>              ReadCubemap(std::string_view filename);
        static std::future<Helpers::Ref<Texture2D>> ReadAsync(std::string_view filename);

        virtual Hardwares::BufferImage&       GetBuffer() override;
        virtual const Hardwares::BufferImage& GetBuffer() const override;
        Helpers::Ref<Buffers::Image2DBuffer>  GetImage2DBuffer() const;
        virtual void                          Dispose() override;

    protected:
        static void FillAsVulkanImage(Helpers::Ref<Texture2D>& texture, const Specifications::TextureSpecification& specification);

    private:
        Helpers::Ref<Buffers::Image2DBuffer> m_image_2d_buffer;
    };
} // namespace ZEngine::Rendering::Textures
