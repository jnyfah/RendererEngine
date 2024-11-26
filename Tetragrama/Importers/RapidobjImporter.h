#pragma once
#include <IAssetImporter.h>
#include <rapidobj/include/rapidobj/rapidobj.hpp>
#include <filesystem>

using namespace ZEngine::Helpers;
using namespace ZEngine::Rendering::Meshes;
using namespace ZEngine::Rendering::Scenes;

namespace Tetragrama::Importers
{

    class RapidobjImporter : public IAssetImporter
    {
    public:
        RapidobjImporter();
        virtual ~RapidobjImporter();
        std::future<void> ImportAsync(std::string_view filename, ImportConfiguration config = {}) override;

    private:
        void                  PreprocessMesh(rapidobj::Result&);
        void                  ExtractMeshes(const rapidobj::Result&, ImporterData&);
        void                  ExtractMaterials(const rapidobj::Result&, ImporterData&);
        void                  ExtractTextures(const rapidobj::Result&, ImporterData&);
        int                   GenerateFileIndex(std::vector<std::string>& data, std::string_view filename);
        void                  ValidateTextureAssignments(MeshMaterial& material);
        void                  ProcessAdditionalTextures(const rapidobj::Material&, MeshMaterial& material);
        bool                  ValidateFileExists(const std::string& filepath);
        void                  CreateHierarchyScene(const rapidobj::Result& result, ImporterData& importer_data);
        void                  ValidateSceneHierarchy(const SceneRawData& scene);
        std::filesystem::path m_base_dir;
    };
} // namespace Tetragrama::Importers