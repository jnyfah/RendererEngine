#pragma once

#include <IAssetImporter.h>
#include <rapidobj/include/rapidobj/rapidobj.hpp>
#include <filesystem> // Add this for path handling

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

        // To aid in Vertex deduplication
        struct VertexData
        {
            float px, py, pz; // Position
            float nx, ny, nz; // Normal
            float u, v;       // Texture 

            bool operator==(const VertexData& other) const noexcept
            {
                return px == other.px && py == other.py && pz == other.pz && nx == other.nx && ny == other.ny && nz == other.nz && u == other.u && v == other.v;
            }
        };

    private:
        // Core processing functions
        void PreprocessMesh(rapidobj::Result& result);
        void ExtractMeshes(const rapidobj::Result& result, ImporterData& importer_data);
        void ExtractMaterials(const rapidobj::Result& result, ImporterData& importer_data);
        void ExtractTextures(const rapidobj::Result& result, ImporterData& importer_data);
        void CreateHierarchyScene(const rapidobj::Result& result, ImporterData& importer_data);

        // Helper functions
        void TraverseNode(const rapidobj::Result& result, SceneRawData* const scene, const std::string& nodeName, int parent_node_id, int depth_level);
        int  GenerateFileIndex(std::vector<std::string>& data, std::string_view filename);
        void ValidateTextureAssignments(MeshMaterial& material);
        void ProcessAdditionalTextures(const rapidobj::Material& obj_material, MeshMaterial& material);
    };
} // namespace Tetragrama::Importers


namespace std
{
    template <>
    struct hash<Tetragrama::Importers::RapidobjImporter::VertexData>
    {
        size_t operator()(const Tetragrama::Importers::RapidobjImporter::VertexData& v) const noexcept
        {
            // Better hash combining function
            size_t seed         = 0;
            auto   hash_combine = [&seed](size_t hash) {
                seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            };

            hash_combine(std::hash<float>{}(v.px));
            hash_combine(std::hash<float>{}(v.py));
            hash_combine(std::hash<float>{}(v.pz));
            hash_combine(std::hash<float>{}(v.nx));
            hash_combine(std::hash<float>{}(v.ny));
            hash_combine(std::hash<float>{}(v.nz));
            hash_combine(std::hash<float>{}(v.u));
            hash_combine(std::hash<float>{}(v.v));

            return seed;
        }
    };
} // namespace std


// Todo
// 1. replace zengine core with log!
// remove unnecassy comments
// if materials is empty ??
// does texture need full file path ??