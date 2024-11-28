#include <pch.h>
#include <AssimpImporter.h>
#include <Core/Coroutine.h>
#include <Helpers/ThreadPool.h>
#include <RapidobjImporter.h>
#include <fmt/format.h>

namespace Tetragrama::Importers
{
    RapidobjImporter::RapidobjImporter() = default;

    RapidobjImporter::~RapidobjImporter() = default;

    std::future<void> RapidobjImporter::ImportAsync(std::string_view filename, ImportConfiguration config)
    {
        ThreadPoolHelper::Submit([this, path = std::string(filename.data()), config] {
            {
                std::unique_lock l(m_mutex);
                m_is_importing = true;
            }
            auto result = rapidobj::ParseFile(path);

            if (result.error)
            {
                if (m_error_callback)
                {
                    m_error_callback(result.error.line);
                }
            }
            else
            {
                ImporterData import_data = {};
                PreprocessMesh(result);
                ExtractMeshes(result, import_data);
                ExtractMaterials(result, import_data);
                ExtractTextures(result, import_data);
                CreateHierarchyScene(result, import_data);
                /*
                 * Serialization of ImporterData
                 */
                REPORT_LOG("Serializing model...")
                SerializeImporterData(import_data, config);

                if (m_complete_callback)
                {
                    m_complete_callback(std::move(import_data));
                }
            }
            {
                std::unique_lock l(m_mutex);
                m_is_importing = false;
            }
        });
        co_return;
    }

    void RapidobjImporter::PreprocessMesh(rapidobj::Result& result)
    {
        // Generate normals if they don't exist
        if (result.attributes.normals.empty())
        {
            rapidobj::Array<float> normals(result.attributes.positions.size());
            for (size_t i = 0; i < normals.size(); ++i)
            {
                normals[i] = 0.0f;
            }

            for (const auto& shape : result.shapes)
            {
                for (size_t f = 0; f < shape.mesh.indices.size(); f += 3)
                {
                    const auto& idx0 = shape.mesh.indices[f];
                    const auto& idx1 = shape.mesh.indices[f + 1];
                    const auto& idx2 = shape.mesh.indices[f + 2];

                    glm::vec3 v0(
                        result.attributes.positions[idx0.position_index * 3],
                        result.attributes.positions[idx0.position_index * 3 + 1],
                        result.attributes.positions[idx0.position_index * 3 + 2]);
                    glm::vec3 v1(
                        result.attributes.positions[idx1.position_index * 3],
                        result.attributes.positions[idx1.position_index * 3 + 1],
                        result.attributes.positions[idx1.position_index * 3 + 2]);
                    glm::vec3 v2(
                        result.attributes.positions[idx2.position_index * 3],
                        result.attributes.positions[idx2.position_index * 3 + 1],
                        result.attributes.positions[idx2.position_index * 3 + 2]);

                    glm::vec3 normal = glm::cross(v1 - v0, v2 - v0);

                    if (glm::dot(normal, normal) > 0.0f)
                    {
                        normal = glm::normalize(normal);
                        for (int i = 0; i < 3; i++)
                        {
                            size_t idx = shape.mesh.indices[f + i].position_index * 3;
                            normals[idx] += normal.x;
                            normals[idx + 1] += normal.y;
                            normals[idx + 2] += normal.z;
                        }
                    }
                }
            }

            for (size_t i = 0; i < normals.size(); i += 3)
            {
                glm::vec3 n(normals[i], normals[i + 1], normals[i + 2]);
                if (glm::dot(n, n) > 0.0f)
                {
                    n = glm::normalize(n);
                }
                else
                {
                    n = glm::vec3(0.0f, 1.0f, 0.0f);
                }
                normals[i]     = n.x;
                normals[i + 1] = n.y;
                normals[i + 2] = n.z;
            }

            result.attributes.normals = std::move(normals);
        }

        // Generate texture coordinates if they don't exist
        if (result.attributes.texcoords.empty())
        {
            REPORT_LOG("Generating default UV coordinates...");

            glm::vec3 min_pos(std::numeric_limits<float>::max());
            glm::vec3 max_pos(std::numeric_limits<float>::lowest());

            for (size_t i = 0; i < result.attributes.positions.size(); i += 3)
            {
                glm::vec3 pos(result.attributes.positions[i], result.attributes.positions[i + 1], result.attributes.positions[i + 2]);
                min_pos = glm::min(min_pos, pos);
                max_pos = glm::max(max_pos, pos);
            }

            glm::vec3 size = max_pos - min_pos;

            rapidobj::Array<float> texcoords(result.attributes.positions.size() / 3 * 2);

            for (size_t i = 0; i < result.attributes.positions.size(); i += 3)
            {
                float u = (result.attributes.positions[i] - min_pos.x) / size.x;
                float v = (result.attributes.positions[i + 2] - min_pos.z) / size.z;

                texcoords[(i / 3) * 2]     = u;
                texcoords[(i / 3) * 2 + 1] = v;
            }

            result.attributes.texcoords = std::move(texcoords);
        }
        // Triangulate
        if (!rapidobj::Triangulate(result))
        {
            REPORT_LOG("Cannot triangulate model...")
        }

        // vertex deduplication
        for (auto& shape : result.shapes)
        {
            std::unordered_map<VertexData, uint32_t> unique_vertices;
            std::vector<rapidobj::Index>             new_indices;
            new_indices.reserve(shape.mesh.indices.size());

            const size_t num_indices = shape.mesh.indices.size();
            for (size_t i = 0; i < num_indices; ++i)
            {
                const auto& index = shape.mesh.indices[i];
                VertexData  vertex{};

                // Position
                vertex.px = result.attributes.positions[index.position_index * 3];
                vertex.py = result.attributes.positions[index.position_index * 3 + 1];
                vertex.pz = result.attributes.positions[index.position_index * 3 + 2];

                // Normal
                vertex.nx = result.attributes.normals[index.normal_index * 3];
                vertex.ny = result.attributes.normals[index.normal_index * 3 + 1];
                vertex.nz = result.attributes.normals[index.normal_index * 3 + 2];

                // UV
                vertex.u = result.attributes.texcoords[index.texcoord_index * 2];
                vertex.v = result.attributes.texcoords[index.texcoord_index * 2 + 1];

                auto it = unique_vertices.find(vertex);
                if (it != unique_vertices.end())
                {
                    rapidobj::Index new_index;
                    new_index.position_index = it->second;
                    new_index.normal_index   = it->second;
                    new_index.texcoord_index = it->second;
                    new_indices.push_back(new_index);
                }
                else
                {
                    uint32_t new_index_value = static_cast<uint32_t>(unique_vertices.size());
                    unique_vertices[vertex]  = new_index_value;

                    rapidobj::Index new_index;
                    new_index.position_index = new_index_value;
                    new_index.normal_index   = new_index_value;
                    new_index.texcoord_index = new_index_value;
                    new_indices.push_back(new_index);
                }
            }

            rapidobj::Array<float> new_positions(unique_vertices.size() * 3);
            rapidobj::Array<float> new_normals(unique_vertices.size() * 3);
            rapidobj::Array<float> new_texcoords(unique_vertices.size() * 2);

            size_t vert_idx = 0;
            for (const auto& [vert, idx] : unique_vertices)
            {
                new_positions[idx * 3]     = vert.px;
                new_positions[idx * 3 + 1] = vert.py;
                new_positions[idx * 3 + 2] = vert.pz;

                new_normals[idx * 3]     = vert.nx;
                new_normals[idx * 3 + 1] = vert.ny;
                new_normals[idx * 3 + 2] = vert.nz;

                new_texcoords[idx * 2]     = vert.u;
                new_texcoords[idx * 2 + 1] = vert.v;
            }

            rapidobj::Array<rapidobj::Index> final_indices(new_indices.size());
            memcpy(final_indices.data(), new_indices.data(), new_indices.size() * sizeof(rapidobj::Index));

            // Update shape and attributes
            result.attributes.positions = std::move(new_positions);
            result.attributes.normals   = std::move(new_normals);
            result.attributes.texcoords = std::move(new_texcoords);
            shape.mesh.indices          = std::move(final_indices);
        }
        REPORT_LOG("Preprocessing complete.");
    }

    void RapidobjImporter::ExtractMeshes(const rapidobj::Result& result, ImporterData& importer_data)
    {
        const auto& attributes = result.attributes;
        const auto& shapes     = result.shapes;

        if (shapes.empty() || attributes.positions.empty())
        {
            REPORT_LOG("No shapes or vertices found in the OBJ file.");
            return;
        }

        importer_data.Scene.Meshes.reserve(shapes.size());

        for (size_t shape_idx = 0; shape_idx < shapes.size(); ++shape_idx)
        {
            const auto&  shape      = shapes[shape_idx];
            const size_t face_count = shape.mesh.indices.size() / 3;

            for (const auto& index : shape.mesh.indices)
            {
                importer_data.Scene.Vertices.push_back(attributes.positions[index.position_index * 3]);
                importer_data.Scene.Vertices.push_back(attributes.positions[index.position_index * 3 + 1]);
                importer_data.Scene.Vertices.push_back(attributes.positions[index.position_index * 3 + 2]);

                importer_data.Scene.Vertices.push_back(attributes.normals[index.normal_index * 3]);
                importer_data.Scene.Vertices.push_back(attributes.normals[index.normal_index * 3 + 1]);
                importer_data.Scene.Vertices.push_back(attributes.normals[index.normal_index * 3 + 2]);

                importer_data.Scene.Vertices.push_back(attributes.texcoords[index.texcoord_index * 2]);
                importer_data.Scene.Vertices.push_back(attributes.texcoords[index.texcoord_index * 2 + 1]);

                importer_data.Scene.Indices.push_back(importer_data.VertexOffset++);
            }

            MeshVNext& mesh           = importer_data.Scene.Meshes.emplace_back();
            mesh.VertexCount          = static_cast<uint32_t>(shape.mesh.indices.size());
            mesh.VertexOffset         = importer_data.VertexOffset - mesh.VertexCount;
            mesh.VertexUnitStreamSize = sizeof(float) * (3 + 3 + 2);
            mesh.StreamOffset         = mesh.VertexUnitStreamSize * mesh.VertexOffset;
            mesh.IndexOffset          = importer_data.IndexOffset;
            mesh.IndexCount           = static_cast<uint32_t>(shape.mesh.indices.size());
            mesh.IndexUnitStreamSize  = sizeof(uint32_t);
            mesh.IndexStreamOffset    = mesh.IndexUnitStreamSize * mesh.IndexOffset;
            mesh.TotalByteSize        = (mesh.VertexCount * mesh.VertexUnitStreamSize) + (mesh.IndexCount * mesh.IndexUnitStreamSize);

            importer_data.IndexOffset += mesh.IndexCount;
        }
    }

    void RapidobjImporter::ExtractMaterials(const rapidobj::Result& result, ImporterData& importer_data)
    {
        static constexpr float OPAQUENESS_THRESHOLD = 0.05f;

        // Handle case where no materials exist
        if (result.materials.empty())
        {
            MeshMaterial& default_material = importer_data.Scene.Materials.emplace_back();
            default_material.AlbedoColor   = ZEngine::Rendering::gpuvec4{0.7f, 0.7f, 0.7f, 1.0f};
            default_material.AmbientColor  = ZEngine::Rendering::gpuvec4{0.2f, 0.2f, 0.2f, 1.0f};
            default_material.SpecularColor = ZEngine::Rendering::gpuvec4{0.5f, 0.5f, 0.5f, 1.0f};
            default_material.EmissiveColor = ZEngine::Rendering::gpuvec4{0.0f, 0.0f, 0.0f, 1.0f};
            default_material.Factors       = ZEngine::Rendering::gpuvec4{0.0f, 0.0f, 0.0f, 0.0f};
            importer_data.Scene.MaterialNames.push_back("<default material>");
            return;
        }

        const uint32_t number_of_materials = static_cast<uint32_t>(result.materials.size());
        importer_data.Scene.Materials.reserve(number_of_materials);
        importer_data.Scene.MaterialNames.reserve(number_of_materials);

        auto toVec4 = [](const rapidobj::Float3& rgb, float alpha = 1.0f) -> ZEngine::Rendering::gpuvec4 {
            return ZEngine::Rendering::gpuvec4{std::clamp(rgb[0], 0.0f, 1.0f), std::clamp(rgb[1], 0.0f, 1.0f), std::clamp(rgb[2], 0.0f, 1.0f), std::clamp(alpha, 0.0f, 1.0f)};
        };

        for (uint32_t mat_idx = 0; mat_idx < number_of_materials; ++mat_idx)
        {
            const auto&   obj_material = result.materials[mat_idx];
            MeshMaterial& material     = importer_data.Scene.Materials.emplace_back();

            importer_data.Scene.MaterialNames.push_back(obj_material.name.empty() ? fmt::format("<unnamed material {}>", mat_idx) : obj_material.name);

            material.AmbientColor  = toVec4(obj_material.ambient);
            material.AlbedoColor   = toVec4(obj_material.diffuse);
            material.SpecularColor = toVec4(obj_material.specular);
            material.EmissiveColor = toVec4(obj_material.emission);

            float opacity      = obj_material.dissolve;
            material.Factors.x = (opacity < 1.0f) ? std::clamp(1.0f - opacity, 0.0f, 1.0f) : 0.0f;

            float transmission = std::max({obj_material.transmittance[0], obj_material.transmittance[1], obj_material.transmittance[2]});
            if (transmission > 0.0f)
            {
                material.Factors.x = std::clamp(transmission, 0.0f, 1.0f);
                material.Factors.z = 0.5f;
            }

            material.Factors.y = std::clamp(obj_material.shininess / 1000.0f, 0.0f, 1.0f);
            material.Factors.w = static_cast<float>(obj_material.illum) / 10.0f;

            // Initialize texture indices
            material.AlbedoMap   = 0xFFFFFFFF;
            material.NormalMap   = 0xFFFFFFFF;
            material.SpecularMap = 0xFFFFFFFF;
            material.EmissiveMap = 0xFFFFFFFF;
            material.OpacityMap  = 0xFFFFFFFF;
        }
    }

    void RapidobjImporter::ExtractTextures(const rapidobj::Result& result, ImporterData& importer_data)
    {
        if (result.materials.empty())
        {
            REPORT_LOG("No materials found for texture extraction.");
            return;
        }

        const uint32_t number_of_materials = static_cast<uint32_t>(result.materials.size());
        REPORT_LOG(fmt::format("Processing textures for {} materials", number_of_materials).c_str());

        auto processTexturePath = [this, &importer_data](const std::string& texname, const std::string& type) -> uint32_t {
            if (texname.empty())
            {
                REPORT_LOG(fmt::format("No {} texture specified.", type));
                return 0xFFFFFFFF;
            }

            std::string normalized_path = texname;
            std::replace(normalized_path.begin(), normalized_path.end(), '\\', '/');

            return GenerateFileIndex(importer_data.Scene.Files, normalized_path);
        };

        for (uint32_t mat_idx = 0; mat_idx < number_of_materials; ++mat_idx)
        {
            const auto& obj_material = result.materials[mat_idx];
            auto&       material     = importer_data.Scene.Materials[mat_idx];

            material.AlbedoMap   = processTexturePath(obj_material.diffuse_texname, "albedo");
            material.SpecularMap = processTexturePath(obj_material.specular_texname, "specular");
            material.EmissiveMap = processTexturePath(obj_material.emissive_texname, "emissive");

            material.NormalMap = !obj_material.bump_texname.empty()           ? processTexturePath(obj_material.bump_texname, "bump")
                                 : !obj_material.normal_texname.empty()       ? processTexturePath(obj_material.normal_texname, "normal")
                                 : !obj_material.displacement_texname.empty() ? processTexturePath(obj_material.displacement_texname, "displacement as normal")
                                                                              : 0xFFFFFFFF;

            material.OpacityMap = processTexturePath(obj_material.alpha_texname, "opacity");
        }
    }

    int RapidobjImporter::GenerateFileIndex(std::vector<std::string>& data, std::string_view filename)
    {
        auto find = std::find(std::begin(data), std::end(data), filename);
        if (find != std::end(data))
        {
            return std::distance(std::begin(data), find);
        }

        data.push_back(filename.data());
        return (data.size() - 1);
    }

    void RapidobjImporter::CreateHierarchyScene(const rapidobj::Result& result, ImporterData& importer_data)
    {
        if (result.shapes.empty())
        {
            return;
        }

        // Create root node ??
        auto root_id                           = SceneRawData::AddNode(&importer_data.Scene, -1, 0);
        importer_data.Scene.NodeNames[root_id] = importer_data.Scene.Names.size();
        importer_data.Scene.Names.push_back("Root");
        importer_data.Scene.GlobalTransformCollection[root_id] = glm::mat4(1.0f);
        importer_data.Scene.LocalTransformCollection[root_id]  = glm::mat4(1.0f);

        // Create model node ??
        auto model_node_id                           = SceneRawData::AddNode(&importer_data.Scene, root_id, 1);
        importer_data.Scene.NodeNames[model_node_id] = importer_data.Scene.Names.size();
        importer_data.Scene.Names.push_back("model_Mesh1_Model");
        importer_data.Scene.GlobalTransformCollection[model_node_id] = glm::mat4(1.0f);
        importer_data.Scene.LocalTransformCollection[model_node_id]  = glm::mat4(1.0f);

        // Collect all unique materials across all shapes
        std::map<int32_t, std::vector<size_t>> material_to_shapes;

        for (size_t shape_idx = 0; shape_idx < result.shapes.size(); ++shape_idx)
        {
            const auto&       shape = result.shapes[shape_idx];
            std::set<int32_t> shape_materials;

            for (int32_t mat_id : shape.mesh.material_ids)
            {
                shape_materials.insert(mat_id);
            }

            for (int32_t mat_id : shape_materials)
            {
                material_to_shapes[mat_id].push_back(shape_idx);
            }
        }

        // Create a node for each material
        for (const auto& [material_id, shape_indices] : material_to_shapes)
        {
            auto material_node_id                           = SceneRawData::AddNode(&importer_data.Scene, model_node_id, 2);
            importer_data.Scene.NodeNames[material_node_id] = importer_data.Scene.Names.size();

            std::string material_name = material_id < static_cast<int32_t>(importer_data.Scene.MaterialNames.size()) ? importer_data.Scene.MaterialNames[material_id]
                                                                                                                     : fmt::format("material_{}", material_id);

            importer_data.Scene.Names.push_back(material_name);

            // Create child nodes for each shape using this material
            for (size_t shape_idx : shape_indices)
            {
                auto shape_node_id                           = SceneRawData::AddNode(&importer_data.Scene, material_node_id, 3);
                importer_data.Scene.NodeNames[shape_node_id] = importer_data.Scene.Names.size();
                importer_data.Scene.Names.push_back(fmt::format("{}_{}", material_name, shape_idx));

                importer_data.Scene.NodeMeshes[shape_node_id]    = static_cast<uint32_t>(shape_idx);
                importer_data.Scene.NodeMaterials[shape_node_id] = static_cast<uint32_t>(material_id);

                importer_data.Scene.GlobalTransformCollection[shape_node_id] = glm::mat4(1.0f);
                importer_data.Scene.LocalTransformCollection[shape_node_id]  = glm::mat4(1.0f);
            }
        }
    }
} // namespace Tetragrama::Importers