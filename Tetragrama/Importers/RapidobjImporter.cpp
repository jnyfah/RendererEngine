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
            ZENGINE_CORE_INFO("Generating default UV coordinates...");

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
            const auto& shape = shapes[shape_idx];

            std::unordered_map<VertexData, uint32_t> unique_vertices;
            const size_t                             face_count = shape.mesh.indices.size() / 3;

            for (size_t f = 0; f < face_count; ++f)
            {
                for (size_t v = 0; v < 3; ++v)
                {
                    const auto& index = shape.mesh.indices[f * 3 + v];
                    VertexData  vertex{};

                    const size_t pos_idx = index.position_index * 3;
                    vertex.px            = attributes.positions[pos_idx];
                    vertex.py            = attributes.positions[pos_idx + 1];
                    vertex.pz            = attributes.positions[pos_idx + 2];

                    const size_t norm_idx = index.normal_index * 3;
                    vertex.nx             = attributes.normals[norm_idx];
                    vertex.ny             = attributes.normals[norm_idx + 1];
                    vertex.nz             = attributes.normals[norm_idx + 2];

                    const size_t tex_idx = index.texcoord_index * 2;
                    vertex.u             = attributes.texcoords[tex_idx];
                    vertex.v             = attributes.texcoords[tex_idx + 1];

                    // Add to unique_vertices or reuse
                    auto     it = unique_vertices.find(vertex);
                    uint32_t vertex_index;

                    if (it != unique_vertices.end())
                    {
                        vertex_index = it->second;
                    }
                    else
                    {
                        vertex_index            = static_cast<uint32_t>(unique_vertices.size());
                        unique_vertices[vertex] = vertex_index;

                        importer_data.Scene.Vertices.insert(
                            importer_data.Scene.Vertices.end(), {vertex.px, vertex.py, vertex.pz, vertex.nx, vertex.ny, vertex.nz, vertex.u, vertex.v});
                    }

                    importer_data.Scene.Indices.push_back(importer_data.VertexOffset + vertex_index);
                }
            }

            MeshVNext& mesh           = importer_data.Scene.Meshes.emplace_back();
            mesh.VertexCount          = static_cast<uint32_t>(unique_vertices.size());
            mesh.VertexOffset         = importer_data.VertexOffset;
            mesh.VertexUnitStreamSize = sizeof(float) * (3 + 3 + 2);
            mesh.StreamOffset         = mesh.VertexUnitStreamSize * mesh.VertexOffset;
            mesh.IndexOffset          = importer_data.IndexOffset;
            mesh.IndexCount           = static_cast<uint32_t>(shape.mesh.indices.size());
            mesh.IndexUnitStreamSize  = sizeof(uint32_t);
            mesh.IndexStreamOffset    = mesh.IndexUnitStreamSize * mesh.IndexOffset;
            mesh.TotalByteSize        = (mesh.VertexCount * mesh.VertexUnitStreamSize) + (mesh.IndexCount * mesh.IndexUnitStreamSize);

            importer_data.VertexOffset += mesh.VertexCount;
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

        // Process materials
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

        // Map each material in the shapes to individual nodes
        for (size_t shape_idx = 0; shape_idx < result.shapes.size(); ++shape_idx)
        {
            const auto& shape = result.shapes[shape_idx];

            // Assign all materials used by the shape to individual nodes
            for (size_t material_idx = 0; material_idx < shape.mesh.material_ids.size(); ++material_idx)
            {
                int32_t mat_id = shape.mesh.material_ids[material_idx];
                if (mat_id >= 0 && static_cast<size_t>(mat_id) < result.materials.size())
                {
                    for (const auto& [node_id, mesh_idx] : importer_data.Scene.NodeMeshes)
                    {
                        if (mesh_idx == shape_idx)
                        {
                            importer_data.Scene.NodeMaterials[node_id] = static_cast<uint32_t>(mat_id);
                        }
                    }
                }
            }
        }
    }

    void RapidobjImporter::ExtractTextures(const rapidobj::Result& result, ImporterData& importer_data)
    {
        if (result.materials.empty())
        {
            ZENGINE_CORE_INFO("No materials found for texture extraction.");
            return;
        }

        const uint32_t number_of_materials = static_cast<uint32_t>(result.materials.size());
        ZENGINE_CORE_INFO(fmt::format("Processing textures for {} materials", number_of_materials).c_str());

        // Helper function to process texture paths and generate file indices
        auto processTexturePath = [this, &importer_data](const std::string& texname, const std::string& type) -> uint32_t {
            if (texname.empty())
            {
                ZENGINE_CORE_INFO(fmt::format("No {} texture specified.", type));
                return 0xFFFFFFFF; // Invalid texture index
            }

            ZENGINE_CORE_INFO(fmt::format("Found texname: {}", texname));

            std::string normalized_path = texname;
            std::replace(normalized_path.begin(), normalized_path.end(), '\\', '/');

            ZENGINE_CORE_INFO(fmt::format("Found {} texture: {}", type, normalized_path));
            return GenerateFileIndex(importer_data.Scene.Files, normalized_path);
        };

        for (uint32_t mat_idx = 0; mat_idx < number_of_materials; ++mat_idx)
        {
            const auto& obj_material = result.materials[mat_idx];
            auto&       material     = importer_data.Scene.Materials[mat_idx];

            ZENGINE_CORE_INFO(fmt::format(
                                  "Processing textures for material {}/{}: {}",
                                  mat_idx + 1,
                                  number_of_materials,
                                  obj_material.name.empty() ? fmt::format("<unnamed material {}>", mat_idx) : obj_material.name)
                                  .c_str());

            // Process standard textures
            material.AlbedoMap   = processTexturePath(obj_material.diffuse_texname, "albedo");
            material.SpecularMap = processTexturePath(obj_material.specular_texname, "specular");
            material.EmissiveMap = processTexturePath(obj_material.emissive_texname, "emissive");

            // Handle multiple normal map definitions
            material.NormalMap = !obj_material.bump_texname.empty()           ? processTexturePath(obj_material.bump_texname, "bump")
                                 : !obj_material.normal_texname.empty()       ? processTexturePath(obj_material.normal_texname, "normal")
                                 : !obj_material.displacement_texname.empty() ? processTexturePath(obj_material.displacement_texname, "displacement as normal")
                                                                              : 0xFFFFFFFF;

            // Process opacity texture
            material.OpacityMap = processTexturePath(obj_material.alpha_texname, "opacity");

            // Handle additional textures (e.g., PBR maps)
            ProcessAdditionalTextures(obj_material, material);

            // Validate texture assignments
            ValidateTextureAssignments(material);
        }

        ZENGINE_CORE_INFO(fmt::format("Total unique textures processed: {}", importer_data.Scene.Files.size()).c_str());
    }

    void RapidobjImporter::ProcessAdditionalTextures(const rapidobj::Material& obj_material, MeshMaterial& material)
    {
        // Process ambient occlusion texture if available
        if (!obj_material.ambient_texname.empty())
        {
            ZENGINE_CORE_INFO("Ambient occlusion texture found but not supported in current material format");
        }

        // Check for PBR textures
        bool has_pbr_textures = false;
        if (!obj_material.roughness_texname.empty())
        {
            has_pbr_textures = true;
            ZENGINE_CORE_INFO("Roughness texture found");
        }
        if (!obj_material.metallic_texname.empty())
        {
            has_pbr_textures = true;
            ZENGINE_CORE_INFO("Metallic texture found");
        }
        if (!obj_material.sheen_texname.empty())
        {
            has_pbr_textures = true;
            ZENGINE_CORE_INFO("Sheen texture found");
        }

        if (has_pbr_textures)
        {
            ZENGINE_CORE_INFO("Note: Some PBR textures found but not supported in current material format");
        }
    }

    void RapidobjImporter::ValidateTextureAssignments(MeshMaterial& material)
    {
        // Check for invalid texture combinations
        if (material.OpacityMap != 0xFFFFFFFF)
        {
            // If we have an opacity map, ensure the material is marked as having transparency
            if (material.Factors.x == 0.0f)
            {
                material.Factors.x = 0.5f;
                ZENGINE_CORE_INFO("Adjusted material transparency factor due to opacity map");
            }
        }

        // Check for normal map without proper factors
        if (material.NormalMap != 0xFFFFFFFF && material.Factors.w == 0.0f)
        {
            material.Factors.w = 1.0f; // Default normal map intensity
            ZENGINE_CORE_INFO("Set default normal map intensity factor");
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
            ZENGINE_CORE_INFO("No shapes found in the OBJ file.");
            return;
        }

        TraverseNode(result, &importer_data.Scene, "filename", -1, 0);
    }

    void RapidobjImporter::TraverseNode(const rapidobj::Result& result, SceneRawData* const scene, const std::string& nodeName, int parent_node_id, int depth_level)
    {
        // Create root node
        auto root_id              = SceneRawData::AddNode(scene, parent_node_id, depth_level);
        scene->NodeNames[root_id] = scene->Names.size();
        scene->Names.push_back(!nodeName.empty() ? nodeName : "<unnamed node>");
        scene->GlobalTransformCollection[root_id] = glm::mat4(1.0f);
        scene->LocalTransformCollection[root_id]  = glm::mat4(1.0f);

        if (depth_level == 0)
        {
            // Create mesh group node
            auto mesh_group_id              = SceneRawData::AddNode(scene, root_id, depth_level + 1);
            scene->NodeNames[mesh_group_id] = scene->Names.size();
            scene->Names.push_back(nodeName + "_Mesh1_Model");
            scene->GlobalTransformCollection[mesh_group_id] = glm::mat4(1.0f);
            scene->LocalTransformCollection[mesh_group_id]  = glm::mat4(1.0f);

            // First, organize shapes by their materials
            std::map<int32_t, std::vector<size_t>> material_to_shapes;

            // Group shapes by material
            for (size_t shape_idx = 0; shape_idx < result.shapes.size(); ++shape_idx)
            {
                const auto& shape = result.shapes[shape_idx];

                // Find the dominant material for this shape
                std::map<int32_t, size_t> material_counts;
                for (int32_t mat_id : shape.mesh.material_ids)
                {
                    if (mat_id >= 0 && mat_id < static_cast<int32_t>(result.materials.size()))
                    {
                        material_counts[mat_id]++;
                    }
                }

                int32_t primary_material_id = -1;
                if (!material_counts.empty())
                {
                    auto most_frequent  = std::max_element(material_counts.begin(), material_counts.end(), [](const auto& p1, const auto& p2) {
                        return p1.second < p2.second;
                    });
                    primary_material_id = most_frequent->first;
                }

                // Group shapes by their primary material
                material_to_shapes[primary_material_id].push_back(shape_idx);
                ZENGINE_CORE_INFO(fmt::format("Shape {} assigned to material {}", shape_idx, primary_material_id).c_str());
            }

            // Create nodes for each material group
            for (const auto& [material_id, shape_indices] : material_to_shapes)
            {
                for (size_t shape_idx : shape_indices)
                {
                    const auto& shape             = result.shapes[shape_idx];
                    auto        sub_node_id       = SceneRawData::AddNode(scene, mesh_group_id, depth_level + 2);
                    scene->NodeNames[sub_node_id] = scene->Names.size();

                    // Get material name
                    std::string material_name;
                    if (material_id >= 0 && material_id < static_cast<int32_t>(result.materials.size()))
                    {
                        material_name = !result.materials[material_id].name.empty() ? result.materials[material_id].name : fmt::format("material_{}", material_id);

                        ZENGINE_CORE_INFO(fmt::format("Creating node with material: {} (ID: {})", material_name, material_id).c_str());
                    }
                    else
                    {
                        material_name = "<default material>";
                    }

                    scene->Names.push_back(material_name);
                    scene->NodeMeshes[sub_node_id]                = static_cast<uint32_t>(shape_idx);
                    scene->NodeMaterials[sub_node_id]             = static_cast<uint32_t>(material_id >= 0 ? material_id : 0);
                    scene->GlobalTransformCollection[sub_node_id] = glm::mat4(1.0f);
                    scene->LocalTransformCollection[sub_node_id]  = glm::mat4(1.0f);

                    ZENGINE_CORE_INFO(fmt::format("Created node for shape {} with material {}", shape_idx, material_id).c_str());
                }
            }

            ZENGINE_CORE_INFO(fmt::format("Created hierarchy with {} material groups", material_to_shapes.size()).c_str());
        }
    }
} // namespace Tetragrama::Importers
