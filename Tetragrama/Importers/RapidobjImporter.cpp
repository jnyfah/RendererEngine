#include <pch.h>
#include <AssimpImporter.h>
#include <Core/Coroutine.h>
#include <Helpers/ThreadPool.h>
#include <RapidobjImporter.h>
#include <fmt/format.h>

namespace Tetragrama::Importers
{
    RapidobjImporter::RapidobjImporter() {}

    RapidobjImporter::~RapidobjImporter() {}

    std::future<void> RapidobjImporter::ImportAsync(std::string_view filename, ImportConfiguration config)
    {
        std::filesystem::path obj_path = filename.data();
        m_base_dir                     = obj_path.parent_path(); // Save the base directory

        ThreadPoolHelper::Submit([this, path = std::string(filename.data()), config] {
            {
                std::unique_lock l(m_mutex);
                m_is_importing = true;
            }

            rapidobj::Result result = rapidobj::ParseFile(path);
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
                ZENGINE_CORE_INFO("Serializing model...")
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
        if (!rapidobj::Triangulate(result))
        {
            ZENGINE_CORE_INFO("Cannot triangulate model...")
        }
        // Step 1: Generate normals if they don't exist
        if (result.attributes.normals.empty())
        {
            ZENGINE_CORE_INFO("Generating normals...");
            rapidobj::Array<float> normals(result.attributes.positions.size());

            // Initialize normals to zero
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

                    glm::vec3 edge1  = v1 - v0;
                    glm::vec3 edge2  = v2 - v0;
                    glm::vec3 normal = glm::cross(edge1, edge2);

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
                    n              = glm::normalize(n);
                    normals[i]     = n.x;
                    normals[i + 1] = n.y;
                    normals[i + 2] = n.z;
                }
                else
                {
                    // Default to "up" vector
                    normals[i]     = 0.0f;
                    normals[i + 1] = 1.0f;
                    normals[i + 2] = 0.0f;
                }
            }

            result.attributes.normals = std::move(normals);
        }

        // Step 2: Generate texture coordinates if they don't exist
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

            // Ensure the UV array is the correct size
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
    }

    void RapidobjImporter::ExtractMeshes(const rapidobj::Result& result, ImporterData& importer_data)
    {
        const auto& attributes = result.attributes;
        const auto& shapes     = result.shapes;

        if (shapes.empty() || attributes.positions.empty())
        {
            ZENGINE_CORE_INFO("No shapes or vertices found in the OBJ file.");
            return;
        }

        importer_data.Scene.Meshes.reserve(shapes.size());

        for (size_t shape_idx = 0; shape_idx < shapes.size(); ++shape_idx)
        {
            const auto& shape = shapes[shape_idx];
            ZENGINE_CORE_INFO(fmt::format("Processing shape {}/{}: {}", shape_idx + 1, shapes.size(), !shape.name.empty() ? shape.name : "<unnamed>").c_str());

            // Track mesh statistics
            uint32_t vertex_count{0};
            uint32_t index_count{0};

            // Process each face
            for (size_t face = 0; face < shape.mesh.indices.size(); face += 3)
            {
                for (size_t v = 0; v < 3; ++v)
                {
                    const auto& index = shape.mesh.indices[face + v];

                    importer_data.Scene.Vertices.push_back(attributes.positions[index.position_index * 3]);
                    importer_data.Scene.Vertices.push_back(attributes.positions[index.position_index * 3 + 1]);
                    importer_data.Scene.Vertices.push_back(attributes.positions[index.position_index * 3 + 2]);
                    importer_data.Scene.Vertices.push_back(attributes.normals[index.normal_index * 3]);
                    importer_data.Scene.Vertices.push_back(attributes.normals[index.normal_index * 3 + 1]);
                    importer_data.Scene.Vertices.push_back(attributes.normals[index.normal_index * 3 + 2]);
                    importer_data.Scene.Vertices.push_back(attributes.texcoords[index.texcoord_index * 2]);
                    importer_data.Scene.Vertices.push_back(attributes.texcoords[index.texcoord_index * 2 + 1]);

                    // Add index
                    importer_data.Scene.Indices.push_back(importer_data.VertexOffset + vertex_count);
                    vertex_count++;
                    index_count++;
                }
            }

            // Create mesh entry
            MeshVNext& mesh           = importer_data.Scene.Meshes.emplace_back();
            mesh.VertexCount          = vertex_count;
            mesh.VertexOffset         = importer_data.VertexOffset;
            mesh.VertexUnitStreamSize = sizeof(float) * (3 + 3 + 2); // pos(3) + normal(3) + uv(2)
            mesh.StreamOffset         = mesh.VertexUnitStreamSize * mesh.VertexOffset;
            mesh.IndexOffset          = importer_data.IndexOffset;
            mesh.IndexCount           = index_count;
            mesh.IndexUnitStreamSize  = sizeof(uint32_t);
            mesh.IndexStreamOffset    = mesh.IndexUnitStreamSize * mesh.IndexOffset;
            mesh.TotalByteSize        = (mesh.VertexCount * mesh.VertexUnitStreamSize) + (mesh.IndexCount * mesh.IndexUnitStreamSize);

            // Update offsets for next mesh
            importer_data.VertexOffset += vertex_count;
            importer_data.IndexOffset += index_count;
        }
    }

    void RapidobjImporter::ExtractMaterials(const rapidobj::Result& result, ImporterData& importer_data)
    {
        static constexpr float OPAQUENESS_THRESHOLD = 0.05f;
        const uint32_t         number_of_materials  = static_cast<uint32_t>(result.materials.size());

        ZENGINE_CORE_INFO(fmt::format("Processing {} materials", number_of_materials).c_str());

        // Pre-allocate space
        importer_data.Scene.Materials.reserve(number_of_materials);
        importer_data.Scene.MaterialNames.reserve(number_of_materials);

        // Helper function to convert Float3 to vec4
        auto toVec4 = [](const rapidobj::Float3& rgb, float alpha = 1.0f) -> ZEngine::Rendering::gpuvec4 {
            return ZEngine::Rendering::gpuvec4{std::clamp(rgb[0], 0.0f, 1.0f), std::clamp(rgb[1], 0.0f, 1.0f), std::clamp(rgb[2], 0.0f, 1.0f), std::clamp(alpha, 0.0f, 1.0f)};
        };

        // Process each material
        for (uint32_t mat_idx = 0; mat_idx < number_of_materials; ++mat_idx)
        {
            const auto&   obj_material = result.materials[mat_idx];
            MeshMaterial& material     = importer_data.Scene.Materials.emplace_back();

            // Store material name
            std::string material_name = obj_material.name.empty() ? fmt::format("<unnamed material {}>", mat_idx) : obj_material.name;
            importer_data.Scene.MaterialNames.push_back(std::move(material_name));

            // Convert material properties
            material.AmbientColor  = toVec4(obj_material.ambient);
            material.AlbedoColor   = toVec4(obj_material.diffuse);
            material.SpecularColor = toVec4(obj_material.specular);
            material.EmissiveColor = toVec4(obj_material.emission);

            // Handle transparency/opacity
            float opacity      = obj_material.dissolve;
            material.Factors.x = (opacity < 1.0f) ? std::clamp(1.0f - opacity, 0.0f, 1.0f) : 0.0f;
            if (material.Factors.x >= (1.0f - OPAQUENESS_THRESHOLD))
            {
                material.Factors.x = 0.0f;
            }

            // Handle transmission filter
            float transmission = std::max({obj_material.transmittance[0], obj_material.transmittance[1], obj_material.transmittance[2]});
            if (transmission > 0.0f)
            {
                material.Factors.x = std::clamp(transmission, 0.0f, 1.0f);
                material.Factors.z = 0.5f;
            }

            // Handle shininess and illumination model
            material.Factors.y = std::clamp(obj_material.shininess / 1000.0f, 0.0f, 1.0f);
            material.Factors.w = static_cast<float>(obj_material.illum) / 10.0f;

            // Initialize texture indices
            material.AlbedoMap   = 0xFFFFFFFF;
            material.NormalMap   = 0xFFFFFFFF;
            material.SpecularMap = 0xFFFFFFFF;
            material.EmissiveMap = 0xFFFFFFFF;
            material.OpacityMap  = 0xFFFFFFFF;
        }

        // Map materials to shapes
        for (size_t shape_idx = 0; shape_idx < result.shapes.size(); ++shape_idx)
        {
            const auto& shape = result.shapes[shape_idx];

            if (!shape.mesh.material_ids.empty())
            {
                std::vector<size_t> material_counts(number_of_materials, 0);
                for (int32_t mat_id : shape.mesh.material_ids)
                {
                    if (mat_id >= 0 && static_cast<size_t>(mat_id) < material_counts.size())
                    {
                        material_counts[mat_id]++;
                    }
                }

                auto most_frequent = std::max_element(material_counts.begin(), material_counts.end());
                if (most_frequent != material_counts.end())
                {
                    size_t material_id = std::distance(material_counts.begin(), most_frequent);
                    for (const auto& [node_id, mesh_idx] : importer_data.Scene.NodeMeshes)
                    {
                        if (mesh_idx == shape_idx)
                        {
                            importer_data.Scene.NodeMaterials[node_id] = static_cast<uint32_t>(material_id);
                        }
                    }
                }
            }
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

    void RapidobjImporter::ExtractTextures(const rapidobj::Result& result, ImporterData& importer_data)
    {
        if (result.materials.empty())
        {
            ZENGINE_CORE_INFO("No materials found for texture extraction");
            return;
        }

        const uint32_t number_of_materials = static_cast<uint32_t>(result.materials.size());
        ZENGINE_CORE_INFO(fmt::format("Processing textures for {} materials", number_of_materials).c_str());

        // Helper function to process texture paths and generate file index
        auto processTexturePath = [this, &importer_data](const std::string& texname, const std::string& type) -> uint32_t {
            if (texname.empty())
            {
                ZENGINE_CORE_INFO(fmt::format("No {} texture specified", type));
                return 0xFFFFFFFF; // Invalid texture index
            }

            // Normalize and resolve the path relative to the base directory
            std::filesystem::path texture_path = texname;
            if (!texture_path.is_absolute())
            {
                texture_path = m_base_dir / texture_path;
            }

            if (!std::filesystem::exists(texture_path))
            {
                ZENGINE_CORE_INFO(fmt::format("Warning: {} texture not found at path: {}", type, texture_path.string()));
                return 0xFFFFFFFF; // Invalid texture index
            }

            ZENGINE_CORE_INFO(fmt::format("Found {} texture: {}", type, texture_path.string()));
            return GenerateFileIndex(importer_data.Scene.Files, texname);
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

            // Diffuse/Albedo texture
            ZENGINE_CORE_INFO(fmt::format("Raw texname value: {}", obj_material.diffuse_texname));
            material.AlbedoMap = processTexturePath(obj_material.diffuse_texname, "albedo");

            // Specular texture
            ZENGINE_CORE_INFO(fmt::format("Raw texname value: {}", obj_material.specular_texname));
            material.SpecularMap = processTexturePath(obj_material.specular_texname, "specular");

            // Emissive texture
            material.EmissiveMap = processTexturePath(obj_material.emissive_texname, "emissive");

            // Normal/Bump map
            material.NormalMap = !obj_material.bump_texname.empty()           ? processTexturePath(obj_material.bump_texname, "bump")
                                 : !obj_material.normal_texname.empty()       ? processTexturePath(obj_material.normal_texname, "normal")
                                 : !obj_material.displacement_texname.empty() ? processTexturePath(obj_material.displacement_texname, "displacement as normal")
                                                                              : 0xFFFFFFFF;

            // Opacity/Alpha texture
            material.OpacityMap = processTexturePath(obj_material.alpha_texname, "opacity");

            // Process additional textures (PBR, unsupported types)
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
    bool RapidobjImporter::ValidateFileExists(const std::string& filepath)
    {
        std::ifstream file(filepath);
        return file.good();
    }

    void RapidobjImporter::CreateHierarchyScene(const rapidobj::Result& result, ImporterData& importer_data)
    {
        if (result.shapes.empty())
        {
            ZENGINE_CORE_INFO("No shapes found in the OBJ file for hierarchy creation.");
            return;
        }

        // Create root node
        auto root_node_id                           = SceneRawData::AddNode(&importer_data.Scene, -1, 0);
        importer_data.Scene.NodeNames[root_node_id] = importer_data.Scene.Names.size();
        importer_data.Scene.Names.push_back("Root");
        importer_data.Scene.GlobalTransformCollection[root_node_id] = glm::mat4(1.0f);
        importer_data.Scene.LocalTransformCollection[root_node_id]  = glm::mat4(1.0f);

        // Group shapes by material and calculate shape bounding boxes
        struct MaterialGroup
        {
            std::string         name;
            std::vector<size_t> shape_indices;
            int32_t             material_id{-1};
            glm::vec3           min_bounds{std::numeric_limits<float>::max()};
            glm::vec3           max_bounds{std::numeric_limits<float>::lowest()};
        };

        std::map<int32_t, MaterialGroup> material_groups;

        for (size_t shape_idx = 0; shape_idx < result.shapes.size(); ++shape_idx)
        {
            const auto& shape               = result.shapes[shape_idx];
            int32_t     primary_material_id = -1;

            // Determine primary material
            if (!shape.mesh.material_ids.empty())
            {
                std::map<int32_t, size_t> material_counts;
                for (int32_t mat_id : shape.mesh.material_ids)
                {
                    material_counts[mat_id]++;
                }

                auto most_frequent = std::max_element(material_counts.begin(), material_counts.end(), [](const auto& p1, const auto& p2) {
                    return p1.second < p2.second;
                });

                if (most_frequent != material_counts.end())
                {
                    primary_material_id = most_frequent->first;
                }
            }

            // Assign to material group
            auto& group       = material_groups[primary_material_id];
            group.material_id = primary_material_id;
            group.shape_indices.push_back(shape_idx);

            if (group.name.empty())
            {
                group.name = (primary_material_id >= 0 && primary_material_id < static_cast<int32_t>(result.materials.size())) ? result.materials[primary_material_id].name
                                                                                                                               : "<default material>";
            }

            // Calculate shape bounds
            for (const auto& index : shape.mesh.indices)
            {
                if (index.position_index >= 0 && index.position_index * 3 + 2 < result.attributes.positions.size())
                {
                    glm::vec3 position(
                        result.attributes.positions[index.position_index * 3],
                        result.attributes.positions[index.position_index * 3 + 1],
                        result.attributes.positions[index.position_index * 3 + 2]);
                    group.min_bounds = glm::min(group.min_bounds, position);
                    group.max_bounds = glm::max(group.max_bounds, position);
                }
            }
        }

        // Create material group nodes
        for (const auto& [material_id, group] : material_groups)
        {
            auto material_node_id                           = SceneRawData::AddNode(&importer_data.Scene, root_node_id, 1);
            importer_data.Scene.NodeNames[material_node_id] = importer_data.Scene.Names.size();
            importer_data.Scene.Names.push_back(fmt::format("Material_Group_{}", group.name));

            glm::vec3 group_center = (group.max_bounds + group.min_bounds) * 0.5f;
            glm::vec3 group_scale  = group.max_bounds - group.min_bounds;

            // Construct translation matrix
            glm::mat4 translation_matrix = glm::mat4(1.0f);
            translation_matrix[3]        = glm::vec4(group_center, 1.0f); // Set translation vector

            // Construct scaling matrix
            glm::mat4 scaling_matrix = glm::mat4(1.0f);
            scaling_matrix[0][0]     = group_scale.x; // Scale X
            scaling_matrix[1][1]     = group_scale.y; // Scale Y
            scaling_matrix[2][2]     = group_scale.z; // Scale Z

            // Combine scaling and translation into a single transform
            glm::mat4 group_transform = scaling_matrix * translation_matrix;

            // Assign transforms
            importer_data.Scene.GlobalTransformCollection[material_node_id] = group_transform;
            importer_data.Scene.LocalTransformCollection[material_node_id]  = group_transform;

            // Create shape nodes
            for (size_t shape_idx : group.shape_indices)
            {
                const auto& shape         = result.shapes[shape_idx];
                auto        shape_node_id = SceneRawData::AddNode(&importer_data.Scene, material_node_id, 2);

                importer_data.Scene.NodeNames[shape_node_id] = importer_data.Scene.Names.size();
                importer_data.Scene.Names.push_back(!shape.name.empty() ? shape.name : fmt::format("Shape_{}", shape_idx));

                importer_data.Scene.NodeMeshes[shape_node_id]    = static_cast<uint32_t>(shape_idx);
                importer_data.Scene.NodeMaterials[shape_node_id] = static_cast<uint32_t>(material_id >= 0 ? material_id : 0);
            }
        }

        ZENGINE_CORE_INFO(fmt::format("Created scene hierarchy with {} material groups and {} shapes.", material_groups.size(), result.shapes.size()).c_str());
        // ValidateSceneHierarchy(importer_data.Scene);
    }

} // namespace Tetragrama::Importers