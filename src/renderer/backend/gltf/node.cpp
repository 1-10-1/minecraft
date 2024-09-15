#include <mc/renderer/backend/gltf/loader.hpp>
#include <mc/renderer/backend/gltf/node.hpp>
#include <mc/renderer/backend/utils.hpp>

#include <glm/ext/quaternion_float.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace renderer::backend
{
    glm::mat4 Node::localMatrix()
    {
        if (!useCachedMatrix)
        {
            cachedLocalMatrix = glm::translate(glm::mat4(1.0f), translation) *
                                glm::mat4(glm::quat { static_cast<float>(rotation.w),
                                                      static_cast<float>(rotation.x),
                                                      static_cast<float>(rotation.y),
                                                      static_cast<float>(rotation.z) }) *
                                glm::scale(glm::mat4(1.0f), scale) * matrix;
        };

        return cachedLocalMatrix;
    }

    glm::mat4 Node::getMatrix()
    {
        // Use a simple caching algorithm to avoid having to recalculate matrices too
        // often while traversing the node hierarchy
        if (!useCachedMatrix)
        {
            glm::mat4 m = localMatrix();
            Node* p     = parent;

            while (p)
            {
                m = p->localMatrix() * m;
                p = p->parent;
            }

            cachedMatrix    = m;
            useCachedMatrix = true;

            return m;
        }
        else
        {
            return cachedMatrix;
        }
    }

    void Node::update()
    {
        useCachedMatrix = false;

        if (mesh)
        {
            glm::mat4 m               = getMatrix();
            mesh->uniformBlock.matrix = m;

            if (skin)
            {
                // Update joint matrices
                glm::mat4 inverseTransform = glm::inverse(m);
                size_t numJoints           = std::min(utils::size(skin->joints), kMaxNumJoints);

                for (size_t i = 0; i < numJoints; i++)
                {
                    Node* jointNode = skin->joints[i];
                    glm::mat4 jointMat =
                        inverseTransform * jointNode->getMatrix() * skin->inverseBindMatrices[i];
                    mesh->uniformBlock.jointMatrix[i] = jointMat;
                }

                mesh->uniformBlock.jointcount = static_cast<uint32_t>(numJoints);
                std::memcpy(mesh->uniformBuffer.mapped, &mesh->uniformBlock, sizeof(mesh->uniformBlock));
            }
            else
            {
                std::memcpy(mesh->uniformBuffer.mapped, &m, sizeof(glm::mat4));
            }
        }

        for (auto& child : children)
        {
            child->update();
        }
    }

    void Model::loadSkins(tinygltf::Model& gltfModel)
    {
        for (tinygltf::Skin& source : gltfModel.skins)
        {
            Skin* newSkin = new Skin {};
            newSkin->name = source.name;

            // Find skeleton root node
            if (source.skeleton > -1)
            {
                newSkin->skeletonRoot = nodeFromIndex(source.skeleton);
            }

            // Find joint nodes
            for (int jointIndex : source.joints)
            {
                Node* node = nodeFromIndex(jointIndex);

                if (node)
                {
                    newSkin->joints.push_back(nodeFromIndex(jointIndex));
                }
            }

            // Get inverse bind matrices from buffer
            if (source.inverseBindMatrices > -1)
            {
                tinygltf::Accessor const& accessor     = gltfModel.accessors[source.inverseBindMatrices];
                tinygltf::BufferView const& bufferView = gltfModel.bufferViews[accessor.bufferView];
                tinygltf::Buffer const& buffer         = gltfModel.buffers[bufferView.buffer];

                newSkin->inverseBindMatrices.resize(accessor.count);

                std::memcpy(newSkin->inverseBindMatrices.data(),
                            &buffer.data[accessor.byteOffset + bufferView.byteOffset],
                            accessor.count * sizeof(glm::mat4));
            }

            skins.push_back(newSkin);
        }
    }

    void Model::getNodeProps(tinygltf::Node const& node,
                             tinygltf::Model const& model,
                             size_t& vertexCount,
                             size_t& indexCount)
    {
        if (node.children.size() > 0)
        {
            for (size_t i = 0; i < node.children.size(); i++)
            {
                getNodeProps(model.nodes[node.children[i]], model, vertexCount, indexCount);
            }
        }

        if (node.mesh > -1)
        {
            tinygltf::Mesh const mesh = model.meshes[node.mesh];

            for (size_t i = 0; i < mesh.primitives.size(); i++)
            {
                auto primitive = mesh.primitives[i];
                vertexCount += model.accessors[primitive.attributes.find("POSITION")->second].count;

                if (primitive.indices > -1)
                {
                    indexCount += model.accessors[primitive.indices].count;
                }
            }
        }
    }

    Node* Model::findNode(Node* parent, uint32_t index)
    {
        Node* nodeFound = nullptr;

        if (parent->index == index)
        {
            return parent;
        }

        for (auto& child : parent->children)
        {
            nodeFound = findNode(child, index);

            if (nodeFound)
            {
                break;
            }
        }

        return nodeFound;
    }

    Node* Model::nodeFromIndex(uint32_t index)
    {
        Node* nodeFound = nullptr;

        for (auto& node : nodes)
        {
            nodeFound = findNode(node, index);

            if (nodeFound)
            {
                break;
            }
        }
        return nodeFound;
    }

    void Model::loadNode(Node* parent,
                         tinygltf::Node const& node,
                         uint32_t nodeIndex,
                         tinygltf::Model const& model,
                         LoaderInfo& loaderInfo,
                         float globalscale)
    {
        Node* newNode      = new Node {};
        newNode->index     = nodeIndex;
        newNode->parent    = parent;
        newNode->name      = node.name;
        newNode->skinIndex = node.skin;
        newNode->matrix    = glm::mat4(1.0f);

        // Generate local node matrix
        glm::vec3 translation = glm::vec3(0.0f);
        if (node.translation.size() == 3)
        {
            translation          = glm::make_vec3(node.translation.data());
            newNode->translation = translation;
        }
        if (node.rotation.size() == 4)
        {
            newNode->rotation = glm::make_quat(node.rotation.data());
        }
        glm::vec3 scale = glm::vec3(1.0f);
        if (node.scale.size() == 3)
        {
            scale          = glm::make_vec3(node.scale.data());
            newNode->scale = scale;
        }
        if (node.matrix.size() == 16)
        {
            newNode->matrix = glm::make_mat4x4(node.matrix.data());
        }

        // Node with children
        if (node.children.size() > 0)
        {
            for (size_t i = 0; i < node.children.size(); i++)
            {
                loadNode(
                    newNode, model.nodes[node.children[i]], node.children[i], model, loaderInfo, globalscale);
            }
        }

        // Node contains mesh data
        if (node.mesh > -1)
        {
            tinygltf::Mesh const mesh     = model.meshes[node.mesh];
            std::unique_ptr<Mesh> newMesh = std::make_unique<Mesh>(*m_bufferManager, newNode->matrix);

            for (size_t j = 0; j < mesh.primitives.size(); j++)
            {
                tinygltf::Primitive const& primitive = mesh.primitives[j];
                uint32_t vertexStart                 = static_cast<uint32_t>(loaderInfo.vertexPos);
                uint32_t indexStart                  = static_cast<uint32_t>(loaderInfo.indexPos);
                uint32_t indexCount                  = 0;
                uint32_t vertexCount                 = 0;
                glm::vec3 posMin {};
                glm::vec3 posMax {};
                bool hasSkin    = false;
                bool hasIndices = primitive.indices > -1;

                // Vertices
                {
                    float const* bufferPos          = nullptr;
                    float const* bufferTangents     = nullptr;
                    float const* bufferNormals      = nullptr;
                    float const* bufferTexCoordSet0 = nullptr;
                    float const* bufferTexCoordSet1 = nullptr;
                    float const* bufferColorSet0    = nullptr;
                    void const* bufferJoints        = nullptr;
                    float const* bufferWeights      = nullptr;

                    int posByteStride;
                    int tangentByteStride;
                    int normByteStride;
                    int uv0ByteStride;
                    int uv1ByteStride;
                    int color0ByteStride;
                    int jointByteStride;
                    int weightByteStride;

                    int jointComponentType;

                    // Position attribute is required
                    MC_ASSERT(primitive.attributes.find("POSITION") != primitive.attributes.end());

                    tinygltf::Accessor const& posAccessor =
                        model.accessors[primitive.attributes.find("POSITION")->second];

                    tinygltf::BufferView const& posView = model.bufferViews[posAccessor.bufferView];

                    bufferPos = reinterpret_cast<float const*>(
                        &(model.buffers[posView.buffer].data[posAccessor.byteOffset + posView.byteOffset]));

                    posMin = glm::vec3(
                        posAccessor.minValues[0], posAccessor.minValues[1], posAccessor.minValues[2]);

                    posMax = glm::vec3(
                        posAccessor.maxValues[0], posAccessor.maxValues[1], posAccessor.maxValues[2]);

                    vertexCount = static_cast<uint32_t>(posAccessor.count);

                    posByteStride = posAccessor.ByteStride(posView)
                                        ? (posAccessor.ByteStride(posView) / sizeof(float))
                                        : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC3);

                    if (primitive.attributes.find("NORMAL") != primitive.attributes.end())
                    {
                        tinygltf::Accessor const& normAccessor =
                            model.accessors[primitive.attributes.find("NORMAL")->second];
                        tinygltf::BufferView const& normView = model.bufferViews[normAccessor.bufferView];
                        bufferNormals                        = reinterpret_cast<float const*>(
                            &(model.buffers[normView.buffer]
                                  .data[normAccessor.byteOffset + normView.byteOffset]));
                        normByteStride = normAccessor.ByteStride(normView)
                                             ? (normAccessor.ByteStride(normView) / sizeof(float))
                                             : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC3);
                    }

                    if (primitive.attributes.find("TANGENT") != primitive.attributes.end())
                    {
                        tinygltf::Accessor const& tanAccessor =
                            model.accessors[primitive.attributes.find("TANGENT")->second];

                        tinygltf::BufferView const& tanView = model.bufferViews[tanAccessor.bufferView];

                        bufferTangents = reinterpret_cast<float const*>(&(
                            model.buffers[tanView.buffer].data[tanAccessor.byteOffset + tanView.byteOffset]));

                        tangentByteStride = tanAccessor.ByteStride(tanView)
                                                ? (tanAccessor.ByteStride(tanView) / sizeof(float))
                                                : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC3);
                    }

                    // UVs
                    if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end())
                    {
                        tinygltf::Accessor const& uvAccessor =
                            model.accessors[primitive.attributes.find("TEXCOORD_0")->second];

                        tinygltf::BufferView const& uvView = model.bufferViews[uvAccessor.bufferView];

                        bufferTexCoordSet0 = reinterpret_cast<float const*>(
                            &(model.buffers[uvView.buffer].data[uvAccessor.byteOffset + uvView.byteOffset]));

                        uv0ByteStride = uvAccessor.ByteStride(uvView)
                                            ? (uvAccessor.ByteStride(uvView) / sizeof(float))
                                            : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC2);
                    }

                    if (primitive.attributes.find("TEXCOORD_1") != primitive.attributes.end())
                    {
                        tinygltf::Accessor const& uvAccessor =
                            model.accessors[primitive.attributes.find("TEXCOORD_1")->second];

                        tinygltf::BufferView const& uvView = model.bufferViews[uvAccessor.bufferView];

                        bufferTexCoordSet1 = reinterpret_cast<float const*>(
                            &(model.buffers[uvView.buffer].data[uvAccessor.byteOffset + uvView.byteOffset]));

                        uv1ByteStride = uvAccessor.ByteStride(uvView)
                                            ? (uvAccessor.ByteStride(uvView) / sizeof(float))
                                            : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC2);
                    }

                    // Vertex colors
                    if (primitive.attributes.find("COLOR_0") != primitive.attributes.end())
                    {
                        tinygltf::Accessor const& accessor =
                            model.accessors[primitive.attributes.find("COLOR_0")->second];

                        tinygltf::BufferView const& view = model.bufferViews[accessor.bufferView];

                        bufferColorSet0 = reinterpret_cast<float const*>(
                            &(model.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));

                        color0ByteStride = accessor.ByteStride(view)
                                               ? (accessor.ByteStride(view) / sizeof(float))
                                               : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC3);
                    }

                    // Skinning
                    // Joints
                    if (primitive.attributes.find("JOINTS_0") != primitive.attributes.end())
                    {
                        tinygltf::Accessor const& jointAccessor =
                            model.accessors[primitive.attributes.find("JOINTS_0")->second];

                        tinygltf::BufferView const& jointView = model.bufferViews[jointAccessor.bufferView];

                        bufferJoints = &(model.buffers[jointView.buffer]
                                             .data[jointAccessor.byteOffset + jointView.byteOffset]);

                        jointComponentType = jointAccessor.componentType;

                        jointByteStride = jointAccessor.ByteStride(jointView)
                                              ? (jointAccessor.ByteStride(jointView) /
                                                 tinygltf::GetComponentSizeInBytes(jointComponentType))
                                              : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC4);
                    }

                    if (primitive.attributes.find("WEIGHTS_0") != primitive.attributes.end())
                    {
                        tinygltf::Accessor const& weightAccessor =
                            model.accessors[primitive.attributes.find("WEIGHTS_0")->second];

                        tinygltf::BufferView const& weightView = model.bufferViews[weightAccessor.bufferView];

                        bufferWeights = reinterpret_cast<float const*>(
                            &(model.buffers[weightView.buffer]
                                  .data[weightAccessor.byteOffset + weightView.byteOffset]));

                        weightByteStride = weightAccessor.ByteStride(weightView)
                                               ? (weightAccessor.ByteStride(weightView) / sizeof(float))
                                               : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC4);
                    }

                    hasSkin = (bufferJoints && bufferWeights);

                    for (size_t v = 0; v < posAccessor.count; v++)
                    {
                        Vertex& vert = loaderInfo.vertexBuffer[loaderInfo.vertexPos] = Vertex {
                            .pos = glm::vec4(glm::make_vec3(&bufferPos[v * posByteStride]), 1.0f),

                            .normal = glm::normalize(
                                glm::vec3(bufferNormals ? glm::make_vec3(&bufferNormals[v * normByteStride])
                                                        : glm::vec3(0.0f))),

                            .uv0 = bufferTexCoordSet0 ? glm::make_vec2(&bufferTexCoordSet0[v * uv0ByteStride])
                                                      : glm::vec3(0.0f),

                            .uv1 = bufferTexCoordSet1 ? glm::make_vec2(&bufferTexCoordSet1[v * uv1ByteStride])
                                                      : glm::vec3(0.0f),

                            .color = bufferColorSet0 ? glm::make_vec4(&bufferColorSet0[v * color0ByteStride])
                                                     : glm::vec4(1.0f),

                            .tangent = bufferTangents ? glm::make_vec4(&bufferTangents[v * tangentByteStride])
                                                      : glm::vec4(0.0),
                        };

                        if (hasSkin)
                        {
                            switch (jointComponentType)
                            {
                                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                                    {
                                        uint16_t const* buf = static_cast<uint16_t const*>(bufferJoints);
                                        vert.joint0 = glm::uvec4(glm::make_vec4(&buf[v * jointByteStride]));
                                        break;
                                    }
                                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                                    {
                                        uint8_t const* buf = static_cast<uint8_t const*>(bufferJoints);
                                        vert.joint0 = glm::vec4(glm::make_vec4(&buf[v * jointByteStride]));
                                        break;
                                    }
                                default:
                                    MC_ASSERT_MSG(false,
                                                  "Joint component type {} not supported by the gltf spec",
                                                  jointComponentType);
                            }
                        }
                        else
                        {
                            vert.joint0 = glm::vec4(0.0f);
                        }

                        vert.weight0 =
                            hasSkin ? glm::make_vec4(&bufferWeights[v * weightByteStride]) : glm::vec4(0.0f);

                        // Fix for all zero weights
                        if (glm::length(vert.weight0) == 0.0f)
                        {
                            vert.weight0 = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
                        }

                        loaderInfo.vertexPos++;
                    }
                }

                // Indices
                if (hasIndices)
                {
                    // NOTE(aether) Why not just primitive.indices?
                    tinygltf::Accessor const& accessor =
                        model.accessors[primitive.indices > -1 ? primitive.indices : 0];

                    tinygltf::BufferView const& bufferView = model.bufferViews[accessor.bufferView];
                    tinygltf::Buffer const& buffer         = model.buffers[bufferView.buffer];

                    indexCount          = static_cast<uint32_t>(accessor.count);
                    void const* dataPtr = &(buffer.data[accessor.byteOffset + bufferView.byteOffset]);

                    switch (accessor.componentType)
                    {
                        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT:
                            {
                                uint32_t const* buf = static_cast<uint32_t const*>(dataPtr);
                                for (size_t index = 0; index < accessor.count; index++)
                                {
                                    loaderInfo.indexBuffer[loaderInfo.indexPos] = buf[index] + vertexStart;
                                    loaderInfo.indexPos++;
                                }
                                break;
                            }
                        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT:
                            {
                                uint16_t const* buf = static_cast<uint16_t const*>(dataPtr);
                                for (size_t index = 0; index < accessor.count; index++)
                                {
                                    loaderInfo.indexBuffer[loaderInfo.indexPos] = buf[index] + vertexStart;
                                    loaderInfo.indexPos++;
                                }
                                break;
                            }
                        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE:
                            {
                                uint8_t const* buf = static_cast<uint8_t const*>(dataPtr);
                                for (size_t index = 0; index < accessor.count; index++)
                                {
                                    loaderInfo.indexBuffer[loaderInfo.indexPos] = buf[index] + vertexStart;
                                    loaderInfo.indexPos++;
                                }
                                break;
                            }
                        default:
                            MC_ASSERT_MSG(
                                false, "Index component type {} not supported", accessor.componentType);
                    }
                }

                Primitive newPrimitive =
                    newMesh->primitives.emplace_back(indexStart,
                                                     indexCount,
                                                     vertexCount,
                                                     // Material #0 is the default material, so we add 1
                                                     primitive.material > -1 ? primitive.material + 1 : 0);

                newPrimitive.setBoundingBox(posMin, posMax);
            }

            // Mesh BB from BBs of primitives
            for (auto& p : newMesh->primitives)
            {
                if (p.bb.valid && !newMesh->bb.valid)
                {
                    newMesh->bb       = p.bb;
                    newMesh->bb.valid = true;
                }

                newMesh->bb.min = glm::min(newMesh->bb.min, p.bb.min);
                newMesh->bb.max = glm::max(newMesh->bb.max, p.bb.max);
            }
            newNode->mesh = std::move(newMesh);
        }

        if (parent)
        {
            parent->children.push_back(newNode);
        }
        else
        {
            nodes.push_back(newNode);
        }

        linearNodes.push_back(newNode);
    }
}  // namespace renderer::backend
