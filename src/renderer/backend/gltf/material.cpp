#include <mc/renderer/backend/gltf/loader.hpp>

#include <glm/gtc/type_ptr.hpp>

namespace renderer::backend
{
    void Model::loadMaterials(tinygltf::Model& gltfModel)
    {
        materials.reserve(gltfModel.materials.size() + 1);

        // Default material
        materials.push_back(Material());

        for (tinygltf::Material& mat : gltfModel.materials)
        {
            Material material {};

            material.doubleSided = mat.doubleSided;

            if (mat.values.find("baseColorTexture") != mat.values.end())
            {
                material.baseColorTexture       = &textures[mat.values["baseColorTexture"].TextureIndex()];
                material.texCoordSets.baseColor = mat.values["baseColorTexture"].TextureTexCoord();
            }

            if (mat.values.find("metallicRoughnessTexture") != mat.values.end())
            {
                material.metallicRoughnessTexture =
                    &textures[mat.values["metallicRoughnessTexture"].TextureIndex()];

                material.texCoordSets.metallicRoughness =
                    mat.values["metallicRoughnessTexture"].TextureTexCoord();
            }

            if (mat.values.find("roughnessFactor") != mat.values.end())
            {
                material.roughnessFactor = static_cast<float>(mat.values["roughnessFactor"].Factor());
            }

            if (mat.values.find("metallicFactor") != mat.values.end())
            {
                material.metallicFactor = static_cast<float>(mat.values["metallicFactor"].Factor());
            }

            if (mat.values.find("baseColorFactor") != mat.values.end())
            {
                material.baseColorFactor = glm::make_vec4(mat.values["baseColorFactor"].ColorFactor().data());
            }

            if (mat.additionalValues.find("normalTexture") != mat.additionalValues.end())
            {
                material.normalTexture = &textures[mat.additionalValues["normalTexture"].TextureIndex()];
                material.texCoordSets.normal = mat.additionalValues["normalTexture"].TextureTexCoord();
            }

            if (mat.additionalValues.find("emissiveTexture") != mat.additionalValues.end())
            {
                material.emissiveTexture = &textures[mat.additionalValues["emissiveTexture"].TextureIndex()];
                material.texCoordSets.emissive = mat.additionalValues["emissiveTexture"].TextureTexCoord();
            }

            if (mat.additionalValues.find("occlusionTexture") != mat.additionalValues.end())
            {
                material.occlusionTexture =
                    &textures[mat.additionalValues["occlusionTexture"].TextureIndex()];

                material.texCoordSets.occlusion = mat.additionalValues["occlusionTexture"].TextureTexCoord();
            }

            if (mat.additionalValues.find("alphaMode") != mat.additionalValues.end())
            {
                tinygltf::Parameter param = mat.additionalValues["alphaMode"];

                if (param.string_value == "BLEND")
                {
                    material.alphaMode = Material::ALPHAMODE_BLEND;
                }

                if (param.string_value == "MASK")
                {
                    material.alphaCutoff = 0.5f;
                    material.alphaMode   = Material::ALPHAMODE_MASK;
                }
            }

            if (mat.additionalValues.find("alphaCutoff") != mat.additionalValues.end())
            {
                material.alphaCutoff = static_cast<float>(mat.additionalValues["alphaCutoff"].Factor());
            }

            if (mat.additionalValues.find("emissiveFactor") != mat.additionalValues.end())
            {
                material.emissiveFactor = glm::vec4(
                    glm::make_vec3(mat.additionalValues["emissiveFactor"].ColorFactor().data()), 1.0);
            }

            // Extensions
            if (mat.extensions.find("KHR_materials_pbrSpecularGlossiness") != mat.extensions.end())
            {
                logger::warn("Application is not prepared to handle the specular glossiness workflow");

                auto ext = mat.extensions.find("KHR_materials_pbrSpecularGlossiness");

                if (ext->second.Has("specularGlossinessTexture"))
                {
                    auto index = ext->second.Get("specularGlossinessTexture").Get("index");
                    material.extension.specularGlossinessTexture = &textures[index.Get<int>()];

                    auto texCoordSet = ext->second.Get("specularGlossinessTexture").Get("texCoord");
                    material.texCoordSets.specularGlossiness = texCoordSet.Get<int>();
                    material.pbrWorkflow                     = PBRWorkflows::specularGlossiness;
                }

                if (ext->second.Has("diffuseTexture"))
                {
                    auto index                        = ext->second.Get("diffuseTexture").Get("index");
                    material.extension.diffuseTexture = &textures[index.Get<int>()];
                }

                if (ext->second.Has("diffuseFactor"))
                {
                    auto factor = ext->second.Get("diffuseFactor");

                    for (uint32_t i = 0; i < factor.ArrayLen(); i++)
                    {
                        auto val = factor.Get(i);
                        material.extension.diffuseFactor[i] =
                            val.IsNumber() ? (float)val.Get<double>() : (float)val.Get<int>();
                    }
                }

                if (ext->second.Has("specularFactor"))
                {
                    auto factor = ext->second.Get("specularFactor");

                    for (uint32_t i = 0; i < factor.ArrayLen(); i++)
                    {
                        auto val = factor.Get(i);
                        material.extension.specularFactor[i] =
                            val.IsNumber() ? (float)val.Get<double>() : (float)val.Get<int>();
                    }
                }
            }

            if (mat.extensions.find("KHR_materials_unlit") != mat.extensions.end())
            {
                material.unlit = true;
            }

            if (mat.extensions.find("KHR_materials_emissive_strength") != mat.extensions.end())
            {
                auto ext = mat.extensions.find("KHR_materials_emissive_strength");

                if (ext->second.Has("emissiveStrength"))
                {
                    auto value                = ext->second.Get("emissiveStrength");
                    material.emissiveStrength = (float)value.Get<double>();
                }
            }

            material.index = static_cast<uint32_t>(materials.size());
            materials.push_back(material);
        }
    }

    void Model::createMaterialBuffer()
    {
        std::vector<ShaderMaterial> shaderMaterials {};

        for (auto& material : materials)
        {
            ShaderMaterial shaderMaterial {};

            shaderMaterial.emissiveFactor   = material.emissiveFactor;
            shaderMaterial.emissiveStrength = material.emissiveStrength;

            // To save space, availabilty and texture coordinate set are combined
            // -1 = texture not used for this material, >= 0 texture used and index of
            // texture coordinate set

            shaderMaterial.colorTextureSet =
                material.baseColorTexture != nullptr ? material.texCoordSets.baseColor : -1;

            shaderMaterial.normalTextureSet =
                material.normalTexture != nullptr ? material.texCoordSets.normal : -1;

            shaderMaterial.occlusionTextureSet =
                material.occlusionTexture != nullptr ? material.texCoordSets.occlusion : -1;

            shaderMaterial.emissiveTextureSet =
                material.emissiveTexture != nullptr ? material.texCoordSets.emissive : -1;

            shaderMaterial.alphaMask = static_cast<float>(material.alphaMode == Material::ALPHAMODE_MASK);
            shaderMaterial.alphaMaskCutoff = material.alphaCutoff;

            if (material.pbrWorkflow == PBRWorkflows::metallicRoughness)
            {
                // Metallic roughness workflow
                shaderMaterial.workflow        = std::to_underlying(PBRWorkflows::metallicRoughness);
                shaderMaterial.baseColorFactor = material.baseColorFactor;
                shaderMaterial.metallicFactor  = material.metallicFactor;
                shaderMaterial.roughnessFactor = material.roughnessFactor;
                shaderMaterial.physicalDescriptorTextureSet = material.metallicRoughnessTexture != nullptr
                                                                  ? material.texCoordSets.metallicRoughness
                                                                  : -1;
                shaderMaterial.colorTextureSet =
                    material.baseColorTexture != nullptr ? material.texCoordSets.baseColor : -1;
            }
            else if (material.pbrWorkflow == PBRWorkflows::specularGlossiness)
            {
                // Specular glossiness workflow
                shaderMaterial.workflow = std::to_underlying(PBRWorkflows::specularGlossiness);
                shaderMaterial.physicalDescriptorTextureSet =
                    material.extension.specularGlossinessTexture != nullptr
                        ? material.texCoordSets.specularGlossiness
                        : -1;
                shaderMaterial.colorTextureSet =
                    material.extension.diffuseTexture != nullptr ? material.texCoordSets.baseColor : -1;
                shaderMaterial.diffuseFactor  = material.extension.diffuseFactor;
                shaderMaterial.specularFactor = glm::vec4(material.extension.specularFactor, 1.0f);
            }

            shaderMaterials.push_back(shaderMaterial);
        }

        vk::DeviceSize bufferSize = shaderMaterials.size() * sizeof(ShaderMaterial);

        auto stagingBufferAccessor = m_bufferManager->create(
            "Material staging buffer",
            bufferSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT);

        std::memcpy(stagingBufferAccessor.getMappedData(), shaderMaterials.data(), bufferSize);

        materialBuffer = m_bufferManager->create("Material buffer",
                                                 bufferSize,
                                                 vk::BufferUsageFlagBits::eShaderDeviceAddress |
                                                     vk::BufferUsageFlagBits::eTransferDst,
                                                 VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                                 VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

        materialBufferAddress =
            m_device->get().getBufferAddress(vk::BufferDeviceAddressInfo().setBuffer(materialBuffer));

        // TODO(aether) the deconstructor will block until the copy is over
        // not the most performant approach
        ScopedCommandBuffer(*m_device, m_cmdManager->getTransferCmdPool(), m_device->getTransferQueue(), true)
            ->copyBuffer(stagingBufferAccessor, materialBuffer, vk::BufferCopy().setSize(bufferSize));
    }
}  // namespace renderer::backend
