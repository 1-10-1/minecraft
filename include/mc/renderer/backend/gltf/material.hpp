#pragma once

#include "gltfTextures.hpp"

#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>

namespace renderer::backend
{
    enum class PBRWorkflows
    {
        metallicRoughness  = 0,
        specularGlossiness = 1
    };

    struct alignas(16) ShaderMaterial
    {
        glm::vec4 baseColorFactor;
        glm::vec4 emissiveFactor;
        glm::vec4 diffuseFactor;
        glm::vec4 specularFactor;

        uint32_t workflow;

        float metallicFactor;
        float emissiveStrength;
        float roughnessFactor;

        int colorTextureSet;
        int normalTextureSet;
        int occlusionTextureSet;
        int emissiveTextureSet;
        int physicalDescriptorTextureSet;

        float alphaMask;
        float alphaMaskCutoff;

        int flags;
    };

    struct Material
    {
        enum AlphaMode
        {
            ALPHAMODE_OPAQUE,
            ALPHAMODE_MASK,
            ALPHAMODE_BLEND
        };

        AlphaMode alphaMode       = ALPHAMODE_OPAQUE;
        float alphaCutoff         = 1.0f;
        float metallicFactor      = 1.0f;
        float roughnessFactor     = 1.0f;
        glm::vec4 baseColorFactor = glm::vec4(1.0f);
        glm::vec4 emissiveFactor  = glm::vec4(0.0f);
        GlTFTexture* baseColorTexture;
        GlTFTexture* metallicRoughnessTexture;
        GlTFTexture* normalTexture;
        GlTFTexture* occlusionTexture;
        GlTFTexture* emissiveTexture;
        bool doubleSided = false;

        struct TexCoordSets
        {
            uint8_t baseColor          = 0;
            uint8_t metallicRoughness  = 0;
            uint8_t specularGlossiness = 0;
            uint8_t normal             = 0;
            uint8_t occlusion          = 0;
            uint8_t emissive           = 0;
        } texCoordSets;

        struct Extension
        {
            GlTFTexture* specularGlossinessTexture;
            GlTFTexture* diffuseTexture;
            glm::vec4 diffuseFactor  = glm::vec4(1.0f);
            glm::vec3 specularFactor = glm::vec3(0.0f);
        } extension;

        PBRWorkflows pbrWorkflow = PBRWorkflows::metallicRoughness;

        int index              = 0;
        bool unlit             = false;
        float emissiveStrength = 1.0f;
    };
}  // namespace renderer::backend
