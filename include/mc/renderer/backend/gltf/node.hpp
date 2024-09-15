#pragma once

#include "mesh.hpp"

#include <memory>
#include <string>
#include <vector>

#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/quaternion_double.hpp>

namespace renderer::backend
{
    struct Node;

    struct Skin
    {
        std::string name;
        Node* skeletonRoot = nullptr;

        std::vector<glm::mat4> inverseBindMatrices;
        std::vector<Node*> joints;
    };

    struct Node
    {
        void update();

        ~Node()
        {
            for (auto& children : children)
            {
                delete children;
            }
        };

        glm::mat4 localMatrix();
        glm::mat4 getMatrix();

        std::string name;

        Node* parent;
        std::vector<Node*> children;

        uint32_t index;
        glm::mat4 matrix;

        std::unique_ptr<Mesh> mesh;

        Skin* skin;
        int32_t skinIndex = -1;

        glm::vec3 translation {};
        glm::vec3 scale { 1.0f };
        glm::dquat rotation {};

        BoundingBox bvh;
        BoundingBox aabb;

        glm::mat4 cachedLocalMatrix { glm::mat4(1.0f) };
        glm::mat4 cachedMatrix { glm::mat4(1.0f) };

        bool useCachedMatrix { false };
    };
}  // namespace renderer::backend
