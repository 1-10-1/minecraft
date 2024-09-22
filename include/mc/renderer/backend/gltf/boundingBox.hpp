#pragma once

#include <glm/ext/matrix_float4x4.hpp>

namespace renderer::backend
{
    struct Node;

    struct BoundingBox
    {
        struct Dimensions
        {
            // what
            glm::vec3 min = glm::vec3(std::numeric_limits<float>::max());
            glm::vec3 max = glm::vec3(-std::numeric_limits<float>::max());
        };

        BoundingBox() {};

        BoundingBox(glm::vec3 min, glm::vec3 max) : min(min), max(max) {};

        BoundingBox getAABB(glm::mat4 m);

        static auto calcNodeHeirarchyBB(std::vector<Node*> const& nodes) -> std::pair<Dimensions, glm::mat4>;

        glm::vec3 min;
        glm::vec3 max;

        bool valid = false;
    };

}  // namespace renderer::backend
