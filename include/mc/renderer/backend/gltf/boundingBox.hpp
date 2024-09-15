#pragma once

#include <glm/ext/matrix_float4x4.hpp>

namespace renderer::backend
{
    struct BoundingBox
    {
        BoundingBox() {};

        BoundingBox(glm::vec3 min, glm::vec3 max) : min(min), max(max) {};

        BoundingBox getAABB(glm::mat4 m);

        glm::vec3 min;
        glm::vec3 max;

        bool valid = false;
    };
}  // namespace renderer::backend
