#include <mc/renderer/backend/gltf/boundingBox.hpp>
#include <mc/renderer/backend/gltf/loader.hpp>

namespace renderer::backend
{
    BoundingBox BoundingBox::getAABB(glm::mat4 m)
    {
        glm::vec3 min = glm::vec3(m[3]);
        glm::vec3 max = min;
        glm::vec3 v0, v1;

        glm::vec3 right = glm::vec3(m[0]);
        v0              = right * this->min.x;
        v1              = right * this->max.x;
        min += glm::min(v0, v1);
        max += glm::max(v0, v1);

        glm::vec3 up = glm::vec3(m[1]);
        v0           = up * this->min.y;
        v1           = up * this->max.y;
        min += glm::min(v0, v1);
        max += glm::max(v0, v1);

        glm::vec3 back = glm::vec3(m[2]);
        v0             = back * this->min.z;
        v1             = back * this->max.z;
        min += glm::min(v0, v1);
        max += glm::max(v0, v1);

        return BoundingBox(min, max);
    }

    void calculateBvhRecursive(Node* node)
    {
        if (node->mesh)
        {
            if (node->mesh->bb.valid)
            {
                node->aabb = node->mesh->bb.getAABB(node->getMatrix());

                if (node->children.size() == 0)
                {
                    node->bvh.min   = node->aabb.min;
                    node->bvh.max   = node->aabb.max;
                    node->bvh.valid = true;
                }
            }
        }

        for (auto& child : node->children)
        {
            calculateBvhRecursive(child);
        }
    }

    auto BoundingBox::calcNodeHeirarchyBB(std::vector<Node*> const& nodes) -> std::pair<Dimensions, glm::mat4>
    {
        for (auto node : nodes)
        {
            calculateBvhRecursive(node);
        }

        Dimensions dimensions {
            .min = glm::vec3(std::numeric_limits<float>::max()),
            .max = glm::vec3(-std::numeric_limits<float>::max()),
        };

        for (auto node : nodes)
        {
            if (node->bvh.valid)
            {
                dimensions.min = glm::min(dimensions.min, node->bvh.min);
                dimensions.max = glm::max(dimensions.max, node->bvh.max);
            }
        }

        glm::mat4 aabb = glm::scale(glm::mat4(1.0f),
                                    glm::vec3(dimensions.max[0] - dimensions.min[0],
                                              dimensions.max[1] - dimensions.min[1],
                                              dimensions.max[2] - dimensions.min[2]));

        aabb[3][0] = dimensions.min[0];
        aabb[3][1] = dimensions.min[1];
        aabb[3][2] = dimensions.min[2];

        return { dimensions, aabb };
    }
}  // namespace renderer::backend
