#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <glm/ext/vector_float4.hpp>

namespace renderer::backend
{
    struct Node;

    struct AnimationChannel
    {
        enum class PathType
        {
            translation,
            rotation,
            scale
        };

        PathType path;
        Node* node;
        uint32_t samplerIndex;
    };

    struct AnimationSampler
    {
        enum class InterpolationType
        {
            linear,
            step,
            cubicSpline
        };

        InterpolationType interpolation;
        std::vector<float> inputs;
        std::vector<glm::vec4> outputsVec4;
        std::vector<float> outputs;

        glm::vec4 cubicSplineInterpolation(size_t index, float time, uint32_t stride);
        void translate(size_t index, float time, Node* node);
        void scale(size_t index, float time, Node* node);
        void rotate(size_t index, float time, Node* node);
    };

    struct Animation
    {
        std::string name;
        std::vector<AnimationSampler> samplers;
        std::vector<AnimationChannel> channels;
        float start = std::numeric_limits<float>::max();
        float end   = std::numeric_limits<float>::min();
    };
}  // namespace renderer::backend
