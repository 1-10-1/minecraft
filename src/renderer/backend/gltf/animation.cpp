#include <mc/renderer/backend/gltf/animation.hpp>
#include <mc/renderer/backend/gltf/loader.hpp>
#include <mc/renderer/backend/gltf/node.hpp>

#include <glm/exponential.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace renderer::backend
{
    // Cube spline interpolation function used for translate/scale/rotate with cubic spline animation samples
    // Details on how this works can be found in the specs
    // https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#appendix-c-spline-interpolation
    glm::vec4 AnimationSampler::cubicSplineInterpolation(size_t index, float time, uint32_t stride)
    {
        float delta          = inputs[index + 1] - inputs[index];
        float t              = (time - inputs[index]) / delta;
        size_t const current = index * stride * 3;
        size_t const next    = (index + 1) * stride * 3;
        size_t const A       = 0;
        size_t const V       = stride * 1;
        // [[maybe_unused]] size_t const B       = stride * 2;

        float t2 = glm::pow(t, 2);
        float t3 = glm::pow(t, 3);

        glm::vec4 pt { 0.0f };
        for (uint32_t i = 0; i < stride; i++)
        {
            float p0 = outputs[current + i + V];          // starting point at t = 0
            float m0 = delta * outputs[current + i + A];  // scaled starting tangent at t = 0
            float p1 = outputs[next + i + V];             // ending point at t = 1
            // [[maybe_unused]] float m1 = delta * outputs[next + i + B];     // scaled ending tangent at t = 1
            pt[i] = ((2.f * t3 - 3.f * t2 + 1.f) * p0) + ((t3 - 2.f * t2 + t) * m0) +
                    ((-2.f * t3 + 3.f * t2) * p1) + ((t3 - t2) * m0);
        }
        return pt;
    }

    // Calculates the translation of this sampler for the given node at a given time point depending on the interpolation type
    void AnimationSampler::translate(size_t index, float time, Node* node)
    {
        switch (interpolation)
        {
            case AnimationSampler::InterpolationType::linear:
                {
                    float u = std::max(0.0f, time - inputs[index]) / (inputs[index + 1] - inputs[index]);
                    node->translation =
                        glm::make_vec3(glm::mix(outputsVec4[index], outputsVec4[index + 1], u));
                    break;
                }
            case AnimationSampler::InterpolationType::step:
                {
                    node->translation = glm::make_vec3(outputsVec4[index]);
                    break;
                }
            case AnimationSampler::InterpolationType::cubicSpline:
                {
                    node->translation = glm::make_vec3(cubicSplineInterpolation(index, time, 3));
                    break;
                }
        }
    }

    // Calculates the scale of this sampler for the given node at a given time point depending on the interpolation type
    void AnimationSampler::scale(size_t index, float time, Node* node)
    {
        switch (interpolation)
        {
            case AnimationSampler::InterpolationType::linear:
                {
                    float u     = std::max(0.0f, time - inputs[index]) / (inputs[index + 1] - inputs[index]);
                    node->scale = glm::make_vec3(glm::mix(outputsVec4[index], outputsVec4[index + 1], u));
                    break;
                }
            case AnimationSampler::InterpolationType::step:
                {
                    node->scale = glm::make_vec3(outputsVec4[index]);
                    break;
                }
            case AnimationSampler::InterpolationType::cubicSpline:
                {
                    node->scale = glm::make_vec3(cubicSplineInterpolation(index, time, 3));
                    break;
                }
        }
    }

    // Calculates the rotation of this sampler for the given node at a given time point depending on the interpolation type
    void AnimationSampler::rotate(size_t index, float time, Node* node)
    {
        switch (interpolation)
        {
            case AnimationSampler::InterpolationType::linear:
                {
                    float u = std::max(0.0f, time - inputs[index]) / (inputs[index + 1] - inputs[index]);
                    glm::quat q1;
                    q1.x = outputsVec4[index].x;
                    q1.y = outputsVec4[index].y;
                    q1.z = outputsVec4[index].z;
                    q1.w = outputsVec4[index].w;
                    glm::quat q2;
                    q2.x           = outputsVec4[index + 1].x;
                    q2.y           = outputsVec4[index + 1].y;
                    q2.z           = outputsVec4[index + 1].z;
                    q2.w           = outputsVec4[index + 1].w;
                    node->rotation = glm::normalize(glm::slerp(q1, q2, u));
                    break;
                }
            case AnimationSampler::InterpolationType::step:
                {
                    glm::quat q1;
                    q1.x           = outputsVec4[index].x;
                    q1.y           = outputsVec4[index].y;
                    q1.z           = outputsVec4[index].z;
                    q1.w           = outputsVec4[index].w;
                    node->rotation = q1;
                    break;
                }
            case AnimationSampler::InterpolationType::cubicSpline:
                {
                    glm::vec4 rot = cubicSplineInterpolation(index, time, 4);
                    glm::quat q;
                    q.x            = rot.x;
                    q.y            = rot.y;
                    q.z            = rot.z;
                    q.w            = rot.w;
                    node->rotation = glm::normalize(q);
                    break;
                }
        }
    }

    void Model::loadAnimations(tinygltf::Model& gltfModel)
    {
        for (tinygltf::Animation& anim : gltfModel.animations)
        {
            Animation animation { .name = anim.name.empty() ? std::to_string(animations.size()) : anim.name };

            for (auto& samp : anim.samplers)
            {
                AnimationSampler sampler {};

                if (samp.interpolation == "LINEAR")
                {
                    sampler.interpolation = AnimationSampler::InterpolationType::linear;
                }

                if (samp.interpolation == "STEP")
                {
                    sampler.interpolation = AnimationSampler::InterpolationType::step;
                }

                if (samp.interpolation == "CUBICSPLINE")
                {
                    sampler.interpolation = AnimationSampler::InterpolationType::cubicSpline;
                }

                // Read sampler input time values
                {
                    tinygltf::Accessor const& accessor     = gltfModel.accessors[samp.input];
                    tinygltf::BufferView const& bufferView = gltfModel.bufferViews[accessor.bufferView];
                    tinygltf::Buffer const& buffer         = gltfModel.buffers[bufferView.buffer];

                    assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);

                    void const* dataPtr = &buffer.data[accessor.byteOffset + bufferView.byteOffset];
                    float const* buf    = static_cast<float const*>(dataPtr);

                    for (size_t index = 0; index < accessor.count; index++)
                    {
                        sampler.inputs.push_back(buf[index]);
                    }

                    for (auto input : sampler.inputs)
                    {
                        if (input < animation.start)
                        {
                            animation.start = input;
                        };
                        if (input > animation.end)
                        {
                            animation.end = input;
                        }
                    }
                }

                // Read sampler output T/R/S values
                {
                    tinygltf::Accessor const& accessor     = gltfModel.accessors[samp.output];
                    tinygltf::BufferView const& bufferView = gltfModel.bufferViews[accessor.bufferView];
                    tinygltf::Buffer const& buffer         = gltfModel.buffers[bufferView.buffer];

                    assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);

                    void const* dataPtr = &buffer.data[accessor.byteOffset + bufferView.byteOffset];

                    switch (accessor.type)
                    {
                        case TINYGLTF_TYPE_VEC3:
                            {
                                glm::vec3 const* buf = static_cast<glm::vec3 const*>(dataPtr);
                                for (size_t index = 0; index < accessor.count; index++)
                                {
                                    sampler.outputsVec4.push_back(glm::vec4(buf[index], 0.0f));
                                    sampler.outputs.push_back(buf[index][0]);
                                    sampler.outputs.push_back(buf[index][1]);
                                    sampler.outputs.push_back(buf[index][2]);
                                }
                                break;
                            }
                        case TINYGLTF_TYPE_VEC4:
                            {
                                glm::vec4 const* buf = static_cast<glm::vec4 const*>(dataPtr);
                                for (size_t index = 0; index < accessor.count; index++)
                                {
                                    sampler.outputsVec4.push_back(buf[index]);
                                    sampler.outputs.push_back(buf[index][0]);
                                    sampler.outputs.push_back(buf[index][1]);
                                    sampler.outputs.push_back(buf[index][2]);
                                    sampler.outputs.push_back(buf[index][3]);
                                }
                                break;
                            }
                        default:
                            {
                                MC_ASSERT_MSG(false, "Unknown type");
                                break;
                            }
                    }
                }

                animation.samplers.push_back(sampler);
            }

            // Channels
            for (auto& source : anim.channels)
            {
                AnimationChannel channel {};

                if (source.target_path == "weights")
                {
                    logger::warn("weights not yet supported, skipping channel");
                    continue;
                }

                if (source.target_path == "rotation")
                {
                    channel.path = AnimationChannel::PathType::rotation;
                }

                if (source.target_path == "translation")
                {
                    channel.path = AnimationChannel::PathType::translation;
                }

                if (source.target_path == "scale")
                {
                    channel.path = AnimationChannel::PathType::scale;
                }

                channel.samplerIndex = source.sampler;
                channel.node         = nodeFromIndex(source.target_node);

                if (!channel.node)
                {
                    continue;
                }

                animation.channels.push_back(channel);
            }

            animations.push_back(animation);
        }
    }

    void Model::updateAnimation(uint32_t index, float time)
    {
        if (animations.empty())
        {
            logger::warn("glTF does not contain animation");
            return;
        }

        if (index > static_cast<uint32_t>(animations.size()) - 1)
        {
            logger::warn("No animation with index {}", index);
            return;
        }

        Animation& animation = animations[index];

        bool updated = false;

        for (auto& channel : animation.channels)
        {
            AnimationSampler& sampler = animation.samplers[channel.samplerIndex];

            if (sampler.inputs.size() > sampler.outputsVec4.size())
            {
                continue;
            }

            for (size_t i = 0; i < sampler.inputs.size() - 1; i++)
            {
                if ((time >= sampler.inputs[i]) && (time <= sampler.inputs[i + 1]))
                {
                    float u = std::max(0.0f, time - sampler.inputs[i]) /
                              (sampler.inputs[i + 1] - sampler.inputs[i]);

                    if (u <= 1.0f)
                    {
                        switch (channel.path)
                        {
                            case AnimationChannel::PathType::translation:
                                sampler.translate(i, time, channel.node);
                                break;
                            case AnimationChannel::PathType::scale:
                                sampler.scale(i, time, channel.node);
                                break;
                            case AnimationChannel::PathType::rotation:
                                sampler.rotate(i, time, channel.node);
                                break;
                        }

                        updated = true;
                    }
                }
            }
        }

        if (updated)
        {
            for (auto& node : nodes)
            {
                node->update();
            }
        }
    }
}  // namespace renderer::backend
