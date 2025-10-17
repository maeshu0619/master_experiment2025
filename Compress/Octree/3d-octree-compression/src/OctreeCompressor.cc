#include "OctreeCompressor.h"

#include <algorithm>

#include "CompressedModel.h"
#include "Model.h"
#include "VertexData.h"

OctreeCompressor::OctreeCompressor(const Settings& settings)
    : settings(settings) {}

OctreeCompressor::OctreeCompressor() : settings(Settings{}) {}

std::unique_ptr<CompressedModel> OctreeCompressor::compress(
    const Model& model) {
    if (!model.isValid()) {
        return nullptr;
    }

    // Calculate octree bounds
    glm::vec3 center = (model.minBounds + model.maxBounds) * 0.5f;
    glm::vec3 extent = model.maxBounds - model.minBounds;
    float maxExtent = std::max({extent.x, extent.y, extent.z});
    float halfSize = maxExtent * 0.5f * 1.1f;  // Add 10% padding

    // Create octree
    auto octree = std::make_unique<Octree<VertexData>>(center, halfSize,
                                                       settings.maxDepth);

    // Insert all vertices
    for (size_t i = 0; i < model.vertices.size(); ++i) {
        VertexData data(model.vertices[i], model.colors[i]);
        octree->insert(data, model.vertices[i]);
    }

    return std::make_unique<CompressedModel>(std::move(octree), model.minBounds,
                                             model.maxBounds);
}
