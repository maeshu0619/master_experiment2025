#include "CompressedModel.h"

#include "Model.h"

CompressedModel::CompressedModel(std::unique_ptr<VertexOctree> octree,
                                 const glm::vec3& minBounds,
                                 const glm::vec3& maxBounds)
    : octree(std::move(octree)), minBounds(minBounds), maxBounds(maxBounds) {}

std::unique_ptr<Model> CompressedModel::decompress() const {
    auto model = std::make_unique<Model>();

    // Query entire bounds to get all vertices
    auto allVertices = octree->query(minBounds, maxBounds);

    model->vertices.reserve(allVertices.size());
    model->colors.reserve(allVertices.size());

    for (const auto& vertexData : allVertices) {
        model->vertices.push_back(vertexData.position);
        model->colors.push_back(vertexData.color);
    }

    model->minBounds = minBounds;
    model->maxBounds = maxBounds;

    return model;
}

size_t CompressedModel::getCompressedSize() const {
    // Estimate based on octree structure
    // This is a simplified calculation
    return sizeof(CompressedModel) + getVertexCount() * sizeof(VertexData);
}

size_t CompressedModel::getVertexCount() const {
    auto allVertices = octree->query(minBounds, maxBounds);
    return allVertices.size();
}
