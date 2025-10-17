#include "OctreeVisualizer.h"

OctreeVisualizer::OctreeVisualizer() {}

std::vector<OctreeVisualizer::BoundingBox>
OctreeVisualizer::extractBoundingBoxes(const Octree<VertexData>* octree,
                                       int maxLevel) const {
    std::vector<BoundingBox> boxes;
    if (!octree || !octree->getRoot()) return boxes;

    extractBoxesRecursive(octree->getRoot(), boxes, 0, maxLevel);
    return boxes;
}

void OctreeVisualizer::extractBoxesRecursive(
    const typename Octree<VertexData>::Node* node,
    std::vector<BoundingBox>& boxes, int currentLevel, int maxLevel) const {
    if (!node) return;

    // Stop if we've reached the max level (unless maxLevel is -1, which means
    // no limit)
    if (maxLevel >= 0 && currentLevel > maxLevel) return;

    // Only add boxes that contain data or have children with data
    bool hasData = !node->data.empty();
    bool hasChildrenWithData = false;

    if (!node->isLeaf()) {
        for (const auto& child : node->children) {
            if (child && (!child->data.empty() || !child->isLeaf())) {
                hasChildrenWithData = true;
                break;
            }
        }
    }

    if (hasData || hasChildrenWithData) {
        BoundingBox box;
        box.center = node->center;
        box.halfSize = glm::vec3(node->halfSize);
        box.hasData = hasData;
        box.level = currentLevel;
        boxes.push_back(box);
    }

    // Recursively process children
    if (!node->isLeaf()) {
        for (const auto& child : node->children) {
            extractBoxesRecursive(child.get(), boxes, currentLevel + 1,
                                  maxLevel);
        }
    }
}

std::vector<glm::vec3> OctreeVisualizer::getWireframeVertices(
    const BoundingBox& box) const {
    std::vector<glm::vec3> vertices;

    glm::vec3 min = box.center - box.halfSize;
    glm::vec3 max = box.center + box.halfSize;

    // Define the 8 corners of the box
    glm::vec3 corners[8] = {
        glm::vec3(min.x, min.y, min.z), glm::vec3(max.x, min.y, min.z),
        glm::vec3(max.x, max.y, min.z), glm::vec3(min.x, max.y, min.z),
        glm::vec3(min.x, min.y, max.z), glm::vec3(max.x, min.y, max.z),
        glm::vec3(max.x, max.y, max.z), glm::vec3(min.x, max.y, max.z)};

    // Define edges as pairs of corner indices
    int edges[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},  // Bottom face
        {4, 5}, {5, 6}, {6, 7}, {7, 4},  // Top face
        {0, 4}, {1, 5}, {2, 6}, {3, 7}   // Vertical edges
    };

    // Add vertices for each edge
    for (int i = 0; i < 12; ++i) {
        vertices.push_back(corners[edges[i][0]]);
        vertices.push_back(corners[edges[i][1]]);
    }

    return vertices;
}

std::vector<glm::vec3> OctreeVisualizer::getSolidBoxVertices(
    const BoundingBox& box) const {
    std::vector<glm::vec3> vertices;

    glm::vec3 min = box.center - box.halfSize;
    glm::vec3 max = box.center + box.halfSize;

    // Front face
    vertices.push_back(glm::vec3(min.x, min.y, max.z));
    vertices.push_back(glm::vec3(max.x, min.y, max.z));
    vertices.push_back(glm::vec3(max.x, max.y, max.z));
    vertices.push_back(glm::vec3(min.x, min.y, max.z));
    vertices.push_back(glm::vec3(max.x, max.y, max.z));
    vertices.push_back(glm::vec3(min.x, max.y, max.z));

    // Back face
    vertices.push_back(glm::vec3(max.x, min.y, min.z));
    vertices.push_back(glm::vec3(min.x, min.y, min.z));
    vertices.push_back(glm::vec3(min.x, max.y, min.z));
    vertices.push_back(glm::vec3(max.x, min.y, min.z));
    vertices.push_back(glm::vec3(min.x, max.y, min.z));
    vertices.push_back(glm::vec3(max.x, max.y, min.z));

    // Left face
    vertices.push_back(glm::vec3(min.x, min.y, min.z));
    vertices.push_back(glm::vec3(min.x, min.y, max.z));
    vertices.push_back(glm::vec3(min.x, max.y, max.z));
    vertices.push_back(glm::vec3(min.x, min.y, min.z));
    vertices.push_back(glm::vec3(min.x, max.y, max.z));
    vertices.push_back(glm::vec3(min.x, max.y, min.z));

    // Right face
    vertices.push_back(glm::vec3(max.x, min.y, max.z));
    vertices.push_back(glm::vec3(max.x, min.y, min.z));
    vertices.push_back(glm::vec3(max.x, max.y, min.z));
    vertices.push_back(glm::vec3(max.x, min.y, max.z));
    vertices.push_back(glm::vec3(max.x, max.y, min.z));
    vertices.push_back(glm::vec3(max.x, max.y, max.z));

    // Top face
    vertices.push_back(glm::vec3(min.x, max.y, max.z));
    vertices.push_back(glm::vec3(max.x, max.y, max.z));
    vertices.push_back(glm::vec3(max.x, max.y, min.z));
    vertices.push_back(glm::vec3(min.x, max.y, max.z));
    vertices.push_back(glm::vec3(max.x, max.y, min.z));
    vertices.push_back(glm::vec3(min.x, max.y, min.z));

    // Bottom face
    vertices.push_back(glm::vec3(min.x, min.y, min.z));
    vertices.push_back(glm::vec3(max.x, min.y, min.z));
    vertices.push_back(glm::vec3(max.x, min.y, max.z));
    vertices.push_back(glm::vec3(min.x, min.y, min.z));
    vertices.push_back(glm::vec3(max.x, min.y, max.z));
    vertices.push_back(glm::vec3(min.x, min.y, max.z));

    return vertices;
}
