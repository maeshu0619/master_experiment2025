#include "Model.h"

#include <algorithm>
#include <limits>

void Model::calculateBounds() {
    if (vertices.empty()) {
        minBounds = glm::vec3(0.0f);
        maxBounds = glm::vec3(0.0f);
        return;
    }

    minBounds = glm::vec3(std::numeric_limits<float>::max());
    maxBounds = glm::vec3(std::numeric_limits<float>::lowest());

    for (const auto& vertex : vertices) {
        minBounds = glm::min(minBounds, vertex);
        maxBounds = glm::max(maxBounds, vertex);
    }
}

bool Model::isValid() const {
    return !vertices.empty() && vertices.size() == colors.size();
}
