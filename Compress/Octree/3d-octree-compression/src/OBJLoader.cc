#include "OBJLoader.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "Model.h"

std::unique_ptr<Model> OBJLoader::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    auto model = std::make_unique<Model>();
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "v") {
            glm::vec3 vertex;
            iss >> vertex.x >> vertex.y >> vertex.z;
            model->vertices.push_back(vertex);

            // Default color (can be extended to read from file)
            model->colors.push_back(glm::vec3(0.7f, 0.7f, 0.7f));
        }
        // Can be extended to handle faces, normals, texture coords, etc.
    }

    model->calculateBounds();

    if (!model->isValid()) {
        throw std::runtime_error("Invalid model loaded from: " + filename);
    }

    return model;
}
