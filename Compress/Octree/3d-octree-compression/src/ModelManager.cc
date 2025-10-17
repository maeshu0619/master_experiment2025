#include "ModelManager.h"

#include "CompressedModel.h"
#include "Model.h"
#include "OBJLoader.h"
#include "OctreeCompressor.h"

ModelManager::ModelManager()
    : loader(std::make_unique<OBJLoader>()),
      compressor(std::make_unique<OctreeCompressor>()) {}

ModelManager::~ModelManager() = default;

std::unique_ptr<Model> ModelManager::loadModel(const std::string& filename) {
    return loader->load(filename);
}

std::unique_ptr<CompressedModel> ModelManager::compressModel(
    const Model& model) {
    return compressor->compress(model);
}

std::unique_ptr<CompressedModel> ModelManager::loadCompressedModel(
    const std::string& filename) {
    auto model = loadModel(filename);
    if (!model) {
        return nullptr;
    }
    return compressModel(*model);
}
