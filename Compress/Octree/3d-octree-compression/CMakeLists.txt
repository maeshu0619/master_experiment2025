cmake_minimum_required(VERSION 3.10)
project(3dOctreeCompression)

set(CMAKE_CXX_STANDARD 17)

# Find packages
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

# Add header files
set(HEADERS
    include/Model.h
    include/IModelLoader.h
    include/OBJLoader.h
    include/Octree.h
    include/VertexData.h
    include/CompressedModel.h
    include/IModelCompressor.h
    include/OctreeCompressor.h
    include/ModelManager.h
    include/OctreeVisualizer.h
)

# Add source files
set(SOURCES
    main.cc
    src/Model.cc
    src/OBJLoader.cc
    src/CompressedModel.cc
    src/OctreeCompressor.cc
    src/ModelManager.cc
    src/OctreeVisualizer.cc
)

# Create a separate object library for glad to control its compilation flags
add_library(glad_lib OBJECT src/glad.c)
target_include_directories(glad_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Disable pedantic warnings for glad.c
if(NOT MSVC)
    target_compile_options(glad_lib PRIVATE -w)  # Disable all warnings for glad
endif()

# Add executable
add_executable(3dOctreeCompression 
    ${SOURCES}
    ${HEADERS}
    $<TARGET_OBJECTS:glad_lib>
)

# Include directories
target_include_directories(3dOctreeCompression PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link libraries
target_link_libraries(3dOctreeCompression 
    OpenGL::GL
    glfw
    glm::glm
)

# Copy models directory to build directory
add_custom_command(TARGET 3dOctreeCompression POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/models $<TARGET_FILE_DIR:3dOctreeCompression>/models)

# Set output directory for the executable
set_target_properties(3dOctreeCompression PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Enable warnings for our code (but not for glad)
if(MSVC)
    target_compile_options(3dOctreeCompression PRIVATE /W4)
else()
    target_compile_options(3dOctreeCompression PRIVATE -Wall -Wextra -Wpedantic)
endif()
