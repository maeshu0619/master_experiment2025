// clang-format off
#include <glad/glad.h>   // MUST be first
#include <GLFW/glfw3.h>  // MUST be after glad
// clang-format on

#include <chrono>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "CompressedModel.h"
#include "Model.h"
#include "ModelManager.h"
#include "OctreeVisualizer.h"

// Vertex shader for octree boxes
const char* boxVertexShaderSource = R"(
#version 460 core
layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
)";

// Fragment shader for octree boxes
const char* boxFragmentShaderSource = R"(
#version 460 core
out vec4 FragColor;

uniform vec4 boxColor;
uniform float alpha;

void main() {
    FragColor = vec4(boxColor.rgb, alpha);
}
)";

// Original vertex shader for points
const char* vertexShaderSource = R"(
#version 460 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec3 vertexColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    gl_PointSize = 3.0;
    vertexColor = aColor;
}
)";

// Original fragment shader for points
const char* fragmentShaderSource = R"(
#version 460 core
in vec3 vertexColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(vertexColor, 1.0);
}
)";

// Camera controls
float cameraDistance = 3.0f;
float cameraAngleX = 0.0f;
float cameraAngleY = 30.0f;
bool mousePressed = false;
double lastMouseX = 0.0;
double lastMouseY = 0.0;

// Visualization state
bool showOctree = true;
bool showBoundaries = true;
bool showPoints = true;
bool showOnlyCurrentLevel = true;  // New: Show only current level by default
bool animateSubdivision = false;
int currentSubdivisionLevel = 0;
int maxSubdivisionLevel = 0;
float animationSpeed = 1.0f;
std::chrono::steady_clock::time_point lastAnimationUpdate;

// Model data
std::unique_ptr<Model> originalModel;
std::unique_ptr<CompressedModel> compressedModel;
std::unique_ptr<OctreeVisualizer> octreeVisualizer;

void updateWindowTitle(GLFWwindow* window) {
    std::stringstream title;
    title << "3D Octree Compression Visualizer - ";
    title << "Level: " << currentSubdivisionLevel << "/" << maxSubdivisionLevel;
    title << " - "
          << (showOnlyCurrentLevel ? "Current Level Only" : "All Levels");
    glfwSetWindowTitle(window, title.str().c_str());
}

void key_callback(GLFWwindow* window, int key, [[maybe_unused]] int scancode,
                  int action, [[maybe_unused]] int mods) {
    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_O:
                showOctree = !showOctree;
                break;
            case GLFW_KEY_B:
                showBoundaries = !showBoundaries;
                break;
            case GLFW_KEY_P:
                showPoints = !showPoints;
                break;
            case GLFW_KEY_L:  // New: Toggle showing only current level
                showOnlyCurrentLevel = !showOnlyCurrentLevel;
                updateWindowTitle(window);
                break;
            case GLFW_KEY_SPACE:
                animateSubdivision = !animateSubdivision;
                lastAnimationUpdate = std::chrono::steady_clock::now();
                break;
            case GLFW_KEY_R:
                currentSubdivisionLevel = 0;
                animateSubdivision = false;
                updateWindowTitle(window);
                break;
            case GLFW_KEY_LEFT:
                if (currentSubdivisionLevel > 0) {
                    currentSubdivisionLevel--;
                    updateWindowTitle(window);
                }
                break;
            case GLFW_KEY_RIGHT:
                if (currentSubdivisionLevel < maxSubdivisionLevel) {
                    currentSubdivisionLevel++;
                    updateWindowTitle(window);
                }
                break;
            case GLFW_KEY_EQUAL:
            case GLFW_KEY_KP_ADD:
                animationSpeed = std::min(animationSpeed * 1.5f, 10.0f);
                break;
            case GLFW_KEY_MINUS:
            case GLFW_KEY_KP_SUBTRACT:
                animationSpeed = std::max(animationSpeed / 1.5f, 0.1f);
                break;
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, true);
                break;
        }
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action,
                           [[maybe_unused]] int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            mousePressed = true;
            glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
        } else if (action == GLFW_RELEASE) {
            mousePressed = false;
        }
    }
}

void cursor_position_callback([[maybe_unused]] GLFWwindow* window, double xpos,
                              double ypos) {
    if (mousePressed) {
        float deltaX = xpos - lastMouseX;
        float deltaY = ypos - lastMouseY;

        cameraAngleX += deltaX * 0.5f;
        cameraAngleY -= deltaY * 0.5f;

        if (cameraAngleY > 89.0f) cameraAngleY = 89.0f;
        if (cameraAngleY < -89.0f) cameraAngleY = -89.0f;

        lastMouseX = xpos;
        lastMouseY = ypos;
    }
}

void scroll_callback([[maybe_unused]] GLFWwindow* window,
                     [[maybe_unused]] double xoffset, double yoffset) {
    cameraDistance -= yoffset * 0.5f;
    if (cameraDistance < 1.0f) cameraDistance = 1.0f;
    if (cameraDistance > 20.0f) cameraDistance = 20.0f;
}

unsigned int createShaderProgram(const char* vertexSource,
                                 const char* fragmentSource) {
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSource, nullptr);
    glCompileShader(vertexShader);

    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        std::cerr << "Vertex shader compilation failed:\n"
                  << infoLog << std::endl;
    }

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSource, nullptr);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        std::cerr << "Fragment shader compilation failed:\n"
                  << infoLog << std::endl;
    }

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cerr << "Shader program linking failed:\n" << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(
        1280, 960, "3D Octree Compression Visualizer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Create shader programs
    unsigned int pointShaderProgram =
        createShaderProgram(vertexShaderSource, fragmentShaderSource);
    unsigned int boxShaderProgram =
        createShaderProgram(boxVertexShaderSource, boxFragmentShaderSource);

    // Load and compress the model
    ModelManager modelManager;
    octreeVisualizer = std::make_unique<OctreeVisualizer>();

    try {
        std::cout << "Loading model..." << std::endl;
        originalModel = modelManager.loadModel("models/bunny.obj");

        if (!originalModel || originalModel->vertices.empty()) {
            std::cerr << "Failed to load model!" << std::endl;
            return -1;
        }

        std::cout << "Compressing model..." << std::endl;
        compressedModel = modelManager.compressModel(*originalModel);

        if (compressedModel) {
            // Get the octree from the compressed model
            auto* octree = compressedModel->getOctree();
            if (octree) {
                maxSubdivisionLevel = octree->getActualMaxDepth();
                std::cout << "Max subdivision level: " << maxSubdivisionLevel
                          << std::endl;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    // Create VAO for point cloud
    unsigned int pointVAO, pointVBO;
    glGenVertexArrays(1, &pointVAO);
    glGenBuffers(1, &pointVBO);

    std::vector<float> vertices;
    vertices.reserve(originalModel->vertices.size() * 6);
    for (size_t i = 0; i < originalModel->vertices.size(); ++i) {
        vertices.push_back(originalModel->vertices[i].x);
        vertices.push_back(originalModel->vertices[i].y);
        vertices.push_back(originalModel->vertices[i].z);
        vertices.push_back(originalModel->colors[i].x);
        vertices.push_back(originalModel->colors[i].y);
        vertices.push_back(originalModel->colors[i].z);
    }

    glBindVertexArray(pointVAO);
    glBindBuffer(GL_ARRAY_BUFFER, pointVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float),
                 vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                          (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                          (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Create VAOs for octree visualization
    unsigned int wireframeVAO, wireframeVBO;
    unsigned int solidVAO, solidVBO;

    glGenVertexArrays(1, &wireframeVAO);
    glGenBuffers(1, &wireframeVBO);
    glGenVertexArrays(1, &solidVAO);
    glGenBuffers(1, &solidVBO);

    updateWindowTitle(window);
    lastAnimationUpdate = std::chrono::steady_clock::now();

    std::cout << "\nControls:" << std::endl;
    std::cout << "  O - Toggle octree visualization" << std::endl;
    std::cout << "  B - Toggle boundaries" << std::endl;
    std::cout << "  P - Toggle points" << std::endl;
    std::cout << "  L - Toggle current level only / all levels"
              << std::endl;  // New
    std::cout << "  Space - Animate subdivision" << std::endl;
    std::cout << "  R - Reset to level 0" << std::endl;
    std::cout << "  Left/Right - Manual level control" << std::endl;
    std::cout << "  +/- - Change animation speed" << std::endl;
    std::cout << "  Mouse - Rotate view" << std::endl;
    std::cout << "  Scroll - Zoom" << std::endl;

    while (!glfwWindowShouldClose(window)) {
        // Update animation
        if (animateSubdivision) {
            auto now = std::chrono::steady_clock::now();
            float deltaTime =
                std::chrono::duration<float>(now - lastAnimationUpdate).count();

            if (deltaTime >= 1.0f / animationSpeed) {
                currentSubdivisionLevel++;
                if (currentSubdivisionLevel > maxSubdivisionLevel) {
                    currentSubdivisionLevel = 0;
                }
                updateWindowTitle(window);
                lastAnimationUpdate = now;
            }
        }

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Setup view and projection matrices
        glm::mat4 model = glm::mat4(1.0f);

        float camX = sin(glm::radians(cameraAngleX)) *
                     cos(glm::radians(cameraAngleY)) * cameraDistance;
        float camY = sin(glm::radians(cameraAngleY)) * cameraDistance;
        float camZ = cos(glm::radians(cameraAngleX)) *
                     cos(glm::radians(cameraAngleY)) * cameraDistance;

        glm::vec3 cameraPos = glm::vec3(camX, camY, camZ);
        glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);
        glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

        glm::mat4 view = glm::lookAt(cameraPos, cameraTarget, cameraUp);

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glm::mat4 projection = glm::perspective(
            glm::radians(45.0f), (float)width / height, 0.1f, 100.0f);

        // Render octree visualization
        if (showOctree && compressedModel) {
            // Extract bounding boxes up to current level
            auto boxes = octreeVisualizer->extractBoundingBoxes(
                compressedModel->getOctree(), currentSubdivisionLevel);

            // Filter boxes based on showOnlyCurrentLevel setting
            std::vector<OctreeVisualizer::BoundingBox> filteredBoxes;
            if (showOnlyCurrentLevel) {
                // Only show boxes at the current level
                for (const auto& box : boxes) {
                    if (box.level == currentSubdivisionLevel) {
                        filteredBoxes.push_back(box);
                    }
                }
            } else {
                // Show all boxes up to current level
                filteredBoxes = boxes;
            }

            // Render solid boxes
            glUseProgram(boxShaderProgram);

            unsigned int modelLoc =
                glGetUniformLocation(boxShaderProgram, "model");
            unsigned int viewLoc =
                glGetUniformLocation(boxShaderProgram, "view");
            unsigned int projLoc =
                glGetUniformLocation(boxShaderProgram, "projection");
            unsigned int colorLoc =
                glGetUniformLocation(boxShaderProgram, "boxColor");
            unsigned int alphaLoc =
                glGetUniformLocation(boxShaderProgram, "alpha");

            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
            glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
            glUniformMatrix4fv(projLoc, 1, GL_FALSE,
                               glm::value_ptr(projection));

            // Render filled boxes with transparency
            glDepthMask(GL_FALSE);  // Don't write to depth buffer for
                                    // transparent objects

            for (const auto& box : filteredBoxes) {
                if (box.hasData) {
                    // Calculate color based on level (deeper = redder)
                    float levelRatio = (float)box.level / maxSubdivisionLevel;
                    glm::vec4 boxColor;

                    if (showOnlyCurrentLevel) {
                        // Use a single color for current level
                        boxColor = glm::vec4(0.2f, 0.6f, 1.0f, 1.0f);  // Blue
                    } else {
                        // Gradient from blue to red based on depth
                        boxColor = glm::vec4(1.0f - levelRatio, 0.2f,
                                             levelRatio, 1.0f);
                    }

                    glUniform4fv(colorLoc, 1, glm::value_ptr(boxColor));
                    glUniform1f(alphaLoc, 0.3f);

                    // Get vertices for this box
                    auto boxVertices =
                        octreeVisualizer->getSolidBoxVertices(box);

                    glBindVertexArray(solidVAO);
                    glBindBuffer(GL_ARRAY_BUFFER, solidVBO);
                    glBufferData(GL_ARRAY_BUFFER,
                                 boxVertices.size() * sizeof(glm::vec3),
                                 boxVertices.data(), GL_DYNAMIC_DRAW);
                    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                                          3 * sizeof(float), (void*)0);
                    glEnableVertexAttribArray(0);

                    glDrawArrays(GL_TRIANGLES, 0, boxVertices.size());
                }
            }

            glDepthMask(GL_TRUE);  // Re-enable depth writing

            // Render wireframe boundaries
            if (showBoundaries) {
                glUniform4f(colorLoc, 1.0f, 1.0f, 1.0f, 1.0f);
                glUniform1f(alphaLoc, 1.0f);
                glLineWidth(1.0f);

                for (const auto& box : filteredBoxes) {
                    auto wireVertices =
                        octreeVisualizer->getWireframeVertices(box);

                    glBindVertexArray(wireframeVAO);
                    glBindBuffer(GL_ARRAY_BUFFER, wireframeVBO);
                    glBufferData(GL_ARRAY_BUFFER,
                                 wireVertices.size() * sizeof(glm::vec3),
                                 wireVertices.data(), GL_DYNAMIC_DRAW);
                    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                                          3 * sizeof(float), (void*)0);
                    glEnableVertexAttribArray(0);

                    glDrawArrays(GL_LINES, 0, wireVertices.size());
                }
            }
        }

        // Render point cloud
        if (showPoints) {
            glUseProgram(pointShaderProgram);

            unsigned int modelLoc =
                glGetUniformLocation(pointShaderProgram, "model");
            unsigned int viewLoc =
                glGetUniformLocation(pointShaderProgram, "view");
            unsigned int projLoc =
                glGetUniformLocation(pointShaderProgram, "projection");

            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
            glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
            glUniformMatrix4fv(projLoc, 1, GL_FALSE,
                               glm::value_ptr(projection));

            glBindVertexArray(pointVAO);
            glDrawArrays(GL_POINTS, 0, originalModel->vertices.size());
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    glDeleteVertexArrays(1, &pointVAO);
    glDeleteBuffers(1, &pointVBO);
    glDeleteVertexArrays(1, &wireframeVAO);
    glDeleteBuffers(1, &wireframeVBO);
    glDeleteVertexArrays(1, &solidVAO);
    glDeleteBuffers(1, &solidVBO);
    glDeleteProgram(pointShaderProgram);
    glDeleteProgram(boxShaderProgram);

    glfwTerminate();

    return 0;
}
