#include "Workshop.h"

#include <glad/gl.h>

#include <iostream>

namespace ws {
Workshop::Workshop() {
  if (!glfwInit())
    throw("Failed to initialize GLFW!");

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE);
  window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
  if (!window) {
    glfwTerminate();
    throw("Failed to create window!");
  }

  glfwMakeContextCurrent(window);

  const int version = gladLoadGL(glfwGetProcAddress);
  if (version == 0)
    throw("Failed to initialize OpenGL context");

  printf("Loaded OpenGL %d.%d\n", GLAD_VERSION_MAJOR(version), GLAD_VERSION_MINOR(version));
}

Workshop::~Workshop() {
  glfwTerminate();
}

bool Workshop::shouldStop() {
  return glfwWindowShouldClose(window);
}

void Workshop::beginFrame() {
}

void Workshop::endFrame() {
  glfwSwapBuffers(window);
  glfwPollEvents();
}
}  // namespace ws