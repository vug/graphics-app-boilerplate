#include "Workshop.hpp"

#include "Workshop/Assets.hpp" // Only Assets.hpp needs to be included via Workshop/
#include "OpenGL.hpp"
#include "Shader.hpp"

#include <glad/gl.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>

#include <ranges>
namespace rng = std::ranges;
namespace vw = std::views;
#include <signal.h>
#include <iostream>

namespace ws {

static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
static inline const char* glMessageSourceToString(GLenum source);
static inline const char* glMessageTypeToString(GLenum type);
static inline const char* glMessageSeverityToString(GLenum severity);
void GLAPIENTRY OpenGLDebugMessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const char* message, const void* userParam);

Workshop::Workshop(int width, int height, const std::string& name) {
  if (!glfwInit())
    throw("Failed to initialize GLFW!");

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, openGLMajorVersion);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, openGLMinorVersion);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  if (shouldDebugOpenGL)
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
  if (useMSAA)
    glfwWindowHint(GLFW_SAMPLES, 8);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE); // default
  glfwWindowHint(GLFW_STENCIL_BITS, 8);  // default

  window = glfwCreateWindow(width, height, name.c_str(), nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    throw("Failed to create window!");
  }
  glfwMakeContextCurrent(window);
  glfwSetWindowUserPointer(window, this);
  if (!shouldVSync)
    glfwSwapInterval(0);
  glfwSetKeyCallback(window, keyCallback);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);
  glfwSetCursorPosCallback(window, mousePositionCallback);
  glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

  const int version = gladLoadGL(glfwGetProcAddress);
  if (version == 0)
    throw("Failed to initialize OpenGL context");

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  // Multi-viewport is not working on Linux. See:
  // The imgui window can't be moved outside the desktop region · Issue #3899 · ocornut/imgui · GitHub https://github.com/ocornut/imgui/issues/3899
#ifdef _WIN32
  io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
#endif
  ImGui::StyleColorsDark();
  // When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
  ImGuiStyle& style = ImGui::GetStyle();
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
    style.WindowRounding = 0.0f;
    style.Colors[ImGuiCol_WindowBg].w = 1.0f;
  }
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glslVersion);
  ImPlot::CreateContext();

  glViewport(0, 0, width, height);
  glEnable(GL_MULTISAMPLE);
  if (shouldDebugOpenGL) {
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(OpenGLDebugMessageCallback, this);
    // Ignore notifications
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, NULL, GL_FALSE);
  }

  Shader::makeNamedStringsFromFolder(ASSETS_FOLDER / "shaders/lib");
}

Workshop::~Workshop() {
  ImPlot::DestroyContext();
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwTerminate();
}

bool Workshop::shouldStop() {
  return glfwWindowShouldClose(window);
}

void Workshop::beginFrame() {
  if (frameNo == 0) time = static_cast<float>(glfwGetTime()); // Just to keep first frame duration low
  ++frameNo;
  const float timeLastFrame = time;
  time = static_cast<float>(glfwGetTime());
  frameDurationSec = time - timeLastFrame;
  frameRate = frameDurationSec == 0.f ? 0 : 1.f / frameDurationSec;

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
}

void Workshop::endFrame() {
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  ImGuiIO& io = ImGui::GetIO();
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
    GLFWwindow* backup_current_context = glfwGetCurrentContext();
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
    glfwMakeContextCurrent(backup_current_context);
  }

  glfwSwapBuffers(window);
  glfwPollEvents();
}

glm::uvec2 Workshop::getWindowSize() const {
  int x, y;
  glfwGetWindowSize(window, &x, &y);
  return {x, y};
}

uint32_t Workshop::getFrameNo() const {
  return frameNo;
}

float Workshop::getFrameDurationMs() const {
  return frameDurationSec * 1000.f;
}

float Workshop::getFrameDurationSec() const {
  return frameDurationSec;
}

float Workshop::getFrameRate() const {
  return frameRate;
}

bool Workshop::getVSync() const {
  return shouldVSync;
}

void Workshop::setVSync(bool should) {
  shouldVSync = should;
  glfwSwapInterval(shouldVSync ? 1 : 0);
}

void Workshop::drawUI() {
  ImGui::Begin("Workshop");
  ImGui::Text("Frame No: %6d, Frame Dur: %.2f, FPS: %.1f", getFrameNo(), getFrameDurationMs(), getFrameRate());
  bool vSync = getVSync();
  if (ImGui::Checkbox("VSync", &vSync))
    setVSync(vSync);
  if (ImGui::Button("OpenGL Features"))
    printFeatures();
  static bool shouldShowImGuiDemo = false;
  ImGui::Checkbox("Show ImGui Demo?", &shouldShowImGuiDemo);
  static int shaderToReloadIx = 0;
  if (!shadersToReload.empty()) {
    auto items = shadersToReload 
      | vw::transform([](auto& s) { return std::format("Shader[{}]", s.get().getId()); })
      | rng::to<std::vector<std::string>>();
    auto items2 = items 
      | vw::transform([](auto& s) { return s.c_str(); })
      | rng::to<std::vector<const char*>>();
    ImGui::Combo("Shader", &shaderToReloadIx, items2.data(), static_cast<int32_t>(items2.size()));
    if (ImGui::Button(std::format("Reload {}", items[shaderToReloadIx]).c_str()))
      shadersToReload[shaderToReloadIx].get().reload();
    ImGui::SameLine();
    if (ImGui::Button("Attributes"))
      shadersToReload[shaderToReloadIx].get().printAttributes();
    ImGui::SameLine();
    if (ImGui::Button("Uniforms"))
      shadersToReload[shaderToReloadIx].get().printUniforms();
    ImGui::SameLine();
    if (ImGui::Button("UBOs"))
      shadersToReload[shaderToReloadIx].get().printUniformBlocks();
    ImGui::SameLine();
    if (ImGui::Button("Source"))
      shadersToReload[shaderToReloadIx].get().printSource();
      
  }
  ImGui::Separator();
  ImGui::End();

  if (shouldShowImGuiDemo)
    ImGui::ShowDemoWindow();
}

void framebufferSizeCallback([[maybe_unused]] GLFWwindow* window, [[maybe_unused]] int width, [[maybe_unused]] int height) {
  // glViewport(0, 0, width, height);
  void* ptr = glfwGetWindowUserPointer(window);
  if (ptr) {
    // Workshop* ws = static_cast<Workshop*>(ptr);
    // ws->width = width;
    // ws->height = height;
  }
}

inline const char* glMessageSourceToString(GLenum source) {
  switch (source) {
    case GL_DEBUG_SOURCE_API:
      return "OpenGL API";
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
      return "Window System";
    case GL_DEBUG_SOURCE_SHADER_COMPILER:
      return "Shader compiler";
    case GL_DEBUG_SOURCE_THIRD_PARTY:
      return "Third-party app associated with OpenGL";
    case GL_DEBUG_SOURCE_APPLICATION:
      return "The user of this application";
    case GL_DEBUG_SOURCE_OTHER:
      return "Unspecified";
    default:
      assert(false);  // unknown source
      return "Unknown";
  }
}

inline const char* glMessageTypeToString(GLenum type) {
  switch (type) {
    case GL_DEBUG_TYPE_ERROR:
      return "Error";
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
      return "Deprecated behavior";
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
      return "Undefined behavior";
    case GL_DEBUG_TYPE_PORTABILITY:
      return "Unportable functionality";
    case GL_DEBUG_TYPE_PERFORMANCE:
      return "Performance issue";
    case GL_DEBUG_TYPE_MARKER:
      return "Command stream annotation";
    case GL_DEBUG_TYPE_PUSH_GROUP:
      return "Group pushing";
    case GL_DEBUG_TYPE_POP_GROUP:
      return "Group popping";
    case GL_DEBUG_TYPE_OTHER:
      return "Unspecified";
    default:
      assert(false);  // unknown type
      return "Unknown";
  }
}

inline const char* glMessageSeverityToString(GLenum severity) {
  switch (severity) {
    case GL_DEBUG_SEVERITY_HIGH:
      return "High";
    case GL_DEBUG_SEVERITY_MEDIUM:
      return "Medium";
    case GL_DEBUG_SEVERITY_LOW:
      return "Low";
    case GL_DEBUG_SEVERITY_NOTIFICATION:
      return "Notification";
    default:
      assert(false);  // unknown type
      return "Unknown";
  }
}

void GLAPIENTRY OpenGLDebugMessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, [[maybe_unused]] GLsizei length, const char* message, const void* userParam) {
  // filter out non-significant error/warning codes
  if (
      id == 131218 // Program/shader state performance warning: Vertex shader in program 9 is being recompiled based on GL state.
      // || id == 131169    //
      // || id == 131185 // Buffer object 2 (bound to GL_ELEMENT_ARRAY_BUFFER_ARB, usage hint is GL_STATIC_DRAW) will use VIDEO memory as the source for buffer object operations
      // || id == 131204 //
      // || id == 2      // INFO: API_ID_RECOMPILE_FRAGMENT_SHADER performance warning has been generated. Fragment shader recompiled due to state change. [ID: 2]
  )
    return;

  std::cout << "OpenGL Debug Message [" << id << "]"
            << ". Severity : " << glMessageSeverityToString(severity)
            << ". Source: " << glMessageSourceToString(source)
            << ". Type: " << glMessageTypeToString(type)
            << ". Message: " << message << ".\n";

  Workshop* thisWorkshop = (Workshop*)userParam;
  if (thisWorkshop->shouldBreakAtOpenGLDebugCallback()) {
#ifdef _MSC_VER
    __debugbreak();
#elif defined(SIGTRAP)
    raise(SIGTRAP);
#endif
  }
}
}  // namespace ws