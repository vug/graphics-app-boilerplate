#pragma once

#include "Framebuffer.hpp"
#include "Texture.hpp"
#include "Scene.hpp"
#include "Shader.hpp"

#include <imgui.h>

#include <vector>

namespace ws {


void ImGuiBeginMouseDragHelper(const char* name, ImVec2 size);
// true only at the first frame of drag (overlaps with isDragging)
bool ImGuiMouseDragHelperIsBeginningDrag();
// true when dragging (overlaps with isBeginningDrag)
bool ImGuiMouseDragHelperIsDragging();
// true one frame after dragging done
// Note that ImGui::GetIO().MouseClickedPos[ImGuiMousebutton_XYZ] has the 
bool ImGuiMouseDragHelperIsEndingDrag();
void ImGuiEndMouseDragHelper();

class TextureViewer {
 public:
  TextureViewer(const std::vector<std::reference_wrapper<ws::Texture>>& textures);
  void draw();

 private:
  std::vector<std::reference_wrapper<ws::Texture>> textures;
  int ix{};
};

class HierarchyWindow {
 public:
  HierarchyWindow(Scene& scene);
  VObjectPtr draw();

 private:
  Scene& scene;
};

class InspectorWindow {
 public:
  void inspectObject(VObjectPtr objPtr);
};

class EditorWindow {
 public:
  EditorWindow(Scene& scene);
	VObjectPtr draw();

 private:
	Camera cam;
  Framebuffer fbo;
  Scene& scene;
  Shader editorShader;
  Shader gridShader;
  uint32_t gridVao;
};
}