#pragma once

#include "../Camera.hpp"
#include "../Framebuffer.hpp"
#include "../Scene.hpp"
#include "../Shader.hpp"

#include <imgui.h>

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

class EditorWindow {
 public:
  EditorWindow(Scene& scene);
  VObjectPtr draw();

 private:
  Camera cam;
  Framebuffer fbo;
  Scene& scene;
  Shader editorShader;
  Shader solidColorShader;
  Shader outlineShader;
  Shader gridShader;
  uint32_t gridVao;
};
}