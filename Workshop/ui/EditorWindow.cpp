#include "ui/EditorWindow.hpp"

#include <Workshop/Assets.hpp>

#include <glm/gtx/rotate_vector.hpp> // for glm::rotate overload for rotating a vector around an axis by an angle

#include <array>
#include <ranges>
#include <variant>

namespace ws {
static bool imguiMouseDragHelperHasBegun = false;

void ImGuiBeginMouseDragHelper(const char* name, ImVec2 size) {
  assert(!imguiMouseDragHelperHasBegun);  // cannot put one into another
  ImVec2 tmpCursor = ImGui::GetCursorPos();
  ImGui::InvisibleButton(name, size, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle | ImGuiButtonFlags_MouseButtonRight);  // ImGuiButtonFlags_AllowItemOverlap needed?
  ImGui::SetCursorPos(tmpCursor);
  imguiMouseDragHelperHasBegun = true;
}

bool ImGuiMouseDragHelperIsBeginningDrag() {
  assert(imguiMouseDragHelperHasBegun);
  return ImGui::IsItemActivated();
}

bool ImGuiMouseDragHelperIsDragging() {
  assert(imguiMouseDragHelperHasBegun);
  return ImGui::IsItemActive();
}

bool ImGuiMouseDragHelperIsEndingDrag() {
  assert(imguiMouseDragHelperHasBegun);
  return ImGui::IsItemDeactivated();
}

void ImGuiEndMouseDragHelper() {
  assert(imguiMouseDragHelperHasBegun);
  imguiMouseDragHelperHasBegun = false;
}

EditorWindow::EditorWindow(Scene& scene)
    : fbo(
        std::vector<Texture::Specs>{
          Texture::Specs{1, 1, Texture::Format::RGBA8, Texture::Filter::Nearest, Texture::Wrap::ClampToBorder}, // Editor Scene Viz
          Texture::Specs{1, 1, Texture::Format::R32i, Texture::Filter::Nearest, Texture::Wrap::ClampToBorder}, // Mesh Ids
        },
        Texture::Specs{1, 1, Texture::Format::Depth32fStencil8, Texture::Filter::Linear, Texture::Wrap::ClampToBorder}
      ),
      outlineSolidFbo(Framebuffer::makeDefaultColorOnly(1, 1)),
      outlineGrowthFbo(Framebuffer::makeDefaultColorOnly(1, 1)),
      scene(scene),
      editorShader(ws::ASSETS_FOLDER / "shaders/editor.vert", ws::ASSETS_FOLDER / "shaders/editor.frag"),
      solidColorShader(ws::ASSETS_FOLDER / "shaders/solid_color.vert", ws::ASSETS_FOLDER / "shaders/solid_color.frag"),
      outlineShader(ws::ASSETS_FOLDER / "shaders/fullscreen_quad_without_vbo.vert", ws::ASSETS_FOLDER / "shaders/fullscreen_quad_outline.frag"),
      copyShader(ws::ASSETS_FOLDER / "shaders/fullscreen_quad_without_vbo.vert", ws::ASSETS_FOLDER / "shaders/fullscreen_quad_texture_sampler.frag"),
      gridShader(ws::ASSETS_FOLDER / "shaders/infinite_grid.vert", ws::ASSETS_FOLDER / "shaders/infinite_grid.frag"),
      gridVao([]() { uint32_t id;glGenVertexArrays(1, &id); return id; }()), 
      emptyVao([]() { uint32_t id;glGenVertexArrays(1, &id); return id; }()) 
  {}

VObjectPtr EditorWindow::draw(VObjectPtr selectedObject, float deltaTimeSec) {
  ImGui::Begin("Editor");

  static bool shouldBeWireframe = false;
  ImGui::Checkbox("Wireframe", &shouldBeWireframe);

  ImGui::SameLine();
  static int shadingModel = 2;
  std::array<const char*, 13> items = {"Pos (Obj)", "Pos (World)", "UV1", "UV2", "Normal (Obj)", "Normal (World)", "Front-Back Faces", "Texture1 (UV1)", "Texture2 (UV2)", "Depth (Ortho)", "Depth (Proj)", "Mesh IDs", "Hemispherical Lighting"};
  ImGui::Combo("Shading Model", &shadingModel, items.data(), static_cast<int>(items.size()));

  ImGui::SameLine();
  static std::string hoveredObjectName = "None";  // static because it'll be set after editor image is drawn
  ImGui::Text("Hovered: %s", hoveredObjectName.c_str());

  ImGui::SameLine();
  static bool showGrid = true;
  ImGui::Checkbox("Grid", &showGrid);

  ImVec2 size = ImGui::GetContentRegionAvail();
  if (size.y < 0) {  // happens when minimized
    ImGui::End();
    return selectedObject;
  }
  glm::ivec2 sizei{size.x, size.y};
  fbo.resizeIfNeeded(sizei.x, sizei.y);
  outlineSolidFbo.resizeIfNeeded(sizei.x, sizei.y);
  outlineGrowthFbo.resizeIfNeeded(sizei.x, sizei.y);

  ImGuiBeginMouseDragHelper("EditorDragDetector", size);
  static Camera cam0;
  static glm::vec3 deltaPos;
  if (ImGuiMouseDragHelperIsBeginningDrag()) {
    cam0 = cam;
    deltaPos = {};
  }
  if (ImGuiMouseDragHelperIsDragging()) {
    const ImVec2 deltaLeft = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0);
    const ImVec2 deltaMiddle = ImGui::GetMouseDragDelta(ImGuiMouseButton_Middle, 0);
    const ImVec2 deltaRight = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right, 0);
    const float mouseSpeed = 0.005f;
    float keySpeed = 8.0f;

    if (ImGui::IsKeyDown(ImGuiKey_LeftShift))
      keySpeed *= 5;

    if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
      if (ImGui::IsKeyDown(ImGuiKey_A))
        deltaPos -= cam.getRight() * keySpeed * deltaTimeSec;
      if (ImGui::IsKeyDown(ImGuiKey_D))
        deltaPos += cam.getRight() * keySpeed * deltaTimeSec;
      if (ImGui::IsKeyDown(ImGuiKey_W))
        deltaPos += cam.getForward() * keySpeed * deltaTimeSec;
      if (ImGui::IsKeyDown(ImGuiKey_S))
        deltaPos -= cam.getForward() * keySpeed * deltaTimeSec;
      if (ImGui::IsKeyDown(ImGuiKey_Q))
        deltaPos -= cam.getUp() * keySpeed * deltaTimeSec;
      if (ImGui::IsKeyDown(ImGuiKey_E))
        deltaPos += cam.getUp() * keySpeed * deltaTimeSec;

      // look around
      if (ImGui::IsKeyDown(ImGuiKey_LeftAlt)) {
        deltaPos = cam.getForward() * deltaRight.y * mouseSpeed;
      }
      // zoom-in-out
      else {
        const glm::vec3 oldPosToOldTarget = cam0.target - cam0.position;
        const float deltaPitch = -deltaRight.y * mouseSpeed;
        glm::vec3 oldPosToNewTarget = glm::rotate(oldPosToOldTarget, deltaPitch, cam0.getRight());
        const float deltaYaw = -deltaRight.x * mouseSpeed;
        oldPosToNewTarget = glm::rotate(oldPosToNewTarget, deltaYaw, glm::vec3{0, 1, 0});
        cam.target = cam0.position + oldPosToNewTarget + deltaPos;
      }

      cam.position = cam0.position + deltaPos;
    }
    // pan around
    else if (ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
      const glm::vec3 deltaPan = (cam0.getRight() * -deltaMiddle.x - cam0.getUp() * -deltaMiddle.y) * mouseSpeed;
      cam.position = cam0.position + deltaPan;
      cam.target = cam0.target + deltaPan;
    }
    // orbit around
    else if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && ImGui::IsKeyDown(ImGuiKey_LeftAlt)) {
      const glm::vec3 oldTargetToOldPos = cam0.position - cam0.target;
      const float deltaPitch = -deltaLeft.y * mouseSpeed;
      glm::vec3 oldTargetToNewPos = glm::rotate(oldTargetToOldPos, deltaPitch, cam0.getRight());
      const float deltaYaw = -deltaLeft.x * mouseSpeed;
      oldTargetToNewPos = glm::rotate(oldTargetToNewPos, deltaYaw, glm::vec3{0, 1, 0});
      cam.position = cam0.target + oldTargetToNewPos;
    }
  }
  ImGuiEndMouseDragHelper();

  cam.aspectRatio = size.x / size.y;

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  fbo.bind();
  // When drawing the scene render into first attachment, and put mesh ids into second attachment
  uint32_t allAttachments[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
  glDrawBuffers(2, allAttachments);
  glViewport(0, 0, sizei.x, sizei.y);
  glDisable(GL_BLEND);
  glEnable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
  fbo.getColorAttachments()[0].clear(glm::vec4(0, 0, 0, 1));
  fbo.getColorAttachments()[1].clear(-1);
  glClear(GL_DEPTH_BUFFER_BIT);
  glPolygonMode(GL_FRONT_AND_BACK, shouldBeWireframe ? GL_LINE : GL_FILL);
  for (auto [ix, renderable] : scene.renderables | std::ranges::views::enumerate) {
    editorShader.bind();
    editorShader.setMatrix4("u_WorldFromObject", renderable.get().transform.getWorldFromObjectMatrix());
    editorShader.setMatrix4("u_ViewFromWorld", cam.getViewFromWorld());
    editorShader.setMatrix4("u_ProjectionFromView", cam.getProjectionFromView());
    editorShader.setVector3("u_CameraPosition", cam.position);
    editorShader.setVector2("u_CameraNearFar", glm::vec2{cam.nearClip, cam.farClip});
    editorShader.setInteger("u_ShadingModel", shadingModel);
    editorShader.setInteger("u_MeshId", static_cast<int>(ix));
    const bool isSelected = std::visit([&renderable](auto&& ptr) { if (ptr == nullptr) return false; return ptr->name == renderable.get().name; }, selectedObject);
    editorShader.setInteger("u_ShouldOutline", static_cast<int>(isSelected));
    renderable.get().texture.bindToUnit(0);
    renderable.get().texture2.bindToUnit(1);
    renderable.get().mesh.bind();
    renderable.get().mesh.draw();
    renderable.get().mesh.unbind();
    renderable.get().texture.unbindFromUnit(0);
    renderable.get().texture2.unbindFromUnit(1);
    editorShader.unbind();
  }

  if (showGrid) {
    // when drawing gizmos (such as coordinate grid) only draw into first attachment, don't touch mesh ids
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_CULL_FACE);
    {
      gridShader.bind();
      gridShader.setMatrix4("u_ViewFromWorld", cam.getViewFromWorld());
      gridShader.setMatrix4("u_ProjectionFromView", cam.getProjectionFromView());
      gridShader.setVector3("u_CameraPosition", cam.position);
      glBindVertexArray(gridVao);
      glDrawArrays(GL_TRIANGLES, 0, 6);
      glBindVertexArray(0);
      gridShader.unbind();
    }
  }
  fbo.unbind();

  // Pass 2: Draw highlighted objects with solid color offscreen
  outlineSolidFbo.bind();
  glClearColor(0, 0, 0, 0);
  glClear(GL_COLOR_BUFFER_BIT);
  glDisable(GL_DEPTH_TEST);
  const bool hasSelectedObject = std::visit([](auto&& ptr) { return ptr != nullptr; }, selectedObject) && std::holds_alternative<RenderableObject*>(selectedObject);
  if (hasSelectedObject) {
    RenderableObject* ptr = std::get<RenderableObject*>(selectedObject);
    solidColorShader.bind();
    solidColorShader.setMatrix4("u_WorldFromObject", ptr->transform.getWorldFromObjectMatrix());
    solidColorShader.setMatrix4("u_ViewFromWorld", cam.getViewFromWorld());
    solidColorShader.setMatrix4("u_ProjectionFromView", cam.getProjectionFromView());
    const glm::vec4 outlineColor{1, 1, 0, 1};
    solidColorShader.setVector4("u_Color", outlineColor);
    ptr->mesh.bind();
    ptr->mesh.draw();
    ptr->mesh.unbind();
    solidColorShader.unbind();
  }
  outlineSolidFbo.unbind();

  // Pass 3: Out-grow highlight solid color area
  outlineGrowthFbo.bind();
  glClearColor(0, 0, 0, 0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glDisable(GL_DEPTH_TEST);
  outlineShader.bind();
  glBindVertexArray(emptyVao);
  outlineSolidFbo.getFirstColorAttachment().bind();
  glDrawArrays(GL_TRIANGLES, 0, 6);
  Texture::unbind();
  glBindVertexArray(0);
  outlineShader.unbind();
  glEnable(GL_DEPTH_TEST);
  outlineGrowthFbo.unbind();

  // Pass 4: Draw highlights as overlay to screen
  fbo.bind();
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glDrawBuffer(GL_COLOR_ATTACHMENT0);
  glDisable(GL_DEPTH_TEST);
  copyShader.bind();
  outlineGrowthFbo.getFirstColorAttachment().bind();
  glBindVertexArray(emptyVao);
  glDrawArrays(GL_TRIANGLES, 0, 6);
  glBindVertexArray(0);
  Texture::unbind();
  copyShader.unbind();
  glEnable(GL_DEPTH_TEST);
  fbo.unbind();

  // IsItemActivated checks whether InvisibleButton was clicked. IsItemClicked on the Image below didn't work for some reason. Probably because it's overlapping with the button.
  bool wasViewportClicked = ImGui::IsItemActivated() && ImGui::IsMouseDown(ImGuiMouseButton_Left);

  ImVec2 uv0 = {0, 0};
  ImVec2 uv1 = {1, 1};
  // flip the texture upside-down via following uv-coordinate transformation: (0, 0), (1, 1) -> (0, 1), (1, 0)
  std::swap(uv0.y, uv1.y);
  ImVec2 imagePosToWin = ImGui::GetCursorPos();
  ImGui::Image((void*)(intptr_t)fbo.getFirstColorAttachment().getId(), size, uv0, uv1, {1, 1, 1, 1}, {1, 1, 0, 1});

  GLint hoveredMeshId{-2};  // -2 means not touched, -1 means no mesh under cursor
  // when clicked on Image is stop being hovered :-O for that case added wasViewportClicked
  if (ImGui::IsItemHovered() || wasViewportClicked) {
    fbo.bind();
    const ImVec2 mousePos = ImGui::GetMousePos();
    const ImVec2 winPos = ImGui::GetWindowPos();
    const ImVec2 pixCoord = ImVec2(mousePos.x - (winPos.x + imagePosToWin.x), mousePos.y - (winPos.y + imagePosToWin.y));

    glReadBuffer(GL_COLOR_ATTACHMENT1);
    const Texture& tex = fbo.getColorAttachments()[1];
    glReadPixels(static_cast<int32_t>(pixCoord.x), static_cast<int32_t>(tex.specs.height - pixCoord.y), 1, 1, GL_RED_INTEGER, GL_INT, &hoveredMeshId);
    hoveredObjectName = hoveredMeshId >= 0 ? scene.renderables[hoveredMeshId].get().name : "None";
    // std::println("hoveredMeshId {}, selectedName {}, pixCoord ({:.1f}, {:.1f})", hoveredMeshId, hoveredObjectName, pixCoord.x, pixCoord.y);
    fbo.unbind();
  }

  VObjectPtr clickedObject{};
  if (wasViewportClicked && !ImGui::IsKeyDown(ImGuiKey_LeftAlt)) {
    if (hoveredMeshId >= 0)
      clickedObject = &scene.renderables[hoveredMeshId].get();
    else if (hoveredMeshId == -1)
      clickedObject = &scene.root;
  }

  ImGui::End();

  return clickedObject;
}
}