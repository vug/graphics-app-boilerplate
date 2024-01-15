#include <Workshop/Assets.hpp>
#include "UI.hpp"

#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/rotate_vector.hpp> // for glm::rotate overload for rotating a vector around an axis by an angle
#include <imgui/imgui.h>
#include <imgui_internal.h> // for PushMultiItemsWidths, GImGui

#include <algorithm>
#include <array>
#include <print>
#include <ranges>

namespace ws {

static bool imguiMouseDragHelperHasBegun = false;

void ImGuiBeginMouseDragHelper(const char* name, ImVec2 size) {
	assert(!imguiMouseDragHelperHasBegun); // cannot put one into another
	ImVec2 tmpCursor = ImGui::GetCursorPos();
	ImGui::InvisibleButton(name, size, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle | ImGuiButtonFlags_MouseButtonRight); // ImGuiButtonFlags_AllowItemOverlap needed?
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

TextureViewer::TextureViewer(const std::vector<std::reference_wrapper<ws::Texture>>& textures) 
  : textures{textures} {
}

void TextureViewer::draw() {

  ImGui::GetIO().ConfigWindowsMoveFromTitleBarOnly = true;
  ImGui::Begin("Texture Viewer", nullptr, ImGuiWindowFlags_NoScrollbar);
  auto items = textures 
    | std::views::transform([](const ws::Texture& tex) { return tex.getName().c_str(); }) 
    | std::ranges::to<std::vector<const char*>>();
  ImGui::Combo("Texture", &ix, items.data(), static_cast<uint32_t>(items.size()));
  const auto& tex = textures[ix].get();
  ImGui::Text("Name: %s, dim: (%d, %d)", tex.getName().c_str(), tex.specs.width, tex.specs.height);
  ImGui::Separator();

  const auto letterSize = ImGui::CalcTextSize("R");
  static std::array<bool, 4> channelSelection = {true, true, true, true};
  ImGui::Selectable("R", &channelSelection[0], ImGuiSelectableFlags_None, letterSize);
  ImGui::SameLine();
  ImGui::Selectable("G", &channelSelection[1], ImGuiSelectableFlags_None, letterSize);
  ImGui::SameLine();
  ImGui::Selectable("B", &channelSelection[2], ImGuiSelectableFlags_None, letterSize);
  ImGui::SameLine();
  ImGui::Selectable("A", &channelSelection[3], ImGuiSelectableFlags_None, letterSize);
  const ImVec4 tintColor = {static_cast<float>(channelSelection[0]), static_cast<float>(channelSelection[1]), static_cast<float>(channelSelection[2]), static_cast<float>(channelSelection[3])};

  static float zoomScale = 0.80f;
  ImGui::SliderFloat("scale", &zoomScale, 0.01f, 1.0f);

  const auto availableSize = ImGui::GetContentRegionAvail();
  const float availableAspectRatio = availableSize.x / availableSize.y;
  const float textureAspectRatio = static_cast<float>(tex.specs.width) / tex.specs.height;
  ImVec2 imgSize = (textureAspectRatio >= availableAspectRatio) ? ImVec2{availableSize.x, availableSize.x / textureAspectRatio} : ImVec2{availableSize.y * textureAspectRatio, availableSize.y};

  static ImVec2 uvOffset{0, 0};
  ImVec2 uvExtend{zoomScale, zoomScale};
  const ImVec2 drag = ImGui::GetMouseDragDelta(ImGuiMouseButton_Middle, 0);
  // one pixel of cursor move should pan the zoomed-in texture by one pixel
  ImVec2 deltaOffset{-drag.x / imgSize.x * zoomScale, drag.y / imgSize.y * zoomScale};

  // Only pan if drag operation starts while being hovered over the image
  static bool dragStartedFromImage = false;
  const ImVec2 pMin = ImGui::GetCursorScreenPos();
  const ImVec2 pMax = {pMin.x + imgSize.x, pMin.y + imgSize.y};
  const bool isHoveringOverImage = ImGui::IsMouseHoveringRect(pMin, pMax);
  if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle))
    dragStartedFromImage = isHoveringOverImage;  

  ImVec2 off = uvOffset;
  if (dragStartedFromImage) {
    off.x += deltaOffset.x;
    off.y += deltaOffset.y;
    // only update uvOffset when dragging ends
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Middle)) {
      uvOffset.x += deltaOffset.x;
      uvOffset.y += deltaOffset.y;
      deltaOffset.x = 0;
      deltaOffset.y = 0;
      dragStartedFromImage = false;
    }
  }
  // top-left corner (uv0) cannot be behind (0, 0) and bottom-right corner (uv1) cannot be beyond (1, 1)
  off.x = std::min(std::max(0.f, off.x), 1.f - uvExtend.x);
  off.y = std::min(std::max(0.f, off.y), 1.f - uvExtend.y);
  ImVec2 uv0 = {off.x, off.y};
  ImVec2 uv1 = {off.x + uvExtend.x, off.y + uvExtend.y};
  // flip the texture upside-down via following uv-coordinate transformation: (0, 0), (1, 1) -> (0, 1), (1, 0) 
  std::swap(uv0.y, uv1.y);
  
  ImGui::Image((void*)(intptr_t)tex.getId(), imgSize, uv0, uv1, tintColor, {0.5, 0.5, 0.5, 1.0});
  ImGui::End();
}


HierarchyWindow::HierarchyWindow(Scene& scene)
    : scene(scene) {}

VObjectPtr HierarchyWindow::draw() {
  ImGui::Begin("Hierarchy");

  struct NodeIdentifier {
    std::string name;
    int siblingId;
  };
  static NodeIdentifier selectedNodeId{"__NONE__", -1};
  static VObjectPtr selectedNode{};

  // ChildNo is relative to parent. It is used to give objects with the same name different ImGui ids
  std::function<void(VObjectPtr, int, int)> drawTree = [&](VObjectPtr node, int depth, int siblingId) {
    std::string nodeName = std::visit([](auto&& ptr) { return ptr->name; }, node);
    const auto& children = std::visit([](auto&& ptr) { return ptr->children; }, node);
    const bool hasChildren = !children.empty();
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow;  // | ImGuiTreeNodeFlags_SpanAvailWidth;
    if (!hasChildren)
      flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    if (selectedNodeId.name == nodeName && selectedNodeId.siblingId == siblingId)
      flags |= ImGuiTreeNodeFlags_Selected;

    // TreeNode works is easier to use but puts the arrow to leaves
    const bool isOpen = ImGui::TreeNodeEx((void*)(intptr_t)siblingId, flags, nodeName.c_str());
    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
      selectedNodeId = {nodeName, siblingId};
      selectedNode = node;
    }
    if (isOpen) {
      int childNo = 1;
      for (auto childPtr : children)
        drawTree(childPtr, depth + 1, childNo++);

      if (hasChildren)
        ImGui::TreePop();
    }
  };

  drawTree(&scene.root, 0, 1);

  const bool isEmptyAreaClicked = ImGui::IsMouseClicked(ImGuiMouseButton_Left) && ImGui::IsWindowHovered() && !ImGui::IsAnyItemHovered();
  if (isEmptyAreaClicked) {
    selectedNodeId = {"__NONE__", -1};
    selectedNode = {};
  }

  ImGui::End();

  return selectedNode;
}

// taken from a The Cherno tutorial
static bool DrawVec3Control(const std::string& label, glm::vec3& values, float resetValue = 0.0f, float columnWidth = 100.0f) {
  bool value_changed = false;
  ImGui::PushID(label.c_str());

  ImGui::Columns(2);
  ImGui::SetColumnWidth(0, columnWidth);
  ImGui::Text(label.c_str());
  ImGui::NextColumn();

  ImGui::PushMultiItemsWidths(3, ImGui::CalcItemWidth());
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});

  float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
  ImVec2 buttonSize = {lineHeight + 3.0f, lineHeight};

  // https://coolors.co/ff595e-ffca3a-8ac926-1982c4-6a4c93
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{1.0f, 0.34901961f, 0.36862745f, 1.0f});
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{1.000000f, 0.521569f, 0.537255f, 1.0f});
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{1.000000f, 0.200000f, 0.227451f, 1.0f});

  if (ImGui::Button("X", buttonSize)) {
    values.x = resetValue;
    value_changed = true;
  }
  ImGui::PopStyleColor(3);

  ImGui::SameLine();
  value_changed |= ImGui::DragFloat("##X", &values.x, 0.1f, 0.0f, 0.0f, "%.2f");
  ImGui::PopItemWidth();
  ImGui::SameLine();

  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.541176f, 0.788235f, 0.149020f, 1.0f});
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.631373f, 0.858824f, 0.262745f, 1.0f});
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.462745f, 0.670588f, 0.129412f, 1.0f});
  if (ImGui::Button("Y", buttonSize)) {
    values.y = resetValue;
    value_changed = true;
  }
  ImGui::PopStyleColor(3);

  ImGui::SameLine();
  value_changed |= ImGui::DragFloat("##Y", &values.y, 0.1f, 0.0f, 0.0f, "%.2f");
  ImGui::PopItemWidth();
  ImGui::SameLine();

  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.098039f, 0.509804f, 0.768627f, 1.0f});
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.149020f, 0.607843f, 0.890196f, 1.0f});
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.082353f, 0.423529f, 0.635294f, 1.0f});
  if (ImGui::Button("Z", buttonSize)) {
    values.z = resetValue;
    value_changed = true;
  }
  ImGui::PopStyleColor(3);

  ImGui::SameLine();
  value_changed |= ImGui::DragFloat("##Z", &values.z, 0.1f, 0.0f, 0.0f, "%.2f");
  ImGui::PopItemWidth();

  ImGui::PopStyleVar();
  ImGui::Columns(1);
  ImGui::PopID();

  return value_changed;
}

void InspectorWindow::inspectObject(VObjectPtr objPtr) {
  ImGui::Begin("Inspector");

  const bool isNull = std::visit([](auto&& objPtr) { return objPtr == nullptr; }, objPtr);
  if (isNull) {
    ImGui::Text("Nothing Selected");
    ImGui::End();
    return;
  }

  const std::string& name = std::visit([](auto&& objPtr) { return objPtr->name; }, objPtr);
  ImGui::Text("%s", name.c_str());

  Transform& transform = std::visit([](auto&& objPtr) -> Transform& { return objPtr->transform; }, objPtr);
  DrawVec3Control("Position", transform.position);
  glm::vec3 eulerXyzDeg = glm::degrees(glm::eulerAngles(transform.rotation));
  if (DrawVec3Control("Rotation", eulerXyzDeg))
    transform.rotation = glm::quat(glm::radians(eulerXyzDeg));
  DrawVec3Control("Scale", transform.scale, 1);

  std::visit(Overloaded{
      [&]([[maybe_unused]] ws::DummyObject* ptr) {
        ImGui::Text("Dummy");
      },
      [&](ws::RenderableObject* renderable) {
        ImGui::Text("Renderable");
        ImGui::Text("Mesh. VOA: %d, VBO: %d, IBO: %d", static_cast<uint32_t>(renderable->mesh.vertexArray), static_cast<uint32_t>(renderable->mesh.vertexBuffer), static_cast<uint32_t>(renderable->mesh.indexBuffer));
        namespace views = std::ranges::views;
        const auto shaderIds = std::ranges::to<std::string>(renderable->shader.getShaderIds() | views::transform([](int n) { return std::to_string(n) + " "; }) | views::join);
        ImGui::Text("Shader. Program: %d, Shaders: %s", renderable->shader.getId(), shaderIds.c_str());
        ImGui::Text("Texture. name: %s, id: %d", renderable->texture.getName().c_str(), renderable->texture.getId());
      },
      [&](ws::CameraObject* cam) {
        ImGui::Text("Camera");
        ImGui::DragFloat("Near", &cam->camera.nearClip);
        ImGui::DragFloat("Far", &cam->camera.farClip);
        ImGui::DragFloat("Fov", &cam->camera.fov);
        // TODO: add directional parameters
      },
      [](auto arg) { throw "Unhandled VObjectPtr variant"; },
  },
  objPtr);

  ImGui::End();
}

EditorWindow::EditorWindow(Scene& scene)
  : 
    fbo(std::vector<Texture::Specs>{
      Texture::Specs{1, 1, Texture::Format::RGBA8, Texture::Filter::Nearest, Texture::Wrap::ClampToBorder},
      Texture::Specs{1, 1, Texture::Format::R32i, Texture::Filter::Nearest, Texture::Wrap::ClampToBorder},
    },
      Texture::Specs{1, 1, Texture::Format::Depth32fStencil8, Texture::Filter::Linear, Texture::Wrap::ClampToBorder}
    ),
    scene(scene),
		editorShader(ws::ASSETS_FOLDER / "shaders/editor.vert", ws::ASSETS_FOLDER / "shaders/editor.frag"),
		gridShader(ws::ASSETS_FOLDER / "shaders/infinite_grid.vert", ws::ASSETS_FOLDER / "shaders/infinite_grid.frag"),
    gridVao([](){ uint32_t id;glGenVertexArrays(1, &id); return id;}())
{ }

VObjectPtr EditorWindow::draw() {
	ImGui::Begin("Editor");

	static bool shouldBeWireframe = false;
	ImGui::Checkbox("Wireframe", &shouldBeWireframe);

  ImGui::SameLine();
  static int shadingModel = 2;
  std::array<const char*, 12> items = {"Pos (Obj)", "Pos (World)", "UV1", "UV2", "Normal (Obj)", "Normal (World)", "Front-Back Faces", "Texture1 (UV1)", "Texture2 (UV2)", "Depth (Ortho)", "Depth (Proj)", "Mesh IDs"};
  ImGui::Combo("Shading Model", &shadingModel, items.data(), static_cast<int>(items.size()));

  ImGui::SameLine();
  static std::string hoveredObjectName = "None"; // static because it'll be set after editor image is drawn
  ImGui::Text("Hovered: %s", hoveredObjectName.c_str());

	ImVec2 size = ImGui::GetContentRegionAvail();
  if (size.y < 0) { // happens when minimized
    ImGui::End();
    return {};
  }
	glm::ivec2 sizei { size.x, size.y };
	fbo.resizeIfNeeded(sizei.x, sizei.y);

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
		float keySpeed = 0.1f;

		if (ImGui::IsKeyDown(ImGuiKey_LeftShift))
			keySpeed *= 5;

		if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
			if (ImGui::IsKeyDown(ImGuiKey_A))
				deltaPos -= cam.getRight() * keySpeed;
			if (ImGui::IsKeyDown(ImGuiKey_D))
				deltaPos += cam.getRight() * keySpeed;
			if (ImGui::IsKeyDown(ImGuiKey_W))
				deltaPos += cam.getForward() * keySpeed;
			if (ImGui::IsKeyDown(ImGuiKey_S))
				deltaPos -= cam.getForward() * keySpeed;
			if (ImGui::IsKeyDown(ImGuiKey_Q))
				deltaPos -= cam.getUp() * keySpeed;
			if (ImGui::IsKeyDown(ImGuiKey_E))
				deltaPos += cam.getUp() * keySpeed;

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

  fbo.bind();
  // When drawing the scene render into first attachment, and put mesh ids into second attachment
  uint32_t bothAttachments[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
  glDrawBuffers(2, bothAttachments);
  glViewport(0, 0, sizei.x, sizei.y);
  glDisable(GL_BLEND);
  glEnable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
  const float red[] = {0, 0, 0, 1};
  glClearTexImage(fbo.getColorAttachments()[0].getId(), 0, GL_RGBA, GL_FLOAT, red);
  const int clearValue = -1;
  glClearTexImage(fbo.getColorAttachments()[1].getId(), 0, GL_RED_INTEGER, GL_INT, &clearValue);
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
    renderable.get().texture.bindToUnit(0);
    renderable.get().texture2.bindToUnit(1);
	  renderable.get().mesh.bind();
	  renderable.get().mesh.draw();
	  renderable.get().mesh.unbind();
    renderable.get().texture.unbindFromUnit(0);
    renderable.get().texture2.unbindFromUnit(1);
    editorShader.unbind();
  }

  // when drawing gizmos (such as coordinate grid) only draw into first attachment, don't touch mesh ids
  uint32_t onlyFirstAttachmetn[] = {GL_COLOR_ATTACHMENT0};
  glDrawBuffers(1, onlyFirstAttachmetn);
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
  fbo.unbind();

  // IsItemActivated checks whether InvisibleButton was clicked. IsItemClicked on the Image below didn't work for some reason. Probably because it's overlapping with the button.
  bool wasViewportClicked = ImGui::IsItemActivated();

  ImVec2 uv0 = {0, 0};
  ImVec2 uv1 = {1, 1};
  // flip the texture upside-down via following uv-coordinate transformation: (0, 0), (1, 1) -> (0, 1), (1, 0) 
  std::swap(uv0.y, uv1.y);
  ImVec2 imagePosToWin = ImGui::GetCursorPos();
  ImGui::Image((void*)(intptr_t)fbo.getFirstColorAttachment().getId(), size, uv0, uv1, { 1, 1, 1, 1 }, { 1, 1, 0, 1 });

  GLint hoveredMeshId{-2}; // -2 means not touched, -1 means no mesh under cursor
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
    //std::println("hoveredMeshId {}, selectedName {}, pixCoord ({:.1f}, {:.1f})", hoveredMeshId, hoveredObjectName, pixCoord.x, pixCoord.y);
    fbo.unbind();
  }

  VObjectPtr clickedObject {};
  if (wasViewportClicked && hoveredMeshId >= 0)
     clickedObject = &scene.renderables[hoveredMeshId].get();
  ImGui::End();

  return clickedObject;
}

}