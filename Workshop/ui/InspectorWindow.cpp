#include "ui/InspectorWindow.hpp"

#include <imgui/imgui.h>
#include <imgui_internal.h> // for PushMultiItemsWidths, GImGui

#include <ranges>

namespace ws {
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
}