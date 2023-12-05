#include "WorkshopTest.hpp"

#include <Workshop/Assets.hpp>
#include <Workshop/Shader.hpp>

#include <print>

TEST_F(WorkshopTest, ShaderInspection) {
  ws::Shader shader{ws::ASSETS_FOLDER / "shaders/solid_color.vert", ws::ASSETS_FOLDER / "shaders/solid_color.frag"};
  GLint params{};

  glGetProgramiv(shader.getId(), GL_DELETE_STATUS, &params);
  ASSERT_EQ(params, 0);

  glGetProgramiv(shader.getId(), GL_LINK_STATUS, &params);
  ASSERT_EQ(params, 1);

  glGetProgramiv(shader.getId(), GL_VALIDATE_STATUS, &params);
  ASSERT_EQ(params, 1);

  glGetProgramiv(shader.getId(), GL_INFO_LOG_LENGTH, &params);
  ASSERT_EQ(params, 0);

  glGetProgramiv(shader.getId(), GL_ATTACHED_SHADERS, &params);
  ASSERT_EQ(params, 2);

  glGetProgramiv(shader.getId(), GL_ACTIVE_ATOMIC_COUNTER_BUFFERS, &params); // ?
  ASSERT_EQ(params, 0);

  GLint longestActiveAttributeNameLength{};
  glGetProgramiv(shader.getId(), GL_ACTIVE_ATTRIBUTE_MAX_LENGTH, &longestActiveAttributeNameLength);
  std::string attributeName(longestActiveAttributeNameLength, '\0');
  GLint numActiveAttributes;
  glGetProgramiv(shader.getId(), GL_ACTIVE_ATTRIBUTES, &numActiveAttributes);
  ASSERT_EQ(numActiveAttributes, 1);
  for (int i = 0; i < numActiveAttributes; ++i) {
    GLsizei length;
    GLint size;
    GLenum type;
    glGetActiveAttrib(shader.getId(), i, static_cast<GLsizei>(sizeof(GLchar) * attributeName.size()), &length, &size, &type, attributeName.data());
    // TODO: instead of printing, have a custom shader for the test and assert based on its content
    std::println("attribute {} {}[{}] {}", i, attributeName, size, ws::Shader::UNIFORM_AND_ATTRIBUTE_TYPES[type]);
  }

  GLint longestActiveUniformNameLength{};
  glGetProgramiv(shader.getId(), GL_ACTIVE_UNIFORM_MAX_LENGTH, &longestActiveUniformNameLength);
  std::string uniformName(longestActiveUniformNameLength, '\0');
  GLint numActiveUniforms;
  glGetProgramiv(shader.getId(), GL_ACTIVE_UNIFORMS, &numActiveUniforms);
  for (int i = 0; i < numActiveUniforms; i++) {
    GLsizei length;
    GLint size;
    GLenum type;
    glGetActiveUniform(shader.getId(), i, static_cast<GLsizei>(sizeof(GLchar) * uniformName.size()), &length, &size, &type, uniformName.data());
    std::println("uniform {} {}[{}] {}", i, uniformName, size, ws::Shader::UNIFORM_AND_ATTRIBUTE_TYPES[type]);
  }

  glGetProgramiv(shader.getId(), GL_ACTIVE_UNIFORM_BLOCKS, &params);
  ASSERT_EQ(params, 0);
  glGetProgramiv(shader.getId(), GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH, &params);
  ASSERT_EQ(params, 0);
}