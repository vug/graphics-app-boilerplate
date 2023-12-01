#include "WorkshopTest.hpp"

#include <Workshop/Texture.hpp>

TEST_F(WorkshopTest, glTextureGenDelete) {
  uint32_t id;
  glGenTextures(1, &id);
  glBindTexture(GL_TEXTURE_2D, id);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 128, 128, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  EXPECT_TRUE(glIsTexture(id));
  glBindTexture(GL_TEXTURE_2D, 0);

  glDeleteTextures(1, &id);
  EXPECT_FALSE(glIsTexture(id));
}

TEST_F(WorkshopTest, TextureIdDimensions) {
  ws::Texture::Specs specs {
    .width = 1024,
    .height = 768,
  };

  ws::Texture tex{specs};
  ASSERT_NE(tex.getId(), ws::INVALID);

  int w, h;
  const int mipLevel = 0;
  glGetTextureLevelParameteriv(tex.getId(), mipLevel, GL_TEXTURE_WIDTH, &w);
  glGetTextureLevelParameteriv(tex.getId(), mipLevel, GL_TEXTURE_HEIGHT, &h);
  ASSERT_EQ(w, 1024);
  ASSERT_EQ(h, 768);
}

TEST_F(WorkshopTest, TextureDestruction) {
  uint32_t id = 0;
  EXPECT_FALSE(glIsTexture(id));
  {
    ws::Texture tex1{};
    id = tex1.getId();
    ASSERT_NE(id, ws::INVALID);
    EXPECT_TRUE(glIsTexture(id));
  }
  EXPECT_FALSE(glIsTexture(id));
}