#include "WorkshopTest.hpp"

#include <Workshop/Texture.hpp>

TEST_F(WorkshopTest, TextureTest) {
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