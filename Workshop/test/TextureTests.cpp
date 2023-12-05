#include "WorkshopTest.hpp"

#include <Workshop/Texture.hpp>

#include <array>
#include <vector>

TEST_F(WorkshopTest, glTextureGenDelete) {
  uint32_t id;
  glGenTextures(1, &id);
  EXPECT_FALSE(glIsTexture(id));

  glBindTexture(GL_TEXTURE_2D, id);
  // Allocating texture is not needed for glIsTexture to return true
  //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 128, 128, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glBindTexture(GL_TEXTURE_2D, 0);
  EXPECT_TRUE(glIsTexture(id));

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

TEST_F(WorkshopTest, TextureMove) {
  ws::Texture tex1{};
  ASSERT_NE(tex1.getId(), ws::INVALID);

  ws::Texture tex2 = std::move(tex1);
  ASSERT_EQ(tex1.getId(), ws::INVALID);

  uint32_t tex4Id;
  {
    ws::Texture tex4{};
    tex4Id = tex4.getId();
  }
  ASSERT_FALSE(glIsTexture(tex4Id));
}


std::vector<ws::Texture> makeVec() {
  std::vector<ws::Texture> result;
  result.push_back(ws::Texture{});
  result.push_back(ws::Texture{});
  return result;
}

TEST_F(WorkshopTest, TextureArrayInitialization) {
  std::array<ws::Texture, 2> arr{ws::Texture{}, ws::Texture{}};
  // std::vector<ws::Texture> arr{ws::Texture{}, ws::Texture{}}; // does not compile because of deleted copy constructor
  std::vector<ws::Texture> vec = makeVec();
  ASSERT_EQ(arr[0].getId() + 1, arr[1].getId());
  ASSERT_EQ(vec[0].getId() + 1, vec[1].getId());
}
