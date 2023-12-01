#include <gtest/gtest.h>

#include <Workshop/Workshop.hpp>

#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

// Make sure that Workshop Constructor and {begin, end}Frame methods won't throw
TEST(AppTest, NoExceptionsBeginEnd) {
  // EXPECT_NO_THROW(ws::Workshop ws{800, 600, "Workshop App"}) won't let ws be defined outside of the macro
  // Therefore unique_ptr is used to be able to access it in later lines
  std::unique_ptr<ws::Workshop> workshop;
  EXPECT_NO_THROW(workshop.reset(new ws::Workshop{800, 600, "Workshop App"}));
  EXPECT_NO_THROW(workshop->beginFrame());
  EXPECT_NO_THROW(workshop->endFrame());
}

// Test glClear actually clears the screen with the color given in glClearColor
TEST(AppTest, ClearColorDepth) {
  ws::Workshop workshop{800, 600, "Workshop App"};
  workshop.beginFrame();

  glClearColor(1, 0, 0, 1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  unsigned char* data = (unsigned char*)malloc(sizeof(unsigned char) * 4);
  glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, data);
  EXPECT_EQ(data[0], 255);
  EXPECT_EQ(data[1], 0);
  EXPECT_EQ(data[2], 0);
  EXPECT_EQ(data[3], 255);

  workshop.endFrame();
}