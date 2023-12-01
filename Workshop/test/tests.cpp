#include <gtest/gtest.h>

#include <Workshop/Workshop.hpp>

#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

// Make sure that Workshop Constructor and {begin, end}Frame methods won't throw
TEST(AppSkeletonTest, NoExceptionsBeginEnd) {
  // EXPECT_NO_THROW(ws::Workshop ws{800, 600, "Workshop App"}) won't let ws be defined outside of the macro
  // Therefore unique_ptr is used to be able to access it in later lines
  std::unique_ptr<ws::Workshop> workshop;
  EXPECT_NO_THROW(workshop.reset(new ws::Workshop{800, 600, "Workshop App"}));
  EXPECT_NO_THROW(workshop->beginFrame());
  EXPECT_NO_THROW(workshop->endFrame());
}

class WorkshopTest : public testing::Test {
 protected:
  ws::Workshop workshop;

  WorkshopTest()
      : workshop{800, 600, "Workshop App"} {}

  void SetUp() override {
    workshop.beginFrame();
  }

  void TearDown() override {
    workshop.endFrame();
  }
};

// Test glClear actually clears the screen with the color given in glClearColor
TEST_F(WorkshopTest, ClearColorDepth) {
  glClearColor(1, 0, 0, 1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  unsigned char* data = (unsigned char*)malloc(sizeof(unsigned char) * 4);
  glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, data);
  EXPECT_EQ(data[0], 255);
  EXPECT_EQ(data[1], 0);
  EXPECT_EQ(data[2], 0);
  EXPECT_EQ(data[3], 255);
}

// Test Workshop::getWindowSize returns initial window size
TEST_F(WorkshopTest, WindowInitialSize) {
  auto winSize = workshop.getWindowSize();
  EXPECT_EQ(winSize.x, 800);
  EXPECT_EQ(winSize.y, 600);
}

// Test Workshop::getWindowSize returns correct window size after a resize
TEST_F(WorkshopTest, WindowResizedSize) {
  glfwSetWindowSize(workshop.getGLFWwindow(), 1024, 768);
  auto winSize = workshop.getWindowSize();
  EXPECT_EQ(winSize.x, 1024);
  EXPECT_EQ(winSize.y, 768);
}

TEST_F(WorkshopTest, ViewportInitialSize) {
  GLint m_viewport[4];

  glGetIntegerv(GL_VIEWPORT, m_viewport);
  EXPECT_EQ(m_viewport[0], 0);
  EXPECT_EQ(m_viewport[1], 0);
  EXPECT_EQ(m_viewport[2], 800);
  EXPECT_EQ(m_viewport[3], 600);
}

TEST_F(WorkshopTest, ViewportResizedSize) {
  GLint m_viewport[4];

   glfwSetWindowSize(workshop.getGLFWwindow(), 1024, 768);
   workshop.endFrame();
   workshop.beginFrame();
   glGetIntegerv(GL_VIEWPORT, m_viewport);
  // TODO: add a glViewport to Workshop::beginFrame
  // EXPECT_EQ(m_viewport[0], 0);
  // EXPECT_EQ(m_viewport[1], 0);
  // EXPECT_EQ(m_viewport[2], 1024);
  // EXPECT_EQ(m_viewport[3], 768);
}
