#pragma once

#include <Workshop/Workshop.hpp>

#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <gtest/gtest.h>

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