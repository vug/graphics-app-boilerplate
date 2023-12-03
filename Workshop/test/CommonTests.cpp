#include "WorkshopTest.hpp"

#include <Workshop/Common.hpp>

TEST(CommonTests, GlHandleMove) {
  ws::GlHandle h1;
  ASSERT_TRUE(h1 == ws::INVALID);
  ASSERT_EQ(h1, ws::INVALID);

  h1 = 42;
  ASSERT_EQ(h1, 42);

  // Move constructor
  ws::GlHandle h2{std::move(h1)};
  ASSERT_EQ(h1, ws::INVALID);
  ASSERT_EQ(h2, 42);

  // Move assignment
  h1 = std::move(h2);
  ASSERT_EQ(h2, ws::INVALID);
  ASSERT_EQ(h1, 42);

  // Handle is equal to itself
  ASSERT_EQ(h1, h1);
  // Different than another handle (with different value)
  ASSERT_NE(h1, h2);

  // Handles with the same value are equal (shouldn't happen in app code)
  ws::GlHandle h3;
  ASSERT_EQ(h2, h3);
}