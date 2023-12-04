#include "WorkshopTest.hpp"

#include <Workshop/Common.hpp>

#include <array>

TEST(CommonTests, GlHandleConstructMoveCompare) {
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

std::array<uint32_t, 8> resources{};

void genResource(uint32_t* name) {
  static uint32_t cnt = 1;
  resources[cnt] = 1;
  *name = cnt;
  ++cnt;
}

void deleteResource(uint32_t* name) {
  if (resources[*name] == 1) {
    resources[*name] = 0;
  } else {
    throw std::exception("Attempting to delete non-existing resource");
  }
}

class GlResource {
  ws::GlHandle id;
  std::vector<uint8_t> managedMember;

public:
  GlResource()
      : id{[]() { uint32_t id;  genResource(&id); return id; }()}
  {}

  ~GlResource() {
    if (id != ws::INVALID)
      deleteResource(&id);
  }

  GlResource(const GlResource& other) = delete;
  GlResource& operator=(const GlResource& other) = delete;

  GlResource(GlResource&& other) = default;
  GlResource& operator=(GlResource&& other) = default;

  uint32_t getId() const { return id; }
};

TEST(CommonTests, GlResourcesWithHandle) {
  auto resources1 = std::array<uint32_t, 8>{0, 0, 0, 0, 0, 0, 0, 0};
  ASSERT_EQ(resources, resources);

  GlResource res1;
  GlResource res2;
  auto resources2 = std::array<uint32_t, 8>{0, 1, 1, 0, 0, 0, 0, 0};
  ASSERT_EQ(resources, resources2);
  ASSERT_EQ(res1.getId(), 1);
  ASSERT_EQ(res2.getId(), 2);

  {
    GlResource res3;
    auto resources3 = std::array<uint32_t, 8>{0, 1, 1, 1, 0, 0, 0, 0};
    ASSERT_EQ(resources, resources3);
  }
  auto resources4 = std::array<uint32_t, 8>{0, 1, 1, 0, 0, 0, 0, 0};
  ASSERT_EQ(resources, resources4);

  GlResource res4;
  auto resources5 = std::array<uint32_t, 8>{0, 1, 1, 0, 1, 0, 0, 0};
  ASSERT_EQ(resources, resources5);

  res2 = std::move(res1);
  ASSERT_EQ(res1.getId(), ws::INVALID);
  ASSERT_EQ(res2.getId(), 1);
  auto resources6 = std::array<uint32_t, 8>{0, 1, 1, 0, 1, 0, 0, 0};
  ASSERT_EQ(resources, resources6);
}