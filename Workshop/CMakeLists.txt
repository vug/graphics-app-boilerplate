set(TARGET Workshop)

if(MSVC)
  # /WX if warnings should be treated as errors
  add_compile_options(/W4 /external:I${PROJECT_SOURCE_DIR}/dependencies /external:W0)
else()
  add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

add_library(${TARGET} STATIC
  includes/Workshop/Common.hpp
  includes/Workshop/Workshop.hpp Workshop.cpp
  includes/Workshop/Shader.hpp Shader.cpp
  includes/Workshop/Texture.hpp Texture.cpp
  includes/Workshop/Cubemap.hpp Cubemap.cpp
  includes/Workshop/Material.hpp Material.cpp
  includes/Workshop/Framebuffer.hpp Framebuffer.cpp
  includes/Workshop/Model.hpp Model.cpp
  includes/Workshop/Transform.hpp Transform.cpp
  includes/Workshop/Camera.hpp Camera.cpp
  includes/Workshop/Lights.hpp Lights.cpp
  includes/Workshop/Input.hpp Input.cpp
  includes/Workshop/OpenGL.hpp OpenGL.cpp
)

target_link_libraries(${TARGET} PUBLIC
  fmt
  # glad_gl_core_31 
  glad_gl_core_46
  glfw
  glm
  GTest::gtest_main
  imgui
  implot
  stb
  tinyobjloader
  vivid
)

# to enable `#include <Workshop/Shader.h>` in projects that depend on Workshop
target_include_directories(${TARGET} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/includes)
# to enable `#include "Shader.h"` inside Workshop project
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/includes/Workshop)
# Create an Assets.h file which holds a filepath variable to workshop assets root 
# and place it somewhere under build/ so that it can be included via <Workshop/Assets.h>
set(WS_ASSETS_FOLDER ${PROJECT_SOURCE_DIR}/assets)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/includes/Workshop/Assets.hpp.in includes/Workshop/Assets.hpp @ONLY)
target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/includes/)

target_compile_features(${TARGET} PRIVATE cxx_std_23)