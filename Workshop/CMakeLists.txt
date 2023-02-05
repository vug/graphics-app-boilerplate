set(TARGET Workshop)

if(MSVC)
  # /WX if warnings should be treated as errors
  add_compile_options(/W4 /external:I${PROJECT_SOURCE_DIR}/dependencies /external:W0)
else()
  add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

add_library(${TARGET} STATIC
  Common.h
  Workshop.h Workshop.cpp
  Shader.h Shader.cpp
  Texture.h Texture.cpp
  Framebuffer.h Framebuffer.cpp
)

target_link_libraries(${TARGET} PUBLIC
  glad_gl_core_31 # glad_gl_core_46
  glfw
  glm
  imgui
  implot
  stb
  tinyobjloader
  vivid
)

# to enable `#include <Workshop.h>` in projects that depend on Workshop
target_include_directories(Workshop PUBLIC .)

target_compile_features(${TARGET} PRIVATE cxx_std_20)