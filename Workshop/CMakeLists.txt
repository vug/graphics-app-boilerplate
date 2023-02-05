set(TARGET Workshop)

if(MSVC)
  # /WX if warnings should be treated as errors
  add_compile_options(/W4 /external:I${PROJECT_SOURCE_DIR}/dependencies /external:W0)
else()
  add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

add_library(${TARGET} STATIC
  include/Workshop/Common.h
  include/Workshop/Workshop.h Workshop.cpp
  include/Workshop/Shader.h Shader.cpp
  include/Workshop/Texture.h Texture.cpp
  include/Workshop/Framebuffer.h Framebuffer.cpp
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

# to enable `#include <Workshop/Shader.h>` in projects that depend on Workshop
target_include_directories(${TARGET} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# to enable `#include "Shader.h"` inside Workshop project
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/Workshop)

set(WS_ASSETS_FOLDER ${PROJECT_SOURCE_DIR}/assets)
configure_file(Assets.h.in Assets.h @ONLY)
target_include_directories(${TARGET} PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")

target_compile_features(${TARGET} PRIVATE cxx_std_20)