set(APP RenderGraph01)

if(MSVC)
  # /WX if warnings should be treated as errors
  add_compile_options(/W4 /external:I${PROJECT_SOURCE_DIR}/dependencies /external:W0)
else()
  add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

add_executable(${APP}
  main.cpp)

target_link_libraries(${APP}
  glad_gl_core_31 # glad_gl_core_46
  glfw
)

target_compile_features(${APP} PRIVATE cxx_std_20)