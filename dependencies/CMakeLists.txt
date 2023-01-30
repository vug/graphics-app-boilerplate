# git submodule add --branch glad2 https://github.com/Dav1dde/glad.git
# requires Python and jinja2
add_subdirectory(glad/cmake)
# glad_add_library(glad_gl_core_46 STATIC API gl:core=4.6)
glad_add_library(glad_gl_core_31 STATIC API gl:core=3.1)

# https://github.com/glfw/glfw
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(glfw)
