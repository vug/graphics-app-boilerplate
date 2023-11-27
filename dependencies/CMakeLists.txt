# git submodule add --branch glad2 https://github.com/Dav1dde/glad.git
# requires Python and jinja2
add_subdirectory(glad/cmake)
glad_add_library(glad_gl_core_46 STATIC API gl:core=4.6)
# glad_add_library(glad_gl_core_31 STATIC API gl:core=3.1)


# https://github.com/glfw/glfw
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(glfw)


# git submodule add https://github.com/g-truc/glm
add_subdirectory(glm)


# git submodule add --branch docking https://github.com/ocornut/imgui.git
add_library(imgui STATIC
  imgui/imgui.cpp
  imgui/imgui_draw.cpp
  imgui/imgui_widgets.cpp
  imgui/imgui_tables.cpp
  imgui/imgui_demo.cpp # for ImGui::ShowDemoWindow()
  imgui/backends/imgui_impl_glfw.cpp
  imgui/backends/imgui_impl_opengl3.cpp
  imgui/misc/cpp/imgui_stdlib.cpp
)
target_compile_definitions(imgui PRIVATE
  -DIMGUI_IMPL_OPENGL_LOADER_GLAD # configure GLAD as the loader
  -DGLFW_INCLUDE_NONE
)
target_link_libraries(imgui PRIVATE
  glfw
)
target_include_directories(
  imgui PUBLIC
  imgui/
  imgui/backends
  imgui/misc/cpp
)


# git submodule add https://github.com/epezent/implot.git 
add_library(
  implot
  implot/implot.cpp
  implot/implot_items.cpp
  implot/implot_demo.cpp
)
target_link_libraries(
  implot PRIVATE
  imgui
)
target_include_directories(
  implot PUBLIC
  implot/
)


# git submodule add https://github.com/nothings/stb.git
# https://stackoverflow.com/questions/55081746/add-target-properties-to-an-existing-imported-library-in-cmake
# Am I making compilation slower by not putting "#define STB_IMAGE_IMPLEMENTATION" into a CPP file?
# STB_IMAGE_STATIC added so that header files can be included by more than one CPP file.
add_library(stb INTERFACE)
set_property(TARGET stb APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS STB_IMAGE_IMPLEMENTATION STB_IMAGE_STATIC)
target_include_directories(
  stb INTERFACE
  stb/
)


# git submodule add https://github.com/tinyobjloader/tinyobjloader.git
add_subdirectory(tinyobjloader)


# git submodule add https://github.com/gurki/vivid.git
# Vivid's cmake is not properly setup wrt its dependency on GLM. We are forcing it to use the GLM from our dependency folder
set(FETCHCONTENT_SOURCE_DIR_GLM "dependencies/glm" CACHE STRING "" FORCE) # Throws a warning about relative directory unsafe. Ignore.
add_subdirectory(vivid)


# https://github.com/Lawrencemm/openmesh
# INTERNAL hides the value. Could have been BOOL, then parameter appears in "cmake .. -L"
# set(BUILD_APPS false CACHE INTERNAL "" FORCE) # we don't need the QT apps that comes with OpenMesh
# set(OPENMESH_DOCS false CACHE INTERNAL "" FORCE)
# add_subdirectory(openmesh)

# git submodule add https://github.com/fmtlib/fmt
add_subdirectory(fmt)