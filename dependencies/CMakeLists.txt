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
target_compile_definitions(glm PUBLIC GLM_ENABLE_EXPERIMENTAL)


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
# Add STB_X_IMPLEMENTATION AND STB_X_STATIC together for each stb component so that its header file can be included by more than one CPP file
add_library(stb INTERFACE)
set_property(TARGET stb APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS 
  STB_IMAGE_IMPLEMENTATION STB_IMAGE_STATIC 
  STB_IMAGE_WRITE_IMPLEMENTATION STB_IMAGE_WRITE_STATIC
)
target_include_directories(
  stb INTERFACE
  stb/
)


# git submodule add https://github.com/tinyobjloader/tinyobjloader.git
add_subdirectory(tinyobjloader)


# git submodule add https://github.com/gurki/vivid.git
# vivid is an old project, and it's subdepedencies were triggering CMake warnings about old versions etc.
# Instead of relying on the CMakeLists.txt that comes with the repo, I explicitly turn that into a static library here myself.
#add_subdirectory(vivid)
include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)
add_library(
  vivid STATIC
  vivid/src/adobe.cpp
  vivid/src/color.cpp
  vivid/src/colormap.cpp
  vivid/src/hex.cpp
  vivid/src/hsl.cpp
  vivid/src/hsv.cpp
  vivid/src/index.cpp
  vivid/src/interpolation.cpp
  vivid/src/lab.cpp
  vivid/src/lch.cpp
  vivid/src/lrgb.cpp
  vivid/src/name.cpp
  vivid/src/oklab.cpp
  vivid/src/rgb.cpp
  vivid/src/rgb32.cpp
  vivid/src/rgb8.cpp
  vivid/src/srgb.cpp
  vivid/src/types.cpp
  vivid/src/utility.cpp
  vivid/src/xyz.cpp
)
target_include_directories(vivid PUBLIC vivid/include/)
target_link_libraries(vivid PUBLIC glm PRIVATE nlohmann_json) # don't know why glm needs to be public here
target_compile_features(vivid PRIVATE cxx_std_17)


# https://github.com/Lawrencemm/openmesh
# INTERNAL hides the value. Could have been BOOL, then parameter appears in "cmake .. -L"
# set(BUILD_APPS false CACHE INTERNAL "" FORCE) # we don't need the QT apps that comes with OpenMesh
# set(OPENMESH_DOCS false CACHE INTERNAL "" FORCE)
# add_subdirectory(openmesh)

# git submodule add https://github.com/fmtlib/fmt
add_subdirectory(fmt)


# git submodule add https://github.com/google/googletest.git
add_subdirectory(googletest)


# git submodule add https://github.com/jpcy/xatlas.git
add_library(xatlas STATIC
  xatlas/source/xatlas/xatlas.cpp
)
target_include_directories(
  xatlas INTERFACE
  xatlas/source/xatlas/
)