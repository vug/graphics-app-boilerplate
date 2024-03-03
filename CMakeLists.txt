# TODO: Use v3.28 when VS upgrades to it to use modules without experimental key
message("[VUGDEBUG] CMake version ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}")
cmake_minimum_required(VERSION 3.27)

include(CheckLanguage)
check_language(C)
check_language(CXX)
check_language(CUDA)
message("Compilers.\n C: ${CMAKE_C_COMPILER}\n CPP: ${CMAKE_CXX_COMPILER}\n CUDA: ${CMAKE_CUDA_COMPILER}")

project(GraphicsAppBoilerplate
  VERSION 0.1
  DESCRIPTION "A bunch of dependencies and some utility functionalities on top of them, so that I won't need to start a graphics program from scratch each time."
  LANGUAGES C CXX CUDA)

add_subdirectory(dependencies)
add_subdirectory(Workshop)
add_subdirectory(Workshop/test)
add_subdirectory(example-apps/00-boilerplate)
add_subdirectory(example-apps/01-clear)
add_subdirectory(example-apps/02-triangle-without-vbo)
add_subdirectory(example-apps/03-cuda-start)
add_subdirectory(example-apps/04-cuda-mandelbrot)
add_subdirectory(example-apps/05-margolus-automata-sand)
add_subdirectory(example-apps/06-learnopengl-stencil-testing)
add_subdirectory(example-apps/07-outlines-via-grow)
add_subdirectory(example-apps/08-learnopengl-cubemaps)
add_subdirectory(example-apps/09-learnopengl-shadow-mapping)
add_subdirectory(example-apps/10-uv-atlas)
add_subdirectory(example-apps/11-rendering-cookbook-infinite-grid)
add_subdirectory(example-apps/12-normal-mapping)
add_subdirectory(example-apps/13-bloom)
add_subdirectory(example-apps/14-material-system)
add_subdirectory(example-apps/15-smallpaint)
add_subdirectory(example-apps/16-embree)

file(GLOB_RECURSE ASSETS ${CMAKE_CURRENT_SOURCE_DIR}/assets/*)
set(WORKSHOP_ASSETS ${ASSETS})
set(WORKSHOP_ASSETS_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/assets/)