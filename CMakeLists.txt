cmake_minimum_required(VERSION 3.18)

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
add_subdirectory(example-apps/01-clear)
add_subdirectory(example-apps/02-triangle-without-vbo)
add_subdirectory(example-apps/03-cuda-start)
add_subdirectory(example-apps/04-cuda-mandelbrot)
add_subdirectory(example-apps/05-margolus-automata-sand)
add_subdirectory(example-apps/06-learnopengl-stencil-testing)
add_subdirectory(example-apps/07-outlines-via-grow)
add_subdirectory(example-apps/08-learnopengl-cubemaps)

file(GLOB_RECURSE ASSETS ${CMAKE_CURRENT_SOURCE_DIR}/assets/*)
set(WORKSHOP_ASSETS ${ASSETS} PARENT_SCOPE)
set(WORKSHOP_ASSETS_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/assets/ PARENT_SCOPE)