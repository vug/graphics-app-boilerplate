cmake_minimum_required(VERSION 3.18)

project(GraphicsAppBoilerplate
  VERSION 0.1
  DESCRIPTION "A bunch of dependencies and some utility functionalities on top of them, so that I won't need to start a graphics program from scratch each time.")

add_subdirectory(dependencies)
add_subdirectory(Workshop)
add_subdirectory(example-apps/01-clear)
add_subdirectory(example-apps/02-triangle-without-vbo)

file(GLOB_RECURSE ASSETS ${CMAKE_CURRENT_SOURCE_DIR}/assets/*)
set(WORKSHOP_ASSETS ${ASSETS} PARENT_SCOPE)
set(WORKSHOP_ASSETS_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/assets/ PARENT_SCOPE)