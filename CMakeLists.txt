cmake_minimum_required(VERSION 3.18)

project(GraphicsAppBoilerplate
  VERSION 0.1
  DESCRIPTION "A bunch of dependencies and some utility functionalities on top of them, so that I won't need to start a graphics program from scratch each time.")

add_subdirectory(dependencies)
add_subdirectory(src)


