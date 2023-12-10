set(TARGET ShadowMapping)

add_executable(${TARGET}
  main.cpp
)

if(MSVC)
  # Add /WX if warnings should be treated as errors
  target_compile_options(${TARGET} PRIVATE /W4 /external:I${PROJECT_SOURCE_DIR}/dependencies /external:W0)

  # Setup for `import std;`
  # Key for experimental feature. TODO: Remove once project minimum version becomes 3.28
  set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API "aa1f7df0-828a-4fcd-9afc-2dc80491aca7")
  set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP 1)
  set(MSVC_PATH ${CMAKE_CXX_COMPILER}) # C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.38.33130/bin/Hostx64/x64/cl.exe
  cmake_path(GET MSVC_PATH PARENT_PATH MSVC_PATH)
  cmake_path(GET MSVC_PATH PARENT_PATH MSVC_PATH)
  cmake_path(GET MSVC_PATH PARENT_PATH MSVC_PATH)
  cmake_path(GET MSVC_PATH PARENT_PATH MSVC_PATH) # C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.38.33130/
  target_compile_options(${TARGET} PRIVATE /experimental:module /ifcSearchDir "${MSVC_PATH}/ifc/x64/$<IF:$<CONFIG:Debug>,Debug,Release>")
  target_include_directories(${TARGET} PRIVATE "${MSVC_PATH}/modules")
else()
  target_compile_options(${TARGET} PRIVATE -Wall -Wextra -pedantic -Werror)
endif()

target_link_libraries(${TARGET} PRIVATE
  Workshop
)

# to access example project source folder via `const std::filesystem::path SRC{SOURCE_DIR};`
target_compile_definitions(${TARGET} PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
# there is no latest option
target_compile_features(${TARGET} PRIVATE cxx_std_23)

add_custom_command(TARGET ${TARGET} POST_BUILD COMMAND ${CMAKE_COMMAND} -E echo "Built Target file: $<TARGET_FILE:${TARGET}>")