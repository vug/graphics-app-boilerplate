set(TARGET EmbreeTest)


add_executable(${TARGET}
  main.cpp
)

if(MSVC)
  # Add /WX if warnings should be treated as errors
  target_compile_options(${TARGET} PRIVATE /W4 /external:I${PROJECT_SOURCE_DIR}/dependencies /external:W0)
else()
  target_compile_options(${TARGET} PRIVATE -Wall -Wextra -pedantic -Werror)
endif()

set(embree_DIR "C:/Users/veliu/repos-other/embree-4.3.1.x64.windows/lib/cmake/embree-4.3.1")
find_package(embree 4 REQUIRED)

target_link_libraries(${TARGET} PRIVATE
  Workshop
  embree
)

# to access example project source folder via `const std::filesystem::path SRC{SOURCE_DIR};`
target_compile_definitions(${TARGET} PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
# there is no latest option
target_compile_features(${TARGET} PRIVATE cxx_std_23)

add_custom_command(TARGET ${TARGET} POST_BUILD COMMAND ${CMAKE_COMMAND} -E echo "Built Target file: $<TARGET_FILE:${TARGET}>")