set(TARGET Mandelbrot)

set(CUDALIB "${TARGET}_CudaLib")
add_library(${CUDALIB} STATIC kernels.cu kernels.h)
# We need to explicitly state that we need all CUDA files in the 
# to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties(${CUDALIB} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(${CUDALIB} INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${CUDALIB} PRIVATE glm) # for #include <glm/vec2.hpp> in kernels.h

add_executable(${TARGET}
  main.cpp
  kernels.h
)
set_target_properties(${TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${TARGET} PRIVATE 
  ${CUDALIB}
  Workshop
)

if(MSVC)
  # Add /WX if warnings should be treated as errors
  # target_compile_options(${TARGET} PRIVATE /W4 /external:I${PROJECT_SOURCE_DIR}/dependencies /external:W0)
else()
  # target_compile_options(${TARGET} PRIVATE -Wall -Wextra -pedantic -Werror)
endif()


# to access example project source folder via `const std::filesystem::path SRC{SOURCE_DIR};`
target_compile_definitions(${TARGET} PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
# there is no latest option
target_compile_features(${TARGET} PRIVATE cxx_std_23)
set_property(TARGET ${TARGET} PROPERTY CUDA_STANDARD 26)

add_custom_command(TARGET ${TARGET} POST_BUILD COMMAND ${CMAKE_COMMAND} -E echo "Built Target file: $<TARGET_FILE:${TARGET}>")