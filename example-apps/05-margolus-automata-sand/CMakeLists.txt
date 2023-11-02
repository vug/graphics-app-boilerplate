set(TARGET MargolusSand)

if(TRUE)
message("Method1: Make a static library out of CU files and link them. dir1 ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
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
else()
message("Method2: Make OBJ files from CU files and link them")
add_executable(${TARGET}
  main.cpp
  kernels.cu kernels.h)
set_target_properties(${TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(${TARGET} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${TARGET} PRIVATE 
  Workshop
)
endif()

if(MSVC)
  # /WX if warnings should be treated as errors
  #target_compile_options(${TARGET} PRIVATE /W4 /external:I${PROJECT_SOURCE_DIR}/dependencies /external:W0)
else()
  #target_compile_options(${TARGET} PRIVATE -Wall -Wextra -pedantic -Werror)
endif()

set_property(TARGET ${TARGET} PROPERTY CUDA_STANDARD 26)
target_compile_features(${TARGET} PRIVATE cxx_std_23)