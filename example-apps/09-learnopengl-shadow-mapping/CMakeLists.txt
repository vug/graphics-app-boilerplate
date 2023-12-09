set(TARGET ShadowMapping)

if(MSVC)
  # /WX if warnings should be treated as errors
  add_compile_options(/W4 /external:I${PROJECT_SOURCE_DIR}/dependencies /external:W0)
else()
  add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

add_executable(${TARGET}
  main.cpp
)

target_link_libraries(${TARGET} PRIVATE
  Workshop
)

target_compile_definitions(${TARGET} PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(${TARGET} PRIVATE cxx_std_23)