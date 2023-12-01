set(TARGET Tests)

enable_testing()

add_executable(${TARGET}
  WorkshopTest.hpp
  WorkshopTests.cpp
  TextureTests.cpp
)

target_link_libraries(${TARGET} PRIVATE
  Workshop
)

target_compile_features(${TARGET} PRIVATE cxx_std_23)

if(MSVC)
  # Add /WX if warnings should be treated as errors
  target_compile_options(${TARGET} PRIVATE /W4 /external:I${PROJECT_SOURCE_DIR}/dependencies /external:W0)
else()
  target_compile_options(${TARGET} PRIVATE -Wall -Wextra -pedantic -Werror)
endif()

include(GoogleTest)
gtest_discover_tests(${TARGET})