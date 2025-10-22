# tests/CMakeLists.txt
cmake_minimum_required(VERSION 3.25)
project(CoreAI3DTests)

# Enable testing
enable_testing()

# Include main project headers
include_directories(
    ${CMAKE_SOURCE_DIR}/CoreAI3D/include
    ${CMAKE_SOURCE_DIR}/CoreAI3D
)

# ---- Catch2 Tests ----
# find_package(Catch2 REQUIRED)

# add_executable(CoreAI3D_Catch2Tests
#     CoreTests.cpp
#     DatabaseTests.cpp
#     test_training.cpp# Your Catch2 test file(s)
# )

# target_link_libraries(CoreAI3D_Catch2Tests
#     PRIVATE CoreAI3DLib Catch2::Catch2WithMain
# )

# Register Catch2 tests with CTest
# include(CTest)
# include(Catch)
# catch_discover_tests(CoreAI3D_Catch2Tests)

# ---- GoogleTest Tests ----
# find_package(GTest REQUIRED)

# add_executable(CoreAI3D_GTest
#     TrainTests.cpp  # Your GoogleTest file(s)
# )

# target_link_libraries(CoreAI3D_GTest
#     PRIVATE CoreAI3DLib GTest::gtest_main
# )

# Register GoogleTest tests with CTest
# include(GoogleTest)
# gtest_discover_tests(CoreAI3D_GTest)
