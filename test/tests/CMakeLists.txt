# tests/CMakeLists.txt
cmake_minimum_required(VERSION 3.25)
project(CoreAI3DTests)

# Enable testing
enable_testing()

# Include main project headers
include_directories(
    ${CMAKE_SOURCE_DIR}/../../CoreAI3D/include
    ${CMAKE_SOURCE_DIR}/../../CoreAI3D
)

# ---- GoogleTest Tests ----
find_package(GTest REQUIRED)

add_executable(CoreAI3D_GTest
    TrainTests.cpp
)

target_link_libraries(CoreAI3D_GTest
    PRIVATE CoreAI3DLib GTest::gtest_main
)

# Register GoogleTest tests with CTest
include(GoogleTest)
gtest_discover_tests(CoreAI3D_GTest)

# ---- Catch2 Tests (header-only) ----
# Since Catch2 is not installed, use header-only version
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(Catch2)

add_executable(CoreAI3D_Catch2Tests
    CoreTests.cpp
    DatabaseTests.cpp
)

target_link_libraries(CoreAI3D_Catch2Tests
    PRIVATE CoreAI3DLib Catch2::Catch2WithMain
)

# Register Catch2 tests with CTest
include(CTest)
include(Catch)
catch_discover_tests(CoreAI3D_Catch2Tests)

# ---- Standalone Test Executables ----
add_executable(simple_test simple_test.cpp)
target_link_libraries(simple_test PRIVATE CoreAI3DLib)
add_test(NAME SimpleTest COMMAND simple_test)

add_executable(test_simple_training ../test_simple_training.cpp)
target_link_libraries(test_simple_training PRIVATE CoreAI3DLib)
add_test(NAME TestSimpleTraining COMMAND test_simple_training)

add_executable(test_data_loading ../test_data_loading.cpp)
target_link_libraries(test_data_loading PRIVATE CoreAI3DLib)
add_test(NAME TestDataLoading COMMAND test_data_loading)

add_executable(test_auto_tuning ../test_auto_tuning.cpp)
target_link_libraries(test_auto_tuning PRIVATE CoreAI3DLib)
add_test(NAME TestAutoTuning COMMAND test_auto_tuning)
