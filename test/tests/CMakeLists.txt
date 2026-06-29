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
    ../src/TrainTests.cpp
    BenchmarkTests.cpp
    PerformanceTests.cpp
    ComparisonTests.cpp
    EdgeCaseTests.cpp
    StressTests.cpp
    RealisticUseCaseTests.cpp
)

target_link_libraries(CoreAI3D_GTest
    PRIVATE CoreAI3DLib GTest::gtest_main
)
# Allow multiple symbol definitions due to mysql-connector-cpp bundling protobuf-Lite
if(NOT WIN32)
    target_link_options(CoreAI3D_GTest PRIVATE -Wl,--allow-multiple-definition)
endif()

# Register GoogleTest tests with CTest
include(GoogleTest)
gtest_discover_tests(CoreAI3D_GTest)

# ---- Catch2 Tests ----
find_package(Catch2 REQUIRED)

add_executable(CoreAI3D_Catch2Tests
    CoreTests.cpp
    DatabaseTests.cpp
    NeuralNetworkTests.cpp
    ActivationTests.cpp
    OptimizerTests.cpp
    RegularizationTests.cpp
    RegressionTests.cpp
    ClassificationTests.cpp
    MathExaminationTests.cpp
)

target_link_libraries(CoreAI3D_Catch2Tests
    PRIVATE CoreAI3DLib Catch2::Catch2WithMain
)

# Register Catch2 tests with CTest
include(CTest)
include(Catch)
catch_discover_tests(CoreAI3D_Catch2Tests)

# ---- Standalone Test Executables ----
# Note: These tests do NOT use MySQL database functionality. The mysql-connector-cpp
# static library bundles its own copy of protobuf-lite which causes symbol conflicts
# with the system protobuf. We use linker flags to allow multiple definitions.

add_executable(simple_test ../src/simple_test.cpp)
target_link_libraries(simple_test PRIVATE CoreAI3DLib)
# Allow multiple symbol definitions due to mysql-connector-cpp bundling protobuf-lite
if(NOT WIN32)
    target_link_options(simple_test PRIVATE -Wl,--allow-multiple-definition)
endif()
add_test(NAME SimpleTest COMMAND simple_test)

add_executable(test_simple_training ../src/test_simple_training.cpp)
target_link_libraries(test_simple_training PRIVATE CoreAI3DLib)
if(NOT WIN32)
    target_link_options(test_simple_training PRIVATE -Wl,--allow-multiple-definition)
endif()
add_test(NAME TestSimpleTraining COMMAND test_simple_training)

add_executable(test_data_loading ../test_data_loading.cpp)
target_link_libraries(test_data_loading PRIVATE CoreAI3DLib)
if(NOT WIN32)
    target_link_options(test_data_loading PRIVATE -Wl,--allow-multiple-definition)
endif()
add_test(NAME TestDataLoading COMMAND test_data_loading)

add_executable(test_auto_tuning ../test_auto_tuning.cpp)
target_link_libraries(test_auto_tuning PRIVATE CoreAI3DLib)
if(NOT WIN32)
    target_link_options(test_auto_tuning PRIVATE -Wl,--allow-multiple-definition)
endif()
add_test(NAME TestAutoTuning COMMAND test_auto_tuning)

add_executable(test_gradient_accumulation ../test_gradient_accumulation.cpp)
target_link_libraries(test_gradient_accumulation PRIVATE CoreAI3DLib)
if(NOT WIN32)
    target_link_options(test_gradient_accumulation PRIVATE -Wl,--allow-multiple-definition)
endif()
add_test(NAME TestGradientAccumulation COMMAND test_gradient_accumulation)

add_executable(test_gradient_accumulation_performance ../test_gradient_accumulation_performance.cpp)
target_link_libraries(test_gradient_accumulation_performance PRIVATE CoreAI3DLib)
if(NOT WIN32)
    target_link_options(test_gradient_accumulation_performance PRIVATE -Wl,--allow-multiple-definition)
endif()
add_test(NAME TestGradientAccumulationPerformance COMMAND test_gradient_accumulation_performance)

add_executable(test_gold_validation ../test_gold_validation.cpp)
target_link_libraries(test_gold_validation PRIVATE CoreAI3DLib)
if(NOT WIN32)
    target_link_options(test_gold_validation PRIVATE -Wl,--allow-multiple-definition)
endif()
add_test(NAME GoldValidation COMMAND test_gold_validation)
