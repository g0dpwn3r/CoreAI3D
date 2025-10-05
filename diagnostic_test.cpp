// Diagnostic test to validate type visibility hypotheses
#include "CoreAI3D/include/SystemModule.hpp"
#include "CoreAI3D/include/WebModule.hpp"
#include <iostream>

int main() {
    // Test 1: Can we use ProcessInfo globally?
    ProcessInfo pi;
    pi.processId = 123;
    std::cout << "ProcessInfo global access: OK" << std::endl;

    // Test 2: Can we use it in a vector?
    std::vector<ProcessInfo> processes;
    processes.push_back(pi);
    std::cout << "std::vector<ProcessInfo>: OK" << std::endl;

    // Test 3: Does SystemModule::ProcessInfo exist?
    // This should fail to compile if my hypothesis is correct
    // SystemModule::ProcessInfo should_not_exist;

    // Test 4: Same for WebModule types
    SearchResult sr;
    sr.title = "test";
    std::cout << "SearchResult global access: OK" << std::endl;

    std::vector<SearchResult> results;
    results.push_back(sr);
    std::cout << "std::vector<SearchResult>: OK" << std::endl;

    std::cout << "All diagnostic tests passed!" << std::endl;
    return 0;
}