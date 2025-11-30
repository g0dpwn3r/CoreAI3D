// tests/DatabaseTests.cpp
#define CATCH_CONFIG_FAST_COMPILE
#include <catch2/catch_all.hpp>
#include <cstdlib>

#include "Database.hpp"

TEST_CASE("Database smoke and dataset flow (requires real DB)", "[database][!mayfail]") {
    const char* run_db = std::getenv("RUN_DB_TESTS");
    if (!run_db || std::string(run_db) != "1") {
        WARN("Skipping database tests because RUN_DB_TESTS!=1");
        return;
    }

    // configure these to match your local test DB or CI DB container
    const std::string host = "0.0.0.0";
    const unsigned int port = 33060;
    const std::string user = "root";
    const std::string password = "password";
    const std::string schema = "coreai3d_test";

    // Try to create a Database instance; if it throws, fail.
    Database db(host, port, user, password, schema, SSLMode::DISABLED, true);

    SECTION("Create tables and CRUD dataset") {
        REQUIRE_NOTHROW(db.createTables());
        int datasetId = db.addDataset("test_dataset", "desc", 2, 3, 2);
        REQUIRE(datasetId > 0);

        // insert a couple of rows (should not throw)
        std::vector<float> f1 = { 0.1f, 0.2f, 0.3f };
        std::vector<float> l1 = { 1.0f, 0.0f };
        REQUIRE_NOTHROW(db.addDatasetRecord(datasetId, 0, f1, l1));

        auto data = db.getDataset(datasetId);
        REQUIRE(data.inputs.size() >= 1);
    }
}
