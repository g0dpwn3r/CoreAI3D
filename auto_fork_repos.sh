#!/bin/bash
# auto_fork_repos.sh
# Fork multiple repos to your GitHub account and generate REPOS array for vcpkg

# GitHub username and token
GITHUB_USER="g0dpwn3r"
GITHUB_TOKEN="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"

# Output array
echo "Generating REPOS array..."

echo "REPOS=("

# List of repositories to fork
# Format: "repo_name github_org_or_user [comma-separated dependencies]"
REPO_LIST=(
  "abseil google/abseil-cpp"
  "utf8-range g0dpwn3r/utf8-range"
  "vcpkg-cmake microsoft/vcpkg-cmake"
  "vcpkg-cmake-config microsoft/vcpkg-cmake-config"
  "boost-filesystem boostorg/filesystem boost,boost-system"
  "boost-system boostorg/system"
  "boost-regex boostorg/regex boost"
  "boost-asio boostorg/asio boost"
  "boost-beast boostorg/beast boost,boost-asio"
  "boost-program-options boostorg/program_options boost"
  "catch2 catchorg/Catch2"
  "mysql-connector-cpp mysql/mysql-connector-cpp boost,openssl,protobuf,zlib"
  "nlohmann-json nlohmann/json"
  "curl curl/curl"
  "google-cloud-cpp googleapis/google-cloud-cpp protobuf,abseil"
  "grpc grpc/grpc protobuf,abseil"
  "gtest google/googletest"
  "openssl openssl/openssl"
  "protobuf protocolbuffers/protobuf"
  "zlib madler/zlib"
  "zstd facebook/zstd"
)

for entry in "${REPO_LIST[@]}"; do
    IFS=' ' read -r NAME ORG_OR_USER DEPS <<< "$entry"

    # Fork the repository via GitHub API
    echo "Forking $ORG_OR_USER -> $GITHUB_USER/$NAME"
    curl -s -X POST \
        -u "$GITHUB_USER:$GITHUB_TOKEN" \
        https://api.github.com/repos/$ORG_OR_USER/forks \
        -d "{\"organization\":\"$GITHUB_USER\"}" > /dev/null

    # Construct REPOS line
    if [ -n "$DEPS" ]; then
        echo "  \"$NAME https://github.com/$GITHUB_USER/$NAME $DEPS\""
    else
        echo "  \"$NAME https://github.com/$GITHUB_USER/$NAME\""
    fi
done

echo ")"
