# CoreAI3D

CoreAI3D is a C++ neural core engine and tooling stack for training and serving AI models. The project uses CMake for cross-platform builds, targets C++20, and is designed to build on Windows (Visual Studio) and Linux/WSL (Ninja). Protobuf, Abseil, Boost, MySQL connector and other libraries are consumed via `vcpkg`.

Highlights
- C++20 codebase
- CMake-based build (minimum CMake 3.20; tested with 3.31.x)
- Supports Visual Studio 2022 and Ninja generators
- Uses `vcpkg` for dependency management
- Protobuf code generation integrated into the build
- Packaging with CPack (NSIS / DEB) and a Dockerfile for Linux builds
- Tests: GTest and Boost.Test may be present in the tree

Getting started

Prerequisites
- Windows: Visual Studio 2022 ("Desktop development with C++" workload) or install the MSVC toolchain and CMake
- Linux / WSL: build-essential, cmake, git and required libs (see `Dockerfile` for a sample apt install list)
- CMake >= 3.20 (3.31 recommended)
- vcpkg (optional but recommended) — set `VCPKG_ROOT` env var or place vcpkg next to the repo

Clone

```bash
git clone https://github.com/g0dpwn3r/CoreAI3D.git
cd CoreAI3D
```

Using vcpkg

If you use `vcpkg` place it in the repo root or set `VCPKG_ROOT` and pass the toolchain to CMake:

```bash
cmake -S . -B out/build -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
```

Build (Linux / WSL, Ninja)

```bash
mkdir -p out/build/linux
cd out/build/linux
cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug ../.. \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Debug
```

Build (Windows Visual Studio)

Open a "x64 Native Tools Command Prompt for VS 2022" or use Visual Studio UI:

CLI (VS generator):

```powershell
cmake -S . -B out/build/vs -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake
cmake --build out/build/vs --config Debug
```

Tip: the workspace also includes `CMakeSettings.json` which defines `Windows-Debug` and `WSL-Debug` configurations for Visual Studio.

Troubleshooting

- "Could not find specified instance of Visual Studio"
  - This commonly occurs when a cached CMake variable such as `CMAKE_GENERATOR_INSTANCE` contains a bad path (typo like `Viual Studio`). Fix:
    - Delete the CMake cache (`CMakeCache.txt` and `CMakeFiles/`) and reconfigure.
    - Ensure Visual Studio 2022 is installed and registered in the Visual Studio Installer.
    - Avoid forcing the generator from inside `CMakeLists.txt`; let the caller/IDE select the generator.

- MSBuild `GetOutOfDateItems` OutOfMemoryException ("System.OutOfMemoryException")
  - Cause: extremely large command-line / huge inline file lists passed to `add_executable` (for example thousands of generated protobuf sources) cause MSBuild tasks to assemble very large strings and exhaust process memory.
  - Mitigations:
    - Use an intermediate library target for generated sources (`add_library(proto_objs ...)`) and `target_link_libraries` the library into the executable. This avoids embedding huge file lists on the linker command line.
    - Use the Ninja generator for heavy code-generation builds (Ninja uses less MSBuild machinery and avoids MSBuild OOM issues).
    - Increase machine RAM / pagefile while building.
    - Split generated sources into smaller logical libraries if possible.

- CMake cache / bad generator selection
  - If you suspect a cached bad generator instance, remove cache files and re-run CMake with the generator you want. Example:

```bash
rm -rf out/build/* CMakeCache.txt CMakeFiles
cmake -S . -B out/build -G "Visual Studio 17 2022" -A x64
```

Testing

If the repository contains tests (GTest / Boost.Test), configure and build tests and run them with CTest:

```bash
cmake -S . -B out/build -DBUILD_TESTING=ON
cmake --build out/build --target RUN_TESTS
# or
ctest --test-dir out/build -C Debug -V
```

Packaging

CPack is configured in the top-level `CMakeLists.txt`. Example generators include NSIS (Windows) and DEB (Linux). After a successful build:

```bash
cd out/build
cpack
```

Docker

A sample `Dockerfile` is included to create a Linux build environment. Build and run:

```bash
docker build -t coreai3d:latest .
docker run --rm coreai3d:latest
```

Developer notes

- The top-level `CMakeLists.txt` was adjusted to avoid forcing a Visual Studio generator from inside CMake and to detect/unset a cached `CMAKE_GENERATOR_INSTANCE` that contains obvious typos. If you maintain custom IDE integration, prefer setting the generator from your tooling rather than from the project file.
- Generated Protobuf sources are collected into `proto_objs` and linked into the main target to reduce command-line length and MSBuild memory pressure.

Contributing

Please open issues or pull requests on the upstream repository. When contributing changes that affect build or packaging, include platform-specific verification steps (Windows/Ninja/WSL).

License

See the `LICENSE` file at the repository root for license details.

Contact

Project maintainer: Carlon <carlonvanspijker@gmail.com>

---

If you want, I can add a short `DEVELOPMENT.md` with detailed instructions for common dev flows (proto generation, vcpkg bootstrap, helping Visual Studio detect instances).
