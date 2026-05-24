# Image Segmentation with CUDA

image-segmentation is a C++ application for benchmarking image segmentation algorithms on CPU and GPU (CUDA).  
The project uses Qt for the graphical interface, CUDA for GPU acceleration, and OpenCV for image processing.

## Requirements

Before building the project, install the following components manually:

### Visual Studio 2022

Install Visual Studio 2022 with the `Desktop development with C++` workload enabled.

### CUDA Toolkit

Install the NVIDIA CUDA Toolkit compatible with your GPU drivers.

After installation:
- verify that the `nvcc` compiler is available in the system `PATH`
- ensure that CUDA integration for Visual Studio is installed correctly

You can verify the installation with:

```bash
nvcc --version
```

### Qt 6.x

Install Qt 6 using the Qt Online Installer.

Only the following component is required:

```text
Desktop MSVC 2022 64-bit
```

After installation, configure the `CMAKE_PREFIX_PATH` environment variable so CMake can locate the Qt installation.

Example:

```bash
$env:CMAKE_PREFIX_PATH=C:/Qt/6.6.2/msvc2022_64
```

## Cloning the Repository

The project uses `vcpkg` as a Git submodule for dependency management.  
Clone the repository recursively:

```bash
git clone --recursive https://github.com/kluczek3/image-segmentation.git
cd image-segmentation
```

If the repository was already cloned without submodules initialized, run:

```bash
git submodule update --init --recursive
```

## Building the Project

Open the repository root directory in Visual Studio 2022.

The project is configured with CMake Manifest Mode.  
During the first configuration step, Visual Studio automatically invokes `vcpkg` and builds the required OpenCV modules from source.

This process may take several minutes on the first run.

Supported build configurations:

```text
x64-Debug
x64-Release
```

Build the solution using:

```text
Build → Build All
```

The generated executable will be located in:

```bash
out/build/x64-<configuration>/
```

## Notes

This project requires both:
- Qt 6.x
- NVIDIA CUDA Toolkit

The application will not compile correctly if either dependency is missing or improperly configured.
