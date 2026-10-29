# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

# The declared minimum is deliberately *below* the real floor. Raising it to
# 3.18 turns CMP0104 ON, and under that policy CMake pre-populates
# CMAKE_CUDA_ARCHITECTURES with nvcc's default architecture (52 on CUDA 12.6)
# while enabling the CUDA language in project(). The architecture block below
# then cannot tell "the caller asked for 52" from "nobody asked anything", so
# every default build would silently target sm_52. Gate the real floor
# explicitly instead of through the policy level.
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
if(CMAKE_VERSION VERSION_LESS 3.18)
    # Keep the lines short: CMake re-wraps FATAL_ERROR text
    # at roughly 70 columns *including* its own 8-space indent.
    message(FATAL_ERROR
        "dietgpu needs CMake >= 3.18, found\n"
        "${CMAKE_VERSION}.\n"
        "\n"
        "CMAKE_CUDA_ARCHITECTURES and the\n"
        "CUDA_ARCHITECTURES target property only exist\n"
        "from 3.18 on. On older CMake every\n"
        "architecture setting in this file is silently\n"
        "ignored and nvcc falls back to its own\n"
        "default, producing a package that installs\n"
        "but will not run.")
endif()
project(dietgpu LANGUAGES CUDA CXX VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# Resolve the interpreter that will actually *run* dietgpu, then use it to
# locate PyTorch. Never probe a bare `python3`: it resolves via PATH, which
#   - under `--no-build-isolation` is the *system* Python (torch not installed
#     there), so detection fails and find_package(Torch) cannot succeed;
#   - under build isolation is the throwaway build environment, whose torch may
#     be a completely different version/ABI from the runtime one, silently
#     producing an extension that fails to load.
# scikit-build-core passes the target environment's interpreter as
# Python_EXECUTABLE / Python3_EXECUTABLE, so prefer that; VIRTUAL_ENV covers a
# plain `cmake` invocation from an activated environment.
set(DIETGPU_PYTHON_EXECUTABLE "")
foreach(_dietgpu_py ${Python_EXECUTABLE} ${Python3_EXECUTABLE} ${PYTHON_EXECUTABLE})
    if(_dietgpu_py AND NOT DIETGPU_PYTHON_EXECUTABLE)
        set(DIETGPU_PYTHON_EXECUTABLE "${_dietgpu_py}")
    endif()
endforeach()
if(NOT DIETGPU_PYTHON_EXECUTABLE AND DEFINED ENV{VIRTUAL_ENV})
    set(DIETGPU_PYTHON_EXECUTABLE "$ENV{VIRTUAL_ENV}/bin/python")
endif()
if(NOT DIETGPU_PYTHON_EXECUTABLE)
    set(DIETGPU_PYTHON_EXECUTABLE "python3")
endif()

# Auto-detect PyTorch CMAKE_PREFIX_PATH if not set
if(NOT CMAKE_PREFIX_PATH OR CMAKE_PREFIX_PATH STREQUAL "")
    message(STATUS "CMAKE_PREFIX_PATH not set, detecting PyTorch via ${DIETGPU_PYTHON_EXECUTABLE} ...")
    execute_process(
        COMMAND "${DIETGPU_PYTHON_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path, end='')"
        OUTPUT_VARIABLE TORCH_CMAKE_PATH
        RESULT_VARIABLE TORCH_DETECT_RESULT
        ERROR_QUIET
    )
    if(TORCH_DETECT_RESULT EQUAL 0 AND TORCH_CMAKE_PATH)
        set(CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")
        message(STATUS "Auto-detected PyTorch CMake path: ${CMAKE_PREFIX_PATH}")
    else()
        message(WARNING "Could not auto-detect PyTorch. Please set CMAKE_PREFIX_PATH manually.")
    endif()
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Legacy FindCUDA, deliberately kept: it is load-bearing on two fronts --
# dietgpu/utils consumes CUDA_INCLUDE_DIRS / CUDA_LIBRARIES, and the
# architecture probe below uses its CUDA_DETECT_INSTALLED_GPUS. find_package()
# already includes select_compute_arch.cmake, so that macro needs no separate
# include(). Moving to find_package(CUDAToolkit) would mean replacing the probe
# too, which is a migration rather than a simplification.
find_package(CUDA REQUIRED)

# ---- CUDA architecture selection ----
# Resolution order:
#   1. CMAKE_CUDA_ARCHITECTURES supplied by the caller -- always wins,
#      including when it is OFF ("add no arch flags of our own").
#   2. TORCH_CUDA_ARCH_LIST supplied by the caller -- find_package(Torch) below
#      turns it into -gencode flags on CMAKE_CUDA_FLAGS, which reach every CUDA
#      target in this project, so we stand down rather than add a second and
#      possibly different list.
#   3. Local GPU detection.
#
# Detection compiles and runs a probe program, so it can only ever describe the
# machine doing the *build*. It is useless when
#   - that machine has no GPU or no device access (CI, containers, cross
#     compilation), or
#   - the target is a CUDA-compatible non-NVIDIA accelerator (e.g. Alibaba
#     T-Head PPU), whose architecture naming only its own toolchain knows.
# The probe result used to be assigned unconditionally, which silently threw
# away whatever was passed via -DCMAKE_CUDA_ARCHITECTURES or CMAKE_ARGS and
# left no way to target anything but the local GPU. Worse, when the probe sees
# no device CMake does not return an empty list -- it falls back to *every*
# architecture that nvcc supports (8 of them on CUDA 12.6), so a GPU-less build
# machine quietly paid for an all-architecture fatbin instead of failing.
#
# Test DEFINED rather than truthiness: "OFF" is a legitimate explicit value but
# is false in a plain if(), and would have fallen through to detection.
#
# This test only works because cmake_minimum_required above stays below 3.18,
# which keeps CMP0104 OLD. Under CMP0104 NEW, project() pre-populates
# CMAKE_CUDA_ARCHITECTURES with nvcc's default (52 on CUDA 12.6), the test
# becomes permanently true, and every default build silently targets sm_52 --
# verified, not hypothetical. If the declared minimum is ever raised, this block
# must switch to its own variable name for the caller-supplied override.
if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    message(STATUS "dietgpu: honoring caller-supplied "
        "CMAKE_CUDA_ARCHITECTURES='${CMAKE_CUDA_ARCHITECTURES}', "
        "local GPU detection skipped")
elseif(TORCH_CUDA_ARCH_LIST OR DEFINED ENV{TORCH_CUDA_ARCH_LIST})
    # OFF is exactly what caffe2/public/cuda.cmake sets for the same purpose:
    # it makes CMake add no CUDA_ARCHITECTURES flags of its own and leaves the
    # field to torch's -gencode injection.
    # Report whichever of the CMake variable / env var actually carries it --
    # concatenating both prints the value twice when both are set.
    set(DIETGPU_TORCH_ARCH_SRC "${TORCH_CUDA_ARCH_LIST}")
    if(NOT DIETGPU_TORCH_ARCH_SRC)
        set(DIETGPU_TORCH_ARCH_SRC "$ENV{TORCH_CUDA_ARCH_LIST}")
    endif()
    message(STATUS "dietgpu: deferring to caller-supplied "
        "TORCH_CUDA_ARCH_LIST='${DIETGPU_TORCH_ARCH_SRC}', "
        "local GPU detection skipped")
    set(CMAKE_CUDA_ARCHITECTURES OFF)
else()
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    # CUDA_DETECT_INSTALLED_GPUS reports one entry per *device*, not per
    # distinct architecture: an 8x RTX 4090 box yields
    # "89;89;89;89;89;89;89;89". CMake expands every list element into its own
    # --generate-code flag, so each .cu file would be compiled once per GPU for
    # the very same architecture -- an 8x waste on the (very common)
    # homogeneous multi-GPU build machine.
    list(REMOVE_DUPLICATES CUDA_ARCH_LIST)
    if(NOT CUDA_ARCH_LIST)
        # Defensive: CMake 3.28 substitutes its "All" fallback instead of
        # returning empty, so this only trips on other CMake versions.
        # Keep the lines short -- CMake re-wraps FATAL_ERROR text at roughly
        # 70 columns *including* its own 8-space indent, breaking long lines
        # mid-sentence.
        message(FATAL_ERROR
            "No CUDA device was detected here and no\n"
            "target architecture was requested, so\n"
            "dietgpu cannot choose what to compile the\n"
            "kernels for.\n"
            "\n"
            "Pin one and retry:\n"
            "  TORCH_CUDA_ARCH_LIST=8.9\n"
            "  DIETGPU_CUDA_ARCHITECTURES=89\n"
            "See README.md, 'Selecting the GPU\n"
            "Architecture'.")
    endif()
    list(LENGTH CUDA_ARCH_LIST DIETGPU_NUM_ARCH)
    if(DIETGPU_NUM_ARCH GREATER 1)
        message(WARNING "dietgpu: compiling every kernel for "
            "${DIETGPU_NUM_ARCH} architectures "
            "(${CUDA_ARCH_LIST}), which multiplies build time "
            "and binary size accordingly. This usually means no "
            "GPU was visible to the probe and CMake fell back to "
            "all architectures it knows. Pin the target instead: "
            "DIETGPU_CUDA_ARCHITECTURES=89 (CMake numeric form) "
            "or TORCH_CUDA_ARCH_LIST=8.9 (PyTorch dotted form).")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
endif()

# dietgpu/{utils,ans,float} take the arch list from this GLOBAL property, not
# from the variable, so it must be published on every path above -- above all
# the caller-supplied one, or those three targets would be left with an empty
# CUDA_ARCHITECTURES and compile no device code at all.
if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Default build type for a bare `cmake ..`. scikit-build-core always passes
# -DCMAKE_BUILD_TYPE (pyproject sets cmake.build-type = "Release"), so this only
# affects manual builds.
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel.")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---- PyTorch C++ ABI alignment (MUST precede every add_subdirectory) ----
# PyTorch wheels are built with a fixed libstdc++ ABI (torch<=2.x manylinux
# wheels use the pre-C++11 ABI: -D_GLIBCXX_USE_CXX11_ABI=0). Only targets that
# link torch inherit this flag; the pure CUDA/C++ sub-libraries (dietgpu_utils,
# gpu_ans, gpu_float_compress) would otherwise use the compiler default (new ABI
# on gcc>=5), so any std::string crossing the boundary (e.g. dietgpu::errorToName)
# gets a different mangled name -> "undefined symbol" at load time.
#
# TORCH_CXX_FLAGS carries the exact -D_GLIBCXX_USE_CXX11_ABI=<v> that matches the
# installed torch. Reading it (instead of hardcoding 0) keeps every target
# consistent automatically across torch/CUDA versions. Applied to BOTH CXX and
# CUDA so the host side of .cu files uses the same ABI.
find_package(Torch QUIET)
if(NOT Torch_FOUND)
    # Keep every line short: CMake re-wraps FATAL_ERROR text at roughly
    # 70 columns *including* its own 8-space indent, and long lines get
    # broken mid-sentence into unreadable fragments.
    message(FATAL_ERROR
        "PyTorch (Torch CMake package) not found.\n"
        "  CMAKE_PREFIX_PATH: '${CMAKE_PREFIX_PATH}'\n"
        "  probed interpreter:\n"
        "    ${DIETGPU_PYTHON_EXECUTABLE}\n"
        "\n"
        "dietgpu links libtorch directly, so it must be\n"
        "built against the SAME torch that imports it.\n"
        "Otherwise it installs fine but fails on import\n"
        "with 'undefined symbol' (version/ABI mismatch).\n"
        "\n"
        "Build isolation hides your runtime torch:\n"
        "uv repoints VIRTUAL_ENV, PATH and the\n"
        "probed interpreter at the throwaway build\n"
        "env, leaving nothing that names the real\n"
        "one. So isolation is unsupported, and\n"
        "torch is not in build-system.requires on\n"
        "purpose.\n"
        "\n"
        "With uv, from inside the project tree,\n"
        "nothing else is needed: the build tools\n"
        "are listed as runtime dependencies and\n"
        "[tool.uv] turns isolation off, so uv\n"
        "installs them first and then compiles.\n"
        "  uv pip install -e .\n"
        "\n"
        "With pip, or with uv run from outside the\n"
        "tree, install the build deps once and\n"
        "disable isolation yourself:\n"
        "  pip install scikit-build-core ninja cmake\n"
        "  pip install --no-build-isolation -e .\n"
        "  pip install --no-build-isolation dietgpu\n"
        "\n"
        "Or point CMake at your torch explicitly:\n"
        "  export CMAKE_PREFIX_PATH=<torch>/share/cmake"
    )
endif()
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "dietgpu: aligning all targets to TORCH_CXX_FLAGS='${TORCH_CXX_FLAGS}'")

# ---- Install-tree RPATH (MUST precede every add_subdirectory) ----
# The wheel layout splits the extension from its CUDA siblings:
#   dietgpu/libdietgpu.so
#   dietgpu/lib/{libgpu_float_compress,libgpu_ans,libdietgpu_utils,libglog}.so
# libdietgpu.so records those as bare sonames in DT_NEEDED, and CMake *drops*
# the build-tree RUNPATH when a target is installed, so the loader is left
# without any search path: `import dietgpu` then dies with
#   OSError: libgpu_float_compress.so: cannot open shared object file
# An editable install hides this, because __init__.py loads
# build/cp*/lib/libdietgpu.so instead, which still carries its build RUNPATH.
#
# $ORIGIN is the directory of the object being loaded, so one value covers both
# the package dir (needs $ORIGIN/lib) and the lib subdir (needs $ORIGIN).
# DT_RUNPATH is *not* transitive -- it only serves the direct DT_NEEDED of the
# object carrying it -- hence every library needs the entry, not just the
# extension; that is why this is set globally rather than per target.
#
# CMAKE_INSTALL_RPATH_USE_LINK_PATH stays OFF on purpose: it would bake
# absolute build-machine paths (torch/lib, /usr/local/cuda-*/lib64) into the
# installed files, making the package non-relocatable. torch and libcudart are
# already in the process by the time we dlopen, because `import torch` loads
# its libraries with RTLD_GLOBAL first.
set(CMAKE_SKIP_BUILD_RPATH OFF)
set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF)
set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/lib")

# glog must not pull in host-system C++ libraries.
# Above we force the pre-C++11 libstdc++ ABI (to match PyTorch), but distro
# packages such as libgflags are built with the compiler default (new) ABI.
# glog links gflags PUBLIC and instantiates
# google::FlagRegisterer<std::string>, whose mangled name differs between the
# two ABIs (old ABI uses "Ss", new ABI uses "SsB5cxx11"). The result is a
# libglog.so carrying an unresolvable reference into a new-ABI system
# libgflags -- it links (shared libs tolerate undefined symbols) but fails at
# dlopen time, and glog's own unittests fail to link outright.
# dietgpu never uses gflags itself, so drop the dependency; glog falls back to
# its built-in flag implementation. glog's unittests are skipped too, since they
# are not part of this project's test suite. BUILD_TESTING is restored
# afterwards so googletest and our own tests are unaffected.
set(WITH_GFLAGS OFF CACHE BOOL "" FORCE)
set(WITH_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF)
add_subdirectory(third_party/glog)
set(BUILD_TESTING ON)

# googletest is needed only by the C++ test executables in dietgpu/{ans,float}.
# Building it costs four shared libraries (gtest, gtest_main, gmock,
# gmock_main) on every configure, and it is also the source of the
# include/gtest/** + lib/libgmock*.so files that leak into a wheel -- the
# reason pyproject.toml needs install.components = ["python"]. Turn this OFF to
# skip both; enable_testing() and GoogleTest (which provides
# gtest_discover_tests to the subdirectories) are enabled from here so the
# sub-projects do not each have to repeat it.
option(DIETGPU_BUILD_TESTS "Build the C++ gtest executables" ON)
if(DIETGPU_BUILD_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(third_party/googletest)
endif()

add_subdirectory(dietgpu)
add_subdirectory(dietgpu/utils)
add_subdirectory(dietgpu/ans)
add_subdirectory(dietgpu/float)

# Installation rules.
# SKBUILD is set by scikit-build-core. In that case the layout has to mirror
# what dietgpu/__init__.py searches for: the torch extension next to the Python
# package, its CUDA siblings in a lib/ subdirectory (see the RPATH block above).
# Every rule is COMPONENT python, which is what lets
# install.components = ["python"] in pyproject.toml drop the sub-projects' own
# rules from the wheel.
# glog declares PUBLIC_HEADER, so installing only its LIBRARY makes CMake warn
# on every configure. Only the runtime .so belongs in the wheel / prefix -- the
# headers are build-time only and are reachable through glog's own CMake package
# in a development build -- so clear the property rather than give it a
# destination. Applies to both branches below.
set_target_properties(glog PROPERTIES PUBLIC_HEADER "")
if(SKBUILD)
    install(TARGETS dietgpu
        LIBRARY DESTINATION dietgpu
        COMPONENT python
    )
    install(TARGETS gpu_float_compress gpu_ans dietgpu_utils glog
        LIBRARY DESTINATION dietgpu/lib
        COMPONENT python
    )
else()
    # Regular cmake build (for development)
    install(TARGETS dietgpu gpu_float_compress gpu_ans dietgpu_utils glog
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()
