#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOLCHAIN_BIN="${TOOLCHAIN_BIN:-/home/mafei/host-tools/gcc/riscv64-linux-musl-x86_64/bin}"
SDK_ROOT="${SDK_ROOT:-/home/mafei/LicheeRV_Nano_Build/LicheeRV-Nano-Build-20260114}"
OPENCV_DIR="${OPENCV_DIR:-/home/mafei/Desktop/QuadrotorPilot/QuadrotorPilot_identify_small/opencv/opencv/install-licheerv/lib/cmake/opencv5}"
BOARD_LIBS="${BOARD_LIBS:-/home/mafei/Desktop/QuadrotorPilot/QuadrotorPilot_identify_small/build-nano/lib}"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/build-nano}"
PREFIX="$TOOLCHAIN_BIN/riscv64-unknown-linux-musl-"
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="${PREFIX}gcc" -DCMAKE_CXX_COMPILER="${PREFIX}g++" -DCMAKE_AR="${PREFIX}ar" -DCMAKE_RANLIB="${PREFIX}ranlib" -DSDK_ROOT="$SDK_ROOT" -DOPENCV_DIR="$OPENCV_DIR" -DBOARD_LIBS="$BOARD_LIBS"
cmake --build "$BUILD_DIR" --parallel "$(nproc)"
printf 'Nano executables:\n  %s/aim_light_test\n  %s/camera_compressed_stream_test\n  %s/camera_calibration_stream\n  %s/main\n  %s/main_test\n' "$BUILD_DIR" "$BUILD_DIR" "$BUILD_DIR" "$BUILD_DIR" "$BUILD_DIR"
# Stage runtime dependencies beside the executable for simple board deployment.
mkdir -p "$BUILD_DIR/lib"
for library in libsample.so libvpu.so libsys.so libvenc.so libvdec.so libmisc.so \
    libsns_full.so libini.so libisp.so libisp_algo.so libae.so libaf.so libawb.so \
    libcvi_bin_isp.so libcvi_bin.so libsns_gc4653.so libjson-c.so libjson-c.so.5 \
    libjson-c.so.5.1.0; do
    if [ -f "$BOARD_LIBS/$library" ]; then cp -a "$BOARD_LIBS/$library" "$BUILD_DIR/lib/"; fi
done
OPENCV_LIB_DIR="$(dirname "$OPENCV_DIR")/../../lib"
for module in core flann imgproc geometry features objdetect video; do
    for candidate in "libopencv_${module}.so.5.1.0" "libopencv_${module}.so.501" "libopencv_${module}.so"; do
        if [ -f "$OPENCV_LIB_DIR/$candidate" ]; then
            cp -a "$OPENCV_LIB_DIR/$candidate" "$BUILD_DIR/lib/"
        fi
    done
done
printf 'Runtime bundle: %s\n' "$BUILD_DIR/lib"

# OpenCV 的动态依赖使用 .so.501 SONAME，部署包中显式补齐对应链接。
for module in core flann imgproc geometry features objdetect video; do
    versioned="$BUILD_DIR/lib/libopencv_${module}.so.5.1.0"
    soname="$BUILD_DIR/lib/libopencv_${module}.so.501"
    if [ -f "$versioned" ] && [ ! -e "$soname" ]; then
        ln -sf "libopencv_${module}.so.5.1.0" "$soname"
    fi
done
