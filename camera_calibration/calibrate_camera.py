#!/usr/bin/env python3
"""Calibrate a pinhole camera from checkerboard images in photo0."""

from pathlib import Path
import argparse
import glob

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", nargs=2, type=int, default=(9, 6),
                        metavar=("COLS", "ROWS"),
                        help="checkerboard inner corners, default: 9 6")
    parser.add_argument("--square-size", type=float, default=0.025,
                        help="checkerboard square size in meters, default: 0.025")
    parser.add_argument("--input", type=Path,
                        default=Path(__file__).resolve().parent / "photo0")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "camera_calibration.yaml")
    args = parser.parse_args()

    cols, rows = args.pattern
    if cols < 2 or rows < 2 or args.square_size <= 0:
        parser.error("pattern dimensions must be >= 2 and square size must be positive")

    files = sorted(glob.glob(str(args.input / "*.png")) +
                   glob.glob(str(args.input / "*.jpg")) +
                   glob.glob(str(args.input / "*.jpeg")))
    if not files:
        print(f"No calibration images found in {args.input}")
        return 1

    object_points = np.zeros((cols * rows, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    object_points *= args.square_size

    object_points_all = []
    image_points_all = []
    image_size = None
    used = 0

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for filename in files:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"skip unreadable image: {filename}")
            continue
        current_size = (image.shape[1], image.shape[0])
        if image_size is not None and current_size != image_size:
            print(f"skip size mismatch: {filename} ({current_size}, expected {image_size})")
            continue
        image_size = current_size
        found, corners = cv2.findChessboardCorners(
            image, (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            print(f"no board: {filename}")
            continue
        corners = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
        object_points_all.append(object_points.copy())
        image_points_all.append(corners)
        used += 1
        print(f"use: {filename}")

    if used < 6 or image_size is None:
        print(f"Need at least 6 usable images; found {used}")
        return 1

    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points_all, image_points_all, image_size, None, None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(args.output), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", image_size[0])
    fs.write("image_height", image_size[1])
    fs.write("camera_matrix", camera_matrix)
    fs.write("dist_coeffs", dist_coeffs)
    fs.write("reprojection_error", float(rms))
    fs.write("checkerboard_cols", cols)
    fs.write("checkerboard_rows", rows)
    fs.write("square_size_m", args.square_size)
    fs.release()

    print(f"Used {used}/{len(files)} images")
    print(f"RMS reprojection error: {rms:.4f} pixels")
    print(f"Wrote {args.output}")
    print("Remember to scale camera_matrix if the runtime image resolution differs from the calibration resolution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
