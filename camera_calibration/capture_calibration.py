#!/usr/bin/env python3
"""Preview Nano's calibration stream and save selected frames into photo0."""

from pathlib import Path
import socket
import sys

import cv2
import numpy as np

HOST = "10.219.85.1"
PORT = 5002
WINDOW_NAME = "Nano camera calibration"
PHOTO_DIR = Path(__file__).resolve().parent / "photo0"


def receive_jpeg(sock: socket.socket, buffer: bytearray):
    while True:
        start = buffer.find(b"\xff\xd8")
        if start >= 0:
            end = buffer.find(b"\xff\xd9", start + 2)
            if end >= 0:
                frame = bytes(buffer[start : end + 2])
                del buffer[: end + 2]
                return frame
            if start > 0:
                del buffer[:start]
        data = sock.recv(256 * 1024)
        if not data:
            raise ConnectionError("Nano closed the stream")
        buffer.extend(data)
        if len(buffer) > 8 * 1024 * 1024:
            buffer.clear()


def photo_index(path: Path) -> int:
    try:
        return int(path.stem.removeprefix("photo"))
    except ValueError:
        return -1


def next_photo_path() -> Path:
    existing = sorted(PHOTO_DIR.glob("photo*.png"))
    index = 0
    while (PHOTO_DIR / f"photo{index}.png").exists():
        index += 1
    return PHOTO_DIR / f"photo{index}.png"


def main() -> int:
    global HOST
    if len(sys.argv) >= 2:
        HOST = sys.argv[1]

    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Connecting to {HOST}:{PORT} ...")
    try:
        with socket.create_connection((HOST, PORT), timeout=10) as sock:
            sock.settimeout(10)
            buffer = bytearray()
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            print("Controls: SPACE/S = save image, D = delete last image, Q/ESC = quit")

            saved = 0
            while True:
                try:
                    jpeg = receive_jpeg(sock, buffer)
                except socket.timeout:
                    print("Stream timeout; check the Nano process and network.")
                    return 1
                image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.putText(display, f"saved={saved}  SPACE/S: save  D: delete  Q: quit",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("s"), ord("S"), 32):
                    path = next_photo_path()
                    if cv2.imwrite(str(path), image):
                        saved += 1
                        print(f"Saved {path}")
                    else:
                        print(f"Failed to save {path}")
                elif key in (ord("d"), ord("D")):
                    paths = sorted(PHOTO_DIR.glob("photo*.png"), key=photo_index)
                    if paths:
                        paths[-1].unlink()
                        saved = max(0, saved - 1)
                        print(f"Deleted {paths[-1]}")

    except (OSError, ConnectionError) as exc:
        print(f"Capture failed: {exc}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
