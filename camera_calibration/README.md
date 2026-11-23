# 相机标定图像采集

这个目录包含两部分：

1. `camera_calibration_stream`：运行在 LicheeRV Nano 上，使用 GC4653 的原生 `1280x720` 图像，以灰度 MJPEG 发送到电脑。
2. `capture_calibration.py`：运行在电脑上，显示 Nano 图像，按键选择截图，并把图片保存到本目录的 `photo0/`。

另外提供 `calibrate_camera.py`，可以用 `photo0/` 中的棋盘格照片计算相机内参和畸变系数。

## 一、Nano 端

编译项目后部署：

```text
build-nano/camera_calibration_stream
build-nano/lib/
```

在 Nano 项目目录中运行：

```bash
LD_LIBRARY_PATH="$PWD/lib:/mnt/system/usr/lib/3rd:/mnt/system/usr/lib:/mnt/system/lib" \
./camera_calibration_stream
```

默认参数直接写在 `camera_calibration_stream.cpp`：

```cpp
kStreamPort = 5002;
kStreamFps = 10;
kJpegQuality = 85;
```

这个程序固定使用相机原生 `1280x720` 采集，不能和 `aim_light_test` 或其他相机程序同时运行。

## 二、电脑端预览和截图

电脑需要安装 Python 3、OpenCV 和 NumPy。进入本目录后运行：

```bash
python3 capture_calibration.py 10.219.85.1
```

如果电脑使用其他 Nano IP，把命令最后的 IP 改成实际地址。

窗口按键：

- `SPACE` 或 `S`：保存当前图像到 `photo0/photoN.png`；
- `D`：删除最近保存的一张图；
- `Q` 或 `ESC`：退出。

截图时建议让棋盘格覆盖画面不同位置、距离和倾角，至少采集 15～20 张有效图片。不要连续保存几乎相同的画面。

## 三、计算标定参数

默认棋盘格为 **9×6 个内角点**，单个方格边长为 **25 mm**。如果棋盘格规格不同，需要修改参数：

```bash
python3 calibrate_camera.py --pattern 9 6 --square-size 0.025
```

例如棋盘格是 8×5 个内角点、方格 30 mm：

```bash
python3 calibrate_camera.py --pattern 8 5 --square-size 0.03
```

结果默认保存为：

```text
camera_calibration.yaml
```

标定输出分辨率是照片的原始分辨率 `1280x720`。如果识别程序使用 `640x360` 或其他尺寸，需要按照宽高缩放比例同步缩放相机内参矩阵；畸变系数通常保持不变。
