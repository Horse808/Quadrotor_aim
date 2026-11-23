#pragma once

#include <memory>
#include <string>

namespace cv {
class Mat;
}

class Camera {
public:
    // VPSS 输出格式。默认保持识别主程序原来的 NV21 行为；需要
    // YUV420P 消费者时可显式选择平面格式。
    enum class OutputFormat {
        Nv21,
        Yuv420Planar,
    };

    Camera();
    ~Camera();

    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;

    // width/height 是传感器 VI 输入分辨率；output_width/output_height 是
    // 送给识别算法的图像分辨率。后两项为 0 时默认使用 1/2 尺寸。
    bool open(int device_id,
              int width,
              int height,
              int fps,
              int output_width = 0,
              int output_height = 0,
              OutputFormat output_format = OutputFormat::Nv21);
    bool readGray(cv::Mat& out_gray, int timeout_ms = 1000);         // 读取一帧图像的 Y 平面
    bool readBgr(cv::Mat& out_bgr, int timeout_ms = 1000);            // 读取一帧并转换为 BGR
    void close();                                                    // 关闭相机，释放所有硬件资源
    void shutdown();                                                 // 同上
    bool isOpened() const;
    bool usesHardwareScaledOutput() const;                           // 告诉主程序当前是否正在使用 VPSS 硬件缩放
    const std::string& lastError() const;                            // 判断错误原因

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
