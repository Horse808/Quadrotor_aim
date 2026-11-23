#pragma once

#include <opencv2/core.hpp>

// 引导灯检测参数。HSV 的 H 范围是 OpenCV 的 0~179。
struct LightDetectorParams {
    int binary_threshold = 200;       // 灰度兼容模式阈值
    int min_area = 10;
    int max_area = 5000;

    float min_circularity = 0.55f;
    float max_aspect_ratio_diff = 0.30f;

    int green_hue_min = 35;
    int green_hue_max = 95;
    int green_saturation_min = 60;
    int green_value_min = 45;
    float min_green_ratio = 0.45f;
};

struct LightResult {
    bool found = false;
    cv::Point2f center{0.f, 0.f};
    cv::Rect bounding_box{};
    float radius = 0.0f;
    float area = 0.0f;
    float circularity = 0.0f;
    float aspect_ratio = 0.0f;
    float green_ratio = 0.0f;
    float confidence = 0.0f;
    float estimated_distance = -1.0f;
};

// 支持 CV_8UC3 BGR（优先，执行绿色分割）和 CV_8UC1 灰度兼容输入。
LightResult detectLight(const cv::Mat& image,
                        const LightDetectorParams& params,
                        float physical_diameter_m = 0.055f,
                        float camera_focal_px = 0.0f);
