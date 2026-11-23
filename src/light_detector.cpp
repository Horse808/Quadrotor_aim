#include "light_detector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry/2d.hpp>

namespace {
bool validParams(const LightDetectorParams& p) {
    return p.binary_threshold >= 0 && p.binary_threshold <= 255 &&
           p.min_area >= 0 && p.max_area >= p.min_area &&
           std::isfinite(p.min_circularity) && p.min_circularity >= 0.0f && p.min_circularity <= 1.0f &&
           std::isfinite(p.max_aspect_ratio_diff) && p.max_aspect_ratio_diff >= 0.0f &&
           p.green_hue_min >= 0 && p.green_hue_min <= 179 &&
           p.green_hue_max >= 0 && p.green_hue_max <= 179 &&
           p.green_hue_min <= p.green_hue_max &&
           p.green_saturation_min >= 0 && p.green_saturation_min <= 255 &&
           p.green_value_min >= 0 && p.green_value_min <= 255 &&
           std::isfinite(p.min_green_ratio) && p.min_green_ratio >= 0.0f && p.min_green_ratio <= 1.0f;
}

float clamp01(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}
}  // namespace

LightResult detectLight(const cv::Mat& image,
                        const LightDetectorParams& params,
                        float physical_diameter_m,
                        float camera_focal_px) {
    LightResult result;
    if (image.empty() || (image.type() != CV_8UC1 && image.type() != CV_8UC3) ||
        !validParams(params) || !std::isfinite(physical_diameter_m) || physical_diameter_m <= 0.0f ||
        !std::isfinite(camera_focal_px) || camera_focal_px < 0.0f) {
        return result;
    }

    cv::Mat mask;
    bool color_mode = image.type() == CV_8UC3;
    if (color_mode) {
        cv::Mat hsv;
        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv,
                    cv::Scalar(params.green_hue_min, params.green_saturation_min, params.green_value_min),
                    cv::Scalar(params.green_hue_max, 255, 255), mask);
    } else {
        // 仅保留旧接口兼容；真正的绿色识别必须使用 CV_8UC3 输入。
        cv::threshold(image, mask, params.binary_threshold, 255, cv::THRESH_BINARY);
    }

    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    float best_score = -std::numeric_limits<float>::infinity();
    for (const auto& contour : contours) {
        const float area = static_cast<float>(cv::contourArea(contour));
        if (!std::isfinite(area) || area < params.min_area || area > params.max_area) continue;

        const float perimeter = static_cast<float>(cv::arcLength(contour, true));
        if (!std::isfinite(perimeter) || perimeter <= 0.0f) continue;
        const float circularity = 4.0f * static_cast<float>(CV_PI) * area / (perimeter * perimeter);
        if (!std::isfinite(circularity) || circularity < params.min_circularity) continue;

        const cv::Rect box = cv::boundingRect(contour);
        if (box.width <= 0 || box.height <= 0) continue;
        const float aspect = static_cast<float>(box.width) / static_cast<float>(box.height);
        if (!std::isfinite(aspect) || std::abs(aspect - 1.0f) > params.max_aspect_ratio_diff) continue;

        float green_ratio = 0.0f;
        if (color_mode) {
            // The contour already came from this mask. Count the mask pixels in
            // its bounding ROI instead of rasterizing the contour a second time.
            // This keeps the score tied to the actual post-morphology mask and
            // avoids a per-candidate allocation plus fillPoly call on Nano.
            const cv::Mat mask_roi = mask(box);
            const int box_pixels = box.area();
            green_ratio = box_pixels > 0
                              ? static_cast<float>(cv::countNonZero(mask_roi)) /
                                    static_cast<float>(box_pixels)
                              : 0.0f;
            if (green_ratio < params.min_green_ratio) continue;
        }

        const cv::Moments moments = cv::moments(contour);
        if (!std::isfinite(moments.m00) || moments.m00 <= 0.0) continue;
        const cv::Point2f center(static_cast<float>(moments.m10 / moments.m00),
                                 static_cast<float>(moments.m01 / moments.m00));
        if (!std::isfinite(center.x) || !std::isfinite(center.y)) continue;

        // 颜色模式下优先选择“绿色、圆、面积合理”的目标，而不是盲选最大面积。
        const float circularity_score = clamp01(circularity);
        const float area_score = clamp01(std::log1p(area) / std::log1p(static_cast<float>(std::max(1, params.max_area))));
        const float score = color_mode
                                ? 0.55f * green_ratio + 0.30f * circularity_score + 0.15f * area_score
                                : 0.70f * circularity_score + 0.30f * area_score;
        if (score <= best_score) continue;

        best_score = score;
        result.found = true;
        result.center = center;
        result.bounding_box = box;
        result.area = area;
        result.radius = std::sqrt(area / static_cast<float>(CV_PI));
        result.circularity = circularity;
        result.aspect_ratio = aspect;
        result.green_ratio = green_ratio;
        result.confidence = clamp01(score);
        if (camera_focal_px > 0.0f && result.radius > 0.5f) {
            result.estimated_distance = (camera_focal_px * physical_diameter_m) / (2.0f * result.radius);
        }
    }
    return result;
}
