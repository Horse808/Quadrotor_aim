#pragma once

#include <opencv2/core.hpp>

// 配置始终保存原始标定内参；按实际算法图像尺寸缩放，避免与预览尺寸混淆。
// 分辨率或矩阵无效时清空输出并拒绝使用，不能静默猜测内参参考尺寸。
inline bool readScaledCameraMatrix(const cv::FileStorage& file,
                                   cv::Size image_size,
                                   cv::Mat& matrix) {
    matrix.release();
    const cv::FileNode width = file["calibration_image_width"];
    const cv::FileNode height = file["calibration_image_height"];
    if (!width.isInt() || !height.isInt() ||
        static_cast<int>(width) <= 0 || static_cast<int>(height) <= 0 ||
        image_size.width <= 0 || image_size.height <= 0) {
        return false;
    }

    cv::Mat scaled;
    try {
        file["camera_matrix"] >> scaled;
        if (scaled.empty() || scaled.rows != 3 || scaled.cols != 3 ||
            scaled.channels() != 1) {
            return false;
        }
        scaled.convertTo(scaled, CV_64F);
        if (!cv::checkRange(scaled) || scaled.at<double>(0, 0) <= 0.0 ||
            scaled.at<double>(1, 1) <= 0.0 || scaled.at<double>(2, 0) != 0.0 ||
            scaled.at<double>(2, 1) != 0.0 || scaled.at<double>(2, 2) != 1.0) {
            return false;
        }
        scaled.row(0) *= static_cast<double>(image_size.width) / static_cast<int>(width);
        scaled.row(1) *= static_cast<double>(image_size.height) / static_cast<int>(height);
        if (!cv::checkRange(scaled)) return false;
    } catch (const cv::Exception&) {
        return false;
    }
    matrix = scaled;
    return true;
}
