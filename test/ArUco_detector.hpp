#pragma once

#include <opencv2/core.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <vector>

enum DetectionMode { MODE_LOST, MODE_DETECT, MODE_TRACK, MODE_CLAHE, MODE_OCCLUDED, MODE_SEARCH };
struct ArucoParams {
    int target_id = 0;
    bool enable_roi = true;
    bool enable_clahe = false;
    bool enable_optical_flow = false;
    float roi_padding_scale = 0.8f;
    int roi_min_margin = 24;
    int max_occluded_frames = 3;
    int max_search_frames = 8;
};
struct DetectorState {
    bool has_prev = false;
    int occluded_frames = 0;
    int search_frames = 0;
    cv::Size prev_size{};
    std::vector<cv::Point2f> prev_corners;
    cv::Mat prev_gray;
    cv::Point2f last_search_dir{};
};
struct ArucoResult {
    bool found = false;
    int id = -1;
    DetectionMode mode = MODE_LOST;
    std::vector<cv::Point2f> corners;
    std::vector<bool> estimated;
    cv::Point2f center{};
    float yaw_rad = 0.f;
    std::vector<int> occluded_idx;
    std::vector<int> out_of_view_idx;
    cv::Point2f search_dir_img{};
    cv::Point2f search_dir_body{};
};

cv::aruco::DetectorParameters makeDetectorParams();
ArucoResult processFrame(const cv::Mat& gray,
                         const cv::aruco::ArucoDetector& detector,
                         const ArucoParams& params,
                         DetectorState& state);
