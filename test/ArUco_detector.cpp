#include "ArUco_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>


#include <algorithm>
#include <cmath>

// ── ArUco 检测器参数 ───────────────────────────────────────────────────────
cv::aruco::DetectorParameters makeDetectorParams() {
    cv::aruco::DetectorParameters params;
    // 减少自适应阈值的处理轮次可显著降低 Nano 上的计算开销。
    params.adaptiveThreshWinSizeMin     = 7;                                 // 自适应二值化窗口大小
    params.adaptiveThreshWinSizeMax     = 31;
    params.adaptiveThreshWinSizeStep    = 8;
    params.polygonalApproxAccuracyRate  = 0.08;                              // 把不规则的轮廓拟合成四边形时的容差率
    params.minMarkerPerimeterRate       = 0.02;                              // 码的周长占画面周长的比例
    params.maxMarkerPerimeterRate       = 4.0;                               // 同上
    params.errorCorrectionRate          = 0.7;                               // 纠错率
    params.cornerRefinementMethod       = cv::aruco::CORNER_REFINE_SUBPIX;   // 角点亚像素优化
    params.cornerRefinementWinSize      = 5;
    params.cornerRefinementMaxIterations = 10;                               // 亚像素级角点精细化的迭代次数
    params.cornerRefinementMinAccuracy  = 0.03;                              // 精度
    params.minCornerDistanceRate        = 0.05;
    params.minDistanceToBorder          = 3;
    return params;
}

// ── 内部：ArUco 检测 + ID 过滤 ────────────────────────────────────────────
static bool tryDetect(const cv::Mat& img,
                      const cv::aruco::ArucoDetector& detector,
                      int target_id,
                      std::vector<cv::Point2f>& out_corners,
                      const cv::Rect* roi = nullptr) {
    const cv::Mat view = roi ? img(*roi) : img;
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    detector.detectMarkers(view, corners, ids);
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] != target_id || i >= corners.size() || corners[i].size() != 4) {
            continue;
        }
        if (!std::all_of(corners[i].begin(), corners[i].end(), [](const cv::Point2f& p) {
                return std::isfinite(p.x) && std::isfinite(p.y);
            })) {
            continue;
        }
        out_corners = corners[i];   // 顺序：TL TR BR BL
        if (roi) {
            for (cv::Point2f& corner : out_corners) {
                corner.x += static_cast<float>(roi->x);
                corner.y += static_cast<float>(roi->y);
            }
        }
        return true;
    }
    return false;
}

static bool makeTrackingRoi(const std::vector<cv::Point2f>& corners,
                            const cv::Size& image_size,
                            float padding_scale,
                            int min_margin,
                            cv::Rect& roi) {
    if (corners.size() != 4 || image_size.width <= 0 || image_size.height <= 0 ||
        !std::all_of(corners.begin(), corners.end(), [](const cv::Point2f& p) {
            return std::isfinite(p.x) && std::isfinite(p.y);
        })) {
        return false;
    }

    float min_x = corners[0].x;
    float max_x = corners[0].x;
    float min_y = corners[0].y;
    float max_y = corners[0].y;
    for (size_t i = 1; i < corners.size(); ++i) {
        min_x = std::min(min_x, corners[i].x);
        max_x = std::max(max_x, corners[i].x);
        min_y = std::min(min_y, corners[i].y);
        max_y = std::max(max_y, corners[i].y);
    }

    const float box_width = std::max(1.0f, max_x - min_x);
    const float box_height = std::max(1.0f, max_y - min_y);
    const float margin_x = std::max(static_cast<float>(min_margin), box_width * padding_scale);
    const float margin_y = std::max(static_cast<float>(min_margin), box_height * padding_scale);

    const int left = std::max(0, static_cast<int>(std::floor(min_x - margin_x)));
    const int top = std::max(0, static_cast<int>(std::floor(min_y - margin_y)));
    const int right = std::min(image_size.width,
                               static_cast<int>(std::ceil(max_x + margin_x)));
    const int bottom = std::min(image_size.height,
                                static_cast<int>(std::ceil(max_y + margin_y)));
    if (right <= left || bottom <= top) {
        return false;
    }

    roi = cv::Rect(left, top, right - left, bottom - top);
    // 标记靠近边缘或尺寸较大时，裁剪区域可能截断目标；
    // 此时改用整幅图像检测。
    const double roi_fraction = static_cast<double>(roi.area()) /
                                static_cast<double>(image_size.area());
    return roi_fraction < 0.85;
}

static void rememberDetection(const cv::Mat& gray,
                              const std::vector<cv::Point2f>& corners,
                              const ArucoParams& ap,
                              DetectorState& state) {
    state.prev_corners = corners;
    state.prev_size = gray.size();
    state.has_prev = true;
    state.occluded_frames = 0;
    state.search_frames = 0;
    if (ap.enable_optical_flow) {
        state.prev_gray = gray.clone();
    } else {
        state.prev_gray.release();
    }
}

// ── 内部：CLAHE 增强 ───────────────────────────────────────────────────────
static void enhanceImage(const cv::Mat& src, cv::Mat& dst) {
    static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(src, dst);
}

// ── 内部：计算 center 和 yaw ──────────────────────────────────────────────
static void fillCenterYaw(ArucoResult& res) {
    res.center = cv::Point2f(0, 0);
    for (const auto& point : res.corners) {
        res.center += point;
    }
    res.center *= 0.25f;

    const cv::Point2f top_edge = res.corners[1] - res.corners[0];  // TL -> TR
    res.yaw_rad = std::atan2(top_edge.y, top_edge.x);
}

// ── 内部：全局位移估算（光流辅助预测）────────────────────────────────────
static cv::Point2f globalShift(const std::vector<cv::Point2f>& prev,
                               const std::vector<cv::Point2f>& next,
                               const std::vector<uchar>& status) {
    cv::Point2f sum(0, 0);
    int cnt = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            sum += next[i] - prev[i];
            ++cnt;
        }
    }
    return cnt > 0 ? sum * (1.0f / cnt) : sum;
}

// ── 内部：图像方向 → 机体方向 ─────────────────────────────────────────────
// 相机朝下：图像上(-y) = 机体前(+x)，图像右(+x) = 机体右(+y)
static cv::Point2f imgToBody(const cv::Point2f& direction) {
    cv::Point2f body_direction(-direction.y, direction.x);
    const float norm = std::sqrt(body_direction.x * body_direction.x +
                                 body_direction.y * body_direction.y);
    if (norm > 1e-6f) {
        body_direction *= 1.0f / norm;
    }
    return body_direction;
}

// ── 内部：根据出界角点的边界位置计算搜索方向 ─────────────────────────────
// 取所有出界角点夹紧到边界后的位置，计算从画面中心指向边界的平均向量（归一化）
static cv::Point2f searchDirFromBorder(const std::vector<cv::Point2f>& border_points,
                                       const cv::Size& image_size) {
    const cv::Point2f image_center(image_size.width * 0.5f,
                                   image_size.height * 0.5f);
    cv::Point2f direction(0, 0);
    for (const auto& point : border_points) {
        direction += point - image_center;
    }
    const float norm = std::sqrt(direction.x * direction.x +
                                 direction.y * direction.y);
    if (norm > 1e-6f) {
        direction *= 1.0f / norm;
    }
    return direction;
}

// ── 四层检测主函数 ─────────────────────────────────────────────────────────
ArucoResult processFrame(const cv::Mat& gray,
                         const cv::aruco::ArucoDetector& detector,
                         const ArucoParams& ap,
                         DetectorState& st) {
    ArucoResult res;
    if (gray.empty() || gray.type() != CV_8UC1) {
        st = DetectorState{};
        return res;
    }
    if (st.has_prev && st.prev_size != gray.size()) {
        st = DetectorState{};
    }
    std::vector<cv::Point2f> detected;
    cv::Rect tracking_roi;
    const bool has_roi = ap.enable_roi && st.has_prev &&
                         makeTrackingRoi(st.prev_corners,
                                         gray.size(),
                                         ap.roi_padding_scale,
                                         ap.roi_min_margin,
                                         tracking_roi);

    // 常规路径：只在标记上一帧附近检测。
    if (has_roi && tryDetect(gray, detector, ap.target_id, detected, &tracking_roi)) {
        res.found = true;
        res.mode = MODE_DETECT;
        res.id = ap.target_id;
        res.corners = detected;
        res.estimated = {false, false, false, false};
        fillCenterYaw(res);
        rememberDetection(gray, res.corners, ap, st);
        return res;
    }

    // 首帧或 ROI 未检测到时，改为全图检测以提高恢复能力。
    if (tryDetect(gray, detector, ap.target_id, detected)) {
        res.found = true;
        res.mode = MODE_DETECT;
        res.id = ap.target_id;
        res.corners = detected;
        res.estimated = {false, false, false, false};
        fillCenterYaw(res);
        rememberDetection(gray, res.corners, ap, st);
        return res;
    }

    // 可选补救层。Nano 降落配置默认关闭，
    // 因为它会额外执行一次全图变换和标记搜索。
    if (ap.enable_clahe) {
        cv::Mat enhanced;
        enhanceImage(gray, enhanced);
        if (tryDetect(enhanced, detector, ap.target_id, detected)) {
            res.found = true;
            res.mode = MODE_CLAHE;
            res.id = ap.target_id;
            res.corners = detected;
            res.estimated = {false, false, false, false};
            fillCenterYaw(res);
            rememberDetection(gray, res.corners, ap, st);
            return res;
        }
    }

    // 以下层需要历史帧
    if (!ap.enable_optical_flow || !st.has_prev) {
        res.mode = MODE_LOST;
        return res;
    }

    // 可选光流补救。默认关闭，因为 LK 会在整幅图像上建立
    // 图像金字塔；该操作发生在标记检测失败后，计算开销较大。
    std::vector<cv::Point2f> next(4);
    std::vector<uchar> status(4, 0);
    std::vector<float> err(4, 0);
    cv::calcOpticalFlowPyrLK(
        st.prev_gray,
        gray,
        st.prev_corners,
        next,
        status,
        err,
        cv::Size(21, 21),
        3,
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01),
        0,
        1e-4);

    if (next.size() != 4 || status.size() != 4 ||
        !std::all_of(next.begin(), next.end(), [](const cv::Point2f& p) {
            return std::isfinite(p.x) && std::isfinite(p.y);
        })) {
        st = DetectorState{};
        res.mode = MODE_LOST;
        return res;
    }
    const cv::Point2f shift = globalShift(st.prev_corners, next, status);

    const int width = gray.cols;
    const int height = gray.rows;
    std::vector<cv::Point2f> final_corners(4);
    std::vector<bool> estimated(4, false);
    std::vector<int> out_idx, occ_idx;
    std::vector<cv::Point2f> border_pts;

    for (int i = 0; i < 4; ++i) {
        const cv::Point2f predicted = st.prev_corners[i] + shift;
        const bool in_image = predicted.x >= 0 && predicted.x < width &&
                              predicted.y >= 0 && predicted.y < height;

        if (!in_image) {
            // 出视野：夹紧到边界记录位置
            cv::Point2f border_point;
            border_point.x = std::max(0.0f, std::min(static_cast<float>(width - 1), predicted.x));
            border_point.y = std::max(0.0f, std::min(static_cast<float>(height - 1), predicted.y));
            out_idx.push_back(i);
            border_pts.push_back(border_point);
            final_corners[i] = predicted;
            estimated[i] = true;
        } else if (!status[i]) {
            // 遮挡
            occ_idx.push_back(i);
            final_corners[i] = predicted;
            estimated[i] = true;
        } else {
            final_corners[i] = next[i];
        }
    }

    // 第四层：目标出视野，进入搜索状态
    if (!out_idx.empty()) {
        if (++st.search_frames > ap.max_search_frames) {
            st.has_prev = false;
            res.mode = MODE_LOST;
            return res;
        }
        const cv::Point2f direction = searchDirFromBorder(border_pts, gray.size());
        res.mode = MODE_SEARCH;
        res.out_of_view_idx = out_idx;
        res.search_dir_img = direction;
        res.search_dir_body = imgToBody(direction);
        st.last_search_dir = direction;
        st.prev_corners = final_corners;
        st.prev_gray = gray.clone();
        st.occluded_frames = 0;
        return res;
    }

    // 第三层：部分角点被遮挡，进入遮挡状态
    if (!occ_idx.empty()) {
        if (++st.occluded_frames > ap.max_occluded_frames) {
            st.has_prev = false;
            res.mode = MODE_LOST;
            return res;
        }
        res.found = true;
        res.mode = MODE_OCCLUDED;
        res.id = ap.target_id;
        res.corners = final_corners;
        res.estimated = estimated;
        res.occluded_idx = occ_idx;
        fillCenterYaw(res);
        st.prev_corners = res.corners;
        st.prev_gray = gray.clone();
        st.search_frames = 0;
        return res;
    }

    // 第三层：所有角点有效，正常跟踪
    res.found = true;
    res.mode = MODE_TRACK;
    res.id = ap.target_id;
    res.corners = final_corners;
    res.estimated = {false, false, false, false};
    fillCenterYaw(res);
    st.prev_corners = res.corners;
    st.prev_gray = gray.clone();
    st.occluded_frames = 0;
    st.search_frames = 0;
    return res;
}
