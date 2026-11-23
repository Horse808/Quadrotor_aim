#include "ArUco_detector.hpp"
#include "camera.hpp"
#include "communicate.hpp"
#include "state_machine.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry/3d.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>

namespace {
std::atomic<bool> g_running{true};
void onSignal(int) { g_running.store(false); }

constexpr int kSensorWidth = 1280;
constexpr int kSensorHeight = 720;
constexpr int kSensorFps = 60;
// 仅使用测试程序内部的视觉处理尺寸，不修改 config 中已有的相机参数。
constexpr int kOutputWidth = 320;
constexpr int kOutputHeight = 180;
constexpr int kTargetId = 10;  // 修改为实际 ArUco 标记 ID。
constexpr float kMarkerSizeM = 0.19f;

struct MarkerPose {
    bool valid = false;
    cv::Vec3d translation{0.0, 0.0, 0.0};
};

bool finitePoint(const cv::Point2f& point) {
    return std::isfinite(point.x) && std::isfinite(point.y);
}

// 使用标记四个角点和实际边长求解相机坐标。当前 OpenCV 5.1
// 的 solvePnP 实现在 geometry 模块中，不需要单独的 calib3d 动态库。
MarkerPose estimateMarkerPose(const ArucoResult& result,
                              const cv::Mat& camera_matrix,
                              const cv::Mat& dist_coeffs) {
    MarkerPose pose;
    if (!result.found || result.corners.size() != 4 || result.estimated.size() != 4 ||
        camera_matrix.empty() || dist_coeffs.empty()) {
        return pose;
    }
    for (size_t i = 0; i < result.corners.size(); ++i) {
        if (result.estimated[i] || !finitePoint(result.corners[i])) return pose;
    }

    const float half = kMarkerSizeM * 0.5f;
    // ArUco_detector 保证角点顺序为 TL、TR、BR、BL；物体坐标的 Y 轴
    // 向下，与相机图像坐标约定一致。
    const std::vector<cv::Point3f> object_points = {
        {-half, -half, 0.0f},
        { half, -half, 0.0f},
        { half,  half, 0.0f},
        {-half,  half, 0.0f},
    };

    cv::Vec3d rvec;
    cv::Vec3d tvec;
    try {
        if (!cv::solvePnP(object_points, result.corners, camera_matrix, dist_coeffs,
                          rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE)) {
            return pose;
        }
    } catch (const cv::Exception& error) {
        std::cerr << "[main] solvePnP 异常：" << error.what() << '\n';
        return pose;
    }

    if (!std::isfinite(tvec[0]) || !std::isfinite(tvec[1]) || !std::isfinite(tvec[2]) ||
        tvec[2] < 0.05 || tvec[2] > 20.0) {
        return pose;
    }
    pose.translation = tvec;
    pose.valid = true;
    return pose;
}

bool readCameraMatrix(cv::Mat& matrix, cv::Mat& dist_coeffs) {
    cv::FileStorage file("config/camera_config.yaml", cv::FileStorage::READ);
    if (!file.isOpened()) {
        std::cerr << "[main] 无法读取 config/camera_config.yaml\n";
        return false;
    }
    file["camera_matrix"] >> matrix;
    file["dist_coeffs"] >> dist_coeffs;
    cv::FileNode width_node = file["input_width"];
    cv::FileNode height_node = file["input_height"];
    if (matrix.empty() || dist_coeffs.empty() || width_node.empty() || height_node.empty()) {
        std::cerr << "[main] 相机内参、畸变参数或处理输入尺寸缺失。\n";
        return false;
    }
    const int configured_width = static_cast<int>(width_node);
    const int configured_height = static_cast<int>(height_node);
    if (configured_width <= 0 || configured_height <= 0 ||
        matrix.rows != 3 || matrix.cols != 3) {
        std::cerr << "[main] 相机内参尺寸无效。\n";
        return false;
    }
    // 配置中的矩阵已经按 config input_width/input_height 缩放；当前算法图像为 320x180。
    const double sx = static_cast<double>(kOutputWidth) / configured_width;
    const double sy = static_cast<double>(kOutputHeight) / configured_height;
    if (matrix.type() != CV_64F) matrix.convertTo(matrix, CV_64F);
    matrix.at<double>(0, 0) *= sx;
    matrix.at<double>(0, 2) *= sx;
    matrix.at<double>(1, 1) *= sy;
    matrix.at<double>(1, 2) *= sy;
    return true;
}

bool waitForPreflight(Communicate& fc, int seconds) {
    const auto stable_duration = std::chrono::seconds(seconds);
    auto valid_since = std::chrono::steady_clock::time_point{};
    while (g_running.load()) {
        const FcState state = fc.readState();
        const auto now = std::chrono::steady_clock::now();
        if (state.heartbeat_valid && state.optical_flow_distance_valid) {
            if (valid_since == std::chrono::steady_clock::time_point{}) valid_since = now;
            if (now - valid_since >= stable_duration) return true;
        } else {
            if (valid_since != std::chrono::steady_clock::time_point{}) {
                std::cout << "[main] 心跳或光流中断，重新累计预检时间。\n";
            }
            valid_since = {};
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
}

const char* modeName(bool normal) {
    return normal ? "正常模式（切换 Offboard 后解锁）" : "测试模式（不切模式、不解锁，但发送速度设定值）";
}
}  // namespace

int main(int argc, char** argv) {
    bool normal_mode = false;
    if (argc == 2 && std::string(argv[1]) == "--normal") {
        normal_mode = true;
    } else if (argc > 1 && std::string(argv[1]) != "--test") {
        std::cerr << "用法：main_test [--test|--normal]\n";
        return 2;
    }

    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);
    std::cout << "[main] " << modeName(normal_mode) << '\n'
              << "[main] ArUco 边长：0.19 m，目标距离：0.50 m。\n"
              << "[main] 注意：速度设定值会影响飞控，必须保留人工接管。\n";

    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    if (!readCameraMatrix(camera_matrix, dist_coeffs)) return 1;

    Camera camera;
    if (!camera.open(0, kSensorWidth, kSensorHeight, kSensorFps,
                     kOutputWidth, kOutputHeight)) {
        std::cerr << "[main] 相机打开失败：" << camera.lastError() << '\n';
        return 1;
    }

    Communicate fc;
    CommunicateConfig communication_config;
    // 两种模式都发送设定值：测试模式不主动切换飞控模式或解锁。
    communication_config.enable_motion = true;
    communication_config.send_rate_hz = 20;
    if (!fc.init(communication_config)) {
        camera.close();
        return 1;
    }

    if (!fc.requestOpticalFlow(20, 5000)) {
        std::cerr << "[main] 请求 20 Hz OPTICAL_FLOW_RAD 失败，停止。\n";
        fc.shutdown();
        camera.close();
        return 1;
    }
    if (!waitForPreflight(fc, 5)) {
        std::cerr << "[main] 心跳或光流未连续有效 5 秒，停止。\n";
        fc.sendCommand(FcCommand{});
        fc.shutdown();
        camera.close();
        return 1;
    }

    if (normal_mode) {
        // 通信线程已经持续发送零速度设定值，先切 Offboard，再解锁。
        if (!fc.setOffboard(5000)) {
            std::cerr << "[main] 切换 Offboard 失败，停止。\n";
            fc.sendCommand(FcCommand{});
            fc.shutdown();
            camera.close();
            return 1;
        }
        if (!fc.arm(5000)) {
            std::cerr << "[main] 解锁失败，停止。\n";
            fc.sendCommand(FcCommand{});
            fc.shutdown();
            camera.close();
            return 1;
        }
        std::cout << "[main] Offboard 和解锁成功，开始执行状态机。\n";
    } else {
        std::cout << "[main] 测试模式：跳过 Offboard 和解锁。\n";
    }

    const auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    cv::aruco::ArucoDetector detector(dictionary, makeDetectorParams());
    ArucoParams detector_params;
    detector_params.target_id = kTargetId;
    DetectorState detector_state;

    StateMachineConfig state_config;
    FlightStateMachine state_machine(state_config);
    cv::Mat gray;
    cv::Mat processed_gray;
    FlightState last_state = state_machine.state();
    auto last_log = std::chrono::steady_clock::now();
    bool disarm_attempted = false;

    while (g_running.load() && fc.isOpen()) {
        const auto now = std::chrono::steady_clock::now();
        if (!camera.readGray(gray, 500)) {
            std::cerr << "[main] 相机取帧失败，进入保护状态。\n";
            fc.sendCommand(FcCommand{});
            break;
        }

        // VPSS 在部分固件上可能不可用，此时 Camera 会回退输出 1280x720。
        // 视觉算法和相机内参统一使用 320x180，因此这里必须显式缩放，
        // 避免回退路径下角点坐标与内参分辨率不一致。
        if (gray.cols != kOutputWidth || gray.rows != kOutputHeight) {
            cv::resize(gray, processed_gray, cv::Size(kOutputWidth, kOutputHeight),
                        0.0, 0.0, cv::INTER_AREA);
        } else {
            processed_gray = gray;
        }

        const ArucoResult detection = processFrame(processed_gray, detector,
                                                   detector_params, detector_state);
        const MarkerPose pose = estimateMarkerPose(detection, camera_matrix, dist_coeffs);
        VisionTarget target;
        target.valid = pose.valid;
        if (pose.valid) {
            target.x_m = static_cast<float>(pose.translation[0]);
            target.y_m = static_cast<float>(pose.translation[1]);
            target.z_m = static_cast<float>(pose.translation[2]);
        }

        const FcState flight_state = fc.readState();
        if (normal_mode && (!flight_state.heartbeat_valid ||
                            !flight_state.offboard_mode ||
                            !flight_state.armed)) {
            // 正常模式下，飞控离开 Offboard 或失去解锁状态时立即停止
            // 视觉控制；通信线程仍会在超时后继续发送零速度。
            std::cerr << "[main] 飞控状态异常（心跳/Offboard/解锁失效），停止运动控制。\n";
            fc.sendCommand(FcCommand{});
            break;
        }
        const FcCommand command = state_machine.update(flight_state, target, now);
        fc.sendCommand(command);

        if (command.land_done && normal_mode && !disarm_attempted) {
            // 只有正常模式允许自动停桨；测试模式即使进入 FINISHED 也不
            // 发送解锁/停桨命令，避免误操作实机。此时通信线程仍持续发送零速度。
            disarm_attempted = true;
            std::cout << "[main] 光流高度低于 10 cm，发送停桨命令。\n";
            if (!fc.disarm(5000)) {
                std::cerr << "[main] 停桨命令未被飞控确认，请立即人工接管。\n";
            }
        }

        if (state_machine.state() != last_state) {
            std::cout << "[main] 状态切换：" << state_machine.stateName() << '\n';
            last_state = state_machine.state();
        }
        if (now - last_log >= std::chrono::seconds(2)) {
            std::cout << "[main] state=" << state_machine.stateName()
                      << " marker=" << (target.valid ? "yes" : "no")
                      << " height=" << (flight_state.height_valid ? flight_state.height_m : -1.0f)
                      << " flow=" << (flight_state.optical_flow_distance_valid ?
                                       flight_state.optical_flow_distance_m : -1.0f)
                      << " target=" << (target.valid ?
                                          (std::to_string(target.x_m) + "," +
                                           std::to_string(target.y_m) + "," +
                                           std::to_string(target.z_m)) : "-")
                      << '\n';
            last_log = now;
        }
        if (state_machine.terminal()) {
            fc.sendCommand(FcCommand{});
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    fc.sendCommand(FcCommand{});
    fc.shutdown();
    camera.close();
    return 0;
}
