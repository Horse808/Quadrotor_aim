#include "camera.hpp"
#include "camera_intrinsics.hpp"
#include "communicate.hpp"
#include "light_detector.hpp"
#include "state_machine.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>

namespace {
constexpr int kSensorWidth = 1280;
constexpr int kSensorHeight = 720;
constexpr int kSensorFps = 60;
constexpr int kImageWidth = 320;
constexpr int kImageHeight = 180;
constexpr int kTargetLoopHz = 20;
std::atomic<bool> g_running{true};

void handleSignal(int) { g_running.store(false); }

struct RuntimeConfig {
    CommunicateConfig communication;
    StateMachineConfig flight;
    LightDetectorParams light;
    int preflight_stable_seconds = 5;
    float light_diameter_m = 0.055f;
    float focal_x_px = 0.0f;
    float focal_y_px = 0.0f;
    float principal_x_px = 0.0f;
    float principal_y_px = 0.0f;
    float log_interval_seconds = 2.0f;
};

bool readNumber(const cv::FileStorage& file, const char* key, float& value) {
    const cv::FileNode node = file[key];
    if (node.empty()) return false;
    node >> value;
    return std::isfinite(value);
}

bool loadConfig(const std::string& fly_path, const std::string& camera_path,
                RuntimeConfig& config) {
    cv::FileStorage fly(fly_path, cv::FileStorage::READ);
    if (!fly.isOpened()) {
        std::cerr << "[配置] 无法打开飞行配置文件：" << fly_path << '\n';
        return false;
    }

    std::string device;
    fly["serial_device"] >> device;
    if (!device.empty()) config.communication.device = device;
    fly["baud_rate"] >> config.communication.baud;
    fly["send_rate_hz"] >> config.communication.send_rate_hz;
    fly["message_timeout_ms"] >> config.communication.command_timeout_ms;
    fly["preflight_stable_seconds"] >> config.preflight_stable_seconds;

    bool ok = true;
    ok &= readNumber(fly, "takeoff_height_m", config.flight.takeoff_height_m);
    ok &= readNumber(fly, "takeoff_speed_mps", config.flight.takeoff_speed_mps);
    ok &= readNumber(fly, "search_height_m", config.flight.search_height_m);
    ok &= readNumber(fly, "height_tolerance_m", config.flight.height_tolerance_m);
    ok &= readNumber(fly, "search_timeout_seconds", config.flight.search_timeout_seconds);
    // 新增参数兼容旧配置：未配置时使用状态机默认值。
    if (!readNumber(fly, "target_loss_hold_seconds", config.flight.target_loss_hold_seconds)) {
        config.flight.target_loss_hold_seconds = StateMachineConfig{}.target_loss_hold_seconds;
    }
    float yaw_deg_s = 0.0f;
    ok &= readNumber(fly, "search_yaw_speed_deg_s", yaw_deg_s);
    config.flight.max_search_yaw_radps = yaw_deg_s * static_cast<float>(CV_PI / 180.0);
    ok &= readNumber(fly, "target_distance_m", config.flight.target_distance_m);
    ok &= readNumber(fly, "target_distance_tolerance_m", config.flight.target_distance_tolerance_m);
    ok &= readNumber(fly, "lateral_tolerance_m", config.flight.lateral_tolerance_m);
    ok &= readNumber(fly, "vertical_tolerance_m", config.flight.vertical_tolerance_m);
    ok &= readNumber(fly, "target_hold_seconds", config.flight.hold_seconds);
    ok &= readNumber(fly, "forward_speed_mps", config.flight.max_forward_speed_mps);
    ok &= readNumber(fly, "lateral_speed_mps", config.flight.max_lateral_speed_mps);
    ok &= readNumber(fly, "vertical_speed_mps", config.flight.max_vertical_speed_mps);
    ok &= readNumber(fly, "landing_speed_mps", config.flight.landing_speed_mps);
    ok &= readNumber(fly, "landing_stop_height_m", config.flight.landing_height_m);
    ok &= readNumber(fly, "position_gain", config.flight.position_gain);
    ok &= readNumber(fly, "vertical_gain", config.flight.vertical_gain);
    ok &= readNumber(fly, "height_gain", config.flight.height_gain);
    ok &= readNumber(fly, "light_diameter_m", config.light_diameter_m);
    ok &= readNumber(fly, "log_interval_seconds", config.log_interval_seconds);

    cv::FileStorage camera(camera_path, cv::FileStorage::READ);
    if (!camera.isOpened()) {
        std::cerr << "[配置] 无法打开相机配置文件：" << camera_path << '\n';
        return false;
    }
    cv::Mat matrix;
    if (!readScaledCameraMatrix(camera, cv::Size(kImageWidth, kImageHeight), matrix)) {
        std::cerr << "[配置] 相机内参或其对应的标定分辨率无效。\n";
        return false;
    }
    config.focal_x_px = static_cast<float>(matrix.at<double>(0, 0));
    config.focal_y_px = static_cast<float>(matrix.at<double>(1, 1));
    config.principal_x_px = static_cast<float>(matrix.at<double>(0, 2));
    config.principal_y_px = static_cast<float>(matrix.at<double>(1, 2));
    if (!std::isfinite(config.focal_x_px) || config.focal_x_px <= 0.0f ||
        !std::isfinite(config.focal_y_px) || config.focal_y_px <= 0.0f ||
        !std::isfinite(config.principal_x_px) || !std::isfinite(config.principal_y_px)) {
        std::cerr << "[配置] 相机内参数值无效。\n";
        return false;
    }
    // 继续使用已有相机配置中的引导灯识别阈值，不改动检测算法本身。
    camera["light_threshold"] >> config.light.binary_threshold;
    camera["light_min_area"] >> config.light.min_area;
    camera["light_max_area"] >> config.light.max_area;
    camera["light_min_circularity"] >> config.light.min_circularity;
    camera["light_max_aspect_ratio_diff"] >> config.light.max_aspect_ratio_diff;
    camera["green_hue_min"] >> config.light.green_hue_min;
    camera["green_hue_max"] >> config.light.green_hue_max;
    camera["green_saturation_min"] >> config.light.green_saturation_min;
    camera["green_value_min"] >> config.light.green_value_min;
    camera["min_green_ratio"] >> config.light.min_green_ratio;

    const auto positive = [](float v) { return std::isfinite(v) && v > 0.0f; };
    if (!ok || !positive(config.flight.takeoff_height_m) ||
        !positive(config.flight.takeoff_speed_mps) || !positive(config.flight.search_height_m) ||
        !positive(config.flight.search_timeout_seconds) ||
        !std::isfinite(config.flight.target_loss_hold_seconds) ||
        config.flight.target_loss_hold_seconds < 0.0f || config.flight.target_loss_hold_seconds > 10.0f ||
        !positive(config.flight.max_search_yaw_radps) ||
        !positive(config.flight.target_distance_m) || !positive(config.flight.hold_seconds) ||
        !positive(config.flight.max_forward_speed_mps) || !positive(config.flight.max_lateral_speed_mps) ||
        !positive(config.flight.max_vertical_speed_mps) || !positive(config.flight.landing_speed_mps) ||
        !positive(config.flight.landing_height_m) || !positive(config.light_diameter_m) ||
        !positive(config.focal_x_px) || !positive(config.focal_y_px) || !positive(config.log_interval_seconds) ||
        config.preflight_stable_seconds <= 0 || config.communication.send_rate_hz < 3 ||
        config.communication.send_rate_hz > 100 || config.communication.command_timeout_ms < 100 ||
        config.communication.command_timeout_ms > 5000 ||
        config.flight.target_distance_tolerance_m < 0.0f || config.flight.lateral_tolerance_m < 0.0f ||
        config.flight.height_tolerance_m < 0.0f ||
        config.flight.vertical_tolerance_m < 0.0f ||
        !std::isfinite(config.flight.position_gain) || config.flight.position_gain <= 0.0f ||
        !std::isfinite(config.flight.vertical_gain) || config.flight.vertical_gain <= 0.0f ||
        !std::isfinite(config.flight.height_gain) || config.flight.height_gain <= 0.0f) {
        std::cerr << "[配置] 飞行参数缺失或超出有效范围，请检查 fly_config.yaml。\n";
        return false;
    }
    config.communication.enable_motion = true;
    return true;
}

bool waitForPreflight(Communicate& fc, int seconds) {
    auto valid_since = std::chrono::steady_clock::time_point{};
    while (g_running.load()) {
        const auto now = std::chrono::steady_clock::now();
        const FcState state = fc.readState();
        if (state.heartbeat_valid && state.optical_flow_distance_valid) {
            if (valid_since == std::chrono::steady_clock::time_point{}) valid_since = now;
            if (now - valid_since >= std::chrono::seconds(seconds)) return true;
        } else {
            valid_since = {};
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
}

VisionTarget makeTarget(const LightResult& light, const RuntimeConfig& config) {
    VisionTarget target;
    if (!light.found || !std::isfinite(light.estimated_distance) ||
        light.estimated_distance <= 0.05f || light.estimated_distance > 20.0f) return target;
    // 用已标定焦距将灯中心像素偏差换算到相机坐标系，并以已知灯直径估距。
    const float distance = light.estimated_distance;
    target.x_m = (light.center.x - config.principal_x_px) * distance / config.focal_x_px;
    target.y_m = (light.center.y - config.principal_y_px) * distance / config.focal_y_px;
    target.z_m = distance;
    target.valid = std::isfinite(target.x_m) && std::isfinite(target.y_m);
    return target;
}
}  // namespace

int main(int argc, char** argv) {
    bool normal_mode = false;
    std::string config_dir = "config";
    std::string config_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--normal") normal_mode = true;
        else if (arg == "--test") normal_mode = false;
        else if ((arg == "--config" || arg == "--config-dir") && i + 1 < argc &&
                 argv[i + 1][0] != '\0' && std::string(argv[i + 1]).rfind("--", 0) != 0) {
            if (arg == "--config-dir") config_dir = argv[++i];
            else config_path = argv[++i];
        }
        else {
            std::cerr << "用法：main [--test|--normal] [--config-dir 配置目录] [--config 飞行配置文件]\n";
            return 2;
        }
    }

    // 两个配置文件共用指定目录；保留 --config 单独覆盖飞行配置的旧用法。
    const std::string config_prefix = config_dir.back() == '/' ? config_dir : config_dir + '/';
    if (config_path.empty()) config_path = config_prefix + "fly_config.yaml";
    const std::string camera_config_path = config_prefix + "camera_config.yaml";

    RuntimeConfig config;
    if (!loadConfig(config_path, camera_config_path, config)) return 2;
    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);
    std::cout << "[主程序] " << (normal_mode ? "正常模式" : "测试模式（不切模式、不解锁、不停桨）") << '\n'
              << "[主程序] 目标距离=" << config.flight.target_distance_m << " 米，灯外径="
              << config.light_diameter_m << " 米，停桨高度=" << config.flight.landing_height_m << " 米。\n";

    Camera camera;
    if (!camera.open(0, kSensorWidth, kSensorHeight, kSensorFps, kImageWidth, kImageHeight)) {
        std::cerr << "[相机] 打开失败：" << camera.lastError() << '\n';
        return 1;
    }
    if (!camera.configureExposureFromFile(camera_config_path)) {
        std::cerr << "[相机] 曝光配置失败：" << camera.lastError() << '\n';
        camera.close();
        return 1;
    }
    if (!camera.usesHardwareScaledOutput()) {
        std::cerr << "[相机] VPSS 硬件缩放未启用；为避免静默回退到高负载 CPU 缩放，停止启动。"
                  << "请检查前面的 VPSS 初始化错误并确认上次进程已完整退出。\n";
        camera.close();
        return 1;
    }
    std::cout << "[相机] 使用 VPSS 硬件缩放输出 " << kImageWidth << "x" << kImageHeight << '\n';
    Communicate fc;
    if (!fc.init(config.communication)) {
        camera.close();
        return 1;
    }
    if (!fc.requestOpticalFlow(config.communication.send_rate_hz, config.communication.command_timeout_ms)) {
        std::cerr << "[飞控] 请求光流数据失败，停止任务。\n";
        fc.shutdown(); camera.close(); return 1;
    }
    if (!waitForPreflight(fc, config.preflight_stable_seconds)) {
        std::cerr << "[飞控] 心跳和光流未能连续通过预检。\n";
        fc.sendCommand(FcCommand{}); fc.shutdown(); camera.close(); return 1;
    }
    if (normal_mode) {
        if (!fc.setOffboard(config.communication.command_timeout_ms) ||
            !fc.arm(config.communication.command_timeout_ms)) {
            std::cerr << "[飞控] 切换 Offboard 或解锁失败；保持零速度并退出。\n";
            fc.sendCommand(FcCommand{}); fc.shutdown(); camera.close(); return 1;
        }
    }

    FlightStateMachine state_machine(config.flight);
    cv::Mat frame, resized;
    auto last_log = std::chrono::steady_clock::now();
    FlightState last_state = state_machine.state();
    bool disarm_attempted = false;
    const auto loop_period = std::chrono::milliseconds(1000 / kTargetLoopHz);

    while (g_running.load() && fc.isOpen()) {
        const auto loop_start = std::chrono::steady_clock::now();
        if (!camera.readBgr(frame, 500)) {
            std::cerr << "[相机] 取帧失败，停止非零运动指令。\n";
            fc.sendCommand(FcCommand{});
            break;
        }
        if (frame.cols != kImageWidth || frame.rows != kImageHeight) {
            std::cerr << "[相机] VPSS 输出尺寸异常：" << frame.cols << "x" << frame.rows
                      << "，停止处理以避免错误尺度下的识别和距离估算。\n";
            fc.sendCommand(FcCommand{});
            break;
        }
        resized = frame;

        const LightResult light = detectLight(resized, config.light, config.light_diameter_m,
                                              config.focal_x_px);
        const VisionTarget target = makeTarget(light, config);
        const FcState flight = fc.readState();
        if (normal_mode && (!flight.heartbeat_valid || !flight.offboard_mode || !flight.armed)) {
            std::cerr << "[飞控] 心跳、Offboard 或解锁状态异常，立即发送零速度。\n";
            fc.sendCommand(FcCommand{});
            break;
        }
        const auto now = std::chrono::steady_clock::now();
        const FcCommand command = state_machine.update(flight, target, now);
        fc.sendCommand(command);
        if (command.land_done && normal_mode && !disarm_attempted) {
            disarm_attempted = true;
            std::cout << "[降落] 达到配置停桨高度，发送 PX4 强制解除武装命令。\n";
            if (!fc.disarm(config.communication.command_timeout_ms, true))
                std::cerr << "[安全] 停桨未获飞控确认，请立即人工接管。\n";
        }
        if (state_machine.state() != last_state) {
            std::cout << "[状态] " << state_machine.stateName() << '\n';
            last_state = state_machine.state();
        }
        if (now - last_log >= std::chrono::duration<float>(config.log_interval_seconds)) {
            std::cout << "[遥测] 状态=" << state_machine.stateName()
                      << " 灯=" << (target.valid ? "已识别" : "未识别")
                      << " 距离=" << (target.valid ? target.z_m : -1.0f) << " 米"
                      << " 高度=" << (flight.optical_flow_distance_valid ? flight.optical_flow_distance_m : -1.0f)
                      << " 米\n";
            last_log = now;
        }
        if (state_machine.terminal()) {
            fc.sendCommand(FcCommand{});
            break;
        }
        std::this_thread::sleep_until(loop_start + loop_period);
    }
    fc.sendCommand(FcCommand{});
    fc.shutdown();
    camera.close();
    return 0;
}
