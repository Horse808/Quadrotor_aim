#include "state_machine.hpp"
#include "camera_intrinsics.hpp"
#include <limits>
#include <opencv2/core.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

void check(bool good, const char* what) {
    if (!good) throw std::runtime_error(what);
}
// 使用不同的起飞/悬停速度，验证两种参数没有混用，也没有叠加。
void checkHeightSpeedControl() {
    StateMachineConfig config;
    config.takeoff_speed_mps = 0.2f;
    config.max_vertical_speed_mps = 0.08f;
    config.height_tolerance_m = 0.0625f;
    const auto start = std::chrono::steady_clock::now();
    const auto time = [&](double seconds) {
        return start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(seconds));
    };
    const auto near = [](float a, float b) { return std::abs(a - b) < 1e-6f; };
    FcState fc;
    fc.heartbeat_valid = fc.optical_flow_distance_valid = fc.height_valid = true;
    fc.height_m = config.takeoff_height_m;
    fc.optical_flow_distance_m = config.takeoff_height_m;
    FlightStateMachine sm(config);
    sm.update(fc, VisionTarget{}, time(0));
    auto command = sm.update(fc, VisionTarget{}, time(0.1));
    check(sm.state() == FlightState::Search && near(command.vz, -config.max_vertical_speed_mps),
          "search sends configured hover compensation at target height");
    fc.height_m = config.search_height_m - 0.2f;
    command = sm.update(fc, VisionTarget{}, time(0.2));
    check(near(command.vz, -config.takeoff_speed_mps),
          "search climb uses takeoff speed, not hover speed or their sum");
    fc.height_m = config.search_height_m + 0.2f;
    command = sm.update(fc, VisionTarget{}, time(0.3));
    check(near(command.vz, config.takeoff_speed_mps),
          "search descent uses takeoff speed, not the visual speed limit");
    // 容差边界包含在悬停区间，越过边界才改用起飞速度修正。
    for (float offset : {-config.height_tolerance_m, 0.0f, config.height_tolerance_m}) {
        fc.height_m = config.search_height_m + offset;
        command = sm.update(fc, VisionTarget{}, time(0.4));
        check(near(command.vz, -config.max_vertical_speed_mps),
              "search hover speed is used throughout inclusive height tolerance");
    }
    fc.height_valid = false;
    fc.optical_flow_distance_m = config.search_height_m - config.height_tolerance_m - 0.001f;
    command = sm.update(fc, VisionTarget{}, time(0.5));
    check(near(command.vz, -config.takeoff_speed_mps), "optical fallback climbs at takeoff speed");
    fc.optical_flow_distance_m = config.search_height_m + config.height_tolerance_m + 0.001f;
    command = sm.update(fc, VisionTarget{}, time(0.6));
    check(near(command.vz, config.takeoff_speed_mps), "optical fallback descends at takeoff speed");
    fc.height_valid = true;
    fc.height_m = fc.optical_flow_distance_m = config.search_height_m;

    VisionTarget target{true, 0.2f, 0.2f, 1.0f};
    sm.update(fc, target, time(1));
    command = sm.update(fc, target, time(1.1));
    check(near(command.vz, config.vertical_gain * target.y_m),
          "tracking vertical alignment overrides hover compensation");
    target.y_m = -0.2f;
    command = sm.update(fc, target, time(1.2));
    check(near(command.vz, config.vertical_gain * target.y_m),
          "tracking upward alignment overrides hover compensation");
    target.y_m = 0;
    command = sm.update(fc, target, time(1.3));
    check(command.vyaw != 0 && near(command.vz, -config.max_vertical_speed_mps),
          "yaw centering keeps hover compensation");
    target.x_m = 0;
    command = sm.update(fc, target, time(1.4));
    check(command.vx != 0 && near(command.vz, -config.max_vertical_speed_mps),
          "forward approach keeps hover compensation");
    command = sm.update(fc, VisionTarget{}, time(1.5));
    check(sm.state() == FlightState::Track && near(command.vz, -config.max_vertical_speed_mps),
          "brief target loss keeps hover compensation with remembered target");
    target.z_m = config.target_distance_m;
    command = sm.update(fc, target, time(2));
    check(near(command.vz, -config.max_vertical_speed_mps), "arrival hold keeps hover compensation");
    command = sm.update(fc, target, time(3));
    check(sm.state() == FlightState::Track && near(command.vz, -config.max_vertical_speed_mps),
          "hover compensation is sent on successive hold updates");
    sm.update(fc, target, time(2 + config.hold_seconds + 0.1));
    command = sm.update(fc, VisionTarget{}, time(2 + config.hold_seconds + 0.2));
    check(sm.state() == FlightState::Land && near(command.vz, config.landing_speed_mps),
          "landing uses descent speed without hover compensation");
    fc.optical_flow_distance_m = config.landing_height_m;
    command = sm.update(fc, VisionTarget{}, time(2 + config.hold_seconds + 0.3));
    check(sm.state() == FlightState::Finished && command.land_done && command.vz == 0,
          "motor stop never applies hover compensation");

    sm.reset();
    fc.height_m = 0;
    fc.optical_flow_distance_m = 0.05f;
    command = sm.update(fc, VisionTarget{}, time(10));
    check(near(command.vz, -config.takeoff_speed_mps), "takeoff ignores hover compensation");
    fc.heartbeat_valid = false;
    command = sm.update(fc, target, time(10.1));
    check(sm.state() == FlightState::Failsafe && command.vz == 0,
          "telemetry failure never applies hover compensation");

    // 悬停始终从垂直速度参数取值，不再使用旧的独立补偿字段。
    fc.heartbeat_valid = true;
    fc.height_m = fc.optical_flow_distance_m = config.takeoff_height_m;
    for (float speed : {0.02f, 0.08f, 0.3f}) {
        config.max_vertical_speed_mps = speed;
        FlightStateMachine checked(config);
        checked.update(fc, VisionTarget{}, time(11));
        command = checked.update(fc, VisionTarget{}, time(11.1));
        check(near(command.vz, -speed), "hover output follows vertical speed directly");
    }
    std::cout << "PASS: takeoff/hover/correction speed selection, thresholds and safety isolation\n";
}

int main() {
    checkHeightSpeedControl();
    cv::FileStorage fly("config/fly_config.yaml", cv::FileStorage::READ);
    StateMachineConfig config;
    fly["target_distance_m"] >> config.target_distance_m;
    fly["target_hold_seconds"] >> config.hold_seconds;
    fly["search_timeout_seconds"] >> config.search_timeout_seconds;
    fly["height_tolerance_m"] >> config.height_tolerance_m;
    fly["target_loss_hold_seconds"] >> config.target_loss_hold_seconds;
    fly["landing_stop_height_m"] >> config.landing_height_m;
    FcState fc;
    fc.heartbeat_valid = fc.optical_flow_distance_valid = true;
    fc.optical_flow_distance_m = 1.0f;
    const auto start = std::chrono::steady_clock::now();
    const auto time = [&](double seconds) {
        return start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(seconds));
    };
    VisionTarget target{true, 0, 0, 0.5f};
    FlightStateMachine sm(config);
    sm.update(fc, target, time(0));
    check(sm.state() == FlightState::Track, "enter Track");
    sm.update(fc, target, time(0.1));
    sm.update(fc, target, time(5.0));
    check(sm.state() == FlightState::Track, "hold not premature");
    sm.update(fc, target, time(5.2));
    check(sm.state() == FlightState::Land, "hold enters Land");
    auto command = sm.update(fc, VisionTarget{}, time(5.3));
    check(command.vx == 0 && command.vy == 0 && command.vyaw == 0 && command.vz > 0, "vertical descent despite vision loss");
    fc.optical_flow_distance_m = config.landing_height_m;
    command = sm.update(fc, VisionTarget{}, time(5.4));
    check(command.land_done && sm.state() == FlightState::Finished, "stop at configured height");
    std::cout << "PASS: target 0.5 m -> hold 5 s -> vertical descent -> land_done at " << config.landing_height_m << " m\n";

    fc.optical_flow_distance_m = 1;
    sm.reset();
    sm.update(fc, target, time(10));
    sm.update(fc, target, time(10.1));
    target.z_m = 0.8f;
    sm.update(fc, target, time(13));
    target.z_m = 0.5f;
    sm.update(fc, target, time(14));
    sm.update(fc, target, time(18.9));
    check(sm.state() == FlightState::Track, "bad alignment resets hold");
    sm.update(fc, target, time(19.1));
    check(sm.state() == FlightState::Land, "renewed hold completes");
    std::cout << "PASS: leaving tolerance resets continuous hold\n";

    sm.reset();
    target = {true, 0.2f, 0.1f, 1.0f};
    sm.update(fc, target, time(25));
    command = sm.update(fc, target, time(25.1));
    check(command.vz > 0 && command.vx == 0 && command.vyaw == 0,
          "vertical alignment precedes yaw and forward motion");
    target.y_m = 0.0f;
    command = sm.update(fc, target, time(25.2));
    check(command.vyaw < 0 && command.vx == 0 && command.vy == 0,
          "yaw alignment precedes forward approach");
    target.x_m = 0.0f;
    target.z_m = 1.0f;
    command = sm.update(fc, target, time(25.3));
    check(command.vx < 0 && command.vyaw == 0, "forward approach after alignment");
    std::cout << "PASS: vertical alignment -> yaw centering -> forward approach\n";

    command = sm.update(fc, VisionTarget{}, time(25.4));
    check(sm.state() == FlightState::Track && command.vx < 0,
          "brief target loss holds last visual target");
    command = sm.update(fc, VisionTarget{}, time(25.3 + config.target_loss_hold_seconds - 0.01));
    check(sm.state() == FlightState::Track, "target hold remains before timeout");
    command = sm.update(fc, VisionTarget{}, time(25.3 + config.target_loss_hold_seconds + 0.01));
    check(sm.state() == FlightState::Search && command.vx == 0,
          "target loss timeout returns to Search without stale motion");
    sm.update(fc, VisionTarget{}, time(28 + config.search_timeout_seconds + 0.1));
    check(sm.state() == FlightState::Land, "search timeout lands without arrival");
    std::cout << "CONFIRMED: search timeout enters Land even without reaching target\n";

    sm.reset();
    fc.height_valid = true;
    const float saved_search_height = config.search_height_m;
    config.search_height_m = config.takeoff_height_m + 0.4f;
    fc.height_m = config.takeoff_height_m;
    fc.optical_flow_distance_m = config.search_height_m + 0.2f;
    FlightStateMachine search_sm(config);
    search_sm.update(fc, VisionTarget{}, time(50));
    command = search_sm.update(fc, VisionTarget{}, time(51));
    check(search_sm.state() == FlightState::Search, "takeoff reaches Search");
    command = search_sm.update(fc, VisionTarget{}, time(51.1));
    check(search_sm.state() == FlightState::Search &&
              std::abs(command.vz + config.takeoff_speed_mps) < 1e-6f && command.vyaw > 0,
          "search climbs at takeoff speed using relative altitude");
    fc.height_m = config.search_height_m + 0.4f;
    command = search_sm.update(fc, VisionTarget{}, time(51.2));
    check(std::abs(command.vz - config.takeoff_speed_mps) < 1e-6f,
          "search descends at takeoff speed when above setpoint");
    fc.height_m = config.search_height_m + config.height_tolerance_m * 0.5f;
    command = search_sm.update(fc, VisionTarget{}, time(51.3));
    check(std::abs(command.vz + config.max_vertical_speed_mps) < 1e-6f,
          "search continuously sends upward vertical speed inside height tolerance");
    config.search_height_m = saved_search_height;
    std::cout << "PASS: search altitude feedback direction and relative-altitude priority\n";

    sm.reset(); fc.height_valid = true; fc.height_m = 0; fc.optical_flow_distance_m = 0.08f;
    command = sm.update(fc, VisionTarget{}, time(60));
    check(sm.state() == FlightState::Takeoff && command.vz == -config.takeoff_speed_mps,
          "zero relative altitude permits takeoff");
    std::cout << "PASS: valid relative altitude 0 m permits takeoff\n";
    fc.height_m = config.takeoff_height_m;
    sm.update(fc, VisionTarget{}, time(61));
    check(sm.state() == FlightState::Search, "configured height enters Search");

    sm.reset(); fc.height_valid = false;
    command = sm.update(fc, VisionTarget{}, time(62));
    check(sm.state() == FlightState::Takeoff && command.vz < 0,
          "valid optical flow permits takeoff without relative altitude");
    fc.optical_flow_distance_valid = false;
    command = sm.update(fc, VisionTarget{}, time(63));
    check(sm.state() == FlightState::Failsafe && command.vz == 0,
          "missing height prevents movement");
    fc.optical_flow_distance_valid = true;
    for (float bad : {std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(), -0.1f}) {
        sm.reset(); fc.height_valid = true; fc.height_m = bad;
        command = sm.update(fc, VisionTarget{}, time(64));
        check(sm.state() == FlightState::Failsafe && command.vz == 0,
              "invalid relative height prevents movement");
    }
    sm.reset(); fc.height_valid = false; fc.heartbeat_valid = false;
    command = sm.update(fc, VisionTarget{}, time(65));
    check(sm.state() == FlightState::Failsafe && command.vz == 0,
          "missing heartbeat prevents movement");
    std::cout << "PASS: target takeoff height, height fallback, missing data and invalid data\n";

    cv::FileStorage camera("config/camera_config.yaml", cv::FileStorage::READ);
    cv::FileStorage calibration("camera_calibration/camera_calibration.yaml", cv::FileStorage::READ);
    check(camera.isOpened() && calibration.isOpened(), "calibration fixtures readable");
    cv::Mat original;
    calibration["camera_matrix"] >> original;
    for (const auto size : {cv::Size(320,180), cv::Size(640,360), cv::Size(1280,720)}) {
        cv::Mat actual, expected = original.clone();
        expected.row(0) *= static_cast<double>(size.width) / static_cast<int>(calibration["image_width"]);
        expected.row(1) *= static_cast<double>(size.height) / static_cast<int>(calibration["image_height"]);
        check(readScaledCameraMatrix(camera, size, actual), "production calibration loader succeeds");
        check(cv::norm(actual, expected, cv::NORM_INF) < 1e-9, "runtime intrinsics match original calibration");
        const double ideal_radius_px = expected.at<double>(0,0) * .055 / (2 * .50);
        const double range = actual.at<double>(0,0) * .055 / (2 * ideal_radius_px);
        check(std::abs(range - .50) < 1e-9, "ideal 50 cm distance remains 50 cm");
        std::cout << "PASS: " << size.width << 'x' << size.height << " intrinsics and 50 cm range\n";
    }
    // 处理分辨率不属于标定参考尺寸，改动后不得使内参被重复缩放。
    const auto fixture = [&](int width, int height, cv::Mat matrix, bool include_width = true) {
        cv::FileStorage out("", cv::FileStorage::WRITE | cv::FileStorage::MEMORY);
        if (include_width) out << "calibration_image_width" << width;
        out << "calibration_image_height" << height;
        out << "input_width" << 640 << "input_height" << 360;
        out << "camera_matrix" << matrix;
        return out.releaseAndGetString();
    };
    cv::Mat loaded;
    cv::FileStorage changed_input(fixture(1280,720,original), cv::FileStorage::READ | cv::FileStorage::MEMORY);
    check(readScaledCameraMatrix(changed_input, cv::Size(320,180), loaded), "changed processing resolution loads");
    check(std::abs(loaded.at<double>(0,0) - original.at<double>(0,0)/4) < 1e-9,
          "processing config does not affect calibration scale");
    check(!readScaledCameraMatrix(changed_input, cv::Size(0,180), loaded) && loaded.empty(),
          "invalid output size rejected and output cleared");
    cv::Mat bad_focal = original.clone(); bad_focal.at<double>(0,0) = -1;
    cv::Mat bad_nan = original.clone(); bad_nan.at<double>(0,2) = std::numeric_limits<double>::quiet_NaN();
    cv::Mat bad_bottom = original.clone(); bad_bottom.at<double>(2,2) = 0;
    for (const auto& text : {fixture(0,720,original), fixture(1280,720,original,false),
                             fixture(1280,720,cv::Mat::eye(2,2,CV_64F)),
                             fixture(1280,720,bad_focal), fixture(1280,720,bad_nan),
                             fixture(1280,720,bad_bottom)}) {
        cv::FileStorage invalid(text, cv::FileStorage::READ | cv::FileStorage::MEMORY);
        loaded = original.clone();
        check(!readScaledCameraMatrix(invalid, cv::Size(320,180), loaded) && loaded.empty(),
              "invalid calibration rejected and output cleared");
    }
    std::cout << "PASS: processing resolution independence and invalid calibration rejection\n";
}
