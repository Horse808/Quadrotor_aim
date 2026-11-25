#pragma once

#include "communicate.hpp"

#include <chrono>
#include <string>

// 状态机只负责把飞控状态和视觉结果转换成速度设定值，
// 不直接操作串口，也不负责切换飞控模式或解锁。
enum class FlightState {
    Takeoff,
    Search,
    Track,
    Land,
    Finished,
    Failsafe,
};

struct VisionTarget {
    bool valid = false;
    // 相机坐标系：x 向右、y 向下、z 向前，单位米。
    float x_m = 0.0f;
    float y_m = 0.0f;
    float z_m = 0.0f;
};

struct StateMachineConfig {
    float takeoff_height_m = 1.0f;
    float search_height_m = 1.0f;
    // 搜索阶段目标高度的上下允许误差，单位米。
    float height_tolerance_m = 0.06f;
    float target_distance_m = 0.50f;
    float target_distance_tolerance_m = 0.06f;
    float lateral_tolerance_m = 0.06f;
    float vertical_tolerance_m = 0.06f;
    float hold_seconds = 5.0f;
    float search_timeout_seconds = 30.0f;
    float target_loss_hold_seconds = 1.5f;
    // 光流/测距高度低于 10 cm 时停止下降并请求停桨。
    float landing_height_m = 0.10f;
    float max_forward_speed_mps = 0.12f;
    float max_lateral_speed_mps = 0.10f;
    // 对应 vertical_speed_mps：悬停向上速度，同时限制视觉垂直对齐速度。
    float max_vertical_speed_mps = 0.08f;
    // 起飞及搜索高度超出容差时的修正速度，不受上述视觉速度上限限制。
    float takeoff_speed_mps = 0.08f;
    float landing_speed_mps = 0.04f;
    float max_search_yaw_radps = 0.15f;
    float position_gain = 0.25f;
    float vertical_gain = 0.25f;
    float height_gain = 0.25f;
};

class FlightStateMachine {
public:
    explicit FlightStateMachine(StateMachineConfig config = {});

    void reset();
    FlightState state() const { return state_; }
    const char* stateName() const;
    bool terminal() const;

    // 当前高度优先使用估计器相对高度；搜索时必须额外使用有效的
    // OPTICAL_FLOW_RAD.distance 进行高度保护。
    FcCommand update(const FcState& fc,
                     const VisionTarget& target,
                     std::chrono::steady_clock::time_point now);

private:
    void changeState(FlightState next, std::chrono::steady_clock::time_point now);
    float clamp(float value, float limit) const;
    bool freshAndUsable(const FcState& fc) const;
    FcCommand zeroCommand() const;

    StateMachineConfig config_;
    FlightState state_ = FlightState::Takeoff;
    std::chrono::steady_clock::time_point state_started_{};
    std::chrono::steady_clock::time_point target_hold_started_{};
    std::chrono::steady_clock::time_point last_target_seen_{};
    VisionTarget last_target_{};
};
