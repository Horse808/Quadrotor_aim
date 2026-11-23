#include "state_machine.hpp"

#include <algorithm>
#include <cmath>

FlightStateMachine::FlightStateMachine(StateMachineConfig config)
    : config_(config) {
    reset();
}

void FlightStateMachine::reset() {
    state_ = FlightState::Takeoff;
    state_started_ = std::chrono::steady_clock::now();
    target_hold_started_ = {};
}

const char* FlightStateMachine::stateName() const {
    switch (state_) {
        case FlightState::Takeoff: return "TAKEOFF";
        case FlightState::Search: return "SEARCH";
        case FlightState::Track: return "TRACK";
        case FlightState::Land: return "LAND";
        case FlightState::Finished: return "FINISHED";
        case FlightState::Failsafe: return "FAILSAFE";
    }
    return "UNKNOWN";
}

bool FlightStateMachine::terminal() const {
    return state_ == FlightState::Finished || state_ == FlightState::Failsafe;
}

void FlightStateMachine::changeState(FlightState next,
                                     std::chrono::steady_clock::time_point now) {
    if (state_ == next) return;
    state_ = next;
    state_started_ = now;
    target_hold_started_ = {};
}

float FlightStateMachine::clamp(float value, float limit) const {
    if (!std::isfinite(value)) return 0.0f;
    return std::max(-limit, std::min(limit, value));
}

bool FlightStateMachine::freshAndUsable(const FcState& fc) const {
    return fc.heartbeat_valid && fc.optical_flow_distance_valid;
}

FcCommand FlightStateMachine::zeroCommand() const {
    return FcCommand{};
}

FcCommand FlightStateMachine::update(const FcState& fc,
                                     const VisionTarget& target,
                                     std::chrono::steady_clock::time_point now) {
    FcCommand command{};

    // 心跳或光流失效时禁止继续输出运动指令。搜索和降落都依赖光流高度
    // 保护，因此不能在数据失效后“保持上一次速度”。
    if (!freshAndUsable(fc)) {
        changeState(FlightState::Failsafe, now);
        return command;
    }

    switch (state_) {
        case FlightState::Takeoff: {
            if (target.valid) {
                changeState(FlightState::Track, now);
                break;
            }
            const float height = fc.height_valid ? fc.height_m : fc.optical_flow_distance_m;
            if (!std::isfinite(height) || height < 0.01f) {
                changeState(FlightState::Failsafe, now);
                break;
            }
            if (height + 0.05f < config_.takeoff_height_m) {
                // 应用层约定 vz > 0 为下降，所以起飞必须是负值。
                command.vz = -config_.takeoff_speed_mps;
            } else {
                changeState(FlightState::Search, now);
            }
            break;
        }

        case FlightState::Search: {
            if (target.valid) {
                changeState(FlightState::Track, now);
                break;
            }
            if (now - state_started_ >=
                std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<float>(config_.search_timeout_seconds))) {
                changeState(FlightState::Land, now);
                break;
            }
            // 搜索过程用光流测距值把高度拉回目标高度，避免旋转期间掉高。
            const float height_error = config_.search_height_m - fc.optical_flow_distance_m;
            command.vz = clamp(-config_.height_gain * height_error,
                               config_.max_vertical_speed_mps);
            // 搜索时只绕偏航轴旋转，不叠加前后或左右速度，避免目标
            // 不在视野内时产生无法验证的水平位移。
            command.vyaw = config_.max_search_yaw_radps;
            break;
        }

        case FlightState::Track: {
            if (!target.valid) {
                changeState(FlightState::Search, now);
                break;
            }
            const float distance_error = target.z_m - config_.target_distance_m;
            command.vx = clamp(-config_.position_gain * distance_error,
                               config_.max_forward_speed_mps);
            command.vy = clamp(-config_.position_gain * target.x_m,
                               config_.max_lateral_speed_mps);

            // 相机朝机头前方时，目标 y 坐标就是机体的上下误差：
            // y > 0 表示目标在画面下方，机体应向下（NED 的 +Z）修正。
            // 跟踪阶段优先把 ArUco 码对准画面中心，避免只满足距离和左右
            // 误差却仍然处在目标的斜上方或斜下方。
            command.vz = clamp(config_.vertical_gain * target.y_m,
                               config_.max_vertical_speed_mps);

            const bool in_position = std::abs(distance_error) <=
                                         config_.target_distance_tolerance_m &&
                                     std::abs(target.x_m) <= config_.lateral_tolerance_m &&
                                     std::abs(target.y_m) <= config_.vertical_tolerance_m;
            if (in_position) {
                if (target_hold_started_ == std::chrono::steady_clock::time_point{}) {
                    target_hold_started_ = now;
                } else if (now - target_hold_started_ >=
                           std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                               std::chrono::duration<float>(config_.hold_seconds))) {
                    changeState(FlightState::Land, now);
                }
            } else {
                target_hold_started_ = {};
            }
            break;
        }

        case FlightState::Land: {
            // 停桨阈值必须使用光流/测距模块的离地距离。GLOBAL_POSITION_INT
            // 在低空时可能有延迟，不能用它替代最后 10 cm 的安全判断。
            const float optical_height = fc.optical_flow_distance_m;
            if (!std::isfinite(optical_height) || optical_height < 0.0f) {
                changeState(FlightState::Failsafe, now);
                break;
            }
            if (optical_height <= config_.landing_height_m) {
                // 主程序在正常模式下看到 land_done 后发送 disarm；测试模式
                // 只结束状态机，不发送任何停桨/解锁相关命令。
                command.land_done = true;
                changeState(FlightState::Finished, now);
                break;
            }
            // 降落阶段不再依赖目标识别，但仍持续检查心跳、光流和高度。
            command.vz = config_.landing_speed_mps;
            break;
        }

        case FlightState::Finished:
        case FlightState::Failsafe:
            break;
    }
    return command;
}
