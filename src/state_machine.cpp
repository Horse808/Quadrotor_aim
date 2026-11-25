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
    last_target_seen_ = {};
    last_target_ = {};
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
    if (next == FlightState::Search) {
        last_target_seen_ = {};
        last_target_ = {};
    }
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
            // 地面上的有效相对高度可以为 0，不要求先达到最小离地高度才起飞。
            // 仍拒绝非有限值及无效负值，并在下方保留起飞目标高度判断。
            if (!std::isfinite(height) || height < 0.0f) {
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
                last_target_ = target;
                last_target_seen_ = now;
                changeState(FlightState::Track, now);
                break;
            }
            if (now - state_started_ >=
                std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<float>(config_.search_timeout_seconds))) {
                changeState(FlightState::Land, now);
                break;
            }
            // GLOBAL_POSITION_INT.relative_alt 是相对起飞点高度，优先用它稳高；
            // 遥测短暂不可用时，才回退到有效的光流测距高度。
            const float height = fc.height_valid ? fc.height_m : fc.optical_flow_distance_m;
            if (!std::isfinite(height) || height < 0.0f) {
                changeState(FlightState::Failsafe, now);
                break;
            }
            const float height_error = config_.search_height_m - height;
            // 容差外使用起飞速度修正高度；容差内持续发送向上的垂直速度。
            // 两种速度互相覆盖，不叠加，也不再由高度增益缩小。
            if (std::abs(height_error) > config_.height_tolerance_m) {
                // BODY_NED 的 +Z 向下，低于目标高度时输出负 vz 上升。
                command.vz = height_error > 0.0f
                                 ? -config_.takeoff_speed_mps
                                 : config_.takeoff_speed_mps;
            } else {
                // 配置为正的速度大小；BODY_NED 的向上速度需要取负号。
                command.vz = -config_.max_vertical_speed_mps;
            }
            command.vyaw = config_.max_search_yaw_radps;
            break;
        }

        case FlightState::Track: {
            if (target.valid) {
                last_target_ = target;
                last_target_seen_ = now;
            } else {
                const auto hold_duration = std::chrono::duration_cast<
                    std::chrono::steady_clock::duration>(
                    std::chrono::duration<float>(config_.target_loss_hold_seconds));
                if (last_target_seen_ == std::chrono::steady_clock::time_point{} ||
                    now - last_target_seen_ > hold_duration) {
                    changeState(FlightState::Search, now);
                    break;
                }
            }
            const VisionTarget& active_target = target.valid ? target : last_target_;
            const float distance_error = active_target.z_m - config_.target_distance_m;
            const float horizontal_angle = std::atan2(active_target.x_m, active_target.z_m);

            // 转向、靠近和到位等待时持续发送向上的配置垂直速度；
            // 下方的视觉上下对齐控制仍会覆盖此值，不叠加两种速度。
            command.vz = -config_.max_vertical_speed_mps;

            // 先調整目標的上下角度，使灯中心与机体高度对齐；垂直误差进入
            // 容差后再通过偏航旋转把灯转到画面中心，最后才前后靠近。
            if (std::abs(active_target.y_m) > config_.vertical_tolerance_m) {
                command.vz = clamp(config_.vertical_gain * active_target.y_m,
                                   config_.max_vertical_speed_mps);
            } else if (std::abs(active_target.x_m) > config_.lateral_tolerance_m) {
                command.vyaw = clamp(-horizontal_angle,
                                     config_.max_search_yaw_radps);
            } else {
                command.vx = clamp(-config_.position_gain * distance_error,
                                   config_.max_forward_speed_mps);
            }

            const bool in_position = std::abs(distance_error) <=
                                         config_.target_distance_tolerance_m &&
                                     std::abs(active_target.x_m) <= config_.lateral_tolerance_m &&
                                     std::abs(active_target.y_m) <= config_.vertical_tolerance_m;
            if (target.valid && in_position) {
                if (target_hold_started_ == std::chrono::steady_clock::time_point{}) {
                    target_hold_started_ = now;
                } else if (now - target_hold_started_ >=
                           std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                               std::chrono::duration<float>(config_.hold_seconds))) {
                    changeState(FlightState::Land, now);
                }
            } else {
                // 位置稳定计时只接受实时识别，丢帧期间不累计稳定时间。
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
                // 主程序在正常模式下看到 land_done 后发送强制停桨指令；测试模式
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
