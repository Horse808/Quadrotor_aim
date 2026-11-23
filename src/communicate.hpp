#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

// 飞控遥测快照。所有 valid 字段都必须同时满足新鲜度和数值范围检查。
struct FcState {
    bool heartbeat_valid = false;
    bool armed = false;
    bool offboard_mode = false;

    // OPTICAL_FLOW_RAD.distance：光流/测距单元报告的距离，单位米。
    float optical_flow_distance_m = 0.0f;
    bool optical_flow_distance_valid = false;

    // GLOBAL_POSITION_INT.relative_alt：相对起飞点高度，单位米。
    float height_m = 0.0f;
    bool height_valid = false;
};

// 应用层坐标约定：vx < 0 向前，vy < 0 向右，vz > 0 下降，vyaw > 0 逆时针。
struct FcCommand {
    float vx = 0.0f;
    float vy = 0.0f;
    float vz = 0.0f;
    float vyaw = 0.0f;
    bool land_done = false;
};

struct CommunicateConfig {
    std::string device = "/dev/ttyS1";
    int baud = 230400;
    int target_system = 1;
    int target_component = 1;
    int source_system = 1;
    int source_component = 191;  // MAV_COMP_ID_ONBOARD_COMPUTER
    int send_rate_hz = 20;
    int command_timeout_ms = 500;
    bool enable_motion = false;  // false 时即使调用 sendCommand 也只发零速度。
};

// MAVLink 2 串口桥接。负责遥测解析、速度设定值以及受控的模式/解锁命令。
class Communicate {
public:
    Communicate() = default;
    ~Communicate();
    Communicate(const Communicate&) = delete;
    Communicate& operator=(const Communicate&) = delete;

    bool init(const CommunicateConfig& config);
    void shutdown();
    FcState readState() const;
    void sendCommand(const FcCommand& command);

    bool waitForHeartbeat(int timeout_ms) const;
    bool requestOpticalFlow(int rate_hz, int timeout_ms);
    bool setOffboard(int timeout_ms);
    bool arm(int timeout_ms);
    bool disarm(int timeout_ms, bool force = false);

    bool isOpen() const { return running_.load(); }
    bool motionEnabled() const { return config_.enable_motion; }

private:
    void workerLoop();
    void consumeBytes(const uint8_t* data, size_t size);
    bool writeSetpoint(const FcCommand& command);
    bool writePacket(const uint8_t* packet, size_t size);
    bool sendCommandLong(uint16_t command,
                         float p1, float p2, float p3,
                         float p4, float p5, float p6, float p7,
                         int timeout_ms);
    bool waitForAck(uint16_t command, uint64_t previous_generation, int timeout_ms);
    bool waitForState(bool (*predicate)(const FcState&), int timeout_ms) const;

    CommunicateConfig config_;
    int fd_ = -1;
    bool initialized_ = false;
    std::atomic<bool> running_{false};
    std::thread worker_;

    mutable std::mutex state_mutex_;
    mutable std::condition_variable state_cv_;
    FcState latest_state_;
    std::chrono::steady_clock::time_point state_received_at_{};
    std::chrono::steady_clock::time_point optical_flow_distance_received_at_{};
    std::chrono::steady_clock::time_point heartbeat_received_at_{};
    uint64_t ack_generation_ = 0;
    uint16_t last_ack_command_ = 0;
    uint8_t last_ack_result_ = 255;

    std::mutex command_mutex_;
    FcCommand latest_command_;
    std::chrono::steady_clock::time_point command_received_at_{};
    std::mutex tx_mutex_;
    std::mutex sequence_mutex_;
    uint8_t tx_sequence_ = 0;
    std::chrono::steady_clock::time_point started_at_{};
    uint8_t rx_buffer_[4096]{};
    size_t rx_size_ = 0;
};
