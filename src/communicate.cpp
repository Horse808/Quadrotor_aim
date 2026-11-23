#include "communicate.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <termios.h>
#include <type_traits>
#include <unistd.h>

namespace {
constexpr uint8_t kMavlinkV2Magic = 0xFD;
constexpr uint32_t kHeartbeatId = 0;
constexpr uint8_t kHeartbeatCrcExtra = 50;
constexpr uint32_t kSetModeId = 11;
constexpr uint8_t kSetModeCrcExtra = 89;
constexpr uint32_t kGlobalPositionIntId = 33;
constexpr uint8_t kGlobalPositionIntCrcExtra = 104;
constexpr uint32_t kCommandLongId = 76;
constexpr uint8_t kCommandLongCrcExtra = 152;
constexpr uint32_t kCommandAckId = 77;
constexpr uint8_t kCommandAckCrcExtra = 143;
constexpr uint32_t kOpticalFlowRadId = 106;
constexpr uint8_t kOpticalFlowRadCrcExtra = 138;
constexpr uint32_t kSetPositionTargetLocalNedId = 84;
constexpr uint8_t kSetPositionTargetLocalNedCrcExtra = 143;
constexpr uint8_t kFrameBodyNed = 8;

constexpr uint16_t kMavCmdSetMessageInterval = 511;
constexpr uint16_t kMavCmdComponentArmDisarm = 400;
constexpr uint16_t kMavCmdDoSetMode = 176;
constexpr uint8_t kMavResultAccepted = 0;
constexpr uint8_t kMavResultInProgress = 5;
constexpr uint8_t kMavModeFlagCustomModeEnabled = 1;
constexpr uint8_t kPx4CustomMainModeOffboard = 6;
constexpr uint8_t kMavModeFlagArmed = 128;

uint16_t crcAccumulate(uint8_t byte, uint16_t crc) {
    uint8_t tmp = byte ^ static_cast<uint8_t>(crc & 0xff);
    tmp ^= static_cast<uint8_t>(tmp << 4);
    return static_cast<uint16_t>((crc >> 8) ^
                                 (static_cast<uint16_t>(tmp) << 8) ^
                                 (static_cast<uint16_t>(tmp) << 3) ^
                                 (tmp >> 4));
}

uint16_t mavlinkCrc(const uint8_t* bytes, size_t size, uint8_t extra) {
    uint16_t crc = 0xffff;
    for (size_t i = 0; i < size; ++i) crc = crcAccumulate(bytes[i], crc);
    return crcAccumulate(extra, crc);
}

template <typename T>
void putLe(uint8_t* dest, T value) {
    static_assert(std::is_trivially_copyable<T>::value, "POD required");
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    uint8_t bytes[sizeof(T)];
    std::memcpy(bytes, &value, sizeof(T));
    std::reverse(bytes, bytes + sizeof(T));
    std::memcpy(dest, bytes, sizeof(T));
#else
    std::memcpy(dest, &value, sizeof(T));
#endif
}

template <typename T>
T getLe(const uint8_t* src) {
    T value{};
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    uint8_t bytes[sizeof(T)];
    std::memcpy(bytes, src, sizeof(T));
    std::reverse(bytes, bytes + sizeof(T));
    std::memcpy(&value, bytes, sizeof(T));
#else
    std::memcpy(&value, src, sizeof(T));
#endif
    return value;
}

speed_t serialSpeed(int baud) {
    switch (baud) {
        case 9600: return B9600;
        case 19200: return B19200;
        case 38400: return B38400;
        case 57600: return B57600;
        case 115200: return B115200;
#ifdef B230400
        case 230400: return B230400;
#endif
#ifdef B460800
        case 460800: return B460800;
#endif
#ifdef B921600
        case 921600: return B921600;
#endif
        default: return 0;
    }
}

bool validCommand(const FcCommand& c) {
    return std::isfinite(c.vx) && std::isfinite(c.vy) &&
           std::isfinite(c.vz) && std::isfinite(c.vyaw);
}

bool heartbeatPredicate(const FcState& state) { return state.heartbeat_valid; }
bool offboardPredicate(const FcState& state) {
    return state.heartbeat_valid && state.offboard_mode;
}
bool armedPredicate(const FcState& state) {
    return state.heartbeat_valid && state.armed;
}
bool disarmedPredicate(const FcState& state) {
    return state.heartbeat_valid && !state.armed;
}
}  // namespace

Communicate::~Communicate() { shutdown(); }

bool Communicate::init(const CommunicateConfig& config) {
    shutdown();
    const speed_t speed = serialSpeed(config.baud);
    if (speed == 0 || config.device.empty() ||
        config.target_system < 1 || config.target_system > 255 ||
        config.target_component < 1 || config.target_component > 255 ||
        config.source_system < 1 || config.source_system > 255 ||
        config.source_component < 1 || config.source_component > 255 ||
        config.send_rate_hz < 3 || config.send_rate_hz > 100 ||
        config.command_timeout_ms < 100 || config.command_timeout_ms > 5000) {
        std::cerr << "[FC] 无效的串口或 MAVLink 配置。\n";
        return false;
    }

    fd_ = ::open(config.device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) {
        std::cerr << "[FC] 无法打开 " << config.device << ": " << std::strerror(errno) << '\n';
        return false;
    }
    termios tty{};
    if (tcgetattr(fd_, &tty) != 0) {
        std::cerr << "[FC] tcgetattr 失败: " << std::strerror(errno) << '\n';
        shutdown();
        return false;
    }
    cfmakeraw(&tty);
    cfsetispeed(&tty, speed);
    cfsetospeed(&tty, speed);
    tty.c_cflag |= CLOCAL | CREAD;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
#ifdef CRTSCTS
    tty.c_cflag &= ~CRTSCTS;
#endif
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 0;
    if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
        std::cerr << "[FC] tcsetattr 失败: " << std::strerror(errno) << '\n';
        shutdown();
        return false;
    }
    tcflush(fd_, TCIOFLUSH);

    config_ = config;
    rx_size_ = 0;
    tx_sequence_ = 0;
    started_at_ = std::chrono::steady_clock::now();
    command_received_at_ = {};
    latest_command_ = {};
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        latest_state_ = {};
        state_received_at_ = {};
        optical_flow_distance_received_at_ = {};
        heartbeat_received_at_ = {};
        ack_generation_ = 0;
        last_ack_command_ = 0;
        last_ack_result_ = 255;
    }
    running_.store(true);
    try {
        worker_ = std::thread(&Communicate::workerLoop, this);
    } catch (const std::system_error& error) {
        running_.store(false);
        std::cerr << "[FC] 无法创建通信线程: " << error.what() << '\n';
        ::close(fd_);
        fd_ = -1;
        return false;
    }
    initialized_ = true;
    std::cout << "[FC] MAVLink 2 UART " << config.device << " @ " << config.baud
              << "，设定值 " << config.send_rate_hz << " Hz，运动控制 "
              << (config.enable_motion ? "开启" : "关闭（只发零速度）") << '\n';
    return true;
}

void Communicate::shutdown() {
    running_.store(false);
    state_cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    if (fd_ >= 0) {
        // 先让通信线程退出，再补发有限次数零速度；即使串口已经
        // 断开也不阻塞清理流程。
        if (initialized_) {
            for (int i = 0; i < 3; ++i) {
                if (!writeSetpoint(FcCommand{})) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        }
        ::close(fd_);
        fd_ = -1;
    }
    initialized_ = false;
}

FcState Communicate::readState() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    FcState result = latest_state_;
    const auto now = std::chrono::steady_clock::now();
    result.heartbeat_valid = heartbeat_received_at_ != std::chrono::steady_clock::time_point{} &&
        now - heartbeat_received_at_ <= std::chrono::seconds(2);
    if (state_received_at_ == std::chrono::steady_clock::time_point{} ||
        now - state_received_at_ > std::chrono::seconds(1)) {
        result.height_valid = false;
    }
    if (optical_flow_distance_received_at_ == std::chrono::steady_clock::time_point{} ||
        now - optical_flow_distance_received_at_ > std::chrono::milliseconds(500)) {
        result.optical_flow_distance_valid = false;
    }
    return result;
}

void Communicate::sendCommand(const FcCommand& command) {
    std::lock_guard<std::mutex> lock(command_mutex_);
    latest_command_ = validCommand(command) ? command : FcCommand{};
    command_received_at_ = std::chrono::steady_clock::now();
}

bool Communicate::waitForHeartbeat(int timeout_ms) const {
    return waitForState(heartbeatPredicate, timeout_ms);
}

bool Communicate::waitForState(bool (*predicate)(const FcState&), int timeout_ms) const {
    if (timeout_ms <= 0) return false;
    std::unique_lock<std::mutex> lock(state_mutex_);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (running_.load()) {
        FcState snapshot = latest_state_;
        const auto now = std::chrono::steady_clock::now();
        snapshot.heartbeat_valid = heartbeat_received_at_ != std::chrono::steady_clock::time_point{} &&
            now - heartbeat_received_at_ <= std::chrono::seconds(2);
        if (optical_flow_distance_received_at_ == std::chrono::steady_clock::time_point{} ||
            now - optical_flow_distance_received_at_ > std::chrono::milliseconds(500)) {
            snapshot.optical_flow_distance_valid = false;
        }
        if (state_received_at_ == std::chrono::steady_clock::time_point{} ||
            now - state_received_at_ > std::chrono::seconds(1)) {
            snapshot.height_valid = false;
        }
        if (predicate(snapshot)) return true;
        if (state_cv_.wait_until(lock, deadline) == std::cv_status::timeout) break;
    }
    return false;
}

bool Communicate::writePacket(const uint8_t* packet, size_t size) {
    if (fd_ < 0 || packet == nullptr || size == 0) return false;
    std::lock_guard<std::mutex> lock(tx_mutex_);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    size_t sent = 0;
    while (sent < size) {
        const ssize_t n = ::write(fd_, packet + sent, size - sent);
        if (n > 0) {
            sent += static_cast<size_t>(n);
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) return false;
            pollfd pfd{fd_, POLLOUT, 0};
            const int wait_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
            if (::poll(&pfd, 1, std::max(1, wait_ms)) > 0 && (pfd.revents & POLLOUT)) continue;
            continue;
        }
        std::cerr << "[FC] UART 写入失败: " << std::strerror(errno) << '\n';
        return false;
    }
    return true;
}

bool Communicate::writeSetpoint(const FcCommand& app) {
    FcCommand command = app;
    bool heartbeat_fresh = false;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        heartbeat_fresh = heartbeat_received_at_ != std::chrono::steady_clock::time_point{} &&
            std::chrono::steady_clock::now() - heartbeat_received_at_ <= std::chrono::seconds(2);
    }
    if (!config_.enable_motion || !heartbeat_fresh) command = {};

    // 应用坐标转换为 PX4 BODY_NED：+X 前、+Y 右、+Z 下。
    const float vx = -command.vx;
    const float vy = -command.vy;
    const float vz = command.vz;
    const float yaw_rate = -command.vyaw;

    // MAVLink SET_POSITION_TARGET_LOCAL_NED 的 type_mask：只使用 vx/vy/vz
    // 和 yaw_rate，忽略位置、加速度以及 yaw。
    constexpr uint16_t kVelocityOnlyTypeMask = 0x05C7;
    uint8_t packet[10 + 53 + 2]{};
    packet[0] = kMavlinkV2Magic;
    packet[1] = 53;
    {
        std::lock_guard<std::mutex> lock(sequence_mutex_);
        packet[4] = tx_sequence_++;
    }
    packet[5] = static_cast<uint8_t>(config_.source_system);
    packet[6] = static_cast<uint8_t>(config_.source_component);
    packet[7] = static_cast<uint8_t>(kSetPositionTargetLocalNedId);
    packet[8] = 0;
    packet[9] = 0;
    uint8_t* p = packet + 10;
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started_at_).count();
    putLe<uint32_t>(p + 0, static_cast<uint32_t>(elapsed));
    putLe<float>(p + 16, vx);
    putLe<float>(p + 20, vy);
    putLe<float>(p + 24, vz);
    putLe<float>(p + 44, yaw_rate);
    putLe<uint16_t>(p + 48, kVelocityOnlyTypeMask);
    p[50] = static_cast<uint8_t>(config_.target_system);
    p[51] = static_cast<uint8_t>(config_.target_component);
    p[52] = kFrameBodyNed;
    const uint16_t crc = mavlinkCrc(packet + 1, 9 + 53, kSetPositionTargetLocalNedCrcExtra);
    packet[63] = static_cast<uint8_t>(crc & 0xff);
    packet[64] = static_cast<uint8_t>(crc >> 8);
    return writePacket(packet, sizeof(packet));
}

bool Communicate::sendCommandLong(uint16_t command,
                                  float p1, float p2, float p3,
                                  float p4, float p5, float p6, float p7,
                                  int timeout_ms) {
    if (fd_ < 0 || !running_.load()) return false;
    uint64_t previous_generation;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        previous_generation = ack_generation_;
    }

    uint8_t packet[10 + 33 + 2]{};
    packet[0] = kMavlinkV2Magic;
    packet[1] = 33;
    {
        std::lock_guard<std::mutex> lock(sequence_mutex_);
        packet[4] = tx_sequence_++;
    }
    packet[5] = static_cast<uint8_t>(config_.source_system);
    packet[6] = static_cast<uint8_t>(config_.source_component);
    packet[7] = static_cast<uint8_t>(kCommandLongId);
    uint8_t* p = packet + 10;
    putLe<float>(p + 0, p1);
    putLe<float>(p + 4, p2);
    putLe<float>(p + 8, p3);
    putLe<float>(p + 12, p4);
    putLe<float>(p + 16, p5);
    putLe<float>(p + 20, p6);
    putLe<float>(p + 24, p7);
    putLe<uint16_t>(p + 28, command);
    p[30] = static_cast<uint8_t>(config_.target_system);
    p[31] = static_cast<uint8_t>(config_.target_component);
    p[32] = 0;
    const uint16_t crc = mavlinkCrc(packet + 1, 9 + 33, kCommandLongCrcExtra);
    packet[43] = static_cast<uint8_t>(crc & 0xff);
    packet[44] = static_cast<uint8_t>(crc >> 8);
    if (!writePacket(packet, sizeof(packet))) return false;
    return waitForAck(command, previous_generation, timeout_ms);
}

bool Communicate::waitForAck(uint16_t command, uint64_t previous_generation, int timeout_ms) {
    std::unique_lock<std::mutex> lock(state_mutex_);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (running_.load()) {
        if (ack_generation_ > previous_generation && last_ack_command_ == command) {
            return last_ack_result_ == kMavResultAccepted || last_ack_result_ == kMavResultInProgress;
        }
        if (state_cv_.wait_until(lock, deadline) == std::cv_status::timeout) break;
    }
    return false;
}

bool Communicate::requestOpticalFlow(int rate_hz, int timeout_ms) {
    if (rate_hz <= 0 || timeout_ms <= 0 || !waitForHeartbeat(timeout_ms)) return false;
    const float interval_us = 1000000.0f / static_cast<float>(rate_hz);
    const bool ok = sendCommandLong(kMavCmdSetMessageInterval,
                                    static_cast<float>(kOpticalFlowRadId), interval_us,
                                    0, 0, 0, 0, 0, timeout_ms);
    if (!ok) std::cerr << "[FC] 请求 OPTICAL_FLOW_RAD 失败。\n";
    return ok;
}

bool Communicate::setOffboard(int timeout_ms) {
    if (!waitForHeartbeat(timeout_ms)) return false;
    // PX4 custom_mode：主模式放在 bit 16..23，6 为 OFFBOARD。
    const float custom_main_mode = static_cast<float>(kPx4CustomMainModeOffboard);
    const bool ack = sendCommandLong(kMavCmdDoSetMode,
                                     static_cast<float>(kMavModeFlagCustomModeEnabled),
                                     custom_main_mode, 0, 0, 0, 0, 0, timeout_ms);
    if (!ack) return false;
    return waitForState(offboardPredicate, timeout_ms);
}

bool Communicate::arm(int timeout_ms) {
    if (!waitForHeartbeat(timeout_ms)) return false;
    const bool ack = sendCommandLong(kMavCmdComponentArmDisarm,
                                     1.0f, 0.0f, 0, 0, 0, 0, 0, timeout_ms);
    if (!ack) return false;
    return waitForState(armedPredicate, timeout_ms);
}

bool Communicate::disarm(int timeout_ms, bool force) {
    if (!waitForHeartbeat(timeout_ms)) return false;
    const bool ack = sendCommandLong(kMavCmdComponentArmDisarm,
                                     0.0f, force ? 21196.0f : 0.0f, 0, 0, 0, 0, 0, timeout_ms);
    if (!ack) return false;
    return waitForState(disarmedPredicate, timeout_ms);
}

void Communicate::consumeBytes(const uint8_t* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (rx_size_ == sizeof(rx_buffer_)) {
            std::memmove(rx_buffer_, rx_buffer_ + 1, --rx_size_);
        }
        rx_buffer_[rx_size_++] = data[i];
        while (rx_size_ >= 2) {
            size_t magic_index = 0;
            while (magic_index < rx_size_ && rx_buffer_[magic_index] != kMavlinkV2Magic &&
                   rx_buffer_[magic_index] != 0xFE) ++magic_index;
            if (magic_index) {
                std::memmove(rx_buffer_, rx_buffer_ + magic_index, rx_size_ - magic_index);
                rx_size_ -= magic_index;
            }
            if (rx_size_ < 2) break;
            const bool v2 = rx_buffer_[0] == kMavlinkV2Magic;
            const size_t header = v2 ? 10 : 6;
            const size_t payload_len = rx_buffer_[1];
            if (v2 && rx_size_ >= 3 && (rx_buffer_[2] & 1)) {
                const size_t signed_len = header + payload_len + 2 + 13;
                if (rx_size_ < signed_len) break;
                std::memmove(rx_buffer_, rx_buffer_ + signed_len, rx_size_ - signed_len);
                rx_size_ -= signed_len;
                continue;
            }
            const size_t frame_len = header + payload_len + 2;
            if (rx_size_ < frame_len) break;

            const uint8_t source_system = v2 ? rx_buffer_[5] : rx_buffer_[3];
            const uint8_t source_component = v2 ? rx_buffer_[6] : rx_buffer_[4];
            const uint32_t msgid = v2 ? (static_cast<uint32_t>(rx_buffer_[7]) |
                                         (static_cast<uint32_t>(rx_buffer_[8]) << 8) |
                                         (static_cast<uint32_t>(rx_buffer_[9]) << 16)) : rx_buffer_[5];
            uint8_t extra = 0;
            if (msgid == kHeartbeatId) extra = kHeartbeatCrcExtra;
            else if (msgid == kSetModeId) extra = kSetModeCrcExtra;
            else if (msgid == kGlobalPositionIntId) extra = kGlobalPositionIntCrcExtra;
            else if (msgid == kCommandAckId) extra = kCommandAckCrcExtra;
            else if (msgid == kOpticalFlowRadId) extra = kOpticalFlowRadCrcExtra;

            bool crc_ok = false;
            if (extra != 0) {
                const size_t crc_header_len = v2 ? 9 : 5;
                const uint16_t expected = mavlinkCrc(rx_buffer_ + 1,
                    crc_header_len + payload_len, extra);
                const uint8_t* checksum = rx_buffer_ + header + payload_len;
                const uint16_t got = static_cast<uint16_t>(checksum[0] |
                    (static_cast<uint16_t>(checksum[1]) << 8));
                crc_ok = got == expected;
            }
            if (crc_ok && source_system == config_.target_system) {
                const uint8_t* payload = rx_buffer_ + header;
                const auto now = std::chrono::steady_clock::now();
                std::lock_guard<std::mutex> lock(state_mutex_);
                if (msgid == kHeartbeatId && payload_len >= 9) {
                    const uint32_t custom_mode = getLe<uint32_t>(payload + 0);
                    const uint8_t base_mode = payload[6];
                    latest_state_.armed = (base_mode & kMavModeFlagArmed) != 0;
                    latest_state_.offboard_mode =
                        (base_mode & kMavModeFlagCustomModeEnabled) != 0 &&
                        (((custom_mode >> 16) & 0xffu) == kPx4CustomMainModeOffboard);
                    latest_state_.heartbeat_valid = true;
                    heartbeat_received_at_ = now;
                    state_cv_.notify_all();
                } else if (msgid == kCommandAckId && payload_len >= 3) {
                    last_ack_command_ = getLe<uint16_t>(payload + 0);
                    last_ack_result_ = payload[2];
                    ++ack_generation_;
                    state_cv_.notify_all();
                } else if (msgid == kGlobalPositionIntId && payload_len >= 20) {
                    const int32_t relative_alt_mm = getLe<int32_t>(payload + 16);
                    const bool valid_height = relative_alt_mm >= 0 && relative_alt_mm <= 100000;
                    latest_state_.height_valid = valid_height;
                    if (valid_height) {
                        latest_state_.height_m = relative_alt_mm * 0.001f;
                        state_received_at_ = now;
                    }
                } else if (msgid == kOpticalFlowRadId && payload_len >= 44) {
                    // distance 和 time_delta_distance_us 均来自 MAVLink 原始消息。
                    const float distance_m = getLe<float>(payload + 36);
                    const uint32_t distance_age_us = getLe<uint32_t>(payload + 32);
                    const bool valid_distance = std::isfinite(distance_m) &&
                        distance_m >= 0.01f && distance_m <= 100.0f &&
                        distance_age_us <= 500000;
                    latest_state_.optical_flow_distance_valid = valid_distance;
                    if (valid_distance) {
                        latest_state_.optical_flow_distance_m = distance_m;
                        optical_flow_distance_received_at_ = now;
                    }
                    state_cv_.notify_all();
                }
                (void)source_component;
            }
            std::memmove(rx_buffer_, rx_buffer_ + frame_len, rx_size_ - frame_len);
            rx_size_ -= frame_len;
        }
    }
}

void Communicate::workerLoop() {
    const auto period = std::chrono::microseconds(1000000 / config_.send_rate_hz);
    auto next_tx = std::chrono::steady_clock::now();
    while (running_.load()) {
        pollfd descriptor{fd_, POLLIN, 0};
        const int ready = ::poll(&descriptor, 1, 10);
        if (ready > 0 && (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL))) {
            std::cerr << "[FC] UART 读取链路不可用。\n";
            running_.store(false);
            state_cv_.notify_all();
            break;
        }
        if (ready > 0 && (descriptor.revents & POLLIN)) {
            uint8_t bytes[1024];
            while (running_.load()) {
                const ssize_t n = ::read(fd_, bytes, sizeof(bytes));
                if (n > 0) {
                    consumeBytes(bytes, static_cast<size_t>(n));
                    continue;
                }
                if (n < 0 && errno == EINTR) continue;
                if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
                if (n == 0) break;
                std::cerr << "[FC] UART 读取失败: " << std::strerror(errno) << '\n';
                running_.store(false);
                state_cv_.notify_all();
                break;
            }
        } else if (ready < 0 && errno != EINTR) {
            std::cerr << "[FC] UART poll 失败: " << std::strerror(errno) << '\n';
            running_.store(false);
            state_cv_.notify_all();
            break;
        }

        const auto now = std::chrono::steady_clock::now();
        if (now >= next_tx) {
            FcCommand command{};
            {
                std::lock_guard<std::mutex> lock(command_mutex_);
                if (command_received_at_ != std::chrono::steady_clock::time_point{} &&
                    now - command_received_at_ <= std::chrono::milliseconds(config_.command_timeout_ms)) {
                    command = latest_command_;
                }
            }
            if (!writeSetpoint(command)) {
                running_.store(false);
                state_cv_.notify_all();
                break;
            }
            next_tx = now + period;
        }
    }
}
