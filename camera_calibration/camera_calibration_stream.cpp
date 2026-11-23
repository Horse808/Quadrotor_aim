#include "camera.hpp"

#include <opencv2/imgproc.hpp>

#include <arpa/inet.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <poll.h>
#include <vector>

#include <sys/socket.h>
#include <unistd.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace {

// Calibration must use the native sensor resolution. Change these values only
// when the sensor mode in /mnt/data/sensor_cfg.ini changes.
constexpr int kSensorWidth = 1280;
constexpr int kSensorHeight = 720;
constexpr int kSensorFps = 60;

// Calibration-stream parameters are intentionally kept in this source file,
// not in config/camera_config.yaml.
constexpr int kStreamPort = 5002;
constexpr int kStreamFps = 10;
constexpr int kJpegQuality = 85;

static_assert(kStreamPort >= 1 && kStreamPort <= 65535);
static_assert(kStreamFps >= 1 && kStreamFps <= kSensorFps);
static_assert(kJpegQuality >= 1 && kJpegQuality <= 100);

std::atomic<bool> g_running{true};

void handleSignal(int) {
    g_running.store(false);
}

void appendJpeg(void* context, void* data, int size) {
    if (context == nullptr || data == nullptr || size <= 0) return;
    auto& output = *static_cast<std::vector<std::uint8_t>*>(context);
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    output.insert(output.end(), bytes, bytes + size);
}

bool encodeGrayJpeg(const cv::Mat& gray, std::vector<std::uint8_t>& jpeg) {
    if (gray.empty() || gray.type() != CV_8UC1 || !gray.isContinuous()) return false;
    jpeg.clear();
    jpeg.reserve(static_cast<std::size_t>(gray.total() / 2));
    return stbi_write_jpg_to_func(appendJpeg, &jpeg, gray.cols, gray.rows,
                                  1, gray.data, kJpegQuality) != 0 && !jpeg.empty();
}

class TcpMjpegServer {
public:
    ~TcpMjpegServer() { stop(); }

    bool start() {
        listener_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listener_ < 0) {
            std::cerr << "[calibration-stream] socket failed: " << std::strerror(errno) << '\n';
            return false;
        }
        int reuse = 1;
        ::setsockopt(listener_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_ANY);
        address.sin_port = htons(static_cast<std::uint16_t>(kStreamPort));
        if (::bind(listener_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0 ||
            ::listen(listener_, 1) < 0) {
            std::cerr << "[calibration-stream] bind/listen failed: "
                      << std::strerror(errno) << '\n';
            stop();
            return false;
        }
        std::cout << "[calibration-stream] grayscale MJPEG server listening on port "
                  << kStreamPort << '\n';
        return true;
    }

    void stop() {
        disconnectClient();
        if (listener_ >= 0) {
            ::shutdown(listener_, SHUT_RDWR);
            ::close(listener_);
            listener_ = -1;
        }
    }

    void acceptClientIfReady() {
        if (listener_ < 0 || client_ >= 0) return;
        pollfd descriptor{listener_, POLLIN, 0};
        if (::poll(&descriptor, 1, 0) <= 0 || !(descriptor.revents & POLLIN)) return;
        const int fd = ::accept(listener_, nullptr, nullptr);
        if (fd < 0) return;
        int send_buffer = 512 * 1024;
        ::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &send_buffer, sizeof(send_buffer));
        client_ = fd;
        std::cout << "[calibration-stream] computer connected\n";
    }

    bool hasClient() const { return client_ >= 0; }

    bool sendFrame(const std::vector<std::uint8_t>& jpeg) {
        if (client_ < 0 || jpeg.empty()) return false;
        std::size_t sent = 0;
        while (sent < jpeg.size() && g_running.load()) {
            const ssize_t result = ::send(client_, jpeg.data() + sent, jpeg.size() - sent,
                                          MSG_NOSIGNAL | MSG_DONTWAIT);
            if (result > 0) {
                sent += static_cast<std::size_t>(result);
                continue;
            }
            if (result < 0 && errno == EINTR) continue;
            if (result < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                pollfd descriptor{client_, POLLOUT, 0};
                const int poll_result = ::poll(&descriptor, 1, 100);
                if (poll_result > 0 && (descriptor.revents & POLLOUT)) continue;
            }
            disconnectClient();
            return false;
        }
        return sent == jpeg.size();
    }

private:
    void disconnectClient() {
        if (client_ >= 0) {
            ::shutdown(client_, SHUT_RDWR);
            ::close(client_);
            client_ = -1;
            std::cout << "[calibration-stream] computer disconnected\n";
        }
    }

    int listener_ = -1;
    int client_ = -1;
};

}  // namespace

int main() {
    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);

    Camera camera;
    if (!camera.open(0, kSensorWidth, kSensorHeight, kSensorFps,
                     kSensorWidth, kSensorHeight)) {
        std::cerr << "[calibration] camera open failed: " << camera.lastError() << '\n';
        return 1;
    }

    if (camera.usesHardwareScaledOutput()) {
        std::cout << "[calibration] VPSS output active at native "
                  << kSensorWidth << 'x' << kSensorHeight << '\n';
    } else {
        std::cout << "[calibration] VPSS unavailable; using native VI frame\n";
    }

    TcpMjpegServer server;
    if (!server.start()) {
        camera.close();
        return 1;
    }

    std::cout << "[calibration] stream=" << kSensorWidth << 'x' << kSensorHeight
              << " @ " << kStreamFps << " fps, JPEG quality=" << kJpegQuality << '\n';
    std::cout << "[calibration] use capture_calibration.py on the computer; press s or SPACE to save\n";

    cv::Mat gray;
    std::vector<std::uint8_t> jpeg;
    const auto frame_period = std::chrono::duration<double>(1.0 / kStreamFps);
    auto next_encode = std::chrono::steady_clock::now();
    auto report_time = std::chrono::steady_clock::now();
    std::uint64_t captured = 0;
    std::uint64_t encoded = 0;
    std::uint64_t sent = 0;
    std::uint64_t failures = 0;

    while (g_running.load()) {
        server.acceptClientIfReady();
        if (!camera.readGray(gray, 250)) {
            ++failures;
            std::cerr << "[calibration] frame read failed: " << camera.lastError() << '\n';
            continue;
        }
        ++captured;

        const auto now = std::chrono::steady_clock::now();
        if (now < next_encode) continue;
        next_encode = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(frame_period);

        if (!encodeGrayJpeg(gray, jpeg)) {
            ++failures;
            continue;
        }
        ++encoded;
        if (server.hasClient() && server.sendFrame(jpeg)) ++sent;

        const auto report_now = std::chrono::steady_clock::now();
        if (report_now - report_time >= std::chrono::seconds(5)) {
            const double elapsed = std::chrono::duration<double>(report_now - report_time).count();
            std::cout << "[calibration-stats] captured=" << captured / elapsed
                      << " fps, encoded=" << encoded / elapsed
                      << " fps, sent=" << sent / elapsed
                      << " fps, failures=" << failures << '\n';
            captured = encoded = sent = 0;
            failures = 0;
            report_time = report_now;
        }
    }

    server.stop();
    camera.close();
    std::cout << "[calibration] stopped\n";
    return 0;
}
