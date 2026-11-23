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

constexpr int kSensorWidth = 1280;
constexpr int kSensorHeight = 720;
constexpr int kSensorFps = 60;

// Test-only parameters. Change these values and rebuild; they are intentionally
// not read from config/camera_config.yaml.
constexpr int kOutputWidth = 360;
constexpr int kOutputHeight = 180;
constexpr int kStreamPort = 5001;
constexpr int kStreamFps = 30;
constexpr int kJpegQuality = 80;

static_assert(kOutputWidth >= 2 && kOutputHeight >= 2,
              "test output dimensions must be at least 2 pixels");
static_assert((kOutputWidth % 2) == 0 && (kOutputHeight % 2) == 0,
              "test output dimensions must be even");
static_assert(kOutputWidth <= kSensorWidth && kOutputHeight <= kSensorHeight,
              "test output dimensions cannot exceed the sensor dimensions");
static_assert(kStreamPort >= 1 && kStreamPort <= 65535,
              "test stream port must be in the TCP port range");
static_assert(kStreamFps >= 1 && kStreamFps <= kSensorFps,
              "test stream fps must be supported by the sensor");
static_assert(kJpegQuality >= 1 && kJpegQuality <= 100,
              "test JPEG quality must be between 1 and 100");

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

bool encodeGrayJpeg(const cv::Mat& gray, int quality, std::vector<std::uint8_t>& jpeg) {
    if (gray.empty() || gray.type() != CV_8UC1 || !gray.isContinuous()) return false;
    jpeg.clear();
    jpeg.reserve(static_cast<std::size_t>(gray.total() / 2));
    return stbi_write_jpg_to_func(appendJpeg, &jpeg, gray.cols, gray.rows,
                                  1, gray.data, quality) != 0 && !jpeg.empty();
}

class TcpMjpegServer {
public:
    ~TcpMjpegServer() { stop(); }

    bool start(int port) {
        listener_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listener_ < 0) {
            std::cerr << "[stream] socket failed: " << std::strerror(errno) << '\n';
            return false;
        }
        int reuse = 1;
        ::setsockopt(listener_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_ANY);
        address.sin_port = htons(static_cast<std::uint16_t>(port));
        if (::bind(listener_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0 ||
            ::listen(listener_, 1) < 0) {
            std::cerr << "[stream] bind/listen failed: " << std::strerror(errno) << '\n';
            stop();
            return false;
        }
        std::cout << "[stream] software MJPEG TCP server listening on port " << port << '\n';
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
        int send_buffer = 256 * 1024;
        ::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &send_buffer, sizeof(send_buffer));
        client_ = fd;
        std::cout << "[stream] laptop connected\n";
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
                const int poll_result = ::poll(&descriptor, 1, 50);
                if (poll_result > 0 && (descriptor.revents & POLLOUT)) continue;
            }
            disconnectClient();
            return false;
        }
        if (sent == jpeg.size()) return true;
        disconnectClient();
        return false;
    }

private:
    void disconnectClient() {
        if (client_ >= 0) {
            ::shutdown(client_, SHUT_RDWR);
            ::close(client_);
            client_ = -1;
        }
    }

    int listener_ = -1;
    int client_ = -1;
};

} // namespace

int main() {
    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);

    Camera camera;
    if (!camera.open(0, kSensorWidth, kSensorHeight, kSensorFps,
                     kOutputWidth, kOutputHeight)) {
        std::cerr << "[camera] open failed: " << camera.lastError() << '\n';
        return 1;
    }
    if (camera.usesHardwareScaledOutput()) {
        std::cout << "[camera] VPSS output active: "
                  << kOutputWidth << 'x' << kOutputHeight << '\n';
    } else {
        std::cout << "[camera] VPSS output unavailable; using VI "
                  << kSensorWidth << 'x' << kSensorHeight
                  << " and CPU resize to " << kOutputWidth << 'x' << kOutputHeight << '\n';
    }

    TcpMjpegServer server;
    if (!server.start(kStreamPort)) {
        camera.close();
        return 1;
    }

    std::cout << "[test] capture-only grayscale MJPEG stream: "
              << kOutputWidth << 'x' << kOutputHeight
              << " @ " << kStreamFps << " fps, quality " << kJpegQuality
              << "; no image processing\n";
    std::cout << "[test] laptop: ffplay -fflags nobuffer -flags low_delay -framedrop "
                 "-f mjpeg -framerate " << kStreamFps
              << " tcp://<nano-ip>:" << kStreamPort << '\n';

    cv::Mat gray_frame;
    cv::Mat resized_gray;
    std::vector<std::uint8_t> jpeg;
    const auto frame_period = std::chrono::duration<double>(1.0 / kStreamFps);
    auto next_encode = std::chrono::steady_clock::now();
    auto report_time = std::chrono::steady_clock::now();
    std::uint64_t captured = 0;
    std::uint64_t encoded = 0;
    std::uint64_t sent = 0;
    std::uint64_t read_failures = 0;
    std::uint64_t encode_failures = 0;
    std::uint64_t send_failures = 0;
    std::uint64_t last_captured = 0;
    std::uint64_t last_encoded = 0;
    std::uint64_t last_sent = 0;
    std::size_t bytes_since_report = 0;

    while (g_running.load()) {
        server.acceptClientIfReady();
        if (!camera.readGray(gray_frame, 250)) {
            ++read_failures;
            std::cerr << "[camera] frame read failed: " << camera.lastError() << '\n';
            continue;
        }
        ++captured;

        const auto now = std::chrono::steady_clock::now();
        if (now < next_encode) continue;
        next_encode = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(frame_period);

        if (gray_frame.cols != kOutputWidth || gray_frame.rows != kOutputHeight) {
            cv::resize(gray_frame, resized_gray, cv::Size(kOutputWidth, kOutputHeight),
                       0.0, 0.0, cv::INTER_AREA);
        } else {
            resized_gray = gray_frame;
        }
        if (!encodeGrayJpeg(resized_gray, kJpegQuality, jpeg)) {
            ++encode_failures;
            continue;
        }
        ++encoded;
        bytes_since_report += jpeg.size();

        if (server.hasClient()) {
            if (server.sendFrame(jpeg)) ++sent;
            else ++send_failures;
        }

        const auto report_now = std::chrono::steady_clock::now();
        if (report_now - report_time >= std::chrono::seconds(5)) {
            const double elapsed = std::chrono::duration<double>(report_now - report_time).count();
            const std::uint64_t captured_delta = captured - last_captured;
            const std::uint64_t encoded_delta = encoded - last_encoded;
            const std::uint64_t sent_delta = sent - last_sent;
            std::cout << "[stats] captured=" << captured_delta / elapsed
                      << " fps, encoded=" << encoded_delta / elapsed
                      << " fps, sent=" << sent_delta / elapsed
                      << " fps, average="
                      << (encoded_delta == 0 ? 0.0 :
                          static_cast<double>(bytes_since_report) / encoded_delta / 1024.0)
                      << " KiB/frame, read_failures=" << read_failures
                      << ", encode_failures=" << encode_failures
                      << ", send_failures=" << send_failures << '\n';
            report_time = report_now;
            last_captured = captured;
            last_encoded = encoded;
            last_sent = sent;
            bytes_since_report = 0;
        }
    }

    server.stop();
    camera.close();
    std::cout << "[test] stopped\n";
    return 0;
}
