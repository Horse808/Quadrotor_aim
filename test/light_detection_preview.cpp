#include "camera.hpp"
#include "light_detector.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <arpa/inet.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <poll.h>
#include <string>
#include <thread>
#include <vector>

#include <sys/socket.h>
#include <unistd.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace {
constexpr int kSensorWidth = 1280;
constexpr int kSensorHeight = 720;
constexpr int kSensorFps = 60;
constexpr int kStreamPort = 5000;
constexpr int kStreamFps = 20;
constexpr int kJpegQuality = 75;
constexpr float kLightDiameterM = 0.055f;
std::atomic<bool> g_running{true};

void handleSignal(int) { g_running.store(false); }

void appendJpeg(void* context, void* data, int size) {
    if (!context || !data || size <= 0) return;
    auto& bytes = *static_cast<std::vector<std::uint8_t>*>(context);
    const auto* begin = static_cast<const std::uint8_t*>(data);
    bytes.insert(bytes.end(), begin, begin + size);
}

bool encodeJpeg(const cv::Mat& bgr, std::vector<std::uint8_t>& jpeg) {
    if (bgr.empty() || bgr.type() != CV_8UC3) return false;
    // OpenCV 使用 BGR 排列，而 stb 的 JPEG 接口要求 RGB；编码前显式换序，避免红蓝通道颠倒。
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    if (!rgb.isContinuous()) return false;
    jpeg.clear();
    jpeg.reserve(static_cast<std::size_t>(rgb.total()));
    return stbi_write_jpg_to_func(appendJpeg, &jpeg, rgb.cols, rgb.rows,
                                  3, rgb.data, kJpegQuality) != 0 && !jpeg.empty();
}

class TcpMjpegServer {
public:
    ~TcpMjpegServer() { stop(); }

    bool start(int port) {
        listener_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listener_ < 0) {
            std::cerr << "[传输] 创建 TCP 套接字失败：" << std::strerror(errno) << '\n';
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
            std::cerr << "[传输] 监听端口失败：" << std::strerror(errno) << '\n';
            stop();
            return false;
        }
        std::cout << "[传输] MJPEG TCP 服务已监听端口 " << port << '\n';
        return true;
    }

    void acceptClientIfReady() {
        if (listener_ < 0 || client_ >= 0) return;
        pollfd descriptor{listener_, POLLIN, 0};
        if (::poll(&descriptor, 1, 0) <= 0 || !(descriptor.revents & POLLIN)) return;
        client_ = ::accept(listener_, nullptr, nullptr);
        if (client_ >= 0) {
            int send_buffer = 256 * 1024;
            ::setsockopt(client_, SOL_SOCKET, SO_SNDBUF, &send_buffer, sizeof(send_buffer));
            std::cout << "[传输] 笔记本已连接\n";
        }
    }

    bool sendFrame(const std::vector<std::uint8_t>& jpeg) {
        if (client_ < 0 || jpeg.empty()) return false;
        std::size_t sent = 0;
        while (sent < jpeg.size() && g_running.load()) {
            const ssize_t count = ::send(client_, jpeg.data() + sent, jpeg.size() - sent,
                                         MSG_NOSIGNAL | MSG_DONTWAIT);
            if (count > 0) {
                sent += static_cast<std::size_t>(count);
                continue;
            }
            if (count < 0 && errno == EINTR) continue;
            if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                pollfd descriptor{client_, POLLOUT, 0};
                if (::poll(&descriptor, 1, 50) > 0 && (descriptor.revents & POLLOUT)) continue;
            }
            disconnectClient();
            return false;
        }
        if (sent == jpeg.size()) return true;
        disconnectClient();
        return false;
    }

    void stop() {
        disconnectClient();
        if (listener_ >= 0) {
            ::shutdown(listener_, SHUT_RDWR);
            ::close(listener_);
            listener_ = -1;
        }
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

bool readInt(const cv::FileStorage& file, const char* key, int& value) {
    const cv::FileNode node = file[key];
    if (node.empty()) return false;
    node >> value;
    return true;
}

bool readFloat(const cv::FileStorage& file, const char* key, float& value) {
    const cv::FileNode node = file[key];
    if (node.empty()) return false;
    node >> value;
    return std::isfinite(value);
}

bool loadCameraConfig(int& width, int& height, LightDetectorParams& params) {
    cv::FileStorage config("config/camera_config.yaml", cv::FileStorage::READ);
    if (!config.isOpened()) {
        std::cerr << "[配置] 无法打开 config/camera_config.yaml，请从项目目录启动程序。\n";
        return false;
    }
    bool ok = readInt(config, "input_width", width) && readInt(config, "input_height", height) &&
              readInt(config, "light_threshold", params.binary_threshold) &&
              readInt(config, "light_min_area", params.min_area) &&
              readInt(config, "light_max_area", params.max_area) &&
              readInt(config, "green_hue_min", params.green_hue_min) &&
              readInt(config, "green_hue_max", params.green_hue_max) &&
              readInt(config, "green_saturation_min", params.green_saturation_min) &&
              readInt(config, "green_value_min", params.green_value_min) &&
              readFloat(config, "light_min_circularity", params.min_circularity) &&
              readFloat(config, "light_max_aspect_ratio_diff", params.max_aspect_ratio_diff) &&
              readFloat(config, "min_green_ratio", params.min_green_ratio);
    if (!ok || width <= 0 || height <= 0 || (width & 1) || (height & 1) ||
        width > kSensorWidth || height > kSensorHeight || params.min_area < 0 ||
        params.max_area < params.min_area || params.green_hue_min < 0 ||
        params.green_hue_max > 179 || params.green_hue_min > params.green_hue_max ||
        params.green_saturation_min < 0 || params.green_saturation_min > 255 ||
        params.green_value_min < 0 || params.green_value_min > 255 ||
        params.min_circularity < 0.0f || params.min_circularity > 1.0f ||
        params.max_aspect_ratio_diff < 0.0f || params.min_green_ratio < 0.0f ||
        params.min_green_ratio > 1.0f) {
        std::cerr << "[配置] 相机尺寸或引导灯识别参数缺失/无效，请检查 camera_config.yaml。\n";
        return false;
    }
    return true;
}

void drawOverlay(cv::Mat& frame, const LightResult& result, double fps) {
    const cv::Point center(frame.cols / 2, frame.rows / 2);
    cv::drawMarker(frame, center, cv::Scalar(255, 255, 0), cv::MARKER_CROSS, 18, 1);
    if (result.found) {
        cv::rectangle(frame, result.bounding_box, cv::Scalar(0, 255, 0), 2);
        cv::drawMarker(frame, result.center, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 18, 2);
        cv::putText(frame, "GREEN FOUND", cv::Point(8, 24), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(0, 255, 0), 2);
        char info[128];
        std::snprintf(info, sizeof(info), "center=(%.0f,%.0f) conf=%.2f area=%.0f",
                      result.center.x, result.center.y, result.confidence, result.area);
        cv::putText(frame, info, cv::Point(8, frame.rows - 10), cv::FONT_HERSHEY_SIMPLEX,
                    0.42, cv::Scalar(255, 255, 0), 1);
    } else {
        cv::putText(frame, "GREEN NOT FOUND", cv::Point(8, 24), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(0, 0, 255), 2);
    }
    char fps_text[48];
    std::snprintf(fps_text, sizeof(fps_text), "FPS %.1f", fps);
    cv::putText(frame, fps_text, cv::Point(8, frame.rows - 10), cv::FONT_HERSHEY_SIMPLEX,
                0.5, cv::Scalar(255, 255, 0), 1);
}
}  // namespace

int main() {
    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);

    int width = 0, height = 0;
    LightDetectorParams params;
    if (!loadCameraConfig(width, height, params)) return 2;

    Camera camera;
    if (!camera.open(0, kSensorWidth, kSensorHeight, kSensorFps, width, height)) {
        std::cerr << "[相机] 打开失败：" << camera.lastError() << '\n';
        return 1;
    }
    if (!camera.configureExposureFromFile("config/camera_config.yaml")) {
        std::cerr << "[相机] 曝光配置失败：" << camera.lastError() << '\n';
        camera.close();
        return 1;
    }
    if (!camera.usesHardwareScaledOutput()) {
        std::cerr << "[相机] VPSS 硬件缩放不可用，停止测试以避免 CPU 回退造成帧率下降。"
                  << "请检查初始化日志及上次程序的资源释放情况。\n";
        camera.close();
        return 1;
    }

    TcpMjpegServer server;
    if (!server.start(kStreamPort)) {
        camera.close();
        return 1;
    }
    std::cout << "[测试] 分辨率=" << width << 'x' << height << "，识别参数来自 camera_config.yaml；"
              << "仅检测并发送预览，不连接飞控。\n";

    cv::Mat frame;
    std::vector<std::uint8_t> jpeg;
    std::uint64_t frames = 0, failures = 0;
    auto report_time = std::chrono::steady_clock::now();
    auto next_frame = report_time;
    auto last_report = report_time;

    while (g_running.load() && camera.isOpened()) {
        next_frame += std::chrono::microseconds(1000000 / kStreamFps);
        const auto start = std::chrono::steady_clock::now();
        server.acceptClientIfReady();
        if (!camera.readBgr(frame, 500)) {
            ++failures;
            if (failures % 20 == 1) std::cerr << "[相机] 取帧失败：" << camera.lastError() << '\n';
            std::this_thread::sleep_until(next_frame);
            continue;
        }
        const LightResult result = detectLight(frame, params, kLightDiameterM);
        const double elapsed = std::chrono::duration<double>(start - last_report).count();
        const double fps = elapsed > 0.0 ? 1.0 / elapsed : 0.0;
        drawOverlay(frame, result, fps);
        last_report = start;

        if (!encodeJpeg(frame, jpeg)) {
            ++failures;
            std::cerr << "[传输] JPEG 编码失败。\n";
        } else {
            server.acceptClientIfReady();
            if (server.sendFrame(jpeg)) ++frames;
        }

        const auto now = std::chrono::steady_clock::now();
        if (now - report_time >= std::chrono::seconds(2)) {
            std::cout << "[统计] 已编码并发送帧=" << frames << "，读取/编码失败=" << failures << '\n';
            frames = failures = 0;
            report_time = now;
        }
        std::this_thread::sleep_until(next_frame);
    }

    server.stop();
    camera.close();
    std::cout << "[测试] 已停止并释放相机及 VPSS 资源。\n";
    return 0;
}
