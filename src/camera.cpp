#include "camera.hpp"

#include <cmath>
#include <cstdio>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>

#include <cstdint>
#include <cstring>
#include <unistd.h>

#include <sample_comm.h>
#include <cvi_ae.h>
#include <cvi_isp.h>
#include <cvi_buffer.h>
#include <cvi_sys.h>
#include <cvi_vi.h>
#include <cvi_vb.h>
#include <cvi_vpss.h>

struct Camera::Impl {
    bool open(int device_id,
              int width,
              int height,
              int fps,
              int output_width,
              int output_height,
              Camera::OutputFormat output_format) {
        // 当前单目传感器由 SDK 的 sensor_cfg.ini 固定选择，只有一路 VI，
        // 因而 device_id 暂不参与选择；fps 同时用于校验和配置 VPSS 输出节拍。
        (void)device_id;
        output_format_ = output_format;
        return openLicheeRV(width, height, fps, output_width, output_height);
    }

    bool configureExposureFromFile(const std::string& config_path) {
        if (!opened) {
            setError("设置曝光前必须先打开相机");
            return false;
        }

        int exposure_time_us = 0;
        try {
            cv::FileStorage config(config_path, cv::FileStorage::READ);
            if (!config.isOpened()) {
                setError("无法打开相机曝光配置：" + config_path);
                return false;
            }
            const cv::FileNode node = config["exposure_time_us"];
            // 旧配置没有此项时仍使用自动曝光，避免影响既有部署。
            if (!node.empty()) {
                if (!node.isInt()) {
                    setError("exposure_time_us 必须是非负整数，单位为微秒");
                    return false;
                }
                node >> exposure_time_us;
                if (exposure_time_us < 0) {
                    setError("exposure_time_us 不能为负数");
                    return false;
                }
            }
        } catch (const cv::Exception& error) {
            setError("读取相机曝光配置失败：" + std::string(error.what()));
            return false;
        }

        // 按 ISP 的真实传感器帧率校验，不能使用 VPSS 缩放后的输出帧率。
        if (exposure_time_us > 0) {
            ISP_PUB_ATTR_S pub_attr{};
            const CVI_S32 rc = CVI_ISP_GetPubAttr(0, &pub_attr);
            if (rc != CVI_SUCCESS) {
                setError("CVI_ISP_GetPubAttr 失败：" + std::to_string(rc));
                return false;
            }
            if (!std::isfinite(pub_attr.f32FrameRate) || pub_attr.f32FrameRate <= 0.0f) {
                setError("ISP 传感器帧率无效，无法校验曝光时间");
                return false;
            }
            const double frame_time_us = 1000000.0 / pub_attr.f32FrameRate;
            if (exposure_time_us > frame_time_us) {
                setError("exposure_time_us 超过传感器单帧时间上限（微秒）：" +
                         std::to_string(frame_time_us));
                return false;
            }
        }

        // 先读取原属性，只修改曝光模式、时间和手动增益，不覆盖其它 ISP 调校。
        ISP_EXPOSURE_ATTR_S attr{};
        CVI_S32 rc = CVI_ISP_GetExposureAttr(0, &attr);
        if (rc != CVI_SUCCESS) {
            setError("CVI_ISP_GetExposureAttr 失败：" + std::to_string(rc));
            return false;
        }
        attr.bByPass = CVI_FALSE;
        attr.enOpType = exposure_time_us == 0 ? OP_TYPE_AUTO : OP_TYPE_MANUAL;
        if (exposure_time_us > 0) {
            attr.stManual.enExpTimeOpType = OP_TYPE_MANUAL;
            attr.stManual.enGainType = AE_TYPE_GAIN;
            attr.stManual.enAGainOpType = OP_TYPE_MANUAL;
            attr.stManual.enDGainOpType = OP_TYPE_MANUAL;
            attr.stManual.enISPDGainOpType = OP_TYPE_MANUAL;
            attr.stManual.u32ExpTime = static_cast<CVI_U32>(exposure_time_us);
            // SDK 中 1024 表示 1 倍增益，防止自动增益抵消缩短曝光的效果。
            attr.stManual.u32AGain = 1024;
            attr.stManual.u32DGain = 1024;
            attr.stManual.u32ISPDGain = 1024;
        }
        rc = CVI_ISP_SetExposureAttr(0, &attr);
        if (rc != CVI_SUCCESS) {
            setError("CVI_ISP_SetExposureAttr 失败：" + std::to_string(rc));
            return false;
        }
        if (exposure_time_us == 0) {
            std::fprintf(stderr, "[相机] 已设置自动曝光\n");
        } else {
            std::fprintf(stderr, "[相机] 已请求手动曝光：%d 微秒，各级增益为 1 倍\n",
                         exposure_time_us);
        }
        last_error.clear();
        return true;
    }

    bool readGray(cv::Mat& out_gray, int timeout_ms) {
        return readLicheeRV(out_gray, timeout_ms, false);
    }

    bool readBgr(cv::Mat& out_bgr, int timeout_ms) {
        return readLicheeRV(out_bgr, timeout_ms, true);
    }

    void shutdown() {
        shutdownLicheeRV();
    }

    bool isOpened() const {
        return opened;
    }

    bool usesHardwareScaledOutput() const {
        return vpss_started;
    }

    void setError(const std::string& message) { last_error = message; }

    static constexpr VPSS_GRP kVpssGroup = 0;
    static constexpr VPSS_CHN kVpssChannel = 0;

    bool openLicheeRV(int width, int height, int fps,
                      int requested_output_width, int requested_output_height) {
        if (opened || width <= 0 || height <= 0 || fps <= 0 ||
            (width & 1) || (height & 1)) {
            return false;
        }

        SAMPLE_INI_CFG_S ini = {};
        SAMPLE_VI_CONFIG_S vi = {};
        CVI_S32 rc = SAMPLE_COMM_VI_ParseIni(&ini);
        if (rc != CVI_SUCCESS) {
            setError("SAMPLE_COMM_VI_ParseIni failed: " + std::to_string(rc));
            return false;
        }

        rc = SAMPLE_COMM_VI_IniToViCfg(&ini, &vi);         // 将 SDK ini 配置转换为 VI 配置结构
        if (rc != CVI_SUCCESS) {
            setError("SAMPLE_COMM_VI_IniToViCfg failed: " + std::to_string(rc));
            return false;
        }

        vi.astViInfo[0].stChnInfo.ViChn = 0;
        vi.astViInfo[0].stChnInfo.enPixFormat = PIXEL_FORMAT_NV21;
        vi.astViInfo[0].stChnInfo.enDynamicRange = DYNAMIC_RANGE_SDR8;
        vi.astViInfo[0].stChnInfo.enVideoFormat = VIDEO_FORMAT_LINEAR;
        vi.astViInfo[0].stChnInfo.enCompressMode = COMPRESS_MODE_NONE;       // 使用非压缩帧

        PIC_SIZE_E pic_size;
        SIZE_S sensor_size;
        rc = SAMPLE_COMM_VI_GetSizeBySensor(ini.enSnsType[0], &pic_size);    // 获取传感器对应的图像尺寸
        if (rc == CVI_SUCCESS) {
            rc = SAMPLE_COMM_SYS_GetPicSize(pic_size, &sensor_size);
        }
        if (rc != CVI_SUCCESS) {
            setError("failed to get GC4653 sensor size: " + std::to_string(rc));
            return false;
        }

        if (static_cast<int>(sensor_size.u32Width) != width ||
            static_cast<int>(sensor_size.u32Height) != height) {
            setError("requested resolution does not match sensor mode " +
                     std::to_string(sensor_size.u32Width) + "x" +
                     std::to_string(sensor_size.u32Height));
            return false;
        }

        vi_config = vi;
        rc = SAMPLE_PLAT_SYS_INIT(sensor_size);                               // 初始化系统级资源
        if (rc != CVI_SUCCESS) {
            setError("SAMPLE_PLAT_SYS_INIT failed: " + std::to_string(rc));
            return false;
        }
        sys_started = true;

        rc = SAMPLE_PLAT_VI_INIT(&vi);                                        // 初始化 VI 图像流水线
        if (rc != CVI_SUCCESS) {
            // SAMPLE_PLAT_VI_INIT 失败时已自行释放 VI/ISP/SYS 资源。
            sys_started = false;
            shutdown();
            setError("SAMPLE_PLAT_VI_INIT failed: " + std::to_string(rc));
            return false;
        }

        vi_config = vi;
        vi_started = true;
        width_ = width;
        height_ = height;

        // 使用 Nano 的 VPSS 硬件生成配置的识别输入图像。
        // 这样可避免在高频处理路径中对全分辨率图像进行 CPU 缩放。
        if (!startVpss(width, height, fps, requested_output_width, requested_output_height)) {
            // 如果当前 SDK/板端固件不支持所选 VPSS 通道，
            // 仍保留相机可用，并由调用方通过 usesHardwareScaledOutput() 判断，
            // 随后回退到原有的 CPU 缩放路径。
            std::fprintf(stderr, "[camera] VPSS output %dx%d unavailable; using VI frame\n",
                         requested_output_width, requested_output_height);
            output_width_ = width;
            output_height_ = height;
        }

        usleep(2000000);
        last_error.clear();
        opened = true;
        return true;
    }

    bool readLicheeRV(cv::Mat& output, int timeout_ms, bool want_bgr) {
        if (!opened) {
            return false;
        }
        if (vpss_started) {
            return readVpssFrame(output, timeout_ms, want_bgr);
        }
        return readViFrame(output, timeout_ms, want_bgr);
    }

    bool readViFrame(cv::Mat& output, int timeout_ms, bool want_bgr) {
        if (frame_acquired) {
            CVI_VI_ReleaseChnFrame(0, 0, &frame);                  // 先释放上一帧
            frame_acquired = false;
        }

        frame = {};
        CVI_S32 rc = CVI_VI_GetChnFrame(0, 0, &frame, timeout_ms); // 拉取新帧
        if (rc != CVI_SUCCESS) {
            setError("CVI_VI_GetChnFrame failed: " + std::to_string(rc));
            return false;
        }
        frame_acquired = true;                                     // 等到下次读取
        return copyFrame(frame, output, width_, height_, want_bgr, "VI");
    }

    bool readVpssFrame(cv::Mat& output, int timeout_ms, bool want_bgr) {
        VIDEO_FRAME_INFO_S vpss_frame{};
        const CVI_S32 rc = CVI_VPSS_GetChnFrame(
            kVpssGroup, kVpssChannel, &vpss_frame, timeout_ms);
        if (rc != CVI_SUCCESS) {
            setError("CVI_VPSS_GetChnFrame failed: " + std::to_string(rc));
            return false;
        }

        const bool ok = copyFrame(
            vpss_frame, output, output_width_, output_height_, want_bgr, "VPSS");
        const CVI_S32 release_rc = CVI_VPSS_ReleaseChnFrame(
            kVpssGroup, kVpssChannel, &vpss_frame);
        if (release_rc != CVI_SUCCESS && ok) {
            setError("CVI_VPSS_ReleaseChnFrame failed: " + std::to_string(release_rc));
            return false;
        }
        return ok;
    }

    bool copyFrame(const VIDEO_FRAME_INFO_S& frame_info,
                   cv::Mat& output,
                   int expected_width,
                   int expected_height,
                   bool want_bgr,
                   const char* source_name) {
        const VIDEO_FRAME_S& video_frame = frame_info.stVFrame;
        const bool is_nv21 = video_frame.enPixelFormat == PIXEL_FORMAT_NV21;
        const bool is_i420 = video_frame.enPixelFormat == PIXEL_FORMAT_YUV_PLANAR_420;
        if ((!is_nv21 && !is_i420) ||
            video_frame.u32Width != static_cast<CVI_U32>(expected_width) ||
            video_frame.u32Height != static_cast<CVI_U32>(expected_height) ||
            video_frame.u32Stride[0] < video_frame.u32Width ||
            video_frame.u32Length[0] == 0 || video_frame.u64PhyAddr[0] == 0 ||
            (want_bgr && (video_frame.u32Stride[1] < (is_i420 ? video_frame.u32Width / 2 : video_frame.u32Width) ||
                          video_frame.u32Length[1] == 0 || video_frame.u64PhyAddr[1] == 0)) ||
            (want_bgr && is_i420 &&
             (video_frame.u32Stride[2] < video_frame.u32Width / 2 ||
              video_frame.u32Length[2] == 0 || video_frame.u64PhyAddr[2] == 0))) {
            setError(std::string("invalid ") + source_name +
                     " frame format, dimensions, stride, or YUV420 planes");
            return false;
        }

        const size_t y_size = static_cast<size_t>(video_frame.u32Stride[0]) * expected_height;
        const size_t y_map_size = std::max<size_t>(video_frame.u32Length[0], y_size);
        const int chroma_height = (expected_height + 1) / 2;
        const int chroma_width = (expected_width + 1) / 2;
        const size_t chroma_size = static_cast<size_t>(video_frame.u32Stride[1]) * chroma_height;
        const size_t chroma_map_size = std::max<size_t>(video_frame.u32Length[1], chroma_size);
        void* y_plane = CVI_SYS_Mmap(video_frame.u64PhyAddr[0], y_map_size);
        void* chroma_plane = want_bgr ? CVI_SYS_Mmap(video_frame.u64PhyAddr[1], chroma_map_size) : nullptr;
        void* v_plane = nullptr;
        size_t v_map_size = 0;
        if (want_bgr && is_i420) {
            const size_t v_size = static_cast<size_t>(video_frame.u32Stride[2]) * chroma_height;
            v_map_size = std::max<size_t>(video_frame.u32Length[2], v_size);
            v_plane = CVI_SYS_Mmap(video_frame.u64PhyAddr[2], v_map_size);
        }
        if (!y_plane || (want_bgr && (!chroma_plane || (is_i420 && !v_plane)))) {
            if (y_plane) CVI_SYS_Munmap(y_plane, y_map_size);
            if (chroma_plane) CVI_SYS_Munmap(chroma_plane, chroma_map_size);
            if (v_plane) CVI_SYS_Munmap(v_plane, v_map_size);
            setError(std::string("CVI_SYS_Mmap failed for ") + source_name + " YUV420 planes");
            return false;
        }

        CVI_S32 rc = CVI_SYS_IonInvalidateCache(video_frame.u64PhyAddr[0], y_plane, y_map_size);
        if (rc == CVI_SUCCESS && want_bgr) {
            rc = CVI_SYS_IonInvalidateCache(video_frame.u64PhyAddr[1], chroma_plane, chroma_map_size);
        }
        if (rc == CVI_SUCCESS && want_bgr && is_i420) {
            rc = CVI_SYS_IonInvalidateCache(video_frame.u64PhyAddr[2], v_plane, v_map_size);
        }
        if (rc != CVI_SUCCESS) {
            CVI_SYS_Munmap(y_plane, y_map_size);
            if (chroma_plane) CVI_SYS_Munmap(chroma_plane, chroma_map_size);
            if (v_plane) CVI_SYS_Munmap(v_plane, v_map_size);
            setError(std::string("CVI_SYS_IonInvalidateCache failed for ") + source_name +
                     ": " + std::to_string(rc));
            return false;
        }

        if (!want_bgr) {
            if (output.rows != expected_height || output.cols != expected_width || output.type() != CV_8UC1) {
                output.create(expected_height, expected_width, CV_8UC1);
            }
            const auto* source = static_cast<const std::uint8_t*>(y_plane);
            for (int row = 0; row < expected_height; ++row) {
                std::memcpy(output.ptr(row), source + static_cast<size_t>(row) * video_frame.u32Stride[0], expected_width);
            }
        } else {
            cv::Mat yuv(expected_height + chroma_height, expected_width, CV_8UC1);
            const auto* y_source = static_cast<const std::uint8_t*>(y_plane);
            for (int row = 0; row < expected_height; ++row) {
                std::memcpy(yuv.ptr<std::uint8_t>(row),
                            y_source + static_cast<size_t>(row) * video_frame.u32Stride[0], expected_width);
            }
            const auto* c_source = static_cast<const std::uint8_t*>(chroma_plane);
            if (is_nv21) {
                for (int row = 0; row < chroma_height; ++row) {
                    std::memcpy(yuv.ptr<std::uint8_t>(expected_height + row),
                                c_source + static_cast<size_t>(row) * video_frame.u32Stride[1], expected_width);
                }
                cv::cvtColor(yuv, output, cv::COLOR_YUV2BGR_NV21);
            } else {
                for (int row = 0; row < chroma_height; ++row) {
                    auto* chroma_row = yuv.ptr<std::uint8_t>(expected_height + row);
                    std::memcpy(chroma_row,
                                c_source + static_cast<size_t>(row) * video_frame.u32Stride[1], chroma_width);
                    std::memcpy(chroma_row + chroma_width,
                                static_cast<const std::uint8_t*>(v_plane) +
                                    static_cast<size_t>(row) * video_frame.u32Stride[2], chroma_width);
                }
                cv::cvtColor(yuv, output, cv::COLOR_YUV2BGR_I420);
            }
        }
        CVI_SYS_Munmap(y_plane, y_map_size);
        if (chroma_plane) CVI_SYS_Munmap(chroma_plane, chroma_map_size);
        if (v_plane) CVI_SYS_Munmap(v_plane, v_map_size);
        return true;
    }

    bool startVpss(int input_width,
                  int input_height,
                  int fps,
                  int requested_output_width,
                  int requested_output_height) {
        if (requested_output_width == 0 && requested_output_height == 0) {
            requested_output_width = input_width / 2;
            requested_output_height = input_height / 2;
        }
        if (requested_output_width <= 0 || requested_output_height <= 0 ||
            (requested_output_width & 1) || (requested_output_height & 1) ||
            requested_output_width > input_width || requested_output_height > input_height) {
            setError("invalid VPSS output resolution " +
                     std::to_string(requested_output_width) + "x" +
                     std::to_string(requested_output_height));
            return false;
        }
        output_width_ = requested_output_width;
        output_height_ = requested_output_height;

        VPSS_GRP_ATTR_S group_attr{};
        group_attr.stFrameRate.s32SrcFrameRate = -1;
        group_attr.stFrameRate.s32DstFrameRate = -1;
        group_attr.enPixelFormat = PIXEL_FORMAT_NV21;
        group_attr.u32MaxW = static_cast<CVI_U32>(input_width);
        group_attr.u32MaxH = static_cast<CVI_U32>(input_height);
        group_attr.u8VpssDev = 0;

        VPSS_CHN_ATTR_S channel_attr[VPSS_MAX_PHY_CHN_NUM]{};
        channel_attr[kVpssChannel].u32Width = static_cast<CVI_U32>(output_width_);
        channel_attr[kVpssChannel].u32Height = static_cast<CVI_U32>(output_height_);
        channel_attr[kVpssChannel].enVideoFormat = VIDEO_FORMAT_LINEAR;
        channel_attr[kVpssChannel].enPixelFormat =
            output_format_ == Camera::OutputFormat::Yuv420Planar
                ? PIXEL_FORMAT_YUV_PLANAR_420
                : PIXEL_FORMAT_NV21;
        channel_attr[kVpssChannel].stFrameRate.s32SrcFrameRate = fps;
        channel_attr[kVpssChannel].stFrameRate.s32DstFrameRate = fps;
        channel_attr[kVpssChannel].u32Depth = 0;
        channel_attr[kVpssChannel].bMirror = CVI_FALSE;
        channel_attr[kVpssChannel].bFlip = CVI_FALSE;
        channel_attr[kVpssChannel].stAspectRatio.enMode = ASPECT_RATIO_NONE;
        channel_attr[kVpssChannel].stNormalize.bEnable = CVI_FALSE;

        const CVI_S32 mode_rc = CVI_SYS_SetVPSSMode(VPSS_MODE_SINGLE);
        if (mode_rc != CVI_SUCCESS) {
            setError("CVI_SYS_SetVPSSMode(SINGLE) failed: " + std::to_string(mode_rc));
            return false;
        }

        vpss_chn_enable_[kVpssChannel] = CVI_TRUE;
        std::fprintf(stderr,
                     "[camera] VPSS init request: input=%dx%d output=%dx%d fps=%d format=%d\n",
                     input_width, input_height, output_width_, output_height_, fps,
                     static_cast<int>(channel_attr[kVpssChannel].enPixelFormat));
        CVI_S32 rc = SAMPLE_COMM_VPSS_Init(
            kVpssGroup, vpss_chn_enable_, &group_attr, channel_attr);
        if (rc != CVI_SUCCESS) {
            std::fprintf(stderr, "[camera] SAMPLE_COMM_VPSS_Init failed: %d\n", rc);
            cleanupFailedVpssInit();
            setError("SAMPLE_COMM_VPSS_Init failed: " + std::to_string(rc));
            return false;
        }
        vpss_initialized = true;

        // 缩放输出不能依赖只按传感器尺寸创建的公共池。官方 VENC
        // sample 会为缩放通道建立独立 VB pool，否则 VPSS 可能启动
        // 成功但没有可供绑定模块消费的输出帧。
        if (output_width_ != input_width || output_height_ != input_height) {
            VB_POOL_CONFIG_S pool_config{};
            pool_config.u32BlkSize = COMMON_GetPicBufferSize(
                static_cast<CVI_U32>(output_width_),
                static_cast<CVI_U32>(output_height_),
                channel_attr[kVpssChannel].enPixelFormat,
                DATA_BITWIDTH_8,
                COMPRESS_MODE_NONE,
                DEFAULT_ALIGN);
            if (pool_config.u32BlkSize == 0) {
                std::fprintf(stderr, "[camera] VPSS output VB size is zero\n");
                setError("COMMON_GetPicBufferSize returned zero for VPSS output");
                stopVpss();
                releaseVpssPool();
                return false;
            }
            pool_config.u32BlkCnt = 4;
            pool_config.enRemapMode = VB_REMAP_MODE_CACHED;
            std::fprintf(stderr, "[camera] VPSS output pool: block_size=%u block_count=%u\n",
                         pool_config.u32BlkSize, pool_config.u32BlkCnt);
            vpss_pool_ = CVI_VB_CreatePool(&pool_config);
            if (vpss_pool_ == VB_INVALID_POOLID) {
                std::fprintf(stderr, "[camera] CVI_VB_CreatePool failed\n");
                setError("CVI_VB_CreatePool for VPSS output failed");
                stopVpss();
                releaseVpssPool();
                return false;
            }
            rc = CVI_VPSS_AttachVbPool(kVpssGroup, kVpssChannel, vpss_pool_);
            if (rc != CVI_SUCCESS) {
                std::fprintf(stderr, "[camera] CVI_VPSS_AttachVbPool failed: %d\n", rc);
                setError("CVI_VPSS_AttachVbPool failed: " + std::to_string(rc));
                stopVpss();
                releaseVpssPool();
                return false;
            }
            std::fprintf(stderr, "[camera] CVI_VPSS_AttachVbPool succeeded\n");
            vpss_pool_attached_ = true;
        }

        rc = SAMPLE_COMM_VPSS_Start(
            kVpssGroup, vpss_chn_enable_, &group_attr, channel_attr);
        if (rc != CVI_SUCCESS) {
            std::fprintf(stderr, "[camera] SAMPLE_COMM_VPSS_Start failed: %d\n", rc);
            setError("SAMPLE_COMM_VPSS_Start failed: " + std::to_string(rc));
            stopVpss();
            releaseVpssPool();
            return false;
        }
        vpss_started = true;
        std::fprintf(stderr, "[camera] SAMPLE_COMM_VPSS_Start succeeded\n");

        rc = SAMPLE_COMM_VI_Bind_VPSS(0, 0, kVpssGroup);
        if (rc != CVI_SUCCESS) {
            std::fprintf(stderr, "[camera] SAMPLE_COMM_VI_Bind_VPSS failed: %d\n", rc);
            setError("VI-to-VPSS bind failed: " + std::to_string(rc));
            stopVpss();
            releaseVpssPool();
            return false;
        }
        vpss_bound = true;
        std::fprintf(stderr, "[camera] SAMPLE_COMM_VI_Bind_VPSS succeeded\n");
        return true;
    }

    void shutdownLicheeRV() {
        releaseFrame();
        if (vpss_bound) {
            const CVI_S32 rc = SAMPLE_COMM_VI_UnBind_VPSS(0, 0, kVpssGroup);
            if (rc != CVI_SUCCESS) {
                std::fprintf(stderr, "[camera] SAMPLE_COMM_VI_UnBind_VPSS failed: %d\n", rc);
            }
            vpss_bound = false;
        }
        stopVpss();
        releaseVpssPool();
        if (vi_started) {
            SAMPLE_COMM_VI_DestroyIsp(&vi_config);
            SAMPLE_COMM_VI_DestroyVi(&vi_config);
            SAMPLE_COMM_VI_CLOSE();
            vi_started = false;
        }
        if (sys_started) {
            SAMPLE_COMM_SYS_Exit();
            sys_started = false;
        }
        opened = false;
        width_ = 0;
        height_ = 0;
        output_width_ = 0;
        output_height_ = 0;
        last_error.clear();
    }

    void releaseFrame() {
        if (frame_acquired) {
            CVI_VI_ReleaseChnFrame(0, 0, &frame);
            frame_acquired = false;
        }
    }

    void cleanupFailedVpssInit() {
        // SAMPLE_COMM_VPSS_Init is not transactional: CreateGrp may have
        // succeeded before a channel attribute/enable operation fails.
        // Destroy the partially-created group so the next camera open can retry.
        const CVI_S32 disable_rc = CVI_VPSS_DisableChn(kVpssGroup, kVpssChannel);
        if (disable_rc != CVI_SUCCESS) {
            std::fprintf(stderr, "[camera] cleanup CVI_VPSS_DisableChn failed: %d\n", disable_rc);
        }
        const CVI_S32 destroy_rc = CVI_VPSS_DestroyGrp(kVpssGroup);
        if (destroy_rc != CVI_SUCCESS) {
            std::fprintf(stderr, "[camera] cleanup CVI_VPSS_DestroyGrp failed: %d\n", destroy_rc);
        }
        vpss_chn_enable_[kVpssChannel] = CVI_FALSE;
        vpss_initialized = false;
        vpss_started = false;
    }

    void stopVpss() {
        if (!vpss_initialized && !vpss_started) return;

        // 分步清理，避免 SDK sample 的 Stop 在 DisableChn 失败后提前返回，
        // 留下仍占用的 VPSS group，导致程序第二次启动时无法创建 group。
        for (int channel = 0; channel < VPSS_MAX_PHY_CHN_NUM; ++channel) {
            if (!vpss_chn_enable_[channel]) continue;
            const CVI_S32 rc = CVI_VPSS_DisableChn(kVpssGroup, channel);
            if (rc != CVI_SUCCESS) {
                std::fprintf(stderr, "[camera] CVI_VPSS_DisableChn(%d) cleanup failed: %d\n",
                             channel, rc);
            }
        }

        if (vpss_started) {
            const CVI_S32 rc = CVI_VPSS_StopGrp(kVpssGroup);
            if (rc != CVI_SUCCESS) {
                std::fprintf(stderr, "[camera] CVI_VPSS_StopGrp cleanup failed: %d\n", rc);
            }
        }

        if (vpss_initialized) {
            const CVI_S32 rc = CVI_VPSS_DestroyGrp(kVpssGroup);
            if (rc != CVI_SUCCESS) {
                std::fprintf(stderr, "[camera] CVI_VPSS_DestroyGrp cleanup failed: %d\n", rc);
            }
        }
        vpss_initialized = false;
        vpss_started = false;
        vpss_chn_enable_[kVpssChannel] = CVI_FALSE;
    }

    void releaseVpssPool() {
        if (vpss_pool_attached_) {
            const CVI_S32 rc = CVI_VPSS_DetachVbPool(kVpssGroup, kVpssChannel);
            if (rc != CVI_SUCCESS) {
                std::fprintf(stderr, "[camera] CVI_VPSS_DetachVbPool failed: %d\n", rc);
            }
            vpss_pool_attached_ = false;
        }
        if (vpss_pool_ != VB_INVALID_POOLID) {
            const CVI_S32 rc = CVI_VB_DestroyPool(vpss_pool_);
            if (rc != CVI_SUCCESS) {
                std::fprintf(stderr, "[camera] CVI_VB_DestroyPool failed: %d\n", rc);
            }
            vpss_pool_ = VB_INVALID_POOLID;
        }
    }

    bool sys_started = false;
    bool vi_started = false;
    bool frame_acquired = false;
    int width_ = 0;
    int height_ = 0;
    int output_width_ = 0;
    int output_height_ = 0;
    bool vpss_initialized = false;
    bool vpss_started = false;
    bool vpss_bound = false;
    VB_POOL vpss_pool_ = VB_INVALID_POOLID;
    bool vpss_pool_attached_ = false;
    Camera::OutputFormat output_format_ = Camera::OutputFormat::Nv21;
    CVI_BOOL vpss_chn_enable_[VPSS_MAX_PHY_CHN_NUM]{};
    SAMPLE_VI_CONFIG_S vi_config{};
    VIDEO_FRAME_INFO_S frame{};
    bool opened = false;
    std::string last_error;
};

Camera::Camera() : impl_(std::make_unique<Impl>()) {}

Camera::~Camera() {
    shutdown();
}

bool Camera::open(int device_id,
                   int width,
                   int height,
                   int fps,
                   int output_width,
                   int output_height,
                   OutputFormat output_format) {
    return impl_->open(device_id, width, height, fps, output_width, output_height, output_format);
}


bool Camera::configureExposureFromFile(const std::string& config_path) {
    return impl_->configureExposureFromFile(config_path);
}

bool Camera::readGray(cv::Mat& out_gray, int timeout_ms) {
    return impl_->readGray(out_gray, timeout_ms);
}

bool Camera::readBgr(cv::Mat& out_bgr, int timeout_ms) {
    return impl_->readBgr(out_bgr, timeout_ms);
}

void Camera::close() {
    shutdown();
}

void Camera::shutdown() {
    impl_->shutdown();
}

bool Camera::isOpened() const {
    return impl_->isOpened();
}

bool Camera::usesHardwareScaledOutput() const {
    return impl_->usesHardwareScaledOutput();
}

const std::string& Camera::lastError() const {
    return impl_->last_error;
}
