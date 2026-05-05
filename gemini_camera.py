from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


class GeminiCameraError(RuntimeError):
    pass


class Gemini335Camera:
    def __init__(
        self,
        sdk_root: Path,
        color_width: int = 1280,
        color_height: int = 720,
        fps: int = 30,
        calib_dir: Path | None = None,
    ) -> None:
        self.sdk_root = sdk_root
        self.color_width = color_width
        self.color_height = color_height
        self.fps = fps
        self._pipeline = None
        self._device = None
        self._ob = None
        self._intrinsic = None
        self._distortion = None
        self._remap_x = None
        self._remap_y = None
        self._load_sdk()
        self._connect()
        self._start_pipeline()
        self._intrinsic = self._read_intrinsic()
        if calib_dir is not None:
            self._load_undistort_maps(calib_dir)

    def _iter_candidate_profiles(self, color_profiles):
        assert self._ob is not None
        tried = set()

        def add(width: int, height: int, fmt, fps: int):
            key = (int(width), int(height), int(fmt), int(fps))
            if key in tried:
                return
            tried.add(key)
            yield width, height, fmt, fps

        yield from add(self.color_width, self.color_height, self._ob.OBFormat.MJPG, self.fps)
        yield from add(self.color_width, self.color_height, self._ob.OBFormat.RGB, self.fps)
        yield from add(640, 480, self._ob.OBFormat.MJPG, 30)
        yield from add(640, 480, self._ob.OBFormat.RGB, 30)
        yield from add(1280, 720, self._ob.OBFormat.MJPG, 30)
        yield from add(1280, 720, self._ob.OBFormat.RGB, 30)

        count = 0
        try:
            count = int(color_profiles.get_count())
        except Exception:
            count = 0

        for index in range(count):
            try:
                profile = color_profiles.get_stream_profile_by_index(index)
            except Exception:
                try:
                    profile = color_profiles.get_profile(index)
                except Exception:
                    continue
            try:
                width = int(profile.get_width())
                height = int(profile.get_height())
                fmt = profile.get_format()
                fps = int(profile.get_fps())
            except Exception:
                continue
            yield from add(width, height, fmt, fps)

    def _select_color_profile(self, color_profiles):
        errors = []
        for width, height, fmt, fps in self._iter_candidate_profiles(color_profiles):
            try:
                profile = color_profiles.get_video_stream_profile(width, height, fmt, fps)
                fmt_name = str(fmt)
                print(f"[INFO] 使用 Gemini335 彩色流: {width}x{height} fmt={fmt_name} fps={fps}")
                self.color_width = width
                self.color_height = height
                self.fps = fps
                return profile
            except Exception as exc:
                errors.append(f"{width}x{height} fmt={fmt} fps={fps}: {exc}")

        try:
            profile = color_profiles.get_default_video_stream_profile()
            width = int(profile.get_width())
            height = int(profile.get_height())
            fps = int(profile.get_fps())
            fmt = profile.get_format()
            print(f"[INFO] 回退到 Gemini335 默认彩色流: {width}x{height} fmt={fmt} fps={fps}")
            self.color_width = width
            self.color_height = height
            self.fps = fps
            return profile
        except Exception as exc:
            errors.append(f"default profile: {exc}")

        raise GeminiCameraError(
            "没有找到可用的 Gemini335 彩色流配置。尝试过:\n" + "\n".join(errors[:12])
        )

    def _load_sdk(self) -> None:
        lib_dir = self.sdk_root / "lib" / "pyorbbecsdk" / ("linux" if sys.platform != "win32" else "windows")
        if not lib_dir.exists():
            raise GeminiCameraError(f"找不到 pyorbbecsdk 动态库目录: {lib_dir}")
        if str(lib_dir) not in sys.path:
            sys.path.insert(0, str(lib_dir))
        try:
            import pyorbbecsdk as ob  # type: ignore
        except Exception as exc:
            raise GeminiCameraError(f"导入 pyorbbecsdk 失败: {exc}") from exc
        self._ob = ob

    def _connect(self) -> None:
        assert self._ob is not None
        context = self._ob.Context()
        context.set_logger_level(self._ob.OBLogLevel.ERROR)
        device_list = context.query_devices()
        if device_list.get_count() == 0:
            raise GeminiCameraError("没有检测到 Gemini335 设备。")
        serial = device_list.get_device_serial_number_by_index(0)
        self._device = device_list.get_device_by_serial_number(serial)

    def _start_pipeline(self) -> None:
        assert self._ob is not None
        if self._device is None:
            raise GeminiCameraError("相机设备未连接。")
        pipeline = self._ob.Pipeline(self._device)
        config = self._ob.Config()
        color_profiles = pipeline.get_stream_profile_list(self._ob.OBSensorType.COLOR_SENSOR)
        color_profile = self._select_color_profile(color_profiles)
        config.enable_stream(color_profile)
        config.set_align_mode(self._ob.OBAlignMode.SW_MODE)
        self._device.set_bool_property(self._ob.OBPropertyID.OB_PROP_LDP_BOOL, False)
        pipeline.start(config)
        self._pipeline = pipeline

    def _read_intrinsic(self) -> np.ndarray:
        if self._pipeline is None:
            raise GeminiCameraError("相机管道未启动。")
        camera_param = self._pipeline.get_camera_param()
        fx = camera_param.rgb_intrinsic.fx
        fy = camera_param.rgb_intrinsic.fy
        cx = camera_param.rgb_intrinsic.cx
        cy = camera_param.rgb_intrinsic.cy
        return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    def _load_undistort_maps(self, calib_dir: Path) -> None:
        remap_x = calib_dir / "remap_x.npy"
        remap_y = calib_dir / "remap_y.npy"
        intrinsic = calib_dir / "M_intrisic.txt"
        distortion = calib_dir / "distor_coeff.txt"
        if not remap_x.exists() or not remap_y.exists():
            raise GeminiCameraError(f"未找到去畸变映射文件: {calib_dir}")
        self._remap_x = np.load(remap_x)
        self._remap_y = np.load(remap_y)
        if intrinsic.exists():
            self._intrinsic = np.loadtxt(intrinsic, delimiter=",")
        if distortion.exists():
            self._distortion = np.asarray(np.loadtxt(distortion, delimiter=","), dtype=np.float64).reshape(-1)

    def read_color_image(self, timeout_ms: int = 3500) -> np.ndarray:
        assert self._ob is not None
        if self._pipeline is None:
            raise GeminiCameraError("相机管道未启动。")
        frames = self._pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            raise GeminiCameraError("读取 Gemini335 图像帧超时。")
        color_frame = frames.get_color_frame()
        if color_frame is None:
            raise GeminiCameraError("未获取到 Gemini335 彩色帧。")
        width = color_frame.get_width()
        height = color_frame.get_height()
        color_format = color_frame.get_format()
        data = np.asanyarray(color_frame.get_data())
        import cv2  # type: ignore

        if color_format == self._ob.OBFormat.RGB:
            image = np.resize(data, (height, width, 3))
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if color_format == self._ob.OBFormat.MJPG:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                raise GeminiCameraError("MJPG 彩色帧解码失败。")
            return image
        raise GeminiCameraError(f"不支持的彩色帧格式: {color_format}")

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        if self._remap_x is None or self._remap_y is None:
            return image
        import cv2  # type: ignore

        return cv2.remap(image, self._remap_x, self._remap_y, cv2.INTER_LINEAR)

    def undistort_pixel(self, pixel_uv: tuple[float, float]) -> tuple[float, float]:
        if self._distortion is None or self._intrinsic is None:
            return pixel_uv
        import cv2  # type: ignore

        pts = np.asarray([[[pixel_uv[0], pixel_uv[1]]]], dtype=np.float64)
        undistorted = cv2.undistortPoints(pts, self._intrinsic, self._distortion, P=self._intrinsic)
        point = undistorted.reshape(-1, 2)[0]
        return float(point[0]), float(point[1])

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
