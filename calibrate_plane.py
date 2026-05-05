from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from gemini_camera import Gemini335Camera, GeminiCameraError
from robot_api import RobotApiError, RobotClient


ROBOT_IP = "192.168.58.2"
TOOL_ID = 1
USER_ID = 0
GRID_SPACING_MM = 0.0
SAFE_Z_OFFSET_MM = 20.0
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
FPS = 30
DISPLAY_SCALE = 1.8
INIT_RX_DEG = 180.0
INIT_RY_DEG = 0.0
INIT_RZ_DEG = 180.0
SESSION_JSON = "planar_session.json"
RESULT_JSON = "planar_calibration_result.json"
ROOT_DIR = Path(__file__).resolve().parents[1]
GEMINI_SDK_ROOT = (
    ROOT_DIR
    / "Gemini335-软件资料(Windows版)-阿凯爱玩机器人-V20240722"
    / "deepsense-gemini335-master"
    / "02.奥比中光-pyobbecsdk示例代码(Gemini335)"
)
CAMERA_CALIB_DIR = None
WINDOW_NAME = "FR3 Gemini335 Planar Calibration"


class PlanarCalibrationError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FR3 + Gemini335 平面标定采样工具")
    parser.add_argument("--robot-ip", default=ROBOT_IP)
    parser.add_argument("--tool-id", type=int, default=TOOL_ID)
    parser.add_argument("--user-id", type=int, default=USER_ID)
    parser.add_argument("--grid-spacing-mm", type=float, default=GRID_SPACING_MM)
    parser.add_argument("--camera-calib-dir", type=Path, default=CAMERA_CALIB_DIR)
    parser.add_argument("--gemini-sdk-root", type=Path, default=GEMINI_SDK_ROOT)
    parser.add_argument("--width", type=int, default=IMAGE_WIDTH)
    parser.add_argument("--height", type=int, default=IMAGE_HEIGHT)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--display-scale", type=float, default=DISPLAY_SCALE)
    parser.add_argument("--session-json", type=Path, default=Path(__file__).with_name(SESSION_JSON))
    parser.add_argument("--result-json", type=Path, default=Path(__file__).with_name(RESULT_JSON))
    return parser.parse_args()


def default_session_data(robot_ip: str, grid_spacing_mm: float, camera_calibration_reference: str | None) -> dict:
    return {
        "robot_ip": robot_ip,
        "image_width": 0,
        "image_height": 0,
        "grid_spacing_mm": grid_spacing_mm,
        "camera_calibration_reference": camera_calibration_reference,
        "samples": [],
    }


def load_session_json(path: Path, robot_ip: str, grid_spacing_mm: float, camera_calibration_reference: str | None) -> dict:
    if not path.exists():
        return default_session_data(robot_ip, grid_spacing_mm, camera_calibration_reference)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return default_session_data(robot_ip, grid_spacing_mm, camera_calibration_reference)
    payload.setdefault("robot_ip", robot_ip)
    payload["grid_spacing_mm"] = grid_spacing_mm
    payload["camera_calibration_reference"] = camera_calibration_reference
    payload.setdefault("image_width", 0)
    payload.setdefault("image_height", 0)
    payload.setdefault("samples", [])
    return payload


def save_session_json(path: Path, session_data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)


def load_result_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else None


def save_result_json(path: Path, result_data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)


def _as_points(points: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(list(points), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise PlanarCalibrationError("Expected Nx2 point array.")
    return arr


def apply_homography(matrix: np.ndarray, points: Iterable[Iterable[float]]) -> np.ndarray:
    pts = _as_points(points)
    homo = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
    mapped = (matrix @ homo.T).T
    scale = mapped[:, 2:3]
    if np.any(np.isclose(scale, 0.0)):
        raise PlanarCalibrationError("Homography produced invalid homogeneous coordinates.")
    return mapped[:, :2] / scale


def compute_error_stats(predicted_xy: np.ndarray, target_xy: np.ndarray) -> dict:
    errors = np.linalg.norm(predicted_xy - target_xy, axis=1)
    if errors.size == 0:
        return {"mean_mm": 0.0, "max_mm": 0.0, "p95_mm": 0.0}
    return {
        "mean_mm": float(np.mean(errors)),
        "max_mm": float(np.max(errors)),
        "p95_mm": float(np.percentile(errors, 95)),
    }


def _fit_homography_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    rows = []
    for (u, v), (x, y) in zip(src, dst, strict=True):
        rows.append([-u, -v, -1.0, 0.0, 0.0, 0.0, u * x, v * x, x])
        rows.append([0.0, 0.0, 0.0, -u, -v, -1.0, u * y, v * y, y])
    a = np.asarray(rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(a)
    h = vt[-1].reshape(3, 3)
    if abs(h[2, 2]) < 1e-12:
        raise PlanarCalibrationError("Homography fit produced a degenerate matrix.")
    return h / h[2, 2]


def fit_homography(src_pixels: Iterable[Iterable[float]], dst_robot_xy: Iterable[Iterable[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    src = _as_points(src_pixels)
    dst = _as_points(dst_robot_xy)
    if len(src) != len(dst):
        raise PlanarCalibrationError("Source and destination point counts do not match.")
    if len(src) < 4:
        raise PlanarCalibrationError("At least 4 samples are required to fit a homography.")

    h = None
    inlier_mask = None
    try:
        import cv2  # type: ignore

        h_cv, mask_cv = cv2.findHomography(src, dst, method=cv2.RANSAC)
        if h_cv is not None:
            h = np.asarray(h_cv, dtype=np.float64)
            if mask_cv is not None:
                inlier_mask = np.asarray(mask_cv, dtype=bool).reshape(-1)
    except Exception:
        h = None

    if h is None:
        h = _fit_homography_dlt(src, dst)

    h = h / h[2, 2]
    h_inv = np.linalg.inv(h)
    predicted = apply_homography(h, src)
    if inlier_mask is None or inlier_mask.shape[0] != len(src):
        inlier_mask = np.ones(len(src), dtype=bool)
    stats = compute_error_stats(predicted[inlier_mask], dst[inlier_mask])
    return h, h_inv / h_inv[2, 2], inlier_mask, stats


def build_result(session_data: dict) -> dict:
    samples = session_data.get("samples", [])
    if len(samples) < 4:
        raise PlanarCalibrationError("At least 4 samples are required to compute calibration.")
    src = [(sample["pixel_u"], sample["pixel_v"]) for sample in samples]
    dst = [(sample["robot_x"], sample["robot_y"]) for sample in samples]
    h, h_inv, inlier_mask, stats = fit_homography(src, dst)
    plane_z_mm = float(np.mean([sample["tcp_pose"][2] for sample in samples]))
    reference_rpy_deg = [
        float(np.mean([sample["tcp_pose"][3] for sample in samples])),
        float(np.mean([sample["tcp_pose"][4] for sample in samples])),
        float(np.mean([sample["tcp_pose"][5] for sample in samples])),
    ]
    return {
        "homography_pixel_to_robot": h.tolist(),
        "homography_robot_to_pixel": h_inv.tolist(),
        "error_stats": stats,
        "image_width": int(session_data.get("image_width", 0)),
        "image_height": int(session_data.get("image_height", 0)),
        "robot_ip": session_data.get("robot_ip", ROBOT_IP),
        "grid_spacing_mm": float(session_data.get("grid_spacing_mm", 0.0)),
        "camera_calibration_reference": session_data.get("camera_calibration_reference"),
        "sample_count": len(samples),
        "inlier_mask": inlier_mask.astype(bool).tolist(),
        "samples": samples,
        "plane_z_mm": plane_z_mm,
        "reference_rpy_deg": reference_rpy_deg,
    }


def pixel_to_robot_xy(result_data: dict, pixel_uv: tuple[float, float]) -> tuple[float, float]:
    matrix = np.asarray(result_data["homography_pixel_to_robot"], dtype=np.float64)
    xy = apply_homography(matrix, [pixel_uv])[0]
    return float(xy[0]), float(xy[1])


def add_sample(session_data: dict, selected_pixel: tuple[float, float] | None, tcp_pose: list[float]) -> dict:
    if selected_pixel is None:
        raise PlanarCalibrationError("请先在图像中点击一个像素点。")
    sample = {
        "point_id": len(session_data["samples"]) + 1,
        "pixel_u": float(selected_pixel[0]),
        "pixel_v": float(selected_pixel[1]),
        "tcp_pose": [float(v) for v in tcp_pose[:6]],
        "robot_x": float(tcp_pose[0]),
        "robot_y": float(tcp_pose[1]),
        "timestamp": __import__("time").time(),
    }
    session_data["samples"].append(sample)
    return sample


def undo_last_sample(session_data: dict) -> dict | None:
    if not session_data.get("samples"):
        return None
    return session_data["samples"].pop()


def move_robot_to_prediction(
    robot: RobotClient,
    result_data: dict | None,
    prediction_data: dict | None,
    verify_rpy_deg: list[float] | None,
    tool_id: int,
    user_id: int,
    safe_z_offset_mm: float = SAFE_Z_OFFSET_MM,
) -> None:
    if prediction_data is None:
        raise PlanarCalibrationError("当前没有可用于验证的预测点。")
    if result_data is None:
        raise PlanarCalibrationError("请先完成标定计算。")
    if result_data.get("plane_z_mm") is None:
        raise PlanarCalibrationError("标定结果缺少平面高度。")
    if verify_rpy_deg is None:
        raise PlanarCalibrationError("验证模式姿态未锁定，请先进入验证模式。")
    x, y = prediction_data["robot_xy"]
    z = float(result_data["plane_z_mm"]) + safe_z_offset_mm
    rx, ry, rz = verify_rpy_deg
    robot.move_ptp([x, y, z, rx, ry, rz], tool_id, user_id, vel=20.0)


def align_robot_to_initial_rpy(
    robot: RobotClient,
    tool_id: int,
    user_id: int,
    rx_deg: float = INIT_RX_DEG,
    ry_deg: float = INIT_RY_DEG,
    rz_deg: float = INIT_RZ_DEG,
) -> list[float]:
    tcp_pose = robot.get_actual_tcp_pose_live()
    target_pose = [
        float(tcp_pose[0]),
        float(tcp_pose[1]),
        float(tcp_pose[2]),
        float(rx_deg),
        float(ry_deg),
        float(rz_deg),
    ]
    robot.move_ptp(target_pose, tool_id, user_id, vel=20.0)
    return target_pose


def draw_overlay(
    frame,
    session_data: dict,
    selected_pixel: tuple[float, float] | None,
    last_status: str,
    current_tcp_pose: list[float] | None,
    verify_mode: bool,
    move_enabled: bool,
    result_data: dict | None,
    prediction_data: dict | None,
    pose_age_s: float | None,
    verify_rpy_deg: list[float] | None,
):
    import cv2  # type: ignore

    canvas = frame.copy()
    overlay = canvas.copy()
    cv2.rectangle(overlay, (8, 8), (530, 220), (0, 0, 0), -1)
    cv2.rectangle(overlay, (8, 228), (530, 470), (0, 0, 0), -1)
    canvas = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)

    pose_state = "unavailable"
    if pose_age_s is not None:
        pose_state = f"live ({pose_age_s:.2f}s ago)"

    lines = [
        f"mode: {'VERIFY' if verify_mode else 'SAMPLE'}",
        f"samples: {len(session_data.get('samples', []))}",
        f"selected pixel: {selected_pixel}",
        f"move enabled: {move_enabled}",
        f"pose refresh: {pose_state}",
        f"status: {last_status}",
    ]
    if current_tcp_pose is not None:
        lines.append(
            "tcp xyzrpy: "
            f"{current_tcp_pose[0]:.2f}, {current_tcp_pose[1]:.2f}, {current_tcp_pose[2]:.2f}, "
            f"{current_tcp_pose[3]:.2f}, {current_tcp_pose[4]:.2f}, {current_tcp_pose[5]:.2f}"
        )
    if result_data is not None:
        stats = result_data["error_stats"]
        lines.append(
            f"fit error mm: mean={stats['mean_mm']:.3f} max={stats['max_mm']:.3f} p95={stats['p95_mm']:.3f}"
        )
    if prediction_data is not None:
        px, py = prediction_data["pixel_uv"]
        rx, ry = prediction_data["robot_xy"]
        lines.append(f"prediction: pixel=({px:.1f},{py:.1f}) -> robot=({rx:.2f},{ry:.2f})")
    if verify_rpy_deg is not None:
        lines.append(
            f"verify rpy lock: ({verify_rpy_deg[0]:.2f}, {verify_rpy_deg[1]:.2f}, {verify_rpy_deg[2]:.2f})"
        )
    y = 24
    for line in lines:
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 1, cv2.LINE_AA)
        y += 22

    help_lines = [
        "Mouse left: select pixel",
        "s: save sample    u: undo last sample",
        "c: compute calibration",
        "v: toggle sample/verify mode",
        "m: enable/disable robot move",
        "g: go to predicted point (safe Z)",
        "p: print TCP pose",
        "q: save and quit",
        "Workflow: sample points -> press c -> press v -> click pixel -> press m -> press g",
    ]
    y = 248
    cv2.putText(canvas, "KEYS", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    y += 26
    for line in help_lines:
        cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22

    if selected_pixel is not None:
        u, v = selected_pixel
        cv2.drawMarker(canvas, (int(round(u)), int(round(v))), (0, 255, 255), markerSize=18, thickness=2)
    return canvas


def main() -> int:
    import cv2  # type: ignore

    args = parse_args()
    session_data = load_session_json(
        args.session_json,
        args.robot_ip,
        args.grid_spacing_mm,
        str(args.camera_calib_dir) if args.camera_calib_dir else None,
    )
    result_data = load_result_json(args.result_json)

    try:
        robot = RobotClient(args.robot_ip)
    except RobotApiError as exc:
        print(f"[ERROR] 连接 FR3 失败: {exc}")
        return 1

    try:
        init_pose = align_robot_to_initial_rpy(
            robot,
            tool_id=args.tool_id,
            user_id=args.user_id,
        )
        print(
            "[INFO] 已将机械臂末端姿态调整到 "
            f"RX={init_pose[3]:.1f}, RY={init_pose[4]:.1f}, RZ={init_pose[5]:.1f}"
        )
    except RobotApiError as exc:
        print(f"[ERROR] 初始化机械臂末端姿态失败: {exc}")
        return 1

    try:
        camera = Gemini335Camera(
            sdk_root=args.gemini_sdk_root,
            color_width=args.width,
            color_height=args.height,
            fps=args.fps,
            calib_dir=args.camera_calib_dir,
        )
    except GeminiCameraError as exc:
        print(f"[ERROR] 连接 Gemini335 失败: {exc}")
        return 1

    selected_pixel = None
    prediction_data = None
    last_status = "Ready to sample"
    verify_mode = False
    move_enabled = False
    last_pose_update_time = None
    verify_rpy_deg = None

    def on_mouse(event, x, y, _flags, _userdata):
        nonlocal selected_pixel, prediction_data, last_status
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        image_x = float(x) / max(args.display_scale, 1e-6)
        image_y = float(y) / max(args.display_scale, 1e-6)
        selected_pixel = camera.undistort_pixel((image_x, image_y))
        prediction_data = None
        last_status = f"Selected pixel ({selected_pixel[0]:.1f}, {selected_pixel[1]:.1f})"
        if verify_mode and result_data is not None:
            try:
                robot_xy = pixel_to_robot_xy(result_data, selected_pixel)
                prediction_data = {"pixel_uv": selected_pixel, "robot_xy": robot_xy}
                last_status = (
                    f"Predicted robot XY: pixel=({selected_pixel[0]:.1f}, {selected_pixel[1]:.1f}) -> "
                    f"robot=({robot_xy[0]:.2f}, {robot_xy[1]:.2f})"
                )
            except PlanarCalibrationError as exc:
                last_status = str(exc)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        WINDOW_NAME,
        max(960, int(args.width * args.display_scale)),
        max(720, int(args.height * args.display_scale)),
    )
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    try:
        while True:
            current_tcp_pose = None
            try:
                current_tcp_pose = robot.get_actual_tcp_pose_live()
                last_pose_update_time = time.time()
            except RobotApiError as exc:
                last_status = f"TCP read failed: {exc}"

            try:
                frame = camera.read_color_image()
            except GeminiCameraError as exc:
                last_status = f"Camera read failed: {exc}"
                frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)

            frame = camera.undistort_image(frame)
            session_data["image_width"] = frame.shape[1]
            session_data["image_height"] = frame.shape[0]
            pose_age_s = None
            if last_pose_update_time is not None:
                pose_age_s = max(0.0, time.time() - last_pose_update_time)
            display = draw_overlay(
                frame,
                session_data,
                selected_pixel,
                last_status,
                current_tcp_pose,
                verify_mode,
                move_enabled,
                result_data,
                prediction_data,
                pose_age_s,
                verify_rpy_deg,
            )
            if abs(args.display_scale - 1.0) > 1e-6:
                display = cv2.resize(
                    display,
                    None,
                    fx=args.display_scale,
                    fy=args.display_scale,
                    interpolation=cv2.INTER_LINEAR,
                )
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(20) & 0xFF
            if key == 0xFF:
                continue
            if key == ord("q"):
                save_session_json(args.session_json, session_data)
                if result_data is not None:
                    save_result_json(args.result_json, result_data)
                break
            if key == ord("p"):
                try:
                    current_tcp_pose = robot.get_actual_tcp_pose_live()
                    last_pose_update_time = time.time()
                    last_status = f"Current TCP: {current_tcp_pose}"
                    print(last_status)
                except RobotApiError as exc:
                    last_status = f"TCP read failed: {exc}"
            elif key == ord("s"):
                if current_tcp_pose is None:
                    last_status = "TCP unavailable, cannot save sample"
                else:
                    try:
                        sample = add_sample(session_data, selected_pixel, current_tcp_pose)
                        last_status = (
                            f"Saved sample #{sample['point_id']}: "
                            f"pixel=({sample['pixel_u']:.1f}, {sample['pixel_v']:.1f}) -> "
                            f"robot=({sample['robot_x']:.2f}, {sample['robot_y']:.2f})"
                        )
                        save_session_json(args.session_json, session_data)
                    except PlanarCalibrationError as exc:
                        last_status = str(exc)
            elif key == ord("u"):
                removed = undo_last_sample(session_data)
                prediction_data = None
                result_data = None
                verify_rpy_deg = None
                if removed is None:
                    last_status = "No sample to undo"
                else:
                    last_status = f"Removed sample #{removed['point_id']}"
                    save_session_json(args.session_json, session_data)
            elif key == ord("c"):
                try:
                    result_data = build_result(session_data)
                    prediction_data = None
                    save_result_json(args.result_json, result_data)
                    save_session_json(args.session_json, session_data)
                    stats = result_data["error_stats"]
                    last_status = (
                        "Calibration done: "
                        f"mean={stats['mean_mm']:.3f}mm, "
                        f"max={stats['max_mm']:.3f}mm, "
                        f"p95={stats['p95_mm']:.3f}mm"
                    )
                except PlanarCalibrationError as exc:
                    last_status = str(exc)
            elif key == ord("v"):
                verify_mode = not verify_mode
                prediction_data = None
                if verify_mode:
                    try:
                        current_tcp_pose = robot.get_actual_tcp_pose_live()
                        last_pose_update_time = time.time()
                        verify_rpy_deg = [
                            float(current_tcp_pose[3]),
                            float(current_tcp_pose[4]),
                            float(current_tcp_pose[5]),
                        ]
                        last_status = (
                            "Switched to VERIFY mode, locked RPY="
                            f"({verify_rpy_deg[0]:.2f}, {verify_rpy_deg[1]:.2f}, {verify_rpy_deg[2]:.2f})"
                        )
                    except RobotApiError as exc:
                        verify_mode = False
                        verify_rpy_deg = None
                        last_status = f"Failed to enter VERIFY mode: {exc}"
                else:
                    verify_rpy_deg = None
                    last_status = "Switched to SAMPLE mode"
            elif key == ord("m"):
                move_enabled = not move_enabled
                last_status = f"Robot move {'enabled' if move_enabled else 'disabled'}"
            elif key == ord("g"):
                if not verify_mode:
                    last_status = "Switch to VERIFY mode first"
                elif not move_enabled:
                    last_status = "Press m to enable robot move first"
                else:
                    try:
                        move_robot_to_prediction(
                            robot,
                            result_data,
                            prediction_data,
                            verify_rpy_deg,
                            tool_id=args.tool_id,
                            user_id=args.user_id,
                            safe_z_offset_mm=SAFE_Z_OFFSET_MM,
                        )
                        last_status = "Robot moved above predicted point"
                    except (PlanarCalibrationError, RobotApiError) as exc:
                        last_status = f"Robot move failed: {exc}"
    finally:
        camera.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
