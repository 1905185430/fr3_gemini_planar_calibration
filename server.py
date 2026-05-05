from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from calibrate_plane import (
    CAMERA_CALIB_DIR,
    FPS,
    GEMINI_SDK_ROOT,
    GRID_SPACING_MM,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    INIT_RX_DEG,
    INIT_RY_DEG,
    INIT_RZ_DEG,
    RESULT_JSON,
    ROBOT_IP,
    SAFE_Z_OFFSET_MM,
    SESSION_JSON,
    TOOL_ID,
    USER_ID,
    PlanarCalibrationError,
    add_sample,
    align_robot_to_initial_rpy,
    build_result,
    default_session_data,
    load_result_json,
    load_session_json,
    move_robot_to_prediction,
    pixel_to_robot_xy,
    save_result_json,
    save_session_json,
    undo_last_sample,
)
from gemini_camera import Gemini335Camera, GeminiCameraError
from robot_api import RobotApiError, RobotClient


ROOT = Path(__file__).resolve().parent


class ClickPayload(BaseModel):
    x: float
    y: float


class BoolPayload(BaseModel):
    enabled: bool


class AppContext:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.robot: RobotClient | None = None
        self.camera: Gemini335Camera | None = None
        self.session_data = load_session_json(
            args.session_json,
            args.robot_ip,
            args.grid_spacing_mm,
            str(args.camera_calib_dir) if args.camera_calib_dir else None,
        )
        self.result_data = load_result_json(args.result_json)
        self.selected_pixel: tuple[float, float] | None = None
        self.prediction_data: dict | None = None
        self.last_status = "等待采样"
        self.verify_mode = False
        self.move_enabled = False
        self.verify_rpy_deg: list[float] | None = None
        self.current_tcp_pose: list[float] | None = None
        self.last_pose_update_time: float | None = None

    def pose_age_s(self) -> float | None:
        if self.last_pose_update_time is None:
            return None
        return max(0.0, time.time() - self.last_pose_update_time)

    def state_payload(self) -> dict:
        return {
            "robot_ip": self.session_data.get("robot_ip"),
            "image_width": self.session_data.get("image_width", 0),
            "image_height": self.session_data.get("image_height", 0),
            "samples": self.session_data.get("samples", []),
            "sample_count": len(self.session_data.get("samples", [])),
            "selected_pixel": self.selected_pixel,
            "last_status": self.last_status,
            "verify_mode": self.verify_mode,
            "move_enabled": self.move_enabled,
            "verify_rpy_deg": self.verify_rpy_deg,
            "current_tcp_pose": self.current_tcp_pose,
            "pose_age_s": self.pose_age_s(),
            "result_data": self.result_data,
            "prediction_data": self.prediction_data,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FR3 + Gemini335 local web UI")
    parser.add_argument("--robot-ip", default=ROBOT_IP)
    parser.add_argument("--tool-id", type=int, default=TOOL_ID)
    parser.add_argument("--user-id", type=int, default=USER_ID)
    parser.add_argument("--grid-spacing-mm", type=float, default=GRID_SPACING_MM)
    parser.add_argument("--camera-calib-dir", type=Path, default=CAMERA_CALIB_DIR)
    parser.add_argument("--gemini-sdk-root", type=Path, default=GEMINI_SDK_ROOT)
    parser.add_argument("--width", type=int, default=IMAGE_WIDTH)
    parser.add_argument("--height", type=int, default=IMAGE_HEIGHT)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--session-json", type=Path, default=ROOT / SESSION_JSON)
    parser.add_argument("--result-json", type=Path, default=ROOT / RESULT_JSON)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


ARGS = parse_args()
CTX = AppContext(ARGS)
app = FastAPI(title="FR3 Gemini335 Planar Calibration Web UI")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


def require_devices() -> tuple[RobotClient, Gemini335Camera]:
    if CTX.robot is None or CTX.camera is None:
        raise HTTPException(status_code=503, detail="机器人或相机尚未初始化")
    return CTX.robot, CTX.camera


def refresh_tcp_pose(robot: RobotClient) -> list[float]:
    pose = robot.get_actual_tcp_pose_live()
    CTX.current_tcp_pose = pose
    CTX.last_pose_update_time = time.time()
    return pose


def draw_stream_frame(frame: np.ndarray) -> np.ndarray:
    canvas = frame.copy()
    if CTX.selected_pixel is not None:
        u, v = CTX.selected_pixel
        cv2.drawMarker(canvas, (int(round(u)), int(round(v))), (0, 255, 255), markerSize=18, thickness=2)
    if CTX.prediction_data is not None and CTX.selected_pixel is not None:
        u, v = CTX.selected_pixel
        cv2.circle(canvas, (int(round(u)), int(round(v))), 22, (0, 255, 0), 2)
    mode_text = "VERIFY" if CTX.verify_mode else "SAMPLE"
    cv2.putText(canvas, f"Mode: {mode_text}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Samples: {len(CTX.session_data.get('samples', []))}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    return canvas


@app.on_event("startup")
def on_startup() -> None:
    CTX.robot = RobotClient(CTX.args.robot_ip)
    align_robot_to_initial_rpy(
        CTX.robot,
        tool_id=CTX.args.tool_id,
        user_id=CTX.args.user_id,
        rx_deg=INIT_RX_DEG,
        ry_deg=INIT_RY_DEG,
        rz_deg=INIT_RZ_DEG,
    )
    CTX.current_tcp_pose = refresh_tcp_pose(CTX.robot)
    CTX.camera = Gemini335Camera(
        sdk_root=CTX.args.gemini_sdk_root,
        color_width=CTX.args.width,
        color_height=CTX.args.height,
        fps=CTX.args.fps,
        calib_dir=CTX.args.camera_calib_dir,
    )


@app.on_event("shutdown")
def on_shutdown() -> None:
    if CTX.camera is not None:
        CTX.camera.close()


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(ROOT / "templates" / "index.html")


@app.get("/api/state")
def api_state():
    robot, _ = require_devices()
    with CTX.lock:
        try:
            refresh_tcp_pose(robot)
        except RobotApiError as exc:
            CTX.last_status = f"读取 TCP 位姿失败: {exc}"
        return JSONResponse(CTX.state_payload())


@app.post("/api/click")
def api_click(payload: ClickPayload):
    _, camera = require_devices()
    with CTX.lock:
        CTX.selected_pixel = camera.undistort_pixel((payload.x, payload.y))
        CTX.prediction_data = None
        CTX.last_status = f"已选择像素点 ({CTX.selected_pixel[0]:.2f}, {CTX.selected_pixel[1]:.2f})"
        if CTX.verify_mode and CTX.result_data is not None:
            try:
                robot_xy = pixel_to_robot_xy(CTX.result_data, CTX.selected_pixel)
                CTX.prediction_data = {"pixel_uv": CTX.selected_pixel, "robot_xy": robot_xy}
                CTX.last_status = (
                    f"预测机器人 XY: pixel=({CTX.selected_pixel[0]:.2f}, {CTX.selected_pixel[1]:.2f}) -> "
                    f"robot=({robot_xy[0]:.2f}, {robot_xy[1]:.2f})"
                )
            except PlanarCalibrationError as exc:
                CTX.last_status = str(exc)
        return JSONResponse(CTX.state_payload())


@app.post("/api/sample/save")
def api_sample_save():
    robot, _ = require_devices()
    with CTX.lock:
        pose = refresh_tcp_pose(robot)
        try:
            sample = add_sample(CTX.session_data, CTX.selected_pixel, pose)
        except PlanarCalibrationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        CTX.last_status = (
            f"已保存采样点 #{sample['point_id']}: "
            f"pixel=({sample['pixel_u']:.2f}, {sample['pixel_v']:.2f}) -> "
            f"robot=({sample['robot_x']:.2f}, {sample['robot_y']:.2f})"
        )
        save_session_json(CTX.args.session_json, CTX.session_data)
        return JSONResponse(CTX.state_payload())


@app.post("/api/sample/undo")
def api_sample_undo():
    with CTX.lock:
        removed = undo_last_sample(CTX.session_data)
        CTX.prediction_data = None
        CTX.result_data = None
        CTX.verify_rpy_deg = None
        if removed is None:
            CTX.last_status = "当前没有可撤销的采样点"
        else:
            CTX.last_status = f"已撤销采样点 #{removed['point_id']}"
            save_session_json(CTX.args.session_json, CTX.session_data)
        return JSONResponse(CTX.state_payload())


@app.post("/api/calibration/compute")
def api_compute():
    with CTX.lock:
        try:
            CTX.result_data = build_result(CTX.session_data)
        except PlanarCalibrationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        save_result_json(CTX.args.result_json, CTX.result_data)
        save_session_json(CTX.args.session_json, CTX.session_data)
        stats = CTX.result_data["error_stats"]
        CTX.last_status = (
            "标定完成: "
            f"mean={stats['mean_mm']:.3f}mm, "
            f"max={stats['max_mm']:.3f}mm, "
            f"p95={stats['p95_mm']:.3f}mm"
        )
        return JSONResponse(CTX.state_payload())


@app.post("/api/mode/verify")
def api_verify_mode(payload: BoolPayload):
    robot, _ = require_devices()
    with CTX.lock:
        CTX.verify_mode = payload.enabled
        CTX.prediction_data = None
        if CTX.verify_mode:
            pose = refresh_tcp_pose(robot)
            CTX.verify_rpy_deg = [float(pose[3]), float(pose[4]), float(pose[5])]
            CTX.last_status = (
                "已进入验证模式，锁定 RPY="
                f"({CTX.verify_rpy_deg[0]:.2f}, {CTX.verify_rpy_deg[1]:.2f}, {CTX.verify_rpy_deg[2]:.2f})"
            )
        else:
            CTX.verify_rpy_deg = None
            CTX.last_status = "已切换回采样模式"
        return JSONResponse(CTX.state_payload())


@app.post("/api/move/enable")
def api_move_enable(payload: BoolPayload):
    with CTX.lock:
        CTX.move_enabled = payload.enabled
        CTX.last_status = f"机器人运动已{'启用' if CTX.move_enabled else '禁用'}"
        return JSONResponse(CTX.state_payload())


@app.post("/api/move/go")
def api_move_go():
    robot, _ = require_devices()
    with CTX.lock:
        if not CTX.verify_mode:
            raise HTTPException(status_code=400, detail="请先切换到验证模式")
        if not CTX.move_enabled:
            raise HTTPException(status_code=400, detail="请先启用机器人运动")
        try:
            move_robot_to_prediction(
                robot,
                CTX.result_data,
                CTX.prediction_data,
                CTX.verify_rpy_deg,
                tool_id=CTX.args.tool_id,
                user_id=CTX.args.user_id,
                safe_z_offset_mm=SAFE_Z_OFFSET_MM,
            )
            CTX.last_status = "机器人已移动到预测点上方"
        except (RobotApiError, PlanarCalibrationError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(CTX.state_payload())


@app.post("/api/robot/align")
def api_robot_align():
    robot, _ = require_devices()
    with CTX.lock:
        pose = align_robot_to_initial_rpy(robot, CTX.args.tool_id, CTX.args.user_id)
        CTX.current_tcp_pose = pose
        CTX.last_pose_update_time = time.time()
        CTX.last_status = (
            f"机械臂姿态已对正到 ({pose[3]:.2f}, {pose[4]:.2f}, {pose[5]:.2f})"
        )
        return JSONResponse(CTX.state_payload())


@app.get("/video_feed")
def video_feed():
    _, camera = require_devices()

    def generate():
        while True:
            try:
                with CTX.lock:
                    frame = camera.read_color_image()
                    frame = camera.undistort_image(frame)
                    CTX.session_data["image_width"] = frame.shape[1]
                    CTX.session_data["image_height"] = frame.shape[0]
                    frame = draw_stream_frame(frame)
                ok, encoded = cv2.imencode(".jpg", frame)
                if not ok:
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
                )
                time.sleep(0.05)
            except GeminiCameraError:
                time.sleep(0.2)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=ARGS.host, port=ARGS.port, reload=False)
