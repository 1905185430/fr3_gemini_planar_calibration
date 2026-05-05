from pathlib import Path

from calibrate_plane import (
    load_result_json,
    load_session_json,
    save_result_json,
    save_session_json,
)


def test_save_and_load_planar_session(tmp_path: Path):
    path = tmp_path / "session.json"
    session = {
        "robot_ip": "192.168.58.2",
        "image_width": 1280,
        "image_height": 720,
        "grid_spacing_mm": 20.0,
        "camera_calibration_reference": "/tmp/camera",
        "samples": [
            {
                "point_id": 1,
                "pixel_u": 100.0,
                "pixel_v": 200.0,
                "tcp_pose": [1, 2, 3, 4, 5, 6],
                "robot_x": 1.0,
                "robot_y": 2.0,
                "timestamp": 123.4,
            }
        ],
    }

    save_session_json(path, session)
    loaded = load_session_json(path, "192.168.58.2", 20.0, "/tmp/camera")

    assert loaded["robot_ip"] == session["robot_ip"]
    assert loaded["image_width"] == 1280
    assert len(loaded["samples"]) == 1
    assert loaded["samples"][0]["pixel_u"] == 100.0


def test_save_and_load_planar_result(tmp_path: Path):
    path = tmp_path / "result.json"
    result = {
        "homography_pixel_to_robot": [[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]],
        "homography_robot_to_pixel": [[1.0, 0.0, -10.0], [0.0, 1.0, -20.0], [0.0, 0.0, 1.0]],
        "error_stats": {"mean_mm": 0.1, "max_mm": 0.2, "p95_mm": 0.19},
        "image_width": 1280,
        "image_height": 720,
        "robot_ip": "192.168.58.2",
        "grid_spacing_mm": 10.0,
        "camera_calibration_reference": "/tmp/camera",
        "sample_count": 1,
        "inlier_mask": [True],
        "samples": [
            {
                "point_id": 1,
                "pixel_u": 100.0,
                "pixel_v": 200.0,
                "tcp_pose": [1, 2, 3, 4, 5, 6],
                "robot_x": 1.0,
                "robot_y": 2.0,
                "timestamp": 123.4,
            }
        ],
        "plane_z_mm": 33.0,
        "reference_rpy_deg": [180.0, 0.0, 0.0],
    }

    save_result_json(path, result)
    loaded = load_result_json(path)

    assert loaded is not None
    assert loaded["sample_count"] == 1
    assert loaded["error_stats"]["mean_mm"] == 0.1
    assert loaded["reference_rpy_deg"] == [180.0, 0.0, 0.0]
