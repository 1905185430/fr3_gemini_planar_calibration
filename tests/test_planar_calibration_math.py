from calibrate_plane import PlanarCalibrationError, build_result, pixel_to_robot_xy


def _sample(point_id: int, u: float, v: float, x: float, y: float) -> dict:
    return {
        "point_id": point_id,
        "pixel_u": u,
        "pixel_v": v,
        "tcp_pose": [x, y, 100.0, 180.0, 0.0, 0.0],
        "robot_x": x,
        "robot_y": y,
        "timestamp": point_id,
    }


def test_build_calibration_result_fits_exact_projective_mapping():
    samples = [
        _sample(1, 10.0, 20.0, 100.0, 200.0),
        _sample(2, 110.0, 20.0, 300.0, 210.0),
        _sample(3, 15.0, 80.0, 110.0, 400.0),
        _sample(4, 120.0, 90.0, 320.0, 420.0),
        _sample(5, 60.0, 55.0, 210.0, 315.0),
    ]

    result = build_result(
        {
            "samples": samples,
            "image_width": 1280,
            "image_height": 720,
            "robot_ip": "192.168.58.2",
            "grid_spacing_mm": 10.0,
            "camera_calibration_reference": None,
        }
    )

    pred_x, pred_y = pixel_to_robot_xy(result, (60.0, 55.0))
    assert abs(pred_x - 210.0) < 1e-6
    assert abs(pred_y - 315.0) < 1e-6
    assert result["sample_count"] == 5
    assert result["error_stats"]["max_mm"] < 1e-6


def test_build_calibration_result_requires_four_points():
    samples = [
        _sample(1, 0.0, 0.0, 0.0, 0.0),
        _sample(2, 1.0, 0.0, 1.0, 0.0),
        _sample(3, 0.0, 1.0, 0.0, 1.0),
    ]

    try:
        build_result(
            {
                "samples": samples,
                "image_width": 640,
                "image_height": 480,
                "robot_ip": "127.0.0.1",
                "grid_spacing_mm": 0.0,
                "camera_calibration_reference": None,
            }
        )
    except PlanarCalibrationError as exc:
        assert "At least 4 samples" in str(exc)
    else:
        raise AssertionError("Expected PlanarCalibrationError")
