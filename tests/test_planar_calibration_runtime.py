from calibrate_plane import (
    PlanarCalibrationError,
    add_sample,
    build_result,
    default_session_data,
    pixel_to_robot_xy,
    undo_last_sample,
)


def test_session_adds_and_undoes_samples():
    session_data = default_session_data("192.168.58.2", 0.0, None)
    session_data["image_width"] = 640
    session_data["image_height"] = 480
    sample = add_sample(session_data, (123.0, 456.0), [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

    assert sample["robot_x"] == 10.0
    assert len(session_data["samples"]) == 1

    removed = undo_last_sample(session_data)
    assert removed is not None
    assert removed["point_id"] == 1
    assert not session_data["samples"]


def test_session_requires_selected_pixel_before_save():
    session_data = default_session_data("192.168.58.2", 0.0, None)
    try:
        add_sample(session_data, None, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    except PlanarCalibrationError as exc:
        assert "点击一个像素点" in str(exc)
    else:
        raise AssertionError("Expected PlanarCalibrationError")


def test_session_predicts_after_compute():
    session_data = default_session_data("192.168.58.2", 0.0, None)
    session_data["image_width"] = 640
    session_data["image_height"] = 480
    points = [
        ((0.0, 0.0), [0.0, 0.0, 100.0, 180.0, 0.0, 0.0]),
        ((10.0, 0.0), [20.0, 0.0, 100.0, 180.0, 0.0, 0.0]),
        ((0.0, 10.0), [0.0, 30.0, 100.0, 180.0, 0.0, 0.0]),
        ((10.0, 10.0), [20.0, 30.0, 100.0, 180.0, 0.0, 0.0]),
    ]
    for pixel, pose in points:
        add_sample(session_data, pixel, pose)

    result = build_result(session_data)
    prediction = pixel_to_robot_xy(result, (5.0, 5.0))
    assert abs(prediction[0] - 10.0) < 1e-6
    assert abs(prediction[1] - 15.0) < 1e-6
