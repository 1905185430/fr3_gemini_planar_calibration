import pytest

from robot_api import RobotApiError, RobotClient


class FakeRobot:
    def __init__(self):
        self._tcp_num = 0

    def SetSpeed(self, speed):
        return 0

    def GetActualTCPNum(self, flag=1):
        return (0, self._tcp_num)

    def SetToolCoord(self, id, t_coord, type, install, toolID, loadNum):
        self._tcp_num = id
        return 0

    def SetToolList(self, id, t_coord, type, install, loadNum):
        return 0

    def SetTcp4RefPoint(self, point_num):
        return 0

    def ComputeTcp4(self):
        return (0, 1, 2, 3, 4, 5, 6)

    def MoveCart(self, *args, **kwargs):
        return 0

    def MoveL(self, *args, **kwargs):
        return 0

    def StopMotion(self):
        return 0


class _RawProxy:
    def GetActualTCPPose(self, flag):
        return (0, 11, 22, 33, 44, 55, 66)


class _StatePkg:
    tl_cur_pos = [1, 2, 3, 4, 5, 6]


class FakeRobotError(FakeRobot):
    def SetSpeed(self, speed):
        return 7


class FakeRobotPoseErrMinus4(FakeRobot):
    def __init__(self):
        super().__init__()
        self.robot = _RawProxy()
        self.robot_state_pkg = _StatePkg()

    def GetActualTCPPose(self, flag=1):
        return -4


class FakeRobotNeedsTcpZero(FakeRobot):
    def __init__(self):
        super().__init__()
        self._tcp_num = 1

    def GetActualTCPNum(self, flag=1):
        return (0, self._tcp_num)

    def SetToolCoord(self, id, t_coord, type, install, toolID, loadNum):
        self._tcp_num = id
        return 0

    def SetTcp4RefPoint(self, point_num):
        if self._tcp_num != 0:
            return 37
        return 0


class _RawStopProxy:
    def __init__(self):
        self.calls = 0

    def StopMotion(self):
        self.calls += 1
        return 0


class FakeRobotStopMotionRetry(FakeRobot):
    def __init__(self):
        super().__init__()
        self.stop_calls = 0

    def StopMotion(self):
        self.stop_calls += 1
        if self.stop_calls == 1:
            raise RuntimeError("Request-sent")
        return 0


class FakeRobotStopMotionRawFallback(FakeRobot):
    def __init__(self):
        super().__init__()
        self.robot = _RawStopProxy()

    def StopMotion(self):
        raise RuntimeError("Request-sent")


class FakeRobotStopMotionHardFail(FakeRobot):
    def StopMotion(self):
        raise RuntimeError("socket closed")


def test_client_unwraps_success_tuple():
    client = RobotClient("127.0.0.1", rpc_factory=lambda _: FakeRobot())
    assert client.compute_tcp4() == [1, 2, 3, 4, 5, 6]


def test_client_raises_on_nonzero_errcode():
    client = RobotClient("127.0.0.1", rpc_factory=lambda _: FakeRobotError())
    with pytest.raises(RobotApiError):
        client.set_speed(20)


def test_get_actual_tcp_pose_falls_back_to_raw_xmlrpc_when_sdk_returns_minus4():
    client = RobotClient("127.0.0.1", rpc_factory=lambda _: FakeRobotPoseErrMinus4())
    assert client.get_actual_tcp_pose() == [11, 22, 33, 44, 55, 66]


def test_set_tcp4_ref_point_auto_switches_to_tcp_zero():
    client = RobotClient("127.0.0.1", rpc_factory=lambda _: FakeRobotNeedsTcpZero())
    client.set_tcp4_ref_point(1)


def test_apply_calibrated_tool_switches_active_tool():
    client = RobotClient("127.0.0.1", rpc_factory=lambda _: FakeRobot())
    tcp = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
    client.apply_calibrated_tool(2, tcp)
    assert client.get_actual_tcp_num() == 2


def test_debug_prints_raw_errcode_on_failure(monkeypatch, capsys):
    monkeypatch.setenv("DEBUG", "1")
    client = RobotClient("127.0.0.1", rpc_factory=lambda _: FakeRobotError())

    with pytest.raises(RobotApiError):
        client.set_speed(20)

    output = capsys.readouterr().out
    assert "robot.SetSpeed(20)" in output
    assert "errcode=7" in output


def test_stop_motion_retries_when_request_sent_exception_occurs():
    fake = FakeRobotStopMotionRetry()
    client = RobotClient("127.0.0.1", rpc_factory=lambda _: fake)
    client.stop_motion()
    assert fake.stop_calls == 2


def test_stop_motion_falls_back_to_raw_xmlrpc_on_repeated_request_sent():
    fake = FakeRobotStopMotionRawFallback()
    client = RobotClient("127.0.0.1", rpc_factory=lambda _: fake)
    client.stop_motion()
    assert fake.robot.calls == 1


def test_stop_motion_raises_on_non_transient_exception():
    client = RobotClient("127.0.0.1", rpc_factory=lambda _: FakeRobotStopMotionHardFail())
    with pytest.raises(RobotApiError, match="socket closed"):
        client.stop_motion()

