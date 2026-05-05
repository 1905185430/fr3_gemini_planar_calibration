from __future__ import annotations

import os
from typing import Any, Callable

from fairino import Robot


class RobotApiError(RuntimeError):
    pass


class RobotClient:
    def __init__(self, ip: str, rpc_factory: Callable[[str], Any] | None = None) -> None:
        self._debug = os.getenv("DEBUG") == "1"
        factory = rpc_factory or (lambda host: Robot.RPC(host))
        try:
            self._robot = factory(ip)
        except Exception as exc:  # pragma: no cover - hardware/network dependent
            raise RobotApiError(f"Failed to connect robot at {ip}: {exc}") from exc

    def _trace(self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if not self._debug:
            return
        arg_text = ", ".join(repr(v) for v in args)
        kw_text = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
        joiner = ", " if arg_text and kw_text else ""
        print(f"[DEBUG] robot.{method}({arg_text}{joiner}{kw_text})")

    @staticmethod
    def _extract_errcode(result: Any) -> int | None:
        if isinstance(result, int):
            return result
        if isinstance(result, (list, tuple)) and result and isinstance(result[0], int):
            return int(result[0])
        return None

    def _trace_result(self, method: str, result: Any) -> None:
        if not self._debug:
            return
        errcode = self._extract_errcode(result)
        if errcode is None:
            print(f"[DEBUG] robot.{method} raw_result={result!r}")
            return
        print(f"[DEBUG] robot.{method} raw_result={result!r} errcode={errcode}")

    def _trace_exception(self, method: str, exc: Exception) -> None:
        if not self._debug:
            return
        print(f"[DEBUG] robot.{method} exception={exc}")

    def _unwrap(self, method: str, result: Any) -> Any:
        if isinstance(result, int):
            if result != 0:
                raise RobotApiError(f"{method} failed with errcode {result}")
            return None

        if isinstance(result, (list, tuple)) and result:
            err = result[0]
            if isinstance(err, int):
                if err != 0:
                    raise RobotApiError(f"{method} failed with errcode {err}")
                if len(result) == 1:
                    return None
                if len(result) == 2:
                    return result[1]
                return tuple(result[1:])

        return result

    @staticmethod
    def _is_errcode(error: Exception, errcode: int) -> bool:
        return f"errcode {errcode}" in str(error)

    @staticmethod
    def _is_request_sent_error(error: Exception) -> bool:
        return "Request-sent" in str(error)

    def call_raw_xmlrpc(self, method: str, *args: Any) -> Any:
        self._trace(f"raw.{method}", args, {})
        robot_proxy = getattr(self._robot, "robot", None)
        if robot_proxy is None:
            raise RobotApiError("Underlying xmlrpc proxy is unavailable")

        try:
            fn = getattr(robot_proxy, method)
            result = fn(*args)
        except Exception as exc:
            self._trace_exception(f"raw.{method}", exc)
            raise RobotApiError(f"raw {method} raised exception: {exc}") from exc
        self._trace_result(f"raw.{method}", result)
        return self._unwrap(f"raw {method}", result)

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        self._trace(method, args, kwargs)
        try:
            fn = getattr(self._robot, method)
            result = fn(*args, **kwargs)
        except Exception as exc:
            self._trace_exception(method, exc)
            raise RobotApiError(f"{method} raised exception: {exc}") from exc
        self._trace_result(method, result)
        return self._unwrap(method, result)

    def set_speed(self, speed: int = 20) -> None:
        self.call("SetSpeed", speed)

    def get_actual_tcp_num(self) -> int:
        return int(self.call("GetActualTCPNum"))

    def ensure_tcp_zero_for_calibration(self) -> None:
        if self.get_actual_tcp_num() == 0:
            return
        self.call("SetToolCoord", 0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0, 0, 0, 0)
        if self.get_actual_tcp_num() != 0:
            raise RobotApiError("Failed to switch active TCP to 0 before 4-point calibration")

    def set_tcp4_ref_point(self, point_num: int) -> None:
        self.ensure_tcp_zero_for_calibration()
        self.call("SetTcp4RefPoint", point_num)

    def compute_tcp4(self) -> list[float]:
        self.ensure_tcp_zero_for_calibration()
        return list(self.call("ComputeTcp4"))

    def set_tool_list(self, tool_id: int, t_coord: list[float]) -> None:
        self.call("SetToolList", tool_id, t_coord, 0, 0, 0)

    def apply_calibrated_tool(self, tool_id: int, t_coord: list[float]) -> None:
        self.call("SetToolList", tool_id, t_coord, 0, 0, 0)
        self.call("SetToolCoord", tool_id, t_coord, 0, 0, 0, 0)
        active = self.get_actual_tcp_num()
        if active != tool_id:
            raise RobotApiError(
                f"Failed to activate tool {tool_id}; current active TCP is {active}"
            )

    def get_actual_tcp_pose(self) -> list[float]:
        try:
            return list(self.call("GetActualTCPPose"))
        except RobotApiError as exc:
            if not self._is_errcode(exc, -4):
                raise

            Robot.RPC.is_conect = True
            try:
                return list(self.call("GetActualTCPPose"))
            except RobotApiError as retry_exc:
                if not self._is_errcode(retry_exc, -4):
                    raise

            try:
                return list(self.call_raw_xmlrpc("GetActualTCPPose", 1))
            except RobotApiError:
                pass

            state_pkg = getattr(self._robot, "robot_state_pkg", None)
            if state_pkg is not None and hasattr(state_pkg, "tl_cur_pos"):
                return [float(v) for v in state_pkg.tl_cur_pos[:6]]

            raise exc

    def get_actual_tcp_pose_live(self) -> list[float]:
        try:
            return list(self.call_raw_xmlrpc("GetActualTCPPose", 1))
        except RobotApiError:
            return self.get_actual_tcp_pose()

    def move_l(
        self,
        pose: list[float],
        tool: int,
        user: int,
        vel: float = 20.0,
        blend_r: float = -1.0,
    ) -> None:
        self.call("MoveL", pose, tool, user, vel=vel, blendR=blend_r)

    def move_ptp(self, pose: list[float], tool: int, user: int, vel: float = 20.0) -> None:
        self.call("MoveCart", pose, tool, user, vel=vel)

    def stop_motion(self) -> None:
        try:
            self.call("StopMotion")
            return
        except RobotApiError as exc:
            if not self._is_request_sent_error(exc):
                raise

        Robot.RPC.is_conect = True
        try:
            self.call("StopMotion")
            return
        except RobotApiError as retry_exc:
            if not self._is_request_sent_error(retry_exc):
                raise

        self.call_raw_xmlrpc("StopMotion")
