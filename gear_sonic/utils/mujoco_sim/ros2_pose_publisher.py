"""ROS2 Odometry publisher for MuJoCo GT root pose.

Publishes the floating-base pose of the robot (`mj_data.qpos[0:7]`) on a
ROS2 topic that mimics the FAST-LIO `/Odometry` output, so downstream
consumers (e.g. the C++ deploy stack) can subscribe to it as if it were
real odometry.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np


class MujocoOdometryPublisher:
    """Publishes MuJoCo GT base pose as nav_msgs/Odometry on a background thread.

    Topic / frames mirror FAST-LIO defaults:
      - topic: ``/Odometry``
      - frame_id (parent): ``camera_init``
      - child_frame_id:    ``body``
    """

    def __init__(
        self,
        sim_env,
        topic: str = "/Odometry",
        rate_hz: float = 100.0,
        node_name: str = "mujoco_odom_publisher",
        frame_id: str = "camera_init",
        child_frame_id: str = "body",
    ):
        self.sim_env = sim_env
        self.topic = topic
        self.rate_hz = rate_hz
        self.node_name = node_name
        self.frame_id = frame_id
        self.child_frame_id = child_frame_id

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._node = None
        self._pub = None
        self._rclpy = None
        self._Odometry = None

    def _setup_ros2(self) -> bool:
        try:
            import rclpy
            from rclpy.node import Node
            from nav_msgs.msg import Odometry
        except ImportError as e:
            print(f"[MujocoOdometryPublisher] rclpy/nav_msgs not available, disabling: {e}")
            return False

        if not rclpy.ok():
            rclpy.init()

        self._rclpy = rclpy
        self._Odometry = Odometry
        self._node = Node(self.node_name)
        self._pub = self._node.create_publisher(Odometry, self.topic, 10)
        print(
            f"[MujocoOdometryPublisher] node='{self.node_name}' topic='{self.topic}' "
            f"rate={self.rate_hz}Hz frame='{self.frame_id}'->'{self.child_frame_id}'"
        )
        return True

    def _shutdown_ros2(self):
        try:
            if self._node is not None:
                self._node.destroy_node()
        except Exception:
            pass
        try:
            if self._rclpy is not None and self._rclpy.ok():
                self._rclpy.shutdown()
        except Exception:
            pass

    def _loop(self):
        if not self._setup_ros2():
            return

        period = 1.0 / max(self.rate_hz, 1.0)
        try:
            while not self._stop_event.is_set():
                step_start = time.monotonic()
                try:
                    qpos = np.asarray(self.sim_env.mj_data.qpos[:7], dtype=np.float64)
                    qvel = np.asarray(self.sim_env.mj_data.qvel[:6], dtype=np.float64)
                except Exception as e:
                    print(f"[MujocoOdometryPublisher] qpos read failed: {e}")
                    time.sleep(period)
                    continue

                msg = self._Odometry()
                stamp = self._node.get_clock().now().to_msg()
                msg.header.stamp = stamp
                msg.header.frame_id = self.frame_id
                msg.child_frame_id = self.child_frame_id

                msg.pose.pose.position.x = float(qpos[0])
                msg.pose.pose.position.y = float(qpos[1])
                msg.pose.pose.position.z = float(qpos[2])
                # MuJoCo qpos quaternion order: [w, x, y, z]
                msg.pose.pose.orientation.w = float(qpos[3])
                msg.pose.pose.orientation.x = float(qpos[4])
                msg.pose.pose.orientation.y = float(qpos[5])
                msg.pose.pose.orientation.z = float(qpos[6])

                msg.twist.twist.linear.x = float(qvel[0])
                msg.twist.twist.linear.y = float(qvel[1])
                msg.twist.twist.linear.z = float(qvel[2])
                msg.twist.twist.angular.x = float(qvel[3])
                msg.twist.twist.angular.y = float(qvel[4])
                msg.twist.twist.angular.z = float(qvel[5])

                self._pub.publish(msg)

                elapsed = time.monotonic() - step_start
                sleep_time = period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            self._shutdown_ros2()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name=self.node_name, daemon=True)
        self._thread.start()

    def stop(self, join_timeout: float = 1.0):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
            self._thread = None