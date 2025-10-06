import rclpy
from rclpy.node import Node
from threading import Lock
from typing import Callable, Optional


class ROS2Subscriber(Node):
    # 类变量，确保 rclpy 只初始化一次
    _rclpy_initialized = False
    _lock = Lock()  # 用于保护 rclpy 初始化
    def __init__(self, node_name: str, topic_name: str, msg_type, call: Optional[Callable] = None):
        with ROS2Subscriber._lock:
            if not ROS2Subscriber._rclpy_initialized:
                rclpy.init()
                ROS2Subscriber._rclpy_initialized = True
        """
        ROS2 Subscriber 封装类
        :param node_name: 节点名称
        :param topic_name: 订阅的话题名
        :param msg_type: 消息类型
        :param call: 可选的回调函数
        """
        super().__init__(node_name)
        self.topic_name = topic_name
        self.msg_type = msg_type
        self.latest_msg = None
        self.lock = Lock()
        self.user_call = call
        
        
        self.subscription = self.create_subscription(
            msg_type,
            topic_name,
            self.callback,
            10  # QoS depth
        )

    def callback(self, msg):
        with self.lock:
            self.latest_msg = msg
            if self.user_call:
                self.user_call(msg)

    def get_latest_data(self):
        rclpy.spin_once(self, timeout_sec=0.5)
        
        with self.lock:
            return self.latest_msg

import time
from sensor_msgs.msg import Image

def custom_callback(msg):
    print(f"Received: SWA={msg.swa}, SWC={msg.swc}")

