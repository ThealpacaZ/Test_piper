import rospy  # 添加这行导入
import threading
from typing import Callable, Optional


class ROSSubscriber:
    def __init__(self, topic_name, msg_type, call: Optional[Callable] = None):
        """
        Initialize ROS subscriber
        :param topic_name: Name of the topic to subscribe to
        :param msg_type: Type of the message
        """
        self.topic_name = topic_name
        self.msg_type = msg_type
        self.latest_msg = None
        self.lock = threading.Lock()
        self.user_call = call
        self.subscriber = None

        # 延迟创建subscriber，确保rospy已经初始化
        self._create_subscriber()

    def _create_subscriber(self):
        """创建订阅者"""
        try:
            self.subscriber = rospy.Subscriber(self.topic_name, self.msg_type, self.callback)
            rospy.loginfo(f"Successfully subscribed to topic: {self.topic_name}")
        except Exception as e:
            rospy.logerr(f"Failed to subscribe to topic {self.topic_name}: {str(e)}")
            raise

    def callback(self, msg):
        """
        Subscriber callback function to receive messages and update the latest data.
        :param msg: The received message
        """
        with self.lock:
            self.latest_msg = msg
            if self.user_call:
                self.user_call(self.latest_msg)

    def get_latest_data(self):
        with self.lock:
            return self.latest_msg

    def is_connected(self):
        """检查是否有发布者连接"""
        if self.subscriber:
            return self.subscriber.get_num_connections() > 0
        return False
