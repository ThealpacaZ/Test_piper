import numpy as np
import time
from sensor.vision_sensor import VisionSensor
import rospy
import threading
from sensor_msgs.msg import Image
from utils.data_handler import debug_print


def ros_image_to_numpy_optimized(ros_image):
    """
    高效的ROS图像到numpy转换（不使用cv_bridge）
    """
    try:
        # 将ROS图像数据转为numpy数组
        if ros_image.encoding == 'bgr8':
            # 8位BGR图像 (3通道)
            dtype = np.uint8
            channels = 3
        elif ros_image.encoding == 'rgb8':
            # 8位RGB图像 (3通道)
            dtype = np.uint8
            channels = 3
        elif ros_image.encoding == 'mono8':
            # 8位灰度图像 (1通道)
            dtype = np.uint8
            channels = 1
        elif ros_image.encoding in ['16UC1', 'mono16']:
            # 16位深度图像 (1通道)
            dtype = np.uint16
            channels = 1
        elif ros_image.encoding == '32FC1':
            # 32位浮点深度图像 (1通道)
            dtype = np.float32
            channels = 1
        else:
            raise ValueError(f"不支持的图像编码: {ros_image.encoding}")

        # 从ROS消息数据创建numpy数组
        np_arr = np.frombuffer(ros_image.data, dtype=dtype)

        # 重新整形为图像尺寸
        if channels == 1:
            # 单通道图像
            expected_size = ros_image.height * ros_image.width
            if len(np_arr) >= expected_size:
                np_arr = np_arr[:expected_size]
                image = np_arr.reshape((ros_image.height, ros_image.width))
            else:
                raise ValueError(f"数据长度不足: 期望{expected_size}, 实际{len(np_arr)}")
        else:
            # 多通道图像
            expected_size = ros_image.height * ros_image.width * channels
            if len(np_arr) >= expected_size:
                np_arr = np_arr[:expected_size]
                image = np_arr.reshape((ros_image.height, ros_image.width, channels))
            else:
                raise ValueError(f"数据长度不足: 期望{expected_size}, 实际{len(np_arr)}")

        return image

    except Exception as e:
        raise Exception(f"图像转换失败: {str(e)}")


class ROSSubscriber:
    def __init__(self, topic_name, msg_type, call=None):
        self.topic_name = topic_name
        self.msg_type = msg_type
        self.latest_msg = None
        self.lock = threading.Lock()
        self.user_call = call
        self.subscriber = None
        self.msg_count = 0  # 添加消息计数器

        self._create_subscriber()

    def _create_subscriber(self):
        try:
            self.subscriber = rospy.Subscriber(self.topic_name, self.msg_type, self.callback)
            print(f"成功创建订阅者: {self.topic_name}")
        except Exception as e:
            print(f"创建订阅者失败: {self.topic_name}, 错误: {e}")
            raise

    def callback(self, msg):
        with self.lock:
            self.latest_msg = msg
            self.msg_count += 1
            if self.msg_count % 30 == 0:  # 每30帧打印一次
                print(f"已接收 {self.msg_count} 帧图像")
            if self.user_call:
                self.user_call(self.latest_msg)

    def get_latest_data(self):
        with self.lock:
            return self.latest_msg


class VisionROSensor(VisionSensor):
    def __init__(self, name):
        super().__init__()
        self.name = name
        print(f"初始化VisionROSensor: {name}")

    def set_up(self, topic, depth_topic=None):
        self.controller = {}
        self.is_depth = depth_topic is not None

        try:
            print(f"正在订阅彩色话题: {topic}")
            self.controller["color"] = ROSSubscriber(topic, Image)

            if self.is_depth:
                print(f"正在订阅深度话题: {depth_topic}")
                self.controller["depth"] = ROSSubscriber(depth_topic, Image)

            # 等待订阅生效
            rospy.sleep(2.0)
            print("订阅设置完成")

        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"初始化相机失败: {str(e)}")

    def get_image(self):
        image = {}

        # 获取彩色图像
        rgb_frame = self.controller["color"].get_latest_data()

        if rgb_frame is not None:
            try:
                # 使用纯Python转换
                rgb_array = ros_image_to_numpy_optimized(rgb_frame)

                # 如果是BGR格式，转换为RGB
                if rgb_frame.encoding == 'bgr8':
                    image["color"] = rgb_array[:, :, ::-1].copy()  # BGR -> RGB
                else:
                    image["color"] = rgb_array.copy()

                # 确保数据类型
                if image["color"].dtype != np.uint8:
                    image["color"] = image["color"].astype(np.uint8)

                print(f"成功转换彩色图像: {image['color'].shape}, 类型: {image['color'].dtype}")

            except Exception as e:
                debug_print(self.name, f"转换彩色图像失败: {str(e)}", "ERROR")
                image["color"] = None
        else:
            debug_print(self.name, "未收到彩色图像数据", "WARNING")
            image["color"] = None

        # 获取深度图像（如果需要）
        if "depth" in self.collect_info:
            if not self.is_depth:
                debug_print(self.name, "需要设置depth_topic来启用深度图像收集", "ERROR")
                raise ValueError("需要提供depth_topic参数")
            else:
                depth_frame = self.controller["depth"].get_latest_data()
                if depth_frame is not None:
                    try:
                        depth_array = ros_image_to_numpy_optimized(depth_frame)
                        image["depth"] = depth_array.copy()
                        print(f"成功转换深度图像: {image['depth'].shape}")
                    except Exception as e:
                        debug_print(self.name, f"转换深度图像失败: {str(e)}", "ERROR")
                        image["depth"] = None
                else:
                    image["depth"] = None

        return image

    def cleanup(self):
        try:
            if hasattr(self, 'controller'):
                for key in self.controller:
                    if hasattr(self.controller[key], 'subscriber') and self.controller[key].subscriber:
                        self.controller[key].subscriber.unregister()
                        print(f"已清理订阅者: {key}")
        except Exception as e:
            print(f"清理时出错: {str(e)}")


if __name__ == "__main__":
    rospy.init_node('test_vision_ros_sensor', anonymous=True)

    cam = VisionROSensor("test_camera")
    cam.set_up("/camera/color/image_raw")
    cam.set_collect_info(["color"])

    print("开始测试...")

    try:
        for i in range(50):
            print(f"测试循环 {i}")
            data = cam.get_image()

            if data["color"] is not None:
                print(f"成功获取图像 {i}: {data['color'].shape}")
                # 可选：保存一张图片验证
                if i == 10:
                    try:
                        import cv2

                        cv2.imwrite(f"test_image_{i}.jpg", cv2.cvtColor(data['color'], cv2.COLOR_RGB2BGR))
                        print(f"已保存测试图像: test_image_{i}.jpg")
                    except ImportError:
                        print("OpenCV不可用，跳过保存")
                break
            else:
                print(f"第{i}次尝试，未获取到图像")

            rospy.sleep(0.1)

    except KeyboardInterrupt:
        print("用户中断")
    finally:
        cam.cleanup()
