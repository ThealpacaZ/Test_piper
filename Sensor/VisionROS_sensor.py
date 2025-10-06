import numpy as np
import os
# 一行解决：强制 OpenCV 使用 X11 后端
os.environ['QT_QPA_PLATFORM'] = 'xcb'
import time
from Sensor.vision_sensor import VisionSensor
from copy import copy
import cv2
from cv_bridge import CvBridge
from Utils.data_handler import debug_print
import rclpy
from Utils.ros2_subscriber import ROS2Subscriber 

from sensor_msgs.msg import Image

class VisionROSensor(VisionSensor):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.cv_bridge = CvBridge()
    
    def set_up(self, topic, depth_topic=None):
        self.controller = {}   
        try:
            self.controller["color"] = ROS2Subscriber(f'{self.name}',topic, Image)
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to initialize camera: {str(e)}")

    def get_image(self):
        image = {}
        rgb_frame = self.controller["color"].get_latest_data()
        if rgb_frame is not None:
            rgb_frame = self.cv_bridge.imgmsg_to_cv2(rgb_frame, desired_encoding="bgr8")
            image["color"] = rgb_frame.copy()
        else:
            debug_print(self.name, "No image data!", "WARNING")
            image["color"]=None
        
        return image

    def cleanup(self):
        try:
            if hasattr(self, 'pipeline'):
                self.pipeline.stop()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    def __del__(self):
        self.cleanup()

if __name__ == "__main__":
   
    cam = VisionROSensor("test1")
    cam2=VisionROSensor("test2")
    cam3=VisionROSensor("test3")
    cam4=VisionROSensor("test4")
    cam.set_up("/camera/left_camera/color/image_raw")
    cam2.set_up("/camera/right_camera/color/image_raw")
    cam3.set_up("/high_camera_node/color/image_raw")
    cam4.set_up("low_camera_node/color/image_raw")
    for i in range(1000):
        if i%100==0:
         print(f"Iteration {i}")

        # 获取图像
        left_image = cam.get_image()
        right_image = cam2.get_image()
        high_image = cam3.get_image()
        low_image=cam4.get_image()

        # 打印调试信息
        if left_image['color'] is not None:
            cv2.imshow("Left Camera", left_image["color"])
        if right_image['color'] is not None:
            cv2.imshow("Right Camera", right_image["color"])
        if high_image['color'] is not None:
            cv2.imshow("High Camera", high_image["color"])
        if low_image['color'] is not None:
             cv2.imshow("Low Camera", low_image["color"])
        #if high_image is not None:
         #   debug_print(cam3.name, f"Image shape: {high_image.shape}, Average pixel value: {np.mean(high_image)}", "INFO")
          #  cv2.imshow("High Camera", high_image)

        cv2.waitKey(1)  # 更新所有OpenCV窗口

        # 延迟以减慢速度
        time.sleep(0.01)  # 每次循环延迟0.5秒

    cv2.destroyAllWindows()
