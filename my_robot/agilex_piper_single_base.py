import os
# 一行解决：强制 OpenCV 使用 X11 后端
os.environ['QT_QPA_PLATFORM'] = 'xcb'
import sys
sys.path.append("./")
from piper_sdk import *

import numpy as np

from my_robot.base_robot import Robot
from Sensor.VisionROS_sensor import VisionROSensor
from Controller.Piper_controller import PiperController
import cv2

from data.collect_any import CollectAny


# Define start position (in degrees)
START_POSITION_ANGLE_LEFT_ARM = [
    0.0,   # Joint 1
    0.85220935,    # Joint 2
    -0.68542569,  # Joint 3
    0.,   # Joint 4
    0.78588684,  # Joint 5
    -0.05256932,    # Joint 6
]

# Define start position (in degrees)
START_POSITION_ANGLE_RIGHT_ARM = [
    0,   # Joint 1
    0,    # Joint 2
    0,  # Joint 3
    0,   # Joint 4
    0,  # Joint 5
    0,    # Joint 6
]

condition = {
    "robot":"piper_single",
    "save_path": "./datasets/", "/camera/high_camera_node/color/image_raw"
    "task_name": "test", 
    "save_format": "hdf5", 
    "save_freq": 10, 
}


class PiperSingle(Robot):
    def __init__(self, start_episode=0):
        super().__init__(start_episode)

        self.condition = condition
        self.controllers = {
            "arm":{
                "left_arm": PiperController("left_arm"),
                "right_arm":PiperController("right_arm")
            },
        }
        self.sensors={
            'image':{
                'cam_right_wrist':VisionROSensor('cam_right_wrist'),
                'cam_left_wrist':VisionROSensor('cam_left_wrist'),
                'cam_high':VisionROSensor('cam_high'),
                'cam_low':VisionROSensor('cam_low')
            }
        }
        self.collection = CollectAny(condition, start_episode=start_episode)

    # ============== init ==============
    def reset(self):
        self.controllers["arm"]["left_arm"].reset(np.array(START_POSITION_ANGLE_RIGHT_ARM))

    def set_up(self):
        self.controllers["arm"]["left_arm"].set_up("can0")
        self.controllers["arm"]["right_arm"].set_up("can1")
        self.sensors['image']['cam_left_wrist'].set_up("/camera/left_camera/color/image_raw")
        self.sensors['image']['cam_right_wrist'].set_up("/camera/right_camera/color/image_raw")
        self.sensors['image']['cam_high'].set_up("/high_camera_node/color/image_raw")
        self.sensors['image']['cam_low'].set_up("/low_camera_node/color/image_raw")
        self.set_collect_type({"arm": ["joint","gripper"],
                               "image": ["color"]
                               })
        
        print("set up success!")
    
if __name__=="__main__":
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
    piper.MotionCtrl_1(0x02,0,0)#恢复
    piper.MotionCtrl_2(0, 0, 0, 0x00)#位置速度模式
    