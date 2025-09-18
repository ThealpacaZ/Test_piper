import sys
sys.path.append('./')


from my_robot.agilex_piper_single_base  import PiperSingle  # 改用Piper机械臂类
import math
import socket
import time
import numpy as np

from utils.bisocket import BiSocket
from utils.data_handler import debug_print, is_enter_pressed

# 关节角度限制（弧度），参考piper_on_PI0.py
joint_limits_rad = [
    (math.radians(-150), math.radians(150)),   # joint1
    (math.radians(0), math.radians(180)),    # joint2
    (math.radians(-170), math.radians(0)),   # joint3
    (math.radians(-100), math.radians(100)),   # joint4
    (math.radians(-70), math.radians(70)),   # joint5
    (math.radians(-120), math.radians(120))    # joint6
]
gripper_limit = [(0.00, 0.07)]  # 夹爪范围

def input_transform(data):
    """适配单臂数据：仅使用右臂（或左臂）数据，忽略另一臂"""
    # 单臂状态：6关节 + 1夹爪
    state = np.concatenate([
        np.array(data[0]["left_arm"]["joint"]).reshape(-1),  # 这个右臂名字可能也有点问题
        np.array(data[0]["left_arm"]["gripper"]).reshape(-1)
    ])
    
    # 图像数据：仅保留手腕相机（参考piper_on_PI0的相机配置）
    img_arr = (data[1]["cam_right_wrist"]["color"],)  
    return img_arr, state

def output_transform(data):
    """处理单臂动作指令，添加限位保护"""
    def clamp(value, min_val, max_val):
        """限制值在安全范围内"""
        return max(min_val, min(value, max_val))
    
    # 处理关节角度（前6位）
    arm_joints = [
        clamp(data[i], joint_limits_rad[i][0], joint_limits_rad[i][1])
        for i in range(6)
    ]
    
    # 处理夹爪（第7位）
    arm_gripper = clamp(data[6], gripper_limit[0][0], gripper_limit[0][1])
    
    # 单臂输出结构
    move_data = {
        "arm": {
            "left_arm": { #111111111111111111
                "joint": arm_joints,
                "gripper": arm_gripper
            }
        }
    }
    return move_data

class PiperClient:
    def __init__(self, robot, control_freq=30):  # 控制频率参考piper_on_PI0的30Hz
        self.robot = robot
        self.control_freq = control_freq
    
    def set_up(self, bisocket: BiSocket):
        self.bisocket = bisocket

    def move(self, message):
        """执行服务器发送的动作指令，逐条控制机械臂"""
        action_chunk = np.array(message["action_chunk"])
        # 限制每次执行的动作数量（参考piper_on_PI0的action_chunk[:30]）
        action_chunk = action_chunk[:30]
        
        for action in action_chunk:
            move_data = output_transform(action)
            self.robot.move(move_data)
            time.sleep(1 / self.control_freq)  # 按频率控制执行间隔

    def play_once(self):
        """采集单臂状态和图像，发送到服务器"""
        raw_data = self.robot.get()  # 获取Piper机械臂的原始数据
        img_arr, state = input_transform(raw_data)
      
        data_send = {
            "img_arr": img_arr,
            "state": state
        }
        self.bisocket.send(data_send)
        time.sleep(1 / self.control_freq)

    def close(self):
        """关闭时重置机械臂"""
        self.robot.reset()
        debug_print("client", "Piper client closed", "INFO")

if __name__ == "__main__":
    import os
    os.environ["INFO_LEVEL"] = "DEBUG"
    
    # 服务器连接配置（根据实际服务器IP修改）
    ip = "127.0.0.1"
    port = 10000

    # 初始化Piper机械臂（参考piper_on_PI0的实例化方式）
    robot = PiperSingle()
    robot.set_up()  # 初始化机械臂连接（如CAN总线）
    
    # 等待用户按Enter开始（参考piper_on_PI0的交互逻辑）
    is_start = False
    while not is_start:
        if is_enter_pressed():
            is_start = True
            print("start client...")
        else:
            print("waiting for start command (press Enter)...")
            time.sleep(1)

    # 创建客户端并连接服务器
    client = PiperClient(robot, control_freq=30)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((ip, port))
    bisocket = BiSocket(client_socket, client.move)
    client.set_up(bisocket)

    # 主循环（执行1000步，参考piper_on_PI0的max_step）
    max_step = 1000
    step = 0
    try:
        while step < max_step and is_start:
            if is_enter_pressed():  # 支持按Enter中断
                debug_print("main", "Interrupted by user", "WARNING")
                break
            client.play_once()
            step += 1
            if step % 10 == 0:
                debug_print("main", f"Step {step}/{max_step}", "INFO")
    except Exception as e:
        debug_print("main", f"Error: {e}", "ERROR")
    finally:
        client.close()
        client_socket.close()