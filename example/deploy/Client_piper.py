import sys

sys.path.append('./')

from my_robot.agilex_piper_single_base import PiperSingle
import math
import socket
import time
import numpy as np

from utils.bisocket import BiSocket
from utils.data_handler import debug_print, is_enter_pressed

# 关节角度限制（弧度）
joint_limits_rad = [
    (math.radians(-150), math.radians(150)),  # joint1
    (math.radians(0), math.radians(180)),  # joint2
    (math.radians(-170), math.radians(0)),  # joint3
    (math.radians(-100), math.radians(100)),  # joint4
    (math.radians(-70), math.radians(70)),  # joint5
    (math.radians(-120), math.radians(120))  # joint6
]
gripper_limit = [(0.00, 0.07)]  # 夹爪范围


def input_transform(data):
    """适配单臂数据，确保数据格式完全符合PI0模型期望"""
    try:
        # 获取关节状态和夹爪状态
        joint_state = np.array(data["controller"]["left_arm"]["joint"]).reshape(-1)
        gripper_state = np.array(data["controller"]["left_arm"]["gripper"]).reshape(-1)

        # 验证关节状态维度
        if joint_state.shape[0] != 6:
            raise ValueError(f"期望6个关节，实际得到{joint_state.shape[0]}个")
        if gripper_state.shape[0] != 1:
            raise ValueError(f"期望1个夹爪值，实际得到{gripper_state.shape[0]}个")

        # 合并状态：6关节 + 1夹爪 = 7维
        combined_state = np.concatenate([joint_state, gripper_state])

        # 获取图像数据
        img_data = data["sensor"]["orbbec_camera"]["color"]

        # 严格验证图像数据格式
        if not isinstance(img_data, np.ndarray):
            debug_print("Client", "图像不是numpy数组，正在转换", "WARNING")
            img_data = np.array(img_data)

        if len(img_data.shape) != 3:
            raise ValueError(f"图像应该是3维数组(H,W,C)，实际形状: {img_data.shape}")

        if img_data.shape[2] != 3:
            raise ValueError(f"图像应该有3个通道(RGB)，实际通道数: {img_data.shape[2]}")

        # 确保图像数据类型正确
        if img_data.dtype != np.uint8:
            debug_print("Client", f"图像类型从{img_data.dtype}转换为uint8", "DEBUG")
            img_data = img_data.astype(np.uint8)

        # 验证像素值范围
        if np.min(img_data) < 0 or np.max(img_data) > 255:
            debug_print("Client", f"图像像素值超出范围[0,255]: [{np.min(img_data)}, {np.max(img_data)}]", "WARNING")
            img_data = np.clip(img_data, 0, 255).astype(np.uint8)

        debug_print("Client", f"输入验证通过 - 关节: {joint_state.shape}, 夹爪: {gripper_state.shape}", "DEBUG")
        debug_print("Client", f"输入验证通过 - 图像: {img_data.shape}, 类型: {img_data.dtype}", "DEBUG")
        debug_print("Client", f"状态数值范围: [{np.min(combined_state):.3f}, {np.max(combined_state):.3f}]", "DEBUG")

        return img_data, combined_state

    except Exception as e:
        debug_print("Client", f"输入转换失败: {e}", "ERROR")
        raise


def output_transform(data):
    """处理单臂动作指令，添加限位保护"""

    def clamp(value, min_val, max_val):
        """限制值在安全范围内"""
        return max(min_val, min(value, max_val))

    try:
        # 验证输入数据
        if not isinstance(data, (np.ndarray, list, tuple)):
            raise ValueError(f"动作数据应为数组或列表，实际类型: {type(data)}")

        if len(data) < 7:
            raise ValueError(f"动作数据应至少包含7个值(6关节+1夹爪)，实际长度: {len(data)}")

        # 处理关节角度（前6位）
        arm_joints = [
            clamp(data[i], joint_limits_rad[i][0], joint_limits_rad[i][1])
            for i in range(6)
        ]

        # 处理夹爪（第7位）
        arm_gripper = clamp(data[6], gripper_limit[0][0], gripper_limit[0][1])

        # 检查是否有值被截断
        for i in range(6):
            if abs(data[i] - arm_joints[i]) > 1e-6:
                debug_print("Client", f"关节{i + 1}被限制: {data[i]:.3f} -> {arm_joints[i]:.3f}", "WARNING")

        if abs(data[6] - arm_gripper) > 1e-6:
            debug_print("Client", f"夹爪被限制: {data[6]:.3f} -> {arm_gripper:.3f}", "WARNING")

        # 单臂输出结构
        move_data = {
            "arm": {
                "left_arm": {
                    "joint": arm_joints,
                    "gripper": arm_gripper
                }
            }
        }
        return move_data

    except Exception as e:
        debug_print("Client", f"输出转换失败: {e}", "ERROR")
        raise


class PiperClient:
    def __init__(self, robot, control_freq=30):
        self.robot = robot
        self.control_freq = control_freq
        self.action_count = 0

    def set_up(self, bisocket: BiSocket):
        self.bisocket = bisocket

    def move(self, message):
        """执行服务器发送的动作指令"""
        try:
            if "error" in message:
                debug_print("Client", f"收到服务器错误: {message['error']}", "ERROR")
                return

            if "action_chunk" not in message:
                debug_print("Client", "服务器响应中缺少action_chunk", "ERROR")
                return

            action_chunk = np.array(message["action_chunk"])
            debug_print("Client", f"收到动作序列，形状: {action_chunk.shape}", "DEBUG")

            # 验证动作数据
            if len(action_chunk) == 0:
                debug_print("Client", "收到空的动作序列", "WARNING")
                return

            # 限制每次执行的动作数量
            action_chunk = action_chunk[:30]
            self.action_count += len(action_chunk)

            debug_print("Client", f"开始执行{len(action_chunk)}个动作", "INFO")

            for i, action in enumerate(action_chunk):
                try:
                    move_data = output_transform(action)
                    self.robot.move(move_data)

                    if i % 5 == 0:  # 每5步打印一次
                        debug_print("Client", f"执行动作 {i + 1}/{len(action_chunk)}", "DEBUG")

                    time.sleep(1 / self.control_freq)

                except Exception as e:
                    debug_print("Client", f"执行第{i + 1}个动作失败: {e}", "ERROR")
                    # 继续执行下一个动作，不中断整个序列

            debug_print("Client", f"动作序列执行完成，累计执行{self.action_count}个动作", "INFO")

        except Exception as e:
            debug_print("Client", f"执行动作失败: {e}", "ERROR")

    def play_once(self):
        """采集单臂状态和图像，发送到服务器"""
        try:
            raw_data = self.robot.get()
            img_data, state = input_transform(raw_data)

            # 构造PI0模型期望的observation格式
            data_send = {
                "observation": {
                    "image": img_data,
                    "state": state
                }
            }

            debug_print("Client", f"发送数据 - 图像: {img_data.shape}, 状态: {len(state)}", "DEBUG")

            # 发送数据
            self.bisocket.send(data_send)
            time.sleep(1 / self.control_freq)

        except Exception as e:
            debug_print("Client", f"数据采集发送失败: {e}", "ERROR")
            raise

    def close(self):
        """关闭时重置机械臂"""
        try:
            debug_print("Client", f"关闭客户端，总共执行了{self.action_count}个动作", "INFO")
            self.robot.reset()
            debug_print("Client", "Piper client closed", "INFO")
        except Exception as e:
            debug_print("Client", f"关闭客户端时出错: {e}", "ERROR")


if __name__ == "__main__":
    import os

    os.environ["INFO_LEVEL"] = "DEBUG"

    # 服务器连接配置
    ip = "127.0.0.1"
    port = 10000

    client_socket = None
    client = None
    robot = None

    try:
        # 初始化Piper机械臂
        debug_print("Main", "初始化Piper机械臂...", "INFO")
        robot = PiperSingle()
        robot.set_up()
        debug_print("Main", "机械臂初始化完成", "INFO")

        # 等待用户开始
        is_start = False
        print("=" * 50)
        print("Piper单臂客户端准备就绪")
        print("按Enter键开始连接服务器...")
        print("=" * 50)

        while not is_start:
            if is_enter_pressed():
                is_start = True
                debug_print("Main", "开始客户端程序...", "INFO")
            else:
                time.sleep(0.1)  # 减少CPU使用

        # 创建客户端并连接服务器
        client = PiperClient(robot, control_freq=30)
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(10)  # 设置连接超时

        debug_print("Main", f"连接服务器 {ip}:{port}...", "INFO")
        client_socket.connect((ip, port))
        client_socket.settimeout(None)  # 连接成功后移除超时
        debug_print("Main", "服务器连接成功", "INFO")

        bisocket = BiSocket(client_socket, client.move)
        client.set_up(bisocket)

        # 主循环
        max_step = 100000
        step = 0

        print(f"开始执行，最大步数: {max_step}")
        print("运行期间按Enter键可以中断程序")

        while step < max_step and is_start:
            if is_enter_pressed():
                debug_print("Main", "用户请求中断程序", "WARNING")
                break

            try:
                client.play_once()
                step += 1

                if step % 10 == 0:
                    debug_print("Main", f"执行步数: {step}/{max_step}", "INFO")

            except Exception as e:
                debug_print("Main", f"执行第{step + 1}步时出错: {e}", "ERROR")
                # 继续尝试下一步，而不是立即退出
                step += 1
                time.sleep(1.0)  # 出错后等待1秒再继续

        debug_print("Main", f"程序执行完成，总共执行了{step}步", "INFO")

    except KeyboardInterrupt:
        debug_print("Main", "程序被用户中断 (Ctrl+C)", "WARNING")
    except socket.timeout:
        debug_print("Main", "连接服务器超时", "ERROR")
    except ConnectionRefusedError:
        debug_print("Main", "无法连接到服务器，请确认服务器已启动", "ERROR")
    except Exception as e:
        debug_print("Main", f"程序运行出错: {e}", "ERROR")
        import traceback

        debug_print("Main", f"详细错误信息:\n{traceback.format_exc()}", "ERROR")
    finally:
        # 确保资源正确释放
        if client:
            try:
                client.close()
            except:
                pass
        if client_socket:
            try:
                client_socket.close()
            except:
                pass
        debug_print("Main", "程序结束，资源已清理", "INFO")
