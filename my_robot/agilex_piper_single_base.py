#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

sys.path.append("./")

import time
import numpy as np
import cv2
import rospy


import socket
import time
from utils.bisocket import BiSocket
from utils.data_handler import debug_print, is_enter_pressed
from my_robot.base_robot import Robot
from controller.Piper_controller import PiperController
from data.collect_any import CollectAny

# 传感器：修改为使用 VisionROSensor
from sensor.VisionROS_sensor import VisionROSensor

# Define start position (in degrees)
START_POSITION_ANGLE_LEFT_ARM = [
    0.0,  # Joint 1
    0.85220935,  # Joint 2
    -0.68542569,  # Joint 3
    0.0,  # Joint 4
    0.78588684,  # Joint 5
    -0.05256932,  # Joint 6
]

START_POSITION_ANGLE_RIGHT_ARM = [0, 0, 0, 0, 0, 0]

condition = {
    "robot": "piper_single",
    "save_path": "./datasets/",
    "task_name": "test",
    "save_format": "hdf5",
    "save_freq": 10,
}


class PiperSingle(Robot):
    def __init__(self, start_episode=0):
        super().__init__(start_episode)

        # 先初始化 ROS 节点（如果还未初始化）
        try:
            if not rospy.get_node_uri():
                rospy.init_node('piper_single_node', anonymous=True)
        except:
            rospy.init_node('piper_single_node', anonymous=True)

        self.condition = condition
        self.controllers = {
            "arm": {
                "left_arm": PiperController("left_arm")
            },
        }

        # === 集成 VisionROSensor 相机 ===
        self.sensors = {
            "image": {
                "orbbec_camera": VisionROSensor("orbbec_camera")
            }
        }

        self.collection = CollectAny(self.condition, start_episode=start_episode)
        # 添加客户端相关属性
        self.client = None
        self.bisocket = None
        self.control_freq = 30  # 控制频率
        self.server_ip = "127.0.0.1"  # 服务器IP
        self.server_port = 10000  # 服务器端口

    # ============== init/robot ==============
    def reset(self):
        self.controllers["arm"]["left_arm"].reset(np.array(START_POSITION_ANGLE_LEFT_ARM))

    def set_up(self):
        # 设置控制器
        self.controllers["arm"]["left_arm"].set_up("can0")

        # 设置相机传感器
        camera_sensor = self.sensors["image"]["orbbec_camera"]

        # 检查话题是否存在并寻找正确的话题
        print("正在检查可用话题...")
        color_topic = None
        depth_topic = None

        try:
            topics = rospy.get_published_topics()
            available_topics = [topic[0] for topic in topics]

            print(f"找到 {len(available_topics)} 个话题")

            # 寻找彩色图像话题
            color_candidates = [
                "/camera/color/image_raw",
                "/camera/rgb/image_raw",
                "/rgb/image_raw",
                "/orbbec/color/image_raw",
                "/camera/image_raw"
            ]

            for candidate in color_candidates:
                if candidate in available_topics:
                    color_topic = candidate
                    print(f"找到彩色话题: {color_topic}")
                    break

            # 如果没找到预定义的话题，搜索所有包含 'image' 和 'color' 或 'rgb' 的话题
            if not color_topic:
                for topic in available_topics:
                    if 'image' in topic.lower() and ('color' in topic.lower() or 'rgb' in topic.lower()):
                        color_topic = topic
                        print(f"自动发现彩色话题: {color_topic}")
                        break

            # 寻找深度图像话题
            depth_candidates = [
                "/camera/depth/image_raw",
                "/camera/depth/image_rect_raw",
                "/depth/image_raw",
                "/orbbec/depth/image_raw"
            ]

            for candidate in depth_candidates:
                if candidate in available_topics:
                    depth_topic = candidate
                    print(f"找到深度话题: {depth_topic}")
                    break

            # 如果没有找到任何图像话题
            if not color_topic:
                print("未找到可用的彩色图像话题!")
                print("可用的话题列表:")
                image_topics = [t for t in available_topics if 'image' in t.lower()]
                if image_topics:
                    for topic in image_topics[:10]:  # 只显示前10个
                        print(f"  - {topic}")
                    if len(image_topics) > 10:
                        print(f"  ... 还有 {len(image_topics) - 10} 个话题")
                else:
                    print("  没有找到包含 'image' 的话题")

                # 手动指定一个话题用于测试
                color_topic = "/camera/color/image_raw"  # 仍然使用默认话题，但会在后面检查连接

        except Exception as e:
            print(f"检查话题时出错: {e}")
            color_topic = "/camera/color/image_raw"

        # 设置相机传感器
        try:
            print(f"尝试设置相机传感器...")
            print(f"彩色话题: {color_topic}")
            if depth_topic:
                print(f"深度话题: {depth_topic}")
            else:
                print("不使用深度话题")

            camera_sensor.set_up(color_topic, depth_topic)
            print("相机传感器设置完成")

            # 等待并检查连接
            print("等待话题连接...")
            for i in range(50):  # 等待5秒
                if hasattr(camera_sensor.controller["color"], 'is_connected') and camera_sensor.controller[
                    "color"].is_connected():
                    print(f"成功连接到彩色话题 {color_topic}")
                    break
                time.sleep(0.1)
            else:
                print(f"警告: 无法连接到彩色话题 {color_topic}")
                print("请检查:")
                print("1. 相机驱动是否启动 (如: roslaunch orbbec_camera XXX.launch)")
                print("2. 话题名称是否正确")
                print("3. 使用 'rostopic list' 查看可用话题")
                print("4. 使用 'rostopic hz <topic_name>' 检查话题数据")

        except Exception as e:
            print(f"设置相机传感器失败: {e}")
            raise

        # 设置收集信息
        self.set_collect_type({
            "arm": ["joint", "qpos", "gripper"],
            "image": ["color"],  # 如果需要深度图，改为 ["color", "depth"]
        })

        print("set up success!")

    def get_images_for_collect(self):
        """
        获取图像数据并确保数据类型兼容HDF5
        返回结构示例：
          {
            "orbbec_camera": {
                "color": array([[[255, 0, 0], ...]], dtype=uint8),
                "depth": array([[[1000], ...]], dtype=uint16)
            }
          }
        """
        images_all = {}
        image_sensors = self.sensors.get("image", {})

        for sensor_name, sensor in image_sensors.items():
            try:
                # 获取图像数据
                img_data = sensor.get_image()

                # 转换为期望的格式
                sensor_images = {}

                for channel, img_array in img_data.items():
                    if img_array is not None:
                        # 确保数据类型兼容HDF5
                        if isinstance(img_array, np.ndarray):
                            # 检查并修正数据类型
                            if img_array.dtype == np.object_:
                                print(f"警告: {sensor_name}_{channel} 数据类型为object，跳过")
                                continue

                            # 确保是标准的numpy类型
                            if img_array.dtype not in [np.uint8, np.uint16, np.int16, np.int32, np.float32, np.float64]:
                                print(f"转换 {sensor_name}_{channel} 数据类型从 {img_array.dtype} 到 uint8")
                                img_array = img_array.astype(np.uint8)

                            # 确保是连续内存布局
                            if not img_array.flags['C_CONTIGUOUS']:
                                img_array = np.ascontiguousarray(img_array)

                            # 直接存储图像数组，不包含时间戳
                            sensor_images[channel] = img_array
                        else:
                            print(f"警告: {sensor_name}_{channel} 不是numpy数组类型: {type(img_array)}")
                            continue

                images_all[sensor_name] = sensor_images

            except Exception as e:
                print(f"获取传感器 {sensor_name} 数据失败: {e}")
                images_all[sensor_name] = {}

        return images_all

    def close_sensors(self):
        """关闭所有传感器"""
        image_sensors = self.sensors.get("image", {})
        for sensor in image_sensors.values():
            try:
                sensor.cleanup()
            except Exception as e:
                print(f"关闭传感器时出错: {e}")





if __name__ == "__main__":
    # ====== 运行前检查 ======
    print("====== 运行前检查 ======")
    print("请确保:")
    print("1. roscore 已启动")
    print("2. 相机驱动已启动 (如: roslaunch orbbec_camera gemini_330_series.launch)")
    print("3. ROS环境变量正确设置")
    print("========================")

    robot = PiperSingle()
    robot.set_up()

    # collection test
    robot.reset()

    printed_once = False  # 收到第一帧时打印一次键名
    saved_once = False  # 无 GUI 时保存一次快照
    window_ready = False  # 懒创建预览窗口
    last_image_time = 0  # 记录上次收到图像的时间

    print("开始数据采集...")

    for i in range(100):
        print(f"[Collect] step {i}")

        try:
            # ---- 兼容两种 get() 返回形态：list 或 dict ----
            got = robot.get()

            if isinstance(got, dict):
                # 新接口：{"controller": {...}, "sensor": {...}}
                controller_data = got.get("controller", {})
                sensor_data = got.get("sensor", {})
            elif isinstance(got, (list, tuple)) and len(got) >= 2:
                # 旧接口：[controller_data, sensor_data]
                controller_data, sensor_data = got[0], got[1]
            else:
                print(f"警告: robot.get() 返回了不支持的类型：{type(got)}")
                controller_data, sensor_data = {}, {}

            # 确保 sensors_data 是字典格式
            if not isinstance(sensor_data, dict):
                print(f"警告: sensors_data 不是字典类型，已转换为字典，实际类型为: {type(sensor_data)}")
                sensor_data = {f"sensor_{i}": data for i, data in enumerate(sensor_data)}

            # 图像数据（函数式获取）
            img_pack = robot.get_images_for_collect()

            # === 扁平化保存到顶层 ===
            if not isinstance(sensor_data, dict):
                sensor_data = {}

            preview_frame = None
            preview_name = None
            has_image_data = False

            for cam_name, chan_dict in img_pack.items():
                if not isinstance(chan_dict, dict):
                    continue
                for chan, payload in chan_dict.items():
                    if isinstance(payload, dict) and "data" in payload:
                        key = f"{cam_name}_{chan}"

                        # 确保数据类型兼容HDF5
                        data = payload["data"]
                        if isinstance(data, np.ndarray) and data.dtype != np.object_:
                            sensor_data[key] = data
                            sensor_data[f"{key}_stamp"] = np.array(
                                payload.get("stamp", 0.0), dtype=np.float64
                            )

                            has_image_data = True
                            last_image_time = time.time()

                            # 预览帧（优先 color）
                            if preview_frame is None or chan == "color":
                                preview_frame = data
                                preview_name = key
                        else:
                            print(
                                f"跳过不兼容的数据类型: {key}, 类型: {type(data)}, dtype: {getattr(data, 'dtype', 'N/A')}"
                            )

            # 首次拿到图像，打印信息
            if not printed_once and has_image_data:
                flat_keys = [k for k in sensor_data.keys() if "orbbec_camera" in k and not k.endswith("_stamp")]
                if flat_keys:
                    print(f"[DEBUG] 成功获取图像数据! 键名: {flat_keys}")
                    if preview_frame is not None:
                        print(f"[DEBUG] 图像信息: 形状={preview_frame.shape}, 类型={preview_frame.dtype}")
                    printed_once = True

            # ——持续预览：每帧尝试显示——
            if preview_frame is not None:
                try:
                    if not window_ready:
                        try:
                            cv2.namedWindow("Preview - Camera Frame", cv2.WINDOW_AUTOSIZE)
                            window_ready = True
                        except Exception:
                            pass

                    # 确保图像格式正确用于显示
                    if len(preview_frame.shape) == 3 and preview_frame.shape[2] == 3:
                        # RGB 转 BGR 用于 OpenCV 显示
                        display_frame = cv2.cvtColor(preview_frame, cv2.COLOR_RGB2BGR)
                    else:
                        display_frame = preview_frame

                    cv2.imshow("Preview - Camera Frame", display_frame)
                    cv2.waitKey(1)  # 刷新 UI

                except Exception as e:
                    if not saved_once:
                        try:
                            cv2.imwrite("preview.png", preview_frame)
                            print(f"[INFO] 无法显示窗口，已保存预览到 ./preview.png")
                            saved_once = True
                        except Exception as save_e:
                            print(f"保存预览图片也失败: {save_e}")
            else:
                # 如果长时间没收到图像，给出提示
                current_time = time.time()
                if current_time - last_image_time > 10 and i % 20 == 0:  # 每20次循环检查一次，且超过10秒没图像
                    print(f"[WARNING] 已经 {current_time - last_image_time:.1f} 秒没收到图像数据")
                    print("请检查:")
                    print("- 运行 'rostopic hz /camera/color/image_raw' 确认有数据")
                    print("- 运行 'rostopic list | grep image' 查看可用话题")

            # ---- 传回数据进行收集 ----
            robot.collect([controller_data, sensor_data])

        except KeyboardInterrupt:
            print("用户中断程序")
            break
        except Exception as e:
            print(f"步骤 {i} 出错: {e}")
            # 继续执行，不中断整个流程

        time.sleep(0.1)

    try:
        robot.finish()
        print("数据收集完成")
    except Exception as e:
        print(f"完成收集时出错: {e}")

    # moving test
    print("开始移动测试...")
    try:
        move_data = {
            "arm": {
                "left_arm": {
                    "qpos": [0.057, 0.0, 0.216, 0.0, 0.085, 0.0],
                    "gripper": 0.2,
                },
            },
        }
        robot.move(move_data)
        time.sleep(1)

        move_data = {
            "arm": {
                "left_arm": {
                    "joint": [0.00, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "gripper": 0.2,
                },
            },
        }
        robot.move(move_data)
        print("移动测试完成")

    except Exception as e:
        print(f"移动测试出错: {e}")

    # 关闭窗口 + 传感器
    print("清理资源...")
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    try:
        robot.close_sensors()
    except Exception as e:
        print(f"关闭传感器出错: {e}")

    print("程序结束")

