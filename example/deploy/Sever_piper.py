import sys

sys.path.append('./')

import socket
import time
import numpy as np
from utils.bisocket import BiSocket
from policy.openpi.inference_model import PI0_SINGLE
from utils.data_handler import debug_print


class Server:
    def __init__(self, model, control_freq=10):
        self.control_freq = control_freq
        self.model = model
        self.inference_count = 0

    def set_up(self, bisocket: BiSocket):
        self.bisocket = bisocket
        # 调用PI0模型的重置方法
        try:
            self.model.reset_obsrvationwindows()
            debug_print("Server", "模型观察窗口重置完成", "INFO")
        except Exception as e:
            debug_print("Server", f"模型重置失败: {e}", "ERROR")

    def infer(self, message):
        self.inference_count += 1
        debug_print("Server", f"开始第 {self.inference_count} 次推理", "INFO")

        try:
            # 检查数据格式
            debug_print("Server", f"收到数据字段: {list(message.keys())}", "DEBUG")

            # 处理observation格式的数据
            if "observation" in message:
                observation = message["observation"]

                if "image" not in observation or "state" not in observation:
                    raise KeyError(f"observation缺少必要字段。现有字段: {list(observation.keys())}")

                img_data = observation["image"]
                state_data = observation["state"]

                debug_print("Server", "使用observation格式", "DEBUG")

            else:
                raise KeyError(f"数据中缺少observation字段。可用字段: {list(message.keys())}")

            # 验证数据
            self._validate_data(img_data, state_data)

            debug_print("Server", "开始模型推理...", "DEBUG")

            # 调用模型进行推理
            self.model.update_observation_window(img_data, state_data)
            debug_print("Server", "观察窗口更新成功", "DEBUG")

            # 获取动作序列
            action_chunk = self.model.get_action()
            debug_print("Server", f"获取动作序列成功，类型: {type(action_chunk)}", "DEBUG")

            if action_chunk is not None:
                if hasattr(action_chunk, 'shape'):
                    debug_print("Server", f"动作序列形状: {action_chunk.shape}", "DEBUG")
                elif hasattr(action_chunk, '__len__'):
                    debug_print("Server", f"动作序列长度: {len(action_chunk)}", "DEBUG")

                return {"action_chunk": action_chunk}
            else:
                debug_print("Server", "模型返回空动作序列", "WARNING")
                return {"action_chunk": [], "warning": "空动作序列"}

        except KeyError as e:
            error_msg = f"数据格式错误: {e}"
            debug_print("Server", error_msg, "ERROR")
            debug_print("Server", f"可用字段: {list(message.keys())}", "ERROR")
            if "observation" in message:
                debug_print("Server", f"observation中的字段: {list(message['observation'].keys())}", "ERROR")
            return {"error": error_msg}

        except Exception as e:
            error_msg = f"推理过程出错: {e}"
            debug_print("Server", error_msg, "ERROR")
            import traceback
            debug_print("Server", f"错误堆栈:\n{traceback.format_exc()}", "ERROR")
            return {"error": error_msg}

    def _validate_data(self, img_data, state_data):
        """验证输入数据的有效性"""
        # 验证图像数据
        if img_data is None:
            raise ValueError("图像数据为空")
        if not hasattr(img_data, 'shape'):
            raise ValueError(f"图像数据无shape属性，类型: {type(img_data)}")
        if len(img_data.shape) != 3:
            raise ValueError(f"图像数据应为3维，实际: {img_data.shape}")

        debug_print("Server", f"图像验证通过 - 形状: {img_data.shape}, 类型: {img_data.dtype}", "DEBUG")
        debug_print("Server", f"图像数值范围: [{np.min(img_data)}, {np.max(img_data)}]", "DEBUG")

        # 验证状态数据
        if state_data is None:
            raise ValueError("状态数据为空")
        if not hasattr(state_data, 'shape'):
            raise ValueError(f"状态数据无shape属性，类型: {type(state_data)}")
        if len(state_data.shape) != 1:
            raise ValueError(f"状态数据应为1维，实际: {state_data.shape}")
        if state_data.shape[0] != 7:
            raise ValueError(f"状态数据应为7维（6关节+1夹爪），实际: {state_data.shape[0]}")

        debug_print("Server", f"状态验证通过 - 形状: {state_data.shape}, 类型: {state_data.dtype}", "DEBUG")
        debug_print("Server", f"状态数值范围: [{np.min(state_data):.3f}, {np.max(state_data):.3f}]", "DEBUG")

    def close(self):
        """关闭服务器并清理资源"""
        try:
            if hasattr(self.model, 'reset_obsrvationwindows'):
                self.model.reset_obsrvationwindows()
            if hasattr(self, "bisocket"):
                self.bisocket.close()
            debug_print("Server", f"服务器关闭，总共处理了{self.inference_count}次推理", "INFO")
        except Exception as e:
            debug_print("Server", f"关闭服务器时出错: {e}", "ERROR")


if __name__ == "__main__":
    import os

    os.environ["INFO_LEVEL"] = "DEBUG"

    # 服务器配置
    ip = "0.0.0.0"
    port = 10000

    server_socket = None
    server = None
    model = None

    try:
        # 初始化PI0模型
        debug_print("Server", "正在初始化PI0模型...", "INFO")
        model = PI0_SINGLE(
            task_name="libero_goal",
            train_config_name="pi0_fast_libero",
            model_name="pi0_fast_libero",
            checkpoint_id="final"
        )
        debug_print("Server", "PI0模型初始化完成", "INFO")

        server = Server(model, control_freq=10)

        # 创建服务器套接字
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((ip, port))
        server_socket.listen(1)

        print("=" * 60)
        print("PI0单臂推理服务器启动成功")
        print(f"监听地址: {ip}:{port}")
        print("等待客户端连接...")
        print("按 Ctrl+C 停止服务器")
        print("=" * 60)

        debug_print("Server", f"服务器监听 {ip}:{port}", "INFO")

        while True:
            try:
                debug_print("Server", "等待客户端连接...", "INFO")
                conn, addr = server_socket.accept()
                debug_print("Server", f"客户端 {addr} 已连接", "INFO")

                try:
                    # 为每个客户端创建独立的处理线程
                    bisocket = BiSocket(conn, server.infer, send_back=True)
                    server.set_up(bisocket)

                    # 保持连接直到客户端断开
                    connection_start_time = time.time()
                    last_activity = time.time()

                    while bisocket.running.is_set():
                        current_time = time.time()

                        # 检查连接活跃度（可选：实现超时机制）
                        if current_time - last_activity > 300:  # 5分钟无活动则断开
                            debug_print("Server", f"客户端 {addr} 超时，断开连接", "WARNING")
                            break

                        time.sleep(0.5)

                    connection_duration = time.time() - connection_start_time
                    debug_print("Server",
                                f"客户端 {addr} 断开连接，连接时长: {connection_duration:.1f}秒",
                                "INFO")

                except Exception as e:
                    debug_print("Server", f"处理客户端 {addr} 时出错: {e}", "ERROR")
                    import traceback

                    debug_print("Server", f"详细错误:\n{traceback.format_exc()}", "ERROR")

                finally:
                    try:
                        conn.close()
                    except:
                        pass
                    debug_print("Server", f"客户端 {addr} 连接资源已清理", "DEBUG")

            except socket.error as e:
                if server_socket:  # 检查socket是否还存在
                    debug_print("Server", f"Socket错误: {e}", "ERROR")
                else:
                    break  # Socket已关闭，退出循环

    except KeyboardInterrupt:
        debug_print("Server", "服务器被用户中断 (Ctrl+C)", "WARNING")
    except Exception as e:
        debug_print("Server", f"服务器启动或运行失败: {e}", "ERROR")
        import traceback

        debug_print("Server", f"详细错误信息:\n{traceback.format_exc()}", "ERROR")
    finally:
        # 确保所有资源都被正确清理
        print("\n正在关闭服务器...")

        if server:
            try:
                server.close()
            except Exception as e:
                debug_print("Server", f"关闭服务器对象时出错: {e}", "ERROR")

        if server_socket:
            try:
                server_socket.close()
            except Exception as e:
                debug_print("Server", f"关闭服务器套接字时出错: {e}", "ERROR")

        debug_print("Server", "服务器已完全关闭", "INFO")
        print("服务器已关闭")
