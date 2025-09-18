import sys
sys.path.append('./')

import socket
import time

from utils.bisocket import BiSocket

from policy.openpi.inference_model import PI0_SINGLE
from utils.data_handler import debug_print

class Server:
    def __init__(self, model, control_freq=10):
        self.control_freq = control_freq
        self.model = model

    def set_up(self, bisocket: BiSocket):
        self.bisocket = bisocket
        # 调用 PI0 模型的重置方法（与原方法名称一致，无需修改）
        self.model.reset_obsrvationwindows()

    def infer(self, message):
        debug_print("Server", "Inference triggered.", "INFO")

        # 接收客户端发送的图像和状态数据
        img_arr, state = message["img_arr"], message["state"]
        # 更新 PI0 模型的观测窗口
        self.model.update_observation_window(img_arr, state)
        # 获取 PI0 模型预测的动作序列
        action_chunk = self.model.get_action()
        return {"action_chunk": action_chunk}

    def close(self):
        if hasattr(self, "bisocket"):
            self.bisocket.close()


if __name__ == "__main__":
    import os
    os.environ["INFO_LEVEL"] = "DEBUG"

    ip = "127.0.0.1"
    port = 10000

    # 初始化 PI0 模型（替换 TestModel）
    # 注意：需要填写实际的模型路径和任务名称
    model = PI0_SINGLE(
        model_path="path/to/pi0_model",  # PI0 模型权重路径
        task_name="your_task_name"       # 任务名称（对应指令文件）
    )

    server = Server(model)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((ip, port))
    server_socket.listen(1)

    debug_print("Server", f"Listening on {ip}:{port}", "INFO")

    try:
        while True:
            debug_print("Server", "Waiting for client connection...", "INFO")
            conn, addr = server_socket.accept()
            debug_print("Server", f"Connected by {addr}", "INFO")

            bisocket = BiSocket(conn, server.infer, send_back=True)
            server.set_up(bisocket)

            while bisocket.running.is_set():
                time.sleep(0.5)

            debug_print("Server", "Client disconnected. Waiting for next client...", "WARNING")

    except KeyboardInterrupt:
        debug_print("Server", "Shutting down.", "WARNING")
    finally:
        server_socket.close()