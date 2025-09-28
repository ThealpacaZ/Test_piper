import socket
import numpy as np
import time
from threading import Thread
import pickle  # 用于复杂数据结构的序列化/反序列化


class RobotServer:
    def __init__(self, host='127.0.0.1', port=10000):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.client_socket = None
        self.running = False
        print(f"Server started on {host}:{port}, waiting for connection...")

    def handle_client(self):
        """处理客户端连接，接收复合数据结构并发送控制指令"""
        while self.running:
            try:
                # 先接收数据长度（4字节，使用大端字节序）
                self.client_socket.setblocking(True)
                self.client_socket.settimeout(None)
                length_data = self.client_socket.recv(4)
                if not length_data:
                    break
                data_length = int.from_bytes(length_data, byteorder='big')

                # 根据长度接收完整数据
                data = b''
                while len(data) < data_length:
                    chunk = self.client_socket.recv(min(4096, data_length - len(data)))
                    if not chunk:
                        break
                    data += chunk

                print("\n===== Received Data =====")
                print(f"Total received data length: {len(data)} bytes")

                # 反序列化数据
                try:
                    data_recv = pickle.loads(data)

                    # 验证数据结构
                    if isinstance(data_recv, dict) and "img_arr" in data_recv and "state" in data_recv:
                        print("\n----- Parsed Data Structure -----")

                        # 处理img_arr（元组内的np数组）
                        print("\nimg_arr (tuple of numpy arrays):")
                        if isinstance(data_recv["img_arr"], tuple):
                            for i, arr in enumerate(data_recv["img_arr"]):
                                if isinstance(arr, np.ndarray):
                                    print(f"  Element {i}:")
                                    print(f"    Shape: {arr.shape}")
                                    print(f"    Dtype: {arr.dtype}")
                                    print(f"    First 5 elements: {arr.flatten()[:5]}...")  # 只显示前5个元素
                                else:
                                    print(f"  Element {i}: Not a numpy array (type: {type(arr)})")
                        else:
                            print("  img_arr is not a tuple")

                        # 处理state（一维np数组）
                        print("\nstate (1D numpy array):")
                        if isinstance(data_recv["state"], np.ndarray) and data_recv["state"].ndim == 1:
                            print(f"  Shape: {data_recv['state'].shape}")
                            print(f"  Dtype: {data_recv['state'].dtype}")
                            print(f"  Values: {data_recv['state']}")
                        else:
                            print("  state is not a 1D numpy array")
                    else:
                        print("Received data does not contain 'img_arr' and 'state' keys")

                except Exception as e:
                    print(f"\nError parsing data: {e}")
                    print("\n----- Binary Data (Hex) -----")
                    print(data.hex()[:200] + "...")  # 只显示前200个十六进制字符
                    print("-----------------------------")

                # 生成动作块，包含多个时间步的动作
                action_chunk = []
                for i in range(30):  # 根据piper_on_PI0.py，action_chunk通常包含30个动作
                    t = time.time() + i * (1 / 30)  # 每个动作时间步增加1/30秒
                    action = np.array([
                        0.5 * np.sin(t),  # joint1
                        0.3 * np.sin(t + 1),  # joint2
                        0.4 * np.sin(t + 2),  # joint3
                        0.2 * np.sin(t + 3),  # joint4
                        0.3 * np.sin(t + 4),  # joint5
                        0.2 * np.sin(t + 5),  # joint6
                        0.05 * (1 + np.sin(t))  # 夹爪
                    ], dtype=np.float32)
                    action_chunk.append(action)

                # 将动作块转换为numpy数组
                action_chunk = np.array(action_chunk)

                # 序列化并发送动作指令
                action_data = pickle.dumps({"action_chunk": action_chunk})
                # 先发送数据长度（4字节）
                self.client_socket.sendall(len(action_data).to_bytes(4, byteorder='big'))
                # 再发送数据
                self.client_socket.sendall(action_data)
                print(f"\nSent action chunk with shape: {action_chunk.shape}")

                time.sleep(0.1)

            except Exception as e:
                print(f"Error handling client: {e}")
                break

        self.client_socket.close()
        print("Client connection closed")

    def start(self):
        """启动服务器"""
        self.running = True
        self.client_socket, addr = self.server_socket.accept()
        print(f"Connected by {addr}")

        client_thread = Thread(target=self.handle_client)
        client_thread.start()

        try:
            while input("Enter 'q' to stop server: ") != 'q':
                time.sleep(1)
        except KeyboardInterrupt:
            pass

        self.running = False
        self.server_socket.close()
        print("Server stopped")


if __name__ == "__main__":
    server = RobotServer()
    server.start()