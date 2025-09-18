import sys
sys.path.append("./")

import h5py
import numpy as np
import os
from tqdm import tqdm

from utils.data_handler import hdf5_groups_to_dict, get_files, get_item

import cv2

'''
dual-arm:

map = {
    "cam_high": "cam_head.color",
    "cam_left_wrist": "cam_left_wrist.color",
    "cam_right_wrist": "cam_right_wrist.color",
    "qpos": ["left_arm.joint","left_arm.gripper","right_arm.joint","right_arm.gripper"],
    "action": ["left_arm.joint","left_arm.gripper","right_arm.joint","right_arm.gripper"],
}

single-arm:

map = {
    # "cam_high": "cam_head.color",
    "cam_wrist": "cam_wrist.color",
    "qpos": ["left_arm.joint","left_arm.gripper"],
    "action": ["left_arm.joint","left_arm.gripper"],
}
'''

map = {
    "cam_high": "slave_cam_head.color",
    "cam_wrist": "slave_cam_wrist.color",
    # qpos 目标维度仍为 14（双臂：6+1+6+1），右臂不存在则补 0
    "qpos": ["slave_left_arm.joint","slave_left_arm.gripper","slave_right_arm.joint","slave_right_arm.gripper"],
    "action": ["slave_left_arm.joint","slave_left_arm.gripper","slave_right_arm.joint","slave_right_arm.gripper"],
}

def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode('.jpg', imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b'\0'))
    return encode_data, max_len

def convert(hdf5_paths, output_path, start_index=0):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    index = start_index
    for hdf5_path in hdf5_paths:
        # 读取数据（坏文件直接跳过）
        try:
            data = hdf5_groups_to_dict(hdf5_path)
        except Exception as e:
            print(f"skip {hdf5_path} due to read error: {e}")
            continue
        
        hdf5_output_path = os.path.join(output_path, f"episode_{index}.hdf5")
        index += 1
        print(data.keys())
        with h5py.File(hdf5_output_path, "w") as f:
            # 降采样
            input_data = {}

            # 读取相机数据（不降采样）
            input_data["cam_high"] = get_item(data, map["cam_high"])
            input_data["cam_wrist"] = get_item(data, map["cam_wrist"])

            # 构造 14 维 qpos：左臂(6)+左爪(1)+右臂(6)+右爪(1)
            def try_get(name):
                try:
                    return get_item(data, name)
                except Exception:
                    return None

            left_joint = try_get("slave_left_arm.joint")
            left_gripper = try_get("slave_left_arm.gripper")
            right_joint = try_get("slave_right_arm.joint")
            right_gripper = try_get("slave_right_arm.gripper")

            # 推断时间步 T（优先以左臂为准）
            for arr in (left_joint, left_gripper, right_joint, right_gripper):
                if arr is not None:
                    T_total = len(arr)
                    break
            else:
                print(f"skip {hdf5_path} because no arm data found")
                continue

            def ensure(arr, T, D):
                if arr is None:
                    out = np.zeros((T, D), dtype=np.float32)
                else:
                    arr = np.asarray(arr)
                    if arr.ndim == 1:
                        arr = arr.reshape(-1, 1)
                    out = arr.astype(np.float32)
                    # 对列进行裁剪/填充
                    if out.shape[1] < D:
                        pad = np.zeros((out.shape[0], D - out.shape[1]), dtype=np.float32)
                        out = np.concatenate([out, pad], axis=1)
                    elif out.shape[1] > D:
                        out = out[:, :D]
                return out

            # 统一时间长度（不降采样）
            T = len(np.asarray(left_joint)) if left_joint is not None else (
                len(np.asarray(left_gripper)) if left_gripper is not None else (
                len(np.asarray(right_joint)) if right_joint is not None else len(np.asarray(right_gripper))
            ))

            left_joint = ensure(left_joint, T, 6)
            left_gripper = ensure(left_gripper, T, 1)
            right_joint = ensure(right_joint, T, 6)
            right_gripper = ensure(right_gripper, T, 1)

            qpos = np.concatenate([left_joint, left_gripper, right_joint, right_gripper], axis=1)
            
            actions = []

            for i in range(len(qpos) - 1):
                actions.append(qpos[i+1])
            
            last_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            # 最后一帧结束无动作
            actions.append(last_action)

            actions = np.array(actions)
            f.create_dataset('action', data=np.array(actions), dtype="float32")

            obs = f.create_group("observations")
            '''
            Basic robot arm parameters: if you’re using joint values, 
            you can rename them to avoid confusion instead of calling them qpos, 
            but remember to update the corresponding model’s data loading phase accordingly.
            '''

            obs.create_dataset('qpos', data=np.array(qpos), dtype="float32")
            obs.create_dataset("left_arm_dim", data=np.array(6))
            obs.create_dataset("right_arm_dim", data=np.array(6))

            images = obs.create_group("images")
            
            # Retrieve data based on your camera/view names, then encode and compress it for storage.

            cam_high = input_data["cam_high"]
            cam_wrist = input_data["cam_wrist"]
            
            
            # head_enc, head_len = images_encoding(cam_high)
            # # wrist_enc, wrist_len = images_encoding(cam_wrist)
            # left_enc, left_len = images_encoding(cam_left_wrist)
            # right_enc, right_len = images_encoding(cam_right_wrist)

            # images.create_dataset('cam_high', data=head_enc, dtype=f'S{head_len}')
            # # images.create_dataset('cam_wrist', data=wrist_enc, dtype=f'S{wrist_len}')
            # images.create_dataset('cam_left_wrist', data=left_enc, dtype=f'S{left_len}')
            # images.create_dataset('cam_right_wrist', data=right_enc, dtype=f'S{right_len}')

            images.create_dataset("cam_high", data=np.stack(cam_high), dtype=np.uint8)
            images.create_dataset("cam_wrist", data=np.stack(cam_wrist), dtype=np.uint8)

        print(f"convert {hdf5_path} to rdt data format at {hdf5_output_path}")

if __name__ == "__main__":
    import argparse
    import json
    # parser = argparse.ArgumentParser(description='Transform datasets typr to HDF5.')
    # parser.add_argument('data_path', type=str,
    #                     help="your data dir like: datasets/task/")
    # parser.add_argument('outout_path', type=str,default=None,
    #                     help='output path commanded like datasets/RDT/...')
    
    # args = parser.parse_args()
    # data_path = args.data_path
    # output_path = args.outout_path
    data_path = "path/to/your/data"
    output_path = "/home/usst/kwj/GitCode/RoboTwin2.0/policy/ACT/processed_data/sim-pick_place_cup/"

    # if output_path is None:
    #     data_config = json.load(os.path.join(data_path, "config.json"))
    #     output_path = f"./datasets/RDT/{data_config['task_name']}"
    
    hdf5_paths = get_files(data_path, "*.hdf5")
    print("hdf5 files:\n",hdf5_paths)
    convert(hdf5_paths, output_path)