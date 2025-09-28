#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
OpenPI inference model classes for dual and single arm robot control
"""
from pathlib import Path

# get current workspace
current_file = Path(__file__)

import sys

parent_dir = current_file.parent
sys.path.append(str(parent_dir))

import json
import sys
import numpy as np
from src.openpi.models import model as _model
from src.openpi.policies import aloha_policy
from src.openpi.policies import policy_config as _policy_config
from src.openpi.shared import download
from src.openpi.training import config as _config
from src.openpi.training import data_loader as _data_loader
import os
import cv2
from PIL import Image


class PI0_DUAL:
    def __init__(self, model_path, task_name):
        self.task_name = task_name
        self.model_path = model_path

        train_config_name = "pi0_base_aloha_robotwin_lora"
        config = _config.get_config(train_config_name)
        print("get config success!")
        self.policy = _policy_config.create_trained_policy(config, model_path)
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.random_set_language()

    def set_img_size(self, img_size):
        self.img_size = img_size

    def random_set_language(self):
        possible_paths = [
            f"datasets/instructions/{self.task_name}.json",
            f"task_instructions/{self.task_name}.json",
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "task_instructions", f"{self.task_name}.json")
        ]

        json_Path = None
        for path in possible_paths:
            if os.path.exists(path):
                json_Path = path
                break

        if json_Path is None:
            print(f"Warning: Could not find instruction file for task {self.task_name}")
            self.instruction = "Grab that green bottle"  # 使用有意义的默认指令
            return

        try:
            with open(json_Path, 'r') as f_instr:
                instruction_dict = json.load(f_instr)
            instructions = instruction_dict['instructions']
            instruction = np.random.choice(instructions)
            self.instruction = instruction
            print(f"successfully set instruction: {instruction}")
        except Exception as e:
            print(f"Error loading instructions: {e}")
            self.instruction = "Grab that green bottle"  # 使用有意义的默认指令

    def update_observation_window(self, img_arr, state):
        """Update observation window with proper validation"""
        try:
            if not isinstance(img_arr, (list, tuple)) or len(img_arr) < 3:
                raise ValueError(
                    f"Expected img_arr to be list/tuple with 3+ elements, got {type(img_arr)} with length {len(img_arr) if hasattr(img_arr, '__len__') else 'unknown'}")

            img_front, img_right, img_left = img_arr[0], img_arr[1], img_arr[2]

            # Validate image dimensions
            for i, img in enumerate([img_front, img_right, img_left]):
                if not hasattr(img, 'shape') or len(img.shape) != 3:
                    raise ValueError(
                        f"Image {i} should have 3 dimensions (H,W,C), got shape: {img.shape if hasattr(img, 'shape') else 'no shape'}")

            # Transpose from (H,W,C) to (C,H,W)
            img_front = np.transpose(img_front, (2, 0, 1))
            img_right = np.transpose(img_right, (2, 0, 1))
            img_left = np.transpose(img_left, (2, 0, 1))

            self.observation_window = {
                "state": state,
                "images": {
                    "cam_high": img_front,
                    "cam_left_wrist": img_left,
                    "cam_right_wrist": img_right,
                },
                "prompt": self.instruction,
            }

        except Exception as e:
            print(f"Error in update_observation_window: {e}")
            raise

    def get_action(self):
        assert (self.observation_window is not None), "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        # 注意：不要重置instruction，因为它需要在整个会话中保持
        self.observation_window = None
        print("successfully unset obs (but kept language instruction)")


class PI0_SINGLE:
    def __init__(self, task_name, train_config_name, model_name, checkpoint_id):
        self.train_config_name = train_config_name
        self.task_name = task_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id

        config = _config.get_config(self.train_config_name)

        # More flexible model path handling
        model_paths = [
            f"/root/.cache/openpi/openpi-assets/checkpoints/{self.model_name}",
            f"./checkpoints/{self.model_name}",
            f"../checkpoints/{self.model_name}",
            self.model_name  # In case full path is provided
        ]

        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            print(f"Warning: Model path not found, using default: {model_paths[0]}")
            model_path = model_paths[0]

        self.policy = _policy_config.create_trained_policy(config, model_path)
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.random_set_language()

        # 验证初始化后的指令
        print(f"Debug: 初始化完成，instruction = '{self.instruction}'")

    def set_img_size(self, img_size):
        self.img_size = img_size

    def random_set_language(self):
        possible_paths = [
            f"datasets/instructions/{self.task_name}.json",
            f"task_instructions/{self.task_name}.json",
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "task_instructions", f"{self.task_name}.json")
        ]

        json_Path = None
        for path in possible_paths:
            if os.path.exists(path):
                json_Path = path
                break

        if json_Path is None:
            print(f"Warning: Could not find instruction file for task {self.task_name}")
            self.instruction = "Grab that green bottle"  # 使用有意义的默认指令
            print(f"Debug: 设置默认指令: {self.instruction}")
            return

        try:
            with open(json_Path, 'r') as f_instr:
                instruction_dict = json.load(f_instr)
            instructions = instruction_dict['instructions']
            instruction = np.random.choice(instructions)
            self.instruction = instruction
            print(f"successfully set instruction: {instruction}")
        except Exception as e:
            print(f"Error loading instructions: {e}")
            self.instruction = "Grab that green bottle"  # 使用有意义的默认指令
            print(f"Debug: 错误后设置默认指令: {self.instruction}")

    def update_observation_window(self, img_data, state_data):
        """接收observation格式的数据并处理"""
        try:
            print(f"Debug: 接收数据 - 图像: {img_data.shape}, 状态: {state_data.shape}")

            # 验证输入
            if len(img_data.shape) != 3:
                raise ValueError(f"图像应该是3维，实际: {img_data.shape}")

            # 存储原始数据
            self.raw_image = img_data

            # 处理状态维度：从7维扩展到8维
            if len(state_data) == 7:
                # 添加一个额外维度，通常是第7个关节位置的占位符
                # 或者复制最后一个值作为占位符
                extended_state = np.append(state_data, 0.0)  # 添加0作为占位符
                print(f"Debug: 状态维度从7扩展到8: {state_data.shape} -> {extended_state.shape}")
            elif len(state_data) == 8:
                extended_state = state_data
            else:
                # 如果维度不是7也不是8，尝试填充或截断到8维
                if len(state_data) < 8:
                    extended_state = np.pad(state_data, (0, 8 - len(state_data)),
                                            mode='constant', constant_values=0)
                else:
                    extended_state = state_data[:8]
                print(f"Debug: 状态维度调整到8: {state_data.shape} -> {extended_state.shape}")

            self.raw_state = extended_state

            # 确保instruction存在
            if not hasattr(self, 'instruction') or not self.instruction:
                self.instruction = "Grab that green bottle"
                print(f"Debug: 在update时重新设置指令: {self.instruction}")

            # 传统格式也需要相应调整
            if len(extended_state) < 15:
                padded_state = np.pad(extended_state, (0, 15 - len(extended_state)),
                                      mode='constant', constant_values=0)
            else:
                padded_state = extended_state[:15]

            self.processed_state = padded_state

            # 转换图像格式
            img_right = np.transpose(img_data, (2, 0, 1))  # (H,W,C) -> (C,H,W)
            img_front = np.zeros_like(img_right)
            img_left = np.zeros_like(img_right)

            self.observation_window = {
                "state": padded_state,
                "images": {
                    "cam_high": img_front,
                    "cam_left_wrist": img_left,
                    "cam_right_wrist": img_right,
                },
                "prompt": self.instruction,
            }

            print("Debug: 观察窗口更新成功")

        except Exception as e:
            print(f"Error in update_observation_window: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            raise

    def get_action(self):
        import numpy as np
        import jax

        print(f"Debug: =====开始诊断=====")

        # 第一步：检查导入
        try:
            from openpi.models import model as _model
            print(f"Debug: ✓ 成功导入 _model")
            print(f"Debug: _model.Observation 类型: {type(_model.Observation)}")

            # 检查Observation是否是NamedTuple
            if hasattr(_model.Observation, '_fields'):
                print(f"Debug: ✓ Observation字段: {_model.Observation._fields}")
            else:
                print(f"Debug: ✗ Observation不是NamedTuple")

        except Exception as import_error:
            print(f"Debug: ✗ 导入失败: {import_error}")
            return np.zeros((1, 7), dtype=np.float32)

        # 第二步：测试创建Observation对象
        try:
            print(f"Debug: 开始创建测试Observation对象...")

            # 准备数据
            test_image = np.random.rand(1, 360, 640, 3).astype(np.float32)
            test_state = np.zeros((1, 32), dtype=np.float32)
            test_tokens = np.zeros((1, 250), dtype=np.int32)
            test_mask = np.zeros((1, 250), dtype=bool)

            # 尝试创建Observation对象
            test_obs = _model.Observation(
                images={
                    "base_0_rgb": test_image,
                    "base_1_rgb": test_image,
                    "wrist_0_rgb": test_image,
                },
                image_masks={
                    "base_0_rgb": np.array([True]),
                    "base_1_rgb": np.array([True]),
                    "wrist_0_rgb": np.array([True]),
                },
                state=test_state,
                tokenized_prompt=test_tokens,
                tokenized_prompt_mask=test_mask,
                token_ar_mask=test_tokens,  # 使用int32类型
                token_loss_mask=test_mask,
            )

            print(f"Debug: ✓ 测试Observation创建成功")
            print(f"Debug: ✓ 对象类型: {type(test_obs)}")
            print(f"Debug: ✓ 有images属性: {hasattr(test_obs, 'images')}")
            print(f"Debug: ✓ images类型: {type(test_obs.images)}")

        except Exception as create_error:
            print(f"Debug: ✗ 创建Observation失败: {create_error}")
            import traceback
            print(f"Debug: 详细错误: {traceback.format_exc()}")
            return np.zeros((1, 7), dtype=np.float32)

        # 第三步：使用真实数据创建Observation
        try:
            assert (self.observation_window is not None), "observation_window为空"

            if not hasattr(self, 'instruction') or not self.instruction:
                self.instruction = "Grab that green bottle"

            print(f"Debug: 开始使用真实数据创建Observation...")

            # 处理真实数据
            processed_image = self.raw_image.astype(np.float32) / 255.0
            processed_image = processed_image[np.newaxis, :]  # [1, H, W, 3]

            processed_state = self.raw_state.reshape(1, -1).astype(np.float32)
            if processed_state.shape[1] < 32:
                padding_width = ((0, 0), (0, 32 - processed_state.shape[1]))
                processed_state = np.pad(processed_state, padding_width, mode='constant', constant_values=0)

            # 创建tokenized数据
            max_token_len = 250
            dummy_tokens = np.zeros((1, max_token_len), dtype=np.int32)
            dummy_mask = np.zeros((1, max_token_len), dtype=bool)
            dummy_ar_mask = np.zeros((1, max_token_len), dtype=np.int32)
            dummy_loss_mask = np.zeros((1, max_token_len), dtype=bool)

            # 创建真实Observation对象
            real_observation = _model.Observation(
                images={
                    "base_0_rgb": processed_image,
                    "base_1_rgb": processed_image,
                    "wrist_0_rgb": processed_image,
                },
                image_masks={
                    "base_0_rgb": np.array([True]),
                    "base_1_rgb": np.array([True]),
                    "wrist_0_rgb": np.array([True]),
                },
                state=processed_state,
                tokenized_prompt=dummy_tokens,
                tokenized_prompt_mask=dummy_mask,
                token_ar_mask=dummy_ar_mask,
                token_loss_mask=dummy_loss_mask,
            )

            print(f"Debug: ✓ 真实Observation创建成功")
            print(f"Debug: ✓ 真实对象类型: {type(real_observation)}")
            print(f"Debug: ✓ 真实对象有images: {hasattr(real_observation, 'images')}")

            # 第四步：调用sample_actions
            model = self.policy._model
            rng = jax.random.PRNGKey(0)

            print(f"Debug: 调用sample_actions...")
            actions = model.sample_actions(rng, real_observation)

            print(f"Debug: ✓ sample_actions成功！")
            print(f"Debug: ✓ 输出类型: {type(actions)}")
            print(f"Debug: ✓ 输出形状: {actions.shape}")

            return self._process_numeric_actions(actions)

        except Exception as real_error:
            print(f"Debug: ✗ 真实数据处理失败: {real_error}")
            import traceback
            print(f"Debug: 详细错误: {traceback.format_exc()}")

        print(f"Debug: 返回零动作")
        return np.zeros((1, 7), dtype=np.float32)

    def _parse_string_actions(self, action_string):
        """解析字符串格式的动作"""
        import numpy as np
        import re

        try:
            print(f"Debug: 尝试解析字符串动作: {action_string}")

            # 提取浮点数
            numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', action_string)
            if len(numbers) >= 7:
                action_values = [float(x) for x in numbers[:7]]
                parsed_actions = np.array([action_values], dtype=np.float32)
                print(f"Debug: 成功解析字符串动作: {parsed_actions.shape}")
                return parsed_actions

            print("Debug: 字符串解析失败，返回零动作")
            return np.zeros((1, 7), dtype=np.float32)

        except Exception as e:
            print(f"Debug: 字符串解析异常: {e}")
            return np.zeros((1, 7), dtype=np.float32)

    def _process_numeric_actions(self, actions):
        """处理Pi0FAST的token输出并解码为实际动作"""
        import numpy as np

        try:
            print(f"Debug: 处理Pi0FAST token输出，类型: {type(actions)}")
            print(f"Debug: Token序列形状: {actions.shape}")

            if not isinstance(actions, np.ndarray):
                actions = np.array(actions)

            # Pi0FAST返回的是token序列 [batch_size, max_decoding_steps]
            if len(actions.shape) == 2 and actions.shape[1] > 7:
                print(f"Debug: 检测到token格式输出，需要解码")

                # 方法1: 尝试使用模型的tokenizer进行解码
                try:
                    # 获取Pi0FAST的配置中的tokenizer
                    model = self.policy._model
                    config = model.config if hasattr(model, 'config') else None

                    if config and hasattr(config, 'fast_model_tokenizer') and config.fast_model_tokenizer:
                        tokenizer = config.fast_model_tokenizer
                        print(f"Debug: 找到tokenizer: {type(tokenizer)}")

                        # 尝试解码token序列为动作
                        if hasattr(tokenizer, 'decode_actions'):
                            decoded_actions = tokenizer.decode_actions(actions)
                            print(f"Debug: 成功解码动作: {decoded_actions.shape}")
                            return decoded_actions
                        elif hasattr(tokenizer, 'extract_actions'):
                            decoded_actions = tokenizer.extract_actions(actions)
                            print(f"Debug: 成功提取动作: {decoded_actions.shape}")
                            return decoded_actions
                        elif hasattr(tokenizer, '__call__'):
                            # 尝试直接调用tokenizer
                            decoded_actions = tokenizer(actions, action_horizon=32, action_dim=7)
                            print(f"Debug: tokenizer调用成功: {decoded_actions.shape}")
                            return decoded_actions

                    print(f"Debug: tokenizer不可用，尝试其他方法")

                except Exception as tokenizer_error:
                    print(f"Debug: tokenizer解码失败: {tokenizer_error}")

                # 方法2: 检查策略是否有解码方法
                try:
                    if hasattr(self.policy, 'decode_tokens'):
                        decoded_actions = self.policy.decode_tokens(actions)
                        print(f"Debug: 策略解码成功: {decoded_actions.shape}")
                        return decoded_actions
                    elif hasattr(self.policy, '_tokenizer'):
                        tokenizer = self.policy._tokenizer
                        if hasattr(tokenizer, 'decode'):
                            decoded_actions = tokenizer.decode(actions)
                            print(f"Debug: 策略tokenizer解码成功: {decoded_actions.shape}")
                            return decoded_actions

                except Exception as policy_decode_error:
                    print(f"Debug: 策略解码失败: {policy_decode_error}")

                # 方法3: 临时解决方案 - 将token映射到合理的动作空间
                print(f"Debug: 使用临时token映射方案")

                # 取第一个batch的前7个token
                token_subset = actions[0, :7].astype(np.float32)

                # 将token ID映射到合理的动作范围 [-1, 1]
                # 假设token范围大约是[0, 50000+]
                normalized_actions = (token_subset % 1000) / 500.0 - 1.0  # 映射到[-1, 1]

                # 进一步缩放到更合理的机械臂动作范围
                scaled_actions = normalized_actions * 0.1  # 小幅动作

                print(f"Debug: 临时映射结果: {scaled_actions}")
                print(f"Debug: 映射后范围: [{np.min(scaled_actions):.3f}, {np.max(scaled_actions):.3f}]")

                return scaled_actions.reshape(1, -1)

            # 如果不是token格式，按原来的方式处理
            if len(actions.shape) == 1:
                actions = actions.reshape(1, -1)

            if actions.shape[-1] > 7:
                actions = actions[:, :7]
            elif actions.shape[-1] < 7:
                padding_width = ((0, 0), (0, 7 - actions.shape[-1]))
                actions = np.pad(actions, padding_width, mode='constant', constant_values=0)

            actions = actions.astype(np.float32)
            return actions

        except Exception as e:
            print(f"Debug: 动作处理失败: {e}")
            return np.zeros((1, 7), dtype=np.float32)

    def _extract_actions_from_policy_error(self, policy_input):
        """当正常流程失败时，尝试从策略内部提取动作"""
        import numpy as np

        try:
            print("Debug: 尝试绕过变换直接获取模型输出...")

            # 尝试直接调用底层模型
            if hasattr(self.policy, '_model'):
                model = self.policy._model

                # 尝试不同的调用方式
                if hasattr(model, 'forward'):
                    outputs = model.forward(policy_input)
                elif hasattr(model, '__call__'):
                    outputs = model(policy_input)
                else:
                    print("Debug: 无法找到合适的调用方法")
                    return np.zeros((1, 7), dtype=np.float32)

                print(f"Debug: 底层模型输出: {type(outputs)}")

                # 如果输出是字典
                if isinstance(outputs, dict):
                    for key in ['actions', 'action', 'output', 'prediction']:
                        if key in outputs:
                            result = outputs[key]
                            if isinstance(result, str):
                                return self._parse_string_actions(result)
                            else:
                                return self._process_numeric_actions(result)

                # 如果输出不是字典，尝试直接处理
                if isinstance(outputs, str):
                    return self._parse_string_actions(outputs)
                else:
                    return self._process_numeric_actions(outputs)

        except Exception as e:
            print(f"Debug: 底层提取也失败: {e}")
            return np.zeros((1, 7), dtype=np.float32)

    def reset_obsrvationwindows(self):
        # 重要修改：不要重置instruction！
        self.observation_window = None
        if hasattr(self, 'raw_image'):
            del self.raw_image
        if hasattr(self, 'raw_state'):
            del self.raw_state
        if hasattr(self, 'processed_state'):
            del self.processed_state
        # 确保instruction存在
        if not hasattr(self, 'instruction') or not self.instruction:
            self.instruction = "Grab that green bottle"
        print(f"successfully unset obs but kept instruction: '{self.instruction}'")
