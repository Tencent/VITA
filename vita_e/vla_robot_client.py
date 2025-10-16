#!/usr/bin/env python3
"""
VLA Robot Client - Python客户端替代demo.html前端
与server_vla.py进行Socket.IO通信，控制GR2机器人进行语音交互和动作执行
"""

import os
import sys
import time
import json
import base64
import threading
import queue
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd
import socketio
from omegaconf import OmegaConf

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "fourier_il_eval"))

from controller.gr2_player import GR2Player


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    dtype: str = 'int16'


@dataclass
class RobotState:
    hand_state: List[float]
    robot_state: List[float]
    image_data: str


class EventBus:
    """简单的事件总线，用于组件间解耦通信"""
    
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}
    
    def subscribe(self, event: str, callback: Callable):
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(callback)
    
    def emit(self, event: str, data: Any = None):
        if event not in self._listeners:
            return
        for callback in self._listeners[event]:
            callback(data)


class AudioManager:
    """音频采集、发送和播放管理器"""
    
    def __init__(self, config: AudioConfig, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.tts_audio_queue = queue.Queue()
        self.is_recording = False
        
        # 初始化音频流
        self.input_stream = None
        self.output_stream = None
        self.output_samplerate = None

        # 启动用于顺序播放音频的线程
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()

        # 初始化持久化输出流（不指定设备，使用系统默认）
        self._init_output_stream()

    def _init_output_stream(self):
        """初始化持久化音频输出流。如果失败则退回到临时播放模式。"""
        try:
            dev_info = sd.query_devices(None, 'output')
            default_sr = dev_info.get('default_samplerate', 16000) if isinstance(dev_info, dict) else 16000
            self.output_samplerate = int(default_sr) if default_sr else 16000
            self.output_stream = sd.OutputStream(
                samplerate=self.output_samplerate,
                channels=1,
                dtype='float32',
            )
            self.output_stream.start()
            print(f"✅ 持久化输出流已启动，采样率: {self.output_samplerate}")
        except Exception as e:
            # 失败时保持 None，后续播放回退到 sd.play/sd.wait
            self.output_stream = None
            self.output_samplerate = None
            print(f"⚠️ 持久化输出流初始化失败，回退到临时播放: {e}")

    def _resample_linear(self, samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        """将一维音频samples从src_sr线性重采样到dst_sr，返回float32[-1,1]"""
        if src_sr == dst_sr:
            return samples.astype(np.float32, copy=False)
        if samples.size == 0:
            return samples.astype(np.float32, copy=False)
        num_output = max(1, int(round(samples.shape[0] * (dst_sr / float(src_sr)))))
        xp = np.arange(samples.shape[0], dtype=np.float32)
        x = np.linspace(0, samples.shape[0] - 1, num=num_output, dtype=np.float32)
        return np.interp(x, xp, samples.astype(np.float32)).astype(np.float32, copy=False)

    def _playback_loop(self):
        """在一个专用线程中，按顺序播放队列中的音频。"""
        while True:
            audio_data = self.tts_audio_queue.get()
            if audio_data is None:  # 使用None作为停止信号
                break
            
            try:
                # 转换为numpy数组
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                # 强制保持 float32，避免被 Python float 提升为 float64
                audio_float = (audio_array.astype(np.float32) * (1.0 / np.float32(32767.0))).astype(np.float32, copy=False)

                if self.output_stream is not None and self.output_samplerate is not None:
                    # 重采样到设备采样率并写入持久化输出流
                    resampled = self._resample_linear(audio_float, 24000, self.output_samplerate).astype(np.float32, copy=False)
                    # 写入期望 (frames, channels)
                    self.output_stream.write(resampled.reshape(-1, 1))
                else:
                    # 回退到一次性播放
                    sd.play(audio_float.astype(np.float32, copy=False), samplerate=24000)
                    sd.wait()
            except Exception as e:
                print(f"❌ 音频播放错误: {e}")

    def start_recording(self):
        """开始音频采集"""
        if self.is_recording:
            return
            
        print("🎤 开始音频采集...")
        self.is_recording = True
        
        def audio_callback(indata, frames, time, status):
            if not self.is_recording:
                return
            # 转换为int16 numpy array
            int16_array = (indata * 32767).astype(np.int16)
            # 转换为字节流，再转换为字节值列表，以匹配JS客户端的行为
            byte_list = list(int16_array.tobytes())
            self.event_bus.emit('audio_data', {
                'sample_rate': self.config.sample_rate,
                'audio': byte_list
            })
        
        self.input_stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            callback=audio_callback,
            blocksize=self.config.chunk_size
        )
        self.input_stream.start()
    
    def stop_recording(self):
        """停止音频采集"""
        if not self.is_recording:
            return
            
        print("🛑 停止音频采集")
        self.is_recording = False
        
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None
    
    def play_audio(self, audio_data: bytes):
        """将接收到的音频数据放入队列以供播放"""
        if audio_data:
            self.tts_audio_queue.put(audio_data)
    
    def stop_playback(self):
        """停止音频播放并清空队列"""
        # 清空队列
        while not self.tts_audio_queue.empty():
            try:
                self.tts_audio_queue.get_nowait()
            except queue.Empty:
                break

        # 方案A：不主动中断持久化输出流，避免跨线程状态机错误；
        # 若没有持久化流（回退路径），停止一次性播放
        if self.output_stream is None:
            try:
                sd.stop()
            except Exception:
                pass

    def shutdown(self):
        """平稳地关闭音频管理器"""
        self.stop_recording()
        self.stop_playback()
        self.tts_audio_queue.put(None)  # 发送停止信号以终止播放线程
        self.playback_thread.join()
        # 关闭持久化输出流
        if self.output_stream is not None:
            try:
                self.output_stream.stop()
                self.output_stream.close()
            except Exception:
                pass
            self.output_stream = None


class VideoStreamManager:
    """视频流管理器，处理连续视频流上传"""
    
    def __init__(self, gr2_player, event_bus: EventBus):
        self.gr2_player = gr2_player
        self.event_bus = event_bus
        self.is_streaming = False
        self.stream_interval = 0.5  # 每500ms发送一帧，与demo.html保持一致
        self.stream_thread = None
    
    def start_video_stream(self):
        """开始连续视频流上传"""
        if self.is_streaming:
            return
            
        print("📹 开始视频流上传...")
        self.is_streaming = True
        
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()
    
    def stop_video_stream(self):
        """停止连续视频流上传"""
        if not self.is_streaming:
            return
            
        print("🛑 停止视频流上传")
        self.is_streaming = False
    
    def _stream_loop(self):
        """视频流循环（在独立线程中运行）"""
        while self.is_streaming:
            try:
                # 获取当前视频帧
                _, left_image, _, _, _ = self.gr2_player.observe(mode='bimanual')
                
                # 编码图像
                image_data = self._encode_image(left_image)
                if image_data:
                    # 发送video_frame事件
                    self.event_bus.emit('video_frame', image_data)
                
                time.sleep(self.stream_interval)
                
            except Exception as e:
                print(f"❌ 视频流处理错误: {e}")
                time.sleep(self.stream_interval)
    
    def _encode_image(self, image: np.ndarray) -> str:
        """将图像编码为Base64 Data URL格式"""
        # 调整图像大小 - 与demo.html中hiddenCanvas尺寸保持一致
        resized_image = cv2.resize(image, (1280, 768))

        # 将图像转换为RGB格式
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # JPEG编码，质量0.7与demo.html保持一致
        _, buffer = cv2.imencode('.jpg', resized_image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        
        # Base64编码
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        # 构造Data URL
        return f"data:image/jpeg;base64,{jpg_as_text}"


class RobotController:
    """机器人控制器，处理状态采集和动作执行"""
    
    def __init__(self, config_path: str, event_bus: EventBus):
        self.event_bus = event_bus
        self.is_active = False
        self.action_queue = queue.Queue()
        self.max_runtime = 100  # 最大运行时间100秒
        
        # 初始化GR2Player
        print("🤖 初始化GR2机器人...")
        config = OmegaConf.load(config_path)
        self.gr2_player = GR2Player(config)
        self.gr2_player.reset_robot()
        print("✅ 机器人初始化完成")
        
        # 启动机器人控制线程
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
    
    def start_action_loop(self):
        """开始动作循环"""
        if self.is_active:
            return
            
        print("🔄 开始机器人动作循环...")
        self.is_active = True
        self.start_time = time.time()
    
    def stop_action_loop(self):
        """停止动作循环"""
        if not self.is_active:
            return
            
        print("⏹️ 停止机器人动作循环")
        self.is_active = False
    
    def execute_action(self, action_data: Dict[str, Any]):
        """执行动作数据"""
        self.action_queue.put(action_data)
    
    def _control_loop(self):
        """机器人控制循环（在独立线程中运行）"""
        while True:
            if not self.is_active:
                time.sleep(0.1)
                continue
            
            # 检查超时
            if time.time() - self.start_time > self.max_runtime:
                print(f"⏰ 达到最大运行时间{self.max_runtime}秒，自动停止")
                self.stop_action_loop()
                continue
            
            # 采集当前状态
            robot_state = self._observe_state()
            if not robot_state:
                continue
                
            # 发送状态到服务器
            self.event_bus.emit('robot_state', robot_state)
            
            # 等待动作响应
            action_data = self._wait_for_action()
            if not action_data:
                continue
                
            # 检查是否是停止信号
            if action_data.get('halt'):
                print("🚨 收到停止信号")
                self.stop_action_loop()
                continue
            
            # 执行动作
            self._execute_action_sequence(action_data)
    
    def _observe_state(self) -> Optional[RobotState]:
        """观察机器人当前状态"""
        qpos, left_image, right_image, _, _ = self.gr2_player.observe(mode='bimanual')
        
        # 提取状态数据
        hand_state = qpos[:12].tolist()
        robot_state = qpos[-14:].tolist()
        
        # 处理图像数据
        image_data = self._encode_image(left_image)
        if not image_data:
            return None
            
        return RobotState(
            hand_state=hand_state,
            robot_state=robot_state,
            image_data=image_data
        )
    
    def _encode_image(self, image: np.ndarray) -> str:      
        """将图像编码为Base64 Data URL格式"""
        # 调整图像大小
        resized_image = cv2.resize(image, (400, 225))

        # JPEG编码
        _, buffer = cv2.imencode('.jpg', resized_image)
        
        # Base64编码
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        # 构造Data URL
        return f"data:image/jpeg;base64,{jpg_as_text}"
    
    def _wait_for_action(self, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        """等待动作响应"""
        try:
            return self.action_queue.get(timeout=timeout)
        except queue.Empty:
            print("⚠️ 等待动作超时")
            return None
    
    def _execute_action_sequence(self, action_data: Dict[str, Any]):
        """执行动作序列"""
        action_type = action_data.get('type', 'unknown')
        if action_type == 'retraction':
            print("⚡ 执行回撤动作序列...")
        elif action_type == 'new':
            print("⚡ 执行新动作序列...")
        else:
            print("⚡ 执行动作序列...")

        # The actual action is inside the 'action' key
        actual_action = action_data.get('action')
        
        if not isinstance(actual_action, dict):
            print(f"❌ 无效的动作内容: {actual_action}")
            return
        
        # 解析动作数据
        if 'action.hand' not in actual_action or 'action.robot' not in actual_action:
            print(f"actual_action: {actual_action}")
            print(f"❌ 动作数据格式错误: 缺少 'action.hand' or 'action.robot'")
            return
        
        hand_actions = actual_action['action.hand']
        robot_actions = actual_action['action.robot']
        
        # 执行动作序列
        for i in range(min(len(hand_actions), len(robot_actions))):
            hand_action = hand_actions[i]
            robot_action = robot_actions[i]
            action = np.concatenate([robot_action, hand_action])
            
            # 执行单步动作
            self.gr2_player.step(action, mode='bimanual', time=0.0)
            time.sleep(0.01)
            
            print(f"✓ 执行动作步骤 {i+1}/{len(hand_actions)}")
        
        print("✅ 动作序列执行完成")


class SocketIOHandler:
    """Socket.IO通信处理器"""
    
    def __init__(self, server_url: str, event_bus: EventBus):
        self.server_url = server_url
        self.event_bus = event_bus
        self.sio = socketio.Client(ssl_verify=False)
        self._setup_event_handlers()
    
    def connect(self):
        """连接到服务器"""
        print(f"🔗 连接到服务器: {self.server_url}")
        self.sio.connect(self.server_url)
        print("✅ 连接成功")
    
    def disconnect(self):
        """断开连接"""
        print("🔌 断开连接")
        self.sio.disconnect()
    
    def _setup_event_handlers(self):
        """设置Socket.IO事件处理器"""
        
        @self.sio.on('connect')
        def on_connect():
            print("🎉 已连接到VLA服务器")
        
        @self.sio.on('disconnect')
        def on_disconnect():
            print("💔 与服务器断开连接")
        
        @self.sio.on('audio')
        def on_audio(data):
            """接收TTS音频数据"""
            self.event_bus.emit('tts_audio', data)
        
        @self.sio.on('start_update_states')
        def on_start_update_states(data):
            """开始更新机器人状态"""
            print("📡 收到开始更新状态信号")
            self.event_bus.emit('start_action_loop')
        
        @self.sio.on('action_feature')
        def on_action_feature(data):
            """接收动作特征数据"""
            print(f"🎯 收到动作数据: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            self.event_bus.emit('action_received', data)
        
        @self.sio.on('stop_tts')
        def on_stop_tts():
            """停止TTS播放"""
            self.event_bus.emit('stop_playback')
    
    def send_audio(self, audio_data: Dict[str, Any]):
        """发送音频数据"""
        self.sio.emit('audio', json.dumps(audio_data))
    
    def send_robot_state(self, robot_state: RobotState):
        """发送机器人状态"""
        state_data = {
            "data": robot_state.image_data,
            "states": {
                "hand": robot_state.hand_state,
                "robot": robot_state.robot_state
            }
        }
        self.sio.emit('states', state_data)
    
    def send_recording_event(self, event_type: str):
        """发送录音事件"""
        self.sio.emit(event_type)
    
    def send_video_frame(self, image_data: str):
        """发送视频帧"""
        self.sio.emit('video_frame', image_data)

    def clear_history(self):
        """发送清除历史记录事件"""
        self.sio.emit('reset_state')


class CommandInterface:
    """命令行交互界面"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.is_running = True
        
        # 启动键盘监听线程
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()
    
    def _input_loop(self):
        """命令输入循环"""
        print("\n" + "="*50)
        print("🎛️  VLA机器人控制面板")
        print("="*50)
        print("命令说明:")
        print("  s - 开始/停止语音交互")
        print("  v - 开始/停止视频流上传")
        print("  r - 机器人回到初始位置")
        print("  c - 清除服务端对话历史")
        print("  q - 退出程序")
        print("  h - 显示帮助")
        print("="*50)
        
        while self.is_running:
            command = input("\n请输入命令: ").strip().lower()
            
            if command == 's':
                self.event_bus.emit('toggle_recording')
            elif command == 'v':
                self.event_bus.emit('toggle_video_stream')
            elif command == 'r':
                self.event_bus.emit('reset_robot')
            elif command == 'c':
                self.event_bus.emit('clear_history')
            elif command == 'q':
                self.event_bus.emit('quit')
                break
            elif command == 'h':
                self._show_help()
            else:
                print("❌ 未知命令，输入 'h' 查看帮助")
    
    def _show_help(self):
        """显示帮助信息"""
        print("\n📖 命令帮助:")
        print("  s - 开始/停止语音交互（切换录音状态）")
        print("  v - 开始/停止视频流上传（独立于语音交互）")
        print("  r - 机器人回到初始位置")
        print("  c - 清除服务端对话历史")
        print("  q - 安全退出程序")
        print("  h - 显示此帮助信息")


class VLARobotClient:
    """VLA机器人客户端主类"""
    
    def __init__(self, 
                 server_url: str = "http://localhost:8081",
                 robot_config_path: str = "fourier_il_eval/controller/configs/gr2t5.yml"):
        
        # 初始化事件总线
        self.event_bus = EventBus()
        
        # 初始化各个组件
        self.audio_config = AudioConfig()
        self.audio_manager = AudioManager(self.audio_config, self.event_bus)
        self.robot_controller = RobotController(robot_config_path, self.event_bus)
        self.video_stream_manager = VideoStreamManager(self.robot_controller.gr2_player, self.event_bus)
        self.socket_handler = SocketIOHandler(server_url, self.event_bus)
        self.command_interface = CommandInterface(self.event_bus)
        
        # 状态变量
        self.is_recording = False
        self.is_video_streaming = False
        self.is_running = True
        
        self._setup_event_subscriptions()
    
    def _setup_event_subscriptions(self):
        """设置事件订阅"""
        
        # 音频相关事件
        self.event_bus.subscribe('audio_data', self._on_audio_data)
        self.event_bus.subscribe('tts_audio', self._on_tts_audio)
        self.event_bus.subscribe('stop_playback', self._on_stop_playback)
        
        # 视频相关事件
        self.event_bus.subscribe('video_frame', self._on_video_frame)
        
        # 机器人相关事件
        self.event_bus.subscribe('robot_state', self._on_robot_state)
        self.event_bus.subscribe('start_action_loop', self._on_start_action_loop)
        self.event_bus.subscribe('action_received', self._on_action_received)
        
        # 用户界面事件
        self.event_bus.subscribe('toggle_recording', self._on_toggle_recording)
        self.event_bus.subscribe('toggle_video_stream', self._on_toggle_video_stream)
        self.event_bus.subscribe('clear_history', self._on_clear_history)
        self.event_bus.subscribe('reset_robot', self._on_reset_robot)
        self.event_bus.subscribe('quit', self._on_quit)
    
    def _on_audio_data(self, audio_data: Dict[str, Any]):
        """处理音频数据"""
        self.socket_handler.send_audio(audio_data)
    
    def _on_tts_audio(self, audio_data: bytes):
        """处理TTS音频"""
        self.audio_manager.play_audio(audio_data)
    
    def _on_stop_playback(self, _):
        """停止音频播放"""
        self.audio_manager.stop_playback()
    
    def _on_video_frame(self, image_data: str):
        """处理视频帧"""
        self.socket_handler.send_video_frame(image_data)
    
    def _on_robot_state(self, robot_state: RobotState):
        """处理机器人状态"""
        self.socket_handler.send_robot_state(robot_state)
    
    def _on_start_action_loop(self, _):
        """开始动作循环"""
        self.robot_controller.start_action_loop()
    
    def _on_action_received(self, action_data: Dict[str, Any]):
        """处理接收到的动作"""
        self.robot_controller.execute_action(action_data)
    
    def _on_toggle_recording(self, _):
        """切换录音状态"""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def _on_toggle_video_stream(self, _):
        """切换视频流状态"""
        if self.is_video_streaming:
            self.stop_video_stream()
        else:
            self.start_video_stream()
    
    def _on_clear_history(self, _):
        """清除历史记录"""
        print("🧹 清除历史记录并重置状态...")
        self.socket_handler.clear_history()

    def _on_reset_robot(self, _):
        """机器人回到初始位置"""
        print("↩️  正在将机器人回到初始位置...")
        try:
            self.robot_controller.gr2_player.reset_robot()
            print("✅ 机器人已回到初始位置")
        except Exception as e:
            print(f"❌ 重置机器人失败: {e}")

    def _on_quit(self, _):
        """退出程序"""
        self.shutdown()
    
    def start_recording(self):
        """开始录音"""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.audio_manager.start_recording()
        self.socket_handler.send_recording_event('recording-started')
        print("🎙️ 语音交互已开始")
    
    def stop_recording(self):
        """停止录音"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        self.audio_manager.stop_recording()
        self.socket_handler.send_recording_event('recording-stopped')
        print("🔇 语音交互已停止")
    
    def start_video_stream(self):
        """开始视频流"""
        if self.is_video_streaming:
            return
            
        self.is_video_streaming = True
        self.video_stream_manager.start_video_stream()
        print("📹 视频流已开始")
    
    def stop_video_stream(self):
        """停止视频流"""
        if not self.is_video_streaming:
            return
            
        self.is_video_streaming = False
        self.video_stream_manager.stop_video_stream()
        print("📹 视频流已停止")
    
    def run(self):
        """运行客户端"""
        # 连接到服务器
        self.socket_handler.connect()
        
        print("🚀 VLA机器人客户端已启动")
        
        # 保持主线程运行
        while self.is_running:
            time.sleep(0.1)
    
    def shutdown(self):
        """安全关闭客户端"""
        print("\n🔄 正在关闭客户端...")
        
        self.is_running = False
        self.stop_recording()
        self.stop_video_stream()
        self.robot_controller.stop_action_loop()
        # 关闭音频资源
        try:
            self.audio_manager.shutdown()
        except Exception:
            pass
        self.socket_handler.disconnect()
        
        print("👋 客户端已关闭")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='VLA Robot Client')
    parser.add_argument('--server', default='http://localhost:8081', 
                       help='VLA服务器地址')
    parser.add_argument('--config', default='fourier_il_eval/controller/configs/gr2t5.yml',
                       help='机器人配置文件路径')
    
    args = parser.parse_args()
    
    # 创建并运行客户端
    client = VLARobotClient(
        server_url=args.server,
        robot_config_path=args.config
    )
    
    client.run()


if __name__ == "__main__":
    main()
