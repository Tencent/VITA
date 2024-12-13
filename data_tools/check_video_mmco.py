import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm

from decord import VideoReader, cpu
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from vita.config import *
from vita.config import FolderDict
from vita.config.dataset_config import *
from vita.constants import MAX_IMAGE_LENGTH, MIN_IMAGE_LENGTH
import numpy as np
import subprocess
from contextlib import redirect_stdout, redirect_stderr
import io
import logging


logging.basicConfig(filename='video_errors.log', level=logging.INFO)
# 定义文件路径
output_file_path = "lost_file_name.txt"

# 将所有字典放入一个列表中
datasets = NaturalCap


# 初始化一个列表来存储丢失的文件名
lost_files = []

# 遍历每个字典
for dataset in datasets:
    keys = list(dataset.keys())
    json_file_path = dataset["chat_path"]

    # 读取JSON文件
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def check_video_file(video_name, set_id):
        video_file_name = video_name
        if video_file_name:
            video_directory = FolderDict[set_id]
            video_file_path = os.path.join(video_directory, video_file_name)
            if not os.path.exists(video_file_path):
                print(f"file lost: {video_file_path}")
                return video_file_name
            else:
                vreader = VideoReader(video_file_path, ctx=cpu(0))
                video_framerate = 1
                fps = vreader.get_avg_fps()
                max_frames = MAX_IMAGE_LENGTH
                min_frames = MIN_IMAGE_LENGTH
                start_time, end_time = None, None
                f_start = 0 if start_time is None else int(start_time * fps)
                f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
                sample_fps = int(video_framerate)
                t_stride = int(round(float(fps) / sample_fps))
                all_pos = list(range(f_start, f_end + 1, t_stride))
                if len(all_pos) > max_frames:
                    sample_pos = [
                        all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)
                    ]
                elif len(all_pos) < min_frames:
                    sample_pos = [
                        all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)
                    ]
                else:
                    sample_pos = all_pos
                print(video_file_name)
                patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
        return None

    # 使用ThreadPoolExecutor进行多线程并行处理
    video_names = [item["video"] for item in data]
    set_id = data[0]['set']
    video_names = list(set(video_names))
    for video_name in video_names:
        check_video_file(video_name, set_id)

# 将丢失的文件名写入到lost_file_name.txt中
with open(output_file_path, "w", encoding="utf-8") as f:
    for file_name in lost_files:
        f.write(file_name + "\n")

print(f"检查完成，共有 {len(lost_files)} 个文件丢失或无法读取，结果已保存到 {output_file_path}")

