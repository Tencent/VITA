import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from vita.config import *
from vita.config import FolderDict
from vita.config.dataset_config import *

# 定义文件路径
output_file_path = "video_file_sizes.txt"

# 将所有字典放入一个列表中
datasets = NaturalCap


# 初始化一个列表来存储视频文件大小信息
video_file_sizes = []
large_video_file_names = []

# 遍历每个字典
for dataset in datasets:
    keys = list(dataset.keys())
    json_file_path = dataset["chat_path"]

    # 读取JSON文件
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def check_video_file_size(video_name, set_id):
        video_file_name = video_name
        if video_file_name:
            video_directory = FolderDict[set_id]
            video_file_path = os.path.join(video_directory, video_file_name)
            if os.path.exists(video_file_path):
                file_size = os.path.getsize(video_file_path) / 1024 / 1024
                # if file_size <= 0.1:
                #     print(video_file_path)
                return video_file_name, file_size
            else:
                return video_file_name, None
        return None, None

    # 使用ThreadPoolExecutor进行多线程并行处理
    video_names = [item["video"] for item in data]
    set_id = data[0]['set']
    video_names = list(set(video_names))
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(check_video_file_size, video_name, set_id) for video_name in video_names]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing", unit="file"
        ):
            result = future.result()
            if result[1]:
                video_file_sizes.append(result[1])
                if result[1] > 50:
                    large_video_file_names.append(result[0])


    distribution = {
        "0-0.1": 0,
        "0.1-0.5": 0,
        "0.5-1": 0,
        "1-10": 0,
        "10-20": 0,
        "20-30": 0,
        "30-40": 0,
        "40-50": 0,
        "50-100": 0,
        "100-200": 0,
        "200-500": 0,
        "500-1000": 0,
        ">1000": 0,
    }

    for length in video_file_sizes:
        if length <= 0.1:
            distribution["0-0.1"] += 1
        elif length <= 0.5:
            distribution["0.1-0.5"] += 1
        elif length <= 1:
            distribution["0.5-1"] += 1
        elif length <= 10:
            distribution["1-10"] += 1
        elif length <= 20:
            distribution["10-20"] += 1
        elif length <= 30:
            distribution["20-30"] += 1
        elif length <= 40:
            distribution["30-40"] += 1
        elif length <= 50:
            distribution["40-50"] += 1
        elif length <= 100:
            distribution["50-100"] += 1
        elif length <= 200:
            distribution["100-200"] += 1
        elif length <= 500:
            distribution["200-500"] += 1
        elif length <= 1000:
            distribution["500-1000"] += 1
        else:
            distribution[">1000"] += 1

    print(f"File size distribution of {json_file_path}:")
    for key, value in distribution.items():
        print(f"{key}: {value}")


with open('lost_file_name.txt', 'w') as file:
    for video_file_name in large_video_file_names:
        file.write(video_file_name + '\n')

print(f'文件名已保存到 lost_file_name.txt')
