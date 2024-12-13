import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import transformers
from PIL import Image
from tqdm import tqdm
import math 
import torchaudio
from decord import VideoReader, cpu
from vita import conversation as conversation_lib
from vita.config import *
from vita.config import AudioFolder, FolderDict
from vita.config.dataset_config import *
from vita.constants import (
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    GLOBAL_WEIGHTS_PATH,
    IGNORE_INDEX,
    MAX_IMAGE_LENGTH,
    MIN_IMAGE_LENGTH,
)
from vita.util.data_utils_video_audio import DataArguments, LazySupervisedDataset
from vita.util.data_utils_video_audio_neg_patch import find_closest_aspect_ratio
from vita.util.mm_utils import tokenizer_image_audio_token, tokenizer_image_token

datasets = NaturalCap


output_file_path = "lost_file_name.txt"

def process_item(item):
    diagonal_list = []
    low_resolution_list = []
    if "image" in item:
        image_file = item["image"]
        if isinstance(image_file, str):
            image_file = [image_file]
        set_id = item["set"]
        if isinstance(set_id, str):
            set_id = [set_id]
        for k, img_file in enumerate(image_file):
            image_directory = FolderDict[set_id[k]]
            img_path = os.path.join(image_directory, img_file.replace("\\", "/"))
            with Image.open(os.path.join(img_path)) as img:
                img.convert("RGB")
                diagonal = math.sqrt(img.size[0]**2 + img.size[1]**2)
                diagonal_list.append(diagonal)
#                if diagonal > 100 and diagonal < 200:
                if diagonal < 10:
                    print(img_path)
                    print(item)
                    low_resolution_list.append(img_file)

    return diagonal_list, low_resolution_list

lost_files = []
for dataset in datasets:
    json_file_path = dataset["chat_path"]

    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    len_list = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_item, item) for item in data]
        for future in tqdm(as_completed(futures), total=len(futures)):
            lens, low_reso = future.result()
            len_list += lens
            lost_files += low_reso

    # assert len(len_list) == len(data)

    distribution = {
        "0-100": 0,
        "100-200": 0,
        "200-300": 0,
        "300-400": 0,
        "400-500": 0,
        "500-600": 0,
        "600-700": 0,
        "700-800": 0,
        "800-900": 0,
        "900-1000": 0,
        "1000-1500": 0,
        "1500-2000": 0,
        "2000-2500": 0,
        "2500-3000": 0,
        "3000-3500": 0,
        "3500-4000": 0,
        "4000-4500": 0,
        "4500-5000": 0,
        "5000-5500": 0,
        "5500-6000": 0,
        "6000-6500": 0,
        "6500-7000": 0,
        "7000-7500": 0,
        "7500-8000": 0,
        "8000-8500": 0,
        "8500-9000": 0,
        "9000-9500": 0,
        "9500-10000": 0,
        ">10000": 0,
    }

    for length in len_list:
        if length <= 100:
            distribution["0-100"] += 1
        elif length <= 200:
            distribution["100-200"] += 1
        elif length <= 300:
            distribution["200-300"] += 1
        elif length <= 400:
            distribution["300-400"] += 1
        elif length <= 500:
            distribution["400-500"] += 1
        elif length <= 600:
            distribution["500-600"] += 1
        elif length <= 700:
            distribution["600-700"] += 1
        elif length <= 800:
            distribution["700-800"] += 1
        elif length <= 900:
            distribution["800-900"] += 1
        elif length <= 1000:
            distribution["900-1000"] += 1
        elif length <= 1500:
            distribution["1000-1500"] += 1
        elif length <= 2000:
            distribution["1500-2000"] += 1
        elif length <= 2500:
            distribution["2000-2500"] += 1
        elif length <= 3000:
            distribution["2500-3000"] += 1
        elif length <= 3500:
            distribution["3000-3500"] += 1
        elif length <= 4000:
            distribution["3500-4000"] += 1
        elif length <= 4500:
            distribution["4000-4500"] += 1
        elif length <= 5000:
            distribution["4500-5000"] += 1
        elif length <= 5500:
            distribution["5000-5500"] += 1
        elif length <= 6000:
            distribution["5500-6000"] += 1
        elif length <= 6500:
            distribution["6000-6500"] += 1
        elif length <= 7000:
            distribution["6500-7000"] += 1
        elif length <= 7500:
            distribution["7000-7500"] += 1
        elif length <= 8000:
            distribution["7500-8000"] += 1
        elif length <= 8500:
            distribution["8000-8500"] += 1
        elif length <= 9000:
            distribution["8500-9000"] += 1
        elif length <= 9500:
            distribution["9000-9500"] += 1
        elif length <= 10000:
            distribution["9500-10000"] += 1
        else:
            distribution[">10000"] += 1

    print(f"Length distribution of {json_file_path}:")
    for key, value in distribution.items():
        print(f"{key}: {value}")

# 将丢失的文件名写入到lost_file_name.txt中
with open(output_file_path, "w", encoding="utf-8") as f:
    for file_name in lost_files:
        f.write(file_name + "\n")

print(f"检查完成，共有 {len(lost_files)} 个文件丢失或无法读取，结果已保存到 {output_file_path}")

# with open(out_file_name, 'w', encoding='utf-8') as file:
#    json.dump(long_json*10, file, ensure_ascii=False, indent=4)

# print(f"处理完成，大于{token_thre}的已保存到{out_file_name}")


