import os
import subprocess
from tqdm import tqdm

# 读取文件名列表
with open('mmco_files.txt', 'r') as file:
    file_names = file.readlines()

# 去除每行末尾的换行符
file_names = [file_name.strip() for file_name in file_names]

# 遍历每个文件名并执行ffmpeg转换
for file_name in tqdm(file_names):
    file_name = 'LLaVA-Video-178K/' + file_name

    if not os.path.exists(file_name) and os.path.exists(file_name + '.mp4'):
        #import pdb; pdb.set_trace()
        os.replace(file_name + '.mp4', file_name)
        print(f"{file_name} 已转换")
        continue  # 跳过不存在的文件

print("所有文件已转换完成。")

