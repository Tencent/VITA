import os
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def convert_file(file_name):
    file_name = 'LLaVA-Video-178K/' + file_name

    if not os.path.exists(file_name):
        return f"文件不存在: {file_name}"

    # 定义临时文件名
    temp_file_name = file_name + '.temp.mp4'

    # 执行ffmpeg转换
    command = [
        'ffmpeg', '-i', file_name, '-c:v', 'libx264', '-crf', '23', '-preset', 'medium', temp_file_name
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 替换原始文件
    os.replace(temp_file_name, file_name)
    return f"转换完成: {file_name}"

# 读取文件名列表
with open('mmco_files_part_4.txt', 'r') as file:
    file_names = file.readlines()

# 去除每行末尾的换行符
file_names = [file_name.strip() for file_name in file_names]

# 使用ThreadPoolExecutor进行多线程处理
with ThreadPoolExecutor(max_workers=1) as executor:  # 你可以根据你的CPU核心数调整max_workers
    futures = {executor.submit(convert_file, file_name): file_name for file_name in file_names}

    for future in tqdm(as_completed(futures), total=len(file_names)):
        result = future.result()
        if result:
            print(result)

print("所有文件已转换完成。")
