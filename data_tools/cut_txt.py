def split_file_into_four(input_file):
    # 读取文件的所有行
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 计算每个文件应包含的行数
    total_lines = len(lines)
    lines_per_file = total_lines // 4
    remainder = total_lines % 4

    # 分配行到四个文件中
    start = 0
    for i in range(4):
        end = start + lines_per_file + (1 if i < remainder else 0)
        output_file = f'mmco_files_part_{i + 1}.txt'
        with open(output_file, 'w', encoding='utf-8') as file:
            file.writelines(lines[start:end])
        start = end

# 调用函数进行文件切分
split_file_into_four('mmco_files.txt')
