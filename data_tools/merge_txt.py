import os

def merge_txt_files(input_files, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for fname in input_files:
            if os.path.isfile(fname):
                with open(fname, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
                    outfile.write("\n")  # 添加换行符以分隔文件内容
            else:
                print(f"文件 {fname} 不存在，跳过该文件。")

if __name__ == "__main__":
    # 输入文件列表
    input_files = [
    'lost_file_name1.txt', 
    'lost_file_name2.txt',
    'lost_file_name3.txt',
    'lost_file_name4.txt',
    'lost_file_name5.txt',
    ]
    # 输出文件名
    output_file = 'lost_file_name.txt'
    
    merge_txt_files(input_files, output_file)
    print(f"文件已合并到 {output_file}")
