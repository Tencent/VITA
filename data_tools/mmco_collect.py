import re

def collect_error_files(log_file_path):
    error_files = []
    with open(log_file_path, 'r', encoding='utf-8') as log_file:
        lines = log_file.readlines()
        previous_line = ""
        for line in lines:
            if "mmco: unref short failure" in line:
                if previous_line != "" and "mmco: unref short failure" not in previous_line:
                    error_files.append(previous_line)

            previous_line = line
    return error_files

def save_error_files(error_files, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for file in error_files:
            output_file.write(file)

def main():
    log_file_path = 'log.txt'
    output_file_path = 'mmco_files.txt'
    error_files = collect_error_files(log_file_path)
    save_error_files(error_files, output_file_path)
    print(f"Collected {len(error_files)} error files. Saved to {output_file_path}")

if __name__ == "__main__":
    main()
