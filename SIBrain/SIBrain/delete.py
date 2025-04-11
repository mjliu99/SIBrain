import os
import glob

# 设定文件模式，匹配所有 all_graphs_epoch_*.pkl 文件
file_pattern = "all_graphs_epoch_*.pkl"

# 找到所有符合模式的文件
files_to_delete = glob.glob(file_pattern)

# 删除文件
for file in files_to_delete:
    os.remove(file)
    print(f"Deleted: {file}")

print("All matching files deleted.")
