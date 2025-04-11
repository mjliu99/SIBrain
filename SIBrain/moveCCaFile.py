import os
import re
import shutil

# 当前目录
current_dir = os.getcwd()

# 正则匹配 pattern
pattern = re.compile(r'embeddings_and_labels_epoch_\d+\.pt$')

# 找到所有符合的文件
matching_files = [f for f in os.listdir(current_dir) if pattern.match(f)]

# 创建目标目录
pt_dir = os.path.join(current_dir, 'ABIDE28001')
os.makedirs(pt_dir, exist_ok=True)

# 移动文件
for filename in matching_files:
    src = os.path.join(current_dir, filename)
    dst = os.path.join(pt_dir, filename)
    shutil.move(src, dst)
    print(f"Moved {filename} to pt/")

print("全部文件已移动完成。")
