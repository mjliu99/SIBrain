import h5py

# 尝试使用 h5py 读取
try:
    with h5py.File('md_AAL_0.4.mat', 'r') as f:
        # 打印文件中的所有键
        print("Keys in HDF5 file:", list(f.keys()))

        # 查看数据的内容（假设数据存储在某个字段中）
        data = f['label'][:]  # 如果 'label' 是存储的数据字段
        print(data)

        # 查看其他字段的内容
        graph_struct = f['graph_struct'][:]
        print(graph_struct)
except Exception as e:
    print(f"Error reading the file: {e}")
