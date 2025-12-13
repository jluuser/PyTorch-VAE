# check_structure.py
import os
import numpy as np
import glob

# 你的数据目录
NPY_DIR = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/prp-dataset/curves_npy_CATH"

print(f"Checking files in: {NPY_DIR}")

files = glob.glob(os.path.join(NPY_DIR, "*.npy"))[:3] # 只看前3个

if not files:
    print("Error: No .npy files found!")
else:
    for f in files:
        print(f"\nFile: {os.path.basename(f)}")
        try:
            # 关键修改：开启 allow_pickle=True
            data = np.load(f, allow_pickle=True)
            
            # 检查是否是对象数组
            if data.dtype == 'O':
                # 提取包裹的内容
                content = data.item()
                print(f"  - Data Type: Python Object ({type(content)})")
                
                if isinstance(content, dict):
                    print(f"  - Keys found: {list(content.keys())}")
                    # 打印每个 Key 的形状
                    for k, v in content.items():
                        if hasattr(v, 'shape'):
                            print(f"    -> Key '{k}': shape={v.shape}")
                        else:
                            print(f"    -> Key '{k}': type={type(v)}")
                else:
                    print(f"  - Content: {content}")
            else:
                print(f"  - Data Type: Pure Array")
                print(f"  - Shape: {data.shape}")
                
        except Exception as e:
            print(f"Error: {e}")