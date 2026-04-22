import numpy as np
import pickle
import os

sub_dir = '/Users/buihung/project_DL/HSTrans/HSTrans/data/sub'

print("\n=== Analyzing .npy files in sub/ ===")
for file in ['SE_sub_0.npy', 'freq_0.npy', 'SE_sub_index_50_2.npy', 'SE_sub_mask_50_2.npy']:
    path = os.path.join(sub_dir, file)
    arr = np.load(path, allow_pickle=True)
    print(f"{file}: shape {arr.shape}, type {arr.dtype}")

print("\n=== Analyzing .pkl files in sub/ ===")
path = os.path.join(sub_dir, 'my_dict_0.pkl')
try:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"my_dict_0.pkl: type {type(data)}")
    if isinstance(data, dict):
        print(f"Keys length: {len(data)}, type of first val: {type(data[next(iter(data))])}")
except Exception as e:
    print(f"Failed to load pkl: {e}")

