import pandas as pd
import numpy as np
import scipy.io as sio
import pickle
import os

data_dir = '/Users/buihung/project_DL/HSTrans/HSTrans/data'

print("=== Analyzing .csv files ===")
for file in ['drug_SMILES_750.csv', 'drug_SMILES_759.csv', 'subword_units_map_chembl_freq_1500.csv']:
    path = os.path.join(data_dir, file)
    df = pd.read_csv(path)
    print(f"\n{file}: shape {df.shape}")
    print(df.head(2))

print("\n=== Analyzing .npy files ===")
for file in ['SE_sub_index_50.npy', 'SE_sub_mask_50.npy']:
    path = os.path.join(data_dir, file)
    arr = np.load(path)
    print(f"\n{file}: shape {arr.shape}, type {arr.dtype}")

print("\n=== Analyzing .mat files ===")
for file in ['raw_frequency_750.mat', 'side_effect_label_750.mat']:
    path = os.path.join(data_dir, file)
    mat = sio.loadmat(path)
    print(f"\n{file}: Keys:")
    for k, v in mat.items():
        if not k.startswith('__'):
            print(f"  {k}: type {type(v)}, shape {v.shape if hasattr(v, 'shape') else 'N/A'}")

print("\n=== Analyzing .pkl files ===")
path = os.path.join(data_dir, 'drug_side.pkl')
try:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"\n{path}: type {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())[:5]} ... ({len(data)} total)")
        # Show structure of first item
        first_k = next(iter(data))
        v = data[first_k]
        print(f"Sample value for '{first_k}': type {type(v)}, len {len(v) if hasattr(v, '__len__') else 'N/A'}")
        if isinstance(v, (list, np.ndarray)) and len(v) > 0:
            print(f"Sample elem: {v[0]}")
    elif isinstance(data, list):
        print(f"List length: {len(data)}")
        if len(data) > 0:
            print(f"First element type: {type(data[0])}")
    elif hasattr(data, 'shape'):
        print(f"Array shape: {data.shape}")
except Exception as e:
    print(f"Failed to load pkl: {e}")

