import scipy.io as sio
import pandas as pd
import numpy as np
import os

def decode_mat_strings(arr):
    """Attempt to decode MATLAB string arrays into Python lists of strings."""
    try:
        # Check if it's an array of objects/strings
        if arr.dtype == np.object_:
            return [str(item[0]) if len(item) > 0 else "" for item in arr.flatten()]
        elif issubclass(arr.dtype.type, np.character):
            return [str(item) for item in arr.flatten()]
    except Exception:
        pass
    return arr.flatten()

def convert_mat_file(file_path, output_dir):
    print(f"Bắt đầu chuyển đổi: {file_path}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    mat_data = sio.loadmat(file_path)
    base_name = os.path.basename(file_path).replace('.mat', '')
    
    # We will try to combine them into pandas DataFrames if possible, 
    # otherwise save individual keys
    arrays = {k: v for k, v in mat_data.items() if not k.startswith('__')}
    
    for key, value in arrays.items():
        if isinstance(value, np.ndarray):
            out_file = os.path.join(output_dir, f"{base_name}_{key}.csv")
            
            # If it's a 1D column/row array of strings or numbers
            if value.shape[1] == 1 or value.shape[0] == 1:
                decoded = decode_mat_strings(value)
                df = pd.DataFrame({key: decoded})
                df.to_csv(out_file, index=False)
                print(f"  - Đã lưu {key} (Kích thước: {value.shape}) vào {out_file}")
                
            # If it's a 2D matrix (like R matrices)
            elif len(value.shape) == 2:
                # If we have dimension names (like 'drugs' and 'sideeffects' in raw_frequency_750)
                # We can try to use them as index/columns if sizes match
                index_labels = None
                col_labels = None
                
                if base_name == 'raw_frequency_750' and key == 'R':
                    if 'drugs' in arrays and len(arrays['drugs']) == value.shape[0]:
                        index_labels = decode_mat_strings(arrays['drugs'])
                    if 'sideeffects' in arrays and len(arrays['sideeffects']) == value.shape[1]:
                        col_labels = decode_mat_strings(arrays['sideeffects'])
                        
                if base_name == 'side_effect_label_750' and key == 'node_label':
                    if 'side_effect' in arrays and len(arrays['side_effect']) == value.shape[0]:
                        index_labels = decode_mat_strings(arrays['side_effect'])
                
                df = pd.DataFrame(value, index=index_labels, columns=col_labels)
                # Keep index if we successfully applied labels
                df.to_csv(out_file, index=(index_labels is not None))
                print(f"  - Đã lưu ma trận {key} (Kích thước: {value.shape}) vào {out_file}")
            else:
                print(f"  * Bỏ qua {key} vì kích thước không phải 1D/2D: {value.shape}")

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    output_dir = os.path.join(data_dir, 'csv_exports')
    
    mat_files = [
        os.path.join(data_dir, 'raw_frequency_750.mat'),
        os.path.join(data_dir, 'side_effect_label_750.mat')
    ]
    
    for mat_file in mat_files:
        if os.path.exists(mat_file):
            convert_mat_file(mat_file, output_dir)
        else:
            print(f"Không tìm thấy file: {mat_file}")
    
    print(f"\nHoàn tất! Các file CSV được lưu tại thư mục: {output_dir}")
