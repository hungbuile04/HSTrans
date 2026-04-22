import pandas as pd
import numpy as np
import os

csv_dir = '/Users/buihung/project_DL/HSTrans/HSTrans/data/csv_exports'

print("=== Analyzing raw_frequency_750_drugs ===")
df_drugs = pd.read_csv(f"{csv_dir}/raw_frequency_750_drugs.csv")
print(df_drugs.head(3))
print("Total drugs:", len(df_drugs))

print("\n=== Analyzing raw_frequency_750_sideeffects ===")
df_se = pd.read_csv(f"{csv_dir}/raw_frequency_750_sideeffects.csv")
print(df_se.head(3))
print("Total side effects (from raw_freq):", len(df_se))

print("\n=== Analyzing raw_frequency_750_R ===")
# Due to the header/index preservation
R_df = pd.read_csv(f"{csv_dir}/raw_frequency_750_R.csv", index_col=0)
print("R Matrix shape:", R_df.shape)
print("Index matches drugs (first 3)?", R_df.index[:3].tolist() == df_drugs.iloc[:3,0].tolist())
print("Columns match SE (first 3)?", R_df.columns[:3].tolist() == df_se.iloc[:3,0].tolist())
# Check the values in R
unique_vals = np.unique(R_df.values)
print(f"Unique values in R (up to 20): {unique_vals[:20]}")
print(f"Percentage of entries that are non-zero: {(R_df.values != 0).mean() * 100:.2f}%")

print("\n=== Analyzing side_effect_label_750_node_label ===")
# node label 
node_label_df = pd.read_csv(f"{csv_dir}/side_effect_label_750_node_label.csv", index_col=0)
print("node_label_df Matrix shape:", node_label_df.shape)
print("First 3 index names:", node_label_df.index[:3].tolist())
unique_vals_nl = np.unique(node_label_df.values)
print(f"Unique values in node_label (up to 20): {unique_vals_nl[:20]}")
print(f"Percentage of entries that are non-zero: {(node_label_df.values != 0).mean() * 100:.2f}%")

