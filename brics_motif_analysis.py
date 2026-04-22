"""
BRICS Motif Analysis - HSTrans Drug Dataset (Optimized v3)
===========================================================
Sử dụng FragmentOnBRICSBonds (nhanh) thay vì BRICSDecompose (chậm).
"""

import os
import re
import signal
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, Descriptors, AllChem, Fragments

# Tắt hoàn toàn log RDKit
RDLogger.logger().setLevel(RDLogger.CRITICAL)
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SMILES_FILE = os.path.join(BASE_DIR, "data", "drug_SMILES_750.csv")
R_MATRIX_FILE = os.path.join(BASE_DIR, "data", "csv_exports", "raw_frequency_750_R.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "csv_exports")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def remove_dummy_atoms(mol):
    """
    Xóa tất cả dummy atoms (atomic number = 0, ký hiệu *) khỏi đồ thị phân tử.
    Thao tác trực tiếp trên Mol object → tránh lỗi hóa trị H khi dùng regex.
    """
    rw = Chem.RWMol(mol)
    # Tìm tất cả dummy atoms (AtomicNum == 0)
    dummy_ids = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
    # Xóa từ index lớn → nhỏ để không bị lệch chỉ số
    for idx in sorted(dummy_ids, reverse=True):
        rw.RemoveAtom(idx)
    try:
        Chem.SanitizeMol(rw)
        return rw.GetMol()
    except Exception:
        return None


def fragment_on_brics(smiles: str):
    """
    Phân tách SMILES bằng FragmentOnBRICSBonds (NHANH).
    Xóa dummy atoms trực tiếp trên đồ thị phân tử (Mol object)
    thay vì thao tác trên chuỗi SMILES → tránh lỗi hóa trị H.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    try:
        # Cắt liên kết BRICS -> 1 phân tử lớn chứa nhiều fragment
        fragmented = AllChem.FragmentOnBRICSBonds(mol)
        # Tách thành từng mảnh Mol riêng lẻ
        frag_mols = Chem.GetMolFrags(fragmented, asMols=True)
    except Exception:
        return []

    cleaned = []
    for frag_mol in frag_mols:
        clean_mol = remove_dummy_atoms(frag_mol)
        if clean_mol is None:
            continue
        if clean_mol.GetNumAtoms() < 2:
            continue
        try:
            canon = Chem.MolToSmiles(clean_mol)
            if canon and len(canon) > 1:
                cleaned.append(canon)
        except Exception:
            continue

    return cleaned


# ===========================================================
# MAIN
# ===========================================================
print("=" * 60)
print("BRICS Motif Analysis (Optimized v3)")
print("=" * 60)

df_smiles = pd.read_csv(SMILES_FILE, header=None, names=["drug_name", "smiles"])
print(f"\nĐã đọc {len(df_smiles)} thuốc")

drug_motifs = {}
motif_counter = Counter()
motif_drug_map = defaultdict(set)
failed_drugs = []

for idx, row in df_smiles.iterrows():
    drug_name = row["drug_name"]
    smiles = row["smiles"]

    if idx % 100 == 0:
        print(f"  Processing {idx+1}/{len(df_smiles)}...")

    motifs = fragment_on_brics(smiles)
    if not motifs:
        failed_drugs.append(drug_name)
        drug_motifs[drug_name] = []
        continue

    drug_motifs[drug_name] = motifs
    motif_counter.update(motifs)
    for m in set(motifs):
        motif_drug_map[m].add(drug_name)

total_drugs = len(df_smiles)
success_drugs = total_drugs - len(failed_drugs)
total_unique_motifs = len(motif_counter)

print(f"\n--- Kết quả ---")
print(f"  Thành công:     {success_drugs}/{total_drugs}")
print(f"  Thất bại:       {len(failed_drugs)}")
print(f"  Motif unique:   {total_unique_motifs}")
print(f"  Tổng fragments: {sum(motif_counter.values())}")

motif_counts = [len(v) for v in drug_motifs.values() if len(v) > 0]
if motif_counts:
    print(f"  Motif/thuốc:    mean={np.mean(motif_counts):.1f}, "
          f"min={np.min(motif_counts)}, max={np.max(motif_counts)}, "
          f"median={np.median(motif_counts):.0f}")

if failed_drugs:
    print(f"\n  Failed (top 10): {failed_drugs[:10]}")


# --- Export 1: Motif theo từng thuốc ---
rows = []
for drug_name, motifs in drug_motifs.items():
    rows.append({
        "drug_name": drug_name,
        "num_motifs": len(motifs),
        "motifs": " | ".join(motifs) if motifs else ""
    })
df_pd = pd.DataFrame(rows)
p1 = os.path.join(OUTPUT_DIR, "brics_motif_per_drug.csv")
df_pd.to_csv(p1, index=False)
print(f"\n[Saved] {p1}")


# --- Export 2: Tần suất motif ---
rows = []
for motif, count in motif_counter.most_common():
    mol = Chem.MolFromSmiles(motif)
    mw = Descriptors.MolWt(mol) if mol else 0
    na = mol.GetNumAtoms() if mol else 0
    nd = len(motif_drug_map[motif])
    rows.append({
        "motif_smiles": motif,
        "total_occurrences": count,
        "num_drugs_containing": nd,
        "percent_drugs": round(nd / max(success_drugs,1) * 100, 2),
        "molecular_weight": round(mw, 2),
        "num_atoms": na
    })
df_freq = pd.DataFrame(rows)
p2 = os.path.join(OUTPUT_DIR, "brics_motif_frequency.csv")
df_freq.to_csv(p2, index=False)
print(f"[Saved] {p2}")

print(f"\n--- Top 20 Motif phổ biến nhất ---")
df_top = df_freq.sort_values("num_drugs_containing", ascending=False).head(20)
for i, (_, r) in enumerate(df_top.iterrows()):
    print(f"  {i+1:2d}. {r['motif_smiles']:35s}  "
          f"| {r['num_drugs_containing']:4d} thuốc ({r['percent_drugs']:5.1f}%)  "
          f"| MW={r['molecular_weight']:7.1f}")


# --- Export 3: Motif ↔ Side Effect ---
print(f"\n--- Tính tương quan Motif ↔ Tác dụng phụ ---")
df_R = pd.read_csv(R_MATRIX_FILE, index_col=0)
print(f"  Ma trận R: {df_R.shape}")

def normalize_name(name):
    return str(name).strip().lower().replace(".", " ")

R_name_map = {normalize_name(n): n for n in df_R.index.tolist()}

MIN_DRUGS = 5
sig_motifs = [m for m, d in motif_drug_map.items() if len(d) >= MIN_DRUGS]
sig_motifs.sort(key=lambda m: len(motif_drug_map[m]), reverse=True)
print(f"  Motif >= {MIN_DRUGS} thuốc: {len(sig_motifs)}")

motif_se_rows = []
for mi, motif in enumerate(sig_motifs):
    if mi % 50 == 0:
        print(f"  Motif-SE progress: {mi}/{len(sig_motifs)}")

    matched = []
    for d in motif_drug_map[motif]:
        norm = normalize_name(d)
        if norm in R_name_map:
            matched.append(R_name_map[norm])

    if len(matched) < 2:
        continue

    sub_R = df_R.loc[matched].values  # numpy for speed
    mean_freq = sub_R.mean(axis=0)

    nonzero_mask = mean_freq > 0
    top5_idx = np.argsort(mean_freq)[-5:][::-1]
    se_cols = df_R.columns
    top5_str = "; ".join([f"{se_cols[j]}({mean_freq[j]:.2f})" for j in top5_idx])

    motif_se_rows.append({
        "motif_smiles": motif,
        "num_drugs_matched": len(matched),
        "mean_se_frequency": round(mean_freq[nonzero_mask].mean(), 3) if nonzero_mask.any() else 0,
        "num_active_se": int(nonzero_mask.sum()),
        "top5_side_effects": top5_str
    })

df_mse = pd.DataFrame(motif_se_rows)
p3 = os.path.join(OUTPUT_DIR, "brics_motif_side_effect.csv")
df_mse.to_csv(p3, index=False)
print(f"[Saved] {p3}")


# --- Tổng kết ---
print(f"\n{'='*60}")
print(f"HOÀN TẤT")
print(f"{'='*60}")
print(f"  Thuốc:          {success_drugs}/{total_drugs}")
print(f"  Motif unique:   {total_unique_motifs}")
print(f"  Motif >= {MIN_DRUGS}:    {len(sig_motifs)}")
print(f"  Motif-SE rows:  {len(motif_se_rows)}")
print(f"\n  1. {p1}")
print(f"  2. {p2}")
print(f"  3. {p3}")
print(f"{'='*60}")
