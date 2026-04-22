# Mô tả dữ liệu — Thư mục `csv_exports`

Thư mục này chứa các file CSV được sinh ra từ hai nguồn:

1. **Chuyển đổi từ file `.mat` gốc** (bằng script `convert_mat_to_csv.py`)
2. **Phân tích BRICS Motif** (bằng script `brics_motif_analysis.py`)

Tất cả dữ liệu xoay quanh bài toán **Dự đoán tần suất tác dụng phụ của thuốc** dựa trên cấu trúc hóa học (SMILES).

---

## Tổng quan

| File | Kích thước | Dòng × Cột | Mô tả ngắn |
|------|-----------|-------------|-------------|
| `raw_frequency_750_R.csv` | 3.0 MB | 750 × 994 | Ma trận tần suất Thuốc–Tác dụng phụ |
| `raw_frequency_750_drugs.csv` | 9 KB | 750 × 1 | Danh sách tên thuốc |
| `raw_frequency_750_sideeffects.csv` | 17 KB | 994 × 1 | Danh sách tên tác dụng phụ |
| `side_effect_label_750_node_label.csv` | 501 KB | 994 × 243 | Vector đặc trưng phân cấp MedDRA |
| `side_effect_label_750_side_effect.csv` | 17 KB | 994 × 1 | Danh sách tên tác dụng phụ |
| `brics_motif_per_drug.csv` | 60 KB | 750 × 3 | Danh sách motif BRICS của từng thuốc |
| `brics_motif_frequency.csv` | 31 KB | 797 × 6 | Bảng tần suất của tất cả motif |
| `brics_motif_side_effect.csv` | 10 KB | 92 × 5 | Tương quan motif ↔ tác dụng phụ |

---

## Nhóm 1: Dữ liệu chuyển đổi từ `.mat`

### `raw_frequency_750_R.csv`

Ma trận tương tác chính giữa **750 loại thuốc** và **994 tác dụng phụ**.

- **Index (cột đầu)**: Tên thuốc (ví dụ: `levocarnitine`, `atorvastatin`)
- **Header (dòng đầu)**: Tên tác dụng phụ (ví dụ: `abdominal pain`, `headache`)
- **Giá trị ô**: Số nguyên từ `0` đến `5`
  - `0`: Không ghi nhận tác dụng phụ
  - `1`: Rất hiếm gặp
  - `2`: Hiếm gặp
  - `3`: Không phổ biến
  - `4`: Phổ biến
  - `5`: Rất phổ biến
- **Độ thưa (Sparsity)**: ~95% giá trị là 0

> Nguồn gốc: Biến `R` trong file `raw_frequency_750.mat`, được thu thập từ cơ sở dữ liệu SIDER và OFFSIDES.

### `raw_frequency_750_drugs.csv`

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `drugs` | string | Tên thuốc (generic name), dấu chấm thay khoảng trắng |

- 750 dòng, mỗi dòng tương ứng 1 hàng trong ma trận R.

### `raw_frequency_750_sideeffects.csv`

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `sideeffects` | string | Tên tác dụng phụ theo chuẩn MedDRA |

- 994 dòng, mỗi dòng tương ứng 1 cột trong ma trận R.

### `side_effect_label_750_node_label.csv`

Ma trận đặc trưng nhị phân mô tả **phân loại y khoa** của 994 tác dụng phụ theo hệ thống phân cấp MedDRA.

- **Index**: Tên tác dụng phụ
- **243 cột**: Mỗi cột đại diện cho 1 nhóm bệnh học trong cây MedDRA
- **Giá trị**: `0` hoặc `1`
  - `1`: Tác dụng phụ thuộc nhóm bệnh học này
  - `0`: Không thuộc
- **Độ thưa**: ~98.8% giá trị là 0
- **Mục đích**: Giúp mô hình học được mối tương quan ngữ nghĩa giữa các tác dụng phụ (ví dụ: "vàng da" và "viêm gan" cùng thuộc nhóm "Hepatobiliary disorders")

### `side_effect_label_750_side_effect.csv`

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `side_effect` | string | Tên tác dụng phụ (giống file `sideeffects.csv`) |

---

## Nhóm 2: Dữ liệu phân tích BRICS Motif

Được sinh ra bởi script `brics_motif_analysis.py`, sử dụng thuật toán **BRICS** (Breaking of Retrosynthetically Interesting Chemical Substructures) từ thư viện RDKit để phân tách cấu trúc hóa học của 750 thuốc thành các mảnh cấu trúc con (motif).

- **Phương pháp**: `FragmentOnBRICSBonds` — cắt liên kết tại 16 loại vị trí phản ứng tổng hợp hữu cơ
- **Tỷ lệ thành công**: 746/750 thuốc (99.5%)
- **Thất bại**: 4 hợp chất vô cơ/kim loại phóng xạ (`arsenic.trioxide`, `samarium`, `technetium`, `radium`)

### `brics_motif_per_drug.csv`

Liệt kê tất cả motif BRICS được tìm thấy trong từng thuốc.

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `drug_name` | string | Tên thuốc |
| `num_motifs` | int | Số lượng motif tìm được |
| `motifs` | string | Danh sách motif SMILES, ngăn cách bằng ` \| ` |

- 750 dòng (tất cả thuốc, kể cả thất bại — `motifs` để trống)
- Trung bình **5.5 motif/thuốc** (min=1, max=80, median=4)

Ví dụ:
```
drug_name,num_motifs,motifs
atorvastatin,7,c1ccccc1 | CC(C)C | Oc1ccccc1 | CCC=O | CC=O | O=CO | CN
```

### `brics_motif_frequency.csv`

Bảng tần suất toàn cục của **797 motif duy nhất** được tìm thấy trong toàn bộ 746 thuốc.

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `motif_smiles` | string | Chuỗi SMILES chuẩn hóa (canonical) của motif |
| `total_occurrences` | int | Tổng số lần xuất hiện (1 thuốc có thể chứa motif nhiều lần) |
| `num_drugs_containing` | int | Số thuốc chứa motif này (unique) |
| `percent_drugs` | float | Tỷ lệ % thuốc chứa motif |
| `molecular_weight` | float | Khối lượng phân tử của motif (Da) |
| `num_atoms` | int | Số nguyên tử nặng (không tính H) |

Top 5 motif phổ biến nhất:

| Motif | SMILES | Số thuốc | Ý nghĩa hóa học |
|-------|--------|----------|------------------|
| Vòng benzen | `c1ccccc1` | 202 (27.1%) | Vòng thơm cơ bản, phổ biến nhất trong dược phẩm |
| Nhóm methoxy | `CO` | 172 (23.1%) | Liên kết C-O, thường gặp trong ether/ester |
| Nhóm carbonyl | `C=O` | 149 (20.0%) | Liên kết đôi C=O, nền tảng của amid/ester/ketone |
| Nhóm ethyl | `CC` | 146 (19.6%) | Chuỗi carbon ngắn nhất |
| Chuỗi propyl | `CCC` | 113 (15.2%) | Chuỗi carbon 3-C |

### `brics_motif_side_effect.csv`

Phân tích tương quan giữa **92 motif có ý nghĩa** (xuất hiện trong ≥ 5 thuốc) với 994 tác dụng phụ. Tính trung bình tần suất tác dụng phụ trên nhóm thuốc cùng chứa 1 motif.

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `motif_smiles` | string | SMILES chuẩn hóa của motif |
| `num_drugs_matched` | int | Số thuốc chứa motif được map thành công với ma trận R |
| `mean_se_frequency` | float | Trung bình tần suất tác dụng phụ (chỉ tính SE > 0) |
| `num_active_se` | int | Số tác dụng phụ có tần suất > 0 |
| `top5_side_effects` | string | Top 5 tác dụng phụ có tần suất cao nhất, định dạng: `tên(giá_trị)` ngăn bằng `; ` |

Ví dụ cột `top5_side_effects`:
```
headache(3.45); nausea(3.21); dizziness(3.10); fatigue(2.98); vomiting(2.85)
```

---

## Cách tái tạo dữ liệu

```bash
# 1. Chuyển đổi .mat → .csv
python convert_mat_to_csv.py

# 2. Phân tích BRICS motif (cần RDKit)
python brics_motif_analysis.py
```

## Ghi chú

- Tất cả tên thuốc sử dụng dấu chấm `.` thay khoảng trắng (ví dụ: `arsenic.trioxide`)
- SMILES tuân theo chuẩn canonical của RDKit
- Giá trị tần suất trong ma trận R theo thang 0–5 từ cơ sở dữ liệu SIDER/OFFSIDES
