import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Hướng dẫn sử dụng HSTrans với Encoder BRICS Motif \n",
    "\n",
    "Notebook này minh họa và so sánh hai chế độ mã hóa SMILES của thuốc cho bài toán dự đoán tác dụng phụ:\n",
    "1. **BPE Subword Encoder (Gốc)**: Cắt chuỗi SMILES theo tần suất ký tự (vocab=2686, độ dài max=50).\n",
    "2. **BRICS Motif Encoder (Mới)**: Cắt chuỗi SMILES thành các nhóm chức năng hóa học thực sự bằng thuật toán BRICS (vocab=799, độ dài max=20).\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Môi trường và Import\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "import torch\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "# Import Subword (gốc)\n",
    "from Net import drug2emb_encoder as bpe_encode\n",
    "\n",
    "# Import Motif (mới)\n",
    "from Net_BRICS import drug2motif_encoder as brics_encode\n",
    "from Net_BRICS import compare_encoders, Trans_BRICS"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. So sánh cách mã hóa SMILES\n",
    "\n",
    "Chúng ta sẽ xem cách mã hóa của 2 phương pháp với một số loại thuốc phổ biến."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Gọi hàm so sánh trực quan đã được viết trong Net_BRICS\n",
    "compare_encoders()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Thử nghiệm Forward Pass của Model BRICS Mới\n",
    "\n",
    "Khởi tạo model `Trans_BRIC` và truyền dữ liệu ảo (dummy data) để kiểm tra luồng hoạt động."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Trans_BRICS().to('cpu')\n",
    "print(f\"Tổng số tham số: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\")\n",
    "\n",
    "# Khởi tạo dữ liệu mẫu: Batch = 2\n",
    "drug_ids = torch.randint(0, 799, (2, 20))       # [B, 20]\n",
    "se_ids = torch.randint(0, 2686, (2, 50))       # [B, 50]\n",
    "drug_mask = torch.ones(2, 20, dtype=torch.long) # [B, 20]\n",
    "se_mask = torch.ones(2, 50, dtype=torch.long)   # [B, 50]\n",
    "\n",
    "# Chạy qua model\n",
    "score, _, _ = model(drug_ids, se_ids, drug_mask, se_mask)\n",
    "\n",
    "print(f\"\\nInput Drug shape: {drug_ids.shape}\")\n",
    "print(f\"Input Side Effect shape: {se_ids.shape}\")\n",
    "print(f\"Output Score shape: {score.shape}\")\n",
    "print(\"Sample Output:\", score.detach().flatten().numpy())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Hướng dẫn chạy Huấn luyện (Training)\n",
    "\n",
    "Để tích hợp hoàn toàn mô hình BRICS vào vòng lặp huấn luyện, bạn cần điều chỉnh trong file `main.py`:\n",
    "\n",
    "1. **Mở file `main.py`**\n",
    "2. **Tìm dòng import (khoảng line 24):**\n",
    "   ```python\n",
    "   from Net import Trans, drug2emb_encoder\n",
    "   ```\n",
    "3. **Thay bằng:**\n",
    "   ```python\n",
    "   from Net_BRICS import Trans_BRICS as Trans\n",
    "   from Net_BRICS import drug2motif_encoder as drug2emb_encoder\n",
    "   ```\n",
    "4. Các thông số input layer (ví dụ: `d_v, input_mask_d = drug2emb_encoder(d)`) có thể giữ nguyên vì API đã được thiết kế Drop-in thay thế hoàn hảo.\n",
    "5. Sau đó, chạy lại lệnh `python main.py` bình thường."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("HSTrans_BRICS_Demo.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("Created HSTrans_BRICS_Demo.ipynb")
