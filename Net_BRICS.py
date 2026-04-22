"""
Net_BRICS.py  —  BRICS Motif Encoder cho HSTrans
==================================================

File này cung cấp một cách encode thuốc THAY THẾ cho encoder gốc (BPE subword).
Thiết kế drop-in: chỉ cần đổi import trong main.py là chuyển được giữa 2 encoder.

    ┌─────────────────────────────────────────────────────────────────┐
    │  ENCODER GỐC (Net.py)             │  ENCODER MỚI (Net_BRICS.py)│
    ├────────────────────────────────────┼────────────────────────────┤
    │  drug2emb_encoder(smile)           │  drug2motif_encoder(smile) │
    │  BPE subword tokenizer             │  BRICS chemical fragments  │
    │  Vocab = 2686 (ký tự ghép)         │  Vocab = ~800 (motif thật) │
    │  Seq_len = 50 tokens               │  Seq_len = 20 motifs       │
    │  Model class: Trans                │  Model class: Trans_BRICS  │
    └────────────────────────────────────┴────────────────────────────┘

HƯỚNG DẪN SỬ DỤNG:
    Trong main.py, chỉ cần thay 2 dòng import:

        # === Dùng BPE gốc ===
        # from Net import Trans, drug2emb_encoder
        # drug_encoder_fn = drug2emb_encoder
        # modeling = Trans

        # === Dùng BRICS motif ===
        from Net_BRICS import Trans_BRICS, drug2motif_encoder
        drug_encoder_fn = drug2motif_encoder
        modeling = Trans_BRICS

    Sau đó trong Data_Encoder.__getitem__, thay:
        d_v, input_mask_d = drug_encoder_fn(d)

GIẢI THÍCH CÁCH HOẠT ĐỘNG:
==========================================================

1. ENCODER GỐC (BPE Subword) — Cắt theo tần suất ký tự
   ─────────────────────────────────────────────────────
   SMILES: "CC(=O)Oc1ccccc1C(=O)O"  (Aspirin)
                │
                ▼  BPE tokenizer (subword_nmt)
   Tokens:  ["CC", "(=", "O)", "Oc", "1ccc", "cc1", "C(", "=O)", "O"]
                │
                │  → 9 token, hầu hết KHÔNG mang ý nghĩa hóa học
                │  → Vòng benzen "c1ccccc1" bị xé thành "1ccc" + "cc1"
                │  → Nhóm carbonyl "C=O" bị xé thành "(=" + "O)"
                │  → Transformer phải tốn nhiều layers để tự ghép lại
                ▼
   IDs:     [42, 88, 3, 109, 205, 17, 91, 3, 2, 0, 0, ..., 0]  (pad đến 50)


2. ENCODER MỚI (BRICS Motif) — Cắt theo quy tắc tổng hợp hóa học
   ──────────────────────────────────────────────────────────────
   SMILES: "CC(=O)Oc1ccccc1C(=O)O"  (Aspirin)
                │
                ▼  RDKit FragmentOnBRICSBonds
                │  Cắt tại các liên kết retrosynthetic (phản ứng tổng hợp ngược)
                │  → Mỗi mảnh là 1 khối chức năng hóa học hoàn chỉnh
                ▼
   Motifs: ["CC", "C=O", "c1ccccc1", "O=CO"]
                │
                │  → 4 token, MỖI token = 1 nhóm chức năng có nghĩa:
                │     "CC"         = nhóm methyl (chuỗi carbon)
                │     "C=O"        = nhóm carbonyl (hoạt tính cao)
                │     "c1ccccc1"   = vòng benzen (vòng thơm)
                │     "O=CO"       = nhóm ester (liên kết thuốc)
                │
                │  → Transformer chỉ cần 4 layers để học tương tác
                │     giữa các nhóm chức ĐÃ CÓ SẴN ý nghĩa
                ▼
   IDs:     [5, 4, 2, 12, 0, 0, 0, ..., 0]  (pad đến 20)


3. TẠI SAO BRICS TỐT HƠN CHO BÀI TOÁN DỰ ĐOÁN TÁC DỤNG PHỤ?
   ─────────────────────────────────────────────────────────────
   Tác dụng phụ của thuốc chủ yếu do CÁC NHÓM CHỨC năng gây ra,
   không phải do từng ký tự riêng lẻ trong chuỗi SMILES.

   Ví dụ: Tất cả thuốc chứa nhóm sulfonamide (S(=O)(=O)N) đều
   có nguy cơ gây dị ứng. BPE có thể cắt nhóm này thành
   "S(=", "O)(", "=O)", "N" → Transformer phải tự tìm ra mối
   liên hệ giữa 4 ký tự vô nghĩa. BRICS giữ nguyên "S(=O)(=O)N"
   thành 1 token duy nhất → Attention weight trực tiếp cho thấy
   "nhóm sulfonamide" liên quan đến "dị ứng da".

4. KIẾN TRÚC SO SÁNH
   ──────────────────

   ┌─── BPE (gốc) ────────────────────────────────────────────────┐
   │ SMILES → BPE(50 tokens) → Emb(2686,304) → Transformer×8     │
   │    → Drug[B,50,304]                                          │
   │                                                              │
   │ SE_subword(50 tokens) → Emb(2686,304) → Transformer×8       │
   │    → SE[B,50,304]                                            │
   │                                                              │
   │ Interaction: Drug⊗SE → [B,50,50,304] → sum → CNN → 23040   │
   │    → Decoder → score                                         │
   └──────────────────────────────────────────────────────────────┘

   ┌─── BRICS (mới) ──────────────────────────────────────────────┐
   │ SMILES → BRICS(20 motifs) → Emb(~800,304) → Transformer×4   │
   │    → Drug[B,20,304]                                          │
   │                                                              │
   │ SE_subword(50 tokens) → Emb(2686,304) → Transformer×8       │
   │    → SE[B,50,304]            (giữ nguyên, không đổi)        │
   │                                                              │
   │ Interaction: Drug⊗SE → [B,20,50,304] → sum → CNN → 8640    │
   │    → Decoder → score                                         │
   └──────────────────────────────────────────────────────────────┘

   Lợi ích:
   - Interaction map nhỏ hơn 2.5× (20×50 vs 50×50) → train nhanh
   - Drug encoder ít layer hơn (4 vs 8) → ít tham số
   - Attention weights giải thích được: motif nào ↔ tác dụng phụ nào
"""

import os
import re
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from Encoder import Encoder_MultipleLayers, Embeddings

# Tắt RDKit warnings
RDLogger.logger().setLevel(RDLogger.CRITICAL)

# ================================================================
# DEVICE & CONSTANTS
# ================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_ID = 0
UNK_ID = 1
MAX_MOTIFS = 20       # Covers >95% drugs (mean=5.5, median=4, p95≈15)
SE_SEQ_LEN = 50       # Giữ nguyên từ model gốc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ================================================================
# BRICS VOCABULARY — Xây dựng từ file thống kê đã tạo
# ================================================================
def _build_brics_vocab():
    """
    Đọc bảng tần suất motif từ brics_motif_frequency.csv,
    xây dựng vocab: motif_smiles → token_id.

    Vocabulary layout:
        ID 0 = [PAD]  (padding token)
        ID 1 = [UNK]  (unknown — motif không có trong vocab)
        ID 2 = motif phổ biến nhất (c1ccccc1 — vòng benzen)
        ID 3 = motif phổ biến thứ 2 (CO — nhóm methoxy)
        ...
    """
    freq_path = os.path.join(BASE_DIR, "data", "csv_exports", "brics_motif_frequency.csv")
    if not os.path.exists(freq_path):
        raise FileNotFoundError(
            f"Không tìm thấy {freq_path}.\n"
            f"Hãy chạy 'python brics_motif_analysis.py' trước để tạo file này."
        )

    df = pd.read_csv(freq_path)
    # Sắp xếp theo num_drugs_containing giảm dần (motif phổ biến nhất → ID nhỏ nhất)
    df = df.sort_values("num_drugs_containing", ascending=False).reset_index(drop=True)

    vocab = {}
    for idx, row in df.iterrows():
        vocab[row["motif_smiles"]] = idx + 2  # +2 vì PAD=0, UNK=1

    vocab_size = len(vocab) + 2
    print(f"[BRICS Vocab] Loaded {len(vocab)} motifs, vocab_size={vocab_size}")
    return vocab, vocab_size


# Global vocabulary (loaded once at import time)
BRICS_VOCAB, BRICS_VOCAB_SIZE = _build_brics_vocab()


# ================================================================
# BRICS FRAGMENTATION — Cắt SMILES thành motif
# ================================================================
def _remove_dummy_atoms(mol):
    """
    Xóa dummy atoms (atomic number = 0) khỏi đồ thị phân tử.
    Dummy atoms là các marker [1*], [2*]... mà BRICS gắn vào điểm cắt.
    Ta xóa chúng vì chỉ cần cấu trúc hóa học thực sự của mỗi mảnh.
    """
    rw = Chem.RWMol(mol)
    dummy_ids = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
    for idx in sorted(dummy_ids, reverse=True):
        rw.RemoveAtom(idx)
    try:
        Chem.SanitizeMol(rw)
        return rw.GetMol()
    except Exception:
        return None


def _smiles_to_motifs(smiles):
    """
    Phân tách 1 chuỗi SMILES thành danh sách canonical SMILES motif.

    Pipeline:
        SMILES → Mol → FragmentOnBRICSBonds → GetMolFrags
        → remove dummy atoms → canonical SMILES

    Returns: list[str] — danh sách motif SMILES, hoặc [] nếu thất bại
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    try:
        fragmented = AllChem.FragmentOnBRICSBonds(mol)
        frag_mols = Chem.GetMolFrags(fragmented, asMols=True)
    except Exception:
        return []

    motifs = []
    for frag_mol in frag_mols:
        clean = _remove_dummy_atoms(frag_mol)
        if clean is None or clean.GetNumAtoms() < 2:
            continue
        try:
            canon = Chem.MolToSmiles(clean)
            if canon and len(canon) > 1:
                motifs.append(canon)
        except Exception:
            continue
    return motifs


# ================================================================
# Pre-compute cache: SMILES → motif IDs (tránh gọi RDKit mỗi batch)
# ================================================================
_MOTIF_CACHE = {}


def drug2motif_encoder(smile, max_motifs=MAX_MOTIFS):
    """
    ★ Drop-in replacement cho drug2emb_encoder() trong Net.py ★

    Encode SMILES → BRICS motif token IDs + attention mask.

    Args:
        smile: str — chuỗi SMILES của thuốc
        max_motifs: int — độ dài tối đa chuỗi token (default=20)

    Returns:
        out:  np.array[int64] shape (max_motifs,) — motif token IDs, padded
        mask: np.array[int64] shape (max_motifs,) — attention mask (1=valid, 0=pad)

    So sánh với drug2emb_encoder gốc:
        drug2emb_encoder("CCO")   → ([42, 3, 0, ..., 0], [1, 1, 0, ..., 0])  # 50-dim
        drug2motif_encoder("CCO") → ([5, 0, 0, ..., 0],  [1, 0, 0, ..., 0])  # 20-dim
    """
    # Check cache first
    if smile in _MOTIF_CACHE:
        motif_ids = _MOTIF_CACHE[smile]
    else:
        motif_strs = _smiles_to_motifs(smile)
        motif_ids = [BRICS_VOCAB.get(m, UNK_ID) for m in motif_strs]
        if not motif_ids:
            motif_ids = [UNK_ID]  # Fallback cho thuốc vô cơ / SMILES lỗi
        _MOTIF_CACHE[smile] = motif_ids

    # Padding / Truncation
    L = min(len(motif_ids), max_motifs)
    out = np.full(max_motifs, PAD_ID, dtype=np.int64)
    out[:L] = motif_ids[:L]

    mask = np.zeros(max_motifs, dtype=np.int64)
    mask[:L] = 1

    return out, mask


# ================================================================
# SUBWORD ENCODER (import lại từ Net.py cho SE encoding)
# ================================================================
# Side effect vẫn dùng subword encoding gốc — không thay đổi
import codecs
try:
    from subword_nmt.apply_bpe import BPE
except Exception:
    class BPE:
        def __init__(self, *args, **kwargs):
            pass
        def process_line(self, line):
            if line is None:
                return ''
            return ' '.join(list(line.strip()))

_sub_csv = pd.read_csv(os.path.join(BASE_DIR, 'data/subword_units_map_chembl_freq_1500.csv'))
_tokens = _sub_csv['index'].astype(str).values
_raw_ids = _sub_csv['level_0'].astype(int).values
_words2idx_d = {t: int(i) + 2 for t, i in zip(_tokens, _raw_ids)}
_SUBWORD_VOCAB_SIZE = int(_raw_ids.max()) + 1 + 2

_bpe_codes = codecs.open(os.path.join(BASE_DIR, 'data/drug_codes_chembl_freq_1500.txt'))
_dbpe = BPE(_bpe_codes, merges=-1, separator='')


def drug2emb_encoder(smile, max_d=SE_SEQ_LEN):
    """SE subword encoder — giữ nguyên từ Net.py gốc."""
    toks = _dbpe.process_line(smile).split()
    if len(toks) == 0:
        ids = np.array([UNK_ID], dtype=np.int64)
    else:
        ids = np.array([_words2idx_d.get(t, UNK_ID) for t in toks], dtype=np.int64)
    L = min(len(ids), max_d)
    out = np.full((max_d,), PAD_ID, dtype=np.int64)
    out[:L] = ids[:L]
    mask = np.zeros((max_d,), dtype=np.int64)
    mask[:L] = 1
    return out, mask


# ================================================================
# MODEL: Trans_BRICS — Thay thế class Trans trong Net.py
# ================================================================
class Trans_BRICS(nn.Module):
    """
    HSTrans với BRICS Motif Encoder thay cho BPE Subword Encoder.

    Thay đổi so với Trans gốc:
    ────────────────────────────────────────────────────
    1. embDrug:     Embedding(2686, 304) → Embedding(~800, 304)
    2. encoderDrug: Transformer × 8 layers → Transformer × 4 layers
    3. Interaction:  [B, 50, 50, 304] → [B, 20, 50, 304]
    4. Decoder:     Linear(23040, ...) → Linear(8640, ...)
    5. embSide, encoderSide: KHÔNG ĐỔI

    API forward() giữ nguyên signature:
        score, Drug, SE = model(Drug, SE, DrugMask, SEMask)
    → Hoàn toàn tương thích với main.py training loop.
    """

    def __init__(self):
        super(Trans_BRICS, self).__init__()
        self.device = DEVICE
        self.relu = nn.ReLU()

        # ─── Hyperparameters ───
        emb_size = 304
        dropout_rate = 0.1
        intermediate_size = 512
        n_heads = 8
        attn_dropout = 0.1
        hidden_dropout = 0.1
        self.dropout = dropout_rate

        # Drug encoder dimensions
        drug_vocab = BRICS_VOCAB_SIZE           # ~800 (motif vocab)
        drug_max_pos = MAX_MOTIFS               # 20
        drug_n_layers = 4                       # Giảm từ 8 vì token đã có nghĩa

        # SE encoder dimensions (giữ nguyên gốc)
        se_vocab = _SUBWORD_VOCAB_SIZE           # 2686
        se_max_pos = SE_SEQ_LEN                  # 50
        se_n_layers = 8

        # ─── Drug Embedding + Encoder (BRICS) ───
        self.embDrug = Embeddings(drug_vocab, emb_size, drug_max_pos, dropout_rate)
        self.encoderDrug = Encoder_MultipleLayers(
            drug_n_layers, emb_size, intermediate_size,
            n_heads, attn_dropout, hidden_dropout
        )

        # ─── SE Embedding + Encoder (subword — giữ nguyên) ───
        self.embSide = Embeddings(se_vocab, emb_size, se_max_pos, dropout_rate)
        self.encoderSide = Encoder_MultipleLayers(
            se_n_layers, emb_size, intermediate_size,
            n_heads, attn_dropout, hidden_dropout
        )

        # ─── Interaction CNN ───
        # Input:  [B, 1, drug_max_pos, se_max_pos] = [B, 1, 20, 50]
        # Output: [B, 10, 18, 48]
        self.icnn = nn.Conv2d(1, 10, 3, padding=0)

        # Flatten size = 10 × (drug_max_pos - 2) × (se_max_pos - 2)
        cnn_out = 10 * (drug_max_pos - 2) * (se_max_pos - 2)  # = 8640

        # ─── Decoder ───
        self.decoder = nn.Sequential(
            nn.Linear(cnn_out, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 1)
        )

        # Cross-attention flag
        self.CrossAttention = False

        self.to(self.device)

    def forward(self, Drug, SE, DrugMask, SEMask):
        """
        Args:
            Drug:     [B, 20]  — BRICS motif token IDs (thay vì [B, 50] subword)
            SE:       [B, 50]  — SE subword token IDs (giữ nguyên)
            DrugMask: [B, 20]  — attention mask cho Drug
            SEMask:   [B, 50]  — attention mask cho SE

        Returns:
            score: [B, 1] — predicted frequency
            Drug:  [B, 20] — drug token IDs (for logging)
            SE:    [B, 50] — SE token IDs (for logging)
        """
        batch = Drug.size(0)

        # ── Prepare masks ──
        Drug = Drug.long().to(self.device)
        SE = SE.long().to(self.device)

        DrugMask = DrugMask.long().to(self.device)
        DrugMask = DrugMask.unsqueeze(1).unsqueeze(2)         # [B,1,1,20]
        DrugMask = (1.0 - DrugMask) * -10000.0

        SEMask = SEMask.long().to(self.device)
        SEMask = SEMask.unsqueeze(1).unsqueeze(2)             # [B,1,1,50]
        SEMask = (1.0 - SEMask) * -10000.0

        # ── Drug encoding (BRICS motifs) ──
        emb_d = self.embDrug(Drug)                            # [B, 20, 304]
        x_d = self.encoderDrug(emb_d.float(), DrugMask.float(), False)

        # ── SE encoding (subword — giữ nguyên) ──
        emb_e = self.embSide(SE)                              # [B, 50, 304]
        x_e = self.encoderSide(emb_e.float(), SEMask.float(), False)

        # ── Interaction map ──
        # Drug: [B, 20, 304], SE: [B, 50, 304]
        d_aug = x_d.unsqueeze(2).repeat(1, 1, SE_SEQ_LEN, 1)  # [B, 20, 50, 304]
        e_aug = x_e.unsqueeze(1).repeat(1, MAX_MOTIFS, 1, 1)   # [B, 20, 50, 304]
        interaction = d_aug * e_aug                             # [B, 20, 50, 304]

        # ── CNN processing ──
        i_v = interaction.permute(0, 3, 1, 2)                  # [B, 304, 20, 50]
        i_v = torch.sum(i_v, dim=1)                            # [B, 20, 50]
        i_v = i_v.unsqueeze(1)                                 # [B, 1, 20, 50]
        i_v = F.dropout(i_v, p=self.dropout, training=self.training)

        f = self.icnn(i_v)                                     # [B, 10, 18, 48]
        f = f.view(batch, -1)                                  # [B, 8640]

        # ── Decoder ──
        score = self.decoder(f)                                # [B, 1]

        return score, Drug, SE


# ================================================================
# UTILITY: In thống kê so sánh 2 encoder
# ================================================================
def compare_encoders(smiles_list=None):
    """
    So sánh trực quan kết quả encode giữa BPE và BRICS cho 1 số thuốc.
    Gọi hàm này để kiểm tra nhanh sự khác biệt.
    """
    if smiles_list is None:
        smiles_list = [
            ("Aspirin",     "CC(=O)Oc1ccccc1C(=O)O"),
            ("Ibuprofen",   "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
            ("Atorvastatin","CC(C)c1c(c(c(n1-c1ccc(cc1)F)c1ccccc1)C(=O)Nc1ccccc1)C(O)CC(CC(=O)O)O"),
        ]

    print("\n" + "=" * 70)
    print("SO SÁNH: BPE Subword vs BRICS Motif Encoder")
    print("=" * 70)

    for name, smi in smiles_list:
        print(f"\n{'─'*70}")
        print(f"Thuốc: {name}")
        print(f"SMILES: {smi}")

        # BPE
        bpe_ids, bpe_mask = drug2emb_encoder(smi)
        bpe_len = int(bpe_mask.sum())
        bpe_tokens = _dbpe.process_line(smi).split()[:bpe_len]

        # BRICS
        motif_ids, motif_mask = drug2motif_encoder(smi)
        motif_len = int(motif_mask.sum())
        motif_strs = _smiles_to_motifs(smi)[:motif_len]

        print(f"\n  BPE   ({bpe_len:2d} tokens, seq_len=50): {bpe_tokens}")
        print(f"  BRICS ({motif_len:2d} motifs, seq_len=20): {motif_strs}")
        print(f"  Compression: {bpe_len}→{motif_len} tokens ({(1 - motif_len/max(bpe_len,1))*100:.0f}% shorter)")

    print(f"\n{'='*70}")
    print(f"BRICS Vocab size: {BRICS_VOCAB_SIZE}")
    print(f"BPE   Vocab size: {_SUBWORD_VOCAB_SIZE}")
    print(f"{'='*70}\n")


# ================================================================
# TEST: Chạy trực tiếp file này để kiểm tra
# ================================================================
if __name__ == "__main__":
    compare_encoders()

    # Test model forward
    print("\n--- Test model forward pass ---")
    model = Trans_BRICS()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # Dummy input
    B = 4
    drug_ids = torch.randint(0, BRICS_VOCAB_SIZE, (B, MAX_MOTIFS))
    se_ids = torch.randint(0, _SUBWORD_VOCAB_SIZE, (B, SE_SEQ_LEN))
    drug_mask = torch.ones(B, MAX_MOTIFS, dtype=torch.long)
    se_mask = torch.ones(B, SE_SEQ_LEN, dtype=torch.long)

    score, _, _ = model(drug_ids, se_ids, drug_mask, se_mask)
    print(f"Input Drug:  {drug_ids.shape}")
    print(f"Input SE:    {se_ids.shape}")
    print(f"Output score: {score.shape}")
    print(f"Sample predictions: {score.flatten().tolist()}")
    print("\n✓ Model forward pass OK!")
