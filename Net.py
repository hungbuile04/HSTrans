import math
import torch.nn as nn
import torch.nn.functional as F
from Encoder import Encoder_MultipleLayers, Embeddings
import torch
import numpy as np
import pandas as pd
import codecs

# DEVICE global
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class Trans(torch.nn.Module):
    def __init__(self):
        super(Trans, self).__init__()

        # dùng device global
        self.device = DEVICE

        # activation and regularization
        self.relu = nn.ReLU()

        input_dim_drug = 2586
        transformer_emb_size_drug = 304
        transformer_dropout_rate = 0.1
        transformer_n_layer_drug = 8
        transformer_intermediate_size_drug = 512
        transformer_num_attention_heads_drug = 8
        transformer_attention_probs_dropout = 0.1
        transformer_hidden_dropout_rate = 0.1

        # 嵌入编码层
        self.embDrug = Embeddings(input_dim_drug,
                              transformer_emb_size_drug,
                              50,
                              transformer_dropout_rate)

        self.embSide = Embeddings(input_dim_drug,
                              transformer_emb_size_drug,
                              50,
                              transformer_dropout_rate)

        # Transformer层
        self.encoderDrug = Encoder_MultipleLayers(transformer_n_layer_drug,
                                              transformer_emb_size_drug,
                                              transformer_intermediate_size_drug,
                                              transformer_num_attention_heads_drug,
                                              transformer_attention_probs_dropout,
                                              transformer_hidden_dropout_rate)

        self.encoderSide = Encoder_MultipleLayers(transformer_n_layer_drug,
                                              transformer_emb_size_drug,
                                              transformer_intermediate_size_drug,
                                              transformer_num_attention_heads_drug,
                                              transformer_attention_probs_dropout,
                                              transformer_hidden_dropout_rate)

        # 位置编码层
        self.position_embeddings = nn.Embedding(500, 200)

        self.dropout = 0.1

        self.decoder = nn.Sequential(
            nn.Linear(23040, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 1)
        )

        self.icnn = nn.Conv2d(1, 10, 3, padding=0)
        self.CrossAttention = False

        # chuyển toàn bộ module lên device ngay khi khởi tạo
        # (điều này đảm bảo tất cả tham số/embedding nằm trên cùng device)
        self.to(self.device)

    def crossAttention_simple(self, x_d, x_e, mask_d=None, mask_e=None, return_attn=False):
        """
        Cross-attention đơn giản (single-head) giữa Drug (x_d) và SE (x_e).

        x_d: [B, Nd, D]  - drug embeddings
        x_e: [B, Ns, D]  - SE embeddings
        mask_d: [B, Nd]  - 1 = real / 0 = pad
        mask_e: [B, Ns]  - 1 = real / 0 = pad
        return_attn: nếu True → trả cả attention weights xoay chiều

        Trả về:
            x_d_new: drug embeddings đã contextualized bằng SE
            x_e_new: SE embeddings đã contextualized bằng drug
        """

        B, Nd, D = x_d.shape
        _, Ns, _ = x_e.shape

        # ---- 1) Chuẩn hóa mask ----
        # mask Additive kiểu HSTrans: (1-mask)*(-10000)
        if mask_d is not None and mask_d.dim() == 4:    # [B,1,1,Nd]
            mask_d = (mask_d.squeeze(1).squeeze(1) == 0)  # True = pad
        elif mask_d is not None:
            mask_d = (mask_d == 0)

        if mask_e is not None and mask_e.dim() == 4:    # [B,1,1,Ns]
            mask_e = (mask_e.squeeze(1).squeeze(1) == 0)
        elif mask_e is not None:
            mask_e = (mask_e == 0)

        # fallback: không mask
        if mask_d is None:
            mask_d = torch.zeros((B, Nd), dtype=torch.bool, device=x_d.device)
        if mask_e is None:
            mask_e = torch.zeros((B, Ns), dtype=torch.bool, device=x_e.device)

        # ---- 2) Drug (Q) → SE (K,V) ----
        # Attention(Q=x_d, K=x_e, V=x_e)
        scores_de = torch.matmul(x_d, x_e.transpose(-2, -1)) / math.sqrt(D)   # [B,Nd,Ns]
        # mask các vị trí SE = pad
        scores_de = scores_de.masked_fill(mask_e.unsqueeze(1), float('-1e9'))
        attn_de = torch.softmax(scores_de, dim=-1)                             # [B,Nd,Ns]
        context_d = torch.matmul(attn_de, x_e)                                 # [B,Nd,D]
        x_d_new = x_d + context_d                                              # residual
        x_d_new = F.layer_norm(x_d_new, (D,))                                  # LN

        # ---- 3) SE (Q) → Drug (K,V) ----
        scores_ed = torch.matmul(x_e, x_d.transpose(-2, -1)) / math.sqrt(D)   # [B,Ns,Nd]
        scores_ed = scores_ed.masked_fill(mask_d.unsqueeze(1), float('-1e9'))
        attn_ed = torch.softmax(scores_ed, dim=-1)
        context_e = torch.matmul(attn_ed, x_d)
        x_e_new = x_e + context_e
        x_e_new = F.layer_norm(x_e_new, (D,))

        if return_attn:
            return x_d_new, x_e_new, attn_de, attn_ed

        return x_d_new, x_e_new

    def forward(self, Drug, SE, DrugMask, SEMsak):

        batch = Drug.size(0)

        # đảm bảo các tensors input là LongTensor/FloatTensor rồi chuyển sang device
        # (nếu bạn đã chuyển batch trong training loop thì .to() ở đây vẫn an toàn)
        Drug = Drug.long().to(self.device)
        DrugMask = DrugMask.long().to(self.device)
        DrugMask = DrugMask.unsqueeze(1).unsqueeze(2)
        DrugMask = (1.0 - DrugMask) * -10000.0

        emb = self.embDrug(Drug)  # embDrug đã ở device vì self.to(self.device) ở __init__
        encoded_layers = self.encoderDrug(emb.float(), DrugMask.float(), False)
        x_d = encoded_layers

        SE = SE.long().to(self.device)
        SEMsak = SEMsak.long().to(self.device)
        SEMsak = SEMsak.unsqueeze(1).unsqueeze(2)
        SEMsak = (1.0 - SEMsak) * -10000.0

        embE = self.embSide(SE)
        encoded_layers = self.encoderSide(embE.float(), SEMsak.float(), False)
        x_e = encoded_layers

        if self.CrossAttention:
            # x_d, x_e = self.crossAttentionencoder([x_d.float(), x_e.float()], DrugMask.float(), True)
            # chuyển mask additive thành mask nhị phân 1/0
            mask_d = (DrugMask.squeeze(1).squeeze(1) != -10000).long()
            mask_e = (SEMsak.squeeze(1).squeeze(1) != -10000).long()

            x_d, x_e = self.crossAttention_simple(
                x_d.float(),
                x_e.float(),
                mask_d=mask_d,
                mask_e=mask_e,
                return_attn=False      # đặt True nếu bạn muốn attention weights
            )

        D = x_d.shape[-1]
        # ===== 1) Tính weight w_ij =====
        scores = torch.matmul(x_d, x_e.transpose(-2, -1)) / math.sqrt(D)  # [B, Nd, Ns]
        # mask padding SE
        if mask_e is not None:
            scores = scores.masked_fill(mask_e.unsqueeze(1) == 0, -1e9)
        w = torch.softmax(scores, dim=-1)   # [B, Nd, Ns]

        # interaction
        d_aug = torch.unsqueeze(x_d, 2).repeat(1, 1, 50, 1)
        e_aug = torch.unsqueeze(x_e, 1).repeat(1, 50, 1, 1)

        i = d_aug * e_aug

        # ===== 3) Weight interaction (KEY) =====
        i = i * w.unsqueeze(-1)                         # [B, Nd, Ns, D]

        i_v = i.permute(0, 3, 1, 2)
        i_v = torch.sum(i_v, dim=1)
        i_v = torch.unsqueeze(i_v, 1)
        i_v = F.dropout(i_v, p=self.dropout)

        f = self.icnn(i_v)
        f = f.view(int(batch), -1)
        score = self.decoder(f)

        return score, Drug, SE



def drug2emb_encoder(smile):
    vocab_path = 'data/drug_codes_chembl_freq_1500.txt'
    sub_csv = pd.read_csv('data/subword_units_map_chembl_freq_1500.csv')

    # 初始化一个BPE编码器，该编码器可以用于将文本进行分词或编码
    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    idx2word_d = sub_csv['index'].values  # 将所有的子结构列表给提取出来
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))  # 构造字典：让子结构与index一一对应

    max_d = 50
    t1 = dbpe.process_line(smile).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # 将该smile的子结构找到对应的index，形成一个index的ndarray
    except:
        i1 = np.array([0])

    l = len(i1)
    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))
    else:
        i = i1[:max_d]  # 进行填充
        input_mask = [1] * max_d  # 构造mask（盖住填充部分）

    return i, np.asarray(input_mask)
