import math
import torch.nn as nn
import torch.nn.functional as F
from Encoder import Encoder_MultipleLayers, Embeddings
import torch
import numpy as np
import pandas as pd
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

# DEVICE global
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_ID = 0
UNK_ID = 1
MAX_D  = 50

sub_csv = pd.read_csv('data/subword_units_map_chembl_freq_1500.csv')

# token string
tokens = sub_csv['index'].astype(str).values
# raw ids (0..len-1)
raw_ids = sub_csv['level_0'].astype(int).values

# shift: token_id = raw_id + 2
words2idx_d = {t: int(i) + 2 for t, i in zip(tokens, raw_ids)}

VOCAB_SIZE = int(raw_ids.max()) + 1 + 2   # (+2 for PAD/UNK)
# nếu raw_ids là 0..2585 thì VOCAB_SIZE = 2586 + 2 = 2588

bpe_codes_drug = codecs.open('data/drug_codes_chembl_freq_1500.txt')
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')


class Trans(torch.nn.Module):
    def __init__(self):
        super(Trans, self).__init__()
        self.device = DEVICE
        self.relu = nn.ReLU()

        input_dim_drug = 2686
        transformer_emb_size_drug = 304
        D = transformer_emb_size_drug  # embedding dim

        transformer_dropout_rate = 0.1
        transformer_n_layer_drug = 8
        transformer_intermediate_size_drug = 512
        transformer_num_attention_heads_drug = 8
        transformer_attention_probs_dropout = 0.1
        transformer_hidden_dropout_rate = 0.1

        self.embDrug = Embeddings(input_dim_drug, D, 50, transformer_dropout_rate)
        self.embSide = Embeddings(input_dim_drug, D, 50, transformer_dropout_rate)

        self.encoderDrug = Encoder_MultipleLayers(
            transformer_n_layer_drug, D, transformer_intermediate_size_drug,
            transformer_num_attention_heads_drug, transformer_attention_probs_dropout,
            transformer_hidden_dropout_rate
        )
        self.encoderSide = Encoder_MultipleLayers(
            transformer_n_layer_drug, D, transformer_intermediate_size_drug,
            transformer_num_attention_heads_drug, transformer_attention_probs_dropout,
            transformer_hidden_dropout_rate
        )

        self.dropout = 0.1
        self.CrossAttention = True

        self.decoder = nn.Sequential(
            nn.Linear(2 * D, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 1)
        )

        self.to(self.device)


    def pooled_interaction(self, i, DrugMask_add, SEMsak_add):
        """
        i: [B, Nd, Ns, D] (đã weighted bằng w)
        DrugMask_add: [B,1,1,Nd] additive mask (0 real, -10000 pad)
        SEMsak_add  : [B,1,1,Ns] additive mask (0 real, -10000 pad)
        Return: feat [B, 2D] = concat(mean_pool, max_pool)
        """
        B, Nd, Ns, D = i.shape

        mask_d = (DrugMask_add.squeeze(1).squeeze(1) == 0)  # [B, Nd] bool
        mask_e = (SEMsak_add.squeeze(1).squeeze(1) == 0)    # [B, Ns] bool

        pair_mask = (mask_d.unsqueeze(2) & mask_e.unsqueeze(1))  # [B, Nd, Ns] bool

        # mean pooling over valid pairs
        i_sum = (i * pair_mask.unsqueeze(-1).float()).sum(dim=(1, 2))  # [B, D]
        denom = pair_mask.sum(dim=(1, 2)).clamp(min=1).unsqueeze(-1).float()  # [B,1]
        g_mean = i_sum / denom  # [B, D]

        # max pooling over valid pairs
        # set invalid pairs to very negative so they don't win max
        i_masked = i.masked_fill(~pair_mask.unsqueeze(-1), float("-1e9"))
        g_max = i_masked.amax(dim=(1, 2))  # [B, D]

        feat = torch.cat([g_mean, g_max], dim=-1)  # [B, 2D]
        return feat

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
        Drug = Drug.long().to(self.device)
        SE   = SE.long().to(self.device)

        DrugMask = DrugMask.long().to(self.device)  # (B,Nd) 1=valid,0=pad
        SEMsak   = SEMsak.long().to(self.device)    # (B,Ns) 1=valid,0=pad

        DrugMask_add = (1.0 - DrugMask.unsqueeze(1).unsqueeze(2)) * -10000.0  # (B,1,1,Nd)
        SEMsak_add   = (1.0 - SEMsak.unsqueeze(1).unsqueeze(2)) * -10000.0    # (B,1,1,Ns)

        x_d = self.encoderDrug(self.embDrug(Drug).float(), DrugMask_add.float(), False)  # (B,Nd,D)
        x_e = self.encoderSide(self.embSide(SE).float(),  SEMsak_add.float(),  False)   # (B,Ns,D)

        if self.CrossAttention:
            # mask đúng: True = pad
            mask_d_pad = (DrugMask == 0)
            mask_e_pad = (SEMsak == 0)
            x_d, x_e = self.crossAttention_simple(
                x_d.float(), x_e.float(),
                mask_d=mask_d_pad, mask_e=mask_e_pad
            )

        D  = x_d.size(-1)
        Nd = x_d.size(1)
        Ns = x_e.size(1)

        scores = torch.matmul(x_d, x_e.transpose(-2, -1)) / math.sqrt(D)  # (B,Nd,Ns)

        # mask pad của SE (keys) đúng 1 lần
        se_add = SEMsak_add.squeeze(1).squeeze(1)        # (B,Ns)
        scores = scores + se_add.unsqueeze(1)            # (B,1,Ns) broadcast -> (B,Nd,Ns)

        w = torch.softmax(scores, dim=-1)                # (B,Nd,Ns)

        d_aug = x_d.unsqueeze(2).expand(-1, -1, Ns, -1)  # (B,Nd,Ns,D)
        e_aug = x_e.unsqueeze(1).expand(-1, Nd, -1, -1)  # (B,Nd,Ns,D)

        i = (d_aug * e_aug) * w.unsqueeze(-1)            # (B,Nd,Ns,D)
        i = F.dropout(i, p=self.dropout, training=self.training)

        feat = self.pooled_interaction(i, DrugMask_add, SEMsak_add)  # (B,2D)
        score = self.decoder(feat)                                   # (B,1)

        return score, Drug, SE



def drug2emb_encoder(smile, max_d=MAX_D):
    toks = dbpe.process_line(smile).split()

    if len(toks) == 0:
        ids = np.array([UNK_ID], dtype=np.int64)
    else:
        ids = np.array([words2idx_d.get(t, UNK_ID) for t in toks], dtype=np.int64)

    L = min(len(ids), max_d)

    out = np.full((max_d,), PAD_ID, dtype=np.int64)
    out[:L] = ids[:L]

    mask = np.zeros((max_d,), dtype=np.int64)
    mask[:L] = 1
    return out, mask


