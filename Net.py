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

bpe_codes_drug = codecs.open('data/drug_codes_chembl_freq_1500.txt')
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')


class Trans(torch.nn.Module):
    def __init__(self):
        super(Trans, self).__init__()
        self.device = DEVICE
        self.relu = nn.ReLU()

        # EXACT parameters from original HSTrans paper
        input_dim_drug = 2686
        transformer_emb_size_drug = 304
        transformer_dropout_rate = 0.1
        transformer_n_layer_drug = 8
        transformer_intermediate_size_drug = 512
        transformer_num_attention_heads_drug = 8
        transformer_attention_probs_dropout = 0.1
        transformer_hidden_dropout_rate = 0.1

        # Embedding layers
        self.embDrug = Embeddings(
            input_dim_drug,
            transformer_emb_size_drug,
            50,
            transformer_dropout_rate
        )
        
        self.embSide = Embeddings(
            input_dim_drug,
            transformer_emb_size_drug,
            50,
            transformer_dropout_rate
        )

        # Transformer encoder layers
        self.encoderDrug = Encoder_MultipleLayers(
            transformer_n_layer_drug,
            transformer_emb_size_drug,
            transformer_intermediate_size_drug,
            transformer_num_attention_heads_drug,
            transformer_attention_probs_dropout,
            transformer_hidden_dropout_rate
        )
        
        self.encoderSide = Encoder_MultipleLayers(
            transformer_n_layer_drug,
            transformer_emb_size_drug,
            transformer_intermediate_size_drug,
            transformer_num_attention_heads_drug,
            transformer_attention_probs_dropout,
            transformer_hidden_dropout_rate
        )

        self.dropout = 0.1

        # CNN layer for interaction processing (ORIGINAL HSTRANS)
        self.icnn = nn.Conv2d(1, 10, 3, padding=0)

        # Decoder (ORIGINAL HSTRANS - input size 23040)
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

        # Cross-attention flag (default OFF in original paper)
        self.CrossAttention = False

        self.to(self.device)

    def crossAttention_simple(self, x_d, x_e, mask_d=None, mask_e=None, return_attn=False):
        """
        Cross-attention đơn giản (single-head) giữa Drug (x_d) và SE (x_e).
        NOTE: In original paper, this is disabled by default (self.CrossAttention = False)
        """
        B, Nd, D = x_d.shape
        _, Ns, _ = x_e.shape

        # Handle mask conversion
        if mask_d is not None and mask_d.dim() == 4:
            mask_d = (mask_d.squeeze(1).squeeze(1) == 0)
        elif mask_d is not None:
            mask_d = (mask_d == 0)

        if mask_e is not None and mask_e.dim() == 4:
            mask_e = (mask_e.squeeze(1).squeeze(1) == 0)
        elif mask_e is not None:
            mask_e = (mask_e == 0)

        if mask_d is None:
            mask_d = torch.zeros((B, Nd), dtype=torch.bool, device=x_d.device)
        if mask_e is None:
            mask_e = torch.zeros((B, Ns), dtype=torch.bool, device=x_e.device)

        # Drug (Q) → SE (K,V)
        scores_de = torch.matmul(x_d, x_e.transpose(-2, -1)) / math.sqrt(D)
        scores_de = scores_de.masked_fill(mask_e.unsqueeze(1), float('-1e9'))
        attn_de = torch.softmax(scores_de, dim=-1)
        context_d = torch.matmul(attn_de, x_e)
        x_d_new = x_d + context_d
        x_d_new = F.layer_norm(x_d_new, (D,))

        # SE (Q) → Drug (K,V)
        scores_ed = torch.matmul(x_e, x_d.transpose(-2, -1)) / math.sqrt(D)
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

        # Convert to device and prepare masks
        Drug = Drug.long().to(self.device)
        SE = SE.long().to(self.device)
        
        DrugMask = DrugMask.long().to(self.device)
        DrugMask = DrugMask.unsqueeze(1).unsqueeze(2)
        DrugMask = (1.0 - DrugMask) * -10000.0

        SEMsak = SEMsak.long().to(self.device)
        SEMsak = SEMsak.unsqueeze(1).unsqueeze(2)
        SEMsak = (1.0 - SEMsak) * -10000.0

        # Drug encoding
        emb = self.embDrug(Drug)
        encoded_layers = self.encoderDrug(emb.float(), DrugMask.float(), False)
        x_d = encoded_layers

        # Side effect encoding
        embE = self.embSide(SE)
        encoded_layers = self.encoderSide(embE.float(), SEMsak.float(), False)
        x_e = encoded_layers

        # Optional cross-attention (disabled in original paper)
        if self.CrossAttention:
            mask_d = (DrugMask.squeeze(1).squeeze(1) != -10000).long()
            mask_e = (SEMsak.squeeze(1).squeeze(1) != -10000).long()
            x_d, x_e = self.crossAttention_simple(
                x_d.float(),
                x_e.float(),
                mask_d=mask_d,
                mask_e=mask_e,
                return_attn=False
            )

        # ===============================================
        # ORIGINAL HSTRANS INTERACTION (without weighting)
        # ===============================================
        d_aug = torch.unsqueeze(x_d, 2).repeat(1, 1, 50, 1)  # [B, 50, 50, 304]
        e_aug = torch.unsqueeze(x_e, 1).repeat(1, 50, 1, 1)  # [B, 50, 50, 304]

        # Element-wise multiplication (NO attention weighting in original)
        i = d_aug * e_aug  # [B, 50, 50, 304]

        # ===============================================
        # ORIGINAL CNN PROCESSING
        # ===============================================
        i_v = i.permute(0, 3, 1, 2)  # [B, 304, 50, 50]
        i_v = torch.sum(i_v, dim=1)  # [B, 50, 50] - sum over embedding dim
        i_v = torch.unsqueeze(i_v, 1)  # [B, 1, 50, 50]
        i_v = F.dropout(i_v, p=self.dropout, training=self.training)

        # Conv2d: input [B, 1, 50, 50] → output [B, 10, 48, 48]
        f = self.icnn(i_v)
        
        # Flatten: [B, 10, 48, 48] → [B, 23040]
        f = f.view(int(batch), -1)
        
        # Decoder: [B, 23040] → [B, 1]
        score = self.decoder(f)

        return score, Drug, SE


def drug2emb_encoder(smile, max_d=MAX_D):
    """
    Encode SMILES string to token indices
    """
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