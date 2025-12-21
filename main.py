import argparse
import os
import pickle
import random
import numpy as np
import pandas as pd

import torch
import torch.utils.data as data

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, accuracy_score

from Net import *                 # Trans, drug2emb_encoder, ...
from smiles2vector import load_drug_smile
from utils import *               # rmse, MAE, compute_metrics, ...

raw_file = 'data/raw_frequency_750.mat'
SMILES_file = 'data/drug_SMILES_750.csv'
mask_mat_file = 'data/mask_mat_750.mat'
side_effect_label = 'data/side_effect_label_750.mat'
input_dim = 109

with open('data/drug_side.pkl', 'rb') as gii:
    drug_side = pickle.load(gii)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# Sampling (author style): positive = DAL!=0, negative = DAL==0
# =========================================================
def Extract_positive_negative_samples(DAL, addition_negative_number=''):
    k = 0
    interaction_target = np.zeros((DAL.shape[0] * DAL.shape[1], 3)).astype(int)
    for i in range(DAL.shape[0]):
        for j in range(DAL.shape[1]):
            interaction_target[k, 0] = i
            interaction_target[k, 1] = j
            interaction_target[k, 2] = DAL[i, j]
            k += 1

    data_shuffle = interaction_target[interaction_target[:, 2].argsort()]  # sort by label
    number_positive = len(np.nonzero(data_shuffle[:, 2])[0])

    final_positive_sample = data_shuffle[interaction_target.shape[0] - number_positive::]  # DAL!=0
    negative_sample = data_shuffle[0:interaction_target.shape[0] - number_positive]        # DAL==0

    a = list(np.arange(interaction_target.shape[0] - number_positive))
    if addition_negative_number == 'all':
        b = random.sample(a, (interaction_target.shape[0] - number_positive))
    else:
        b = random.sample(a, (1 + addition_negative_number) * number_positive)

    final_negtive_sample = negative_sample[b[0:number_positive], :]
    addition_negative_sample = negative_sample[b[number_positive::], :]

    # balanced set = positive + same-size negative
    final_positive_sample = np.concatenate((final_positive_sample, final_negtive_sample), axis=0)
    return addition_negative_sample, final_positive_sample, final_negtive_sample

def loss_fun(output, label):
    # sum-squared-error
    return torch.sum((output - label) ** 2)


def identify_sub_fold(data_list, fold_id: int,
                      n_se: int = 994,
                      vocab_size: int = 2686,
                      topk: int = 50,
                      percentile: float = 95.0,
                      out_dir: str = "data/sub"):

    print(f"[Fold {fold_id}] Building SE_sub_index from TRAIN only (no leakage)")
    os.makedirs(out_dir, exist_ok=True)

    drug_smile = [item[1] for item in data_list]      # SMILES
    side_id    = [int(item[0]) for item in data_list] # SE_id
    labels     = [float(item[2]) for item in data_list]

    # 1) Encode SMILES -> sub tokens
    sub_dict = {}
    for i in range(len(drug_smile)):
        drug_sub, mask = drug2emb_encoder(drug_smile[i])
        sub_dict[i] = drug_sub.tolist()

    # 2) Aggregate SE_sub
    SE_sub = np.zeros((n_se, vocab_size), dtype=np.float32)
    for j in range(len(drug_smile)):
        sid = side_id[j]
        y = labels[j]
        for tok in sub_dict[j]:
            if tok == 0:
                continue
            if 0 <= tok < vocab_size:
                SE_sub[sid, tok] += y

    n = float(np.sum(SE_sub)) + 1e-12
    SE_sum  = np.sum(SE_sub, axis=1)   # (n_se,)
    Sub_sum = np.sum(SE_sub, axis=0)   # (vocab_size,)

    SE_p  = SE_sum / n
    Sub_p = Sub_sum / n
    SE_sub_p = SE_sub / n

    # Vectorized freq
    denom = np.sqrt(
        (SE_p[:, None] * Sub_p[None, :] / n) *
        (1 - SE_p)[:, None] *
        (1 - Sub_p)[None, :]
    ) + 1e-12
    freq = (SE_sub_p - (SE_p[:, None] * Sub_p[None, :])) / denom
    freq = freq + 1e-5

    non_nan_values = freq[~np.isnan(freq)]
    thr = np.percentile(non_nan_values, percentile)
    print(f"[Fold {fold_id}] percentile@{percentile}% = {thr}")

    # 3) Select topk per SE
    SE_sub_index = np.zeros((n_se, topk), dtype=np.int32)
    for sid in range(n_se):
        row = freq[sid]
        sorted_idx = np.argsort(row)[::-1]
        filtered = sorted_idx[row[sorted_idx] > thr]
        if filtered.size == 0:
            filtered = sorted_idx[:topk]
        else:
            filtered = filtered[:topk]

        if filtered.size < topk:
            pad = np.zeros((topk - filtered.size,), dtype=np.int32)
            filtered = np.concatenate([filtered.astype(np.int32), pad], axis=0)

        SE_sub_index[sid] = filtered

    SE_sub_mask = (SE_sub_index > 0).astype(np.int32)

    np.save(os.path.join(out_dir, f"SE_sub_index_50_{fold_id}.npy"), SE_sub_index)
    np.save(os.path.join(out_dir, f"SE_sub_mask_50_{fold_id}.npy"),  SE_sub_mask)

    print(f"[Fold {fold_id}] Saved: SE_sub_index_50_{fold_id}.npy / SE_sub_mask_50_{fold_id}.npy")


class Data_Encoder(data.Dataset):
    def __init__(self, list_IDs, labels, df_dti, fold_id: int):
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti
        self.fold_id = fold_id

        self.SE_index = np.load(f"data/sub/SE_sub_index_50_{fold_id}.npy").astype(np.int32)
        self.SE_mask  = np.load(f"data/sub/SE_sub_mask_50_{fold_id}.npy").astype(np.int32)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        index = self.list_IDs[idx]
        d = self.df.iloc[index]['Drug_smile']
        s = int(self.df.iloc[index]['SE_id'])
        y = float(self.labels[index])

        d_v, input_mask_d = drug2emb_encoder(d)     # (50,), (50,)
        s_v = self.SE_index[s, :]                   # (50,)
        input_mask_s = self.SE_mask[s, :]           # (50,)

        # return y as float tensor later in collate
        return d_v, s_v, input_mask_d, input_mask_s, y


# Train
def trainfun(model, device, train_loader, optimizer, epoch, log_interval):
    # print(f"Training on {len(train_loader.dataset)} samples...")
    model.train()
    avg_loss = []

    # lấy vocab từ embedding
    vocab = model.embDrug.word_embeddings.num_embeddings

    for batch_idx, (Drug, SE, DrugMask, SEMsak, Label) in enumerate(train_loader):
        # move to device
        Drug     = Drug.to(device, non_blocking=True).long()
        SE       = SE.to(device, non_blocking=True).long()
        DrugMask = DrugMask.to(device, non_blocking=True)
        SEMsak   = SEMsak.to(device, non_blocking=True)
        Label    = torch.tensor([int(x) for x in Label], device=device).float()

        # ====== CHECK out-of-range (OOR) ======
        dmin, dmax = int(Drug.min().item()), int(Drug.max().item())
        emin, emax = int(SE.min().item()), int(SE.max().item())

        if dmax >= vocab or dmin < 0:
            raise RuntimeError(f"[Drug OOR] min={dmin} max={dmax} vocab={vocab}")
        if emax >= vocab or emin < 0:
            raise RuntimeError(f"[SE OOR] min={emin} max={emax} vocab={vocab}")

        optimizer.zero_grad(set_to_none=True)
        out, _, _ = model(Drug, SE, DrugMask, SEMsak)
        pred = out.view(-1)

        loss = loss_fun(pred, Label)
        loss.backward()
        optimizer.step()

        avg_loss.append(loss.item())

        # if batch_idx % log_interval == 0:
        #     print(
        #         f"Train epoch: {epoch} "
        #         f"[{(batch_idx+1)*len(Label)}/{len(train_loader.dataset)} "
        #         f"({100.*(batch_idx+1)/len(train_loader):.0f}%)]\t"
        #         f"Loss: {loss.item():.6f}"
        #     )

    return sum(avg_loss) / max(1, len(avg_loss))


# Regression evaluation (RMSE/MAE/SCC/Overlap)
def evaluate_regression(model, device, loader, only_nonzero=True):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for Drug, SE, DrugMask, SEMsak, Label in loader:
            Drug     = Drug.to(device, non_blocking=True)
            SE       = SE.to(device, non_blocking=True)
            DrugMask = DrugMask.to(device, non_blocking=True)
            SEMsak   = SEMsak.to(device, non_blocking=True)

            Label = torch.tensor(Label, device=device, dtype=torch.float32).view(-1)

            out, _, _ = model(Drug, SE, DrugMask, SEMsak)
            pred = out.view(-1)

            all_preds.append(pred.detach().cpu())
            all_labels.append(Label.detach().cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    if only_nonzero:
        mask = (labels != 0)
        labels_use = labels[mask]
        preds_use = preds[mask]
    else:
        labels_use = labels
        preds_use = preds

    m = {}
    m["rmse"] = rmse(labels_use, preds_use)
    m["mae"]  = MAE(labels_use, preds_use)
    scc, ov1, ov5, ov10, ov20 = compute_metrics(labels_use, preds_use)
    m["scc"] = scc
    m["ov1"] = ov1
    m["ov5"] = ov5
    m["ov10"] = ov10
    m["ov20"] = ov20
    m["labels"] = labels_use
    m["preds"] = preds_use
    return m


#   - y_true = (label!=0)
#   - score  = pred (continuous)  -> AUC/AUPR should use continuous score
def evaluate_binary_from_regression(model, device, loader, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for Drug, SE, DrugMask, SEMsak, Label in loader:
            Drug     = Drug.to(device, non_blocking=True)
            SE       = SE.to(device, non_blocking=True)
            DrugMask = DrugMask.to(device, non_blocking=True)
            SEMsak   = SEMsak.to(device, non_blocking=True)
            Label = torch.tensor(Label, device=device, dtype=torch.float32).view(-1)

            out, _, _ = model(Drug, SE, DrugMask, SEMsak)
            pred = out.view(-1)

            all_preds.append(pred.detach().cpu())
            all_labels.append(Label.detach().cpu())

    score = torch.cat(all_preds).numpy()               # continuous
    y_raw = torch.cat(all_labels).numpy()
    y_true = (y_raw != 0).astype(int)

    # AUC/AUPR: use continuous scores
    auc_all = roc_auc_score(y_true, score)
    aupr_all = average_precision_score(y_true, score)

    # precision/recall/acc: need binary prediction
    y_pred = (score > threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    return auc_all, aupr_all, precision, recall, accuracy


def main_fold(train_loader, val_loader, test_loader,
              modeling, lr, num_epoch, weight_decay, log_interval,
              cuda_name, save_model, fold_id,
              early_stop_patience=100,
              select_metric="rmse"):

    print('\n=======================================================================================')
    print('model: ', modeling.__name__)
    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)
    print('weight_decay: ', weight_decay)

    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    model = modeling().to(device)

    try:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    except ValueError:
        model.eval()
        with torch.no_grad():
            for Drug, SE, DrugMask, SEMsak, _ in train_loader:
                Drug     = Drug.to(device, non_blocking=True)
                SE       = SE.to(device, non_blocking=True)
                DrugMask = DrugMask.to(device, non_blocking=True)
                SEMsak   = SEMsak.to(device, non_blocking=True)
                _ = model(Drug, SE, DrugMask, SEMsak)
                break
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Total parameters: {total_params}')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min" if select_metric == "rmse" else "max",
        factor=0.5,
        patience=3,
        verbose=True
    )

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("predictResult", exist_ok=True)

    best_epoch = -1
    best_state = None
    bad_epochs = 0
    best_score = float("inf") if select_metric == "rmse" else -float("inf")

    for epoch in range(num_epoch):
        train_loss = trainfun(model, device, train_loader, optimizer, epoch + 1, log_interval)

        val_m = evaluate_regression(model, device, val_loader, only_nonzero=True)

        print(f"\n===== Epoch {epoch + 1} summary =====")
        print(f"Train Loss: {train_loss:.5f}")
        print('Validation:\tRMSE: {:.5f}\tMAE: {:.5f}\tSCC: {:.5f}'.format(val_m["rmse"], val_m["mae"], val_m["scc"]))
        print('Overlap@1%: {:.5f}\t5%: {:.5f}\t10%: {:.5f}\t20%: {:.5f}'.format(
            val_m["ov1"], val_m["ov5"], val_m["ov10"], val_m["ov20"]
        ))

        cur = val_m["rmse"] if select_metric == "rmse" else val_m["scc"]
        scheduler.step(cur)

        improved = (cur < best_score) if select_metric == "rmse" else (cur > best_score)
        if improved:
            best_score = cur
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0

            if save_model:
                torch.save(best_state, f"checkpoints/fold{fold_id}_best_val_{select_metric}.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= early_stop_patience:
                print(f"Early stopping. Best val {select_metric} at epoch {best_epoch}: {best_score:.5f}")
                break

    # Load best and final test once
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    print(f"\n[Fold {fold_id}] Loaded best model from epoch {best_epoch} (best val {select_metric}={best_score:.5f})")

    test_m = evaluate_regression(model, device, test_loader, only_nonzero=True)

    print("\n===== FINAL TEST =====")
    print('Test:\tRMSE: {:.5f}\tMAE: {:.5f}\tSCC: {:.5f}'.format(test_m["rmse"], test_m["mae"], test_m["scc"]))
    print('Overlap@1%: {:.5f}\t5%: {:.5f}\t10%: {:.5f}\t20%: {:.5f}'.format(
        test_m["ov1"], test_m["ov5"], test_m["ov10"], test_m["ov20"]
    ))

    # Optional: binary metrics derived from regression
    auc_all, aupr_all, precision, recall, accuracy = evaluate_binary_from_regression(model, device, test_loader, threshold=0.5)
    print('Binary-from-regression:\tAUC: {:.5f}\tAUPR: {:.5f}\tPrec: {:.5f}\tRecall: {:.5f}\tACC: {:.5f}'.format(
        auc_all, aupr_all, precision, recall, accuracy
    ))

    np.save(f'predictResult/test_labels_fold{fold_id}.npy', test_m["labels"])
    np.save(f'predictResult/test_preds_fold{fold_id}.npy',  test_m["preds"])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--model', type=int, required=False, default=0)
    parser.add_argument('--lr', type=float, required=False, default=1e-4)
    parser.add_argument('--wd', type=float, required=False, default=0.01)
    parser.add_argument('--epoch', type=int, required=False, default=40)
    parser.add_argument('--log_interval', type=int, required=False, default=40)
    parser.add_argument('--cuda_name', type=str, required=False, default='cuda')
    parser.add_argument('--save_model', action='store_true', default=True)

    # val split + early stop
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--select_metric', type=str, default='rmse', choices=['rmse', 'scc'])

    args = parser.parse_args()

    modeling = [Trans][args.model]
    lr = args.lr
    num_epoch = args.epoch
    weight_decay = args.wd
    log_interval = args.log_interval
    cuda_name = args.cuda_name
    save_model = args.save_model

    # -----------------------------
    # 1) Build balanced samples (author)
    # -----------------------------
    addition_negative_sample, final_positive_sample, final_negative_sample = Extract_positive_negative_samples(
        drug_side, addition_negative_number='all'
    )

    final_sample = final_positive_sample
    X = final_sample[:, :]
    # labels for stratification
    data_y = [int(float(X[i, 2])) for i in range(X.shape[0])]

    drug_dict, drug_smile = load_drug_smile(SMILES_file)

    # data_x: (SE_id, Drug_id) ; data: (SE_id, SMILES, Label)
    data_x = []
    data = []
    for i in range(X.shape[0]):
        drug_id = int(X[i, 0])
        se_id   = int(X[i, 1])
        label   = float(X[i, 2])

        data_x.append((se_id, drug_id))
        data.append((se_id, drug_smile[drug_id], label))

    data = np.array(data, dtype=object)
    data_x = np.array(data_x, dtype=object)
    data_y = np.array(data_y, dtype=int)

    # -----------------------------
    # 2) Outer 5-fold CV
    # -----------------------------
    kfold = StratifiedKFold(5, random_state=1, shuffle=True)

    train_params = {'batch_size': 128, 'shuffle': True,  'num_workers': 8, 'pin_memory': True}
    eval_params  = {'batch_size': 128, 'shuffle': False, 'num_workers': 8, 'pin_memory': True}

    for fold_id, (trainval_idx, test_idx) in enumerate(kfold.split(data_x, data_y)):
        print(f"\n====================== FOLD {fold_id} ======================")

        # -----------------------------
        # 3) Inner split: train/val from trainval (stratified)
        # -----------------------------
        trainval_y = data_y[trainval_idx]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=1 + fold_id)
        train_rel, val_rel = next(sss.split(np.zeros_like(trainval_y), trainval_y))

        train_idx = trainval_idx[train_rel]
        val_idx   = trainval_idx[val_rel]

        data_train = data[train_idx]
        data_val   = data[val_idx]
        data_test  = data[test_idx]

        # -----------------------------
        # 4) Build SE features from TRAIN ONLY (NO LEAKAGE)
        # -----------------------------
        identify_sub_fold(data_train.tolist(), fold_id=fold_id)

        # -----------------------------
        # 5) Build DataFrames
        # -----------------------------
        df_train = pd.DataFrame(data=data_train.tolist(), columns=['SE_id', 'Drug_smile', 'Label'])
        df_val   = pd.DataFrame(data=data_val.tolist(),   columns=['SE_id', 'Drug_smile', 'Label'])
        df_test  = pd.DataFrame(data=data_test.tolist(),  columns=['SE_id', 'Drug_smile', 'Label'])

        # -----------------------------
        # 6) Datasets + Loaders
        # -----------------------------
        training_set = Data_Encoder(df_train.index.values, df_train.Label.values, df_train, fold_id)
        val_set      = Data_Encoder(df_val.index.values,   df_val.Label.values,   df_val,   fold_id)
        testing_set  = Data_Encoder(df_test.index.values,  df_test.Label.values,  df_test,  fold_id)

        training_loader = torch.utils.data.DataLoader(training_set, **train_params)
        val_loader      = torch.utils.data.DataLoader(val_set,      **eval_params)
        testing_loader  = torch.utils.data.DataLoader(testing_set,  **eval_params)

        # -----------------------------
        # 7) Train with validation selection, test once at end
        # -----------------------------
        main_fold(
            train_loader=training_loader,
            val_loader=val_loader,
            test_loader=testing_loader,
            modeling=modeling,
            lr=lr,
            num_epoch=num_epoch,
            weight_decay=weight_decay,
            log_interval=log_interval,
            cuda_name=cuda_name,
            save_model=save_model,
            fold_id=fold_id,
            early_stop_patience=args.patience,
            select_metric=args.select_metric
        )
