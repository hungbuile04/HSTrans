import argparse
import os
import pickle
import random
import numpy as np
import pandas as pd

import torch
import torch.utils.data as data

from sklearn.model_selection import StratifiedKFold
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

    data_shuffle = interaction_target[interaction_target[:, 2].argsort()]
    number_positive = len(np.nonzero(data_shuffle[:, 2])[0])

    final_positive_sample = data_shuffle[interaction_target.shape[0] - number_positive::]
    negative_sample = data_shuffle[0:interaction_target.shape[0] - number_positive]

    a = list(np.arange(interaction_target.shape[0] - number_positive))
    if addition_negative_number == 'all':
        b = random.sample(a, (interaction_target.shape[0] - number_positive))
    else:
        b = random.sample(a, (1 + addition_negative_number) * number_positive)

    final_negtive_sample = negative_sample[b[0:number_positive], :]
    addition_negative_sample = negative_sample[b[number_positive::], :]

    final_positive_sample = np.concatenate((final_positive_sample, final_negtive_sample), axis=0)
    return addition_negative_sample, final_positive_sample, final_negtive_sample


# =========================================================
# SIMPLE MSE LOSS (như paper gốc)
# =========================================================
def loss_fun(output, label):
    """Simple MSE loss - exactly as in original HSTrans paper"""
    return torch.sum((output - label) ** 2)


# =========================================================
# Build SE_sub_index from TRAIN only (no leakage)
# =========================================================
def identify_sub_fold(data_list, fold_id: int,
                      n_se: int = 994,
                      vocab_size: int = 2686,
                      topk: int = 50,
                      percentile: float = 95.0,
                      out_dir: str = "data/sub"):

    print(f"[Fold {fold_id}] Building SE_sub_index from TRAIN only (no leakage)")
    os.makedirs(out_dir, exist_ok=True)

    drug_smile = [item[1] for item in data_list]
    side_id    = [int(item[0]) for item in data_list]
    labels     = [float(item[2]) for item in data_list]

    # 1) Encode SMILES -> sub tokens
    sub_dict = {}
    for i in range(len(drug_smile)):
        drug_sub, _ = drug2emb_encoder(drug_smile[i])
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
    SE_sum  = np.sum(SE_sub, axis=1)
    Sub_sum = np.sum(SE_sub, axis=0)

    SE_p  = SE_sum / n
    Sub_p = Sub_sum / n
    SE_sub_p = SE_sub / n

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

    print(f"[Fold {fold_id}] Saved SE features")


# =========================================================
# Dataset
# =========================================================
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
        y = self.labels[index]  # Keep as original float value

        d_v, input_mask_d = drug2emb_encoder(d)
        s_v = self.SE_index[s, :]
        input_mask_s = self.SE_mask[s, :]

        return d_v, s_v, input_mask_d, input_mask_s, y


# =========================================================
# Train one epoch (SIMPLE - như paper gốc)
# =========================================================
def trainfun(model, device, train_loader, optimizer, epoch, log_interval):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    avg_loss = []

    for batch_idx, (Drug, SE, DrugMask, SEMsak, Label) in enumerate(train_loader):
        Drug     = Drug.to(device, non_blocking=True)
        SE       = SE.to(device, non_blocking=True)
        DrugMask = DrugMask.to(device, non_blocking=True)
        SEMsak   = SEMsak.to(device, non_blocking=True)
        Label    = torch.FloatTensor([float(item) for item in Label]).to(device, non_blocking=True)

        optimizer.zero_grad()
        out, _, _ = model(Drug, SE, DrugMask, SEMsak)
        pred = out.flatten()

        # SIMPLE MSE LOSS - exactly as original paper
        loss = loss_fun(pred, Label)

        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())

        # if batch_idx % log_interval == 0:
        #     print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch,
        #         (batch_idx + 1) * len(Label),
        #         len(train_loader.dataset),
        #         100. * (batch_idx + 1) / len(train_loader),
        #         loss.item()
        #     ))

    return sum(avg_loss) / len(avg_loss)


# =========================================================
# Predict (như paper gốc)
# =========================================================
def predict(model, device, test_loader):
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()

    model.eval()
    torch.cuda.manual_seed(42)

    with torch.no_grad():
        for batch_idx, (Drug, SE, DrugMask, SEMsak, Label) in enumerate(test_loader):
            Drug     = Drug.to(device)
            SE       = SE.to(device)
            DrugMask = DrugMask.to(device)
            SEMsak   = SEMsak.to(device)
            Label    = torch.FloatTensor([float(item) for item in Label])

            out, _, _ = model(Drug, SE, DrugMask, SEMsak)

            # Filter non-zero labels (như paper gốc)
            location = torch.where(Label != 0)
            pred = out[location]
            label = Label[location]

            total_preds = torch.cat((total_preds, pred.cpu()), 0)
            total_labels = torch.cat((total_labels, label.cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


# =========================================================
# Evaluate (như paper gốc - binary metrics)
# =========================================================
def evaluate(model, device, test_loader):
    total_preds = torch.Tensor()
    total_label = torch.Tensor()
    singleDrug_auc = []
    singleDrug_aupr = []
    
    model.eval()
    torch.cuda.manual_seed(42)

    with torch.no_grad():
        for batch_idx, (Drug, SE, DrugMask, SEMsak, Label) in enumerate(test_loader):
            Drug     = Drug.to(device)
            SE       = SE.to(device)
            DrugMask = DrugMask.to(device)
            SEMsak   = SEMsak.to(device)
            Label    = torch.FloatTensor([float(item) for item in Label])
            
            output, _, _ = model(Drug, SE, DrugMask, SEMsak)
            pred = output.cpu()

            total_preds = torch.cat((total_preds, pred), 0)
            total_label = torch.cat((total_label, Label), 0)

            # Per-drug metrics
            pred_np = pred.numpy().flatten()
            pred_binary = np.where(pred_np > 0.5, 1, 0)
            label_binary = (Label.numpy().flatten() != 0).astype(int)

            if len(np.unique(label_binary)) > 1:  # Need both classes for AUC
                singleDrug_auc.append(roc_auc_score(label_binary, pred_binary))
                singleDrug_aupr.append(average_precision_score(label_binary, pred_binary))

    drugAUC = sum(singleDrug_auc) / len(singleDrug_auc) if singleDrug_auc else 0.0
    drugAUPR = sum(singleDrug_aupr) / len(singleDrug_aupr) if singleDrug_aupr else 0.0
    
    total_preds = total_preds.numpy()
    total_label = total_label.numpy()

    # Binary conversion
    total_pre_binary = np.where(total_preds > 0.5, 1, 0)
    label01 = np.where(total_label != 0, 1, total_label)

    precision = precision_score(label01, total_pre_binary)
    recall = recall_score(label01, total_pre_binary)
    accuracy = accuracy_score(label01, total_pre_binary)

    # Overall AUC/AUPR
    pos = np.squeeze(total_preds[np.where(total_label)])
    pos_label = np.ones(len(pos))
    neg = np.squeeze(total_preds[np.where(total_label == 0)])
    neg_label = np.zeros(len(neg))

    y = np.hstack((pos, neg))
    y_true = np.hstack((pos_label, neg_label))
    
    auc_all = roc_auc_score(y_true, y)
    aupr_all = average_precision_score(y_true, y)

    return auc_all, aupr_all, drugAUC, drugAUPR, precision, recall, accuracy


# =========================================================
# Main training loop (SIMPLIFIED - như paper gốc)
# =========================================================
def main_fold(train_loader, test_loader, modeling, lr, num_epoch, 
              weight_decay, log_interval, cuda_name, save_model, fold_id):

    print('\n=======================================================================================')
    print('model: ', modeling.__name__)
    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)
    print('weight_decay: ', weight_decay)

    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    model = modeling().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params}')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("predictResult", exist_ok=True)

    best_rmse = 1e9
    best_epoch = -1
    best_scc = -1e9
    best_ov1, best_ov5, best_ov10, best_ov20 = 0, 0, 0, 0
    best_epoch_metrics = -1

    # Training loop - NO VALIDATION, NO EARLY STOPPING (như paper gốc)
    for epoch in range(num_epoch):
        train_loss = trainfun(
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch + 1,
            log_interval=log_interval
        )

        print(f"\n===== Epoch {epoch + 1} summary =====")
        print(f"Train Loss: {train_loss:.5f}")

        # Intermediate evaluation on test set
        test_labels, test_preds = predict(model=model, device=device, test_loader=test_loader)
        test_rMSE = rmse(test_labels, test_preds)
        test_MAE = MAE(test_labels, test_preds)
        scc, ov1, ov5, ov10, ov20 = compute_metrics(test_labels, test_preds)

        if test_rMSE < best_rmse:
            best_rmse = test_rMSE
            best_epoch = epoch + 1

        if scc > best_scc:
            best_scc = scc
            best_ov1, best_ov5, best_ov10, best_ov20 = ov1, ov5, ov10, ov20
            best_epoch_metrics = epoch + 1

        auc_all, aupr_all, drugAUC, drugAUPR, precision, recall, accuracy = evaluate(
            model=model, device=device, test_loader=test_loader
        )

        print('Test:\trMSE: {:.5f}\tMAE: {:.5f}\tSCC: {:.5f}'.format(test_rMSE, test_MAE, scc))
        print('Overlap@1%: {:.5f}\t5%: {:.5f}\t10%: {:.5f}\t20%: {:.5f}'.format(ov1, ov5, ov10, ov20))
        print('AUC: {:.5f}\tAUPR: {:.5f}\tDrugAUC: {:.5f}\tDrugAUPR: {:.5f}\t'
              'Prec: {:.5f}\tRecall: {:.5f}\tACC: {:.5f}'.format(
            auc_all, aupr_all, drugAUC, drugAUPR, precision, recall, accuracy
        ))

    # Final prediction after all epochs
    print("\n正在预测")
    test_labels, test_preds = predict(model=model, device=device, test_loader=test_loader)

    np.save(f'predictResult/test_labels_fold{fold_id}.npy', test_labels)
    np.save(f'predictResult/test_preds_fold{fold_id}.npy', test_preds)

    test_rMSE = rmse(test_labels, test_preds)
    test_MAE = MAE(test_labels, test_preds)
    scc, ov1, ov5, ov10, ov20 = compute_metrics(test_labels, test_preds)

    auc_all, aupr_all, drugAUC, drugAUPR, precision, recall, accuracy = evaluate(
        model=model, device=device, test_loader=test_loader
    )

    print("\n===== FINAL TEST RESULTS =====")
    print('RMSE: {:.5f}\tMAE: {:.5f}\tSCC: {:.5f}'.format(test_rMSE, test_MAE, scc))
    print('Overlap@1%: {:.5f}\t5%: {:.5f}\t10%: {:.5f}\t20%: {:.5f}'.format(ov1, ov5, ov10, ov20))
    print('AUC: {:.5f}\tAUPR: {:.5f}\tDrugAUC: {:.5f}\tDrugAUPR: {:.5f}\t'
          'Prec: {:.5f}\tRecall: {:.5f}\tACC: {:.5f}'.format(
        auc_all, aupr_all, drugAUC, drugAUPR, precision, recall, accuracy
    ))

    print(f"\n>>>>> FOLD {fold_id} FINISHED")
    print(f">>>>> BEST RMSE: {best_rmse:.5f} at epoch {best_epoch}")
    print(f">>>>> BEST SCC: {best_scc:.5f} at epoch {best_epoch_metrics}")
    print(f">>>>> BEST Overlaps: 1%={best_ov1:.5f}, 5%={best_ov5:.5f}, "
          f"10%={best_ov10:.5f}, 20%={best_ov20:.5f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--model', type=int, required=False, default=0)
    parser.add_argument('--lr', type=float, required=False, default=1e-4)
    parser.add_argument('--wd', type=float, required=False, default=0.01)
    parser.add_argument('--epoch', type=int, required=False, default=40)
    parser.add_argument('--log_interval', type=int, required=False, default=40)
    parser.add_argument('--cuda_name', type=str, required=False, default='cuda')
    parser.add_argument('--save_model', action='store_true', default=True)

    args = parser.parse_args()

    modeling = [Trans][args.model]
    lr = args.lr
    num_epoch = args.epoch
    weight_decay = args.wd
    log_interval = args.log_interval
    cuda_name = args.cuda_name
    save_model = args.save_model

    # Build balanced samples
    addition_negative_sample, final_positive_sample, final_negative_sample = \
        Extract_positive_negative_samples(drug_side, addition_negative_number='all')

    final_sample = final_positive_sample
    X = final_sample[:, :]
    data_y = [int(float(X[i, 2])) for i in range(X.shape[0])]

    drug_dict, drug_smile = load_drug_smile(SMILES_file)

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

    # 5-fold CV (NO inner validation split)
    kfold = StratifiedKFold(5, random_state=1, shuffle=True)

    params = {'batch_size': 128, 'shuffle': True, 'num_workers': 4, 'pin_memory': True}

    for fold_id, (train_idx, test_idx) in enumerate(kfold.split(data_x, data_y)):
        print(f"\n====================== FOLD {fold_id} ======================")

        data_train = data[train_idx]
        data_test  = data[test_idx]

        # Build SE features from TRAIN ONLY
        identify_sub_fold(data_train.tolist(), fold_id=fold_id)

        # Build DataFrames
        df_train = pd.DataFrame(data=data_train.tolist(), columns=['SE_id', 'Drug_smile', 'Label'])
        df_test  = pd.DataFrame(data=data_test.tolist(),  columns=['SE_id', 'Drug_smile', 'Label'])

        # Datasets + Loaders
        training_set = Data_Encoder(df_train.index.values, df_train.Label.values, df_train, fold_id)
        testing_set  = Data_Encoder(df_test.index.values,  df_test.Label.values,  df_test,  fold_id)

        training_loader = torch.utils.data.DataLoader(training_set, **params)
        testing_loader  = torch.utils.data.DataLoader(testing_set,  **params)

        # Train (NO validation set, NO early stopping)
        main_fold(
            train_loader=training_loader,
            test_loader=testing_loader,
            modeling=modeling,
            lr=lr,
            num_epoch=num_epoch,
            weight_decay=weight_decay,
            log_interval=log_interval,
            cuda_name=cuda_name,
            save_model=save_model,
            fold_id=fold_id
        )