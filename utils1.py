import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, balanced_accuracy_score,
    precision_score, recall_score, confusion_matrix, RocCurveDisplay
)

def get_module_idx(device):
    # Yeo-7 网络编号：
    # 0: Visual
    # 1: Somatomotor
    # 2: DorsalAttention
    # 3: VentralAttention
    # 4: Limbic
    # 5: Frontoparietal (Control)
    # 6: DefaultMode

    region_names = [
        # 左半球 45
        "Precentral_L","Frontal_Sup_L","Frontal_Sup_Orb_L","Frontal_Mid_L","Frontal_Mid_Orb_L",
        "Frontal_Inf_Oper_L","Frontal_Inf_Tri_L","Frontal_Inf_Orb_L","Rolandic_Oper_L","Supp_Motor_Area_L",
        "Olfactory_L","Frontal_Sup_Medial_L","Frontal_Med_Orb_L","Rectus_L","Insula_L",
        "Cingulum_Ant_L","Cingulum_Mid_L","Cingulum_Post_L","Hippocampus_L","ParaHippocampal_L",
        "Amygdala_L","Calcarine_L","Cuneus_L","Lingual_L","Occipital_Sup_L",
        "Occipital_Mid_L","Occipital_Inf_L","Fusiform_L","Postcentral_L","Parietal_Sup_L",
        "Parietal_Inf_L","SupraMarginal_L","Angular_L","Precuneus_L","Paracentral_Lobule_L",
        "Caudate_L","Putamen_L","Pallidum_L","Thalamus_L","Heschl_L",
        "Temporal_Sup_L","Temporal_Pole_Sup_L","Temporal_Mid_L","Temporal_Pole_Mid_L","Temporal_Inf_L",
        # 右半球 45
        "Precentral_R","Frontal_Sup_R","Frontal_Sup_Orb_R","Frontal_Mid_R","Frontal_Mid_Orb_R",
        "Frontal_Inf_Oper_R","Frontal_Inf_Tri_R","Frontal_Inf_Orb_R","Rolandic_Oper_R","Supp_Motor_Area_R",
        "Olfactory_R","Frontal_Sup_Medial_R","Frontal_Med_Orb_R","Rectus_R","Insula_R",
        "Cingulum_Ant_R","Cingulum_Mid_R","Cingulum_Post_R","Hippocampus_R","ParaHippocampal_R",
        "Amygdala_R","Calcarine_R","Cuneus_R","Lingual_R","Occipital_Sup_R",
        "Occipital_Mid_R","Occipital_Inf_R","Fusiform_R","Postcentral_R","Parietal_Sup_R",
        "Parietal_Inf_R","SupraMarginal_R","Angular_R","Precuneus_R","Paracentral_Lobule_R",
        "Caudate_R","Putamen_R","Pallidum_R","Thalamus_R","Heschl_R",
        "Temporal_Sup_R","Temporal_Pole_Sup_R","Temporal_Mid_R","Temporal_Pole_Mid_R","Temporal_Inf_R",
    ]

    network_map = {
        # Visual (0)
        "Calcarine_L":0, "Calcarine_R":0, "Cuneus_L":0, "Cuneus_R":0,
        "Lingual_L":0,   "Lingual_R":0,   "Occipital_Sup_L":0, "Occipital_Sup_R":0,
        "Occipital_Mid_L":0,"Occipital_Mid_R":0,"Occipital_Inf_L":0,"Occipital_Inf_R":0,
        "Fusiform_L":0,  "Fusiform_R":0,

        # Somatomotor (1)
        "Precentral_L":1,"Precentral_R":1,"Postcentral_L":1,"Postcentral_R":1,
        "Supp_Motor_Area_L":1,"Supp_Motor_Area_R":1,
        "Paracentral_Lobule_L":1,"Paracentral_Lobule_R":1,
        "Heschl_L":1,"Heschl_R":1,  # 听觉也归到躯体感觉

        # DorsalAttention (2)
        "Frontal_Sup_L":2,"Frontal_Sup_R":2,
        "Parietal_Sup_L":2,"Parietal_Sup_R":2,
        "Precuneus_L":2,"Precuneus_R":2,
        "Parietal_Inf_L":2,"Parietal_Inf_R":2,

        # VentralAttention (3)
        "Frontal_Inf_Oper_L":3,"Frontal_Inf_Oper_R":3,
        "Frontal_Inf_Tri_L":3,  "Frontal_Inf_Tri_R":3,
        "Rolandic_Oper_L":3,    "Rolandic_Oper_R":3,
        "Insula_L":3,           "Insula_R":3,

        # Limbic (4)
        "Olfactory_L":4, "Olfactory_R":4,
        "ParaHippocampal_L":4,"ParaHippocampal_R":4,
        "Hippocampus_L":4,      "Hippocampus_R":4,
        "Amygdala_L":4,         "Amygdala_R":4,
        "Caudate_L":4, "Caudate_R":4,
        "Putamen_L":4,"Putamen_R":4,
        "Pallidum_L":4,"Pallidum_R":4,
        "Thalamus_L":4,"Thalamus_R":4,

        # Frontoparietal / Control (5)
        "Frontal_Mid_L":5,      "Frontal_Mid_R":5,
        "Frontal_Mid_Orb_L":5,  "Frontal_Mid_Orb_R":5,
        "Frontal_Inf_Orb_L":5,  "Frontal_Inf_Orb_R":5,
        "SupraMarginal_L":5,    "SupraMarginal_R":5,
        "Angular_L":5,          "Angular_R":5,

        # DefaultMode (6)
        "Frontal_Sup_Medial_L":6,"Frontal_Sup_Medial_R":6,
        "Rectus_L":6,           "Rectus_R":6,
        "Cingulum_Ant_L":6,     "Cingulum_Ant_R":6,
        "Cingulum_Mid_L":6,     "Cingulum_Mid_R":6,
        "Cingulum_Post_L":6,    "Cingulum_Post_R":6,
        "Temporal_Mid_L":6,     "Temporal_Mid_R":6,
        "Temporal_Pole_Mid_L":6,"Temporal_Pole_Mid_R":6,
        "Temporal_Inf_L":6,     "Temporal_Inf_R":6,
        "Temporal_Sup_L":6,     "Temporal_Sup_R":6,
        "Temporal_Pole_Sup_L":6,"Temporal_Pole_Sup_R":6,
        "Frontal_Sup_Orb_L":6,  "Frontal_Sup_Orb_R":6,
        "Frontal_Med_Orb_L":6,  "Frontal_Med_Orb_R":6,
    }

    # 检查有没有遗漏
    missing = [r for r in region_names if r not in network_map]
    if missing:
        raise KeyError(f"These ROIs are not in network_map: {missing}")

    module_idx = torch.tensor(
        [network_map[r] for r in region_names],
        dtype=torch.long,
        device=device
    )
    return module_idx




def compute_channel_stats(data_list, eps=1e-5):
    """
    Compute per-channel mean and std without concatenating all data in memory,
    using a two-pass sum / sumsq approach.
    data_list: list of np.array with shape (T_i, D)
    Returns:
      mean: (D,), std: (D,)
    """
    total_count = 0
    sum_ = None
    sumsq = None
    for x in data_list:
        x = np.asarray(x)
        T, D = x.shape
        if sum_ is None:
            sum_ = np.zeros(D, dtype=np.float64)
            sumsq = np.zeros(D, dtype=np.float64)
        sum_ += x.sum(axis=0)
        sumsq += (x ** 2).sum(axis=0)
        total_count += T
    if total_count == 0:
        raise ValueError("Empty data_list passed to compute_channel_stats")
    mean = sum_ / total_count
    var = sumsq / total_count - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0)) + eps
    return mean, std


def standardize_all(data_list, mean, std, clip_val=5.0):
    standardized = []
    for x in data_list:
        x = (x - mean) / std
        x = np.clip(x, -clip_val, clip_val)
        standardized.append(x)
    return standardized


def compute_local_robust_stats(data_list):
    """
    Compute median and MAD per sample.
    Returns:
      locals_m:   (N, D)
      locals_mad: (N, D)
    """
    locals_m = []
    locals_mad = []
    for x in data_list:
        x = np.asarray(x)
        m = np.median(x, axis=0)
        mad = np.median(np.abs(x - m), axis=0) + 1e-6
        locals_m.append(m)
        locals_mad.append(mad)
    return np.stack(locals_m), np.stack(locals_mad)


def compute_global_robust_stats(locals_m, locals_mad):
    """
    Compute global median-of-locals and MAD-of-locals across samples.
    """
    global_med = np.median(locals_m, axis=0)
    global_mad = np.median(locals_mad, axis=0) + 1e-6
    return global_med, global_mad


def robust_local_then_global(data_list, global_med, global_mad, clip_val=5.0):
    """
    Two-step robust standardization: local median/MAD then global median/MAD.
    """
    out = []
    for x in data_list:
        x = np.asarray(x)
        m = np.median(x, axis=0)
        mad = np.median(np.abs(x - m), axis=0) + 1e-6
        x_loc = (x - m) / mad
        x_glb = (x_loc - global_med) / global_mad
        out.append(np.clip(x_glb, -clip_val, clip_val))
    return out


def bootstrap_ci(metric_func, y_true, y_pred, probs=None, n_bootstrap=1000, alpha=0.05, random_state=None):
    """
    Bootstrap confidence interval for a metric. If all predictions are correct or all incorrect,
    returns the metric with a degenerate CI (metric, metric).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if probs is not None:
        probs = np.asarray(probs)
    # Only one class -> can't compute metrics
    if len(np.unique(y_true)) == 1:
        return float('nan'), (float('nan'), float('nan'))
    # All correct or all incorrect -> degenerate CI
    metric_val = metric_func(y_true, y_pred) if probs is None else metric_func(y_true, y_pred, probs)
    if np.all(y_true == y_pred) or np.all(y_true != y_pred):
        return float(metric_val), (float(metric_val), float(metric_val))
    # bootstrap
    rng = np.random.RandomState(random_state)
    boot_scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]
        if probs is not None:
            ps = probs[idx]
            score = metric_func(yt, yp, ps)
        else:
            score = metric_func(yt, yp)
        boot_scores.append(score)
    boot = np.array(boot_scores)
    # remove nan
    boot = boot[~np.isnan(boot)]
    if boot.size == 0:
        return float('nan'), (float('nan'), float('nan'))
    lower = np.percentile(boot, 100 * (alpha / 2))
    upper = np.percentile(boot, 100 * (1 - alpha / 2))
    return float(np.mean(boot)), (float(lower), float(upper))


def compute_metrics_with_ci(y_true, y_pred, probs):
    metrics = {}
    # metric definitions
    def acc(y, yhat): return accuracy_score(y, yhat)
    def f1m(y, yhat): return f1_score(y, yhat, average='macro', zero_division=0)
    def prec(y, yhat): return precision_score(y, yhat, average='macro', zero_division=0)
    def rec(y, yhat): return recall_score(y, yhat, average='macro', zero_division=0)
    def balacc(y, yhat): return balanced_accuracy_score(y, yhat)
    def auc(y, _, p):
        # binary or multiclass
        if p.ndim == 2 and p.shape[1] == 2:
            return roc_auc_score(y, p[:, 1])
        try:
            return roc_auc_score(y, p, multi_class='ovr', average='macro')
        except Exception:
            return float('nan')

    funcs = {
        'accuracy': acc,
        'f1': f1m,
        'precision': prec,
        'recall': rec,
        'roc_auc': auc,
        'balanced_acc': balacc,
    }

    for name, func in funcs.items():
        try:
            if name == 'roc_auc':
                mean, (lo, hi) = bootstrap_ci(func, y_true, y_pred, probs=probs)
            else:
                mean, (lo, hi) = bootstrap_ci(func, y_true, y_pred)
        except Exception:
            mean, lo, hi = 0.0, 0.0, 0.0
        # sanitize
        if np.isnan(mean): mean = 0.0
        if np.isnan(lo) or np.isnan(hi): lo, hi = 0.0, 0.0
        metrics[name] = mean
        metrics[f"{name}_ci"] = (lo, hi)
    return metrics


class Dataset(Dataset):
    def __init__(self, data_list, labels, win_len=1000, hop_len=1000, fs=500):
        self.segments = []
        self.segment_labels = []
        self.segment_subject_ids = []
        self.subject_true_labels = {}
        self.label_counts = {}
        print(f"Initializing EEGDataset with {len(data_list)} subjects")
        for subj_idx, (X, y) in enumerate(zip(data_list, labels)):
            if X is None or len(X) == 0: continue
            T, D = X.shape
            start = 0
            while start + win_len <= T:
                seg = X[start:start+win_len]
                self.segments.append(seg)
                self.segment_labels.append(y)
                self.segment_subject_ids.append(subj_idx)
                self.label_counts[y] = self.label_counts.get(y, 0) + 1
                if subj_idx not in self.subject_true_labels:
                    self.subject_true_labels[subj_idx] = y
                start += hop_len
        if not self.segments:
            raise ValueError("No valid segments found.")
        self.segments = np.stack(self.segments)
        self.segment_labels = np.array(self.segment_labels, dtype=np.int64)
        self.segment_subject_ids = np.array(self.segment_subject_ids, dtype=np.int64)
        print(f"Created {len(self.segments)} segments. Shape: {self.segments.shape}")
        dist = ", ".join([f"Label {l}: {c}" for l, c in sorted(self.label_counts.items())])
        print(f"  Class distribution: {dist}")

    def __len__(self): return len(self.segments)
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.segments[idx]).float(),
            torch.tensor(self.segment_labels[idx], dtype=torch.long),
            torch.tensor(self.segment_subject_ids[idx], dtype=torch.long)
        )


class fMRIDataset(Dataset):
    def __init__(self, data_list, labels, dataset_type="Unknown"):
        self.data = [torch.tensor(d, dtype=torch.float32) for d in data_list]
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.subject_ids = list(range(len(labels)))
        self.subject_true_labels = {i: int(label) for i, label in enumerate(labels)}
        self.label_counts = {}
        for label in labels:
            self.label_counts[label] = self.label_counts.get(label, 0) + 1
        print(f"{dataset_type} Dataset: {len(self.data)} samples, fMRI shape: {self.data[0].shape if self.data else 'N/A'}")
        dist = ", ".join([f"Class {l}: {c}" for l, c in sorted(self.label_counts.items())])
        print(f"  Class distribution: {dist}")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.subject_ids[idx]


class MetricsTracker:
    def __init__(self):
        self.metrics = {k: [] for k in (
            'accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'balanced_acc', 'loss'
        )}
        self.train_samples = []
        self.test_samples = []
        self.train_windows = []
        self.test_windows = []

    def update(self, metrics, n_train, n_test, n_train_windows=None, n_test_windows=None):
        for k in self.metrics:
            self.metrics[k].append(metrics.get(k, np.nan))
        self.train_samples.append(n_train)
        self.test_samples.append(n_test)
        if n_train_windows is not None:
            self.train_windows.append(n_train_windows)
        if n_test_windows is not None:
            self.test_windows.append(n_test_windows)

    def get_averages(self):
        result = {}
        for k, vals in self.metrics.items():
            vals = [v for v in vals if v is not None and not np.isnan(v)]
            if vals:
                result[k] = (np.mean(vals), np.std(vals))
        for name, lst in (
            ('train_samples', self.train_samples),
            ('test_samples', self.test_samples),
            ('train_windows', self.train_windows),
            ('test_windows', self.test_windows)
        ):
            vals = [v for v in lst if v is not None]
            if vals:
                result[name] = (np.mean(vals), np.std(vals))
        return result


def save_latent_outputs(save_dir, outputs_dict, prefix=""):
    os.makedirs(save_dir, exist_ok=True)
    for key, value in outputs_dict.items():
        arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        path = os.path.join(save_dir, f"{prefix}{key}")
        # For high-dim arrays, use compressed .npz
        if arr.ndim > 2:
            np.savez_compressed(path + '.npz', arr)
        else:
            np.save(path + '.npy', arr)
    print("Saved latent outputs:")
    for key, value in outputs_dict.items():
        shape = (value.detach().cpu().numpy().shape if torch.is_tensor(value)
                 else np.asarray(value).shape)
        print(f"  {key}: {shape}")


def evaluate_model(model, dataloader, device, save_dir=None, prefix="val_"):
    model.eval()
    total_loss = 0.0
    seg_preds, seg_probs, seg_labels, seg_sids = [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                x, y_seg, y_sid = batch
            else:
                x, y_seg = batch
                y_sid = torch.arange(len(y_seg))
            x = x.to(device)
            y_seg = y_seg.to(device)
            # seg_sids.extend(y_sid.cpu().numpy().tolist())  # save subject IDs

            cfg = getattr(model, 'config', {})
            a, b, l = cfg.get('alpha_recon',1.0), cfg.get('beta_kl',1.0), cfg.get('lambda_cls',1.0)
            out = model(x, y_seg)
            loss = a*out['recon_loss'] + b*out['kl_loss'] + l*out['cls_loss']
            total_loss += loss.item()

            probs = torch.softmax(out['logits'], dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            seg_preds .extend(preds.tolist())
            seg_probs .append(probs)
            seg_labels.extend(y_seg.cpu().numpy().tolist())
            seg_sids  .extend(y_sid.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(dataloader),1)
    seg_sids = np.array(seg_sids)
    unique_sids = np.unique(seg_sids)
    print(f"[DEBUG] Unique subject IDs in evaluation: {unique_sids}")  # DEBUG

    seg_probs = np.vstack(seg_probs)  
    seg_labels = np.array(seg_labels)
    seg_sids   = np.array(seg_sids)
    subj_true, subj_preds, subj_probs = [], [], []
    for sid in np.unique(seg_sids):
        idx = (seg_sids == sid)
        subj_true .append(seg_labels[idx][0])
        avg_p = seg_probs[idx].mean(axis=0)
        subj_probs.append(avg_p)
        subj_preds .append(int(avg_p.argmax()))

    subj_true = np.array(subj_true)
    subj_preds= np.array(subj_preds)
    subj_probs= np.vstack(subj_probs)
    print(f"[DEBUG] Unique subject IDs in final preds: {unique_sids}, preds for {len(subj_preds)} subjects")  # DEBUG

    metrics = compute_metrics_with_ci(subj_true, subj_preds, subj_probs)
    metrics.update({
        'loss': avg_loss,
        'labels': subj_true,
        'preds':  subj_preds,
        'probs':  subj_probs
    })

    if save_dir:
        np.savez_compressed(
            os.path.join(save_dir, f"{prefix}subject_metrics.npz"),
            true=subj_true, pred=subj_preds, prob=subj_probs
        )

    return metrics



def visualize_latent_space(model, dataloader, save_path, device, epoch=None, tsne_seed=42):
    model.eval()
    latents, labels = [], []
    with torch.no_grad():
        for x, y, *_ in dataloader:
            x = x.to(device)
            y = y.to(device).long()
            out = model(x, y)
            latents.append(out['z'].mean(dim=1).cpu().numpy())
            labels.append(y.cpu().numpy())
    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels, axis=0)
    tsne = TSNE(n_components=2, perplexity=30, max_iter=500, random_state=tsne_seed)
    lat2 = tsne.fit_transform(latents)
    plt.figure(figsize=(10,8))
    scatter = plt.scatter(lat2[:,0], lat2[:,1], c=labels, alpha=0.6)
    plt.colorbar(scatter, label='Class')
    title = f'Latent Space (Epoch {epoch})' if epoch is not None else 'Latent Space'
    plt.title(title)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path, class_names=None, normalize=False, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:,None]
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names)) if class_names else None
    if class_names:
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                     ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_loss_curve(history, save_path):
    plt.figure(figsize=(8,6))
    plt.plot(history.get('train_loss', []), label='Train Loss')
    plt.plot(history.get('val_loss', []), label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true, y_scores, save_path, title="ROC Curve"):
    if len(np.unique(y_true))<2:
        print("ROC curve requires at least two classes.")
        return
    RocCurveDisplay.from_predictions(y_true, y_scores[:,1])
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_per_fold_metrics(metrics_tracker, save_path):
    metrics = metrics_tracker.metrics
    plt.figure(figsize=(10,6))
    for key in ['accuracy','f1','roc_auc']:
        if key in metrics and metrics[key]:
            plt.plot(range(1,len(metrics[key])+1), metrics[key], marker='o', label=key.upper())
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Per-Fold Classification Metrics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_ci_bar(metrics_dict, save_path, metric='f1'):
    raw = metrics_dict.get(metric, [])
    data = [(v, ci) for v, ci in raw if v is not None and not np.isnan(v)]
    if not data:
        print(f"No valid data for {metric}")
        return
    values, cis = zip(*data)
    lowers = [ci[0] for ci in cis]
    uppers = [ci[1] for ci in cis]
    folds = np.arange(1, len(values)+1)
    errs = np.array(uppers) - np.array(values)
    plt.figure(figsize=(8,6))
    plt.bar(folds, values, yerr=errs, capsize=5)
    plt.xlabel('Fold')
    plt.ylabel(metric.replace('_',' ').title())
    plt.title(f'{metric.upper()} with 95% CI')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_final_results(metrics_tracker, results_dir, all_ci_metrics):
    os.makedirs(results_dir, exist_ok=True)
    # align lengths
    max_len = max(len(lst) for lst in metrics_tracker.metrics.values())
    for k,v in metrics_tracker.metrics.items():
        while len(v)<max_len: v.append(np.nan)
    # per-fold CSV
    df = pd.DataFrame(metrics_tracker.metrics)
    df.to_csv(os.path.join(results_dir,'per_fold_metrics.csv'), index=False)
    # summary stats
    final = metrics_tracker.get_averages()
    # write text
    with open(os.path.join(results_dir,'cross_validation_metrics.txt'),'w') as f:
        f.write("=== Cross-Validation Results ===\n")
        for i,row in df.iterrows():
            f.write(f"Fold {i+1}: "+", ".join([f"{c}={row[c]:.4f}" for c in df.columns])+"\n")
        f.write("\nSummary (mean ± std):\n")
        for k,(m,s) in final.items():
            f.write(f"{k}: {m:.4f} ± {s:.4f}\n")
    # save JSON
    with open(os.path.join(results_dir,'metrics_summary.json'),'w') as jf:
        json.dump(final, jf, indent=2)
    print("Saved final results to", results_dir)