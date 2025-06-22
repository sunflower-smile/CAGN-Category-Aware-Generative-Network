import os
import json
import numpy as np
from glob import glob
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from datetime import datetime

from trainer1 import Trainer
from trainer1_nsga2 import run_nsga2
from utils1 import (
    MetricsTracker,
    plot_roc_curve,
    plot_confusion_matrix,
    Dataset,
    visualize_latent_space,
    compute_channel_stats,
    standardize_all,
    plot_per_fold_metrics,
    save_final_results,
    evaluate_model
)


def load_subject_data(folder):
    files = sorted(glob(os.path.join(folder, '*.npy')))
    data = []
    for f in files:
        arr = np.load(f)
        if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
            arr = arr.T
        data.append(arr)
    return data


def convert_ndarrays_to_lists(obj):
    if isinstance(obj, dict):
        return {k: convert_ndarrays_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays_to_lists(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    else:
        return obj


def create_results_dir(base_dir="results_MD_HC"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, f"cls_EEG_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = create_results_dir()

    # 1) load all subjects
    ds1 = load_subject_data(config['ds1_FOLDER'])
    ds2 = load_subject_data(config['ds2_FOLDER'])
    all_data = ds1 + ds2
    all_labels = [0]*len(ds1) + [1]*len(ds2)
    config['data_dim'] = all_data[0].shape[1]

    skf = StratifiedKFold(n_splits=config['n_splits'],
                          shuffle=True, random_state=config['random_state'])
    metrics_tracker = MetricsTracker()
    all_ci_metrics = {m: [] for m in ['accuracy','f1','precision','recall','roc_auc','balanced_acc']}

    subj_idx = np.arange(len(all_data))
    for fold, (train_idxs, test_idxs) in enumerate(skf.split(subj_idx, all_labels), start=1):
        print(f"\n=== Fold {fold}/{config['n_splits']} ===")
        fold_dir = os.path.join(results_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # A) subject‐level train/test split
        train_subj, test_subj = train_idxs.tolist(), test_idxs.tolist()
        assert set(train_subj).isdisjoint(test_subj)

        # B) inner split train→train/val
        y_train = [all_labels[i] for i in train_subj]
        tr_subj, val_subj = train_test_split(
            train_subj, test_size=0.2, random_state=config['random_state'],
            stratify=y_train
        )
        assert set(tr_subj).isdisjoint(val_subj)
        assert set(val_subj).isdisjoint(test_subj)

        # C) assemble lists
        train_data = [all_data[i] for i in tr_subj]
        train_labels = [all_labels[i] for i in tr_subj]
        val_data = [all_data[i] for i in val_subj]
        val_labels = [all_labels[i] for i in val_subj]
        test_data = [all_data[i] for i in test_subj]
        test_labels = [all_labels[i] for i in test_subj]

        # D) normalize
        mean, std = compute_channel_stats(train_data)
        train_data = standardize_all(train_data, mean, std)
        val_data   = standardize_all(val_data, mean, std)
        test_data  = standardize_all(test_data, mean, std)

        # E) build A_norm (unused by Trainer here if model doesn't need it)
        Cs = [np.corrcoef(X, rowvar=False) for X in train_data]
        A = np.mean(Cs, axis=0)
        np.fill_diagonal(A, 1.0)
        Dsum = np.sum(A, axis=1)
        A_norm = np.diag(1/np.sqrt(Dsum)) @ A @ np.diag(1/np.sqrt(Dsum))
        A_tensor = torch.from_numpy(A_norm).float().to(device)

        # F) windowing
        win_len, hop_len = config['win_len'], config['hop_len']
        train_ds = Dataset(train_data, train_labels, win_len, hop_len)
        val_ds   = Dataset(val_data,   val_labels,   win_len, hop_len)
        test_ds  = Dataset(test_data,  test_labels,  win_len, hop_len)
        if len(train_ds)==0 or len(val_ds)==0 or len(test_ds)==0:
            print(f"[WARNING] fold {fold} empty segments, skip")
            continue

        # G) loaders
        loader_params = dict(batch_size=config['batch_size'],
                             num_workers=config['num_workers'],
                             pin_memory=True)
        train_loader = DataLoader(train_ds, shuffle=True, **loader_params)
        val_loader   = DataLoader(val_ds,   shuffle=False, **loader_params)
        test_loader  = DataLoader(test_ds,  shuffle=False, **loader_params)

        # H) Trainer init & first training on train/val split
        trainer = Trainer(config)
        print("→ First training on train / val …")
        best_val = trainer.train(train_loader, val_loader, fold_dir)
        # save first model
        torch.save(dict(
            model_state=trainer.model.state_dict(),
            config=config,
            best_val=best_val
        ), os.path.join(fold_dir, "first_model.pth"))

        # I) NSGA-II hyperparam search on validation only
        print("→ NSGA-II hyperparam search …")
        config, pareto = run_nsga2(config, trainer.model, val_loader, device,
                                  strategy='cls_min', save_all=True)
        with open(os.path.join(fold_dir, "pareto_front.json"), 'w') as f:
            json.dump(convert_ndarrays_to_lists(pareto), f, indent=2)

        # I.2) retrain from scratch on train+val with new α,β,λ
        print("→ Retraining from scratch on train+val …")
        trainval_data   = train_data + val_data
        trainval_labels = train_labels + val_labels
        trainval_ds = Dataset(trainval_data, trainval_labels, win_len, hop_len)
        trainval_loader = DataLoader(trainval_ds, shuffle=True, **loader_params)

        # new Trainer instance, skip validation inside train()
        final_trainer = Trainer(config)
        final_trainer.config['patience'] = 0
        final_trainer.train(trainval_loader, val_loader=None, fold_dir=fold_dir)
        final_model = final_trainer.model

        # J) final evaluation on test set
        print("→ Final evaluation on test set …")
        test_metrics = evaluate_model(final_model, test_loader,
                                      device, save_dir=fold_dir, prefix='test_')
        with open(os.path.join(fold_dir, "fold_test_metrics.json"), 'w') as f:
            json.dump(convert_ndarrays_to_lists(test_metrics), f, indent=2)

        # K) visualize & track
        visualize_latent_space(final_model, test_loader,
                               os.path.join(fold_dir,"latent_space.png"),
                               device, epoch=config['num_epochs'])
        plot_roc_curve(test_metrics['labels'], np.array(test_metrics['probs']),
                       os.path.join(fold_dir,"roc_curve.png"),
                       title=f"ROC Fold {fold}")
        plot_confusion_matrix(test_metrics['labels'], test_metrics['preds'],
                              os.path.join(fold_dir,"conf_matrix.png"),
                              class_names=["HC","MD"], normalize=True,
                              title=f"CM Fold {fold}")

        # update cross‐fold trackers
        track = {k:test_metrics[k] for k in
                 ['loss','accuracy','f1','precision','recall','roc_auc','balanced_acc']}
        metrics_tracker.update(track,
            n_train=len(tr_subj), n_test=len(test_subj),
            n_train_windows=len(train_ds), n_test_windows=len(test_ds)
        )
        for m in all_ci_metrics:
            all_ci_metrics[m].append((test_metrics[m], test_metrics[f"{m}_ci"]))

    # L) summary & save
    save_final_results(metrics_tracker, results_dir, all_ci_metrics)
    plot_per_fold_metrics(metrics_tracker, os.path.join(results_dir,"per_fold.png"))
    print("All results saved in:", results_dir)


if __name__ == "__main__":
    config = {
        'batch_size': 32, 'num_epochs': 20, 'learning_rate': 1e-3,
        'n_splits': 10, 'random_state': 42, 'num_workers': 4,
        'patience': 5, 'warmup_epochs': 10, 'max_grad_norm': 1.0,
        'rnn_hidden': 64, 'rnn_layers': 2, 'latent_dim': 8,
        'num_components': 8, 'cls_hidden': 64, 'num_classes': 2,
        'tau': 1.0, 'dropout': 0.3,
        'alpha_recon': 1.0, 'beta_kl': 1.0, 'lambda_cls': 1.0,
        'win_len': 500, 'hop_len': 500,
        'ds1_FOLDER': '/home/uais1/Echo/HBN_newData/HC_MD-dataset/MD_HC/HC',
        'ds2_FOLDER': '/home/uais1/Echo/HBN_newData/HC_MD-dataset/MD_HC/MD'
    }
    main(config)
