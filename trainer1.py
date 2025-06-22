# trainer.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
from torchinfo import summary
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import LAVAE
from utils import evaluate_model, plot_loss_curve


def kl_anneal(epoch, warmup=20, max_beta=1.0):
    """
    KL annealing: ramp from 0 to max_beta over 'warmup' epochs, then stay at max_beta.
    """
    if epoch >= warmup:
        return max_beta
    # linear ramp
    return max_beta * (epoch / warmup)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LAVAE(config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.initialize_weights()

        # summary: pass label indices, not one-hot
        dummy_x = torch.randn(config['batch_size'], config['win_len'], config['data_dim']).to(self.device)
        dummy_y = torch.zeros(config['batch_size'], dtype=torch.long).to(self.device)
        summary(self.model, input_data=(dummy_x, dummy_y))

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config['num_epochs'], eta_min=1e-6)

    def initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LSTM, nn.GRU)):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def train(self, train_loader, val_loader=None, fold_dir=None):
        num_epochs = self.config.get('num_epochs')
        patience = self.config.get('patience')
        max_grad_norm = self.config.get('max_grad_norm')
        warmup_epochs = self.config.get('warmup_epochs')

        best_val_loss = float('inf')
        best_val_f1 = 0.0
        patience_counter = 0
        best_metrics = None

        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = recon_sum = kl_sum = cls_sum = 0.0

            alpha = self.config.get('alpha_recon', 1.0)
            beta  = self.config.get('beta_kl', 1.0)
            lamb  = self.config.get('lambda_cls', 1.0)

            for x, y_segment_label, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                x = x.to(self.device)
                y = y_segment_label.to(self.device)

                self.optimizer.zero_grad()

                curr_beta = kl_anneal(epoch, warmup=warmup_epochs, max_beta=beta)
                outputs = self.model(x, y)

                recon_loss = outputs['recon_loss']
                kl_loss    = outputs['kl_loss']
                cls_loss   = outputs['cls_loss']

                loss = alpha * recon_loss + curr_beta * kl_loss + lamb * cls_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                recon_sum  += recon_loss.item()
                kl_sum     += kl_loss.item()
                cls_sum    += cls_loss.item()

            # average per batch
            num_batches = len(train_loader)
            avg_loss  = total_loss / num_batches
            avg_recon = recon_sum  / num_batches
            avg_kl    = kl_sum     / num_batches
            avg_cls   = cls_sum    / num_batches

            # val_metrics = self.evaluate(val_loader, save_dir=fold_dir)
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, save_dir=fold_dir)
                val_loss    = val_metrics['val_loss']
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_metrics['val_acc'])
                self.history['val_f1'].append(val_metrics['val_f1'])
            else:
                val_metrics = None
                val_loss    = None
                self.history['val_loss'].append(None)
                self.history['val_acc'].append(None)
                self.history['val_f1'].append(None)

            # plot_loss_curve(self.history, os.path.join(fold_dir, "loss_curve.png"))
            if fold_dir is not None:
                plot_loss_curve(self.history, os.path.join(fold_dir, "loss_curve.png"))

            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
           
            # always print train stats
            print(f"\nEpoch {epoch+1}: lr={lr:.6f}")
            print(f"Train loss={avg_loss:.4f} (Recon={avg_recon:.4f}, KL={avg_kl:.4f}, Cls={avg_cls:.4f})")
            # only print val stats if we actually did validation
            if val_loader is not None:
                print(f"Val loss={val_loss:.4f}, "
                      f"Acc={val_metrics['val_acc']:.4f}, "
                      f"F1={val_metrics['val_f1']:.4f}, "
                      f"Precision={val_metrics.get('val_precision', 0):.4f}, "
                      f"Recall={val_metrics.get('val_recall', 0):.4f}, "
                      f"AUC={val_metrics.get('val_roc_auc', 0):.4f}, "
                      f"Balanced Acc={val_metrics.get('val_balanced_acc', 0):.4f}")
            # early stopping & checkpoint
            should_save = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                should_save = True
            if val_metrics['val_f1'] > best_val_f1:
                best_val_f1 = val_metrics['val_f1']
                should_save = True

            if should_save:
                best_metrics = val_metrics
                self._save_model(os.path.join(fold_dir, 'best_model.pth'),
                                  epoch, best_metrics)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        return best_metrics

    def evaluate(self, dataloader, save_dir=None):
        metrics = evaluate_model(self.model, dataloader, self.device, save_dir=save_dir)
        return {
            'val_loss': metrics['loss'],
            'val_acc': metrics['accuracy'],
            'val_f1': metrics['f1'],
            'val_precision': metrics.get('precision'),
            'val_recall': metrics.get('recall'),
            'val_roc_auc': metrics.get('roc_auc'),
            'val_balanced_acc': metrics.get('balanced_acc'),
            'labels': metrics.get('labels'),
            'probs': metrics.get('probs'),
            'preds': metrics.get('preds'),
        }

    def _save_model(self, path, epoch, metrics):
        # 保存模型参数和学习到的协方差矩阵（如果有）
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        # 若模型有协方差矩阵属性，则保存
        if hasattr(self.model, 'covariance_matrix'):
            cov = self.model.covariance_matrix
            if torch.is_tensor(cov):
                state['covariance_matrix'] = cov.detach().cpu().numpy()
            else:
                state['covariance_matrix'] = cov
        elif hasattr(self.model, 'get_covariance_matrix'):
            cov = self.model.get_covariance_matrix()
            if torch.is_tensor(cov):
                state['covariance_matrix'] = cov.detach().cpu().numpy()
            else:
                state['covariance_matrix'] = cov
        torch.save(state, path)
