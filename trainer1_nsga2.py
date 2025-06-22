# trainer_nsga2.py
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LAVAEObjective(Problem):
    def __init__(self, model, dataloader, device):
        super().__init__(n_var=3, n_obj=3, n_constr=0, xl=0.0, xu=1.0)
        self.model = model.to(device).eval()
        self.device = device

        recon_sum = kl_sum = cls_sum = 0.0
        num_batches = 0
        with torch.no_grad():
            for x_batch, y_batch, _ in dataloader:
                x = x_batch.to(self.device)
                y = y_batch.to(self.device).long()
                outputs = self.model(x, y)
                recon_sum += outputs['recon_loss'].item()
                kl_sum    += outputs['kl_loss'].item()
                cls_sum   += outputs['cls_loss'].item()
                num_batches += 1

        # average per batch
        self.base_recon = recon_sum / num_batches
        self.base_kl    = kl_sum    / num_batches
        self.base_cls   = cls_sum   / num_batches

    def _evaluate(self, X, out, *args, **kwargs):
        # X: (pop, 3) weights in [0,1]
        f1 = X[:, 0] * self.base_recon
        f2 = X[:, 1] * self.base_kl
        f3 = X[:, 2] * self.base_cls
        out['F'] = np.column_stack([f1, f2, f3])


def run_nsga2(config, model_to_evaluate, data_loader, device,
              strategy='cls_min', min_lambda=0.2, save_all=False):
    problem = LAVAEObjective(model=model_to_evaluate, dataloader=data_loader, device=device)
    algorithm = NSGA2(pop_size=20)
    res = minimize(problem, algorithm, ('n_gen', 10), seed=1, verbose=False)

    weights = res.X
    losses  = res.F

    # select best
    if strategy == 'cls_min':
        best_idx = int(np.argmin(losses[:, 2]))
    elif strategy == 'recon_min':
        best_idx = int(np.argmin(losses[:, 0]))
    elif strategy == 'f1_cls_sum':
        norm = (losses - losses.min(axis=0)) / (losses.ptp(axis=0) + 1e-8)
        scores = norm[:,0] + norm[:,2]
        best_idx = int(np.argmin(scores))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    alpha, beta, lamb = weights[best_idx]
    lamb = max(lamb, min_lambda)
    print(f"[NSGA-II] Strategy={strategy} | α={alpha:.4f}, β={beta:.4f}, λ={lamb:.4f}")

    config['alpha_recon'] = float(alpha)
    config['beta_kl']     = float(beta)
    config['lambda_cls']  = float(lamb)

    if save_all:
        plot_pareto_front(weights, losses, best_idx, save_path='pareto_front.png')
        save_weights_csv(weights, path='nsga2_weights.csv')
        return config, {'X': weights, 'F': losses}
    return config


def plot_pareto_front(X, F, best_idx=None, save_path='pareto_front.png'):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(F[:,0], F[:,1], F[:,2], s=30, label='Solutions')
    if best_idx is not None:
        ax.scatter(F[best_idx,0], F[best_idx,1], F[best_idx,2], s=80, marker='^', label='Selected')
    ax.set_xlabel('Recon Loss')
    ax.set_ylabel('KL Div')
    ax.set_zlabel('Cls Loss')
    ax.set_title('NSGA-II Pareto Front')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_weights_csv(X, path='nsga2_weights.csv'):
    df = pd.DataFrame(X, columns=['alpha','beta','lambda'])
    df.to_csv(path, index=False)