#!/usr/bin/env python3
"""
Evaluation and plotting code for basic RandOpt for toy experiments
"""

import copy
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import simple_1D_signals_expts.datasets as datasets


def _save_fig(args, suffix, timestamp=None):
    """Save figure to logging directory."""
    if args.pretrain_dataset is None:
        pretrain_dataset = args.base_init
    else:
        pretrain_dataset = args.pretrain_dataset
    name = f"{pretrain_dataset}_{args.posttrain_dataset}_{args.test_dataset}_{suffix}"
    #ts = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f"{args.logging_dir}/{name}.png", dpi=150, bbox_inches='tight')
    #plt.savefig(f"{args.logging_dir}/{name}_{ts}.pdf", bbox_inches='tight')


def _set_ylim(ax, y_true, margin=0.2):
    """Set y-axis limits with margin."""
    y_min, y_max = y_true.min(), y_true.max()
    y_margin = (y_max - y_min) * margin
    ax.set_ylim([y_min - y_margin, y_max + y_margin])


def compute_mse(y_pred, y_true):
    """Compute mean squared error."""
    squared_error = ((y_pred - y_true) ** 2).flatten()
    mse = squared_error.mean()
    std = squared_error.std()
    se = std / np.sqrt(squared_error.shape[0])
    return mse, se

def eval_model(model, ctx_y, fut_y, args):
    """Evaluate model on a dataset."""
    y_pred = model.AR_rollout(ctx_y, args.fut_sz)
    return compute_mse(y_pred, fut_y)

def plot_predictions(base_preds, top_k_preds, random_models_preds, ensemble_preds, ctx_x, ctx_y, fut_x, fut_y, args):

    ctx_color = [0.16, 0.44, 0.73] # deep blue
    base_color = 'k' # black
    top_k_color = [0.85, 0.33, 0.25] # muted coral
    top_1_color = [0.85, 0.33, 0.25] # muted coral
    random_model_color = 'k'
    ensemble_color = [0.941, 0.765, 0.157] # gold yellow
    gt_color = [0.16, 0.44, 0.73] # deep blue

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    for i in range(3):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        ax.plot(fut_x[i].cpu().numpy(), base_preds[i].detach().cpu().numpy(), '-', linewidth=4.5, label=f'Base model', c=base_color)

        if args.plot_random_models:
            for j in range(args.num_random_models_to_plot):
                label = 'Random perturbs' if j == 0 else None
                ax.plot(fut_x[i].cpu().numpy(), random_models_preds[j][i].detach().cpu().numpy(), '-', linewidth=2.0, alpha=0.2, label=label, c=random_model_color)

        if args.plot_top_k:
            for j in range(args.num_top_k_models_to_plot):
                label = 'Top-k perturbs' if j == 0 else None
                ax.plot(fut_x[i].cpu().numpy(), top_k_preds[j][i].detach().cpu().numpy(), '-', linewidth=2.0, label=label, c=top_k_color)

        if args.plot_top_1:
            ax.plot(fut_x[i].cpu().numpy(), top_k_preds[0][i].detach().cpu().numpy(), '-', linewidth=4.0, label=f'Best guess', c=top_1_color)

        if args.plot_ensemble:
            ax.plot(fut_x[i].cpu().numpy(), ensemble_preds[i].detach().cpu().numpy(), '-', linewidth=4.0, label=f'Ensemble', c=ensemble_color)

        ax.plot(ctx_x[i].cpu().numpy(), ctx_y[i].cpu().numpy(), '-', linewidth=4.0, label='Context', c=ctx_color)
        ax.plot(fut_x[i].cpu().numpy(), fut_y[i].cpu().numpy(), label='Ground truth', linewidth=4.0, linestyle='--', color=gt_color)

        ax.set_ylim([-2*datasets.SCALE, 2*datasets.SCALE])
        if i == 0:
            ax.legend(ncol=2, fontsize=18, loc='lower left')
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

        _save_fig(args, f"sample{i}", timestamp=ts)
        plt.show()
        plt.close(fig)

def plot_performance(base_mse, base_se, top_1_mse, top_1_se, ensemble_mse, ensemble_se, args):
    plt.rcParams.update({'font.size': 24})

    ctx_color = [0.16, 0.44, 0.73] # deep blue
    base_color = 'k' # black
    top_k_color = [0.85, 0.33, 0.25] # muted coral
    top_1_color = [0.85, 0.33, 0.25] # muted coral
    random_model_color = 'k'
    ensemble_color = [0.941, 0.765, 0.157] # gold yellow
    gt_color = [0.16, 0.44, 0.73] # deep blue

    """Plot performance."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 10))
    ax.bar(x=['Base model', 'Top-1 perturb', 'Ensemble'], height=[base_mse, top_1_mse, ensemble_mse], yerr=[base_se, top_1_se, ensemble_se], capsize=5, color=[base_color, top_1_color, ensemble_color])
    #ax.set_ylim([0.0, np.max([base_mse, top_1_mse, ensemble_mse]) + 0.1])
    ax.set_ylim([0.0, 0.65])
    ax.set_aspect('auto')
    ax.set_ylabel('MSE ± SE')
    plt.xticks(rotation=45, ha='right')

    _save_fig(args, "performance")
    plt.show()
    plt.close(fig)
