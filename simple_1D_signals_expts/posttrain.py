#!/usr/bin/env python3
"""
Post-training code for RandOpt toy experiments
"""

import copy
import time
import numpy as np
import torch
from simple_1D_signals_expts import datasets
from simple_1D_signals_expts import eval


def RandOpt(base_model, posttrain_dataset, args, N=10, sigma=0.01, K=5, weighted=True, temperature=1.0):
    """Post-train model with RandOpt on the post-training train set."""
    print(f"\n{'='*60}\nPOST-TRAINING MODEL\n{'='*60}")

    t0 = time.time()

    dataset = datasets.load_data(args.posttrain_dataset_sz, posttrain_dataset, args)

    # Sample N perturbed models and evaluate on the post-train dataset
    ctx_x, ctx_y, fut_x, fut_y = dataset
    model_scores = []  # (seed, mse)

    for seed in range(N):
        perturbed = copy.deepcopy(base_model)
        perturbed.perturb_weights(seed, sigma)
        perturbed.eval()

        with torch.no_grad():
            mse = eval.eval_model(perturbed, ctx_y, fut_y, args)

        model_scores.append((seed, mse))
        del perturbed  # Free memory immediately

    # Select top-K based on MSE
    model_scores.sort(key=lambda x: x[1])
    top_k_seeds = [model_scores[i][0] for i in range(K)]

    print(f"Completed in {time.time() - t0:.2f}s")

    return top_k_seeds
