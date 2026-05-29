#!/usr/bin/env python3
"""Basic RandOpt for toy experiments"""

import argparse
from datetime import datetime
import json
import os
import random

import numpy as np
import torch

import copy

from simple_1D_signals_expts import datasets
from simple_1D_signals_expts import models
from simple_1D_signals_expts import pretrain
from simple_1D_signals_expts import posttrain
from simple_1D_signals_expts import eval

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Toy Expt v3", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Datasets
    p.add_argument("--pretrain_dataset", type=str, default=None)
    p.add_argument("--posttrain_dataset", type=str, default=None)
    p.add_argument("--test_dataset", type=str, default=None)
    p.add_argument("--res_x", type=float, default=0.1)

    # Pretraining
    p.add_argument("--pretrain_bsz", type=int, default=256)
    p.add_argument("--posttrain_dataset_sz", type=int, default=1024)
    p.add_argument("--pretrain_iters", type=int, default=1000)
    p.add_argument("--pretraining_lr", type=float, default=0.001)
    p.add_argument("--test_bsz", type=int, default=256)
    p.add_argument("--base_init", type=str, default="xavier")

    # Model
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--ctx_sz", type=int, default=50)
    p.add_argument("--fut_sz", type=int, default=100)

    # Post-training (RandOpt)
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--N", type=int, default=300)
    p.add_argument("--K", type=int, default=30)

    # Plotting
    p.add_argument("--num_random_models_to_plot", type=int, default=25)
    p.add_argument("--num_top_k_models_to_plot", type=int, default=5)
    p.add_argument("--plot_top_k", type=lambda x: x.lower() == 'true', default=False)
    p.add_argument("--plot_top_1", type=lambda x: x.lower() == 'true', default=False)
    p.add_argument("--plot_random_models", type=lambda x: x.lower() == 'true', default=False)
    p.add_argument("--plot_ensemble", type=lambda x: x.lower() == 'true', default=False)

    # Misc
    p.add_argument("--logging_dir", type=str, default="log")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--global_seed", type=int, default=42)

    if argv is None:
        argv = []
    args = p.parse_args(argv)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(args):
    """Setup logging directory and save args."""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging_dir = f"{args.logging_dir}"
    os.makedirs(logging_dir, exist_ok=True)

    # Save args (handle device serialization)
    args_dict = vars(args).copy()
    args_dict['device'] = str(args.device)
    with open(f"{logging_dir}/args.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    return logging_dir


def create_model(args, dim_in):
    """Create and initialize a model."""
    model = models.Net(
        width=args.width, depth=args.depth, dim_in=dim_in,
        dim_out=1, init_type=args.base_init, device=args.device
    )
    model.init_weights()
    return model

def create_model_from_seed(seed, base, args):
    """Create and initialize a model from a seed."""
    model = copy.deepcopy(base)
    model.perturb_weights(seed, args.sigma)
    return model


def main(args):
    set_seed(args.global_seed)

    print(f"{'='*60}\n1D curves experiment\n{'='*60}")
    print(f"N: {args.N} | K: {args.K}")

    args.logging_dir = setup_logging(args)

    ## Create and pretrain base model
    base_model = create_model(args, dim_in=args.ctx_sz)
    if args.pretrain_iters > 0 and args.pretrain_dataset is not None:
        base_model = pretrain.pretrain_base_model(base_model, args.pretrain_dataset, args)

    ## Post-train with RandOpt
    top_k_seeds = posttrain.RandOpt(base_model, args.posttrain_dataset, args,
        N=args.N, sigma=args.sigma, K=args.K
    )

    ## Evaluate
    test_dataset = datasets.load_data(args.test_bsz, args.test_dataset, args)
    ctx_x, ctx_y, fut_x, fut_y = test_dataset

    # base model predictions
    base_preds = base_model.AR_rollout(ctx_y, args.fut_sz)

    # top-k models predictions
    top_k_preds = []
    for k in range(args.K):
        seed = top_k_seeds[k]  # top_k_seeds is list of seeds
        model_k = create_model_from_seed(seed, base_model, args)
        top_k_preds.append(model_k.AR_rollout(ctx_y, args.fut_sz))

    # random models predictions
    random_models_preds = []
    for i in range(args.num_random_models_to_plot):
        model_i = create_model_from_seed(random.randint(0, args.N - 1), base_model, args)
        random_models_preds.append(model_i.AR_rollout(ctx_y, args.fut_sz))

    # ensemble's predictions
    ensemble_preds = torch.stack(top_k_preds).mean(axis=0)

    # evaluate prediction mses
    base_mse, base_se = eval.compute_mse(base_preds, fut_y)
    top_1_mse, top_1_se = eval.compute_mse(top_k_preds[0], fut_y)
    ensemble_mse, ensemble_se = eval.compute_mse(ensemble_preds, fut_y)

    print(f"Base model MSE: {base_mse:.4f}")
    print(f"Top-1 model MSE: {top_1_mse:.4f}")
    print(f"Ensemble MSE: {ensemble_mse:.4f}")

    ## Plot results
    eval.plot_predictions(base_preds, top_k_preds, random_models_preds, ensemble_preds, ctx_x, ctx_y, fut_x,fut_y, args)
    eval.plot_performance(base_mse.item(), base_se.item(), top_1_mse.item(), top_1_se.item(), ensemble_mse.item(), ensemble_se.item(), args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
