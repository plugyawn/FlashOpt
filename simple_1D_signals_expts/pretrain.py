#!/usr/bin/env python3
"""
Pretraining code for basic RandOpt for toy experiments
"""

import time
import numpy as np
import torch
from simple_1D_signals_expts import datasets


def pretrain_base_model(model, pretrain_dataset, args):
    """Pretrain base model with SGD on the pretraining train set."""
    print(f"\n{'='*60}\nPRETRAINING BASE MODEL\n{'='*60}")
    print(f"Batch size: {args.pretrain_bsz}, Iterations: {args.pretrain_iters}")

    t0 = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.pretraining_lr)
    log_interval = max(1, args.pretrain_iters // 10)

    for i in range(args.pretrain_iters):
        model.train()
        optimizer.zero_grad()
        _, ctx_y, _, fut_y = datasets.load_data(args.pretrain_bsz, pretrain_dataset, args)
        loss = model.compute_loss(ctx_y, fut_y[:,[0]]) # just next token prediction, so x_fut_0
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print(f"Iter {i+1}/{args.pretrain_iters} - Loss: {loss.item():.4f}")

    print(f"Completed in {time.time() - t0:.2f}s")
    return model
