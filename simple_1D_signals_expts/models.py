#!/usr/bin/env python3
"""
Models for basic RandOpt for toy experiments
"""

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, width, depth, dim_in, dim_out, init_type, device):
        super().__init__()
        self.width, self.depth, self.dim_in,self.dim_out, self.device = width, depth, dim_in, dim_out, device
        self.init_type = init_type

        # Build layers: input -> [hidden + ReLU] * (depth-1) -> output
        layers = [nn.Linear(self.dim_in, width, device=device)]
        for _ in range(depth - 2):
            layers.extend([nn.ReLU(), nn.Linear(width, width, device=device)])
        layers.extend([nn.ReLU(), nn.Linear(width, dim_out, device=device)])
        self.layers = nn.ModuleList(layers)

    def forward(self, ctx):
        """Forward pass. Predicts y_next given ctx.
        
        Args:
            ctx: tensor of shape [batch_size, ctx_sz] or [ctx_sz] (raw values)
        """
        # Handle both batched and non-batched inputs
        was_1d = ctx.dim() == 1
        if was_1d:
            if ctx is not None:
                ctx = ctx.unsqueeze(0)
        
        h = ctx
        for layer in self.layers:
            h = layer(h)
        
        if was_1d:
            h = h.squeeze(0)
        
        return h.squeeze(-1)
    
    def compute_loss(self, ctx, y):
        """Compute loss.
        
        Args:
            ctx: tensor of shape [batch_size, ctx_sz]
            y: tensor of shape [batch_size, 1] or [batch_size]
        """
        y_pred = self.forward(ctx)  # [batch_size]
        return nn.MSELoss()(y_pred, y.squeeze(-1))

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if self.init_type == "xavier":
                    nn.init.xavier_uniform_(layer.weight)
                elif self.init_type == "kaiming":
                    nn.init.kaiming_uniform_(layer.weight)
                elif self.init_type == "ortho":
                    nn.init.orthogonal_(layer.weight)
                else:
                    raise ValueError(f"Invalid initialization type: {self.init_type}")
                nn.init.zeros_(layer.bias)
    
    def perturb_weights(self, seed, sigma):
        torch.manual_seed(seed)
        for p in self.parameters():
            p.data.add_(torch.randn_like(p.data) * sigma)

    def AR_rollout(self, ctx, T):
        """AR rollout.
        
        Args:
            ctx: tensor of shape [batch_size, ctx_sz]
            T: number of steps to roll out

        Returns:
            y_preds: tensor of shape [batch_size, T]
        """
        y_preds = []
        for t in range(T):
            y_pred = self.forward(ctx)  # [batch_size]
            ctx = torch.cat([ctx, y_pred.unsqueeze(-1)], dim=1)
            ctx = ctx[:, 1:]
            y_preds.append(y_pred)

        y_preds = torch.stack(y_preds, dim=1)  # [batch_size, T]
        
        return y_preds