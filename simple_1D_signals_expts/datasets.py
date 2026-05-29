#!/usr/bin/env python3
"""
Datasets for basic RandOpt for toy experiments
"""

import numpy as np
import torch

FREQ = 4.0
SCALE = 1.0

def generate_sigmoid():
    """Generate sigmoid/tanh S-curve."""
    phase = np.random.uniform(0, 2*np.pi) - np.pi
    amp = np.random.uniform(0.5, 1.5)
    y_offset = np.random.uniform(-0.5, 0.5)

    def fn(x):
        x = np.asarray(x)
        return amp * np.tanh(0.1 * x + phase) + y_offset
    return fn

def generate_line():
    """Generate line."""
    slope = np.random.uniform(-0.5, 0.5)
    intercept = np.random.uniform(-1.0, 1.0)
    return lambda x: slope*x + intercept

def generate_one_line():
    """Generate one line."""
    slope = -0.25
    intercept = 0.0
    return lambda x: slope*x + intercept
    
def generate_harmonic():
    """Generate sum of two harmonics (sin + sin(2x))."""
    phase = np.random.uniform(0, 2*np.pi)
    amp = np.random.uniform(0.8, 1.2)
    y_offset = np.random.uniform(-0.5, 0.5)

    def fn(x):
        x = np.asarray(x)
        return amp * (0.5 * np.sin(FREQ * x + phase) + 0.3 * np.sin(2 * FREQ * x)) + y_offset
    return fn

def generate_sinusoid():
    """Generate sinusoid."""
    phase = np.random.uniform(0, 2*np.pi)
    amp = np.random.uniform(0.8, 1.2)
    y_offset = np.random.uniform(-0.5, 0.5)

    def fn(x):
        x = np.asarray(x)
        return amp * np.sin(FREQ * x + phase) + y_offset
    return fn

def generate_one_sinusoid():
    """Generate one sinusoid."""
    phase = 0.0
    amp = 0.5
    y_offset = 0.0
    
    def fn(x):
        x = np.asarray(x)
        return amp * np.sin(FREQ * x + phase) + y_offset
    return fn

def generate_sinusoid2():
    """Generate sinusoid."""
    phase = np.random.uniform(0, 2*np.pi)
    amp = np.random.uniform(1.2, 1.6)
    y_offset = np.random.uniform(-0.5, 0.5)

    def fn(x):
        x = np.asarray(x)
        return amp * np.sin(FREQ * x + phase) + y_offset
    return fn

def generate_one_sinusoid2():
    """Generate one sinusoid."""
    phase = 0.0
    amp = 0.5
    y_offset = 0.0
    
    def fn(x):
        x = np.asarray(x)
        return amp * np.sin(FREQ * x + phase) + y_offset
    return fn

def generate_sinusoid_with_texture():
    """Generate sinusoid with texture."""
    phase = np.random.uniform(0, 2*np.pi)
    amp = np.random.uniform(0.8, 1.2)
    y_offset = np.random.uniform(-0.5, 0.5)

    texture_phase = np.random.uniform(0, 2*np.pi)
    texture_amp = np.random.uniform(0.1, 0.3)
    texture_freq = np.random.uniform(2.0, 8.0)
    texture_y_offset = np.random.uniform(-0.1, 0.1)

    def fn(x):
        x = np.asarray(x)
        return amp * np.sin(FREQ * x + phase) + y_offset + texture_amp * np.sin(texture_freq * x + texture_phase) + texture_y_offset
    return fn

def generate_one_sinusoid_with_texture():
    """Generate one sinusoid with texture."""
    phase = 0.0
    amp = 0.5
    y_offset = 0.0  
    texture_phase = 0.0
    texture_amp = 0.2
    texture_freq = 5.0
    texture_y_offset = 0.0  

    def fn(x):
        x = np.asarray(x)
        return amp * np.sin(FREQ * x + phase) + y_offset + texture_amp * np.sin(texture_freq * x + texture_phase) + texture_y_offset
    return fn

def generate_squarewave():
    """Generate soft squarewave using tanh."""
    phase = np.random.uniform(0, 2*np.pi)
    amp = np.random.uniform(0.2, 0.4)
    y_offset = np.random.uniform(-0.5, 0.5)
    sharpness = np.random.uniform(4.0, 6.0)

    def fn(x):
        x = np.asarray(x)
        return amp * np.tanh(sharpness * np.sin(FREQ * x + phase)) + y_offset
    return fn

def generate_one_squarewave():
    """Generate one squarewave."""
    phase = 0.0
    amp = 0.3
    y_offset = 0.0
    sharpness = 5.0

    def fn(x):
        x = np.asarray(x)
        return amp * np.tanh(sharpness * np.sin(FREQ * x + phase)) + y_offset
    return fn

def generate_sawtooth():
    """Generate soft sawtooth (same as datasets_old.py)."""
    phase = np.random.uniform(0, 2*np.pi)
    amp = np.random.uniform(0.8, 1.2)
    y_offset = np.random.uniform(-0.5, 0.5)

    def fn(x):
        x = np.asarray(x)
        t = FREQ * x + phase
        saw = np.sin(t) - 0.5 * np.sin(2*t) + 0.33 * np.sin(3*t) - 0.25 * np.sin(4*t)
        return amp * saw * 0.5 + y_offset
    return fn


# Mixed datasets for pretraining
def generate_mixed():
    """Mixed: sinusoid + squarewave + sawtooth + harmonic + sigmoid."""
    generators = [generate_sinusoid, generate_squarewave, generate_sawtooth, 
                  generate_harmonic, generate_sigmoid, generate_line]
    return np.random.choice(generators)()

def generate_mixed_sinusoid_and_squarewave():
    """Mixed: sinusoid + squarewave."""
    generators = [generate_sinusoid, generate_squarewave]
    return np.random.choice(generators)()

def generate_mixed_no_sinusoid():
    """Mixed without sinusoid (held-out for transfer)."""
    generators = [generate_squarewave, generate_sawtooth, generate_harmonic, generate_sigmoid]
    return np.random.choice(generators)()

def generate_mixed_no_squarewave():
    """Mixed without squarewave (held-out for transfer)."""
    generators = [generate_sinusoid, generate_sawtooth, generate_harmonic, generate_sigmoid]
    return np.random.choice(generators)()

def generate_mixed_no_sawtooth():
    """Mixed without sawtooth (held-out for transfer)."""
    generators = [generate_sinusoid, generate_squarewave, generate_harmonic, generate_sigmoid]
    return np.random.choice(generators)()

def generate_mixed_no_harmonic():
    """Mixed without harmonic (held-out for transfer)."""
    generators = [generate_sigmoid, generate_sinusoid, generate_squarewave, generate_sawtooth]
    return np.random.choice(generators)()

def generate_mixed_no_sigmoid():
    """Mixed without sigmoid (held-out for transfer)."""
    generators = [generate_harmonic, generate_sinusoid, generate_squarewave, generate_sawtooth]
    return np.random.choice(generators)()


# Composite datasets
def generate_composite_sin_square():
    """Composite: sinusoid + soft squarewave.
    """
    phase1 = np.random.uniform(0, 2*np.pi)
    phase_diff = np.random.uniform(np.pi/6, np.pi/3)
    phase2 = phase1 + phase_diff
    
    alpha = np.random.uniform(0.2, 0.3)
    
    amp = np.random.uniform(0.7, 1.0)
    y_offset = np.random.uniform(-0.3, 0.3)
    sharpness = np.random.uniform(3.0, 4.5)
    
    def fn(x):
        x = np.asarray(x)
        sin_part = np.sin(FREQ * x + phase1)
        square_part = np.tanh(sharpness * np.sin(FREQ * x + phase2))
        return amp * (alpha * sin_part + (1 - alpha) * square_part) + y_offset
    
    return fn

def generate_composite_sin_sawtooth():
    """Composite: sinusoid + soft sawtooth."""
    phase1 = np.random.uniform(0, 2*np.pi)
    phase_diff = np.random.uniform(np.pi/6, np.pi/3)
    phase2 = phase1 + phase_diff
    
    alpha = np.random.uniform(0.5, 0.6)
    amp = np.random.uniform(0.7, 1.0)
    y_offset = np.random.uniform(-0.3, 0.3)
    
    def fn(x):
        x = np.asarray(x)
        sin_part = np.sin(FREQ * x + phase1)
        t = FREQ * x + phase2
        saw_part = np.sin(t) - 0.45 * np.sin(2*t) + 0.25 * np.sin(3*t)
        saw_part = saw_part * 0.55
        return amp * (alpha * sin_part + (1 - alpha) * saw_part) + y_offset
    
    return fn

def generate_composite_square_sawtooth():
    """Composite: soft squarewave + soft sawtooth."""
    phase1 = np.random.uniform(0, 2*np.pi)
    phase_diff = np.random.uniform(np.pi/6, np.pi/3)
    phase2 = phase1 + phase_diff
    
    alpha = np.random.uniform(0.45, 0.55)
    amp = np.random.uniform(0.7, 1.0)
    y_offset = np.random.uniform(-0.3, 0.3)
    sharpness = np.random.uniform(3.0, 4.5)
    
    def fn(x):
        x = np.asarray(x)
        square_part = np.tanh(sharpness * np.sin(FREQ * x + phase1))
        t = FREQ * x + phase2
        saw_part = (np.sin(t) - 0.45 * np.sin(2*t) + 0.25 * np.sin(3*t)) * 0.55
        return amp * (alpha * square_part + (1 - alpha) * saw_part) + y_offset
    
    return fn

def generate_composite_triple():
    """Composite: sinusoid + soft squarewave + soft sawtooth."""
    phase1 = np.random.uniform(0, 2*np.pi)
    phase2 = phase1 + np.random.uniform(np.pi/6, np.pi/4)
    phase3 = phase2 + np.random.uniform(np.pi/6, np.pi/4)
    
    w = np.array([0.45, 0.30, 0.25]) + np.random.uniform(-0.08, 0.08, 3)
    w = w / w.sum()
    
    amp = np.random.uniform(0.6, 0.9)
    y_offset = np.random.uniform(-0.2, 0.2)
    sharpness = np.random.uniform(3.0, 4.5)
    
    def fn(x):
        x = np.asarray(x)
        sin_part = np.sin(FREQ * x + phase1)
        square_part = np.tanh(sharpness * np.sin(FREQ * x + phase2))
        t = FREQ * x + phase3
        saw_part = (np.sin(t) - 0.45 * np.sin(2*t) + 0.25 * np.sin(3*t)) * 0.55
        return amp * (w[0] * sin_part + w[1] * square_part + w[2] * saw_part) + y_offset
    
    return fn

# Dataset registry
DATASET_GENERATORS = {
    "line": generate_line,
    "one_line": generate_one_line,
    "sigmoid": generate_sigmoid,
    "harmonic": generate_harmonic,
    "sinusoid": generate_sinusoid,
    "one_sinusoid": generate_one_sinusoid,
    "sinusoid2": generate_sinusoid2,
    "one_sinusoid2": generate_one_sinusoid2,
    "sinusoid_with_texture": generate_sinusoid_with_texture,
    "one_sinusoid_with_texture": generate_one_sinusoid_with_texture,
    "squarewave": generate_squarewave,
    "one_squarewave": generate_one_squarewave,
    "sawtooth": generate_sawtooth,
    "sinusoid_and_squarewave": generate_mixed_sinusoid_and_squarewave,
    "mixed": generate_mixed,
    "mixed_no_sinusoid": generate_mixed_no_sinusoid,
    "mixed_no_squarewave": generate_mixed_no_squarewave,
    "mixed_no_sawtooth": generate_mixed_no_sawtooth,
    "mixed_no_harmonic": generate_mixed_no_harmonic,
    "mixed_no_sigmoid": generate_mixed_no_sigmoid,
    "composite_sin_square": generate_composite_sin_square,
    "composite_sin_sawtooth": generate_composite_sin_sawtooth,
    "composite_square_sawtooth": generate_composite_square_sawtooth,
    "composite_triple": generate_composite_triple,
}


def load_data(bsz, dataset, args):
    """Load data from dataset."""
    if dataset not in DATASET_GENERATORS:
        raise ValueError(f"Dataset {dataset} not supported")
    generator = DATASET_GENERATORS[dataset]
    
    ctx_x_list, ctx_y_list, fut_x_list, fut_y_list = [], [], [], []
    
    for _ in range(bsz):
        gt_fn = generator()
        start_x = -2.5#np.random.uniform(-10, 10)
        x_vals = start_x + np.arange(args.ctx_sz + args.fut_sz) * args.res_x
        y_vals = [float(gt_fn(x)) for x in x_vals]
        ctx_x_list.append(x_vals[:args.ctx_sz])
        ctx_y_list.append(y_vals[:args.ctx_sz])
        fut_x_list.append(x_vals[args.ctx_sz:])
        fut_y_list.append(y_vals[args.ctx_sz:])
    
    ctx_x = torch.tensor(ctx_x_list, dtype=torch.float32, device=args.device)*SCALE
    ctx_y = torch.tensor(ctx_y_list, dtype=torch.float32, device=args.device)*SCALE
    fut_x = torch.tensor(fut_x_list, dtype=torch.float32, device=args.device)*SCALE
    fut_y = torch.tensor(fut_y_list, dtype=torch.float32, device=args.device)*SCALE

    #ctx_y += torch.randn(ctx_y.shape, device=args.device) * 0.05
    #fut_y += torch.randn(fut_y.shape, device=args.device) * 0.05
    
    return ctx_x, ctx_y, fut_x, fut_y