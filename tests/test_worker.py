"""
CPU test of the rewired vLLM WorkerExtension (utils/worker_extn.py) using a fake
model. vLLM imports in that module are guarded, so it imports on a CPU box; here
we drive the perturbation methods directly (no vLLM generate) against a small
nn.Module and check the weight math end-to-end.

Run:  .venv/bin/python tests/test_worker.py
"""
import os
import sys
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import perturb as P
import utils.worker_extn as WE


def _make_worker():
    torch.manual_seed(0)

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer0 = nn.Linear(16, 8, bias=False)
            self.layer1 = nn.Linear(8, 8, bias=False)
            self.visual = nn.Linear(4, 4, bias=False)  # 'visual.weight' -> skipped

    model = Fake().to(torch.float32)
    w = WE.WorkerExtension()
    w.model_runner = types.SimpleNamespace(model=model)
    w.configure_perturbation(noise="rademacher", kernel="torch")
    w.store_base_weights()
    base = {n: p.data.clone() for n, p in model.named_parameters()}
    return w, model, base


def _expected(model, base, seed, sigma, should_perturb):
    plan = P.iter_perturb_params(model.named_parameters(), should_perturb)
    exp = {n: base[n].clone() for n in base}
    for name, p, off in plan:
        signs = P.rademacher_signs(seed, off, p.numel(), p.device).reshape(p.shape)
        exp[name] = base[name] + sigma * signs
    return exp


def test_perturb_reconstructs_from_base():
    w, model, base = _make_worker()
    sigma = 1e-2
    w.perturb_self_weights(123, sigma)
    exp = _expected(model, base, 123, sigma, w._should_perturb)
    for name, p in model.named_parameters():
        assert torch.allclose(p.data, exp[name], atol=0), name
    # visual.weight must be untouched
    assert torch.equal(model.visual.weight.data, base["visual.weight"])


def test_restore_is_noop_and_next_perturb_overwrites():
    w, model, base = _make_worker()
    sigma = 1e-2
    w.perturb_self_weights(123, sigma)
    snap = {n: p.data.clone() for n, p in model.named_parameters()}
    w.restore_self_weights(123, sigma)  # no-op
    for name, p in model.named_parameters():
        assert torch.equal(p.data, snap[name]), f"restore must be a no-op, {name} changed"
    # A different seed reconstructs absolutely from base (not from current live weights)
    w.perturb_self_weights(999, sigma)
    exp = _expected(model, base, 999, sigma, w._should_perturb)
    for name, p in model.named_parameters():
        assert torch.allclose(p.data, exp[name], atol=0), name
    # Returning to seed 123 is bit-identical to the first time (drift-free)
    w.perturb_self_weights(123, sigma)
    for name, p in model.named_parameters():
        assert torch.equal(p.data, snap[name]), f"{name} not drift-free across seed switches"


def test_negate_flips_perturbation():
    w, model, base = _make_worker()
    sigma = 5e-3
    w.perturb_self_weights(7, sigma, negate=True)
    exp = _expected(model, base, 7, -sigma, w._should_perturb)
    for name, p in model.named_parameters():
        assert torch.allclose(p.data, exp[name], atol=0), name


def test_reset_to_base():
    w, model, base = _make_worker()
    w.perturb_self_weights(5, 1e-2)
    w.reset_to_base_weights()
    for name, p in model.named_parameters():
        assert torch.equal(p.data, base[name]), name


def test_apply_perturbation_matches_perturb():
    w, model, base = _make_worker()
    sigma = 2e-3
    w.apply_perturbation(321, sigma)
    exp = _expected(model, base, 321, sigma, w._should_perturb)
    for name, p in model.named_parameters():
        assert torch.allclose(p.data, exp[name], atol=0), name


def test_apply_averaged_perturbations():
    w, model, base = _make_worker()
    seeds_sigmas = [(11, 1e-2), (22, 2e-2), (33, 1e-2)]
    w.apply_averaged_perturbations(seeds_sigmas)            # equal weights 1/3
    # Reference: base + sum_i (1/K * sigma_i) * R(seed_i), fp32 accumulate
    K = len(seeds_sigmas)
    plan = P.iter_perturb_params(model.named_parameters(), w._should_perturb)
    for name, p, off in plan:
        acc = torch.zeros(p.numel(), dtype=torch.float32)
        for seed, sigma in seeds_sigmas:
            acc += (1.0 / K) * sigma * P.rademacher_signs(seed, off, p.numel(), p.device)
        exp = (base[name].to(torch.float32) + acc.reshape(p.shape)).to(p.dtype)
        assert torch.allclose(p.data, exp, atol=1e-6), name
    assert torch.equal(model.visual.weight.data, base["visual.weight"])  # skipped


def test_apply_linear_combined_perturbations_raw_coefficients():
    w, model, base = _make_worker()
    seeds_sigmas = [(11, 1e-2), (22, 2e-2)]
    coeffs = [1.0, -0.25]
    w.apply_linear_combined_perturbations(seeds_sigmas, coeffs)
    plan = P.iter_perturb_params(model.named_parameters(), w._should_perturb)
    for name, p, off in plan:
        acc = torch.zeros(p.numel(), dtype=torch.float32)
        for (seed, sigma), coeff in zip(seeds_sigmas, coeffs):
            acc += coeff * sigma * P.rademacher_signs(seed, off, p.numel(), p.device)
        exp = (base[name].to(torch.float32) + acc.reshape(p.shape)).to(p.dtype)
        assert torch.allclose(p.data, exp, atol=1e-6), name
    assert torch.equal(model.visual.weight.data, base["visual.weight"])


def test_switch_to_seed_matches_reconstruct_fp32():
    w, model, base = _make_worker()
    sigma = 1e-3
    w.perturb_self_weights(11, sigma)          # at A
    w.switch_to_seed(11, 22, sigma)            # A -> B in place
    exp = _expected(model, base, 22, sigma, w._should_perturb)
    for name, p in model.named_parameters():
        assert torch.allclose(p.data, exp[name], atol=1e-5), name


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa
            failed += 1
            import traceback
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
