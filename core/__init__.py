"""RandOpt core package.

``engine`` pulls in heavy GPU-only deps (ray, vllm); its public names are
exposed lazily so that the pure-torch ``core.perturb`` submodule can be imported
on a CPU box (for tests) without ray/vllm installed.

``from core import launch_engines``  -> lazily imports core.engine
``from core import perturb``          -> normal submodule import (no engine deps)
"""
import importlib

__all__ = ["RandOptNcclLLM", "launch_engines", "cleanup_engines", "perturb"]

_ENGINE_EXPORTS = {"RandOptNcclLLM", "launch_engines", "cleanup_engines"}


def __getattr__(name):
    if name in _ENGINE_EXPORTS:
        engine = importlib.import_module(".engine", __name__)
        return getattr(engine, name)
    # Anything else (e.g. the 'perturb' submodule) falls through to the normal
    # import machinery, which imports it as a submodule.
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
