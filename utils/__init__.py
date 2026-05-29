"""RandOpt utils package.

Made an explicit package (rather than a PEP-420 namespace package) so that
vLLM's Ray worker processes can reliably import ``utils.worker_extn`` as the
``worker_extension_cls`` — namespace resolution was failing in spawned workers
("'utils' is not a package"). The repo root is additionally propagated to Ray
workers via ``runtime_env`` PYTHONPATH at the ray.init sites.
"""
