# RandOpt Speedrun Records

Each row is one run of the fixed standard. Reproduce with:
`python speedrun.py --config configs/standard_8xh100_qwen72b.yaml`

See [docs/SPEEDRUN.md](docs/SPEEDRUN.md) for the standard, the metric
(held-out FineWeb bits-per-byte), and the methodology. Rows are appended
automatically by `speedrun.py`; `seeds/s` is the optimization throughput.

| date | commit | config | model | hardware | pop | seeds/s | base acc | ens acc | base bpb | ens bpb | total |
|------|--------|--------|-------|----------|-----|---------|----------|---------|----------|---------|-------|
