# HW Lockin Emulation Cost

This repo include scripts for measuring emulation cost in the HW lockin paper.

## Env Setup

- Python 3.11
- Install dependecies using `./requirements.txt`

## Scripts

The following two scripts measures the latency and throughput of quantization-aware/pruning-aware GEMMs and LLM inference.

1. Quantization emulation cost:

```bash
python profile-q-int.py
```

2. Pruning emulation cost

```bash
python profile-p.py
```
