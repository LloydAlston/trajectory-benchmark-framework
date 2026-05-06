# Trajectory Benchmark Framework

Unified benchmarking pipeline for trajectory prediction.

## Overview
This project compares linear and LSTM models under identical conditions to understand trade-offs in trajectory prediction. It identifies failure modes such as linear model breakdown under sharp turns and error accumulation in autoregressive sequence models.

## Benchmark Outcomes
Model performance is dependent on the chosen metric (ADE vs FDE):

- **Linear Baseline**: Lower ADE (good short-term fit) but higher FDE (fails on long-horizon turns).
- **LSTM Model**: Higher ADE (less precise locally) but lower FDE (better long-term adaptation), though it accumulates autoregressive error.

### Key Observations
- Model ranking depends on the metric (ADE vs FDE).
- Simple models can outperform sequence models locally but fail globally.
- Sequence models capture complex dynamics but introduce instability.

## Features
- Multiple models (LSTM, Linear baseline)
- Reproducible experiments (seed control)
- ADE / FDE evaluation
- Failure case analysis

## Run
```bash
pip install -r requirements.txt
python run.py
```