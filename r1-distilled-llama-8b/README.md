# R1-Distilled LLaMA Experiments

This directory contains experiments with an **R1-distilled LLaMA-3.1-8B** model adapted for Clembench.

## Setup

- **Model:** LLaMA-3.1-8B distilled from R1.  
- **Backend:** Implemented in [`vllm-r1-d-l-8b_api.py`](vllm-r1-d-l-8b_api.py) using vLLM.  
- **Modification:** The API was slightly adjusted to **remove `<think>` tokens** from outputs to ensure compatibility with Clembench parsing.

## Results

Evaluation shows **low performance**, with the model being overly verbose despite filtering:

| Model                  | Clemscore | Statscore |
|-------------------------|-----------|-----------|
| r1-d-l-8b               | 6.77 | 46.37 |

The gap between Clemscore and Statscore suggests that verbosity and excessive commentary often break the strict parsing protocol required in games.
