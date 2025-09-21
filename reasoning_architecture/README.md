# Reasoning Architecture Experiments

This repository contains experiments with reasoning-based search strategies for LLMs in Clembench, implemented via the [search-and-learn](https://github.com/huggingface/search-and-learn) framework.

## Implementation

The reasoning backend is defined in [`vllm-reasoning-sal_api.py`](vllm-reasoning-sal_api.py):contentReference[oaicite:0]{index=0}.  
It integrates:
- **LLM** (`meta-llama/Llama-3.1-8B-Instruct`)
- **PRM (Reward Model):** `RLHFlow/Llama3.1-8B-PRM-Deepseek-Data`
- **Search strategies:** Best-of-N, Beam Search, DVTS

To use this backend, clone and install the dependencies of [search-and-learn](https://github.com/huggingface/search-and-learn).

## Key Results

Initial results showed **degraded performance relative to the baseline** when applying search-based reasoning:

| Search Method | Clemscore | Statscore |
|---------------|-----------|-----------|
| Llama-8B-Instruct (baseline) | 29.05 | 55.45 |
| Best-of-N | 13.27:contentReference[oaicite:1]{index=1} | 27.48:contentReference[oaicite:2]{index=2} |
| Beam Search | 16.50:contentReference[oaicite:3]{index=3} | 46.87:contentReference[oaicite:4]{index=4} |
| DVTS | 13.74:contentReference[oaicite:5]{index=5} | 43.92:contentReference[oaicite:6]{index=6} |

## Error Analysis and Adjustments

Trajectory inspection revealed two dominant error types:
1. **Parser mismatch**: unsupported or malformed verbs (e.g., *look, notice, explore, move, go*).
2. **Overly verbose outputs**: multi-line answers wrapping the core action in commentary, causing parsing errors.

To mitigate this, we tested:
- Lower temperature (**0.3**) to reduce drift into invalid actions.
- Larger `n` (**12**) to preserve candidate diversity.
- Strict **12-token cap** to suppress commentary.

However, these adjustments (beam-style search) did **not improve results**:

| Search Method | Clemscore | Statscore |
|---------------|-----------|-----------|
| Llama-8B-Instruct (baseline) | 29.05 | 55.45 |
| Best-of-N (adjusted) | 14.68 | 18.84 |

## Conclusions

A generic test-time reasoning stack did not yield improvements under Clembench’s strict parsing rules.  
Instead, multi-step search and reranking introduced more points of failure than benefit in this short, turn-based dialogue setting. For such environments, reasoning architectures may add unnecessary complexity without improving outcomes.

