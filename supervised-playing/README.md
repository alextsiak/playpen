# Supervised Playing Architecture

This repository contains experiments on **protocol-aware supervision** in Clembench, where mid-episode feedback is injected into gameplay to improve instruction adherence.

## Idea

The player (LLaMA-3.1-8B-Instruct) interacts with the Game Master as usual. At selected turns, an **Advisor model** analyzes recent trajectories and provides targeted, protocol-specific feedback.  
The Advisor:
- Identifies mistakes and provides corrections,
- Suggests 2–3 actionable recommendations,
- Proposes one valid next action.

## Implementation

### Pipeline
- The Clemcore sequential runner was extended so that, at **turns 10 and 25**, the Advisor’s feedback is injected alongside the Game Master’s message.  
- Feedback prompt format and details are defined in `clemcore-clemcore-clemgame-runners-sequential.py`.  
- To reproduce the results make sure to use slighlty modified files: `clemcore-clemcore-clemgame-runners-dispatch.py` and  `clemcore-clemcore-cli.py`.  

### Models
- **Player (LLM):** `meta-llama/Llama-3.1-8B-Instruct`  
- **Advisor (LLM):** `Qwen/Qwen2.5-32B-Instruct`  

The Advisor runs with 4-bit quantization (NF4) and injects advice at configurable turns.

## Key Results

Injecting advice at **turns 10 and 25** improved Clembench performance:

| Setup                                        | Clemscore | Statscore |
|----------------------------------------------|-----------|-----------|
| Llama-8B-Instruct (baseline)                 | 29.05     | 55.45     |
| Supervised Playing (advice @ 10 & 25)        | 32.30 | 55.50 |

This corresponds to a **+11.19% improvement in Clemscore** with Statscore essentially unchanged (as expected, since core decoding was unaltered).

## Advising Frequency Experiments

We tested different advice schedules:

| Setup                                        | Clemscore | Statscore |
|----------------------------------------------|-----------|-----------|
| Llama-8B-Instruct (baseline)                 | 29.05     | 55.45     |
| Advice @ 10 & 25                             | 32.30     | 55.50     |
| Advice @ 15 & 30                             | 31.19     | –         |
| Advice every 10 turns                        | 31.09     | –         |
| Advice @ 3 and every 10 turns                | 30.21     | –         |

Two injections at turns **10 and 25** proved most effective.

## Conclusions

- **Supervised Playing** boosts Clembench scores by injecting **protocol-focused corrections** at key points.  
- The timing of advice is crucial: too frequent interventions reduce effectiveness, while two well-placed injections strike a good balance.  
- Because the advice targets **general instruction patterns** (formatting, valid actions, turn discipline) rather than specific game trajectories, this approach avoids overfitting and complements dataset-based instruction tuning.  
- These results highlight the importance of **targeted corrective feedback**. Richer or denser forms of feedback may further enhance performance in future setups.

