# Game-Specific Instructions Tuning for LLMs

This repository directory contains experiments on improving instruction-following abilities of LLMs in interactive environments using Direct Preference Optimization (DPO) with a custom dataset of game-specific instructions.

## Motivation

Error analysis of Clembench runs showed that many failed games resulted from models not adhering to the Game Master’s rules. To address this, we designed a dataset targeting 14 instruction categories (e.g., single-word responses, JSON formatting, forbidden words, dialogue state maintenance). Each category was represented by two patterns, and GPT-4o generated 100 examples per pattern, resulting in 2,800 training examples.

## Dataset

- **Path:** `data/dpo_dataset_2800/`  
- **Content:** 2,800 preference-formatted examples for instruction adherence.  

## Training Setup

- **Tuning method:** DPO with PEFT (LoRA)  
- **Base model:** `unsloth/Meta-Llama-3.1-8B-Instruct` (4bit)  
- **Reference model:** same as above  
- **Batch size:** 4 (with gradient accumulation = 4)  
- **Learning rate:** 5e-7  
- **Beta:** 0.2  
- **Dropout:** 0.1  
- **Max seq length:** 256  
- **Influence / matrix size:** 32  
- **Training steps:** 525  

Hyperparameters were selected based on compact experiments with a 1B model before scaling to 8B.

## Results

### Clemcore 3.2.1 (T=0.0, greedy decoding)

| Model                       | Clemscore | Statscore |
|-----------------------------|-----------|-----------|
| Llama-8B-it-4bit (baseline) | 27.35     | 49.16     |
| Llama-8B-it-4bit-dpo-2800   | 27.66 (+1.13%) | 50.47 (+2.66%) |

**Scores are stored in:**  
- `llama3-8b-it-4bit.val.json`  
- `llama3-8b-it-4bit-pref-2800.val.json`  

### Clemcore 3.3.2 (T=0.0, greedy decoding)

| Model                       | Clemscore | Statscore |
|-----------------------------|-----------|-----------|
| Llama-8B-it-4bit (baseline) | 27.67     | 50.65     |
| Llama-8B-it-4bit-dpo-2800   | 28.09 (+1.52%) | 50.36 (-0.57%) |

**Scores are stored in:**  
- `llama3-8b-it-4bit-check_results.val.json`  
- `llama3-8b-it-4bit-pref-2800_check_results.val.json`  

## Conclusion

Small-scale tuning on structured, game-specific instructions leads to consistent improvements in Clembench performance and better instruction adherence. Unlike fine-tuning on full game trajectories, this approach generalizes across games, reducing the risk of overfitting while enhancing rule-following abilities.

