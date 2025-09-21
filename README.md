 Playpen Experiments – Post-Training Architectures for Text Games  

This repository contains experiments on improving the performance of large language models (LLMs) in **Clembench** and **Playpen** interactive text-game environments.  
We evaluate both **post-training methods** (e.g., DPO/SFT with Tülu and custom datasets) and **architectural methods** (reasoning search, supervised playing) to strengthen protocol-following, rule compliance, and decision-making.  

---

## Repository Structure  

- **`curriculum_training/`, `curriculum_training_2/`**  
  Implementations of curriculum learning, where games are ordered from easier to harder to test difficulty-based training strategies.  

- **`failure_driven_learning/`**  
  Pipelines for collecting failed trajectories and replaying them with teacher models to build corrective datasets.  

- **`game-instructions/`**  
  Custom dataset of 14 categories of strict game-specific rules used for specialized fine-tuning.  

- **`playpen/`**  
 LM Playpen Environment for Learning in Interaction. Interacts with Clemcore and Clembench.  

- **`r1-distilled-llama-8b/`**  
  Experiments with an R1-distilled variant of LLaMA-8B.  

- **`reasoning_architecture/`**  
  Implementation of a reasoning pipeline, combining the base LLM, a preference/reward model (PRM), and search strategies such as Best-of-N, Beam Search, and Diverse Verifier Tree Search (DVTS).  

- **`supervised-playing/`**  
  Pipelines for supervised playing, where an Advisor LLM injects corrective feedback mid-episode.  

- **`tulu/`**  
  Experiments with Tülu datasets to improve general model abilities such as instruction-following, diversity, and preference alignment.  

---

## Getting Started  

1. Clone the repo:  
   ```bash
   git clone https://github.com/alextsiak/playpen.git
   cd playpen

2. Please follow the installation steps in the [Playpen README](https://github.com/phisad/playpen/tree/main)
