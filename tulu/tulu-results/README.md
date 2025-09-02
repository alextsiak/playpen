## Experiment description 

The goal of the experiments below is to explore how different types of tuning affect the ability of large language models to perform interactive tasks within the Clembench benchmark. We focus on whether improving a model's general abilities — including instruction-following, conversational capabilities, and broader preference tuning — can lead to better performance in interactive, game-like environments, even without exposing the model to game-specific data.

To test this, we use the LLaMA 3.1 8B Instruct model as the base and apply several tuning strategies, including instruction-following datasets, real-world conversational data (WildChat), and mixture preference datasets. Across most experiments, we rely on DPO (Direct Preference Optimization) as the main tuning approach, with some additional tests using supervised fine-tuning (SFT).

We use Tulu 3 datasets to run our experiments: https://huggingface.co/collections/allenai/tulu-3-datasets-673b8df14442393f7213f372

### Potsdam Blitz Team

Arefeva, Tsiakalou, Zarev, Zorin

### Our Best Model:

Model: LLaMA 3.1 8B Instruct 4-bit tuned with unsloth and DPO
Dataset: allenai/tulu-3-pref-personas-instruction-following
Batch size: 4×4, steps: 700
Results: clemscore 14.39, Avg % Played 34.79, Avg Quality Score 41.35

### The Baseline to compare

The Baseline (Llama3-8b-it-4bit): clemscore 19.58, Avg % Played 49.43, Avg Quality Score 39.62
