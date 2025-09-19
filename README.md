# CLEMbench text-only evals (taboo + referencegame)

This branch contains:
- scripts to run & score two text games at T=0.0 and T=0.7 for both the base model
  (`Meta-Llama-3.1-8B-Instruct`) and our SFT model (`llama3-8b-sft-playpen`).
- a summarizer that prints %played, quality, and clemscore (text-only).
- `results_min/` with the summary text so we don't push large artifacts.

## Usage

```bash
export CLEM_MODEL_REGISTRY="$(pwd)/model_registry.json"
bash scripts/run_text_evals.sh

