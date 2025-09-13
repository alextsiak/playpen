import yaml
import argparse
from unsloth import FastLanguageModel
from transformers import BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
import torch
from datasets import load_dataset
import json
from utils import formatting_func, run_wandb
import wandb
import os

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/home/users/rzorin/.cache/torchinductor"

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    type=str,
    required=True,
    help="Please set your training parameters in config-sft.yaml")
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

#type tuning
tuning = config['tuning']['type']
#model and dataset
base_model_name = config['base_model']['name']
sft_dataset_name = config['sft_dataset']['name']
sft_model_name = config['sft_model']['name']
preference_dataset_name = config['preference_dataset']['name']
#context length
max_seq_length = int(config['max_seq_length']['length'])
#lora specific parameters
matrice_size = int(config['low_rank_matrice']['size'])
influence = int(config['influence']['rate'])
dropout = float(config['dropout']['rate'])
#hyperparameters
lr = float(config['training']['learning_rate'])
batch_size = int(config['training']['batch_size'])
grad_acc_steps = int(config['training']['grad_acc_steps'])
max_training_steps = int(config['training']['max_training_steps'])
warmup_steps = int(config["training"]["warmup_steps"])
beta = float(config['training']['beta'])

print("*****Config*****")
if tuning == 'sft':
    print(f"tuning type: {tuning}\n",
          f"base_model: {base_model_name}\n",
          f"sft_dataset: {sft_dataset_name}\n"
          )
else: #if tuning == 'dpo'
    print(f"tuning type: {tuning}\n",
          f"reference model: {base_model_name}\n",
          f"sft_model: {sft_model_name}\n",
          f"preference dataset: {preference_dataset_name}\n",
          )
print(f"max_seq_lenght: {max_seq_length}\n",
      f"matrice_size: {matrice_size}\n",
      f"influence: {influence}\n",
      f"dropout: {dropout}\n",
      f"learning_rate: {lr}\n",
      f"batch size: {batch_size}\n",
      f"gradient accumulation steps: {grad_acc_steps}\n",
      f"max_training_steps: {max_training_steps}\n",
      f"warmup_steps: {warmup_steps}\n")
if tuning == 'dpo':
    print(f"beta: {beta}")
print("*****End of config*****")

if tuning == 'sft':

    save_dir = f"{base_model_name.split('/')[-1]}_finetuned_{sft_dataset_name.split('/')[-1]}"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name, max_seq_length=max_seq_length, dtype=None, load_in_4bit=True
        )

    model = FastLanguageModel.get_peft_model(
        model,
        r=matrice_size, #the size of the low-rank matrices
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=influence, # adjusts how strongly the LoRA modifications affect the model
        lora_dropout=dropout,  # Supports any, but = 0 is optimized. 
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        # max_seq_length=max_seq_length,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
        )

    # model.config.quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=False,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float16,
    #     )

    #train_dataset = load_dataset(sft_dataset_name, split="train")
    train_dataset = load_dataset("json", data_files=sft_dataset_name)["train"]

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    def formatting_func(examples):
        messages = examples["messages"]
        texts = [
            "".join([m["content"].strip() + "\n" for m in convo]).strip() + EOS_TOKEN for convo in messages
        ]
        return {"text": texts}

    train_dataset = train_dataset.map(formatting_func, batched=True)

    # Shuffle the dataset before training
    train_dataset = train_dataset.shuffle(seed=42)

    run_wandb(base_model_name, sft_dataset_name, max_training_steps, lr, batch_size, grad_acc_steps, save_dir)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_acc_steps,
            warmup_steps=warmup_steps,
            max_steps=max_training_steps,
            report_to="wandb",
            run_name=f"{base_model_name}_{sft_dataset_name}",
            output_dir=os.path.join(save_dir, "outputs"),
            optim="adamw_8bit",
        ),
        )

else: #if tuning == dpo

    save_dir = f"{sft_model_name.split('/')[-1]}_finetuned_{preference_dataset_name.split('/')[-1]}"

    #this is the original ref model
    ref_model, _ = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        #max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    
    # this is our LoRA-adapted SFT model
    model, tokenizer = FastLanguageModel.from_pretrained(
	model_name=sft_model_name,
	#max_seq_length=2048,
	dtype=None,
	load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
	model,
        #r=matrice_size, #the size of the low-rank matrices
        #target_modules=[
        #    "q_proj",
        #    "k_proj",
        #    "v_proj",
        #    "o_proj",
        #    "gate_proj",
        #    "up_proj",
        #    "down_proj",
        #],
        #lora_alpha=influence, # adjusts how strongly the LoRA modifications affect the model
        #lora_dropout=dropout,  # Supports any, but = 0 is optimized. 
        #bias="none",  # Supports any, but = "none" is optimized
        #use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        #max_seq_length=max_seq_length,
        #use_rslora=False,  # We support rank stabilized LoRA
        #loftq_config=None,  # And LoftQ
        )

#    if tokenizer.chat_template is None:
    	#tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    #don't want to train the reference model
    for param in ref_model.parameters():
        param.requires_grad = False
    
    #model.config.quantization_config = BitsAndBytesConfig(
    #load_in_4bit=True,
    #bnb_4bit_use_double_quant=False,
    #bnb_4bit_quant_type="nf4",
    #bnb_4bit_compute_dtype=torch.float16,
    #)   
    
    #train_dataset = load_dataset(preference_dataset_name, split="train")
    train_dataset = load_dataset("json", data_files=preference_dataset_name)["train"]

    # Shuffle the dataset before training
    train_dataset = train_dataset.shuffle(seed=42)

    run_wandb(sft_model_name, preference_dataset_name, max_training_steps, lr, batch_size, grad_acc_steps, save_dir)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=DPOConfig(
            beta=beta,
            max_length=max_seq_length,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_acc_steps,
            warmup_steps=warmup_steps,
            max_steps=max_training_steps,
            report_to="wandb",
            run_name=f"{sft_model_name}_{preference_dataset_name}",
            output_dir=os.path.join(save_dir, "outputs"),
            optim="adamw_8bit",
        ),
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

trainer.train()

wandb.finish()


model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

if tuning == 'sft':
    filtered_config = {
        'tuning': config['tuning']['type'],
        'base_model': config['base_model']['name'],
        'sft_dataset': config['sft_dataset']['name'],
        'max_seq_length': config['max_seq_length']['length'],
        'matrice_size': config['low_rank_matrice']['size'],
        'influence': config['influence']['rate'],
        'dropout': config['dropout']['rate'],
        'lr': config['training']['learning_rate'],
        'batch_size': config['training']['batch_size'],
        'grad_acc_steps': config['training']['grad_acc_steps'],
        'max_training_steps': config['training']['max_training_steps'],
        'warmup_steps': config["training"]["warmup_steps"]
        }
    
else: #if tuning == 'dpo'
    filtered_config = {
        'tuning': config['tuning']['type'],
        'reference_model': config['base_model']['name'],
        'sft_model': config['sft_model']['name'],
        'preference_dataset': config['preference_dataset']['name'],
        'max_seq_length': config['max_seq_length']['length'],
        'matrice_size': config['low_rank_matrice']['size'],
        'influence': config['influence']['rate'],
        'dropout': config['dropout']['rate'],
        'lr': config['training']['learning_rate'],
        'batch_size': config['training']['batch_size'],
        'grad_acc_steps': config['training']['grad_acc_steps'],
        'max_training_steps': config['training']['max_training_steps'],
        'warmup_steps': config["training"]["warmup_steps"],
        'beta': config['training']['beta']
        }
save_path = os.path.join(save_dir, "used_config.yaml")
with open(save_path, 'w') as f:
    yaml.dump(filtered_config, f)
