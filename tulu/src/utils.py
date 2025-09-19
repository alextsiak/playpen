import os

def formatting_func(examples):
    messages = examples["messages"]
    texts = [
        "".join([m["content"].strip() + "\n" for m in convo]).strip()
        for convo in messages
    ]
    return {"text": texts}

def run_wandb(model_name, dataset_name, max_steps, lr, batch_size, grad_acc_steps, save_dir):
    os.environ["WANDB_DIR"] = save_dir
    import wandb
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
    project="pm-pt",
    name=f"{model_name}_{dataset_name}",
    config={
        "model": model_name,
        "dataset": dataset_name,
        "max_steps": max_steps,
        "learning_rate": lr,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_acc_steps,
    },
    )