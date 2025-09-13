import json

# This script unites several files of sft-data (batches) in a single sft-dataset
all_data = []
batch_number = 7

for i in range(1, batch_number):
    with open(f"../data/sft_batch_{i}.json", "r") as f:
        batch = json.load(f)
        all_data.extend(batch)

# Save full dataset
with open("../data/sft_dataset.json", "w") as f:
    json.dump(all_data, f, indent=2)