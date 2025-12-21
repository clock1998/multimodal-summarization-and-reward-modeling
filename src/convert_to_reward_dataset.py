from datasets import load_dataset
import json

def convert_to_reward_dataset(dataset):
    # Create 'chosen' and 'rejected' fields based on the winner column
    def response_1_2_to_chosen_rejected(example):
        if example["winner"] == 1:
            example["chosen"] = example["response1"]
            example["rejected"] = example["response2"]
        else:  # winner == 2
            example["chosen"] = example["response2"]
            example["rejected"] = example["response1"]
        return example

    dataset = dataset.map(response_1_2_to_chosen_rejected)

    # Keep only necessary columns
    dataset = dataset.select_columns(["chosen", "rejected"])

    return dataset
