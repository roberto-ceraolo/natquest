import json
import random
from datasets import load_dataset
from tqdm import tqdm
import argparse

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def sample_huggingface_dataset(dataset_name, subset, feature_names, split, num_samples):
    print(f"\nLoading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, subset, split=split, streaming=True, trust_remote_code=True)
    sampled_data = []
    
    print(f"Sampling {num_samples} datapoints from {dataset_name}")

    for item in tqdm(dataset.shuffle(seed=RANDOM_SEED).take(num_samples), total=num_samples, desc=f"Sampling {dataset_name}"):
        sample = {}
        for feature_name in feature_names:
            if isinstance(feature_name, list):  # For nested features like choices[text]
                value = item
                for key in feature_name:
                    value = value[key]
                sample['.'.join(feature_name)] = value
            else:
                sample[feature_name] = item[feature_name]
        # if sample has more than one key, unify them into a single key, question
        if len(sample.keys()) > 1:
            # if there are "Correct Answer" and "Incorrect Answer" keys, unify them into the question, after "Choices" key
            if "Correct Answer" in sample.keys() and "Incorrect Answer 1" in sample.keys():
                sample["question"] = sample["Question"] + "\n Choices: " + sample["Correct Answer"] + ", " + sample["Incorrect Answer 1"] + ", " + sample["Incorrect Answer 2"] + ", " + sample["Incorrect Answer 3"]
                # drop the original keys
                sample.pop("Question")
                sample.pop("Correct Answer")
                sample.pop("Incorrect Answer 1")
                sample.pop("Incorrect Answer 2")
                sample.pop("Incorrect Answer 3")
            if "options" in sample.keys():
                sample["question"] = sample["question"] + "\n Choices: " + ", ".join(sample["options"])
                sample.pop("options")
            if "choices.text" in sample.keys():
                sample["question"] = sample["question"] + "\n Choices: " + ", ".join(sample["choices.text"])
                sample.pop("choices.text")
            if "context" in sample.keys():
                sample["question"] = sample["context"] + "\n" + sample["question"] 
                sample.pop("context")

        sampled_data.append(sample)

    print(f"Completed sampling from {dataset_name}. Collected {len(sampled_data)} samples.")
    return sampled_data

def main(datasets_to_process):
    # Total number of samples and number of samples per dataset
    SAMPLES_PER_DATASET = 500

    datasets = [
        ("truthfulqa/truthful_qa", "generation", ["question"], "validation"),
        ("Idavidrein/gpqa", "gpqa_extended", ["Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"], "train"),
        ("TIGER-Lab/MMLU-Pro", None, ["question", "options"], "test"),
        ("allenai/ai2_arc", "ARC-Challenge", ["question", ["choices", "text"]], "test"),
        ("openai/gsm8k", "main", ["question"], "test"),
        ("rajpurkar/squad_v2", None, ["context", "question"], "validation")
    ]

    print(f"Starting the dataset sampling process.")
    TOTAL_SAMPLES = SAMPLES_PER_DATASET * len(datasets)
    print(f"Total samples to be collected: {TOTAL_SAMPLES}")
    print(f"Samples per dataset: {SAMPLES_PER_DATASET}")
    print(f"Random seed: {RANDOM_SEED}")
    
    all_sampled_data = {}

    # Sample from Hugging Face datasets
    for dataset_name, subset, feature_names, split in datasets:
        if dataset_name.split('/')[-1] in datasets_to_process:
            all_sampled_data[dataset_name] = sample_huggingface_dataset(dataset_name, subset, feature_names, split, SAMPLES_PER_DATASET)
    
    save = True
    if save: 
        # Save sampled data to JSON files
        print("\nSaving sampled data to JSON files:")
        for dataset_name, sampled_data in all_sampled_data.items():
            filename = f"{dataset_name.split('/')[-1]}_sampled.json"
            with open(filename, 'w') as f:
                json.dump(sampled_data, f, indent=2)
            print(f"  - Saved {len(sampled_data)} samples to {filename}")
        
    total_samples = sum(len(data) for data in all_sampled_data.values())
    print(f"\nSampling complete. Total samples collected: {total_samples}")
    if total_samples < TOTAL_SAMPLES:
        print(f"Note: Collected {TOTAL_SAMPLES - total_samples} fewer samples than intended.")
        print("This may be due to some datasets having fewer examples than requested.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample datasets")
    parser.add_argument('--datasets', nargs='+', default=["truthful_qa", "gpqa", "MMLU-Pro", "ai2_arc", "gsm8k", "squad_v2"],
                        help='List of datasets to process')
    args = parser.parse_args()

    # usage
    # python sampling_rq2_noncausal.py 
    
    main(args.datasets)