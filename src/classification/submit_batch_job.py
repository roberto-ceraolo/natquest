# submit_batch_job.py

import json
from typing import List, Dict
from openai import OpenAI
import pandas as pd
import argparse

from classification_utils import (
    define_bloom_taxonomy_function,
    define_needs_function,
    define_uniqueness_function,
    define_domain_function,
    get_prompt_needs,
    get_prompt_uniqueness,
    get_prompt_bloom_taxonomy,
    get_prompt_domain,
    define_causality_function,
    get_prompt_causality
)

JOB_CONFIGS = {
    "user_needs": {
        "function": define_needs_function(),
        "prompt_function": get_prompt_needs
    },
    "bloom_taxonomy": {
        "function": define_bloom_taxonomy_function(),
        "prompt_function": get_prompt_bloom_taxonomy
    },
    "uniqueness": {
        "function": define_uniqueness_function(),
        "prompt_function": get_prompt_uniqueness
    },
    "domain": {
        "function": define_domain_function(),
        "prompt_function": get_prompt_domain
    },
    "causality": {
        "function": define_causality_function(),
        "prompt_function": get_prompt_causality
    },
}

def prepare_batch_input(questions: List[Dict[str, str]], job_type: str, model: str = "gpt-4o-mini-2024-07-18") -> List[Dict]:
    job_config = JOB_CONFIGS[job_type]
    function = job_config["function"]
    prompt_function = job_config["prompt_function"]

    batch_input = []
    for question_data in questions:
        prompt = prompt_function(question_data["question"])

        batch_request = {
            "custom_id": str(question_data['id']),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [{"role": "system","content": (
                        "You are an expert in psychology, cognitive science, and linguistics."
                    )
                    },
                      {"role": "user", "content": prompt}],
                "functions": [function],
                "function_call": {"name": function["name"]},
                "max_tokens": 1000
            }
        }
        batch_input.append(json.dumps(batch_request))

    return batch_input

def submit_batch_job(batch_input: List[Dict], client: OpenAI, job_type_name: str) -> str:
    # Write batch input to a file
    print("Writing batch input to file...")
    with open(f"batch_input_{job_type_name}.jsonl", "w") as f:
        for item in batch_input:
            f.write(f"{item}\n")

    # Upload the batch input file
    print("Uploading batch input file...")
    with open(f"batch_input_{job_type_name}.jsonl", "rb") as f:
        batch_input_file = client.files.create(file=f, purpose="batch")

    # Create the batch
    print("Creating batch...")
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"Question analysis batch - {job_type_name}"}
    )

    print(f"Batch created with ID: {batch.id}")
    return batch.id

def main():
    parser = argparse.ArgumentParser(description="Submit a batch job for question analysis.")
    parser.add_argument("--job_type", choices=JOB_CONFIGS.keys(), default="causality", help="Type of job to submit")
    args = parser.parse_args()

    job_type = args.job_type  
    job_type_name = job_type + "_nonnatural"
    

    client = OpenAI()

    # Load the dataframe from data/causalquest.csv
    df = pd.read_csv('data/causalquest.csv')

    # Prepare batch input
    df_list = df.to_dict('records')
    batch_input = prepare_batch_input(df_list, job_type)

    # Submit batch job
    batch_id = submit_batch_job(batch_input, client, job_type_name)

    # Write batch ID to log file
    with open(f'batch_job_log_{job_type_name}.txt', 'w') as f:
        f.write(f"Batch ID: {batch_id}")

    print(f"Batch job submitted for {job_type_name}. Batch ID: {batch_id}")
    print(f"Batch ID has been saved to 'batch_job_log_{job_type_name}.txt'")

if __name__ == "__main__":
    main()