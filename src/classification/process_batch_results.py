# process_batch_results.py

import json
from openai import OpenAI
import time
import argparse


from classification_utils import (
    define_bloom_taxonomy_function,
    define_needs_function,
    define_uniqueness_function,
    get_prompt_needs,
    get_prompt_uniqueness,
    get_prompt_bloom_taxonomy,
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
    "causality": {
        "function": define_causality_function(),
        "prompt_function": get_prompt_causality
    },
}

def parse_llm_output(output: str) -> dict:
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {
            "error": "Failed to parse LLM output",
            "raw_output": output
        }

def process_batch_results(batch_id: str, client: OpenAI) -> list:
    # Check batch status
    while True:
        batch_status = client.batches.retrieve(batch_id)
        print(f"Batch status: {batch_status.status}")
        if batch_status.status == "completed":
            break
        elif batch_status.status == "failed":
            raise Exception("Batch processing failed")
        time.sleep(60)  # Wait for 60 seconds before checking again

    # Retrieve and process the results
    print("Batch processing complete. Retrieving results...")
    output_file_id = batch_status.output_file_id
    file_response = client.files.content(output_file_id)
    
    results = []
    
    for line in file_response.text.split('\n'):
        if line:
            batch_result = json.loads(line)
            function_call = batch_result['response']['body']['choices'][0]['message']['function_call']
            analysis = json.loads(function_call['arguments'])
            
            result = {
                "query_id": batch_result['custom_id'],
                **analysis
            }
            results.append(result)

    return results

def main():
    #parser = argparse.ArgumentParser(description="Process batch results for question analysis.")
    #parser.add_argument("--job_type", choices=JOB_CONFIGS.keys(), default="uniqueness", help="Type of job to process")
    #args = parser.parse_args()

    client = OpenAI()

    
    job_type = "causality_nonnatural"

    # Read batch ID from log file
    with open(f'batch_job_log_{job_type}.txt', 'r') as f:
        batch_id = f.read().strip().split(': ')[1]

    # Process batch results
    results = process_batch_results(batch_id, client)

    # Save results to JSONL file
    with open(f'question_classifications_{job_type}_batch_result.jsonl', 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"Classification complete for {job_type}. Results saved to 'question_classifications_{job_type}.jsonl'")

if __name__ == "__main__":
    main()