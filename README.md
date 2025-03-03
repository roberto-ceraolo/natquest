# _Quriosity_: Analyzing Human Questioning Behavior and Causal Inquiry through Curiosity-Driven Queries

![Related Work](related_work.png)

<p align="center"> Table 1: Comparison of the source of questions for various datasets, including datasets commonly used in LLM
testing. Given their curated nature, the questions in these datasets are not suitable for use in studying curiosity-driven
questioning behavior.
</p>


This repository accompanies our research paper titled "**_Quriosity_: Analyzing Human Questioning Behavior and Causal Inquiry through Curiosity-Driven Queries**" by *Roberto Ceraolo\*, Dmitrii Kharlapenko\*, Ahmad Khan\*, Amélie Reymond, Rada Mihalcea, Bernhard Schölkopf, Mrinmaya Sachan, Zhijing Jin*.
The repo contains the code that we used to generate the dataset, the results and the plots that we used in the paper. The dataset will soon be published on the Hugging Face Hub.

## Repository Structure

The following is the structure of the repo:

```
- src/
  - causalQuest_generation/
    - causalquest_download_sources.py: Scripts to download and preprocess data from various sources
    - nq_sampling.py: Script to sample data from the Natural Questions dataset
  - classification/
    - submit_batch_job.py: Script to submit batch jobs for question classification
    - process_batch_results.py: Script to process the results from batch classification jobs
    - classification_utils.py: Utility functions for classification tasks
  - fine_tuning/
    - train_phi_lora.py: Script to fine-tune the Phi-1.5 model with LoRA
    - train_flan_lora.py: Script to fine-tune FLAN model with LoRA
  - analyses_clusters.py: Script for performing clustering analysis on embeddings
  - embedding_generation.py: Script for generating embeddings for questions
  - evaluate.py: Script for evaluating model predictions against human annotations
  - linguistic_baseline.py: Script implementing linguistic baseline analysis
  - plots.py: Script for creating various plots used in the paper
  - politeness.py: Script for analyzing politeness in questions
  - sampling_testing_dataset.py: Script for sampling datasets for testing

```

This structure reflects the main components of the project, including data generation, classification, fine-tuning, analysis, and evaluation scripts. The `src/` directory contains the core source code. 

## Dataset

### Data Schema

The dataset is structured with the following fields for each question:

- `query_id`: A unique identifier for each question (integer)
- `causal_type`: The type of causal question, if applicable (string)
- `source`: The origin of the question (e.g., 'nq', 'quora', 'msmarco', 'wildchat', 'sharegpt')
- `query`: The original question text
- `shortened_query`: A condensed version of the question
- `is_causal_raw`: Raw classification of causal nature, with CoT reasoning
- `is_causal`: Boolean indicating whether the question is causal
- `domain_class`: Classification of the question's domain
- `action_class`: Classification of the action type
- `is_subjective`: Indicates if the question is subjective
- `bloom_taxonomy`: Categorization based on Bloom's taxonomy
- `user_needs`: Classification of the user's needs
- `uniqueness`: Measure of the answer's uniqueness

This structure allows for comprehensive analysis of question types, sources, and characteristics as detailed in the paper.

### Dataset Statistics
| Nature of Data | Dataset | # Samples |
|----------------|---------|-----------|
| H-to-SE (33.3%) | MSMarco (2018) | 2,250 |
| | NaturalQuestions (2019) | 2,250 |
| H-to-H (33.3%) | Quora Question Pairs | 4,500 |
| H-to-LLM (33.3%) | ShareGPT | 2,250 |
| | WildChat (2024) | 2,250 |

Table 1: Our dataset equally covers questions from the three source types: human-to-search-engine queries (H-to-SE), human-to-human interactions (H-to-H), and human-to-LLM interactions (H-to-LLM).

| Category | Curated Questions | Quriosity |
|----------|------------------------|----------|
| **Open-Endedness** |
| Open-Ended | 30% | 68% |
| **Cognitive Complexity** |
| Remembering | 30.51% | 36.82% |
| Understanding | 7.47% | 13.47% |
| Applying | 36.41% | 13.54% |
| Analyzing | 13.90% | 8.8% |
| Evaluating | 11.54% | 13.74% |
| Creating | 0.17% | 13.62% |

Table 2: Comparison of questions in curated tests and our Quriosity in terms of open-endedness and cognitive complexity.


## Code Setup

1. Clone this repository:
   ```
   git clone https://github.com/roberto-ceraolo/natquest.git
   cd natquest
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   ```
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Code Usage

### Data Collection and Preprocessing

The data collection and preprocessing scripts are located in the `src/causalQuest_generation/` directory. Key files include:

- `causalquest_download_sources.py`: Downloads and preprocesses data from various sources.
- `nq_sampling.py`: Samples data from the Natural Questions dataset.

### Analysis

The main analysis scripts are located in the `src/` directory:

- `analyses_clusters.py`: Performs clustering analysis on the embeddings.
- `embedding_generation.py`: Generates embeddings for the questions.
- `linguistic_baseline.py`: Implements linguistic baseline analysis.
- `plots.py`: Creates various plots for the paper.


### Classification

The classification scripts are in the `src/classification/` directory:

- `submit_batch_job.py`: Submits batch jobs for question classification.
- `process_batch_results.py`: Processes the results from batch classification jobs.

To run the classification:
```
python src/classification/submit_batch_job.py --job_type causality

python src/classification/process_batch_results.py
```

### Evaluation

The evaluation script is located in `src/evaluate.py`.

### Fine-tuning

The fine-tuning script for the Phi-1.5 model is in `src/fine_tuning/train_phi_lora.py` and for the FLAN models is in `src/fine_tuning/train_flan_lora.py`.


## License 

This repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please open an issue on this repository or contact the authors.