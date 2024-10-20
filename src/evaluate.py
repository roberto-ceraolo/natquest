import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, List, Tuple

def calculate_weighted_kappa(y_true: List[str], y_pred: List[str]) -> float:
    """Calculate weighted Cohen's Kappa for Bloom's Taxonomy classifications."""
    bloom_levels = ["Remembering", "Understanding", "Applying", "Analyzing", "Evaluating", "Creating"]
    n_levels = len(bloom_levels)
    level_to_index = {level: index for index, level in enumerate(bloom_levels)}
    
    # Convert string labels to numeric indices
    y_true_num = [level_to_index[label] for label in y_true]
    y_pred_num = [level_to_index[label] for label in y_pred]
    
    # Create weight matrix
    weights = np.zeros((n_levels, n_levels))
    for i in range(n_levels):
        for j in range(n_levels):
            weights[i, j] = 1 - (abs(i - j) / (n_levels - 1))
    
    # Calculate observed agreement
    observed = sum(weights[i, j] for i, j in zip(y_true_num, y_pred_num)) / len(y_true)
    
    # Calculate expected agreement
    true_counts = np.bincount(y_true_num, minlength=n_levels)
    pred_counts = np.bincount(y_pred_num, minlength=n_levels)
    expected = sum(weights[i, j] * true_counts[i] * pred_counts[j] 
                   for i in range(n_levels) 
                   for j in range(n_levels)) / (len(y_true) ** 2)
    
    # Calculate weighted Cohen's Kappa
    kappa = (observed - expected) / (1 - expected)
    return kappa


def load_human_annotations(file_path: str) -> pd.DataFrame:
    """Load human annotations from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        agreement_columns = [col for col in df.columns if 'agreement' in col]

        # where df["uniqueness_agreement"] is Unique answer, replace with Unique Answer
        df["uniqueness_agreement"] = df["uniqueness_agreement"].replace("Unique answer", "Unique Answer")

        return df[['query_id', 'shortened_query'] + agreement_columns]
    except FileNotFoundError:
        raise FileNotFoundError(f"Human annotations file not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file {file_path} is empty.")

def load_model_predictions(predictions_file: str) -> pd.DataFrame:
    """Load model predictions from a JSONL file."""
    try:
        with open(predictions_file, 'r') as f:
            predictions = [json.loads(line) for line in f]
        df = pd.DataFrame(predictions)
        if 'primary_need_category' in df.columns:
            df.rename(columns={"primary_need_category": "category"}, inplace=True)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Model predictions file not found: {predictions_file}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in file: {predictions_file}")

def calculate_accuracy(human_df: pd.DataFrame, model_df: pd.DataFrame, annotator_category: str) -> Tuple[float, pd.DataFrame]:
    """Calculate accuracy and return classification report and merged dataframe."""
    human_df = human_df[['query_id', annotator_category, "shortened_query"]].rename(columns={annotator_category: 'category'})
    model_df = model_df[['query_id', 'category']]
    merged_df = pd.merge(human_df, model_df, how='inner', on='query_id', suffixes=('_human', '_model'))
    
    y_true = merged_df['category_human']
    y_pred = merged_df['category_model']
    
    if annotator_category == "cognitive_complexity_agreement":
        weighted_f1 = calculate_weighted_kappa(y_true, y_pred)
        report = {'bloom_weighted_kappa': weighted_f1}
    else:
        report = classification_report(y_true, y_pred, output_dict=True)

    return report, merged_df

def main():
    # add a parser to get the human annotations file
    parser = argparse.ArgumentParser(description="Evaluate model predictions against human annotations.")
    parser.add_argument("--human_annotations file", help="Path to the human annotations file", required=True)
    args = parser.parse_args()

    human_annotations_file = args.human_annotations_file

    model_predictions = {
        "uniqueness_classifications.jsonl": "uniqueness_agreement",
        "answerability_classifications.jsonl": "answerability_agreement",
        "bloom_classifications.jsonl": "cognitive_complexity_agreement",
        "question_classifications_user_needs.jsonl": "needs_agreement"
    }

    try:
        human_df = load_human_annotations(human_annotations_file)
        
        for pred_file, annotator_category in model_predictions.items():
            print(f"Calculating accuracy for {pred_file}")
            model_df = load_model_predictions(pred_file)
            report, merged_df = calculate_accuracy(human_df, model_df, annotator_category)
            print(f"Classification report for {pred_file}:")
            print(report)
            if annotator_category == "cognitive_complexity_agreement":
                print(f"Weighted kappa score for {pred_file}: {report['bloom_weighted_kappa']}")
            else:
                print(f"Weighted F1 score for {pred_file}: {report['weighted avg']['f1-score']}")
            print()

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()