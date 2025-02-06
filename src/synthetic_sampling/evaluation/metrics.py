from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import EvaluationConfig
from data_processing import Mapper, SurveyDataset, simple_collate_fn
from evaluation import calculate_batch_perplexity


def evaluate_single_question(
    df: pd.DataFrame,
    qid: str,
    mapper: Mapper,
    config: EvaluationConfig,
    survey_mappings: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluate one question (identified by qid) using the full evaluation pipeline.

    For the given question, build a dataset and DataLoader, run batched inference,
    compute probability estimates from perplexities, and then calculate evaluation
    metrics (Brier score, macro and weighted F1, and per-category F1). The returned
    dictionary includes the question description as well as the model name.

    Returns:
        A dictionary of metrics for the question.
    """
    try:
        # Build the dataset for this question.
        dataset = SurveyDataset(df, qid, mapper, config, survey_mappings)
    except ValueError as e:
        print(f"Skipping question {qid}: {e}")
        return {}

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=simple_collate_fn,
        num_workers=4,
        shuffle=False
    )

    all_true_labels = []
    all_pred_labels = []
    all_probabilities = []

    for batch in tqdm(dataloader, desc=f"Evaluating question {qid}"):
        ppl_matrix = calculate_batch_perplexity(
            batch,
            dataset.label_options,
            config.model,
            config.tokenizer
        )
        # Convert perplexity to probability estimates.
        # (Lower perplexity should mean higher probability; use the inverse and normalize.)
        inverse_ppl = 1.0 / ppl_matrix
        probs = inverse_ppl / inverse_ppl.sum(dim=1, keepdim=True)

        # Predicted label is the candidate with the highest probability.
        pred_indices = torch.argmax(probs, dim=1)
        batch_pred_labels = [dataset.label_options[idx] for idx in pred_indices.tolist()]
        batch_true_labels = [sample["true_label"] for sample in batch]

        all_true_labels.extend(batch_true_labels)
        all_pred_labels.extend(batch_pred_labels)
        all_probabilities.append(probs.cpu().numpy())

    # Concatenate probability distributions.
    all_probabilities = np.concatenate(all_probabilities, axis=0)  # Shape: (num_samples, num_label_options)

    # Compute Brier score.
    num_samples, num_labels = all_probabilities.shape
    y_true_onehot = np.zeros_like(all_probabilities)
    for i, true_label in enumerate(all_true_labels):
        idx = dataset.label_options.index(true_label)
        y_true_onehot[i, idx] = 1
    brier_scores = np.mean((all_probabilities - y_true_onehot) ** 2, axis=1)
    avg_brier_score = np.mean(brier_scores)

    # Compute F1 scores.
    macro_f1 = f1_score(all_true_labels, all_pred_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_true_labels, all_pred_labels, average="weighted", zero_division=0)
    class_report = classification_report(all_true_labels, all_pred_labels, output_dict=True)

    # Retrieve the question description from the survey mappings.
    question_info = survey_mappings.get(qid, {})
    question_description = question_info.get("description", "No description available")

    model_name = config.model.__class__.__name__

    metrics = {
        "question_id": qid,
        "question_description": question_description,
        "num_samples": num_samples,
        "avg_brier_score": avg_brier_score,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "class_report": class_report,
        "model_name": model_name
    }
    return metrics


def evaluate_questions(
    df: pd.DataFrame,
    question_ids: List[str],
    mapper: Mapper,
    config: EvaluationConfig,
    survey_mappings: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Evaluate the model over a list of questions.

    Returns:
        A list of dictionaries, each containing evaluation metrics for one question,
        including the question description and model name.
    """
    all_metrics = []
    for qid in question_ids:
        print(f"Evaluating question: {qid}")
        metrics = evaluate_single_question(df, qid, mapper, config, survey_mappings)
        if metrics:
            all_metrics.append(metrics)
    return all_metrics
