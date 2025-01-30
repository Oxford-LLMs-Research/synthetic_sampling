import numpy as np
from sklearn.metrics import f1_score, brier_score_loss
from typing import List, Dict, Any

def compute_metrics(all_results: List[Dict[str, Any]], label_options: List[str]) -> Dict[str, Any]:
    true_labels = []
    predicted_probabilities = []
    predicted_labels = []

    for result in all_results:
        true_label = result['true_label']
        perplexities = result['perplexities']
        inverse_perplexities = {label: 1 / perplexity for label, perplexity in perplexities.items()}
        total = sum(inverse_perplexities.values())
        probabilities = {label: inv_ppl / total for label, inv_ppl in inverse_perplexities.items()}
        true_labels.append(true_label)
        predicted_probabilities.append([probabilities.get(label, 0.0) for label in label_options])
        predicted_label = max(probabilities, key=probabilities.get)
        predicted_labels.append(predicted_label)

    true_labels = np.array(true_labels)
    predicted_probabilities = np.array(predicted_probabilities)
    predicted_labels = np.array(predicted_labels)

    brier_scores = []
    for i, label in enumerate(label_options):
        true_binary = (true_labels == label).astype(int)
        prob_binary = predicted_probabilities[:, i]
        brier_scores.append(brier_score_loss(true_binary, prob_binary))
    brier_score = np.mean(brier_scores)

    macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
    weighted_f1 = f1_score(true_labels, predicted_labels, average='weighted')
    per_label_f1 = {label: f1_score(true_labels, predicted_labels, labels=[label], average=None)[0] for label in label_options}

    return {
        'brier_score': brier_score,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_label_f1': per_label_f1
    }
