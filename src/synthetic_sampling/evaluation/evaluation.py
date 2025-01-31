import torch
import torch.nn.functional as F
from .config import DEFAULT_CONFIG

config = DEFAULT_CONFIG
tokenizer = config.tokenizer
model = config.model

def compute_manual_perplexity(input_ids, label):
    """
    Computes the perplexity of a given label using manual token-level probability calculations.

    This function concatenates the provided input prompt (`input_ids`) with the tokenized label,
    retrieves model logits for each label token, and calculates perplexity based on the average
    log probability of the label tokens.

    Args:
        input_ids (torch.Tensor): The input token IDs (prompt) in tensor format.
        label (str): The label text whose perplexity is being computed.

    Returns:
        float: The computed perplexity score for the given label.
    """

    # Tokenize label (without special tokens)
    label_tokens = tokenizer.encode(label, add_special_tokens=False, return_tensors="pt").to(model.device)

    # Concatenate prompt and label tokens
    full_input_ids = torch.cat([input_ids, label_tokens], dim=-1)

    # Compute logits
    with torch.no_grad():
        outputs = model(full_input_ids)
        logits = outputs.logits  # Shape: [1, sequence_length, vocab_size]

    # Extract the log probabilities for the label tokens
    log_probs = []
    for i in range(label_tokens.shape[1]):  # Iterate over label tokens
        token_id = label_tokens[0, i]
        token_logits = logits[0, input_ids.shape[1] + i - 1]  # Get logits for this position
        token_log_prob = F.log_softmax(token_logits, dim=-1)[token_id]  # Log probability of the correct token
        log_probs.append(token_log_prob.item())

    # Compute perplexity: PPL = exp(-sum(log_probs) / N)
    avg_log_prob = -sum(log_probs) / len(log_probs)
    perplexity = torch.exp(torch.tensor(avg_log_prob)).item()
    return perplexity


def calculate_perplexity(input_ids, label):
    """
    Computes the perplexity of a given label using the model's built-in loss function.

    This function tokenizes the label, concatenates it with the input prompt (`input_ids`),
    and computes perplexity by passing the full input through the model with label-based loss masking.
    The model's cross-entropy loss (averaged over label tokens) is exponentiated to obtain perplexity.

    Args:
        input_ids (torch.Tensor): The input token IDs (prompt) in tensor format.
        label (str): The label text whose perplexity is being computed.

    Returns:
        float: The computed perplexity score for the given label.
    """

    # Tokenize label (without special tokens)
    label_tokens = tokenizer.encode(label, add_special_tokens=False, return_tensors="pt").to(model.device)

    # Concatenate prompt with label tokens
    full_input_ids = torch.cat([input_ids, label_tokens], dim=-1)

    # Create labels tensor (only the label tokens should contribute to loss)
    labels = full_input_ids.clone()
    labels[:, :input_ids.shape[1]] = -100  # Ignore the prompt part in loss calculation

    # Get model outputs
    with torch.no_grad():
        outputs = model(full_input_ids, labels=labels)
        loss = outputs.loss  # Cross-entropy loss (mean over tokens)

    # Compute perplexity
    perplexity = torch.exp(loss).item()
    return perplexity
