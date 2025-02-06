from typing import Dict, List, Any

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .config import DEFAULT_CONFIG

config = DEFAULT_CONFIG
tokenizer = config.tokenizer
model = config.model

def compute_manual_perplexity(input_ids, label):
    """
    Compute the perplexity of a given label using manual token-level probability calculations.

    This function concatenates the provided input prompt (`input_ids`) with the tokenized label,
    retrieves model logits for each label token, and calculates perplexity based on the average
    log probability of the label tokens.

    Args:
        input_ids (torch.Tensor): The input token IDs (prompt) in tensor format.
        label (str): The label text whose perplexity is being computed.

    Returns:
        float: The computed perplexity score for the given label.
    """

    label_tokens = tokenizer.encode(label, add_special_tokens=False, return_tensors="pt").to(model.device)
    full_input_ids = torch.cat([input_ids, label_tokens], dim=-1)

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
    Compute the perplexity of a given label using the model's built-in loss function.

    This function tokenizes the label, concatenates it with the input prompt (`input_ids`),
    and computes perplexity by passing the full input through the model with label-based loss masking.
    The model's cross-entropy loss (averaged over label tokens) is exponentiated to obtain perplexity.

    Args:
        input_ids (torch.Tensor): The input token IDs (prompt) in tensor format.
        label (str): The label text whose perplexity is being computed.

    Returns:
        float: The computed perplexity score for the given label.
    """

    label_tokens = tokenizer.encode(label, add_special_tokens=False, return_tensors="pt").to(model.device)
    full_input_ids = torch.cat([input_ids, label_tokens], dim=-1)

    # Create labels tensor (only the label tokens should contribute to loss)
    labels = full_input_ids.clone()
    labels[:, :input_ids.shape[1]] = -100

    with torch.no_grad():
        outputs = model(full_input_ids, labels=labels)
        loss = outputs.loss  # Cross-entropy loss (mean over tokens)

    perplexity = torch.exp(loss).item()
    return perplexity

def calculate_batch_perplexity(batch: List[Dict[str, Any]], label_options: List[str],
                               model: torch.nn.Module, tokenizer: Any) -> torch.Tensor:
    """
    Compute per-example perplexity for each candidate label.
    For each sample in the batch, tokenizes the prompt (using apply_chat_template) and concatenates
    it with each candidate label (via encode). Uses vectorized masking to ignore prompt/padding tokens,
    shifts logits/labels for causal LM prediction, and computes perplexity.

    Returns a tensor of shape (num_examples, num_label_options).
    """

    label_to_ids = {
        label: tokenizer.encode(label, add_special_tokens=False, return_tensors="pt").squeeze(0)
        for label in label_options
    }
    full_ids_list = []  # List to hold concatenated sequences (prompt + label).
    prompt_lengths = []  # To record prompt length per sample.
    num_examples = len(batch)

    for example in batch:
        # Tokenize the prompt. (apply_chat_template may return a dict or a tensor.)
        prompt_enc = tokenizer.apply_chat_template(
            example['chat'],
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        prompt_ids = prompt_enc["input_ids"] if isinstance(prompt_enc, dict) else prompt_enc
        # Remove extra batch dimension: [1, seq_len] → [seq_len]
        prompt_ids = prompt_ids.squeeze(0)
        pl = prompt_ids.size(0)
        for label in label_options:
            label_ids = label_to_ids[label]
            full_ids = torch.cat([prompt_ids, label_ids], dim=0)
            full_ids_list.append(full_ids)
            prompt_lengths.append(pl)

    padded_input_ids = pad_sequence(full_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    total_sequences, max_seq_len = padded_input_ids.shape

    # Build a labels tensor that mirrors padded_input_ids.
    labels_tensor = padded_input_ids.clone()
    # Vectorized masking: create a (total_sequences, max_seq_len) grid.
    prompt_lengths_tensor = torch.tensor(prompt_lengths, dtype=torch.long)
    seq_range = torch.arange(max_seq_len).unsqueeze(0).expand(total_sequences, max_seq_len)
    mask_prompt = seq_range < prompt_lengths_tensor.unsqueeze(1)
    labels_tensor[mask_prompt] = -100
    # Mask out padding tokens.
    labels_tensor[labels_tensor == tokenizer.pad_token_id] = -100

    padded_input_ids = padded_input_ids.to(model.device)
    labels_tensor = labels_tensor.to(model.device)

    with torch.no_grad():
        outputs = model(padded_input_ids)
        logits = outputs.logits  # Shape: (total_sequences, max_seq_len, vocab_size)

    # Shift logits and labels for causal LM loss calculation.
    shift_logits = logits[:, :-1, :]   # Remove final time step.
    shift_labels = labels_tensor[:, 1:]  # Remove first token.
    vocab_size = shift_logits.size(-1)
    logits_flat = shift_logits.reshape(-1, vocab_size)
    labels_flat = shift_labels.reshape(-1)

    losses_flat = F.cross_entropy(logits_flat, labels_flat, reduction='none', ignore_index=-100)
    losses = losses_flat.view(total_sequences, max_seq_len - 1)
    mask = (shift_labels != -100).float()
    loss_per_sequence = (losses * mask).sum(dim=1) / mask.sum(dim=1)
    ppl_per_sequence = torch.exp(loss_per_sequence)
    num_labels = len(label_options)
    ppl_matrix = ppl_per_sequence.view(num_examples, num_labels)
    return ppl_matrix
