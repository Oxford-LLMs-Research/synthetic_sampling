import torch
import torch.nn.functional as F


def compute_manual_perplexity_batch(batch, label_options, config):
    """
    Compute perplexity for each label option for all examples in the batch simultaneously.

    Args:
        batch: A batch of examples, each containing a 'chat' and 'true_label'.
        label_options: List of possible answer labels.
        config: Evaluation config.

    Returns:
        A list of dictionaries, where each dictionary corresponds to an example and contains:
            - 'true_label': The true label for the example.
            - 'perplexities': A dictionary mapping each label option to its perplexity.
    """
    # Tokenize all labels once
    label_tokens = {
        label: config.tokenizer.encode(label, add_special_tokens=False, return_tensors="pt").to(config.device)
        for label in label_options
    }

    # Tokenize all prompts in the batch
    input_ids_list = [
        config.tokenizer.apply_chat_template(
            example['chat'],
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(config.device)
        for example in batch
    ]

    # Prepare inputs for all examples and labels
    full_input_ids_list = []
    label_lengths = []
    for label in label_options:
        label_token_ids = label_tokens[label]
        label_length = label_token_ids.shape[1]
        label_lengths.append(label_length)

        # Concatenate each prompt with the current label
        for input_ids in input_ids_list:
            full_input_ids = torch.cat([input_ids, label_token_ids], dim=-1)
            full_input_ids_list.append(full_input_ids)

    # Pad all sequences to the same length
    max_length = max([x.shape[1] for x in full_input_ids_list])
    padded_input_ids_list = [
        torch.nn.functional.pad(x, (0, max_length - x.shape[1]), value=config.tokenizer.pad_token_id)
        for x in full_input_ids_list
    ]

    # Stack all inputs into a single batch
    full_input_ids_batch = torch.cat(padded_input_ids_list, dim=0)

    # Compute logits in a single forward pass
    with torch.no_grad():
        outputs = config.model(full_input_ids_batch)
        logits = outputs.logits  # Shape: [num_examples * num_labels, sequence_length, vocab_size]

    # Reshape logits to separate examples and labels
    num_examples = len(batch)
    num_labels = len(label_options)
    logits = logits.view(num_examples, num_labels, -1, logits.shape[-1])  # Shape: [num_examples, num_labels, sequence_length, vocab_size]

    # Compute perplexity for each example and label
    results = []
    for i, example in enumerate(batch):
        example_perplexities = {}
        for j, label in enumerate(label_options):
            # Extract logits for the current label
            label_logits = logits[i, j, -label_lengths[j]:]  # Shape: [label_length, vocab_size]

            # Compute log probabilities for the correct tokens
            label_token_ids = label_tokens[label][0]
            log_probs = F.log_softmax(label_logits, dim=-1)[range(label_lengths[j]), label_token_ids]

            # Compute perplexity: PPL = exp(-sum(log_probs) / N)
            avg_log_prob = -log_probs.mean().item()
            perplexity = torch.exp(torch.tensor(avg_log_prob)).item()
            example_perplexities[label] = perplexity

        # Store results for this example
        results.append({
            'true_label': example['true_label'],
            'perplexities': example_perplexities
        })

    return results
