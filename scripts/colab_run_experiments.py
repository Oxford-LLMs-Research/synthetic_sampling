"""Colab GPU runner for the two paired follow-up experiments.

Usage on Colab (A100/GH200-class GPU, ~65 GB VRAM for bf16 Qwen3-32B):

    !pip install -q transformers accelerate
    # upload the two *_instances.jsonl files (or mount Drive), then:
    !python colab_run_experiments.py --experiment temporal \
        --input temporal_context_instances.jsonl --out temporal_results.jsonl
    !python colab_run_experiments.py --experiment country \
        --input country_injection_instances.jsonl --out country_results.jsonl

Each instance is scored under conditions that share the identical
pipeline and differ only in the manipulated element:

  temporal:  baseline | with_year (true) | with_year_placebo | with_date
             (fine interview date, only when interview_date is present).
  country:   original profile vs  the same profile with one appended item
             "In which country do you live?" -> country name.

Scoring follows the paper's perplexity method: each answer option is scored
by the mean log-probability of its tokens conditioned on the prompt (plain
completion, no chat template, matching the method used uniformly for base
and instruct models); the prediction is the option with the lowest
perplexity. Results are appended per instance so interrupted runs resume
where they stopped.

Output records (one per instance x condition):
    example_id, condition, ground_truth, predicted, correct,
    option_logprobs (mean per-token logprob per option)
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = (
    "You are a helpful assistant. Predict how the respondent would answer "
    "the target question using their prior answers.\n\n"
    "Profile: {profile}\n\n"
    "{extra}"
    "Question: {question}\n\n"
    "Instructions: Reply with a short concise answer. No reasoning. "
    "No explanation. No extra text.\n\n"
    "Answer:"
)


def render_profile(questions: dict) -> str:
    return "\n\n".join(f"Q: {q}\nA: {a}" for q, a in questions.items())


def temporal_conditions(instance: dict) -> list[str]:
    conds = ["baseline", "with_year", "with_year_placebo"]
    if instance.get("interview_date"):
        conds.append("with_date")
    return conds


def build_prompt(instance: dict, condition: str, experiment: str) -> str:
    questions = dict(instance["questions"])
    extra = ""
    if experiment == "temporal":
        if condition == "with_year":
            extra = f"The survey was conducted in {instance['survey_year']}.\n\n"
        elif condition == "with_year_placebo":
            extra = (
                f"The survey was conducted in {instance['survey_year_placebo']}.\n\n"
            )
        elif condition == "with_date":
            extra = (
                f"The interview took place on {instance['interview_date']}.\n\n"
            )
    if experiment == "country" and condition == "with_country":
        questions[instance["country_question"]] = instance["country_name"]
    if experiment == "country" and condition == "with_country_placebo":
        questions[instance["country_question"]] = instance["country_placebo_name"]
    return PROMPT_TEMPLATE.format(
        profile=render_profile(questions),
        extra=extra,
        question=instance["target_question"],
    )


@torch.no_grad()
def score_options(model, tokenizer, prompt: str, options: list, device) -> dict:
    """Mean per-token logprob of each option, conditioned on the prompt."""
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    scores = {}
    seqs, opt_lens = [], []
    for opt in options:
        opt_ids = tokenizer(" " + opt, add_special_tokens=False,
                            return_tensors="pt").input_ids[0]
        seqs.append(torch.cat([prompt_ids, opt_ids]))
        opt_lens.append(len(opt_ids))

    maxlen = max(len(s) for s in seqs)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    batch = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        batch[i, :len(s)] = s
        attn[i, :len(s)] = 1

    logits = model(batch.to(device), attention_mask=attn.to(device)).logits
    logprobs = torch.log_softmax(logits.float(), dim=-1)

    for i, opt in enumerate(options):
        n_opt = opt_lens[i]
        seq_len = int(attn[i].sum())
        # tokens at positions [seq_len - n_opt, seq_len) are the option;
        # each is predicted by the logits at the previous position
        positions = range(seq_len - n_opt, seq_len)
        lp = [logprobs[i, p - 1, batch[i, p]].item() for p in positions]
        scores[opt] = sum(lp) / len(lp)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["temporal", "country"], required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N instances (smoke test)")
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done.add((r["example_id"], r["condition"]))
                except json.JSONDecodeError:
                    pass
        print(f"resuming: {len(done)} (instance, condition) pairs already scored")

    instances = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    if args.limit:
        instances = instances[:args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto")
    except (ValueError, ImportError):  # no accelerate: single-device fallback
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype).to(dev)
    model.eval()
    device = next(model.parameters()).device

    n_scored = 0
    with open(args.out, "a", encoding="utf-8") as out_fh:
        for k, inst in enumerate(instances):
            conditions = (temporal_conditions(inst) if args.experiment == "temporal"
                          else ["baseline", "with_country", "with_country_placebo"])
            for cond in conditions:
                if (inst["example_id"], cond) in done:
                    continue
                prompt = build_prompt(inst, cond, args.experiment)
                scores = score_options(model, tokenizer, prompt,
                                       inst["options"], device)
                predicted = max(scores, key=scores.get)
                gt = inst.get("answer") or inst.get("ground_truth")
                rec = {
                    "example_id": inst["example_id"],
                    "condition": cond,
                    "ground_truth": gt,
                    "predicted": predicted,
                    "correct": predicted == gt,
                    "option_logprobs": scores,
                }
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_fh.flush()
                n_scored += 1
            if (k + 1) % 50 == 0:
                print(f"{k + 1}/{len(instances)} instances "
                      f"({n_scored} new condition-scores)")
    print(f"done: {n_scored} new condition-scores written to {args.out}")


if __name__ == "__main__":
    main()
