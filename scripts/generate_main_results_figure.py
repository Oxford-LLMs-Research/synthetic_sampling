#!/usr/bin/env python
"""
Generate three-panel figure for ICML paper:
  Panel A: Accuracy vs Model Size (log scale)
  Panel B: Median Variance Ratio vs Model Size (log scale)
  Panel C: Median JS Divergence vs Model Size (log scale)

Panels A and B show horizontal reference lines for baselines.

Usage:
    python generate_figure.py --results-dir /path/to/results/
    python generate_figure.py --results-dir /path/to/results/ --output figure1.pdf
    
Requirements:
    pip install matplotlib numpy pandas
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.spatial.distance import jensenshannon


# =============================================================================
# MODEL METADATA
# =============================================================================

@dataclass
class ModelInfo:
    """Metadata for a model."""
    display_name: str
    family: str
    size_b: float  # Size in billions (active params for MoE)
    model_type: str  # 'base' or 'instruct'
    is_moe: bool = False
    total_params_b: Optional[float] = None  # For MoE models

# Model folder name -> ModelInfo
MODEL_METADATA: Dict[str, ModelInfo] = {
    # Llama 3.1
    'llama3.1-8b-base': ModelInfo('Llama 3.1 8B', 'Llama', 8, 'base'),
    'llama3.1-8b-instruct': ModelInfo('Llama 3.1 8B', 'Llama', 8, 'instruct'),
    'llama3.1-70b-base': ModelInfo('Llama 3.1 70B', 'Llama', 70, 'base'),
    'llama3.1-70b-instruct': ModelInfo('Llama 3.1 70B', 'Llama', 70, 'instruct'),
    
    # OLMo 3
    'olmo3-7b-base': ModelInfo('OLMo 3 7B', 'OLMo', 7, 'base'),
    'olmo3-7b-dpo': ModelInfo('OLMo 3 7B', 'OLMo', 7, 'instruct'),
    'olmo3-32b-base': ModelInfo('OLMo 3 32B', 'OLMo', 32, 'base'),
    'olmo3-32b-dpo': ModelInfo('OLMo 3 32B', 'OLMo', 32, 'instruct'),
    
    # Qwen 3
    'qwen3-4b': ModelInfo('Qwen 3 4B', 'Qwen', 4, 'instruct'),
    'qwen3-32b': ModelInfo('Qwen 3 32B', 'Qwen', 32, 'instruct'),
    
    # GPT-OSS (OpenAI)
    'gpt_oss': ModelInfo('GPT-OSS 120B', 'GPT-OSS', 120, 'instruct'),
    
    # DeepSeek (MoE - using total params in memory)
    'deepseek-v3p1-terminus': ModelInfo('DeepSeek-V3.1', 'DeepSeek', 685, 'instruct', 
                                         is_moe=True, total_params_b=685),
    
    # Gemma 3
    'gemma-3-27b-instruct': ModelInfo('Gemma 3 27B', 'Gemma', 27, 'instruct'),
}

# Visual styling per family - use only color, same marker for all
# Professional color palette (colorblind-friendly, high contrast)
FAMILY_STYLES = {
    'Llama': {'color': '#2E86AB'},      # Blue
    'OLMo': {'color': '#A23B72'},       # Purple
    'Qwen': {'color': '#F18F01'},       # Orange
    'GPT-OSS': {'color': '#C73E1D'},    # Red
    'DeepSeek': {'color': '#6A994E'},   # Green
    'Gemma': {'color': '#BC4749'},      # Dark red
}


# =============================================================================
# DATA LOADING
# =============================================================================

@dataclass
class ModelResults:
    """Aggregated results for a single model."""
    model_name: str
    info: ModelInfo
    n_instances: int
    accuracy: float
    random_baseline: float
    majority_baseline: float
    macro_f1: float
    median_vr_soft: Optional[float] = None
    flattening_rate: Optional[float] = None
    median_js_soft: Optional[float] = None  # Median JS divergence


def load_jsonl_files(folder: Path, profile_filter: Optional[str] = 's6m4') -> List[dict]:
    """
    Load all JSONL files from a folder.
    
    Args:
        folder: Path to folder containing JSONL files
        profile_filter: Only include instances with this profile type in their ID.
                       Default 's6m4' (rich profile). Use None to include all.
                       Profile types: 's3m2' (sparse), 's4m3' (medium), 's6m4' (rich)
    """
    instances = []
    skipped = 0
    
    for jsonl_file in folder.glob("*.jsonl"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    inst = json.loads(line)
                    
                    # Filter by profile type if specified
                    if profile_filter:
                        example_id = inst.get('example_id', '')
                        if not example_id.endswith(f'_{profile_filter}'):
                            skipped += 1
                            continue
                    
                    instances.append(inst)
    
    if profile_filter and skipped > 0:
        print(f"    Filtered to '{profile_filter}' profile: kept {len(instances):,}, skipped {skipped:,}")
    
    return instances


def compute_accuracy(instances: List[dict]) -> Tuple[float, int]:
    """Compute accuracy from instances."""
    if not instances:
        return 0.0, 0
    
    correct = sum(1 for inst in instances if inst.get('correct', False))
    return correct / len(instances), len(instances)


def extract_question_id(instance: dict) -> str:
    """
    Extract the question identifier from an instance.
    
    example_id format: "{survey}_{respondent}_{target}_{profile_type}"
    We want: "{survey}_{target}" as the question identifier
    (same question across all respondents)
    
    Handles target codes with underscores (e.g., Q725_4, Q601_21B)
    """
    example_id = instance.get('example_id', '')
    
    # Parse: ends with _s\dm\d (profile type)
    import re
    match = re.match(r'^(.+)_(s\d+m\d+)$', example_id)
    if not match:
        # Fallback to full ID
        return example_id
    
    prefix = match.group(1)  # "{survey}_{respondent}_{target}"
    
    # Known surveys (check longer ones first to avoid partial matches)
    known_surveys = [
        'ess_wave_11', 'ess_wave_10',  # Multi-word surveys first
        'afrobarometer', 'arabbarometer', 'asianbarometer', 
        'latinobarometer', 'wvs'
    ]
    
    # Find which survey this is
    survey = None
    survey_prefix = None
    for s in known_surveys:
        if prefix.startswith(s + '_'):
            survey = s
            survey_prefix = s + '_'
            break
    
    if survey_prefix:
        # Remove survey prefix: remainder is "{respondent_id}_{target_code}"
        remainder = prefix[len(survey_prefix):]
        
        # Target codes typically start with Q, P, S, or are lowercase identifiers
        # Find where target code starts by working backwards
        parts = remainder.split('_')
        
        if len(parts) == 1:
            # Only one part - must be the target (no respondent ID)
            return f"{survey}_{remainder}"
        elif len(parts) == 2:
            # Two parts: likely {respondent}_{target}
            # Check if second part looks like target code start
            if parts[1] and (parts[1][0] in 'QPS' or parts[1][0].islower()):
                return f"{survey}_{parts[1]}"
            else:
                # Assume first is respondent, second is target
                return f"{survey}_{parts[1]}"
        else:
            # Multiple parts: find where target code starts
            # Target codes typically start with Q/P/S or are lowercase
            target_start_idx = len(parts) - 1
            for i in range(len(parts) - 1, -1, -1):
                if parts[i] and (parts[i][0] in 'QPS' or parts[i][0].islower()):
                    target_start_idx = i
                    break
            
            # Everything from target_start_idx onwards is target_code
            target_code = '_'.join(parts[target_start_idx:])
            return f"{survey}_{target_code}"
    
    # Fallback: take first and last underscore-separated parts
    parts = prefix.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[-1]}"
    
    return example_id


def compute_baselines(instances: List[dict]) -> Tuple[float, float]:
    """Compute random and majority baselines."""
    if not instances:
        return 0.0, 0.0
    
    # Random baseline: average of 1/n_options
    random_scores = []
    for inst in instances:
        n_opts = inst.get('n_options') or len(inst.get('options', []))
        if n_opts > 0:
            random_scores.append(1.0 / n_opts)
    random_baseline = np.mean(random_scores) if random_scores else 0.0
    
    # Majority baseline: for each question, accuracy if always predicting majority
    # Group by question (not by instance!)
    from collections import defaultdict, Counter
    question_answers = defaultdict(list)
    
    for inst in instances:
        qid = extract_question_id(inst)
        gt = inst.get('ground_truth')
        if gt:
            question_answers[qid].append(gt)
    
    # Debug: show question grouping stats
    n_questions = len(question_answers)
    responses_per_q = [len(v) for v in question_answers.values()]
    print(f"    Questions found: {n_questions}, avg responses/question: {np.mean(responses_per_q):.1f}")
    
    majority_scores = []
    for qid, answers in question_answers.items():
        if len(answers) >= 2:  # Need multiple responses to compute majority
            most_common_count = Counter(answers).most_common(1)[0][1]
            majority_scores.append(most_common_count / len(answers))
    
    majority_baseline = np.mean(majority_scores) if majority_scores else 0.0
    
    return random_baseline, majority_baseline


def compute_macro_f1(instances: List[dict]) -> float:
    """Compute macro F1 score."""
    from collections import defaultdict
    
    # Group by ground truth class
    class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    for inst in instances:
        gt = inst.get('ground_truth')
        pred = inst.get('predicted')
        
        if gt == pred:
            class_stats[gt]['tp'] += 1
        else:
            class_stats[gt]['fn'] += 1
            class_stats[pred]['fp'] += 1
    
    # Compute F1 per class
    f1_scores = []
    for cls, stats in class_stats.items():
        precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
        recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    return np.mean(f1_scores) if f1_scores else 0.0


def compute_heterogeneity(instances: List[dict]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute median variance ratio and JS divergence (soft predictions)."""
    from collections import defaultdict
    
    # Group by question_id (properly extracted)
    question_data = defaultdict(lambda: {'empirical': [], 'predicted_probs': []})
    
    for inst in instances:
        qid = extract_question_id(inst)
        gt = inst.get('ground_truth')
        
        # Get probabilities - try different field names
        probs = inst.get('probabilities') or inst.get('probs') or {}
        
        # If no probabilities, try to compute from logprobs
        if not probs and 'option_logprobs' in inst:
            logprobs = inst['option_logprobs']
            if logprobs:
                # Convert logprobs to probs
                max_lp = max(logprobs.values())
                probs = {k: np.exp(v - max_lp) for k, v in logprobs.items()}
                total = sum(probs.values())
                probs = {k: v / total for k, v in probs.items()}
        
        if qid and gt and probs:
            question_data[qid]['empirical'].append(gt)
            question_data[qid]['predicted_probs'].append(probs)
    
    # Compute variance ratio and JS divergence for each question
    variance_ratios = []
    js_divergences = []
    
    for qid, data in question_data.items():
        if len(data['empirical']) < 10:  # Need enough samples
            continue
        
        # Empirical distribution
        from collections import Counter
        emp_counts = Counter(data['empirical'])
        total = sum(emp_counts.values())
        options = list(emp_counts.keys())
        
        if len(options) < 2:
            continue
        
        emp_probs = np.array([emp_counts.get(opt, 0) / total for opt in options])
        emp_var = np.var(emp_probs)
        
        if emp_var < 1e-10:
            continue
        
        # Predicted distribution (average probabilities)
        pred_probs_list = []
        for prob_dict in data['predicted_probs']:
            probs = [prob_dict.get(opt, 0) for opt in options]
            pred_probs_list.append(probs)
        
        avg_pred_probs = np.mean(pred_probs_list, axis=0)
        pred_var = np.var(avg_pred_probs)
        
        vr = pred_var / emp_var
        variance_ratios.append(vr)
        
        # Compute JS divergence
        # Normalize to ensure they sum to 1
        emp_probs_norm = emp_probs / emp_probs.sum() if emp_probs.sum() > 0 else emp_probs
        pred_probs_norm = avg_pred_probs / avg_pred_probs.sum() if avg_pred_probs.sum() > 0 else avg_pred_probs
        # Handle zeros
        eps = 1e-10
        emp_probs_norm = np.clip(emp_probs_norm, eps, 1.0)
        pred_probs_norm = np.clip(pred_probs_norm, eps, 1.0)
        emp_probs_norm = emp_probs_norm / emp_probs_norm.sum()
        pred_probs_norm = pred_probs_norm / pred_probs_norm.sum()
        # scipy returns sqrt(JSD), so square it
        js = jensenshannon(emp_probs_norm, pred_probs_norm, base=2) ** 2
        js_divergences.append(js)
    
    if not variance_ratios:
        return None, None, None
    
    median_vr = np.median(variance_ratios)
    flattening_rate = sum(1 for vr in variance_ratios if vr < 1) / len(variance_ratios)
    median_js = np.median(js_divergences) if js_divergences else None
    
    return median_vr, flattening_rate, median_js


def load_model_results(model_folder: Path, model_name: str, 
                       profile_filter: Optional[str] = 's6m4',
                       skip_heterogeneity: bool = False) -> Optional[ModelResults]:
    """Load and compute all metrics for a single model."""
    
    if model_name not in MODEL_METADATA:
        print(f"  Warning: Unknown model '{model_name}', skipping")
        return None
    
    info = MODEL_METADATA[model_name]
    
    # Load all instances (filtered by profile)
    instances = load_jsonl_files(model_folder, profile_filter=profile_filter)
    
    if not instances:
        print(f"  Warning: No instances found in {model_folder}")
        return None
    
    # Compute metrics
    accuracy, n_instances = compute_accuracy(instances)
    random_baseline, majority_baseline = compute_baselines(instances)
    macro_f1 = compute_macro_f1(instances)
    
    # Heterogeneity (can be slow)
    median_vr, flattening_rate, median_js = None, None, None
    if not skip_heterogeneity:
        print(f"    Computing heterogeneity metrics...")
        median_vr, flattening_rate, median_js = compute_heterogeneity(instances)
    
    return ModelResults(
        model_name=model_name,
        info=info,
        n_instances=n_instances,
        accuracy=accuracy,
        random_baseline=random_baseline,
        majority_baseline=majority_baseline,
        macro_f1=macro_f1,
        median_vr_soft=median_vr,
        flattening_rate=flattening_rate,
        median_js_soft=median_js,
    )


def load_all_models(results_dir: Path, profile_filter: Optional[str] = 's6m4',
                    skip_heterogeneity: bool = False) -> List[ModelResults]:
    """Load results for all models in the results directory."""
    
    all_results = []
    
    # Find all model folders
    model_folders = [f for f in results_dir.iterdir() if f.is_dir()]
    
    print(f"Found {len(model_folders)} model folders")
    if profile_filter:
        print(f"Filtering to profile: {profile_filter}")
    if skip_heterogeneity:
        print("Skipping heterogeneity computation")
    print()
    
    for folder in sorted(model_folders):
        model_name = folder.name
        print(f"Loading {model_name}...")
        
        result = load_model_results(folder, model_name, 
                                    profile_filter=profile_filter,
                                    skip_heterogeneity=skip_heterogeneity)
        if result:
            all_results.append(result)
            print(f"  ✓ {result.n_instances:,} instances, {result.accuracy:.1%} accuracy")
        else:
            print(f"  ✗ Failed to load")
    
    return all_results


# =============================================================================
# PLOTTING
# =============================================================================

def create_figure(results: List[ModelResults], output_path: Optional[Path] = None,
                  profile_name: str = 'rich'):
    """Create the three-panel figure following ICML guidelines."""
    
    # Set matplotlib style for ICML compliance
    plt.rcParams.update({
        'font.size': 9,
        'axes.linewidth': 0.5,
        'grid.linewidth': 0.5,
        'lines.linewidth': 0.5,
        'patch.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
    })
    
    # Set up figure - three panels side by side
    # Adjust layout to make room for legend on the right
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor('white')
    # Leave room for legend on the right, moderate space between panels
    plt.subplots_adjust(wspace=0.3, right=0.85)
    
    # Get global baselines (should be same across models, take mean)
    random_baseline = np.mean([r.random_baseline for r in results])
    majority_baseline = np.mean([r.majority_baseline for r in results])
    
    # Compute dodge offsets for models at same size
    # Dodge factor in log space (multiply/divide size by this factor)
    DODGE_FACTOR = 1.08  # 8% offset in log space
    
    def get_dodged_x(size: float, model_type: str) -> float:
        """Apply horizontal dodge for base vs instruct at same size."""
        if model_type == 'base':
            return size / DODGE_FACTOR
        else:
            return size * DODGE_FACTOR
    
    # ===================
    # PANEL A: Accuracy
    # ===================
    
    # Plot reference lines first (behind points) - dark, thick lines
    ax1.axhline(y=majority_baseline, color='#000000', linestyle='--', 
                linewidth=0.7, label=f'Majority baseline ({majority_baseline:.1%})', zorder=1)
    ax1.axhline(y=random_baseline, color='#666666', linestyle='--', 
                linewidth=0.7, label=f'Random baseline ({random_baseline:.1%})', zorder=1)
    
    # Plot each model - use color for family, filled/hollow for base/instruct
    for r in results:
        style = FAMILY_STYLES.get(r.info.family, {'color': '#808080'})
        color = style['color']
        
        # Get dodged x position
        x_pos = get_dodged_x(r.info.size_b, r.info.model_type)
        
        # Marker style: filled circle for instruct, hollow circle for base
        # All use same marker shape (circle), only color and fill differ
        if r.info.model_type == 'base':
            facecolor = 'white'
            edgecolor = color
            edgewidth = 0.7
        else:
            facecolor = color
            edgecolor = color
            edgewidth = 0.5
        
        ax1.scatter(x_pos, r.accuracy, 
                   marker='o', s=70, 
                   facecolors=facecolor, edgecolors=edgecolor, 
                   linewidths=edgewidth, zorder=3)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Model Size (Billion Parameters)', fontsize=9)
    ax1.set_ylabel('Accuracy', fontsize=9)
    ax1.set_xlim(3, 800)
    ax1.set_ylim(0.15, 0.55)
    
    # Format y-axis as percentage
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Custom x-ticks
    ax1.set_xticks([4, 8, 32, 70, 120, 685])
    ax1.set_xticklabels(['4B', '8B', '32B', '70B', '120B', '685B'])
    
    # Baseline legend in upper right (consistent with VR panel)
    baseline_legend = ax1.legend(loc='upper right', fontsize=8, frameon=True, fancybox=False, 
                                  edgecolor='black', framealpha=1.0, handlelength=1.5)
    ax1.set_facecolor('white')
    ax1.grid(False)  # No grid per ICML guidelines (cleaner look)
    
    # ===================
    # PANEL B: Variance Ratio
    # ===================
    
    # Reference line at VR = 1 (no flattening) - dark, thick line
    ax2.axhline(y=1.0, color='#000000', linestyle='--', 
                linewidth=0.7, label='No flattening (VR = 1)', zorder=1)
    
    # Subtle shading for flattening zone (below VR=1) - very light
    ax2.axhspan(0, 1.0, alpha=0.05, color='gray', zorder=0)
    
    # Plot each model (only those with VR data)
    for r in results:
        if r.median_vr_soft is None:
            continue
            
        style = FAMILY_STYLES.get(r.info.family, {'color': '#808080'})
        color = style['color']
        
        # Get dodged x position
        x_pos = get_dodged_x(r.info.size_b, r.info.model_type)
        
        if r.info.model_type == 'base':
            facecolor = 'white'
            edgecolor = color
            edgewidth = 0.7
        else:
            facecolor = color
            edgecolor = color
            edgewidth = 0.5
        
        ax2.scatter(x_pos, r.median_vr_soft, 
                   marker='o', s=70,
                   facecolors=facecolor, edgecolors=edgecolor, 
                   linewidths=edgewidth, zorder=3)
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Model Size (Billion Parameters)', fontsize=9)
    ax2.set_ylabel('Median Variance Ratio', fontsize=9)
    ax2.set_xlim(3, 800)
    ax2.set_ylim(0.3, 1.2)
    
    # Custom x-ticks
    ax2.set_xticks([4, 8, 32, 70, 120, 685])
    ax2.set_xticklabels(['4B', '8B', '32B', '70B', '120B', '685B'])
    
    # Reference line legend in upper right
    ax2.legend(loc='upper right', fontsize=8, frameon=True, fancybox=False,
               edgecolor='black', framealpha=1.0, handlelength=1.5)
    ax2.set_facecolor('white')
    ax2.grid(False)  # No grid per ICML guidelines
    
    # ===================
    # PANEL C: JS Divergence
    # ===================
    
    # Plot each model (only those with JS divergence data)
    for r in results:
        if r.median_js_soft is None:
            continue
            
        style = FAMILY_STYLES.get(r.info.family, {'color': '#808080'})
        color = style['color']
        
        # Get dodged x position
        x_pos = get_dodged_x(r.info.size_b, r.info.model_type)
        
        if r.info.model_type == 'base':
            facecolor = 'white'
            edgecolor = color
            edgewidth = 0.7
        else:
            facecolor = color
            edgecolor = color
            edgewidth = 0.5
        
        ax3.scatter(x_pos, r.median_js_soft, 
                   marker='o', s=70,
                   facecolors=facecolor, edgecolors=edgecolor, 
                   linewidths=edgewidth, zorder=3)
    
    ax3.set_xscale('log')
    ax3.set_xlabel('Model Size (Billion Parameters)', fontsize=9)
    ax3.set_ylabel('Median JS Divergence', fontsize=9)
    ax3.set_xlim(3, 800)
    # JS divergence range is [0, 1], but typically much smaller
    # Auto-scale based on data
    js_values = [r.median_js_soft for r in results if r.median_js_soft is not None]
    if js_values:
        js_min, js_max = min(js_values), max(js_values)
        js_range = js_max - js_min
        ax3.set_ylim(max(0, js_min - 0.1 * js_range), min(1.0, js_max + 0.1 * js_range))
    else:
        ax3.set_ylim(0, 0.1)
    
    # Custom x-ticks
    ax3.set_xticks([4, 8, 32, 70, 120, 685])
    ax3.set_xticklabels(['4B', '8B', '32B', '70B', '120B', '685B'])
    
    ax3.set_facecolor('white')
    ax3.grid(False)  # No grid per ICML guidelines
    
    # ===================
    # MODEL LEGEND (inside panel A, upper right)
    # ===================
    
    # Create legend for specific models (color encoding)
    # Base/Instruct distinction (filled vs hollow) will be explained in caption
    model_legend_elements = []
    
    # Get unique model display names (ignore base/instruct distinction for legend)
    # Group by display_name to avoid duplicates
    unique_models = {}
    for r in results:
        model_name = r.info.display_name
        if model_name not in unique_models:
            unique_models[model_name] = r.info.family
    
    # Sort by family then by model name for consistent ordering
    sorted_models = sorted(unique_models.items(), key=lambda x: (x[1], x[0]))
    
    for model_name, family in sorted_models:
        style = FAMILY_STYLES.get(family, {'color': '#808080'})
        # Use filled circle to show color (matches data point style)
        marker = plt.Line2D([0], [0], marker='o', color='black',
                           markerfacecolor=style['color'], markeredgecolor=style['color'],
                           markersize=5.5, linestyle='None', 
                           markeredgewidth=0.5, label=model_name)
        model_legend_elements.append(marker)
    
    # Add model legend on the right side of the figure (outside plot area)
    # Place legend on the right side of the figure, similar to profile richness figure
    fig.legend(
        handles=model_legend_elements,
        loc='center right',
        bbox_to_anchor=(0.92, 0.5),
        fontsize=8.5,
        frameon=True,
        fancybox=False,
        edgecolor='black',
        framealpha=1.0,
        handlelength=1.5,
        columnspacing=0.5,
        ncol=1
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10, right=0.85, wspace=0.3)  # Less space needed without bottom legend, keep room for right legend
    
    # Save or show
    if output_path:
        # Save in multiple formats
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
                    edgecolor='none', pad_inches=0.1)
        print(f"\n✓ Saved figure to {output_path}")
        
        # Also save PDF for paper
        pdf_path = output_path.with_suffix('.pdf')
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', 
                    edgecolor='none', pad_inches=0.1)
        print(f"✓ Saved PDF to {pdf_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary_table(results: List[ModelResults]):
    """Print a summary table of all results."""
    
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    # Sort by accuracy descending
    results_sorted = sorted(results, key=lambda r: r.accuracy, reverse=True)
    
    # Get baselines
    random_bl = np.mean([r.random_baseline for r in results])
    majority_bl = np.mean([r.majority_baseline for r in results])
    
    print(f"\nBaselines: Random = {random_bl:.1%}, Majority = {majority_bl:.1%}")
    print(f"N ≈ {results[0].n_instances:,} per model\n")
    
    print(f"{'Model':<25} {'Size':>6} {'Type':<8} {'Acc':>7} {'vs Maj':>8} {'VR':>6} {'JS':>6}")
    print("-" * 78)
    
    for r in results_sorted:
        vs_maj = r.accuracy - majority_bl
        vr_str = f"{r.median_vr_soft:.2f}" if r.median_vr_soft else "N/A"
        js_str = f"{r.median_js_soft:.3f}" if r.median_js_soft else "N/A"
        print(f"{r.info.display_name:<25} {r.info.size_b:>5.0f}B {r.info.model_type:<8} "
              f"{r.accuracy:>6.1%} {vs_maj:>+7.1%} {vr_str:>6} {js_str:>6}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate ICML figure for LLM survey prediction results')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Path to results directory containing model folders')
    parser.add_argument('--output', type=str, default='figure1.png',
                       help='Output filename (default: figure1.png)')
    parser.add_argument('--profile', type=str, default='s6m4',
                       choices=['s3m2', 's4m3', 's6m4', 'all'],
                       help='Profile type to include: s3m2 (sparse), s4m3 (medium), s6m4 (rich), or all (default: s6m4)')
    parser.add_argument('--skip-heterogeneity', action='store_true',
                       help='Skip variance ratio computation (faster)')
    parser.add_argument('--no-show', action='store_true',
                       help="Don't display the figure, just save it")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    # Profile filter
    profile_filter = None if args.profile == 'all' else args.profile
    profile_name = {'s3m2': 'sparse', 's4m3': 'medium', 's6m4': 'rich', 'all': 'all'}.get(args.profile, args.profile)
    
    # Load all model results
    print("Loading model results...")
    print("=" * 60)
    results = load_all_models(results_dir, profile_filter=profile_filter,
                              skip_heterogeneity=args.skip_heterogeneity)
    
    if not results:
        print("Error: No results loaded!")
        return
    
    print(f"\n{'=' * 60}")
    print(f"Loaded {len(results)} models ({profile_name} profile only)")
    print("=" * 60)
    
    # Print summary
    print_summary_table(results)
    
    # Create figure
    print("\nGenerating figure...")
    output_path = Path(args.output)
    create_figure(results, output_path, profile_name=profile_name)


if __name__ == '__main__':
    main()