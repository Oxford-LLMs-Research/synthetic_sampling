#!/usr/bin/env python3
"""
Analyze profile richness effect across models.

Computes metrics by profile level (sparse/medium/rich) for each model:
- Accuracy
- Variance ratio (heterogeneity preservation)
- Brier score (calibration)
- Expected Calibration Error (ECE)

Generates visualizations following ICML guidelines:
- Dark lines (≥0.5pt)
- No gray backgrounds
- Proper axis labels and legends
- No titles inside figures

Usage:
    python analyze_profile_richness.py --results-dir results/ --output analysis/profile_richness/
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add src to path
_script_dir = Path(__file__).resolve().parent
_src_dir = _script_dir.parent / "src"
sys.path.insert(0, str(_src_dir))

from synthetic_sampling.evaluation import (
    load_results,
    ResultsAnalyzer,
    ParsedInstance,
)
from shared_data_cache import get_all_models_enriched


# =============================================================================
# MODEL METADATA
# =============================================================================

@dataclass
class ModelInfo:
    """Metadata for a model."""
    display_name: str
    family: str
    size_b: float  # Size in billions
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

# Profile type mapping
PROFILE_TYPES = {
    's3m2': {'name': 'Sparse', 'features': 6, 'order': 0},
    's4m3': {'name': 'Medium', 'features': 12, 'order': 1},
    's6m4': {'name': 'Rich', 'features': 24, 'order': 2},
}


# =============================================================================
# DATA LOADING AND ANALYSIS
# =============================================================================

@dataclass
class ProfileMetrics:
    """Metrics for a specific profile level."""
    profile_type: str
    n_instances: int
    accuracy: float
    macro_f1: Optional[float]
    variance_ratio_soft: Optional[float]  # Median (to match generate_main_results_figure.py)
    variance_ratio_hard: Optional[float]  # Median
    variance_ratio_soft_mean: Optional[float]  # Mean (also saved for comparison)
    variance_ratio_hard_mean: Optional[float]  # Mean
    js_divergence_soft: Optional[float]  # Median JS divergence
    js_divergence_hard: Optional[float]  # Median JS divergence
    js_divergence_soft_mean: Optional[float]  # Mean JS divergence
    js_divergence_hard_mean: Optional[float]  # Mean JS divergence
    brier_score: Optional[float]
    ece: Optional[float]  # Expected Calibration Error
    mean_log_loss: Optional[float]


def load_model_results(
    model_folder: Path,
    profile_filter: Optional[str] = None
) -> List[ParsedInstance]:
    """Load results from a model folder, optionally filtered by profile type."""
    instances = []
    for jsonl_file in sorted(model_folder.glob("*.jsonl")):
        try:
            batch = load_results(str(jsonl_file))
            for inst in batch:
                if profile_filter is None or inst.profile_type == profile_filter:
                    instances.append(inst)
        except Exception as e:
            print(f"    Warning: Could not load {jsonl_file.name}: {e}")
            continue
    return instances


def compute_profile_metrics(
    instances: List[ParsedInstance],
    profile_type: str
) -> Optional[ProfileMetrics]:
    """Compute metrics for a specific profile level."""
    if not instances:
        return None
    
    analyzer = ResultsAnalyzer(instances)
    overall = analyzer.overall_metrics()
    
    # Heterogeneity analysis
    # Use min_n=10 to match generate_main_results_figure.py
    hetero = analyzer.heterogeneity_analysis(min_n=10)
    # Use median to match generate_main_results_figure.py (which uses median)
    vr_soft = hetero.get('variance_ratio_soft', {}).get('median')
    vr_hard = hetero.get('variance_ratio_hard', {}).get('median')
    # Also save mean for comparison
    vr_soft_mean = hetero.get('variance_ratio_soft', {}).get('mean')
    vr_hard_mean = hetero.get('variance_ratio_hard', {}).get('mean')
    # JS divergence (median and mean)
    js_soft = hetero.get('js_divergence_soft', {}).get('median')
    js_hard = hetero.get('js_divergence_hard', {}).get('median')
    js_soft_mean = hetero.get('js_divergence_soft', {}).get('mean')
    js_hard_mean = hetero.get('js_divergence_hard', {}).get('mean')
    
    # Calibration
    cal = analyzer.calibration_curve(n_bins=10)
    ece = cal.get('ece')
    
    return ProfileMetrics(
        profile_type=profile_type,
        n_instances=len(instances),
        accuracy=overall.accuracy,
        macro_f1=overall.macro_f1,
        variance_ratio_soft=vr_soft,
        variance_ratio_hard=vr_hard,
        variance_ratio_soft_mean=vr_soft_mean,
        variance_ratio_hard_mean=vr_hard_mean,
        js_divergence_soft=js_soft,
        js_divergence_hard=js_hard,
        js_divergence_soft_mean=js_soft_mean,
        js_divergence_hard_mean=js_hard_mean,
        brier_score=overall.brier_score,
        ece=ece,
        mean_log_loss=overall.mean_log_loss,
    )


def analyze_all_models(
    results_dir: Path,
    model_whitelist: Optional[List[str]] = None,
    input_paths: Optional[List[Path]] = None,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
) -> Dict[str, Dict[str, ProfileMetrics]]:
    """
    Analyze all models and compute metrics by profile level.
    
    Parameters
    ----------
    results_dir : Path
        Directory containing model folders
    model_whitelist : Optional[List[str]]
        Restrict to these model names
    input_paths : Optional[List[Path]]
        Input data paths for enrichment (enables caching)
    cache_dir : Optional[Path]
        Cache directory for enriched instances
    use_cache : bool
        Use cache if available (requires input_paths)
    
    Returns
    -------
    Dict mapping model_name -> Dict mapping profile_type -> ProfileMetrics
    """
    # Check if directory exists
    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results directory not found: {results_dir}\n"
            f"Please provide a valid path to the results directory containing model folders."
        )
    
    if not results_dir.is_dir():
        raise ValueError(f"Path is not a directory: {results_dir}")
    
    results = {}
    
    # Use shared cache if input_paths provided
    if input_paths and cache_dir and use_cache:
        print("Using shared cache for enriched instances...")
        instances_by_model = get_all_models_enriched(
            results_dir=results_dir,
            input_paths=input_paths,
            cache_dir=cache_dir,
            profile_filter=None,
            model_whitelist=model_whitelist,
            force_reload=False,
            verbose=True,
        )
        
        for model_name, all_instances in instances_by_model.items():
            print(f"Analyzing {model_name}...")
            if not all_instances:
                print(f"  No instances found, skipping.")
                continue
    else:
        # Fall back to direct loading (no enrichment/cache)
        model_folders = [f for f in results_dir.iterdir() if f.is_dir()]
        if model_whitelist:
            model_folders = [f for f in model_folders if f.name in model_whitelist]
        model_folders = sorted(model_folders)
        
        if not model_folders:
            raise ValueError(
                f"No model folders found in {results_dir}\n"
                f"Expected structure: {results_dir}/<model_name>/*.jsonl"
            )
        
        instances_by_model = {}
        for folder in model_folders:
            model_name = folder.name
            print(f"Loading {model_name}...")
            all_instances = load_model_results(folder, profile_filter=None)
            if all_instances:
                instances_by_model[model_name] = all_instances
    
    # Process each model
    for model_name, all_instances in instances_by_model.items():
        print(f"Analyzing {model_name}...")
        
        if not all_instances:
            print(f"  No instances found, skipping.")
            continue
        
        # Group by profile type
        by_profile = defaultdict(list)
        for inst in all_instances:
            if inst.profile_type in PROFILE_TYPES:
                by_profile[inst.profile_type].append(inst)
        
        if not by_profile:
            print(f"  No valid profile types found, skipping.")
            continue
        
        # Compute metrics for each profile level
        profile_metrics = {}
        for profile_type in PROFILE_TYPES.keys():
            if profile_type in by_profile:
                metrics = compute_profile_metrics(by_profile[profile_type], profile_type)
                if metrics:
                    profile_metrics[profile_type] = metrics
                    vr_str = f"{metrics.variance_ratio_soft:.3f}" if metrics.variance_ratio_soft is not None else "N/A"
                    js_str = f"{metrics.js_divergence_soft:.4f}" if metrics.js_divergence_soft is not None else "N/A"
                    brier_str = f"{metrics.brier_score:.3f}" if metrics.brier_score is not None else "N/A"
                    f1_str = f"{metrics.macro_f1:.3f}" if metrics.macro_f1 is not None else "N/A"
                    print(f"  {PROFILE_TYPES[profile_type]['name']}: "
                          f"n={metrics.n_instances:,}, acc={metrics.accuracy:.1%}, "
                          f"F1={f1_str}, VR={vr_str}, JS={js_str}, Brier={brier_str}")
        
        if profile_metrics:
            results[model_name] = profile_metrics
    
    return results
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_model_legend_elements(
    sorted_models: List[str],
    results: Dict[str, Dict[str, ProfileMetrics]],
    family_colors: Dict[str, str]
) -> List:
    """
    Create legend elements for all models, grouped by display_name (one entry per model).
    Sorted alphabetically by display_name.
    
    Returns list of Line2D objects for matplotlib legend.
    """
    from matplotlib.lines import Line2D
    
    # Group by display_name (combine base and instruct versions)
    legend_dict = {}
    for model_name in sorted_models:
        info = MODEL_METADATA.get(model_name)
        if not info:
            continue
        # Check if this model has data
        if model_name not in results:
            continue
        
        # Use display_name as key (groups base and instruct together)
        display_name = info.display_name
        if display_name not in legend_dict:
            color = family_colors.get(info.family, '#000000')
            # Use solid line style (instruct style) for legend
            legend_dict[display_name] = (color, display_name)
    
    # Sort alphabetically by display_name
    sorted_legend = sorted(legend_dict.items(), key=lambda x: x[1][1])
    
    return [
        Line2D([0], [0], color=color, linestyle='-', 
              linewidth=0.5, label=label)
        for _, (color, label) in sorted_legend
    ]


def get_model_size(model_name: str) -> float:
    """Get model size in billions for sorting/plotting."""
    info = MODEL_METADATA.get(model_name)
    if info:
        return info.size_b
    # Try to extract from name
    import re
    match = re.search(r'(\d+(?:\.\d+)?)b', model_name.lower())
    if match:
        return float(match.group(1))
    return 1.0  # Default


def create_profile_richness_figure(
    results: Dict[str, Dict[str, ProfileMetrics]],
    output_path: Path,
    include_brier: bool = False
) -> None:
    """
    Create 2x2 or 3-panel figure showing profile richness effects.
    
    Panels:
    (a) Accuracy by profile level
    (b) Variance ratio by profile level
    (c) Optional: Brier score by profile level
    """
    # ICML style: dark lines, no gray background, proper labels
    plt.rcParams.update({
        'font.size': 9,
        'font.family': 'serif',
        'axes.linewidth': 0.5,
        'lines.linewidth': 0.5,
        'patch.linewidth': 0.5,
        'figure.dpi': 300,
    })
    
    # Sort models by size
    sorted_models = sorted(
        results.keys(),
        key=lambda m: (get_model_size(m), m)
    )
    
    # Profile order
    profile_order = ['s3m2', 's4m3', 's6m4']
    profile_names = [PROFILE_TYPES[p]['name'] for p in profile_order]
    profile_features = [PROFILE_TYPES[p]['features'] for p in profile_order]
    
    # Prepare data
    model_sizes = [get_model_size(m) for m in sorted_models]
    
    # Group by model family for color coding - match main figure colors
    # Use same colors as generate_main_results_figure.py for consistency
    family_colors = {
        'Llama': '#2E86AB',      # Blue (matches main figure)
        'OLMo': '#A23B72',       # Purple (matches main figure)
        'Qwen': '#F18F01',       # Orange (matches main figure)
        'GPT-OSS': '#C73E1D',    # Red (matches main figure)
        'DeepSeek': '#6A994E',   # Green (matches main figure)
        'Gemma': '#BC4749',      # Dark red (matches main figure)
    }
    
    # Get unique families
    families = set()
    for model_name in sorted_models:
        info = MODEL_METADATA.get(model_name)
        if info:
            families.add(info.family)
    
    # Create figure: (a) Accuracy, (b) Variance Ratio, (c) JS Divergence, (d) Optional Brier
    # Adjust layout to make room for legend on the right
    n_panels = 4 if include_brier else 3
    # Create figure with extra space on the right for legend
    # Use subplot_adjust to leave room for legend instead of tight_layout
    fig, axes = plt.subplots(1, n_panels, figsize=(16 if n_panels == 4 else 12, 3))
    if n_panels == 1:
        axes = [axes]
    # Adjust subplot parameters: add space between plots (wspace) and leave room for legend (right)
    # Match spacing from main results figure, more room for legend to avoid overlap
    plt.subplots_adjust(wspace=0.3, right=0.80)
    
    # Create legend elements once (shared across all panels)
    legend_elements = create_model_legend_elements(sorted_models, results, family_colors)
    legend_handles = [elem for elem in legend_elements]
    legend_labels = [elem.get_label() for elem in legend_elements]
    
    # Panel (a): Accuracy by profile level
    ax = axes[0]
    for model_name in sorted_models:
        info = MODEL_METADATA.get(model_name)
        if not info:
            continue
        
        model_results = results[model_name]
        accuracies = []
        sizes = []
        
        for profile_type in profile_order:
            if profile_type in model_results:
                metrics = model_results[profile_type]
                accuracies.append(metrics.accuracy)
                sizes.append(get_model_size(model_name))
            else:
                accuracies.append(np.nan)
                sizes.append(get_model_size(model_name))
        
        color = family_colors.get(info.family, '#000000')
        # Use different line styles for base vs instruct
        linestyle = '--' if info.model_type == 'base' else '-'
        
        # Plot lines connecting profile levels
        valid_mask = ~np.isnan(accuracies)
        if valid_mask.sum() > 0:
            ax.plot(
                [profile_features[i] for i in range(len(profile_order)) if valid_mask[i]],
                [accuracies[i] for i in range(len(profile_order)) if valid_mask[i]],
                color=color,
                linestyle=linestyle,
                linewidth=0.5,
                alpha=0.7,
                marker='o',
                markersize=2,
            )
    
    ax.set_xlabel('Number of Profile Features')
    ax.set_ylabel('Accuracy')
    ax.set_xticks(profile_features)
    ax.set_xticklabels(profile_features)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_ylim(bottom=0.15)  # Start at 15% to reduce empty space
    # Format y-axis ticks to show 2 decimal places
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
    
    # Panel (b): Variance ratio by profile level
    ax = axes[1]
    for model_name in sorted_models:
        info = MODEL_METADATA.get(model_name)
        if not info:
            continue
        
        model_results = results[model_name]
        variance_ratios = []
        
        for profile_type in profile_order:
            if profile_type in model_results:
                metrics = model_results[profile_type]
                variance_ratios.append(metrics.variance_ratio_soft)
            else:
                variance_ratios.append(np.nan)
        
        color = family_colors.get(info.family, '#000000')
        linestyle = '--' if info.model_type == 'base' else '-'
        
        valid_mask = ~np.isnan(variance_ratios)
        if valid_mask.sum() > 0:
            ax.plot(
                [profile_features[i] for i in range(len(profile_order)) if valid_mask[i]],
                [variance_ratios[i] for i in range(len(profile_order)) if valid_mask[i]],
                color=color,
                linestyle=linestyle,
                linewidth=0.5,
                alpha=0.7,
                marker='o',
                markersize=2,
            )
    
    ax.set_xlabel('Number of Profile Features')
    ax.set_ylabel('Variance Ratio (Soft)')
    ax.set_xticks(profile_features)
    ax.set_xticklabels(profile_features)
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_ylim(bottom=0)
    
    # Panel (c): JS Divergence by profile level
    ax = axes[2]
    for model_name in sorted_models:
        info = MODEL_METADATA.get(model_name)
        if not info:
            continue
        
        model_results = results[model_name]
        js_divergences = []
        
        for profile_type in profile_order:
            if profile_type in model_results:
                metrics = model_results[profile_type]
                js_divergences.append(metrics.js_divergence_soft)
            else:
                js_divergences.append(np.nan)
        
        color = family_colors.get(info.family, '#000000')
        linestyle = '--' if info.model_type == 'base' else '-'
        
        valid_mask = ~np.isnan(js_divergences)
        if valid_mask.sum() > 0:
            ax.plot(
                [profile_features[i] for i in range(len(profile_order)) if valid_mask[i]],
                [js_divergences[i] for i in range(len(profile_order)) if valid_mask[i]],
                color=color,
                linestyle=linestyle,
                linewidth=0.5,
                alpha=0.7,
                marker='o',
                markersize=2,
            )
    
    ax.set_xlabel('Number of Profile Features')
    ax.set_ylabel('JS Divergence (Soft)')
    ax.set_xticks(profile_features)
    ax.set_xticklabels(profile_features)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_ylim(bottom=0)
    
    # Panel (d): Brier score by profile level (optional)
    if include_brier and n_panels >= 4:
        ax = axes[3]
        for model_name in sorted_models:
            info = MODEL_METADATA.get(model_name)
            if not info:
                continue
            
            model_results = results[model_name]
            brier_scores = []
            
            for profile_type in profile_order:
                if profile_type in model_results:
                    metrics = model_results[profile_type]
                    brier_scores.append(metrics.brier_score)
                else:
                    brier_scores.append(np.nan)
            
            color = family_colors.get(info.family, '#000000')
            linestyle = '--' if info.model_type == 'base' else '-'
            
            valid_mask = ~np.isnan(brier_scores)
            if valid_mask.sum() > 0:
                ax.plot(
                    [profile_features[i] for i in range(len(profile_order)) if valid_mask[i]],
                    [brier_scores[i] for i in range(len(profile_order)) if valid_mask[i]],
                    color=color,
                    linestyle=linestyle,
                    linewidth=0.5,
                    alpha=0.7,
                    marker='o',
                    markersize=2,
                )
        
        ax.set_xlabel('Number of Profile Features')
        ax.set_ylabel('Brier Score')
        ax.set_xticks(profile_features)
        ax.set_xticklabels(profile_features)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        # Lower Brier score is better, so invert y-axis might be helpful
        # But we'll keep it normal for consistency
    
    # Add single shared legend on the right side (outside the plot area)
    if legend_handles:
        # Place legend on the right side of the figure, closer to plots
        fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc='center right',
            bbox_to_anchor=(0.94, 0.5),
            fontsize=8.5,
            frameon=True,
            fancybox=False,
            edgecolor='black',
            framealpha=1.0,
            ncol=1,
            columnspacing=0.5,
            handlelength=1.5
        )
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.80, wspace=0.3)  # Maintain spacing after tight_layout, more room for legend
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to {output_path}")
    
    # Also save PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved PDF to {pdf_path}")
    
    plt.close()


def create_model_size_figure(
    results: Dict[str, Dict[str, ProfileMetrics]],
    output_path: Path
) -> None:
    """
    Create alternative figure with model size on x-axis, separate lines for sparse/medium/rich.
    """
    plt.rcParams.update({
        'font.size': 9,
        'font.family': 'serif',
        'axes.linewidth': 0.5,
        'lines.linewidth': 0.5,
        'patch.linewidth': 0.5,
        'figure.dpi': 300,
    })
    
    # Sort models by size
    sorted_models = sorted(
        results.keys(),
        key=lambda m: (get_model_size(m), m)
    )
    
    model_sizes = [get_model_size(m) for m in sorted_models]
    
    # Profile order
    profile_order = ['s3m2', 's4m3', 's6m4']
    profile_styles = {
        's3m2': {'label': 'Sparse (6 feat)', 'linestyle': '-', 'color': '#1f77b4'},
        's4m3': {'label': 'Medium (12 feat)', 'linestyle': '--', 'color': '#ff7f0e'},
        's6m4': {'label': 'Rich (24 feat)', 'linestyle': ':', 'color': '#2ca02c'},
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 3))
    
    # Panel (a): Accuracy
    ax = axes[0]
    for profile_type in profile_order:
        accuracies = []
        sizes = []
        
        for model_name in sorted_models:
            model_results = results.get(model_name, {})
            if profile_type in model_results:
                metrics = model_results[profile_type]
                accuracies.append(metrics.accuracy)
                sizes.append(get_model_size(model_name))
        
        if accuracies:
            style = profile_styles[profile_type]
            ax.plot(
                sizes,
                accuracies,
                label=style['label'],
                linestyle=style['linestyle'],
                color=style['color'],
                linewidth=0.5,
                marker='o',
                markersize=2,
            )
    
    ax.set_xlabel('Model Size (Billions of Parameters)')
    ax.set_ylabel('Accuracy')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc='best', fontsize=7, frameon=True)
    ax.set_ylim(bottom=0)
    
    # Panel (b): Variance ratio
    ax = axes[1]
    for profile_type in profile_order:
        variance_ratios = []
        sizes = []
        
        for model_name in sorted_models:
            model_results = results.get(model_name, {})
            if profile_type in model_results:
                metrics = model_results[profile_type]
                if metrics.variance_ratio_soft is not None:
                    variance_ratios.append(metrics.variance_ratio_soft)
                    sizes.append(get_model_size(model_name))
        
        if variance_ratios:
            style = profile_styles[profile_type]
            ax.plot(
                sizes,
                variance_ratios,
                label=style['label'],
                linestyle=style['linestyle'],
                color=style['color'],
                linewidth=0.5,
                marker='o',
                markersize=2,
            )
    
    ax.set_xlabel('Model Size (Billions of Parameters)')
    ax.set_ylabel('Variance Ratio (Soft)')
    ax.set_xscale('log')
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc='best', fontsize=7, frameon=True)
    ax.set_ylim(bottom=0)
    
    # Panel (c): JS Divergence
    ax = axes[2]
    for profile_type in profile_order:
        js_divergences = []
        sizes = []
        
        for model_name in sorted_models:
            model_results = results.get(model_name, {})
            if profile_type in model_results:
                metrics = model_results[profile_type]
                if metrics.js_divergence_soft is not None:
                    js_divergences.append(metrics.js_divergence_soft)
                    sizes.append(get_model_size(model_name))
        
        if js_divergences:
            style = profile_styles[profile_type]
            ax.plot(
                sizes,
                js_divergences,
                label=style['label'],
                linestyle=style['linestyle'],
                color=style['color'],
                linewidth=0.5,
                marker='o',
                markersize=2,
            )
    
    ax.set_xlabel('Model Size (Billions of Parameters)')
    ax.set_ylabel('JS Divergence (Soft)')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc='best', fontsize=7, frameon=True)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to {output_path}")
    
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved PDF to {pdf_path}")
    
    plt.close()


# =============================================================================
# DATA LOADING/SAVING
# =============================================================================

def load_results_from_json(json_path: Path) -> Dict[str, Dict[str, ProfileMetrics]]:
    """
    Load previously computed results from JSON file.
    
    Returns:
        Dict mapping model_name -> Dict mapping profile_type -> ProfileMetrics
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Results file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = {}
    for model_name, profile_data in data.items():
        profile_metrics = {}
        for profile_type, metrics_dict in profile_data.items():
            profile_metrics[profile_type] = ProfileMetrics(
                profile_type=metrics_dict['profile_type'],
                n_instances=metrics_dict['n_instances'],
                accuracy=metrics_dict['accuracy'],
                macro_f1=metrics_dict.get('macro_f1'),
                variance_ratio_soft=metrics_dict.get('variance_ratio_soft'),
                variance_ratio_hard=metrics_dict.get('variance_ratio_hard'),
                variance_ratio_soft_mean=metrics_dict.get('variance_ratio_soft_mean'),
                variance_ratio_hard_mean=metrics_dict.get('variance_ratio_hard_mean'),
                js_divergence_soft=metrics_dict.get('js_divergence_soft'),
                js_divergence_hard=metrics_dict.get('js_divergence_hard'),
                js_divergence_soft_mean=metrics_dict.get('js_divergence_soft_mean'),
                js_divergence_hard_mean=metrics_dict.get('js_divergence_hard_mean'),
                brier_score=metrics_dict.get('brier_score'),
                ece=metrics_dict.get('ece'),
                mean_log_loss=metrics_dict.get('mean_log_loss'),
            )
        results[model_name] = profile_metrics
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze profile richness effect across models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=None,
        help='Results root directory (containing model folders with JSONL files). Required unless using --load-results'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('analysis/profile_richness'),
        help='Output directory for analysis results and figures'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='*',
        help='Restrict to these model folder names (default: all)'
    )
    parser.add_argument(
        '--no-brier',
        action='store_true',
        help='Exclude Brier score panel from figure'
    )
    parser.add_argument(
        '--alt-layout',
        action='store_true',
        help='Use alternative layout with model size on x-axis'
    )
    parser.add_argument(
        '--load-results',
        type=Path,
        default=None,
        help='Load previously computed results from JSON file (skips calculation)'
    )
    parser.add_argument(
        '--skip-figures',
        action='store_true',
        help='Skip figure generation (only compute and save results)'
    )
    parser.add_argument(
        '--skip-summary',
        action='store_true',
        help='Skip summary analysis printing'
    )
    parser.add_argument(
        '--inputs',
        type=Path,
        default=None,
        help='Input data directory (main_data) for enrichment. If provided, enables caching.'
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=None,
        help='Cache directory for enriched instances (default: analysis/.cache/enriched - shared with other scripts)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable cache (force reload)'
    )
    
    args = parser.parse_args()
    
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'profile_richness_results.json'
    
    # Load or compute results
    if args.load_results:
        # Load from specified file
        print(f"Loading results from {args.load_results}...")
        try:
            results = load_results_from_json(args.load_results)
            print(f"Loaded results for {len(results)} models")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # Compute results from data
        # Resolve and validate results directory
        # If path is relative, resolve it relative to current working directory
        if args.results_dir.is_absolute():
            results_dir = args.results_dir
        else:
            results_dir = Path.cwd() / args.results_dir
        results_dir = results_dir.resolve()
        
        # Analyze all models
        print("Analyzing profile richness effects...")
        print(f"Results directory: {results_dir}")
        if args.inputs:
            print(f"Input data: {args.inputs} (enables caching)")
        print("=" * 80)
        
        # Set up cache
        # Use shared cache directory so all analysis scripts can reuse the same cache
        input_paths = [args.inputs] if args.inputs and args.inputs.exists() else None
        if args.cache_dir:
            cache_dir = args.cache_dir
        elif args.inputs:
            # Default to shared cache location (analysis/.cache/enriched)
            # This allows profile_richness, profile_richness_by_topic, and disaggregated to share cache
            shared_cache = Path('analysis/.cache/enriched')
            cache_dir = shared_cache
        else:
            cache_dir = None
        
        try:
            results = analyze_all_models(
                results_dir,
                model_whitelist=args.models,
                input_paths=input_paths,
                cache_dir=cache_dir,
                use_cache=not args.no_cache,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"\nError: {e}")
            sys.exit(1)
        
        if not results:
            print("\nError: No results found. Check that --results-dir contains model folders.")
            sys.exit(1)
        
        # Save results to JSON
        # Convert to serializable format
        serializable_results = {}
        for model_name, profile_metrics in results.items():
            serializable_results[model_name] = {}
            for profile_type, metrics in profile_metrics.items():
                serializable_results[model_name][profile_type] = {
                    'profile_type': metrics.profile_type,
                    'n_instances': metrics.n_instances,
                    'accuracy': metrics.accuracy,
                    'macro_f1': metrics.macro_f1,
                    'variance_ratio_soft': metrics.variance_ratio_soft,  # Median (matches main figure)
                    'variance_ratio_hard': metrics.variance_ratio_hard,  # Median
                    'variance_ratio_soft_mean': metrics.variance_ratio_soft_mean,  # Mean (for comparison)
                    'variance_ratio_hard_mean': metrics.variance_ratio_hard_mean,  # Mean
                    'js_divergence_soft': metrics.js_divergence_soft,  # Median JS divergence
                    'js_divergence_hard': metrics.js_divergence_hard,  # Median
                    'js_divergence_soft_mean': metrics.js_divergence_soft_mean,  # Mean (for comparison)
                    'js_divergence_hard_mean': metrics.js_divergence_hard_mean,  # Mean
                    'brier_score': metrics.brier_score,
                    'ece': metrics.ece,
                    'mean_log_loss': metrics.mean_log_loss,
                }
        
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nSaved results to {json_path}")
    
    # Generate figures
    if not args.skip_figures:
        print("\nGenerating figures...")
        
        if args.alt_layout:
            # Alternative layout: model size on x-axis
            figure_path = output_dir / 'figure_profile_richness_by_size.png'
            create_model_size_figure(results, figure_path)
        else:
            # Standard layout: profile features on x-axis
            figure_path = output_dir / 'figure_profile_richness.png'
            create_profile_richness_figure(
                results,
                figure_path,
                include_brier=False  # Default: exclude Brier score panel
            )
    else:
        print("\nSkipping figure generation (use without --skip-figures to generate)")
    
    # Print summary analysis
    if not args.skip_summary:
        print("\n" + "=" * 80)
        print("SUMMARY ANALYSIS")
        print("=" * 80)
        
        # Question 1: Does accuracy improve at all?
        print("\n1. Does accuracy improve with profile richness?")
        print("-" * 80)
    accuracy_improvements = []
    for model_name, profile_metrics in results.items():
        if 's3m2' in profile_metrics and 's6m4' in profile_metrics:
            sparse_acc = profile_metrics['s3m2'].accuracy
            rich_acc = profile_metrics['s6m4'].accuracy
            improvement = rich_acc - sparse_acc
            accuracy_improvements.append((model_name, improvement))
            direction = "improves" if improvement > 0 else "worsens"
            print(f"  {model_name:30s} {direction:8s} by {abs(improvement):.2%} "
                  f"(sparse: {sparse_acc:.1%} → rich: {rich_acc:.1%})")
    
    if accuracy_improvements:
        avg_improvement = np.mean([imp for _, imp in accuracy_improvements])
        n_improve = sum(1 for _, imp in accuracy_improvements if imp > 0)
        print(f"\n  Overall: {n_improve}/{len(accuracy_improvements)} models show improvement")
        print(f"  Average change: {avg_improvement:+.2%}")
    
        # Question 2: Do larger models benefit more?
        print("\n2. Do larger models benefit more from profile richness?")
        print("-" * 80)
        size_benefit = []
        for model_name, profile_metrics in results.items():
            if 's3m2' in profile_metrics and 's6m4' in profile_metrics:
                size = get_model_size(model_name)
                sparse_acc = profile_metrics['s3m2'].accuracy
                rich_acc = profile_metrics['s6m4'].accuracy
                improvement = rich_acc - sparse_acc
                size_benefit.append((size, improvement))
        
        if len(size_benefit) >= 3:
            # Split into small and large models
            sizes = [s for s, _ in size_benefit]
            median_size = np.median(sizes)
            small_models = [imp for s, imp in size_benefit if s <= median_size]
            large_models = [imp for s, imp in size_benefit if s > median_size]
            
            avg_small = np.mean(small_models) if small_models else 0
            avg_large = np.mean(large_models) if large_models else 0
            
            print(f"  Small models (≤{median_size:.0f}B): average improvement = {avg_small:+.2%}")
            print(f"  Large models (>{median_size:.0f}B): average improvement = {avg_large:+.2%}")
            if avg_large > avg_small:
                print(f"  → Larger models benefit more ({avg_large - avg_small:+.2%} difference)")
            else:
                print(f"  → No clear size effect")
        
        # Question 3: Does variance ratio improve?
        print("\n3. Does variance ratio improve with profile richness?")
        print("-" * 80)
        vr_improvements = []
        for model_name, profile_metrics in results.items():
            if 's3m2' in profile_metrics and 's6m4' in profile_metrics:
                sparse_vr = profile_metrics['s3m2'].variance_ratio_soft
                rich_vr = profile_metrics['s6m4'].variance_ratio_soft
                if sparse_vr is not None and rich_vr is not None:
                    improvement = rich_vr - sparse_vr
                    vr_improvements.append((model_name, improvement))
                    direction = "improves" if improvement > 0 else "worsens"
                    print(f"  {model_name:30s} {direction:8s} by {abs(improvement):.3f} "
                          f"(sparse: {sparse_vr:.3f} → rich: {rich_vr:.3f})")
        
        if vr_improvements:
            avg_improvement = np.mean([imp for _, imp in vr_improvements])
            n_improve = sum(1 for _, imp in vr_improvements if imp > 0)
            print(f"\n  Overall: {n_improve}/{len(vr_improvements)} models show improvement")
            print(f"  Average change: {avg_improvement:+.3f}")
        
        # Question 3b: Does JS divergence improve?
        print("\n3b. Does JS divergence improve with profile richness?")
        print("-" * 80)
        print("(Lower JS divergence = better match to empirical distribution)")
        js_improvements = []
        for model_name, profile_metrics in results.items():
            if 's3m2' in profile_metrics and 's6m4' in profile_metrics:
                sparse_js = profile_metrics['s3m2'].js_divergence_soft
                rich_js = profile_metrics['s6m4'].js_divergence_soft
                if sparse_js is not None and rich_js is not None:
                    # Lower JS is better, so improvement = sparse_js - rich_js
                    improvement = sparse_js - rich_js
                    js_improvements.append((model_name, improvement))
                    direction = "improves" if improvement > 0 else "worsens"
                    print(f"  {model_name:30s} {direction:8s} by {abs(improvement):.4f} "
                          f"(sparse: {sparse_js:.4f} → rich: {rich_js:.4f})")
        
        if js_improvements:
            avg_improvement = np.mean([imp for _, imp in js_improvements])
            n_improve = sum(1 for _, imp in js_improvements if imp > 0)
            print(f"\n  Overall: {n_improve}/{len(js_improvements)} models show improvement")
            print(f"  Average change: {avg_improvement:+.4f} (positive = better match)")
        
        # Question 4: Does calibration improve?
        print("\n4. Does calibration improve with profile richness?")
        print("-" * 80)
        cal_improvements = []
        for model_name, profile_metrics in results.items():
            if 's3m2' in profile_metrics and 's6m4' in profile_metrics:
                sparse_ece = profile_metrics['s3m2'].ece
                rich_ece = profile_metrics['s6m4'].ece
                if sparse_ece is not None and rich_ece is not None:
                    # Lower ECE is better, so improvement = sparse_ece - rich_ece
                    improvement = sparse_ece - rich_ece
                    cal_improvements.append((model_name, improvement))
                    direction = "improves" if improvement > 0 else "worsens"
                    print(f"  {model_name:30s} {direction:8s} by {abs(improvement):.4f} "
                          f"(sparse ECE: {sparse_ece:.4f} → rich ECE: {rich_ece:.4f})")
        
        if cal_improvements:
            avg_improvement = np.mean([imp for _, imp in cal_improvements])
            n_improve = sum(1 for _, imp in cal_improvements if imp > 0)
            print(f"\n  Overall: {n_improve}/{len(cal_improvements)} models show improvement")
            print(f"  Average change: {avg_improvement:+.4f} (positive = better calibration)")
    else:
        print("\nSkipping summary analysis (use without --skip-summary to print)")
    
    print("\nDone!")
    print(f"\nResults saved to: {json_path}")
    print(f"You can regenerate figures with:")
    print(f"  python scripts/analyze_profile_richness.py --load-results {json_path} --output {output_dir}")


if __name__ == '__main__':
    main()
