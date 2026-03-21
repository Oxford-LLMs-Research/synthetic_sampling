#!/usr/bin/env python3
"""
Create summary scatter plots for profile richness analysis.

Shows sparse (6 features) vs rich (24 features) accuracy for:
1. Overall (all data)
2. By section (each section separately)

Usage:
    python plot_profile_richness_summary.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# Model metadata for size and display names
MODEL_METADATA = {
    'deepseek-v3p1-terminus': {'size': 37, 'display': 'DeepSeek-V3', 'family': 'DeepSeek'},
    'gemma-3-27b-instruct': {'size': 27, 'display': 'Gemma 3 27B', 'family': 'Gemma'},
    'gpt_oss': {'size': 120, 'display': 'GPT-OSS 120B', 'family': 'GPT-OSS'},
    'llama3.1-70b-base': {'size': 70, 'display': 'Llama 3.1 70B Base', 'family': 'Llama'},
    'llama3.1-70b-instruct': {'size': 70, 'display': 'Llama 3.1 70B Instruct', 'family': 'Llama'},
    'llama3.1-8b-base': {'size': 8, 'display': 'Llama 3.1 8B Base', 'family': 'Llama'},
    'llama3.1-8b-instruct': {'size': 8, 'display': 'Llama 3.1 8B Instruct', 'family': 'Llama'},
    'olmo3-32b-base': {'size': 32, 'display': 'OLMo 3 32B Base', 'family': 'OLMo'},
    'olmo3-32b-dpo': {'size': 32, 'display': 'OLMo 3 32B Instruct', 'family': 'OLMo'},
    'olmo3-7b-base': {'size': 7, 'display': 'OLMo 3 7B Base', 'family': 'OLMo'},
    'olmo3-7b-dpo': {'size': 7, 'display': 'OLMo 3 7B Instruct', 'family': 'OLMo'},
    'qwen3-32b': {'size': 32, 'display': 'Qwen 3 32B', 'family': 'Qwen'},
    'qwen3-4b': {'size': 4, 'display': 'Qwen 3 4B', 'family': 'Qwen'},
}

# Key models to label on plots
KEY_MODELS = ["GPT-OSS 120B", "DeepSeek-V3", "OLMo 3 7B Base", "Qwen 3 32B"]

# ICML style
plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'axes.linewidth': 0.5,
    'lines.linewidth': 0.5,
    'patch.linewidth': 0.5,
    'figure.dpi': 300,
})


def load_profile_richness_data(json_path: Path) -> Dict:
    """Load profile richness results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_sparse_rich_data(data: Dict, by_section: bool = False, metric: str = 'accuracy') -> Dict:
    """
    Extract sparse (s3m2) and rich (s6m4) values for specified metric.
    
    Parameters
    ----------
    data : dict
        Profile richness data (overall or by section)
    by_section : bool
        If True, data is structured by section
    metric : str
        Metric to extract ('accuracy', 'js_divergence_soft', 'brier_score', 'macro_f1')
    
    Returns
    -------
    dict
        {model: {'sparse': float, 'rich': float, 'size': float, 'display': str}}
        or {section: {model: {...}}} if by_section
    """
    results = {}
    
    # Metric conversion factors (accuracy and F1 are percentages, others are raw)
    convert_to_pct = metric in ['accuracy', 'macro_f1']
    
    if by_section:
        # Data structure: {model: {section: {s3m2: {...}, s6m4: {...}}}}
        for model_name, sections in data.items():
            for section, profiles in sections.items():
                if section not in results:
                    results[section] = {}
                
                sparse_val = profiles.get('s3m2', {}).get(metric)
                rich_val = profiles.get('s6m4', {}).get(metric)
                
                if sparse_val is not None and rich_val is not None:
                    model_info = MODEL_METADATA.get(model_name, {'size': 1, 'display': model_name, 'family': 'Unknown'})
                    results[section][model_name] = {
                        'sparse': sparse_val * 100 if convert_to_pct else sparse_val,
                        'rich': rich_val * 100 if convert_to_pct else rich_val,
                        'size': model_info['size'],
                        'display': model_info['display'],
                        'family': model_info['family']
                    }
    else:
        # Data structure: {model: {s3m2: {...}, s6m4: {...}}}
        for model_name, profiles in data.items():
            sparse_val = profiles.get('s3m2', {}).get(metric)
            rich_val = profiles.get('s6m4', {}).get(metric)
            
            if sparse_val is not None and rich_val is not None:
                model_info = MODEL_METADATA.get(model_name, {'size': 1, 'display': model_name, 'family': 'Unknown'})
                results[model_name] = {
                    'sparse': sparse_val * 100 if convert_to_pct else sparse_val,
                    'rich': rich_val * 100 if convert_to_pct else rich_val,
                    'size': model_info['size'],
                    'display': model_info['display'],
                    'family': model_info['family']
                }
    
    return results


def create_scatter_plot(data: Dict, title: str, output_path: Path, 
                        xlabel: str = 'Accuracy with Sparse Profile (6 features)',
                        ylabel: str = 'Accuracy with Rich Profile (24 features)',
                        xlim: Tuple[float, float] = None,
                        ylim: Tuple[float, float] = None,
                        reference_lines: bool = True,
                        annotate_all: bool = False):
    """
    Create a scatter plot of sparse vs rich accuracy.
    
    Parameters
    ----------
    data : dict
        {model: {'sparse': float, 'rich': float, 'size': float, 'display': str}}
    title : str
        Plot title (None to omit, per ICML guidelines)
    output_path : Path
        Output file path
    xlabel, ylabel : str
        Axis labels
    xlim, ylim : tuple
        Axis limits (auto if None)
    reference_lines : bool
        Whether to add reference lines
    annotate_all : bool
        If True, annotate all models; if False, only key models
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    
    # Extract data
    models = list(data.keys())
    sparse = [data[m]['sparse'] for m in models]
    rich = [data[m]['rich'] for m in models]
    sizes = [data[m]['size'] for m in models]
    displays = [data[m]['display'] for m in models]
    
    # Size-based coloring (log scale)
    size_colors = np.log10(np.array(sizes))
    scatter = ax.scatter(sparse, rich, c=size_colors, s=80, cmap='viridis', 
                        edgecolors='black', linewidth=0.5, alpha=0.8)
    
    # Set axis limits first (needed for annotation positioning)
    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(min(sparse) - 1, max(sparse) + 1)
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(min(rich) - 1, max(rich) + 1)
    
    # Get axis limits for annotation bounds checking
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # Diagonal line (no improvement)
    diag_min = min(x_min, y_min)
    diag_max = max(x_max, y_max)
    ax.plot([diag_min, diag_max], [diag_min, diag_max], 'k--', 
           alpha=0.3, linewidth=0.5, label='No improvement')
    
    # Reference lines (mean sparse and rich if reference_lines is True)
    if reference_lines:
        mean_sparse = np.mean(sparse)
        mean_rich = np.mean(rich)
        ax.axhline(y=mean_rich, color='gray', linestyle=':', alpha=0.5, 
                  linewidth=0.5)
        ax.axvline(x=mean_sparse, color='gray', linestyle=':', alpha=0.5, 
                  linewidth=0.5)
    
    # Annotate models
    if annotate_all:
        # Annotate all models with compact labels
        for model_name, model_data in data.items():
            x, y = model_data['sparse'], model_data['rich']
            display = model_data['display']
            
            # Create compact label (remove "3.1", "3", shorten names)
            label = display.replace("3.1 ", "").replace("3 ", "").replace(" Base", " (b)").replace(" Instruct", " (i)")
            # Split into lines if too long
            if len(label) > 12:
                parts = label.split()
                if len(parts) > 2:
                    label = '\n'.join([' '.join(parts[:2]), ' '.join(parts[2:])])
            
            # Smart offset: try different positions to stay in bounds
            offsets = [
                (0.02 * x_range, 0.02 * y_range),  # Top-right
                (0.02 * x_range, -0.02 * y_range),  # Bottom-right
                (-0.02 * x_range, 0.02 * y_range),  # Top-left
                (-0.02 * x_range, -0.02 * y_range), # Bottom-left
                (0, 0.02 * y_range),  # Top
                (0, -0.02 * y_range), # Bottom
            ]
            
            # Find first offset that keeps annotation in bounds
            text_x, text_y = x, y
            for offset_x, offset_y in offsets:
                candidate_x = x + offset_x
                candidate_y = y + offset_y
                if x_min <= candidate_x <= x_max and y_min <= candidate_y <= y_max:
                    text_x, text_y = candidate_x, candidate_y
                    break
            
            ax.annotate(label, (x, y), 
                       xytext=(text_x, text_y), 
                       fontsize=6, alpha=0.7,
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                edgecolor='none', alpha=0.7))
    else:
        # Only annotate key models (for main plot)
        for model_name, model_data in data.items():
            display = model_data['display']
            if display in KEY_MODELS or any(key in display for key in KEY_MODELS):
                x, y = model_data['sparse'], model_data['rich']
                
                # Smart offset that stays in bounds
                x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                
                if 'GPT-OSS' in display:
                    offset = (0.02 * x_range, -0.02 * y_range)
                else:
                    offset = (0.02 * x_range, 0.02 * y_range)
                
                # Clean up display name for annotation
                label = display.replace("3 ", "").replace(" ", "\n")
                
                # Check bounds
                text_x = x + offset[0]
                text_y = y + offset[1]
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()
                
                # Clamp to bounds
                text_x = max(x_min, min(x_max, text_x))
                text_y = max(y_min, min(y_max, text_y))
                
                ax.annotate(label, (x, y), 
                           xytext=(text_x, text_y), 
                           fontsize=7, alpha=0.7,
                           ha='left', va='bottom')
    
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Model Size (log₁₀ B params)')
    # Set reasonable tick positions
    size_min, size_max = min(sizes), max(sizes)
    ticks = []
    tick_labels = []
    for size in [4, 8, 30, 70, 120]:
        if size_min <= size <= size_max:
            ticks.append(np.log10(size))
            tick_labels.append(f'{size}B')
    if not ticks:
        # Fallback: use min and max
        ticks = [np.log10(size_min), np.log10(size_max)]
        tick_labels = [f'{int(size_min)}B', f'{int(size_max)}B']
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(width=0.5, labelsize=8)
    
    if title:
        ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(output_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
    print(f"Saved scatter plot to {output_path}")
    plt.close(fig)


def create_multi_panel_section_plot(by_section_extracted: Dict, metric: str, output_path: Path):
    """
    Create a multi-panel figure with all sections as subplots.
    ICML style: no titles, subplot labels (A, B, C, ...)
    """
    sections = sorted(by_section_extracted.keys())
    n_sections = len(sections)
    
    # Determine grid layout (try to make it roughly square)
    n_cols = int(np.ceil(np.sqrt(n_sections)))
    n_rows = int(np.ceil(n_sections / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3))
    if n_sections == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Get global axis limits for consistency
    all_sparse = []
    all_rich = []
    for section_data in by_section_extracted.values():
        all_sparse.extend([d['sparse'] for d in section_data.values()])
        all_rich.extend([d['rich'] for d in section_data.values()])
    
    xlim = (min(all_sparse) - 1, max(all_sparse) + 1)
    ylim = (min(all_rich) - 1, max(all_rich) + 1)
    
    # Metric-specific labels
    metric_labels = {
        'accuracy': ('Accuracy with Sparse Profile (6 features)', 'Accuracy with Rich Profile (24 features)'),
        'js_divergence_soft': ('JS Divergence (Sparse)', 'JS Divergence (Rich)'),
        'brier_score': ('Brier Score (Sparse)', 'Brier Score (Rich)'),
        'macro_f1': ('F1 Score (Sparse)', 'F1 Score (Rich)'),
    }
    xlabel, ylabel = metric_labels.get(metric, ('Sparse', 'Rich'))
    
    for idx, section in enumerate(sections):
        ax = axes[idx]
        section_data = by_section_extracted[section]
        
        if not section_data:
            ax.axis('off')
            continue
        
        # Extract data
        models = list(section_data.keys())
        sparse = [section_data[m]['sparse'] for m in models]
        rich = [section_data[m]['rich'] for m in models]
        sizes = [section_data[m]['size'] for m in models]
        
        # Size-based coloring
        size_colors = np.log10(np.array(sizes))
        scatter = ax.scatter(sparse, rich, c=size_colors, s=60, cmap='viridis', 
                            edgecolors='black', linewidth=0.5, alpha=0.8)
        
        # Diagonal line
        diag_min = min(xlim[0], ylim[0])
        diag_max = max(xlim[1], ylim[1])
        ax.plot([diag_min, diag_max], [diag_min, diag_max], 'k--', 
               alpha=0.3, linewidth=0.5)
        
        # Reference lines
        mean_sparse = np.mean(sparse)
        mean_rich = np.mean(rich)
        ax.axhline(y=mean_rich, color='gray', linestyle=':', alpha=0.5, linewidth=0.5)
        ax.axvline(x=mean_sparse, color='gray', linestyle=':', alpha=0.5, linewidth=0.5)
        
        # Annotate all models (compact)
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        for model_name, model_data in section_data.items():
            x, y = model_data['sparse'], model_data['rich']
            display = model_data['display']
            
            # Compact label
            label = display.replace("3.1 ", "").replace("3 ", "").replace(" Base", " (b)").replace(" Instruct", " (i)")
            if len(label) > 10:
                parts = label.split()
                if len(parts) > 2:
                    label = '\n'.join([' '.join(parts[:2]), ' '.join(parts[2:])])
            
            # Smart offset
            offset_x = 0.015 * x_range
            offset_y = 0.015 * y_range
            text_x = min(xlim[1], max(xlim[0], x + offset_x))
            text_y = min(ylim[1], max(ylim[0], y + offset_y))
            
            ax.annotate(label, (x, y), 
                       xytext=(text_x, text_y), 
                       fontsize=5, alpha=0.7,
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                                edgecolor='none', alpha=0.7))
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        
        # Only add labels to leftmost and bottom subplots
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel(xlabel, fontsize=9)
        if idx % n_cols == 0:
            ax.set_ylabel(ylabel, fontsize=9)
        
        # Subplot label (A, B, C, ...)
        section_name_clean = section.replace('_', ' ').title()
        ax.text(0.02, 0.98, chr(65 + idx), transform=ax.transAxes,
               fontsize=12, fontweight='bold', va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Hide unused subplots
    for idx in range(n_sections, len(axes)):
        axes[idx].axis('off')
    
    # Add colorbar to the last subplot
    if n_sections > 0:
        cbar = plt.colorbar(scatter, ax=axes[-1], label='Model Size (log₁₀ B params)')
        size_min, size_max = min(sizes), max(sizes)
        ticks = []
        tick_labels = []
        for size in [4, 8, 30, 70, 120]:
            if size_min <= size <= size_max:
                ticks.append(np.log10(size))
                tick_labels.append(f'{size}B')
        if ticks:
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(width=0.5, labelsize=7)
    
    plt.tight_layout()
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(output_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
    print(f"Saved multi-panel plot to {output_path}")
    plt.close(fig)


def main():
    """Generate summary scatter plots for profile richness."""
    analysis_dir = Path(__file__).parent.parent / 'analysis'
    figures_dir = analysis_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Metrics to plot
    metrics = {
        'accuracy': {'xlabel': 'Accuracy with Sparse Profile (6 features)', 
                    'ylabel': 'Accuracy with Rich Profile (24 features)'},
        'js_divergence_soft': {'xlabel': 'JS Divergence (Sparse)', 
                              'ylabel': 'JS Divergence (Rich)'},
        'brier_score': {'xlabel': 'Brier Score (Sparse)', 
                       'ylabel': 'Brier Score (Rich)'},
        'macro_f1': {'xlabel': 'F1 Score (Sparse)', 
                    'ylabel': 'F1 Score (Rich)'},
    }
    
    # Load overall profile richness data
    overall_path = analysis_dir / 'profile_richness' / 'profile_richness_results.json'
    if overall_path.exists():
        print(f"Loading overall data from {overall_path}")
        overall_data = load_profile_richness_data(overall_path)
        
        # Create plots for each metric
        for metric, labels in metrics.items():
            overall_extracted = extract_sparse_rich_data(overall_data, by_section=False, metric=metric)
            
            if overall_extracted:
                # Determine axis limits from data
                all_sparse = [d['sparse'] for d in overall_extracted.values()]
                all_rich = [d['rich'] for d in overall_extracted.values()]
                xlim = (min(all_sparse) - 1, max(all_sparse) + 1)
                ylim = (min(all_rich) - 1, max(all_rich) + 1)
                
                output_path = figures_dir / f'figure_profile_richness_scatter_{metric}'
                create_scatter_plot(
                    overall_extracted,
                    title=None,  # No titles per ICML guidelines (use caption instead)
                    output_path=output_path,
                    xlim=xlim,
                    ylim=ylim,
                    xlabel=labels['xlabel'],
                    ylabel=labels['ylabel'],
                    annotate_all=True  # Annotate all models for consistency
                )
    else:
        print(f"Overall data file not found: {overall_path}")
    
    # Load by-section data
    by_section_path = analysis_dir / 'profile_richness_by_section' / 'profile_richness_by_section.json'
    if by_section_path.exists():
        print(f"\nLoading by-section data from {by_section_path}")
        by_section_data = load_profile_richness_data(by_section_path)
        
        # Create multi-panel plot for accuracy (main metric)
        by_section_extracted = extract_sparse_rich_data(by_section_data, by_section=True, metric='accuracy')
        if by_section_extracted:
            sections_dir = figures_dir / 'profile_richness_by_section'
            sections_dir.mkdir(parents=True, exist_ok=True)
            
            # Multi-panel figure (ICML style: no titles, subplot labels)
            output_path = sections_dir / 'figure_profile_richness_scatter_by_section'
            create_multi_panel_section_plot(by_section_extracted, 'accuracy', output_path)
            
            print(f"\nGenerated multi-panel plot with {len(by_section_extracted)} sections")
    else:
        print(f"By-section data file not found: {by_section_path}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
