#!/usr/bin/env python3
"""
Generate comprehensive tables from profile richness analysis results.

Creates tables showing all metrics, profile levels, and models in multiple formats:
- LaTeX (for papers)
- Markdown (for documentation)
- CSV (for Excel/spreadsheets)

Usage:
    python generate_profile_richness_table.py
    python generate_profile_richness_table.py --format latex
    python generate_profile_richness_table.py --format markdown --metrics accuracy macro_f1 variance_ratio_soft
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

# Model metadata for display names and sorting
MODEL_METADATA = {
    'deepseek-v3p1-terminus': {'display': 'DeepSeek-V3', 'size': 37, 'family': 'DeepSeek'},
    'gemma-3-27b-instruct': {'display': 'Gemma 3 27B', 'size': 27, 'family': 'Gemma'},
    'gpt_oss': {'display': 'GPT-OSS 120B', 'size': 120, 'family': 'GPT-OSS'},
    'llama3.1-70b-base': {'display': 'Llama 3.1 70B Base', 'size': 70, 'family': 'Llama'},
    'llama3.1-70b-instruct': {'display': 'Llama 3.1 70B Instruct', 'size': 70, 'family': 'Llama'},
    'llama3.1-8b-base': {'display': 'Llama 3.1 8B Base', 'size': 8, 'family': 'Llama'},
    'llama3.1-8b-instruct': {'display': 'Llama 3.1 8B Instruct', 'size': 8, 'family': 'Llama'},
    'olmo3-32b-base': {'display': 'OLMo 3 32B Base', 'size': 32, 'family': 'OLMo'},
    'olmo3-32b-dpo': {'display': 'OLMo 3 32B Instruct', 'size': 32, 'family': 'OLMo'},
    'olmo3-7b-base': {'display': 'OLMo 3 7B Base', 'size': 7, 'family': 'OLMo'},
    'olmo3-7b-dpo': {'display': 'OLMo 3 7B Instruct', 'size': 7, 'family': 'OLMo'},
    'qwen3-32b': {'display': 'Qwen 3 32B', 'size': 32, 'family': 'Qwen'},
    'qwen3-4b': {'display': 'Qwen 3 4B', 'size': 4, 'family': 'Qwen'},
}

# Profile type metadata
PROFILE_TYPES = {
    's3m2': {'name': 'Sparse', 'features': 6, 'order': 0},
    's4m3': {'name': 'Medium', 'features': 12, 'order': 1},
    's6m4': {'name': 'Rich', 'features': 24, 'order': 2},
}

# Metric metadata: name, format string, decimal places
# Note: {:.1%} format already multiplies by 100 and adds % sign
METRIC_METADATA = {
    'accuracy': {'name': 'Accuracy', 'format': '{:.1%}', 'decimals': 1, 'multiply': 1},
    'macro_f1': {'name': 'Macro F1', 'format': '{:.1%}', 'decimals': 1, 'multiply': 1},
    'variance_ratio_soft': {'name': 'Variance Ratio (Soft)', 'format': '{:.3f}', 'decimals': 3, 'multiply': 1},
    'variance_ratio_hard': {'name': 'Variance Ratio (Hard)', 'format': '{:.3f}', 'decimals': 3, 'multiply': 1},
    'js_divergence_soft': {'name': 'JS Divergence (Soft)', 'format': '{:.4f}', 'decimals': 4, 'multiply': 1},
    'js_divergence_hard': {'name': 'JS Divergence (Hard)', 'format': '{:.4f}', 'decimals': 4, 'multiply': 1},
    'brier_score': {'name': 'Brier Score', 'format': '{:.3f}', 'decimals': 3, 'multiply': 1},
    'ece': {'name': 'ECE', 'format': '{:.4f}', 'decimals': 4, 'multiply': 1},
    'mean_log_loss': {'name': 'Mean Log Loss', 'format': '{:.3f}', 'decimals': 3, 'multiply': 1},
}


def load_results(json_path: Path) -> Dict:
    """Load profile richness results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def format_value(value: Optional[float], metric_info: Dict) -> str:
    """Format a metric value according to its metadata."""
    if value is None:
        return '—'
    if metric_info['multiply'] != 1:
        value = value * metric_info['multiply']
    return metric_info['format'].format(value)


def generate_table_data(
    results: Dict,
    metrics: List[str],
    profile_order: List[str] = ['s3m2', 's4m3', 's6m4']
) -> List[Dict]:
    """
    Generate table data structure.
    
    Returns list of rows, where each row is a dict with:
    - 'model': model display name
    - 'model_key': model folder name
    - 'size': model size
    - 'family': model family
    - For each metric/profile: '{metric}_{profile}': formatted value
    """
    rows = []
    
    # Sort models by size, then by name
    sorted_models = sorted(
        results.keys(),
        key=lambda m: (
            MODEL_METADATA.get(m, {}).get('size', 0),
            MODEL_METADATA.get(m, {}).get('display', m)
        )
    )
    
    for model_key in sorted_models:
        model_info = MODEL_METADATA.get(model_key, {
            'display': model_key,
            'size': 0,
            'family': 'Unknown'
        })
        
        row = {
            'model': model_info['display'],
            'model_key': model_key,
            'size': model_info['size'],
            'family': model_info['family'],
        }
        
        model_results = results[model_key]
        
        # Add values for each metric/profile combination
        for metric in metrics:
            if metric not in METRIC_METADATA:
                continue
            
            metric_info = METRIC_METADATA[metric]
            
            for profile in profile_order:
                if profile in model_results:
                    value = model_results[profile].get(metric)
                    row[f'{metric}_{profile}'] = format_value(value, metric_info)
                else:
                    row[f'{metric}_{profile}'] = '—'
        
        rows.append(row)
    
    return rows


def generate_latex_table(
    rows: List[Dict],
    metrics: List[str],
    profile_order: List[str],
    caption: str = "Profile Richness Analysis Results",
    label: str = "tab:profile_richness"
) -> str:
    """Generate LaTeX table code following ICML guidelines."""
    lines = []
    lines.append("\\begin{table*}[t]")
    # Caption goes BEFORE the table (ICML guidelines)
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("  \\begin{center}")
    lines.append("    \\begin{small}")
    lines.append("      \\begin{sc}")
    lines.append("        \\begin{tabular}{l" + "c" * len(metrics) * len(profile_order) + "}")
    lines.append("          \\toprule")
    
    # Header row 1: Profile levels
    header1 = ["Model"]
    for metric in metrics:
        for profile in profile_order:
            profile_name = PROFILE_TYPES[profile]['name']
            header1.append(profile_name)
    lines.append("          " + " & ".join(header1) + " \\\\")
    lines.append("          \\midrule")
    
    # Header row 2: Metrics
    header2 = [""]
    for metric in metrics:
        metric_name = METRIC_METADATA[metric]['name']
        header2.extend([metric_name] * len(profile_order))
    lines.append("          " + " & ".join(header2) + " \\\\")
    lines.append("          \\midrule")
    
    # Data rows
    for row in rows:
        row_data = [row['model']]
        for metric in metrics:
            for profile in profile_order:
                key = f'{metric}_{profile}'
                row_data.append(row.get(key, '—'))
        lines.append("          " + " & ".join(row_data) + " \\\\")
    
    lines.append("          \\bottomrule")
    lines.append("        \\end{tabular}")
    lines.append("      \\end{sc}")
    lines.append("    \\end{small}")
    lines.append("  \\end{center}")
    lines.append("  \\vskip -0.1in")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def generate_markdown_table(
    rows: List[Dict],
    metrics: List[str],
    profile_order: List[str]
) -> str:
    """Generate Markdown table."""
    lines = []
    
    # Build header
    header = ["Model"]
    for metric in metrics:
        metric_name = METRIC_METADATA[metric]['name']
        for profile in profile_order:
            profile_name = PROFILE_TYPES[profile]['name']
            header.append(f"{metric_name}\n({profile_name})")
    
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    
    # Data rows
    for row in rows:
        row_data = [row['model']]
        for metric in metrics:
            for profile in profile_order:
                key = f'{metric}_{profile}'
                row_data.append(row.get(key, '—'))
        lines.append("| " + " | ".join(row_data) + " |")
    
    return "\n".join(lines)


def generate_csv_table(
    rows: List[Dict],
    metrics: List[str],
    profile_order: List[str]
) -> str:
    """Generate CSV table."""
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.writer(output)
    
    # Header
    header = ["Model", "Size", "Family"]
    for metric in metrics:
        metric_name = METRIC_METADATA[metric]['name']
        for profile in profile_order:
            profile_name = PROFILE_TYPES[profile]['name']
            header.append(f"{metric_name} ({profile_name})")
    
    writer.writerow(header)
    
    # Data rows
    for row in rows:
        row_data = [row['model'], row['size'], row['family']]
        for metric in metrics:
            for profile in profile_order:
                key = f'{metric}_{profile}'
                row_data.append(row.get(key, '—'))
        writer.writerow(row_data)
    
    return output.getvalue()


def generate_compact_latex_table(
    rows: List[Dict],
    metrics: List[str],
    profile_order: List[str],
    caption: str = "Profile Richness Analysis Results",
    label: str = "tab:profile_richness"
) -> str:
    """
    Generate a more compact LaTeX table with profile levels as rows.
    Structure: Model | Metric | Sparse | Medium | Rich
    Follows ICML guidelines.
    """
    lines = []
    lines.append("\\begin{table*}[t]")
    # Caption goes BEFORE the table (ICML guidelines)
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("  \\begin{center}")
    lines.append("    \\begin{small}")
    lines.append("      \\begin{sc}")
    lines.append("        \\begin{tabular}{l" + "c" * (len(profile_order) + 1) + "}")
    lines.append("          \\toprule")
    
    # Header
    header = ["Model", "Metric"]
    for profile in profile_order:
        header.append(PROFILE_TYPES[profile]['name'])
    lines.append("          " + " & ".join(header) + " \\\\")
    lines.append("          \\midrule")
    
    # Data rows: one row per model/metric combination
    for row in rows:
        model_name = row['model']
        for i, metric in enumerate(metrics):
            metric_name = METRIC_METADATA[metric]['name']
            
            row_data = [model_name if i == 0 else "", metric_name]
            for profile in profile_order:
                key = f'{metric}_{profile}'
                row_data.append(row.get(key, '—'))
            
            lines.append("          " + " & ".join(row_data) + " \\\\")
        
        # Add spacing between models (except last)
        if row != rows[-1]:
            lines.append("          \\midrule")
    
    lines.append("          \\bottomrule")
    lines.append("        \\end{tabular}")
    lines.append("      \\end{sc}")
    lines.append("    \\end{small}")
    lines.append("  \\end{center}")
    lines.append("  \\vskip -0.1in")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Generate tables from profile richness analysis results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('analysis/profile_richness/profile_richness_results.json'),
        help='Input JSON file with profile richness results'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('analysis/profile_richness'),
        help='Output directory for tables'
    )
    parser.add_argument(
        '--format',
        choices=['latex', 'markdown', 'csv', 'all', 'latex-compact'],
        default='all',
        help='Output format(s)'
    )
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=None,
        help='Metrics to include (default: all main metrics)'
    )
    parser.add_argument(
        '--caption',
        type=str,
        default='Profile Richness Analysis Results',
        help='Table caption (for LaTeX)'
    )
    parser.add_argument(
        '--label',
        type=str,
        default='tab:profile_richness',
        help='Table label (for LaTeX)'
    )
    
    args = parser.parse_args()
    
    # Default metrics if not specified
    if args.metrics is None:
        default_metrics = [
            'accuracy',
            'macro_f1',
            'variance_ratio_soft',
            'js_divergence_soft',
            'brier_score',
        ]
    else:
        default_metrics = args.metrics
    
    # Validate metrics
    metrics = []
    for metric in default_metrics:
        if metric in METRIC_METADATA:
            metrics.append(metric)
        else:
            print(f"Warning: Unknown metric '{metric}', skipping")
    
    if not metrics:
        print("Error: No valid metrics specified")
        return
    
    # Load results
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    print(f"Loading results from {args.input}")
    results = load_results(args.input)
    print(f"Loaded results for {len(results)} models")
    
    # Generate table data
    profile_order = ['s3m2', 's4m3', 's6m4']
    rows = generate_table_data(results, metrics, profile_order)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    formats_to_generate = []
    if args.format == 'all':
        formats_to_generate = ['latex', 'markdown', 'csv', 'latex-compact']
    else:
        formats_to_generate = [args.format]
    
    # Generate tables
    for fmt in formats_to_generate:
        if fmt == 'latex':
            table = generate_latex_table(rows, metrics, profile_order, args.caption, args.label)
            output_path = args.output_dir / 'profile_richness_table.tex'
            with open(output_path, 'w') as f:
                f.write(table)
            print(f"Generated LaTeX table: {output_path}")
        
        elif fmt == 'latex-compact':
            table = generate_compact_latex_table(rows, metrics, profile_order, args.caption, args.label)
            output_path = args.output_dir / 'profile_richness_table_compact.tex'
            with open(output_path, 'w') as f:
                f.write(table)
            print(f"Generated compact LaTeX table: {output_path}")
        
        elif fmt == 'markdown':
            table = generate_markdown_table(rows, metrics, profile_order)
            output_path = args.output_dir / 'profile_richness_table.md'
            with open(output_path, 'w') as f:
                f.write(table)
            print(f"Generated Markdown table: {output_path}")
        
        elif fmt == 'csv':
            table = generate_csv_table(rows, metrics, profile_order)
            output_path = args.output_dir / 'profile_richness_table.csv'
            with open(output_path, 'w') as f:
                f.write(table)
            print(f"Generated CSV table: {output_path}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
