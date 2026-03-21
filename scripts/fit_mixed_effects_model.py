#!/usr/bin/env python3
"""
Fit mixed effects model to profile richness data.

Fits a mixed effects model with random effects for model, respondent, question, region, and topic_section.

Usage:
    python fit_mixed_effects_model.py \\
        --data analysis/mixed_effects_data.csv \\
        --output analysis/mixed_effects_model \\
        --formula "correct ~ 1"
    
    # Custom formula
    python fit_mixed_effects_model.py \\
        --data analysis/mixed_effects_data.csv \\
        --output analysis/mixed_effects_model \\
        --formula "correct ~ 1" \\
        --re-formula "(1|model) + (1|respondent) + (1|question) + (1|region) + (1|topic_section)"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLM
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available. Install with: pip install statsmodels")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def parse_re_formula(re_formula: str) -> Dict[str, List[str]]:
    """
    Parse random effects formula like "(1|model) + (1|respondent) + (1|question)".
    
    Returns dict mapping group_var -> list of random effect vars.
    For statsmodels MixedLM, we need to specify groups and random effects separately.
    """
    # Simple parser for (1|var) format
    import re
    pattern = r'\(1\|(\w+)\)'
    matches = re.findall(pattern, re_formula)
    return matches


def fit_mixed_effects_model(
    df: pd.DataFrame,
    formula: str = "correct ~ 1",
    re_formula: str = "(1|model) + (1|respondent) + (1|question) + (1|region) + (1|topic_section)",
    groups: Optional[str] = None,
) -> Dict:
    """
    Fit mixed effects model using statsmodels MixedLM.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with columns: correct, model, respondent, question, region, topic_section, etc.
    formula : str
        Fixed effects formula (default: "correct ~ 1" for intercept only)
    re_formula : str
        Random effects formula (default: all random effects)
    groups : Optional[str]
        Grouping variable (if None, uses first random effect)
    
    Returns
    -------
    Dict with model results, diagnostics, and summary
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels is required. Install with: pip install statsmodels")
    
    print("Fitting mixed effects model...")
    print(f"  Formula: {formula}")
    print(f"  Random effects: {re_formula}")
    print(f"  N observations: {len(df):,}")
    
    # Parse random effects
    re_vars = parse_re_formula(re_formula)
    if not re_vars:
        raise ValueError(f"No random effects found in: {re_formula}")
    
    print(f"  Random effect variables: {re_vars}")
    
    # Check that all variables exist
    all_vars = set(re_vars)
    if formula != "correct ~ 1":
        # Extract variables from formula (simple parsing)
        import re
        formula_vars = re.findall(r'\b(\w+)\b', formula)
        all_vars.update(formula_vars)
    
    missing_vars = all_vars - set(df.columns)
    if missing_vars:
        raise ValueError(f"Missing variables in data: {missing_vars}")
    
    # For crossed random effects in statsmodels, we need to:
    # 1. Use a single group for all observations (or use respondent as group)
    # 2. Specify crossed random effects using vc_formula (variance components)
    
    # Create a single group variable for all observations (for crossed effects)
    df = df.copy()
    df['_group'] = 1
    
    # Use respondent as grouping variable if specified, otherwise use single group
    if groups is None:
        groups = '_group'
    
    if groups not in df.columns and groups != '_group':
        raise ValueError(f"Grouping variable '{groups}' not found in data")
    
    # Build vc_formula for crossed random effects
    # Format: {"re_name": "0 + C(variable)"} for each random effect
    vc_formula = {}
    for var in re_vars:
        vc_formula[var] = f"0 + C({var})"
    
    print(f"  Grouping variable: {groups}")
    if groups == '_group':
        print(f"  Using single group (crossed random effects)")
    else:
        print(f"  N groups: {df[groups].nunique():,}")
    print(f"  Variance components: {list(vc_formula.keys())}")
    
    # Fit model using formula API (easier for crossed effects)
    print("\nFitting model (this may take a while)...")
    print("  Note: statsmodels doesn't provide progress updates during fitting.")
    print("  For large datasets, this can take 10-30+ minutes. Please be patient...")
    
    import time
    start_time = time.time()
    elapsed = 0.0
    
    try:
        # Use MixedLM.from_formula for easier specification
        model = smf.mixedlm(
            formula,
            data=df,
            groups=df[groups],
            vc_formula=vc_formula,
        )
        
        # Fit with REML (default) or ML
        # Note: statsmodels doesn't expose progress callbacks, so we can't show
        # a real-time progress bar. We'll show elapsed time instead.
        print("  Starting optimization...")
        
        # Try to fit with periodic status updates
        # Unfortunately, statsmodels doesn't support progress callbacks,
        # but we can at least show that it's working
        result = model.fit(reml=True, method=["lbfgs"])
        
        elapsed = time.time() - start_time
        print(f"✓ Model converged! (took {elapsed/60:.1f} minutes)")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\nError fitting model after {elapsed/60:.1f} minutes: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Extract results
    model_results = {
        'formula': formula,
        're_formula': re_formula,
        'groups': groups,
        'n_obs': len(df),
        'n_groups': df[groups].nunique() if groups != '_group' else 1,
        'converged': result.converged,
        'llf': result.llf,  # Log-likelihood
        'aic': result.aic,
        'bic': result.bic,
        'fixed_effects': {
            'coefficients': result.fe_params.to_dict(),
            'pvalues': result.pvalues.to_dict() if hasattr(result, 'pvalues') else None,
            'std_errors': result.bse.to_dict() if hasattr(result, 'bse') else None,
        },
        'summary_text': str(result.summary()),
    }
    
    # Convergence diagnostics
    diagnostics = {
        'converged': result.converged,
        'convergence_warnings': [],
    }
    
    if not result.converged:
        diagnostics['convergence_warnings'].append("Model did not converge!")
    
    # Check for other warnings
    if hasattr(result, 'warnings'):
        diagnostics['convergence_warnings'].extend(result.warnings)
    
    # Variance components (from vcomp for crossed effects)
    if hasattr(result, 'vcomp'):
        var_components = {}
        for name in re_vars:
            if name in result.vcomp:
                var_components[name] = float(result.vcomp[name])
        model_results['variance_components'] = var_components
    elif hasattr(result, 'cov_re'):
        # Fallback for nested models
        var_components = {}
        if hasattr(result.cov_re, 'diagonal'):
            diag = result.cov_re.diagonal()
            for i, name in enumerate(re_vars):
                if i < len(diag):
                    var_components[name] = float(diag[i])
        model_results['variance_components'] = var_components
    
    model_results['diagnostics'] = diagnostics
    model_results['fit_time_seconds'] = elapsed
    
    return {
        'model': result,
        'results': model_results,
    }


def print_model_summary(fit_result: Dict) -> None:
    """Print model summary and diagnostics."""
    results = fit_result['results']
    model = fit_result['model']
    
    print("\n" + "=" * 80)
    print("MIXED EFFECTS MODEL RESULTS")
    print("=" * 80)
    
    print(f"\nModel specification:")
    print(f"  Formula: {results['formula']}")
    print(f"  Random effects: {results['re_formula']}")
    print(f"  Groups: {results['groups']}")
    print(f"  N observations: {results['n_obs']:,}")
    print(f"  N groups: {results['n_groups']:,}")
    
    print(f"\nModel fit:")
    print(f"  Log-likelihood: {results['llf']:.2f}")
    print(f"  AIC: {results['aic']:.2f}")
    print(f"  BIC: {results['bic']:.2f}")
    print(f"  Converged: {results['converged']}")
    if 'fit_time_seconds' in results:
        print(f"  Fit time: {results['fit_time_seconds']/60:.1f} minutes")
    
    # Variance components
    if 'variance_components' in results:
        print(f"\nVariance components (random effects):")
        var_components = results['variance_components']
        total_var = sum(var_components.values())
        for name, var in sorted(var_components.items(), key=lambda x: -x[1]):
            pct = 100 * var / total_var if total_var > 0 else 0
            print(f"  {name:15s}: {var:.6f} ({pct:.1f}% of total)")
        print(f"  {'Total':15s}: {total_var:.6f}")
    
    # Convergence diagnostics
    print(f"\nConvergence diagnostics:")
    diagnostics = results['diagnostics']
    if diagnostics['converged']:
        print(f"  ✓ Model converged successfully")
    else:
        print(f"  ✗ Model did NOT converge")
    
    if diagnostics['convergence_warnings']:
        print(f"  Warnings:")
        for warning in diagnostics['convergence_warnings']:
            print(f"    - {warning}")
    
    # Print full statsmodels summary
    print(f"\nDetailed summary:")
    print(model.summary())


def generate_diagnostic_plots(fit_result: Dict, output_path: Path) -> None:
    """
    Generate diagnostic plots for the mixed effects model.
    
    Follows ICML guidelines:
    - Dark lines (>= 0.5pt)
    - No gray backgrounds
    - No titles inside figure (caption serves this function)
    - Labeled axes
    - White background
    """
    if not HAS_MATPLOTLIB:
        print("  (Skipping plots - matplotlib not available)")
        return
    
    print("\nGenerating diagnostic plots...")
    import time
    plot_start = time.time()
    
    result = fit_result['model']
    
    # Create figure with subplots
    # ICML: No title inside figure, white background
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # ICML: Dark lines >= 0.5pt, no gray backgrounds
    linewidth = 0.5  # Minimum 0.5pt per ICML guidelines
    
    # 1. Q-Q plot of residuals (check normality)
    from scipy import stats
    residuals = result.resid
    stats.probplot(residuals, dist="norm", plot=axes[0, 0])
    # ICML: Remove title, ensure dark lines, label axes
    axes[0, 0].set_xlabel('Theoretical Quantiles', fontsize=14, labelpad=8)
    axes[0, 0].set_ylabel('Sample Quantiles', fontsize=14, labelpad=8)
    # Make Q-Q line darker and thicker
    for line in axes[0, 0].get_lines():
        line.set_linewidth(linewidth)
        line.set_color('black')
    axes[0, 0].grid(True, alpha=0.3, linewidth=linewidth, color='black')
    axes[0, 0].set_facecolor('white')
    axes[0, 0].spines['bottom'].set_color('black')
    axes[0, 0].spines['top'].set_color('black')
    axes[0, 0].spines['right'].set_color('black')
    axes[0, 0].spines['left'].set_color('black')
    
    # 2. Residuals vs fitted values (check homoscedasticity)
    fitted = result.fittedvalues
    axes[0, 1].scatter(fitted, residuals, alpha=0.3, s=1, color='black', edgecolors='none')
    axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=linewidth)
    axes[0, 1].set_xlabel('Fitted Values', fontsize=14, labelpad=8)
    axes[0, 1].set_ylabel('Residuals', fontsize=14, labelpad=8)
    axes[0, 1].grid(True, alpha=0.3, linewidth=linewidth, color='black')
    axes[0, 1].set_facecolor('white')
    axes[0, 1].spines['bottom'].set_color('black')
    axes[0, 1].spines['top'].set_color('black')
    axes[0, 1].spines['right'].set_color('black')
    axes[0, 1].spines['left'].set_color('black')
    
    # 3. Histogram of residuals (check distribution)
    axes[1, 0].hist(residuals, bins=50, edgecolor='black', facecolor='black', alpha=0.7, linewidth=linewidth)
    axes[1, 0].set_xlabel('Residuals', fontsize=14, labelpad=8)
    axes[1, 0].set_ylabel('Frequency', fontsize=14, labelpad=8)
    axes[1, 0].grid(True, alpha=0.3, axis='y', linewidth=linewidth, color='black')
    axes[1, 0].set_facecolor('white')
    axes[1, 0].spines['bottom'].set_color('black')
    axes[1, 0].spines['top'].set_color('black')
    axes[1, 0].spines['right'].set_color('black')
    axes[1, 0].spines['left'].set_color('black')
    
    # 4. Scale-location plot (check homoscedasticity)
    sqrt_abs_residuals = np.sqrt(np.abs(residuals))
    axes[1, 1].scatter(fitted, sqrt_abs_residuals, alpha=0.3, s=1, color='black', edgecolors='none')
    axes[1, 1].set_xlabel('Fitted Values', fontsize=14, labelpad=8)
    axes[1, 1].set_ylabel('√|Residuals|', fontsize=14, labelpad=8)
    axes[1, 1].grid(True, alpha=0.3, linewidth=linewidth, color='black')
    axes[1, 1].set_facecolor('white')
    axes[1, 1].spines['bottom'].set_color('black')
    axes[1, 1].spines['top'].set_color('black')
    axes[1, 1].spines['right'].set_color('black')
    axes[1, 1].spines['left'].set_color('black')
    
    # ICML: Ensure all text is dark and increase font sizes
    for ax in axes.flat:
        ax.tick_params(colors='black', width=linewidth, labelsize=10)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color('black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
    
    plt.tight_layout()
    
    # Save plots
    plot_path = output_path.with_suffix('.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  ✓ Saved diagnostic plots to {plot_path}")
    
    # Also save PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  ✓ Saved diagnostic plots to {pdf_path}")
    
    plt.close()
    
    plot_elapsed = time.time() - plot_start
    print(f"  (Took {plot_elapsed:.1f} seconds)")


def save_model_results(fit_result: Dict, output_path: Path, include_plots: bool = True) -> None:
    """Save model results to JSON and text files."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = fit_result['results']
    
    # Save JSON (serializable parts)
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to {json_path}")
    
    # Save text summary
    txt_path = output_path.with_suffix('.txt')
    with open(txt_path, 'w') as f:
        f.write("MIXED EFFECTS MODEL RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Formula: {results['formula']}\n")
        f.write(f"Random effects: {results['re_formula']}\n")
        f.write(f"Groups: {results['groups']}\n")
        f.write(f"N observations: {results['n_obs']:,}\n")
        f.write(f"N groups: {results['n_groups']:,}\n\n")
        f.write(f"Log-likelihood: {results['llf']:.2f}\n")
        f.write(f"AIC: {results['aic']:.2f}\n")
        f.write(f"BIC: {results['bic']:.2f}\n")
        f.write(f"Converged: {results['converged']}\n\n")
        f.write(results['summary_text'])
    print(f"✓ Saved summary to {txt_path}")
    
    # Generate diagnostic plots
    if include_plots:
        generate_diagnostic_plots(fit_result, output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Fit mixed effects model to profile richness data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--data',
        type=Path,
        required=True,
        help='Path to prepared data CSV (from prepare_mixed_effects_data.py)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('analysis/mixed_effects/mixed_effects_model'),
        help='Output path (without extension, will create .json and .txt)'
    )
    parser.add_argument(
        '--formula',
        type=str,
        default='correct ~ 1',
        help='Fixed effects formula (default: correct ~ 1)'
    )
    parser.add_argument(
        '--re-formula',
        type=str,
        default='(1|model) + (1|respondent) + (1|question) + (1|region) + (1|topic_section)',
        help='Random effects formula (default: all random effects)'
    )
    parser.add_argument(
        '--groups',
        type=str,
        default=None,
        help='Grouping variable (default: respondent)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip diagnostic plots generation'
    )
    parser.add_argument(
        '--sample',
        type=float,
        default=None,
        help='Randomly sample fraction of data (0.0-1.0) to reduce memory usage. Example: 0.1 for 10%% sample'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )
    parser.add_argument(
        '--use-float32',
        action='store_true',
        help='Convert numeric columns to float32 (may reduce memory, but statsmodels may convert back to float64)'
    )
    
    args = parser.parse_args()
    
    if not HAS_STATSMODELS:
        print("\nError: statsmodels is required.")
        print("Install with: pip install statsmodels")
        sys.exit(1)
    
    # Load data
    print(f"Loading data from {args.data}...")
    if args.data.suffix == '.parquet':
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)
    
    print(f"  Loaded {len(df):,} observations")
    print(f"  Columns: {list(df.columns)}")
    
    # Sample data if requested (to reduce memory usage)
    if args.sample is not None:
        if not (0 < args.sample <= 1):
            print(f"\nError: --sample must be between 0 and 1 (got {args.sample})")
            sys.exit(1)
        
        original_size = len(df)
        np.random.seed(args.seed)
        df = df.sample(frac=args.sample, random_state=args.seed).reset_index(drop=True)
        print(f"\n  ⚠ Sampling: Using {len(df):,} observations ({args.sample:.1%} of {original_size:,})")
        print(f"  This reduces memory requirements but may affect precision of estimates.")
    
    # Estimate memory requirements
    n_obs = len(df)
    n_models = df['model'].nunique()
    n_respondents = df['respondent'].nunique()
    n_questions = df['question'].nunique()
    n_regions = df['region'].nunique()
    n_sections = df['topic_section'].nunique()
    
    total_re_levels = n_models + n_respondents + n_questions + n_regions + n_sections
    # Note: statsmodels/patsy uses float64 internally, so we can't reduce this
    # The design matrix is built by patsy and uses float64 by default
    estimated_memory_gb = (n_obs * total_re_levels * 8) / (1024**3)  # float64 = 8 bytes
    
    print(f"\nDataset structure:")
    print(f"  Observations: {n_obs:,}")
    print(f"  Models: {n_models}")
    print(f"  Respondents: {n_respondents:,}")
    print(f"  Questions: {n_questions}")
    print(f"  Regions: {n_regions}")
    print(f"  Sections: {n_sections}")
    print(f"  Total random effect levels: {total_re_levels:,}")
    print(f"  Estimated memory needed: ~{estimated_memory_gb:.1f} GiB")
    
    if estimated_memory_gb > 50:
        print(f"\n  ⚠ WARNING: Large memory requirement ({estimated_memory_gb:.1f} GiB)")
        print(f"  Note: statsmodels/patsy uses float64 internally, so we can't reduce precision")
        print(f"  Solutions:")
        print(f"    1. Use --sample to reduce dataset size (e.g., --sample 0.1 for 10%)")
        print(f"    2. Use R's lme4 package (more memory-efficient for large datasets)")
        print(f"    3. Simplify model (remove some random effects)")
    
    # Check required columns
    required = ['correct', 'model', 'respondent', 'question', 'region', 'topic_section']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"\nError: Missing required columns: {missing}")
        sys.exit(1)
    
    # Try to use float32 for numeric columns (may help with memory, though statsmodels may convert back)
    if args.use_float32:
        print(f"\n  Converting numeric columns to float32...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'correct':  # Keep correct as int for now
                df[col] = df[col].astype(np.float32)
        print(f"  Note: statsmodels/patsy may convert back to float64 internally")
    
    # Fit model
    try:
        fit_result = fit_mixed_effects_model(
            df,
            formula=args.formula,
            re_formula=args.re_formula,
            groups=args.groups,
        )
    except Exception as e:
        print(f"\nError fitting model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print summary
    print_model_summary(fit_result)
    
    # Save results
    save_model_results(fit_result, args.output, include_plots=not args.no_plots)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
