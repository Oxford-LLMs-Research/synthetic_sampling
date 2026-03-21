#!/usr/bin/env python3
"""
Fit mixed effects model using R's lme4 package (wrapper script).

This is a Python wrapper that calls an R script. R's lme4 is more
memory-efficient than statsmodels for large datasets with crossed random effects.

Prerequisites:
    - R installed and in PATH
    - R packages: lme4, jsonlite, optparse
    - Install with: Rscript -e "install.packages(c('lme4', 'jsonlite', 'optparse'))"

Usage:
    python fit_mixed_effects_model_r.py \\
        --data analysis/mixed_effects/mixed_effects_data.csv \\
        --output analysis/mixed_effects/mixed_effects_model
"""

import argparse
import subprocess
import sys
from pathlib import Path

def find_rscript():
    """Find Rscript executable on Windows or Unix."""
    import platform
    import os
    
    # Try direct call first
    try:
        result = subprocess.run(['Rscript', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return 'Rscript'
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # On Windows, try common locations
    if platform.system() == 'Windows':
        common_paths = [
            r'C:\Program Files\R\R-*\bin\Rscript.exe',
            r'C:\Program Files (x86)\R\R-*\bin\Rscript.exe',
        ]
        import glob
        for pattern in common_paths:
            matches = glob.glob(pattern)
            if matches:
                # Get the latest version
                matches.sort(reverse=True)
                rscript_path = matches[0]
                # Verify it works
                try:
                    result = subprocess.run([rscript_path, '--version'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        return rscript_path
                except:
                    pass
    
    return None

def check_r_available():
    """Check if R is available. Returns (is_available, rscript_path)."""
    rscript = find_rscript()
    if rscript is None:
        return False, None
    return True, rscript

def check_r_packages(rscript):
    """Check if required R packages are installed."""
    script = """
    required <- c('lme4', 'jsonlite', 'optparse')
    missing <- required[!required %in% rownames(installed.packages())]
    if (length(missing) > 0) {
        cat(paste('Missing packages:', paste(missing, collapse=', '), '\n'))
        quit(status=1)
    } else {
        cat('All required packages installed\n')
        quit(status=0)
    }
    """
    try:
        result = subprocess.run([rscript, '--vanilla', '-'], 
                              input=script, capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Fit mixed effects model using R\'s lme4 (wrapper)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python fit_mixed_effects_model_r.py --data analysis/mixed_effects/mixed_effects_data.csv

  # With sampling
  python fit_mixed_effects_model_r.py --data analysis/mixed_effects/mixed_effects_data.csv --sample 0.1

  # Custom formula (different specification)
  python fit_mixed_effects_model_r.py --data analysis/mixed_effects/mixed_effects_data.csv \\
      --formula "correct ~ n_options + modal_share + model + region + topic_section" \\
      --re-formula "(1|survey)"
        """
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
        help='Output path prefix (without extension, will create .json and .txt)'
    )
    parser.add_argument(
        '--formula',
        type=str,
        default='correct ~ n_options + modal_share + model + region + topic_section',
        help='Fixed effects formula (default: correct ~ n_options + modal_share + model + region + topic_section)'
    )
    parser.add_argument(
        '--re-formula',
        type=str,
        default='(1|survey)',
        help='Random effects formula (default: (1|survey))'
    )
    parser.add_argument(
        '--sample',
        type=float,
        default=None,
        help='Randomly sample fraction of data (0.0-1.0) to reduce memory usage'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip diagnostic plots generation'
    )
    
    args = parser.parse_args()
    
    # Check R availability
    print("Checking R installation...")
    r_available, rscript = check_r_available()
    if not r_available:
        print("\nError: R is not installed or Rscript is not in PATH")
        print("Please install R from https://cran.r-project.org/")
        print("Or add R's bin directory to your PATH")
        sys.exit(1)
    print(f"  OK: R is available ({rscript})")
    
    # Check R packages
    print("Checking R packages...")
    if not check_r_packages(rscript):
        print("\nError: Required R packages are missing")
        print("Install with:")
        print(f'  {rscript} -e "install.packages(c(\'lme4\', \'jsonlite\', \'optparse\'))"')
        sys.exit(1)
    print("  OK: All required packages installed")
    
    # Find R script
    script_dir = Path(__file__).parent
    r_script = script_dir / 'fit_mixed_effects_model_r.R'
    
    if not r_script.exists():
        print(f"\nError: R script not found at {r_script}")
        sys.exit(1)
    
    # Build command
    # Always pass --re-formula explicitly to avoid R script fallback issues
    cmd = [
        rscript,
        '--vanilla',
        str(r_script),
        '--data', str(args.data),
        '--output', str(args.output),
        '--formula', args.formula,
        '--re-formula', args.re_formula,  # Always pass explicitly
    ]
    
    if args.sample is not None:
        cmd.extend(['--sample', str(args.sample)])
    
    cmd.extend(['--seed', str(args.seed)])
    
    if args.no_plots:
        cmd.append('--no-plots')
    
    # For display, quote arguments that might have special characters
    display_cmd = []
    for arg in cmd:
        if any(c in arg for c in '()[]{}'):
            display_cmd.append(f'"{arg}"')
        else:
            display_cmd.append(arg)
    
    print(f"\nRunning R script: {' '.join(display_cmd)}\n")
    print("=" * 80)
    
    # Run R script
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 80)
        print("OK: Model fitting completed successfully!")
        print(f"\nResults saved to:")
        print(f"  {args.output}.json")
        print(f"  {args.output}.txt")
    except subprocess.CalledProcessError as e:
        print(f"\nError: R script failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)

if __name__ == '__main__':
    main()
