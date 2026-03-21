#!/usr/bin/env python3
"""
Generate LaTeX table with country-level results (medians across models).

Usage:
    python generate_country_table.py --input analysis/disaggregated/by_country.json --output analysis/country_table.tex
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

try:
    import pycountry
    HAS_PYCOUNTRY = True
except ImportError:
    HAS_PYCOUNTRY = False
    print("Warning: pycountry not available. Installing or using fallback mapping...")
    # Try to install pycountry
    try:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pycountry", "-q"])
        import pycountry
        HAS_PYCOUNTRY = True
        print("Successfully installed pycountry")
    except:
        pass

# Basic fallback mapping for common countries (ISO-2 -> Name)
FALLBACK_COUNTRY_NAMES = {
    'AD': 'Andorra', 'AE': 'United Arab Emirates', 'AF': 'Afghanistan', 'AG': 'Antigua and Barbuda',
    'AL': 'Albania', 'AM': 'Armenia', 'AO': 'Angola', 'AR': 'Argentina', 'AT': 'Austria', 'AU': 'Australia',
    'AZ': 'Azerbaijan', 'BA': 'Bosnia and Herzegovina', 'BD': 'Bangladesh', 'BE': 'Belgium', 'BF': 'Burkina Faso',
    'BG': 'Bulgaria', 'BI': 'Burundi', 'BJ': 'Benin', 'BO': 'Bolivia', 'BR': 'Brazil', 'BW': 'Botswana',
    'BY': 'Belarus', 'CA': 'Canada', 'CD': 'Democratic Republic of the Congo', 'CF': 'Central African Republic',
    'CG': 'Republic of the Congo', 'CH': 'Switzerland', 'CI': "Côte d'Ivoire", 'CL': 'Chile', 'CM': 'Cameroon',
    'CN': 'China', 'CO': 'Colombia', 'CR': 'Costa Rica', 'CV': 'Cape Verde', 'CY': 'Cyprus', 'CZ': 'Czechia',
    'DE': 'Germany', 'DJ': 'Djibouti', 'DK': 'Denmark', 'DO': 'Dominican Republic', 'DZ': 'Algeria',
    'EC': 'Ecuador', 'EE': 'Estonia', 'EG': 'Egypt', 'ER': 'Eritrea', 'ES': 'Spain', 'ET': 'Ethiopia',
    'FI': 'Finland', 'FR': 'France', 'GA': 'Gabon', 'GB': 'United Kingdom', 'GE': 'Georgia', 'GH': 'Ghana',
    'GM': 'Gambia', 'GN': 'Guinea', 'GQ': 'Equatorial Guinea', 'GR': 'Greece', 'GT': 'Guatemala',
    'GW': 'Guinea-Bissau', 'GY': 'Guyana', 'HK': 'Hong Kong', 'HN': 'Honduras', 'HR': 'Croatia',
    'HT': 'Haiti', 'HU': 'Hungary', 'ID': 'Indonesia', 'IE': 'Ireland', 'IL': 'Israel', 'IN': 'India',
    'IQ': 'Iraq', 'IR': 'Iran', 'IS': 'Iceland', 'IT': 'Italy', 'JM': 'Jamaica', 'JO': 'Jordan',
    'JP': 'Japan', 'KE': 'Kenya', 'KG': 'Kyrgyzstan', 'KH': 'Cambodia', 'KM': 'Comoros', 'KN': 'Saint Kitts and Nevis',
    'KR': 'South Korea', 'KW': 'Kuwait', 'KZ': 'Kazakhstan', 'LA': 'Laos', 'LB': 'Lebanon', 'LC': 'Saint Lucia',
    'LK': 'Sri Lanka', 'LR': 'Liberia', 'LS': 'Lesotho', 'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia',
    'LY': 'Libya', 'MA': 'Morocco', 'MD': 'Moldova', 'ME': 'Montenegro', 'MG': 'Madagascar', 'MK': 'North Macedonia',
    'ML': 'Mali', 'MM': 'Myanmar', 'MN': 'Mongolia', 'MO': 'Macao', 'MR': 'Mauritania', 'MT': 'Malta',
    'MU': 'Mauritius', 'MV': 'Maldives', 'MW': 'Malawi', 'MX': 'Mexico', 'MY': 'Malaysia', 'MZ': 'Mozambique',
    'NA': 'Namibia', 'NE': 'Niger', 'NG': 'Nigeria', 'NI': 'Nicaragua', 'NL': 'Netherlands', 'NO': 'Norway',
    'NP': 'Nepal', 'NZ': 'New Zealand', 'OM': 'Oman', 'PA': 'Panama', 'PE': 'Peru', 'PG': 'Papua New Guinea',
    'PH': 'Philippines', 'PK': 'Pakistan', 'PL': 'Poland', 'PR': 'Puerto Rico', 'PS': 'Palestine', 'PT': 'Portugal',
    'PY': 'Paraguay', 'QA': 'Qatar', 'RO': 'Romania', 'RS': 'Serbia', 'RU': 'Russia', 'RW': 'Rwanda',
    'SA': 'Saudi Arabia', 'SB': 'Solomon Islands', 'SC': 'Seychelles', 'SD': 'Sudan', 'SE': 'Sweden',
    'SG': 'Singapore', 'SI': 'Slovenia', 'SK': 'Slovakia', 'SL': 'Sierra Leone', 'SN': 'Senegal', 'SO': 'Somalia',
    'SR': 'Suriname', 'SS': 'South Sudan', 'ST': 'São Tomé and Príncipe', 'SV': 'El Salvador', 'SY': 'Syria',
    'SZ': 'Eswatini', 'TD': 'Chad', 'TG': 'Togo', 'TH': 'Thailand', 'TJ': 'Tajikistan', 'TL': 'Timor-Leste',
    'TN': 'Tunisia', 'TR': 'Turkey', 'TT': 'Trinidad and Tobago', 'TW': 'Taiwan', 'TZ': 'Tanzania',
    'UA': 'Ukraine', 'UG': 'Uganda', 'US': 'United States', 'UY': 'Uruguay', 'UZ': 'Uzbekistan',
    'VE': 'Venezuela', 'VN': 'Vietnam', 'YE': 'Yemen', 'ZA': 'South Africa', 'ZM': 'Zambia', 'ZW': 'Zimbabwe',
}


def get_country_name(iso_code: str) -> str:
    """Convert ISO 3166-1 alpha-2 code to country name."""
    if HAS_PYCOUNTRY:
        try:
            country = pycountry.countries.get(alpha_2=iso_code)
            if country:
                return country.name
        except (KeyError, AttributeError):
            pass
    
    # Fallback: use mapping dictionary
    return FALLBACK_COUNTRY_NAMES.get(iso_code, iso_code)


def compute_median_across_models(
    data: Dict[str, Dict[str, Dict]], 
    country: str, 
    metric: str
) -> Optional[float]:
    """Compute median of a metric across all models for a given country."""
    values = []
    for model_name, countries_data in data.items():
        if country in countries_data:
            value = countries_data[country].get(metric)
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                values.append(float(value))
    
    if not values:
        return None
    
    return float(np.median(values))


def format_value(value: Optional[float], format_str: str = "{:.2f}") -> str:
    """Format a numeric value for LaTeX table."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "---"
    return format_str.format(value)


def generate_latex_table(
    data: Dict[str, Dict[str, Dict]],
    output_path: Path,
    metrics: List[tuple] = None
) -> None:
    """
    Generate LaTeX table with countries as rows and metrics as columns.
    
    Args:
        data: Nested dict {model_name: {country_code: {metric: value}}}
        output_path: Path to output LaTeX file
        metrics: List of (metric_key, display_name, format_str) tuples
    """
    if metrics is None:
        # Default metrics matching main figure and other analyses
        metrics = [
            ('accuracy', 'Accuracy', '{:.1%}'),
            ('macro_f1', 'Macro F1', '{:.1%}'),
            ('variance_ratio_soft_median', 'VR (Soft)', '{:.2f}'),
            ('js_divergence_soft_median', 'JS Divergence', '{:.3f}'),
        ]
    
    # Get all countries (sorted)
    all_countries = set()
    for model_data in data.values():
        all_countries.update(model_data.keys())
    countries_sorted = sorted(all_countries)
    
    # Compute medians for each country
    country_results = {}
    for country in countries_sorted:
        country_results[country] = {}
        for metric_key, _, _ in metrics:
            median_value = compute_median_across_models(data, country, metric_key)
            country_results[country][metric_key] = median_value
    
    # Generate LaTeX following ICML guidelines
    # Note: longtable replaces table environment, but we maintain formatting
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write longtable with proper formatting
        f.write("\\begin{center}\n")
        f.write("  \\begin{small}\n")
        f.write("    \\begin{sc}\n")
        f.write("      \\begin{longtable}{l" + "c" * len(metrics) + "}\n")
        f.write("        \\caption{Classification accuracies and other metrics by country (medians across models).}\n")
        f.write("        \\label{tab:country_results}\n")
        f.write("        \\\\\n")
        f.write("        \\toprule\n")
        
        # Column headers
        header_parts = ["Country"]
        for _, display_name, _ in metrics:
            header_parts.append(display_name)
        f.write("        " + " & ".join(header_parts) + " \\\\\n")
        f.write("        \\midrule\n")
        f.write("        \\endfirsthead\n")
        f.write("        \\multicolumn{" + str(len(metrics) + 1) + "}{c}{\\tablename\\ \\thetable\\ -- \\textit{continued from previous page}} \\\\\n")
        f.write("        \\toprule\n")
        f.write("        " + " & ".join(header_parts) + " \\\\\n")
        f.write("        \\midrule\n")
        f.write("        \\endhead\n")
        f.write("        \\midrule\n")
        f.write("        \\multicolumn{" + str(len(metrics) + 1) + "}{r}{\\textit{continued on next page}} \\\\\n")
        f.write("        \\endfoot\n")
        f.write("        \\bottomrule\n")
        f.write("        \\endlastfoot\n")
        
        # Data rows
        for country in countries_sorted:
            country_name = get_country_name(country)
            row_parts = [country_name]
            
            for metric_key, _, format_str in metrics:
                value = country_results[country][metric_key]
                formatted = format_value(value, format_str)
                row_parts.append(formatted)
            
            f.write("        " + " & ".join(row_parts) + " \\\\\n")
        
        f.write("      \\end{longtable}\n")
        f.write("    \\end{sc}\n")
        f.write("  \\end{small}\n")
        f.write("\\end{center}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table with country-level results"
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('analysis/disaggregated/by_country.json'),
        help='Path to by_country.json file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('analysis/country_table.tex'),
        help='Path to output LaTeX file'
    )
    parser.add_argument(
        '--metrics',
        nargs='+',
        help='Custom metrics to include (format: key:display_name:format)'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} models")
    
    # Parse custom metrics if provided
    metrics = None
    if args.metrics:
        metrics = []
        for m in args.metrics:
            parts = m.split(':')
            if len(parts) == 3:
                metrics.append((parts[0], parts[1], parts[2]))
            else:
                raise ValueError(f"Invalid metric format: {m}. Expected 'key:display_name:format'")
    
    # Generate table
    print(f"Generating LaTeX table...")
    generate_latex_table(data, args.output, metrics)
    
    print(f"Table written to {args.output}")
    print(f"Note: Make sure to include \\usepackage{{longtable}} in your LaTeX preamble")


if __name__ == '__main__':
    main()
