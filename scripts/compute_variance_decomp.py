"""
Compute partial R² for 'question content' and 'model choice' from the EMNLP MEM (V2).

Approach: Nakagawa & Schielzeth (2013) for LMMs.
  partial R²_X = sigma2_f_X / (sigma2_f_total + sigma2_epsilon)
  where sigma2_f_X = Var(beta_X * X) = variance of fitted values from predictor X

Run from project root. Requires per_question_norm_acc.csv and majority_class_norm_acc.csv.
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NORM_PATH = ROOT / "analysis/normalized_accuracy/per_question_norm_acc.csv"
MAJ_PATH  = ROOT / "analysis/normalized_accuracy/majority_class_norm_acc.csv"

# ---- MEM V2 residual variance (from mem_v2_normalized_accuracy.txt) ----
SIGMA2_EPS = 0.0711   # Scale from MEM output
SIGMA2_RE  = 0.0      # Group Var ≈ 0 (collapsed)

# ---- MEM V2 fixed-effect coefficients ----
# model: 13 levels (deepseek is reference = 0)
BETA_MODEL = {
    "deepseek":              0.000,
    "gemma3-27b":           -0.033,
    "gpt-oss":              -0.017,
    "llama3.1_70b_base":   -0.115,
    "llama3.1_70b_instruct":-0.031,
    "llama3.1_8b_base":    -0.137,
    "llama3.1_8b_instruct": -0.027,
    "olmo3_32b_base":      -0.104,
    "olmo3_32b_dpo":       -0.121,
    "olmo3_7b_base":       -0.088,
    "olmo3_7b_dpo":        -0.062,
    "qwen3-32b":           -0.002,
    "qwen3-4b":            -0.056,
}

# topic_section: 7 levels (contemporary_issues is reference = 0)
BETA_TOPIC = {
    "contemporary_issues":       0.000,
    "institutional_trust":      -0.035,
    "political_attitudes":      -0.026,
    "political_participation":  -0.012,
    "social_attitudes":         -0.080,
    "values_identity":          -0.065,
    "wellbeing":                -0.029,
}

# region: 20 levels (Caribbean is reference = 0)
BETA_REGION = {
    "Caribbean":         0.000,
    "Central Africa":    0.011,
    "Central America":   0.043,
    "Central Asia":      0.037,
    "East Africa":       0.011,
    "East Asia":         0.037,
    "Eastern Europe":    0.021,
    "Middle East":       0.017,
    "North Africa":      0.018,
    "North America":     0.025,
    "Northern Europe":   0.016,
    "Oceania":           0.018,
    "South America":     0.041,
    "South Asia":        0.001,
    "Southeast Asia":    0.045,
    "Southern Africa":  -0.010,
    "Southern Europe":   0.020,
    "Unknown":           0.038,
    "West Africa":       0.016,
    "Western Europe":    0.013,
}

# modal_share coefficient
BETA_MODAL = 0.051


def group_variance(coeff_dict):
    """Variance of group means (balanced-design approximation)."""
    vals = np.array(list(coeff_dict.values()))
    return vals.var()   # population variance = np.var (ddof=0)


def main():
    # ---- Load data for computing Var(modal_share) ----
    df_norm = pd.read_csv(NORM_PATH).dropna(subset=["norm_acc"])
    df_maj  = pd.read_csv(MAJ_PATH)

    # majority_acc per (survey, target_code) serves as modal_share proxy
    # Use s6m4 profile (richest) or mean across profiles — shouldn't matter
    modal = (df_maj
             .groupby(["survey", "target_code"])["majority_acc"]
             .mean()
             .reset_index()
             .rename(columns={"majority_acc": "modal_share"}))
    var_modal = modal["modal_share"].var(ddof=1)
    print(f"Var(modal_share) from majority_acc: {var_modal:.6f}  "
          f"(mean={modal['modal_share'].mean():.3f}, SD={modal['modal_share'].std():.3f})")

    # ---- Compute sigma2_f for each predictor ----
    sigma2_f_model  = group_variance(BETA_MODEL)
    sigma2_f_topic  = group_variance(BETA_TOPIC)
    sigma2_f_region = group_variance(BETA_REGION)
    sigma2_f_modal  = BETA_MODAL**2 * var_modal

    sigma2_f_total  = sigma2_f_model + sigma2_f_topic + sigma2_f_region + sigma2_f_modal

    sigma2_total = sigma2_f_total + SIGMA2_RE + SIGMA2_EPS

    print(f"\n--- Variance components (sigma^2) ---")
    print(f"  model:         {sigma2_f_model:.6f}")
    print(f"  topic_section: {sigma2_f_topic:.6f}")
    print(f"  region:        {sigma2_f_region:.6f}")
    print(f"  modal_share:   {sigma2_f_modal:.6f}")
    print(f"  fixed_total:   {sigma2_f_total:.6f}")
    print(f"  random_effect: {SIGMA2_RE:.6f}")
    print(f"  residual:      {SIGMA2_EPS:.6f}")
    print(f"  TOTAL:         {sigma2_total:.6f}")

    # ---- Partial R² (Nakagawa formula) ----
    R2_model         = sigma2_f_model  / sigma2_total
    R2_question_cont = (sigma2_f_topic + sigma2_f_modal) / sigma2_total
    R2_region        = sigma2_f_region / sigma2_total
    R2_marginal_all  = sigma2_f_total  / sigma2_total

    print(f"\n--- Partial R² (Nakagawa) ---")
    print(f"  Model choice:     {R2_model*100:.2f}%  (paper claims 1.3%)")
    print(f"  Question content: {R2_question_cont*100:.2f}%  (paper claims 6.0%)")
    print(f"    topic_section:  {(sigma2_f_topic/sigma2_total)*100:.2f}%")
    print(f"    modal_share:    {(sigma2_f_modal/sigma2_total)*100:.2f}%")
    print(f"  Region:           {R2_region*100:.2f}%")
    print(f"  Marginal R² all:  {R2_marginal_all*100:.2f}%")

    # ---- Also compute from the aggregated data directly ----
    print(f"\n--- Simple ANOVA decomposition on per_question_norm_acc.csv ---")
    # Aggregate to per-(question, model) means — average across profiles
    agg = (df_norm.groupby(["survey", "target_code", "model"])["norm_acc"]
                  .mean().reset_index())
    grand_mean = agg["norm_acc"].mean()
    SS_total = ((agg["norm_acc"] - grand_mean) ** 2).sum()
    n = len(agg)
    print(f"  n={n}, grand_mean={grand_mean:.4f}, SS_total={SS_total:.4f}")

    # Between-question SS
    q_means = agg.groupby(["survey", "target_code"])["norm_acc"].mean()
    q_sizes = agg.groupby(["survey", "target_code"])["norm_acc"].count()
    SS_q = ((q_means - grand_mean)**2 * q_sizes).sum()

    # Between-model SS
    m_means = agg.groupby("model")["norm_acc"].mean()
    m_sizes = agg.groupby("model")["norm_acc"].count()
    SS_m = ((m_means - grand_mean)**2 * m_sizes).sum()

    print(f"  SS_question / SS_total = {SS_q/SS_total*100:.1f}%  "
          f"(eta2, not what paper reports)")
    print(f"  SS_model    / SS_total = {SS_m/SS_total*100:.1f}%  "
          f"(eta2, not what paper reports)")

    # ICC approach (variance components)
    K_q = agg.groupby(["survey","target_code"]).ngroups
    MS_q = SS_q / (K_q - 1)
    MS_w_q = (SS_total - SS_q) / (n - K_q)
    n_per_q = n / K_q
    vc_q = max((MS_q - MS_w_q) / n_per_q, 0)
    vc_total_q = vc_q + MS_w_q
    ICC_q = vc_q / vc_total_q if vc_total_q > 0 else 0

    K_m = 13
    MS_m = SS_m / (K_m - 1)
    MS_w_m = (SS_total - SS_m) / (n - K_m)
    n_per_m = n / K_m
    vc_m = max((MS_m - MS_w_m) / n_per_m, 0)
    vc_total_m = vc_m + MS_w_m
    ICC_m = vc_m / vc_total_m if vc_total_m > 0 else 0

    print(f"\n  ICC question = {ICC_q*100:.1f}%")
    print(f"  ICC model    = {ICC_m*100:.1f}%")

    # ---- Summary of what values to use in paper ----
    print(f"\n=== PAPER CLAIMS: question=6%, model=1.3% ===")
    print(f"=== COMPUTED (Nakagawa from MEM):  "
          f"question={R2_question_cont*100:.1f}%, model={R2_model*100:.1f}% ===")


if __name__ == "__main__":
    main()
