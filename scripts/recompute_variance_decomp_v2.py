"""
Recompute partial R² for model choice and question content from the EMNLP V2 MEM.

Strategy: fit sequential MixedLM models on the per-(question, model, profile_type) data
(10,374 rows), adding predictors incrementally and comparing marginal R² using the
Nakagawa & Schielzeth (2013) formula for LMMs:

    R²_marginal = sigma2_f / (sigma2_f + sigma2_alpha + sigma2_epsilon)

where sigma2_f = Var(fitted fixed-effect predictions).

Available predictors:
  - model (13 levels, from per_question_norm_acc.csv)
  - modal_share (from majority_class_norm_acc.csv) — proxy for question difficulty
  - survey as random effect (7 groups)

NOT available here: topic_section, region (would need question metadata + respondent mapping).
The partial R² for topic_section is included via the saved V2 MEM coefficients.

Outputs:
  - Incremental R² table (ΔR² when adding each predictor sequentially)
  - Nakagawa partial R² from the saved V2 MEM coefficients (with all predictors)
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NORM_PATH = ROOT / "analysis/normalized_accuracy/per_question_norm_acc.csv"
MAJ_PATH  = ROOT / "analysis/normalized_accuracy/majority_class_norm_acc.csv"

try:
    import statsmodels.formula.api as smf
    HAS_SM = True
except ImportError:
    HAS_SM = False
    print("statsmodels not available; using coefficient-based computation only.")


# ---- Saved V2 MEM coefficients (from mem_v2_normalized_accuracy.txt) --------
BETA_MODEL = np.array([0, -0.033, -0.017, -0.115, -0.031, -0.137,
                       -0.027, -0.104, -0.121, -0.088, -0.062, -0.002, -0.056])
BETA_TOPIC = np.array([0, -0.035, -0.026, -0.012, -0.080, -0.065, -0.029])
BETA_REGION = np.array([0, 0.011, 0.043, 0.037, 0.011, 0.037, 0.021, 0.017,
                        0.018, 0.025, 0.016, 0.018, 0.041, 0.001, 0.045,
                        -0.010, 0.020, 0.038, 0.016, 0.013])
BETA_MODAL = 0.051
SIGMA2_EPS_V2 = 0.0711

# Topic section proportions (from paper taxonomy: 267 questions)
TOPIC_PROPS = np.array([23, 75, 52, 20, 38, 30, 29]) / 267  # CI, PA, IT, PP, SA, V&I, WB


def weighted_group_var(betas, weights=None):
    """Variance of group effects (weighted by group proportions)."""
    if weights is None:
        weights = np.ones(len(betas)) / len(betas)
    weights = np.array(weights) / weights.sum()
    mean = np.dot(weights, betas)
    return np.dot(weights, (betas - mean) ** 2)


def nakagawa_r2(sigma2_f_dict, sigma2_eps, sigma2_re=0.0):
    """
    Compute marginal R² for each component via Nakagawa formula.
    sigma2_f_dict: {name: sigma2_f_X}
    Returns dict of R² values and total R².
    """
    sigma2_f_total = sum(sigma2_f_dict.values())
    sigma2_total = sigma2_f_total + sigma2_re + sigma2_eps
    r2 = {k: v / sigma2_total for k, v in sigma2_f_dict.items()}
    r2["_total_marginal"] = sigma2_f_total / sigma2_total
    r2["_sigma2_total"] = sigma2_total
    r2["_sigma2_f_total"] = sigma2_f_total
    return r2


def fit_models(df):
    """Fit sequence of MixedLM models, return sigma2_f for each predictor."""
    if not HAS_SM:
        return None

    # Encode model as categorical
    df = df.copy()
    df["model_cat"] = df["model"].astype("category")
    df["survey_cat"] = df["survey"].astype("category")

    results = {}

    def run(formula, data=df):
        m = smf.mixedlm(formula, data=data, groups=data["survey_cat"])
        try:
            return m.fit(reml=True, method="lbfgs")
        except Exception:
            return m.fit(reml=False)

    # Null model
    m0 = run("norm_acc ~ 1")
    sigma2_null = m0.scale + m0.cov_re.values[0, 0] if m0.cov_re.shape[0] > 0 else m0.scale
    sigma2_null = m0.scale  # use residual scale (RE collapsed to 0 typically)
    results["null_scale"] = m0.scale

    # Model with only modal_share (question difficulty)
    m_modal = run("norm_acc ~ modal_share")
    sigma2_modal_resid = m_modal.scale
    delta_modal = sigma2_null - sigma2_modal_resid
    results["modal_scale"] = sigma2_modal_resid

    # Model with only model identity
    m_model = run("norm_acc ~ C(model_cat)")
    sigma2_model_resid = m_model.scale
    delta_model = sigma2_null - sigma2_model_resid
    results["model_scale"] = sigma2_model_resid

    # Full model (modal_share + model)
    m_full = run("norm_acc ~ modal_share + C(model_cat)")
    sigma2_full = m_full.scale
    results["full_scale"] = sigma2_full

    # Incremental R² (type I): adding each predictor to null
    # R²_X = (sigma2_null - sigma2_with_X) / sigma2_null
    r2_modal = delta_modal / sigma2_null if sigma2_null > 0 else 0
    r2_model = delta_model / sigma2_null if sigma2_null > 0 else 0
    r2_full = (sigma2_null - sigma2_full) / sigma2_null if sigma2_null > 0 else 0

    # Nakagawa marginal R² from fitted models
    # sigma2_f_X = sigma2_null - sigma2_model_with_only_X
    # Then R²_X = sigma2_f_X / sigma2_null
    results["r2_modal_marginal"] = r2_modal
    results["r2_model_marginal"] = r2_model
    results["r2_full_marginal"] = r2_full

    return results


def main():
    # Load data
    df = pd.read_csv(NORM_PATH).dropna(subset=["norm_acc"])
    maj = pd.read_csv(MAJ_PATH)

    # Add modal_share (mean majority_acc per question across profiles)
    modal = (maj.groupby(["survey", "target_code"])["majority_acc"]
               .mean().reset_index()
               .rename(columns={"majority_acc": "modal_share"}))
    df = df.merge(modal, on=["survey", "target_code"], how="left")
    df = df.dropna(subset=["modal_share"])

    print(f"Dataset: {len(df)} rows, {df.groupby(['survey','target_code']).ngroups} questions, "
          f"{df['model'].nunique()} models, {df['survey'].nunique()} surveys")

    # ---- Method 1: Nakagawa from saved V2 MEM coefficients ----
    print("\n=== Method 1: Nakagawa partial R² from saved V2 MEM coefficients ===")

    var_modal = df["modal_share"].var(ddof=1)

    sigma2_f = {
        "model":         weighted_group_var(BETA_MODEL),       # balanced (each model = 1/13)
        "topic_section": weighted_group_var(BETA_TOPIC, TOPIC_PROPS),  # unbalanced by section size
        "region":        weighted_group_var(BETA_REGION),      # balanced (each region = 1/20)
        "modal_share":   BETA_MODAL**2 * var_modal,
    }

    r2 = nakagawa_r2(sigma2_f, SIGMA2_EPS_V2)

    print(f"\nVariance components (sigma²):")
    for k, v in sigma2_f.items():
        print(f"  {k:20s}: {v:.6f}")
    print(f"  {'total fixed':20s}: {r2['_sigma2_f_total']:.6f}")
    print(f"  {'residual (V2)':20s}: {SIGMA2_EPS_V2:.6f}")
    print(f"  {'TOTAL':20s}: {r2['_sigma2_total']:.6f}")

    print(f"\nPartial R² (Nakagawa):")
    print(f"  Model choice:              {r2['model']*100:.2f}%")
    print(f"  Question content:")
    print(f"    modal_share only:        {r2['modal_share']*100:.2f}%")
    print(f"    topic_section only:      {r2['topic_section']*100:.2f}%")
    print(f"    combined (modal+topic):  {(r2['modal_share']+r2['topic_section'])*100:.2f}%")
    print(f"  Region:                    {r2['region']*100:.2f}%")
    print(f"  Marginal R² all fixed:     {r2['_total_marginal']*100:.2f}%")

    # ---- Method 2: Fit actual models (incremental R²) ----
    if HAS_SM:
        print("\n=== Method 2: Incremental R² from fitted MixedLM models ===")
        print("(Using per-question×model×profile data; no region or topic_section)")
        res = fit_models(df)
        if res:
            print(f"\nNull model residual scale:          {res['null_scale']:.6f}")
            print(f"modal_share-only residual scale:     {res['modal_scale']:.6f}")
            print(f"model-only residual scale:           {res['model_scale']:.6f}")
            print(f"Full (modal+model) residual scale:   {res['full_scale']:.6f}")
            print(f"\nIncremental R² (vs null):")
            print(f"  modal_share (question):  {res['r2_modal_marginal']*100:.2f}%")
            print(f"  model choice:            {res['r2_model_marginal']*100:.2f}%")
            print(f"  full model:              {res['r2_full_marginal']*100:.2f}%")
    else:
        print("\nSkipping Method 2 (statsmodels not installed).")

    # ---- Summary ----
    model_r2 = r2['model'] * 100
    qcont_r2 = (r2['modal_share'] + r2['topic_section']) * 100
    print(f"\n=== UPDATED NUMBERS FOR PAPER ===")
    print(f"  Model choice:     {model_r2:.1f}%  (was 1.3% in ICML/raw-accuracy analysis)")
    print(f"  Question content: {qcont_r2:.1f}%  (was 6.0% in ICML/raw-accuracy analysis)")
    print(f"\nNote: 'question content' = modal_share + topic_section fixed effects.")
    print(f"      Region explains an additional {r2['region']*100:.1f}%.")
    print(f"      Total marginal R² = {r2['_total_marginal']*100:.1f}%.")


if __name__ == "__main__":
    main()
