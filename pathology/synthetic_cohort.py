# -*- coding: utf-8 -*-
"""
Synthetic Patient Cohort Generator
====================================
Generates a fake but statistically rigorous n=47 TNBC patient dataset
linking TAM nuclear aspect ratio to survival outcome.

Outputs:
  - cohort_data.csv          : per-patient data table
  - kaplan_meier.png         : KM survival curves (high vs low TAM AR)
  - cox_regression.csv       : Cox PH model summary
  - forest_plot.png          : hazard ratio forest plot
  - cohort_table1.csv        : Table 1 style demographics summary
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mannwhitneyu

# Optional: lifelines for KM and Cox (install if not present)
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("lifelines not found — install with: pip install lifelines")
    print("KM and Cox plots will be skipped, CSV output still generated.")

OUTPUT_DIR = "synthetic_cohort_results/"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

STYLE = {
    "font":      "Arial",
    "fontsize":  8,
    "dpi":       300,
    "color_high": "#D62728",   # red = high TAM AR (poor prognosis)
    "color_low":  "#1F77B4",   # blue = low TAM AR (better prognosis)
}

plt.rcParams.update({
    'font.family': STYLE['font'],
    'font.size':   STYLE['fontsize'],
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

np.random.seed(42)   # reproducible fake data


# ---------------------------------------------------------------------------
# Step 1: Generate patient demographics
# ---------------------------------------------------------------------------

N = 47

patient_ids = [f"TNBC-{str(i+1).zfill(3)}" for i in range(N)]

age        = np.random.normal(52, 10, N).clip(28, 78).astype(int)
grade      = np.random.choice([2, 3], N, p=[0.25, 0.75])
stage      = np.random.choice(['II', 'III', 'IV'], N, p=[0.35, 0.45, 0.20])
chemo      = np.random.choice(['Yes', 'No'], N, p=[0.85, 0.15])

# TAM nuclear aspect ratio: drawn from distributions matching our image analysis
# High AR group (hypoxic-like): mean ~1.88, sd ~0.35
# Low AR group (normoxic-like): mean ~1.48, sd ~0.28
# Split roughly 55/45
group_assign = np.random.choice(['High', 'Low'], N, p=[0.55, 0.45])
tam_ar = np.where(
    group_assign == 'High',
    np.random.normal(1.88, 0.35, N).clip(1.2, 3.2),
    np.random.normal(1.48, 0.28, N).clip(1.05, 2.4)
)

# Median split for analysis (in a real paper you'd use the cohort median)
ar_median  = np.median(tam_ar)
ar_group   = np.where(tam_ar >= ar_median, 'High AR', 'Low AR')

# HIF-1alpha IHC score (0-3): correlated with TAM AR
hif1a_score = np.where(
    tam_ar > ar_median,
    np.random.choice([2, 3], N, p=[0.35, 0.65]),
    np.random.choice([0, 1, 2], N, p=[0.40, 0.45, 0.15])
)

# CD68 density (macrophage marker, cells/mm2): higher in hypoxic regions
cd68_density = np.where(
    tam_ar > ar_median,
    np.random.normal(48, 12, N).clip(20, 90),
    np.random.normal(31, 10, N).clip(10, 65)
)

# ---------------------------------------------------------------------------
# Step 2: Generate survival data
# ---------------------------------------------------------------------------
# Survival time in months, censoring at 60 months (5 years)
# High TAM AR → shorter survival (HR ~2.4 relative to low AR)

baseline_survival = np.random.exponential(scale=48, size=N)   # months

# High AR patients survive ~40% shorter on average
hazard_multiplier = np.where(tam_ar > ar_median, 1.0, 1.8)
survival_time = (baseline_survival * hazard_multiplier).clip(1, 120)

# Censoring: ~30% of patients censored at last follow-up
censored       = np.random.choice([0, 1], N, p=[0.30, 0.70])   # 1 = event observed
# Patients who survive past 60 months are censored at 60
survival_time  = np.where(survival_time > 60, 60, survival_time)
censored       = np.where(survival_time == 60, 0, censored)

# ---------------------------------------------------------------------------
# Step 3: Assemble dataframe
# ---------------------------------------------------------------------------

df = pd.DataFrame({
    'Patient_ID':     patient_ids,
    'Age':            age,
    'Grade':          grade,
    'Stage':          stage,
    'Chemotherapy':   chemo,
    'TAM_AR':         np.round(tam_ar, 4),
    'AR_Group':       ar_group,
    'HIF1a_Score':    hif1a_score,
    'CD68_Density':   np.round(cd68_density, 1),
    'Survival_months': np.round(survival_time, 1),
    'Event':          censored,          # 1 = death/recurrence, 0 = censored
})

df.to_csv(os.path.join(OUTPUT_DIR, "cohort_data.csv"), index=False)
print(f"Saved: cohort_data.csv  (n={len(df)})")
print(df[['Patient_ID','Age','Stage','TAM_AR','AR_Group',
          'HIF1a_Score','Survival_months','Event']].to_string(index=False))


# ---------------------------------------------------------------------------
# Step 4: Table 1 — demographics summary
# ---------------------------------------------------------------------------

def summarize_table1(df):
    rows = []
    high = df[df['AR_Group'] == 'High AR']
    low  = df[df['AR_Group'] == 'Low AR']

    def fmt_mean(col):
        return (f"{df[col].mean():.1f} ± {df[col].std():.1f}",
                f"{high[col].mean():.1f} ± {high[col].std():.1f}",
                f"{low[col].mean():.1f} ± {low[col].std():.1f}")

    def fmt_cat(col, val):
        n_all  = (df[col] == val).sum()
        n_high = (high[col] == val).sum()
        n_low  = (low[col] == val).sum()
        return (f"{n_all} ({100*n_all/len(df):.0f}%)",
                f"{n_high} ({100*n_high/len(high):.0f}%)",
                f"{n_low} ({100*n_low/len(low):.0f}%)")

    rows.append(('n', str(len(df)), str(len(high)), str(len(low))))
    rows.append(('Age (mean ± SD)', *fmt_mean('Age')))
    for g in [2, 3]:
        rows.append((f'Grade {g}', *fmt_cat('Grade', g)))
    for s in ['II', 'III', 'IV']:
        rows.append((f'Stage {s}', *fmt_cat('Stage', s)))
    rows.append(('Chemotherapy', *fmt_cat('Chemotherapy', 'Yes')))
    rows.append(('TAM AR (mean ± SD)', *fmt_mean('TAM_AR')))
    rows.append(('HIF-1α score ≥2', *fmt_cat('HIF1a_Score', 2)))

    t1 = pd.DataFrame(rows, columns=['Variable', 'All (n=47)',
                                      'High AR (n=26)', 'Low AR (n=21)'])
    t1.to_csv(os.path.join(OUTPUT_DIR, "cohort_table1.csv"), index=False)
    print(f"\nSaved: cohort_table1.csv")
    print(t1.to_string(index=False))
    return t1

summarize_table1(df)


# ---------------------------------------------------------------------------
# Step 5: Mann-Whitney on TAM AR between groups (sanity check)
# ---------------------------------------------------------------------------

high_ar = df[df['AR_Group'] == 'High AR']['TAM_AR'].values
low_ar  = df[df['AR_Group'] == 'Low AR']['TAM_AR'].values
u_stat, p_mw = mannwhitneyu(high_ar, low_ar, alternative='two-sided')
print(f"\nTAM AR: High={high_ar.mean():.3f} vs Low={low_ar.mean():.3f} "
      f"| Mann-Whitney p={p_mw:.4f}")


# ---------------------------------------------------------------------------
# Step 6: Kaplan-Meier curves
# ---------------------------------------------------------------------------

if HAS_LIFELINES:
    kmf_high = KaplanMeierFitter(label='High TAM AR')
    kmf_low  = KaplanMeierFitter(label='Low TAM AR')

    high_mask = df['AR_Group'] == 'High AR'
    low_mask  = df['AR_Group'] == 'Low AR'

    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    kmf_high.fit(df.loc[high_mask, 'Survival_months'],
                 df.loc[high_mask, 'Event'])
    kmf_high.plot_survival_function(ax=ax, ci_show=True,
                                    color=STYLE['color_high'], linewidth=1.5)

    kmf_low.fit(df.loc[low_mask, 'Survival_months'],
                df.loc[low_mask, 'Event'])
    kmf_low.plot_survival_function(ax=ax, ci_show=True,
                                   color=STYLE['color_low'], linewidth=1.5)

    # Log-rank test p-value
    from lifelines.statistics import logrank_test
    lr = logrank_test(
        df.loc[high_mask, 'Survival_months'], df.loc[low_mask, 'Survival_months'],
        event_observed_A=df.loc[high_mask, 'Event'],
        event_observed_B=df.loc[low_mask, 'Event'])

    p_lr = lr.p_value
    ax.text(0.97, 0.95, f"Log-rank p = {p_lr:.4f}",
            transform=ax.transAxes, ha='right', va='top', fontsize=7)

    ax.set_xlabel('Time (months)', fontsize=8)
    ax.set_ylabel('Survival probability', fontsize=8)
    ax.set_title('Overall survival by TAM nuclear aspect ratio\n'
                 'Triple-negative breast cancer (n=47)', fontsize=8, pad=6)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, frameon=False, loc='upper right',
              bbox_to_anchor=(0.97, 0.88))

    fig.tight_layout()
    km_path = os.path.join(OUTPUT_DIR, "kaplan_meier.png")
    fig.savefig(km_path, dpi=STYLE['dpi'], bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: kaplan_meier.png  (log-rank p={p_lr:.4f})")

    # ---------------------------------------------------------------------------
    # Step 7: Cox proportional hazards model
    # ---------------------------------------------------------------------------

    cox_df = df[['Survival_months', 'Event', 'TAM_AR',
                 'Age', 'HIF1a_Score', 'CD68_Density']].copy()
    # Encode stage numerically
    stage_map = {'II': 2, 'III': 3, 'IV': 4}
    cox_df['Stage_num'] = df['Stage'].map(stage_map)

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_df, duration_col='Survival_months', event_col='Event')
    print("\nCox PH model summary:")
    cph.print_summary()

    cox_summary = cph.summary.reset_index()
    cox_summary.to_csv(os.path.join(OUTPUT_DIR, "cox_regression.csv"), index=False)
    print(f"Saved: cox_regression.csv")

    # ---------------------------------------------------------------------------
    # Step 8: Forest plot of hazard ratios
    # ---------------------------------------------------------------------------

    covariates   = cph.summary.index.tolist()
    hr_values    = cph.summary['exp(coef)'].values
    hr_lower     = cph.summary['exp(coef) lower 95%'].values
    hr_upper     = cph.summary['exp(coef) upper 95%'].values
    p_values     = cph.summary['p'].values

    covariate_labels = {
        'TAM_AR':      'TAM nuclear AR',
        'Age':         'Age (per year)',
        'HIF1a_Score': 'HIF-1α score',
        'CD68_Density':'CD68 density',
        'Stage_num':   'Stage',
    }
    display_labels = [covariate_labels.get(c, c) for c in covariates]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    y_pos = np.arange(len(covariates))

    for i, (hr, lo, hi, pv) in enumerate(zip(hr_values, hr_lower, hr_upper, p_values)):
        color = '#D62728' if hr > 1 else '#1F77B4'
        ax.plot([lo, hi], [i, i], color=color, linewidth=1.2, zorder=2)
        ax.scatter(hr, i, color=color, s=40, zorder=3,
                   marker='D' if pv < 0.05 else 'o')

    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.8, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_labels, fontsize=7)
    ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=8)
    ax.set_title('Multivariable Cox regression: TNBC survival', fontsize=8, pad=6)

    # Add HR and p-value text on the right
    x_max = ax.get_xlim()[1]
    for i, (hr, pv) in enumerate(zip(hr_values, p_values)):
        pstr = f"p={pv:.3f}" if pv >= 0.001 else "p<0.001"
        ax.text(x_max * 1.02, i,
                f"HR={hr:.2f}  {pstr}",
                va='center', ha='left', fontsize=6)

    ax.text(0.98, -0.18,
            '◆ p<0.05   ● p≥0.05',
            transform=ax.transAxes, ha='right', fontsize=6, color='gray')

    fig.tight_layout()
    fp_path = os.path.join(OUTPUT_DIR, "forest_plot.png")
    fig.savefig(fp_path, dpi=STYLE['dpi'], bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: forest_plot.png")

else:
    print("\nInstall lifelines to generate KM curves and Cox model:")
    print("  pip install lifelines")

print("\nAll synthetic cohort outputs saved to:", OUTPUT_DIR)
