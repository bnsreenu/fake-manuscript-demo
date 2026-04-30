# -*- coding: utf-8 -*-
"""
XRM Particle Morphology Analysis: NMC811 Cathode Degradation
=============================================================
Generates synthetic per-particle morphology data for low-cycle and
high-cycle NMC811 cathode particles, mimicking metrics extracted from
X-ray microscopy (XRM) tomography segmentation.

Metrics per particle (as would be measured from XRM segmentation):
  - crack_density     : total crack length / particle volume (um^-2)
  - crack_vol_frac    : crack volume / total particle volume (%)
  - porosity          : intraparticle pore volume fraction (%)
  - particle_diameter : equivalent sphere diameter (um)
  - n_cracks          : number of distinct crack segments detected
  - delamination_frac : fraction of particle surface showing delamination (%)

Outputs (saved to xrm_results/):
  - particle_data.csv         : full per-particle dataset
  - morphology_summary.csv    : group summary statistics
  - stats_results.csv         : Mann-Whitney U + Cohen's d per metric
  - violin_crack_density.png  : publication-ready violin plot
  - violin_porosity.png       : publication-ready violin plot
  - correlation_plot.png      : crack density vs capacity fade correlation
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

OUTPUT_DIR = "xrm_results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)

STYLE = {
    "font":       "Arial",
    "fontsize":   8,
    "dpi":        300,
    "color_low":  "#1F77B4",   # blue = low cycle (pristine)
    "color_high": "#D62728",   # red  = high cycle (degraded)
    "alpha":      0.65,
}

plt.rcParams.update({
    'font.family': STYLE['font'],
    'font.size':   STYLE['fontsize'],
    'axes.spines.top':   False,
    'axes.spines.right': False,
})


# ---------------------------------------------------------------------------
# Step 1: Generate synthetic particle data
# ---------------------------------------------------------------------------
# Based on published XRM studies of cycled NMC811:
#   Mu et al., Nano Energy 2018; Xu et al., ACS Nano 2020
#
# Low cycle group:  cells cycled 50 times   (n=32 particles analyzed)
# High cycle group: cells cycled 300 times  (n=35 particles analyzed)

N_LOW  = 32
N_HIGH = 35

# Particle diameter: similar in both groups (same batch of material)
diameter_low  = np.random.normal(11.2, 2.4, N_LOW).clip(4, 22)
diameter_high = np.random.normal(10.8, 2.6, N_HIGH).clip(4, 22)

# Crack density (um^-2): low cycle near zero, high cycle substantially higher
crack_density_low  = np.random.gamma(shape=1.2, scale=0.008, size=N_LOW).clip(0, 0.05)
crack_density_high = np.random.gamma(shape=3.5, scale=0.018, size=N_HIGH).clip(0.01, 0.15)

# Crack volume fraction (%): correlated with crack density
crack_vf_low  = crack_density_low  * np.random.normal(8.5, 1.2, N_LOW).clip(5, 12)
crack_vf_high = crack_density_high * np.random.normal(9.1, 1.4, N_HIGH).clip(5, 14)

# Intraparticle porosity (%): increases with cracking
porosity_low  = np.random.normal(1.8, 0.6, N_LOW).clip(0.4, 3.5)
porosity_high = np.random.normal(4.9, 1.2, N_HIGH).clip(1.5, 9.0)

# Number of distinct crack segments
n_cracks_low  = np.random.poisson(lam=1.8, size=N_LOW)
n_cracks_high = np.random.poisson(lam=7.4, size=N_HIGH)

# Surface delamination fraction (%)
delam_low  = np.random.gamma(shape=1.1, scale=1.5, size=N_LOW).clip(0, 8)
delam_high = np.random.gamma(shape=2.8, scale=4.2, size=N_HIGH).clip(2, 35)

# Capacity retention at end of cycling (% of initial): correlated with crack density
# Low cycle: 94-98%, high cycle: 72-88%
cap_low  = np.random.normal(96.2, 1.4, N_LOW).clip(91, 99)
cap_high = 100 - (crack_density_high * 180 + np.random.normal(0, 2, N_HIGH)).clip(12, 30)

# Assemble dataframe
df_low = pd.DataFrame({
    'group':             'Low cycle (50 cyc)',
    'particle_id':       [f'LC-{i+1:03d}' for i in range(N_LOW)],
    'diameter_um':       np.round(diameter_low, 2),
    'crack_density':     np.round(crack_density_low, 5),
    'crack_vol_frac':    np.round(crack_vf_low, 3),
    'porosity':          np.round(porosity_low, 3),
    'n_cracks':          n_cracks_low,
    'delamination_pct':  np.round(delam_low, 2),
    'capacity_retention':np.round(cap_low, 1),
})

df_high = pd.DataFrame({
    'group':             'High cycle (300 cyc)',
    'particle_id':       [f'HC-{i+1:03d}' for i in range(N_HIGH)],
    'diameter_um':       np.round(diameter_high, 2),
    'crack_density':     np.round(crack_density_high, 5),
    'crack_vol_frac':    np.round(crack_vf_high, 3),
    'porosity':          np.round(porosity_high, 3),
    'n_cracks':          n_cracks_high,
    'delamination_pct':  np.round(delam_high, 2),
    'capacity_retention':np.round(cap_high, 1),
})

df = pd.concat([df_low, df_high], ignore_index=True)
df.to_csv(os.path.join(OUTPUT_DIR, "particle_data.csv"), index=False)
print(f"Saved particle_data.csv  (n_low={N_LOW}, n_high={N_HIGH})")


# ---------------------------------------------------------------------------
# Step 2: Summary statistics
# ---------------------------------------------------------------------------

metrics = ['crack_density', 'crack_vol_frac', 'porosity', 'n_cracks', 'delamination_pct']

summary_rows = []
for grp, gdf in df.groupby('group'):
    row = {'Group': grp, 'N': len(gdf)}
    for m in metrics:
        row[f'{m}_mean'] = round(gdf[m].mean(), 4)
        row[f'{m}_std']  = round(gdf[m].std(), 4)
        row[f'{m}_median'] = round(gdf[m].median(), 4)
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "morphology_summary.csv"), index=False)
print("Saved morphology_summary.csv")
print(summary_df.to_string(index=False))


# ---------------------------------------------------------------------------
# Step 3: Statistics — Mann-Whitney U + Cohen's d
# ---------------------------------------------------------------------------

def cohens_d(a, b):
    pooled = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else np.nan

stat_rows = []
low_vals  = df[df['group'] == 'Low cycle (50 cyc)']
high_vals = df[df['group'] == 'High cycle (300 cyc)']

print("\n--- Statistical comparisons (Low cycle vs High cycle) ---")
for m in metrics:
    a = low_vals[m].values
    b = high_vals[m].values
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    d = cohens_d(a, b)
    stat_rows.append({
        'Metric': m,
        'Mean_low':  round(a.mean(), 4),
        'Mean_high': round(b.mean(), 4),
        'MannWhitneyU': u,
        'p_value': p,
        'cohens_d': round(d, 4),
    })
    print(f"  {m}: low={a.mean():.4f}, high={b.mean():.4f}, "
          f"U={u:.0f}, p={p:.2e}, d={d:.3f}")

stats_df = pd.DataFrame(stat_rows)
stats_df.to_csv(os.path.join(OUTPUT_DIR, "stats_results.csv"), index=False)
print("Saved stats_results.csv")


# ---------------------------------------------------------------------------
# Step 4: Plots
# ---------------------------------------------------------------------------

def pval_label(p):
    if p < 0.0001: return "p < 0.0001"
    elif p < 0.001: return "p < 0.001"
    elif p < 0.01:  return "p < 0.01"
    elif p < 0.05:  return "p < 0.05"
    else:           return f"p = {p:.3f}"


def violin_plot(metric, ylabel, filename, unit=""):
    low_data  = low_vals[metric].values
    high_data = high_vals[metric].values

    fig, ax = plt.subplots(figsize=(3.5, 3.8))
    parts = ax.violinplot([low_data, high_data], positions=[0, 1],
                          showmedians=True, showextrema=False)

    colors = [STYLE['color_low'], STYLE['color_high']]
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(STYLE['alpha'])
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)

    for i, (d, color) in enumerate(zip([low_data, high_data], colors)):
        jitter = np.random.normal(0, 0.06, size=len(d))
        ax.scatter(i + jitter, d, s=4, color=color, alpha=0.4, linewidths=0)

    # p-value bracket
    u, p = stats.mannwhitneyu(low_data, high_data, alternative='two-sided')
    y_top = max(max(low_data), max(high_data))
    y_br  = y_top * 1.08
    h     = y_top * 0.04
    ax.plot([0, 0, 1, 1], [y_br, y_br+h, y_br+h, y_br],
            lw=0.8, color='black')
    ax.text(0.5, y_br + h * 1.2, pval_label(p),
            ha='center', va='bottom', fontsize=7)

    # n= labels inside violin
    y_min = ax.get_ylim()[0]
    y_max = ax.get_ylim()[1]
    y_n   = y_min + 0.04 * (y_max - y_min)
    for i, (d, color) in enumerate(zip([low_data, high_data], colors)):
        ax.text(i, y_n, f'n={len(d)}',
                ha='center', va='bottom', fontsize=6,
                color='white', fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Low cycle\n(50 cyc)', 'High cycle\n(300 cyc)'], fontsize=7)
    ax.set_ylabel(f'{ylabel}{" (" + unit + ")" if unit else ""}', fontsize=8)
    ax.set_title(f'NMC811 cathode: {ylabel}', fontsize=8, pad=8)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out, dpi=STYLE['dpi'], bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}")


violin_plot('crack_density',  'Crack density',        'violin_crack_density.png',   unit='$\\mu m^{-2}$')
violin_plot('porosity',       'Intraparticle porosity','violin_porosity.png',        unit='%')
violin_plot('crack_vol_frac', 'Crack volume fraction', 'violin_crack_vol_frac.png',  unit='%')
violin_plot('delamination_pct','Surface delamination', 'violin_delamination.png',    unit='%')


# ---------------------------------------------------------------------------
# Step 5: Correlation — crack density vs capacity retention
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(4.0, 3.5))

for grp, color, marker in [
    ('Low cycle (50 cyc)',   STYLE['color_low'],  'o'),
    ('High cycle (300 cyc)', STYLE['color_high'], 's'),
]:
    gdf = df[df['group'] == grp]
    ax.scatter(gdf['crack_density'], gdf['capacity_retention'],
               c=color, marker=marker, s=20, alpha=0.7,
               label=grp.split(' (')[0])

# Pearson r on full dataset
r, p_r = stats.pearsonr(df['crack_density'], df['capacity_retention'])
x_line = np.linspace(df['crack_density'].min(), df['crack_density'].max(), 100)
slope, intercept = np.polyfit(df['crack_density'], df['capacity_retention'], 1)
ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=0.8, alpha=0.6)
ax.text(0.97, 0.97, f'r = {r:.3f}\n{pval_label(p_r)}',
        transform=ax.transAxes, ha='right', va='top', fontsize=7)

ax.set_xlabel('Crack density (um\u207B\u00B2)', fontsize=8)
ax.set_ylabel('Capacity retention (%)', fontsize=8)
ax.set_title('Crack density vs capacity retention\nNMC811 cathode (n=67 particles)', fontsize=8, pad=6)
ax.legend(fontsize=7, frameon=False)

fig.tight_layout()
corr_path = os.path.join(OUTPUT_DIR, "correlation_crack_capacity.png")
fig.savefig(corr_path, dpi=STYLE['dpi'], bbox_inches='tight')
plt.close(fig)
print(f"Saved: correlation_crack_capacity.png")

print(f"\nAll XRM morphology outputs saved to: {OUTPUT_DIR}")
print("\nKey results:")
for row in stat_rows:
    print(f"  {row['Metric']}: p={row['p_value']:.2e}, d={row['cohens_d']:.3f}")
