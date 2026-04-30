# -*- coding: utf-8 -*-
"""
Electrochemical Data Generator: NMC811 Cycling Degradation
===========================================================
Generates synthetic but physically realistic electrochemical data
for NMC811 half-cells cycled to 50 (low) and 300 (high) cycles.

Physics basis:
  - Capacity fade: linear + SEI growth term (sqrt dependence on cycle)
  - Coulombic efficiency: asymptotic approach to ~99.5% after formation
  - EIS: Randles circuit with frequency-dependent impedance
    Z = Rs + Rct/(1 + jw*Rct*Cdl) + Zw
    Parameters shift with cycle count as SEI grows and Rct increases

Outputs (saved to electrochemical_results/):
  - cycling_data.csv           : per-cycle capacity and CE for each cell
  - eis_data.csv               : EIS spectra at 4 cycle checkpoints
  - capacity_fade_curves.png   : publication-ready cycling performance plot
  - coulombic_efficiency.png   : CE vs cycle number
  - eis_nyquist.png            : Nyquist plot at cycle 1, 100, 200, 300
  - dqdv_curves.png            : differential capacity (dQ/dV) at checkpoints
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

OUTPUT_DIR = "electrochemical_results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(123)

STYLE = {
    "font":       "Arial",
    "fontsize":   8,
    "dpi":        300,
    "alpha":      0.85,
}

plt.rcParams.update({
    'font.family': STYLE['font'],
    'font.size':   STYLE['fontsize'],
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

# Color palette for cycle checkpoints
CYCLE_COLORS = {
    1:   '#2166AC',   # deep blue
    100: '#4DAF4A',   # green
    200: '#FF7F00',   # orange
    300: '#D62728',   # red
}

N_CELLS_LOW  = 5   # replicate cells, low cycle group
N_CELLS_HIGH = 5   # replicate cells, high cycle group
MAX_CYCLES   = 300
CHECKPOINTS  = [1, 100, 200, 300]


# ---------------------------------------------------------------------------
# Step 1: Capacity fade model
# ---------------------------------------------------------------------------
# Q(n) = Q0 * (1 - a*n - b*sqrt(n)) + noise
# a: linear degradation (electrolyte oxidation, Li plating)
# b: sqrt term (SEI growth, diffusion-limited)
# High cycle cells have higher a and b

Q0 = 200.0   # mAh/g, typical NMC811 initial discharge capacity

def capacity_curve(n_cycles, a, b, noise_std=0.4):
    n = np.arange(1, n_cycles + 1)
    Q = Q0 * (1 - a * n - b * np.sqrt(n))
    Q = Q + np.random.normal(0, noise_std, len(n))
    return Q.clip(Q0 * 0.5, Q0)   # floor at 50% capacity

def ce_curve(n_cycles, ce_plateau=99.52, formation_cycles=3, noise_std=0.04):
    n = np.arange(1, n_cycles + 1)
    # Formation cycles: CE rises from ~85% to plateau
    ce = ce_plateau - (ce_plateau - 85) * np.exp(-n / formation_cycles)
    # Late cycle: slight CE decrease as degradation accelerates
    ce = ce - 0.0008 * np.maximum(n - 150, 0)
    ce = ce + np.random.normal(0, noise_std, len(n))
    return ce.clip(84, 99.9)


# Parameters: low cycle cells degrade more slowly
params_low  = dict(a=0.000120, b=0.000650, noise_std=0.40)
params_high = dict(a=0.000280, b=0.001200, noise_std=0.55)

all_rows = []
for cell_i in range(N_CELLS_LOW):
    Q = capacity_curve(MAX_CYCLES, **params_low)
    CE = ce_curve(MAX_CYCLES, ce_plateau=np.random.normal(99.54, 0.04))
    for cyc in range(MAX_CYCLES):
        all_rows.append({
            'cell_id':  f'LC-cell{cell_i+1:02d}',
            'group':    'Low cycle',
            'cycle':    cyc + 1,
            'capacity': round(Q[cyc], 3),
            'capacity_retention': round(Q[cyc] / Q[0] * 100, 3),
            'coulombic_efficiency': round(CE[cyc], 4),
        })

for cell_i in range(N_CELLS_HIGH):
    Q = capacity_curve(MAX_CYCLES, **params_high)
    CE = ce_curve(MAX_CYCLES, ce_plateau=np.random.normal(99.48, 0.05))
    for cyc in range(MAX_CYCLES):
        all_rows.append({
            'cell_id':  f'HC-cell{cell_i+1:02d}',
            'group':    'High cycle',
            'cycle':    cyc + 1,
            'capacity': round(Q[cyc], 3),
            'capacity_retention': round(Q[cyc] / Q[0] * 100, 3),
            'coulombic_efficiency': round(CE[cyc], 4),
        })

cycling_df = pd.DataFrame(all_rows)
cycling_df.to_csv(os.path.join(OUTPUT_DIR, "cycling_data.csv"), index=False)
print(f"Saved cycling_data.csv  ({len(cycling_df)} rows)")


# ---------------------------------------------------------------------------
# Step 2: EIS model — Randles circuit with Warburg element
# ---------------------------------------------------------------------------
# Z_total = Rs + 1/(1/Rct + j*w*Cdl) + Aw/sqrt(w) * (1-j)
#
# Parameters evolve with cycle number:
#   Rs  (solution resistance): ~constant
#   Rsei (SEI resistance): grows with sqrt(cycle)
#   Rct (charge transfer resistance): increases with cycling
#   Cdl (double layer capacitance): slight decrease
#   Aw  (Warburg coefficient): increases as diffusion path lengthens

def randles_impedance(freq, Rs, Rsei, Csei, Rct, Cdl, Aw):
    w = 2 * np.pi * freq
    # SEI element
    Z_sei = Rsei / (1 + 1j * w * Rsei * Csei)
    # Charge transfer + double layer
    Z_ct  = Rct / (1 + 1j * w * Rct * Cdl)
    # Warburg (semi-infinite diffusion)
    Z_w   = Aw / np.sqrt(w) * (1 - 1j)
    return Rs + Z_sei + Z_ct + Z_w

# Base parameters at cycle 1
base_params = dict(
    Rs   = 2.1,      # ohm
    Rsei = 3.5,
    Csei = 8e-5,
    Rct  = 12.0,
    Cdl  = 2.5e-4,
    Aw   = 15.0,
)

def params_at_cycle(cycle, group='low'):
    scale = 1.0 if group == 'low' else 1.35
    p = base_params.copy()
    p['Rsei'] += scale * 0.065 * np.sqrt(cycle)
    p['Rct']  += scale * 0.18 * cycle ** 0.6
    p['Aw']   += scale * 0.040 * cycle
    p['Cdl']  *= max(0.6, 1 - scale * 0.0008 * cycle)
    return p

freq = np.logspace(5, -2, 80)   # 100 kHz to 0.01 Hz

eis_rows = []
for cycle in CHECKPOINTS:
    for group in ['low', 'high']:
        p = params_at_cycle(cycle, group)
        Z = randles_impedance(freq, **p)
        noise_r = np.random.normal(0, 0.08, len(freq))
        noise_i = np.random.normal(0, 0.08, len(freq))
        for i, f in enumerate(freq):
            eis_rows.append({
                'group':     'Low cycle' if group == 'low' else 'High cycle',
                'cycle':     cycle,
                'freq_hz':   round(f, 6),
                'Z_real':    round(Z[i].real + noise_r[i], 4),
                'Z_imag':    round(-Z[i].imag + noise_i[i], 4),  # convention: -Im(Z)
            })

eis_df = pd.DataFrame(eis_rows)
eis_df.to_csv(os.path.join(OUTPUT_DIR, "eis_data.csv"), index=False)
print(f"Saved eis_data.csv")


# ---------------------------------------------------------------------------
# Step 3: dQ/dV curves
# ---------------------------------------------------------------------------
# NMC811 characteristic peaks:
#   Charge peaks at ~3.68V (H1-M), ~4.0V (M-H2), ~4.18V (H2-H3)
#   H2-H3 peak diminishes with cycling (structural transition suppression)

def dqdv_curve(voltage, cycle, group='low'):
    scale_fade = 0.0012 if group == 'high' else 0.0006
    # Three peaks: H1-M, M-H2, H2-H3
    peak1 = 45  * np.exp(-((voltage - 3.675) ** 2) / (2 * 0.018**2))
    peak2 = 38  * np.exp(-((voltage - 3.995) ** 2) / (2 * 0.022**2))
    # H2-H3 peak fades with cycling
    h2h3_amp = max(5, 62 - scale_fade * cycle * 280)
    peak3 = h2h3_amp * np.exp(-((voltage - 4.195) ** 2) / (2 * 0.015**2))
    noise = np.random.normal(0, 0.8, len(voltage))
    return peak1 + peak2 + peak3 + noise

voltage = np.linspace(3.0, 4.3, 500)


# ---------------------------------------------------------------------------
# Step 4: Plots
# ---------------------------------------------------------------------------

# --- 4A: Capacity retention curves ---
fig, ax = plt.subplots(figsize=(5.0, 3.5))

for cell_id in cycling_df[cycling_df['group'] == 'Low cycle']['cell_id'].unique():
    cdf = cycling_df[cycling_df['cell_id'] == cell_id]
    ax.plot(cdf['cycle'], cdf['capacity_retention'],
            color=CYCLE_COLORS[1], alpha=0.35, linewidth=0.8)

for cell_id in cycling_df[cycling_df['group'] == 'High cycle']['cell_id'].unique():
    cdf = cycling_df[cycling_df['cell_id'] == cell_id]
    ax.plot(cdf['cycle'], cdf['capacity_retention'],
            color=CYCLE_COLORS[300], alpha=0.35, linewidth=0.8)

# Mean lines
for grp, color, label in [
    ('Low cycle',  CYCLE_COLORS[1],   'Low rate (0.5C)'),
    ('High cycle', CYCLE_COLORS[300], 'High rate (2C)'),
]:
    mean_ret = cycling_df[cycling_df['group'] == grp].groupby('cycle')['capacity_retention'].mean()
    ax.plot(mean_ret.index, mean_ret.values, color=color, linewidth=1.8, label=label)

ax.axhline(80, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax.text(305, 80.5, '80% threshold', fontsize=6, color='gray', va='bottom')
ax.set_xlabel('Cycle number', fontsize=8)
ax.set_ylabel('Capacity retention (%)', fontsize=8)
ax.set_title('NMC811 capacity retention during cycling', fontsize=8, pad=6)
ax.legend(fontsize=7, frameon=False)
ax.set_xlim(0, 305)
ax.set_ylim(55, 102)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "capacity_fade_curves.png"),
            dpi=STYLE['dpi'], bbox_inches='tight')
plt.close(fig)
print("Saved capacity_fade_curves.png")


# --- 4B: Coulombic efficiency ---
fig, ax = plt.subplots(figsize=(5.0, 3.0))

for grp, color, label in [
    ('Low cycle',  CYCLE_COLORS[1],   'Low rate (0.5C)'),
    ('High cycle', CYCLE_COLORS[300], 'High rate (2C)'),
]:
    mean_ce = cycling_df[cycling_df['group'] == grp].groupby('cycle')['coulombic_efficiency'].mean()
    # Only plot from cycle 3 onward (formation cycles excluded)
    ax.plot(mean_ce.index[3:], mean_ce.values[3:],
            color=color, linewidth=1.5, label=label)

ax.set_xlabel('Cycle number', fontsize=8)
ax.set_ylabel('Coulombic efficiency (%)', fontsize=8)
ax.set_title('Coulombic efficiency during cycling', fontsize=8, pad=6)
ax.legend(fontsize=7, frameon=False)
ax.set_xlim(3, 305)
ax.set_ylim(98.8, 99.8)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "coulombic_efficiency.png"),
            dpi=STYLE['dpi'], bbox_inches='tight')
plt.close(fig)
print("Saved coulombic_efficiency.png")


# --- 4C: EIS Nyquist plots ---
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=False)

for ax_idx, (group, grp_label) in enumerate([
    ('Low cycle',  'Low rate (0.5C)'),
    ('High cycle', 'High rate (2C)'),
]):
    ax = axes[ax_idx]
    for cycle in CHECKPOINTS:
        edf = eis_df[(eis_df['group'] == group) & (eis_df['cycle'] == cycle)]
        # Only plot the relevant frequency range (semicircle region)
        mask = edf['Z_imag'] >= -2
        ax.plot(edf.loc[mask, 'Z_real'], edf.loc[mask, 'Z_imag'],
                color=CYCLE_COLORS[cycle], linewidth=1.2,
                label=f'Cycle {cycle}')
        # Mark high-freq intercept
        ax.scatter(edf['Z_real'].iloc[0], edf['Z_imag'].iloc[0],
                   color=CYCLE_COLORS[cycle], s=12, zorder=5)

    ax.set_xlabel("Z' (Ohm)", fontsize=8)
    ax.set_ylabel("-Z'' (Ohm)", fontsize=8)
    ax.set_title(f'EIS Nyquist: {grp_label}', fontsize=8, pad=6)
    ax.legend(fontsize=6, frameon=False)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=-2)
    ax.set_aspect('equal', adjustable='datalim')

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "eis_nyquist.png"),
            dpi=STYLE['dpi'], bbox_inches='tight')
plt.close(fig)
print("Saved eis_nyquist.png")


# --- 4D: dQ/dV curves ---
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=True)

for ax_idx, (group, grp_label) in enumerate([
    ('low',  'Low rate (0.5C)'),
    ('high', 'High rate (2C)'),
]):
    ax = axes[ax_idx]
    for cycle in CHECKPOINTS:
        dqdv = dqdv_curve(voltage, cycle, group)
        ax.plot(voltage, dqdv, color=CYCLE_COLORS[cycle],
                linewidth=1.2, label=f'Cycle {cycle}')

    ax.set_xlabel('Voltage (V vs Li/Li+)', fontsize=8)
    if ax_idx == 0:
        ax.set_ylabel('dQ/dV (mAh $g^{-1}$ $V^{-1}$)', fontsize=8)
    ax.set_title(f'Differential capacity: {grp_label}', fontsize=8, pad=6)
    ax.legend(fontsize=6, frameon=False)
    ax.set_xlim(3.0, 4.3)
    ax.set_ylim(-5, 85)

    # Annotate phase transitions
    for v, label in [(3.675, 'H1-M'), (3.995, 'M-H2'), (4.195, 'H2-H3')]:
        ax.axvline(v, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
        ax.text(v, 78, label, fontsize=5, ha='center', color='gray')

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "dqdv_curves.png"),
            dpi=STYLE['dpi'], bbox_inches='tight')
plt.close(fig)
print("Saved dqdv_curves.png")


# ---------------------------------------------------------------------------
# Step 5: Summary table of electrochemical metrics at end of cycling
# ---------------------------------------------------------------------------

final_rows = []
for grp in ['Low cycle', 'High cycle']:
    final_cycle = cycling_df[
        (cycling_df['group'] == grp) & (cycling_df['cycle'] == MAX_CYCLES)
    ]
    final_rows.append({
        'Group': grp,
        'N_cells': final_cycle['cell_id'].nunique(),
        'Final_capacity_mAhg': round(final_cycle['capacity'].mean(), 1),
        'Final_retention_pct': round(final_cycle['capacity_retention'].mean(), 1),
        'Retention_std':       round(final_cycle['capacity_retention'].std(), 2),
        'Mean_CE_pct':         round(cycling_df[cycling_df['group'] == grp]['coulombic_efficiency'].mean(), 3),
    })

final_df = pd.DataFrame(final_rows)
final_df.to_csv(os.path.join(OUTPUT_DIR, "electrochemical_summary.csv"), index=False)
print("\nSaved electrochemical_summary.csv")
print(final_df.to_string(index=False))

print(f"\nAll electrochemical outputs saved to: {OUTPUT_DIR}")
