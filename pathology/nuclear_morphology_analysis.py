# -*- coding: utf-8 -*-
"""
Nuclear Morphology Analysis: TAM Aspect Ratio in Hypoxic vs Normoxic TNBC
==========================================================================
Updated violin plot includes p-value bracket annotation and tightened aesthetics.
All other pipeline logic unchanged from last working version.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from skimage import measure, img_as_ubyte
from skimage.color import rgb2hed, hed2rgb
from skimage.exposure import rescale_intensity
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from stardist.plot import render_label
from scipy import stats


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_CONFIGS = {
    "Hypoxic_40x": {
        "path": "figures/fig1C_hypoxic_hm.png",
        "min_area": 150,
        "max_area": 10000,
        "min_ar": 1.1,
        "max_ar": 5.0,
        "solidity_min": 0.82,
        "large_nucleus_area": 2000,
        "circular_circ": 0.85,
        "circular_area": 400,
        "stromal_filter": False,
        "stromal_percentile": 60,
    },
    "Normoxic_40x": {
        "path": "figures/fig1D_normoxic_hm.png",
        "min_area": 50,
        "max_area": 1000,
        "min_ar": 1.1,
        "max_ar": 5.0,
        "solidity_min": 0.85,
        "large_nucleus_area": 1500,
        "circular_circ": 0.90,
        "circular_area": 100,
        "stromal_filter": False,
        "stromal_percentile": 60,
    },
}

OUTPUT_DIR = "nuclear_morphology_results/"

MIN_NUCLEUS_AREA        = 50
MAX_NUCLEUS_AREA        = 10000
MIN_ASPECT_RATIO        = 1.05
MAX_ASPECT_RATIO        = 5.0
NMS_THRESH              = 0.3
PROB_THRESH             = 0.5
MIN_NUCLEI_FOR_ANALYSIS = 30

STYLE = {
    "font":       "Arial",
    "fontsize":   8,
    "dpi":        300,
    "color_hyp":  "#D62728",
    "color_norm": "#1F77B4",
    "alpha":      0.6,
}

plt.rcParams.update({
    'font.family': STYLE['font'],
    'font.size':   STYLE['fontsize'],
    'axes.spines.top':   False,
    'axes.spines.right': False,
})


# ---------------------------------------------------------------------------
# Stain separation
# ---------------------------------------------------------------------------

def color_separate(image_rgb):
    image_hed = rgb2hed(image_rgb)
    null = np.zeros_like(image_hed[:, :, 0])
    image_h = img_as_ubyte(hed2rgb(np.stack((image_hed[:, :, 0], null, null), axis=-1)))
    image_e = img_as_ubyte(hed2rgb(np.stack((null, image_hed[:, :, 1], null), axis=-1)))
    image_d = img_as_ubyte(hed2rgb(np.stack((null, null, image_hed[:, :, 2]), axis=-1)))
    h = rescale_intensity(image_hed[:, :, 0], out_range=(0, 1),
                          in_range=(0, np.percentile(image_hed[:, :, 0], 99)))
    d = rescale_intensity(image_hed[:, :, 2], out_range=(0, 1),
                          in_range=(0, np.percentile(image_hed[:, :, 2], 99)))
    zdh = img_as_ubyte(np.dstack((null, d, h)))
    return image_h, image_e, image_d, zdh


def save_color_separated(image_rgb, label, output_dir):
    H_img, E_img, D_img, zdh = color_separate(image_rgb)
    for channel_img, suffix in [
        (H_img, "H_hematoxylin"),
        (E_img, "E_eosin"),
        (D_img, "DAB_brown"),
        (zdh,   "pseudocolor_HDABoverlay"),
    ]:
        out = os.path.join(output_dir, f"{label}_{suffix}.png")
        plt.imsave(out, channel_img)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment_he_model(image_rgb, prob_thresh=None, nms_thresh=None):
    model = StarDist2D.from_pretrained('2D_versatile_he')
    labels, _ = model.predict_instances(
        normalize(image_rgb),
        nms_thresh=nms_thresh or NMS_THRESH,
        prob_thresh=prob_thresh or PROB_THRESH)
    return labels


def segment_fluo_fallback(image_rgb, prob_thresh=None, nms_thresh=None,
                          save_inverted_h=None):
    print("  Using stain separation + fluo model fallback...")
    H_img, _, _, _ = color_separate(image_rgb)
    H_inverted = np.invert(H_img)
    H_gray = H_inverted[:, :, 0].astype(np.float32)
    p2  = np.percentile(H_gray, 2)
    p98 = np.percentile(H_gray, 98)
    H_gray = np.clip(H_gray, p2, p98)
    H_gray = (H_gray - H_gray.min()) / (H_gray.max() - H_gray.min() + 1e-8)
    if save_inverted_h:
        plt.imsave(save_inverted_h, H_gray, cmap='gray')
        print(f"  Saved inverted H: {save_inverted_h}")
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    labels, _ = model.predict_instances(
        normalize(H_gray),
        nms_thresh=nms_thresh or NMS_THRESH,
        prob_thresh=prob_thresh or PROB_THRESH)
    return labels


def prescale_image(image_rgb, scale):
    from PIL import Image as PILImage
    h, w = image_rgb.shape[:2]
    scaled = np.array(PILImage.fromarray(image_rgb).resize(
        (int(w * scale), int(h * scale)), PILImage.LANCZOS))
    return scaled


def downscale_labels(labels_scaled, original_shape, scale):
    from PIL import Image as PILImage
    h, w = original_shape[:2]
    labels_pil = PILImage.fromarray(labels_scaled.astype(np.int32))
    return np.array(labels_pil.resize((w, h), PILImage.NEAREST))


def segment_nuclei_with_fallback(image_rgb, label="", prob_thresh=None, nms_thresh=None,
                                 prescale=1.0, force_fluo=False, save_inverted_h=None):
    pt = prob_thresh or PROB_THRESH
    nt = nms_thresh  or NMS_THRESH
    if prescale and prescale != 1.0:
        print(f"  Prescaling {prescale}x...")
        image_for_seg = prescale_image(image_rgb, prescale)
    else:
        image_for_seg = image_rgb

    if not force_fluo:
        print(f"  Trying 2D_versatile_he (prob={pt}, nms={nt})...")
        try:
            labels_scaled = segment_he_model(image_for_seg, prob_thresh=pt, nms_thresh=nt)
            n = len(np.unique(labels_scaled)) - 1
            print(f"  Detected {n} raw nuclei")
            if n >= MIN_NUCLEI_FOR_ANALYSIS:
                labels = downscale_labels(labels_scaled, image_rgb.shape, prescale) \
                         if prescale != 1.0 else labels_scaled
                return labels, "2D_versatile_he"
            print(f"  Yield too low, switching to fluo fallback...")
        except Exception as e:
            print(f"  HE model failed ({e}), switching to fluo fallback...")
    else:
        print(f"  force_fluo=True: stain separation + inverted H + fluo model...")

    labels_scaled = segment_fluo_fallback(image_for_seg, prob_thresh=pt, nms_thresh=nt,
                                          save_inverted_h=save_inverted_h)
    n = len(np.unique(labels_scaled)) - 1
    print(f"  Fluo fallback detected {n} raw nuclei")
    labels = downscale_labels(labels_scaled, image_rgb.shape, prescale) \
             if prescale != 1.0 else labels_scaled
    return labels, "2D_versatile_fluo_fallback"


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def classify_nuclei_by_population(df, profile=None):
    before = len(df)
    p = profile or {}
    SOLIDITY_MIN       = p.get('solidity_min',       0.82)
    LARGE_NUCLEUS_AREA = p.get('large_nucleus_area', 2000)
    CIRCULAR_CIRC      = p.get('circular_circ',      0.85)
    CIRCULAR_AREA      = p.get('circular_area',       400)

    fragment_mask = (df['solidity'] < SOLIDITY_MIN) & (df['area'] < 800)
    large_mask    = df['area'] > LARGE_NUCLEUS_AREA
    circular_mask = (df['circularity'] > CIRCULAR_CIRC) & (df['area'] < CIRCULAR_AREA) \
                    if 'circularity' in df.columns \
                    else pd.Series(False, index=df.index)

    df_clean = df[~fragment_mask & ~large_mask & ~circular_mask].copy().reset_index(drop=True)
    print(f"  classify_nuclei_by_population:")
    print(f"    removed {fragment_mask.sum()} fragments, "
          f"{large_mask.sum()} large, "
          f"{circular_mask.sum()} circular debris")
    print(f"    retained {len(df_clean)} / {before}")
    return df_clean


def build_stromal_mask(image_rgb, percentile=70):
    from skimage.filters import gaussian
    from skimage.morphology import binary_dilation, disk
    image_hed = rgb2hed(image_rgb)
    eosin = image_hed[:, :, 1]
    eosin_smooth = gaussian(eosin, sigma=5)
    thresh = np.percentile(eosin_smooth, percentile)
    stromal_mask = binary_dilation(eosin_smooth > thresh, disk(8))
    return stromal_mask


def filter_by_stromal_mask(df, stromal_mask):
    if stromal_mask is None:
        return df
    h, w = stromal_mask.shape
    cy = df['centroid-0'].values.astype(int).clip(0, h - 1)
    cx = df['centroid-1'].values.astype(int).clip(0, w - 1)
    in_stroma = stromal_mask[cy, cx]
    df_stromal = df[in_stroma].copy().reset_index(drop=True)
    print(f"  Stromal filter: removed {(~in_stroma).sum()}, retained {len(df_stromal)}")
    return df_stromal


def extract_shape_features(labels, image_rgb, profile=None):
    p = profile or {}
    min_area = p.get('min_area', MIN_NUCLEUS_AREA)
    max_area = p.get('max_area', MAX_NUCLEUS_AREA)
    min_ar   = p.get('min_ar',   MIN_ASPECT_RATIO)
    max_ar   = p.get('max_ar',   MAX_ASPECT_RATIO)

    props = measure.regionprops_table(
        labels, image_rgb,
        properties=['label', 'area', 'perimeter',
                    'major_axis_length', 'minor_axis_length',
                    'eccentricity', 'solidity', 'centroid',
                    'equivalent_diameter', 'mean_intensity'])
    df = pd.DataFrame(props)
    df['aspect_ratio'] = df['major_axis_length'] / df['minor_axis_length'].replace(0, np.nan)
    df['circularity']  = (4 * np.pi * df['area']) / (df['perimeter'] ** 2 + 1e-6)

    print(f"  Raw: {len(df)} nuclei | "
          f"area median={df['area'].median():.0f} | "
          f"below min_area: {(df['area'] < min_area).sum()}")

    df = df[(df['area'] > min_area) & (df['area'] < max_area)]
    df = df[(df['aspect_ratio'] >= min_ar) & (df['aspect_ratio'] <= max_ar)]
    df = df.dropna(subset=['aspect_ratio']).reset_index(drop=True)
    df = classify_nuclei_by_population(df, profile=p)

    stromal_mask = None
    if p.get('stromal_filter', False):
        print(f"  Building stromal mask (percentile={p.get('stromal_percentile', 70)})...")
        stromal_mask = build_stromal_mask(image_rgb, p.get('stromal_percentile', 70))
        df = filter_by_stromal_mask(df, stromal_mask)

    print(f"  {len(df)} nuclei retained after all filtering")
    return df, stromal_mask


# ---------------------------------------------------------------------------
# Diagnostic image saving
# ---------------------------------------------------------------------------

def save_segmentation_images(image_rgb, labels, df_nuclei, label, output_dir,
                             stromal_mask=None):
    if stromal_mask is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(image_rgb)
        om = np.zeros((*stromal_mask.shape, 4), dtype=np.float32)
        om[stromal_mask]  = [0.2, 0.8, 0.2, 0.35]
        om[~stromal_mask] = [0.8, 0.2, 0.2, 0.35]
        ax.imshow(om); ax.axis('off')
        ax.set_title(f'Stromal mask: {label} (green=included)', fontsize=7)
        fig.savefig(os.path.join(output_dir, f"{label}_stromal_mask.png"),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    h, w = labels.shape
    kept_labels    = set(df_nuclei['label'].values) if df_nuclei is not None else set()
    filtered_labels = np.where(np.isin(labels, list(kept_labels)), labels, 0)

    overlay = render_label(filtered_labels, img=image_rgb, alpha=0.35, alpha_boundary=0.9)
    if overlay.dtype != np.uint8:
        overlay = (overlay / overlay.max()).clip(0, 1) if overlay.max() > 1 else overlay
    plt.imsave(os.path.join(output_dir, f"{label}_segmentation_overlay.png"), overlay)
    print(f"  Saved: {label}_segmentation_overlay.png")

    cmap_inst    = plt.get_cmap('nipy_spectral')
    colored_mask = np.zeros((h, w, 4), dtype=np.float32)
    for lbl in kept_labels:
        colored_mask[labels == lbl] = cmap_inst((lbl * 37) % 256 / 256.0)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_rgb); ax.imshow(colored_mask, alpha=0.5); ax.axis('off')
    ax.set_title(f'Instance segmentation: {label.replace("_", " ")}', fontsize=8)
    fig.savefig(os.path.join(output_dir, f"{label}_instance_mask.png"),
                dpi=STYLE['dpi'], bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"  Saved: {label}_instance_mask.png")

    if df_nuclei is not None and len(df_nuclei) > 0:
        ar_map = np.zeros((h, w), dtype=np.float32)
        for lbl, ar in zip(df_nuclei['label'].values, df_nuclei['aspect_ratio'].values):
            ar_map[labels == lbl] = ar
        background = labels == 0
        vmin, vmax = 1.0, df_nuclei['aspect_ratio'].quantile(0.95)
        norm_ar = Normalize(vmin=vmin, vmax=vmax)
        cmap_ar = plt.get_cmap('RdYlBu_r')
        ar_colored = cmap_ar(norm_ar(ar_map))
        ar_colored[background] = [1, 1, 1, 0]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5),
                                 gridspec_kw={'width_ratios': [1, 0.05]})
        axes[0].imshow(image_rgb); axes[0].imshow(ar_colored, alpha=0.65); axes[0].axis('off')
        axes[0].set_title(f'Aspect ratio heatmap: {label.replace("_", " ")}', fontsize=8)
        sm = ScalarMappable(cmap=cmap_ar, norm=norm_ar); sm.set_array([])
        cbar = fig.colorbar(sm, cax=axes[1])
        cbar.set_label('Aspect Ratio\n(major/minor)', fontsize=7)
        cbar.ax.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{label}_aspect_ratio_heatmap.png"),
                    dpi=STYLE['dpi'], bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {label}_aspect_ratio_heatmap.png")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def analyze_image(image_path, label, output_dir, profile=None, save_diagnostics=True):
    print(f"\nAnalyzing: {label} ({image_path})")
    if profile:
        print(f"  Profile: min_area={profile.get('min_area')}, "
              f"solidity_min={profile.get('solidity_min')}, "
              f"large_nucleus_area={profile.get('large_nucleus_area')}")
    image_rgb = np.array(Image.open(image_path).convert('RGB'))

    if save_diagnostics:
        save_color_separated(image_rgb, label, output_dir)

    prob_thresh = profile.get('prob_thresh', None) if profile else None
    nms_thresh  = profile.get('nms_thresh',  None) if profile else None
    prescale    = profile.get('prescale',    1.0)  if profile else 1.0
    force_fluo  = profile.get('force_fluo', False) if profile else False
    inv_h_path  = os.path.join(output_dir, f"{label}_inverted_H_input.png")

    labels, method = segment_nuclei_with_fallback(
        image_rgb, label,
        prob_thresh=prob_thresh, nms_thresh=nms_thresh,
        prescale=prescale, force_fluo=force_fluo,
        save_inverted_h=inv_h_path)

    df, stromal_mask = extract_shape_features(labels, image_rgb, profile=profile)
    df['image_label']        = label
    df['segmentation_method'] = method

    print(f"  AR: mean={df['aspect_ratio'].mean():.3f}, "
          f"median={df['aspect_ratio'].median():.3f}, "
          f"std={df['aspect_ratio'].std():.3f}")

    if save_diagnostics:
        save_segmentation_images(image_rgb, labels, df, label, output_dir,
                                 stromal_mask=stromal_mask)
    return df, labels


def run_analysis(image_configs, output_dir, save_diagnostics=True):
    os.makedirs(output_dir, exist_ok=True)
    all_dfs, all_labels = {}, {}
    for label, profile in image_configs.items():
        path = profile.get('path', '')
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping {label}")
            continue
        df, labels = analyze_image(path, label, output_dir,
                                   profile=profile, save_diagnostics=save_diagnostics)
        all_dfs[label]   = df
        all_labels[label] = labels
        df.to_csv(os.path.join(output_dir, f"{label}_nuclei.csv"), index=False)
        print(f"  Saved CSV: {label}_nuclei.csv")
    return all_dfs, all_labels


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def cohens_d(a, b):
    pooled = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else np.nan


def compare_groups(df_a, df_b, label_a, label_b, output_dir):
    ar_a = df_a['aspect_ratio'].values
    ar_b = df_b['aspect_ratio'].values
    stat, pval = stats.mannwhitneyu(ar_a, ar_b, alternative='two-sided')
    d = cohens_d(ar_a, ar_b)
    summary = {
        'Group_A': label_a, 'Group_B': label_b,
        'n_A': len(ar_a), 'n_B': len(ar_b),
        'mean_AR_A': np.mean(ar_a), 'mean_AR_B': np.mean(ar_b),
        'median_AR_A': np.median(ar_a), 'median_AR_B': np.median(ar_b),
        'MannWhitneyU': stat, 'p_value': pval, 'cohens_d': d,
    }
    print(f"\n--- {label_a} vs {label_b} ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, f"stats_{label_a}_vs_{label_b}.csv"), index=False)
    return summary


# ---------------------------------------------------------------------------
# Plots — updated with p-value annotation
# ---------------------------------------------------------------------------

def _pval_label(pval):
    """Convert p-value to annotation string."""
    if pval < 0.0001:
        return "p < 0.0001"
    elif pval < 0.001:
        return "p < 0.001"
    elif pval < 0.01:
        return "p < 0.01"
    elif pval < 0.05:
        return "p < 0.05"
    else:
        return f"p = {pval:.3f}"


def add_significance_bracket(ax, x1, x2, y, pval, fontsize=7):
    """Draw a bracket with p-value label between two violin positions."""
    h    = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
    pad  = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01
    ax.plot([x1, x1, x2, x2],
            [y, y + h, y + h, y],
            lw=0.8, color='black')
    ax.text((x1 + x2) / 2, y + h + pad,
            _pval_label(pval),
            ha='center', va='bottom', fontsize=fontsize, color='black')


def plot_aspect_ratio_violin(all_dfs, output_dir, stats_results=None):
    """
    Violin plot with jittered strip and optional p-value brackets.
    stats_results: list of dicts from compare_groups, each with
                   Group_A, Group_B, p_value keys.
    """
    labels = list(all_dfs.keys())
    colors = [STYLE['color_hyp'] if 'Hyp' in l else STYLE['color_norm'] for l in labels]
    data   = [all_dfs[l]['aspect_ratio'].values for l in labels]
    label_to_pos = {l: i for i, l in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(max(len(labels) * 1.6 + 0.8, 4.5), 4.0))

    parts = ax.violinplot(data, positions=range(len(labels)),
                          showmedians=True, showextrema=False)
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(STYLE['alpha'])
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)

    for i, (d, color) in enumerate(zip(data, colors)):
        jitter = np.random.normal(0, 0.06, size=len(d))
        ax.scatter(i + jitter, d, s=2, color=color, alpha=0.25, linewidths=0)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace('_', '\n') for l in labels], fontsize=7)
    ax.set_ylabel('Nuclear Aspect Ratio\n(major / minor axis)', fontsize=8)
    ax.set_title('TAM Nuclear Elongation: Hypoxic vs Normoxic Regions',
                 fontsize=9, pad=10)

    # Add n= annotation just inside the bottom of each violin
    y_min = ax.get_ylim()[0]
    y_max = ax.get_ylim()[1]
    y_n   = y_min + 0.04 * (y_max - y_min)   # 4% up from bottom of plot
    for i, (l, d) in enumerate(zip(labels, data)):
        ax.text(i, y_n, f'n={len(d)}',
                ha='center', va='bottom', fontsize=6, color='white',
                fontweight='bold')

    # Add p-value brackets
    if stats_results:
        y_bracket = max(max(d) for d in data) * 1.02
        for sr in stats_results:
            a, b = sr['Group_A'], sr['Group_B']
            if a in label_to_pos and b in label_to_pos:
                add_significance_bracket(ax,
                                         label_to_pos[a], label_to_pos[b],
                                         y_bracket, sr['p_value'])
                y_bracket *= 1.10   # stack brackets if multiple pairs

    fig.tight_layout()
    out = os.path.join(output_dir, 'aspect_ratio_violin.png')
    fig.savefig(out, dpi=STYLE['dpi'], bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out}")


def plot_aspect_ratio_histogram(all_dfs, output_dir):
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    for label, df in all_dfs.items():
        color = STYLE['color_hyp'] if 'Hyp' in label else STYLE['color_norm']
        ax.hist(df['aspect_ratio'], bins=40, color=color,
                alpha=0.55, density=True, label=label.replace('_', ' '))
    ax.set_xlabel('Nuclear Aspect Ratio')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Nuclear Aspect Ratios', fontsize=9, pad=8)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out = os.path.join(output_dir, 'aspect_ratio_histogram.png')
    fig.savefig(out, dpi=STYLE['dpi'], bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("Nuclear Morphology Analysis: TAM Elongation in TNBC")
    print("=" * 60)

    all_dfs, all_labels = run_analysis(IMAGE_CONFIGS, OUTPUT_DIR, save_diagnostics=True)

    if len(all_dfs) < 2:
        print("\nOnly one image loaded.")
        for label, df in all_dfs.items():
            print(f"  {label}: {len(df)} nuclei | "
                  f"mean AR={df['aspect_ratio'].mean():.3f} | "
                  f"median AR={df['aspect_ratio'].median():.3f}")
    else:
        # Run statistics first so we can pass results to the violin plot
        pairs = [
            ("Hypoxic_40x", "Normoxic_40x"),
            ("Hypoxic_20x", "Normoxic_20x"),
        ]
        stats_results = []
        for a, b in pairs:
            if a in all_dfs and b in all_dfs:
                sr = compare_groups(all_dfs[a], all_dfs[b], a, b, OUTPUT_DIR)
                stats_results.append(sr)

        # Violin plot with p-value brackets
        plot_aspect_ratio_violin(all_dfs, OUTPUT_DIR, stats_results=stats_results)
        plot_aspect_ratio_histogram(all_dfs, OUTPUT_DIR)

        # Summary table
        summary_rows = []
        for label, df in all_dfs.items():
            summary_rows.append({
                'Image':                label,
                'N_nuclei':             len(df),
                'Mean_AR':              round(df['aspect_ratio'].mean(), 4),
                'Median_AR':            round(df['aspect_ratio'].median(), 4),
                'Std_AR':               round(df['aspect_ratio'].std(), 4),
                'Mean_eccentricity':    round(df['eccentricity'].mean(), 4),
                'Mean_solidity':        round(df['solidity'].mean(), 4),
                'Segmentation_method':  df['segmentation_method'].iloc[0],
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(OUTPUT_DIR, "morphology_summary.csv"), index=False)
        print(f"\nSaved summary: morphology_summary.csv")
        print("\n" + summary_df.to_string(index=False))

    print("\nDone.")
