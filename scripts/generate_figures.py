"""
generate_figures.py
-------------------
Generates three analysis figures for the WeightedKgBlend paper:

  Fig A: Per-fold MRR bar chart (test set)
  Fig B: Box plot of reciprocal ranks across all 5 folds (test set)
  Fig C: Wilcoxon signed-rank test p-value table

Usage:
    python scripts/generate_figures.py

Outputs saved to: results/figures/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import wilcoxon
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
RESULTS_DIR = Path('/Users/meghamala/projects/WeightedKgBlend/results')
PREDS_DIR   = Path('/Users/meghamala/projects/WeightedKgBlend/results_updated/predictions')
SPLITS_DIR  = Path('/Users/meghamala/projects/WeightedKgBlend/data/splits_updated')
OUT_DIR     = RESULTS_DIR / 'figures'
OUT_DIR.mkdir(exist_ok=True)

# ── Per-slice MRR data (test set) from results_table_updated.md ────────────
data = {
    'TransE':          [0.0164, 0.0155, 0.0127, 0.0201, 0.0143],
    'ProbCBR':         [0.0091, 0.0255, 0.0131, 0.0088, 0.0124],
    'RotatE':          [0.0430, 0.0410, 0.0568, 0.0512, 0.0382],
    'Simple Ensemble': [0.0456, 0.0439, 0.0594, 0.0541, 0.0413],
    'Path-Gated':      [0.0627, 0.0762, 0.0837, 0.0653, 0.0529],
}
slices = ['Slice 0', 'Slice 1', 'Slice 2', 'Slice 3', 'Slice 4']
models = list(data.keys())
colors = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7', '#C4AD66']

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE A — Per-fold MRR bar chart
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(11, 15))
fig.subplots_adjust(hspace=0.4)

# ── Panel (a): Per-fold MRR bar chart ──────────────────────────────────────
ax = axes[0]
n_models  = len(models)
n_slices  = len(slices)
bar_width = 0.15
x         = np.arange(n_slices)

for i, (model, color) in enumerate(zip(models, colors)):
    offset = (i - n_models / 2 + 0.5) * bar_width
    ax.bar(x + offset, data[model], bar_width,
           label=model, color=color, alpha=0.85, edgecolor='white')

ax.set_xlabel('Cross-validation fold', fontsize=12)
ax.set_ylabel('MRR (test set)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(slices)
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0, 0.105)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
ax.text(-0.05, 1.04, '(a)', transform=ax.transAxes,
        fontsize=14, fontweight='bold')

print('Saved: fig_perfold_mrr')

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE B — Box plot of reciprocal ranks (load from prediction files)
# ═══════════════════════════════════════════════════════════════════════════

def load_reciprocal_ranks(model_dir, n_slices=5):
    """Load reciprocal ranks from predictions_test.tsv across all slices."""
    all_rr = []
    for sl in range(n_slices):
        pred_file = model_dir / f'slice_{sl}' / 'predictions_test.tsv'
        if not pred_file.exists():
            continue
        df = pd.read_csv(pred_file, sep='\t')
        if 'reciprocal_rank' in df.columns:
            rr = df['reciprocal_rank'].dropna().values
            all_rr.extend(rr.tolist())
    return all_rr

# Load from prediction files; fall back to per-slice MRR as proxy
rr_data = {}
model_dirs = {
    'TransE':          PREDS_DIR / 'TransE',
    'ProbCBR':         PREDS_DIR / 'ProbCBR',
    'RotatE':          PREDS_DIR / 'RotatE',
    'Simple Ensemble': PREDS_DIR / 'Ensemble',
    'Path-Gated':      PREDS_DIR / 'PathGated',
}
for model in models:
    rr = load_reciprocal_ranks(model_dirs[model])
    if rr:
        rr_data[model] = rr
        print(f'{model}: {len(rr)} reciprocal rank values loaded')
    else:
        rr_data[model] = data[model]
        print(f'{model}: fallback to per-fold MRR')

ax = axes[1]
box_data  = [rr_data[m] for m in models]
bp = ax.boxplot(box_data, patch_artist=True, notch=False,
                medianprops=dict(color='black', linewidth=2))

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel('Reciprocal Rank', fontsize=12)
ax.set_ylim(0, 0.20)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
ax.text(-0.05, 1.04, '(b)', transform=ax.transAxes,
        fontsize=14, fontweight='bold')

print('Saved: panel (b)')

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE C — Wilcoxon signed-rank test table
# ═══════════════════════════════════════════════════════════════════════════
pg_mrr      = np.array(data['Path-Gated'])
baselines   = ['TransE', 'ProbCBR', 'RotatE', 'Simple Ensemble']
wilcoxon_rows = []

for baseline in baselines:
    bl_mrr = np.array(data[baseline])
    diff   = pg_mrr - bl_mrr
    # Wilcoxon requires non-zero differences
    if np.all(diff == 0):
        stat, pval = np.nan, np.nan
    else:
        stat, pval = wilcoxon(pg_mrr, bl_mrr, alternative='greater')
    mean_pg = pg_mrr.mean()
    mean_bl = bl_mrr.mean()
    improve = (mean_pg - mean_bl) / mean_bl * 100
    wilcoxon_rows.append({
        'Baseline': baseline,
        'Baseline MRR': f'{mean_bl:.4f}',
        'Path-Gated MRR': f'{mean_pg:.4f}',
        'Improvement': f'+{improve:.1f}%',
        'p-value': f'{pval:.4f}' if not np.isnan(pval) else 'N/A',
        'Significant (p<0.05)': 'Yes' if (not np.isnan(pval) and pval < 0.05) else 'No',
    })

df_wilcoxon = pd.DataFrame(wilcoxon_rows)
print('\nWilcoxon Signed-Rank Test — Path-Gated vs. Baselines (test MRR, 5 folds):')
print(df_wilcoxon.to_string(index=False))

ax = axes[2]
ax.axis('off')
cols = ['Baseline', 'Baseline MRR', 'Path-Gated MRR', 'Improvement',
        'p-value', 'Sig. (p<0.05)']
# rename last column in dataframe to match
df_wilcoxon_plot = df_wilcoxon.rename(
    columns={'Significant (p<0.05)': 'Sig. (p<0.05)'})
table = ax.table(
    cellText=df_wilcoxon_plot[cols].values,
    colLabels=cols,
    cellLoc='center',
    loc='upper center',
    bbox=[0, 0.05, 1, 0.85]
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.0)

# Header styling
for j in range(len(cols)):
    table[0, j].set_facecolor('#2c3e50')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Highlight significant rows
for i, row in enumerate(wilcoxon_rows):
    color = '#d5f5e3' if row['Significant (p<0.05)'] == 'Yes' else '#fdfefe'
    for j in range(len(cols)):
        table[i + 1, j].set_facecolor(color)

ax.text(-0.05, 0.98, '(c)', transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top')

fig.savefig(OUT_DIR / 'fig_performance.pdf', dpi=300, bbox_inches='tight')
fig.savefig(OUT_DIR / 'fig_performance.png', dpi=300, bbox_inches='tight')
plt.close()
df_wilcoxon.to_csv(OUT_DIR / 'wilcoxon_results.csv', index=False)
print('Saved: fig_performance (a+b+c stacked)')

print(f'\nAll figures saved to: {OUT_DIR}')
