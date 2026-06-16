"""
generate_workflow.py
--------------------
Generates a two-panel workflow figure for the WeightedKgBlend paper:
  (a) Simple Ensemble
  (b) Path-Gated Re-ranking

Output: results/figures/fig_workflow.pdf / .png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path

OUT_DIR = Path('/Users/meghamala/projects/WeightedKgBlend/results/figures')
OUT_DIR.mkdir(exist_ok=True)

# ── Colours ──────────────────────────────────────────────────────────────────
C_KG      = '#2C3E7A'   # dark blue  — KG input
C_KGE     = '#27AE60'   # green      — KGE models
C_CBR     = '#E67E22'   # orange     — CBR / path
C_ENS     = '#8E44AD'   # purple     — ensemble output
C_PG      = '#B7950B'   # gold       — path-gated
C_LIGHT   = '#F4F6F7'   # light grey — panel background
C_ARROW   = '#555555'   # dark grey  — arrows

def draw_box(ax, x, y, w, h, label, color, fontsize=9, text_color='white', radius=0.04):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f'round,pad=0.01,rounding_size={radius}',
                         linewidth=0, facecolor=color, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold',
            color=text_color, zorder=4, multialignment='center')

def draw_arrow(ax, x1, y1, x2, y2, label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=C_ARROW,
                                lw=1.5, mutation_scale=14),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.025, label, ha='center', va='bottom',
                fontsize=7, color=C_ARROW, style='italic', zorder=5)

def panel_background(ax, x1, y1, x2, y2, color=C_LIGHT):
    rect = FancyBboxPatch((x1, y1), x2-x1, y2-y1,
                          boxstyle='round,pad=0.01,rounding_size=0.05',
                          linewidth=1.2, edgecolor='#CCCCCC',
                          facecolor=color, zorder=0)
    ax.add_patch(rect)

# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(12, 9))
fig.subplots_adjust(hspace=0.15)

# ────────────────────────────────────────────────────────────────────────────
# PANEL (a) — Simple Ensemble
# ────────────────────────────────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.2)
ax.axis('off')

# Background
panel_background(ax, 0.1, 0.1, 9.9, 3.0)

# Panel label + title
ax.text(0.22, 2.85, '(a)', fontsize=13, fontweight='bold', va='top')
ax.text(5.0, 3.08, 'Simple Ensemble', ha='center', va='bottom',
        fontsize=11, fontweight='bold', color=C_ENS)

# MIND KG
draw_box(ax, 1.1, 1.5, 1.4, 0.6, 'MIND KG', C_KG, fontsize=9)

# Models
draw_box(ax, 3.0, 2.5, 1.4, 0.55, 'TransE', C_KGE)
draw_box(ax, 3.0, 1.5, 1.4, 0.55, 'RotatE', C_KGE)
draw_box(ax, 3.0, 0.5, 1.4, 0.55, 'ProbCBR', C_CBR)

# Reciprocal ranks
draw_box(ax, 5.1, 2.5, 1.5, 0.55, '1/rank\u209c', C_KGE, fontsize=8)
draw_box(ax, 5.1, 1.5, 1.5, 0.55, '1/rank\u1d63', C_KGE, fontsize=8)
draw_box(ax, 5.1, 0.5, 1.5, 0.55, '1/rank\u1d9c', C_CBR, fontsize=8)

# Optuna
draw_box(ax, 7.1, 1.5, 1.5, 0.65, 'Optuna\n(200 trials)', C_ENS, fontsize=8)

# Output
draw_box(ax, 9.1, 1.5, 1.5, 0.65, 'Ranked\ndisease list', C_ENS, fontsize=8)

# Arrows: KG → models
draw_arrow(ax, 1.8, 1.7, 2.3, 2.5)
draw_arrow(ax, 1.8, 1.5, 2.3, 1.5)
draw_arrow(ax, 1.8, 1.3, 2.3, 0.5)

# Arrows: models → rr
draw_arrow(ax, 3.7, 2.5, 4.35, 2.5, 'recip. rank')
draw_arrow(ax, 3.7, 1.5, 4.35, 1.5, 'recip. rank')
draw_arrow(ax, 3.7, 0.5, 4.35, 0.5, 'recip. rank')

# Arrows: rr → optuna
draw_arrow(ax, 5.85, 2.5, 6.35, 1.75)
draw_arrow(ax, 5.85, 1.5, 6.35, 1.5)
draw_arrow(ax, 5.85, 0.5, 6.35, 1.25)

# Arrow: optuna → output
draw_arrow(ax, 7.85, 1.5, 8.35, 1.5, '\u03bb\u2081,\u03bb\u2082,\u03bb\u2083')

# ────────────────────────────────────────────────────────────────────────────
# PANEL (b) — Path-Gated Re-ranking
# ────────────────────────────────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.2)
ax.axis('off')

# Background
panel_background(ax, 0.1, 0.1, 9.9, 3.0)

# Panel label + title
ax.text(0.22, 2.85, '(b)', fontsize=13, fontweight='bold', va='top')
ax.text(5.0, 3.08, 'Path-Gated Re-ranking', ha='center', va='bottom',
        fontsize=11, fontweight='bold', color=C_PG)

# MIND KG
draw_box(ax, 1.1, 1.9, 1.4, 0.6, 'MIND KG', C_KG, fontsize=9)

# Stage 1
draw_box(ax, 3.0, 1.9, 1.4, 0.55, 'RotatE', C_KGE)
draw_box(ax, 5.0, 1.9, 1.6, 0.55, 'Top-50\ncandidates', C_KGE, fontsize=8)

# Stage 2
draw_box(ax, 7.1, 1.9, 1.6, 0.55, 'ProbCBR\npath scoring', C_CBR, fontsize=8)

# Optuna + output
draw_box(ax, 3.0, 0.65, 1.5, 0.55, 'Optuna\n(300 trials)', C_PG, fontsize=8)
draw_box(ax, 5.0, 0.65, 1.6, 0.55, 'Re-ranked list\n+ path', C_PG, fontsize=8)

# Path example box
path_box = FancyBboxPatch((6.5, 0.2), 3.3, 0.85,
                           boxstyle='round,pad=0.01,rounding_size=0.04',
                           linewidth=1.2, edgecolor=C_PG,
                           facecolor='#FEF9E7', zorder=3)
ax.add_patch(path_box)
# drug → gene → disease with labels above arrows
ax.text(6.85, 0.62, 'drug', ha='center', va='center',
        fontsize=8.5, color=C_KG, fontweight='bold', zorder=5)
ax.annotate('', xy=(7.55, 0.62), xytext=(7.15, 0.62),
            arrowprops=dict(arrowstyle='->', color=C_PG, lw=1.4), zorder=5)
ax.text(7.35, 0.74, 'inhibits', ha='center', va='bottom',
        fontsize=7, color=C_CBR, style='italic', zorder=5)
ax.text(7.9, 0.62, 'gene', ha='center', va='center',
        fontsize=8.5, color=C_KGE, fontweight='bold', zorder=5)
ax.annotate('', xy=(8.6, 0.62), xytext=(8.2, 0.62),
            arrowprops=dict(arrowstyle='->', color=C_PG, lw=1.4), zorder=5)
ax.text(8.4, 0.74, 'assoc. with', ha='center', va='bottom',
        fontsize=7, color=C_CBR, style='italic', zorder=5)
ax.text(9.1, 0.62, 'disease', ha='center', va='center',
        fontsize=8.5, color=C_KG, fontweight='bold', zorder=5)

# Stage labels
ax.text(4.0, 2.35, 'Stage 1', ha='center', fontsize=7.5,
        color='#555555', style='italic')
ax.text(6.05, 2.35, 'Stage 2', ha='center', fontsize=7.5,
        color='#555555', style='italic')

# Arrows panel b
draw_arrow(ax, 1.8, 1.9, 2.3, 1.9)
draw_arrow(ax, 3.7, 1.9, 4.2, 1.9)
draw_arrow(ax, 5.8, 1.9, 6.3, 1.9)

# Down arrow from ProbCBR
draw_arrow(ax, 7.1, 1.62, 7.1, 1.1)
ax.text(7.22, 1.38, '\u03b1,\u03b2', fontsize=8, color=C_ARROW, style='italic')

# Optuna connections
draw_arrow(ax, 6.35, 0.85, 5.8, 0.75)
draw_arrow(ax, 3.75, 0.65, 4.2, 0.65)

# Path example arrow
draw_arrow(ax, 5.8, 0.55, 6.48, 0.55)

plt.savefig(OUT_DIR / 'fig_workflow.pdf', dpi=300, bbox_inches='tight')
plt.savefig(OUT_DIR / 'fig_workflow.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: fig_workflow.pdf / .png → {OUT_DIR}')
