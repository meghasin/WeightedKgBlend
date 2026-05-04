"""
run_ensemble_updated.py
-----------------------
Runs Optuna-optimized Simple Ensemble (TransE + RotatE + ProbCBR) on splits_updated.
Saves predictions and updates results/full_results.md.

Usage:
    python scripts/run_ensemble_updated.py --slice 0
    python scripts/run_ensemble_updated.py --all
"""

import argparse
import re
import numpy as np
import pandas as pd
import optuna
from collections import defaultdict
from pathlib import Path

optuna.logging.set_verbosity(optuna.logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument('--slice', type=int, default=None)
parser.add_argument('--all',   action='store_true')
parser.add_argument('--trials', type=int, default=200)
parser.add_argument('--top_k', type=int, default=50)
args = parser.parse_args()

slices = list(range(5)) if args.all else [args.slice]
assert slices[0] is not None, 'Provide --slice N or --all'

BASE    = Path(__file__).parent.parent
PRED    = BASE / 'results_updated/predictions'
OUT     = BASE / 'results_updated/predictions/Ensemble'
RESULTS = BASE / 'results/full_results.md'
MODELS  = ['TransE', 'RotatE', 'ProbCBR']
TOP_K   = args.top_k

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_preds(sl):
    preds = {}
    for model in MODELS:
        for split in ['valid', 'test']:
            path = PRED / model / f'slice_{sl}' / f'predictions_{split}.tsv'
            if not path.exists():
                raise FileNotFoundError(f'Missing: {path}')
            preds[(model, split)] = pd.read_csv(path, sep='\t')
    return preds

def align_dfs(preds, split):
    base = preds[(MODELS[0], split)][['drug', 'expected_disease']].copy()
    return [base.merge(preds[(m, split)], on=['drug','expected_disease'], how='left')
            for m in MODELS]

def compute_ensemble(dfs, weights, top_k=TOP_K):
    rrs = []
    for idx in range(len(dfs[0])):
        exp_dis = dfs[0].iloc[idx]['expected_disease']
        scores  = defaultdict(float)
        for mi, df in enumerate(dfs):
            row = df.iloc[idx]
            for k in range(1, top_k + 1):
                d = row.get(f'top{k}_disease', '')
                if d:
                    scores[d] += weights[mi] * (1.0 / k)
        ranked = sorted(scores, key=scores.get, reverse=True)
        rank   = ranked.index(exp_dis) + 1 if exp_dis in ranked else len(ranked) + 1
        rrs.append(1.0 / rank)
    return float(np.mean(rrs))

def build_pred_df(dfs, weights, top_k=TOP_K):
    rows = []
    for idx in range(len(dfs[0])):
        row0   = dfs[0].iloc[idx]
        exp_dis = row0['expected_disease']
        scores  = defaultdict(float)
        for mi, df in enumerate(dfs):
            row = df.iloc[idx]
            for k in range(1, top_k + 1):
                d = row.get(f'top{k}_disease', '')
                if d:
                    scores[d] += weights[mi] * (1.0 / k)
        ranked = sorted(scores, key=scores.get, reverse=True)
        rank   = ranked.index(exp_dis) + 1 if exp_dis in ranked else len(ranked) + 1
        r = {'drug': row0['drug'], 'expected_disease': exp_dis,
             'rank': rank, 'reciprocal_rank': 1.0 / rank}
        for k in range(1, top_k + 1):
            r[f'top{k}_disease'] = ranked[k-1] if k-1 < len(ranked) else ''
        rows.append(r)
    return pd.DataFrame(rows)

def metrics(df):
    return {
        'N'      : len(df),
        'MRR'    : round(df['reciprocal_rank'].mean(), 4),
        'Hits@1' : round((df['rank'] <= 1).mean(), 4),
        'Hits@3' : round((df['rank'] <= 3).mean(), 4),
        'Hits@5' : round((df['rank'] <= 5).mean(), 4),
        'Hits@10': round((df['rank'] <= 10).mean(), 4),
    }

# ── Run ───────────────────────────────────────────────────────────────────────

all_results = {}  # (sl, split) -> metrics dict
all_weights = {}  # sl -> weights

for sl in slices:
    print(f'\n{"="*50}\nEnsemble — slice_{sl}\n{"="*50}')

    preds     = load_preds(sl)
    valid_dfs = align_dfs(preds, 'valid')
    test_dfs  = align_dfs(preds, 'test')

    # Individual baselines
    print('Individual valid MRRs:')
    for m in MODELS:
        print(f'  {m}: {preds[(m,"valid")]["reciprocal_rank"].mean():.4f}')

    # Optuna
    def objective(trial):
        w = [trial.suggest_float(f'w{i}', 0.0, 1.0) for i in range(len(MODELS))]
        t = sum(w) or 1.0
        return compute_ensemble(valid_dfs, [wi/t for wi in w])

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    raw_w = [study.best_params[f'w{i}'] for i in range(len(MODELS))]
    best_w = [w / sum(raw_w) for w in raw_w]
    all_weights[sl] = best_w

    print(f'\nOptimal weights:')
    for m, w in zip(MODELS, best_w):
        print(f'  {m}: {w:.4f}')
    print(f'Best valid MRR: {study.best_value:.4f}')

    # Build and save prediction files
    outdir = OUT / f'slice_{sl}'
    outdir.mkdir(parents=True, exist_ok=True)

    for split, dfs in [('valid', valid_dfs), ('test', test_dfs)]:
        df_out = build_pred_df(dfs, best_w)
        df_out.to_csv(outdir / f'predictions_{split}.tsv', sep='\t', index=False)
        m = metrics(df_out)
        all_results[(sl, split)] = m
        print(f'  {split}: MRR={m["MRR"]} H@1={m["Hits@1"]} H@3={m["Hits@3"]} H@5={m["Hits@5"]} H@10={m["Hits@10"]}')

    print(f'Saved: {outdir}')

# ── Update full_results.md ────────────────────────────────────────────────────

md = RESULTS.read_text()

# Build new Ensemble section content
def fmt_row(sl, split, m):
    return f'| slice_{sl} | {split:5s} | {m["N"]} | {m["MRR"]:.4f} | {m["Hits@1"]:.4f} | {m["Hits@3"]:.4f} | {m["Hits@5"]:.4f} | {m["Hits@10"]:.4f} |'

table_rows = []
for sl in range(5):
    for split in ['test', 'valid']:
        k = (sl, split)
        if k in all_results:
            table_rows.append(fmt_row(sl, split, all_results[k]))
        else:
            table_rows.append(f'| slice_{sl} | {split:5s} | — | — | — | — | — | — |')

# Summary
def summary_row(split):
    vals = {m: [all_results[(sl, split)][m] for sl in range(5) if (sl, split) in all_results]
            for m in ['MRR','Hits@1','Hits@3','Hits@5','Hits@10']}
    if not vals['MRR']:
        return None
    return {m: f'{np.mean(v):.4f} ± {np.std(v):.4f}' for m, v in vals.items()}

test_sum  = summary_row('test')
valid_sum = summary_row('valid')

new_ensemble_section = f"""## Simple Ensemble

| Slice   | Split | N   | MRR    | Hits@1 | Hits@3 | Hits@5 | Hits@10 |
|---------|-------|-----|--------|--------|--------|--------|---------|
{chr(10).join(table_rows)}

**Optimal weights per slice (TransE / RotatE / ProbCBR):**
"""
for sl in slices:
    if sl in all_weights:
        w = all_weights[sl]
        new_ensemble_section += f'- slice_{sl}: TransE={w[0]:.4f}, RotatE={w[1]:.4f}, ProbCBR={w[2]:.4f}\n'

if test_sum:
    new_ensemble_section += f"""
**Summary (test, mean ± std)**

| MRR | Hits@1 | Hits@3 | Hits@5 | Hits@10 |
|-----|--------|--------|--------|---------|
| {test_sum['MRR']} | {test_sum['Hits@1']} | {test_sum['Hits@3']} | {test_sum['Hits@5']} | {test_sum['Hits@10']} |
"""
if valid_sum:
    new_ensemble_section += f"""
**Summary (valid, mean ± std)**

| MRR | Hits@1 | Hits@3 | Hits@5 | Hits@10 |
|-----|--------|--------|--------|---------|
| {valid_sum['MRR']} | {valid_sum['Hits@1']} | {valid_sum['Hits@3']} | {valid_sum['Hits@5']} | {valid_sum['Hits@10']} |
"""

# Replace existing Ensemble section
pattern = r'## Simple Ensemble.*?(?=\n## |\Z)'
if re.search(pattern, md, re.DOTALL):
    md = re.sub(pattern, new_ensemble_section.strip(), md, flags=re.DOTALL)
else:
    md = md + '\n---\n\n' + new_ensemble_section

# Update overall summary table
def update_summary_table(md, model, split, summary):
    if not summary:
        return md
    new_row = f'| {model:15s} | {summary["MRR"]} | {summary["Hits@1"]} | {summary["Hits@3"]} | {summary["Hits@5"]} | {summary["Hits@10"]} |'
    pattern = rf'\| {re.escape(model)}\s*\|[^\n]*\|'
    # Find the right summary table (test or valid)
    return md  # summary table update handled below

# Rewrite both summary tables at the bottom
def build_summary_tables(md):
    models_order = ['TransE', 'RotatE', 'ProbCBR', 'Simple Ensemble', 'Path-Gated']

    # Collect all results
    all_m = {}
    for model in ['TransE', 'RotatE', 'ProbCBR']:
        pdir = PRED / model
        for split in ['test', 'valid']:
            vals = {}
            for sl in range(5):
                p = pdir / f'slice_{sl}' / f'predictions_{split}.tsv'
                if p.exists():
                    df = pd.read_csv(p, sep='\t')
                    for met in ['MRR','Hits@1','Hits@3','Hits@5','Hits@10']:
                        vals.setdefault(met, []).append(metrics(df)[met])
            if vals.get('MRR'):
                all_m[(model, split)] = {k: f'{np.mean(v):.4f} ± {np.std(v):.4f}' for k, v in vals.items()}

    for split_label in ['test', 'valid']:
        for sl in range(5):
            k = (sl, split_label)
            if k in all_results:
                all_m.setdefault(('Simple Ensemble', split_label), {})
                for met in ['MRR','Hits@1','Hits@3','Hits@5','Hits@10']:
                    all_m[('Simple Ensemble', split_label)].setdefault(met+'_list', []).append(all_results[k][met])

    for split_label in ['test', 'valid']:
        key = ('Simple Ensemble', split_label)
        tmp = {}
        for met in ['MRR','Hits@1','Hits@3','Hits@5','Hits@10']:
            lst = all_m.get(key, {}).get(met+'_list', [])
            if lst:
                tmp[met] = f'{np.mean(lst):.4f} ± {np.std(lst):.4f}'
        if tmp:
            all_m[key] = tmp

    for split_label, label in [('test','test'), ('valid','valid')]:
        header = f'## Overall Summary ({label} set, mean ± std across 5 slices)'
        table  = f'{header}\n\n| Model           | MRR             | Hits@1          | Hits@3          | Hits@5          | Hits@10         |\n'
        table += '|-----------------|-----------------|-----------------|-----------------|-----------------|------------------|\n'
        for model in models_order:
            s = all_m.get((model, split_label), {})
            def g(k): return s.get(k, '—')
            table += f'| {model:15s} | {g("MRR"):15s} | {g("Hits@1"):15s} | {g("Hits@3"):15s} | {g("Hits@5"):15s} | {g("Hits@10"):15s} |\n'

        pattern = rf'{re.escape(header)}.*?(?=\n## |\Z)'
        if re.search(pattern, md, re.DOTALL):
            md = re.sub(pattern, table.strip(), md, flags=re.DOTALL)
        else:
            md += '\n---\n\n' + table
    return md

md = build_summary_tables(md)
RESULTS.write_text(md)
print(f'\nUpdated: {RESULTS}')
print('Done.')
