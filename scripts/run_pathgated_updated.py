"""
run_pathgated_updated.py
------------------------
Runs Path-Gated re-ranking (Optuna) on splits_updated for all 5 slices.
Updates results/full_results.md.

Usage:
    python scripts/run_pathgated_updated.py --all
    python scripts/run_pathgated_updated.py --slice 0
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
parser.add_argument('--slice',  type=int, default=None)
parser.add_argument('--all',    action='store_true')
parser.add_argument('--trials', type=int, default=300)
parser.add_argument('--top_k',  type=int, default=50)
args = parser.parse_args()

slices = list(range(5)) if args.all else [args.slice]
assert slices[0] is not None, 'Provide --slice N or --all'

BASE    = Path(__file__).parent.parent
PRED    = BASE / 'results_updated/predictions'
OUT     = PRED / 'PathGated'
RESULTS = BASE / 'results/full_results.md'
TOP_K   = args.top_k

# ── Helpers ───────────────────────────────────────────────────────────────

def load_path_scores(sl, split):
    path = OUT / f'slice_{sl}' / f'path_lookup_rotate_{split}_slice{sl}.tsv'
    df   = pd.read_csv(path, sep='\t')
    scores   = df.groupby(['drug','disease'])['path_score'].sum().to_dict()
    best_path= df[df['path_rank']==1].set_index(['drug','disease'])['path'].to_dict()
    return scores, best_path

def compute_metrics_pg(rotate_df, path_scores, alpha, beta, top_k=TOP_K):
    rrs, ranks = [], []
    for _, row in rotate_df.iterrows():
        drug, exp_dis = row['drug'], row['expected_disease']
        cands = {}
        for k in range(1, top_k + 1):
            d = row.get(f'top{k}_disease', '')
            if d:
                cands[d] = alpha * (1.0/k) + beta * path_scores.get((drug,d), 0.0)
        if exp_dis not in cands:
            cands[exp_dis] = alpha * (1.0/(top_k+1)) + beta * path_scores.get((drug,exp_dis), 0.0)
        ranked = sorted(cands, key=cands.get, reverse=True)
        rank   = ranked.index(exp_dis) + 1
        rrs.append(1.0/rank)
        ranks.append(rank)
    ranks = np.array(ranks)
    return {'N': len(ranks),
            'MRR'    : round(float(np.mean(rrs)), 4),
            'Hits@1' : round(float((ranks<=1).mean()), 4),
            'Hits@3' : round(float((ranks<=3).mean()), 4),
            'Hits@5' : round(float((ranks<=5).mean()), 4),
            'Hits@10': round(float((ranks<=10).mean()), 4)}

def build_pred_df(rotate_df, path_scores, best_path, alpha, beta, top_k=TOP_K):
    rows = []
    for _, row in rotate_df.iterrows():
        drug, exp_dis = row['drug'], row['expected_disease']
        cands = {}
        for k in range(1, top_k + 1):
            d = row.get(f'top{k}_disease', '')
            if d:
                cands[d] = alpha * (1.0/k) + beta * path_scores.get((drug,d), 0.0)
        if exp_dis not in cands:
            cands[exp_dis] = alpha * (1.0/(top_k+1)) + beta * path_scores.get((drug,exp_dis), 0.0)
        ranked = sorted(cands, key=cands.get, reverse=True)
        rank   = ranked.index(exp_dis) + 1
        r = {'drug': drug, 'expected_disease': exp_dis,
             'rank': rank, 'reciprocal_rank': 1.0/rank}
        for k in range(1, top_k+1):
            d = ranked[k-1] if k-1 < len(ranked) else ''
            r[f'top{k}_disease'] = d
            r[f'top{k}_path']    = best_path.get((drug,d),'') if d else ''
        rows.append(r)
    return pd.DataFrame(rows)

# ── Run ───────────────────────────────────────────────────────────────────

all_results = {}
all_weights = {}

for sl in slices:
    print(f'\n{"="*50}\nPath-Gated — slice_{sl}\n{"="*50}')

    rotate_test  = pd.read_csv(PRED / f'RotatE/slice_{sl}/predictions_test.tsv',  sep='\t')
    rotate_valid = pd.read_csv(PRED / f'RotatE/slice_{sl}/predictions_valid.tsv', sep='\t')

    ps_test,  bp_test  = load_path_scores(sl, 'test')
    ps_valid, bp_valid = load_path_scores(sl, 'valid')

    # Coverage
    for split, rotate_df, ps in [('test', rotate_test, ps_test), ('valid', rotate_valid, ps_valid)]:
        cov = sum(1 for _, r in rotate_df.iterrows()
                  for k in range(1,11) if (r['drug'], r.get(f'top{k}_disease','')) in ps)
        tot = sum(1 for _, r in rotate_df.iterrows()
                  for k in range(1,11) if r.get(f'top{k}_disease',''))
        print(f'  {split} top-10 path coverage: {cov}/{tot} = {cov/tot:.1%}')

    # Optuna on valid
    def objective(trial):
        alpha = trial.suggest_float('alpha', 0.0, 1.0)
        beta  = trial.suggest_float('beta',  0.0, 1.0)
        return compute_metrics_pg(rotate_valid, ps_valid, alpha, beta)['MRR']

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best_alpha = study.best_params['alpha']
    best_beta  = study.best_params['beta']
    all_weights[sl] = (best_alpha, best_beta)

    print(f'  alpha={best_alpha:.4f}  beta={best_beta:.4f}  valid MRR={study.best_value:.4f}')

    # Save predictions
    outdir = OUT / f'slice_{sl}'
    outdir.mkdir(parents=True, exist_ok=True)

    for split, rotate_df, ps, bp in [
        ('test',  rotate_test,  ps_test,  bp_test),
        ('valid', rotate_valid, ps_valid, bp_valid),
    ]:
        df_out = build_pred_df(rotate_df, ps, bp, best_alpha, best_beta)
        df_out.to_csv(outdir / f'predictions_{split}.tsv', sep='\t', index=False)
        m = compute_metrics_pg(rotate_df, ps, best_alpha, best_beta)
        all_results[(sl, split)] = m
        base = compute_metrics_pg(rotate_df, {}, 1.0, 0.0)
        print(f'  {split}: MRR={m["MRR"]} H@10={m["Hits@10"]}  '
              f'(RotatE base: MRR={base["MRR"]} H@10={base["Hits@10"]}  '
              f'ΔMRR={m["MRR"]-base["MRR"]:+.4f})')

# ── Update full_results.md ────────────────────────────────────────────────

md = RESULTS.read_text()

def fmt_row(sl, split, m):
    return (f'| slice_{sl} | {split:5s} | {m["N"]} | {m["MRR"]:.4f} | '
            f'{m["Hits@1"]:.4f} | {m["Hits@3"]:.4f} | {m["Hits@5"]:.4f} | {m["Hits@10"]:.4f} |')

table_rows = []
for sl in range(5):
    for split in ['test', 'valid']:
        k = (sl, split)
        if k in all_results:
            table_rows.append(fmt_row(sl, split, all_results[k]))
        else:
            table_rows.append(f'| slice_{sl} | {split:5s} | — | — | — | — | — | — |')

def summary(split):
    vals = {m: [all_results[(sl,split)][m] for sl in range(5) if (sl,split) in all_results]
            for m in ['MRR','Hits@1','Hits@3','Hits@5','Hits@10']}
    if not vals['MRR']: return None
    return {m: f'{np.mean(v):.4f} ± {np.std(v):.4f}' for m,v in vals.items()}

ts, vs = summary('test'), summary('valid')

new_section = f"""## Path-Gated

| Slice   | Split | N   | MRR    | Hits@1 | Hits@3 | Hits@5 | Hits@10 |
|---------|-------|-----|--------|--------|--------|--------|---------|
{chr(10).join(table_rows)}

**Optimal weights per slice (alpha=RotatE / beta=path_score):**
"""
for sl in slices:
    if sl in all_weights:
        a, b = all_weights[sl]
        new_section += f'- slice_{sl}: alpha={a:.4f}, beta={b:.4f}\n'

if ts:
    new_section += f"""
**Summary (test, mean ± std)**

| MRR | Hits@1 | Hits@3 | Hits@5 | Hits@10 |
|-----|--------|--------|--------|---------|
| {ts['MRR']} | {ts['Hits@1']} | {ts['Hits@3']} | {ts['Hits@5']} | {ts['Hits@10']} |
"""
if vs:
    new_section += f"""
**Summary (valid, mean ± std)**

| MRR | Hits@1 | Hits@3 | Hits@5 | Hits@10 |
|-----|--------|--------|--------|---------|
| {vs['MRR']} | {vs['Hits@1']} | {vs['Hits@3']} | {vs['Hits@5']} | {vs['Hits@10']} |
"""

pattern = r'## Path-Gated.*?(?=\n## |\Z)'
if re.search(pattern, md, re.DOTALL):
    md = re.sub(pattern, new_section.strip(), md, flags=re.DOTALL)
else:
    md = md + '\n---\n\n' + new_section

# Update overall summary tables
def rebuild_summary_tables(md):
    models_order = ['TransE','RotatE','ProbCBR','Simple Ensemble','Path-Gated']

    collected = {}
    for model in ['TransE','RotatE','ProbCBR']:
        for split in ['test','valid']:
            vals = defaultdict(list)
            for sl in range(5):
                p = PRED / model / f'slice_{sl}' / f'predictions_{split}.tsv'
                if p.exists():
                    df = pd.read_csv(p, sep='\t')
                    vals['MRR'].append(round(df['reciprocal_rank'].mean(),4))
                    for h in [1,3,5,10]:
                        vals[f'Hits@{h}'].append(round((df['rank']<=h).mean(),4))
            if vals['MRR']:
                collected[(model,split)] = {k: f'{np.mean(v):.4f} ± {np.std(v):.4f}'
                                            for k,v in vals.items()}

    # Ensemble
    ens_vals = defaultdict(lambda: defaultdict(list))
    for sl in range(5):
        for split in ['test','valid']:
            p = PRED / 'Ensemble' / f'slice_{sl}' / f'predictions_{split}.tsv'
            if p.exists():
                df = pd.read_csv(p, sep='\t')
                ens_vals[split]['MRR'].append(round(df['reciprocal_rank'].mean(),4))
                for h in [1,3,5,10]:
                    ens_vals[split][f'Hits@{h}'].append(round((df['rank']<=h).mean(),4))
    for split in ['test','valid']:
        if ens_vals[split]['MRR']:
            collected[('Simple Ensemble',split)] = {
                k: f'{np.mean(v):.4f} ± {np.std(v):.4f}'
                for k,v in ens_vals[split].items()}

    # Path-Gated
    pg_vals = defaultdict(lambda: defaultdict(list))
    for sl in range(5):
        for split in ['test','valid']:
            k = (sl,split)
            if k in all_results:
                m = all_results[k]
                pg_vals[split]['MRR'].append(m['MRR'])
                for h in [1,3,5,10]:
                    pg_vals[split][f'Hits@{h}'].append(m[f'Hits@{h}'])
    for split in ['test','valid']:
        if pg_vals[split]['MRR']:
            collected[('Path-Gated',split)] = {
                k: f'{np.mean(v):.4f} ± {np.std(v):.4f}'
                for k,v in pg_vals[split].items()}

    for split_label in ['test','valid']:
        header = f'## Overall Summary ({split_label} set, mean ± std across 5 slices)'
        table  = (f'{header}\n\n'
                  f'| Model           | MRR             | Hits@1          | Hits@3          | Hits@5          | Hits@10         |\n'
                  f'|-----------------|-----------------|-----------------|-----------------|-----------------|------------------|\n')
        for model in models_order:
            s = collected.get((model, split_label), {})
            def g(k): return s.get(k,'—')
            table += (f'| {model:15s} | {g("MRR"):15s} | {g("Hits@1"):15s} | '
                      f'{g("Hits@3"):15s} | {g("Hits@5"):15s} | {g("Hits@10"):15s} |\n')

        pat = rf'{re.escape(header)}.*?(?=\n## |\Z)'
        if re.search(pat, md, re.DOTALL):
            md = re.sub(pat, table.strip(), md, flags=re.DOTALL)
        else:
            md += '\n---\n\n' + table
    return md

md = rebuild_summary_tables(md)
RESULTS.write_text(md)
print(f'\nUpdated: {RESULTS}')
print('Done.')
