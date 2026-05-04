"""
run_probcbr_local.py
--------------------
Runs ProbCBR locally on a given slice of splits_updated.
Saves: slice_N.pkl, predictions_test.tsv, predictions_valid.tsv, path_lookup.tsv

Usage:
    python scripts/run_probcbr_local.py --slice 0
    python scripts/run_probcbr_local.py --slice 0 --splits_dir data/splits_updated
"""

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from prob_cbr_proper import ProbCBR

# ── Args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--slice',      type=int, default=0)
parser.add_argument('--splits_dir', type=str, default='data/splits_updated')
parser.add_argument('--out_dir',    type=str, default='results/predictions_updated/ProbCBR')
parser.add_argument('--k',          type=int, default=100)
parser.add_argument('--max_path_len', type=int, default=3)
parser.add_argument('--n_clusters', type=int, default=10)
args = parser.parse_args()

BASE    = Path(__file__).parent.parent
SPLITS  = BASE / args.splits_dir / f'slice_{args.slice}'
OUT     = BASE / args.out_dir / f'slice_{args.slice}'
PKL_DIR = BASE / 'results/models_updated/ProbCBR'

OUT.mkdir(parents=True, exist_ok=True)
PKL_DIR.mkdir(parents=True, exist_ok=True)

print(f'Slice      : {args.slice}')
print(f'Splits dir : {SPLITS}')
print(f'Output dir : {OUT}')

# ── Load data ─────────────────────────────────────────────────────────────
print('\nLoading splits...')
kge_train = pd.read_csv(SPLITS / 'kge_train.tsv', sep='\t', header=None,
                         names=['head','relation','tail'], low_memory=False)
ind_train = pd.read_csv(SPLITS / 'ind_train.tsv', sep='\t', header=None,
                         names=['head','relation','tail'])
ind_test  = pd.read_csv(SPLITS / 'ind_test.tsv',  sep='\t', header=None,
                         names=['head','relation','tail'])
ind_valid = pd.read_csv(SPLITS / 'ind_valid.tsv', sep='\t', header=None,
                         names=['head','relation','tail'])
entities  = pd.read_csv(SPLITS / 'entities.txt',  header=None)[0].tolist()

print(f'kge_train : {len(kge_train):,} triples')
print(f'ind_train : {len(ind_train):,}')
print(f'ind_test  : {len(ind_test):,}')
print(f'ind_valid : {len(ind_valid):,}')
print(f'entities  : {len(entities):,}')

# ── Fit ProbCBR ───────────────────────────────────────────────────────────
print('\nFitting ProbCBR...')
INDICATION_REL = 'indication'

train_triples = list(kge_train[['head','relation','tail']].itertuples(index=False, name=None))
all_entities  = set(entities)

model = ProbCBR(k=args.k, max_path_len=args.max_path_len,
                n_clusters=args.n_clusters)
model.fit(train_triples, all_entities, query_relation=INDICATION_REL)

# Save pickle
pkl_path = PKL_DIR / f'slice_{args.slice}.pkl'
with open(pkl_path, 'wb') as f:
    pickle.dump(model, f)
print(f'Saved model: {pkl_path}')

# ── Predict ───────────────────────────────────────────────────────────────
INDICATION_REL = 'indication'

for split, ind_df in [('test', ind_test), ('valid', ind_valid)]:
    print(f'\nPredicting {split} ({len(ind_df):,} queries)...')
    queries = list(ind_df[['head','relation','tail']].itertuples(index=False, name=None))
    queries = [(h, r, t) for h, r, t in queries]

    pred_df = model.predict(queries, all_entities)

    # Add top-K disease columns (top 50)
    TOP_K = 50
    print(f'  Building top-{TOP_K} columns...')
    top_diseases_list = []
    for _, row in tqdm(pred_df.iterrows(), total=len(pred_df), desc='  top-K'):
        drug = row['drug']
        scores = model.predict_one(drug, INDICATION_REL, all_entities)
        sorted_cands = sorted(scores, key=scores.get, reverse=True)[:TOP_K]
        top_diseases_list.append(sorted_cands)

    for k in range(1, TOP_K + 1):
        pred_df[f'top{k}_disease'] = [
            cands[k-1] if k-1 < len(cands) else ''
            for cands in top_diseases_list
        ]

    out_path = OUT / f'predictions_{split}.tsv'
    pred_df.to_csv(out_path, sep='\t', index=False)
    print(f'  Saved: {out_path}')

    mrr = pred_df['reciprocal_rank'].mean()
    h10 = (pred_df['rank'] <= 10).mean()
    print(f'  MRR={mrr:.4f}  Hits@10={h10:.4f}')

# ── Build path lookup ─────────────────────────────────────────────────────
print('\nBuilding path lookup...')
path_rows = []

for split, ind_df in [('test', ind_test), ('valid', ind_valid)]:
    for _, row in tqdm(ind_df.iterrows(), total=len(ind_df), desc=f'  paths {split}'):
        drug, _, disease = row['head'], row['relation'], row['tail']
        cluster = model.entity_clusters.get(drug, 0)
        cluster_paths = model.path_stats.get(cluster, {}).get(INDICATION_REL, {})

        for rank, (path, (freq, precision)) in enumerate(
            sorted(cluster_paths.items(),
                   key=lambda x: x[1][0]*x[1][1], reverse=True)[:10], 1
        ):
            path_score = freq * precision
            path_rows.append({
                'drug'       : drug,
                'disease'    : disease,
                'path_rank'  : rank,
                'path_score' : path_score,
                'path'       : ' -> '.join(path),
                'is_expected': True,
                'split'      : split,
            })

path_df = pd.DataFrame(path_rows)
path_path = OUT / 'path_lookup.tsv'
path_df.to_csv(path_path, sep='\t', index=False)
print(f'Saved path lookup: {path_path} ({len(path_df):,} rows)')
print('\nDone.')
