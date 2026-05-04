"""
build_pathgated_lookup.py
-------------------------
Generates path lookup files for Path-Gated re-ranking.
For each drug in RotatE's top-50 candidates, scores mechanistic paths
using the saved ProbCBR model.

Usage:
    python scripts/build_pathgated_lookup.py --slice 0
    python scripts/build_pathgated_lookup.py --all
"""

import argparse
import pickle
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--slice', type=int, default=None)
parser.add_argument('--all',   action='store_true')
parser.add_argument('--top_k', type=int, default=50)
args = parser.parse_args()

slices = list(range(5)) if args.all else [args.slice]
assert slices[0] is not None, 'Provide --slice N or --all'

BASE      = Path(__file__).parent.parent
PRED      = BASE / 'results_updated/predictions'
MODEL_DIR = BASE / 'results_updated/models/ProbCBR'
OUT_DIR   = PRED / 'PathGated'
TOP_K     = args.top_k

sys.path.insert(0, str(Path(__file__).parent))
from prob_cbr_proper import ProbCBR  # needed for pickle.load

# ── Fast helpers (reuse from Kaggle notebook logic) ───────────────────────

def build_rel_index(bio_graph):
    idx = defaultdict(lambda: defaultdict(list))
    for node, edges in bio_graph.items():
        for rel, nb in edges:
            idx[rel][node].append(nb)
    return idx

def build_cluster_matrices(model):
    dim = len(model.relation_to_id)
    matrices = {}
    for cid, drugs in model.cluster_drugs.items():
        vecs = np.array([model.entity_vectors.get(d, np.zeros(dim, dtype=np.float32))
                         for d in drugs], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrices[cid] = (drugs, vecs / norms)
    return matrices, dim

def assign_cluster(drug, model, cluster_matrices, dim):
    if drug in model.entity_clusters:
        return model.entity_clusters[drug]
    vec  = model.entity_vectors.get(drug, np.zeros(dim, dtype=np.float32))
    norm = np.linalg.norm(vec)
    if norm == 0 or model.cluster_centroids is None:
        return 0
    sims = np.dot(model.cluster_centroids, vec / norm)
    return int(np.argmax(sims))

def fast_contextual(drug, model, cluster_matrices, dim):
    cluster = assign_cluster(drug, model, cluster_matrices, dim)
    if cluster not in cluster_matrices:
        return []
    drugs_list, normed = cluster_matrices[cluster]
    q_vec  = model.entity_vectors.get(drug, np.zeros(dim, dtype=np.float32))
    q_norm = np.linalg.norm(q_vec)
    if q_norm == 0:
        return drugs_list[:model.k]
    sims    = normed @ (q_vec / q_norm)
    top_idx = np.argsort(sims)[::-1][:model.k]
    return [drugs_list[j] for j in top_idx]

MAX_TRACES = 200

def score_candidates(drug, candidates, model, rel_index, cluster_matrices, dim):
    """Score a specific set of candidate diseases for a drug using ProbCBR paths."""
    contextual = fast_contextual(drug, model, cluster_matrices, dim)
    if not contextual:
        return {}

    path_freq_all     = defaultdict(int)
    path_freq_correct = defaultdict(int)
    for ctx in contextual:
        info = model.drug_paths.get(ctx, {})
        for path_type, ends in info.get('paths', {}).items():
            path_freq_all[path_type]     += len(ends)
            path_freq_correct[path_type] += sum(1 for e in ends if e in info.get('correct', set()))

    total_correct = sum(path_freq_correct.values()) or 1
    scores        = defaultdict(float)
    best_paths    = {}

    for path_type in path_freq_all:
        freq_all     = path_freq_all[path_type]
        freq_correct = path_freq_correct[path_type]
        if freq_all < model.min_path_freq or freq_correct == 0:
            continue

        precision    = freq_correct / freq_all
        prior        = freq_correct / total_correct
        contribution = prior * precision

        curr_nodes = {drug}
        valid = True
        for rel in path_type:
            nxt_nodes = set()
            for node in curr_nodes:
                nxt_nodes.update(rel_index[rel].get(node, []))
            if not nxt_nodes:
                valid = False
                break
            if len(nxt_nodes) > MAX_TRACES:
                nxt_nodes = set(list(nxt_nodes)[:MAX_TRACES])
            curr_nodes = nxt_nodes

        if valid:
            reached = curr_nodes & set(candidates)
            for disease in reached:
                scores[disease] += contribution
                path_str = ' '.join(
                    f'{drug if i==0 else "?"} --[{path_type[i]}]-->' for i in range(len(path_type))
                ) + f' {disease}'
                if disease not in best_paths or contribution > best_paths[disease][0]:
                    best_paths[disease] = (contribution, path_str)

    return {d: (scores[d], best_paths[d][1]) for d in scores if d in best_paths}


# ── Main loop ─────────────────────────────────────────────────────────────

for sl in slices:
    print(f'\n{"="*50}\nPath lookup — slice_{sl}\n{"="*50}')

    pkl_path = MODEL_DIR / f'slice_{sl}.pkl'
    print(f'Loading model: {pkl_path}')
    with open(pkl_path, 'rb') as f:
        model_data = pickle.load(f)

    # Handle both (model, n_ents) tuple and bare model
    if isinstance(model_data, tuple):
        model, _ = model_data
    else:
        model = model_data

    rel_index             = build_rel_index(model.bio_graph)
    cluster_matrices, dim = build_cluster_matrices(model)
    print(f'  rel_index: {sum(len(v) for v in rel_index.values()):,} entries')

    outdir = OUT_DIR / f'slice_{sl}'
    outdir.mkdir(parents=True, exist_ok=True)

    for split in ['test', 'valid']:
        rotate_df = pd.read_csv(PRED / f'RotatE/slice_{sl}/predictions_{split}.tsv', sep='\t')
        ind_df    = pd.read_csv(BASE / f'data/splits_updated/slice_{sl}/ind_{split}.tsv',
                                sep='\t', header=None, names=['head','relation','tail'])

        known = set(zip(ind_df['head'], ind_df['tail']))
        path_rows = []
        covered = 0

        for _, row in tqdm(rotate_df.iterrows(), total=len(rotate_df), desc=f'  {split}'):
            drug    = row['drug']
            exp_dis = row['expected_disease']

            # Collect RotatE's top-K candidates
            candidates = []
            for k in range(1, TOP_K + 1):
                d = row.get(f'top{k}_disease', '')
                if d:
                    candidates.append(d)
            if exp_dis not in candidates:
                candidates.append(exp_dis)

            scored = score_candidates(drug, candidates, model, rel_index, cluster_matrices, dim)
            if scored:
                covered += 1

            # Sort by score descending → path_rank
            sorted_cands = sorted(scored.items(), key=lambda x: x[1][0], reverse=True)
            for rank, (disease, (score, path_str)) in enumerate(sorted_cands, 1):
                path_rows.append({
                    'drug'       : drug,
                    'disease'    : disease,
                    'path_rank'  : rank,
                    'path_score' : round(score, 6),
                    'path'       : path_str,
                    'is_expected': (drug, disease) in known,
                    'split'      : split,
                })

        path_df = pd.DataFrame(path_rows)
        out_path = outdir / f'path_lookup_rotate_{split}_slice{sl}.tsv'
        path_df.to_csv(out_path, sep='\t', index=False)

        pairs    = len(path_df.groupby(['drug','disease']))
        coverage = covered / len(rotate_df)
        print(f'  {split}: {len(path_df):,} path rows | {pairs:,} drug-disease pairs | '
              f'coverage {coverage:.1%} | saved → {out_path.name}')

print('\nDone.')
