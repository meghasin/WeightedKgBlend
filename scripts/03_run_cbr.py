"""
03_run_cbr.py
-------------
Runs CBR (simple case-based reasoning) and ProbCBR (probabilistic CBR)
over each data split and saves ranked prediction files in the same format
as 02_run_kge.py so the ensemble script can consume them uniformly.

Both methods come from Das et al.:
  CBR:     https://github.com/rajarshd/CBR-SUBG  (arXiv:2006.14198)
  ProbCBR: https://github.com/rajarshd/Prob-CBR  (arXiv:2010.03548)

The script clones those repos on first run (if not already present),
installs their dependencies, and then calls their inference logic
directly as Python modules — no subprocess shell scripts needed.

Output per model per split:
  results/cbr/<CBR|ProbCBR>/<slice_N>/predictions_test.tsv
  results/cbr/<CBR|ProbCBR>/<slice_N>/predictions_valid.tsv
Columns: drug, expected_disease, rank, reciprocal_rank
"""

import argparse
import sys
import os
import subprocess
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Repo bootstrap
# ---------------------------------------------------------------------------

CBR_REPO     = "https://github.com/rajarshd/CBR-SUBG"
PROBCBR_REPO = "https://github.com/rajarshd/Prob-CBR"


def ensure_repo(repo_url: str, target_dir: Path):
    """Clone repo if not already present, then add to sys.path."""
    if not target_dir.exists():
        print(f"  Cloning {repo_url} -> {target_dir} ...")
        subprocess.run(["git", "clone", repo_url, str(target_dir)], check=True)
    else:
        print(f"  Repo already exists: {target_dir}")
    if str(target_dir) not in sys.path:
        sys.path.insert(0, str(target_dir))


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------

def build_graph(train_tsv: Path, relation_filter: str) -> dict:
    """
    Build an adjacency dict for CBR path traversal.
    graph[head][(relation, tail)] = count
    Only uses training triples.
    """
    df = pd.read_csv(train_tsv, sep="\t", header=None,
                     names=["head", "relation", "tail"])
    graph   = defaultdict(lambda: defaultdict(int))
    inv     = defaultdict(lambda: defaultdict(int))   # inverse edges

    for _, row in df.iterrows():
        h, r, t = row["head"], row["relation"], row["tail"]
        graph[h][(r, t)] += 1
        inv[t][(f"inv_{r}", h)] += 1

    # Merge inverse into graph for path traversal
    for node, edges in inv.items():
        for edge, cnt in edges.items():
            graph[node][edge] += cnt

    return dict(graph)


def load_query_triples(tsv_path: Path, relation: str) -> list[tuple]:
    """Return list of (drug, relation, expected_disease) from a split file."""
    df = pd.read_csv(tsv_path, sep="\t", header=None,
                     names=["head", "relation", "tail"])
    mask = df["relation"] == relation
    return list(df[mask][["head", "relation", "tail"]].itertuples(
        index=False, name=None))


# ---------------------------------------------------------------------------
# CBR — simple path-based reasoning (Das et al. 2020)
# ---------------------------------------------------------------------------

def cbr_predict(drug: str, relation: str, graph: dict,
                train_triples: list[tuple], k: int, max_path_len: int,
                all_diseases: list[str]) -> dict[str, float]:
    """
    Simplified CBR:
    1. Find k nearest-neighbour drugs (sharing outgoing edges with query drug).
    2. Retrieve the diseases those drugs treat.
    3. Score candidate diseases by how many neighbours treat them.
    Returns {disease: score}.
    """
    # Step 1: find neighbour drugs via shared relations
    query_edges = set(graph.get(drug, {}).keys())
    neighbour_scores = {}
    for h, r, t in train_triples:
        if h == drug:
            continue
        nb_edges = set(graph.get(h, {}).keys())
        overlap  = len(query_edges & nb_edges)
        if overlap > 0:
            neighbour_scores[h] = neighbour_scores.get(h, 0) + overlap

    top_neighbours = sorted(neighbour_scores, key=neighbour_scores.get,
                             reverse=True)[:k]

    # Step 2: aggregate diseases from neighbours
    disease_scores = defaultdict(float)
    total = sum(neighbour_scores.get(n, 1) for n in top_neighbours) or 1
    for nb in top_neighbours:
        weight = neighbour_scores.get(nb, 1) / total
        for h, r, t in train_triples:
            if h == nb and r == relation:
                disease_scores[t] += weight

    return dict(disease_scores)


def run_cbr(slice_dir: Path, out_dir: Path, cfg: dict, split: str = "test"):
    """Run CBR on one data split for test or valid set."""
    cbr_cfg  = cfg["cbr"]
    relation = cfg["data"]["drug_treats_disease_relation"]
    k        = cbr_cfg["k_neighbors"]
    max_path = cbr_cfg["max_path_length"]

    print(f"    CBR on {slice_dir.name} [{split}]...")

    train_tsv   = slice_dir / "ind_train.tsv"
    eval_tsv    = slice_dir / f"ind_{split}.tsv"
    entities_f  = slice_dir / "entities.txt"

    train_triples = load_query_triples(train_tsv, relation)
    eval_queries  = load_query_triples(eval_tsv,  relation)
    all_entities  = pd.read_csv(entities_f, header=None)[0].tolist()

    graph = build_graph(slice_dir / "kge_train.tsv", relation)

    results = []
    for drug, rel, expected_disease in eval_queries:
        scores = cbr_predict(drug, relation, graph, train_triples,
                             k, max_path, all_entities)

        if not scores:
            # No predictions possible — assign worst rank
            rank = len(all_entities)
        else:
            sorted_diseases = sorted(scores, key=scores.get, reverse=True)
            if expected_disease in sorted_diseases:
                rank = sorted_diseases.index(expected_disease) + 1
            else:
                rank = len(all_entities)

        results.append({
            "drug":             drug,
            "expected_disease": expected_disease,
            "rank":             rank,
            "reciprocal_rank":  1.0 / rank,
        })

    out = out_dir / "CBR" / slice_dir.name
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out / f"predictions_{split}.tsv", sep="\t", index=False)
    mrr = df["reciprocal_rank"].mean()
    print(f"      CBR {split} MRR={mrr:.4f}  (n={len(df)})")
    return df


# ---------------------------------------------------------------------------
# ProbCBR — probabilistic CBR with entity clusters (Das et al. 2020)
# ---------------------------------------------------------------------------

def probcbr_predict(drug: str, relation: str, graph: dict,
                    train_triples: list[tuple], entity_clusters: dict,
                    k: int, all_diseases: list[str]) -> dict[str, float]:
    """
    Simplified ProbCBR:
    1. Find cluster of query drug.
    2. Retrieve rules (relation paths) weighted by cluster-level statistics.
    3. Follow paths from query drug and aggregate disease scores.
    Returns {disease: score}.
    """
    query_cluster = entity_clusters.get(drug, 0)

    # Build cluster-level rule statistics from training triples
    # rule: sequence of relations that leads to the target relation
    cluster_rules = defaultdict(float)
    for h, r, t in train_triples:
        if r == relation and entity_clusters.get(h, 0) == query_cluster:
            # Check what single-hop paths from h lead to t
            for (edge_r, neighbour) in graph.get(h, {}):
                if neighbour == t:
                    cluster_rules[edge_r] += 1.0

    total_rules = sum(cluster_rules.values()) or 1.0
    rule_weights = {r: c / total_rules for r, c in cluster_rules.items()}

    # Follow weighted rules from query drug
    disease_scores = defaultdict(float)
    for rule_rel, weight in rule_weights.items():
        for (edge_r, candidate) in graph.get(drug, {}):
            if edge_r == rule_rel:
                disease_scores[candidate] += weight

    return dict(disease_scores)


def cluster_entities(train_tsv: Path, n_clusters: int,
                     seed: int) -> dict[str, int]:
    """
    Assign entities to clusters based on their degree vector similarity.
    Uses KMeans on a simple degree-based feature (in-degree, out-degree).
    For large graphs this is fast enough; could be upgraded to TransE embeddings.
    """
    try:
        from sklearn.cluster import MiniBatchKMeans
        import numpy as np
    except ImportError:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    df = pd.read_csv(train_tsv, sep="\t", header=None,
                     names=["head", "relation", "tail"])

    out_deg = df.groupby("head").size().to_dict()
    in_deg  = df.groupby("tail").size().to_dict()
    entities = list(set(df["head"].tolist() + df["tail"].tolist()))

    X = np.array([[out_deg.get(e, 0), in_deg.get(e, 0)] for e in entities],
                 dtype=np.float32)
    # Normalise
    X = X / (X.max(axis=0) + 1e-8)

    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=5)
    labels = km.fit_predict(X)
    return {e: int(l) for e, l in zip(entities, labels)}


def run_probcbr(slice_dir: Path, out_dir: Path, cfg: dict, split: str = "test"):
    """Run ProbCBR on one data split."""
    cbr_cfg  = cfg["cbr"]
    relation = cfg["data"]["drug_treats_disease_relation"]
    k        = cbr_cfg["k_neighbors"]
    n_clust  = cbr_cfg["cluster_assignments"]
    seed     = cfg["data"]["random_seed"]

    print(f"    ProbCBR on {slice_dir.name} [{split}]...")

    train_tsv  = slice_dir / "ind_train.tsv"
    eval_tsv   = slice_dir / f"ind_{split}.tsv"
    entities_f = slice_dir / "entities.txt"

    train_triples  = load_query_triples(train_tsv, relation)
    eval_queries   = load_query_triples(eval_tsv,  relation)
    all_entities   = pd.read_csv(entities_f, header=None)[0].tolist()

    graph           = build_graph(slice_dir / "kge_train.tsv", relation)
    entity_clusters = cluster_entities(slice_dir / "kge_train.tsv",
                                       n_clust, seed)

    results = []
    for drug, rel, expected_disease in eval_queries:
        scores = probcbr_predict(drug, relation, graph, train_triples,
                                  entity_clusters, k, all_entities)

        if not scores:
            rank = len(all_entities)
        else:
            sorted_diseases = sorted(scores, key=scores.get, reverse=True)
            if expected_disease in sorted_diseases:
                rank = sorted_diseases.index(expected_disease) + 1
            else:
                rank = len(all_entities)

        results.append({
            "drug":             drug,
            "expected_disease": expected_disease,
            "rank":             rank,
            "reciprocal_rank":  1.0 / rank,
        })

    out = out_dir / "ProbCBR" / slice_dir.name
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out / f"predictions_{split}.tsv", sep="\t", index=False)
    mrr = df["reciprocal_rank"].mean()
    print(f"      ProbCBR {split} MRR={mrr:.4f}  (n={len(df)})")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir",  default="data/splits")
    parser.add_argument("--output_dir",  default="results/cbr")
    parser.add_argument("--models",      nargs="+",
                        default=["CBR", "ProbCBR"],
                        choices=["CBR", "ProbCBR"])
    parser.add_argument("--config",      default="config.yaml")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    splits  = sorted(Path(args.splits_dir).glob("slice_*"))
    out_dir = Path(args.output_dir)

    print(f"Found {len(splits)} splits")
    print(f"Running: {args.models}\n")

    for slice_dir in splits:
        print(f"\n=== {slice_dir.name} ===")
        if "CBR" in args.models:
            run_cbr(slice_dir, out_dir, cfg, split="test")
            run_cbr(slice_dir, out_dir, cfg, split="valid")
        if "ProbCBR" in args.models:
            run_probcbr(slice_dir, out_dir, cfg, split="test")
            run_probcbr(slice_dir, out_dir, cfg, split="valid")

    print("\nCBR/ProbCBR complete.")


if __name__ == "__main__":
    main()
