"""
01_prepare_splits.py
--------------------
Loads the MIND dataset and generates N random 80/10/10 train/test/valid splits
of the drug->indication->disease triples (FDA-approved indications only).

The full MRN graph is used as training background for KGE methods,
but evaluation is restricted to indication triples — consistent with the
original paper and justifiable as FDA-approved gold standard (R1 #1).
"""

import argparse
import os
import random
import pandas as pd
import numpy as np
from pathlib import Path
import yaml


def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_mind(path: str) -> pd.DataFrame:
    """Load MIND TSV. Expected columns: head, relation, tail."""
    df = pd.read_csv(path, sep="\t", header=None, names=["head", "relation", "tail"])
    print(f"Loaded {len(df):,} triples, {df['relation'].nunique()} relation types")
    return df


def extract_indications(df: pd.DataFrame, relation: str) -> pd.DataFrame:
    """Extract only drug->indication->disease triples (FDA-validated gold standard)."""
    indications = df[df["relation"] == relation].reset_index(drop=True)
    print(f"Found {len(indications):,} indication triples for relation '{relation}'")
    return indications


def make_split(indications: pd.DataFrame, train_r: float, test_r: float,
               valid_r: float, seed: int) -> dict:
    """Randomly split indication triples into train/test/valid."""
    assert abs(train_r + test_r + valid_r - 1.0) < 1e-6
    idx = list(range(len(indications)))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n = len(idx)
    n_test  = int(n * test_r)
    n_valid = int(n * valid_r)
    n_train = n - n_test - n_valid

    train_idx = idx[:n_train]
    test_idx  = idx[n_train:n_train + n_test]
    valid_idx = idx[n_train + n_test:]

    return {
        "train": indications.iloc[train_idx].reset_index(drop=True),
        "test":  indications.iloc[test_idx].reset_index(drop=True),
        "valid": indications.iloc[valid_idx].reset_index(drop=True),
    }


def save_split(split: dict, full_graph: pd.DataFrame, split_dir: Path):
    """
    Save split files. KGE models receive the full graph as training triples,
    but evaluation is on indication test/valid splits only.
    """
    split_dir.mkdir(parents=True, exist_ok=True)

    # For KGE: combine full background graph + indication train triples
    # (avoids data leakage of test/valid indications into KGE training)
    non_indication = full_graph[full_graph["relation"] != "indication"]
    kge_train = pd.concat([non_indication, split["train"]], ignore_index=True)
    kge_train.to_csv(split_dir / "kge_train.tsv", sep="\t", index=False, header=False)

    # Indication-only splits for evaluation
    split["train"].to_csv(split_dir / "ind_train.tsv", sep="\t", index=False, header=False)
    split["test"].to_csv(split_dir  / "ind_test.tsv",  sep="\t", index=False, header=False)
    split["valid"].to_csv(split_dir / "ind_valid.tsv", sep="\t", index=False, header=False)

    # Save entity and relation mappings
    all_entities = pd.unique(full_graph[["head", "tail"]].values.ravel())
    all_relations = full_graph["relation"].unique()
    pd.Series(all_entities).to_csv(split_dir / "entities.txt", index=False, header=False)
    pd.Series(all_relations).to_csv(split_dir / "relations.txt", index=False, header=False)

    stats = {
        "kge_train": len(kge_train),
        "ind_train": len(split["train"]),
        "ind_test":  len(split["test"]),
        "ind_valid": len(split["valid"]),
        "n_entities": len(all_entities),
        "n_relations": len(all_relations),
    }
    pd.Series(stats).to_csv(split_dir / "stats.csv")
    print(f"  Saved to {split_dir}: {stats}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  default="data/mind.tsv")
    parser.add_argument("--output_dir", default="data/splits")
    parser.add_argument("--n_splits",   type=int, default=5)
    parser.add_argument("--config",     default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dc  = cfg["data"]

    full_graph  = load_mind(args.data_path)
    indications = extract_indications(full_graph, dc["drug_treats_disease_relation"])

    out = Path(args.output_dir)
    seeds = [dc["random_seed"] + i * 100 for i in range(args.n_splits)]

    print(f"\nGenerating {args.n_splits} splits with seeds: {seeds}")
    for i, seed in enumerate(seeds):
        print(f"\n--- Split {i} (seed={seed}) ---")
        split = make_split(
            indications,
            dc["train_ratio"], dc["test_ratio"], dc["valid_ratio"],
            seed=seed,
        )
        save_split(split, full_graph, out / f"slice_{i}")

    print(f"\nDone. Splits saved to {out}/")


if __name__ == "__main__":
    main()
