"""
02_run_kge.py
-------------
Trains TransE, RotatE, DistMult, ComplEx, and Rephetio on each data split
using PyKEEN (PyTorch backend). Saves ranked prediction files per split.

Each model outputs: results/kge/<model>/<slice_N>/predictions_test.tsv
Columns: drug, disease, reciprocal_rank, rank
"""

import argparse
import os
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.models import TransE, RotatE, DistMult, ComplEx


MODEL_MAP = {
    "TransE":    TransE,
    "RotatE":    RotatE,
    "DistMult":  DistMult,
    "ComplEx":   ComplEx,
}


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_triples_factory(tsv_path: Path, entity_to_id=None, relation_to_id=None):
    """Load a TSV triple file into a PyKEEN TriplesFactory."""
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["head", "relation", "tail"])
    triples = df[["head", "relation", "tail"]].values.astype(str)
    if entity_to_id is None:
        return TriplesFactory.from_labeled_triples(triples)
    else:
        return TriplesFactory.from_labeled_triples(
            triples,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )


def get_ranked_predictions(model, test_factory: TriplesFactory,
                           relation_filter: str, device: str) -> pd.DataFrame:
    """
    For every (drug, indication, ?) query in the test set,
    score all possible tail entities and return ranked results.
    Returns DataFrame with columns: drug, expected_disease, rank, reciprocal_rank
    """
    model.eval()
    results = []

    rel_id = test_factory.relation_to_id.get(relation_filter)
    if rel_id is None:
        raise ValueError(f"Relation '{relation_filter}' not found in factory")

    test_triples = test_factory.mapped_triples
    indication_mask = test_triples[:, 1] == rel_id
    indication_triples = test_triples[indication_mask]

    entity_ids = torch.arange(test_factory.num_entities, device=device)

    with torch.no_grad():
        for triple in indication_triples:
            h, r, t = triple[0].item(), triple[1].item(), triple[2].item()

            # Score all tail entities for this (h, r, ?) query
            h_tensor = torch.tensor([h], device=device).repeat(test_factory.num_entities)
            r_tensor = torch.tensor([r], device=device).repeat(test_factory.num_entities)
            scores = model.score_t(h_tensor, r_tensor)  # shape: (n_entities,)

            # Rank (descending score = rank 1 is best)
            sorted_idx = torch.argsort(scores, descending=True).cpu().numpy()
            rank = int(np.where(sorted_idx == t)[0][0]) + 1  # 1-indexed
            rr   = 1.0 / rank

            drug_name     = test_factory.entity_id_to_label[h]
            disease_name  = test_factory.entity_id_to_label[t]
            results.append({
                "drug": drug_name,
                "expected_disease": disease_name,
                "rank": rank,
                "reciprocal_rank": rr,
            })

    return pd.DataFrame(results)


def run_kge_model(model_name: str, slice_dir: Path, out_dir: Path, cfg: dict):
    """Train a single KGE model on one data split and save predictions."""
    kge_cfg     = cfg["kge"]
    model_cfg   = kge_cfg[model_name]
    device      = kge_cfg["device"] if torch.cuda.is_available() else "cpu"
    relation    = cfg["data"]["drug_treats_disease_relation"]

    print(f"\n  Training {model_name} on {slice_dir.name}...")

    train_factory = load_triples_factory(slice_dir / "kge_train.tsv")
    test_factory  = load_triples_factory(
        slice_dir / "ind_test.tsv",
        entity_to_id=train_factory.entity_to_id,
        relation_to_id=train_factory.relation_to_id,
    )
    valid_factory = load_triples_factory(
        slice_dir / "ind_valid.tsv",
        entity_to_id=train_factory.entity_to_id,
        relation_to_id=train_factory.relation_to_id,
    )

    result = pipeline(
        training=train_factory,
        testing=test_factory,
        validation=valid_factory,
        model=model_name,
        model_kwargs=dict(embedding_dim=model_cfg["embedding_dim"]),
        optimizer="Adam",
        optimizer_kwargs=dict(lr=model_cfg["lr"]),
        training_kwargs=dict(
            num_epochs=kge_cfg["num_epochs"],
            batch_size=kge_cfg["batch_size"],
        ),
        stopper="early",
        stopper_kwargs=dict(patience=kge_cfg["early_stopping_patience"]),
        evaluator_kwargs=dict(filtered=True),
        device=device,
        random_seed=42,
    )

    # Save model
    model_out = out_dir / model_name / slice_dir.name
    model_out.mkdir(parents=True, exist_ok=True)
    result.save_to_directory(str(model_out))

    # Generate and save ranked predictions on test set
    preds = get_ranked_predictions(
        result.model, test_factory, relation, device
    )
    pred_path = model_out / "predictions_test.tsv"
    preds.to_csv(pred_path, sep="\t", index=False)
    print(f"  Saved {len(preds)} test predictions -> {pred_path}")

    # Also save validation predictions (needed for ensemble weight optimisation)
    valid_preds = get_ranked_predictions(
        result.model, valid_factory, relation, device
    )
    valid_preds.to_csv(model_out / "predictions_valid.tsv", sep="\t", index=False)

    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir",  default="data/splits")
    parser.add_argument("--output_dir",  default="results/kge")
    parser.add_argument("--models",      nargs="+",
                        default=["TransE", "RotatE", "DistMult", "ComplEx"])
    parser.add_argument("--config",      default="config.yaml")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    splits   = sorted(Path(args.splits_dir).glob("slice_*"))
    out_dir  = Path(args.output_dir)

    print(f"Found {len(splits)} splits: {[s.name for s in splits]}")
    print(f"Models to train: {args.models}")

    for slice_dir in splits:
        for model_name in args.models:
            run_kge_model(model_name, slice_dir, out_dir, cfg)

    print("\nAll KGE models complete.")


if __name__ == "__main__":
    main()
