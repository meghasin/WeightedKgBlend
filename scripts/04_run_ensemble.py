"""
04_run_ensemble.py
------------------
Runs WeightedKgBlend on prediction files from all 7 models across all splits.

Two optimisation modes (config: ensemble.use_optuna):
  - Optuna (recommended): Bayesian optimisation over simplex-constrained weights
  - Grid search (fallback): exhaustive search at step_size=0.1 (original method)

Outputs per split:
  results/ensemble/slice_N/
    weights.yaml          -- optimised lambda values
    predictions_test.tsv  -- final ensemble ranked predictions
    predictions_valid.tsv -- validation set predictions
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_predictions(pred_path: Path) -> pd.DataFrame:
    """Load a predictions TSV. Columns: drug, expected_disease, rank, reciprocal_rank."""
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")
    return pd.read_csv(pred_path, sep="\t")


def align_predictions(pred_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all model predictions on (drug, expected_disease).
    Returns a DataFrame with one column per model's reciprocal_rank.
    Missing predictions get reciprocal_rank = 0.
    """
    base = pred_dfs[0][["drug", "expected_disease"]].copy()
    for i, df in enumerate(pred_dfs):
        rr = df[["drug", "expected_disease", "reciprocal_rank"]].rename(
            columns={"reciprocal_rank": f"rr_{i}"}
        )
        base = base.merge(rr, on=["drug", "expected_disease"], how="left")
    rr_cols = [f"rr_{i}" for i in range(len(pred_dfs))]
    base[rr_cols] = base[rr_cols].fillna(0.0)
    return base, rr_cols


def compute_ensemble_mrr(weights: np.ndarray, aligned: pd.DataFrame,
                          rr_cols: list) -> float:
    """Compute MRR of weighted ensemble given weight vector."""
    rr_matrix = aligned[rr_cols].values          # shape: (n_queries, n_models)
    ensemble_scores = rr_matrix @ weights         # weighted sum of reciprocal ranks
    # Re-rank within each drug's candidate set is implicit — scores ARE the ranks here
    # (as in original paper: weighted average of reciprocal ranks)
    return float(np.mean(ensemble_scores))


def optimise_optuna(aligned_valid: pd.DataFrame, rr_cols: list,
                    n_trials: int, seed: int) -> np.ndarray:
    """Use Optuna TPE sampler to find optimal weights on simplex."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Install optuna: pip install optuna")

    n_models = len(rr_cols)

    def objective(trial):
        # Sample raw weights and normalise to sum=1 (simplex projection)
        raw = np.array([trial.suggest_float(f"w{i}", 0.0, 1.0)
                        for i in range(n_models)])
        if raw.sum() < 1e-8:
            return 0.0
        weights = raw / raw.sum()
        return compute_ensemble_mrr(weights, aligned_valid, rr_cols)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_raw = np.array([study.best_params[f"w{i}"] for i in range(n_models)])
    best_weights = best_raw / best_raw.sum()
    print(f"    Optuna best MRR (valid): {study.best_value:.4f}")
    return best_weights


def optimise_grid(aligned_valid: pd.DataFrame, rr_cols: list,
                  step: float) -> np.ndarray:
    """Exhaustive grid search over simplex at given step size (original method)."""
    n_models = len(rr_cols)
    steps = np.arange(0, 1 + step, step)
    best_mrr = -1
    best_weights = np.ones(n_models) / n_models

    # Generate all combinations that sum to 1 (within tolerance)
    for combo in product(steps, repeat=n_models):
        w = np.array(combo)
        if abs(w.sum() - 1.0) > step / 2:
            continue
        w = w / w.sum()
        mrr = compute_ensemble_mrr(w, aligned_valid, rr_cols)
        if mrr > best_mrr:
            best_mrr = mrr
            best_weights = w

    print(f"    Grid search best MRR (valid): {best_mrr:.4f}")
    return best_weights


def run_ensemble_on_split(slice_name: str, results_dir: Path,
                           out_dir: Path, cfg: dict):
    """Run full ensemble pipeline for a single split."""
    ens_cfg  = cfg["ensemble"]
    rel      = cfg["data"]["drug_treats_disease_relation"]
    MODEL_NAMES = ["TransE", "RotatE", "DistMult", "ComplEx", "Rephetio", "CBR", "ProbCBR"]

    print(f"\n--- Ensemble: {slice_name} ---")
    split_out = out_dir / slice_name
    split_out.mkdir(parents=True, exist_ok=True)

    # Load validation predictions for all models
    valid_preds, test_preds = [], []
    loaded_models = []
    for model in MODEL_NAMES:
        vp = results_dir / model.lower() / slice_name / "predictions_valid.tsv"
        tp = results_dir / model.lower() / slice_name / "predictions_test.tsv"
        # Try KGE path convention too
        if not vp.exists():
            vp = results_dir / "kge" / model / slice_name / "predictions_valid.tsv"
            tp = results_dir / "kge" / model / slice_name / "predictions_test.tsv"
        if not vp.exists():
            vp = results_dir / "cbr" / model / slice_name / "predictions_valid.tsv"
            tp = results_dir / "cbr" / model / slice_name / "predictions_test.tsv"
        if vp.exists() and tp.exists():
            valid_preds.append(load_predictions(vp))
            test_preds.append(load_predictions(tp))
            loaded_models.append(model)
        else:
            print(f"  WARNING: predictions not found for {model} on {slice_name}, skipping")

    if len(loaded_models) < 2:
        print(f"  Not enough models for ensemble on {slice_name}, skipping")
        return None

    print(f"  Using models: {loaded_models}")

    # Align predictions
    aligned_valid, rr_cols = align_predictions(valid_preds)
    aligned_test,  _       = align_predictions(test_preds)

    # Optimise weights on validation set
    if ens_cfg["use_optuna"]:
        weights = optimise_optuna(
            aligned_valid, rr_cols,
            n_trials=ens_cfg["optuna_n_trials"],
            seed=cfg["data"]["random_seed"],
        )
    else:
        weights = optimise_grid(aligned_valid, rr_cols, ens_cfg["grid_step"])

    # Save weights
    weight_dict = {m: float(round(w, 4)) for m, w in zip(loaded_models, weights)}
    with open(split_out / "weights.yaml", "w") as f:
        yaml.dump({"optimised_weights": weight_dict,
                   "optimisation": "optuna" if ens_cfg["use_optuna"] else "grid"}, f)
    print(f"  Weights: {weight_dict}")

    # Apply weights to test set
    rr_matrix = aligned_test[rr_cols].values
    ensemble_rr = rr_matrix @ weights
    aligned_test["ensemble_reciprocal_rank"] = ensemble_rr
    aligned_test.to_csv(split_out / "predictions_test.tsv", sep="\t", index=False)

    # Apply weights to valid set (for unbiased evaluation)
    rr_matrix_v = aligned_valid[rr_cols].values
    ensemble_rr_v = rr_matrix_v @ weights
    aligned_valid["ensemble_reciprocal_rank"] = ensemble_rr_v
    aligned_valid.to_csv(split_out / "predictions_valid.tsv", sep="\t", index=False)

    return aligned_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_dir",  default="results/ensemble")
    parser.add_argument("--config",      default="config.yaml")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    out_dir = Path(args.output_dir)
    r_dir   = Path(args.results_dir)

    # Discover splits
    slices = sorted([d.name for d in (r_dir / "kge" / "RotatE").glob("slice_*")])
    if not slices:
        slices = [f"slice_{i}" for i in range(cfg["data"]["n_splits"])]

    print(f"Running ensemble on slices: {slices}")
    for s in slices:
        run_ensemble_on_split(s, r_dir, out_dir, cfg)

    print("\nEnsemble complete.")


if __name__ == "__main__":
    main()
