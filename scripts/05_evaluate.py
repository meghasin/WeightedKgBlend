"""
05_evaluate.py
--------------
Reads prediction files for all 7 models + WeightedKgBlend across all splits.
Computes MRR, Hits@1/3/10 per split, then reports:
  - Mean ± std across splits (addresses R1 #2)
  - Wilcoxon signed-rank test vs WeightedKgBlend (addresses R1 #5)
  - Final Table 1 as CSV and LaTeX

Output: results/final_table.csv, results/final_table.tex, results/stats.csv
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def hits_at_k(rr_series: pd.Series, k: int) -> float:
    """Hits@k: fraction of queries where rank <= k (RR >= 1/k)."""
    return float((rr_series >= 1.0 / k).mean())


def compute_metrics(rr_series: pd.Series) -> dict:
    return {
        "MRR":     float(rr_series.mean()),
        "Hits@1":  hits_at_k(rr_series, 1),
        "Hits@3":  hits_at_k(rr_series, 3),
        "Hits@10": hits_at_k(rr_series, 10),
    }


def load_model_rr(results_dir: Path, model: str, slice_name: str,
                   split: str = "test") -> pd.Series | None:
    """Load per-query reciprocal ranks for a model on a given split."""
    # Try various path conventions
    candidates = [
        results_dir / "kge"      / model / slice_name / f"predictions_{split}.tsv",
        results_dir / "cbr"      / model / slice_name / f"predictions_{split}.tsv",
        results_dir / model.lower() / slice_name      / f"predictions_{split}.tsv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p, sep="\t")
            rr_col = "ensemble_reciprocal_rank" if "ensemble" in str(p) else "reciprocal_rank"
            if rr_col not in df.columns:
                rr_col = [c for c in df.columns if "reciprocal" in c][0]
            return df[rr_col]
    return None


def load_ensemble_rr(results_dir: Path, slice_name: str,
                      split: str = "test") -> pd.Series | None:
    p = results_dir / "ensemble" / slice_name / f"predictions_{split}.tsv"
    if p.exists():
        df = pd.read_csv(p, sep="\t")
        return df["ensemble_reciprocal_rank"]
    return None


def run_wilcoxon(ensemble_rr: pd.Series, model_rr: pd.Series) -> dict:
    """
    Wilcoxon signed-rank test: H0 = ensemble and model have same distribution.
    Uses per-query RR from a single split (or concatenated across splits).
    """
    min_len = min(len(ensemble_rr), len(model_rr))
    e = ensemble_rr.values[:min_len]
    m = model_rr.values[:min_len]
    diff = e - m
    if np.all(diff == 0):
        return {"statistic": np.nan, "p_value": np.nan, "significant": False}
    stat, p = wilcoxon(diff, alternative="greater")
    return {"statistic": float(stat), "p_value": float(p), "significant": bool(p < 0.05)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_dir",  default="results")
    parser.add_argument("--split",       default="test", choices=["test", "valid"])
    parser.add_argument("--config",      default="config.yaml")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    r_dir    = Path(args.results_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    MODEL_NAMES = ["TransE", "RotatE", "DistMult", "ComplEx", "Rephetio", "CBR", "ProbCBR"]
    ALL_MODELS  = MODEL_NAMES + ["WeightedKgBlend"]
    slices      = [f"slice_{i}" for i in range(cfg["data"]["n_splits"])]
    METRICS     = ["MRR", "Hits@1", "Hits@3", "Hits@10"]

    # Collect per-split metrics for each model
    # structure: {model: {metric: [val_slice0, val_slice1, ...]}}
    results = {m: {k: [] for k in METRICS} for m in ALL_MODELS}
    # Also collect raw RR series for Wilcoxon (concatenated across splits)
    raw_rr  = {m: [] for m in ALL_MODELS}

    for slice_name in slices:
        for model in MODEL_NAMES:
            rr = load_model_rr(r_dir, model, slice_name, args.split)
            if rr is None:
                print(f"  WARNING: no predictions for {model}/{slice_name}")
                for k in METRICS:
                    results[model][k].append(np.nan)
            else:
                m = compute_metrics(rr)
                for k in METRICS:
                    results[model][k].append(m[k])
                raw_rr[model].extend(rr.tolist())

        ens_rr = load_ensemble_rr(r_dir, slice_name, args.split)
        if ens_rr is None:
            print(f"  WARNING: no ensemble predictions for {slice_name}")
            for k in METRICS:
                results["WeightedKgBlend"][k].append(np.nan)
        else:
            m = compute_metrics(ens_rr)
            for k in METRICS:
                results["WeightedKgBlend"][k].append(m[k])
            raw_rr["WeightedKgBlend"].extend(ens_rr.tolist())

    # Build summary table: mean ± std
    rows = []
    for model in ALL_MODELS:
        row = {"Algorithm": model}
        for k in METRICS:
            vals = [v for v in results[model][k] if not np.isnan(v)]
            if vals:
                row[k] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
                row[f"{k}_mean"] = np.mean(vals)
                row[f"{k}_std"]  = np.std(vals)
            else:
                row[k] = "N/A"
                row[f"{k}_mean"] = np.nan
                row[f"{k}_std"]  = np.nan
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "final_table.csv", index=False)
    print("\n=== Results (mean ± std across splits) ===")
    print(summary_df[["Algorithm"] + METRICS].to_string(index=False))

    # Wilcoxon signed-rank tests: each model vs WeightedKgBlend
    ens_rr_all = pd.Series(raw_rr["WeightedKgBlend"])
    stat_rows = []
    for model in MODEL_NAMES:
        if not raw_rr[model]:
            continue
        model_rr_all = pd.Series(raw_rr[model])
        w = run_wilcoxon(ens_rr_all, model_rr_all)
        stat_rows.append({
            "Model": model,
            "Wilcoxon_stat": w["statistic"],
            "p_value": w["p_value"],
            "significant (p<0.05)": w["significant"],
        })

    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(out_dir / "stats.csv", index=False)
    print("\n=== Wilcoxon signed-rank: WeightedKgBlend vs each model ===")
    print(stats_df.to_string(index=False))

    # LaTeX table
    latex_cols = ["Algorithm"] + METRICS
    latex = summary_df[latex_cols].to_latex(
        index=False, escape=False,
        caption="Comparative results (mean ± std across 5 random splits). "
                "Bold = best per metric. * = WeightedKgBlend significantly better "
                "(Wilcoxon p<0.05).",
        label="tab:results",
    )
    with open(out_dir / "final_table.tex", "w") as f:
        f.write(latex)

    print(f"\nOutputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
