# WeightedKgBlend

**Weighted Ensemble Approach for Knowledge Graph Completion in Drug Repurposing**

Meghamala Sinha, Roger Tu, Carolina González, Andrew I. Su
Department of Integrative Structural and Computational Biology, The Scripps Research Institute

[![bioRxiv](https://img.shields.io/badge/bioRxiv-2024.07.16.603664-blue)](https://doi.org/10.1101/2024.07.16.603664)

---

## Overview

WeightedKgBlend is a weighted ensemble method for link prediction in biomedical knowledge graphs, combining Knowledge Graph Embedding (KGE) methods with Case-Based Reasoning (CBR) for drug-disease association prediction and drug repurposing.

**Key contributions:**
- Ensemble combining KGE + path-based reasoning for biomedical KG completion
- Novel **Path-Gated Re-ranking**: RotatE for candidate generation, ProbCBR for mechanistic re-ranking
- Evaluated on updated MIND with **6,761 indication edges** across **5-fold cross-validation**
- Every prediction accompanied by a mechanistic biological path explaining *why* a drug may treat a disease
- Updated MIND dataset with 1,391 new FDA-approved indication edges from DrugCentral
- Drug repurposing candidate extraction with resolved drug/disease names and mechanistic paths

---

## Dataset

**MIND (Mechanistic Repositioning Network with Indications) — splits_updated**

Available on Zenodo: [DOI: 10.5281/zenodo.8117748](https://doi.org/10.5281/zenodo.8117748)

| Property | Value |
|----------|-------|
| Total edges | 9,652,116 |
| Total nodes | 250,035 |
| Node types | 9 |
| Relation types | 22 |
| Indication edges (updated) | 6,761 |
| Test / valid edges per slice | 676 |
| Cross-validation folds | 5 (stratified) |

The `splits_updated` dataset adds 1,391 DrugCentral indication edges to the original MIND,
remapped via the DrugCentral → MeSH → DOID pipeline. Available on Kaggle:
[megha90/wkb-splits-updated](https://www.kaggle.com/datasets/megha90/wkb-splits-updated).

---

## Repository Structure

```
WeightedKgBlend/
├── notebooks/
│   ├── 01_main_pipeline.ipynb            # KGE models (TransE, RotatE) — Kaggle
│   ├── 02_cbr_with_paths.ipynb           # CBR with ranked mechanistic path extraction
│   ├── 03_drugcentral_remapping.ipynb    # Update MIND indication edges from DrugCentral
│   ├── 04_ensemble_slice0.ipynb          # Simple ensemble (Optuna-tuned weights)
│   ├── 05_path_gated_reranking.ipynb     # Path-Gated Re-ranking (novel method)
│   ├── 06_repurposing_candidates.ipynb   # Novel drug repurposing candidate extraction
│   ├── weightedkgblend-splits.ipynb      # 5-fold cross-validation split generation
│   └── parallel/
│       ├── weightedkgblend-transe.ipynb  # TransE (all slices, Kaggle)
│       ├── weightedkgblend-rotate.ipynb  # RotatE (all slices, Kaggle)
│       ├── weightedkgblend-probcbr.ipynb # ProbCBR (all slices, Kaggle)
│       ├── weightedkgblend-cbr.ipynb     # CBR (all slices, Kaggle)
│       ├── weightedkgblend-distmult.ipynb
│       └── weightedkgblend-complex.ipynb
│
├── scripts/
│   ├── 01_prepare_splits.py              # Generate 5-fold train/test/valid splits
│   ├── 02_run_kge.py                     # Train KGE models via PyKEEN
│   ├── 03_run_cbr.py                     # Run CBR and ProbCBR
│   ├── run_ensemble_updated.py           # Simple Ensemble (Optuna, splits_updated)
│   ├── run_pathgated_updated.py          # Path-Gated Re-ranking (Optuna, splits_updated)
│   ├── 05_evaluate.py                    # Evaluate across splits + Wilcoxon tests
│   ├── 06_update_mind_mapping.py         # Re-map DrugCentral indications to MIND
│   ├── prob_cbr_proper.py                # ProbCBR implementation (Das et al. 2020)
│   └── prob_cbr_kaggle_cell.py           # ProbCBR Kaggle-ready cell
│
├── results/
│   ├── results_table_updated.md          # Per-slice results table (all models, all slices)
│   └── full_results.md                   # Full results with descriptions and key findings
│
├── config.yaml                           # All hyperparameters (single source of truth)
├── requirements.txt                      # Python dependencies
└── README.md
```

---

## Methods

### Models

| Model | Type | Reference |
|-------|------|-----------|
| TransE | KGE | Bordes et al. 2013 |
| RotatE | KGE | Sun et al. 2019 |
| ProbCBR | Path-based | Das et al. 2020 |
| Simple Ensemble | Weighted sum | This work |
| Path-Gated | Re-ranking | This work |

### Simple Ensemble

WeightedKgBlend combines reciprocal ranks from multiple models using Optuna-optimised weights:

```
score(drug, disease) = Σ λᵢ × (1 / rank_model_i(drug, disease))
subject to: λᵢ ≥ 0
```

Weights optimised on the validation set using Bayesian optimisation (200 trials, TPE sampler).

### Path-Gated Re-ranking (novel contribution)

A two-stage method that uses RotatE for high-recall candidate generation and ProbCBR mechanistic paths for precision re-ranking:

```
score(drug, disease) = α × (1/RotatE_rank) + β × ProbCBR_path_score
```

- RotatE retrieves top-50 candidate diseases per drug
- ProbCBR computes mechanistic path scores for those candidates
- Optuna tunes α and β on the validation set (300 trials)
- Optimal α ≈ 0 across all slices — path scores alone drive re-ranking

Every prediction is accompanied by a mechanistic biological path, e.g.:

```
nicardipine -[inhibits_CinG]-> CACNA1C -[associated_with_GawD]-> hypertensive disorder
```

### Evaluation

- 5-fold cross-validation (stratified, 676 test/valid per fold)
- Metrics: MRR, Hits@1, Hits@3, Hits@5, Hits@10
- Results reported as mean ± std across 5 slices

---

## Results

### Summary (mean ± std across 5 slices)

**Test set:**

| Model | MRR | Hits@1 | Hits@3 | Hits@5 | Hits@10 |
|-------|-----|--------|--------|--------|---------|
| TransE | 0.0158 ± 0.0025 | 0.0036 ± 0.0015 | 0.0114 ± 0.0039 | 0.0191 ± 0.0042 | 0.0311 ± 0.0080 |
| ProbCBR | 0.0138 ± 0.0061 | 0.0050 ± 0.0052 | 0.0136 ± 0.0058 | 0.0198 ± 0.0079 | 0.0296 ± 0.0091 |
| RotatE | 0.0460 ± 0.0069 | 0.0135 ± 0.0036 | 0.0403 ± 0.0054 | 0.0648 ± 0.0129 | 0.1100 ± 0.0225 |
| Simple Ensemble | 0.0489 ± 0.0068 | 0.0135 ± 0.0036 | 0.0400 ± 0.0052 | 0.0645 ± 0.0130 | 0.1097 ± 0.0225 |
| **Path-Gated** | **0.0682 ± 0.0107** | **0.0188 ± 0.0073** | **0.0493 ± 0.0131** | **0.0734 ± 0.0188** | **0.1291 ± 0.0263** |

**Valid set:**

| Model | MRR | Hits@1 | Hits@3 | Hits@5 | Hits@10 |
|-------|-----|--------|--------|--------|---------|
| TransE | 0.0146 ± 0.0030 | 0.0039 ± 0.0021 | 0.0102 ± 0.0034 | 0.0158 ± 0.0054 | 0.0263 ± 0.0056 |
| ProbCBR | 0.0110 ± 0.0046 | 0.0041 ± 0.0032 | 0.0092 ± 0.0043 | 0.0133 ± 0.0057 | 0.0246 ± 0.0087 |
| RotatE | 0.0452 ± 0.0066 | 0.0128 ± 0.0043 | 0.0389 ± 0.0060 | 0.0631 ± 0.0130 | 0.1094 ± 0.0228 |
| Simple Ensemble | 0.0506 ± 0.0066 | 0.0149 ± 0.0041 | 0.0433 ± 0.0092 | 0.0678 ± 0.0106 | 0.1087 ± 0.0165 |
| **Path-Gated** | **0.0704 ± 0.0067** | **0.0209 ± 0.0047** | **0.0502 ± 0.0071** | **0.0747 ± 0.0140** | **0.1353 ± 0.0188** |

Path-Gated achieves **+48% MRR** and **+17% Hits@10** over RotatE alone.
Full per-slice results in [`results/results_table_updated.md`](results/results_table_updated.md).

---

## Drug Repurposing Candidates

Novel drug-disease pairs extracted from Path-Gated test predictions across all 5 slices,
filtered against all known indication pairs (71,518 pairs excluded).

- **455 high-confidence candidates**: rank ≤ 5 with named drugs and mechanistic path
- **30 top candidates**: fully resolved drug/disease names + intermediate node names

Selected examples with mechanistic paths:

| Drug | Disease | Rank | Mechanistic Path |
|------|---------|------|-----------------|
| nicardipine | hypertensive disorder | 1 | inhibits CACNA1C → associated_with → hypertensive disorder |
| vismodegib | basal cell carcinoma | 1 | inhibits SMO → marker_mechanism → BCC |
| benzatropine | Parkinson's disease | 1 | inhibits DAT → associated_with → Parkinson's |
| metformin | lung benign neoplasm | 1 | targets SLC22A2 → associated_with → lung benign neoplasm |
| tofacitinib | kidney disease | 1 | inhibits ABL1 → associated_with → kidney disease |
| everolimus | malignant neoplasm of breast | 2 | inhibits CDH1 → associated_with → breast cancer |
| frovatriptan | essential tremor | 1 | activates HTR1A → associated_with → essential tremor |

Full candidate list in `results/repurposing_candidates_top.tsv` (455 candidates)
and `results/repurposing_candidates_resolved.tsv` (30 candidates, resolved names).

---

## Key Findings

1. **Path-Gated Re-ranking is the best method** — consistently outperforms all baselines across all 5 slices. The near-zero α (RotatE weight) means ProbCBR path scores alone are driving re-ranking, confirming that mechanistic paths add genuine discriminative signal beyond embedding-based ranking.

2. **Simple Ensemble adds little over RotatE alone** — optimal weights give RotatE ~92% of the weight, suggesting KGE dominates and CBR path scores are noisy in the simple weighted sum formulation.

3. **Path coverage is the bottleneck** — only ~41% of RotatE top-10 candidates have a ProbCBR path. Improving path coverage (e.g., with longer paths, relaxed matching) is the main lever for further gains.

4. **44 rank-1 predictions with mechanistic paths** across 5 slices — these are the strongest repurposing candidates, with the model's top-ranked prediction supported by an interpretable biological route.

---

## Follow-up Work

This project motivated a follow-up study on neural path reasoning for drug repurposing:

**NeuralPathReasoning-MIND** — Zero-shot inference with ULTRA (Neural Bellman-Ford Networks)
- Repository: [meghasin/NeuralPathReasoning-MIND](https://github.com/meghasin/NeuralPathReasoning-MIND)
- ULTRA zero-shot achieves **MRR 0.3569 ± 0.0092** — 5.2× improvement over Path-Gated with no fine-tuning

---

## Quick Start

### Option A — Kaggle (recommended, free GPU)

1. Upload any notebook from `notebooks/parallel/` to Kaggle
2. Add the MIND dataset: [megha90/wkb-splits-updated](https://www.kaggle.com/datasets/megha90/wkb-splits-updated)
3. Settings → Accelerator → GPU T4 x2
4. Run all cells

### Option B — Local

```bash
git clone https://github.com/meghasin/WeightedKgBlend
cd WeightedKgBlend
pip install -r requirements.txt

# Download MIND from https://zenodo.org/records/8117748 → place in data/
python scripts/01_prepare_splits.py
python scripts/02_run_kge.py
python scripts/03_run_cbr.py
python scripts/run_ensemble_updated.py --all
python scripts/run_pathgated_updated.py --all
```

---

## Citation

```bibtex
@article{sinha2024weightedkgblend,
  title={Weighted Ensemble Approach for Knowledge Graph completion improves performance},
  author={Sinha, Meghamala and Tu, Roger and Gonz{\'a}lez, Carolina and Su, Andrew I},
  journal={bioRxiv},
  year={2024},
  doi={10.1101/2024.07.16.603664}
}
```

---

## License

MIT License

## Acknowledgements

Supported by NIH grant 1R01AG066750-01.
MIND dataset created at The Scripps Research Institute, Department of Integrative Structural and Computational Biology.
