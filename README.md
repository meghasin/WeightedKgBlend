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
- Novel **Path-Gated Re-ranking**: RotatE for candidate generation, ProbCBR for mechanistic re-ranking (MRR +14%, Hits@10 +47% on slice_0)
- Evaluated on MIND — a large-scale biomedical knowledge graph with FDA-validated indication edges
- Every prediction comes with a mechanistic biological path explaining *why* a drug may treat a disease
- Updated MIND dataset with 2,763 new FDA-approved indication edges (8,133 total)
- Statistically validated across 5-fold cross-validation with Wilcoxon significance testing

---

## Dataset

**MIND (Mechanistic Repositioning Network with Indications)**
Available on Zenodo: [DOI: 10.5281/zenodo.8117748](https://doi.org/10.5281/zenodo.8117748)

- 9,652,116 edges
- 250,035 nodes
- 9 node types, 22 relation types
- 8,133 FDA-approved drug-disease indication edges (updated 2025)

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
│   ├── 04_run_ensemble.py                # WeightedKgBlend ensemble (Optuna optimisation)
│   ├── 05_evaluate.py                    # Evaluate across splits + Wilcoxon tests
│   ├── 06_update_mind_mapping.py         # Re-map DrugCentral indications to MIND
│   ├── prob_cbr_proper.py                # ProbCBR implementation (Das et al. 2020)
│   └── prob_cbr_kaggle_cell.py           # ProbCBR Kaggle-ready cell
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
| DistMult | KGE | Yang et al. 2014 |
| ComplEx | KGE | Trouillon et al. 2016 |
| ProbCBR | Path-based | Das et al. 2020 |
| CBR | Path-based | Das et al. 2020 |

### Simple Ensemble

WeightedKgBlend combines reciprocal ranks from multiple models using Optuna-optimised weights:

```
score(drug, disease) = Σ λᵢ × (1 / rank_model_i(drug, disease))
subject to: λᵢ ≥ 0
```

Weights are optimised on the validation set using Bayesian optimisation (200 trials, TPE sampler).
Optimal weights (slice_0): TransE=0.065, RotatE=0.924, ProbCBR=0.010

### Path-Gated Re-ranking (novel)

A two-stage method that uses RotatE for high-recall candidate generation and ProbCBR mechanistic paths for precision re-ranking:

```
score(drug, disease) = α × (1/RotatE_rank) + β × ProbCBR_path_score
```

- RotatE retrieves top-50 candidate diseases per drug
- ProbCBR computes mechanistic path scores specifically for those candidates (41.1% path coverage vs. 7.5% baseline)
- Optuna tunes α and β on the validation set (300 trials)
- Optimal weights (slice_0): α=0.0008, β=0.8845
- **Result:** MRR 0.0661 → 0.0754 (+14%), Hits@10 0.1109 → 0.1635 (+47%)

Every prediction is accompanied by a mechanistic biological path, e.g.:

```
paclitaxel --[activates_CaG]--> BRCA1 --[treats_GtD]--> oral squamous cell carcinoma
```

### Evaluation

- 5-fold cross-validation (80/10/10 train/test/valid per fold)
- Metrics: MRR, Hits@1, Hits@3, Hits@5, Hits@10
- Statistical significance: Wilcoxon signed-rank test across folds
- Results reported as mean ± std across 5 slices

---

## Results (slice_0)

| Method | Split | MRR | Hits@1 | Hits@3 | Hits@5 | Hits@10 |
|--------|-------|-----|--------|--------|--------|---------|
| TransE | test | 0.0151 | 0.0019 | 0.0075 | 0.0188 | 0.0320 |
| RotatE | test | 0.0551 | 0.0169 | 0.0602 | 0.0808 | 0.1109 |
| ProbCBR | test | 0.0132 | 0.0037 | 0.0130 | 0.0168 | 0.0242 |
| Simple Ensemble | test | 0.0577 | 0.0169 | 0.0602 | 0.0808 | 0.1109 |
| **Path-Gated** | **test** | **0.0754** | **0.0188** | **0.0658** | **0.1071** | **0.1635** |

*Full 5-slice results in progress.*

---

## Repurposing Candidates

`notebooks/06_repurposing_candidates.ipynb` extracts novel drug repurposing candidates from the Path-Gated predictions, tiered by confidence:

- **Tier 1** — RotatE rank ≤ 5 + mechanistic path (highest confidence)
- **Tier 2** — RotatE rank ≤ 10 + mechanistic path
- **Tier 3** — RotatE rank ≤ 20 + mechanistic path

Example candidates (slice_0, Tier 1):

| Drug | Disease | RotatE rank | Mechanistic path |
|------|---------|-------------|-----------------|
| cabozantinib | neuroblastoma | 4 | inhibits MET → marker_mechanism → neuroblastoma |
| topiramate | bipolar disorder | 2 | inhibits KCNB1 → associated_with → bipolar disorder |
| pomalidomide | myelodysplastic syndrome | 5 | inhibits IKZF1 → associated_with → MDS |
| cytarabine | stomach cancer | 5 | inhibits ABCB1 → marker_mechanism → stomach cancer |

---

## Quick Start

### Option A — Kaggle (recommended, free GPU)

1. Upload any notebook from `notebooks/parallel/` to Kaggle
2. Add the MIND dataset (Zenodo) and model datasets via Add Data
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
python scripts/04_run_ensemble.py
python scripts/05_evaluate.py
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
