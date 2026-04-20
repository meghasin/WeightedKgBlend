# WeightedKgBlend

**Weighted Ensemble Approach for Knowledge Graph Completion in Drug Repurposing**

Meghamala Sinha, Roger Tu, Carolina González, Andrew I. Su  
Department of Integrative Structural and Computational Biology, The Scripps Research Institute

[![bioRxiv](https://img.shields.io/badge/bioRxiv-2024.07.16.603664-blue)](https://doi.org/10.1101/2024.07.16.603664)

---

## Overview

WeightedKgBlend is a weighted ensemble method for link prediction in biomedical knowledge graphs, combining Knowledge Graph Embedding (KGE) methods with Case-Based Reasoning (CBR) for drug-disease association prediction and drug repurposing.

**Key contributions:**
- First ensemble combining KGE + path-based reasoning for biomedical KG completion
- Evaluated on MIND — a large-scale biomedical knowledge graph with FDA-validated indication edges
- Produces interpretable mechanistic paths explaining *why* a drug may treat a disease
- Updated MIND dataset with 2,763 new FDA-approved indication edges (8,133 total)
- Statistically validated across multiple random splits with Wilcoxon significance testing

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
│   ├── 01_main_pipeline.ipynb       # Full KGE + ensemble pipeline (Kaggle)
│   ├── 02_cbr_with_paths.ipynb      # CBR with ranked mechanistic path extraction
│   └── 03_drugcentral_remapping.ipynb  # Update MIND indication edges from DrugCentral
│
├── scripts/
│   ├── 01_prepare_splits.py         # Generate N random train/test/valid splits
│   ├── 02_run_kge.py                # Train KGE models via PyKEEN
│   ├── 03_run_cbr.py                # Run CBR and ProbCBR
│   ├── 04_run_ensemble.py           # WeightedKgBlend ensemble (Optuna optimisation)
│   ├── 05_evaluate.py               # Evaluate across splits + Wilcoxon tests
│   ├── 06_update_mind_mapping.py    # Re-map DrugCentral indications to MRN
│   ├── prob_cbr_proper.py           # Faithful ProbCBR implementation (Das et al. 2020)
│   └── prob_cbr_kaggle_cell.py      # ProbCBR Kaggle cell
│
├── config.yaml                      # All hyperparameters (single source of truth)
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Quick Start

### Option A — Kaggle (recommended, free GPU)

1. Go to [kaggle.com](https://kaggle.com) → Code → New Notebook → Import Notebook
2. Upload `notebooks/01_main_pipeline.ipynb`
3. Add MIND dataset (from Zenodo) via Add Data button
4. Settings → Accelerator → GPU T4 x2
5. Run all cells

### Option B — Local machine

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/WeightedKgBlend
cd WeightedKgBlend

# Install dependencies
pip install -r requirements.txt

# Download MIND dataset
# Go to https://zenodo.org/records/8117748 and place files in data/

# Run pipeline
python scripts/01_prepare_splits.py
python scripts/02_run_kge.py
python scripts/03_run_cbr.py
python scripts/04_run_ensemble.py
python scripts/05_evaluate.py
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
| Rephetio | Path-based | Himmelstein et al. 2017 |
| CBR | Path-based | Das et al. 2020 |
| ProbCBR | Path-based | Das et al. 2020 |

### Ensemble

WeightedKgBlend combines predictions from all 7 models using Optuna-optimised weights:

```
score(drug, disease) = Σ λᵢ × RR_model_i(drug, disease)
subject to: Σ λᵢ = 1, λᵢ ≥ 0
```

Weights are optimised on the validation set using Bayesian optimisation (200 trials).

### Evaluation

- 3 random 80/10/10 train/test/valid splits
- Metrics: MRR, Hits@1, Hits@3, Hits@10
- Statistical significance: Wilcoxon signed-rank test across splits
- Results reported as mean ± std

---

## Results

| Method | MRR | Hits@1 | Hits@3 | Hits@10 |
|--------|-----|--------|--------|---------|
| TransE | 0.154 | 0.000 | 0.242 | 0.442 |
| RotatE | 0.173 | 0.105 | 0.147 | 0.253 |
| Rephetio | 0.202 | 0.143 | 0.266 | 0.498 |
| ProbCBR | 0.191 | 0.116 | 0.326 | 0.453 |
| **WeightedKgBlend** | **0.239** | **0.158** | **0.244** | **0.370** |

*Original single-split results. Multi-split results with mean ± std in progress.*

---

## Mechanistic Path Example

```
Drug:    Carvedilol
Disease: Acute coronary syndrome
Rank:    1

Path 1 (score=0.84, 2 hops):
  Carvedilol --[inhibits]--> BCL2 --[marker_or_mechanism]--> ACS

Path 2 (score=0.61, 2 hops):
  Carvedilol --[activates]--> ADRB1 --[marker_or_mechanism]--> ACS
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

This project is licensed under the MIT License.

## Acknowledgements

Supported by NIH grant 1R01AG066750-01.  
MIND dataset created at The Scripps Research Institute, Department of Integrative Structural and Computational Biology.
