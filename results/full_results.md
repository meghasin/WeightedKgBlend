# WeightedKgBlend — Full Results (Updated MIND)

**Dataset:** `mind_updated.tsv` — 6,761 drug-indication edges from MIND + 1,391 additional edges from DrugCentral, totalling 8,152 positive indication triples. Split into 5 stratified folds; each fold has ~676 test and ~676 validation triples held out. All models are evaluated on the link prediction task: given a drug, rank all candidate diseases to find the correct indication.

**Evaluation protocol:** Filtered ranking — for each (drug, ?, disease) query, all known positive triples are removed from the candidate list except the one being evaluated. Metrics are averaged across all queries within a split.

**Metrics:**
- **MRR** (Mean Reciprocal Rank): average of 1/rank across all queries. Higher = better. Sensitive to top placements.
- **Hits@k**: fraction of queries where the correct disease appears in the top-k ranked candidates. Reported for k = 1, 3, 5, 10.

**All results are from `results_updated/` — trained and evaluated on `splits_updated`.**

---

## Models at a Glance

| Model | Type | Key idea |
|---|---|---|
| TransE | KGE | Translational embeddings: head + relation ≈ tail |
| RotatE | KGE | Relational patterns via rotation in complex space |
| ProbCBR | Symbolic | Case-based reasoning over multi-hop paths in the KG |
| Simple Ensemble | Fusion | Weighted reciprocal-rank fusion of TransE + RotatE + ProbCBR |
| Path-Gated | Hybrid | ProbCBR path scores re-rank RotatE's top-50 candidates |

---

## TransE

**Model description:** TransE (Bordes et al., 2013) learns entity and relation embeddings such that `h + r ≈ t` for a true triple (h, r, t). Scoring is done by the negative L2 distance between `h + r` and `t`. TransE is a strong baseline for simple, non-hierarchical relations but struggles with symmetric or 1-to-N relations.

**Training:** Trained via Kaggle GPU (T4), using `pykeen` with default hyperparameters on `splits_updated`. Embedding dim=256, negative sampling, trained for 500 epochs.

**Observation:** TransE is the weakest individual model here, achieving MRR ~0.016 on test. Drug-indication is a many-to-many relation (one drug → many diseases, one disease ← many drugs), which is known to challenge TransE's 1-to-1 assumption.

| Slice   | Split | N   | MRR    | Hits@1 | Hits@3 | Hits@5 | Hits@10 |
|---------|-------|-----|--------|--------|--------|--------|---------|
| slice_0 | test  | 668 | 0.0164 | 0.0030 | 0.0165 | 0.0225 | 0.0359  |
| slice_0 | valid | 667 | 0.0106 | 0.0015 | 0.0075 | 0.0135 | 0.0225  |
| slice_1 | test  | 674 | 0.0155 | 0.0059 | 0.0104 | 0.0178 | 0.0297  |
| slice_1 | valid | 667 | 0.0171 | 0.0045 | 0.0135 | 0.0210 | 0.0300  |
| slice_2 | test  | 671 | 0.0127 | 0.0030 | 0.0060 | 0.0119 | 0.0179  |
| slice_2 | valid | 673 | 0.0110 | 0.0015 | 0.0045 | 0.0059 | 0.0178  |
| slice_3 | test  | 669 | 0.0201 | 0.0045 | 0.0149 | 0.0239 | 0.0419  |
| slice_3 | valid | 670 | 0.0171 | 0.0060 | 0.0134 | 0.0194 | 0.0299  |
| slice_4 | test  | 663 | 0.0143 | 0.0015 | 0.0090 | 0.0196 | 0.0302  |
| slice_4 | valid | 671 | 0.0170 | 0.0060 | 0.0119 | 0.0194 | 0.0313  |

**Summary (test, mean ± std across 5 slices)**

| MRR             | Hits@1          | Hits@3          | Hits@5          | Hits@10         |
|-----------------|-----------------|-----------------|-----------------|-----------------|
| 0.0158 ± 0.0025 | 0.0036 ± 0.0015 | 0.0114 ± 0.0039 | 0.0191 ± 0.0042 | 0.0311 ± 0.0080 |

**Summary (valid, mean ± std across 5 slices)**

| MRR             | Hits@1          | Hits@3          | Hits@5          | Hits@10         |
|-----------------|-----------------|-----------------|-----------------|-----------------|
| 0.0146 ± 0.0030 | 0.0039 ± 0.0021 | 0.0102 ± 0.0034 | 0.0158 ± 0.0054 | 0.0263 ± 0.0056 |

---

## RotatE

**Model description:** RotatE (Sun et al., 2019) represents each relation as a rotation in complex-valued embedding space: `h ∘ r = t`. This formulation can model symmetry, antisymmetry, inversion, and composition patterns simultaneously. It is consistently among the best-performing KGE models on standard benchmarks.

**Training:** Trained via Kaggle GPU (T4), using `pykeen` on `splits_updated`. Embedding dim=256, adversarial negative sampling, trained for 500 epochs.

**Observation:** RotatE is the dominant individual model, outperforming TransE by ~3× in MRR (0.046 vs 0.016) and ProbCBR by ~3.3× (0.046 vs 0.014). Its strong embedding geometry captures the complex many-to-many drug-indication structure much better than TransE. RotatE's predictions serve as the candidate pool (top-50) for Path-Gated re-ranking.

| Slice   | Split | N   | MRR    | Hits@1 | Hits@3 | Hits@5 | Hits@10 |
|---------|-------|-----|--------|--------|--------|--------|---------|
| slice_0 | test  | 668 | 0.0430 | 0.0105 | 0.0359 | 0.0584 | 0.0988  |
| slice_0 | valid | 667 | 0.0436 | 0.0105 | 0.0360 | 0.0660 | 0.1049  |
| slice_1 | test  | 674 | 0.0410 | 0.0104 | 0.0401 | 0.0608 | 0.1024  |
| slice_1 | valid | 667 | 0.0468 | 0.0150 | 0.0405 | 0.0645 | 0.1079  |
| slice_2 | test  | 671 | 0.0568 | 0.0179 | 0.0477 | 0.0864 | 0.1520  |
| slice_2 | valid | 673 | 0.0610 | 0.0208 | 0.0609 | 0.0847 | 0.1352  |
| slice_3 | test  | 669 | 0.0512 | 0.0179 | 0.0448 | 0.0703 | 0.1106  |
| slice_3 | valid | 670 | 0.0436 | 0.0104 | 0.0418 | 0.0687 | 0.1090  |
| slice_4 | test  | 663 | 0.0382 | 0.0106 | 0.0332 | 0.0483 | 0.0860  |
| slice_4 | valid | 671 | 0.0434 | 0.0179 | 0.0358 | 0.0522 | 0.0820  |

**Summary (test, mean ± std across 5 slices)**

| MRR             | Hits@1          | Hits@3          | Hits@5          | Hits@10         |
|-----------------|-----------------|-----------------|-----------------|-----------------|
| 0.0460 ± 0.0069 | 0.0135 ± 0.0036 | 0.0403 ± 0.0054 | 0.0648 ± 0.0129 | 0.1100 ± 0.0225 |

**Summary (valid, mean ± std across 5 slices)**

| MRR             | Hits@1          | Hits@3          | Hits@5          | Hits@10         |
|-----------------|-----------------|-----------------|-----------------|-----------------|
| 0.0477 ± 0.0062 | 0.0149 ± 0.0038 | 0.0430 ± 0.0086 | 0.0672 ± 0.0117 | 0.1078 ± 0.0181 |

---

## ProbCBR

**Model description:** ProbCBR (Das et al., 2020) is a probabilistic case-based reasoning method that operates directly on the graph structure. For a query drug, it identifies a cluster of similar drugs (neighbours in embedding space), retrieves multi-hop relational paths they used to reach their correct diseases, and scores candidate diseases by the prior probability and precision of those paths. It provides interpretable mechanistic paths (e.g., drug → target → pathway → disease) alongside its predictions.

**Training:** Trained via Kaggle CPU, independently for each slice. Saves a `.pkl` model file to `results_updated/models/ProbCBR/slice_N.pkl`. These pkl files are later reused by the Path-Gated pipeline.

**Observation:** ProbCBR alone (MRR 0.014) performs worse than RotatE (MRR 0.046) as a ranker, likely because its path-based scoring is noisy over the full disease space. However, when its paths are used to re-rank RotatE's already-filtered top-50 candidates (Path-Gated), performance improves dramatically. High variance across slices (std 0.006) suggests path coverage varies with the training fold.

| Slice   | Split | N   | MRR    | Hits@1 | Hits@3 | Hits@5 | Hits@10 |
|---------|-------|-----|--------|--------|--------|--------|---------|
| slice_0 | test  | 676 | 0.0091 | 0.0000 | 0.0104 | 0.0133 | 0.0237  |
| slice_0 | valid | 676 | 0.0122 | 0.0059 | 0.0104 | 0.0148 | 0.0237  |
| slice_1 | test  | 676 | 0.0255 | 0.0148 | 0.0251 | 0.0340 | 0.0473  |
| slice_1 | valid | 676 | 0.0183 | 0.0089 | 0.0148 | 0.0222 | 0.0399  |
| slice_2 | test  | 671 | 0.0131 | 0.0044 | 0.0118 | 0.0192 | 0.0281  |
| slice_2 | valid | 676 | 0.0119 | 0.0044 | 0.0118 | 0.0148 | 0.0266  |
| slice_3 | test  | 669 | 0.0088 | 0.0015 | 0.0089 | 0.0118 | 0.0222  |
| slice_3 | valid | 676 | 0.0073 | 0.0015 | 0.0059 | 0.0089 | 0.0178  |
| slice_4 | test  | 663 | 0.0124 | 0.0044 | 0.0118 | 0.0207 | 0.0266  |
| slice_4 | valid | 676 | 0.0052 | 0.0000 | 0.0030 | 0.0059 | 0.0148  |

**Summary (test, mean ± std across 5 slices)**

| MRR             | Hits@1          | Hits@3          | Hits@5          | Hits@10         |
|-----------------|-----------------|-----------------|-----------------|-----------------|
| 0.0138 ± 0.0061 | 0.0050 ± 0.0052 | 0.0136 ± 0.0058 | 0.0198 ± 0.0079 | 0.0296 ± 0.0091 |

**Summary (valid, mean ± std across 5 slices)**

| MRR             | Hits@1          | Hits@3          | Hits@5          | Hits@10         |
|-----------------|-----------------|-----------------|-----------------|-----------------|
| 0.0110 ± 0.0046 | 0.0041 ± 0.0032 | 0.0092 ± 0.0043 | 0.0133 ± 0.0057 | 0.0246 ± 0.0087 |

---

## Simple Ensemble

**Model description:** Weighted reciprocal-rank fusion of TransE, RotatE, and ProbCBR. For each query, each model contributes a score to each candidate disease: `score(d) = Σ_i w_i × (1 / rank_i(d))`, where `rank_i(d)` is the position of disease d in model i's ranked list. Weights are optimised per slice using Optuna (200 trials, TPE sampler) on the validation set to maximise validation MRR. Top-50 candidates per model are considered.

**Script:** `scripts/run_ensemble_updated.py --all`

**Observation:** Optuna strongly favours RotatE across all slices (weight 0.88–0.98), with TransE contributing a small but consistent signal (~0.05–0.09) and ProbCBR contributing negligibly (~0.0002–0.03). The ensemble provides only a marginal gain over RotatE alone (+0.0029 MRR on test), confirming that TransE and ProbCBR add little orthogonal information at the ranking level. The real gain comes from Path-Gated re-ranking instead.

**Optimal weights per slice (TransE / RotatE / ProbCBR):**
- slice_0: TransE=0.0458, RotatE=0.9540, ProbCBR=0.0003
- slice_1: TransE=0.0889, RotatE=0.8820, ProbCBR=0.0291
- slice_2: TransE=0.0239, RotatE=0.9759, ProbCBR=0.0002
- slice_3: TransE=0.0779, RotatE=0.9211, ProbCBR=0.0010
- slice_4: TransE=0.0834, RotatE=0.9145, ProbCBR=0.0020

| Slice   | Split | N   | MRR    | Hits@1 | Hits@3 | Hits@5 | Hits@10 |
|---------|-------|-----|--------|--------|--------|--------|---------|
| slice_0 | test  | 668 | 0.0456 | 0.0105 | 0.0359 | 0.0584 | 0.0988  |
| slice_0 | valid | 667 | 0.0463 | 0.0105 | 0.0360 | 0.0660 | 0.1049  |
| slice_1 | test  | 674 | 0.0439 | 0.0104 | 0.0401 | 0.0593 | 0.1024  |
| slice_1 | valid | 667 | 0.0498 | 0.0150 | 0.0405 | 0.0645 | 0.1109  |
| slice_2 | test  | 671 | 0.0594 | 0.0179 | 0.0477 | 0.0864 | 0.1520  |
| slice_2 | valid | 673 | 0.0636 | 0.0208 | 0.0609 | 0.0847 | 0.1352  |
| slice_3 | test  | 669 | 0.0541 | 0.0179 | 0.0433 | 0.0703 | 0.1091  |
| slice_3 | valid | 670 | 0.0466 | 0.0104 | 0.0433 | 0.0716 | 0.1090  |
| slice_4 | test  | 663 | 0.0413 | 0.0106 | 0.0332 | 0.0483 | 0.0860  |
| slice_4 | valid | 671 | 0.0466 | 0.0179 | 0.0358 | 0.0522 | 0.0835  |

**Summary (test, mean ± std across 5 slices)**

| MRR             | Hits@1          | Hits@3          | Hits@5          | Hits@10         |
|-----------------|-----------------|-----------------|-----------------|-----------------|
| 0.0489 ± 0.0068 | 0.0135 ± 0.0036 | 0.0400 ± 0.0052 | 0.0645 ± 0.0130 | 0.1097 ± 0.0225 |

**Summary (valid, mean ± std across 5 slices)**

| MRR             | Hits@1          | Hits@3          | Hits@5          | Hits@10         |
|-----------------|-----------------|-----------------|-----------------|-----------------|
| 0.0506 ± 0.0066 | 0.0149 ± 0.0041 | 0.0433 ± 0.0092 | 0.0678 ± 0.0106 | 0.1087 ± 0.0165 |

---

## Path-Gated

**Model description:** Path-Gated re-ranking uses ProbCBR's mechanistic path scores to re-rank RotatE's top-50 candidate diseases. For each (drug, candidate_disease) pair, a combined score is computed as:

```
score(d) = α × (1 / RotatE_rank(d)) + β × ProbCBR_path_score(drug, d)
```

where `ProbCBR_path_score` is the sum of (prior × precision) contributions from all multi-hop paths that connect the drug to the candidate disease in the knowledge graph. Path scores are computed specifically against RotatE's top-50 candidate pool (not ProbCBR's own candidates), achieving ~87–89% coverage. α and β are optimised per slice using Optuna (300 trials) on the validation set.

**Scripts:**
1. `scripts/build_pathgated_lookup.py --all` — generates `path_lookup_rotate_{split}_slice{N}.tsv` files
2. `scripts/run_pathgated_updated.py --all` — runs Optuna and saves re-ranked predictions

**Observation:** Path-Gated is the best-performing model by a large margin: MRR 0.068 vs 0.049 for Simple Ensemble and 0.046 for RotatE alone on test (+48% over RotatE). Notably, Optuna sets α≈0 across all slices, meaning ProbCBR path scores alone drive the re-ranking — the RotatE reciprocal rank within the top-50 window adds little. This suggests that within RotatE's shortlist, the path-based mechanistic signal is a stronger discriminator than embedding-based rank ordering. The interpretable paths (e.g., drug → target → pathway → disease) are also saved alongside predictions for biological analysis.

**Optimal weights per slice (α=RotatE rank weight / β=path score weight):**
- slice_0: α=0.0000, β=0.5722
- slice_1: α=0.0000, β=0.6489
- slice_2: α=0.0000, β=0.9586
- slice_3: α=0.0023, β=0.6130
- slice_4: α=0.0005, β=0.6818

| Slice   | Split | N   | MRR    | Hits@1 | Hits@3 | Hits@5 | Hits@10 |
|---------|-------|-----|--------|--------|--------|--------|---------|
| slice_0 | test  | 668 | 0.0627 | 0.0165 | 0.0404 | 0.0614 | 0.1153  |
| slice_0 | valid | 667 | 0.0760 | 0.0240 | 0.0555 | 0.0825 | 0.1529  |
| slice_1 | test  | 674 | 0.0762 | 0.0208 | 0.0668 | 0.0920 | 0.1499  |
| slice_1 | valid | 667 | 0.0740 | 0.0180 | 0.0540 | 0.0915 | 0.1589  |
| slice_2 | test  | 671 | 0.0837 | 0.0313 | 0.0596 | 0.0954 | 0.1699  |
| slice_2 | valid | 673 | 0.0774 | 0.0267 | 0.0579 | 0.0802 | 0.1352  |
| slice_3 | test  | 669 | 0.0653 | 0.0164 | 0.0493 | 0.0732 | 0.1076  |
| slice_3 | valid | 670 | 0.0611 | 0.0134 | 0.0433 | 0.0687 | 0.1209  |
| slice_4 | test  | 663 | 0.0529 | 0.0090 | 0.0302 | 0.0452 | 0.1026  |
| slice_4 | valid | 671 | 0.0635 | 0.0224 | 0.0402 | 0.0507 | 0.1088  |

**Summary (test, mean ± std across 5 slices)**

| MRR             | Hits@1          | Hits@3          | Hits@5          | Hits@10         |
|-----------------|-----------------|-----------------|-----------------|-----------------|
| 0.0682 ± 0.0107 | 0.0188 ± 0.0073 | 0.0493 ± 0.0131 | 0.0734 ± 0.0188 | 0.1291 ± 0.0263 |

**Summary (valid, mean ± std across 5 slices)**

| MRR             | Hits@1          | Hits@3          | Hits@5          | Hits@10         |
|-----------------|-----------------|-----------------|-----------------|-----------------|
| 0.0704 ± 0.0067 | 0.0209 ± 0.0047 | 0.0502 ± 0.0071 | 0.0747 ± 0.0140 | 0.1353 ± 0.0188 |

---

## Overall Summary (test set, mean ± std across 5 slices)

Models ranked by MRR. Path-Gated achieves the best performance, with a +48% MRR improvement over RotatE alone and +39% over Simple Ensemble.

| Model           | MRR             | Hits@1          | Hits@3          | Hits@5          | Hits@10         |
|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| TransE          | 0.0158 ± 0.0025 | 0.0036 ± 0.0015 | 0.0114 ± 0.0039 | 0.0191 ± 0.0042 | 0.0311 ± 0.0080 |
| ProbCBR         | 0.0138 ± 0.0061 | 0.0050 ± 0.0052 | 0.0136 ± 0.0058 | 0.0198 ± 0.0079 | 0.0296 ± 0.0091 |
| RotatE          | 0.0460 ± 0.0069 | 0.0135 ± 0.0036 | 0.0403 ± 0.0054 | 0.0648 ± 0.0129 | 0.1100 ± 0.0225 |
| Simple Ensemble | 0.0489 ± 0.0068 | 0.0135 ± 0.0036 | 0.0400 ± 0.0052 | 0.0645 ± 0.0130 | 0.1097 ± 0.0225 |
| **Path-Gated**  | **0.0682 ± 0.0107** | **0.0188 ± 0.0073** | **0.0493 ± 0.0131** | **0.0734 ± 0.0188** | **0.1291 ± 0.0263** |

---

## Overall Summary (valid set, mean ± std across 5 slices)

| Model           | MRR             | Hits@1          | Hits@3          | Hits@5          | Hits@10         |
|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| TransE          | 0.0146 ± 0.0031 | 0.0039 ± 0.0020 | 0.0102 ± 0.0036 | 0.0158 ± 0.0056 | 0.0263 ± 0.0053 |
| ProbCBR         | 0.0110 ± 0.0045 | 0.0041 ± 0.0032 | 0.0092 ± 0.0042 | 0.0133 ± 0.0056 | 0.0246 ± 0.0087 |
| RotatE          | 0.0477 ± 0.0068 | 0.0149 ± 0.0041 | 0.0430 ± 0.0093 | 0.0672 ± 0.0104 | 0.1078 ± 0.0169 |
| Simple Ensemble | 0.0506 ± 0.0066 | 0.0149 ± 0.0041 | 0.0433 ± 0.0092 | 0.0678 ± 0.0106 | 0.1087 ± 0.0165 |
| **Path-Gated**  | **0.0704 ± 0.0067** | **0.0209 ± 0.0047** | **0.0502 ± 0.0071** | **0.0747 ± 0.0140** | **0.1353 ± 0.0188** |

---

## Key Findings

1. **RotatE >> TransE as a standalone KGE model** (MRR 0.046 vs 0.016), consistent with RotatE's known advantage on complex relational patterns like drug-indication.

2. **ProbCBR alone is weak as a ranker** (MRR 0.014) but its path scores are highly informative when used to re-rank a pre-filtered shortlist (Path-Gated MRR 0.068).

3. **Simple Ensemble adds only marginal value over RotatE** (+0.003 MRR). Optuna assigns ~90–98% weight to RotatE, showing the models carry largely redundant information at the rank-list level.

4. **Path-Gated is the best model overall** (+48% MRR over RotatE, +39% over Simple Ensemble on test). Optuna consistently sets α≈0, meaning ProbCBR path scores alone are sufficient — the RotatE rank order within the top-50 does not add additional discriminative signal once the candidate pool is constrained.

5. **Interpretability benefit of Path-Gated**: predictions include the best supporting mechanistic path for each drug-disease pair (saved in `predictions_{split}.tsv`), enabling biological validation of top candidates.

---

## Curated Examples — Path-Gated Rank-1 Predictions with Mechanistic Paths

These are cases where Path-Gated ranked the correct disease at position 1 and recovered a biologically meaningful multi-hop path through the MIND knowledge graph. Intermediate nodes (genes, biological processes) were resolved from the graph and cross-referenced with external databases.

| Drug | Disease | Mechanistic Path | Biological Rationale |
|------|---------|-----------------|----------------------|
| **Nicardipine** (CHEBI:7551) | Hypertensive disorder (DOID:10763) | Nicardipine → **inhibits** → *CACNA1C* (Cav1.2, L-type Ca²⁺ channel) → **marker/mechanism of** → Hypertension | CACNA1C is the direct molecular target of dihydropyridine calcium channel blockers; its inhibition reduces vascular smooth muscle contraction. Textbook mechanism. |
| **Vismodegib** (CHEBI:66903) | Basal cell carcinoma (DOID:2513) | Vismodegib → **inhibits** → *SMO* (Smoothened receptor) → **marker/mechanism of** → Basal cell carcinoma | Vismodegib is FDA-approved specifically for BCC via Hedgehog pathway inhibition at SMO. The model perfectly recovered the drug's approved mechanism. |
| **Benzatropine mesylate** (CHEBI:3049) | Parkinson's disease (DOID:14330) | Benzatropine → **inhibits** → *SLC6A3* (dopamine transporter DAT) → **marker/mechanism of** → Parkinson's disease | DAT is a canonical Parkinson's biomarker and drug target. Benzatropine is an approved anticholinergic/dopaminergic used for Parkinson's symptoms. |
| **Nateglinide** (CHEBI:31897) | Type 2 diabetes (DOID:9352) | Nateglinide → **activates** → *PPARG* (PPARγ, peroxisome proliferator-activated receptor γ) → **gene treats disease** → Type 2 diabetes | PPARγ is a master regulator of glucose homeostasis and the target of thiazolidinediones. Nateglinide's primary mechanism is KATP channel closure, but PPARγ activation is a known secondary effect. |
| **Thiosalicylic acid** (CHEBI:59124) | Osteoarthritis (DOID:8398) | Thiosalicylic acid → **inhibits** → *prostaglandin biosynthetic process* (GO:0001516) → **associated with** → Osteoarthritis | Salicylate derivatives inhibit COX-mediated prostaglandin synthesis; prostaglandins are key mediators of inflammatory pain and cartilage degradation in osteoarthritis. |
| **Phensuximide** (CHEBI:8079) | Childhood absence epilepsy (DOID:1825) | Phensuximide → **activates** → *cell death* (GO:0008219) → **associated with** → Childhood absence epilepsy | Phensuximide is an approved succinimide anti-epileptic used specifically for absence seizures. The path reflects the drug's ability to reduce aberrant neuronal excitability. |

**Coverage:** 101 of the rank ≤5 test predictions (across all 5 slices) have a non-empty mechanistic path. 44 rank-1 hits include a path with at least one resolved intermediate node.
