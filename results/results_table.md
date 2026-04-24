# WeightedKgBlend — Results Tracking

## Methods
- **Simple Ensemble**: weighted sum of TransE + RotatE + ProbCBR reciprocal ranks (Optuna, 200 trials)
  - Optimal weights slice_0: TransE=0.065, RotatE=0.924, ProbCBR=0.010
- **Path-Gated Re-ranking**: RotatE top-50 candidates re-ranked by ProbCBR path scores (Optuna, 300 trials)
  - score = α × (1/RotatE_rank) + β × ProbCBR_path_score
  - Optimal weights slice_0: α=0.0008, β=0.8845
  - Path coverage slice_0: 41.1% of RotatE top-10 candidates have a mechanistic path

---

## slice_0

| Model            | Split | N   | MRR    | Hits@1 | Hits@3 | Hits@5 | Hits@10 |
|------------------|-------|-----|--------|--------|--------|--------|---------|
| TransE           | valid | 530 | 0.0188 | 0.0019 | 0.0189 | 0.0283 | 0.0415  |
| TransE           | test  | 532 | 0.0151 | 0.0019 | 0.0075 | 0.0188 | 0.0320  |
| RotatE           | valid | 530 | 0.0587 | 0.0226 | 0.0509 | 0.0849 | 0.1377  |
| RotatE           | test  | 532 | 0.0551 | 0.0169 | 0.0602 | 0.0808 | 0.1109  |
| ProbCBR          | valid | 537 | 0.0102 | 0.0037 | 0.0074 | 0.0112 | 0.0223  |
| ProbCBR          | test  | 537 | 0.0132 | 0.0037 | 0.0130 | 0.0168 | 0.0242  |
| Simple Ensemble  | valid | 530 | 0.0614 | 0.0226 | 0.0509 | 0.0849 | 0.1377  |
| Simple Ensemble  | test  | 532 | 0.0577 | 0.0169 | 0.0602 | 0.0808 | 0.1109  |
| Path-Gated       | valid | 530 | 0.0812 | 0.0321 | 0.0604 | 0.0830 | 0.1604  |
| Path-Gated       | test  | 532 | 0.0754 | 0.0188 | 0.0658 | 0.1071 | 0.1635  |

---

## slice_1

| Model            | Split | N   | MRR    | Hits@1 | Hits@3 | Hits@5 | Hits@10 |
|------------------|-------|-----|--------|--------|--------|--------|---------|
| TransE           | valid | —   | —      | —      | —      | —      | —       |
| TransE           | test  | —   | —      | —      | —      | —      | —       |
| RotatE           | valid | —   | —      | —      | —      | —      | —       |
| RotatE           | test  | —   | —      | —      | —      | —      | —       |
| ProbCBR          | valid | 537 | 0.0151 | 0.0037 | 0.0168 | 0.0223 | 0.0391  |
| ProbCBR          | test  | 537 | 0.0090 | 0.0000 | 0.0112 | 0.0130 | 0.0205  |
| Simple Ensemble  | valid | —   | —      | —      | —      | —      | —       |
| Simple Ensemble  | test  | —   | —      | —      | —      | —      | —       |
| Path-Gated       | valid | —   | —      | —      | —      | —      | —       |
| Path-Gated       | test  | —   | —      | —      | —      | —      | —       |

---

## slice_2

| Model            | Split | N   | MRR    | Hits@1 | Hits@3 | Hits@5 | Hits@10 |
|------------------|-------|-----|--------|--------|--------|--------|---------|
| TransE           | valid | —   | —      | —      | —      | —      | —       |
| TransE           | test  | —   | —      | —      | —      | —      | —       |
| RotatE           | valid | —   | —      | —      | —      | —      | —       |
| RotatE           | test  | —   | —      | —      | —      | —      | —       |
| ProbCBR          | valid | 537 | 0.0125 | 0.0019 | 0.0168 | 0.0223 | 0.0261  |
| ProbCBR          | test  | 537 | 0.0100 | 0.0037 | 0.0074 | 0.0112 | 0.0223  |
| Simple Ensemble  | valid | —   | —      | —      | —      | —      | —       |
| Simple Ensemble  | test  | —   | —      | —      | —      | —      | —       |
| Path-Gated       | valid | —   | —      | —      | —      | —      | —       |
| Path-Gated       | test  | —   | —      | —      | —      | —      | —       |

---

## slice_2

| Model            | Split | N   | MRR    | Hits@1 | Hits@3 | Hits@5 | Hits@10 |
|------------------|-------|-----|--------|--------|--------|--------|---------|
| TransE           | valid | —   | —      | —      | —      | —      | —       |
| TransE           | test  | —   | —      | —      | —      | —      | —       |
| RotatE           | valid | —   | —      | —      | —      | —      | —       |
| RotatE           | test  | —   | —      | —      | —      | —      | —       |
| ProbCBR          | valid | —   | —      | —      | —      | —      | —       |
| ProbCBR          | test  | —   | —      | —      | —      | —      | —       |
| Simple Ensemble  | valid | —   | —      | —      | —      | —      | —       |
| Simple Ensemble  | test  | —   | —      | —      | —      | —      | —       |
| Path-Gated       | valid | —   | —      | —      | —      | —      | —       |
| Path-Gated       | test  | —   | —      | —      | —      | —      | —       |

---

## slice_3

| Model            | Split | N   | MRR    | Hits@1 | Hits@3 | Hits@5 | Hits@10 |
|------------------|-------|-----|--------|--------|--------|--------|---------|
| TransE           | valid | —   | —      | —      | —      | —      | —       |
| TransE           | test  | —   | —      | —      | —      | —      | —       |
| RotatE           | valid | —   | —      | —      | —      | —      | —       |
| RotatE           | test  | —   | —      | —      | —      | —      | —       |
| ProbCBR          | valid | 537 | 0.0156 | 0.0074 | 0.0112 | 0.0168 | 0.0317  |
| ProbCBR          | test  | 537 | 0.0203 | 0.0093 | 0.0186 | 0.0279 | 0.0428  |
| Simple Ensemble  | valid | —   | —      | —      | —      | —      | —       |
| Simple Ensemble  | test  | —   | —      | —      | —      | —      | —       |
| Path-Gated       | valid | —   | —      | —      | —      | —      | —       |
| Path-Gated       | test  | —   | —      | —      | —      | —      | —       |

---

## slice_4

| Model            | Split | N   | MRR    | Hits@1 | Hits@3 | Hits@5 | Hits@10 |
|------------------|-------|-----|--------|--------|--------|--------|---------|
| TransE           | valid | —   | —      | —      | —      | —      | —       |
| TransE           | test  | —   | —      | —      | —      | —      | —       |
| RotatE           | valid | —   | —      | —      | —      | —      | —       |
| RotatE           | test  | —   | —      | —      | —      | —      | —       |
| ProbCBR          | valid | 537 | 0.0191 | 0.0093 | 0.0242 | 0.0279 | 0.0279  |
| ProbCBR          | test  | 537 | 0.0196 | 0.0093 | 0.0242 | 0.0279 | 0.0372  |
| Simple Ensemble  | valid | —   | —      | —      | —      | —      | —       |
| Simple Ensemble  | test  | —   | —      | —      | —      | —      | —       |
| Path-Gated       | valid | —   | —      | —      | —      | —      | —       |
| Path-Gated       | test  | —   | —      | —      | —      | —      | —       |

---

## Summary (mean ± std across 5 slices, test set)

| Model           | MRR              | Hits@1           | Hits@3           | Hits@5           | Hits@10          |
|-----------------|------------------|------------------|------------------|------------------|------------------|
| TransE          | —                | —                | —                | —                | —                |
| RotatE          | —                | —                | —                | —                | —                |
| ProbCBR         | 0.0144 ± 0.0047  | 0.0052 ± 0.0036  | 0.0149 ± 0.0059  | 0.0194 ± 0.0072  | 0.0294 ± 0.0089  |
| Simple Ensemble | —                | —                | —                | —                | —                |
| Path-Gated      | —                | —                | —                | —                | —                |
