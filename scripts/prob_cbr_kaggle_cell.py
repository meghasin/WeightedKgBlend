"""
PASTE THIS AS A NEW CELL IN YOUR KAGGLE NOTEBOOK
REPLACE the ProbCBR section in Cell 6 with this.
Run this cell AFTER Cell 5 (KGE) finishes.
"""

# ── Proper ProbCBR implementation (Das et al. EMNLP 2020) ─────────────────

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


class ProbCBR:
    def __init__(self, k=100, max_path_len=3, n_clusters=10,
                 min_path_freq=2, seed=42):
        self.k             = k
        self.max_path_len  = max_path_len
        self.n_clusters    = n_clusters
        self.min_path_freq = min_path_freq
        self.seed          = seed
        self.entity_vectors  = {}
        self.entity_clusters = {}
        self.path_stats      = {}
        self.graph           = {}
        self.inv_graph       = {}
        self.relation_to_id  = {}

    def _build_graph(self, triples):
        g, ig = defaultdict(list), defaultdict(list)
        for h,r,t in triples:
            g[h].append((r,t))
            ig[t].append((f"inv_{r}",h))
        self.graph, self.inv_graph = dict(g), dict(ig)

    def _build_entity_vectors(self, triples, all_entities):
        rels     = list(set(r for _,r,_ in triples))
        all_rels = rels + [f"inv_{r}" for r in rels]
        self.relation_to_id = {r:i for i,r in enumerate(all_rels)}
        n = len(all_rels)
        vecs = {}
        for e in all_entities:
            v = np.zeros(n, dtype=np.float32)
            for r,_ in self.graph.get(e,[]):
                if r in self.relation_to_id: v[self.relation_to_id[r]] = 1.0
            for r,_ in self.inv_graph.get(e,[]):
                if r in self.relation_to_id: v[self.relation_to_id[r]] = 1.0
            vecs[e] = v
        self.entity_vectors = vecs

    def _cluster_entities(self, all_entities):
        ents = list(all_entities)
        X    = np.array([self.entity_vectors.get(e, np.zeros(
                         len(self.relation_to_id))) for e in ents])
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms==0] = 1
        km = MiniBatchKMeans(n_clusters=self.n_clusters,
                             random_state=self.seed, n_init=5)
        labels = km.fit_predict(X/norms)
        self.entity_clusters = {e:int(l) for e,l in zip(ents,labels)}

    def _get_paths(self, entity):
        paths   = defaultdict(list)
        queue   = [(entity, ())]
        visited = {(entity,())}
        while queue:
            curr, path = queue.pop(0)
            if len(path) >= self.max_path_len: continue
            for rel,nb in self.graph.get(curr,[]):
                np_ = path+(rel,)
                paths[np_].append(nb)
                s = (nb,np_)
                if s not in visited:
                    visited.add(s); queue.append((nb,np_))
            for rel,nb in self.inv_graph.get(curr,[]):
                np_ = path+(rel,)
                paths[np_].append(nb)
                s = (nb,np_)
                if s not in visited:
                    visited.add(s); queue.append((nb,np_))
        return dict(paths)

    def _compute_path_stats(self, triples, query_rel):
        cluster_triples = defaultdict(list)
        for h,r,t in triples:
            if r == query_rel:
                cluster_triples[self.entity_clusters.get(h,0)].append((h,t))

        raw = defaultdict(lambda: defaultdict(lambda:[0,0]))
        for cluster, ind_t in cluster_triples.items():
            for drug, correct in ind_t:
                for path, ends in self._get_paths(drug).items():
                    raw[cluster][path][0] += 1
                    if correct in ends:
                        raw[cluster][path][1] += 1

        result = {}
        for cluster, p_dict in raw.items():
            total = len(cluster_triples.get(cluster,[]))
            if total == 0: continue
            result[cluster] = {}
            for path,(freq,hits) in p_dict.items():
                if freq >= self.min_path_freq:
                    result[cluster][path] = (freq/total, hits/freq)
        return result

    def fit(self, triples, all_entities, query_rel):
        print("  ProbCBR: building graph...")
        self._build_graph(triples)
        print("  ProbCBR: building entity vectors...")
        self._build_entity_vectors(triples, all_entities)
        print("  ProbCBR: clustering entities...")
        self._cluster_entities(all_entities)
        print("  ProbCBR: computing path statistics...")
        self.path_stats = self._compute_path_stats(triples, query_rel)
        n_paths = sum(len(v) for v in self.path_stats.values())
        print(f"  ProbCBR: {n_paths} paths across {len(self.path_stats)} clusters")
        return self

    def predict_one(self, drug, all_entities):
        cluster       = self.entity_clusters.get(drug, 0)
        cluster_paths = self.path_stats.get(cluster, {})
        if not cluster_paths: return {}

        # Cosine similarity to find neighbours
        qv   = self.entity_vectors.get(drug, np.zeros(len(self.relation_to_id)))
        qn   = np.linalg.norm(qv)
        sims = {}
        for e,v in self.entity_vectors.items():
            if e == drug: continue
            en = np.linalg.norm(v)
            if qn > 0 and en > 0:
                s = float(np.dot(qv,v)/(qn*en))
                if s > 0: sims[e] = s

        scores = defaultdict(float)
        for path,(freq,prec) in cluster_paths.items():
            curr = {drug}
            valid = True
            for rel in path:
                nxt = set()
                for e in curr:
                    if rel.startswith("inv_"):
                        for r,nb in self.inv_graph.get(e,[]):
                            if r == rel[4:]: nxt.add(nb)
                    else:
                        for r,nb in self.graph.get(e,[]):
                            if r == rel: nxt.add(nb)
                if not nxt: valid=False; break
                curr = nxt
            if valid:
                for c in curr:
                    scores[c] += freq * prec
        return dict(scores)

    def predict(self, test_queries, all_entities):
        rows   = []
        n_ents = len(all_entities)
        for drug,rel,expected in tqdm(test_queries, desc="  ProbCBR"):
            scores = self.predict_one(drug, all_entities)
            if not scores:
                rank = n_ents
            else:
                sd = sorted(scores, key=scores.get, reverse=True)
                rank = sd.index(expected)+1 if expected in sd else n_ents
            rows.append({"drug":drug,"expected_disease":expected,
                         "rank":rank,"reciprocal_rank":1.0/rank})
        return pd.DataFrame(rows)


# ── Run proper ProbCBR on all slices ──────────────────────────────────────

from pathlib import Path

SPLITS  = Path('/kaggle/working/data/splits')
RES     = Path('/kaggle/working/results')
INDICATION_REL = 'indication'   # update if needed

for i in range(N_SPLITS):
    sl     = SPLITS / f'slice_{i}'
    outdir = RES / 'cbr' / 'ProbCBR' / f'slice_{i}'
    outdir.mkdir(parents=True, exist_ok=True)

    for split in ['test','valid']:
        done = outdir / f'predictions_{split}.tsv'
        if done.exists():
            print(f'  SKIP ProbCBR/slice_{i}/{split}')
            continue

        print(f'\n>>> Proper ProbCBR / slice_{i} / {split}')

        # Load triples
        tr_df = pd.read_csv(sl/'ind_train.tsv', sep='\t', header=None,
                            names=['h','r','t'])
        ev_df = pd.read_csv(sl/f'ind_{split}.tsv', sep='\t', header=None,
                            names=['h','r','t'])
        all_ents = pd.read_csv(sl/'entities.txt', header=None)[0].tolist()

        tr_triples = list(tr_df.itertuples(index=False, name=None))
        ev_queries = list(ev_df.itertuples(index=False, name=None))

        # Fit and predict
        model = ProbCBR(k=100, max_path_len=3, n_clusters=10,
                        min_path_freq=2, seed=42)
        model.fit(tr_triples, all_ents, INDICATION_REL)
        preds = model.predict(ev_queries, all_ents)

        preds.to_csv(done, sep='\t', index=False)
        print(f'  ProbCBR/slice_{i}/{split} MRR={preds.reciprocal_rank.mean():.4f}')

print('\nProper ProbCBR complete!')
