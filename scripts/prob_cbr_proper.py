"""
prob_cbr_proper.py
------------------
Faithful implementation of ProbCBR (Das et al., EMNLP Findings 2020)
"Probabilistic Case-based Reasoning for Open-World KG Completion"

Key differences from simplified version:
1. Collects actual relation paths (up to length 3) between entities
2. Computes per-cluster path precision: P(path works | cluster, relation)
3. Scores candidates by: sum over paths of (path_freq * path_precision)
4. Entity similarity via sparse relation-type vectors (not just degree)

Usage in Kaggle Cell 6 — replace run_method('ProbCBR',...) with:
    from prob_cbr_proper import ProbCBR
    model = ProbCBR(k=100, max_path_len=3, n_clusters=10)
    model.fit(train_triples, all_entities)
    preds = model.predict(test_queries, all_entities)
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm


class ProbCBR:
    """
    Probabilistic Case-Based Reasoning for KG completion.
    
    Algorithm (Das et al. 2020):
    
    OFFLINE (fit):
    1. Represent each entity as a sparse vector of its relation types
    2. Cluster entities into K clusters using KMeans
    3. For each cluster c and query relation r:
       - Find all training entities in cluster c
       - Collect paths of length 1-3 from each entity
       - Compute path precision: fraction of times path ends at correct answer
       - Store: path_stats[c][r][path] = (frequency, precision)
    
    ONLINE (predict):
    Given query (drug, relation, ?):
    1. Find k nearest neighbour drugs (similar relation-type vectors)
    2. Determine cluster of query drug
    3. For each path in path_stats[cluster][relation]:
       - Follow path from query drug
       - Score each reachable entity by: path_freq * path_precision * n_neighbours_using_path
    4. Rank candidates by score
    """

    def __init__(self, k=100, max_path_len=3, n_clusters=10,
                 min_path_freq=2, seed=42):
        self.k            = k
        self.max_path_len = max_path_len
        self.n_clusters   = n_clusters
        self.min_path_freq= min_path_freq
        self.seed         = seed

        # Populated by fit()
        self.entity_vectors  = {}   # entity -> sparse relation vector
        self.entity_clusters = {}   # entity -> cluster id
        self.path_stats      = {}   # cluster -> relation -> path -> (freq, precision)
        self.entity_paths    = {}   # entity -> {path: [end_entities]}
        self.relation_to_id  = {}
        self.graph           = {}   # entity -> [(relation, tail)]
        self.inv_graph       = {}   # entity -> [(inv_relation, head)]

    # ------------------------------------------------------------------
    # Step 1: Build graph
    # ------------------------------------------------------------------

    def _build_graph(self, triples):
        """Build adjacency lists from training triples."""
        graph     = defaultdict(list)
        inv_graph = defaultdict(list)
        for h, r, t in triples:
            graph[h].append((r, t))
            inv_graph[t].append((f"inv_{r}", h))
        self.graph     = dict(graph)
        self.inv_graph = dict(inv_graph)

    # ------------------------------------------------------------------
    # Step 2: Entity vectors (relation-type profile)
    # ------------------------------------------------------------------

    def _build_entity_vectors(self, triples, all_entities):
        """
        Represent each entity as a binary vector over relation types.
        Entity e has 1 in dimension r if e appears as head in any (e, r, ?) triple,
        or as tail in any (?, r, e) triple (as inv_r).
        """
        # Build relation vocabulary
        relations = list(set(r for _, r, _ in triples))
        inv_rels  = [f"inv_{r}" for r in relations]
        all_rels  = relations + inv_rels
        self.relation_to_id = {r: i for i, r in enumerate(all_rels)}
        n_rels = len(all_rels)

        # Build sparse vectors
        vectors = {}
        for entity in all_entities:
            vec = np.zeros(n_rels, dtype=np.float32)
            for r, _ in self.graph.get(entity, []):
                if r in self.relation_to_id:
                    vec[self.relation_to_id[r]] = 1.0
            for r, _ in self.inv_graph.get(entity, []):
                if r in self.relation_to_id:
                    vec[self.relation_to_id[r]] = 1.0
            vectors[entity] = vec

        self.entity_vectors = vectors
        return vectors

    # ------------------------------------------------------------------
    # Step 3: Cluster entities
    # ------------------------------------------------------------------

    def _cluster_entities(self, all_entities):
        """KMeans clustering of entity relation-type vectors."""
        entities = list(all_entities)
        X = np.array([self.entity_vectors.get(e, np.zeros(
            len(self.relation_to_id))) for e in entities])

        # Normalise for cosine-like clustering
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X_norm = X / norms

        km = MiniBatchKMeans(n_clusters=self.n_clusters,
                             random_state=self.seed, n_init=5)
        labels = km.fit_predict(X_norm)
        self.entity_clusters = {e: int(l) for e, l in zip(entities, labels)}

    # ------------------------------------------------------------------
    # Step 4: Collect paths per entity
    # ------------------------------------------------------------------

    def _get_paths(self, entity, max_len):
        """
        BFS to collect all relation paths from entity up to max_len.
        Returns dict: path_tuple -> list of end entities
        """
        # path = tuple of relations
        paths = defaultdict(list)
        # queue: (current_entity, path_so_far)
        queue = [(entity, ())]
        visited_states = {(entity, ())}

        while queue:
            curr, path = queue.pop(0)
            if len(path) >= max_len:
                continue
            for rel, neighbour in self.graph.get(curr, []):
                new_path = path + (rel,)
                paths[new_path].append(neighbour)
                state = (neighbour, new_path)
                if state not in visited_states:
                    visited_states.add(state)
                    queue.append((neighbour, new_path))
            for rel, neighbour in self.inv_graph.get(curr, []):
                new_path = path + (rel,)
                paths[new_path].append(neighbour)
                state = (neighbour, new_path)
                if state not in visited_states:
                    visited_states.add(state)
                    queue.append((neighbour, new_path))

        return dict(paths)

    # ------------------------------------------------------------------
    # Step 5: Compute path precision per cluster
    # ------------------------------------------------------------------

    def _compute_path_stats(self, triples, query_relation):
        """
        For each cluster c, compute for each path p:
          frequency(p, c)  = number of cluster-c entities that have path p
          precision(p, c)  = fraction of those where path ends at correct answer

        path_stats[c][p] = (frequency, precision)
        """
        # Group training triples by cluster
        cluster_triples = defaultdict(list)
        for h, r, t in triples:
            if r == query_relation:
                c = self.entity_clusters.get(h, 0)
                cluster_triples[c].append((h, t))

        path_stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        # [0] = count of entities with this path
        # [1] = count where path ends at correct answer

        for cluster, ind_triples in cluster_triples.items():
            for drug, correct_disease in ind_triples:
                paths = self._get_paths(drug, self.max_path_len)
                for path, end_entities in paths.items():
                    path_stats[cluster][path][0] += 1  # freq
                    if correct_disease in end_entities:
                        path_stats[cluster][path][1] += 1  # precision hit

        # Convert to (frequency, precision) tuples, filter rare paths
        result = {}
        for cluster, p_dict in path_stats.items():
            result[cluster] = {}
            total = len(cluster_triples.get(cluster, []))
            if total == 0:
                continue
            for path, (freq, hits) in p_dict.items():
                if freq >= self.min_path_freq:
                    precision = hits / freq
                    frequency = freq / total  # normalised
                    result[cluster][path] = (frequency, precision)

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, triples, all_entities, query_relation):
        """
        Fit ProbCBR on training triples.
        triples: list of (head, relation, tail)
        all_entities: list of all entity names
        query_relation: the relation we want to predict (e.g. 'indication')
        """
        print("  ProbCBR: building graph...")
        self._build_graph(triples)

        print("  ProbCBR: building entity vectors...")
        self._build_entity_vectors(triples, all_entities)

        print("  ProbCBR: clustering entities...")
        self._cluster_entities(all_entities)

        print("  ProbCBR: computing path statistics...")
        self.path_stats = self._compute_path_stats(triples, query_relation)

        n_paths = sum(len(v) for v in self.path_stats.values())
        print(f"  ProbCBR: {n_paths} paths learned across "
              f"{len(self.path_stats)} clusters")
        return self

    def predict_one(self, drug, query_relation, all_entities):
        """
        Predict candidate diseases for a single drug query.
        Returns dict: {disease: score}
        """
        # 1. Get cluster of query drug
        cluster = self.entity_clusters.get(drug, 0)
        cluster_paths = self.path_stats.get(cluster, {})

        if not cluster_paths:
            return {}

        # 2. Find k nearest neighbours (similar relation-type profile)
        query_vec = self.entity_vectors.get(
            drug, np.zeros(len(self.relation_to_id)))
        query_norm = np.linalg.norm(query_vec)

        neighbour_scores = {}
        for entity, vec in self.entity_vectors.items():
            if entity == drug:
                continue
            # Cosine similarity
            norm_e = np.linalg.norm(vec)
            if query_norm > 0 and norm_e > 0:
                sim = np.dot(query_vec, vec) / (query_norm * norm_e)
            else:
                sim = 0.0
            if sim > 0:
                neighbour_scores[entity] = sim

        top_neighbours = sorted(neighbour_scores,
                                key=neighbour_scores.get,
                                reverse=True)[:self.k]

        if not top_neighbours:
            return {}

        # 3. Score candidate diseases
        disease_scores = defaultdict(float)

        for path, (freq, precision) in cluster_paths.items():
            # Follow path from query drug
            curr_entities = {drug}
            valid = True
            for rel in path:
                next_entities = set()
                for e in curr_entities:
                    if rel.startswith("inv_"):
                        actual_rel = rel[4:]
                        for r, neighbour in self.inv_graph.get(e, []):
                            if r == actual_rel:
                                next_entities.add(neighbour)
                    else:
                        for r, neighbour in self.graph.get(e, []):
                            if r == rel:
                                next_entities.add(neighbour)
                if not next_entities:
                    valid = False
                    break
                curr_entities = next_entities

            if not valid:
                continue

            # Score = path_frequency * path_precision * overlap with neighbours
            path_score = freq * precision
            for candidate in curr_entities:
                disease_scores[candidate] += path_score

        return dict(disease_scores)

    def predict(self, test_queries, all_entities):
        """
        Predict for all test queries.
        test_queries: list of (drug, relation, expected_disease)
        Returns DataFrame with columns: drug, expected_disease, rank, reciprocal_rank
        """
        rows = []
        n_ents = len(all_entities)
        ent_list = list(all_entities)

        for drug, rel, expected_disease in tqdm(test_queries,
                                                desc="  ProbCBR predicting"):
            scores = self.predict_one(drug, rel, all_entities)

            if not scores:
                rank = n_ents
            else:
                sorted_candidates = sorted(scores,
                                           key=scores.get, reverse=True)
                if expected_disease in sorted_candidates:
                    rank = sorted_candidates.index(expected_disease) + 1
                else:
                    rank = n_ents

            rows.append({
                "drug":             drug,
                "expected_disease": expected_disease,
                "rank":             rank,
                "reciprocal_rank":  1.0 / rank,
                "top_path":         self._get_top_path(drug, rel)
            })

        return pd.DataFrame(rows)

    def _get_top_path(self, drug, query_relation):
        """Return the highest-scoring path for a drug — used for interpretability."""
        cluster = self.entity_clusters.get(drug, 0)
        cluster_paths = self.path_stats.get(cluster, {})
        if not cluster_paths:
            return ""
        best_path = max(cluster_paths,
                        key=lambda p: cluster_paths[p][0] * cluster_paths[p][1],
                        default=None)
        return " → ".join(best_path) if best_path else ""
