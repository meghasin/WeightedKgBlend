"""
06_update_mind_mapping.py
-------------------------
Re-maps DrugCentral indications to MRN using current DrugCentral data.
Uses a cascade of matching strategies:
  1. UMLS CUI match (most reliable)
  2. MeSH ID match
  3. Exact name match (lowercase, stripped)
  4. Fuzzy name match (fallback)

Produces an updated MIND TSV with more indication edges than the
original manual mapping (5,558 edges from 2021).

Usage:
    python scripts/06_update_mind_mapping.py \
        --mrn_path data/mind.tsv \
        --output_path data/mind_updated.tsv \
        --report_path data/mapping_report.csv

Requirements:
    pip install requests pandas rapidfuzz tqdm
"""

import argparse
import io
import re
import requests
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

try:
    from rapidfuzz import fuzz, process as rfprocess
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("WARNING: rapidfuzz not installed. Fuzzy matching disabled.")
    print("Install with: pip install rapidfuzz")

DRUGCENTRAL_INDICATIONS_URL = (
    "https://unmtid-shinyapps.net/download/drugcentral/drug_indications.tsv"
)
FALLBACK_URL = (
    "https://drugcentral.org/static/download/drug_indications.tsv"
)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_drugcentral() -> pd.DataFrame:
    for url in [DRUGCENTRAL_INDICATIONS_URL, FALLBACK_URL]:
        try:
            print(f"Downloading DrugCentral from:\n  {url}")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), sep="\t", low_memory=False)
            print(f"Downloaded {len(df):,} rows, columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"  Failed: {e}")
    raise RuntimeError(
        "Could not download DrugCentral.\n"
        "Download manually from https://drugcentral.org/download\n"
        "and pass with --dc_path data/drug_indications.tsv"
    )


# ---------------------------------------------------------------------------
# MRN analysis
# ---------------------------------------------------------------------------

def analyse_mrn(path: str) -> tuple:
    print(f"\nLoading MRN from {path}...")
    mrn = pd.read_csv(path, sep="\t", header=None, names=["head","relation","tail"])
    print(f"Triples: {len(mrn):,} | Relations: {mrn.relation.nunique()}")
    print("\nRelation counts:")
    print(mrn.relation.value_counts().to_string())
    nodes = pd.Series(pd.unique(mrn[["head","tail"]].values.ravel())).dropna().unique()
    print(f"\nUnique nodes: {len(nodes):,}")
    return mrn, nodes


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def norm(s):
    if pd.isna(s): return ""
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower().strip())

def get_cui(s):
    if pd.isna(s): return None
    m = re.search(r'C\d{7}', str(s))
    return m.group(0) if m else None

def get_mesh(s):
    if pd.isna(s): return None
    m = re.search(r'D\d{6}', str(s))
    return m.group(0) if m else None


# ---------------------------------------------------------------------------
# Build indexes
# ---------------------------------------------------------------------------

def build_indexes(nodes) -> dict:
    name_idx, cui_idx, mesh_idx = {}, {}, {}
    for node in nodes:
        n = norm(node)
        if n: name_idx[n] = node
        c = get_cui(node)
        if c: cui_idx[c] = node
        m = get_mesh(node)
        if m: mesh_idx[m] = node
    print(f"\nIndexes — name: {len(name_idx):,} | CUI: {len(cui_idx):,} | MeSH: {len(mesh_idx):,}")
    return {"name": name_idx, "cui": cui_idx, "mesh": mesh_idx,
            "all_names": list(name_idx.keys())}


# ---------------------------------------------------------------------------
# Match entity
# ---------------------------------------------------------------------------

def match_entity(name, cui, mesh, idx, threshold=88):
    # 1. UMLS CUI
    if cui:
        c = get_cui(str(cui))
        if c and c in idx["cui"]: return idx["cui"][c], "umls_cui"
    # 2. MeSH
    if mesh:
        m = get_mesh(str(mesh))
        if m and m in idx["mesh"]: return idx["mesh"][m], "mesh"
    # 3. Exact name
    n = norm(name)
    if n and n in idx["name"]: return idx["name"][n], "exact_name"
    # 4. Fuzzy
    if FUZZY_AVAILABLE and n and idx["all_names"]:
        res = rfprocess.extractOne(n, idx["all_names"],
                                   scorer=fuzz.token_sort_ratio,
                                   score_cutoff=threshold)
        if res:
            matched, score, _ = res
            return idx["name"][matched], f"fuzzy_{score}"
    return None, None


# ---------------------------------------------------------------------------
# Map indications
# ---------------------------------------------------------------------------

def map_indications(dc: pd.DataFrame, idx: dict) -> pd.DataFrame:
    # Detect column names
    drug_col    = next((c for c in dc.columns if "drug" in c.lower() and "name" in c.lower()), dc.columns[0])
    disease_col = next((c for c in dc.columns if any(x in c.lower() for x in ["concept","indication","disease","snomed_full"])), None)
    cui_col     = next((c for c in dc.columns if "umls" in c.lower() and "cui" in c.lower()), None)
    mesh_col    = next((c for c in dc.columns if "mesh" in c.lower()), None)
    status_col  = next((c for c in dc.columns if "status" in c.lower()), None)

    print(f"\nColumns: drug={drug_col} | disease={disease_col} | CUI={cui_col} | MeSH={mesh_col} | status={status_col}")

    # Filter FDA approved
    if status_col:
        approved = dc[dc[status_col].str.upper().isin(["APPROVED","FDA","EMA"])].copy()
        print(f"FDA/EMA approved: {len(approved):,} of {len(dc):,}")
    else:
        approved = dc.copy()
        print(f"No status column — using all {len(approved):,} rows")

    rows = []
    drug_stats, dis_stats = defaultdict(int), defaultdict(int)

    print(f"\nMatching {len(approved):,} indications...")
    for _, row in tqdm(approved.iterrows(), total=len(approved)):
        drug_name = row.get(drug_col, "")
        dis_name  = row.get(disease_col, "") if disease_col else ""
        cui       = row.get(cui_col, None) if cui_col else None
        mesh      = row.get(mesh_col, None) if mesh_col else None

        drug_node, drug_m = match_entity(drug_name, None, None, idx)
        dis_node,  dis_m  = match_entity(dis_name, cui, mesh, idx)

        drug_stats[drug_m or "unmatched"] += 1
        dis_stats[dis_m or "unmatched"]   += 1

        if drug_node and dis_node:
            rows.append({
                "head": drug_node, "relation": "indication", "tail": dis_node,
                "dc_drug": drug_name, "dc_disease": dis_name,
                "drug_match": drug_m, "dis_match": dis_m,
            })

    result = pd.DataFrame(rows).drop_duplicates(subset=["head","relation","tail"])
    print(f"\nMatched: {len(result):,} unique indication triples")
    print(f"Drug match methods:    {dict(drug_stats)}")
    print(f"Disease match methods: {dict(dis_stats)}")
    return result


# ---------------------------------------------------------------------------
# Update MIND
# ---------------------------------------------------------------------------

def update_mind(mrn, matched):
    existing = mrn[mrn.relation == "indication"]
    print(f"\nExisting indication edges: {len(existing):,}")
    existing_set = set(zip(existing.head, existing.tail))
    new_triples = matched[["head","relation","tail"]].copy()
    new_triples["is_new"] = new_triples.apply(
        lambda r: (r.head, r.tail) not in existing_set, axis=1)
    genuinely_new = new_triples[new_triples.is_new].drop("is_new", axis=1)
    print(f"Genuinely new:             {len(genuinely_new):,}")
    print(f"Already in MRN:            {len(new_triples)-len(genuinely_new):,}")
    updated = pd.concat([mrn, genuinely_new], ignore_index=True)
    print(f"Updated MRN: {len(updated):,} triples (was {len(mrn):,})")
    return updated, genuinely_new


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mrn_path",        default="data/mind.tsv")
    parser.add_argument("--output_path",     default="data/mind_updated.tsv")
    parser.add_argument("--report_path",     default="data/mapping_report.csv")
    parser.add_argument("--dc_path",         default=None)
    parser.add_argument("--fuzzy_threshold", type=int, default=88)
    args = parser.parse_args()

    mrn, nodes = analyse_mrn(args.mrn_path)
    idx = build_indexes(nodes)

    if args.dc_path and Path(args.dc_path).exists():
        dc = pd.read_csv(args.dc_path, sep="\t", low_memory=False)
        print(f"Loaded local DrugCentral: {len(dc):,} rows")
    else:
        dc = download_drugcentral()

    matched     = map_indications(dc, idx)
    updated, new_edges = update_mind(mrn, matched)

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    updated[["head","relation","tail"]].to_csv(
        args.output_path, sep="\t", index=False, header=False)
    matched.to_csv(args.report_path, index=False)

    orig_count = len(mrn[mrn.relation == "indication"])
    total      = orig_count + len(new_edges)

    print(f"\n{'='*50}")
    print(f"MANUSCRIPT NUMBERS:")
    print(f"  Original (2021 manual): 5,558")
    print(f"  Current MIND:           {orig_count:,}")
    print(f"  New from re-mapping:    {len(new_edges):,}")
    print(f"  Total updated:          {total:,}")
    print(f"  Improvement:            +{len(new_edges):,} ({len(new_edges)/max(orig_count,1)*100:.1f}%)")
    print(f"{'='*50}")
    print(f"\nSaved: {args.output_path}")
    print(f"Report: {args.report_path}")


if __name__ == "__main__":
    main()
