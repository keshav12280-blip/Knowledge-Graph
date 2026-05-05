"""
Stage 8 — Prerequisite DAG Construction
Build directed prerequisite graph from Stage 6 Pass 3 relations.
Apply Stage 7 ID remap. Detect + break cycles. Assign depth levels.
Add temporal cross-segment prerequisite edges.
Output: outputs/{chapter_id}/stage8_prerequisite_dag.json
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    OUTPUTS_DIR, LOG_FORMAT, LOG_DATE_FORMAT,
    TEMPORAL_PREREQ_SIM_THRESHOLD,
    EMBED_MODEL_ID, EMBED_MODEL_DIR, EMBED_BATCH_SIZE,
)


def _setup_logger(out_dir: Path) -> logging.Logger:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage8")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        fh = logging.FileHandler(log_dir / "stage8.log")
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


def _out_path(out_dir: Path) -> Path:
    return out_dir / "stage8_prerequisite_dag.json"


def _is_valid(out_dir: Path) -> bool:
    p = _out_path(out_dir)
    return p.exists() and p.stat().st_size > 0


def _load(out_dir: Path):
    relations_file = out_dir / "stage6_raw_relations.jsonl"
    canonical_file = out_dir / "stage7_canonical_concepts.jsonl"
    remap_file     = out_dir / "stage7_id_remap.json"

    relations = []
    with open(relations_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    relations.append(json.loads(line))
                except Exception:
                    pass

    canonical = {}
    with open(canonical_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    c = json.loads(line)
                    canonical[c["concept_id"]] = c
                except Exception:
                    pass

    id_remap = json.loads(remap_file.read_text()) if remap_file.exists() else {}
    return relations, canonical, id_remap


def _resolve_id(raw_id: str, id_remap: Dict[str, str]) -> str:
    return id_remap.get(raw_id, raw_id)


def _name_to_id(name: str, canonical: Dict[str, Dict]) -> Optional[str]:
    """Find concept_id by name (case-insensitive, also checks aliases)."""
    name_lower = name.lower().strip()
    for cid, c in canonical.items():
        if c["name"].lower().strip() == name_lower:
            return cid
        for alias in c.get("aliases", []):
            if alias.lower().strip() == name_lower:
                return cid
    return None


def _detect_and_break_cycles(
    edges: List[Tuple[str, str, float]],
    logger: logging.Logger,
) -> List[Tuple[str, str, float]]:
    """
    Use networkx to detect cycles. Remove lowest-confidence edge in each cycle.
    Returns clean edge list.
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx not installed — skipping cycle detection")
        return edges

    G = nx.DiGraph()
    for src, tgt, conf in edges:
        G.add_edge(src, tgt, confidence=conf)

    cycles_removed = 0
    while True:
        try:
            cycle = nx.find_cycle(G)
            # cycle is list of (u,v) tuples; find weakest edge
            weakest = min(cycle, key=lambda e: G[e[0]][e[1]].get("confidence", 0))
            G.remove_edge(*weakest)
            cycles_removed += 1
        except nx.NetworkXNoCycle:
            break

    if cycles_removed:
        logger.info(f"  Removed {cycles_removed} cycle edges")

    return [(u, v, G[u][v].get("confidence", 0.8)) for u, v in G.edges()]


def _compute_depths(
    concept_ids: List[str],
    edges: List[Tuple[str, str, float]],
) -> Dict[str, int]:
    """Compute prerequisite depth for each concept node."""
    try:
        import networkx as nx
    except ImportError:
        return {cid: 0 for cid in concept_ids}

    G = nx.DiGraph()
    G.add_nodes_from(concept_ids)
    for src, tgt, _ in edges:
        G.add_edge(src, tgt)

    depths = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        if not preds:
            depths[node] = 0
        else:
            depths[node] = max(depths.get(p, 0) for p in preds) + 1
    return depths


def _add_temporal_prereqs(
    canonical: Dict[str, Dict],
    existing_edges: Set[Tuple[str, str]],
    logger: logging.Logger,
) -> List[Tuple[str, str, float]]:
    """
    Add cross-segment prerequisite edges based on:
    - Semantic similarity > threshold
    - Earlier segment_index appears before later one
    """
    new_edges = []
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        return []

    if EMBED_MODEL_DIR.exists() and (EMBED_MODEL_DIR / "config.json").exists():
        model_path = str(EMBED_MODEL_DIR)
    elif EMBED_MODEL_DIR.exists():
        cache_name = "models--" + EMBED_MODEL_ID.replace("/", "--")
        snapshots = sorted((EMBED_MODEL_DIR / cache_name / "snapshots").glob("*"))
        model_path = str(snapshots[-1]) if snapshots else EMBED_MODEL_ID
    else:
        model_path = EMBED_MODEL_ID
    embed_model = SentenceTransformer(model_path)

    concepts_list = list(canonical.values())
    if not concepts_list:
        return []

    names = [c["name"] for c in concepts_list]
    embeddings = embed_model.encode(names, batch_size=EMBED_BATCH_SIZE, normalize_embeddings=True)

    # Group by segment index (approximate ordering)
    seg_to_concepts: Dict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(concepts_list):
        for seg_id in c.get("source_segments", [c.get("segment_id", "")]):
            seg_to_concepts[seg_id].append(i)

    seg_ids = sorted(seg_to_concepts.keys())

    added = 0
    for seg_i_idx in range(len(seg_ids)):
        for seg_j_idx in range(seg_i_idx + 1, len(seg_ids)):
            seg_i_concepts = seg_to_concepts[seg_ids[seg_i_idx]]
            seg_j_concepts = seg_to_concepts[seg_ids[seg_j_idx]]
            for i in seg_i_concepts:
                for j in seg_j_concepts:
                    if i == j:
                        continue
                    cid_i = concepts_list[i]["concept_id"]
                    cid_j = concepts_list[j]["concept_id"]
                    if (cid_i, cid_j) in existing_edges:
                        continue
                    sim = float(np.dot(embeddings[i], embeddings[j]))
                    if sim >= TEMPORAL_PREREQ_SIM_THRESHOLD:
                        # Earlier concept (seg_i) is prerequisite for later (seg_j)
                        new_edges.append((cid_j, cid_i, round(sim * 0.7, 4)))
                        added += 1
                        if added > 200:  # cap temporal edges
                            return new_edges

    logger.info(f"  Added {len(new_edges)} temporal prerequisite edges")
    return new_edges


def run(chapter_dir: Path, out_dir: Path, force: bool = False) -> Path:
    logger = _setup_logger(out_dir)
    out_file = _out_path(out_dir)

    if _is_valid(out_dir) and not force:
        logger.info("Stage 8 output exists — skipping")
        return out_file

    logger.info(f"Stage 8 start | chapter={chapter_dir.name}")
    relations, canonical, id_remap = _load(out_dir)
    logger.info(f"  {len(relations)} raw relations  |  {len(canonical)} canonical concepts")

    # Extract prerequisite relations only
    prereq_relations = [r for r in relations if r.get("is_prereq") or r.get("relation") == "requires"]
    logger.info(f"  {len(prereq_relations)} prerequisite relations")

    # Apply ID remap + name→ID resolution
    edges: List[Tuple[str, str, float]] = []
    skipped = 0
    for rel in prereq_relations:
        # source is the concept that NEEDS the prerequisite
        # target is the PREREQUISITE concept
        src_name = rel.get("source_name", "")
        tgt_name = rel.get("target_name", "")
        conf = rel.get("confidence", 0.8)

        src_id = _name_to_id(src_name, canonical)
        tgt_id = _name_to_id(tgt_name, canonical)

        if not src_id or not tgt_id:
            skipped += 1
            continue
        if src_id == tgt_id:
            continue

        # Apply remap
        src_id = _resolve_id(src_id, id_remap)
        tgt_id = _resolve_id(tgt_id, id_remap)
        edges.append((src_id, tgt_id, conf))

    logger.info(f"  Resolved {len(edges)} edges  ({skipped} skipped — name not found)")

    # Remove duplicates, keep highest confidence
    edge_map: Dict[Tuple[str, str], float] = {}
    for src, tgt, conf in edges:
        key = (src, tgt)
        edge_map[key] = max(edge_map.get(key, 0.0), conf)
    edges = [(src, tgt, conf) for (src, tgt), conf in edge_map.items()]

    # Detect and break cycles
    edges = _detect_and_break_cycles(edges, logger)

    # Add temporal cross-segment prerequisites
    existing_edge_set = {(src, tgt) for src, tgt, _ in edges}
    temporal_edges = _add_temporal_prereqs(canonical, existing_edge_set, logger)
    edges.extend(temporal_edges)

    # Temporal edges may introduce new cycles — re-run cycle detection
    edges = _detect_and_break_cycles(edges, logger)

    # Compute depth levels
    all_concept_ids = list(canonical.keys())
    depths = _compute_depths(all_concept_ids, edges)

    # Assign depth back to canonical concepts
    for cid, c in canonical.items():
        c["depth_level"] = depths.get(cid, 0)

    # Topological sort = recommended learning order
    try:
        import networkx as nx
        G = nx.DiGraph()
        G.add_nodes_from(all_concept_ids)
        for src, tgt, _ in edges:
            G.add_edge(src, tgt)
        learning_order = list(nx.topological_sort(G))
    except Exception:
        learning_order = sorted(all_concept_ids, key=lambda cid: depths.get(cid, 0))

    output = {
        "chapter_id":    chapter_dir.name,
        "node_count":    len(canonical),
        "edge_count":    len(edges),
        "edges": [
            {"source": src, "target": tgt, "confidence": conf, "type": "requires"}
            for src, tgt, conf in edges
        ],
        "depths":        depths,
        "learning_order": learning_order,
        "depth_distribution": {
            str(d): sum(1 for v in depths.values() if v == d)
            for d in sorted(set(depths.values()))
        },
    }

    tmp_file = out_file.with_suffix(".json.tmp")
    tmp_file.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    tmp_file.rename(out_file)

    logger.info(
        f"Stage 8 done  | {len(edges)} prereq edges  "
        f"max_depth={max(depths.values(), default=0)} → {out_file}"
    )
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 8 — Prerequisite DAG")
    parser.add_argument("--chapter", required=True)
    parser.add_argument("--input",   default=None)
    parser.add_argument("--output",  default=None)
    parser.add_argument("--force",   action="store_true")
    args = parser.parse_args()

    from config import INPUT_ROOT
    chapter_dir = Path(args.chapter)
    if not chapter_dir.is_absolute():
        root = Path(args.input) if args.input else INPUT_ROOT
        chapter_dir = root / chapter_dir
    out_dir = Path(args.output) if args.output else OUTPUTS_DIR / chapter_dir.name
    run(chapter_dir, out_dir, force=args.force)


if __name__ == "__main__":
    main()
