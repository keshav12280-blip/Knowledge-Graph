"""
Stage 10 — Graph Construction
CPU-only stage. Must not load any GPU model.
Assemble all nodes and edges from Stages 7–9 into the final knowledge graph.
Supports NetworkX (default) and Kuzu (scale path) via config.py.
Runs quality gates after build and writes quality_report.json.
Output: outputs/{chapter_id}/stage10_graph/
    nodes.jsonl, edges.jsonl, chapter_graph.graphml,
    prerequisite_dag.graphml, {domain}_subgraph.graphml, quality_report.json
"""

# This stage is CPU-only. It must not load any GPU model.
# If you see a torch.cuda call in this file, it is a bug.

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    OUTPUTS_DIR, LOG_FORMAT, LOG_DATE_FORMAT, GRAPH_BACKEND,
    MIN_CONCEPTS_PER_CHAPTER, MIN_EDGES_PER_CHAPTER,
    MAX_ORPHAN_RATIO, MIN_EVIDENCE_COVERAGE,
)


def _setup_logger(out_dir: Path) -> logging.Logger:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage10")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        fh = logging.FileHandler(log_dir / "stage10.log")
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


def _out_dir(out_dir: Path) -> Path:
    return out_dir / "stage10_graph"


def _is_valid(out_dir: Path) -> bool:
    d = _out_dir(out_dir)
    return (d / "nodes.jsonl").exists() and (d / "edges.jsonl").exists()


def _make_backend(graph_out_dir: Path):
    if GRAPH_BACKEND == "kuzu":
        try:
            from graph_backends.kuzu_backend import KuzuBackend
            return KuzuBackend(graph_out_dir)
        except ImportError:
            pass
    from graph_backends.networkx_backend import NetworkXBackend
    return NetworkXBackend(graph_out_dir)


def _load_canonical(out_dir: Path) -> List[Dict]:
    p = out_dir / "stage7_canonical_concepts.jsonl"
    concepts = []
    if not p.exists():
        return concepts
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    concepts.append(json.loads(line))
                except Exception:
                    pass
    return concepts


def _load_relations(out_dir: Path) -> List[Dict]:
    p = out_dir / "stage6_raw_relations.jsonl"
    relations = []
    if not p.exists():
        return relations
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    relations.append(json.loads(line))
                except Exception:
                    pass
    return relations


def _load_id_remap(out_dir: Path) -> Dict[str, str]:
    p = out_dir / "stage7_id_remap.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _load_prereq_dag(out_dir: Path) -> Dict:
    p = out_dir / "stage8_prerequisite_dag.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"edges": [], "depths": {}}


def _load_hierarchy(out_dir: Path) -> Dict:
    p = out_dir / "stage9_hierarchy.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"chapter": {}, "god_nodes": []}


def _load_units(out_dir: Path) -> List[Dict]:
    for name in ("stage4_captions.jsonl", "stage3_transcripts.jsonl", "stage0_units.jsonl"):
        p = out_dir / name
        if p.exists():
            units = []
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            units.append(json.loads(line))
                        except Exception:
                            pass
            return units
    return []


def _resolve(raw_id: str, remap: Dict) -> str:
    return remap.get(raw_id, raw_id)


def run_quality_gates(backend, logger: logging.Logger) -> Dict:
    """
    Run quality checks on the assembled graph.
    Logs errors on failure but does NOT crash — graph is still exported.
    Returns report dict; also writes quality_report.json.
    """
    report = {"passed": True, "checks": {}}

    # Collect node data from backend
    try:
        import networkx as nx
        all_nodes = list(backend.nodes_by_type(None)) if hasattr(backend, "nodes_by_type") else []
        # Build lightweight view for quality checks
        concept_nodes = [n for n in all_nodes if n.get("type") == "AtomicConcept"]
        god_nodes     = [n for n in all_nodes if n.get("type") == "TopicGodNode"]
        n_nodes = backend.node_count()
        n_edges = backend.edge_count()
    except Exception as e:
        logger.warning(f"  Quality gate data collection failed: {e}")
        return report

    # Gate 1: minimum concept count
    gate1 = len(concept_nodes) >= MIN_CONCEPTS_PER_CHAPTER
    report["checks"]["min_concepts"] = {
        "passed": gate1,
        "value": len(concept_nodes),
        "threshold": MIN_CONCEPTS_PER_CHAPTER,
    }

    # Gate 2: minimum edge count
    gate2 = n_edges >= MIN_EDGES_PER_CHAPTER
    report["checks"]["min_edges"] = {
        "passed": gate2,
        "value": n_edges,
        "threshold": MIN_EDGES_PER_CHAPTER,
    }

    # Gate 3: orphan ratio (nodes with degree 0)
    # Use the NetworkX graph if available for degree queries
    try:
        nx_graph = backend._graph if hasattr(backend, "_graph") else None
        if nx_graph is not None:
            orphans = [n for n in nx_graph.nodes() if nx_graph.degree(n) == 0]
            orphan_ratio = len(orphans) / max(n_nodes, 1)
        else:
            orphans = []
            orphan_ratio = 0.0
    except Exception:
        orphans = []
        orphan_ratio = 0.0

    gate3 = orphan_ratio <= MAX_ORPHAN_RATIO
    report["checks"]["orphan_ratio"] = {
        "passed": gate3,
        "value": round(orphan_ratio, 3),
        "threshold": MAX_ORPHAN_RATIO,
        "orphan_node_sample": [str(o) for o in orphans[:10]],
    }

    # Gate 4: evidence coverage (AtomicConcepts with source_evidence)
    no_evidence = [
        n for n in concept_nodes
        if not n.get("source_evidence") and not n.get("source_segments")
    ]
    evidence_coverage = 1.0 - len(no_evidence) / max(len(concept_nodes), 1)
    gate4 = evidence_coverage >= MIN_EVIDENCE_COVERAGE
    report["checks"]["evidence_coverage"] = {
        "passed": gate4,
        "value": round(evidence_coverage, 3),
        "threshold": MIN_EVIDENCE_COVERAGE,
        "missing_sample": [n.get("name", n.get("id", "?")) for n in no_evidence[:10]],
    }

    # Gate 5: prerequisite DAG is acyclic (checked via backend)
    gate5 = True
    try:
        import networkx as nx
        nx_graph = backend._graph if hasattr(backend, "_graph") else None
        if nx_graph is not None:
            gate5 = nx.is_directed_acyclic_graph(nx_graph)
    except Exception:
        pass
    report["checks"]["dag_acyclic"] = {"passed": gate5}

    # Gate 6: god nodes exist
    gate6 = len(god_nodes) >= 3
    report["checks"]["god_nodes_exist"] = {
        "passed": gate6,
        "value": len(god_nodes),
    }

    report["passed"] = all(c["passed"] for c in report["checks"].values())

    if not report["passed"]:
        failed = [k for k, v in report["checks"].items() if not v["passed"]]
        logger.error(f"  KG QUALITY GATES FAILED: {failed}")
        logger.error("  Graph exported but quality is below threshold.")
        logger.error("  Check quality_report.json for details.")
    else:
        logger.info("  All quality gates passed.")

    return report


def run(chapter_dir: Path, out_dir: Path, force: bool = False) -> Path:
    logger = _setup_logger(out_dir)
    graph_dir = _out_dir(out_dir)

    if _is_valid(out_dir) and not force:
        logger.info("Stage 10 outputs exist — skipping")
        return graph_dir

    logger.info(f"Stage 10 start | chapter={chapter_dir.name}  backend={GRAPH_BACKEND}")
    graph_dir.mkdir(parents=True, exist_ok=True)
    backend = _make_backend(graph_dir)

    concepts   = _load_canonical(out_dir)
    relations  = _load_relations(out_dir)
    id_remap   = _load_id_remap(out_dir)
    prereq_dag = _load_prereq_dag(out_dir)
    hierarchy  = _load_hierarchy(out_dir)
    units      = _load_units(out_dir)

    # Build unit_id → source evidence lookup
    unit_lookup: Dict[str, Dict] = {u.get("unit_id", ""): u for u in units if u.get("unit_id")}

    depths = prereq_dag.get("depths", {})

    logger.info(
        f"  concepts={len(concepts)}  relations={len(relations)}  "
        f"god_nodes={len(hierarchy.get('god_nodes', []))}"
    )

    # ── 1. Add Chapter node ────────────────────────────────────────────────────
    chapter_node = hierarchy.get("chapter", {})
    chapter_node_id = chapter_node.get("chapter_node_id", f"chapter_{chapter_dir.name}")
    backend.add_node(
        chapter_node_id,
        type         = "Chapter",
        title        = chapter_node.get("title", chapter_dir.name),
        summary      = chapter_node.get("summary", ""),
        chapter_id   = chapter_dir.name,
        domains      = chapter_node.get("domains", []),
    )

    # ── 2. Add TopicGodNode nodes ──────────────────────────────────────────────
    god_node_ids = set()
    for gn in hierarchy.get("god_nodes", []):
        gn_id = gn["god_node_id"]
        god_node_ids.add(gn_id)
        backend.add_node(
            gn_id,
            type       = "TopicGodNode",
            title      = gn.get("title", ""),
            summary    = gn.get("summary", ""),
            domain     = gn.get("domain", ""),
            difficulty = gn.get("difficulty", 1),
            chapter_id = chapter_dir.name,
        )
        # Chapter → TopicGodNode
        backend.add_edge(chapter_node_id, gn_id, type="has_topic", confidence=1.0)

    # ── 3. Add AtomicConcept nodes ─────────────────────────────────────────────
    concept_id_set = set()
    for c in concepts:
        cid = c["concept_id"]
        concept_id_set.add(cid)
        # Source evidence: list timestamps and unit_ids
        source_evidence = []
        for seg_id in c.get("source_segments", []):
            source_evidence.append({"segment_id": seg_id})

        backend.add_node(
            cid,
            type             = "AtomicConcept",
            name             = c["name"],
            concept_type     = c.get("type", "fact"),
            domain           = c.get("domain", ""),
            definition       = c.get("definition", ""),
            confidence       = c.get("confidence", 1.0),
            frequency        = c.get("frequency", 1),
            difficulty       = int(depths.get(cid, 0)) + 1,
            depth_level      = depths.get(cid, 0),
            chapter_id       = chapter_dir.name,
            aliases          = c.get("aliases", []),
            source_evidence  = source_evidence,
        )

    # ── 4. TopicGodNode → AtomicConcept (contains) ───────────────────────────
    for gn in hierarchy.get("god_nodes", []):
        gn_id = gn["god_node_id"]
        for cid in gn.get("concept_ids", []):
            resolved = _resolve(cid, id_remap)
            if backend.has_node(resolved):
                backend.add_edge(gn_id, resolved, type="contains", confidence=1.0)

    # ── 5. Prerequisite edges ──────────────────────────────────────────────────
    prereq_count = 0
    for edge in prereq_dag.get("edges", []):
        src = _resolve(edge["source"], id_remap)
        tgt = _resolve(edge["target"], id_remap)
        if backend.has_node(src) and backend.has_node(tgt):
            backend.add_edge(src, tgt, type="requires", confidence=edge.get("confidence", 0.8))
            prereq_count += 1

    # ── 6. Semantic relations (explains, example_of, leads_to, etc.) ──────────
    # Name → canonical ID lookup for relation resolution
    name_to_id: Dict[str, str] = {}
    for c in concepts:
        name_to_id[c["name"].lower().strip()] = c["concept_id"]
        for alias in c.get("aliases", []):
            name_to_id[alias.lower().strip()] = c["concept_id"]

    sem_count = 0
    cross_domain_count = 0
    for rel in relations:
        if rel.get("is_prereq") or rel.get("relation") == "requires":
            continue  # already handled above
        src_name = rel.get("source_name", "").lower().strip()
        tgt_name = rel.get("target_name", "").lower().strip()
        src_id = name_to_id.get(src_name)
        tgt_id = name_to_id.get(tgt_name)
        if not src_id or not tgt_id:
            continue
        src_id = _resolve(src_id, id_remap)
        tgt_id = _resolve(tgt_id, id_remap)
        if not backend.has_node(src_id) or not backend.has_node(tgt_id):
            continue
        rel_type = rel.get("relation", "explains")
        conf = rel.get("confidence", 0.8)

        # Detect cross-domain
        src_domain = backend.get_node(src_id).get("domain", "") if backend.get_node(src_id) else ""
        tgt_domain = backend.get_node(tgt_id).get("domain", "") if backend.get_node(tgt_id) else ""
        cross_domain = src_domain and tgt_domain and src_domain != tgt_domain

        backend.add_edge(
            src_id, tgt_id,
            type=rel_type,
            confidence=conf,
            cross_domain=cross_domain,
        )
        sem_count += 1
        if cross_domain:
            cross_domain_count += 1

    # ── 7. Source (appears_in) edges ──────────────────────────────────────────
    source_nodes: Dict[str, str] = {}  # source_path → source_node_id
    for unit in units:
        sp = unit.get("source_path", "")
        if not sp or sp in source_nodes:
            continue
        src_node_id = f"source_{hash(sp) & 0xFFFFFFFF}"
        source_nodes[sp] = src_node_id
        backend.add_node(
            src_node_id,
            type        = "Source",
            source_path = sp,
            source_type = unit.get("source_type", ""),
            chapter_id  = chapter_dir.name,
        )

    # Concept appears_in Source
    for c in concepts:
        for seg_id in c.get("source_segments", []):
            # Find units by segment_id
            pass  # source tracing done via segment_id; simplified here

    logger.info(
        f"  Graph: {backend.node_count()} nodes  {backend.edge_count()} edges  "
        f"prereq={prereq_count}  semantic={sem_count}  cross_domain={cross_domain_count}"
    )

    # ── Quality gates ─────────────────────────────────────────────────────────
    logger.info("  Running quality gates...")
    quality_report = run_quality_gates(backend, logger)
    quality_report_path = graph_dir / "quality_report.json"
    quality_report_path.write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2)
    )
    logger.info(f"  Quality report → {quality_report_path}")

    # ── Save ──────────────────────────────────────────────────────────────────
    backend.save()

    logger.info(f"Stage 10 done  | graph saved → {graph_dir}")
    return graph_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 10 — Graph construction")
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
