"""
Stage 11 — Visualization and Export
Generates interactive HTML visualizations from the knowledge graph.
Output: outputs/{chapter_id}/stage11_outputs/
    chapter_overview.html
    {domain}_topic.html
    prerequisite_dag.html
    full_concept_graph.html
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OUTPUTS_DIR, LOG_FORMAT, LOG_DATE_FORMAT, ENABLE_QUERY_INDEX


# Domain → color palette (consistent across all views)
DOMAIN_COLORS = {
    "optics":         "#4FC3F7",
    "mechanics":      "#EF9A9A",
    "chemistry":      "#A5D6A7",
    "electricity":    "#FFE082",
    "thermodynamics": "#FFAB91",
    "waves":          "#CE93D8",
    "quantum":        "#80DEEA",
    "biology":        "#C5E1A5",
    "mathematics":    "#F48FB1",
    "general":        "#B0BEC5",
}
DEFAULT_COLOR = "#CFD8DC"

NODE_TYPE_SIZES = {
    "Chapter":      60,
    "TopicGodNode": 35,
    "AtomicConcept": 15,
    "Source":       10,
}
NODE_TYPE_BORDER = {
    "Chapter":      4,
    "TopicGodNode": 3,
    "AtomicConcept": 1,
    "Source":       1,
}

EDGE_TYPE_COLORS = {
    "requires":      "#E53935",
    "explains":      "#1E88E5",
    "leads_to":      "#43A047",
    "example_of":    "#FB8C00",
    "contains":      "#757575",
    "has_topic":     "#5E35B1",
    "contrasts_with":"#D81B60",
    "formula_for":   "#00ACC1",
    "part_of":       "#6D4C41",
    "applied_in":    "#558B2F",
    "appears_in":    "#9E9E9E",
}


def _setup_logger(out_dir: Path) -> logging.Logger:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage11")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        fh = logging.FileHandler(log_dir / "stage11.log")
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


def _out_dir(out_dir: Path) -> Path:
    return out_dir / "stage11_outputs"


def _is_valid(out_dir: Path) -> bool:
    d = _out_dir(out_dir)
    return (d / "chapter_overview.html").exists()


def _load_graph(out_dir: Path):
    graph_dir = out_dir / "stage10_graph"
    nodes = []
    edges = []
    nodes_file = graph_dir / "nodes.jsonl"
    edges_file = graph_dir / "edges.jsonl"
    if nodes_file.exists():
        with open(nodes_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        nodes.append(json.loads(line))
                    except Exception:
                        pass
    if edges_file.exists():
        with open(edges_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        edges.append(json.loads(line))
                    except Exception:
                        pass
    return nodes, edges


def _node_color(node: Dict) -> str:
    domain = node.get("domain", "")
    return DOMAIN_COLORS.get(domain, DEFAULT_COLOR)


def _node_size(node: Dict) -> int:
    node_type = node.get("type", "AtomicConcept")
    base = NODE_TYPE_SIZES.get(node_type, 15)
    freq = node.get("frequency", 1)
    return base + min(20, freq * 2)


def _node_label(node: Dict) -> str:
    node_type = node.get("type", "AtomicConcept")
    if node_type == "Chapter":
        return node.get("title", node["id"])[:40]
    elif node_type == "TopicGodNode":
        return node.get("title", node["id"])[:35]
    elif node_type == "Source":
        return Path(node.get("source_path", node["id"])).name[:30]
    return node.get("name", node["id"])[:40]


def _node_tooltip(node: Dict) -> str:
    lines = []
    ntype = node.get("type", "")
    if ntype == "AtomicConcept":
        lines.append(f"<b>{node.get('name','')}</b>")
        lines.append(f"Type: {node.get('concept_type','')}")
        lines.append(f"Domain: {node.get('domain','')}")
        lines.append(f"Depth: {node.get('depth_level', 0)}")
        lines.append(f"Frequency: {node.get('frequency',1)}")
        defn = node.get('definition','')
        if defn:
            lines.append(f"<i>{defn[:120]}</i>")
    elif ntype in ("Chapter", "TopicGodNode"):
        lines.append(f"<b>{node.get('title','')}</b>")
        summary = node.get("summary", "")
        if summary:
            lines.append(f"<i>{summary[:200]}</i>")
    return "<br>".join(lines)


def _build_pyvis_network(nodes, edges, title="Knowledge Graph", height="900px"):
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError("pyvis not installed. pip install pyvis")

    net = Network(
        height=height,
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=True,
        notebook=False,
    )
    net.set_options("""
    {
      "nodes": {"font": {"size": 12, "face": "Helvetica"}, "borderWidth": 2},
      "edges": {"arrows": {"to": {"enabled": true, "scaleFactor": 0.8}},
                "smooth": {"type": "dynamic"}},
      "physics": {"stabilization": {"iterations": 100},
                  "barnesHut": {"gravitationalConstant": -12000, "springLength": 120}},
      "interaction": {"hover": true, "tooltipDelay": 200}
    }
    """)

    node_ids = {n["id"] for n in nodes}
    for node in nodes:
        net.add_node(
            node["id"],
            label      = _node_label(node),
            title      = _node_tooltip(node),
            color      = _node_color(node),
            size       = _node_size(node),
            borderWidth= NODE_TYPE_BORDER.get(node.get("type","AtomicConcept"), 1),
            shape      = "dot" if node.get("type") == "AtomicConcept" else "box",
        )

    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        if src not in node_ids or tgt not in node_ids:
            continue
        etype = edge.get("type", "explains")
        conf  = edge.get("confidence", 0.8)
        dashed = bool(edge.get("cross_domain"))
        net.add_edge(
            src, tgt,
            title  = f"{etype} ({conf:.2f})",
            color  = EDGE_TYPE_COLORS.get(etype, "#888"),
            width  = max(1, conf * 4),
            dashes = dashed,
        )
    return net


def build_query_index(
    nodes: List[Dict],
    edges: List[Dict],
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Build optional LlamaIndex KnowledgeGraphIndex for natural language queries.
    Only runs when ENABLE_QUERY_INDEX = True in config.py.
    Enables queries like: "What are prerequisites for Total Internal Reflection?"
    """
    try:
        from llama_index.core import KnowledgeGraphIndex, StorageContext, Settings
        from llama_index.core.graph_stores import SimpleGraphStore
        from llama_index.core.schema import TextNode
    except ImportError:
        logger.warning(
            "  LlamaIndex not installed — skipping query index. "
            "pip install llama-index-core"
        )
        return

    logger.info("  Building LlamaIndex KnowledgeGraphIndex...")
    try:
        graph_store = SimpleGraphStore()
        storage_context = StorageContext.from_defaults(graph_store=graph_store)

        # Build (subject, relation, object) triples for KG index
        # Map node id → name for readable triples
        id_to_name: Dict[str, str] = {}
        for n in nodes:
            label = n.get("name") or n.get("title") or n.get("id", "")
            id_to_name[n["id"]] = label

        kg_triples = []
        for edge in edges:
            src_name = id_to_name.get(edge["source"], edge["source"])
            tgt_name = id_to_name.get(edge["target"], edge["target"])
            rel = edge.get("type", "relates_to")
            if src_name and tgt_name:
                kg_triples.append((src_name, rel, tgt_name))

        # Populate graph store with triples
        for subj, rel, obj in kg_triples:
            graph_store.upsert_triplet(subj, rel, obj)

        # Build index from concept definitions as text nodes
        concept_nodes_data = [n for n in nodes if n.get("type") == "AtomicConcept"]
        text_nodes = []
        for n in concept_nodes_data:
            content = f"{n.get('name','')}: {n.get('definition','')}"
            if content.strip():
                text_nodes.append(TextNode(text=content, id_=n["id"]))

        index = KnowledgeGraphIndex(
            nodes=text_nodes,
            storage_context=storage_context,
            max_triplets_per_chunk=10,
            include_embeddings=False,
        )

        # Persist index
        index_dir = output_dir / "query_index"
        index_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(index_dir))
        logger.info(
            f"  Query index built: {len(kg_triples)} triples, "
            f"{len(text_nodes)} concept nodes → {index_dir}"
        )
        logger.info(
            "  Example queries:\n"
            "    from llama_index.core import load_index_from_storage, StorageContext\n"
            "    from llama_index.core.graph_stores import SimpleGraphStore\n"
            "    storage = StorageContext.from_defaults(persist_dir='query_index', "
            "graph_store=SimpleGraphStore())\n"
            "    index = load_index_from_storage(storage)\n"
            "    query_engine = index.as_query_engine()\n"
            "    print(query_engine.query('What are prerequisites for Snell\\'s Law?'))"
        )
    except Exception as e:
        logger.warning(f"  Query index build failed: {e}")


def _render_html(net, out_path: Path, page_title: str) -> None:
    # pyvis requires .html extension, so use a different temp naming scheme
    tmp = out_path.parent / f".tmp_{out_path.name}"
    net.save_graph(str(tmp))
    # Inject page title into the saved HTML
    html = tmp.read_text(encoding="utf-8")
    html = html.replace("<title>", f"<title>{page_title} — ", 1)
    tmp.write_text(html, encoding="utf-8")
    tmp.rename(out_path)


def run(chapter_dir: Path, out_dir: Path, force: bool = False) -> Path:
    logger = _setup_logger(out_dir)
    exports_dir = _out_dir(out_dir)

    if _is_valid(out_dir) and not force:
        logger.info("Stage 11 outputs exist — skipping")
        return exports_dir

    logger.info(f"Stage 11 start | chapter={chapter_dir.name}")
    exports_dir.mkdir(parents=True, exist_ok=True)

    nodes, edges = _load_graph(out_dir)
    if not nodes:
        logger.warning("  No graph data found — nothing to visualize")
        return exports_dir

    logger.info(f"  {len(nodes)} nodes  {len(edges)} edges")

    # ── Guard: pyvis requires at least 3 nodes to save HTML ───────────────────
    if len(nodes) < 3:
        logger.warning(
            f"  Too few nodes ({len(nodes)}) to render — pyvis requires ≥3 nodes. "
            "Writing summary.json only."
        )
        summary = {
            "chapter_id":    chapter_dir.name,
            "total_nodes":   len(nodes),
            "total_edges":   len(edges),
            "domains":       [],
            "node_types":    {},
            "edge_types":    {},
            "outputs":       [],
            "note":          "Graph too small to render (needs ≥3 nodes).",
        }
        (exports_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2)
        )
        return exports_dir

    # ── 1. Chapter overview (Chapter + TopicGodNodes only) ────────────────────
    overview_nodes = [n for n in nodes if n.get("type") in ("Chapter", "TopicGodNode")]
    overview_node_ids = {n["id"] for n in overview_nodes}
    overview_edges = [
        e for e in edges
        if e["source"] in overview_node_ids and e["target"] in overview_node_ids
    ]
    net = _build_pyvis_network(overview_nodes, overview_edges, title="Chapter Overview", height="600px")
    _render_html(net, exports_dir / "chapter_overview.html", "Chapter Overview")
    logger.info("  chapter_overview.html")

    # ── 2. Per-domain topic views ──────────────────────────────────────────────
    domains = set()
    for n in nodes:
        d = n.get("domain", "")
        if d and n.get("type") == "AtomicConcept":
            domains.add(d)

    for domain in domains:
        # Include god nodes for this domain + all atomic concepts in domain
        domain_nodes = [
            n for n in nodes
            if n.get("domain") == domain or
               (n.get("type") == "TopicGodNode" and n.get("domain") == domain)
        ]
        domain_ids = {n["id"] for n in domain_nodes}
        domain_edges = [
            e for e in edges
            if e["source"] in domain_ids and e["target"] in domain_ids
        ]
        if not domain_nodes:
            continue
        net = _build_pyvis_network(domain_nodes, domain_edges, title=f"{domain.title()} Domain")
        _render_html(net, exports_dir / f"{domain}_topic.html", f"{domain.title()} Topic")
        logger.info(f"  {domain}_topic.html  ({len(domain_nodes)} nodes)")

    # ── 3. Prerequisite DAG ────────────────────────────────────────────────────
    prereq_edges = [e for e in edges if e.get("type") == "requires"]
    if prereq_edges:
        prereq_node_ids = set()
        for e in prereq_edges:
            prereq_node_ids.add(e["source"])
            prereq_node_ids.add(e["target"])
        prereq_nodes = [n for n in nodes if n["id"] in prereq_node_ids]
        net = _build_pyvis_network(
            prereq_nodes, prereq_edges,
            title="Prerequisite Learning Path", height="800px",
        )
        _render_html(net, exports_dir / "prerequisite_dag.html", "Learning Path")
        logger.info(f"  prerequisite_dag.html  ({len(prereq_nodes)} nodes)")

    # ── 4. Full concept graph (AtomicConcepts only to keep it renderable) ─────
    concept_nodes = [n for n in nodes if n.get("type") == "AtomicConcept"]
    concept_ids = {n["id"] for n in concept_nodes}
    concept_edges = [
        e for e in edges
        if e["source"] in concept_ids and e["target"] in concept_ids
    ]
    if concept_nodes:
        net = _build_pyvis_network(
            concept_nodes, concept_edges, title="Full Concept Graph", height="1000px"
        )
        _render_html(net, exports_dir / "full_concept_graph.html", "Full Concept Graph")
        logger.info(f"  full_concept_graph.html  ({len(concept_nodes)} nodes)")

    # ── 5. Write summary JSON ─────────────────────────────────────────────────
    summary = {
        "chapter_id":    chapter_dir.name,
        "total_nodes":   len(nodes),
        "total_edges":   len(edges),
        "domains":       list(domains),
        "node_types":    {
            t: sum(1 for n in nodes if n.get("type") == t)
            for t in ("Chapter","TopicGodNode","AtomicConcept","Source")
        },
        "edge_types": {
            t: sum(1 for e in edges if e.get("type") == t)
            for t in ("requires","explains","leads_to","example_of","contains","has_topic")
        },
        "outputs": [
            "chapter_overview.html",
            "prerequisite_dag.html",
            "full_concept_graph.html",
        ] + [f"{d}_topic.html" for d in sorted(domains)],
    }
    (exports_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )

    # ── Optional LlamaIndex query index ───────────────────────────────────────
    if ENABLE_QUERY_INDEX:
        build_query_index(nodes, edges, exports_dir, logger)
    else:
        logger.info(
            "  Query index skipped (ENABLE_QUERY_INDEX=False in config.py). "
            "Set to True to build a natural language query layer."
        )

    logger.info(f"Stage 11 done  | {exports_dir}")
    return exports_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 11 — Visualization and export")
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
