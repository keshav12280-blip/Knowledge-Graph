"""
Stage 9 — God Node Synthesis and Hierarchy
CPU-only stage (LLM calls go to vLLM HTTP API).
Cluster canonical concepts per domain → Topic God Nodes via LLM summaries.
Build 3-level hierarchy: Chapter → TopicGodNodes → AtomicConcepts.
Output: outputs/{chapter_id}/stage9_god_nodes.json
        outputs/{chapter_id}/stage9_hierarchy.json
"""

# This stage is CPU-only. LLM calls go via HTTP to vLLM server.
# It must not load any GPU model into this Python process.

import argparse
import json
import logging
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    OUTPUTS_DIR, LOG_FORMAT, LOG_DATE_FORMAT,
    EMBED_MODEL_ID, EMBED_MODEL_DIR, EMBED_BATCH_SIZE,
    VLLM_BASE_URL, VLLM_LLM_MODEL, VLLM_TIMEOUT_SEC,
    LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE,
    GODNODE_HDBSCAN_MIN_CLUSTER, GODNODE_MAX_CONCEPTS_IN_PROMPT,
    GODNODE_SUMMARY_PROMPT, CHAPTER_NODE_PROMPT,
)


def _setup_logger(out_dir: Path) -> logging.Logger:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage9")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        fh = logging.FileHandler(log_dir / "stage9.log")
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


def _out_godnodes(out_dir: Path) -> Path:
    return out_dir / "stage9_god_nodes.json"


def _out_hierarchy(out_dir: Path) -> Path:
    return out_dir / "stage9_hierarchy.json"


def _is_valid(out_dir: Path) -> bool:
    return _out_godnodes(out_dir).exists() and _out_hierarchy(out_dir).exists()


def _load_canonical(out_dir: Path) -> List[Dict[str, Any]]:
    p = out_dir / "stage7_canonical_concepts.jsonl"
    concepts = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    concepts.append(json.loads(line))
                except Exception:
                    pass
    return concepts


def _load_prereq_dag(out_dir: Path) -> Dict[str, Any]:
    p = out_dir / "stage8_prerequisite_dag.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"depths": {}}


def _cluster_domain_concepts(
    concepts: List[Dict],
    embed_model,
    logger: logging.Logger,
) -> List[List[Dict]]:
    """Cluster concepts within a domain using HDBSCAN. Returns list of clusters."""
    if len(concepts) < GODNODE_HDBSCAN_MIN_CLUSTER:
        return [concepts]

    try:
        import hdbscan
    except ImportError:
        logger.warning("hdbscan not installed — treating entire domain as one cluster")
        return [concepts]

    names = [c["name"] for c in concepts]
    embeddings = embed_model.encode(names, batch_size=EMBED_BATCH_SIZE, normalize_embeddings=True)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=GODNODE_HDBSCAN_MIN_CLUSTER,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(embeddings)

    cluster_map: Dict[int, List[Dict]] = defaultdict(list)
    for concept, label in zip(concepts, labels):
        cluster_map[int(label)].append(concept)

    # Noise (-1) → individual clusters or merge into nearest
    noise = cluster_map.pop(-1, [])
    clusters = list(cluster_map.values())

    # Add noise concepts to nearest cluster (by name similarity) or as their own
    for c in noise:
        if clusters:
            clusters[0].append(c)  # simplest: attach to first cluster
        else:
            clusters.append([c])

    return clusters if clusters else [concepts]


def _parse_godnode_line(raw: str) -> Tuple[str, str, List[str], int]:
    """
    Parse 'TITLE | 2-SENTENCE-SUMMARY | TOP3_CONCEPTS | DIFFICULTY(1-5)'
    Returns (title, summary, top_concepts, difficulty).
    """
    parts = [p.strip() for p in raw.split("|")]
    title    = parts[0] if len(parts) > 0 else "Topic"
    summary  = parts[1] if len(parts) > 1 else ""
    top_str  = parts[2] if len(parts) > 2 else ""
    diff_str = parts[3] if len(parts) > 3 else "2"

    top_concepts = [x.strip() for x in top_str.split(",") if x.strip()]
    try:
        difficulty = int("".join(filter(str.isdigit, diff_str)) or "2")
        difficulty = max(1, min(5, difficulty))
    except Exception:
        difficulty = 2

    return title[:80], summary, top_concepts[:3], difficulty


def _parse_chapter_line(raw: str) -> Tuple[str, str, List[str]]:
    """Parse 'CHAPTER_TITLE | 2-SENTENCE-SUMMARY | DOMAINS'"""
    parts = [p.strip() for p in raw.split("|")]
    title   = parts[0] if len(parts) > 0 else "Chapter"
    summary = parts[1] if len(parts) > 1 else ""
    domains_str = parts[2] if len(parts) > 2 else ""
    domains = [d.strip() for d in domains_str.split(",") if d.strip()]
    return title[:100], summary, domains


def run(chapter_dir: Path, out_dir: Path, force: bool = False) -> Path:
    logger = _setup_logger(out_dir)
    godnodes_file = _out_godnodes(out_dir)
    hierarchy_file = _out_hierarchy(out_dir)

    if _is_valid(out_dir) and not force:
        logger.info("Stage 9 outputs exist — skipping")
        return godnodes_file

    logger.info(f"Stage 9 start | chapter={chapter_dir.name}")
    concepts = _load_canonical(out_dir)
    prereq_dag = _load_prereq_dag(out_dir)
    depths = prereq_dag.get("depths", {})
    logger.info(f"  {len(concepts)} canonical concepts")

    if not concepts:
        logger.warning("  No concepts — writing empty outputs")
        godnodes_file.write_text("[]")
        hierarchy_file.write_text("{}")
        return godnodes_file

    # Load embedding model (CPU)
    from sentence_transformers import SentenceTransformer
    model_path = str(EMBED_MODEL_DIR) if EMBED_MODEL_DIR.exists() else EMBED_MODEL_ID
    embed_model = SentenceTransformer(model_path)

    # Group by domain
    domain_groups: Dict[str, List[Dict]] = defaultdict(list)
    for c in concepts:
        domain_groups[c.get("domain", "general")].append(c)
    logger.info(f"  Domains: {list(domain_groups.keys())}")

    # Cluster within each domain
    all_clusters: List[Dict[str, Any]] = []  # {domain, concepts, cluster_id}
    for domain, domain_concepts in domain_groups.items():
        clusters = _cluster_domain_concepts(domain_concepts, embed_model, logger)
        logger.info(f"  Domain '{domain}': {len(domain_concepts)} concepts → {len(clusters)} clusters")
        for cluster in clusters:
            all_clusters.append({
                "cluster_id":  str(uuid.uuid4()),
                "domain":      domain,
                "concepts":    cluster,
            })

    logger.info(f"  Total clusters (future god nodes): {len(all_clusters)}")

    def _call_vllm(prompt: str) -> str:
        """Call vLLM HTTP API. No model loaded in this process."""
        response = requests.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json={
                "model":       VLLM_LLM_MODEL,
                "messages":    [{"role": "user", "content": prompt}],
                "max_tokens":  200,
                "temperature": LLM_TEMPERATURE,
                "stream":      False,
            },
            timeout=VLLM_TIMEOUT_SEC,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    logger.info(f"  Using vLLM HTTP API at {VLLM_BASE_URL} (model: {VLLM_LLM_MODEL})")

    # Generate god node summaries
    god_nodes = []
    for cluster in all_clusters:
        concept_names = [c["name"] for c in cluster["concepts"][:GODNODE_MAX_CONCEPTS_IN_PROMPT]]
        concept_list = ", ".join(concept_names)
        prompt = GODNODE_SUMMARY_PROMPT.format(concept_list=concept_list)
        try:
            raw = _call_vllm(prompt)
            first_line = next((l.strip() for l in raw.splitlines() if l.strip()), raw)
            title, summary, top_concepts, difficulty = _parse_godnode_line(first_line)
        except Exception as e:
            logger.warning(f"  God node LLM failed cluster {cluster['cluster_id']}: {e}")
            title = concept_names[0] if concept_names else "Topic"
            summary = ""
            top_concepts = concept_names[:3]
            difficulty = 2

        # Compute cluster difficulty from depth distribution
        cluster_depths = [depths.get(c["concept_id"], 0) for c in cluster["concepts"]]
        avg_depth = sum(cluster_depths) / max(len(cluster_depths), 1)
        difficulty = max(1, min(5, int(avg_depth) + 1))

        god_node = {
            "god_node_id":    str(uuid.uuid4()),
            "type":           "TopicGodNode",
            "title":          title,
            "summary":        summary,
            "domain":         cluster["domain"],
            "difficulty":     difficulty,
            "top_concepts":   top_concepts,
            "concept_ids":    [c["concept_id"] for c in cluster["concepts"]],
            "concept_count":  len(cluster["concepts"]),
            "cluster_id":     cluster["cluster_id"],
        }
        god_nodes.append(god_node)
        logger.debug(f"  God node: '{title}' ({cluster['domain']}) {len(cluster['concepts'])} concepts")

    # Generate chapter node
    topic_titles = ", ".join(gn["title"] for gn in god_nodes[:40])
    chapter_prompt = CHAPTER_NODE_PROMPT.format(topic_titles=topic_titles)
    try:
        raw = _call_vllm(chapter_prompt)
        first_line = next((l.strip() for l in raw.splitlines() if l.strip()), raw)
        chapter_title, chapter_summary, chapter_domains = _parse_chapter_line(first_line)
    except Exception as e:
        logger.warning(f"  Chapter node LLM failed: {e}")
        chapter_title   = chapter_dir.name
        chapter_summary = ""
        chapter_domains = list(domain_groups.keys())

    chapter_node = {
        "chapter_node_id": str(uuid.uuid4()),
        "type":            "Chapter",
        "title":           chapter_title,
        "summary":         chapter_summary,
        "chapter_id":      chapter_dir.name,
        "domains":         chapter_domains,
        "god_node_ids":    [gn["god_node_id"] for gn in god_nodes],
        "god_node_count":  len(god_nodes),
        "concept_count":   len(concepts),
    }

    # Build hierarchy dict
    hierarchy = {
        "chapter":   chapter_node,
        "god_nodes": god_nodes,
        "concept_count": len(concepts),
    }

    # Write outputs
    tmp_gn = godnodes_file.with_suffix(".json.tmp")
    tmp_gn.write_text(json.dumps(god_nodes, ensure_ascii=False, indent=2))
    tmp_gn.rename(godnodes_file)

    tmp_h = hierarchy_file.with_suffix(".json.tmp")
    tmp_h.write_text(json.dumps(hierarchy, ensure_ascii=False, indent=2))
    tmp_h.rename(hierarchy_file)

    logger.info(
        f"Stage 9 done  | {len(god_nodes)} god nodes  "
        f"chapter='{chapter_title}' → {godnodes_file}"
    )
    return godnodes_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 9 — God nodes and hierarchy")
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
