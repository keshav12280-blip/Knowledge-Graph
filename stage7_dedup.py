"""
Stage 7 — Entity Resolution and Deduplication
CPU-only stage (except LLM Pass 3 which uses vLLM HTTP API).
Pass 1: RapidFuzz lexical dedup
Pass 2: MiniLM semantic clustering (HDBSCAN)
Pass 3: LLM verification for ambiguous clusters (vLLM HTTP API)
Deduplication runs across ALL source types simultaneously — not per-source.
Output: outputs/{chapter_id}/stage7_canonical_concepts.jsonl
        outputs/{chapter_id}/stage7_id_remap.json
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
    RAPIDFUZZ_THRESHOLD, SEMANTIC_SIM_MERGE_THRESHOLD,
    SEMANTIC_SIM_LLM_CHECK_THRESHOLD, HDBSCAN_MIN_CLUSTER_SIZE,
    DEDUP_LLM_BATCH_SIZE, CONFIDENCE_BASE_WEIGHT, CONFIDENCE_FREQUENCY_WEIGHT,
    VLLM_BASE_URL, VLLM_LLM_MODEL, VLLM_TIMEOUT_SEC,
    LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE,
    DEDUP_LLM_PROMPT,
)


def _setup_logger(out_dir: Path) -> logging.Logger:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage7")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        fh = logging.FileHandler(log_dir / "stage7.log")
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


def _out_canonical(out_dir: Path) -> Path:
    return out_dir / "stage7_canonical_concepts.jsonl"


def _out_remap(out_dir: Path) -> Path:
    return out_dir / "stage7_id_remap.json"


def _is_valid(out_dir: Path) -> bool:
    return _out_canonical(out_dir).exists() and _out_remap(out_dir).exists()


def _load_concepts(out_dir: Path) -> List[Dict[str, Any]]:
    p = out_dir / "stage6_raw_concepts.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"stage6_raw_concepts.jsonl not found at {p}")
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


def _normalize(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9 ]", "", name.lower().strip())


# ── Pass 1: RapidFuzz lexical dedup ──────────────────────────────────────────
def _lexical_dedup(concepts: List[Dict], logger: logging.Logger) -> List[List[int]]:
    """
    Returns a list of groups (lists of indices) where indices share the same
    canonical concept by fuzzy string matching.
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        logger.warning("RapidFuzz not installed — skipping lexical dedup")
        return [[i] for i in range(len(concepts))]

    names = [_normalize(c["name"]) for c in concepts]
    groups: List[List[int]] = []
    assigned = [False] * len(names)

    for i, name_i in enumerate(names):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        for j in range(i + 1, len(names)):
            if assigned[j]:
                continue
            score = fuzz.ratio(name_i, names[j])
            if score >= RAPIDFUZZ_THRESHOLD:
                group.append(j)
                assigned[j] = True
        groups.append(group)

    logger.info(f"  Lexical dedup: {len(concepts)} → {len(groups)} groups")
    return groups


# ── Pass 2: Semantic dedup via HDBSCAN ────────────────────────────────────────
def _semantic_dedup(
    concepts: List[Dict],
    groups: List[List[int]],
    logger: logging.Logger,
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Further merge groups using MiniLM embeddings + HDBSCAN.
    Returns (merged_groups, ambiguous_groups) where ambiguous needs LLM check.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import hdbscan
    except ImportError as e:
        logger.warning(f"Missing library for semantic dedup: {e} — using lexical groups only")
        return groups, []

    model_path = str(EMBED_MODEL_DIR) if EMBED_MODEL_DIR.exists() else EMBED_MODEL_ID
    logger.info(f"  Loading embed model for dedup: {model_path}")
    embed_model = SentenceTransformer(model_path)

    # Get canonical name per lexical group (most frequent name)
    group_names = []
    for group in groups:
        name_counts = defaultdict(int)
        for idx in group:
            name_counts[concepts[idx]["name"]] += 1
        canonical = max(name_counts, key=name_counts.get)
        group_names.append(canonical)

    logger.info(f"  Embedding {len(group_names)} group names")
    embeddings = embed_model.encode(
        group_names,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # HDBSCAN clustering
    if len(embeddings) < HDBSCAN_MIN_CLUSTER_SIZE + 1:
        logger.info("  Too few groups for HDBSCAN — skipping semantic dedup")
        return groups, []

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    cluster_labels = clusterer.fit_predict(embeddings)

    # Build cluster_id → list of group indices
    cluster_to_groups: Dict[int, List[int]] = defaultdict(list)
    for gi, label in enumerate(cluster_labels):
        cluster_to_groups[int(label)].append(gi)

    merged_groups: List[List[int]] = []
    ambiguous_groups: List[List[int]] = []

    # Noise points (label=-1) stay as individual groups
    for gi in cluster_to_groups.get(-1, []):
        merged_groups.append(groups[gi])

    for cluster_id, group_indices in cluster_to_groups.items():
        if cluster_id == -1:
            continue
        if len(group_indices) == 1:
            merged_groups.append(groups[group_indices[0]])
            continue

        # Compute within-cluster similarity range
        cluster_embs = embeddings[group_indices]
        sims = []
        for a in range(len(cluster_embs)):
            for b in range(a + 1, len(cluster_embs)):
                sims.append(float(np.dot(cluster_embs[a], cluster_embs[b])))

        min_sim = min(sims) if sims else 1.0
        combined_indices = [idx for gi in group_indices for idx in groups[gi]]

        if min_sim >= SEMANTIC_SIM_MERGE_THRESHOLD:
            # High confidence merge
            merged_groups.append(combined_indices)
        elif min_sim >= SEMANTIC_SIM_LLM_CHECK_THRESHOLD:
            # Ambiguous — send to LLM
            ambiguous_groups.append(combined_indices)
        else:
            # Low similarity — keep separate
            for gi in group_indices:
                merged_groups.append(groups[gi])

    logger.info(
        f"  Semantic dedup: {len(groups)} → "
        f"{len(merged_groups)} groups + {len(ambiguous_groups)} ambiguous"
    )
    return merged_groups, ambiguous_groups


# ── Pass 3: LLM verification via vLLM HTTP API ────────────────────────────────
def _llm_verify(
    ambiguous_groups: List[List[int]],
    concepts: List[Dict],
    logger: logging.Logger,
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Ask vLLM HTTP API whether ambiguous groups should merge.
    Returns (merge_these, keep_separate_these).
    No model loaded into this process.
    """
    if not ambiguous_groups:
        return [], ambiguous_groups

    to_merge = []
    to_keep  = []

    for batch_start in range(0, len(ambiguous_groups), DEDUP_LLM_BATCH_SIZE):
        batch = ambiguous_groups[batch_start: batch_start + DEDUP_LLM_BATCH_SIZE]
        for group in batch:
            names = list(dict.fromkeys(concepts[i]["name"] for i in group))[:8]
            concept_list = ", ".join(f'"{n}"' for n in names)
            prompt = DEDUP_LLM_PROMPT.format(concept_list=concept_list)
            try:
                response = requests.post(
                    f"{VLLM_BASE_URL}/chat/completions",
                    json={
                        "model":       VLLM_LLM_MODEL,
                        "messages":    [{"role": "user", "content": prompt}],
                        "max_tokens":  30,
                        "temperature": 0.0,
                        "stream":      False,
                    },
                    timeout=VLLM_TIMEOUT_SEC,
                )
                response.raise_for_status()
                text = re.sub(r"<think>.*?</think>", "", response.json()["choices"][0]["message"]["content"], flags=re.DOTALL).strip().lower()
                if text.startswith("yes"):
                    to_merge.append(group)
                else:
                    to_keep.append(group)
            except Exception as e:
                logger.warning(f"  LLM dedup call failed: {e}")
                to_keep.append(group)

    logger.info(f"  LLM verified: {len(to_merge)} merge, {len(to_keep)} keep separate")
    return to_merge, to_keep


# ── Canonical concept builder ─────────────────────────────────────────────────
def _build_canonical(
    all_groups: List[List[int]],
    concepts: List[Dict],
) -> Tuple[List[Dict], Dict[str, str]]:
    """
    For each group, pick canonical concept.
    Returns (canonical_concepts, id_remap {old_id → canonical_id}).
    """
    canonical_concepts = []
    id_remap: Dict[str, str] = {}
    max_freq = max((c.get("frequency", 1) for c in concepts), default=1)

    for group in all_groups:
        if not group:
            continue
        group_concepts = [concepts[i] for i in group]

        # Canonical = highest frequency + confidence
        def score(c):
            freq = c.get("frequency", 1)
            conf = c.get("confidence", 1.0)
            return (
                CONFIDENCE_BASE_WEIGHT * conf
                + CONFIDENCE_FREQUENCY_WEIGHT * (freq / max_freq)
            )

        canonical = max(group_concepts, key=score)
        canonical_id = canonical["concept_id"]

        # Aggregate frequency across all group members
        total_freq = sum(c.get("frequency", 1) for c in group_concepts)
        # Collect aliases
        aliases = list(dict.fromkeys(
            c["name"] for c in group_concepts if c["name"] != canonical["name"]
        ))
        # Aggregate source segments
        source_segments = list(dict.fromkeys(
            c.get("segment_id", "") for c in group_concepts if c.get("segment_id")
        ))

        final_confidence = (
            CONFIDENCE_BASE_WEIGHT * canonical.get("confidence", 1.0)
            + CONFIDENCE_FREQUENCY_WEIGHT * (total_freq / max(max_freq, 1))
        )

        canon = {
            **canonical,
            "concept_id":       canonical_id,
            "frequency":        total_freq,
            "aliases":          aliases,
            "source_segments":  source_segments,
            "confidence":       round(min(1.0, final_confidence), 4),
            "difficulty":       1,  # will be inferred from prerequisite depth in Stage 8
        }
        canonical_concepts.append(canon)

        # Build remap for all non-canonical IDs
        for c in group_concepts:
            if c["concept_id"] != canonical_id:
                id_remap[c["concept_id"]] = canonical_id

    return canonical_concepts, id_remap


def run(chapter_dir: Path, out_dir: Path, force: bool = False) -> Path:
    logger = _setup_logger(out_dir)
    canonical_file = _out_canonical(out_dir)
    remap_file     = _out_remap(out_dir)

    if _is_valid(out_dir) and not force:
        logger.info("Stage 7 outputs exist — skipping")
        return canonical_file

    logger.info(f"Stage 7 start | chapter={chapter_dir.name}")
    concepts = _load_concepts(out_dir)
    logger.info(f"  Raw concepts: {len(concepts)}")

    # Validate cross-material coverage — dedup must span ALL source types
    source_types_seen = set(c.get("source_type", "") for c in concepts if c.get("source_type"))
    # Also check segment IDs for diversity (concepts from different segments = different sources)
    logger.info(f"  Deduplicating across source types: {source_types_seen or '(source_type not set in concepts)'}")
    if not source_types_seen or len(source_types_seen) < 2:
        logger.warning(
            "  Only one source type found in concept pool. "
            "Cross-material fusion may not be working correctly. "
            "Check that all stage6 outputs are being loaded."
        )

    if not concepts:
        logger.warning("  No concepts to deduplicate")
        canonical_file.write_text("")
        remap_file.write_text("{}")
        return canonical_file

    # Pass 1: lexical
    lexical_groups = _lexical_dedup(concepts, logger)

    # Pass 2: semantic
    merged_groups, ambiguous_groups = _semantic_dedup(concepts, lexical_groups, logger)

    # Pass 3: LLM on ambiguous
    llm_merge, llm_keep = _llm_verify(ambiguous_groups, concepts, logger)

    # Final group list
    all_groups = merged_groups + llm_merge
    for group in llm_keep:
        # Split into individual concepts
        for idx in group:
            all_groups.append([idx])

    canonical_concepts, id_remap = _build_canonical(all_groups, concepts)
    logger.info(
        f"  Dedup: {len(concepts)} raw → {len(canonical_concepts)} canonical  "
        f"({len(id_remap)} remapped)"
    )

    # Write canonical concepts
    tmp_c = canonical_file.with_suffix(".jsonl.tmp")
    with open(tmp_c, "w", encoding="utf-8") as f:
        for c in canonical_concepts:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    tmp_c.rename(canonical_file)

    # Write remap
    tmp_r = remap_file.with_suffix(".json.tmp")
    tmp_r.write_text(json.dumps(id_remap, ensure_ascii=False, indent=2))
    tmp_r.rename(remap_file)

    logger.info(f"Stage 7 done  | {len(canonical_concepts)} concepts → {canonical_file}")
    return canonical_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 7 — Entity resolution and dedup")
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
