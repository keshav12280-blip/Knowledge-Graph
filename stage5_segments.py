"""
Stage 5 — Segment Formation
CPU-only stage. Must not load any GPU model.
Merge transcripts + VLM captions → detect topic boundaries → assign domains.
Aligns transcript word-timestamps to keyframes for source evidence linking.
Uses LlamaIndex SentenceWindowNodeParser for text chunking.
Uses langdetect to route multilingual content to BGE-M3 instead of MiniLM.
Output: outputs/{chapter_id}/stage5_segments.jsonl
"""

# This stage is CPU-only. It must not load any GPU model.
# If you see a torch.cuda call in this file, it is a bug.

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    OUTPUTS_DIR, LOG_FORMAT, LOG_DATE_FORMAT,
    EMBED_MODEL_ID, EMBED_MODEL_DIR,
    MULTILINGUAL_EMBED_ID, MULTILINGUAL_EMBED_DIR,
    EMBED_BATCH_SIZE,
    SIMILARITY_BOUNDARY_THRESHOLD, SEGMENT_MIN_UNITS, SEGMENT_MAX_UNITS,
    SEGMENT_MAX_DURATION_SEC,
    DOMAIN_CONFIDENCE_THRESHOLD, DOMAIN_SEED_PHRASES,
    MAX_SEGMENT_TOKENS,
    EvidenceUnit,
)


def _setup_logger(out_dir: Path) -> logging.Logger:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage5")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        fh = logging.FileHandler(log_dir / "stage5.log")
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


def _out_path(out_dir: Path) -> Path:
    return out_dir / "stage5_segments.jsonl"


def _is_valid(out_dir: Path) -> bool:
    p = _out_path(out_dir)
    return p.exists() and p.stat().st_size > 0


def _load_units(out_dir: Path) -> List[EvidenceUnit]:
    for name in ("stage4_captions.jsonl", "stage3_transcripts.jsonl", "stage0_units.jsonl"):
        p = out_dir / name
        if p.exists():
            units = []
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            units.append(EvidenceUnit.model_validate_json(line))
                        except Exception:
                            pass
            return units
    raise FileNotFoundError("No usable stage output found for Stage 5 input")


def _load_frame_index(out_dir: Path) -> List[Dict[str, Any]]:
    """Load stage2 selected frames for transcript-to-keyframe alignment."""
    p = out_dir / "stage2_selected_frames.jsonl"
    if not p.exists():
        return []
    frames = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    # Use the copy in selected_frames/ if present
                    if "selected_frame_path" in entry:
                        entry["frame_path"] = entry["selected_frame_path"]
                    frames.append(entry)
                except Exception:
                    pass
    return frames


def align_transcript_to_keyframes(
    transcript_segments: List[Dict],
    keyframes: List[Dict],
) -> List[Dict]:
    """
    For each transcript segment, find keyframes whose timestamp falls within
    [segment.start, segment.end]. Attach them as visual evidence.
    """
    for seg in transcript_segments:
        seg["aligned_keyframes"] = [
            kf for kf in keyframes
            if seg.get("start", 0) <= kf.get("timestamp_sec", -1) <= seg.get("end", 0)
        ]
    return transcript_segments


def _detect_language(text: str) -> str:
    """Detect language of text sample. Returns ISO 639-1 code or 'en' on failure."""
    try:
        from langdetect import detect
        return detect(text[:500])
    except Exception:
        return "en"


def _load_embed_model(text_sample: str, logger: logging.Logger):
    """
    Return MiniLM for English content, BGE-M3 for multilingual content.
    Checks language of text_sample to decide which model to load.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("sentence-transformers not installed. pip install sentence-transformers")

    def _resolve_model_path(model_dir: Path, model_id: str) -> str:
        if not model_dir.exists():
            return model_id
        if (model_dir / "config.json").exists():
            return str(model_dir)
        # HF cache layout: snapshots/<hash>/
        cache_name = "models--" + model_id.replace("/", "--")
        snapshots = sorted((model_dir / cache_name / "snapshots").glob("*"))
        if snapshots:
            return str(snapshots[-1])
        return model_id

    lang = _detect_language(text_sample)
    if lang == "en":
        model_path = _resolve_model_path(EMBED_MODEL_DIR, EMBED_MODEL_ID)
        logger.info(f"  Language detected: {lang} → using MiniLM: {model_path}")
    else:
        model_path = _resolve_model_path(MULTILINGUAL_EMBED_DIR, MULTILINGUAL_EMBED_ID)
        logger.info(f"  Language detected: {lang} → using BGE-M3: {model_path}")

    return SentenceTransformer(model_path)


def _chunk_text_llamaindex(text: str, max_tokens: int = MAX_SEGMENT_TOKENS) -> str:
    """
    Use LlamaIndex SentenceWindowNodeParser to chunk text.
    Returns merged text capped at max_tokens words.
    Falls back to simple word truncation if LlamaIndex unavailable.
    """
    try:
        from llama_index.core.node_parser import SentenceWindowNodeParser
        from llama_index.core import Document

        parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        doc = Document(text=text)
        nodes = parser.get_nodes_from_documents([doc])

        # Combine nodes until max_tokens word limit
        combined = ""
        for node in nodes:
            candidate = (combined + " " + node.text).strip()
            if len(candidate.split()) > max_tokens:
                break
            combined = candidate
        return combined.strip() if combined else text[:max_tokens * 6]
    except ImportError:
        # Fallback: simple word truncation
        words = text.split()
        return " ".join(words[:max_tokens])


def _merge_context(unit: EvidenceUnit) -> str:
    """Build merged text context for a unit (transcript + caption + ocr)."""
    parts = []
    if unit.raw_text:
        parts.append(unit.raw_text)
    if unit.visual_caption:
        parts.append(unit.visual_caption)
    if unit.ocr_text:
        parts.append(unit.ocr_text)
    return " ".join(parts).strip()


def validate_segment(segment: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate a segment before sending to Stage 6 LLM.
    Returns (is_valid, reason).
    """
    merged = segment.get("merged_text", "")
    word_count = len(merged.split())

    if word_count < 20:
        return False, f"Too short: {word_count} words"
    if word_count > MAX_SEGMENT_TOKENS * 2:
        return False, f"Too long: {word_count} words (cap at {MAX_SEGMENT_TOKENS * 2})"
    if not segment.get("domain"):
        return False, "No domain assigned"
    if not merged.strip():
        return False, "Empty merged text"

    return True, "ok"


def _embed_texts(texts: List[str], model) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    )


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _detect_boundaries_ruptures(embeddings: np.ndarray, logger: logging.Logger) -> List[int]:
    """Use ruptures PELT to find changepoints in embedding sequence."""
    try:
        import ruptures as rpt
    except ImportError:
        logger.warning("ruptures not installed — using cosine-sim thresholding only")
        return []

    n = len(embeddings)
    if n < 4:
        return []

    try:
        from sklearn.decomposition import PCA
        n_components = min(10, embeddings.shape[1], n)
        pca = PCA(n_components=n_components)
        signal = pca.fit_transform(embeddings)
    except Exception:
        signal = embeddings[:, :10]

    try:
        algo = rpt.Pelt(model="rbf", jump=1).fit(signal)
        penalty = max(5, n // 5)
        breakpoints = algo.predict(pen=penalty)
        boundaries = [bp - 1 for bp in breakpoints if 0 < bp < n]
        return sorted(set(boundaries))
    except Exception as e:
        logger.warning(f"  ruptures failed: {e}")
        return []


def _detect_boundaries_cosine(embeddings: np.ndarray) -> List[int]:
    """Simple cosine similarity drop detection."""
    boundaries = []
    for i in range(1, len(embeddings)):
        sim = _cosine_sim(embeddings[i-1], embeddings[i])
        if sim < SIMILARITY_BOUNDARY_THRESHOLD:
            boundaries.append(i)
    return boundaries


def _detect_domain(merged_text: str, embed_model, domain_vectors: Dict[str, np.ndarray]) -> tuple:
    """Return (domain_label, confidence)."""
    if not merged_text.strip():
        return "general", 0.0
    text_vec = embed_model.encode([merged_text], normalize_embeddings=True)[0]
    best_domain = "general"
    best_sim = 0.0
    for domain, dvec in domain_vectors.items():
        sim = _cosine_sim(text_vec, dvec)
        if sim > best_sim:
            best_sim = sim
            best_domain = domain
    return best_domain, round(float(best_sim), 4)


def run(chapter_dir: Path, out_dir: Path, force: bool = False) -> Path:
    logger = _setup_logger(out_dir)
    out_file = _out_path(out_dir)

    if _is_valid(out_dir) and not force:
        logger.info("Stage 5 output exists — skipping")
        return out_file

    logger.info(f"Stage 5 start | chapter={chapter_dir.name}")
    units = _load_units(out_dir)
    logger.info(f"  Loaded {len(units)} units")

    # Load selected frames for transcript-to-keyframe alignment
    keyframes = _load_frame_index(out_dir)
    logger.info(f"  Loaded {len(keyframes)} selected keyframes for alignment")

    # Build merged contexts per unit
    contexts = [_merge_context(u) for u in units]

    # Sample text for language detection (first non-empty context)
    sample_text = next((c for c in contexts if c), "educational content")
    embed_model = _load_embed_model(sample_text, logger)

    # Pre-embed domain seed phrases
    logger.info("  Embedding domain seeds")
    domain_vectors: Dict[str, np.ndarray] = {}
    for domain, phrase in DOMAIN_SEED_PHRASES.items():
        domain_vectors[domain] = embed_model.encode([phrase], normalize_embeddings=True)[0]

    logger.info(f"  Embedding {len(contexts)} unit contexts")
    embeddings = _embed_texts(
        [c if c else "no content" for c in contexts],
        embed_model,
    )

    # Detect topic boundaries
    boundaries_cosine = set(_detect_boundaries_cosine(embeddings))
    boundaries_ruptures = set(_detect_boundaries_ruptures(embeddings, logger))
    # Force boundary on new source file
    prev_source = None
    boundaries_source = set()
    for i, u in enumerate(units):
        src = u.source_path
        if prev_source is not None and src != prev_source:
            boundaries_source.add(i)
        prev_source = src

    # Force time-based boundaries for long single-source videos
    # This ensures we get multiple segments even when only one source file exists
    boundaries_time = set()
    for i, u in enumerate(units):
        duration = u.span_end - u.span_start
        if duration > SEGMENT_MAX_DURATION_SEC:
            # Split into chunks of SEGMENT_MAX_DURATION_SEC
            n_splits = int(duration // SEGMENT_MAX_DURATION_SEC)
            for split_idx in range(1, n_splits + 1):
                # Mark this unit as needing a split boundary
                boundaries_time.add(i)
            logger.info(
                f"  Time-based split at unit {i}: {duration:.0f}s > {SEGMENT_MAX_DURATION_SEC}s "
                f"(will create {n_splits + 1} sub-segments)"
            )

    all_boundaries = sorted(
        boundaries_cosine | boundaries_ruptures | boundaries_source | boundaries_time
    )
    logger.info(
        f"  Boundaries: {len(boundaries_cosine)} cosine + "
        f"{len(boundaries_ruptures)} ruptures + "
        f"{len(boundaries_source)} source + "
        f"{len(boundaries_time)} time = "
        f"{len(all_boundaries)} total"
    )

    # Group units into segments
    def make_segments(units_list, boundaries_set):
        segs = []
        current = []
        for i, unit in enumerate(units_list):
            if i in boundaries_set and current:
                segs.append(current)
                current = []
            current.append(i)
            if len(current) >= SEGMENT_MAX_UNITS:
                segs.append(current)
                current = []
        if current:
            segs.append(current)
        return segs

    segment_groups = make_segments(units, set(all_boundaries))

    # Handle single-unit segments with very long duration by splitting text content
    def split_long_single_unit_segments(groups, units_list, contexts):
        """For segments with a single long-duration unit, split the text into chunks."""
        new_groups = []

        for indices in groups:
            if len(indices) != 1:
                new_groups.append(indices)
                continue

            i = indices[0]
            u = units_list[i]
            duration = u.span_end - u.span_start

            if duration <= SEGMENT_MAX_DURATION_SEC:
                new_groups.append(indices)
                continue

            # Split this single unit's content into time-based chunks
            n_splits = int(duration // SEGMENT_MAX_DURATION_SEC)
            chunk_duration = duration / (n_splits + 1)

            text = contexts[i]
            if not text:
                new_groups.append(indices)
                continue

            words = text.split()
            words_per_chunk = max(50, len(words) // (n_splits + 1))

            for split_idx in range(n_splits + 1):
                sub_start = u.span_start + split_idx * chunk_duration
                sub_end = u.span_start + (split_idx + 1) * chunk_duration

                # Extract chunk of words
                word_start = split_idx * words_per_chunk
                word_end = word_start + words_per_chunk if split_idx < n_splits else len(words)
                chunk_text = " ".join(words[word_start:word_end])

                if chunk_text.strip():
                    # Create a synthetic sub-unit with split content
                    new_unit_id = f"{u.unit_id}_split_{split_idx}"
                    new_unit = u.model_copy(update={
                        "unit_id": new_unit_id,
                        "span_start": sub_start,
                        "span_end": sub_end,
                        "raw_text": chunk_text if u.raw_text else "",
                    })
                    units_list.append(new_unit)
                    contexts.append(chunk_text)
                    new_groups.append([len(units_list) - 1])

        return new_groups

    segment_groups = split_long_single_unit_segments(segment_groups, units, contexts)
    logger.info(f"  Formed {len(segment_groups)} raw segments")

    # Build segment objects with validation
    segments = []
    skipped = 0
    for seg_idx, indices in enumerate(segment_groups):
        seg_units = [units[i] for i in indices]
        merged_text_parts = [contexts[i] for i in indices if contexts[i]]
        raw_merged = " ".join(merged_text_parts)

        # Domain detection on merged segment text
        domain, domain_conf = _detect_domain(raw_merged, embed_model, domain_vectors)

        # Transcript-to-keyframe alignment for video units
        all_aligned_kf_paths = []
        for u in seg_units:
            if u.source_type == "video" and u.keyframe_paths:
                all_aligned_kf_paths.extend(u.keyframe_paths)

        # Additionally align by timestamp using stage3 word timestamps
        # (keyframes from stage2 already have timestamp_sec)
        video_unit_ids = {u.unit_id for u in seg_units if u.source_type == "video"}
        span_start = min(u.span_start for u in seg_units)
        span_end   = max(u.span_end   for u in seg_units)
        timestamp_aligned = [
            kf["frame_path"] for kf in keyframes
            if kf.get("unit_id") in video_unit_ids
            and span_start <= kf.get("timestamp_sec", -1) <= span_end
        ]
        all_aligned_kf_paths.extend(timestamp_aligned)
        all_aligned_kf_paths = list(dict.fromkeys(all_aligned_kf_paths))  # deduplicate

        # Chunk/truncate merged text to MAX_SEGMENT_TOKENS
        merged_text = _chunk_text_llamaindex(raw_merged, MAX_SEGMENT_TOKENS)

        seg = {
            "segment_id":            str(uuid.uuid4()),
            "segment_index":         seg_idx,
            "chapter_id":            seg_units[0].chapter_id,
            "domain":                domain,
            "domain_confidence":     domain_conf,
            "needs_llm_domain_check": domain_conf < DOMAIN_CONFIDENCE_THRESHOLD,
            "unit_ids":              [u.unit_id for u in seg_units],
            "source_paths":          list(dict.fromkeys(u.source_path for u in seg_units)),
            "source_types":          list(dict.fromkeys(u.source_type for u in seg_units)),
            "span_start":            span_start,
            "span_end":              span_end,
            "merged_text":           merged_text,
            "unit_count":            len(seg_units),
            "aligned_keyframe_paths": all_aligned_kf_paths[:10],
        }

        valid, reason = validate_segment(seg)
        if not valid:
            logger.debug(f"  Segment {seg_idx} skipped: {reason}")
            skipped += 1
            continue

        # Update domain on units
        for u in seg_units:
            u_idx = units.index(u)
            units[u_idx] = u.model_copy(update={
                "domain":     domain,
                "confidence": domain_conf,
                # Attach aligned keyframes to units for downstream source evidence
                "keyframe_paths": list(dict.fromkeys(
                    u.keyframe_paths + all_aligned_kf_paths
                ))[:20],
            })

        segments.append(seg)

    logger.info(f"  Valid segments: {len(segments)}  skipped: {skipped}")

    # Save segments
    tmp_file = out_file.with_suffix(".jsonl.tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")
    tmp_file.rename(out_file)

    # Also update the units file with domain and aligned keyframe info
    updated_units_file = out_dir / "stage5_units_with_domain.jsonl"
    with open(updated_units_file, "w", encoding="utf-8") as f:
        for unit in units:
            f.write(unit.model_dump_json() + "\n")

    logger.info(f"Stage 5 done  | {len(segments)} segments → {out_file}")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5 — Segment formation")
    parser.add_argument("--chapter", required=True)
    parser.add_argument("--input",   default=None)
    parser.add_argument("--output",  default=None)
    parser.add_argument("--data-dir", default=None, dest="data_dir")
    parser.add_argument("--force",   action="store_true")
    args = parser.parse_args()

    from config import INPUT_ROOT
    chapter_dir = Path(args.chapter)
    if not chapter_dir.is_absolute():
        root = Path(args.input or args.data_dir) if (args.input or args.data_dir) else INPUT_ROOT
        chapter_dir = root / chapter_dir
    out_dir = Path(args.output) if args.output else OUTPUTS_DIR / chapter_dir.name
    run(chapter_dir, out_dir, force=args.force)


if __name__ == "__main__":
    main()