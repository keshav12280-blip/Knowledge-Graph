"""
Stage 6 — Concept Extraction — 3-Pass LLM
Runs three sequential LLM calls per segment via vLLM HTTP API.
Stage 6 is CPU-only in the Python process — all LLM inference is done by
the vLLM server over HTTP. Never loads a model into this process.
All output is plain text parsed by string split — never JSON from LLM.
Output: outputs/{chapter_id}/stage6_raw_concepts.jsonl
        outputs/{chapter_id}/stage6_raw_relations.jsonl
"""

# This stage calls vLLM via HTTP API only. It must not load any GPU model
# into this Python process. The vLLM server handles GPU inference.

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    OUTPUTS_DIR, LOG_FORMAT, LOG_DATE_FORMAT,
    VLLM_BASE_URL, VLLM_LLM_MODEL, VLLM_TIMEOUT_SEC,
    LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE,
    LLM_MAX_CONCEPTS, LLM_MAX_RELATIONS, LLM_MAX_PREREQS,
    SEGMENT_MERGED_TEXT_LIMIT, VALID_CONCEPT_TYPES, VALID_RELATION_TYPES,
    LLM_PASS1_PROMPT, LLM_PASS2_PROMPT, LLM_PASS3_PROMPT,
)


def _setup_logger(out_dir: Path) -> logging.Logger:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage6")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        fh = logging.FileHandler(log_dir / "stage6.log")
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


def _out_concepts(out_dir: Path) -> Path:
    return out_dir / "stage6_raw_concepts.jsonl"


def _out_relations(out_dir: Path) -> Path:
    return out_dir / "stage6_raw_relations.jsonl"


def _is_valid(out_dir: Path) -> bool:
    return _out_concepts(out_dir).exists() and _out_relations(out_dir).exists()


def _load_segments(out_dir: Path) -> List[Dict[str, Any]]:
    p = out_dir / "stage5_segments.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"stage5_segments.jsonl not found at {p}")
    segs = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    segs.append(json.loads(line))
                except Exception:
                    pass
    return segs


# ── vLLM HTTP API inference helper ───────────────────────────────────────────

def call_vllm(prompt: str, model: str = VLLM_LLM_MODEL) -> str:
    """
    Call vLLM server via HTTP OpenAI-compatible API.
    Never loads a model into this Python process.
    Raises requests.HTTPError on server errors.
    """
    response = requests.post(
        f"{VLLM_BASE_URL}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": LLM_MAX_NEW_TOKENS,
            "temperature": LLM_TEMPERATURE,
            "stream": False,
        },
        timeout=VLLM_TIMEOUT_SEC,
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"].strip()
    # Strip chain-of-thought blocks some models emit before the answer
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


class LLMClient:
    """vLLM HTTP API client. No model loaded in this process."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def load(self) -> None:
        """Verify vLLM server is reachable."""
        try:
            resp = requests.get(f"{VLLM_BASE_URL}/models", timeout=10)
            resp.raise_for_status()
            models = [m.get("id","") for m in resp.json().get("data", [])]
            self.logger.info(f"  vLLM server ready. Available models: {models}")
        except Exception as e:
            self.logger.warning(
                f"  vLLM server not reachable at {VLLM_BASE_URL}: {e}\n"
                f"  Start vLLM server before running Stage 6."
            )
            raise RuntimeError(f"vLLM server not reachable: {e}") from e

    def generate_one(self, prompt: str) -> str:
        return call_vllm(prompt)

    def unload(self) -> None:
        """No-op: HTTP client has nothing to unload."""
        pass


# ── Parse helpers ─────────────────────────────────────────────────────────────

def _parse_pass1(
    raw: str,
    segment_id: str,
    domain: str,
    source_types: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """Parse 'NAME | TYPE | DEFINITION' lines."""
    concepts = []
    # Use first source_type in the segment for cross-material tracking
    primary_source_type = (source_types[0] if source_types else "unknown")
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        name, ctype, definition = parts[0], parts[1].lower(), parts[2]
        if not name or len(name) > 120:
            continue
        if ctype not in VALID_CONCEPT_TYPES:
            ctype = "fact"
        concepts.append({
            "concept_id":  str(uuid.uuid4()),
            "name":        name,
            "type":        ctype,
            "definition":  definition,
            "domain":      domain,
            "segment_id":  segment_id,
            "source_type": primary_source_type,
            "source_types": source_types or [],
            "confidence":  1.0,
            "frequency":   1,
            "raw_line":    line,
        })
    return concepts[:LLM_MAX_CONCEPTS]


def _parse_pass2(raw: str, segment_id: str, concept_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """Parse 'SOURCE | RELATION | TARGET | CONFIDENCE' lines."""
    relations = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        source = parts[0]
        relation = parts[1].lower().replace(" ", "_") if len(parts) > 1 else "explains"
        target = parts[2] if len(parts) > 2 else ""
        confidence = 0.8
        if len(parts) >= 4:
            try:
                confidence = float(parts[3])
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                pass
        if not source or not target:
            continue
        if relation not in VALID_RELATION_TYPES:
            relation = "explains"
        relations.append({
            "relation_id": str(uuid.uuid4()),
            "source_name": source,
            "relation":    relation,
            "target_name": target,
            "confidence":  confidence,
            "segment_id":  segment_id,
            "raw_line":    line,
        })
    return relations[:LLM_MAX_RELATIONS]


def _parse_pass3(raw: str, segment_id: str) -> List[Dict[str, Any]]:
    """Parse 'CONCEPT | requires | PREREQUISITE' lines."""
    prereqs = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        concept = parts[0]
        prerequisite = parts[2]
        if not concept or not prerequisite:
            continue
        prereqs.append({
            "relation_id": str(uuid.uuid4()),
            "source_name": concept,
            "relation":    "requires",
            "target_name": prerequisite,
            "confidence":  0.85,
            "segment_id":  segment_id,
            "raw_line":    line,
            "is_prereq":   True,
        })
    return prereqs[:LLM_MAX_PREREQS]


def _process_questionbank_segment(seg: Dict[str, Any]) -> Tuple[List, List]:
    """
    Simplified single-pass for Q&A structured segments (from JSON questionbanks).
    No LLM needed — structure is already explicit.
    """
    concepts = []
    relations = []
    merged_text = seg.get("merged_text", "")

    # Parse Q|A|Wrong lines
    lines = [l.strip() for l in merged_text.splitlines() if l.strip()]
    current_q = current_a = None
    wrong_list = []
    for line in lines:
        if line.startswith("Q:"):
            current_q = line[2:].strip()
        elif line.startswith("A:"):
            current_a = line[2:].strip()
        elif line.startswith("Wrong:"):
            wrong_list = [w.strip() for w in line[6:].split(",")]

    if current_q and current_a:
        cid = str(uuid.uuid4())
        concepts.append({
            "concept_id":  cid,
            "name":        current_q[:100],
            "type":        "fact",
            "definition":  current_a,
            "domain":      seg.get("domain", "general"),
            "segment_id":  seg["segment_id"],
            "confidence":  0.9,
            "frequency":   1,
            "raw_line":    f"Q: {current_q}",
        })
        for wrong in wrong_list:
            if wrong:
                wrong_id = str(uuid.uuid4())
                concepts.append({
                    "concept_id":  wrong_id,
                    "name":        wrong[:100],
                    "type":        "example",
                    "definition":  f"Incorrect answer for: {current_q[:80]}",
                    "domain":      seg.get("domain", "general"),
                    "segment_id":  seg["segment_id"],
                    "confidence":  0.6,
                    "frequency":   1,
                    "raw_line":    f"Wrong: {wrong}",
                })
                relations.append({
                    "relation_id": str(uuid.uuid4()),
                    "source_name": wrong[:100],
                    "relation":    "contrasts_with",
                    "target_name": current_q[:100],
                    "confidence":  0.7,
                    "segment_id":  seg["segment_id"],
                    "raw_line":    f"Wrong: {wrong}",
                    "is_prereq":   False,
                })
    return concepts, relations


def run(chapter_dir: Path, out_dir: Path, force: bool = False) -> Path:
    logger = _setup_logger(out_dir)
    concepts_file  = _out_concepts(out_dir)
    relations_file = _out_relations(out_dir)

    if _is_valid(out_dir) and not force:
        logger.info("Stage 6 outputs exist — skipping")
        return concepts_file

    logger.info(f"Stage 6 start | chapter={chapter_dir.name}")
    segments = _load_segments(out_dir)
    logger.info(f"  {len(segments)} segments to process")

    # Separate questionbank segments (no LLM needed)
    llm_segments = []
    direct_segments = []
    for seg in segments:
        merged = seg.get("merged_text", "")
        if merged.startswith("Q:") and "A:" in merged:
            direct_segments.append(seg)
        else:
            llm_segments.append(seg)

    all_concepts: List[Dict] = []
    all_relations: List[Dict] = []

    # Process direct (questionbank) segments without LLM
    for seg in direct_segments:
        c, r = _process_questionbank_segment(seg)
        all_concepts.extend(c)
        all_relations.extend(r)
    logger.info(f"  Questionbank segments: {len(direct_segments)} → {len(all_concepts)} concepts")

    # Check for partial progress (resumability per segment)
    progress_file = out_dir / "stage6_progress.json"
    done_segments = set()
    if progress_file.exists() and not force:
        try:
            done_segments = set(json.loads(progress_file.read_text()))
            logger.info(f"  Resuming from {len(done_segments)} completed segments")
            # Load already-processed concepts/relations
            if concepts_file.exists():
                with open(concepts_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                all_concepts.append(json.loads(line))
                            except Exception:
                                pass
            if relations_file.exists():
                with open(relations_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                all_relations.append(json.loads(line))
                            except Exception:
                                pass
        except Exception:
            done_segments = set()

    pending = [s for s in llm_segments if s["segment_id"] not in done_segments]
    logger.info(f"  LLM segments pending: {len(pending)}")

    if pending:
        client = LLMClient(logger)
        client.load()

        for i, seg in enumerate(pending):
            seg_id = seg["segment_id"]
            domain = seg.get("domain", "general")
            merged_text = seg.get("merged_text", "")[:SEGMENT_MERGED_TEXT_LIMIT]
            logger.info(
                f"  Segment {i+1}/{len(pending)}  "
                f"domain={domain}  text_len={len(merged_text)}"
            )

            # ── Pass 1: Concept names + types + definitions ───────────────
            prompt1 = LLM_PASS1_PROMPT.format(
                domain=domain,
                max_concepts=LLM_MAX_CONCEPTS,
                merged_text=merged_text,
            )
            try:
                raw1 = client.generate_one(prompt1)
                logger.debug(f"    Pass1 raw (first 300 chars): {raw1[:300]!r}")
                pass1_concepts = _parse_pass1(
                    raw1, seg_id, domain,
                    source_types=seg.get("source_types", []),
                )
                logger.debug(f"    Pass1: {len(pass1_concepts)} concepts")
            except Exception as e:
                logger.warning(f"    Pass1 failed seg {seg_id}: {e}")
                pass1_concepts = []
                raw1 = ""

            if not pass1_concepts:
                done_segments.add(seg_id)
                progress_file.write_text(json.dumps(list(done_segments)))
                continue

            # ── Pass 2: Relationships ─────────────────────────────────────
            concept_list = "\n".join(f"- {c['name']}" for c in pass1_concepts)
            context_short = merged_text[:800]
            prompt2 = LLM_PASS2_PROMPT.format(
                concept_list=concept_list,
                max_relations=LLM_MAX_RELATIONS,
                context_short=context_short,
            )
            try:
                raw2 = client.generate_one(prompt2)
                pass2_relations = _parse_pass2(raw2, seg_id, {})
                logger.debug(f"    Pass2: {len(pass2_relations)} relations")
            except Exception as e:
                logger.warning(f"    Pass2 failed seg {seg_id}: {e}")
                pass2_relations = []

            # ── Pass 3: Prerequisites ─────────────────────────────────────
            prompt3 = LLM_PASS3_PROMPT.format(
                concept_list=concept_list,
                max_prereqs=LLM_MAX_PREREQS,
            )
            try:
                raw3 = client.generate_one(prompt3)
                pass3_prereqs = _parse_pass3(raw3, seg_id)
                logger.debug(f"    Pass3: {len(pass3_prereqs)} prereqs")
            except Exception as e:
                logger.warning(f"    Pass3 failed seg {seg_id}: {e}")
                pass3_prereqs = []

            all_concepts.extend(pass1_concepts)
            all_relations.extend(pass2_relations)
            all_relations.extend(pass3_prereqs)
            done_segments.add(seg_id)

            # Flush progress after each segment (crash safety)
            progress_file.write_text(json.dumps(list(done_segments)))

            # Flush intermediate results (so partial runs are usable)
            if (i + 1) % 10 == 0:
                _flush(all_concepts, all_relations, out_dir, logger)

        client.unload()
        # Clean up progress file on full completion
        if progress_file.exists():
            progress_file.unlink()

    # Final flush
    _flush(all_concepts, all_relations, out_dir, logger, final=True)
    logger.info(
        f"Stage 6 done  | {len(all_concepts)} concepts, "
        f"{len(all_relations)} relations → {concepts_file}"
    )
    return concepts_file


def _flush(
    concepts: List[Dict],
    relations: List[Dict],
    out_dir: Path,
    logger: logging.Logger,
    final: bool = False,
) -> None:
    c_file = _out_concepts(out_dir)
    r_file = _out_relations(out_dir)
    suffix = ".jsonl" if final else ".jsonl.partial"

    tmp_c = c_file.with_suffix(suffix + ".tmp")
    with open(tmp_c, "w", encoding="utf-8") as f:
        for c in concepts:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    if final:
        tmp_c.rename(c_file)

    tmp_r = r_file.with_suffix(suffix + ".tmp")
    with open(tmp_r, "w", encoding="utf-8") as f:
        for r in relations:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if final:
        tmp_r.rename(r_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6 — 3-pass LLM concept extraction")
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
