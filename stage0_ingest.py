"""
Stage 0 — Input Scan and Routing
CPU-only stage. Must not load any GPU model.
Scans chapter folder, detects file types, creates skeleton EvidenceUnits.
Output: outputs/{chapter_id}/stage0_units.jsonl
"""

# This stage is CPU-only. It must not load any GPU model.
# If you see a torch.cuda call in this file, it is a bug.

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    OUTPUTS_DIR, FRAMES_DIR, INPUT_ROOT, LOG_FORMAT, LOG_DATE_FORMAT,
    SUPPORTED_VIDEO_EXTS, SUPPORTED_PDF_EXTS, SUPPORTED_JSON_EXTS,
    SUPPORTED_TEXT_EXTS, SUPPORTED_IMAGE_EXTS,
    PDF_MIN_TEXT_CHARS_PER_PAGE, EvidenceUnit,
    VLLM_BASE_URL, VLLM_LLM_MODEL, VLLM_TIMEOUT_SEC,
)
from ingest_adapters.video_adapter import probe_video, detect_video_subtype
from ingest_adapters.pdf_adapter import classify_pdf_pages
from ingest_adapters.json_adapter import extract_json, handle_unknown_json
from ingest_adapters.text_adapter import extract_text


def _setup_logger(out_dir: Path) -> logging.Logger:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage0")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        fh = logging.FileHandler(log_dir / "stage0.log")
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


def _out_path(out_dir: Path) -> Path:
    return out_dir / "stage0_units.jsonl"


def _is_valid(out_dir: Path) -> bool:
    p = _out_path(out_dir)
    return p.exists() and p.stat().st_size > 0


def _load_txt_with_llamaindex(txt_paths: List[Path]) -> List[dict]:
    """
    Use LlamaIndex SimpleDirectoryReader for TXT files.
    Falls back to plain text_adapter if LlamaIndex is unavailable.
    """
    try:
        from llama_index.core import SimpleDirectoryReader
        docs = SimpleDirectoryReader(input_files=[str(p) for p in txt_paths]).load_data()
        return [{"text": d.text, "source": d.metadata.get("file_path", "")} for d in docs]
    except ImportError:
        # Fallback: use plain text_adapter
        results = []
        for p in txt_paths:
            parsed = extract_text(p)
            for seg in parsed["segments"]:
                if seg.get("text", "").strip():
                    results.append({"text": seg["text"], "source": str(p)})
        return results


def run(chapter_dir: Path, out_dir: Path, force: bool = False) -> Path:
    logger = _setup_logger(out_dir)
    out_file = _out_path(out_dir)

    if _is_valid(out_dir) and not force:
        logger.info("Stage 0 output exists — skipping (use --force to rerun)")
        return out_file

    logger.info(f"Stage 0 start | chapter={chapter_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    chapter_id = chapter_dir.name

    units = []
    files = sorted(chapter_dir.iterdir())
    logger.info(f"  Found {len(files)} files")

    # Collect TXT paths for batch LlamaIndex load
    txt_paths: List[Path] = []

    for fpath in files:
        if not fpath.is_file():
            continue
        suffix = fpath.suffix.lower()

        # ── Video ──────────────────────────────────────────────────────────
        if suffix in SUPPORTED_VIDEO_EXTS:
            try:
                meta = probe_video(fpath)
                subtype = detect_video_subtype(fpath, meta)
            except Exception as e:
                logger.warning(f"  ffprobe failed {fpath.name}: {e}")
                meta = {"duration_sec": 0, "fps": 1, "width": 0, "height": 0}
                subtype = "unknown"

            unit = EvidenceUnit(
                unit_id=str(uuid.uuid4()),
                source_path=str(fpath),
                source_type="video",
                video_subtype=subtype,
                chapter_id=chapter_id,
                span_start=0.0,
                span_end=meta.get("duration_sec", 0.0),
            )
            units.append(unit)
            logger.info(f"  VIDEO  {fpath.name}  subtype={subtype}  dur={meta['duration_sec']:.0f}s")

        # ── PDF ────────────────────────────────────────────────────────────
        elif suffix in SUPPORTED_PDF_EXTS:
            frames_dir = FRAMES_DIR / chapter_id
            try:
                pages = classify_pdf_pages(fpath, frames_dir)
            except Exception as e:
                logger.warning(f"  PDF classify failed {fpath.name}: {e}")
                pages = []

            text_pages  = [p for p in pages if p["page_type"] == "text"]
            image_pages = [p for p in pages if p["page_type"] == "visual"]
            sparse_pages = [p for p in pages if p["page_type"] == "sparse"]

            # One unit per text-page chunk (groups of 3)
            chunk_size = 3
            for i in range(0, len(text_pages), chunk_size):
                chunk = text_pages[i: i + chunk_size]
                merged_text = "\n\n".join(p["text"] for p in chunk)
                unit = EvidenceUnit(
                    unit_id=str(uuid.uuid4()),
                    source_path=str(fpath),
                    source_type="pdf",
                    video_subtype="null",
                    chapter_id=chapter_id,
                    span_start=float(chunk[0]["page_num"]),
                    span_end=float(chunk[-1]["page_num"]),
                    raw_text=merged_text,
                )
                units.append(unit)

            # One unit per visual page (VLM will caption via the pre-rendered JPEG)
            for p in image_pages:
                unit = EvidenceUnit(
                    unit_id=str(uuid.uuid4()),
                    source_path=str(fpath),
                    source_type="pdf",
                    video_subtype="null",
                    chapter_id=chapter_id,
                    span_start=float(p["page_num"]),
                    span_end=float(p["page_num"]),
                    raw_text="",
                    # Store the pre-rendered image path so Stage 4 can find it
                    keyframe_paths=[p["image_path"]] if p.get("image_path") else [],
                )
                units.append(unit)

            logger.info(
                f"  PDF    {fpath.name}  "
                f"{len(pages)} pages  "
                f"{len(text_pages)} text  {len(image_pages)} visual  {len(sparse_pages)} sparse"
            )

        # ── JSON / JSONL ───────────────────────────────────────────────────
        elif suffix in SUPPORTED_JSON_EXTS:
            try:
                parsed = extract_json(fpath)
            except Exception as e:
                logger.warning(f"  JSON parse failed {fpath.name}: {e}")
                parsed = {"schema": "error", "items": [], "raw": None}

            schema = parsed.get("schema", "unknown")
            items  = parsed.get("items", [])

            if schema == "questionbank":
                for item in items:
                    text = (
                        f"Q: {item['question']}\n"
                        f"A: {item['answer']}\n"
                        + (f"Wrong: {', '.join(item['wrong'])}" if item["wrong"] else "")
                    )
                    unit = EvidenceUnit(
                        unit_id=str(uuid.uuid4()),
                        source_path=str(fpath),
                        source_type="json",
                        video_subtype="null",
                        chapter_id=chapter_id,
                        span_start=float(item.get("index", 0)),
                        span_end=float(item.get("index", 0)),
                        raw_text=text,
                        confidence=0.95,
                    )
                    units.append(unit)
            elif schema == "transcript":
                for item in items:
                    unit = EvidenceUnit(
                        unit_id=str(uuid.uuid4()),
                        source_path=str(fpath),
                        source_type="text",
                        video_subtype="null",
                        chapter_id=chapter_id,
                        span_start=float(item.get("start", 0)),
                        span_end=float(item.get("end", 0)),
                        raw_text=item.get("text", ""),
                    )
                    units.append(unit)
            else:
                # Unknown schema — call vLLM HTTP API for structure description
                raw_desc = ""
                if parsed.get("raw") is not None:
                    logger.info(f"  JSON unknown schema {fpath.name} — querying vLLM for structure")
                    try:
                        raw_desc = handle_unknown_json(
                            parsed["raw"],
                            VLLM_BASE_URL,
                            VLLM_LLM_MODEL,
                            VLLM_TIMEOUT_SEC,
                        )
                    except Exception as e:
                        logger.warning(f"  vLLM JSON structure query failed: {e}")

                raw_text = raw_desc or json.dumps(items, ensure_ascii=False)[:4000]
                unit = EvidenceUnit(
                    unit_id=str(uuid.uuid4()),
                    source_path=str(fpath),
                    source_type="json",
                    video_subtype="null",
                    chapter_id=chapter_id,
                    span_start=0.0,
                    span_end=float(len(items)),
                    raw_text=raw_text,
                )
                units.append(unit)

            n_units_from_file = sum(1 for u in units if u.source_path == str(fpath))
            logger.info(f"  JSON   {fpath.name}  schema={schema}  {len(items)} items → {n_units_from_file} units")

        # ── Plain text / subtitles ─────────────────────────────────────────
        elif suffix in SUPPORTED_TEXT_EXTS:
            txt_paths.append(fpath)  # batched below

        # ── Images ─────────────────────────────────────────────────────────
        elif suffix in SUPPORTED_IMAGE_EXTS:
            unit = EvidenceUnit(
                unit_id=str(uuid.uuid4()),
                source_path=str(fpath),
                source_type="image",
                video_subtype="null",
                chapter_id=chapter_id,
                span_start=0.0,
                span_end=0.0,
            )
            units.append(unit)
            logger.info(f"  IMAGE  {fpath.name}")

        else:
            logger.debug(f"  SKIP   {fpath.name}  (unsupported extension)")

    # ── Process TXT files via LlamaIndex SimpleDirectoryReader ────────────
    if txt_paths:
        # SRT/VTT files need timed parsing — handle separately
        srt_vtt = [p for p in txt_paths if p.suffix.lower() in (".srt", ".vtt")]
        plain_txt = [p for p in txt_paths if p.suffix.lower() not in (".srt", ".vtt")]

        for fpath in srt_vtt:
            try:
                parsed = extract_text(fpath)
                for seg in parsed["segments"]:
                    if not seg.get("text", "").strip():
                        continue
                    unit = EvidenceUnit(
                        unit_id=str(uuid.uuid4()),
                        source_path=str(fpath),
                        source_type="text",
                        video_subtype="null",
                        chapter_id=chapter_id,
                        span_start=float(seg.get("start", 0)),
                        span_end=float(seg.get("end", 0)),
                        raw_text=seg["text"],
                    )
                    units.append(unit)
                logger.info(f"  TEXT   {fpath.name}  {len(parsed['segments'])} segments (SRT/VTT)")
            except Exception as e:
                logger.warning(f"  Text parse failed {fpath.name}: {e}")

        if plain_txt:
            logger.info(f"  Loading {len(plain_txt)} TXT file(s) via LlamaIndex SimpleDirectoryReader")
            try:
                loaded = _load_txt_with_llamaindex(plain_txt)
                # Group loaded docs back by source file for unit creation
                from collections import defaultdict
                by_source = defaultdict(list)
                for doc in loaded:
                    by_source[doc["source"]].append(doc["text"])

                for fpath in plain_txt:
                    texts = by_source.get(str(fpath), [])
                    if not texts:
                        # Fallback: re-read directly
                        parsed = extract_text(fpath)
                        texts = [s["text"] for s in parsed["segments"] if s.get("text", "").strip()]

                    for idx, text in enumerate(texts):
                        if not text.strip():
                            continue
                        unit = EvidenceUnit(
                            unit_id=str(uuid.uuid4()),
                            source_path=str(fpath),
                            source_type="text",
                            video_subtype="null",
                            chapter_id=chapter_id,
                            span_start=float(idx),
                            span_end=float(idx),
                            raw_text=text,
                        )
                        units.append(unit)
                    logger.info(f"  TEXT   {fpath.name}  {len(texts)} chunks")
            except Exception as e:
                logger.warning(f"  LlamaIndex load failed: {e} — falling back to text_adapter")
                for fpath in plain_txt:
                    try:
                        parsed = extract_text(fpath)
                        for seg in parsed["segments"]:
                            if not seg.get("text", "").strip():
                                continue
                            unit = EvidenceUnit(
                                unit_id=str(uuid.uuid4()),
                                source_path=str(fpath),
                                source_type="text",
                                video_subtype="null",
                                chapter_id=chapter_id,
                                span_start=float(seg.get("start", seg.get("index", 0))),
                                span_end=float(seg.get("end", seg.get("index", 0))),
                                raw_text=seg["text"],
                            )
                            units.append(unit)
                        logger.info(f"  TEXT   {fpath.name}  {len(parsed['segments'])} segments (fallback)")
                    except Exception as e2:
                        logger.warning(f"  Text fallback failed {fpath.name}: {e2}")

    # ── Cross-material coverage log ────────────────────────────────────────
    source_types_seen = set(u.source_type for u in units)
    logger.info(f"  Source types in chapter: {sorted(source_types_seen)}")
    if len(source_types_seen) < 2:
        logger.warning(
            "  Only one source type found. Multi-material fusion may be limited. "
            "Expected: video + pdf + json for best results."
        )

    # ── Write output ────────────────────────────────────────────────────────
    tmp_file = out_file.with_suffix(".jsonl.tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        for unit in units:
            f.write(unit.model_dump_json() + "\n")
    tmp_file.rename(out_file)

    logger.info(f"Stage 0 done  | {len(units)} EvidenceUnits → {out_file}")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0 — Input scan and routing")
    parser.add_argument("--chapter", required=True, help="Path to chapter input folder")
    parser.add_argument("--input",   default=None,  help="Override INPUT_ROOT")
    parser.add_argument("--output",  default=None,  help="Override output dir")
    parser.add_argument("--data-dir", default=None, dest="data_dir", help="Override data dir")
    parser.add_argument("--force",   action="store_true")
    args = parser.parse_args()

    chapter_dir = Path(args.chapter)
    if not chapter_dir.is_absolute():
        root = Path(args.input or args.data_dir) if (args.input or args.data_dir) else INPUT_ROOT
        chapter_dir = root / chapter_dir

    out_dir = Path(args.output) if args.output else OUTPUTS_DIR / chapter_dir.name
    run(chapter_dir, out_dir, force=args.force)


if __name__ == "__main__":
    main()
