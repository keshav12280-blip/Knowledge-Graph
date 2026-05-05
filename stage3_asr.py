"""
Stage 3 — ASR Transcription
Transcribe all videos using openai-whisper (medium, English).
Word-level timestamps enabled for transcript-frame alignment in Stage 5.
Existing .txt/.srt transcripts in chapter folder are used directly.
Output: outputs/{chapter_id}/stage3_transcripts.jsonl  (updates EvidenceUnits)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    OUTPUTS_DIR, LOG_FORMAT, LOG_DATE_FORMAT,
    WHISPER_MODEL_NAME, WHISPER_MODEL_DIR,
    EvidenceUnit,
)

WHISPER_LANGUAGE = "en"


def _setup_logger(out_dir: Path) -> logging.Logger:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage3")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        fh = logging.FileHandler(log_dir / "stage3.log")
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


def _out_path(out_dir: Path) -> Path:
    return out_dir / "stage3_transcripts.jsonl"


def _is_valid(out_dir: Path) -> bool:
    p = _out_path(out_dir)
    return p.exists() and p.stat().st_size > 0


def _load_units(out_dir: Path) -> List[EvidenceUnit]:
    p = out_dir / "stage0_units.jsonl"
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


def _find_existing_transcript(video_path: Path) -> Optional[Path]:
    """Look for a .txt or .srt file with the same stem in the same folder."""
    parent = video_path.parent
    for ext in (".srt", ".vtt", ".txt"):
        candidate = parent / (video_path.stem + ext)
        if candidate.exists():
            return candidate
    return None


def _load_whisper(logger: logging.Logger):
    """Load openai-whisper model on GPU."""
    try:
        import whisper
    except ImportError:
        raise ImportError("openai-whisper not installed. pip install openai-whisper")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Stage 3 requires a GPU. No CUDA device found.\n"
            "Check your CUDA installation."
        )

    logger.info(f"  Loading openai-whisper '{WHISPER_MODEL_NAME}' → {WHISPER_MODEL_DIR}")
    model = whisper.load_model(
        WHISPER_MODEL_NAME,
        device="cuda",
        download_root=str(WHISPER_MODEL_DIR),
    )
    return model


def _transcribe(video_path: Path, model, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Transcribe a single video with openai-whisper.
    Returns list of segment dicts: {start, end, text, words}.
    """
    logger.info(f"  Transcribing {video_path.name}")
    result = model.transcribe(
        str(video_path),
        language=WHISPER_LANGUAGE,
        word_timestamps=True,
        verbose=False,
    )

    segments_data: List[Dict[str, Any]] = []
    for seg in result["segments"]:
        words = []
        for w in seg.get("words", []):
            words.append({
                "word":  w["word"],
                "start": round(w["start"], 3),
                "end":   round(w["end"],   3),
            })
        segments_data.append({
            "start": round(seg["start"], 3),
            "end":   round(seg["end"],   3),
            "text":  seg["text"].strip(),
            "words": words,
        })

    logger.info(f"  {video_path.name} → {len(segments_data)} segments")
    return segments_data


def run(chapter_dir: Path, out_dir: Path, force: bool = False) -> Path:
    logger = _setup_logger(out_dir)
    out_file = _out_path(out_dir)

    if _is_valid(out_dir) and not force:
        logger.info("Stage 3 output exists — skipping")
        return out_file

    logger.info(f"Stage 3 start | chapter={chapter_dir.name}")
    units = _load_units(out_dir)
    video_units = [u for u in units if u.source_type == "video"]
    logger.info(f"  {len(video_units)} video units")

    path_to_units: Dict[str, List[EvidenceUnit]] = {}
    for u in video_units:
        path_to_units.setdefault(u.source_path, []).append(u)

    non_video = [u for u in units if u.source_type != "video"]
    updated_units: List[EvidenceUnit] = []

    whisper_model = None

    for video_path_str, unit_list in path_to_units.items():
        video_path = Path(video_path_str)

        existing = _find_existing_transcript(video_path)
        if existing:
            logger.info(f"  Using existing transcript: {existing.name}")
            from ingest_adapters.text_adapter import extract_text
            parsed = extract_text(existing)
            segs = parsed["segments"]
            segments_data = [
                {
                    "start": float(s.get("start", s.get("index", 0))),
                    "end":   float(s.get("end",   s.get("index", 0))),
                    "text":  s.get("text", ""),
                    "words": [],
                }
                for s in segs
            ]
            full_text = " ".join(s["text"] for s in segments_data)
        else:
            try:
                if whisper_model is None:
                    whisper_model = _load_whisper(logger)
                segments_data = _transcribe(video_path, whisper_model, logger)
                full_text = " ".join(s["text"] for s in segments_data)
            except Exception as e:
                logger.warning(f"  Transcription failed {video_path.name}: {e}")
                segments_data = []
                full_text = ""

        for u in unit_list:
            if u.span_start == 0.0 and u.span_end == 0.0:
                span_segs = segments_data
            else:
                span_segs = [
                    s for s in segments_data
                    if s["end"] >= u.span_start and s["start"] <= u.span_end
                ]
            span_text = " ".join(s["text"] for s in span_segs)
            updated_units.append(u.model_copy(update={"raw_text": span_text or full_text}))

    if whisper_model is not None:
        del whisper_model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("  Model unloaded from VRAM.")

    all_units = non_video + updated_units

    tmp_file = out_file.with_suffix(".jsonl.tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        for unit in all_units:
            f.write(unit.model_dump_json() + "\n")
    tmp_file.rename(out_file)

    logger.info(f"Stage 3 done  | {len(all_units)} units (incl. non-video) → {out_file}")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 — ASR transcription")
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
