"""
Stage 1 — Frame Extraction
Extract frames from all videos in EvidenceUnits at subtype-specific FPS.
Uses ThreadPoolExecutor for parallel per-video extraction.
Output: frames/{chapter_id}/ + outputs/{chapter_id}/stage1_frame_index.jsonl
"""

import argparse
import json
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    OUTPUTS_DIR, FRAMES_DIR, LOG_FORMAT, LOG_DATE_FORMAT,
    FRAME_RATE_BY_SUBTYPE, FRAME_WIDTH_PX, FRAME_JPEG_QUALITY,
    EvidenceUnit,
)


def _setup_logger(out_dir: Path) -> logging.Logger:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage1")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        fh = logging.FileHandler(log_dir / "stage1.log")
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


def _out_path(out_dir: Path) -> Path:
    return out_dir / "stage1_frame_index.jsonl"


def _is_valid(out_dir: Path) -> bool:
    p = _out_path(out_dir)
    return p.exists() and p.stat().st_size > 0


def _load_units(out_dir: Path) -> List[EvidenceUnit]:
    units_file = out_dir / "stage0_units.jsonl"
    if not units_file.exists():
        raise FileNotFoundError(f"stage0_units.jsonl not found at {units_file}")
    units = []
    with open(units_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    units.append(EvidenceUnit.model_validate_json(line))
                except Exception:
                    pass
    return units


def _extract_video_frames(
    video_path: Path,
    frames_dir: Path,
    subtype: str,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """
    Extract frames for one video via FFmpeg subprocess.
    Returns list of {frame_path, video_path, timestamp_sec, subtype}.
    """
    fps = FRAME_RATE_BY_SUBTYPE.get(subtype, 1.0)
    vid_frames_dir = frames_dir / video_path.stem
    vid_frames_dir.mkdir(parents=True, exist_ok=True)

    # FFmpeg: extract at target FPS, resize width to FRAME_WIDTH_PX
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps},scale={FRAME_WIDTH_PX}:-2",
        "-q:v", str(int(100 - FRAME_JPEG_QUALITY)),  # ffmpeg quality scale inverted
        "-f", "image2",
        str(vid_frames_dir / "frame_%06d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        logger.warning(f"  FFmpeg error {video_path.name}: {result.stderr[-300:]}")
        return []

    frames = sorted(vid_frames_dir.glob("frame_*.jpg"))
    index = []
    for i, fp in enumerate(frames):
        timestamp = i / fps  # approximate; good enough for clustering
        index.append({
            "frame_path":    str(fp),
            "video_path":    str(video_path),
            "timestamp_sec": round(timestamp, 2),
            "subtype":       subtype,
            "frame_num":     i,
        })

    logger.info(
        f"  {video_path.name}  fps={fps}  → {len(frames)} frames  → {vid_frames_dir}"
    )
    return index


def run(chapter_dir: Path, out_dir: Path, force: bool = False) -> Path:
    logger = _setup_logger(out_dir)
    out_file = _out_path(out_dir)

    if _is_valid(out_dir) and not force:
        logger.info("Stage 1 output exists — skipping")
        return out_file

    logger.info(f"Stage 1 start | chapter={chapter_dir.name}")
    units = _load_units(out_dir)
    video_units = [u for u in units if u.source_type == "video"]
    logger.info(f"  {len(video_units)} video units to extract")

    if not video_units:
        logger.warning("  No video units found — writing empty index")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text("")
        return out_file

    frames_dir = FRAMES_DIR / chapter_dir.name
    frames_dir.mkdir(parents=True, exist_ok=True)

    all_entries: List[Dict[str, Any]] = []
    # Parallel extraction — CPU-bound I/O, safe to parallelize
    max_workers = min(4, len(video_units))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _extract_video_frames,
                Path(u.source_path),
                frames_dir,
                u.video_subtype,
                logger,
            ): u
            for u in video_units
        }
        for future in as_completed(futures):
            unit = futures[future]
            try:
                entries = future.result()
                # Attach unit_id to each frame entry for downstream linking
                for e in entries:
                    e["unit_id"] = unit.unit_id
                all_entries.extend(entries)
            except Exception as exc:
                logger.warning(f"  Frame extraction failed {unit.source_path}: {exc}")

    logger.info(f"  Total raw frames extracted: {len(all_entries)}")

    tmp_file = out_file.with_suffix(".jsonl.tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")
    tmp_file.rename(out_file)

    logger.info(f"Stage 1 done  | {len(all_entries)} frames → {out_file}")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 — Frame extraction")
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
