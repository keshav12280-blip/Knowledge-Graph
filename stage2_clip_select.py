"""
Stage 2 — Frame Selection
Keeps every visually distinct frame; near-duplicates are discarded.

Pipeline:
  Step 1 — Garbage removal (dark / blurry / pure talking head)
  Step 2 — OpenCV histogram similarity dedup
            Reject frame if corr-coeff similarity to ANY kept frame >= CV_SIMILARITY_THRESHOLD (0.98)
  Step 3 — CLIP embedding + KMeans + centroid-nearest (with text-density boost)
  Step 4 — Copy selected frames to selected_frames/{chapter_id}/
Output: outputs/{chapter_id}/stage2_selected_frames.jsonl
        selected_frames/{chapter_id}/  (copies of selected JPEGs)
"""

import argparse
import json
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    OUTPUTS_DIR, FRAMES_DIR, SELECTED_FRAMES_DIR, LOG_FORMAT, LOG_DATE_FORMAT,
    CLIP_MODEL_NAME, CLIP_PRETRAINED, CLIP_MODEL_DIR, CLIP_EMBED_BATCH_SIZE,
    CV_SIMILARITY_THRESHOLD,
    DARK_FRAME_MEAN_THRESHOLD,
    BLURRY_FRAME_LAPLACIAN_THRESHOLD, TALKING_HEAD_FACE_RATIO,
    TALKING_HEAD_TEXT_DENSITY_MIN, KMEANS_FRAMES_BY_SUBTYPE,
    CLIP_TARGET_FRAMES_MIN, CLIP_TARGET_FRAMES_MAX,
)


def _setup_logger(out_dir: Path) -> logging.Logger:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage2")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        fh = logging.FileHandler(log_dir / "stage2.log")
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


def _out_path(out_dir: Path) -> Path:
    return out_dir / "stage2_selected_frames.jsonl"


def _is_valid(out_dir: Path) -> bool:
    p = _out_path(out_dir)
    return p.exists() and p.stat().st_size > 0


def _load_frame_index(out_dir: Path) -> List[Dict[str, Any]]:
    p = out_dir / "stage1_frame_index.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"stage1_frame_index.jsonl not found at {p}")
    entries = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    return entries


# ── Text density scoring ──────────────────────────────────────────────────────
def _text_density_score(image_path: str) -> float:
    try:
        import cv2
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    except Exception:
        return 0.0


# ── Step 1: Garbage frame removal ────────────────────────────────────────────
def _remove_garbage(entries: List[Dict], logger: logging.Logger) -> List[Dict]:
    try:
        import cv2
    except ImportError:
        logger.warning("opencv not installed — skipping garbage removal")
        return entries

    kept = []
    face_cascade = None
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    except Exception:
        pass

    for entry in tqdm(entries, desc="Garbage removal", leave=False):
        fp = Path(entry["frame_path"])
        if not fp.exists():
            continue
        try:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            total_pixels = h * w

            if gray.mean() < DARK_FRAME_MEAN_THRESHOLD:
                continue

            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if lap_var < BLURRY_FRAME_LAPLACIAN_THRESHOLD:
                continue

            # Pure talking head: large face + low text density
            if face_cascade is not None and lap_var < TALKING_HEAD_TEXT_DENSITY_MIN:
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                if len(faces) > 0:
                    face_area = sum(fw * fh for (_, _, fw, fh) in faces)
                    if face_area / total_pixels > TALKING_HEAD_FACE_RATIO:
                        continue

            entry["text_density"] = round(lap_var, 2)
            kept.append(entry)
        except Exception:
            kept.append(entry)

    logger.info(f"  Garbage removal: {len(entries)} → {len(kept)}")
    return kept


# ── Step 2: OpenCV histogram similarity dedup ─────────────────────────────────
def _cv_histogram(img_bgr) -> np.ndarray:
    """Compute normalised 3-channel histogram for one BGR image."""
    import cv2
    hist = cv2.calcHist([img_bgr], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _cv_similarity_dedup(entries: List[Dict], logger: logging.Logger) -> List[Dict]:
    """
    Reject a frame if its histogram correlation-coefficient similarity to ANY already-kept
    frame is >= CV_SIMILARITY_THRESHOLD (default 0.98).
    Frames are processed in timestamp order so the first occurrence of a scene is kept.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("opencv not installed — skipping similarity dedup")
        return entries

    # Sort by timestamp to keep the earliest frame of each scene
    sorted_entries = sorted(entries, key=lambda e: e.get("timestamp_sec", 0))

    kept = []
    kept_hists: List[np.ndarray] = []

    for entry in tqdm(sorted_entries, desc="CV similarity dedup", leave=False):
        fp = Path(entry["frame_path"])
        if not fp.exists():
            continue
        try:
            img = cv2.imread(str(fp))
            if img is None:
                kept.append(entry)
                continue
            hist = _cv_histogram(img)
            # Compare against all retained frames
            is_duplicate = False
            for ref_hist in kept_hists:
                # cv2.HISTCMP_CORREL: 1.0 = identical, -1.0 = opposite
                sim = cv2.compareHist(
                    hist.reshape(-1, 1).astype(np.float32),
                    ref_hist.reshape(-1, 1).astype(np.float32),
                    cv2.HISTCMP_CORREL,
                )
                if sim >= CV_SIMILARITY_THRESHOLD:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(entry)
                kept_hists.append(hist)
        except Exception:
            kept.append(entry)

    logger.info(
        f"  CV similarity dedup (threshold={CV_SIMILARITY_THRESHOLD}): "
        f"{len(entries)} → {len(kept)} (removed {len(entries) - len(kept)} duplicates)"
    )
    return kept


# ── Step 3: CLIP embed → KMeans → select ─────────────────────────────────────
def _clip_select(entries: List[Dict], logger: logging.Logger) -> List[Dict]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 requires a GPU.")

    try:
        import open_clip
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as e:
        raise ImportError(f"Required library missing: {e}")

    device = "cuda"
    logger.info(f"  Loading CLIP {CLIP_MODEL_NAME} / {CLIP_PRETRAINED}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
        cache_dir=str(CLIP_MODEL_DIR),
    )
    model = model.to(device).eval()

    # Group by video
    video_groups: Dict[str, List[int]] = defaultdict(list)
    for i, entry in enumerate(entries):
        key = entry.get("video_path", entry["frame_path"])
        video_groups[key].append(i)

    # Embed all frames
    logger.info(f"  Embedding {len(entries)} frames  batch={CLIP_EMBED_BATCH_SIZE}")
    all_embeddings = np.zeros((len(entries), 512), dtype=np.float32)

    for batch_start in tqdm(range(0, len(entries), CLIP_EMBED_BATCH_SIZE), desc="CLIP embed", leave=False):
        batch = entries[batch_start: batch_start + CLIP_EMBED_BATCH_SIZE]
        images, valid_indices = [], []
        for j, entry in enumerate(batch):
            fp = Path(entry["frame_path"])
            if not fp.exists():
                continue
            try:
                img = preprocess(Image.open(fp).convert("RGB"))
                images.append(img)
                valid_indices.append(batch_start + j)
            except Exception:
                pass
        if not images:
            continue
        tensor = torch.stack(images).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            feats = model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embs = feats.cpu().float().numpy()
        for k, idx in enumerate(valid_indices):
            all_embeddings[idx] = embs[k]

    del model
    torch.cuda.empty_cache()
    logger.info("  CLIP model unloaded.")

    selected_entries = []
    for video_path, indices in video_groups.items():
        if not indices:
            continue
        video_entries = [entries[i] for i in indices]
        subtype = video_entries[0].get("subtype", "unknown")
        n_clusters = min(KMEANS_FRAMES_BY_SUBTYPE.get(subtype, 120), len(indices))

        vembs = all_embeddings[indices]

        if n_clusters <= 1:
            best = max(range(len(video_entries)), key=lambda i: video_entries[i].get("text_density", 0))
            entry = video_entries[best].copy()
            entry["cluster"] = 0
            selected_entries.append(entry)
            continue

        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        labels = km.fit_predict(vembs)
        centers = km.cluster_centers_

        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
            if not cluster_indices:
                continue
            cluster_embs = vembs[cluster_indices]
            center = centers[cluster_id]
            dists = np.linalg.norm(cluster_embs - center, axis=1)
            best_local = int(np.argmin(dists))

            td_scores = [video_entries[cluster_indices[i]].get("text_density", 0) for i in range(len(cluster_indices))]
            max_td = max(td_scores)
            best_td_local = int(np.argmax(td_scores))
            # Prefer high-text slide frame if clearly better than centroid pick
            if max_td > 60 and td_scores[best_local] < max_td * 0.5:
                chosen_local = best_td_local
            else:
                chosen_local = best_local

            chosen_global = cluster_indices[chosen_local]
            entry = video_entries[chosen_global].copy()
            entry["cluster"] = cluster_id
            entry["text_density"] = entry.get("text_density", 0.0)
            selected_entries.append(entry)

    total = len(selected_entries)
    logger.info(f"  CLIP selected: {total}")
    if total > CLIP_TARGET_FRAMES_MAX:
        selected_entries.sort(key=lambda e: e.get("text_density", 0), reverse=True)
        selected_entries = selected_entries[:CLIP_TARGET_FRAMES_MAX]
        logger.info(f"  Trimmed to cap {CLIP_TARGET_FRAMES_MAX}")
    elif total < CLIP_TARGET_FRAMES_MIN:
        logger.warning(f"  Only {total} frames selected (below target {CLIP_TARGET_FRAMES_MIN})")

    return selected_entries


# ── Step 4: Copy selected frames to selected_frames/{chapter_id}/ ────────────
def _copy_selected_frames(
    selected: List[Dict],
    chapter_name: str,
    logger: logging.Logger,
) -> List[Dict]:
    dest_dir = SELECTED_FRAMES_DIR / chapter_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    updated = []
    for entry in selected:
        src = Path(entry["frame_path"])
        if not src.exists():
            updated.append(entry)
            continue
        dst = dest_dir / src.name
        try:
            shutil.copy2(src, dst)
            entry = entry.copy()
            entry["selected_frame_path"] = str(dst)
        except Exception as exc:
            logger.warning(f"  Could not copy {src.name}: {exc}")
        updated.append(entry)

    logger.info(f"  Copied {len(updated)} selected frames → {dest_dir}")
    return updated


def run(chapter_dir: Path, out_dir: Path, force: bool = False) -> Path:
    logger = _setup_logger(out_dir)
    out_file = _out_path(out_dir)

    if _is_valid(out_dir) and not force:
        logger.info("Stage 2 output exists — skipping")
        return out_file

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Stage 2 requires a GPU. No CUDA device found.\n"
            "Check your CUDA installation."
        )

    logger.info(f"Stage 2 start | chapter={chapter_dir.name}")
    raw_entries = _load_frame_index(out_dir)
    logger.info(f"  Raw frames: {len(raw_entries)}")

    if not raw_entries:
        logger.warning("  No frames in index — writing empty output")
        out_file.write_text("")
        return out_file

    # Step 1: quality filter
    after_garbage = _remove_garbage(raw_entries, logger)

    # Step 2: OpenCV histogram similarity dedup (replaces pHash)
    after_dedup = _cv_similarity_dedup(after_garbage, logger)

    # Step 3: CLIP embed → KMeans → select representative frames
    selected = _clip_select(after_dedup, logger)

    # Step 4: Copy to selected_frames/ folder
    selected = _copy_selected_frames(selected, chapter_dir.name, logger)

    logger.info(f"  Final selected frames: {len(selected)}")

    tmp_file = out_file.with_suffix(".jsonl.tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        for entry in selected:
            f.write(json.dumps(entry) + "\n")
    tmp_file.rename(out_file)

    logger.info(f"Stage 2 done  | {len(selected)} frames → {out_file}")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 — Frame selection")
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
