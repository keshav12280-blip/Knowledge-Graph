"""
Stage 4 — VLM Captioning
Caption selected keyframes + PDF image pages using gemma-4-E4B-it via vLLM.
Plain text output only — never asks VLM for JSON.
PDF visual pages use images pre-rendered by Stage 0 (stored in unit.keyframe_paths).
Output: outputs/{chapter_id}/stage4_captions.jsonl  (updates EvidenceUnits)
"""

import argparse
import base64
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    OUTPUTS_DIR, LOG_FORMAT, LOG_DATE_FORMAT,
    VLM_MODEL_DIR, VLM_MODEL_ID,
    VLM_BATCH_SIZE, VLM_MAX_NEW_TOKENS, VLM_TEMPERATURE,
    VLM_GPU_MEMORY_UTILIZATION, VLM_MAX_MODEL_LEN, VLM_PROMPT,
    EvidenceUnit,
)


def _setup_logger(out_dir: Path) -> logging.Logger:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage4")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        fh = logging.FileHandler(log_dir / "stage4.log")
        fh.setFormatter(fmt)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


def _out_path(out_dir: Path) -> Path:
    return out_dir / "stage4_captions.jsonl"


def _is_valid(out_dir: Path) -> bool:
    p = _out_path(out_dir)
    return p.exists() and p.stat().st_size > 0


def _load_units(out_dir: Path) -> List[EvidenceUnit]:
    # Stage 4 reads from stage 3 output (which has ASR-updated raw_text)
    p = out_dir / "stage3_transcripts.jsonl"
    if not p.exists():
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


def _load_selected_frames(out_dir: Path) -> Dict[str, List[str]]:
    """Returns unit_id → list of frame paths from stage2 output."""
    p = out_dir / "stage2_selected_frames.jsonl"
    if not p.exists():
        return {}
    unit_to_frames: Dict[str, List[str]] = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                uid = entry.get("unit_id", "")
                if uid:
                    # Prefer the copy in selected_frames/ if it exists
                    fp = entry.get("selected_frame_path") or entry["frame_path"]
                    unit_to_frames.setdefault(uid, []).append(fp)
            except Exception:
                pass
    return unit_to_frames


def _img_to_b64(path: Path) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def _pdf_page_to_image(pdf_path: Path, page_num: int) -> Optional[bytes]:
    """Render a PDF page to PNG bytes for VLM captioning."""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        page = doc[int(page_num)]
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        doc.close()
        return img_bytes
    except Exception:
        return None


class VLMClient:
    """Thin wrapper around vLLM offline LLM for vision inference."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._llm = None

    def load(self) -> None:
        import os
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Stage 4 requires a GPU. No CUDA device found.\n"
                "Check your CUDA installation."
            )
        # vLLM v1 EngineCore forks a subprocess for the engine worker.
        # If CUDA was already initialized in the parent (e.g. by Stage 3),
        # the fork fails with "Cannot re-initialize CUDA in forked subprocess".
        # spawn creates a fresh process instead of forking, avoiding this.
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vllm not installed. pip install vllm")

        model_path = str(VLM_MODEL_DIR) if VLM_MODEL_DIR.exists() else VLM_MODEL_ID
        self.logger.info(f"  Loading VLM: {model_path}")
        self._llm = LLM(
            model=model_path,
            max_model_len=VLM_MAX_MODEL_LEN,
            gpu_memory_utilization=VLM_GPU_MEMORY_UTILIZATION,
            max_num_seqs=VLM_BATCH_SIZE,
            trust_remote_code=True,
            dtype="auto",
            limit_mm_per_prompt={"image": 1},
        )
        self.logger.info("  VLM loaded")

    def caption_batch(self, image_b64_list: List[str]) -> List[str]:
        """Caption a batch of base64-encoded images. Returns plain text list."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        sampling = SamplingParams(
            temperature=VLM_TEMPERATURE,
            max_tokens=VLM_MAX_NEW_TOKENS,
        )

        # Build multimodal inputs for vLLM — gemma-4 chat template
        # gemma-4 uses: <|turn>user\n<|image|>text<turn|>\n<|turn>model\n
        inputs = []
        for b64 in image_b64_list:
            prompt = f"<|turn>user\n<|image|>{VLM_PROMPT}<turn|>\n<|turn>model\n"
            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {
                    "image": self._b64_to_pil(b64),
                },
            })

        outputs = self._llm.generate(inputs, sampling)
        return [o.outputs[0].text.strip() for o in outputs]

    def _b64_to_pil(self, b64: str):
        import io
        from PIL import Image
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    def unload(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        self.logger.info("  Model unloaded from VRAM.")


def _caption_items(
    items: List[Dict[str, Any]],  # each: {unit_id, source, img_b64, page_num}
    client: VLMClient,
    logger: logging.Logger,
) -> Dict[str, List[str]]:
    """Caption all items in batches. Returns unit_id → list of captions (one per frame)."""
    from collections import defaultdict
    captions: Dict[str, List[str]] = defaultdict(list)
    total = len(items)
    for i in range(0, total, VLM_BATCH_SIZE):
        batch = items[i: i + VLM_BATCH_SIZE]
        b64s = [x["img_b64"] for x in batch]
        logger.info(f"  VLM batch {i//VLM_BATCH_SIZE + 1}/{(total+VLM_BATCH_SIZE-1)//VLM_BATCH_SIZE}  ({len(batch)} items)")
        try:
            results = client.caption_batch(b64s)
            for item, caption in zip(batch, results):
                if caption.strip():
                    captions[item["unit_id"]].append(caption.strip())
        except Exception as e:
            logger.warning(f"  VLM batch failed: {e}")
    return dict(captions)


def run(chapter_dir: Path, out_dir: Path, force: bool = False) -> Path:
    logger = _setup_logger(out_dir)
    out_file = _out_path(out_dir)

    if _is_valid(out_dir) and not force:
        logger.info("Stage 4 output exists — skipping")
        return out_file

    logger.info(f"Stage 4 start | chapter={chapter_dir.name}")
    units = _load_units(out_dir)
    unit_to_frames = _load_selected_frames(out_dir)

    # Build caption queue
    caption_items: List[Dict[str, Any]] = []

    # 1) Selected video frames
    for unit in units:
        if unit.source_type != "video":
            continue
        frames = unit_to_frames.get(unit.unit_id, [])
        if not frames:
            continue
        for fp in frames:  # caption every selected frame
            fp_path = Path(fp)
            if not fp_path.exists():
                logger.warning(f"  Missing frame: {fp}")
                continue
            b64 = _img_to_b64(fp_path)
            if b64:
                caption_items.append({
                    "unit_id":  unit.unit_id,
                    "source":   fp,
                    "img_b64":  b64,
                    "page_num": None,
                })

    # 2) PDF image pages — use pre-rendered images from Stage 0 (stored in keyframe_paths)
    for unit in units:
        if unit.source_type != "pdf" or unit.raw_text:
            continue  # skip text-rich PDF pages
        # Stage 0 pre-renders visual PDF pages and stores the path in keyframe_paths
        pdf_images = unit.keyframe_paths
        if pdf_images:
            # Use pre-rendered image from Stage 0
            for img_path_str in pdf_images[:1]:  # one representative image per page
                img_path = Path(img_path_str)
                if img_path.exists():
                    b64 = _img_to_b64(img_path)
                    if b64:
                        caption_items.append({
                            "unit_id":  unit.unit_id,
                            "source":   img_path_str,
                            "img_b64":  b64,
                            "page_num": int(unit.span_start),
                        })
                else:
                    logger.warning(f"  Pre-rendered PDF image missing: {img_path_str} — re-rendering")
                    pdf_path = Path(unit.source_path)
                    page_num = int(unit.span_start)
                    img_bytes = _pdf_page_to_image(pdf_path, page_num)
                    if img_bytes:
                        b64 = base64.b64encode(img_bytes).decode("utf-8")
                        caption_items.append({
                            "unit_id":  unit.unit_id,
                            "source":   f"{pdf_path}:page{page_num}",
                            "img_b64":  b64,
                            "page_num": page_num,
                        })
        else:
            # Fallback: render at runtime (pre-rendering missed this page)
            pdf_path = Path(unit.source_path)
            page_num = int(unit.span_start)
            img_bytes = _pdf_page_to_image(pdf_path, page_num)
            if img_bytes:
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                caption_items.append({
                    "unit_id":  unit.unit_id,
                    "source":   f"{pdf_path}:page{page_num}",
                    "img_b64":  b64,
                    "page_num": page_num,
                })

    # 3) Standalone images
    for unit in units:
        if unit.source_type != "image":
            continue
        b64 = _img_to_b64(Path(unit.source_path))
        if b64:
            caption_items.append({
                "unit_id":  unit.unit_id,
                "source":   unit.source_path,
                "img_b64":  b64,
                "page_num": None,
            })

    logger.info(f"  VLM queue: {len(caption_items)} items")

    if not caption_items:
        logger.info("  No visual items to caption — copying stage3 output")
        import shutil
        shutil.copy(out_dir / "stage3_transcripts.jsonl", out_file)
        return out_file

    # Run VLM
    client = VLMClient(logger)
    client.load()
    unit_to_caption = _caption_items(caption_items, client, logger)
    client.unload()

    # Build unit_id → list of frame paths
    unit_to_all_frames: Dict[str, List[str]] = {}
    for unit in units:
        if unit.source_type == "video":
            unit_to_all_frames[unit.unit_id] = unit_to_frames.get(unit.unit_id, [])

    # Update units with captions and keyframe_paths
    updated_units = []
    for unit in units:
        caption_list = unit_to_caption.get(unit.unit_id, [])
        # Join all per-frame captions into one string separated by newlines
        caption = "\n".join(caption_list) if caption_list else unit.visual_caption
        kf_paths = unit_to_all_frames.get(unit.unit_id, unit.keyframe_paths)
        if caption or kf_paths:
            updated = unit.model_copy(update={
                "visual_caption": caption,
                "keyframe_paths": kf_paths or unit.keyframe_paths,
            })
            updated_units.append(updated)
        else:
            updated_units.append(unit)

    tmp_file = out_file.with_suffix(".jsonl.tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        for unit in updated_units:
            f.write(unit.model_dump_json() + "\n")
    tmp_file.rename(out_file)

    captioned_count = sum(1 for u in updated_units if u.visual_caption)
    logger.info(f"Stage 4 done  | {captioned_count} units with captions → {out_file}")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 — VLM captioning")
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
