from __future__ import annotations

import json
import io
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from .base_hook import Hook
from .registry import register_hook


def _safe_name(name: str) -> str:
    name = (name or "unknown").strip().replace("\\", "/")
    return name.split("/")[-1]


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _decode_hw(payload: Optional[bytes]) -> Optional[tuple[int, int]]:
    if not payload:
        return None
    try:
        from PIL import Image  # type: ignore

        with Image.open(io.BytesIO(payload)) as im:
            w, h = im.size
        return int(h), int(w)
    except Exception:
        pass

    try:
        import cv2  # type: ignore

        arr = np.frombuffer(payload, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        return int(h), int(w)
    except Exception:
        return None


def _logits_to_mask(logits: torch.Tensor, *, threshold: float) -> torch.Tensor:
    if logits.ndim == 3:
        # [C, H, W] or [H, W]
        if logits.dtype.is_floating_point:
            prob = torch.sigmoid(logits)
            return (prob > float(threshold)).to(torch.uint8)
        return logits.to(torch.uint8)

    if logits.ndim == 4:
        # [B, C, H, W]
        c = int(logits.shape[1])
        if c == 1:
            prob = torch.sigmoid(logits[:, 0])
            return (prob > float(threshold)).to(torch.uint8)
        return torch.argmax(logits, dim=1).to(torch.uint8)

    raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")


def _resize_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.shape[:2] == (h, w):
        return mask
    try:
        from PIL import Image  # type: ignore

        pil = Image.fromarray(mask, mode="L")
        pil = pil.resize((int(w), int(h)), resample=Image.NEAREST)
        return np.asarray(pil, dtype=np.uint8)
    except Exception:
        pass

    try:
        import cv2  # type: ignore

        return cv2.resize(mask, (int(w), int(h)), interpolation=cv2.INTER_NEAREST)
    except Exception:
        # Fallback: return original mask even if size mismatches.
        return mask


def _write_mask_png(path: Path, mask01: np.ndarray) -> bool:
    mask01 = np.asarray(mask01, dtype=np.uint8)
    img = (mask01 > 0).astype(np.uint8) * 255
    try:
        from PIL import Image  # type: ignore

        Image.fromarray(img, mode="L").save(path)
        return True
    except Exception:
        pass

    try:
        import cv2  # type: ignore

        return bool(cv2.imwrite(str(path), img))
    except Exception:
        return False


def _extract_img_ref(meta: dict) -> Optional[str]:
    for key in ("img_file", "img_path", "path"):
        value = meta.get(key)
        if value:
            return str(value)
    return None


def _normalize_metas(metas: Any, batch_size: int) -> list[dict]:
    if isinstance(metas, list):
        # Preserve ordering to keep alignment with `outputs[i]`.
        return [m if isinstance(m, dict) else {} for m in metas]

    # torch default_collate may produce a dict-of-lists for "meta".
    if isinstance(metas, dict):
        out: list[dict] = []
        for i in range(max(0, int(batch_size))):
            item: dict = {}
            for k, v in metas.items():
                if isinstance(v, (list, tuple)) and len(v) == batch_size:
                    item[k] = v[i]
                else:
                    item[k] = v
            out.append(item)
        return out

    return []


class RawRecordCallback:
    def __init__(
        self,
        *,
        base_dir: Path,
        split: str,
        epoch: int,
        num_samples: int,
        save_pred_mask: bool,
        save_gt_mask: bool,
        threshold: float,
    ):
        self.base_dir = Path(base_dir)
        self.split = str(split).lower()
        self.epoch = int(epoch)
        self.num_samples = int(num_samples)
        self.save_pred_mask = bool(save_pred_mask)
        self.save_gt_mask = bool(save_gt_mask)
        self.threshold = float(threshold)

        self.out_epoch_dir = self.base_dir / self.split / f"epoch_{self.epoch}"
        self.out_epoch_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.base_dir / "manifest.jsonl"

        self.saved = 0
        self.done = False

    def on_batch(self, *, batch, outputs, masks, dataset, mode: str, epoch: Optional[int] = None, **_kwargs):
        if self.done or self.saved >= self.num_samples:
            self.done = True
            return

        if not isinstance(batch, dict) or "meta" not in batch:
            return
        metas = _normalize_metas(batch["meta"], int(outputs.shape[0]) if torch.is_tensor(outputs) and outputs.ndim >= 1 else 0)
        if not metas:
            return

        # outputs: [B, C, H, W], masks: [B, H, W]
        for i, meta in enumerate(metas):
            if self.saved >= self.num_samples:
                self.done = True
                return
            if not isinstance(meta, dict) or not torch.is_tensor(outputs) or i >= int(outputs.shape[0]):
                continue

            src_ref, payload = self._read_raw_bytes(dataset, meta)
            raw_hw = _decode_hw(payload)
            if payload is None or raw_hw is None:
                continue
            raw_h, raw_w = raw_hw

            img_ref = _extract_img_ref(meta)
            fname = _safe_name(str(img_ref or f"sample_{self.saved}.png"))
            out_img_path = self.out_epoch_dir / fname
            if out_img_path.exists():
                out_img_path = self.out_epoch_dir / f"{out_img_path.stem}_{self.saved}{out_img_path.suffix}"
            out_img_path.write_bytes(payload)

            out_pred_path = None
            out_gt_path = None
            pred_nz = None
            gt_nz = None

            if self.save_pred_mask and torch.is_tensor(outputs):
                sample_logits = outputs[i : i + 1]
                pred = _logits_to_mask(sample_logits, threshold=self.threshold)
                pred_np = pred.squeeze(0).detach().to("cpu").numpy().astype(np.uint8)
                pred_np = _resize_mask(pred_np, raw_h, raw_w)
                out_pred_path = self.out_epoch_dir / f"{out_img_path.stem}_pred.png"
                if not _write_mask_png(out_pred_path, pred_np):
                    out_pred_path = None
                else:
                    pred_nz = int((pred_np > 0).sum())

            if self.save_gt_mask and torch.is_tensor(masks):
                gt = masks[i]
                gt_np = gt.detach().to("cpu").numpy().astype(np.uint8)
                gt_np = (gt_np > 0).astype(np.uint8)
                gt_np = _resize_mask(gt_np, raw_h, raw_w)
                out_gt_path = self.out_epoch_dir / f"{out_img_path.stem}_gt.png"
                if not _write_mask_png(out_gt_path, gt_np):
                    out_gt_path = None
                else:
                    gt_nz = int((gt_np > 0).sum())

            _append_jsonl(
                self.manifest_path,
                {
                    "split": self.split,
                    "epoch": self.epoch,
                    "src": src_ref,
                    "dst": str(out_img_path),
                    "pred": str(out_pred_path) if out_pred_path else None,
                    "gt": str(out_gt_path) if out_gt_path else None,
                    "threshold": self.threshold,
                    "pred_nz": pred_nz,
                    "gt_nz": gt_nz,
                    "raw_h": raw_h,
                    "raw_w": raw_w,
                },
            )
            self.saved += 1

    @staticmethod
    def _read_raw_bytes(dataset: Any, meta: dict) -> tuple[str, Optional[bytes]]:
        img_ref = _extract_img_ref(meta)
        if not img_ref:
            return "unknown", None

        # Prefer a public dataset API if available.
        reader = getattr(dataset, "get_raw_image_bytes", None)
        if callable(reader):
            try:
                payload = reader(str(img_ref))
                if not payload:
                    return str(img_ref), None
                ref_fn = getattr(dataset, "get_raw_image_ref", None)
                resolved = ref_fn(str(img_ref)) if callable(ref_fn) else None
                src = str(resolved) if resolved else str(img_ref)
                return src, payload
            except Exception:
                return str(img_ref), None

        img_dir = getattr(dataset, "img_dir", None)
        if img_dir:
            path = Path(str(img_dir)) / str(img_ref)
            if path.exists():
                return str(path), path.read_bytes()

        if hasattr(dataset, "_resolve_img_key") and hasattr(dataset, "_read_bytes"):
            try:
                key = dataset._resolve_img_key(str(img_ref))
                payload = dataset._read_bytes(key)
                bucket = getattr(dataset, "bucket", None)
                src = f"s3://{bucket}/{key}" if bucket else key
                return src, payload
            except Exception:
                return f"s3:{img_ref}", None

        return str(img_ref), None


@register_hook("RecordRawHook")
class RecordRawHook(Hook):
    """
    Record raw images + optional pred/gt masks via Evaluator callbacks.

    Hook does NOT iterate or forward; it only installs a callback before eval/test.
    """

    priority = 40

    def __init__(
        self,
        *,
        enabled: bool = False,
        splits: tuple[str, ...] = ("val", "test"),
        interval_epochs: int = 1,
        num_samples: int = 16,
        out_dir: str = "records/raw",
        save_pred_mask: bool = True,
        save_gt_mask: bool = True,
        threshold: float = 0.5,
    ):
        self.enabled = bool(enabled)
        self.splits = tuple(str(s).lower() for s in splits)
        self.interval_epochs = max(1, int(interval_epochs))
        self.num_samples = max(0, int(num_samples))
        self.out_dir = str(out_dir).strip().strip("/")
        self.save_pred_mask = bool(save_pred_mask)
        self.save_gt_mask = bool(save_gt_mask)
        self.threshold = float(threshold)

    def before_val(self, trainer):
        self._install(trainer, split="val")

    def before_test(self, trainer):
        self._install(trainer, split="test")

    def _install(self, trainer, *, split: str) -> None:
        if not self.enabled or self.num_samples <= 0:
            return
        split = str(split).lower()
        if split not in self.splits:
            return

        epoch = int(getattr(trainer, "epoch", 0)) + 1
        if split == "val" and (epoch % self.interval_epochs) != 0:
            return

        ctx = getattr(trainer, "ctx", None)
        run_dir = getattr(ctx, "run_dir", None) if ctx is not None else None
        base_dir = (Path(run_dir) if run_dir else Path("logs")) / self.out_dir

        cb = RawRecordCallback(
            base_dir=base_dir,
            split=split,
            epoch=epoch,
            num_samples=self.num_samples,
            save_pred_mask=self.save_pred_mask,
            save_gt_mask=self.save_gt_mask,
            threshold=self.threshold,
        )
        add_cb = getattr(trainer, "add_evaluator_callback", None)
        if callable(add_cb):
            add_cb(cb)
        else:
            callbacks = getattr(trainer, "_evaluator_callbacks", None)
            if callbacks is None:
                trainer._evaluator_callbacks = []
                callbacks = trainer._evaluator_callbacks
            callbacks.append(cb)
