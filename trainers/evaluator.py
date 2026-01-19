from __future__ import annotations

from typing import Any, Optional

import torch

from trainers.metrics import ConfusionMatrix


class Evaluator:
    """
    Shared evaluation loop for val/test.

    Trainer keeps orchestration + hooks; Evaluator implements the loop body.
    """

    def __init__(self, *, device: str | torch.device):
        self.device = device

    @staticmethod
    def _unpack_batch(batch):
        if isinstance(batch, dict):
            if "image" in batch:
                images = batch["image"]
            elif "img" in batch:
                images = batch["img"]
            else:
                images = list(batch.values())[0]

            if "mask" in batch:
                masks = batch["mask"]
            elif "label" in batch:
                masks = batch["label"]
            else:
                masks = list(batch.values())[1]
            return images, masks

        images, masks = batch
        return images, masks

    @staticmethod
    def _build_confmat(cfg: Any) -> Optional[ConfusionMatrix]:
        metrics_list = cfg.get("metrics") if cfg is not None else None
        if not metrics_list:
            return None

        model_cfg = cfg.get("model") or {}
        num_classes = None
        if isinstance(model_cfg, dict):
            num_classes = model_cfg.get("num_classes")
        else:
            num_classes = getattr(model_cfg, "num_classes", None)
        if num_classes is None:
            return None

        loss_cfg = cfg.get("loss") or {}
        if isinstance(loss_cfg, dict):
            ignore_index = loss_cfg.get("ignore_index")
        else:
            ignore_index = getattr(loss_cfg, "ignore_index", None)

        return ConfusionMatrix(num_classes=int(num_classes), ignore_index=ignore_index)

    @torch.no_grad()
    def run(
        self,
        *,
        model,
        loader,
        criterion,
        cfg: Any,
        mode: str = "val",
        epoch: Optional[int] = None,
        ctx: Optional[Any] = None,
        callbacks: Optional[list[Any]] = None,
    ) -> dict[str, float]:
        if loader is None:
            raise ValueError(f"Evaluator.run() requires a loader (mode={mode})")

        model.eval()
        expected_batches = len(loader)
        total_loss = 0.0
        seen_batches = 0
        confmat = self._build_confmat(cfg)

        callbacks = list(callbacks or [])
        for cb in callbacks:
            fn = getattr(cb, "on_start", None)
            if callable(fn):
                fn(mode=mode, epoch=epoch, ctx=ctx, loader=loader)

        for batch in loader:
            seen_batches += 1
            images, masks = self._unpack_batch(batch)
            images = images.to(self.device)
            masks = masks.to(self.device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += float(loss.item())

            if confmat is not None:
                confmat.update(outputs, masks)

            if callbacks:
                dataset = getattr(loader, "dataset", None)
                for cb in callbacks:
                    fn = getattr(cb, "on_batch", None)
                    if callable(fn):
                        fn(
                            batch=batch,
                            outputs=outputs,
                            masks=masks,
                            dataset=dataset,
                            mode=mode,
                            epoch=epoch,
                        )
                # Early stop is opt-in per callback (default off). This avoids
                # breaking full validation/test metrics when recorders finish.
                if any(
                    bool(getattr(cb, "request_stop", False)) and bool(getattr(cb, "done", False))
                    for cb in callbacks
                ):
                    break

        # If callbacks stop the loop early, use the number of seen batches to avoid
        # under-reporting the loss (dividing by len(loader)).
        denom = max(1, seen_batches if seen_batches > 0 else expected_batches)
        avg_loss = total_loss / denom
        metrics = {f"{mode}_loss" if mode in {"val", "test"} else "loss": float(avg_loss)}
        if confmat is not None:
            try:
                metrics_list = cfg.get("metrics")
                metrics.update(confmat.compute(metrics_list))
            except Exception as exc:
                print(f"[Evaluator] Metric compute skipped: {exc}")

        for cb in callbacks:
            fn = getattr(cb, "on_end", None)
            if callable(fn):
                fn(metrics=metrics, mode=mode, epoch=epoch, ctx=ctx, loader=loader)
        return metrics
