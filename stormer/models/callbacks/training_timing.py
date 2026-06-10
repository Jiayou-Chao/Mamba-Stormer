"""Callback to log epoch timing and throughput metrics."""
import time
from typing import Any

import torch
from lightning.pytorch.callbacks import Callback


class TrainingTiming(Callback):
    """Log epoch wall time and throughput to configured loggers."""

    def __init__(self) -> None:
        self._epoch_start_time = None
        self._epoch_samples = 0
        self._last_batch_size = None

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._epoch_start_time = time.time()
        self._epoch_samples = 0
        self._last_batch_size = None

    def on_train_batch_end(self, trainer, pl_module, outputs: Any, batch: Any, batch_idx: int) -> None:
        batch_size = self._extract_batch_size(batch)
        if batch_size is None:
            return
        self._epoch_samples += batch_size
        self._last_batch_size = batch_size

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self._epoch_start_time is None:
            return

        elapsed = time.time() - self._epoch_start_time
        world_size = trainer.world_size if trainer.world_size is not None else 1
        global_samples = self._epoch_samples * world_size
        throughput = (global_samples / elapsed) if elapsed > 0 else 0.0
        global_batch_size = (
            self._last_batch_size * world_size
            if self._last_batch_size is not None
            else None
        )

        if trainer.is_global_zero:
            metrics = {
                "train/epoch_time_sec": float(elapsed),
                "train/epoch_samples": int(global_samples),
                "train/world_size": int(world_size),
                "train/throughput_samples_per_sec": float(throughput),
            }
            if global_batch_size is not None:
                metrics["train/global_batch_size"] = int(global_batch_size)
            self._log_metrics(trainer, metrics)

    @staticmethod
    def _extract_batch_size(batch: Any):
        if torch.is_tensor(batch):
            return batch.shape[0]
        if isinstance(batch, (list, tuple)) and batch:
            first = batch[0]
            if torch.is_tensor(first):
                return first.shape[0]
        return None

    @staticmethod
    def _log_metrics(trainer, metrics) -> None:
        if trainer.logger is None:
            return
        if isinstance(trainer.logger, list):
            for logger in trainer.logger:
                logger.log_metrics(metrics, step=trainer.global_step)
                if hasattr(logger, "flush"):
                    logger.flush()
        else:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)
            if hasattr(trainer.logger, "flush"):
                trainer.logger.flush()
