import os.path
import os
from typing import Optional, Dict, Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint


def _get_env_positive_int(var_name: str) -> Optional[int]:
    raw_value = os.environ.get(var_name)
    if raw_value is None or raw_value == "":
        return None

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{var_name} must be an integer, got: {raw_value}") from exc

    if value < 1:
        raise ValueError(f"{var_name} must be >= 1, got: {value}")
    return value


class CheckpointHook(ModelCheckpoint):
    """Save checkpoint with only the incremental part of the model"""
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        ckpt_dir = os.environ.get("PIXNERD_CKPT_DIR") or self.dirpath or trainer.default_root_dir
        self.dirpath = ckpt_dir
        save_every_n_train_steps = _get_env_positive_int("PIXNERD_SAVE_EVERY_N_TRAIN_STEPS")
        if save_every_n_train_steps is not None:
            self._every_n_train_steps = save_every_n_train_steps
        self.exception_ckpt_path = os.path.join(self.dirpath, "on_exception.pt")
        pl_module.strict_loading = False

    def on_save_checkpoint(
            self, trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            checkpoint: Dict[str, Any]
    ) -> None:
        del checkpoint["callbacks"]

    # def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
    #     if not "debug" in self.exception_ckpt_path:
    #         trainer.save_checkpoint(self.exception_ckpt_path)
