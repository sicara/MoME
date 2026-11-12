"""MoME+ fine-tuning trainer selecting checkpoints on the tumor core labels.

Subclass of nnUNetTrainerFT that changes:
- checkpoint selection and early stopping: EMA of the mean pseudo Dice of NCR and ET
  (labels 1 and 3), instead of the mean over all foreground labels
- no best checkpoint before SELECTION_WARMUP_EPOCHS, so the EMA is not anchored on epoch 0
- foreground oversampling raised to favour small lesions
- latest checkpoint saved every 2 epochs
- validation restricted to the inference modes (t1ce-only and t1ce+flair)
"""

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerFT import nnUNetTrainerFT

# Indices of NCR (label 1) and ET (label 3) in dice_per_class_or_region.
CORE_LABEL_INDICES = (0, 2)
EMA_DECAY = 0.9
SELECTION_WARMUP_EPOCHS = 15
CORE_EARLY_STOPPING_PATIENCE = 40
CORE_OVERSAMPLE_FOREGROUND_PERCENT = 0.5
CORE_SAVE_EVERY = 2
# Sup_Loss indices drawn for validation: the inference modes only, t1ce-only
# (MultiMod [0,1,0,0]) and t1ce+flair ([0,1,0,1]).
CORE_VAL_MODALITY_COMBINATIONS = (1, 8)


class nnUNetTrainerFTCore(nnUNetTrainerFT):
    """MoME+ fine-tuning trainer with tumor-core checkpoint selection."""

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self._early_stopping_patience = CORE_EARLY_STOPPING_PATIENCE
        self.oversample_foreground_percent = CORE_OVERSAMPLE_FOREGROUND_PERCENT
        self.save_every = CORE_SAVE_EVERY

    def get_plain_dataloaders(self, initial_patch_size, dim):
        """Parent dataloaders, validation restricted to the inference modalities.

        Validation drives checkpoint selection and early stopping, so it skips the
        flair-only combination that inference never runs.
        """
        dl_tr, dl_val = super().get_plain_dataloaders(initial_patch_size, dim)
        dl_val.modality_combinations = CORE_VAL_MODALITY_COMBINATIONS
        return dl_tr, dl_val

    def _core_ema_history(self) -> list:
        """EMA of the mean NCR/ET pseudo Dice, one value per logged epoch."""
        ema, history = None, []
        for dice_per_label in self.logger.my_fantastic_logging['dice_per_class_or_region']:
            core_dice = float(np.nanmean([dice_per_label[i] for i in CORE_LABEL_INDICES]))
            ema = core_dice if ema is None else EMA_DECAY * ema + (1 - EMA_DECAY) * core_dice
            history.append(ema)
        return history

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        """Parent load_checkpoint + best core EMA and early-stopping counter rebuilt from the log."""
        super().load_checkpoint(filename_or_checkpoint)
        eligible = self._core_ema_history()[SELECTION_WARMUP_EPOCHS:self.current_epoch]
        self._best_ema = max(eligible) if eligible else None
        self._epochs_since_improvement = len(eligible) - 1 - int(np.argmax(eligible)) if eligible else 0
        self.print_to_log_file(
            f"Resumed at epoch {self.current_epoch}, best core EMA {self._best_ema}, "
            f"{self._epochs_since_improvement} epochs since improvement"
        )

    def on_epoch_end(self):
        """Logging, periodic checkpoint, best core-EMA checkpoint and early stopping."""
        logging = self.logger.my_fantastic_logging
        self.logger.log('epoch_end_timestamps', __import__('time').time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in logging['dice_per_class_or_region'][-1]])
        current_core_ema = self._core_ema_history()[-1]
        self.print_to_log_file('Core (NCR+ET) EMA pseudo dice', np.round(current_core_ema, decimals=4))
        self.print_to_log_file(
            f"Epoch time: {np.round(logging['epoch_end_timestamps'][-1] - logging['epoch_start_timestamps'][-1], decimals=2)} s")

        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        if current_epoch >= SELECTION_WARMUP_EPOCHS:
            if self._best_ema is None or current_core_ema > self._best_ema:
                self._best_ema = current_core_ema
                self._epochs_since_improvement = 0
                self.print_to_log_file(f"Yayy! New best core EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
                self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))
            else:
                self._epochs_since_improvement += 1

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        if self._epochs_since_improvement >= self._early_stopping_patience:
            self.print_to_log_file(
                f"Early stopping: no improvement for {self._early_stopping_patience} epochs. "
                f"Best core EMA Dice: {np.round(self._best_ema, decimals=4)}"
            )
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))
            self.num_epochs = current_epoch + 1

        self.current_epoch += 1
