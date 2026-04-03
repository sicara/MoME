"""MoME+ fine-tuning trainer.

Subclass of the MoME+ nnUNetTrainer that overrides:
- initialize(): points expert weights to local MoME+ pretrained checkpoints
- do_split(): loads from a custom datasplit.json
- on_epoch_end(): adds early stopping
- Lower LR (1e-4) and fewer epochs (200) for stable fine-tuning
"""

import json

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.Dispatch_network import ClsDispatchNet
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

# ─── Configuration ────────────────────────────────────────────────────────────
EXPERT_BASE = (
    "/workspace/models_weights/mome_brain_tumor/"
    "nnUNetTrainer__nnUNetPlans__3d_fullres/fold_MoME_plus"
)
DATASPLIT_PATH = "/workspace/data/mome_training/Dataset100_BrainTumor/datasplit.json"

# Fine-tuning hyperparameters
FT_INITIAL_LR = 1e-4
FT_NUM_EPOCHS = 200
FT_EARLY_STOPPING_PATIENCE = 30


class nnUNetTrainerFT(nnUNetTrainer):
    """MoME+ trainer for fine-tuning with local expert checkpoints."""

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self.initial_lr = FT_INITIAL_LR
        self.num_epochs = FT_NUM_EPOCHS
        self._early_stopping_patience = FT_EARLY_STOPPING_PATIENCE
        self._epochs_since_improvement = 0

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )
            num_experts = 4
            output_channels = 32

            # Main network (receives 4 modality channels + 4*32 expert features)
            self.network = self.build_network_architecture(
                self.plans_manager,
                self.dataset_json,
                self.configuration_manager,
                4 + output_channels * num_experts,
                enable_deep_supervision=True,
                num_experts=num_experts,
                Decoder_only=False,
                Mod_Prior=True,
            ).to(self.device)

            # Expert 1 (T1)
            self.network1 = self.build_network_architecture(
                self.plans_manager, self.dataset_json, self.configuration_manager,
                self.num_input_channels, enable_deep_supervision=True,
            ).to(self.device)
            load_pretrained_weights(
                self.network1, f"{EXPERT_BASE}/checkpoint_best1.pth", verbose=True
            )

            # Expert 2 (T1ce)
            self.network2 = self.build_network_architecture(
                self.plans_manager, self.dataset_json, self.configuration_manager,
                self.num_input_channels, enable_deep_supervision=True,
            ).to(self.device)
            load_pretrained_weights(
                self.network2, f"{EXPERT_BASE}/checkpoint_best2.pth", verbose=True
            )

            # Expert 3 (T2)
            self.network3 = self.build_network_architecture(
                self.plans_manager, self.dataset_json, self.configuration_manager,
                self.num_input_channels, enable_deep_supervision=True,
            ).to(self.device)
            load_pretrained_weights(
                self.network3, f"{EXPERT_BASE}/checkpoint_best3.pth", verbose=True
            )

            # Expert 4 (FLAIR)
            self.network4 = self.build_network_architecture(
                self.plans_manager, self.dataset_json, self.configuration_manager,
                self.num_input_channels, enable_deep_supervision=True,
            ).to(self.device)
            load_pretrained_weights(
                self.network4, f"{EXPERT_BASE}/checkpoint_best4.pth", verbose=True
            )

            # Dispatch network
            self.dispatch_network = ClsDispatchNet(input_dim=num_experts).to(self.device)

            # Compile if requested
            if self._do_i_compile():
                self.print_to_log_file('Compiling network...')
                self.network = torch.compile(self.network)
                self.network1 = torch.compile(self.network1)
                self.network2 = torch.compile(self.network2)
                self.network3 = torch.compile(self.network3)
                self.network4 = torch.compile(self.network4)
                self.dispatch_network = torch.compile(self.dispatch_network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError(
                "You have called self.initialize even though the trainer was already initialized. "
                "That should not happen."
            )

    def load_pretrained_mome_weights(self, checkpoint_path: str) -> None:
        """Load only network weights from a MoME+ checkpoint (all 6 networks).

        Unlike load_checkpoint(), this does NOT restore optimizer state, epoch,
        logger, or _best_ema — so fine-tuning starts fresh from epoch 0.
        """
        if not self.was_initialized:
            self.initialize()

        self.print_to_log_file(f"Loading MoME+ weights from {checkpoint_path}")

        ckpt_main = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        ckpt1 = torch.load(checkpoint_path.replace("best", "best1"), map_location=self.device, weights_only=False)
        ckpt2 = torch.load(checkpoint_path.replace("best", "best2"), map_location=self.device, weights_only=False)
        ckpt3 = torch.load(checkpoint_path.replace("best", "best3"), map_location=self.device, weights_only=False)
        ckpt4 = torch.load(checkpoint_path.replace("best", "best4"), map_location=self.device, weights_only=False)
        ckpt_dispatch = torch.load(checkpoint_path.replace("best", "best_dispatch"), map_location=self.device, weights_only=False)

        def _strip_module_prefix(state_dict: dict) -> dict:
            out = {}
            for k, v in state_dict.items():
                key = k[7:] if k.startswith("module.") else k
                out[key] = v
            return out

        self.network.load_state_dict(_strip_module_prefix(ckpt_main["network_weights"]))
        self.network1.load_state_dict(_strip_module_prefix(ckpt1["network_weights"]))
        self.network2.load_state_dict(_strip_module_prefix(ckpt2["network_weights"]))
        self.network3.load_state_dict(_strip_module_prefix(ckpt3["network_weights"]))
        self.network4.load_state_dict(_strip_module_prefix(ckpt4["network_weights"]))
        self.dispatch_network.load_state_dict(ckpt_dispatch["network_weights"])

        self.print_to_log_file("All 6 MoME+ networks loaded. Starting fresh from epoch 0.")

    def on_epoch_end(self):
        """Parent on_epoch_end + early stopping."""
        # Run parent logic (logging, checkpointing, best EMA tracking)
        self.logger.log('epoch_end_timestamps', __import__('time').time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # Best checkpoint tracking
        current_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
        if self._best_ema is None or current_ema > self._best_ema:
            self._best_ema = current_ema
            self._epochs_since_improvement = 0
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))
        else:
            self._epochs_since_improvement += 1

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        # Early stopping check
        if self._epochs_since_improvement >= self._early_stopping_patience:
            self.print_to_log_file(
                f"Early stopping: no improvement for {self._early_stopping_patience} epochs. "
                f"Best EMA Dice: {np.round(self._best_ema, decimals=4)}"
            )
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))
            self.num_epochs = current_epoch + 1  # Force training loop to stop

        self.current_epoch += 1

    def do_split(self):
        with open(DATASPLIT_PATH, "r") as f:
            data = json.load(f)

        tr_keys = data["train"]["BrainTumorFT"]
        val_keys = data["val"]["BrainTumorFT"]

        self.print_to_log_file("train keys: ", tr_keys)
        self.print_to_log_file("val keys: ", val_keys)
        self.print_to_log_file("num of tr_keys: ", len(tr_keys))
        self.print_to_log_file("num of val_keys: ", len(val_keys))
        self.print_to_log_file("num of batch size: ", self.batch_size)
        self.print_to_log_file("num of total epoch: ", self.num_epochs)

        return tr_keys, val_keys
