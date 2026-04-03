"""Entry point for MoME+ fine-tuning training.

Bypasses the standard run_training fold validation to accept 'BrainTumorFT'
and uses load_checkpoint() to restore all 6 MoME+ networks (not just the main one).

Usage:
    python -m nnunetv2.run.run_training_ft 100 3d_fullres BrainTumorFT
    python -m nnunetv2.run.run_training_ft 100 3d_fullres BrainTumorFT --c  # continue
"""

import argparse
import multiprocessing

import torch
from batchgenerators.utilities.file_and_folder_operations import join, isfile
from torch.backends import cudnn

from nnunetv2.run.run_training import get_trainer_from_args


def run_training_ft_entry():
    parser = argparse.ArgumentParser(description="MoME+ fine-tuning training")
    parser.add_argument("dataset_name_or_id", type=str)
    parser.add_argument("configuration", type=str)
    parser.add_argument("fold", type=str)
    parser.add_argument("-tr", type=str, default="nnUNetTrainerFT")
    parser.add_argument("-p", type=str, default="nnUNetPlans")
    parser.add_argument(
        "-pretrained_weights",
        type=str,
        default=None,
        help="Path to MoME+ checkpoint_best.pth. Loads ALL 6 networks via load_checkpoint().",
    )
    parser.add_argument("-num_gpus", type=int, default=1)
    parser.add_argument("--use_compressed", default=False, action="store_true")
    parser.add_argument("--c", action="store_true", help="Continue from latest checkpoint")
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--val_best", action="store_true")
    parser.add_argument("--disable_checkpointing", action="store_true")
    parser.add_argument("-device", type=str, default="cuda")
    args = parser.parse_args()

    # Device setup
    if args.device == "cpu":
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device("cpu")
    elif args.device == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device("cuda")
    else:
        device = torch.device(args.device)

    # Fold: keep as string (no int conversion for BrainTumorFT)
    fold = args.fold

    # Build trainer (finds nnUNetTrainerFT class by name)
    nnunet_trainer = get_trainer_from_args(
        args.dataset_name_or_id, args.configuration, fold, args.tr, args.p,
        args.use_compressed, device=device,
    )

    if args.disable_checkpointing:
        nnunet_trainer.disable_checkpointing = True

    # ─── Checkpoint loading ───────────────────────────────────────────────────
    if args.c:
        # Continue training: load the full MoME+ checkpoint (all 6 networks)
        for ckpt_name in ("checkpoint_final.pth", "checkpoint_latest.pth", "checkpoint_best.pth"):
            ckpt_path = join(nnunet_trainer.output_folder, ckpt_name)
            if isfile(ckpt_path):
                print(f"Continuing from {ckpt_path}")
                nnunet_trainer.load_checkpoint(ckpt_path)
                break
        else:
            print("WARNING: No checkpoint found to continue from. Starting fresh.")
    elif args.pretrained_weights is not None:
        # Load only network weights (not optimizer/epoch/logger) so FT starts fresh
        nnunet_trainer.load_pretrained_mome_weights(args.pretrained_weights)

    # ─── Training ─────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    if not args.val:
        nnunet_trainer.run_training()

    if args.val_best:
        nnunet_trainer.load_checkpoint(
            join(nnunet_trainer.output_folder, "checkpoint_best.pth")
        )


if __name__ == "__main__":
    run_training_ft_entry()
