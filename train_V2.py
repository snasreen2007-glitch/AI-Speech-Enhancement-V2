import os
import json
import random
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

from model_V2 import DTLNNoiseCanceller
from losses_V2 import EnhancementLossV2


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path.home() / "sih"

DATA_DIR = BASE / "data/processed"
CHECKPOINT_DIR = BASE / "checkpoints"

SR = 16000
NUM_SAMPLES = 32000

BATCH_SIZE = 8
EPOCHS = 5

LEARNING_RATE = 1e-3

SEED = 42

# CPU settings
NUM_WORKERS = 0

# Gradient clipping
GRAD_CLIP = 5.0


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DATASET
# ============================================================

class SpeechNoiseDataset(Dataset):

    def __init__(
        self,
        split,
        limit=None
    ):

        self.split = split

        split_dir = DATA_DIR / split

        clean_dir = split_dir / "clean"
        noisy_dir = split_dir / "noisy"

        self.clean_files = sorted(
            clean_dir.glob("*.wav")
        )

        self.noisy_files = sorted(
            noisy_dir.glob("*.wav")
        )

        if len(self.clean_files) != len(
            self.noisy_files
        ):
            raise RuntimeError(
                f"{split}: clean/noisy count mismatch"
            )

        if limit is not None:

            self.clean_files = (
                self.clean_files[:limit]
            )

            self.noisy_files = (
                self.noisy_files[:limit]
            )

        print(
            f"{split.upper()} samples: "
            f"{len(self.clean_files)}"
        )

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, index):

        clean_path = self.clean_files[index]
        noisy_path = self.noisy_files[index]

        clean, clean_sr = sf.read(
            clean_path,
            dtype="float32"
        )

        noisy, noisy_sr = sf.read(
            noisy_path,
            dtype="float32"
        )

        if clean_sr != SR:
            raise RuntimeError(
                f"Unexpected clean sample rate: "
                f"{clean_sr}"
            )

        if noisy_sr != SR:
            raise RuntimeError(
                f"Unexpected noisy sample rate: "
                f"{noisy_sr}"
            )

        # Make mono
        if clean.ndim > 1:
            clean = clean.mean(axis=1)

        if noisy.ndim > 1:
            noisy = noisy.mean(axis=1)

        # Make exactly 64000 samples
        clean = self.fix_length(clean)
        noisy = self.fix_length(noisy)

        clean = torch.from_numpy(
            clean.copy()
        ).float()

        noisy = torch.from_numpy(
            noisy.copy()
        ).float()

        # [T] -> [1,T]
        clean = clean.unsqueeze(0)
        noisy = noisy.unsqueeze(0)

        # Simple waveform-based VAD target
        frame_energy = clean.abs()

        vad_target = (
            frame_energy > 0.01
        ).float()

        return (
            noisy,
            clean,
            vad_target
        )

    @staticmethod
    def fix_length(audio):

        if len(audio) > NUM_SAMPLES:

            audio = audio[:NUM_SAMPLES]

        elif len(audio) < NUM_SAMPLES:

            pad = (
                NUM_SAMPLES -
                len(audio)
            )

            audio = np.pad(
                audio,
                (0, pad)
            )

        return audio


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_loss
):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "scheduler_state_dict":
                scheduler.state_dict(),
            "best_val_loss":
                best_val_loss
        },
        path
    )


def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler
):

    checkpoint = torch.load(
        path,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    epoch = checkpoint["epoch"]

    best_val_loss = checkpoint[
        "best_val_loss"
    ]

    print(
        f"Resumed from epoch "
        f"{epoch + 1}"
    )

    return epoch + 1, best_val_loss


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    for batch_index, (
        noisy,
        clean,
        vad_target
    ) in enumerate(loader):

        noisy = noisy.to(device)
        clean = clean.to(device)
        vad_target = vad_target.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        enhanced, vad_probability = model(
            noisy
        )

        losses = criterion(
            enhanced,
            clean,
            vad_probability,
            vad_target
        )

        loss = losses["total"]

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()

        running_loss += loss.item()

        if (
            batch_index + 1
        ) % 100 == 0:

            print(
                f"  Batch "
                f"{batch_index + 1}/"
                f"{len(loader)} | "
                f"Loss: "
                f"{loss.item():.4f}"
            )

    return (
        running_loss /
        len(loader)
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    for (
        noisy,
        clean,
        vad_target
    ) in loader:

        noisy = noisy.to(device)
        clean = clean.to(device)
        vad_target = vad_target.to(device)

        enhanced, vad_probability = model(
            noisy
        )

        losses = criterion(
            enhanced,
            clean,
            vad_probability,
            vad_target
        )

        running_loss += (
            losses["total"].item()
        )

    return (
        running_loss /
        len(loader)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a very small training test"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    device = torch.device("cpu")

    torch.set_num_threads(
        min(
            14,
            os.cpu_count() or 1
        )
    )

    print("=" * 60)
    print("SIH AI/ML ADAPTIVE NOISE CANCELLATION")
    print("TRAINING")
    print("=" * 60)

    print(
        f"Device        : {device}"
    )

    print(
        f"CPU threads   : "
        f"{torch.get_num_threads()}"
    )

    # --------------------------------------------------------
    # Smoke test configuration
    # --------------------------------------------------------

    if args.smoke_test:

        train_limit = 32
        val_limit = 8

        epochs = 1

        print("\nSMOKE TEST MODE")
        print(
            f"Training samples: {train_limit}"
        )
        print(
            f"Validation samples: {val_limit}"
        )

    else:

        train_limit = None
        val_limit = None

        epochs = EPOCHS

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = SpeechNoiseDataset(
        "train",
        limit=train_limit
    )

    val_dataset = SpeechNoiseDataset(
        "val",
        limit=val_limit
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = DTLNNoiseCanceller().to(
        device
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\nModel parameters: "
        f"{parameters:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = EnhancementLossV2()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    # --------------------------------------------------------
    # Checkpoint directory
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    last_checkpoint = (
        CHECKPOINT_DIR /
        "last.pt"
    )

    best_checkpoint = (
        CHECKPOINT_DIR /
        "best.pt"
    )

    start_epoch = 1

    best_val_loss = float("inf")

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    if (
        args.resume
        and
        last_checkpoint.exists()
    ):

        start_epoch, best_val_loss = (
            load_checkpoint(
                last_checkpoint,
                model,
                optimizer,
                scheduler
            )
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nStarting training...\n")

    for epoch in range(
        start_epoch,
        epochs + 1
    ):

        print("=" * 60)

        print(
            f"EPOCH {epoch}/{epochs}"
        )

        print("=" * 60)

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(
            val_loss
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"\nEpoch {epoch} results:"
        )

        print(
            f"Train loss : "
            f"{train_loss:.6f}"
        )

        print(
            f"Val loss   : "
            f"{val_loss:.6f}"
        )

        print(
            f"Learning rate: "
            f"{current_lr:.6e}"
        )

        # ----------------------------------------------------
        # Save last checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_loss
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                best_checkpoint
            )

            print(
                f"\n*** NEW BEST MODEL ***"
            )

            print(
                f"Saved: "
                f"{best_checkpoint}"
            )

    print("\n")
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )


if __name__ == "__main__":
    main()
