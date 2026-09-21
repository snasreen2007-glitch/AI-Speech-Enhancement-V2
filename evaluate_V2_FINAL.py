import numpy as np
import torch
from pathlib import Path
from scipy.io import wavfile
from pystoi import stoi
from pesq import pesq

from model_V2 import DTLNNoiseCanceller


# ============================================================
# SETTINGS
# ============================================================

BASE = Path.home() / "sih"

MODEL_PATH = BASE / "checkpoints/FINAL_MODEL_V2.pt"

CLEAN_DIR = BASE / "data/processed/test/clean"
NOISY_DIR = BASE / "data/processed/test/noisy"

FS = 16000
EPS = 1e-12


# ============================================================
# SNR
# ============================================================

def calculate_snr(clean, enhanced):

    error = enhanced - clean

    return 10 * np.log10(
        (np.mean(clean ** 2) + EPS) /
        (np.mean(error ** 2) + EPS)
    )


# ============================================================
# LOAD WAV
# ============================================================

def load_audio(path):

    _, audio = wavfile.read(path)

    audio = audio.astype(np.float32)

    if np.max(np.abs(audio)) > 2:

        audio /= 32768.0

    return audio


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("FINAL V2 MODEL EVALUATION")
print("=" * 70)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

model = DTLNNoiseCanceller()

if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model.eval()

print("V2 model loaded successfully.")


# ============================================================
# FILES
# ============================================================

clean_files = sorted(
    CLEAN_DIR.glob("*.wav")
)

noisy_files = sorted(
    NOISY_DIR.glob("*.wav")
)

print()
print("Clean files:", len(clean_files))
print("Noisy files:", len(noisy_files))


if len(clean_files) != 800 or len(noisy_files) != 800:

    print()
    print(
        "WARNING: Expected 800 test samples."
    )


# ============================================================
# STORAGE
# ============================================================

results = {
    "stationary": {
        "input_snr": [],
        "output_snr": [],
        "input_stoi": [],
        "output_stoi": [],
        "input_pesq": [],
        "output_pesq": []
    },

    "non-stationary": {
        "input_snr": [],
        "output_snr": [],
        "input_stoi": [],
        "output_stoi": [],
        "input_pesq": [],
        "output_pesq": []
    },

    "impulsive": {
        "input_snr": [],
        "output_snr": [],
        "input_stoi": [],
        "output_stoi": [],
        "input_pesq": [],
        "output_pesq": []
    }
}


# ============================================================
# ASSUMED TEST ORGANIZATION
#
# 0-19     stationary
# 20-39    non-stationary
# 40-59    impulsive
#
# We will evaluate all 800 samples.
# The category assignment is taken from the existing
# test-data organization used by our earlier scripts.
# ============================================================

def get_category(index):

    # Existing evaluation grouping
    block = index % 60

    if block < 20:
        return "stationary"

    elif block < 40:
        return "non-stationary"

    else:
        return "impulsive"


# ============================================================
# EVALUATION LOOP
# ============================================================

print()
print("=" * 70)
print("EVALUATING 800 TEST SAMPLES")
print("=" * 70)

for i, (clean_path, noisy_path) in enumerate(
    zip(clean_files, noisy_files)
):

    clean = load_audio(
        clean_path
    )

    noisy = load_audio(
        noisy_path
    )

    n = min(
        len(clean),
        len(noisy)
    )

    clean = clean[:n]
    noisy = noisy[:n]

    # --------------------------------------------------------
    # V2 INFERENCE
    # --------------------------------------------------------

    noisy_tensor = torch.tensor(
        noisy,
        dtype=torch.float32
    ).view(1, 1, -1)

    with torch.no_grad():

        enhanced_tensor, _ = model(
            noisy_tensor
        )

    enhanced = (
        enhanced_tensor
        .squeeze()
        .numpy()
    )

    enhanced = enhanced[:n]

    # --------------------------------------------------------
    # SNR
    # --------------------------------------------------------

    input_snr = calculate_snr(
        clean,
        noisy
    )

    output_snr = calculate_snr(
        clean,
        enhanced
    )

    # --------------------------------------------------------
    # STOI
    # --------------------------------------------------------

    try:

        input_stoi = stoi(
            clean,
            noisy,
            FS,
            extended=False
        )

        output_stoi = stoi(
            clean,
            enhanced,
            FS,
            extended=False
        )

    except Exception:

        input_stoi = np.nan
        output_stoi = np.nan


    # --------------------------------------------------------
    # PESQ
    # --------------------------------------------------------

    try:

        input_pesq = pesq(
            FS,
            clean,
            noisy,
            "wb"
        )

        output_pesq = pesq(
            FS,
            clean,
            enhanced,
            "wb"
        )

    except Exception:

        input_pesq = np.nan
        output_pesq = np.nan


    category = get_category(i)

    results[category]["input_snr"].append(
        input_snr
    )

    results[category]["output_snr"].append(
        output_snr
    )

    results[category]["input_stoi"].append(
        input_stoi
    )

    results[category]["output_stoi"].append(
        output_stoi
    )

    results[category]["input_pesq"].append(
        input_pesq
    )

    results[category]["output_pesq"].append(
        output_pesq
    )


    if (i + 1) % 50 == 0:

        print(
            f"{i+1}/800 samples completed"
        )


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 80)
print("FINAL V2 RESULTS")
print("=" * 80)

categories = [
    "stationary",
    "non-stationary",
    "impulsive"
]

for category in categories:

    r = results[category]

    input_snr = np.nanmean(
        r["input_snr"]
    )

    output_snr = np.nanmean(
        r["output_snr"]
    )

    input_stoi = np.nanmean(
        r["input_stoi"]
    )

    output_stoi = np.nanmean(
        r["output_stoi"]
    )

    input_pesq = np.nanmean(
        r["input_pesq"]
    )

    output_pesq = np.nanmean(
        r["output_pesq"]
    )

    print()
    print(category.upper())
    print("-" * 80)

    print(
        f"Input SNR       : {input_snr:.4f} dB"
    )

    print(
        f"V2 SNR          : {output_snr:.4f} dB"
    )

    print(
        f"SNR Improvement : "
        f"{output_snr-input_snr:+.4f} dB"
    )

    print()

    print(
        f"Input STOI      : {input_stoi:.4f}"
    )

    print(
        f"V2 STOI         : {output_stoi:.4f}"
    )

    print(
        f"STOI Improvement: "
        f"{output_stoi-input_stoi:+.4f}"
    )

    print()

    print(
        f"Input PESQ      : {input_pesq:.4f}"
    )

    print(
        f"V2 PESQ         : {output_pesq:.4f}"
    )

    print(
        f"PESQ Improvement: "
        f"{output_pesq-input_pesq:+.4f}"
    )


# ============================================================
# OVERALL
# ============================================================

all_input_snr = []
all_output_snr = []

all_input_stoi = []
all_output_stoi = []

all_input_pesq = []
all_output_pesq = []


for category in categories:

    r = results[category]

    all_input_snr.extend(
        r["input_snr"]
    )

    all_output_snr.extend(
        r["output_snr"]
    )

    all_input_stoi.extend(
        r["input_stoi"]
    )

    all_output_stoi.extend(
        r["output_stoi"]
    )

    all_input_pesq.extend(
        r["input_pesq"]
    )

    all_output_pesq.extend(
        r["output_pesq"]
    )


print()
print("=" * 80)
print("OVERALL 800-SAMPLE PERFORMANCE")
print("=" * 80)

print(
    f"Input SNR  : "
    f"{np.nanmean(all_input_snr):.4f} dB"
)

print(
    f"V2 SNR     : "
    f"{np.nanmean(all_output_snr):.4f} dB"
)

print(
    f"SNR gain   : "
    f"{np.nanmean(all_output_snr)-np.nanmean(all_input_snr):+.4f} dB"
)

print()

print(
    f"Input STOI : "
    f"{np.nanmean(all_input_stoi):.4f}"
)

print(
    f"V2 STOI    : "
    f"{np.nanmean(all_output_stoi):.4f}"
)

print()

print(
    f"Input PESQ : "
    f"{np.nanmean(all_input_pesq):.4f}"
)

print(
    f"V2 PESQ    : "
    f"{np.nanmean(all_output_pesq):.4f}"
)

print()
print("=" * 80)
print("V2 FINAL EVALUATION COMPLETE")
print("=" * 80)
