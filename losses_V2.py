import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SI-SNR LOSS
# ============================================================

def si_snr_loss(
    enhanced,
    clean,
    eps=1e-8
):
    """
    Scale-Invariant Signal-to-Noise Ratio loss.

    Higher SI-SNR = better speech reconstruction.
    """

    enhanced = enhanced - enhanced.mean(dim=-1, keepdim=True)
    clean = clean - clean.mean(dim=-1, keepdim=True)

    target_energy = torch.sum(
        clean * clean,
        dim=-1,
        keepdim=True
    ) + eps

    projection = (
        torch.sum(
            enhanced * clean,
            dim=-1,
            keepdim=True
        )
        / target_energy
    ) * clean

    noise = enhanced - projection

    ratio = (
        torch.sum(projection ** 2, dim=-1)
        /
        (
            torch.sum(noise ** 2, dim=-1)
            + eps
        )
    )

    si_snr = 10.0 * torch.log10(
        ratio + eps
    )

    return -si_snr.mean()


# ============================================================
# MULTI-RESOLUTION STFT LOSS
# ============================================================

def stft_loss(
    enhanced,
    clean,
    fft_sizes=(256, 512, 1024),
    eps=1e-7
):
    """
    Multi-resolution spectral loss.

    Helps preserve:
        - speech harmonics
        - consonants
        - formants
        - transient information
    """

    enhanced = enhanced.squeeze(1)
    clean = clean.squeeze(1)

    total_loss = 0.0

    for n_fft in fft_sizes:

        hop = n_fft // 4
        win = n_fft

        window = torch.hann_window(
            win,
            device=enhanced.device
        )

        enh_spec = torch.stft(
            enhanced,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=window,
            center=True,
            return_complex=True
        )

        clean_spec = torch.stft(
            clean,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=window,
            center=True,
            return_complex=True
        )

        enh_mag = torch.abs(enh_spec)
        clean_mag = torch.abs(clean_spec)

        # Spectral convergence
        diff = torch.norm(
            clean_mag - enh_mag,
            p="fro"
        )

        reference = torch.norm(
            clean_mag,
            p="fro"
        ) + eps

        spectral_convergence = (
            diff / reference
        )

        # Log magnitude difference
        log_enh = torch.log(
            enh_mag + eps
        )

        log_clean = torch.log(
            clean_mag + eps
        )

        log_mag_loss = F.l1_loss(
            log_enh,
            log_clean
        )

        total_loss = (
            total_loss
            + spectral_convergence
            + log_mag_loss
        )

    return total_loss / len(fft_sizes)


# ============================================================
# WAVEFORM LOSS
# ============================================================

def waveform_loss(
    enhanced,
    clean
):
    """
    Direct waveform reconstruction loss.
    """

    return F.l1_loss(
        enhanced,
        clean
    )


# ============================================================
# ENERGY LOSS
# ============================================================

def energy_loss(
    enhanced,
    clean
):
    """
    Prevents the model from excessively
    suppressing or amplifying speech energy.
    """

    enhanced_energy = torch.sqrt(
        torch.mean(
            enhanced ** 2,
            dim=-1
        ) + 1e-8
    )

    clean_energy = torch.sqrt(
        torch.mean(
            clean ** 2,
            dim=-1
        ) + 1e-8
    )

    return F.l1_loss(
        enhanced_energy,
        clean_energy
    )


# ============================================================
# VAD LOSS
# ============================================================

def vad_loss(
    vad_probability,
    vad_target
):
    """
    VAD loss using the speech-activity target
    already provided by the dataset.
    """

    # Make target shape compatible with model output.
    if vad_target.dim() == 2:
        vad_target = vad_target.unsqueeze(1)

    if vad_target.shape[-1] != vad_probability.shape[-1]:
        vad_target = F.interpolate(
            vad_target.float(),
            size=vad_probability.shape[-1],
            mode="linear",
            align_corners=False
        )

    vad_target = vad_target.float()

    return F.binary_cross_entropy(
        vad_probability.clamp(
            1e-6,
            1.0 - 1e-6
        ),
        vad_target
    )


# ============================================================
# V2 TOTAL LOSS
# ============================================================

class EnhancementLossV2(nn.Module):

    def __init__(
        self,
        si_snr_weight=1.0,
        stft_weight=1.0,
        waveform_weight=0.5,
        energy_weight=0.15,
        vad_weight=0.10
    ):

        super().__init__()

        self.si_snr_weight = si_snr_weight
        self.stft_weight = stft_weight
        self.waveform_weight = waveform_weight
        self.energy_weight = energy_weight
        self.vad_weight = vad_weight

    def forward(
        self,
        enhanced,
        clean,
        vad_probability=None,
        vad_target=None
    ):

        loss_si_snr = si_snr_loss(
            enhanced,
            clean
        )

        loss_stft = stft_loss(
            enhanced,
            clean
        )

        loss_waveform = waveform_loss(
            enhanced,
            clean
        )

        loss_energy = energy_loss(
            enhanced,
            clean
        )

        total = (
            self.si_snr_weight * loss_si_snr
            +
            self.stft_weight * loss_stft
            +
            self.waveform_weight * loss_waveform
            +
            self.energy_weight * loss_energy
        )

        loss_vad = torch.tensor(
            0.0,
            device=enhanced.device
        )

        if (
            vad_probability is not None
            and vad_target is not None
        ):

            loss_vad = vad_loss(
                vad_probability,
                vad_target
            )

            total = (
                total
                +
                self.vad_weight * loss_vad
            )

        return {
            "total": total,
            "si_snr": loss_si_snr.detach(),
            "stft": loss_stft.detach(),
            "waveform": loss_waveform.detach(),
            "energy": loss_energy.detach(),
            "vad": loss_vad.detach()
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("TESTING V2 ENHANCEMENT LOSS")
    print("=" * 65)

    device = torch.device("cpu")

    clean = torch.randn(
        2,
        1,
        32000,
        device=device
    )

    enhanced = clean + (
        0.1 * torch.randn_like(clean)
    )

    vad = torch.rand(
        2,
        1,
        32000,
        device=device
    )

    criterion = EnhancementLossV2()

    losses = criterion(
        enhanced,
        clean,
        vad
    )

    for name, value in losses.items():

        print(
            f"{name:10s}: "
            f"{value.item():.6f}"
        )

    print("=" * 65)
    print("V2 LOSS TEST PASSED")
    print("=" * 65)
