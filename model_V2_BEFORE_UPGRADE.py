import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                padding=2 * dilation,
                dilation=dilation
            ),
            nn.GroupNorm(8, channels),
            nn.PReLU(),

            nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                padding=2 * dilation,
                dilation=dilation
            ),
            nn.GroupNorm(8, channels)
        )

        self.activation = nn.PReLU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class DTLNNoiseCanceller(nn.Module):
    """
    V2 Phase-Aware Lightweight Speech Enhancement Network

    Input:
        noisy: [B, 1, T]

    Output:
        enhanced: [B, 1, T]
        vad_probability: [B, 1, T]

    Main improvements over V1:
        1. Stronger feature encoder
        2. Residual dilated temporal blocks
        3. Magnitude mask estimation
        4. Complex residual correction
        5. Speech-preservation pathway
        6. VAD estimation
    """

    def __init__(self):
        super().__init__()

        self.n_fft = 256
        self.hop_length = 128
        self.win_length = 256

        self.freq_bins = self.n_fft // 2 + 1

        # --------------------------------------------------
        # FEATURE ENCODER
        # --------------------------------------------------

        self.encoder = nn.Sequential(

            nn.Conv1d(
                self.freq_bins,
                128,
                kernel_size=5,
                padding=2
            ),
            nn.GroupNorm(8, 128),
            nn.PReLU(),

            nn.Conv1d(
                128,
                96,
                kernel_size=5,
                padding=2
            ),
            nn.GroupNorm(8, 96),
            nn.PReLU(),

            nn.Conv1d(
                96,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.GroupNorm(8, 64),
            nn.PReLU()
        )

        # --------------------------------------------------
        # TEMPORAL MODEL
        # --------------------------------------------------

        self.temporal = nn.Sequential(

            ResidualBlock(64, 1),
            ResidualBlock(64, 2),
            ResidualBlock(64, 4),
            ResidualBlock(64, 8),
            ResidualBlock(64, 16)
        )

        # --------------------------------------------------
        # MULTI-HEAD OUTPUTS
        # --------------------------------------------------

        # Magnitude mask
        self.mask_head = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(64, self.freq_bins, kernel_size=1)
        )

        # Real residual correction
        self.real_head = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(64, self.freq_bins, kernel_size=1)
        )

        # Imaginary residual correction
        self.imag_head = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(64, self.freq_bins, kernel_size=1)
        )

        # Speech-presence estimate
        self.vad_head = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(32, 1, kernel_size=1)
        )

        self.register_buffer(
            "window",
            torch.hann_window(self.win_length)
        )

    def forward(self, noisy):

        original_length = noisy.shape[-1]

        x = noisy.squeeze(1)

        # --------------------------------------------------
        # STFT
        # --------------------------------------------------

        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True
        )

        noisy_real = spec.real
        noisy_imag = spec.imag

        magnitude = torch.sqrt(
            noisy_real ** 2 +
            noisy_imag ** 2 +
            1e-8
        )

        # --------------------------------------------------
        # LOG SPECTRAL FEATURES
        # --------------------------------------------------

        log_mag = torch.log1p(magnitude)

        features = self.encoder(log_mag)

        # --------------------------------------------------
        # TEMPORAL PROCESSING
        # --------------------------------------------------

        temporal = self.temporal(features)

        # --------------------------------------------------
        # MAGNITUDE MASK
        # --------------------------------------------------

        mask = torch.sigmoid(
            self.mask_head(temporal)
        )

        # Prevent complete speech removal.
        mask = 0.02 + 0.98 * mask

        enhanced_mag = magnitude * mask

        # --------------------------------------------------
        # COMPLEX RESIDUAL CORRECTION
        # --------------------------------------------------

        real_residual = torch.tanh(
            self.real_head(temporal)
        )

        imag_residual = torch.tanh(
            self.imag_head(temporal)
        )

        # Residual correction is deliberately bounded.
        correction_scale = 0.15 * magnitude

        enhanced_real = (
            enhanced_mag *
            noisy_real /
            (magnitude + 1e-8)
        )

        enhanced_imag = (
            enhanced_mag *
            noisy_imag /
            (magnitude + 1e-8)
        )

        enhanced_real = (
            enhanced_real +
            correction_scale * real_residual
        )

        enhanced_imag = (
            enhanced_imag +
            correction_scale * imag_residual
        )

        # --------------------------------------------------
        # LIMIT COMPLEX MAGNITUDE
        # --------------------------------------------------

        enhanced_complex_mag = torch.sqrt(
            enhanced_real ** 2 +
            enhanced_imag ** 2 +
            1e-8
        )

        max_mag = magnitude * 1.05

        scale = torch.minimum(
            torch.ones_like(enhanced_complex_mag),
            max_mag / (enhanced_complex_mag + 1e-8)
        )

        enhanced_real = enhanced_real * scale
        enhanced_imag = enhanced_imag * scale

        enhanced_spec = torch.complex(
            enhanced_real,
            enhanced_imag
        )

        # --------------------------------------------------
        # ISTFT
        # --------------------------------------------------

        enhanced = torch.istft(
            enhanced_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            length=original_length
        )

        enhanced = enhanced.unsqueeze(1)

        # --------------------------------------------------
        # VAD
        # --------------------------------------------------

        vad_low = self.vad_head(temporal)

        vad_probability = torch.sigmoid(
            F.interpolate(
                vad_low,
                size=original_length,
                mode="linear",
                align_corners=False
            )
        )

        return enhanced, vad_probability


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("TESTING V2 PHASE-AWARE SPEECH ENHANCEMENT MODEL")
    print("=" * 65)

    device = torch.device("cpu")

    model = DTLNNoiseCanceller().to(device)

    x = torch.randn(
        1,
        1,
        32000,
        device=device
    )

    print("Input shape:", x.shape)

    with torch.no_grad():

        enhanced, vad = model(x)

    print("Enhanced shape:", enhanced.shape)
    print("VAD shape:", vad.shape)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Total parameters: {parameters:,}")

    print("=" * 65)
    print("V2 MODEL TEST PASSED")
    print("=" * 65)
