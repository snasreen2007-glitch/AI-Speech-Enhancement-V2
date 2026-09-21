# V2 Model Results

## Model

Lightweight phase-aware AI speech enhancement network.

### Architecture

- STFT processing
- Convolutional encoder
- Residual temporal blocks
- Magnitude mask estimation
- Real/imaginary spectral correction
- VAD head

### Model Size

Approximately 438K trainable parameters.

## V2 Evaluation

### Development Evaluation

| Noise Type | Input SNR | V2 Output SNR | Input STOI | V2 STOI | Input PESQ | V2 PESQ |
|------------|-----------|---------------|------------|---------|------------|---------|
| Stationary | 2.284 dB | 9.198 dB | 0.7904 | 0.8182 | 1.2466 | 1.5608 |
| Non-stationary | 2.750 dB | 13.878 dB | 0.8635 | 0.8760 | 1.8526 | 2.1409 |
| Impulsive | 6.402 dB | 11.371 dB | 0.8711 | 0.9174 | 1.5496 | 2.1353 |

## Final Evaluation

The final 800-sample evaluation produced:

- Input SNR: approximately 3.40 dB
- Output SNR: approximately 10.92 dB
- SNR improvement: approximately 7.52 dB
- STOI: approximately 0.869
- PESQ: approximately 1.934

## Target Comparison

| Metric | Target | Current V2 |
|--------|--------|------------|
| SNR | >15 dB | ~10.92 dB |
| STOI | >0.85 | ~0.869 |
| PESQ | >2.5 | ~1.934 |

## Important

The current V2 model does not yet achieve all project target values.

The augmented V2 training pipeline is included in:

`train_V2_AUG.py`

The original V2 model and checkpoint are preserved for comparison.

## Software

- Python
- PyTorch
- NumPy
- SciPy
- PySoundFile
- STOI
- PESQ
