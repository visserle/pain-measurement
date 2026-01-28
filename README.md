# Multimodal Pain Measurement Dataset

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and data for a multimodal pain measurement study examining physiological and behavioral correlates of experimental heat pain in healthy adults. The dataset includes synchronized recordings of electrodermal activity (EDA), heart rate (HR), pupillometry, facial expressions, and electroencephalography (EEG), alongside continuous pain intensity ratings.

## Study Overview

### Experimental Design

Participants (*N* = 50; mean age = 25.8 ± 5.0 years) underwent:

1. **Pain Calibration**: Bayesian adaptive estimation of individual pain thresholds (VAS 0) and moderate pain intensities (VAS 70) using contact heat stimulation
2. **Measurement Session**: 12 trials of individualized temperature stimuli with continuous pain ratings on a visual analogue scale (VAS 0–100)

Temperature stimuli were generated procedurally using parametric waveforms combining half-cycle cosine functions with varying periods, amplitudes, plateaus, and decreasing phases. Each participant received identical stimulus patterns scaled to their individual pain sensitivity as determined during calibration.

### Recorded Modalities

| Modality | Sampling Rate | Features |
|----------|---------------|----------|
| EEG | 250 Hz | 8 channels (F3, F4, C3, Cz, C4, P3, P4, Oz) |
| Electrodermal Activity | 10 Hz | Raw skin conductance |
| Heart Rate | 10 Hz | Beats per minute from PPG |
| Pupillometry | 10 Hz | Bilateral pupil diameter |
| Facial Expressions | 10 Hz | 5 pain-related action units (brow furrow, cheek raise, mouth open, nose wrinkle, upper lip raise) |
| Temperature | 10 Hz | Applied thermode temperature |
| Pain Rating | 10 Hz | Continuous VAS (0–100) |

## Repository Structure

```text
├── data/                    # Raw and processed data
│   ├── experiments/         # Calibration and measurement results
│   └── imotions/           # Per-participant iMotions recordings
├── notebooks/              # Analysis and visualization notebooks
├── reports/                # Generated figures and statistics
├── results/                # Model training results
├── src/
│   ├── data/               # Data loading and database management
│   ├── experiments/        # Experiment control software
│   │   ├── calibration/    # Bayesian pain threshold estimation
│   │   └── measurement/    # Main measurement protocol
│   ├── features/           # Signal preprocessing pipelines
│   ├── models/             # Classification models and training
│   └── plots/              # Visualization utilities
└── tests/                  # Unit tests
```

## Database

The dataset is distributed as a DuckDB database file (`pain-measurement.duckdb`) containing:

- **Raw tables**: Unprocessed sensor data per modality
- **Feature tables**: Causally preprocessed data for predictive modeling
- **Explore tables**: Non-causally preprocessed data for exploratory analysis
- **Metadata tables**: Trial information, calibration results, and questionnaire responses

### Data Access

```python
from src.data.database_manager import DatabaseManager

db = DatabaseManager()
with db:
    # Retrieve preprocessed data with automatic invalid trial filtering
    df = db.get_trials("Feature_Data", exclude_problematic=True)
    
    # Direct SQL queries
    result = db.execute("SELECT * FROM Trials_Info").pl()
```

## Preprocessing Pipeline

All preprocessing is implemented using causal (forward-only) operations to ensure temporal validity for predictive modeling:

| Modality | Processing Steps |
|----------|------------------|
| EEG | Decimation (500→250 Hz), highpass filter (0.5 Hz), 50 Hz notch filter |
| EDA | Decimation (100→10 Hz) |
| Heart Rate | Artifact removal (>120 BPM), forward fill |
| Pupillometry | Blink removal, median filter, bilateral averaging |
| Facial Expressions | Exponential moving average smoothing (α=0.05) |

## Classification Framework

The repository includes a deep learning framework for binary classification of pain states:

### Task Definition

Samples are extracted from stimulus intervals and labeled based on temperature trajectory:

- **Class 0**: Temperature increasing or at plateau
- **Class 1**: Temperature decreasing (pain relief anticipation)

### Implemented Architectures

- **Transformer-based**: PatchTST, iTransformer, TimesNet
- **Lightweight**: LightTS, MLP
- **EEG-specific**: EEGNet
- **Ensemble**: EEGPhysioEnsemble, FacePhysioEnsemble

### Training

```bash
python -m src.models.main --features eda_raw heart_rate --models PatchTST MLP
```

Model selection uses Optuna for hyperparameter optimization with group-aware cross-validation (participant-level splits).

## Installation

### Requirements

- Python ≥ 3.12
- DuckDB ≥ 1.2
- Polars ≥ 1.0
- PyTorch ≥ 2.0
- SciPy ≥ 1.15

### Setup

```bash
conda env create -f requirements.yaml
conda activate pain
```

## Experiment Software

The experiment was implemented using [Expyriment](https://www.expyriment.org/) with:

- **Thermal stimulation**: QST.Lab Thermoino device
- **Multimodal recording**: iMotions platform
- **Continuous ratings**: Custom visual analogue scale

## Questionnaires

The following validated instruments were administered:

- Pain Catastrophizing Scale (PCS)
- Pain Vigilance and Awareness Questionnaire (PVAQ)
- State-Trait Anxiety Inventory (STAI-T-10)
- Positive and Negative Affect Schedule (PANAS; pre/post)
- Mindful Attention Awareness Scale (MAAS)

## Citation

If you use this dataset or code, please cite:

```bibtex
@article{,
  title={},
  author={},
  journal={PAIN},
  year={},
  doi={}
}
```

## License

This project is licensed under the MIT License. See [LICENCE](LICENCE) for details.

## Contact

For questions regarding the dataset or code, please open an issue or contact the corresponding author.
