# Noninvasive and Objective Near Real-Time Detection of Pain Changes in Fluctuating Pain

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Preprint](https://img.shields.io/badge/bioRxiv-preprint-b31b1b.svg)](https://www.biorxiv.org/content/10.64898/2026.01.26.701710v1)

Code and data accompanying [Visser & Büchel (2026, in submission)](https://doi.org/10.64898/2026.01.26.701710), a study on objective pain assessment using multimodal physiological recordings. The central goal is to reliably detect when pain intensity decreases—a building block for future interventions that leverage perceived control over pain.

Synchronized recordings cover EDA, heart rate, pupil size, facial action units, and EEG, captured while participants experienced tonic heat pain with continuously varying intensity.

> [!NOTE]
> Note that all data will be made available in a forthcoming data publication. In the meantime, data will be available upon reasonable request to the corresponding author. 

## Study Overview

### Participants

Fifty healthy adults (right-handed, aged 18–40, BMI 18–30) were recruited. After excluding five individuals and 69 problematic trials, the dataset contains **471 valid trials from 42 participants** (23 female; mean age 26.2 years).

### Protocol

1. **Calibration** — A Bayesian staircase procedure determined each participant's pain threshold and the temperature eliciting moderate pain (VAS 70). Resulting averages: threshold 44.5 °C (SD 1.5), VAS 70 at 46.8 °C (SD 0.8).

2. **Main task** — Twelve 3-minute heat stimuli per participant, with temperatures oscillating between the individually calibrated bounds. Participants provided continuous pain ratings on a 0–70 VAS.

### Stimulus Design

Temperature trajectories were constructed by chaining half-cycle cosines of randomized period (5–20 s) and amplitude:

$$y(t) = A \cos(\pi f t) + y_0 - A$$

Plateaus at two rising segments (15 s each) and one local minimum (5 s) discouraged anticipation. Every curve contained three identical large decreases, enabling focused evaluation of pain-decrease detection.

### Modalities

| Signal | Hardware | Acquired at | Stored at |
| ------ | -------- | ----------- | --------- |
| EEG (8 ch) | Neuroelectrics Enobio 8 | 500 Hz | 250 Hz |
| Skin conductance | Shimmer3 GSR+ | 100 Hz | 10 Hz |
| Heart rate | Shimmer3 GSR+ (PPG) | 100 Hz | 10 Hz |
| Pupil diameter | Smart Eye AI-X | 60 Hz | 10 Hz |
| Facial expressions | Affectiva / iMotions | ≈10 Hz | 10 Hz |
| Thermode temperature | Medoc Pathways ATS | — | 10 Hz |
| Pain rating | Custom VAS | 10 Hz | 10 Hz |

## Repository Structure

```text
├── notebooks/               # Analysis and visualization notebooks
├── results/                 # Model weigths and training results
├── src/
│   ├── data/                # Data loading and database management
│   ├── experiments/         # Experiment control software
│   │   ├── calibration/     # Bayesian pain threshold estimation
│   │   └── measurement/     # Main measurement protocol
│   ├── features/            # Signal preprocessing functions
│   ├── models/              # Classification models and training
│   └── plots/               # Visualization utilities
└── pain-measurement.duckdb  # DuckDB database with dataset
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

## Classification

A deep-learning pipeline trains binary classifiers to distinguish temperature phases:

| Label | Condition |
| ----- | --------- |
| 0 | Temperature rising or stable |
| 1 | Temperature falling |

### Models

Transformer variants (PatchTST, iTransformer),lightweight networks (LightTS, MLP), a CNN (TimesNet), an EEG-tailored architecture (EEGNet), and multimodal ensembles.

### Usage

```bash
python -m src.models.main --features eda_raw heart_rate --models PatchTST MLP
```

Hyperparameters are tuned with Optuna; splits respect participant identity to avoid data leakage.

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

## Citation

If you use this dataset or code, please cite:

```bibtex
@article{visser2026noninvasive,
  title={Noninvasive and Objective Near Real-Time Detection of Pain Changes in Fluctuating Pain},
  author={Visser, Leonard and B{\"u}chel, Christian},
  journal={bioRxiv},
  year={2026},
  doi={10.64898/2026.01.26.701710},
  url={https://www.biorxiv.org/content/10.64898/2026.01.26.701710v1}
}
```

## License

This github repository is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). See [LICENCE](LICENCE) for details.

## Contact

For questions regarding the dataset or code, please open an issue or contact the corresponding author.
