# Noninvasive and Objective Near Real-Time Detection of Pain Changes

[![Preprint](https://img.shields.io/badge/bioRxiv-preprint-b31b1b.svg)](https://www.biorxiv.org/content/10.64898/2026.01.26.701710v1)

This repository contains the source code and data management utilities for the study **"Noninvasive and Objective Near Real-Time Detection of Pain Changes in Fluctuating Pain"** ([Visser & Büchel, 2026](https://doi.org/10.64898/2026.01.26.701710)).

The codebase implements a multimodal machine learning pipeline to detect decreasing pain intensity using physiological signals (EEG, EDA, Heart Rate, Pupil Diameter, Facial Expressions).

> [!NOTE]
> The full dataset will be made available in a forthcoming data publication. Currently, data is available upon reasonable request to the corresponding author.

## Repository Structure

The project is organized into modular components for data handling, experimentation, and modeling.

```text
├── notebooks/               # Analysis and visualization notebooks
├── results/                 # Model weights and training results
├── src/
│   ├── data/                # Database management and data loading
│   ├── experiments/         # Experiment control software
│   ├── features/            # Signal preprocessing and feature extraction
│   ├── models/              # Deep learning models and training loop
│   └── plots/               # Visualization utilities
└── pain-measurement.duckdb  # DuckDB database (not included in repo)
```

## Installation

### Prerequisites
-   **Python**: ≥ 3.12
-   **Conda**: Recommended for environment management

### Setup
Create and activate the environment using the provided `requirements.yaml`:

```bash
conda env create -f requirements.yaml
conda activate pain
```

## Usage

### 1. Data Access
The project uses a local DuckDB database for efficient data querying. You can interact with it using the provided `DatabaseManager`:

```python
from src.data.database_manager import DatabaseManager

db = DatabaseManager()
with db:
    # Retrieve preprocessed feature data, automatically filtering invalid trials
    df = db.get_trials("Feature_Data", exclude_problematic=True)
    
    # Or execute direct SQL queries
    participant_info = db.execute("SELECT * FROM Participants_Info").pl()
    print(participant_info)
```

### 2. Training Models
The classification pipeline allows you to train and evaluate various deep learning architectures (e.g., PatchTST, TimesNet, MLP) on specific feature sets.

To train a model using EDA and Heart Rate features:

```bash
python -m src.models.main --features eda_raw heart_rate --models PatchTST MLP
```

See the README and code comments for more details on available options and configurations.

### 3. Experiments
The repository includes the experimental scripts used to collect the physiological data.
-   **Calibration**: `src.experiments.calibration`
-   **Measurement**: `src.experiments.measurement`


## Contact

For questions regarding the code or dataset, please open an issue or contact the corresponding author.
