# Altair figure notebooks

Each notebook is organized around one manuscript figure. The composite notebook
keeps its five dependent panels together. The grand-average pair and the
participant-correlation pair each share one notebook for easier joint tweaking,
but remain separate rendered and exported figures.

| Figure | Notebook |
| --- | --- |
| Main composite (panels A–E) | `composite_figure.ipynb` |
| Methods stimulus temperature | `methods_stimulus_temperature.ipynb` |
| Model inference | `model_inference.ipynb` |
| Supplementary stimulus intervals | `supplementary_stimulus_intervals.ipynb` |
| Supplementary temperature curves | `supplementary_temperature_curves.ipynb` |
| Supplementary grand averages: physiology and facial expressions | `supplementary_grand_averages.ipynb` |
| Supplementary participant correlations: physiology and facial expressions | `supplementary_participant_correlations.ipynb` |
| Supplementary facial-expression correlation matrix | `supplementary_facial_correlation_matrix.ipynb` |

Reusable plotting implementations remain in `src/plots_altair/`. The notebooks
only prepare the figure-specific data, call the plotting function, and render or
export the result.

The aggregate notebooks under `notebooks/` are retained as legacy working
documents; `notebooks_altair/` is the figure-oriented entry point going forward.
