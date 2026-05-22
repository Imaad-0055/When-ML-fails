# Mini-Project: When ML Fails

This repository investigates a failure mode on the Paddy dataset for the ECC Spring 2026 mini-project.

## Research question

Does a non-linear Random Forest regressor trained to predict total paddy yield learn a scale shortcut from cultivated-area and input-quantity features, such that it appears excellent under a random split but fails when evaluated on unseen cultivated-area groups?

## Project structure

- `src/run_experiments.py`: reproducible experiment runner.
- `src/run_reference_pipeline.py`: runs only the broken/reference total-yield pipeline.
- `src/run_corrected_pipeline.py`: runs only the corrected per-hectare pipeline.
- `notebooks/paddy_failure_analysis.ipynb`: notebook version of the investigation.
- `report.md`: written report following the required project structure.
- `report.pdf`: PDF version of the written report.
- `results/`: generated tables and figures.
- `requirements.txt`: Python dependencies.

## Reproduce

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\run_experiments.py --data data\paddydataset.csv
```

The script uses fixed random seeds and writes all outputs to `results/`.

If `python` is not available in PowerShell, use the installed interpreter path that was used to generate the results:

```powershell
& "C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe" src\run_experiments.py --data data\paddydataset.csv
```

To run the broken and corrected pipelines independently:

```powershell
python src\run_reference_pipeline.py --data data\paddydataset.csv --output-dir results_reference
python src\run_corrected_pipeline.py --data data\paddydataset.csv --output-dir results_corrected
```
