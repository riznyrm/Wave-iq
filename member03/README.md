# Wave IQ Member 03 model development

This folder contains Member 03's training notebook, code, datasets, saved models and results.

## Files to understand

| Item | Purpose |
| --- | --- |
| `03_model_development.ipynb` | Explained Colab workflow and prediction example |
| `train.py` | Run the same training locally |
| `requirements.txt` | Python dependencies |
| `src/` | Feature preparation, grouped splitting, training and prediction code |
| `data/` | Supplied input, development/test subsets, split assignments and data summary |
| `artifacts/` | Ridge, Decision Tree, Random Forest and baseline models with their settings |
| `reports/` | Model comparisons, tuning scores, test predictions and three graphs |

## Run in Google Colab

1. Upload `03_model_development.ipynb` to Google Colab and start a fresh runtime.
2. Run all cells. If the code is not already present, upload the updated `Wave-IQ-Member03.zip` when prompted.
3. Leave `RUN_TRAINING = True` to reproduce training. Set it to `False` to inspect the saved run.
4. Review the results and edit the raw inputs in the prediction example.

The notebook installs the dependencies from `requirements.txt`. It has been executed locally; the hosted Colab upload flow is checked separately in a local simulation.

## Run locally

Use Python 3.10–3.13. From this folder:

```bash
python -m pip install -r requirements.txt
python train.py
```

## Member 03 tasks

The workflow prepares development and test data, keeps order families separate, fits a median DummyRegressor baseline, tunes Ridge/Decision Tree/Random Forest with five grouped folds, compares MAE/RMSE/R², and saves complete model pipelines. The DummyRegressor is separate from the three main models.

Training settings are in `src/waveiq/training.py` under `DEFAULT_CONFIG`. All feature preparation is fitted inside training folds. Model selection uses development MAE and is frozen before test predictions. The dataset source hash prevents accidentally training on a different file.

The input has repeated summary records and uncertain target units. This run counts identical export rows once and excludes related development families from the inherited test set. The observed performance remains limited; see `reports/RESULTS.md`. A working implementation does not establish a highly accurate operational model.

## Give the results to Member 04

Use the models in `artifacts/`, `reports/holdout_comparison.csv`, `reports/holdout_predictions.csv` and the notebook prediction example. The saved pipelines compute engineered features and preprocessing from raw inputs. Install the pinned requirements before loading them.

## Git branch

Copy this folder into the existing Wave-iq repository and use the branch `model-development`.
