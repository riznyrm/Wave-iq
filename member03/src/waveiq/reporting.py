"""Write a concise model comparison from the saved run."""
import json
from pathlib import Path
import pandas as pd

LABELS = {"dummy_median":"Median baseline", "ridge":"Ridge", "decision_tree":"Decision tree", "random_forest":"Random forest"}

def write_results(project_root):
    root = Path(project_root)
    reports = root / "reports"
    cv = pd.read_csv(reports / "cv_comparison.csv").set_index("model")
    test = pd.read_csv(reports / "holdout_comparison.csv").set_index("model")
    selection = json.loads((root / "artifacts/selection.json").read_text())
    audit = json.loads((root / "data/processed/data_audit.json").read_text())
    chosen = selection["selected_predictor"]
    lines = ["# Member 03 model results", "",
             f"Selected using development CV MAE: **{LABELS[chosen]}**.", "",
             "| Model | CV MAE | Test MAE | Test RMSE | Test R² |",
             "| --- | --- | --- | --- | --- |"]
    for name,label in LABELS.items():
        lines.append(f"| {label} | {cv.loc[name,'cv_MAE']:.3f} | {test.loc[name,'MAE']:.3f} | {test.loc[name,'RMSE']:.3f} | {test.loc[name,'R2']:.3f} |")
    lines += ["", "MAE and RMSE use the original recorded target scale. The physical unit still needs confirmation. CV scores were used for tuning; the test set was evaluated after selection was frozen.", "",
              "The measured improvement over the median baseline is modest. Negative R² and the scatter plot show that the current inputs do not explain the large rejection amounts well. The model remains a research prototype.", "",
              f"Development: {audit['development_rows']:,} rows. Test: {audit['test_rows']:,} rows. Related development rows excluded to protect the inherited test set: {audit['purged_alias_rows']:,}. No order family crosses the development/test boundary.", "",
              "The same five grouped folds compare 72 Ridge settings, 40 tree settings and 32 forest settings. Best parameters are in artifacts/best_parameters.json. Full candidate scores are in search_results/.", "",
              "![Model comparison](figures/model_comparison.png)", "",
              "![Actual versus predicted](figures/actual_vs_predicted.png)", "",
              "![Training and validation](figures/training_validation_gap.png)", "",
              "The source event definition, duplicate semantics, order-family relationships and target unit still need upstream validation. The supplied handover is used with one copy per identical exported row. Production outcome columns are excluded from model inputs."]
    (reports / "RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
