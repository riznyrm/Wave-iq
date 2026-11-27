from pathlib import Path
import sys, json, joblib, matplotlib.pyplot as plt, numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def _fix_imputers(obj):
    if isinstance(obj, SimpleImputer) and not hasattr(obj, '_fill_dtype'):
        obj._fill_dtype = getattr(obj, '_fit_dtype', getattr(getattr(obj, 'statistics_', None), 'dtype', np.float64))
    for attr in ('steps', 'transformers', 'transformers_'):
        val = getattr(obj, attr, None)
        if isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, (list, tuple)):
                    for sub in item:
                        _fix_imputers(sub)
                else:
                    _fix_imputers(item)
    for attr in ('regressor', 'estimator', 'base_estimator', 'named_steps'):
        val = getattr(obj, attr, None)
        if isinstance(val, dict):
            for v in val.values():
                _fix_imputers(v)
        elif val is not None:
            _fix_imputers(val)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'member03' / 'src'))
M03, M04 = ROOT / 'member03', ROOT / 'member04'
FIG_DIR = M04 / 'reports' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
contract = json.loads((M03 / 'artifacts' / 'feature_contract.json').read_text(encoding='utf-8'))
raw_cols = list(contract['raw_numeric_inputs']) + list(contract['raw_categorical_inputs'])
target_col = contract['target']
lower_bound = contract.get('prediction_lower_bound', 0)
selected_model_name = contract.get('selected_predictor', 'random_forest')
test_df = pd.read_csv(M03 / 'data' / 'processed' / 'test.csv')
missing_cols = sorted(set(raw_cols + [target_col]) - set(test_df.columns))
if missing_cols: raise ValueError(f'Missing expected columns in test.csv: {missing_cols}')
X_test = test_df[raw_cols].copy()
y_test = pd.to_numeric(test_df[target_col], errors='coerce')
mask = y_test.notna()
X_test, y_test = X_test.loc[mask].reset_index(drop=True), y_test.loc[mask].reset_index(drop=True)
rows, preds_by_model = [], {}
for mpath in sorted((M03 / 'artifacts').glob('*.joblib')):
    model = joblib.load(mpath)
    _fix_imputers(model)
    pred = np.clip(np.asarray(model.predict(X_test), dtype=float), lower_bound, None)
    preds_by_model[mpath.stem] = pred
    rows.append({'model': mpath.stem, 'mae': float(mean_absolute_error(y_test, pred)), 'rmse': float(np.sqrt(mean_squared_error(y_test, pred))), 'r2': float(r2_score(y_test, pred))})
metrics_df = pd.DataFrame(rows).sort_values('mae', ascending=True).reset_index(drop=True)
metrics_df.to_csv(M04 / 'reports' / 'holdout_metrics_comparison.csv', index=False)
print('=== Member 04 Holdout Evaluation Comparison ===')
print(metrics_df.to_string(index=False))
best_name = selected_model_name if selected_model_name in preds_by_model else metrics_df.loc[0, 'model']
best_pred = preds_by_model[best_name]
print(f'\nSelected / Best Model: {best_name}')
plt.figure(figsize=(8, 4.5))
plt.bar(metrics_df['model'], metrics_df['mae'])
plt.ylabel('Holdout MAE')
plt.title('Holdout MAE by Model (Lower is Better)')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig(FIG_DIR / 'holdout_mae_comparison.png', dpi=150)
plt.close()
plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_pred, alpha=0.6)
max_val = max(float(np.max(y_test)), float(np.max(best_pred)), 1.0)
plt.plot([0, max_val], [0, max_val], linestyle='--')
plt.xlabel('Actual Rejection')
plt.ylabel(f'Predicted Rejection ({best_name})')
plt.title(f'Actual vs. Predicted Rejection - {best_name}')
plt.tight_layout()
plt.savefig(FIG_DIR / 'actual_vs_predicted.png', dpi=150)
plt.close()
demo_df = X_test.head(5).copy()
demo_df['Actual_Rejection'] = y_test.head(5).values
demo_df['Predicted_Rejection'] = np.round(best_pred[:5], 4)
demo_df.to_csv(M04 / 'reports' / 'sample_predictions_demo.csv', index=False)
print('\n=== Sample Prediction Demo (First 5 Holdout Rows) ===')
print(demo_df[['Construction', 'Req_Finish_Fabrics', 'epi', 'ppi', 'Actual_Rejection', 'Predicted_Rejection']].to_string(index=False))