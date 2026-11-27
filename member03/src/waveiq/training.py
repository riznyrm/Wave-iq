"""Grouped tuning of exactly three model families plus a median baseline."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import parallel_backend
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV, GroupKFold, RandomizedSearchCV, cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from .data import GROUP, TARGET, export_prepared, prepare_data, sha256_file
from .features import (
    BASE_NUM, CAT, ENGINEERED, RAW_FEATURES, VARIANTS, FeatureBuilder,
    build_preprocessor, feature_lists, identity_target, nonnegative_expm1, nonnegative_identity,
)

DEFAULT_CONFIG = {'data_path': 'data/input/weaving_rejection_eda_input.csv',
 'expected_source_sha256': '554143538c452416b86c8402cc59d48c288c074c53229d854ad690c6bc29c07f',
 'random_state': 42,
 'test_size': 0.2,
 'cv_folds': 5,
 'n_jobs': 2,
 'decision_tree_iterations': 40,
 'random_forest_iterations': 32}

SCORING = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
}
MAIN_MODELS = ("ridge", "decision_tree", "random_forest")


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        return f"{value.__module__}.{value.__name__}"
    if isinstance(value, Path):
        return value.as_posix()
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False), encoding="utf-8")


def model_pipeline(name: str, seed: int = 42) -> Pipeline:
    if name == "ridge":
        estimator = Ridge(alpha=1.0, solver="lsqr", tol=1e-6)
    elif name == "decision_tree":
        estimator = DecisionTreeRegressor(random_state=seed)
    elif name == "random_forest":
        # Search parallelism is outside the forest; avoid nested CPU oversubscription.
        estimator = RandomForestRegressor(random_state=seed, n_jobs=1)
    elif name == "dummy_median":
        estimator = DummyRegressor(strategy="median")
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([
        ("features", FeatureBuilder("engineered")),
        ("preprocessing", build_preprocessor(scale_numeric=name == "ridge")),
        ("model", TransformedTargetRegressor(
            regressor=estimator, func=identity_target, inverse_func=nonnegative_identity,
            check_inverse=False,
        )),
    ])


def parameter_spaces(name: str) -> list[dict]:
    common = {"features__variant": list(VARIANTS)}
    if name == "ridge":
        common.update({"model__regressor__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]})
    elif name == "decision_tree":
        common.update({
            "model__regressor__max_depth": [3, 5, 8, 12, 18, None],
            "model__regressor__min_samples_leaf": [2, 5, 10, 20, 40],
            "model__regressor__min_samples_split": [2, 10, 20],
            "model__regressor__criterion": ["squared_error", "absolute_error"],
            "model__regressor__ccp_alpha": [0.0, 0.01, 0.1, 1.0],
        })
    elif name == "random_forest":
        common.update({
            "model__regressor__n_estimators": [150, 250],
            "model__regressor__max_depth": [5, 8, 12, 18, None],
            "model__regressor__min_samples_leaf": [2, 5, 10, 20, 40],
            "model__regressor__max_features": [0.5, 0.8, 1.0, "sqrt"],
            "model__regressor__max_samples": [0.7, 1.0],
        })
    else:
        raise ValueError(name)
    # A log target is a setting of the same regressor, not a fourth model family.
    # Inverse transformation happens before scoring, so all metrics use target units.
    return [
        {**common, "model__func": [identity_target], "model__inverse_func": [nonnegative_identity]},
        {**common, "model__func": [np.log1p], "model__inverse_func": [nonnegative_expm1]},
    ]


def metric_values(actual, predicted) -> dict:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def cv_summary(name: str, results: dict, index: int | None = None) -> dict:
    row = {"model": name, "role": "baseline" if name == "dummy_median" else "main"}
    for metric in SCORING:
        sign = 1 if metric == "R2" else -1
        if index is None:
            row[f"cv_{metric}"] = sign * float(np.mean(results[f"test_{metric}"]))
            row[f"cv_{metric}_std"] = float(np.std(results[f"test_{metric}"]))
            row[f"train_{metric}"] = sign * float(np.mean(results[f"train_{metric}"]))
        else:
            row[f"cv_{metric}"] = sign * float(results[f"mean_test_{metric}"][index])
            row[f"cv_{metric}_std"] = float(results[f"std_test_{metric}"][index])
            row[f"train_{metric}"] = sign * float(results[f"mean_train_{metric}"][index])
    return row


def save_search_results(search, path: Path) -> None:
    records = []
    for index, params in enumerate(search.cv_results_["params"]):
        record = {"candidate": index, "parameters": json.dumps(jsonable(params), sort_keys=True)}
        for key, values in search.cv_results_.items():
            if key == "params" or key.startswith("param_"):
                continue
            record[key] = jsonable(values[index])
        records.append(record)
    pd.DataFrame(records).to_csv(path, index=False)


def evaluate_frozen_models(project_root: str | Path) -> pd.DataFrame:
    """Member 04 entry point. Reuses the frozen models and test rows; never fits."""
    root = Path(project_root).resolve()
    artifacts, reports = root / "artifacts", root / "reports"
    selection = json.loads((artifacts / "selection.json").read_text(encoding="utf-8"))
    test = pd.read_csv(root / "data/processed/test.csv", dtype={"ID": str})
    X, y = test[RAW_FEATURES], test[TARGET]
    predictions = test[["Record_ID", "ID", "Month", GROUP]].copy()
    predictions["actual_Rejection"] = y
    rows, loaded = [], {}
    for name in ("dummy_median",) + MAIN_MODELS:
        path = artifacts / f"{name}.joblib"
        if sha256_file(path) != selection["model_sha256"][name]:
            raise ValueError(f"Artifact hash mismatch for {name}. Re-run training or restore the original artifact.")
        model = joblib.load(path)
        pred = model.predict(X)
        if not np.isfinite(pred).all() or np.any(pred < 0):
            raise ValueError(f"Invalid predictions from {name}")
        loaded[name] = pred
        predictions[f"predicted_{name}"] = pred
        rows.append({"model": name, "role": "baseline" if name == "dummy_median" else "main",
                     **metric_values(y, pred),
                     "selected_using_development_cv": name == selection["selected_predictor"]})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(reports / "holdout_comparison.csv", index=False)
    predictions.to_csv(reports / "holdout_predictions.csv", index=False)
    # Test outcomes are reported without changing the selected model.
    return comparison


def run_training(project_root: str | Path) -> dict:
    root = Path(project_root).resolve()
    config = DEFAULT_CONFIG.copy()
    seed, jobs = config["random_state"], config["n_jobs"]
    artifacts, reports = root / "artifacts", root / "reports"
    artifacts.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    (reports / "search_results").mkdir(exist_ok=True)
    print("Preparing the reviewed handover and preserving the Member 02 test orders...", flush=True)
    prepared = prepare_data(
        root / config["data_path"], seed=seed, test_size=config["test_size"],
        expected_sha256=config["expected_source_sha256"],
    )
    export_prepared(prepared, root / "data/processed")
    # Only development data enters fitting, feature choice or tuning.
    dev = prepared.development
    X, y, groups = dev[RAW_FEATURES], dev[TARGET], dev[GROUP]
    folds = list(GroupKFold(n_splits=config["cv_folds"]).split(X, y, groups))
    fold_rows = []
    fold_membership = dev[["Record_ID", GROUP]].copy()
    fold_membership["Validation_Fold"] = 0
    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        overlap = set(groups.iloc[train_idx]) & set(groups.iloc[valid_idx])
        assert not overlap
        assert set(groups.iloc[train_idx]).isdisjoint(prepared.test[GROUP])
        fold_membership.loc[valid_idx, "Validation_Fold"] = fold
        fold_rows.append({"fold": fold, "training_rows": len(train_idx),
                          "validation_rows": len(valid_idx),
                          "training_families": groups.iloc[train_idx].nunique(),
                          "validation_families": groups.iloc[valid_idx].nunique(),
                          "family_overlap": len(overlap)})
    assert fold_membership["Validation_Fold"].between(1, config["cv_folds"]).all()
    pd.DataFrame(fold_rows).to_csv(reports / "cv_fold_audit.csv", index=False)
    print(json.dumps(prepared.audit, indent=2), flush=True)

    models, summaries, details = {}, [], {}
    with threadpool_limits(limits=1), parallel_backend("threading", n_jobs=jobs):
        baseline = model_pipeline("dummy_median", seed)
        baseline_cv = cross_validate(
            baseline, X, y, cv=folds, scoring=SCORING, n_jobs=jobs,
            return_train_score=True, error_score="raise",
        )
        summaries.append(cv_summary("dummy_median", baseline_cv))
        baseline.fit(X, y)
        models["dummy_median"] = baseline
        print(f"Median baseline CV MAE: {summaries[-1]['cv_MAE']:.4f}", flush=True)
        for name in MAIN_MODELS:
            start = time.monotonic()
            arguments = dict(
                estimator=model_pipeline(name, seed), scoring=SCORING,
                refit="MAE", cv=folds, n_jobs=jobs, return_train_score=True,
                error_score="raise", verbose=0,
            )
            if name == "ridge":
                search = GridSearchCV(param_grid=parameter_spaces(name), **arguments)
            else:
                search = RandomizedSearchCV(
                    param_distributions=parameter_spaces(name),
                    n_iter=config[f"{name}_iterations"], random_state=seed, **arguments,
                )
            print(f"Tuning {name} using {len(folds)} order-family folds...", flush=True)
            search.fit(X, y)
            elapsed = time.monotonic() - start
            model = search.best_estimator_
            models[name] = model
            summary = cv_summary(name, search.cv_results_, search.best_index_)
            summary.update({
                "feature_variant": model.named_steps["features"].variant,
                "target_transform": "log1p" if model.named_steps["model"].func is np.log1p else "identity",
                "candidates": len(search.cv_results_["params"]), "tuning_seconds": elapsed,
            })
            summaries.append(summary)
            details[name] = {
                "best_parameters": jsonable(search.best_params_),
                "selected_candidate": search.best_index_,
                "candidate_count": len(search.cv_results_["params"]),
                "feature_variant": summary["feature_variant"],
                "target_transform": summary["target_transform"],
                "selected_features": sum(feature_lists(summary["feature_variant"]), []),
            }
            save_search_results(search, reports / "search_results" / f"{name}.csv")
            print(f"{name}: CV MAE={summary['cv_MAE']:.4f}, "
                  f"{summary['target_transform']} target, {summary['feature_variant']} features "
                  f"({elapsed:.1f} seconds)", flush=True)

    comparison = pd.DataFrame(summaries).sort_values("cv_MAE", kind="stable").reset_index(drop=True)
    comparison.to_csv(reports / "cv_comparison.csv", index=False)
    chosen = str(comparison.iloc[0]["model"])
    best_main = str(comparison.loc[comparison["role"].eq("main")].iloc[0]["model"])
    hashes = {}
    for name, model in models.items():
        path = artifacts / f"{name}.joblib"
        joblib.dump(model, path, compress=3)
        hashes[name] = sha256_file(path)
        restored = joblib.load(path)
        np.testing.assert_allclose(model.predict(X.iloc[:20]), restored.predict(X.iloc[:20]), rtol=1e-12)

    # Selection is written BEFORE any test predictions or test metrics are calculated.
    selection = {
        "selection_metric": "mean_5_fold_development_MAE_original_target_scale",
        "selected_predictor": chosen, "best_main_model": best_main,
        "selection_includes_baseline": True,
        "model_sha256": hashes, "seed": seed,
        "fit_population": "development_only; test_and_purged_rows_never_fitted",
        "status": "research_prototype_pending_upstream_semantic_validation",
    }
    write_json(artifacts / "selection.json", selection)
    write_json(artifacts / "best_parameters.json", details)
    pd.DataFrame([
        {"model": name, "feature": feature, "feature_variant": detail["feature_variant"]}
        for name, detail in details.items() for feature in detail["selected_features"]
    ]).to_csv(reports / "final_feature_list.csv", index=False)
    write_json(artifacts / "feature_contract.json", {
        "raw_numeric_inputs": BASE_NUM, "raw_categorical_inputs": CAT,
        "engineered_candidates": ENGINEERED, "target": TARGET,
        "target_unit": "unverified_recorded_amount", "prediction_lower_bound": 0,
        "missing_input_values": "training_fold_median_or_Missing_category",
        "unseen_construction": "OneHotEncoder(handle_unknown=ignore)",
        "selected_predictor": chosen,
        "selected_features": details[chosen]["selected_features"] if chosen in details else [],
    })
    write_json(artifacts / "environment.json", {
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__, "joblib": joblib.__version__,
    })
    print(f"Selection frozen: {chosen}; now evaluating the untouched test rows.", flush=True)
    holdout = evaluate_frozen_models(root)
    print(holdout.to_string(index=False), flush=True)
    return {"prepared": prepared, "models": models, "cv_comparison": comparison,
            "holdout_comparison": holdout, "selection": selection}
