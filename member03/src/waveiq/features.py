"""Member 02 candidate features, fitted inside Member 03 training pipelines.

The formulas match cell 11 of 02_eda_feature_engineering.ipynb. This module
adds an importable sklearn transformer so saved models accept raw inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

BASE_NUM = [
    "Req_Finish_Fabrics", "Fabric_Allowance", "Shrink_allow",
    "Req_grey_fabric", "Req_beam_length(yds)", "warp_count",
    "weft_count", "epi", "ppi",
]
CAT = ["Construction"]
ENGINEERED = [
    "Thread_Density", "EPI_PPI_Ratio", "Log_Order_Size",
    "Fabric_Expansion_Ratio",
]
RAW_FEATURES = BASE_NUM + CAT
VARIANTS = ("original", "engineered", "reduced", "compact")
FORBIDDEN_FEATURES = {
    "ID", "Order_Group", "Month", "Previous_pdn", "Total_Pdn(yds)",
    "Rec_Beam_length(yds)", "Rejection", "Rej_and_cut_Piece",
    "act_shrink%", "Total_pdn_m/c", "Total_pdn_per_order",
    "Source_Row", "Record_ID", "Split",
}


def feature_lists(variant: str = "engineered") -> tuple[list[str], list[str]]:
    if variant == "original":
        return BASE_NUM.copy(), CAT.copy()
    if variant == "engineered":
        return BASE_NUM + ENGINEERED, CAT.copy()
    if variant == "reduced":
        return [c for c in BASE_NUM if c not in
                ["Req_grey_fabric", "Req_beam_length(yds)"]] + ENGINEERED, CAT.copy()
    if variant == "compact":
        return ["Log_Order_Size", "Fabric_Allowance", "Shrink_allow",
                "warp_count", "weft_count", "epi", "ppi",
                "Thread_Density", "EPI_PPI_Ratio"], CAT.copy()
    raise ValueError(f"Unknown feature variant: {variant!r}")


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Stateless formulas only; no statistics are fitted across rows."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Supply a pandas DataFrame with named raw input columns.")
    missing = sorted(set(RAW_FEATURES) - set(data.columns))
    if missing:
        raise ValueError(f"Missing raw input columns: {missing}")
    result = data[RAW_FEATURES].copy()
    for column in BASE_NUM:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result[BASE_NUM] = result[BASE_NUM].replace([np.inf, -np.inf], np.nan)
    result["Thread_Density"] = result["epi"] + result["ppi"]
    result["EPI_PPI_Ratio"] = result["epi"] / result["ppi"].where(result["ppi"] > 0)
    result["Log_Order_Size"] = np.log1p(
        result["Req_Finish_Fabrics"].where(result["Req_Finish_Fabrics"] >= 0)
    )
    result["Fabric_Expansion_Ratio"] = (
        result["Req_grey_fabric"] /
        result["Req_Finish_Fabrics"].where(result["Req_Finish_Fabrics"] > 0)
    )
    result[ENGINEERED] = result[ENGINEERED].replace([np.inf, -np.inf], np.nan)
    for column in CAT:
        result[column] = result[column].map(
            lambda value: str(value).strip() if pd.notna(value) else np.nan
        ).replace("", np.nan).astype(object)
    return result


class FeatureBuilder(TransformerMixin, BaseEstimator):
    """Choose a Member 02 candidate feature set without admitting outcomes or IDs."""

    def __init__(self, variant: str = "engineered"):
        self.variant = variant

    def fit(self, X: pd.DataFrame, y=None):
        numeric, categorical = feature_lists(self.variant)
        engineer_features(X.iloc[:1])  # Validate schema without learning statistics.
        self.output_features_ = numeric + categorical
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "output_features_")
        return engineer_features(X)[self.output_features_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "output_features_")
        return np.asarray(self.output_features_, dtype=object)


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale_numeric:
        steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(steps), make_column_selector(dtype_include=np.number)),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing",
                                      keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), CAT),
    ], remainder="drop")


def identity_target(values):
    return np.asarray(values, dtype=float)


def nonnegative_identity(values):
    """Apply the same zero lower bound during CV, testing and demonstration."""
    return np.maximum(0.0, np.asarray(values, dtype=float))


def nonnegative_expm1(values):
    # The cap prevents floating point overflow for extreme extrapolation.
    values = np.minimum(np.asarray(values, dtype=float), np.log(np.finfo(float).max) - 2)
    return np.maximum(0.0, np.expm1(values))
