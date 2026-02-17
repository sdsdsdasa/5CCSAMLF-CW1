from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer


# ---------- Configuration  --------
DEFAULT_CATEGORICAL_COLS = ["cut", "color", "clarity"]
TARGET_COL = "outcome"


@dataclass
class PreprocessConfig:
    """
    Config for preprocessing.
    - log1p_numeric: applies log1p to numeric columns (useful if many are skewed)
      WARNING: Only safe if numeric values are >= 0. If unsure, keep False.
    """
    categorical_cols: Optional[List[str]] = None
    target_col: str = TARGET_COL
    log1p_numeric: bool = False
    scale_numeric: bool = True
    numeric_impute_strategy: str = "median"
    categorical_impute_strategy: str = "most_frequent"
    onehot_drop: Optional[str] = "first"  # "first" to match baseline; or None


def split_X_y(df: pd.DataFrame, config: PreprocessConfig = PreprocessConfig()) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into X (features) and y (target)."""
    if config.target_col not in df.columns:
        raise ValueError(f"Target column '{config.target_col}' not found. Columns: {list(df.columns)}")
    y = df[config.target_col]
    X = df.drop(columns=[config.target_col])
    return X, y


def _safe_categorical_cols(X: pd.DataFrame, config: PreprocessConfig) -> List[str]:
    """Return categorical columns that actually exist in X (avoid KeyErrors)."""
    cols = config.categorical_cols if config.categorical_cols is not None else DEFAULT_CATEGORICAL_COLS
    return [c for c in cols if c in X.columns]


def make_preprocessor(X_example: Optional[pd.DataFrame] = None, config: PreprocessConfig = PreprocessConfig()) -> ColumnTransformer:
    """
    Build a ColumnTransformer for numeric + categorical preprocessing.

    If you pass X_example, categorical columns are restricted to those that exist.
    Otherwise it assumes DEFAULT_CATEGORICAL_COLS (if present at fit time).
    """
    # Decide categorical columns (only those that exist if X_example provided)
    categorical_cols = (
        _safe_categorical_cols(X_example, config) if X_example is not None
        else (config.categorical_cols if config.categorical_cols is not None else DEFAULT_CATEGORICAL_COLS)
    )

    # Numeric selector: everything except the categorical columns
    numeric_selector = make_column_selector(dtype_include=np.number)
    # Categorical selector: explicitly named columns (safer for coursework)
    # Any leftover non-numeric columns not in categorical_cols would be ignored by default.

    # --- Numeric pipeline ---
    numeric_steps = [
        ("imputer", SimpleImputer(strategy=config.numeric_impute_strategy)),
    ]

    if config.log1p_numeric:
        # Applies log1p to numeric features after imputation.
        # Note: log1p requires values >= -1; usually you want >= 0.
        numeric_steps.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))

    if config.scale_numeric:
        numeric_steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))

    numeric_pipe = Pipeline(steps=numeric_steps)

    # --- Categorical pipeline ---
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=config.categorical_impute_strategy)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop=config.onehot_drop, sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_selector),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",  # keep it strict/reproducible
        verbose_feature_names_out=False,
    )
    return preprocessor


def fit_preprocessor(preprocessor: ColumnTransformer, X_train: pd.DataFrame) -> ColumnTransformer:
    """Fit the preprocessor on training features only (no leakage)."""
    preprocessor.fit(X_train)
    return preprocessor


def transform_features(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    """
    Transform features into a numeric matrix suitable for sklearn models.
    Returns a numpy array.
    """
    return preprocessor.transform(X)


def fit_transform_features(
    X_train: pd.DataFrame,
    config: PreprocessConfig = PreprocessConfig(),
) -> Tuple[ColumnTransformer, np.ndarray]:
    """Convenience: create + fit + transform on training set."""
    pre = make_preprocessor(X_train, config=config)
    pre = fit_preprocessor(pre, X_train)
    X_train_p = transform_features(pre, X_train)
    return pre, X_train_p


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Get output feature names after preprocessing (useful for report / debugging).
    Works after preprocessor is fit.
    """
    try:
        names = preprocessor.get_feature_names_out()
        return list(names)
    except Exception:
        return []


# --- Optional: basic sanity checker ---
def sanity_check_alignment(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    config: PreprocessConfig = PreprocessConfig(),
) -> None:
    """
    Quick check that the preprocessor can fit on train and transform test without errors.
    This is the 'execution marks insurance' check.
    """
    pre = make_preprocessor(X_train, config=config)
    pre.fit(X_train)
    _ = pre.transform(X_test)
