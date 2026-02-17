
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer


# --- Dataset-specific defaults (based on your uploaded CW1_train.csv) ---
TARGET_COL = "outcome"
DEFAULT_CATEGORICAL_COLS = ["cut", "color", "clarity"]  # exactly the object columns in your train data

NumericTransform = Literal["none", "yeo_johnson"]  # safe options for negatives


@dataclass
class PreprocessConfig:
    target_col: str = TARGET_COL

    # For your dataset, keep these as categoricals:
    categorical_cols: Optional[List[str]] = None  # if None, defaults to DEFAULT_CATEGORICAL_COLS

    # Imputation (your current train file has 0 missing values, but keep this for robustness)
    numeric_impute_strategy: str = "median"
    categorical_impute_strategy: str = "most_frequent"

    # Encoding
    onehot_drop: Optional[str] = "first"  # matches the provided baseline eval script; set None if you want all levels
    onehot_sparse: bool = False

    # Numeric handling
    numeric_transform: NumericTransform = "none"   # "yeo_johnson" can help with skew and is safe for negatives
    scale_numeric: bool = True                    # generally helps linear/SVM models


def split_X_y(df: pd.DataFrame, config: PreprocessConfig = PreprocessConfig()) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into X (features) and y (target)."""
    if config.target_col not in df.columns:
        raise ValueError(f"Target column '{config.target_col}' not found. Columns: {list(df.columns)}")
    y = df[config.target_col]
    X = df.drop(columns=[config.target_col])
    return X, y


def _categorical_cols_present(X: pd.DataFrame, config: PreprocessConfig) -> List[str]:
    """Use known categoricals if present; otherwise fall back to object dtype columns."""
    cols = config.categorical_cols if config.categorical_cols is not None else DEFAULT_CATEGORICAL_COLS
    present = [c for c in cols if c in X.columns]

    # Safety fallback: if someone passes a different dataset, auto-detect object columns
    if len(present) == 0:
        present = X.select_dtypes(include=["object", "category"]).columns.tolist()

    return present


def make_preprocessor(
    X_example: Optional[pd.DataFrame] = None,
    config: PreprocessConfig = PreprocessConfig(),
) -> ColumnTransformer:
    """
    Build a ColumnTransformer for numeric + categorical preprocessing.

    Pass X_example to lock categorical columns to those that exist (avoids KeyErrors).
    """
    if X_example is None:
        # We'll resolve categoricals at fit-time; still fine.
        categorical_cols = config.categorical_cols if config.categorical_cols is not None else DEFAULT_CATEGORICAL_COLS
    else:
        categorical_cols = _categorical_cols_present(X_example, config)

    # Selectors
    numeric_selector = make_column_selector(dtype_include=np.number)

    # --- Numeric pipeline ---
    numeric_steps = [
        ("imputer", SimpleImputer(strategy=config.numeric_impute_strategy)),
    ]

    if config.numeric_transform == "yeo_johnson":
        # Safe for zeros and negatives (unlike log/log1p)
        numeric_steps.append(("power", PowerTransformer(method="yeo-johnson", standardize=False)))

    if config.scale_numeric:
        numeric_steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))

    numeric_pipe = Pipeline(steps=numeric_steps)

    # --- Categorical pipeline ---
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=config.categorical_impute_strategy)),
            ("onehot", OneHotEncoder(handle_unknown="ignore",
                                     drop=config.onehot_drop,
                                     sparse_output=config.onehot_sparse)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_selector),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def fit_preprocessor(preprocessor: ColumnTransformer, X_train: pd.DataFrame) -> ColumnTransformer:
    """Fit the preprocessor on training features only (no leakage)."""
    preprocessor.fit(X_train)
    return preprocessor


def transform_features(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    """Transform features into a numeric matrix suitable for sklearn models."""
    Xt = preprocessor.transform(X)
    # If OneHotEncoder is sparse, it may return sparse matrix; convert to array for simplicity
    try:
        return Xt.toarray()
    except Exception:
        return np.asarray(Xt)


def fit_transform_features(
    X_train: pd.DataFrame,
    config: PreprocessConfig = PreprocessConfig(),
) -> Tuple[ColumnTransformer, np.ndarray]:
    """Convenience: create + fit + transform on training set."""
    pre = make_preprocessor(X_train, config=config)
    pre = fit_preprocessor(pre, X_train)
    X_train_p = transform_features(pre, X_train)
    return pre, X_train_p


def sanity_check(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    config: PreprocessConfig = PreprocessConfig(),
) -> None:
    """
    Insurance check: confirms we can fit on train and transform test (prevents submission-time crashes).
    """
    pre = make_preprocessor(X_train, config=config)
    pre.fit(X_train)
    _ = pre.transform(X_test)
