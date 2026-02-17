import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Tuple, Optional, List, Literal

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer


TARGET_COL = "outcome"
CATEGORICAL_COLS = ["cut", "color", "clarity"]

NumericTransform = Literal["none", "yeo_johnson"]


@dataclass
class PreprocessConfig:
    target_col: str = TARGET_COL
    categorical_cols: Optional[List[str]] = None

    numeric_impute_strategy: str = "median"
    categorical_impute_strategy: str = "most_frequent"

    numeric_transform: NumericTransform = "none"   # "yeo_johnson" handles negatives safely
    scale_numeric: bool = True

    onehot_drop: Optional[str] = "first"


def split_X_y(df: pd.DataFrame, config: PreprocessConfig = PreprocessConfig()) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[config.target_col]
    X = df.drop(columns=[config.target_col])
    return X, y


def make_preprocessor(X_example: pd.DataFrame, config: PreprocessConfig = PreprocessConfig()) -> ColumnTransformer:

    categorical_cols = config.categorical_cols or CATEGORICAL_COLS
    categorical_cols = [c for c in categorical_cols if c in X_example.columns]

    numeric_selector = make_column_selector(dtype_include=np.number)

    numeric_steps = [
        ("imputer", SimpleImputer(strategy=config.numeric_impute_strategy))
    ]

    if config.numeric_transform == "yeo_johnson":
        numeric_steps.append(("power", PowerTransformer(method="yeo-johnson", standardize=False)))

    if config.scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=config.categorical_impute_strategy)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop=config.onehot_drop, sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_selector),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    return preprocessor


def fit_transform(X_train: pd.DataFrame, config: PreprocessConfig = PreprocessConfig()):
    pre = make_preprocessor(X_train, config)
    Xp = pre.fit_transform(X_train)
    return pre, Xp


def transform(pre, X):
    return pre.transform(X)
