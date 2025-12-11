# src/ml_models.py

from typing import List, Tuple
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .config import DATA_DIR


def load_labeled_data() -> pd.DataFrame:
    """
    Load the labeled market features dataset.

    Expected columns include at least:
        - timestamp
        - asset
        - technical indicator features
        - future_close
        - future_return
        - true_action
    """
    data_dir = Path(DATA_DIR)
    path = data_dir / "processed" / "market_features_labeled.csv"
    df = pd.read_csv(path)

    # Ensure timestamp is datetime
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Determine which columns should be used as features for the ML model.

    By default we exclude columns that are identifiers or targets.
    """
    exclude = {"timestamp", "asset", "future_close", "future_return", "true_action"}
    return [c for c in df.columns if c not in exclude]


def train_classifier(df_train: pd.DataFrame) -> Tuple[RandomForestClassifier, List[str]]:
    """
    Train a RandomForest classifier to predict true_action.

    Args:
        df_train: training DataFrame containing features and true_action.

    Returns:
        (clf, feature_cols)
    """
    feature_cols = get_feature_columns(df_train)
    X_train = df_train[feature_cols]
    y_train = df_train["true_action"]

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    return clf, feature_cols

