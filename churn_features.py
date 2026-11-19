"""Feature engineering, kept in its own module so the saved model can be
unpickled from any script (a function defined in a script that was run as
__main__ cannot be found again when loading from somewhere else)."""
import numpy as np

ADDON_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def add_features(X):
    """Domain-motivated features. Row-wise only, so there is no train/test leakage."""
    import pandas as pd

    X = X.copy()
    X["n_addons"] = (X[ADDON_COLS] == "Yes").sum(axis=1)
    X["avg_monthly_spend"] = X["TotalCharges"] / X["tenure"].clip(lower=1)
    X["charge_vs_history"] = X["MonthlyCharges"] - X["avg_monthly_spend"]
    X["autopay"] = X["PaymentMethod"].str.contains("automatic").astype(int)
    X["tenure_bucket"] = pd.cut(
        X["tenure"], [-1, 6, 12, 24, 48, np.inf], labels=["0-6", "7-12", "13-24", "25-48", "49+"]
    ).astype(str)
    return X
