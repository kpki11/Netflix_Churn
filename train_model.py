import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json

# Default training data path (sample dataset included in the repo)
DATA_PATH = "sample_netflix_churn.csv"

# Columns we expect in the training dataset
EXPECTED_COLUMNS = [
    "customer_id",
    "age",
    "gender",
    "subscription_type",
    "watch_hours",
    "last_login_days",
    "region",
    "device",
    "monthly_fee",
    "churned",
    "payment_method",
    "number_of_profiles",
    "avg_watch_time_per_day",
    "favorite_genre",
]


def logical_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply logical cleaning rules similar to the original notebook."""
    # Protect against divide-by-zero
    implied_daily_watch = df["watch_hours"] / df["last_login_days"].replace(0, 1)

    # Category 1: Impossible usage
    cat1_a = df["avg_watch_time_per_day"] > 24
    cat1_b = implied_daily_watch > 24
    mask_cat1 = cat1_a | cat1_b

    # Category 2: Zero average but non-zero total
    mask_cat2 = (df["avg_watch_time_per_day"] == 0) & (df["watch_hours"] > 0)

    # Category 3: Behavioural outliers (example logic)
    cat3_a = (df["avg_watch_time_per_day"] > 12) & (df["last_login_days"] <= 2)
    cat3_b = (df["age"] > 60) & (df["avg_watch_time_per_day"] > 16)
    mask_cat3 = cat3_a | cat3_b

    total_outliers_mask = mask_cat1 | mask_cat2 | mask_cat3
    df_clean = df[~total_outliers_mask].copy()
    return df_clean


def winsorize_column(df: pd.DataFrame, col: str,
                     lower_quantile: float = 0.05,
                     upper_quantile: float = 0.95) -> pd.DataFrame:
    lower_bound = df[col].quantile(lower_quantile)
    upper_bound = df[col].quantile(upper_quantile)
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    return df


def load_and_preprocess(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"The following expected columns are missing from the CSV: {missing}. "
            "Please make sure your dataset follows the documented format."
        )

    df = df[EXPECTED_COLUMNS].copy()

    # Logical cleaning
    df_clean = logical_cleaning(df)

    # Winsorize high-variance usage columns
    df_wins = df_clean.copy()
    for col in ["watch_hours", "avg_watch_time_per_day"]:
        df_wins = winsorize_column(df_wins, col, 0.05, 0.95)

    return df_wins


def train_and_save(path: str = DATA_PATH):
    df_wins = load_and_preprocess(path)

    # Features and target
    X = df_wins.drop(columns=["churned"])
    y = df_wins["churned"].astype(int)

    # One-hot encode categoricals
    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestClassifier(
        n_estimators=150, random_state=42
    )
    rf_model.fit(X_train_scaled, y_train)

    y_pred = rf_model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print("\n--- Random Forest Classification Report ---")
    print(classification_report(y_test, y_pred))
    print(f"Test accuracy: {acc:.4f}")

    # Save model artefacts
    joblib.dump(rf_model, "rf_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(X_encoded.columns.tolist(), "feature_columns.pkl")

    meta = {
        "test_accuracy": float(acc),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "data_path": path,
    }
    with open("model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved: rf_model.pkl, scaler.pkl, feature_columns.pkl, model_meta.json")


if __name__ == "__main__":
    train_and_save()
