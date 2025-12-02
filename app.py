import os
from pathlib import Path
from typing import Tuple, List

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

from train_model import train_and_save, DATA_PATH

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

SAMPLE_CSV_PATH = "sample_netflix_churn.csv"


def ensure_model_files() -> None:
    """Train model on sample data if model files do not exist."""
    needed = ["rf_model.pkl", "scaler.pkl", "feature_columns.pkl"]
    if all(os.path.exists(f) for f in needed):
        return

    st.sidebar.info("Model files not found. Training model now on sample data...")
    train_and_save(DATA_PATH)


def load_model() -> Tuple[object, object, List[str], dict]:
    """Load trained model, scaler, feature columns and metadata."""
    ensure_model_files()

    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_columns.pkl")

    meta: dict = {}
    meta_path = Path("model_meta.json")
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

    return model, scaler, feature_cols, meta


def load_sample_data() -> pd.DataFrame:
    """Load the built-in sample dataset."""
    return pd.read_csv(SAMPLE_CSV_PATH)


def validate_columns(df: pd.DataFrame) -> List[str]:
    """Return list of missing required columns."""
    return [c for c in EXPECTED_COLUMNS if c not in df.columns]


def run_prediction(
    df_row: pd.Series,
    model,
    scaler,
    feature_cols: List[str],
) -> float:
    """Run churn prediction for a single user row and return probability of churn."""
    if "churned" in df_row.index:
        df_row = df_row.drop(labels=["churned"])

    df_input = df_row.to_frame().T

    encoded = pd.get_dummies(df_input, drop_first=True)
    encoded = encoded.reindex(columns=feature_cols, fill_value=0)

    scaled = scaler.transform(encoded)
    proba = model.predict_proba(scaled)[0, 1]
    return float(proba)


def main() -> None:
    st.set_page_config(page_title="Netflix Churn App", layout="wide")

    st.title("Netflix Churn Prediction App")
    st.write(
        "A Python-based churn prediction demo built using Netflix-style user data "
        "for a project under Professor Alekh Gaur. You can use the built-in "
        "sample dataset or upload your own CSV in the same format."
    )

    # Load (and if needed, train) the model
    try:
        model, scaler, feature_cols, meta = load_model()
        st.sidebar.success("✅ Model ready.")
        if meta:
            st.sidebar.write(
                f"Test accuracy: {meta.get('test_accuracy', 0) * 100:.1f}% "
                f"(n_train={meta.get('n_train', '?')}, n_test={meta.get('n_test', '?')})"
            )
    except Exception as e:
        st.sidebar.error(
            "Model loading/training failed. Check logs and ensure sample_netflix_churn.csv "
            "and train_model.py are present."
        )
        st.sidebar.exception(e)
        return

    st.sidebar.header("Data source")

    mode = st.sidebar.radio(
        "Choose data source:",
        ("Use sample Netflix data (100 users)", "Upload your own CSV"),
    )

    df_data = None
    source_label = ""

    if mode == "Use sample Netflix data (100 users)":
        try:
            df_data = load_sample_data()
            source_label = "Sample dataset (sample_netflix_churn.csv)"
        except Exception as e:
            st.error(
                "Could not load the sample dataset. "
                "Make sure sample_netflix_churn.csv is present in the same folder as app.py."
            )
            st.exception(e)
            return
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
        if uploaded is not None:
            try:
                df_data = pd.read_csv(uploaded)
                source_label = f"Uploaded file: {uploaded.name}"
            except Exception as e:
                st.error("Could not read the uploaded CSV file.")
                st.exception(e)
                return
        else:
            st.info("Upload a CSV file to continue, or switch back to the sample dataset.")
            return

    st.subheader("Dataset overview")
    st.write(f"**Source:** {source_label}")

    missing = validate_columns(df_data)
    if missing:
        st.error(
            "The dataset is missing these required columns: "
            + ", ".join(missing)
            + ". Please check the README / instructions for the expected format."
        )
        return

    st.write("First 10 rows of the dataset:")
    st.dataframe(df_data.head(10))

    st.markdown(
        "#### Select a user\n"
        "Choose a specific user (row) from the dataset to predict churn probability."
    )

    if "customer_id" in df_data.columns:
        id_options = df_data["customer_id"].astype(str).tolist()
        selected_id = st.selectbox("Select customer_id:", id_options)
        row = df_data[df_data["customer_id"].astype(str) == selected_id].iloc[0]
        selected_label = f"customer_id = {selected_id}"
    else:
        idx_options = list(range(len(df_data)))
        selected_idx = st.selectbox("Select row index:", idx_options)
        row = df_data.iloc[selected_idx]
        selected_label = f"row index = {selected_idx}"

    st.write(f"**Selected user:** {selected_label}")

    with st.expander("Show selected row data"):
        st.json(row.to_dict())

    if st.button("Predict churn probability for this user"):
        proba = run_prediction(row, model, scaler, feature_cols)
        label = "Likely to churn" if proba >= 0.5 else "Not likely to churn"

        st.markdown("### Prediction result")
        st.write(f"**Churn probability:** {proba * 100:.1f}%")
        st.write(f"**Prediction:** {label}")

        if meta:
            st.caption(
                f"Model trained on {meta.get('n_train', '?')} users, "
                f"tested on {meta.get('n_test', '?')} users, "
                f"accuracy: {meta.get('test_accuracy', 0) * 100:.1f}%."
            )

    st.markdown("---")
    st.markdown("#### Expected dataset format")
    st.write(
        "Your CSV must contain **at least** the following columns with these meanings:"
    )
    st.markdown(
        "- `customer_id` (string/integer): Unique user identifier.\n"
        "- `age` (integer): Age of the user.\n"
        "- `gender` (string): e.g. Male, Female.\n"
        "- `subscription_type` (string): e.g. Basic, Standard, Premium.\n"
        "- `watch_hours` (float): Total watch hours in recent period.\n"
        "- `last_login_days` (integer): Days since last login.\n"
        "- `region` (string): Region or country.\n"
        "- `device` (string): Primary device (TV, Mobile, etc.).\n"
        "- `monthly_fee` (float): Subscription fee.\n"
        "- `churned` (0/1 integer): 1 if the user churned, 0 otherwise (required for training).\n"
        "- `payment_method` (string): e.g. Credit Card, Debit Card, PayPal.\n"
        "- `number_of_profiles` (integer): Number of profiles on the account.\n"
        "- `avg_watch_time_per_day` (float): Average daily watch time in hours.\n"
        "- `favorite_genre` (string): e.g. Drama, Comedy, Action.\n"
    )
    st.info(
        "For **training** (`train_model.py`), `churned` must be present. "
        "For **prediction-only** uploads here, `churned` can be included or omitted; "
        "it is not used in the prediction."
    )


if __name__ == "__main__":
    main()
