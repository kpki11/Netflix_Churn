# Netflix Churn Prediction App

A Python-based churn prediction demo built using Netflix-style user data for a project under **Professor Alekh Gaur**.

The app:
- Trains a Random Forest classifier on Netflix-style churn data.
- Automatically trains the model on the **sample dataset** if no model files are found.
- Lets users explore a built-in sample dataset (100 users).
- Lets users upload their own CSV in the same format and pick any row to get a churn probability.

## Files

- `train_model.py` – Loads CSV data, applies logical cleaning and winsorization, trains a Random Forest model, and saves model artefacts.
- `app.py` – Streamlit app that:
  - Ensures model files exist (auto-trains on `sample_netflix_churn.csv` if needed).
  - Uses the sample dataset or an uploaded CSV.
  - Predicts churn probability for a selected user row.
- `sample_netflix_churn.csv` – Dummy dataset with 100 Netflix-style users.
- `requirements.txt` – Python dependencies.
- `.gitignore` – Ignore cache and model artefacts in git.
- `model_meta.json`, `rf_model.pkl`, `scaler.pkl`, `feature_columns.pkl` – Created at runtime (or by running `python train_model.py`).

## Expected dataset format

Any dataset used for training or upload must contain at least these columns:

- `customer_id` (string/integer): Unique user identifier.
- `age` (integer): Age of the user.
- `gender` (string): e.g. `Male`, `Female`.
- `subscription_type` (string): e.g. `Basic`, `Standard`, `Premium`.
- `watch_hours` (float): Total watch hours in a recent period.
- `last_login_days` (integer): Days since last login.
- `region` (string): Region / country (e.g. `India`, `US`, `Europe`).
- `device` (string): Primary device (`TV`, `Mobile`, `Tablet`, `Desktop`).
- `monthly_fee` (float): Subscription fee.
- `churned` (0/1 integer): 1 if the user churned, 0 otherwise (required for training).
- `payment_method` (string): e.g. `Credit Card`, `Debit Card`, `PayPal`.
- `number_of_profiles` (integer): Number of profiles on the account.
- `avg_watch_time_per_day` (float): Average daily watch time (hours per day).
- `favorite_genre` (string): e.g. `Action`, `Drama`, `Comedy`.

For **training** (`train_model.py`), `churned` is required. For **prediction-only** uploads in the app, `churned` can be present or missing; the model ignores it when predicting.

The repo includes `sample_netflix_churn.csv` that follows this schema and contains 100 dummy users.

## How it works on Streamlit Cloud

1. Streamlit runs `app.py`.
2. `app.py` calls `load_model()`, which calls `ensure_model_files()`.
3. If no `rf_model.pkl` / `scaler.pkl` / `feature_columns.pkl` exist, it calls `train_and_save(DATA_PATH)` from `train_model.py`.
4. The model is trained on `sample_netflix_churn.csv` on the server, and the files are saved.
5. The app then loads the model and is ready to use.

You do **not** need to run `train_model.py` locally or upload `.pkl` files if you don't want to; everything can be trained on first run in the Streamlit environment.

## Local usage (optional)

```bash
pip install -r requirements.txt
python train_model.py        # optional; trains and saves model locally
streamlit run app.py
```
