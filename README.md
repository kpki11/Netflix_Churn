# Netflix Churn Prediction App

A Python-based churn prediction demo built using Netflix-style user data for a project under **Professor Alekh Gaur**.

The app:
- Trains a Random Forest classifier on Netflix-style churn data.
- Lets users **explore a built-in sample dataset** (100 users).
- Lets users **upload their own CSV** in the same format and pick any row to get a churn probability.

## 1. Files

- `train_model.py` – Loads CSV data, applies logical cleaning and winsorization, trains a Random Forest model, and saves model artefacts.
- `app.py` – Streamlit app that:
  - Loads the trained model.
  - Uses the sample dataset or an uploaded CSV.
  - Predicts churn probability for a selected user row.
- `sample_netflix_churn.csv` – Dummy dataset with 100 Netflix-style users.
- `requirements.txt` – Python dependencies.
- `.gitignore` – Ignore cache and model artefacts in git.
- `model_meta.json`, `rf_model.pkl`, `scaler.pkl`, `feature_columns.pkl` – Created after training.

## 2. Expected dataset format

Any dataset used for **training** or **upload** must contain at least these columns:

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

> For **training** (`train_model.py`), `churned` is required.  
> For **prediction-only** uploads in the app, `churned` can be present or missing; the model ignores it when predicting.

The repo includes `sample_netflix_churn.csv` that follows this schema and contains 100 dummy users.  
You can use it as-is or replace it with your own data in the same format.

## 3. Local setup

1. Clone/download this repo.
2. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## 4. Train the model

By default, `train_model.py` trains on `sample_netflix_churn.csv`.

```bash
python train_model.py
```

This will:

- Check that all required columns exist.
- Apply logical cleaning (impossible watch times, behavioural outliers, etc.).
- Winsorize key usage columns.
- One-hot encode categorical variables.
- Train a `RandomForestClassifier`.
- Save:
  - `rf_model.pkl`
  - `scaler.pkl`
  - `feature_columns.pkl`
  - `model_meta.json`

If you want to train on **your own dataset**, either:
- Replace `sample_netflix_churn.csv` with your own CSV that has the same columns, **or**
- Change the `DATA_PATH` constant at the top of `train_model.py` to point to your file.

## 5. Run the app locally

After training:

```bash
streamlit run app.py
```

In the app you can:

- Choose **Use sample Netflix data (100 users)** to explore the dummy dataset.
- Or choose **Upload your own CSV** (with the same column structure) and then pick any row.

For the selected user, the app shows:

- Input features for that user.
- Predicted churn probability (0–100%).
- A churn / no-churn label.

## 6. Hosting on GitHub + Streamlit Cloud

1. Initialise git in your local folder (if not already):

   ```bash
   git init
   git add .
   git commit -m "Initial Netflix churn app"
   ```

2. Add the GitHub remote for your repo (example):

   ```bash
   git remote add origin https://github.com/kpki11/Netflix_Churn.git
   git branch -M main
   git push -u origin main
   ```

3. Go to Streamlit Community Cloud, sign in with GitHub, and click **New app**.
   - Repository: `kpki11/Netflix_Churn`
   - Branch: `main`
   - Main file: `app.py`

4. Deploy. Streamlit will install dependencies from `requirements.txt` and run `app.py`.

Make sure you have already created and committed the model files (`rf_model.pkl`, `scaler.pkl`, `feature_columns.pkl`, `model_meta.json`) by running `python train_model.py` locally before deploying, so that the app works immediately when someone opens it.
