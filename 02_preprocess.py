"""
preprocessor.py
─────────────────────────────────
One job: take the raw DataFrame and add useful columns.
Import load_data from loader.py first, then pass it here.
─────────────────────────────────
Usage:
    from loader import load_data
    from preprocessor import preprocess

    df = load_data()
    df = preprocess(df)
"""

import pandas as pd


def preprocess(df):
    """
    Takes raw df, returns df with extra feature columns.
    Original columns are never removed or renamed.
    """
    df = df.copy()

    # ── Calendar features ──────────────────────────────
    df["year"]          = df["ds"].dt.year
    df["month_num"]     = df["ds"].dt.month
    df["day_of_month"]  = df["ds"].dt.day
    df["week_of_year"]  = df["ds"].dt.isocalendar().week.astype(int)
    df["quarter"]       = df["ds"].dt.quarter
    df["dow_num"]       = df["ds"].dt.dayofweek   # 0=Mon, 6=Sun

    # ── Business flags ─────────────────────────────────
    df["is_weekend"]    = (df["dow_num"] >= 5).astype(int)
    df["is_month_start"]= df["ds"].dt.is_month_start.astype(int)
    df["is_month_end"]  = df["ds"].dt.is_month_end.astype(int)

    # How many days until/since the GST deadline (20th)
    df["days_to_gst"]   = (20 - df["day_of_month"]).clip(-10, 10)

    # ── Lag features (yesterday, last week, last month) ─
    df["lag_1"]  = df["y"].shift(1).bfill().astype(int)
    df["lag_7"]  = df["y"].shift(7).bfill().astype(int)
    df["lag_30"] = df["y"].shift(30).bfill().astype(int)

    # ── Rolling averages ───────────────────────────────
    # shift(1) so we only use past data — no leakage
    df["rolling_7_mean"]  = df["y"].shift(1).rolling(7,  min_periods=1).mean().bfill()
    df["rolling_30_mean"] = df["y"].shift(1).rolling(30, min_periods=1).mean().bfill()

    print(f"Preprocessed | shape={df.shape} | new cols added: {df.shape[1] - 7}")
    return df


def get_train_test(df, split_date="2024-09-30"):
    """Split into train and test by date."""
    train = df[df["ds"] <= split_date].copy()
    test  = df[df["ds"] >  split_date].copy()
    print(f"Train: {len(train)} rows | Test: {len(test)} rows | split={split_date}")
    return train, test


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    import importlib
    loader = importlib.import_module("01_data_loading")
    load_data = loader.load_data
    df_raw = load_data("msme_cashflow_2yr.csv")
    df_clean = preprocess(df_raw)
    print(df_clean.columns.tolist())
    print(df_clean.head(3).to_string())
    train, test = get_train_test(df_clean)