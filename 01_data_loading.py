

import pandas as pd
DATA_PATH = "msme_cashflow_2yr.csv"


def load_data(path=DATA_PATH):
    """
    Load the raw MSME CSV and return a typed DataFrame.
    """
    df = pd.read_csv(path, parse_dates=["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    # Fix types
    df["y"]               = df["y"].astype(int)
    df["is_festival"]     = df["is_festival"].astype(int)
    df["is_market_drop"]  = df["is_market_drop"].astype(int)
    df["gst_filing_flag"] = df["gst_filing_flag"].astype(int)

    print(f"Loaded {len(df)} rows | {df['ds'].min().date()} → {df['ds'].max().date()}")
    return df


def summary(df):
    """Quick look at what's in the data."""
    print("\n── Data Summary ──────────────────────")
    print(f"  Shape     : {df.shape}")
    print(f"  Date range: {df['ds'].min().date()} → {df['ds'].max().date()}")
    print(f"  Revenue   : min=₹{df['y'].min():,}  max=₹{df['y'].max():,}  mean=₹{df['y'].mean():,.0f}")
    print(f"  Festivals : {df['is_festival'].sum()} days")
    print(f"  Drops     : {df['is_market_drop'].sum()} days")
    print(f"  GST days  : {df['gst_filing_flag'].sum()} days")
    print(f"  Nulls     : {df.isnull().sum().sum()}")
    print("─────────────────────────────────────\n")


# Run directly to test
if __name__ == "__main__":
    df = load_data()
    summary(df)
    print(df.head())
