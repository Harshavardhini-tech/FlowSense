import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import date, timedelta
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from loader import load_data
from preprocessor import preprocess, get_train_test

st.set_page_config(page_title="FlowSense", page_icon="💰", layout="wide")
st.title("💰 FlowSense — Cash Flow Forecasting")
st.caption("AI-Powered Liquidity Risk Manager for MSMEs")
st.info("📌 Model: LightGBM | Explainability: Feature Importance | Enter your revenue data to get forecasts")

# ── SIDEBAR ────────────────────────────────────────────
st.sidebar.title("📋 Enter Your Business Data")
st.sidebar.markdown("---")
st.sidebar.subheader("Recent Revenue (Last 7 Days)")
today = date.today()
user_inputs = []
for i in range(6, -1, -1):
    d = today - timedelta(days=i)
    val = st.sidebar.number_input(
        f"{d.strftime('%a, %d %b')}",
        min_value=0,
        value=50000,
        step=1000,
        key=f"day_{i}"
    )
    user_inputs.append({"ds": pd.Timestamp(d), "y": val})

st.sidebar.markdown("---")
st.sidebar.subheader("Business Flags")
has_festival    = st.sidebar.checkbox("Festival day this week?")
has_market_drop = st.sidebar.checkbox("Market slowdown expected?")
gst_due         = st.sidebar.checkbox("GST filing due soon?")
forecast_days   = st.sidebar.slider("Forecast how many days ahead?", 7, 90, 30)
run_button      = st.sidebar.button("🚀 Generate Forecast", type="primary")

# ── TRAIN MODEL ────────────────────────────────────────
@st.cache_resource
def train_model():
    df          = load_data()
    df          = preprocess(df)
    train, test = get_train_test(df)

    FEATURES = [
        "year", "month_num", "day_of_month", "week_of_year",
        "quarter", "dow_num", "is_weekend", "is_month_start",
        "is_month_end", "days_to_gst", "is_festival",
        "is_market_drop", "gst_filing_flag",
        "lag_1", "lag_7", "lag_30",
        "rolling_7_mean", "rolling_30_mean"
    ]

    X_train = train[FEATURES]
    y_train = train["y"]
    X_test  = test[FEATURES]
    y_test  = test["y"]

    model = LGBMRegressor(
        n_estimators=200, max_depth=6,
        learning_rate=0.05, random_state=42, verbose=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    test  = test.copy()
    test["predicted"] = preds.astype(int)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = np.mean(np.abs((y_test.values - preds) / y_test.values)) * 100

    return df, test, model, mae, rmse, mape, X_test, FEATURES

with st.spinner("Loading and training model..."):
    df, test, model, mae, rmse, mape, X_test, FEATURES = train_model()

# ── KPIs ───────────────────────────────────────────────
st.subheader("📊 Historical Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Daily Revenue", f"₹{df['y'].mean():,.0f}")
col2.metric("Max Revenue Day",   f"₹{df['y'].max():,.0f}")
col3.metric("Min Revenue Day",   f"₹{df['y'].min():,.0f}")
col4.metric("Model Accuracy",    f"{100 - mape:.1f}%")

# ── MODEL PERFORMANCE ──────────────────────────────────
with st.expander("📈 Model Performance Details"):
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE",  f"₹{mae:,.0f}")
    m2.metric("RMSE", f"₹{rmse:,.0f}")
    m3.metric("MAPE", f"{mape:.2f}%")

    fig_test, ax_test = plt.subplots(figsize=(12, 3))
    ax_test.plot(test["ds"], test["y"],        color="steelblue", linewidth=0.8, label="Actual")
    ax_test.plot(test["ds"], test["predicted"], color="red",       linewidth=0.8, label="Predicted")
    ax_test.set_xlabel("Date"); ax_test.set_ylabel("Revenue (₹)"); ax_test.legend()
    fig_test.tight_layout()
    st.pyplot(fig_test)

st.divider()

# ── USER FORECAST ──────────────────────────────────────
if run_button:
    st.subheader("🔮 Your Personalised Forecast")

    user_df = pd.DataFrame(user_inputs)
    last_7  = [r["y"] for r in user_inputs]

    future_dates = pd.date_range(today + timedelta(days=1), periods=forecast_days)
    future = pd.DataFrame({"ds": future_dates})
    future["year"]           = future["ds"].dt.year
    future["month_num"]      = future["ds"].dt.month
    future["day_of_month"]   = future["ds"].dt.day
    future["week_of_year"]   = future["ds"].dt.isocalendar().week.astype(int)
    future["quarter"]        = future["ds"].dt.quarter
    future["dow_num"]        = future["ds"].dt.dayofweek
    future["is_weekend"]     = (future["dow_num"] >= 5).astype(int)
    future["is_month_start"] = future["ds"].dt.is_month_start.astype(int)
    future["is_month_end"]   = future["ds"].dt.is_month_end.astype(int)
    future["days_to_gst"]    = (20 - future["day_of_month"]).clip(-10, 10)
    future["is_festival"]    = int(has_festival)
    future["is_market_drop"] = int(has_market_drop)
    future["gst_filing_flag"]= int(gst_due)
    future["lag_1"]          = last_7[-1]
    future["lag_7"]          = last_7[0]
    future["lag_30"]         = df["y"].iloc[-30]
    future["rolling_7_mean"] = np.mean(last_7)
    future["rolling_30_mean"]= df["y"].iloc[-30:].mean()
    future["predicted"]      = model.predict(future[FEATURES]).astype(int)

    # Forecast KPIs
    f1, f2, f3 = st.columns(3)
    f1.metric("Avg Forecasted Revenue", f"₹{future['predicted'].mean():,.0f}")
    f2.metric("Best Day Forecast",      f"₹{future['predicted'].max():,.0f}")
    f3.metric("Worst Day Forecast",     f"₹{future['predicted'].min():,.0f}")

    # Forecast chart
    fig_f, ax_f = plt.subplots(figsize=(12, 3))
    ax_f.bar(user_df["ds"], user_df["y"], color="steelblue", label="Your Recent Revenue", width=0.6)
    ax_f.plot(future["ds"], future["predicted"], color="green", linewidth=2, marker="o", markersize=3, label="Forecast")
    ax_f.axvline(pd.Timestamp(today), color="gray", linestyle="--", label="Today")
    ax_f.set_xlabel("Date"); ax_f.set_ylabel("Revenue (₹)"); ax_f.legend()
    fig_f.tight_layout()
    st.pyplot(fig_f)

    # Alerts
    st.subheader("🚨 Liquidity Alerts")
    avg_rev  = np.mean(last_7)
    low_days = future[future["predicted"] < avg_rev * 0.7]
    if len(low_days) > 0:
        st.warning(f"⚠️ {len(low_days)} days in your forecast are predicted below 70% of your recent average.")
        alerts_display = low_days[["ds", "predicted"]].copy()
        alerts_display["ds"] = alerts_display["ds"].dt.strftime("%a, %d %b %Y")
        alerts_display["predicted"] = alerts_display["predicted"].apply(lambda x: f"₹{x:,}")
        alerts_display = alerts_display.rename(columns={"ds": "Date", "predicted": "Predicted Revenue"})
        st.dataframe(alerts_display, use_container_width=True)
    else:
        st.success("✅ No major liquidity risks detected in your forecast period.")

    # Forecast table
    with st.expander("📋 Full Forecast Table"):
        forecast_display = future[["ds", "predicted"]].copy()
        forecast_display["ds"] = forecast_display["ds"].dt.strftime("%a, %d %b %Y")
        forecast_display["predicted"] = forecast_display["predicted"].apply(lambda x: f"₹{x:,}")
        forecast_display = forecast_display.rename(columns={"ds": "Date", "predicted": "Predicted Revenue"})
        st.dataframe(forecast_display, use_container_width=True)

else:
    st.info("👈 Enter your revenue data in the sidebar and click 'Generate Forecast' to get started.")

# ── FEATURE IMPORTANCE ─────────────────────────────────
st.divider()
st.subheader("🔍 What Drives Your Revenue?")
st.caption("These are the top factors that influence revenue predictions")

label_map = {
    "lag_1": "Yesterday's Revenue",
    "lag_7": "Last Week Same Day",
    "lag_30": "Last Month Same Day",
    "rolling_7_mean": "7-Day Average",
    "rolling_30_mean": "30-Day Average",
    "is_festival": "Festival Day",
    "is_market_drop": "Market Slowdown",
    "gst_filing_flag": "GST Filing Day",
    "dow_num": "Day of Week",
    "month_num": "Month of Year",
    "day_of_month": "Day of Month",
    "is_weekend": "Weekend",
    "days_to_gst": "Days to GST",
    "week_of_year": "Week of Year",
    "quarter": "Quarter",
    "year": "Year",
    "is_month_start": "Month Start",
    "is_month_end": "Month End",
}

importance = pd.DataFrame({
    "Factor": FEATURES,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=True).tail(10)

importance["Factor"] = importance["Factor"].map(label_map).fillna(importance["Factor"])

fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
ax_imp.barh(importance["Factor"], importance["Importance"], color="steelblue")
ax_imp.set_xlabel("Impact on Revenue Prediction")
ax_imp.set_title("Top 10 Revenue Drivers")
fig_imp.tight_layout()
st.pyplot(fig_imp)

st.divider()
st.caption("FlowSense | Major Project | Harshavardhini | UID: 111723049009")