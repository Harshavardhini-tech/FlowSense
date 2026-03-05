import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import date, timedelta
import sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from loader import load_data
from preprocessor import preprocess, get_train_test

# ── PAGE CONFIG ────────────────────────────────────────
st.set_page_config(page_title="FlowSense", page_icon="💰", layout="wide")

# ── CUSTOM CSS ─────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f8f9fb; }

    /* Title */
    h1 { color: #1a1a2e; font-weight: 800; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: white;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-left: 4px solid #4f8ef7;
    }

    /* Metric label */
    [data-testid="metric-container"] label {
        font-size: 0.78rem !important;
        color: #666 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Metric value */
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #1a1a2e !important;
    }

    /* Green metric border for best */
    .metric-best [data-testid="metric-container"] { border-left-color: #22c55e; }
    /* Red metric border for worst */
    .metric-worst [data-testid="metric-container"] { border-left-color: #ef4444; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid #e8eaed;
    }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f8ef7, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        width: 100%;
        font-size: 1rem;
        transition: opacity 0.2s;
    }
    .stButton > button[kind="primary"]:hover { opacity: 0.9; }

    /* Info/success/warning boxes */
    .stAlert { border-radius: 10px; }

    /* Divider */
    hr { border-color: #e8eaed; }

    /* Expander */
    .streamlit-expanderHeader { font-weight: 600; color: #1a1a2e; }

    /* Section headers */
    h2, h3 { color: #1a1a2e; }

    /* Footer */
    .footer {
        text-align: center;
        color: #999;
        font-size: 0.8rem;
        padding: 20px 0 10px 0;
    }
    .footer strong { color: #4f8ef7; }

    /* Accuracy badge */
    .accuracy-badge {
        display: inline-block;
        background: #dcfce7;
        color: #166534;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ── HEADER ─────────────────────────────────────────────
st.markdown("# 💰 FlowSense — Cash Flow Forecasting")
st.markdown("**AI-Powered Liquidity Risk Manager for MSMEs** · Predict revenue · Spot risks early · Plan smarter")

st.markdown("""
<div style="background: linear-gradient(135deg, #eff6ff, #f5f3ff); border-radius: 10px; padding: 12px 18px; margin: 8px 0 18px 0; border: 1px solid #dbeafe;">
    🚀 <b>Model:</b> LightGBM &nbsp;|&nbsp; 🔍 <b>Explainability:</b> Feature Importance &nbsp;|&nbsp; 📅 <b>Trained on:</b> MSME data Jan 2023 – Dec 2024
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Enter Your Business Data")
    st.markdown("---")
    st.markdown("### 💵 Recent Revenue (Last 7 Days)")
    st.caption("Enter daily revenue in ₹")

    today = date.today()
    user_inputs = []
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        val = st.number_input(
            f"{d.strftime('%a, %d %b')}",
            min_value=0,
            value=50000,
            step=1000,
            key=f"day_{i}"
        )
        user_inputs.append({"ds": pd.Timestamp(d), "y": val})

    st.markdown("---")
    st.markdown("### 🏷️ Business Flags")
    st.caption("These affect the forecast accuracy")
    has_festival    = st.checkbox("🎉 Festival day this week?")
    has_market_drop = st.checkbox("📉 Market slowdown expected?")
    gst_due         = st.checkbox("🧾 GST filing due soon?")

    st.markdown("---")
    st.markdown("### 📆 Forecast Horizon")
    forecast_days = st.slider("Days ahead to forecast", 7, 90, 30,
                               help="Longer forecasts are less precise but useful for planning")
    st.caption(f"Forecasting **{forecast_days} days** from today")

    st.markdown("---")
    run_button = st.button("🚀 Generate Forecast", type="primary")

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
    X_train = train[FEATURES]; y_train = train["y"]
    X_test  = test[FEATURES];  y_test  = test["y"]
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

with st.spinner("⚙️ Training model on MSME data..."):
    df, test, model, mae, rmse, mape, X_test, FEATURES = train_model()

accuracy = 100 - mape

# ── BUSINESS OVERVIEW ──────────────────────────────────
last_7_values = [r["y"] for r in user_inputs]
avg_val  = np.mean(last_7_values)
best_val = max(last_7_values)
worst_val= min(last_7_values)

st.markdown("### 📊 Your Business Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Your 7-Day Avg", f"₹{avg_val:,.0f}")

with col2:
    delta_best = best_val - avg_val
    st.metric("Your Best Day", f"₹{best_val:,.0f}",
              delta=f"+₹{delta_best:,.0f} vs avg" if delta_best > 0 else None)

with col3:
    delta_worst = worst_val - avg_val
    st.metric("Your Worst Day", f"₹{worst_val:,.0f}",
              delta=f"₹{delta_worst:,.0f} vs avg",
              delta_color="inverse")

with col4:
    acc_label = "Excellent" if accuracy >= 85 else "Good" if accuracy >= 70 else "Fair"
    st.metric("Model Accuracy", f"{accuracy:.1f}%",
              delta=f"{acc_label} for time-series",
              delta_color="normal" if accuracy >= 70 else "inverse")

# ── MODEL PERFORMANCE ──────────────────────────────────
with st.expander("📈 Model Performance Details — click to expand"):
    st.caption("These metrics show how well the model predicts on historical test data")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE  (Mean Abs Error)",  f"₹{mae:,.0f}",
              help="On average, predictions are off by this amount")
    m2.metric("RMSE (Root Mean Sq Err)", f"₹{rmse:,.0f}",
              help="Penalises large errors more heavily")
    m3.metric("MAPE (Mean Abs % Err)",  f"{mape:.2f}%",
              help="Percentage error — lower is better")

    st.markdown("**Historical Actual vs Predicted**")
    fig_test, ax_test = plt.subplots(figsize=(12, 3))
    fig_test.patch.set_facecolor('#f8f9fb')
    ax_test.set_facecolor('#f8f9fb')
    ax_test.plot(test["ds"], test["y"],         color="#4f8ef7", linewidth=1.2, label="Actual", alpha=0.9)
    ax_test.plot(test["ds"], test["predicted"],  color="#f97316", linewidth=1.2, label="Predicted", alpha=0.9, linestyle="--")
    ax_test.set_xlabel("Date", fontsize=9); ax_test.set_ylabel("Revenue (₹)", fontsize=9)
    ax_test.legend(fontsize=9); ax_test.spines[['top','right']].set_visible(False)
    fig_test.tight_layout()
    st.pyplot(fig_test)

st.divider()

# ── USER FORECAST ──────────────────────────────────────
if run_button:
    with st.spinner("🔮 Generating your personalised forecast..."):
        user_df  = pd.DataFrame(user_inputs)
        last_7   = [r["y"] for r in user_inputs]
        future_dates = pd.date_range(today + timedelta(days=1), periods=forecast_days)
        future = pd.DataFrame({"ds": future_dates})

        future["year"]            = future["ds"].dt.year
        future["month_num"]       = future["ds"].dt.month
        future["day_of_month"]    = future["ds"].dt.day
        future["week_of_year"]    = future["ds"].dt.isocalendar().week.astype(int)
        future["quarter"]         = future["ds"].dt.quarter
        future["dow_num"]         = future["ds"].dt.dayofweek
        future["is_weekend"]      = (future["dow_num"] >= 5).astype(int)
        future["is_month_start"]  = future["ds"].dt.is_month_start.astype(int)
        future["is_month_end"]    = future["ds"].dt.is_month_end.astype(int)
        future["days_to_gst"]     = (20 - future["day_of_month"]).clip(-10, 10)
        future["is_festival"]     = int(has_festival)
        future["is_market_drop"]  = int(has_market_drop)
        future["gst_filing_flag"] = int(gst_due)

        # ── FIX: rolling lags updated day-by-day ──────────
        rolling_window = list(last_7)
        predictions = []
        for idx, row in future.iterrows():
            row_feat = row.copy()
            row_feat["lag_1"]           = rolling_window[-1]
            row_feat["lag_7"]           = rolling_window[-7] if len(rolling_window) >= 7 else rolling_window[0]
            row_feat["lag_30"]          = df["y"].iloc[-30] if len(rolling_window) < 30 else rolling_window[-30]
            row_feat["rolling_7_mean"]  = np.mean(rolling_window[-7:])
            row_feat["rolling_30_mean"] = np.mean(rolling_window[-30:]) if len(rolling_window) >= 30 else np.mean(rolling_window)
            pred = int(model.predict(pd.DataFrame([row_feat[FEATURES]]))[0])
            predictions.append(pred)
            rolling_window.append(pred)

        future["predicted"] = predictions

    st.markdown("### 🔮 Your Personalised Forecast")

    # Forecast KPIs
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Avg Forecasted Revenue", f"₹{future['predicted'].mean():,.0f}")
    f2.metric("Best Day Forecast",      f"₹{future['predicted'].max():,.0f}",
              delta=f"+₹{future['predicted'].max() - future['predicted'].mean():,.0f} vs avg")
    f3.metric("Worst Day Forecast",     f"₹{future['predicted'].min():,.0f}",
              delta=f"₹{future['predicted'].min() - future['predicted'].mean():,.0f} vs avg",
              delta_color="inverse")
    low_count = len(future[future["predicted"] < np.mean(last_7) * 0.7])
    f4.metric("⚠️ Risk Days", f"{low_count} days",
              delta="below 70% avg" if low_count > 0 else "No risk days",
              delta_color="inverse" if low_count > 0 else "normal")

    # Forecast chart
    fig_f, ax_f = plt.subplots(figsize=(12, 4))
    fig_f.patch.set_facecolor('#f8f9fb')
    ax_f.set_facecolor('#f8f9fb')
    ax_f.bar(user_df["ds"], user_df["y"], color="#4f8ef7", label="Your Recent Revenue", width=0.6, alpha=0.85)
    ax_f.plot(future["ds"], future["predicted"], color="#22c55e", linewidth=2.2,
              marker="o", markersize=3.5, label=f"{forecast_days}-Day Forecast", zorder=5)

    # Shade risk days
    avg_rev = np.mean(last_7)
    for _, row in future[future["predicted"] < avg_rev * 0.7].iterrows():
        ax_f.axvspan(row["ds"] - timedelta(hours=12), row["ds"] + timedelta(hours=12),
                     color="#fee2e2", alpha=0.6, zorder=1)

    ax_f.axvline(pd.Timestamp(today), color="#94a3b8", linestyle="--", linewidth=1.2, label="Today")
    ax_f.set_xlabel("Date", fontsize=9); ax_f.set_ylabel("Revenue (₹)", fontsize=9)
    ax_f.spines[['top','right']].set_visible(False)
    risk_patch = mpatches.Patch(color='#fee2e2', label='⚠️ Risk Day')
    handles, labels = ax_f.get_legend_handles_labels()
    ax_f.legend(handles=handles + [risk_patch], fontsize=9)
    fig_f.tight_layout()
    st.pyplot(fig_f)

    # ── ALERTS ──────────────────────────────────────────
    st.markdown("### 🚨 Liquidity Alerts")
    low_days = future[future["predicted"] < avg_rev * 0.7]
    if len(low_days) > 0:
        st.warning(f"⚠️ **{len(low_days)} days** in your forecast are predicted to fall below **70% of your recent average** (₹{avg_rev * 0.7:,.0f}). Consider maintaining a cash buffer.")
        alerts_display = low_days[["ds", "predicted"]].copy()
        alerts_display["ds"] = alerts_display["ds"].dt.strftime("%a, %d %b %Y")
        alerts_display["Shortfall (₹)"] = (avg_rev - low_days["predicted"]).apply(lambda x: f"₹{x:,.0f}")
        alerts_display["predicted"] = alerts_display["predicted"].apply(lambda x: f"₹{x:,}")
        alerts_display = alerts_display.rename(columns={"ds": "Date", "predicted": "Predicted Revenue"})
        st.dataframe(alerts_display, use_container_width=True, hide_index=True)
    else:
        st.success("✅ **No major liquidity risks** detected in your forecast period. Revenue looks stable!")

    # ── FULL FORECAST TABLE ──────────────────────────────
    with st.expander("📋 Full Forecast Table"):
        forecast_display = future[["ds", "predicted"]].copy()
        forecast_display["ds"] = forecast_display["ds"].dt.strftime("%a, %d %b %Y")
        forecast_display["predicted"] = forecast_display["predicted"].apply(lambda x: f"₹{x:,}")
        forecast_display = forecast_display.rename(columns={"ds": "Date", "predicted": "Predicted Revenue"})
        st.dataframe(forecast_display, use_container_width=True, hide_index=True)

else:
    st.markdown("""
    <div style="background: white; border-radius: 12px; padding: 32px; text-align: center;
                border: 2px dashed #cbd5e1; margin: 20px 0;">
        <div style="font-size: 2.5rem;">🔮</div>
        <h3 style="color: #1a1a2e; margin: 8px 0;">Your Forecast Awaits</h3>
        <p style="color: #64748b;">Enter your revenue data in the sidebar and hit <b>Generate Forecast</b> to see predictions, alerts, and insights.</p>
    </div>
    """, unsafe_allow_html=True)

# ── FEATURE IMPORTANCE ─────────────────────────────────
st.divider()
st.markdown("### 🔍 What Drives Your Revenue?")
st.caption("Top factors the model uses to predict your cash flow")

label_map = {
    "lag_1": "Yesterday's Revenue", "lag_7": "Last Week Same Day",
    "lag_30": "Last Month Same Day", "rolling_7_mean": "7-Day Average",
    "rolling_30_mean": "30-Day Average", "is_festival": "Festival Day",
    "is_market_drop": "Market Slowdown", "gst_filing_flag": "GST Filing Day",
    "dow_num": "Day of Week", "month_num": "Month of Year",
    "day_of_month": "Day of Month", "is_weekend": "Weekend",
    "days_to_gst": "Days to GST", "week_of_year": "Week of Year",
    "quarter": "Quarter", "year": "Year",
    "is_month_start": "Month Start", "is_month_end": "Month End",
}

importance = pd.DataFrame({
    "Factor": FEATURES,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=True).tail(10)
importance["Factor"] = importance["Factor"].map(label_map).fillna(importance["Factor"])

colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(importance)))
fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
fig_imp.patch.set_facecolor('#f8f9fb')
ax_imp.set_facecolor('#f8f9fb')
bars = ax_imp.barh(importance["Factor"], importance["Importance"], color=colors, height=0.6)
ax_imp.set_xlabel("Impact on Revenue Prediction", fontsize=9)
ax_imp.set_title("Top 10 Revenue Drivers", fontweight="bold", fontsize=11)
ax_imp.spines[['top','right']].set_visible(False)
# Add value labels
for bar, val in zip(bars, importance["Importance"]):
    ax_imp.text(bar.get_width() + max(importance["Importance"]) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va='center', fontsize=8, color='#555')
fig_imp.tight_layout()
st.pyplot(fig_imp)

# ── FOOTER ─────────────────────────────────────────────
st.divider()
st.markdown("""
<div class="footer">
    <strong>FlowSense</strong> · AI-Powered Cash Flow Forecasting for MSMEs<br>
    Major Project · Harshavardhini · UID: 111723049009 · Built with LightGBM + Streamlit
</div>
""", unsafe_allow_html=True)