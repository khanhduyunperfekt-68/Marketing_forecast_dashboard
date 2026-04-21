# app.py
# Marketing AK-47 Dashboard - Seasonal Forecast Version
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Marketing AK-47 Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Marketing AK-47 Dashboard")
st.caption("Diagnostic & Optimization Engine - Seasonal Forecast")

# ============================
# LOAD DATA
# ============================
@st.cache_data
def load_data():
    df = pd.read_csv('data_processed_module1.csv')
    return df

df = load_data()

# ============================
# SEASONAL FORECAST
# ============================
def generate_seasonal_forecast(df):
    """Generate forecast using same week from previous year (52-week lag)"""
    df = df.copy()
    df['Sales_Forecast'] = df['Sales'].shift(52)
    avg_sales = df['Sales'].mean()
    df['Sales_Forecast'].fillna(avg_sales, inplace=True)
    df['Forecast_Error'] = df['Sales'] - df['Sales_Forecast']
    df['Forecast_Error_Pct'] = (df['Forecast_Error'] / df['Sales']) * 100
    return df

df = generate_seasonal_forecast(df)

# ============================
# METRICS
# ============================
mae = mean_absolute_error(df['Sales'], df['Sales_Forecast'])
mape = mean_absolute_percentage_error(df['Sales'], df['Sales_Forecast']) * 100

# ============================
# SIDEBAR
# ============================
st.sidebar.header("Settings")

# Alert threshold
error_threshold = st.sidebar.slider(
    "Alert Threshold (%)", 
    min_value=5, 
    max_value=30, 
    value=15, 
    step=5,
    help="Alert when forecast error exceeds this percentage"
)

# Budget optimization section
st.sidebar.subheader("Budget Optimization")
budget = st.sidebar.number_input("Total Budget", min_value=1000, max_value=50000, value=10000, step=1000)
promo = st.sidebar.selectbox("Promo Campaign", ["No", "Yes"], index=0)

# ============================
# MAIN DASHBOARD - KPI ROW
# ============================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Sales", f"{df['Sales'].sum():,.0f}")

with col2:
    st.metric("Avg Weekly Sales", f"{df['Sales'].mean():.0f}")

with col3:
    st.metric("Forecast MAE", f"{mae:.1f}")

with col4:
    st.metric("Forecast MAPE", f"{mape:.1f}%")

with col5:
    st.metric("Alert Threshold", f"±{error_threshold}%")

# ============================
# ACTUAL VS FORECAST CHART
# ============================
st.subheader("Actual vs Forecast Sales")

fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(df['Week'], df['Sales'], 'o-', label='Actual', linewidth=2, markersize=4, alpha=0.8)
ax1.plot(df['Week'], df['Sales_Forecast'], 's--', label='Forecast (Seasonal)', linewidth=2, markersize=4, alpha=0.8)
ax1.set_xlabel('Week')
ax1.set_ylabel('Sales')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)
st.pyplot(fig1)

# ============================
# ERROR CHART
# ============================
st.subheader("Forecast Errors")

fig2, ax2 = plt.subplots(figsize=(12, 4))
colors = ['red' if e < 0 else 'green' for e in df['Forecast_Error']]
ax2.bar(df['Week'], df['Forecast_Error'], color=colors, alpha=0.7)
ax2.axhline(y=0, color='black', linewidth=1)
ax2.axhline(y=error_threshold * df['Sales'].mean() / 100, color='orange', linestyle='--', 
            label=f'Alert threshold (±{error_threshold}%)')
ax2.axhline(y=-error_threshold * df['Sales'].mean() / 100, color='orange', linestyle='--')
ax2.set_xlabel('Week')
ax2.set_ylabel('Error (Actual - Forecast)')
ax2.set_title('Forecast Errors')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)
st.pyplot(fig2)

# ============================
# ALERTS TABLE
# ============================
st.subheader("Alert Dashboard")

# Generate alerts
alerts = []
for i, row in df.iterrows():
    if abs(row['Forecast_Error_Pct']) > error_threshold:
        if row['Forecast_Error'] < 0:
            alert_type = "Below Forecast"
            action = "Review campaign performance"
        else:
            alert_type = "Above Forecast"
            action = "Consider increasing budget"
    else:
        alert_type = "Normal"
        action = "Continue current strategy"
    
    alerts.append({
        'Week': row['Week'],
        'Actual': round(row['Sales']),
        'Forecast': round(row['Sales_Forecast']),
        'Error_%': round(row['Forecast_Error_Pct'], 1),
        'Alert': alert_type,
        'Action': action
    })

alerts_df = pd.DataFrame(alerts)

# Show only weeks with alerts
alerts_triggered = alerts_df[alerts_df['Alert'] != 'Normal']
if len(alerts_triggered) > 0:
    st.warning(f"{len(alerts_triggered)} weeks triggered alerts")
    st.dataframe(alerts_triggered, use_container_width=True)
else:
    st.success("No alerts triggered in the current period")

# ============================
# BUDGET OPTIMIZATION (Simple)
# ============================
if st.sidebar.button("Run Optimization", type="primary"):
    st.subheader("Budget Optimization Results")
    
    # Calculate average ROI
    avg_roi_fb = (df['Sales'] / df['FB_Spend']).mean()
    avg_roi_gg = (df['Sales'] / df['GG_Spend']).mean()
    
    # Find optimal allocation
    best_sales = 0
    best_fb = 0
    best_gg = 0
    
    for fb_pct in range(0, 101, 10):
        fb_spend = budget * fb_pct / 100
        gg_spend = budget - fb_spend
        sales = fb_spend * avg_roi_fb + gg_spend * avg_roi_gg
        
        if sales > best_sales:
            best_sales = sales
            best_fb = fb_spend
            best_gg = gg_spend
            best_pct = fb_pct
    
    # Promo impact
    promo_value = 1 if promo == "Yes" else 0
    avg_promo_lift = df[df['Promo'] == 1]['Sales'].mean() - df[df['Promo'] == 0]['Sales'].mean()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Optimal FB Spend", f"{best_fb:,.0f}")
    with col2:
        st.metric("Optimal GG Spend", f"{best_gg:,.0f}")
    with col3:
        st.metric("Expected Sales", f"{best_sales:.0f}")
    
    st.info(f"Optimal allocation: {best_pct}% FB, {100-best_pct}% GG")
    
    if promo_value == 1:
        st.success(f"Promo lift estimate: +{avg_promo_lift:.0f} sales")
    
    # Allocation chart
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.bar(['FB', 'GG'], [best_fb, best_gg], color=['blue', 'orange'], alpha=0.7)
    ax3.set_ylabel('Spend')
    ax3.set_title('Optimal Budget Allocation')
    for i, v in enumerate([best_fb, best_gg]):
        ax3.text(i, v + 100, f'{v:,.0f}', ha='center')
    st.pyplot(fig3)

# ============================
# DATA TABLE
# ============================
with st.expander("View Raw Data"):
    st.dataframe(df.tail(20), use_container_width=True)

# ============================
# FOOTER
# ============================
st.divider()
st.caption("Marketing AK-47 | Seasonal Forecast | Built with Streamlit")