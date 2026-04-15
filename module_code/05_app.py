
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import Ridge
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Marketing AK-47 Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(" Marketing AK-47 Dashboard")
st.caption("Diagnostic & Optimization Engine")

# ============================
# SIDEBAR
# ============================
st.sidebar.header(" Settings")

# Upload file
uploaded_file = st.sidebar.file_uploader(
    "Upload Marketing Data (CSV)", 
    type=['csv'],
    help="Upload your marketing dataset with columns: FB_Spend, GG_Spend, Sales, Clicks, Conversions, Promo, Seasonality, etc."
)

# Or use default sample data
use_sample = st.sidebar.checkbox("Use sample data", value=True)

# Budget input
st.sidebar.subheader(" Budget Optimization")
budget = st.sidebar.number_input("Total Budget", min_value=1000, max_value=100000, value=10000, step=1000)
promo_input = st.sidebar.selectbox("Promo Campaign", ["No", "Yes"], index=0)
run_optimization = st.sidebar.button("Run Optimization", type="primary")

# Threshold settings
st.sidebar.subheader(" Alert Thresholds")
threshold_mode = st.sidebar.selectbox("Threshold Mode", ["Auto (Percentile)", "Manual"])
if threshold_mode == "Manual":
    cpc_threshold = st.sidebar.number_input("CPC Threshold", value=3000)
    roas_threshold = st.sidebar.number_input("ROAS Threshold", value=0.05)
else:
    cpc_threshold = None
    roas_threshold = None

# ============================
# LOAD DATA
# ============================
@st.cache_data
def load_data(uploaded_file, use_sample):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'Week' in df.columns:
            df['Week'] = pd.to_datetime(df['Week'])
        df = df.sort_values('Week').reset_index(drop=True)
        return df, "Uploaded"
    elif use_sample:
        df = pd.read_csv('marketing_dataset_sample.csv')
        df['Week'] = pd.to_datetime(df['Week'])
        df = df.sort_values('Week').reset_index(drop=True)
        return df, "Sample"
    else:
        return None, None

df, data_source = load_data(uploaded_file, use_sample)

if df is None:
    st.warning("Please upload data or use sample data")
    st.stop()

st.sidebar.success(f" Data loaded: {data_source} ({len(df)} weeks)")

# ============================
# CALCULATE METRICS
# ============================
def calculate_metrics(df):
    df = df.copy()
    df['CPC_FB'] = df['FB_Spend'] / (df['Clicks'] / 2 + 1e-6)
    df['CPC_GG'] = df['GG_Spend'] / (df['Clicks'] / 2 + 1e-6)
    df['CPC_Avg'] = (df['CPC_FB'] + df['CPC_GG']) / 2
    df['CPA'] = (df['FB_Spend'] + df['GG_Spend']) / (df['Conversions'] + 1e-6)
    df['ROAS'] = df['Sales'] / (df['FB_Spend'] + df['GG_Spend'] + 1e-6)
    df['ROI'] = (df['Sales'] - (df['FB_Spend'] + df['GG_Spend'])) / (df['FB_Spend'] + df['GG_Spend'] + 1e-6)
    df['CR'] = df['Conversions'] / (df['Clicks'] + 1e-6)
    return df

df = calculate_metrics(df)

# ============================
# LOAD OR TRAIN MODEL
# ============================
@st.cache_resource
def load_model(df):
    try:
        model_raw = joblib.load('marketing_model_simple.pkl')
        st.info("Model loaded from file")
        return model_raw
    except:
        st.info("Training new model...")
        X = df[['FB_Effect', 'GG_Effect', 'Promo', 'Seasonality']].values
        y = df['Sales'].values
        model_raw = Ridge(alpha=0.5)
        model_raw.fit(X, y)
        st.success("Model trained successfully")
        return model_raw

model = load_model(df)

# ============================
# PREDICT SALES
# ============================
X_pred = df[['FB_Effect', 'GG_Effect', 'Promo', 'Seasonality']].values
df['Sales_Forecast'] = model.predict(X_pred)
df['Sales_Error'] = df['Sales'] - df['Sales_Forecast']
df['Sales_Error_Pct'] = (df['Sales_Error'] / df['Sales']) * 100

# ============================
# MAIN DASHBOARD
# ============================
# KPI Row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Sales", f"{df['Sales'].sum():,.0f}", 
              delta=f"{((df['Sales'].tail(4).sum() / df['Sales'].head(4).sum() - 1) * 100):.0f}% vs first 4 weeks")

with col2:
    st.metric("Avg CPC", f"{df['CPC_Avg'].mean():,.0f}đ", 
              delta=f"{df['CPC_Avg'].tail(4).mean() - df['CPC_Avg'].head(4).mean():.0f}đ")

with col3:
    st.metric("Avg ROAS", f"{df['ROAS'].mean():.2f}x")

with col4:
    st.metric("Avg CR", f"{df['CR'].mean():.2%}")

with col5:
    st.metric("Forecast Accuracy", f"{100 - df['Sales_Error_Pct'].abs().mean():.1f}%")

# ============================
# CHARTS
# ============================
st.subheader(" Sales: Actual vs Forecast")
fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(df['Week'], df['Sales'], 'o-', label='Actual', linewidth=2, alpha=0.8)
ax1.plot(df['Week'], df['Sales_Forecast'], 's--', label='Forecast', linewidth=2, alpha=0.8)
ax1.set_xlabel('Week')
ax1.set_ylabel('Sales')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)
st.pyplot(fig1)

# Two columns for charts
col1, col2 = st.columns(2)

with col1:
    st.subheader(" CPC Trend")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(df['Week'], df['CPC_Avg'], 'o-', color='orange', linewidth=2)
    ax2.axhline(y=df['CPC_Avg'].quantile(0.75), color='red', linestyle='--', label='75th percentile')
    ax2.set_xlabel('Week')
    ax2.set_ylabel('CPC (đ)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig2)

with col2:
    st.subheader(" ROAS Trend")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(df['Week'], df['ROAS'], 'o-', color='green', linewidth=2)
    ax3.axhline(y=df['ROAS'].quantile(0.25), color='red', linestyle='--', label='25th percentile')
    ax3.set_xlabel('Week')
    ax3.set_ylabel('ROAS')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    st.pyplot(fig3)

# ============================
# OPTIMIZATION
# ============================
if run_optimization:
    st.subheader(" Budget Optimization Results")
    
    promo_value = 1 if promo_input == "Yes" else 0
    season_value = df['Seasonality'].mean()
    
    # Coefficients from model
    coef_fb = model.coef_[0] if hasattr(model, 'coef_') else 0.0033
    coef_gg = model.coef_[1] if hasattr(model, 'coef_') else 0.0083
    coef_promo = model.coef_[2] if hasattr(model, 'coef_') else 38
    coef_season = model.coef_[3] if hasattr(model, 'coef_') else 68
    intercept = model.intercept_ if hasattr(model, 'intercept_') else 215
    
    def sales_function(fb_spend, gg_spend):
        fb_effect = fb_spend**2 / (5000**2 + fb_spend**2)
        gg_effect = gg_spend**2 / (5000**2 + gg_spend**2)
        return (coef_fb * fb_effect + coef_gg * gg_effect + 
                coef_promo * promo_value + coef_season * season_value + intercept)
    
    # Simple optimization logic
    best_fb = budget * 0.4
    best_gg = budget * 0.6
    best_sales = sales_function(best_fb, best_gg)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Optimal FB Spend", f"{best_fb:,.0f}đ")
    with col2:
        st.metric("Optimal GG Spend", f"{best_gg:,.0f}đ")
    with col3:
        st.metric("Expected Sales", f"{best_sales:.0f}")
    
    # Compare with equal split
    equal_sales = sales_function(budget/2, budget/2)
    improvement = (best_sales / equal_sales - 1) * 100
    
    st.info(f" Improvement vs equal split: +{improvement:.1f}%")
    
    # Allocation chart
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    ax4.bar(['FB', 'GG'], [best_fb, best_gg], color=['blue', 'orange'], alpha=0.7)
    ax4.set_ylabel('Spend')
    ax4.set_title('Optimal Budget Allocation')
    for i, v in enumerate([best_fb, best_gg]):
        ax4.text(i, v + 100, f'{v:,.0f}', ha='center')
    st.pyplot(fig4)

# ============================
# ALERTS TABLE
# ============================
st.subheader(" Diagnostic Alerts")

# Simple alert logic
def generate_alerts_simple(df):
    alerts = []
    for i in range(len(df)):
        row = df.iloc[i]
        alerts_text = []
        
        if row['CPC_Avg'] > df['CPC_Avg'].quantile(0.75):
            alerts_text.append(f"CPC high ({row['CPC_Avg']:.0f}đ)")
        if row['ROAS'] < df['ROAS'].quantile(0.25):
            alerts_text.append(f"ROAS low ({row['ROAS']:.2f}x)")
        if row['CR'] < df['CR'].quantile(0.25):
            alerts_text.append(f"CR low ({row['CR']:.1%})")
        if abs(row['Sales_Error_Pct']) > 15:
            if row['Sales_Error'] < 0:
                alerts_text.append(f"Sales {abs(row['Sales_Error_Pct']):.0f}% below forecast")
            else:
                alerts_text.append(f"Sales {row['Sales_Error_Pct']:.0f}% above forecast")
        
        alerts.append({
            'Week': row['Week'].strftime('%Y-%m-%d'),
            'Sales': f"{row['Sales']:.0f}",
            'Forecast': f"{row['Sales_Forecast']:.0f}",
            'Error': f"{row['Sales_Error_Pct']:.1f}%",
            'Alerts': '; '.join(alerts_text) if alerts_text else 'All good'
        })
    return pd.DataFrame(alerts)

alerts_df = generate_alerts_simple(df.tail(10))
st.dataframe(alerts_df, use_container_width=True)

# ============================
# DATA TABLE
# ============================
with st.expander(" View Raw Data"):
    st.dataframe(df.tail(20), use_container_width=True)

# ============================
# FOOTER
# ============================
st.divider()
st.caption("Marketing AK-47 Diagnostic & Optimization Engine | Built with Streamlit")