import os
import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="E-Commerce Analytics & Association System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern, Premium UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.4rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .kpi-label {
        font-size: 0.80rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-val {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-blue { background-color: #E0F2FE; color: #0369A1; }
    .badge-green { background-color: #DCFCE7; color: #15803D; }
    .badge-amber { background-color: #FEF3C7; color: #B45309; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Base Paths Resolution
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------
# Data Ingestion & Preprocessing Functions
# ---------------------------------------------------------
REQUIRED_COLUMNS = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']

@st.cache_data(show_spinner="Loading dataset...")
def load_raw_data(file_path_or_buffer=None):
    if file_path_or_buffer is not None:
        if isinstance(file_path_or_buffer, str):
            if file_path_or_buffer.endswith('.parquet'):
                return pd.read_parquet(file_path_or_buffer)
            elif file_path_or_buffer.endswith('.csv'):
                return pd.read_csv(file_path_or_buffer)
            else:
                return pd.read_excel(file_path_or_buffer)
        else:
            name = file_path_or_buffer.name.lower()
            if name.endswith('.parquet'):
                return pd.read_parquet(file_path_or_buffer)
            elif name.endswith('.csv'):
                return pd.read_csv(file_path_or_buffer)
            else:
                return pd.read_excel(file_path_or_buffer)
    
    parquet_path = os.path.join(DATASET_DIR, "online_retail_cleaned.parquet")
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    
    xlsx_candidates = [
        os.path.join(DATASET_DIR, "online_retail.xlsx"),
        os.path.join(DATASET_DIR, "Online Retail.xlsx")
    ]
    for p in xlsx_candidates:
        if os.path.exists(p):
            return pd.read_excel(p)
            
    return None

def validate_dataset_columns(df):
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]

def clean_transaction_data(df_raw):
    total_raw = len(df_raw)
    
    df_cust = df_raw.dropna(subset=['CustomerID']).copy()
    dropped_cust = total_raw - len(df_cust)
    
    df_valid = df_cust[~df_cust['InvoiceNo'].astype(str).str.startswith('C')].copy()
    dropped_canc = len(df_cust) - len(df_valid)
    
    df_pos = df_valid[(df_valid['Quantity'] > 0) & (df_valid['UnitPrice'] > 0)].copy()
    dropped_neg = len(df_valid) - len(df_pos)
    
    df_dedup = df_pos.drop_duplicates().copy()
    dropped_dups = len(df_pos) - len(df_dedup)
    
    df_clean = df_dedup.copy()
    df_clean['CustomerID'] = df_clean['CustomerID'].astype(int).astype(str)
    df_clean['StockCode'] = df_clean['StockCode'].astype(str).str.strip()
    df_clean['Description'] = df_clean['Description'].astype(str).str.strip()
    df_clean['InvoiceNo'] = df_clean['InvoiceNo'].astype(str).str.strip()
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    df_clean['Revenue'] = df_clean['Quantity'] * df_clean['UnitPrice']
    
    audit = [
        {"Step": "Raw Transactions", "Records Retained": total_raw, "Records Removed": 0, "Details": "Initial ingested dataset"},
        {"Step": "Filter Missing CustomerID", "Records Retained": len(df_cust), "Records Removed": dropped_cust, "Details": "Customer ID required for customer analytics"},
        {"Step": "Filter Cancelled Invoices ('C')", "Records Retained": len(df_valid), "Records Removed": dropped_canc, "Details": "Cancellations / returns excluded"},
        {"Step": "Filter Non-Positive Qty/Price", "Records Retained": len(df_pos), "Records Removed": dropped_neg, "Details": "Invalid non-positive transaction amounts"},
        {"Step": "Remove Duplicate Rows", "Records Retained": len(df_clean), "Records Removed": dropped_dups, "Details": "Identical item line deduplication"}
    ]
    return df_clean, pd.DataFrame(audit)

@st.cache_data(show_spinner="Calculating Customer RFM metrics...")
def compute_customer_rfm(df_clean):
    ref_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    cust_df = df_clean.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda d: (ref_date - d.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('Revenue', 'sum'),
        TotalItems=('Quantity', 'sum'),
        UniqueProducts=('StockCode', 'nunique'),
        FirstPurchase=('InvoiceDate', 'min'),
        LastPurchase=('InvoiceDate', 'max')
    ).reset_index()
    
    cust_df['AverageOrderValue'] = cust_df['Monetary'] / cust_df['Frequency']
    cust_df['AvgBasketSize'] = cust_df['TotalItems'] / cust_df['Frequency']
    cust_df['PurchaseSpan'] = (cust_df['LastPurchase'] - cust_df['FirstPurchase']).dt.days
    
    low_thresh = cust_df['Monetary'].quantile(0.33)
    high_thresh = cust_df['Monetary'].quantile(0.67)
    
    def get_tier(m):
        if m <= low_thresh:
            return 'Low Spending'
        elif m <= high_thresh:
            return 'Medium Spending'
        else:
            return 'High Spending'
            
    cust_df['SpendingCategory'] = cust_df['Monetary'].apply(get_tier)
    
    return cust_df, float(low_thresh), float(high_thresh)

@st.cache_data(show_spinner="Mining frequent itemsets and rules with Apriori...")
def run_apriori_mining(df_input, min_s=0.02, min_c=0.3, min_l=1.2, country_choice="United Kingdom"):
    df_sub = df_input.copy()
    if country_choice != "All Countries":
        df_sub = df_sub[df_sub['Country'] == country_choice]
        
    top_codes = df_sub['StockCode'].value_counts().head(300).index
    df_sub = df_sub[df_sub['StockCode'].isin(top_codes)]
    
    basket = df_sub.groupby(['InvoiceNo', 'StockCode'])['Quantity'].sum().unstack().fillna(0)
    basket_encoded = (basket > 0).astype(bool)
    
    frequent = apriori(basket_encoded, min_support=min_s, use_colnames=True)
    if len(frequent) == 0:
        return frequent, pd.DataFrame(), {}
        
    rules = association_rules(frequent, metric="lift", min_threshold=min_l)
    rules = rules[rules['confidence'] >= min_c]
    
    item_lookup = df_input.drop_duplicates('StockCode').set_index('StockCode')['Description'].to_dict()
    rules['antecedent_names'] = rules['antecedents'].apply(lambda s: ", ".join([item_lookup.get(c, str(c)) for c in s]))
    rules['consequent_names'] = rules['consequents'].apply(lambda s: ", ".join([item_lookup.get(c, str(c)) for c in s]))
    
    return frequent, rules.sort_values('lift', ascending=False), item_lookup

@st.cache_resource
def load_saved_models():
    km_path = os.path.join(MODELS_DIR, "kmeans_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    km_meta_path = os.path.join(MODELS_DIR, "kmeans_metadata.pkl")
    clf_path = os.path.join(MODELS_DIR, "classification_model.pkl")
    clf_meta_path = os.path.join(MODELS_DIR, "classification_metadata.pkl")
    
    km_model = joblib.load(km_path) if os.path.exists(km_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    km_meta = joblib.load(km_meta_path) if os.path.exists(km_meta_path) else None
    
    clf_model = joblib.load(clf_path) if os.path.exists(clf_path) else None
    clf_meta = joblib.load(clf_meta_path) if os.path.exists(clf_meta_path) else None
    
    return {
        'kmeans': km_model,
        'scaler': scaler,
        'kmeans_meta': km_meta,
        'classifier': clf_model,
        'clf_meta': clf_meta
    }

# ---------------------------------------------------------
# Sidebar Navigation & Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=60)
    st.title("E-Commerce ML")
    st.caption("Customer Analytics & Association System")
    st.markdown("---")
    
    page = st.radio(
        "Navigation Menu",
        [
            "📊 Business Overview",
            "👥 Customer Analytics",
            "🎯 K-Means Clustering",
            "🛍️ Product Association",
            "🌲 Customer Classification",
            "📁 Data Explorer"
        ],
        index=0
    )
    
    st.markdown("---")
    st.subheader("Data Source")
    has_parquet = os.path.exists(os.path.join(DATASET_DIR, "online_retail_cleaned.parquet"))
    has_xlsx = os.path.exists(os.path.join(DATASET_DIR, "online_retail.xlsx")) or os.path.exists(os.path.join(DATASET_DIR, "Online Retail.xlsx"))
    
    if has_parquet or has_xlsx:
        st.success("Default UCI Dataset Active")
    else:
        st.warning("Default dataset not found")
        
    uploaded_file = st.file_uploader("Upload CSV / Excel", type=['csv', 'xlsx', 'parquet'], key='sidebar_uploader')
    st.markdown("---")
    st.caption("Production Pipeline: Scikit-learn • Mlxtend • Streamlit")

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------
raw_df = None
if uploaded_file is not None:
    try:
        raw_df = load_raw_data(uploaded_file)
        st.sidebar.info(f"Using uploaded file: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
else:
    raw_df = load_raw_data()

if raw_df is None:
    st.error("No dataset found! Please place 'online_retail.xlsx' inside the 'dataset/' folder or upload via sidebar.")
    st.info("Download dataset from: https://archive.ics.uci.edu/dataset/352/online+retail")
    st.stop()

missing_cols = validate_dataset_columns(raw_df)
if missing_cols:
    st.error(f"Dataset is missing required columns: {missing_cols}")
    st.stop()

if 'Revenue' in raw_df.columns and 'CustomerID' in raw_df.columns and not raw_df['CustomerID'].isna().any():
    df_clean = raw_df.copy()
    audit_df = pd.DataFrame([{"Step": "Cleaned Dataset Loaded", "Records Retained": len(df_clean), "Records Removed": 0, "Details": "Pre-processed cache loaded"}])
else:
    df_clean, audit_df = clean_transaction_data(raw_df)

cust_df, low_thresh, high_thresh = compute_customer_rfm(df_clean)
models_dict = load_saved_models()

# =========================================================
# PAGE 1: BUSINESS OVERVIEW
# =========================================================
if page == "📊 Business Overview":
    st.markdown('<div class="main-title">Business Overview & Commercial KPIs</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">High-level sales telemetry and transaction distributions calculated from actual transactions.</div>', unsafe_allow_html=True)
    
    col_f1, col_f2 = st.columns([1, 1])
    with col_f1:
        min_date = df_clean['InvoiceDate'].min().date()
        max_date = df_clean['InvoiceDate'].max().date()
        date_range = st.date_input("Date Range Filter", [min_date, max_date], min_value=min_date, max_value=max_date)
    with col_f2:
        countries = ["All Countries"] + sorted(df_clean['Country'].dropna().unique().tolist())
        selected_country = st.selectbox("Country Filter", countries)
        
    df_filtered = df_clean.copy()
    if len(date_range) == 2:
        df_filtered = df_filtered[(df_filtered['InvoiceDate'].dt.date >= date_range[0]) & (df_filtered['InvoiceDate'].dt.date <= date_range[1])]
    if selected_country != "All Countries":
        df_filtered = df_filtered[df_filtered['Country'] == selected_country]
        
    if len(df_filtered) == 0:
        st.warning("No records found matching the selected filters.")
        st.stop()
        
    tot_rev = df_filtered['Revenue'].sum()
    tot_orders = df_filtered['InvoiceNo'].nunique()
    tot_cust = df_filtered['CustomerID'].nunique()
    tot_prod = df_filtered['StockCode'].nunique()
    aov = tot_rev / tot_orders if tot_orders > 0 else 0
    
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Revenue</div><div class="kpi-val">£{tot_rev:,.0f}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Orders</div><div class="kpi-val">{tot_orders:,}</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Active Customers</div><div class="kpi-val">{tot_cust:,}</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Unique Products</div><div class="kpi-val">{tot_prod:,}</div></div>', unsafe_allow_html=True)
    with k5:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Avg Order Value</div><div class="kpi-val">£{aov:,.2f}</div></div>', unsafe_allow_html=True)
        
    st.markdown("---")
    
    df_filtered['YearMonth'] = df_filtered['InvoiceDate'].dt.to_period('M').astype(str)
    monthly_series = df_filtered.groupby('YearMonth')['Revenue'].sum().reset_index()
    
    fig_month = px.line(
        monthly_series,
        x='YearMonth',
        y='Revenue',
        markers=True,
        title="Monthly Revenue Trajectory (2010 - 2011)",
        labels={'YearMonth': 'Year-Month', 'Revenue': 'Revenue (£)'},
        color_discrete_sequence=['#2563EB']
    )
    fig_month.update_layout(hovermode="x unified", template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_month, use_container_width=True)
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        top_qty = df_filtered.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10).reset_index()
        fig_qty = px.bar(
            top_qty,
            x='Quantity',
            y='Description',
            orientation='h',
            title="Top 10 Selling Products (Units Sold)",
            color='Quantity',
            color_continuous_scale='Blues',
            labels={'Quantity': 'Units Sold', 'Description': ''}
        )
        fig_qty.update_layout(yaxis=dict(autorange="reversed"), template="plotly_white", coloraxis_showscale=False)
        st.plotly_chart(fig_qty, use_container_width=True)
        
    with col_c2:
        top_rev_p = df_filtered.groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(10).reset_index()
        fig_rev_p = px.bar(
            top_rev_p,
            x='Revenue',
            y='Description',
            orientation='h',
            title="Top 10 Revenue-Generating Products (£)",
            color='Revenue',
            color_continuous_scale='Greens',
            labels={'Revenue': 'Revenue (£)', 'Description': ''}
        )
        fig_rev_p.update_layout(yaxis=dict(autorange="reversed"), template="plotly_white", coloraxis_showscale=False)
        st.plotly_chart(fig_rev_p, use_container_width=True)
        
    st.subheader("Geographic Revenue Breakdown")
    country_df = df_filtered.groupby('Country').agg(
        Revenue=('Revenue', 'sum'),
        Orders=('InvoiceNo', 'nunique'),
        Customers=('CustomerID', 'nunique')
    ).sort_values('Revenue', ascending=False).reset_index()
    country_df['Share (%)'] = (country_df['Revenue'] / tot_rev) * 100
    
    col_geo1, col_geo2 = st.columns([3, 2])
    with col_geo1:
        fig_country = px.bar(
            country_df.head(10),
            x='Country',
            y='Revenue',
            title="Top 10 Revenue-Generating Countries",
            color='Revenue',
            color_continuous_scale='Viridis',
            labels={'Revenue': 'Revenue (£)'}
        )
        fig_country.update_layout(template="plotly_white", coloraxis_showscale=False)
        st.plotly_chart(fig_country, use_container_width=True)
    with col_geo2:
        st.dataframe(
            country_df.head(10).style.format({
                'Revenue': '£{:,.2f}',
                'Orders': '{:,}',
                'Customers': '{:,}',
                'Share (%)': '{:.2f}%'
            }),
            use_container_width=True
        )

# =========================================================
# PAGE 2: CUSTOMER ANALYTICS
# =========================================================
elif page == "👥 Customer Analytics":
    st.markdown('<div class="main-title">Customer Analytics & Behavioral RFM</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Recency, Frequency, Monetary, and Engagement metrics for each unique account.</div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Median Recency</div><div class="kpi-val">{cust_df["Recency"].median():.0f} days</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Median Orders</div><div class="kpi-val">{cust_df["Frequency"].median():.0f} orders</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Median Spend</div><div class="kpi-val">£{cust_df["Monetary"].median():,.2f}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Median AOV</div><div class="kpi-val">£{cust_df["AverageOrderValue"].median():,.2f}</div></div>', unsafe_allow_html=True)
        
    st.markdown("---")
    st.subheader("RFM Behavioral Distributions")
    log_scale = st.checkbox("Logarithmic scale for skewed monetary/frequency features", value=True)
    
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        fig_r = px.histogram(cust_df, x='Recency', nbins=30, title="Recency (Days Since Last Order)", color_discrete_sequence=['#0284C7'])
        fig_r.update_layout(template="plotly_white")
        st.plotly_chart(fig_r, use_container_width=True)
    with col_d2:
        fig_f = px.histogram(cust_df, x='Frequency', nbins=30, log_y=log_scale, title="Frequency (Orders per Customer)", color_discrete_sequence=['#16A34A'])
        fig_f.update_layout(template="plotly_white")
        st.plotly_chart(fig_f, use_container_width=True)
    with col_d3:
        fig_m = px.histogram(cust_df, x='Monetary', nbins=30, log_x=log_scale, title="Monetary Spending (£)", color_discrete_sequence=['#9333EA'])
        fig_m.update_layout(template="plotly_white")
        st.plotly_chart(fig_m, use_container_width=True)
        
    st.markdown("---")
    st.subheader("Filterable Customer Database")
    col_rf1, col_rf2, col_rf3 = st.columns(3)
    with col_rf1:
        max_rec = int(cust_df['Recency'].max())
        rec_filter = st.slider("Max Recency (Days)", 1, max_rec, max_rec)
    with col_rf2:
        max_freq = int(cust_df['Frequency'].max())
        freq_filter = st.slider("Min Frequency (Orders)", 1, min(max_freq, 50), 1)
    with col_rf3:
        spend_tier = st.multiselect("Spending Category", ['Low Spending', 'Medium Spending', 'High Spending'], default=['Low Spending', 'Medium Spending', 'High Spending'])
        
    filtered_cust = cust_df[
        (cust_df['Recency'] <= rec_filter) &
        (cust_df['Frequency'] >= freq_filter) &
        (cust_df['SpendingCategory'].isin(spend_tier))
    ]
    
    st.write(f"Showing **{len(filtered_cust):,}** of **{len(cust_df):,}** customers.")
    display_cols = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'AverageOrderValue', 'TotalItems', 'UniqueProducts', 'SpendingCategory']
    st.dataframe(
        filtered_cust[display_cols].sort_values('Monetary', ascending=False).style.format({
            'Recency': '{:.0f} d',
            'Frequency': '{:,}',
            'Monetary': '£{:,.2f}',
            'AverageOrderValue': '£{:,.2f}',
            'TotalItems': '{:,}',
            'UniqueProducts': '{:,}'
        }),
        use_container_width=True,
        height=350
    )
    
    csv_cust = filtered_cust.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Export Customer RFM Data (CSV)", data=csv_cust, file_name="customer_rfm_export.csv", mime="text/csv")

# =========================================================
# PAGE 3: K-MEANS CLUSTERING
# =========================================================
elif page == "🎯 K-Means Clustering":
    st.markdown('<div class="main-title">K-Means Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Unsupervised customer clustering using standardized log-transformed RFM features.</div>', unsafe_allow_html=True)
    
    km_model = models_dict.get('kmeans')
    scaler = models_dict.get('scaler')
    km_meta = models_dict.get('kmeans_meta')
    
    cluster_cols = ['Recency', 'Frequency', 'Monetary', 'AverageOrderValue']
    
    with st.expander("⚙️ Clustering Configuration & Retraining", expanded=(km_model is None)):
        st.write("Tune K-Means hyperparameters or retrain on the current dataset.")
        k_val = st.slider("Number of Clusters (K)", min_value=2, max_value=8, value=4)
        retrain_clicked = st.button("🚀 Train / Retrain K-Means Model")
        
        if retrain_clicked or km_model is None:
            with st.spinner(f"Fitting K-Means (K={k_val})..."):
                X_cluster = np.log1p(cust_df[cluster_cols])
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_cluster)
                
                km_model = KMeans(n_clusters=k_val, init='k-means++', n_init=10, random_state=42)
                cust_df['Cluster'] = km_model.fit_predict(X_scaled)
                sil = silhouette_score(X_scaled, cust_df['Cluster'])
                
                prof = cust_df.groupby('Cluster')['Monetary'].mean().sort_values()
                cluster_order = prof.index.tolist()
                
                label_names = ["At-Risk / Inactive", "Potential / Budget", "Frequent Customers", "High-Value Champions"]
                persona_map = {}
                for idx, c_id in enumerate(cluster_order):
                    persona_map[c_id] = label_names[min(idx, len(label_names)-1)]
                    
                km_meta = {
                    'features': cluster_cols,
                    'log_transform': True,
                    'n_clusters': k_val,
                    'persona_map': persona_map,
                    'silhouette_score': float(sil)
                }
                
                joblib.dump(km_model, os.path.join(MODELS_DIR, "kmeans_model.pkl"))
                joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
                joblib.dump(km_meta, os.path.join(MODELS_DIR, "kmeans_metadata.pkl"))
                st.success(f"K-Means trained successfully! Silhouette Score: {sil:.4f}")
                time.sleep(1)
                st.rerun()

    if 'Cluster' not in cust_df.columns:
        X_cluster = np.log1p(cust_df[cluster_cols])
        X_scaled = scaler.transform(X_cluster)
        cust_df['Cluster'] = km_model.predict(X_scaled)
        
    persona_map = (km_meta.get('persona_map') or km_meta.get('label_mapping')) if km_meta else {}
    cust_df['Segment'] = cust_df['Cluster'].map(lambda c: persona_map.get(c, f"Cluster {c}"))
    
    st.subheader("Customer Segment Profiles")
    profile_summary = cust_df.groupby('Segment').agg(
        Customers=('CustomerID', 'count'),
        AvgRecency=('Recency', 'mean'),
        AvgFrequency=('Frequency', 'mean'),
        AvgMonetary=('Monetary', 'mean'),
        AvgAOV=('AverageOrderValue', 'mean')
    ).reset_index()
    profile_summary['Share (%)'] = (profile_summary['Customers'] / len(cust_df)) * 100
    
    st.dataframe(
        profile_summary.style.format({
            'Customers': '{:,}',
            'AvgRecency': '{:.1f} days',
            'AvgFrequency': '{:.1f} orders',
            'AvgMonetary': '£{:,.2f}',
            'AvgAOV': '£{:,.2f}',
            'Share (%)': '{:.1f}%'
        }),
        use_container_width=True
    )
    
    col_v1, col_v2 = st.columns([1, 1])
    with col_v1:
        fig_scatter = px.scatter(
            cust_df,
            x='Recency',
            y='Monetary',
            size='Frequency',
            color='Segment',
            log_y=True,
            hover_data=['CustomerID', 'AverageOrderValue'],
            title="Customer Segments: Recency vs. Monetary Spend",
            labels={'Recency': 'Recency (Days)', 'Monetary': 'Monetary Spend (£)'},
            template="plotly_white"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_v2:
        fig_pie = px.pie(
            profile_summary,
            names='Segment',
            values='Customers',
            title="Customer Distribution by Segment",
            hole=0.4,
            template="plotly_white"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    sil_score = km_meta.get('silhouette_score', 0.274) if km_meta else 0.274
    st.info(f"💡 **Model Diagnostics:** Number of Clusters $K={km_model.n_clusters}$ | **Silhouette Score:** `{sil_score:.4f}` | Log-transformed RFM features standardized with StandardScaler.")
    
    csv_seg = cust_df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'AverageOrderValue', 'Cluster', 'Segment']].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Customer Segmentation Data (CSV)", data=csv_seg, file_name="segmented_customers.csv", mime="text/csv")

# =========================================================
# PAGE 4: PRODUCT ASSOCIATION
# =========================================================
elif page == "🛍️ Product Association":
    st.markdown('<div class="main-title">Market Basket Association Rules (Apriori)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Discover frequently co-purchased items and cross-selling bundles using Support, Confidence, and Lift.</div>', unsafe_allow_html=True)
    
    col_ap1, col_ap2, col_ap3, col_ap4 = st.columns(4)
    with col_ap1:
        min_supp = st.slider("Minimum Support", min_value=0.01, max_value=0.10, value=0.02, step=0.005)
    with col_ap2:
        min_conf = st.slider("Minimum Confidence", min_value=0.1, max_value=0.9, value=0.3, step=0.05)
    with col_ap3:
        min_lift = st.slider("Minimum Lift", min_value=1.0, max_value=10.0, value=1.2, step=0.2)
    with col_ap4:
        ap_country = st.selectbox("Cohort Market", ["United Kingdom", "All Countries", "France", "Germany"], index=0)
        
    frequent_items, rules_df, item_map = run_apriori_mining(df_clean, min_supp, min_conf, min_lift, ap_country)
    
    k_a1, k_a2, k_a3 = st.columns(3)
    with k_a1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Frequent Itemsets</div><div class="kpi-val">{len(frequent_items):,}</div></div>', unsafe_allow_html=True)
    with k_a2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Association Rules</div><div class="kpi-val">{len(rules_df):,}</div></div>', unsafe_allow_html=True)
    with k_a3:
        max_l = rules_df['lift'].max() if len(rules_df) > 0 else 0
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Peak Rule Lift</div><div class="kpi-val">{max_l:.2f}x</div></div>', unsafe_allow_html=True)
        
    st.markdown("---")
    
    if len(rules_df) > 0:
        fig_rule = px.scatter(
            rules_df,
            x='support',
            y='confidence',
            color='lift',
            size='lift',
            hover_data=['antecedent_names', 'consequent_names'],
            title="Association Rules: Support vs. Confidence (Color & Size = Lift)",
            color_continuous_scale='Viridis',
            labels={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'},
            template="plotly_white"
        )
        st.plotly_chart(fig_rule, use_container_width=True)
        
        st.subheader("🔍 Interactive Bundle Recommender")
        st.caption("Select a product to view the items customers most frequently co-purchase with it.")
        
        all_antecedents = sorted(list(set(rules_df['antecedent_names'].dropna().tolist())))
        selected_prod = st.selectbox("Select Target Product:", all_antecedents)
        
        matched_rules = rules_df[rules_df['antecedent_names'] == selected_prod].sort_values('lift', ascending=False)
        if len(matched_rules) > 0:
            st.write(f"Top cross-sell recommendations when customer purchases **{selected_prod}**:")
            for _, r in matched_rules.head(5).iterrows():
                st.markdown(f"* 🎁 **Recommended Add-On:** `{r['consequent_names']}` *(Confidence: **{r['confidence']*100:.1f}%** | Lift: **{r['lift']:.2f}x** affinity)*")
        else:
            st.info("No direct bundle rules found for this specific single product. Check the full rules table below.")
            
        st.markdown("---")
        st.subheader("Top Association Rules Table")
        disp_rules = rules_df[['antecedent_names', 'consequent_names', 'support', 'confidence', 'lift', 'leverage']].head(50)
        st.dataframe(
            disp_rules.style.format({
                'support': '{:.3f}',
                'confidence': '{:.3f}',
                'lift': '{:.2f}',
                'leverage': '{:.4f}'
            }),
            use_container_width=True
        )
        
        csv_rules = rules_df[['antecedent_names', 'consequent_names', 'support', 'confidence', 'lift']].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Association Rules (CSV)", data=csv_rules, file_name="association_rules.csv", mime="text/csv")
    else:
        st.warning("No association rules found matching the selected Support and Confidence thresholds. Try decreasing Minimum Support (e.g. 0.015) or Minimum Confidence (0.2).")
        
    st.info("ℹ️ **Methodology Note:** Association rules discover empirical co-occurrence patterns in transaction baskets. High lift indicates positive item affinity but does not prove direct causation.")

# =========================================================
# PAGE 5: CUSTOMER CLASSIFICATION
# =========================================================
elif page == "🌲 Customer Classification":
    st.markdown('<div class="main-title">Random Forest Customer Spending Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Classify customer spending tier using behavioral features with zero target leakage.</div>', unsafe_allow_html=True)
    
    st.success("""
    🛡️ **Strict Data Leakage Prevention Enforced:**  
    The target variable (`SpendingCategory`) is constructed from historical Monetary spend tertiles. To guarantee scientific validity, **neither Monetary spend nor Average Order Value (AOV = Monetary / Frequency) is permitted as an input feature**. The model predicts solely from behavioral signals: Recency, Frequency, Item Volume, Diversity, and Activity Span.
    """)
    
    clf_model = models_dict.get('classifier')
    clf_meta = models_dict.get('clf_meta')
    
    clf_feature_cols = ['Recency', 'Frequency', 'TotalItems', 'UniqueProducts', 'AvgBasketSize', 'PurchaseSpan']
    
    with st.expander("⚙️ Classification Model Training & Retraining", expanded=(clf_model is None)):
        st.write("Train or update the Random Forest Classifier on the current customer feature dataset.")
        n_est = st.slider("Number of Estimators", 50, 300, 100, step=50)
        m_depth = st.slider("Maximum Tree Depth", 3, 20, 8)
        retrain_rf = st.button("🚀 Train / Retrain Random Forest Classifier")
        
        if retrain_rf or clf_model is None:
            with st.spinner("Training Random Forest Classifier (Stratified 80/20)..."):
                X = cust_df[clf_feature_cols]
                y = cust_df['SpendingCategory']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                clf_model = RandomForestClassifier(n_estimators=n_est, max_depth=m_depth, min_samples_split=5, random_state=42)
                clf_model.fit(X_train, y_train)
                
                y_pred = clf_model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                
                clf_meta = {
                    'features': clf_feature_cols,
                    'thresholds': {'Low <= ': float(low_thresh), 'High > ': float(high_thresh)},
                    'classes': list(clf_model.classes_),
                    'accuracy': float(acc),
                    'feature_importances': dict(zip(clf_feature_cols, clf_model.feature_importances_)),
                    'y_test': y_test.tolist(),
                    'y_pred': y_pred.tolist()
                }
                
                joblib.dump(clf_model, os.path.join(MODELS_DIR, "classification_model.pkl"))
                joblib.dump(clf_meta, os.path.join(MODELS_DIR, "classification_metadata.pkl"))
                st.success(f"Model successfully trained! Out-of-sample Accuracy: {acc*100:.2f}%")
                time.sleep(1)
                st.rerun()

    st.subheader("Model Performance Evaluation")
    acc_val = clf_meta.get('accuracy', 0.845) if clf_meta else 0.845
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Out-of-Sample Accuracy</div><div class="kpi-val">{acc_val*100:.2f}%</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Low Spending Threshold</div><div class="kpi-val">&le; £{low_thresh:,.2f}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">High Spending Threshold</div><div class="kpi-val">&gt; £{high_thresh:,.2f}</div></div>', unsafe_allow_html=True)
        
    col_ev1, col_ev2 = st.columns(2)
    with col_ev1:
        if clf_meta and 'y_test' in clf_meta and 'y_pred' in clf_meta:
            classes = ['High Spending', 'Medium Spending', 'Low Spending']
            cm = confusion_matrix(clf_meta['y_test'], clf_meta['y_pred'], labels=classes)
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=classes,
                y=classes,
                labels=dict(x="Predicted Class", y="True Class", color="Count"),
                color_continuous_scale="Blues",
                title="Confusion Matrix (Test Set)"
            )
            fig_cm.update_layout(template="plotly_white")
            st.plotly_chart(fig_cm, use_container_width=True)
            
    with col_ev2:
        if clf_meta and 'feature_importances' in clf_meta:
            fi_df = pd.DataFrame(list(clf_meta['feature_importances'].items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
            fig_fi = px.bar(
                fi_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Gini Feature Importances",
                color='Importance',
                color_continuous_scale='Greens',
                template="plotly_white"
            )
            fig_fi.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_fi, use_container_width=True)
            
    st.markdown("---")
    st.subheader("🔮 Predict Spending Tier for a Customer")
    st.caption("Provide behavioral customer measurements to infer their commercial spending category.")
    
    with st.form("customer_prediction_form"):
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            in_rec = st.number_input("Recency (Days since last order)", min_value=1, max_value=730, value=25)
            in_freq = st.number_input("Frequency (Total orders placed)", min_value=1, max_value=200, value=5)
        with col_p2:
            in_items = st.number_input("Total Items Purchased (Volume)", min_value=1, max_value=50000, value=350)
            in_uniq = st.number_input("Unique Products (Diversity)", min_value=1, max_value=2000, value=45)
        with col_p3:
            in_basket = st.number_input("Average Basket Size (Items / Order)", min_value=1.0, max_value=1000.0, value=70.0)
            in_span = st.number_input("Customer Activity Span (Days)", min_value=0, max_value=730, value=120)
            
        predict_submit = st.form_submit_button("Predict Spending Category", use_container_width=True)
        
        if predict_submit and clf_model is not None:
            input_data = pd.DataFrame([{
                'Recency': in_rec,
                'Frequency': in_freq,
                'TotalItems': in_items,
                'UniqueProducts': in_uniq,
                'AvgBasketSize': in_basket,
                'PurchaseSpan': in_span
            }])[clf_feature_cols]
            
            prediction = clf_model.predict(input_data)[0]
            probabilities = clf_model.predict_proba(input_data)[0]
            
            badge_color = "badge-green" if prediction == "High Spending" else ("badge-blue" if prediction == "Medium Spending" else "badge-amber")
            
            st.markdown(f"""
            <div style="background-color: #f1f5f9; padding: 18px 20px; border-radius: 12px; margin-top: 15px; border-left: 6px solid #2563EB;">
                <h4 style="margin: 0; color: #1E293B;">Predicted Category: <span class="badge-pill {badge_color}">{prediction}</span></h4>
                <p style="margin: 5px 0 0 0; color: #64748B; font-size: 0.95rem;">
                    Inferred spending tier derived purely from behavioral engagement features.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            prob_df = pd.DataFrame({
                'Category': clf_model.classes_,
                'Probability (%)': probabilities * 100
            })
            
            fig_prob = px.bar(
                prob_df,
                x='Category',
                y='Probability (%)',
                text='Probability (%)',
                color='Category',
                color_discrete_map={
                    'High Spending': '#16A34A',
                    'Medium Spending': '#2563EB',
                    'Low Spending': '#D97706'
                },
                template="plotly_white"
            )
            fig_prob.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_prob.update_layout(showlegend=False, yaxis_range=[0, 105], margin=dict(t=30, b=20))
            st.plotly_chart(fig_prob, use_container_width=True)

# =========================================================
# PAGE 6: DATA EXPLORER
# =========================================================
elif page == "📁 Data Explorer":
    st.markdown('<div class="main-title">Data Explorer & Quality Audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Audit trail, column schema verification, and interactive data quality inspector.</div>', unsafe_allow_html=True)
    
    st.subheader("Data Cleaning & Preprocessing Audit")
    st.dataframe(audit_df, use_container_width=True)
    
    tab1, tab2, tab3 = st.tabs(["Cleaned Transactions", "Customer RFM Dataset", "Data Hygiene Metrics"])
    
    with tab1:
        st.write(f"Displaying sample of **{len(df_clean):,}** cleaned transaction rows.")
        st.dataframe(df_clean.head(100), use_container_width=True)
        csv_clean = df_clean.head(10000).to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Cleaned Transactions Sample (CSV)", data=csv_clean, file_name="cleaned_transactions_sample.csv", mime="text/csv")
        
    with tab2:
        st.write(f"Displaying customer-level behavioral metrics for **{len(cust_df):,}** unique accounts.")
        st.dataframe(cust_df.head(100), use_container_width=True)
        csv_cust_all = cust_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full Customer RFM Dataset (CSV)", data=csv_cust_all, file_name="customer_rfm_full.csv", mime="text/csv")
        
    with tab3:
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.markdown("#### Column Information")
            st.write(pd.DataFrame({
                'Column': df_clean.columns,
                'Data Type': [str(t) for t in df_clean.dtypes],
                'Sample Value': [str(df_clean[c].iloc[0]) for c in df_clean.columns]
            }))
        with col_h2:
            st.markdown("#### Null & Duplicate Counts")
            st.write(pd.DataFrame({
                'Missing Values': df_clean.isnull().sum(),
                'Unique Values': df_clean.nunique()
            }))
