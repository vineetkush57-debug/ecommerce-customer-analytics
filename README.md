# E-Commerce Customer Analytics & Product Association System

An end-to-end Machine Learning, Customer Segmentation, and Market Basket Analytics application built with **Python**, **Scikit-learn**, **Mlxtend**, and **Streamlit**.

---

## 📌 Project Overview

This project implements a complete, practical data science solution for e-commerce transactional data using the **UCI Online Retail dataset**. It moves beyond static reporting to deliver three core machine learning and data mining algorithms:

1. **K-Means Clustering:** RFM (Recency, Frequency, Monetary) customer segmentation to discover distinct behavioral customer segments.
2. **Apriori Association Rule Mining:** Market Basket Analysis to uncover co-purchasing patterns, bundle affinity, and cross-selling opportunities using Support, Confidence, and Lift.
3. **Random Forest Classification:** Supervised classification predicting customer spending tiers (`Low Spending`, `Medium Spending`, `High Spending`) using behavioral signals while **strictly preventing data leakage**.

The project includes an exploratory and modeling **Jupyter Notebook**, saved machine learning models (`.pkl`), and an interactive multi-page **Streamlit Web Application**.

---

## 🚀 Key Features

* **Interactive Commercial Telemetry (Page 1):** Real-time calculation of total revenue, order count, active customers, unique products, and Average Order Value (AOV) with monthly trends and country breakdowns.
* **Customer Behavioral Profiling (Page 2):** Distribution analysis of Recency, Frequency, and Monetary spend, combined with interactive multi-parameter filtering and CSV data export.
* **Unsupervised Customer Segmentation (Page 3):** Standardized, log-transformed RFM clustering with 2D/3D visualizations, persona profiles, Silhouette score evaluation, and on-the-fly model retraining.
* **Market Basket Analysis & Bundle Recommender (Page 4):** Configurable Support, Confidence, and Lift thresholds, top association rules visualization, and an interactive "Customers who bought X also bought Y" recommendation tool.
* **Leak-Free Customer Spending Classifier (Page 5):** Random Forest classifier trained strictly on non-monetary behavioral metrics (Recency, Frequency, Quantity volume, Product diversity, Basket size, Activity span), complete with confusion matrix, feature importances, and a live prediction form.
* **Data Explorer & Quality Audit (Page 6):** Complete audit trail tracking dropped missing records, cancellations, non-positive items, and duplicates, with custom CSV/Excel upload support.

---

## 🛠️ Technologies Used

* **Language:** Python (3.10+)
* **Data Manipulation & Analysis:** Pandas, NumPy, OpenPyXL, PyArrow
* **Machine Learning:** Scikit-learn (KMeans, StandardScaler, RandomForestClassifier, metrics)
* **Association Mining:** Mlxtend (Apriori, Association Rules)
* **Visualizations:** Plotly Express, Plotly Graph Objects, Matplotlib, Seaborn
* **Web Application:** Streamlit
* **Model Serialization:** Joblib
* **Interactive Notebook:** Jupyter, nbformat

---

## 📁 Project File Structure

```text
ecommerce-customer-analytics/
│
├── dataset/
│   ├── online_retail.xlsx               # Primary raw UCI dataset
│   └── online_retail_cleaned.parquet    # High-speed preprocessed cache
│
├── models/
│   ├── kmeans_model.pkl                 # Fitted K-Means clustering model
│   ├── scaler.pkl                       # Fitted StandardScaler for RFM features
│   ├── kmeans_metadata.pkl              # Cluster persona mappings & metrics
│   ├── classification_model.pkl         # Fitted Random Forest classifier
│   └── classification_metadata.pkl      # Feature definitions & thresholds
│
├── notebooks/
│   └── customer_analysis.ipynb          # Step-by-step Jupyter Notebook
│
├── app.py                               # Streamlit Web Application
├── requirements.txt                     # Project Python dependencies
├── README.md                            # Comprehensive documentation
└── .gitignore                           # Git exclusion rules
```

---

## 📊 Dataset Information

* **Name:** UCI Online Retail Dataset
* **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail)
* **Period:** 01/12/2010 to 09/12/2011
* **Attributes:**
  * `InvoiceNo`: 6-digit transaction number (invoices starting with 'C' indicate cancellations).
  * `StockCode`: Unique 5-digit product identifier.
  * `Description`: Product name.
  * `Quantity`: Quantities of each product per transaction.
  * `InvoiceDate`: Transaction generation timestamp.
  * `UnitPrice`: Product unit price (£).
  * `CustomerID`: 5-digit customer identifier.
  * `Country`: Country of customer residence.

---

## 💻 Installation & Setup

### 1. Clone or Open Project Directory

```powershell
cd C:\Users\hp\.gemini\antigravity\scratch\ecommerce-customer-analytics
```

### 2. Create Virtual Environment

Create an isolated Python virtual environment named `venv`:

```powershell
python -m venv venv
```

### 3. Activate Virtual Environment (Windows PowerShell)

```powershell
.\venv\Scripts\Activate.ps1
```

*(If script execution is disabled on your system, enable it for the current session with: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`)*

### 4. Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 📓 Running the Jupyter Notebook

To explore the step-by-step data cleaning, exploratory data analysis, and model training:

```powershell
jupyter notebook notebooks/customer_analysis.ipynb
```

The notebook contains 7 structured, executable steps:
1. **Step 1:** Library imports & reproducible random seed configuration (`random_state=42`).
2. **Step 2:** Ingestion & structural summary (missing values, shapes, column types).
3. **Step 3:** Cleaning pipeline (removal of missing `CustomerID`, cancellations `'C'`, non-positive quantities/prices, and duplicates).
4. **Step 4:** Exploratory Data Analysis (monthly revenue trajectory, top items, geographic share).
5. **Step 5:** RFM feature engineering, log-transformation, Elbow & Silhouette evaluation, K-Means fitting, and persona profiling.
6. **Step 6:** Apriori market basket construction, frequent itemsets, and association rule generation.
7. **Step 7:** Leak-free Random Forest spending classification, confusion matrix, feature importances, and serialization.

---

## 🎯 How the Algorithms Work

### 1. K-Means Customer Clustering
* **Goal:** Group accounts into homogeneous purchasing cohorts without predefined labels.
* **Features Used:**
  * **Recency:** Days since most recent purchase relative to reference snapshot date ($T_{\text{ref}} = \max(\text{Date}) + 1\text{ day}$).
  * **Frequency:** Count of unique invoices purchased.
  * **Monetary:** Total revenue spent across all orders.
  * **Average Order Value (AOV):** Spending per order ($\text{Monetary} / \text{Frequency}$).
* **Why Log Transformation?** RFM metrics have heavy right-skewed tails spanning multiple orders of magnitude. Using $\log(1+x)$ stabilizes variance so extreme spenders do not skew Euclidean distances.
* **Optimal K:** Evaluated via Elbow Method (Within-Cluster Sum of Squares) and Silhouette Scores ($K=4$ selected).
* **Cluster Personas:**
  * **High-Value Champions:** High spend (£8,000+), high frequency (~13 orders), low recency (~20 days).
  * **Frequent Customers:** Reliable re-purchasers with solid monetary contribution.
  * **Potential / Budget Customers:** Moderate frequency and basket size.
  * **At-Risk / Inactive Customers:** Long elapsed recency (>150 days) and low lifetime spend.

### 2. Apriori Association Rule Mining
* **Goal:** Discover which products customers purchase together in the same basket.
* **Key Metrics:**
  * **Support:** $\frac{\text{Orders containing } A \text{ and } B}{\text{Total Orders}}$ (measures prevalence).
  * **Confidence:** $\frac{\text{Support}(A \cup B)}{\text{Support}(A)}$ (probability of buying $B$ given $A$).
  * **Lift:** $\frac{\text{Confidence}(A \to B)}{\text{Support}(B)}$ (strength of affinity; $\text{Lift} > 1$ denotes true complementary association beyond chance).
* **Bundle Suggestions:** Enables dynamic cross-sell recommendations (e.g., matching teacups with teapots, gift bags with wrapping paper).
* **Methodology Caveat:** Association rules indicate empirical co-occurrence, not causal necessity.

### 3. Random Forest Classification (Leakage-Free)
* **Goal:** Classify customers into commercial spending tiers (`Low Spending`, `Medium Spending`, `High Spending`).
* **Critical Scientific Constraint:** The target classes are defined by monetary tertiles. Therefore, **Monetary spend and Average Order Value are strictly excluded as features**.
* **Permitted Behavioral Features:**
  * `Recency`: Elapsed days since last purchase.
  * `Frequency`: Total number of orders placed.
  * `TotalItems`: Total quantity volume of units purchased.
  * `UniqueProducts`: Diversity of unique catalog items explored.
  * `AvgBasketSize`: Average physical units per order ($\text{TotalItems}/\text{Frequency}$).
  * `PurchaseSpan`: Days elapsed between first recorded and most recent order.
* **Performance:** Achieves **~85% out-of-sample accuracy** on stratified 80/20 test split.

---

## 🖥️ Launching the Streamlit Web Application

Run the application with:

```powershell
streamlit run app.py
# Or directly via the virtual environment python:
python -m streamlit run app.py
```

The application will launch automatically in your web browser at:
`http://localhost:8501`

---

## 🔍 Evaluation Metrics Summary

| Algorithm | Primary Metric | Out-of-Sample Score | Interpretation |
| :--- | :--- | :--- | :--- |
| **K-Means Clustering** | Silhouette Score ($K=4$) | **0.2740** | Meaningful separation across log-scaled RFM space. |
| **Random Forest** | Accuracy | **84.56%** | High predictive power using purely behavioral features. |
| **Random Forest** | Macro F1-Score | **0.85** | Balanced performance across Low, Med, and High spending tiers. |
| **Apriori Mining** | Max Lift | **7.35x** | Strong non-random complementary product affinity. |

---

## ⚠️ Limitations & Considerations

1. **Historical Snapshot Nature:** The dataset spans December 2010 to December 2011. Market conditions and product assortments reflect this historical window.
2. **Missing Customer IDs:** Approximately 25% of raw transactions did not have a recorded `CustomerID` (guest checkouts) and were properly filtered from customer-level RFM modeling.
3. **Wholesale vs. Retail Mix:** The dataset includes both individual retail consumers and small wholesale distributors (indicated by high unit volumes).
4. **Association vs. Causation:** Apriori identifies correlation in market baskets; A/B testing is recommended before modifying physical inventory or promotional bundling.

---

## 🔮 Future Improvements

* **Predictive Churn Modeling:** Survival analysis or gradient boosting models to predict probability of churn over a 90-day forward horizon.
* **Customer Lifetime Value (CLV):** Implementation of BG/NBD and Gamma-Gamma probabilistic models.
* **Deep Collaborative Filtering:** Neural matrix factorization and two-tower recommender models for individualized personalization.
* **Automated Scheduled Ingestion:** Connecting the pipeline to a live SQL warehouse or transactional API.
