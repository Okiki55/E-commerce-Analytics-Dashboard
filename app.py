# app.py - E-commerce Analytics Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="E-commerce Analytics Dashboard", layout="wide")

# ----------------------------
# Paths & helpers
# ----------------------------
paths = {
    "rfm_segments": os.path.join( "rfm_segments.csv"),
    "rfm_table": os.path.join("rfm_table.csv"),
    "cust_churn": os.path.join( "cust_with_churn.csv"),
    "product_nlp": os.path.join( "models", "product_nlp.joblib"),
    "cust_clv": os.path.join("cust_with_clv.csv"),
    "market_rules": os.path.join("market_rules_top50.csv"),
    "cohort": os.path.join( "cohort_retention.csv"),
    "product_clusters": os.path.join("product_clusters.csv"),
    "churn_pipe": os.path.join("churn_pipeline.joblib"),
    "clv_pipe": os.path.join( "clv_reg_pipeline.joblib"),
}

def load_csv_safe(p):
    if os.path.exists(p):
        return pd.read_csv(p)
    return None

# ----------------------------
# Load artifacts (if present)
# ----------------------------
rfm_segments = load_csv_safe(paths["rfm_segments"])
rfm_table = load_csv_safe(paths["rfm_table"])
cust_churn = load_csv_safe(paths["cust_churn"])
cust_clv = load_csv_safe(paths["cust_clv"])
market_rules = load_csv_safe(paths["market_rules"])
cohort = load_csv_safe(paths["cohort"])
product_clusters = load_csv_safe(paths["product_clusters"])

# Load models (optional) - wrapped in try/except
churn_pipe = None
clv_pipe = None
product_nlp = None
if os.path.exists(paths["churn_pipe"]):
    try:
        churn_pipe = joblib.load(paths["churn_pipe"])
    except Exception as e:
        churn_pipe = None

if os.path.exists(paths["clv_pipe"]):
    try:
        clv_pipe = joblib.load(paths["clv_pipe"])
    except Exception as e:
        clv_pipe = None

if os.path.exists(paths["product_nlp"]):
    try:
        product_nlp = joblib.load(paths["product_nlp"])
    except Exception:
        product_nlp = None

# ----------------------------
# Sidebar & navigation
# ----------------------------
st.sidebar.title("E-commerce Analytics")
st.sidebar.markdown("Artifacts folder: `./output/`")
tab = st.sidebar.radio("Choose view", ["Overview", "RFM Segments", "Churn", "CLV", "Recommendations", "Cohorts", "Products"])

st.title("E-commerce Analytics Dashboard")
st.markdown("Explore customer segments, churn risk, CLV, product recommendations, and cohort retention.")

# ----------------------------
# Overview tab
# ----------------------------
if tab == "Overview":
    st.header("Project summary & available artifacts")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Available CSV artifacts")
        for k, p in paths.items():
            if k.endswith(("pipe","product_nlp")):
                continue
            exists = os.path.exists(p)
            st.write(f"- {os.path.basename(p)} — {'✅' if exists else '❌'}")
    with col2:
        st.subheader("Models present")
        st.write(f"- churn pipeline: {'✅' if churn_pipe is not None else '❌'}")
        st.write(f"- clv pipeline: {'✅' if clv_pipe is not None else '❌'}")
        st.write(f"- product nlp: {'✅' if product_nlp is not None else '❌'}")

    st.markdown("---")
    st.markdown("**Quick actions**")
    st.write("• Use the RFM tab to inspect segments. • Use Churn / CLV to find at-risk or high-value customers. • Use Recommendations to find product associations.")

# ----------------------------
# RFM Segments tab
# ----------------------------
elif tab == "RFM Segments":
    st.header("RFM Segmentation")
    if rfm_segments is None:
        st.error("`rfm_segments.csv` not found in output folder.")
    else:
        st.dataframe(rfm_segments.head(200))

        st.markdown("### Segment counts")
        seg_counts = rfm_segments["SegmentName"].value_counts().rename_axis("segment").reset_index(name="count")
        st.bar_chart(seg_counts.set_index("segment")["count"])

        st.markdown("### Segment profiling (mean values)")
        profile = rfm_segments.groupby("SegmentName")[["Recency","Frequency","Monetary"]].mean().round(2)
        st.dataframe(profile)

        st.markdown("### PCA scatter (2D) of segments")
        if "pc1" in rfm_segments.columns and "pc2" in rfm_segments.columns:
            fig, ax = plt.subplots(figsize=(8,5))
            sns.scatterplot(data=rfm_segments, x="pc1", y="pc2", hue="SegmentName", ax=ax, s=40, alpha=0.7)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            st.pyplot(fig)
        else:
            st.info("pc1/pc2 columns not found — PCA coords not available. Use the pipeline script to save them.")

# ----------------------------
# Churn tab
# ----------------------------
elif tab == "Churn":
    st.header("Churn risk & top at-risk customers")
    if cust_churn is None:
        st.error("`cust_with_churn.csv` not found.")
    else:
        st.write("Top customers by churn probability")
        cust_churn_sorted = cust_churn.sort_values("churn_proba", ascending=False).head(100)
        st.dataframe(cust_churn_sorted[["CustomerID","recency_days","frequency","monetary_total","churn_proba","churn"]].head(50))

        st.markdown("### Churn distribution")
        fig, ax = plt.subplots(figsize=(6,3))
        sns.histplot(cust_churn["churn_proba"].dropna(), bins=30, ax=ax)
        ax.set_xlabel("Churn Probability")
        st.pyplot(fig)

        st.markdown("### Filter & inspect one customer")
        cid = st.number_input("CustomerID", min_value=int(cust_churn["CustomerID"].min()), max_value=int(cust_churn["CustomerID"].max()), value=int(cust_churn["CustomerID"].iloc[0]))
        sel = cust_churn[cust_churn["CustomerID"]==cid]
        if sel.shape[0] == 0:
            st.warning("Customer not found in churn file.")
        else:
            st.write(sel.T)

        if churn_pipe is not None:
            st.success("Churn model loaded — you may run on a feature vector to re-check predictions.")
            if st.button("Show sample model prediction (first row)"):
                sample = cust_churn.head(1)[churn_pipe.named_steps['pre'].transformers[0][2]]
                # It's hard to generically run through pipeline inside UI without consistent column ordering,
                # so we just show the stored churn_proba from the file instead.

# ----------------------------
# CLV tab
# ----------------------------
elif tab == "CLV":
    st.header("Customer Lifetime Value (CLV) & predicted future value")
    if cust_clv is None:
        st.error("`cust_with_clv.csv` not found.")
    else:
        st.markdown("### CLV distribution")
        fig, ax = plt.subplots(figsize=(8,3))
        sns.histplot(cust_clv["CLV_score"].replace([np.inf, -np.inf], np.nan).dropna(), bins=50, ax=ax)
        ax.set_xlabel("CLV Score")
        st.pyplot(fig)

        st.markdown("### Top High CLV Customers")
        top = cust_clv.sort_values("CLV_score", ascending=False).head(50)
        st.dataframe(top[["CustomerID","CLV_score","predicted_future_value","churn_proba"]])

        st.markdown("### Compare churn vs CLV")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(data=cust_clv, x="churn_proba", y="CLV_score", hue="CLV_score", palette="viridis", alpha=0.6)
        ax.set_xlabel("Churn Probability"); ax.set_ylabel("CLV Score")
        st.pyplot(fig)

        if clv_pipe is not None:
            st.write("CLV regression model loaded for predictions.")
            sample_row = cust_clv.head(1)
            st.write(sample_row.T)

# ----------------------------
# Recommendations tab
# ----------------------------
elif tab == "Recommendations":
    st.header("Product Recommendations (Market Basket)")
    if market_rules is None:
        st.error("`market_rules_top50.csv` not found.")
    else:
        st.write("Top association rules (preview):")
        st.dataframe(market_rules.head(30))

        st.markdown("### Recommend by product")
        # The CSV produced by apriori stores antecedents/consequents as strings; we need to parse them
        # Try to load antecedents column; if they are stored as frozensets or strings, we adapt.
        try:
            # If antecedents column exists as text like "{'a'}"
            market_rules["antecedents_str"] = market_rules["antecedents"].astype(str)
            products_all = sorted({p.strip().strip("frozenset(){}'") for row in market_rules["antecedents_str"] for p in row.split(",") if p.strip()})
        except Exception:
            products_all = None

        if products_all:
            prod = st.selectbox("Select product (example token)", products_all)
            st.markdown("Recommendations (top rules where selected product is in antecedents):")
            # Show rules that include the product text
            mask = market_rules["antecedents_str"].str.contains(prod, na=False)
            st.dataframe(market_rules[mask][["antecedents","consequents","support","confidence","lift"]].head(10))
        else:
            st.info("Product parsing from rules failed. Inspect `market_rules_top50.csv` for the rule format.")

# ----------------------------
# Cohorts tab
# ----------------------------
elif tab == "Cohorts":
    st.header("Cohort retention heatmap")
    if cohort is None:
        st.error("`cohort_retention.csv` not found.")
    else:
        df_cohort = cohort.copy()
        # Attempt to parse index/columns if imported as strings
        try:
            # If first column got read as index label, ensure formatted matrix
            cohort_mat = pd.read_csv(paths["cohort"], index_col=0)
        except Exception:
            cohort_mat = cohort

        st.write("Cohort retention snapshot (first 12x12):")
        display_mat = cohort_mat.iloc[:12, :12]
        st.dataframe(display_mat.round(3))

        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(display_mat.astype(float), cmap="Blues", annot=False, ax=ax)
        st.pyplot(fig)

# ----------------------------
# Products tab
# ----------------------------
elif tab == "Products":
    st.header("Product clusters (from descriptions)")
    if product_clusters is None:
        st.error("`product_clusters.csv` not found.")
    else:
        st.write("Sample product clusters:")
        st.dataframe(product_clusters.head(200))

        st.markdown("Show cluster group")
        cluster_id = st.number_input("Cluster ID", min_value=int(product_clusters["prod_cluster"].min()), max_value=int(product_clusters["prod_cluster"].max()), value=int(product_clusters["prod_cluster"].min()))
        sel = product_clusters[product_clusters["prod_cluster"]==cluster_id]
        st.write(f"Products in cluster {cluster_id} (sample):")
        st.dataframe(sel.head(200))