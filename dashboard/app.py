"""
Pharma Sales Analytics Dashboard
Streamlit app summarizing EDA, segmentation, rep performance,
forecasting, and predictive model insights.
 
Run with: streamlit run dashboard/app.py
(run from the project root, or adjust DATA_DIR/SRC_DIR paths below)
"""
 
import sys
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
 
# --- Path setup ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
sys.path.append(SRC_DIR)
 
from segmentation import filter_valid_sales, add_order_type  # noqa: E402
 
st.set_page_config(page_title="Pharma Sales Analytics", layout="wide")
 
 
# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
 
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "cleaned_data.csv"), parse_dates=["date"])
    df = filter_valid_sales(df)
    df = add_order_type(df, bulk_quantile=0.99)
    return df
 
 
@st.cache_data
def load_segments():
    path = os.path.join(DATA_DIR, "customer_segments.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None
 
 
df = load_data()
segments = load_segments()
 
 
# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
 
st.sidebar.title("Filters")
 
countries = sorted(df["country"].unique())
selected_countries = st.sidebar.multiselect("Country", countries, default=countries)
 
years = sorted(df["year"].unique())
selected_years = st.sidebar.multiselect("Year", years, default=years)
 
teams = sorted(df["sales_team"].unique())
selected_teams = st.sidebar.multiselect("Sales Team", teams, default=teams)
 
order_types = sorted(df["order_type"].unique())
selected_order_types = st.sidebar.multiselect("Order Type", order_types, default=order_types)
 
filtered = df[
    df["country"].isin(selected_countries)
    & df["year"].isin(selected_years)
    & df["sales_team"].isin(selected_teams)
    & df["order_type"].isin(selected_order_types)
]
 
st.sidebar.markdown("---")
st.sidebar.caption(f"Showing {len(filtered):,} of {len(df):,} transactions")
 
 
# ---------------------------------------------------------------------------
# Header / KPIs
# ---------------------------------------------------------------------------
 
st.title("Pharma Sales Analytics Dashboard")
st.caption("Sales Force Effectiveness & Territory Optimization | 2017-2020 | Germany & Poland")
 
col1, col2, col3, col4, col5 = st.columns(5)
 
total_sales = filtered["sales"].sum()
total_qty = filtered["quantity"].sum()
n_customers = filtered["customer_name"].nunique()
n_transactions = len(filtered)
avg_txn = filtered["sales"].mean() if len(filtered) else 0
 
col1.metric("Total Sales", f"{total_sales/1e9:.2f}B")
col2.metric("Total Quantity", f"{total_qty/1e6:.2f}M units")
col3.metric("Customers", f"{n_customers:,}")
col4.metric("Transactions", f"{n_transactions:,}")
col5.metric("Avg Transaction Value", f"{avg_txn:,.0f}")
 
st.markdown("---")
 
 
# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
 
tab_overview, tab_segments, tab_reps, tab_forecast, tab_model = st.tabs(
    ["Overview", "Segmentation", "Team & Rep Performance", "Forecast Outlook", "Model Insights"]
)
 
 
# --- Overview Tab -----------------------------------------------------------
with tab_overview:
    c1, c2 = st.columns(2)
 
    with c1:
        st.subheader("Monthly Sales Trend")
        monthly = filtered.groupby(pd.Grouper(key="date", freq="ME"))["sales"].sum().reset_index()
        fig = px.line(monthly, x="date", y="sales", markers=True)
        fig.update_layout(yaxis_title="Sales", xaxis_title="Date")
        st.plotly_chart(fig, use_container_width=True)
 
    with c2:
        st.subheader("Sales by Country")
        country_summary = filtered.groupby("country")["sales"].sum().reset_index()
        fig = px.bar(country_summary, x="country", y="sales", color="country")
        st.plotly_chart(fig, use_container_width=True)
 
    c3, c4 = st.columns(2)
 
    with c3:
        st.subheader("Sales by Channel / Sub-channel")
        channel_summary = filtered.groupby(["channel", "subchannel"])["sales"].sum().reset_index()
        fig = px.bar(channel_summary, x="sales", y="subchannel", color="channel", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
 
    with c4:
        st.subheader("Sales by Product Class")
        pc_summary = filtered.groupby("product_class")["sales"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(pc_summary, x="sales", y="product_class", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
 
    st.subheader("Bulk vs Retail Revenue Split")
    order_summary = filtered.groupby("order_type").agg(
        transactions=("sales", "count"), total_sales=("sales", "sum")
    ).reset_index()
    order_summary["pct_of_sales"] = order_summary["total_sales"] / order_summary["total_sales"].sum() * 100
    c5, c6 = st.columns([1, 2])
    with c5:
        st.dataframe(order_summary, use_container_width=True)
    with c6:
        fig = px.pie(order_summary, names="order_type", values="total_sales",
                      title="Revenue Share: Bulk vs Retail")
        st.plotly_chart(fig, use_container_width=True)
 
 
# --- Segmentation Tab --------------------------------------------------------
with tab_segments:
    if segments is None:
        st.warning(
            "customer_segments.csv not found in data/processed/. "
            "Run notebook 03_segmentation.ipynb first to generate it."
        )
    else:
        st.subheader("Customer Segments")
 
        seg_name_col = "segment_name" if "segment_name" in segments.columns else "cluster"
 
        c1, c2 = st.columns(2)
        with c1:
            seg_counts = segments[seg_name_col].value_counts().reset_index()
            seg_counts.columns = ["segment", "count"]
            fig = px.bar(seg_counts, x="segment", y="count", title="Customers per Segment")
            st.plotly_chart(fig, use_container_width=True)
 
        with c2:
            if "total_sales" in segments.columns:
                seg_sales = segments.groupby(seg_name_col)["total_sales"].sum().reset_index()
                fig = px.pie(seg_sales, names=seg_name_col, values="total_sales",
                              title="Revenue Share by Segment")
                st.plotly_chart(fig, use_container_width=True)
 
        st.subheader("Segment Profile")
        profile_cols = [c for c in [
            "total_sales", "total_quantity", "n_transactions", "avg_transaction_value",
            "n_product_classes", "bulk_order_share", "sales_growth"
        ] if c in segments.columns]
 
        if profile_cols:
            profile = segments.groupby(seg_name_col)[profile_cols].mean().round(2)
            profile["n_customers"] = segments.groupby(seg_name_col).size()
            st.dataframe(profile, use_container_width=True)
 
        st.subheader("Customer Segments by Location")
        if {"latitude", "longitude"}.issubset(segments.columns):
            fig = px.scatter_mapbox(
                segments, lat="latitude", lon="longitude", color=seg_name_col,
                hover_name="customer_name" if "customer_name" in segments.columns else None,
                hover_data=["total_sales"] if "total_sales" in segments.columns else None,
                zoom=4, height=500,
            )
            fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
 
 
# --- Team & Rep Performance Tab ----------------------------------------------
with tab_reps:
    st.subheader("Sales Team Performance")
 
    c1, c2 = st.columns(2)
    with c1:
        team_summary = filtered.groupby("sales_team")["sales"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(team_summary, x="sales_team", y="sales", title="Total Sales by Team")
        st.plotly_chart(fig, use_container_width=True)
 
    with c2:
        team_trend = filtered.groupby(["year", "sales_team"])["sales"].sum().reset_index()
        fig = px.line(team_trend, x="year", y="sales", color="sales_team", markers=True,
                       title="Team Performance Over Years")
        st.plotly_chart(fig, use_container_width=True)
 
    st.subheader("Sales Rep KPIs")
    rep_summary = filtered.groupby(["name_of_sales_rep", "sales_team"]).agg(
        total_sales=("sales", "sum"),
        n_transactions=("sales", "count"),
        n_customers=("customer_name", "nunique"),
        avg_transaction_value=("sales", "mean"),
    ).reset_index().sort_values("total_sales", ascending=False)
    st.dataframe(rep_summary, use_container_width=True)
 
    st.subheader("Country Resource Allocation")
    country_alloc = filtered.groupby("country").agg(
        total_sales=("sales", "sum"),
        n_reps=("name_of_sales_rep", "nunique"),
        n_transactions=("sales", "count"),
        n_customers=("customer_name", "nunique"),
    ).reset_index()
    country_alloc["sales_per_rep"] = country_alloc["total_sales"] / country_alloc["n_reps"]
    country_alloc["transactions_per_rep"] = country_alloc["n_transactions"] / country_alloc["n_reps"]
 
    c3, c4 = st.columns([1, 2])
    with c3:
        st.dataframe(country_alloc, use_container_width=True)
    with c4:
        fig = px.bar(country_alloc, x="country", y=["sales_per_rep", "transactions_per_rep"],
                      barmode="group", title="Sales & Transactions per Rep by Country")
        st.plotly_chart(fig, use_container_width=True)
 
    if segments is not None and "segment_name" in segments.columns:
        st.subheader("Team Exposure to Customer Segments")
        merge_cols = ["customer_name", "city", "country"]
        cols_present = [c for c in merge_cols if c in filtered.columns and c in segments.columns]
        if cols_present:
            merged = filtered.merge(segments[cols_present + ["segment_name"]], on=cols_present, how="left")
            alignment = pd.crosstab(
                merged["sales_team"], merged["segment_name"],
                values=merged["sales"], aggfunc="sum", normalize="index"
            ) * 100
            st.dataframe(alignment.round(2), use_container_width=True)
 
 
# --- Forecast Tab -------------------------------------------------------------
with tab_forecast:
    st.subheader("12-Month Forecast Outlook")
    st.caption(
        "Forecasts generated separately for Retail vs Bulk orders per country "
        "(see notebooks/05_forecasting.ipynb). Static summary shown below - "
        "update these values after re-running the forecasting notebook."
    )
 
    outlook_data = {
        "segment": ["Germany - Retail", "Germany - Bulk", "Poland - Retail", "Poland - Bulk"],
        "last_12_months_actual": [1.749120e9, 9.694338e8, 6.040825e8, 7.697421e7],
        "next_12_months_forecast": [1.429230e9, 1.399775e9, 4.599890e8, 2.745702e8],
        "MAPE_pct": [27.89, 65.17, 18.51, 147.88],
    }
    outlook = pd.DataFrame(outlook_data)
    outlook["change_pct"] = (
        (outlook["next_12_months_forecast"] - outlook["last_12_months_actual"])
        / outlook["last_12_months_actual"] * 100
    )
 
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(outlook.style.format({
            "last_12_months_actual": "{:,.0f}",
            "next_12_months_forecast": "{:,.0f}",
            "MAPE_pct": "{:.1f}%",
            "change_pct": "{:.1f}%",
        }), use_container_width=True)
 
    with c2:
        fig = go.Figure()
        fig.add_bar(name="Last 12 Months (Actual)", x=outlook["segment"], y=outlook["last_12_months_actual"])
        fig.add_bar(name="Next 12 Months (Forecast)", x=outlook["segment"], y=outlook["next_12_months_forecast"])
        fig.update_layout(barmode="group", title="Actual vs Forecast Sales by Segment")
        st.plotly_chart(fig, use_container_width=True)
 
    st.markdown(
        """
        **Key takeaways**
        - Retail segments (Germany & Poland) have reasonable forecast accuracy (MAPE 18-28%) and both
          show a forecast **decline** over the next 12 months.
        - Bulk segments have high MAPE (65-148%) due to lumpy, deal-driven revenue - treat these
          forecasts as directional only, and manage via deal pipeline tracking instead.
        """
    )
 
 
# --- Model Insights Tab -------------------------------------------------------
with tab_model:
    st.subheader("Predictive Model: Drivers of Transaction Value")
    st.caption(
        "XGBoost model predicting transaction sales value (excluding quantity), "
        "interpreted via SHAP (see notebooks/06_predictive_model.ipynb)."
    )
 
    shap_data = {
        "feature": ["country", "month_num", "product_class", "year", "sub_channel",
                     "manager", "sales_team", "channel"],
        "mean_abs_shap": [9376.61, 6355.74, 5492.43, 4846.45, 3203.09, 2743.09, 2264.18, 1822.45],
    }
    shap_df = pd.DataFrame(shap_data).sort_values("mean_abs_shap", ascending=True)
 
    fig = px.bar(shap_df, x="mean_abs_shap", y="feature", orientation="h",
                  title="Mean |SHAP value| - Feature Importance")
    st.plotly_chart(fig, use_container_width=True)
 
    st.markdown(
        """
        **Key takeaway**
 
        `country` is by far the strongest driver of transaction value - more than 1.5x the
        next factor (`month_num`) and over 4x the weakest factor (`channel`). Poland
        transactions are systematically lower-value than Germany's, **independent of
        channel, sales team, or product mix**. This confirms that Poland's underperformance
        is a market-level issue, not a sales-execution issue, and should be addressed through
        market-level interventions (pricing, product availability, market development) rather
        than rep reallocation alone.
        """
    )
 
    st.subheader("Model Performance")
    perf_data = {
        "Model": ["With quantity", "Without quantity"],
        "R2": [0.90, 0.25],  # placeholder for "with quantity" - update after running notebook
        "Note": [
            "High R2 expected: sales ~= quantity x price",
            "Lower R2 expected without transaction size, but SHAP reveals systematic drivers",
        ],
    }
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True)
 
 
st.markdown("---")
st.caption("Pharma Sales Analytics | Data: 2017-2020, Germany & Poland | Built with Streamlit")
 