"""
Segmentation Module
Builds customer/territory-level features and applies K-Means clustering
to identify segments (e.g. key accounts, high-potential retail, underserved territories).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sns.set_style("whitegrid")


def add_order_type(df: pd.DataFrame, bulk_quantile: float = 0.99) -> pd.DataFrame:
    """Flag rows as bulk vs retail based on quantity percentile."""
    threshold = df["quantity"].quantile(bulk_quantile)
    df = df.copy()
    df["order_type"] = np.where(df["quantity"] >= threshold, "bulk", "retail")
    return df


def filter_valid_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Remove negative-sales rows (returns/credits) before aggregation."""
    before = len(df)
    df = df[df["sales"] >= 0].copy()
    print(f"Filtered out {before - len(df)} rows with negative sales")
    return df


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction-level data to customer level for segmentation.
    One row per (customer_name, city, country).
    """
    grp_cols = ["customer_name", "city", "country", "latitude", "longitude"]

    agg = df.groupby(grp_cols).agg(
        total_sales=("sales", "sum"),
        total_quantity=("quantity", "sum"),
        n_transactions=("sales", "count"),
        avg_transaction_value=("sales", "mean"),
        n_product_classes=("product_class", "nunique"),
        n_products=("product_name", "nunique"),
    ).reset_index()

    # Bulk order share per customer
    if "order_type" in df.columns:
        bulk_share = (
            df.groupby(grp_cols)["order_type"]
            .apply(lambda x: (x == "bulk").mean())
            .reset_index(name="bulk_order_share")
        )
        agg = agg.merge(bulk_share, on=grp_cols, how="left")

    # Dominant channel / sub-channel per customer
    channel_mode = (
        df.groupby(grp_cols)["channel"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index(name="primary_channel")
    )
    agg = agg.merge(channel_mode, on=grp_cols, how="left")

    subchannel_mode = (
        df.groupby(grp_cols)["subchannel"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index(name="primary_sub_channel")
    )
    agg = agg.merge(subchannel_mode, on=grp_cols, how="left")

    # Year-over-year trend: compare last available year vs first
    yearly = df.groupby(grp_cols + ["year"])["sales"].sum().reset_index()
    pivot = yearly.pivot_table(index=grp_cols, columns="year", values="sales", fill_value=0)
    years = sorted(pivot.columns)
    if len(years) >= 2:
        pivot["sales_growth"] = (pivot[years[-1]] - pivot[years[0]]) / pivot[years[0]].replace(0, np.nan)
        pivot["sales_growth"] = pivot["sales_growth"].fillna(0)
        growth = pivot[["sales_growth"]].reset_index()
        agg = agg.merge(growth, on=grp_cols, how="left")
    else:
        agg["sales_growth"] = 0

    return agg


def scale_features(df: pd.DataFrame, feature_cols: list):
    """Standardize numeric features for clustering."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])
    return X, scaler


def find_optimal_k(X, k_range=range(2, 8)):
    """Run elbow method and silhouette scores to help choose number of clusters."""
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(k_range), inertias, marker="o")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(list(k_range), silhouettes, marker="o", color="orange")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Score")

    plt.tight_layout()
    return fig, dict(zip(k_range, inertias)), dict(zip(k_range, silhouettes))


def run_kmeans(X, df: pd.DataFrame, k: int, label_col: str = "cluster"):
    """Fit K-Means and attach cluster labels to dataframe."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df = df.copy()
    df[label_col] = km.fit_predict(X)
    return df, km


def profile_clusters(df: pd.DataFrame, feature_cols: list, label_col: str = "cluster") -> pd.DataFrame:
    """Summarize mean feature values per cluster."""
    profile = df.groupby(label_col)[feature_cols].mean()
    profile["n_customers"] = df.groupby(label_col).size()
    profile["pct_of_total_sales"] = (
        df.groupby(label_col)["total_sales"].sum() / df["total_sales"].sum() * 100
    )
    return profile.round(2)


def plot_cluster_geo(df: pd.DataFrame, label_col: str = "cluster"):
    """Scatter plot of clusters on lat/long."""
    fig, ax = plt.subplots(figsize=(8, 8))
    for cluster_id in sorted(df[label_col].unique()):
        sub = df[df[label_col] == cluster_id]
        ax.scatter(sub["longitude"], sub["latitude"], label=f"Cluster {cluster_id}", alpha=0.6, s=40)
    ax.set_title("Customer Segments by Location")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_cluster_scatter(df: pd.DataFrame, x: str, y: str, label_col: str = "cluster"):
    """Scatter plot of two features colored by cluster."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=label_col, palette="tab10", alpha=0.6, ax=ax)
    ax.set_title(f"Clusters: {x} vs {y}")
    plt.tight_layout()
    return fig
