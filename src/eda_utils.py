"""
EDA Utilities
Reusable functions for exploratory data analysis on pharma sales data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
FIGURES_PATH = "reports/figures"


def load_clean_data(path: str = "data/processed/cleaned_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def save_fig(fig, name: str, path: str = FIGURES_PATH):
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, f"{name}.png"), bbox_inches="tight", dpi=150)


# ---------- Time trends ----------

def sales_trend_over_time(df: pd.DataFrame, freq: str = "ME"):
    """Plot total sales over time (monthly or yearly)."""
    trend = df.groupby(pd.Grouper(key="date", freq=freq))["sales"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=trend, x="date", y="sales", marker="o", ax=ax)
    ax.set_title("Total Sales Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    plt.tight_layout()
    return fig, trend


def yoy_growth(df: pd.DataFrame):
    """Compute year-over-year sales growth."""
    yearly = df.groupby("year")["sales"].sum().reset_index()
    yearly["yoy_growth_pct"] = yearly["sales"].pct_change() * 100
    return yearly


# ---------- Country / Channel ----------

def sales_by_country(df: pd.DataFrame):
    summary = df.groupby("country")["sales"].sum().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=summary, x="country", y="sales", ax=ax)
    ax.set_title("Total Sales by Country")
    plt.tight_layout()
    return fig, summary


def sales_by_channel(df: pd.DataFrame):
    summary = df.groupby(["channel", "subchannel"])["sales"].sum().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=summary, x="sales", y="subchannel", hue="channel", ax=ax)
    ax.set_title("Sales by Channel / Sub-channel")
    plt.tight_layout()
    return fig, summary


# ---------- Product ----------

def sales_by_product_class(df: pd.DataFrame, top_n: int = 10):
    summary = df.groupby("product_class")["sales"].sum().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=summary.head(top_n), x="sales", y="product_class", ax=ax)
    ax.set_title(f"Top {top_n} Product Classes by Sales")
    plt.tight_layout()
    return fig, summary


def top_products(df: pd.DataFrame, top_n: int = 10):
    summary = df.groupby("product_name")["sales"].sum().sort_values(ascending=False).reset_index()
    return summary.head(top_n)


# ---------- Sales Team / Rep / Manager ----------

def sales_by_team(df: pd.DataFrame):
    summary = df.groupby("sales_team")["sales"].sum().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=summary, x="sales_team", y="sales", ax=ax)
    ax.set_title("Total Sales by Sales Team")
    plt.tight_layout()
    return fig, summary


def sales_by_rep(df: pd.DataFrame, top_n: int = 13):
    summary = df.groupby(["name_of_sales_rep", "sales_team"])["sales"].sum().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=summary.head(top_n), x="sales", y="name_of_sales_rep", hue="sales_team", ax=ax, dodge=False)
    ax.set_title("Total Sales by Sales Rep")
    plt.tight_layout()
    return fig, summary


def sales_by_manager(df: pd.DataFrame):
    summary = df.groupby("manager")["sales"].sum().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=summary, x="manager", y="sales", ax=ax)
    ax.set_title("Total Sales by Manager")
    plt.tight_layout()
    return fig, summary


def team_yearly_trend(df: pd.DataFrame):
    """Sales team performance trend across years."""
    summary = df.groupby(["year", "sales_team"])["sales"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=summary, x="year", y="sales", hue="sales_team", marker="o", ax=ax)
    ax.set_title("Sales Team Performance Over Years")
    plt.tight_layout()
    return fig, summary


# ---------- Geography ----------

def geo_scatter(df: pd.DataFrame):
    """Scatter plot of sales by location (lat/long), sized/colored by sales volume."""
    geo = df.groupby(["city", "country", "latitude", "longitude"])["sales"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(geo["longitude"], geo["latitude"], s=geo["sales"] / geo["sales"].max() * 300,
                     c=geo["sales"], cmap="viridis", alpha=0.6)
    ax.set_title("Sales Distribution by Location")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(sc, label="Total Sales")
    plt.tight_layout()
    return fig, geo


# ---------- Summary stats ----------

def overall_summary(df: pd.DataFrame) -> dict:
    return {
        "total_sales": df["sales"].sum(),
        "total_quantity": df["quantity"].sum(),
        "avg_price": df["price"].mean(),
        "n_customers": df["customer_name"].nunique(),
        "n_distributors": df["distributor"].nunique(),
        "n_products": df["product_name"].nunique(),
        "date_range": (df["date"].min(), df["date"].max()),
        "countries": df["country"].unique().tolist(),
        "teams": df["sales_team"].unique().tolist(),
    }
