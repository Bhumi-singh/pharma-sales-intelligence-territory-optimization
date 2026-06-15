"""
Sales Rep / Team Performance Module
Analyzes rep, team, and manager performance, with a focus on
country-level resource allocation (Germany vs Poland) following
the segmentation finding that Poland is an underserved market.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def rep_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute core KPIs per sales rep."""
    kpis = df.groupby(["name_of_sales_rep", "sales_team", "manager"]).agg(
        total_sales=("sales", "sum"),
        total_quantity=("quantity", "sum"),
        n_transactions=("sales", "count"),
        avg_transaction_value=("sales", "mean"),
        n_customers=("customer_name", "nunique"),
        n_cities=("city", "nunique"),
        n_product_classes=("product_class", "nunique"),
    ).reset_index()

    # Country mix per rep
    country_mix = pd.crosstab(df["name_of_sales_rep"], df["country"], normalize="index") * 100
    country_mix = country_mix.add_prefix("pct_sales_").reset_index()
    kpis = kpis.merge(country_mix, on="name_of_sales_rep", how="left")

    # Sales per customer
    kpis["sales_per_customer"] = kpis["total_sales"] / kpis["n_customers"]

    return kpis.sort_values("total_sales", ascending=False)


def yoy_growth_by_rep(df: pd.DataFrame) -> pd.DataFrame:
    """YoY sales growth per rep (first year vs last year in data)."""
    yearly = df.groupby(["name_of_sales_rep", "year"])["sales"].sum().reset_index()
    pivot = yearly.pivot(index="name_of_sales_rep", columns="year", values="sales").fillna(0)
    years = sorted(pivot.columns)
    pivot["growth_pct"] = (pivot[years[-1]] - pivot[years[0]]) / pivot[years[0]].replace(0, np.nan) * 100
    return pivot.reset_index()


def country_resource_allocation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Key analysis: compare reps/teams covering each country vs revenue generated.
    Tells us whether Poland is under-resourced or genuinely a smaller market.
    """
    summary = df.groupby("country").agg(
        total_sales=("sales", "sum"),
        n_reps=("name_of_sales_rep", "nunique"),
        n_teams=("sales_team", "nunique"),
        n_transactions=("sales", "count"),
        n_customers=("customer_name", "nunique"),
        n_cities=("city", "nunique"),
    ).reset_index()

    summary["sales_per_rep"] = summary["total_sales"] / summary["n_reps"]
    summary["sales_per_customer"] = summary["total_sales"] / summary["n_customers"]
    summary["transactions_per_rep"] = summary["n_transactions"] / summary["n_reps"]

    return summary


def rep_country_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """For each rep, how much of their activity is in Germany vs Poland."""
    breakdown = df.groupby(["name_of_sales_rep", "sales_team", "country"]).agg(
        sales=("sales", "sum"),
        transactions=("sales", "count"),
        customers=("customer_name", "nunique"),
    ).reset_index()
    return breakdown


def plot_rep_kpis(kpis: pd.DataFrame, metric: str = "total_sales", top_n: int = 13):
    """Bar chart of a rep-level KPI."""
    fig, ax = plt.subplots(figsize=(10, 6))
    data = kpis.sort_values(metric, ascending=False).head(top_n)
    sns.barplot(data=data, x=metric, y="name_of_sales_rep", hue="sales_team", dodge=False, ax=ax)
    ax.set_title(f"Sales Reps by {metric.replace('_', ' ').title()}")
    plt.tight_layout()
    return fig


def plot_country_allocation(summary: pd.DataFrame):
    """Side-by-side comparison: revenue vs rep count vs sales-per-rep by country."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    sns.barplot(data=summary, x="country", y="total_sales", ax=axes[0])
    axes[0].set_title("Total Sales by Country")

    sns.barplot(data=summary, x="country", y="n_reps", ax=axes[1])
    axes[1].set_title("Number of Reps Covering Country")

    sns.barplot(data=summary, x="country", y="sales_per_rep", ax=axes[2])
    axes[2].set_title("Sales per Rep by Country")

    plt.tight_layout()
    return fig


def plot_rep_country_heatmap(breakdown: pd.DataFrame):
    """Heatmap of sales by rep x country."""
    pivot = breakdown.pivot_table(index="name_of_sales_rep", columns="country", values="sales", fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 8))
    sns.heatmap(pivot, annot=True, fmt=",.0f", cmap="YlGnBu", ax=ax)
    ax.set_title("Sales by Rep x Country")
    plt.tight_layout()
    return fig


def team_segment_alignment(df: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """
    Merge customer segments back into transaction data to see which
    sales teams/reps cover which customer segments (Key Accounts, Rising Stars, etc).
    """
    merge_cols = ["customer_name", "city", "country"]
    merged = df.merge(
        segments[merge_cols + ["segment_name"]],
        on=merge_cols,
        how="left"
    )
    alignment = pd.crosstab(merged["sales_team"], merged["segment_name"], values=merged["sales"],
                             aggfunc="sum", normalize="index") * 100
    return alignment.round(2)
