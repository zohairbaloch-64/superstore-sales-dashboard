
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Superstore Sales Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)

@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    df["order_date"] = pd.to_datetime(df["order_date"], dayfirst=True, errors="coerce")
    df["ship_date"] = pd.to_datetime(df["ship_date"], dayfirst=True, errors="coerce")
    df["sales"] = (
        df["sales"].astype(str).str.replace(",", "", regex=False).astype(float)
    )
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce", downcast="integer")
    df["discount"] = pd.to_numeric(df["discount"], errors="coerce")
    df["shipping_cost"] = pd.to_numeric(df["shipping_cost"], errors="coerce")
    df["profit_margin"] = df["profit"] / df["sales"]
    df["profit_margin"] = df["profit_margin"].replace([np.inf, -np.inf], np.nan)
    df["discount_pct"] = df["discount"] * 100
    df["order_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    df["order_quarter"] = df["order_date"].dt.to_period("Q").dt.to_timestamp()
    df["order_dayofweek"] = df["order_date"].dt.day_name()
    df["is_loss"] = df["profit"] < 0
    return df

def format_currency(value):
    if pd.isna(value):
        return "-"
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"${value/1_000:,.2f}K"
    return f"${value:,.2f}"

def filter_data(df):
    min_date = df["order_date"].min()
    max_date = df["order_date"].max()
    date_range = st.sidebar.date_input(
        "Order Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    regions = sorted(df["region"].dropna().unique())
    categories = sorted(df["category"].dropna().unique())
    segments = sorted(df["segment"].dropna().unique())

    selected_region = st.sidebar.multiselect("Region", options=regions, default=regions)
    selected_category = st.sidebar.multiselect("Category", options=categories, default=categories)
    selected_segment = st.sidebar.multiselect("Segment", options=segments, default=segments)

    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = df["order_date"].between(pd.to_datetime(start_date), pd.to_datetime(end_date))
    else:
        mask = pd.Series(True, index=df.index)

    if selected_region:
        mask &= df["region"].isin(selected_region)
    if selected_category:
        mask &= df["category"].isin(selected_category)
    if selected_segment:
        mask &= df["segment"].isin(selected_segment)

    return df.loc[mask].copy()

def build_kpis(df):
    total_sales = df["sales"].sum()
    total_profit = df["profit"].sum()
    total_orders = df["order_id"].nunique()
    avg_margin = df["profit_margin"].mean()

    c1, c2, c3, c4 = st.columns(4, gap="large")
    c1.metric("💰 Total Sales", format_currency(total_sales))
    c2.metric("📈 Total Profit", format_currency(total_profit))
    c3.metric("🧾 Total Orders", f"{total_orders:,}")
    c4.metric(
        "📊 Avg Profit Margin",
        f"{avg_margin:.1%}" if not pd.isna(avg_margin) else "-",
        delta=None,
        delta_color="normal",
    )
    if total_profit < 0:
        st.warning("Current filters return negative profit. Adjust the filters or focus on higher-margin categories.")

def draw_sales_trend(df):
    trend = (
        df.dropna(subset=["order_month"])
        .groupby("order_month")
        .agg(sales=("sales", "sum"), profit=("profit", "sum"))
        .reset_index()
    )
    fig = px.line(
        trend,
        x="order_month",
        y=["sales", "profit"],
        markers=True,
        title="Sales and Profit Trend",
        labels={"value": "Amount (USD)", "order_month": "Order Month", "variable": "Metric"},
    )
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), legend_title_text="Metric")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** A strong business line shows both sales and profit rising together. Divergence indicates margin pressure.")

def draw_profit_by_category(df):
    category = (
        df.groupby("category")
        .agg(total_profit=("profit", "sum"))
        .reset_index()
        .sort_values("total_profit", ascending=False)
    )
    fig = px.bar(
        category,
        x="category",
        y="total_profit",
        title="Profit by Category",
        labels={"total_profit": "Total Profit", "category": "Category"},
        color="total_profit",
        color_continuous_scale="Teal",
    )
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

def draw_region_sales(df):
    region = (
        df.groupby("region")
        .agg(total_sales=("sales", "sum"), total_profit=("profit", "sum"))
        .reset_index()
        .sort_values("total_sales", ascending=True)
    )
    fig = px.bar(
        region,
        x="total_sales",
        y="region",
        orientation="h",
        title="Region-wise Sales",
        labels={"total_sales": "Total Sales", "region": "Region"},
        color="total_profit",
        color_continuous_scale="Blues",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** Strong regional sales with weak profit may require local pricing or operational review.")


def draw_discount_profit(df):
    discount_bins = [0, 5, 10, 15, 20, 25, 50, 100]
    discount_labels = ["0-5%", "5-10%", "10-15%", "15-20%", "20-25%", "25-50%", "50-100%"]
    discount_df = df.dropna(subset=["discount_pct"]).copy()
    discount_df["discount_band"] = pd.cut(
        discount_df["discount_pct"], bins=discount_bins, labels=discount_labels, include_lowest=True
    )
    discount_summary = (
        discount_df.groupby("discount_band")
        .agg(avg_margin=("profit_margin", "mean"), total_sales=("sales", "sum"))
        .reset_index()
        .dropna()
    )
    fig = px.bar(
        discount_summary,
        x="discount_band",
        y="avg_margin",
        title="Discount Band vs Average Profit Margin",
        labels={"discount_band": "Discount Band", "avg_margin": "Average Profit Margin"},
        color="avg_margin",
        color_continuous_scale="OrRd",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** This chart reveals how deeper discounts impact profitability.")


def draw_top_customers(df):
    customers = (
        df.groupby("customer_name")
        .agg(total_sales=("sales", "sum"), total_profit=("profit", "sum"))
        .reset_index()
        .sort_values("total_sales", ascending=False)
        .head(10)
    )
    fig = px.bar(
        customers,
        x="total_sales",
        y="customer_name",
        orientation="h",
        title="Top 10 Customers by Revenue",
        labels={"total_sales": "Sales", "customer_name": "Customer"},
        color="total_profit",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** Top customers drive a significant portion of revenue and should be prioritized for retention.")

def draw_profit_vs_sales(df):
    product_perf = (
        df.groupby("product_name")
        .agg(total_sales=("sales", "sum"), total_profit=("profit", "sum"), avg_margin=("profit_margin", "mean"))
        .reset_index()
        .sort_values("total_sales", ascending=False)
        .head(80)
    )
    product_perf["size_margin"] = product_perf["avg_margin"].abs()
    fig = px.scatter(
        product_perf,
        x="total_sales",
        y="total_profit",
        size="size_margin",
        color="avg_margin",
        hover_data=["product_name"],
        title="Profit vs Sales by Top Products",
        labels={"total_sales": "Sales", "total_profit": "Profit", "avg_margin": "Avg Profit Margin", "size_margin": "Abs Avg Profit Margin"},
        color_continuous_scale="RdYlGn",
        size_max=18,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** This filtered view keeps the chart actionable and highlights products where revenue is not matching profit.")

def draw_loss_table(df):
    losses = (
        df.loc[df["profit"] < 0, ["product_name", "category", "region", "sales", "profit", "profit_margin"]]
        .sort_values("profit")
        .head(15)
    )
    if losses.empty:
        st.info("No loss-making products found for the selected filters.")
        return
    losses_display = losses.copy()
    losses_display["sales"] = losses_display["sales"].map(format_currency)
    losses_display["profit"] = losses_display["profit"].map(format_currency)
    losses_display["profit_margin"] = losses_display["profit_margin"].map(
        lambda x: f"{x:.1%}" if pd.notna(x) else "-"
    )
    st.subheader("Loss-Making Products")
    st.dataframe(losses_display, use_container_width=True)

def draw_advanced_analysis(df):
    high_sales_threshold = df["sales"].quantile(0.90)
    low_margin_threshold = df["profit_margin"].quantile(0.25)
    high_sales_low_profit = (
        df.loc[(df["sales"] >= high_sales_threshold) & (df["profit_margin"] <= low_margin_threshold)]
        .groupby("product_name")
        .agg(total_sales=("sales", "sum"), total_profit=("profit", "sum"), avg_margin=("profit_margin", "mean"))
        .reset_index()
        .sort_values("total_sales", ascending=False)
        .head(15)
    )
    loss_categories = (
        df.groupby("category")
        .agg(total_profit=("profit", "sum"), total_sales=("sales", "sum"))
        .reset_index()
        .assign(loss_ratio=lambda x: x["total_profit"] / x["total_sales"])
        .sort_values("loss_ratio")
    )
    if not loss_categories[loss_categories["total_profit"] < 0].empty:
        st.warning("Loss-making category detected. Review pricing, discounts, and operational costs in these segments.")
    if high_sales_low_profit.empty:
        st.info("No high-sales, low-profit products were detected for the current filters.")
    else:
        high_sales_low_profit_display = high_sales_low_profit.copy()
        high_sales_low_profit_display["total_sales"] = high_sales_low_profit_display["total_sales"].map(format_currency)
        high_sales_low_profit_display["total_profit"] = high_sales_low_profit_display["total_profit"].map(format_currency)
        high_sales_low_profit_display["avg_margin"] = high_sales_low_profit_display["avg_margin"].map(
            lambda x: f"{x:.1%}" if pd.notna(x) else "-"
        )
        st.subheader("High Sales, Low Profit Products")
        st.dataframe(high_sales_low_profit_display, use_container_width=True)
    loss_categories_display = loss_categories.copy()
    loss_categories_display["total_sales"] = loss_categories_display["total_sales"].map(format_currency)
    loss_categories_display["total_profit"] = loss_categories_display["total_profit"].map(format_currency)
    loss_categories_display["loss_ratio"] = loss_categories_display["loss_ratio"].map(
        lambda x: f"{x:.1%}" if pd.notna(x) else "-"
    )
    st.subheader("Category Profitability Summary")
    st.dataframe(loss_categories_display, use_container_width=True)

def main():
    st.markdown("# Superstore Sales Dashboard")
    st.markdown("#### A polished analytics experience for revenue, profit, and customer performance.")
    st.sidebar.header("Dashboard Filters")
    st.sidebar.markdown("Use the sidebar to update date range, region, category, and segment filters.")

    df = load_data("SuperStoreOrders.csv")
    filtered_df = filter_data(df)

    if filtered_df.empty:
        st.warning("No data available for the selected filters. Adjust the filters to restore insights.")
        return

    build_kpis(filtered_df)
    st.markdown("---")

    overview_tab, customer_tab, advanced_tab = st.tabs(["Overview", "Customer Insights", "Profitability"])

    with overview_tab:
        st.subheader("Trends and Category Performance")
        draw_sales_trend(filtered_df)
        st.markdown("---")
        left, right = st.columns(2, gap="large")
        with left:
            draw_profit_by_category(filtered_df)
        with right:
            draw_discount_profit(filtered_df)
        st.markdown("---")
        draw_region_sales(filtered_df)

    with customer_tab:
        st.subheader("Customer and Product Dynamics")
        draw_top_customers(filtered_df)
        st.markdown("---")
        draw_profit_vs_sales(filtered_df)
        st.markdown("**Note:** The scatter view is limited to top products by sales to keep the chart actionable.")

    with advanced_tab:
        st.subheader("Loss and Profitability Diagnostics")
        draw_loss_table(filtered_df)
        st.markdown("---")
        draw_advanced_analysis(filtered_df)

    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Data",
        data=csv,
        file_name="filtered_superstore_data.csv",
        mime="text/csv",
    )

if __name__ == "__main__":
    main()