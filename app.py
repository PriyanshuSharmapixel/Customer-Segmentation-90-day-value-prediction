from pathlib import Path
import json

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Value Dashboard",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "project_outputs"


@st.cache_data
def load_data():
    customers = pd.read_csv(
        DATA_DIR / "customer_results.csv",
        dtype={"CustomerID": str}
    )

    summary = pd.read_csv(DATA_DIR / "segment_summary.csv")

    with open(DATA_DIR / "project_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    return customers, summary, metadata


customers, summary, metadata = load_data()

st.title("Customer Segmentation & 90-Day Value")
st.caption(
    "Historical evaluation at 1 September 2011 · "
    "Forecast window: 1 September–29 November 2011 · "
    "Spending in GBP, excluding refunds"
)

col1, col2, col3 = st.columns(3)

col1.metric("Customers analysed", f"{len(customers):,}")
col2.metric("Customer segments", customers["Segment"].nunique())
col3.metric(
    "Baseline MAE",
    f"£{metadata['baseline_mae_gbp']:,.2f}"
)

st.info(
    "Previous 90-day spending was selected as the prediction method. "
    "It outperformed XGBoost in this exploratory evaluation."
)

tab1, tab2, tab3 = st.tabs(
    ["Segment overview", "Customer explorer", "Method & limitations"]
)

with tab1:
    indexed_summary = summary.set_index("Segment")

    st.subheader("Customers by segment")
    st.bar_chart(indexed_summary[["Customers"]])

    st.subheader("Historical purchase spending share (%)")
    st.bar_chart(indexed_summary[["HistoricalSpendingSharePct"]])

    st.subheader("Predicted versus actual 90-day spending (£)")
    st.bar_chart(
        indexed_summary[
            ["Predicted90DaySpending", "Actual90DaySpending"]
        ]
    )

    st.subheader("Suggested business actions")
    for _, row in summary.iterrows():
        st.markdown(f"**{row['Segment']}**")
        st.write(row["SuggestedAction"])

with tab2:
    selected_segments = st.multiselect(
        "Choose customer segments",
        options=sorted(customers["Segment"].unique()),
        default=sorted(customers["Segment"].unique())
    )

    filtered = customers[
        customers["Segment"].isin(selected_segments)
    ].copy()

    customer_search = st.text_input("Search customer ID")

    if customer_search.strip():
        filtered = filtered[
            filtered["CustomerID"].str.contains(
                customer_search.strip(), regex=False, na=False
            )
        ]

    st.write(f"{len(filtered):,} matching customers")

    st.dataframe(
        filtered.sort_values("PredictedSpend90Days", ascending=False),
        hide_index=True,
        use_container_width=True
    )

    st.download_button(
        "Download filtered customers",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_customers.csv",
        mime="text/csv"
    )

with tab3:
    st.write(
        "Customers were segmented using log-transformed, standardised "
        "Recency, Frequency and Monetary features with K-Means."
    )

    comparison = pd.DataFrame({
        "Approach": ["Previous 90-day spending", "Updated XGBoost"],
        "MAE (£)": [
            metadata["baseline_mae_gbp"],
            metadata["updated_xgboost_mae_gbp"]
        ]
    })

    st.dataframe(comparison, hide_index=True)

    st.caption("Lower MAE is better. MAE is not a percentage accuracy.")

    st.subheader("Limitations")
    for limitation in metadata["limitations"]:
        st.write(f"• {limitation}")

    st.write(
        "This dashboard displays saved historical predictions. "
        "It does not forecast spending for newly uploaded data."
    )