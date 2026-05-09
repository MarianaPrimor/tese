import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Production Planning Dashboard",
    layout="wide"
)

st.title("Production Planning Dashboard — Mock-up")

st.markdown(
    "This is a visual mock-up of how the final decision-support tool may look."
)

# ============================================================
# FAKE KPI DATA
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Orders", 28)
col2.metric("Planning Days", 5)
col3.metric("Lines", "L1 / L2")
col4.metric("Operator Violations", 2)
col5.metric("Capacity Violations", 3)

st.divider()

# ============================================================
# FAKE PLAN DATA
# ============================================================

data = [
    ["Day 1", "L1", "DC035072", 100, 240, 8, "High"],
    ["Day 1", "L1", "DC118826", 300, 360, 10, "Medium"],
    ["Day 1", "L2", "DC028072", 500, 420, 12, "Low"],
    ["Day 2", "L1", "DC125621", 150, 180, 7, "Low"],
    ["Day 2", "L2", "DC226083", 500, 390, 11, "High"],
    ["Day 3", "L1", "DC024072N", 100, 90, 5, "Medium"],
    ["Day 4", "L2", "DC117001", 300, 300, 9, "Medium"],
    ["Day 5", "L1", "DC436621", 500, 450, 13, "High"],
]

df = pd.DataFrame(
    data,
    columns=[
        "Day",
        "Line",
        "Order",
        "Master Boxes",
        "Production Time (min)",
        "Operators Required",
        "Priority"
    ]
)

st.subheader("Production Plan Overview")

st.dataframe(df, use_container_width=True)

st.divider()

# ============================================================
# PLAN BY DAY AND LINE
# ============================================================

st.subheader("Allocation by Day and Line")

days = df["Day"].unique()

for day in days:
    st.markdown(f"### {day}")

    day_df = df[df["Day"] == day]

    col_l1, col_l2 = st.columns(2)

    with col_l1:
        st.markdown("#### Line L1")
        l1 = day_df[day_df["Line"] == "L1"]

        if l1.empty:
            st.info("No orders allocated.")
        else:
            st.dataframe(l1, use_container_width=True)

    with col_l2:
        st.markdown("#### Line L2")
        l2 = day_df[day_df["Line"] == "L2"]

        if l2.empty:
            st.info("No orders allocated.")
        else:
            st.dataframe(l2, use_container_width=True)

st.divider()

# ============================================================
# CAPACITY MOCK-UP
# ============================================================

st.subheader("Capacity Usage by Line and Day")

capacity_data = pd.DataFrame({
    "Day": ["Day 1", "Day 1", "Day 2", "Day 2", "Day 3", "Day 4", "Day 5"],
    "Line": ["L1", "L2", "L1", "L2", "L1", "L2", "L1"],
    "Used Capacity (%)": [92, 108, 70, 95, 35, 80, 115],
})

st.bar_chart(
    capacity_data,
    x="Day",
    y="Used Capacity (%)",
    color="Line"
)

st.dataframe(capacity_data, use_container_width=True)

st.divider()

# ============================================================
# OPERATORS MOCK-UP
# ============================================================

st.subheader("Operators Required vs Available")

operators_data = pd.DataFrame({
    "Day": ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"],
    "Required Operators": [30, 25, 12, 20, 35],
    "Available Operators": [29, 29, 30, 29, 29],
})

st.bar_chart(
    operators_data,
    x="Day",
    y=["Required Operators", "Available Operators"]
)

st.dataframe(operators_data, use_container_width=True)

st.divider()

# ============================================================
# WARNINGS
# ============================================================

st.subheader("Planning Warnings")

st.warning("Day 1 - Line L2 exceeds available capacity.")
st.warning("Day 5 - Line L1 exceeds available capacity.")
st.warning("Day 5 requires more operators than available.")

st.success("This mock-up will later be connected to the genetic algorithm output.")