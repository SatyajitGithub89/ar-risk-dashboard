import streamlit as st
import pandas as pd
import plotly.express as px

from ar_analysis import run_analysis

# 🔹 Page config
st.set_page_config(layout="wide")
st.title("📊 AR Risk Monitoring Dashboard")

uploaded_file = st.file_uploader("Upload AR Data", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # 🔹 Run analysis
    agg_data, cluster_summary, sil_score = run_analysis(df)

    # =========================
    # 🔹 SIDEBAR FILTERS (IMPROVED)
    # =========================
    st.sidebar.header("Filters")

    age_filter = st.sidebar.slider("Minimum Age", 0, 365, 0)

    rule_filter = st.sidebar.selectbox(
        "Rule-Based Risk",
        ["All"] + list(agg_data["risk_category"].unique())
    )

    ml_filter = st.sidebar.selectbox(
        "ML-Based Risk",
        ["All"] + list(agg_data["cluster_label"].unique())
    )

    decision_filter = st.sidebar.selectbox(
        "Final Decision",
        ["All"] + list(agg_data["final_decision"].unique())
    )

    # =========================
    # 🔹 APPLY FILTERS
    # =========================
    filtered_data = agg_data.copy()

    filtered_data = filtered_data[filtered_data["max_age"] >= age_filter]

    if rule_filter != "All":
        filtered_data = filtered_data[
            filtered_data["risk_category"] == rule_filter
        ]

    if ml_filter != "All":
        filtered_data = filtered_data[
            filtered_data["cluster_label"] == ml_filter
        ]

    if decision_filter != "All":
        filtered_data = filtered_data[
            filtered_data["final_decision"] == decision_filter
        ]

    # =========================
    # 🔹 KPI SECTION (DYNAMIC)
    # =========================
    total_ar = filtered_data["total_AR"].sum()
    avg_age = filtered_data["max_age"].mean()
    count_accounts = len(filtered_data)

    review_cases = filtered_data[
        filtered_data["final_decision"] == "Review Needed"
    ]

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total AR (Filtered)", f"{total_ar:,.0f}")
    col2.metric("Avg Age", f"{avg_age:.1f}" if not pd.isna(avg_age) else "0")
    col3.metric("Accounts", count_accounts)
    col4.metric("Review Needed", len(review_cases))
    col5.metric("Silhouette Score", f"{sil_score:.2f}")

    # =========================
    # 🔹 TABS FOR DIFFERENT VIEWS (VERY IMPORTANT)
    # =========================
    tab1, tab2, tab3 = st.tabs(
        ["📘 Rule-Based", "🤖 ML-Based", "⚖️ Final Decision"]
    )

    # -------------------------
    # 🔹 RULE-BASED VIEW
    # -------------------------
    with tab1:
        fig1 = px.scatter(
            filtered_data,
            x="total_AR",
            y="max_age",
            color="risk_category",
            hover_data=["project_id"],
            title="Rule-Based Risk"
        )
        st.plotly_chart(fig1, use_container_width=True)

    # -------------------------
    # 🔹 ML-BASED VIEW
    # -------------------------
    with tab2:
        fig2 = px.scatter(
            filtered_data,
            x="total_AR",
            y="max_age",
            color="cluster_label",
            hover_data=["project_id"],
            title="ML-Based Segmentation"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # -------------------------
    # 🔹 FINAL DECISION VIEW
    # -------------------------
    with tab3:
        fig3 = px.scatter(
            filtered_data,
            x="total_AR",
            y="max_age",
            color="final_decision",
            hover_data=["project_id"],
            title="Agreement vs Mismatch"
        )
        st.plotly_chart(fig3, use_container_width=True)

    # =========================
    # 🔹 REVIEW CASES
    # =========================
    st.subheader("⚠️ Cases Needing Review")

    st.dataframe(
        filtered_data[
            filtered_data["final_decision"] == "Review Needed"
        ],
        use_container_width=True
    )

    # =========================
    # 🔹 CLUSTER SUMMARY
    # =========================
    st.subheader("📊 Cluster Summary")

    st.dataframe(cluster_summary, use_container_width=True)

    # =========================
    # 🔹 FULL DATA
    # =========================
    st.subheader("📄 Full Data")

    st.dataframe(filtered_data, use_container_width=True)

    # =========================
    # 🔹 DOWNLOAD
    # =========================
    csv = filtered_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Filtered Data",
        csv,
        "ar_analysis_output.csv",
        "text/csv"
    )