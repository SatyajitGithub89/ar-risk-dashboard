# ar_analysis.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_analysis(df):

    # 🔹 Aggregation
    agg_data = df.groupby("Billable_WBS").agg({
        "Net_AR": "sum",
        "Age_in_Days": ["max", "mean"],
        "Billing_Document": "count"
    }).reset_index()

    agg_data.columns = ["project_id","total_AR", "max_age", "mean_age", "invoice_count"]

    # 🔹 Rule-based risk
    def risk_category(age):
        if age > 180:
            return "High Risk"
        elif age > 90:
            return "Medium Risk"
        else:
            return "Low Risk"

    agg_data["risk_category"] = agg_data["max_age"].apply(risk_category)

    # 🔹 Scaling
    features = agg_data[["total_AR", "max_age", "mean_age", "invoice_count"]]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 🔹 KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    agg_data["cluster"] = kmeans.fit_predict(scaled_features)

    sil_score = silhouette_score(scaled_features, agg_data["cluster"])

    # 🔹 Cluster summary
    cluster_summary = agg_data.groupby("cluster")[[
        "total_AR", "max_age", "mean_age", "invoice_count"
    ]].mean()

    # 🔹 Dynamic cluster labeling (IMPORTANT FIX)
    sorted_clusters = cluster_summary.sort_values("max_age").index.tolist()

    cluster_map = {
        sorted_clusters[0]: "Low Risk",
        sorted_clusters[1]: "High Risk",
        sorted_clusters[2]: "Medium Risk"
    }

    agg_data["cluster_label"] = agg_data["cluster"].map(cluster_map)

    # 🔹 Final decision
    def final_decision(row):
        if row["risk_category"] == row["cluster_label"]:
            return row["risk_category"]
        else:
            return "Review Needed"

    agg_data["final_decision"] = agg_data.apply(final_decision, axis=1)

    return agg_data, cluster_summary, sil_score