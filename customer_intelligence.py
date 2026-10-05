import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def get_customer_intelligence():

    # Load cleaned data
    df = pd.read_csv("data/cleaned_retail_data.csv")

    # Convert date
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Calculate Recency
    reference_date = df["InvoiceDate"].max()

    recency = (
        df.groupby("CustomerID")["InvoiceDate"]
        .max()
        .reset_index()
    )

    recency["Recency"] = (
        reference_date - recency["InvoiceDate"]
    ).dt.days

    # Calculate Frequency
    frequency = (
        df.groupby("CustomerID")["InvoiceNo"]
        .nunique()
        .reset_index()
    )

    frequency.rename(
        columns={"InvoiceNo": "Frequency"},
        inplace=True
    )

    # Calculate Monetary
    monetary = (
        df.groupby("CustomerID")["TotalAmount"]
        .sum()
        .reset_index()
    )

    monetary.rename(
        columns={"TotalAmount": "Monetary"},
        inplace=True
    )

    # Combine RFM
    rfm = recency.merge(frequency, on="CustomerID")
    rfm = rfm.merge(monetary, on="CustomerID")

    # Scale RFM
    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    # K-Means
    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    # Cluster names
    cluster_names = {
        0: "Loyal High-Value",
        1: "Inactive Low-Value",
        2: "VIP",
        3: "Regular"
    }

    rfm["ClusterName"] = rfm["Cluster"].map(cluster_names)

    return rfm
