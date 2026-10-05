import pandas as pd

# Load the Online Retail dataset
file_path = "data/Online Retail.xlsx.xlsx"

df = pd.read_excel(file_path)

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:")
print(df.duplicated().sum())
print("\nNegative Quantity:")
print((df["Quantity"] < 0).sum())
print("\nNegative UnitPrice:")
print((df["UnitPrice"] < 0).sum())
print("\nRemoving duplicate rows...")
df = df.drop_duplicates()
print("Duplicates removed successfully!")
print("New shape:", df.shape)
print("\nRemoving negative UnitPrice rows...")
df = df[df["UnitPrice"] >= 0]
print("Negative UnitPrice rows removed!")
print("New shape:", df.shape)
print("\nRemoving rows with missing CustomerID...")
df = df.dropna(subset=["CustomerID"])
print("Missing CustomerID rows removed!")
print("New shape:", df.shape)
print("\nRemoving rows with missing Description...")
df = df.dropna(subset=["Description"])
print("Missing Description rows removed!")
print("New shape:", df.shape)
print("\nQuantity values less than or equal to zero:")
print((df["Quantity"] <= 0).sum())
print("\nRemoving zero and negative Quantity rows...")
df = df[df["Quantity"] > 0]
print("Zero and negative Quantity rows removed!")
print("New shape:", df.shape)
print("\nInvoiceDate data type:")
print(df["InvoiceDate"].dtype)
print("\nConverting InvoiceDate...")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
print("InvoiceDate converted successfully!")
print(df["InvoiceDate"].dtype)
print("\nCalculating Total Amount...")
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
print("Total Amount calculated successfully!")
print(df[["Quantity", "UnitPrice", "TotalAmount"]].head())
print("\nUnique Customers:")
print(df["CustomerID"].nunique())
print("\nUnique Products:")
print(df["StockCode"].nunique())
print("\nSaving cleaned data...")
df.to_csv("data/cleaned_retail_data.csv", index=False)
print("Cleaned data saved successfully!")
print("\nTop 10 Countries by Sales:")
country_sales = df.groupby("Country")["TotalAmount"].sum().sort_values(ascending=False)
print(country_sales.head(10))
print("\nTop 10 Products by Sales:")
product_sales = df.groupby("Description")["TotalAmount"].sum().sort_values(ascending=False)
print(product_sales.head(10))
print("\nTop 10 Customers by Spending:")
customer_spending = df.groupby("CustomerID")["TotalAmount"].sum().sort_values(ascending=False)
print(customer_spending.head(10))
print("\nLatest Purchase Date:")
latest_date = df["InvoiceDate"].max()
print(latest_date)
print("\nCalculating Recency...")
reference_date = df["InvoiceDate"].max()
recency = df.groupby("CustomerID")["InvoiceDate"].max().reset_index()
recency["Recency"] = (reference_date - recency["InvoiceDate"]).dt.days
print("\nRecency calculated successfully!")
print(recency.head())
print("\nCalculating Frequency...")
frequency = df.groupby("CustomerID")["InvoiceNo"].nunique().reset_index()
frequency.rename(columns={"InvoiceNo": "Frequency"}, inplace=True)
print("\nFrequency calculated successfully!")
print(frequency.head())
print("\nCalculating Monetary...")
monetary = df.groupby("CustomerID")["TotalAmount"].sum().reset_index()
monetary.rename(columns={"TotalAmount": "Monetary"}, inplace=True)
print("\nMonetary calculated successfully!")
print(monetary.head())
print("\nCreating RFM table...")
rfm = recency.merge(frequency, on="CustomerID")
rfm = rfm.merge(monetary, on="CustomerID")
print("RFM table created successfully!")
print(rfm.head())
print("\nCalculating RFM Scores...")
rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])
print("RFM Scores calculated successfully!")
print(rfm.head())
print("\nCreating RFM Score...")
rfm["RFM_Score"] = (
    rfm["R_Score"].astype(int)
    + rfm["F_Score"].astype(int)
    + rfm["M_Score"].astype(int)
)
print("RFM Score created successfully!")
print(rfm[["CustomerID", "R_Score", "F_Score", "M_Score", "RFM_Score"]].head())
print("\nCreating Customer Segments...")
def segment_customer(score):
    if score >= 12:
        return "High Value"
    elif score >= 8:
        return "Regular"
    else:
        return "Low Value"
rfm["CustomerSegment"] = rfm["RFM_Score"].apply(segment_customer)
print("Customer Segments created successfully!")
print(rfm[["CustomerID", "RFM_Score", "CustomerSegment"]].head(10))
print("\nCustomer Segment Count:")
print(rfm["CustomerSegment"].value_counts())
import matplotlib.pyplot as plt
print("\nCreating Customer Segment Chart...")
rfm["CustomerSegment"].value_counts().plot(kind="bar")
plt.title("Customer Segments")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
print("\nOne-Time vs Repeat Customers:")
one_time_customers = (frequency["Frequency"] == 1).sum()
repeat_customers = (frequency["Frequency"] > 1).sum()
print("One-Time Customers:", one_time_customers)
print("Repeat Customers:", repeat_customers)
import matplotlib.pyplot as plt
print("\nCreating One-Time vs Repeat Customer Chart...")
customer_type = {
    "One-Time": one_time_customers,
    "Repeat": repeat_customers
}
plt.bar(customer_type.keys(), customer_type.values())
plt.title("One-Time vs Repeat Customers")
plt.xlabel("Customer Type")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()
print("\nCustomer Purchase Frequency:")
print(frequency["Frequency"].value_counts().sort_index())
from sklearn.preprocessing import StandardScaler
print("\nPreparing RFM data for K-Means...")
# Select RFM features
rfm_features = rfm[["Recency", "Frequency", "Monetary"]]
# Scale the data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features)
print("RFM data scaled successfully!")
print("Scaled data shape:", rfm_scaled.shape)
from sklearn.cluster import KMeans
print("\nFinding the best K using Elbow Method...")
inertia = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)

print("Elbow Method calculation completed!")
plt.plot(range(2, 11), inertia, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(range(2, 11))
plt.tight_layout()
plt.show()
print("\nRunning K-Means Clustering...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
print("K-Means clustering completed!")
print("\nCluster Counts:")
print(rfm["Cluster"].value_counts().sort_index())
print("\nCluster Analysis:")
cluster_analysis = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
print(cluster_analysis)
print("\nAssigning Cluster Names...")
cluster_names = {
    0: "Loyal High-Value",
    1: "Inactive Low-Value",
    2: "VIP",
    3: "Regular"
}
rfm["ClusterName"] = rfm["Cluster"].map(cluster_names)
print("Cluster names assigned successfully!")
print(rfm[["CustomerID", "Cluster", "ClusterName"]].head(10))
print("\nFinal Customer Segment Summary:")
segment_summary = rfm["ClusterName"].value_counts()
print(segment_summary)
print("\nCreating K-Means Customer Segment Chart...")
plt.figure(figsize=(8, 6))

for cluster in rfm["ClusterName"].unique():
    cluster_data = rfm[rfm["ClusterName"] == cluster]
    plt.scatter(
        cluster_data["Recency"],
        cluster_data["Monetary"],
        label=cluster
    )
plt.title("K-Means Customer Segments")
plt.xlabel("Recency")
plt.ylabel("Monetary")
plt.legend()
plt.tight_layout()
plt.show()
print("\nPreparing Product Recommendation Data...")
customer_product = df[["CustomerID", "StockCode", "Description"]]
print("Product recommendation data prepared!")
print(customer_product.head(10))
print("\nNumber of Products Purchased by Each Customer:")
customer_product_count = (
    customer_product.groupby("CustomerID")["StockCode"]
    .nunique()
)
print(customer_product_count.head(10))
print("\nCreating Customer-Product Matrix...")
customer_product_matrix = customer_product.pivot_table(
    index="CustomerID",
    columns="StockCode",
    values="Description",
    aggfunc="count",
    fill_value=0
)
print("Customer-Product Matrix created!")
print("Matrix shape:", customer_product_matrix.shape)
print(customer_product_matrix.head())
from sklearn.metrics.pairwise import cosine_similarity
print("\nCalculating Product Similarity...")
product_similarity = cosine_similarity(customer_product_matrix.T)
print("Product similarity calculated!")
print("Similarity matrix shape:", product_similarity.shape)
print("\nCreating Product Similarity DataFrame...")
product_similarity_df = pd.DataFrame(
    product_similarity,
    index=customer_product_matrix.columns,
    columns=customer_product_matrix.columns
)
print("Product Similarity DataFrame created!")
print(product_similarity_df.head())
print("\nFinding Similar Products...")
product_code = "85123A"
similar_products = product_similarity_df[product_code].sort_values(
    ascending=False
)
print("\nTop Similar Products:")
print(similar_products.head(6))
print("\nAdding Product Names to Recommendations...")
product_descriptions = (
    customer_product[["StockCode", "Description"]]
    .drop_duplicates("StockCode")
    .set_index("StockCode")["Description"]
)
recommended_products = similar_products.iloc[1:6].reset_index()
recommended_products.columns = ["StockCode", "Similarity"]
recommended_products["Description"] = (
    recommended_products["StockCode"].map(product_descriptions)
)
print("\nRecommended Products:")
print(recommended_products)
print("\nGenerating Customer Recommendations...")
customer_id = 12347
purchased_products = customer_product[
    customer_product["CustomerID"] == customer_id
]["StockCode"].unique()
recommendation_scores = product_similarity_df[purchased_products].sum(axis=1)
recommendation_scores = recommendation_scores.drop(
    labels=purchased_products,
    errors="ignore"
)
top_recommendations = recommendation_scores.sort_values(
    ascending=False
).head(5)
print("\nTop Recommendations for Customer", customer_id)
print(top_recommendations)
print("\nAdding Product Names to Customer Recommendations...")
customer_recommendations = top_recommendations.reset_index()
customer_recommendations.columns = ["StockCode", "RecommendationScore"]

customer_recommendations["Description"] = (
    customer_recommendations["StockCode"].map(product_descriptions)
)
print("\nFinal Customer Recommendations:")
print(customer_recommendations)
def recommend_products(customer_id, top_n=5):
    purchased_products = customer_product[
        customer_product["CustomerID"] == customer_id
    ]["StockCode"].unique()

    recommendation_scores = product_similarity_df[purchased_products].sum(axis=1)
    recommendation_scores = recommendation_scores.drop(
        labels=purchased_products,
        errors="ignore"
    )
    top_recommendations = recommendation_scores.sort_values(
        ascending=False
    ).head(top_n)
    recommendations = top_recommendations.reset_index()
    recommendations.columns = ["StockCode", "RecommendationScore"]
    recommendations["Description"] = (
        recommendations["StockCode"].map(product_descriptions)
    )
    return recommendations
print("\nTesting Recommendation Function...")
test_recommendations = recommend_products(12347)
print("\nRecommendations for Customer 12347:")
print(test_recommendations)
print("\nInteractive Recommendation System")
customer_id = int(input("Enter Customer ID: "))
if customer_id not in customer_product["CustomerID"].unique():
    print("\nCustomer not found. Please enter a valid Customer ID.")
else:
    recommendations = recommend_products(customer_id)
    print("\nRecommended Products:")
    print(recommendations)
    print("\nSaving Recommendations...")

recommendations.to_csv(
    "data/customer_recommendations.csv",
    index=False
)
print("Recommendations saved successfully!")