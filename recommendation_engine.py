import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load cleaned data
df = pd.read_csv("data/cleaned_retail_data.csv")

# Select customer and product information
customer_product = df[["CustomerID", "StockCode", "Description"]]

# Create customer-product matrix
customer_product_matrix = customer_product.pivot_table(
    index="CustomerID",
    columns="StockCode",
    values="Description",
    aggfunc="count",
    fill_value=0
)

# Calculate product similarity
product_similarity = cosine_similarity(customer_product_matrix.T)

# Convert similarity matrix into DataFrame
product_similarity_df = pd.DataFrame(
    product_similarity,
    index=customer_product_matrix.columns,
    columns=customer_product_matrix.columns
)

# Create product code → product name mapping
product_descriptions = (
    customer_product[["StockCode", "Description"]]
    .drop_duplicates("StockCode")
    .set_index("StockCode")["Description"]
)


def recommend_products(customer_id, top_n=5):

    # Find products already purchased by the customer
    purchased_products = customer_product[
        customer_product["CustomerID"] == customer_id
    ]["StockCode"].unique()

    # Calculate recommendation scores
    recommendation_scores = product_similarity_df[
        purchased_products
    ].sum(axis=1)

    # Remove products already purchased
    recommendation_scores = recommendation_scores.drop(
        labels=purchased_products,
        errors="ignore"
    )

    # Get top recommendations
    top_recommendations = recommendation_scores.sort_values(
        ascending=False
    ).head(top_n)

    # Create final recommendation table
    recommendations = top_recommendations.reset_index()
    recommendations.columns = ["StockCode", "RecommendationScore"]

    recommendations["Description"] = (
        recommendations["StockCode"].map(product_descriptions)
    )

    return recommendations
