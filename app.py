import streamlit as st
from recommendation_engine import recommend_products, customer_product
from customer_intelligence import get_customer_intelligence

st.title("AI E-commerce Product Recommendation System")
with st.sidebar:
    st.header("About This Project")

    st.write(
        "This system analyzes customer purchase behavior "
        "and recommends relevant products."
    )

    st.write("**Techniques Used:**")
    st.write("• RFM Analysis")
    st.write("• K-Means Clustering")
    st.write("• Cosine Similarity")
    st.divider()

st.write("**Project Flow:**")
st.write("Customer Data → RFM Analysis → K-Means Segmentation → Product Recommendation")
st.write(
    "This application uses customer purchase behavior to analyze customer segments "
    "and recommend relevant products."
)

customer_id = st.number_input(
    "Enter Customer ID to get personalized recommendations",
    min_value=1,
    step=1
)

customer_data = get_customer_intelligence()

if st.button("Get Personalized Recommendations", type="primary"):

    if customer_id not in customer_product["CustomerID"].unique():
        st.error("Customer ID not found. Please enter a valid Customer ID.")

    else:
        recommendations = recommend_products(customer_id)

        customer_info = customer_data[
            customer_data["CustomerID"] == customer_id
        ]

        st.success("Recommendations generated successfully!")

        st.subheader("Customer Intelligence")
        st.caption("Customer behavior summary based on RFM analysis and K-Means segmentation.")
        st.write(f"**Customer ID:** {int(customer_id)}")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Recency",
            f"{int(customer_info['Recency'].iloc[0])} days"
        )

        col2.metric(
            "Frequency",
            int(customer_info["Frequency"].iloc[0])
        )

        col3.metric(
            "Monetary",
            f"£{customer_info['Monetary'].iloc[0]:,.2f}"
        )

        col4.metric(
            "Customer Segment",
            customer_info["ClusterName"].iloc[0]
        )

        st.subheader("Recommended Products")
        st.caption("Products recommended based on similarity with the customer's previous purchases.")

        st.dataframe(
    recommendations,
    width="stretch",
    hide_index=True
)
        st.divider()

st.subheader("How It Works")

st.write("1. Enter a valid Customer ID.")
st.write("2. The system analyzes the customer's purchase behavior.")
st.write("3. RFM analysis and K-Means identify the customer segment.")
st.write("4. Cosine Similarity finds similar products.")
st.write("5. The system displays personalized product recommendations.")