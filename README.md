# AI E-commerce Customer Intelligence & Product Recommendation System

An AI-based e-commerce project that helps businesses understand their customers and provide more relevant product recommendations.

The project uses customer purchase history to analyze buying behavior, identify different types of customers, and recommend products based on products they have purchased before.

## What This Project Does

This project mainly focuses on two areas:

- Customer Intelligence
- Product Recommendation

For customer intelligence, the project uses RFM analysis and K-Means clustering to group customers according to their purchasing behavior.

For product recommendations, it uses product purchase patterns and cosine similarity to find products that are similar to the products a customer has already purchased.

## Main Features

- Customer purchase data cleaning and preprocessing
- Customer behavior analysis
- RFM (Recency, Frequency, Monetary) analysis
- Customer segmentation using K-Means clustering
- Identification of VIP, Loyal High-Value, Regular, and Inactive Low-Value customers
- Product similarity calculation using Cosine Similarity
- Personalized product recommendations
- Interactive Streamlit interface
- Customer validation and recommendation display

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Git & GitHub

## Machine Learning Approach

### Customer Segmentation

The project calculates three important customer metrics:

- **Recency:** How recently a customer made a purchase
- **Frequency:** How often a customer placed an order
- **Monetary:** How much a customer spent

These values are then scaled and given to the K-Means clustering algorithm to identify groups of customers with similar purchasing behavior.

### Product Recommendation

The recommendation system creates a customer-product purchase matrix and uses Cosine Similarity to compare products.

When a customer enters their Customer ID, the system checks their previous purchases and recommends products that have similar purchasing patterns.

## Dataset

The project uses the **Online Retail dataset** containing transaction information such as:

- Invoice number
- Product code
- Product description
- Quantity
- Invoice date
- Unit price
- Customer ID
- Country

The dataset was cleaned before performing customer analysis and recommendations.

## Project Structure

```text
AI-ECOMMERCE-ML/
│
├── data/
│   ├── cleaned_retail_data.csv
│   └── customer_recommendations.csv
│
├── app.py
├── main.py
├── customer_intelligence.py
├── recommendation_engine.py
├── requirements.txt
└── .gitignore
