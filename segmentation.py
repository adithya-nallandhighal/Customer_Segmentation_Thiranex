import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Segmentation", page_icon="📊")

# Always load the .pkl files from the same folder as this script,
# no matter which folder you launch Streamlit from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_models():
    kmeans = joblib.load(os.path.join(BASE_DIR, "kmeans_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    return kmeans, scaler


try:
    kmeans, scaler = load_models()
except FileNotFoundError:
    st.error("kmeans_model.pkl / scaler.pkl not found. Put them in the same folder as this file.")
    st.stop()

# Exact features and order used in training (read from the saved scaler)
FEATURES = list(scaler.feature_names_in_)
# ['Age', 'Income', 'Total_Spending', 'NumWebPurchases', 'NumWebVisitsMonth', 'Recency']

# Cluster profiles taken from the notebook's cluster_summary. Rename as you like.
SEGMENTS = {
    0: "Older customers, moderate income, low spending",
    1: "Older customers, high income, high spending, rarely visit the website",
    2: "Younger customers, high income, top spenders",
    3: "Budget shoppers, frequent web visitors, bought recently",
    4: "Budget shoppers, frequent web visitors, inactive for a long time",
    5: "Mid-age, upper-middle income, heavy web purchasers",
}

st.title("Customer Segmentation App")
st.write("Enter customer details to predict the segment.")

age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Income", min_value=0, max_value=200000, value=50000)
total_spending = st.number_input("Total Spending (sum of purchases)", min_value=0, max_value=5000, value=1000)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)
num_web_visits = st.number_input("Number of Web Visits Per Month", min_value=0, max_value=50, value=3)
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)

input_data = pd.DataFrame([{
    "Age": age,
    "Income": income,
    "Total_Spending": total_spending,
    "NumWebPurchases": num_web_purchases,
    "NumWebVisitsMonth": num_web_visits,
    "Recency": recency,
}])[FEATURES]  # enforce the training column order

if st.button("Predict Segment"):
    input_scaled = scaler.transform(input_data)  # DataFrame keeps column names -> sklearn validates them
    cluster = int(kmeans.predict(input_scaled)[0])
    st.success(f"Predicted Segment: Cluster {cluster}")
    st.info(SEGMENTS.get(cluster, ""))