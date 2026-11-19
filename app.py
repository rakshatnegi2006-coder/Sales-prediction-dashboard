import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Sales Prediction Dashboard", layout="wide")
st.title("📊 Sales Prediction Dashboard")
st.write("Is dashboard me hum pichli sales se aane wali sales ka prediction karte hai.")

# Sample Data Banaya
np.random.seed(42)
months = pd.date_range(start="2023-01-01", periods=24, freq="ME")
sales = np.random.randint(40000, 120000, size=24) + np.arange(24)*2000
df = pd.DataFrame({"Month": months, "Sales": sales})

# Sidebar
st.sidebar.header("Settings")
adv = st.sidebar.slider("Advertising Spend (₹)", 5000, 50000, 20000)
price = st.sidebar.slider("Product Price (₹)", 100, 1000, 500)

# Graph 1
st.subheader("Monthly Sales Trend")
st.line_chart(df.set_index("Month"))

# Prediction Model
X = np.array(range(len(df))).reshape(-1, 1)
y = df["Sales"].values
model = LinearRegression()
model.fit(X, y)

future_months = np.array(range(len(df), len(df)+6)).reshape(-1, 1)
pred = model.predict(future_months)

st.subheader("Next 6 Months Prediction")
pred_df = pd.DataFrame({
    "Month": pd.date_range(start="2025-01-01", periods=6, freq="ME"),
    "Predicted Sales": pred.astype(int)
})
st.dataframe(pred_df, use_container_width=True)
st.bar_chart(pred_df.set_index("Month"))

st.success(f"Advertising ₹{adv} aur Price ₹{price} pe, Agle Mahine ki Approx Sales: ₹{int(pred[0]):,} hogi")

st.write("Made by Priyanshu Negi | Python, Streamlit, Machine Learning")