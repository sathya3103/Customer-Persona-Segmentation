import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Base directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")
DATA_FILE = os.path.join(BASE_DIR, "customers.csv")

st.set_page_config(page_title="Customer Persona Segmenter", layout="wide")

@st.cache_resource
def load_or_train_model():
    if not os.path.exists(MODEL_FILE):
        from train import train_kmeans
        return train_kmeans(DATA_FILE, MODEL_FILE)
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_FILE):
        from dataset import generate_customer_data
        generate_customer_data(output_file=DATA_FILE)
    return pd.read_csv(DATA_FILE)

# Header
st.title("Customer Persona Segmenter")
st.write("Unsupervised Machine Learning model using K-Means Clustering for automated customer segmentation.")

# Load Artifacts & Model
artifact = load_or_train_model()
df = load_dataset()

from train import detect_columns
income_col, spending_col = detect_columns(df)

kmeans = artifact["kmeans"]
scaler = artifact["scaler"]
centroids = artifact["centroids"]
personas = artifact["personas"]

# Apply cluster labels
X_scaled = scaler.transform(df[[income_col, spending_col]].values)
df["Cluster"] = kmeans.predict(X_scaled)

# Sidebar Inputs
st.sidebar.header("Predict Customer Segment")
input_income = st.sidebar.slider("Annual Income ($k)", min_value=15.0, max_value=150.0, value=75.0, step=1.0)
input_spending = st.sidebar.slider("Spending Score (1-100)", min_value=1.0, max_value=100.0, value=60.0, step=1.0)

# Real-time Prediction
raw_input = np.array([[input_income, input_spending]])
scaled_input = scaler.transform(raw_input)
predicted_cluster = int(kmeans.predict(scaled_input)[0])
persona = personas.get(predicted_cluster, {
    "name": f"Cluster {predicted_cluster}",
    "badge": "Segment",
    "desc": "Standard Customer Segment",
    "color": "#6B7280"
})

col1, col2 = st.columns([1, 1.4])

with col1:
    st.subheader("Prediction Result")
    st.metric("Predicted Segment", persona["name"])
    st.write(f"**Segment Badge:** {persona['badge']}")
    st.write(f"**Description:** {persona['desc']}")
    st.write(f"**Cluster ID:** {predicted_cluster}")
    st.write(f"**Target Income:** ${input_income:.1f}k")
    st.write(f"**Target Spending Score:** {input_spending:.1f}")

    st.subheader("Cluster Centroids")
    centroid_data = []
    for cid, info in personas.items():
        inc, spd = centroids[cid]
        centroid_data.append({
            "Cluster": cid,
            "Persona": info["name"],
            "Avg Income ($k)": round(inc, 1),
            "Avg Spending Score": round(spd, 1)
        })
    st.dataframe(pd.DataFrame(centroid_data), hide_index=True)

with col2:
    st.subheader("Cluster Distribution Plot")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    palette = {cid: info["color"] for cid, info in personas.items()}
    
    sns.scatterplot(
        data=df,
        x=income_col,
        y=spending_col,
        hue="Cluster",
        palette=palette,
        style="Cluster",
        s=70,
        alpha=0.7,
        ax=ax
    )
    
    ax.scatter(centroids[:, 0], centroids[:, 1], c="black", s=180, marker="X", label="Centroids", zorder=5)
    ax.scatter([input_income], [input_spending], c="red", edgecolor="black", s=200, marker="*", label="Input Point", zorder=6)

    ax.set_title("Customer Segments (K-Means)")
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle="--", alpha=0.5)

    st.pyplot(fig)

with st.expander("View Customer Dataset"):
    st.dataframe(df)
