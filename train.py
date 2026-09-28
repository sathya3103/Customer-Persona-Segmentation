import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(BASE_DIR, "customers.csv")
DEFAULT_MODEL = os.path.join(BASE_DIR, "model.pkl")

def ensure_file_exists(filepath):
    """Ensures file exists so Windows OneDrive file hooks permit write operations."""
    if not os.path.exists(filepath):
        with open(filepath, 'a'):
            pass

def detect_columns(df):
    """Detects income and spending score column names flexibly."""
    income_col, spending_col = None, None
    for col in df.columns:
        c = col.lower()
        if "income" in c or "salary" in c:
            income_col = col
        elif "spending" in c or "score" in c:
            spending_col = col
    
    if not income_col or not spending_col:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) >= 2:
            income_col, spending_col = num_cols[0], num_cols[1]
        else:
            raise ValueError(f"Could not identify income and spending columns. Available: {df.columns.tolist()}")
            
    return income_col, spending_col

def train_kmeans(csv_file=DEFAULT_CSV, model_output=DEFAULT_MODEL):
    """
    Trains K-Means clustering model on customer income and spending score.
    Saves model, scaler, and cluster persona mappings to model.pkl.
    """
    if not os.path.exists(csv_file):
        from dataset import generate_customer_data
        generate_customer_data(output_file=csv_file)

    df = pd.read_csv(csv_file)
    income_col, spending_col = detect_columns(df)
    feature_cols = [income_col, spending_col]
    X = df[feature_cols].values

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train K-Means
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # Get unscaled centroids
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)

    # Map clusters to meaningful business personas based on centroids
    centroid_tuples = [(i, centroids[i][0], centroids[i][1]) for i in range(n_clusters)]
    
    # VIP: highest income + spending
    vip_id = max(centroid_tuples, key=lambda c: c[1] + c[2])[0]
    
    # Budget: lowest spending among remaining
    remaining = [c for c in centroid_tuples if c[0] != vip_id]
    budget_id = min(remaining, key=lambda c: c[2])[0]
    
    # Core: remaining cluster
    core_id = [c[0] for c in centroid_tuples if c[0] not in (vip_id, budget_id)][0]

    personas = {
        vip_id: {
            "name": "VIP / High Spender",
            "badge": "High Value",
            "desc": "High income with high spending. Priority target for premium & luxury campaigns.",
            "color": "#8B5CF6"
        },
        core_id: {
            "name": "Core / Balanced Customer",
            "badge": "Standard",
            "desc": "Moderate income and moderate spending. Steady repeat customers.",
            "color": "#10B981"
        },
        budget_id: {
            "name": "Budget / Frugal Customer",
            "badge": "Price Sensitive",
            "desc": "Low-to-moderate spending. Responsive to discount deals and promotions.",
            "color": "#EF4444"
        }
    }

    # Package everything into a single dictionary artifact
    artifact = {
        "kmeans": kmeans,
        "scaler": scaler,
        "features": feature_cols,
        "centroids": centroids,
        "personas": personas
    }

    ensure_file_exists(model_output)
    joblib.dump(artifact, model_output)
    print(f"Successfully trained K-Means (K={n_clusters}). Model saved to '{model_output}'.")
    for cid, persona in personas.items():
        inc, spd = centroids[cid]
        print(f" - Cluster {cid}: {persona['name']} (Avg Income: ${inc:.1f}k, Avg Spending Score: {spd:.1f})")

    return artifact

if __name__ == "__main__":
    train_kmeans()
