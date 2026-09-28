import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_FILE = os.path.join(BASE_DIR, "customers.csv")

def ensure_file_exists(filepath):
    """Ensures file exists so Windows OneDrive file hooks permit write operations."""
    if not os.path.exists(filepath):
        with open(filepath, 'a'):
            pass

def generate_customer_data(n_samples=300, seed=42, output_file=DEFAULT_DATA_FILE):
    """
    Generates synthetic customer dataset with Annual Income and Spending Score.
    """
    np.random.seed(seed)
    n_per_cluster = n_samples // 3

    # Cluster 0: Budget / Frugal (Low/Moderate Income, Low Spending)
    income_c0 = np.random.normal(loc=35, scale=10, size=n_per_cluster)
    spending_c0 = np.random.normal(loc=25, scale=8, size=n_per_cluster)

    # Cluster 1: VIP / High Spenders (High Income, High Spending)
    income_c1 = np.random.normal(loc=90, scale=15, size=n_per_cluster)
    spending_c1 = np.random.normal(loc=82, scale=9, size=n_per_cluster)

    # Cluster 2: Core / Balanced (Moderate Income, Moderate Spending)
    income_c2 = np.random.normal(loc=58, scale=12, size=n_per_cluster)
    spending_c2 = np.random.normal(loc=50, scale=10, size=n_per_cluster)

    # Combine clusters
    income = np.concatenate([income_c0, income_c1, income_c2])
    spending = np.concatenate([spending_c0, spending_c1, spending_c2])

    income = np.clip(np.round(income, 1), 15.0, 150.0)
    spending = np.clip(np.round(spending, 1), 1.0, 100.0)

    df = pd.DataFrame({
        "CustomerID": [f"CUST-{1000 + i}" for i in range(len(income))],
        "Annual Income (k$)": income,
        "Spending Score (1-100)": spending
    }).sample(frac=1, random_state=seed).reset_index(drop=True)

    ensure_file_exists(output_file)
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} customer records saved to '{output_file}'.")
    return df

if __name__ == "__main__":
    generate_customer_data()
