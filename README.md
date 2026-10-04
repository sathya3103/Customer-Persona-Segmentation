# 🛍️ Customer Persona Segmenter

> **An easy-to-use, interactive AI tool that automatically categorizes customers into meaningful business groups (personas) based on their buying habits and income.**

---

## 💡 What is This Project? (In Plain English)

Imagine you own a retail store with thousands of customers. Some customers spend freely on luxury products, others are careful bargain hunters, and many fall somewhere in between. 

If you send the exact same email promotion to everyone, you miss opportunities! 
* VIP shoppers might ignore discount sales.
* Budget-conscious shoppers might ignore high-end luxury items.

The **Customer Persona Segmenter** uses **Artificial Intelligence (Unsupervised Machine Learning)** to analyze customer data—specifically **Annual Income** and **Spending Score**—and automatically divide customers into 3 distinct groups. This helps business owners and marketers tailor their marketing campaigns effectively.

---

## 👥 The 3 Customer Segments Explained

Our machine learning model automatically identifies three main customer profiles:

| Icon | Customer Persona | Characteristics | Business Strategy |
| :---: | :--- | :--- | :--- |
| 👑 | **VIP / High Spender** | High Income + High Spending Score | **Target for Luxury & Exclusive Offers:** Give priority customer support, early access to new collections, and premium rewards. |
| ⚖️ | **Core / Balanced Customer** | Moderate Income + Moderate Spending Score | **Target for Loyalty Programs:** Maintain engagement with steady rewards, personalized recommendations, and subscription perks. |
| 🏷️ | **Budget / Frugal Customer** | Low-to-Moderate Income + Low Spending Score | **Target for Discounts & Value Deals:** Offer coupon codes, seasonal sales, clearance events, and bundled savings packages. |

---

## ✨ Web Application Features

The project includes an interactive web dashboard built with **Streamlit**:

1. 🎛️ **Live Customer Predictor (Sidebar Sliders):** Drag the sliders to input an annual income and spending score. The AI instantly predicts which segment the customer belongs to.
2. 📊 **Interactive Visual Chart:** Displays a color-coded scatter plot of all customers, showing cluster centers (marked with **X**) and highlighting your predicted customer point (marked with a red **★**).
3. 📋 **Centroid Summary Table:** Displays average annual income and average spending score for each group.
4. 🔍 **Dataset Inspector:** View and filter the raw customer dataset right inside your browser.

---

## 📁 Project Files & What They Do

Here is a simple breakdown of the project files:

```text
project/
├── app.py              # 🌐 The interactive web dashboard (run with Streamlit)
├── train.py            # 🤖 Machine learning script that trains the K-Means model
├── dataset.py          # 📊 Helper script that generates realistic customer data
├── customers.csv       # 📄 CSV file containing customer records
├── model.pkl           # 💾 Saved AI model artifact (stores trained clusters & rules)
├── requirements.txt    # 📦 List of Python packages required to run the project
└── README.md           # 📖 Documentation & User Guide (this file)
```

---

## 🚀 How to Run the App (Step-by-Step)

Follow these simple steps to run the application on your computer.

### Step 1: Install Python
Ensure you have **Python 3.8 or higher** installed on your computer.
* You can download Python from [python.org](https://www.python.org/).
* To check if Python is installed, open your command prompt/terminal and run:
  ```bash
  python --version
  ```

### Step 2: Open Command Prompt / Terminal
Open your Terminal (macOS/Linux) or Command Prompt / PowerShell (Windows) and navigate to the project directory:
```bash
cd path/to/project
```

### Step 3: Install Required Packages
Install all needed dependencies with a single command:
```bash
pip install -r requirements.txt
```

### Step 4: Launch the Web App
Run the following command to start the interactive dashboard:
```bash
streamlit run app.py
```

🎉 **That's it!** A browser tab will automatically open at `http://localhost:8501` showing your live **Customer Persona Segmenter** app.

---

## ⚙️ Advanced Usage (Optional)

* **Re-generating the Customer Dataset:**
  If you want to generate fresh synthetic customer data, run:
  ```bash
  python dataset.py
  ```
* **Re-training the AI Model manually:**
  If you want to retrain the K-Means clustering model on new data, run:
  ```bash
  python train.py
  ```
*(Note: `app.py` automatically trains the model if `model.pkl` is missing, so you don't need to do this manually unless you want to!)*

---

## 🛠️ Built With

* **Python** - Core programming language
* **Streamlit** - Framework for building interactive web apps
* **Scikit-Learn** - Machine learning library (K-Means Clustering & StandardScaler)
* **Pandas & NumPy** - Data manipulation and analysis
* **Matplotlib & Seaborn** - Data visualization and charting

---

## ❓ Frequently Asked Questions (FAQ)

<details>
<summary><b>1. What is a "Spending Score"?</b></summary>
A Spending Score is a score from 1 to 100 assigned to a customer based on their purchasing behavior, frequency of visits, and average spending habits.
</details>

<details>
<summary><b>2. Do I need coding experience to use the app?</b></summary>
Not at all! Once launched, the web application is 100% visual with sliders and charts.
</details>

<details>
<summary><b>3. What if 'streamlit' command is not recognized?</b></summary>
Try running `python -m streamlit run app.py` instead.
</details>

---
