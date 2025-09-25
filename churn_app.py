import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# -----------------------
# 1. Simulate Customer Data
# -----------------------
np.random.seed(42)
n_customers = 1000

data = pd.DataFrame({
    "tenure": np.random.randint(1, 60, n_customers),
    "monthly_spend": np.random.randint(20, 200, n_customers),
    "complaints": np.random.poisson(1, n_customers),
    "support_calls": np.random.poisson(2, n_customers),
    "is_premium": np.random.choice([0,1], n_customers, p=[0.7,0.3])
})

data["churn"] = (
    (data["tenure"] < 12).astype(int) |
    (data["complaints"] > 3).astype(int) |
    ((data["monthly_spend"] < 50) & (data["is_premium"]==0)).astype(int)
).astype(int)

# -----------------------
# 2. Train ML Model
# -----------------------
X = data.drop("churn", axis=1)
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:,1]

# -----------------------
# 3. Streamlit UI
# -----------------------
st.title("📊 Customer Churn Prevention Decision Tool")
st.write("A Decision Science demo with Machine Learning + Optimization")

# Budget slider
budget = st.slider("💰 Retention Budget ($)", 1000, 10000, 5000, step=500)

# Action sliders
st.sidebar.header("🎯 Retention Actions")
discount_10_prob = st.sidebar.slider("Save Prob (10% Discount)", 0.0, 1.0, 0.3, 0.05)
discount_20_prob = st.sidebar.slider("Save Prob (20% Discount)", 0.0, 1.0, 0.5, 0.05)
support_prob = st.sidebar.slider("Save Prob (Premium Support)", 0.0, 1.0, 0.4, 0.05)

actions = {
    "discount_10": {"cost": 10, "save_prob": discount_10_prob},
    "discount_20": {"cost": 20, "save_prob": discount_20_prob},
    "premium_support": {"cost": 15, "save_prob": support_prob}
}

# -----------------------
# 4. Decision Framework
# -----------------------
results = X_test.copy()
results["churn_prob"] = y_prob
recommendations = []
total_cost = 0
saved_customers = 0

# Sort by risk × value
for idx, row in results.sort_values(by="churn_prob", ascending=False).iterrows():
    best_action, best_ev = None, 0
    for act, info in actions.items():
        ev = row["monthly_spend"] * info["save_prob"] - info["cost"]
        if ev > best_ev:
            best_ev, best_action = ev, act

    if best_action and total_cost + actions[best_action]["cost"] <= budget:
        recommendations.append(best_action)
        total_cost += actions[best_action]["cost"]
        if np.random.rand() < actions[best_action]["save_prob"]:
            saved_customers += 1
    else:
        recommendations.append("none")

results["action"] = recommendations

# -----------------------
# 5. Dashboard Outputs
# -----------------------
col1, col2, col3 = st.columns(3)
col1.metric("💵 Total Cost", f"${total_cost}")
col2.metric("🙋 Customers Saved", f"{saved_customers}")
col3.metric("📈 Retention Rate", f"{saved_customers/len(results):.2%}")

st.subheader("📌 Sample Recommendations")
st.write(results.head(10))

# Histogram
st.subheader("📊 Churn Probability Distribution")
fig, ax = plt.subplots()
ax.hist(results["churn_prob"], bins=20, alpha=0.7, color="skyblue", edgecolor="black")
ax.axvline(0.5, color="red", linestyle="--", label="Threshold 0.5")
ax.set_xlabel("Churn Probability")
ax.set_ylabel("Number of Customers")
ax.set_title("Churn Probability Distribution")
ax.legend()
st.pyplot(fig)

