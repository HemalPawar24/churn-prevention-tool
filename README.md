# churn-prevention-tool
Customer Churn Prevention Decision Tool using Streamlit + Machine Learning + Decision Science!
Customer churn is a critical problem in telecom, SaaS, and subscription-based businesses. This project demonstrates how to:

Predict churn using a machine learning model.

Quantify the business impact of different retention actions.

Optimize decisions under a budget using expected value calculations.

Visualize results interactively through a Streamlit web app.

🔍 Key Features

Synthetic dataset simulation (tenure, spend, complaints, support calls, premium customers).

Random Forest model for churn probability prediction.

Decision framework using Expected Value (EV):

𝐸
𝑉
=
(
monthly spend
×
save probability
)
−
action cost
EV=(monthly spend×save probability)−action cost

Budget-aware allocation of retention strategies (10% discount, 20% discount, premium support).

Interactive Streamlit dashboard with:

Retention budget slider.

Adjustable action save probabilities.

KPIs: Total cost, customers saved, retention rate.

Churn probability histogram.

Sample recommendations table.

🛠️ Tech Stack

Python (NumPy, Pandas, scikit-learn, Matplotlib)

Streamlit (for UI & interactive analytics)
