import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('Salary_Data.csv')
data = data.dropna()

# Prepare data
X = data[['Years of Experience']].values
y = data['Salary'].values

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predict and calculate R²
y_pred = model.predict(X_test_poly)
r2 = r2_score(y_test, y_pred)

# Streamlit app
st.title("Salary Predictor")
st.markdown("### Polynomial Regression (Degree 2)")
st.write(f"**R² Score:** {r2:.4f}")

# User input
years_exp = st.number_input("Enter Years of Experience", min_value=0.0, max_value=50.0, step=0.5)
if years_exp:
    val_poly = poly.transform(np.array([[years_exp]]))
    predicted_salary = model.predict(val_poly)[0]
    st.success(f"Predicted Salary for {years_exp} years of experience: ${predicted_salary:,.2f}")

# Plot
st.markdown("### Salary vs Years of Experience")
fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', label='Data')
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range_pred = model.predict(poly.transform(X_range))
ax.plot(X_range, y_range_pred, color='red', label='Polynomial Fit')
ax.set_xlabel("Years of Experience")
ax.set_ylabel("Salary")
ax.legend()
st.pyplot(fig)

