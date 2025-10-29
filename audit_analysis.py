# audit_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import streamlit as st

# --- 1. LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/Vamsi/Downloads/audit_data.csv")
    return df

df = load_data()
st.write("### Raw Data Preview", df.head())

# --- 2. CLEANING ---
# Fix duplicate Score_B column (common in this dataset)
if df.columns.duplicated().any():
    df = df.loc[:, ~df.columns.duplicated()]

# Convert Risk to int (0/1)
df['Risk'] = df['Risk'].astype(int)

# --- 3. EDA ---
st.write("## Exploratory Data Analysis")

# Risk Distribution
fig1, ax1 = plt.subplots()
df['Risk'].value_counts().plot(kind='bar', color=['green', 'red'], ax=ax1)
ax1.set_title("Audit Risk Distribution")
ax1.set_xlabel("Risk (0=Low, 1=High)")
ax1.set_ylabel("Count")
st.pyplot(fig1)

# Correlation Heatmap
st.write("### Risk Factor Correlations")
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()

fig2, ax2 = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, ax=ax2)
ax2.set_title("Feature Correlation Matrix")
st.pyplot(fig2)

# Top Risky Locations
st.write("### Top 10 Riskiest Locations")
location_risk = df.groupby('LOCATION_ID')['Risk'].mean().sort_values(ascending=False).head(10)
fig3, ax3 = plt.subplots()
location_risk.plot(kind='bar', color='orange', ax=ax3)
ax3.set_title("Average Risk by Location")
ax3.set_ylabel("Risk Probability")
st.pyplot(fig3)

# --- 4. FEATURE ENGINEERING ---
# Inherent Risk should = sum of A to F risks
risk_cols = ['Risk_A', 'Risk_B', 'Risk_C', 'Risk_D', 'RiSk_E', 'Risk_F']
df['Calculated_Inherent_Risk'] = df[risk_cols].sum(axis=1)

# Audit Risk = Inherent × Control × Detection
df['Calculated_Audit_Risk'] = df['Inherent_Risk'] * df['CONTROL_RISK'] * df['Detection_Risk']

# Add feature: High Money Value?
df['High_Money'] = (df['Money_Value'] > df['Money_Value'].quantile(0.75)).astype(int)

# Select features
features = ['Sector_score', 'PARA_A', 'PARA_B', 'Money_Value', 'History', 
            'District_Loss', 'PROB', 'Inherent_Risk', 'CONTROL_RISK', 
            'High_Money', 'Calculated_Inherent_Risk']
X = df[features]
y = df['Risk']

# --- 5. ML MODEL ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- 6. RESULTS ---
st.write("## Machine Learning Results")
accuracy = accuracy_score(y_test, y_pred)
st.write(f"**Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)**")

# Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
st.write("### Classification Report")
st.dataframe(pd.DataFrame(report).transpose())

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig4, ax4 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
ax4.set_title("Confusion Matrix")
ax4.set_xlabel("Predicted")
ax4.set_ylabel("Actual")
st.pyplot(fig4)

# Feature Importance
st.write("### Top Predictors of Audit Risk")
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
fig5, ax5 = plt.subplots()
feat_imp.plot(kind='barh', color='teal', ax=ax5)
ax5.set_title("Feature Importance")
st.pyplot(fig5)

# --- 7. PREDICTION TOOL ---
st.write("## Predict Audit Risk")
st.write("Enter values to predict if an audit is **High Risk**")

col1, col2 = st.columns(2)
with col1:
    sector = st.slider("Sector Score", 1.0, 60.0, 3.89)
    para_a = st.number_input("PARA_A", 0.0, 100.0, 4.18)
    para_b = st.number_input("PARA_B", 0.0, 100.0, 2.5)
    money = st.number_input("Money Value", 0.0, 1000.0, 3.38)
    history = st.number_input("Past Issues (History)", 0, 10, 0)

with col2:
    district_loss = st.slider("District Loss", 0, 10, 2)
    prob = st.slider("Probability (PROB)", 0.2, 0.6, 0.2)
    control_risk = st.slider("Control Risk", 0.4, 2.0, 0.4)
    inherent = st.number_input("Inherent Risk", 0.0, 200.0, 8.57)
    high_money = 1 if money > 10 else 0

# Predict
input_data = pd.DataFrame([{
    'Sector_score': sector,
    'PARA_A': para_a,
    'PARA_B': para_b,
    'Money_Value': money,
    'History': history,
    'District_Loss': district_loss,
    'PROB': prob,
    'Inherent_Risk': inherent,
    'CONTROL_RISK': control_risk,
    'High_Money': high_money,
    'Calculated_Inherent_Risk': para_a*0.6 + para_b*0.2 + money*0.2 + history*0.2 + district_loss*0.2 + prob*2
}])

pred = model.predict(input_data)[0]
prob = model.predict_proba(input_data)[0]

st.write(f"### Prediction: **{'HIGH RISK' if pred == 1 else 'LOW RISK'}**")
st.write(f"Probability of High Risk: **{prob[1]:.2%}**")

# --- 8. DOWNLOAD MODEL (Optional) ---
import joblib
joblib.dump(model, "audit_risk_model.pkl")
st.success("Model saved as `audit_risk_model.pkl`")