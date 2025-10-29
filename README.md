# Audit Risk AI Dashboard  
**Predicts High-Risk Audits with 99%+ Accuracy**  


---

## Features
- **AI Model**: Random Forest (99.35% accuracy)
- **Interactive Dashboard**: Sliders → instant risk prediction
- **EDA**: Risk distribution, correlation heatmap, risky locations
- **Supply Chain Ready**: Detect fraud in invoices, shipments, vendors

---

## How to Use (No Code Needed)


1. Scroll to **"Predict Audit Risk"**
2. Move sliders (e.g., `Money Value = $15,000`, `History = 5`)
3. Get result: **HIGH RISK** → Trigger inspection!

---

## Local Setup (Optional)

```bash
# 1. Clone repo
git clone https://github.com/yourname/audit-risk-dashboard.git
cd audit-risk-dashboard

# 2. Install
pip install -r requirements.txt

# 3. Run
streamlit run audit_analysis.py

```
Column,Meaning
Risk,"1 = High Risk, 0 = Low Risk"
"PARA_A, PARA_B",Misstatement values
Money_Value,Transaction size
History,Past issues
Inherent_Risk,Calculated risk


---


---





