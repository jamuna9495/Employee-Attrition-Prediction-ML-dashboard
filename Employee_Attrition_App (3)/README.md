# Employee Attrition Intelligence — Streamlit App

100% Python-la build panna full-featured Employee Attrition Prediction app.
Dashboard + EDA + ML model + live prediction + batch prediction — ellame oru app-la.

## Enna irukku (What's inside)

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit app — dashboard, EDA, model performance, prediction UI |
| `logic.py` | Data loading + prediction logic (no Streamlit dependency, testable) |
| `train_model.py` | Trains & compares 3 ML models, saves the best one |
| `model_artifacts.pkl` | Pre-trained model (already trained on your CSV — 1470 employees) |
| `HR_data.csv` | Local copy of your dataset (backup, if the E:\ path is not found) |
| `requirements.txt` | Python libraries needed |

## Setup panna (First time only)

1. Intha folder full-a unga computer-la copy pannunga (e.g. `D:\attrition_app\`)
2. Command Prompt / PowerShell open pannunga, andha folder-ku poyi:
   ```
   cd D:\attrition_app
   ```
3. Libraries install pannunga:
   ```
   pip install -r requirements.txt
   ```

## App run panna

```
streamlit run app.py
```

Browser-la automatic-a open aagum (`http://localhost:8501`).

## Data path

App default-a intha path-la CSV thedum:
```
E:\DS and ML\HR analysis (2).csv
```
Andha path-la file irundha, adhu automatic-a load aagum. Illana, `HR_data.csv`
(intha folder-lame irukku) use pannum. Wanting na, app sidebar-la
**"Data source"** expander open pannitu, vera CSV-yum upload pannalam.

## App-la enna features irukku

1. **Overview Dashboard** — Headcount, attrition rate, department/role/age-wise
   attrition charts, filters (department, role, gender)
2. **Deep-Dive Explorer** — Scatter plot, category breakdown, correlation heatmap
   — ungalukku edhavadhu two variables compare panna
3. **Model Performance** — 3 models (Random Forest, Gradient Boosting, Logistic
   Regression) compare panni best model select pannirukom. ROC curve, confusion
   matrix, multi-metric radar chart, feature importance ellam irukku
4. **Predict Attrition** —
   - **Single Employee**: Oru employee details fill pannitu, attrition risk
     probability (gauge chart) + satisfaction radar (employee vs workforce avg)
     + top contributing factors (SHAP if installed, else heuristic) +
     recommendation kidaikum. Intha session-la panna ella predictions-um oru
     mini-table-la track aagum.
   - **What-If Simulator**: Last prediction-oda profile-a base-a vechi, oru
     feature (e.g. Monthly Income) vary pannina attrition risk eppadi maarum
     nu live line chart-la kaatum.
   - **Batch Upload**: Multiple employees CSV upload pannitu, ella perukum
     prediction oru click-la kidaikum (vectorized — fast), download pannalam
5. **Risk Leaderboard** — Whole workforce-a probability order-la rank pannitu
   kaatum, department/risk-bucket filters, top-N slider, color-coded table,
   CSV download, risk-bucket pie chart + department-wise avg risk chart
6. **About** — Model details, tech stack explanation

### Theme switcher (font + colour)

Sidebar-la **"Appearance"** section-la 4 themes irukku — ovvondrukum vera
font pairing + colour palette:

| Theme | Fonts | Vibe |
|---|---|---|
| Pine & Brass | Fraunces + Inter | Warm, editorial (default) |
| Midnight Indigo | Space Grotesk + Inter | Bold, modern dashboard |
| Sunset Clay | Fraunces + Inter | Warm terracotta/teal |
| Deep Ocean | Space Grotesk + Inter | Cool teal/navy |

Dropdown-la select pannina udane whole app — headers, KPI cards, charts,
buttons ellame — andha theme-oda colours/fonts-ku maarum.

## Model-a retrain panna venuma?

Data change aana or accuracy improve panna, intha command run pannunga:

```
python train_model.py
```

Idhu `model_artifacts.pkl` file-a refresh pannum (best model + metrics).

## Tech stack (100% Python)

- **pandas / numpy** — data handling
- **scikit-learn** — ML models (Random Forest, Gradient Boosting, Logistic
  Regression), trained + compared automatically, best one auto-selected
- **plotly** — ella interactive charts-um
- **streamlit** — dashboard/app framework, session-state, caching
- **joblib** — trained model save/load panna
- **shap** *(optional)* — install pannina, Predict page-la exact per-feature
  SHAP contribution kidaikum; illana automatic-a heuristic ranking use pannum
  (app never breaks either way)

## Notes

- Dataset-la target leakage create panna columns (`CF_current Employee`,
  `CF_attrition label`) training-la irundhu remove pannirukom — antha rendu
  columns Attrition-oda direct copy, ava irundha model "cheat" pannidum.
- Predict page-la **decision threshold** slider irukku — adha adjust pannina,
  evlo aggressive-a "at risk" nu flag panradhu nu control pannalam.
