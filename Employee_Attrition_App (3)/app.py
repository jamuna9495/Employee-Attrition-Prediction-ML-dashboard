"""
Employee Attrition Intelligence
================================
A Streamlit dashboard + ML prediction app for employee attrition,
built entirely in Python (pandas, scikit-learn, plotly, streamlit).

Advanced features:
- Live-switchable color/font themes
- Vectorized risk leaderboard across the whole workforce
- What-if simulator (sweep one feature, watch risk change live)
- SHAP explainability when available, heuristic fallback otherwise
- Session-scoped prediction history
- Cached bulk scoring for performance

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import logic

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Attrition Intelligence",
    page_icon="\U0001F4CA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# THEME SYSTEM — pick a palette + font pairing, applied live via CSS
# ----------------------------------------------------------------------------
THEMES = {
    "Pine & Brass": {
        "ink": "#12172B", "slate": "#5B6478", "paper": "#F6F5F1", "card": "#FFFFFF",
        "accent": "#2D5C4E", "accent_soft": "#E4EEE9", "secondary": "#B98B2E",
        "risk_high": "#B3432B", "risk_med": "#B98B2E", "risk_low": "#2D5C4E",
        "palette": ["#2D5C4E", "#B98B2E", "#5B6478", "#7CA697", "#D9C6A0", "#B3432B"],
        "display_font": "Fraunces", "display_url": "Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700",
    },
    "Midnight Indigo": {
        "ink": "#0E1030", "slate": "#6A6E9B", "paper": "#F2F1FA", "card": "#FFFFFF",
        "accent": "#4338CA", "accent_soft": "#E6E4FB", "secondary": "#DB9F2E",
        "risk_high": "#C1274D", "risk_med": "#DB9F2E", "risk_low": "#4338CA",
        "palette": ["#4338CA", "#DB9F2E", "#6A6E9B", "#8C86EE", "#C7C2F5", "#C1274D"],
        "display_font": "Space Grotesk", "display_url": "Space+Grotesk:wght@500;600;700",
    },
    "Sunset Clay": {
        "ink": "#2B1B14", "slate": "#8A6F63", "paper": "#FBF3EA", "card": "#FFFFFF",
        "accent": "#C25E3A", "accent_soft": "#F4E1D5", "secondary": "#3B6E63",
        "risk_high": "#B3432B", "risk_med": "#D9A441", "risk_low": "#3B6E63",
        "palette": ["#C25E3A", "#3B6E63", "#8A6F63", "#E0966F", "#9BC1B7", "#8B2E2E"],
        "display_font": "Fraunces", "display_url": "Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700",
    },
    "Deep Ocean": {
        "ink": "#081A24", "slate": "#4F7080", "paper": "#EEF5F6", "card": "#FFFFFF",
        "accent": "#0E7C86", "accent_soft": "#DCEFF0", "secondary": "#C97A2B",
        "risk_high": "#B3432B", "risk_med": "#C97A2B", "risk_low": "#0E7C86",
        "palette": ["#0E7C86", "#C97A2B", "#4F7080", "#6FBCC4", "#E3BE8C", "#8C2E2E"],
        "display_font": "Space Grotesk", "display_url": "Space+Grotesk:wght@500;600;700",
    },
}

if "theme_name" not in st.session_state:
    st.session_state.theme_name = "Pine & Brass"
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "last_input" not in st.session_state:
    st.session_state.last_input = None

T = THEMES[st.session_state.theme_name]
PLOTLY_TEMPLATE = "plotly_white"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family={T['display_url']}&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {T['ink']};
}}
.stApp {{
    background-color: {T['paper']};
    color: {T['ink']};
}}
/* Force dark, readable text everywhere in the main content area —
   overrides browser/OS dark-mode defaults that otherwise render
   Streamlit's native widget text as white-on-light (invisible). */
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] div,
[data-testid="stMarkdownContainer"] * ,
[data-testid="stWidgetLabel"] * ,
.stSelectbox *, .stMultiSelect *, .stNumberInput *, .stTextInput *,
.stSlider *, .stRadio *, .stCheckbox *, .stDataFrame *,
.stTabs *, .stExpander *, .stCaption {{
    color: {T['ink']} !important;
}}
h1, h2, h3 {{
    font-family: '{T['display_font']}', sans-serif;
    color: {T['ink']};
    font-weight: 600;
    letter-spacing: -0.01em;
}}
[data-testid="stSidebar"] {{
    background-color: {T['ink']};
}}
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
[data-testid="stSidebar"] .stSelectbox *, [data-testid="stSidebar"] .stRadio *,
[data-testid="stSidebar"] .stExpander *, [data-testid="stSidebar"] .stCaption {{
    color: #EDEBE3 !important;
}}
[data-testid="stMetricValue"] {{
    font-family: '{T['display_font']}', sans-serif;
    color: {T['ink']};
}}
[data-testid="stMetricLabel"] {{
    color: {T['slate']};
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
.kpi-card {{
    background: {T['card']};
    border: 1px solid {T['accent']}22;
    border-left: 4px solid {T['accent']};
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 1px 2px rgba(18,23,43,0.04);
}}
.section-eyebrow {{
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.72rem;
    color: {T['accent']};
    font-weight: 700;
    margin-bottom: -0.6rem;
    font-family: 'JetBrains Mono', monospace;
}}
.risk-pill {{
    display: inline-block;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.95rem;
}}
.mono-tag {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: {T['slate']};
}}
hr {{ border-color: {T['accent']}22; }}
.stButton>button {{
    background-color: {T['accent']};
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.55rem 1.4rem;
    font-weight: 600;
}}
.stButton>button:hover {{ filter: brightness(0.9); color: white; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; color: {T['slate']}; }}
.stTabs [aria-selected="true"] {{ color: {T['accent']} !important; }}
[data-testid="stMetricDelta"] svg {{ display: none; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# DATA + MODEL LOADING
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_data(uploaded_bytes=None):
    if uploaded_bytes is not None:
        return pd.read_csv(uploaded_bytes)
    return logic.load_raw_data()


@st.cache_resource(show_spinner=False)
def get_model():
    return logic.load_model()


@st.cache_data(show_spinner="Scoring the full workforce...")
def get_bulk_scores(df):
    return logic.score_bulk(df, get_model())


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Attrition Intelligence")
    st.caption("Employee Attrition Prediction Suite")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Overview Dashboard", "Deep-Dive Explorer", "Model Performance",
         "Predict Attrition", "Risk Leaderboard", "About"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    st.markdown("**Appearance**")
    chosen_theme = st.selectbox("Theme", list(THEMES.keys()),
                                 index=list(THEMES.keys()).index(st.session_state.theme_name))
    if chosen_theme != st.session_state.theme_name:
        st.session_state.theme_name = chosen_theme
        st.rerun()

    st.markdown("---")
    with st.expander("Data source", expanded=False):
        st.caption("Default path:")
        st.code(logic.WINDOWS_DEFAULT_PATH, language=None)
        uploaded = st.file_uploader("...or upload a CSV", type=["csv"])

df = get_data(uploaded) if uploaded is not None else get_data()

if df is None:
    st.error(
        "Couldn't find the HR dataset. Place **HR analysis (2).csv** at "
        f"`{logic.WINDOWS_DEFAULT_PATH}`, or upload a CSV from the sidebar."
    )
    st.stop()

artifacts = get_model()
if artifacts is None:
    with st.spinner("Training model for the first time..."):
        artifacts = logic.train_and_save_model(df)
    st.cache_resource.clear()
    artifacts = logic.load_model()

bulk_scores = get_bulk_scores(df)
df_scored = df.copy()
df_scored["Risk Probability"] = bulk_scores
df_scored["Risk Bucket"] = df_scored["Risk Probability"].apply(lambda p: logic.risk_bucket(p)[0])


# ----------------------------------------------------------------------------
# SHARED HELPERS
# ----------------------------------------------------------------------------
def kpi(label, value, delta=None, help_text=None):
    delta_html = ""
    if delta is not None:
        color = T["risk_high"] if delta > 0 else T["risk_low"]
        arrow = "\u2191" if delta > 0 else "\u2193"
        delta_html = f'<div style="color:{color};font-size:0.8rem;font-weight:600;">{arrow} {abs(delta):.1f} vs avg</div>'
    st.markdown(
        f"""<div class="kpi-card">
            <div style="color:{T['slate']};font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em;">{label}</div>
            <div style="font-family:'{T['display_font']}',sans-serif;font-size:1.9rem;color:{T['ink']};margin-top:0.2rem;">{value}</div>
            {delta_html}
            {f'<div style="color:{T["slate"]};font-size:0.78rem;margin-top:0.1rem;">{help_text}</div>' if help_text else ''}
        </div>""",
        unsafe_allow_html=True,
    )


def style_fig(fig, height=380):
    fig.update_layout(
        template=PLOTLY_TEMPLATE, font_family="Inter", font_color=T["ink"], height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", legend_title_text="",
    )
    return fig


# ============================================================================
# PAGE 1 — OVERVIEW DASHBOARD
# ============================================================================
if page == "Overview Dashboard":
    st.markdown('<div class="section-eyebrow">Workforce Snapshot</div>', unsafe_allow_html=True)
    st.title("Overview Dashboard")
    st.caption("A live pulse on headcount, attrition rate, and where risk concentrates across the org.")

    with st.expander("Filter this view", expanded=False):
        c1, c2, c3 = st.columns(3)
        dept_f = c1.multiselect("Department", sorted(df["Department"].unique()))
        role_f = c2.multiselect("Job Role", sorted(df["Job Role"].unique()))
        gender_f = c3.multiselect("Gender", sorted(df["Gender"].unique()))

    fdf = df_scored.copy()
    if dept_f: fdf = fdf[fdf["Department"].isin(dept_f)]
    if role_f: fdf = fdf[fdf["Job Role"].isin(role_f)]
    if gender_f: fdf = fdf[fdf["Gender"].isin(gender_f)]

    total = len(fdf)
    attr_rate = (fdf["Attrition"] == "Yes").mean() * 100 if total else 0
    overall_rate = (df["Attrition"] == "Yes").mean() * 100
    avg_tenure = fdf["Years At Company"].mean() if total else 0
    avg_income = fdf["Monthly Income"].mean() if total else 0
    high_risk_n = (fdf["Risk Bucket"] == "High Risk").sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("Headcount", f"{total:,}")
    with c2: kpi("Attrition Rate", f"{attr_rate:.1f}%", delta=(attr_rate - overall_rate) if (dept_f or role_f or gender_f) else None, help_text="Yes / total")
    with c3: kpi("Avg. Tenure", f"{avg_tenure:.1f} yrs")
    with c4: kpi("Avg. Monthly Income", f"${avg_income:,.0f}")
    with c5: kpi("Model-Flagged High Risk", f"{high_risk_n:,}", help_text=f"{high_risk_n/total*100:.1f}% of view" if total else None)

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Attrition rate by department")
        g = fdf.groupby("Department")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        g.columns = ["Department", "Attrition Rate %"]
        fig = px.bar(g.sort_values("Attrition Rate %"), x="Attrition Rate %", y="Department", orientation="h",
                     color_discrete_sequence=[T["accent"]], text_auto=".1f")
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)
    with col2:
        st.subheader("Attrition rate by job role")
        g = fdf.groupby("Job Role")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        g.columns = ["Job Role", "Attrition Rate %"]
        fig = px.bar(g.sort_values("Attrition Rate %"), x="Attrition Rate %", y="Job Role", orientation="h",
                     color_discrete_sequence=[T["secondary"]], text_auto=".1f")
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Overtime vs. attrition")
        g = fdf.groupby(["Over Time", "Attrition"]).size().reset_index(name="Count")
        fig = px.bar(g, x="Over Time", y="Count", color="Attrition", barmode="group",
                     color_discrete_sequence=[T["accent"], T["risk_high"]])
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)
    with col4:
        st.subheader("Attrition by age band")
        g = fdf.groupby("CF_age band")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        g.columns = ["Age Band", "Attrition Rate %"]
        order = ["Under 25", "25 - 34", "35 - 44", "45 - 54", "Over 55"]
        g["Age Band"] = pd.Categorical(g["Age Band"], categories=order, ordered=True)
        g = g.sort_values("Age Band")
        fig = px.line(g, x="Age Band", y="Attrition Rate %", markers=True, color_discrete_sequence=[T["risk_high"]])
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)

    st.subheader("Monthly income distribution: stayed vs. left")
    fig = px.box(fdf, x="Attrition", y="Monthly Income", color="Attrition",
                 color_discrete_sequence=[T["accent"], T["risk_high"]])
    st.plotly_chart(style_fig(fig, 350), use_container_width=True)


# ============================================================================
# PAGE 2 — DEEP-DIVE EXPLORER
# ============================================================================
elif page == "Deep-Dive Explorer":
    st.markdown('<div class="section-eyebrow">Exploratory Analysis</div>', unsafe_allow_html=True)
    st.title("Deep-Dive Explorer")
    st.caption("Cross-cut any two variables to find where attrition risk hides.")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in df.select_dtypes(include="object").columns if c != "Attrition"]

    tab1, tab2, tab3 = st.tabs(["Scatter Explorer", "Category Breakdown", "Correlation Heatmap"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        x_axis = c1.selectbox("X axis", numeric_cols, index=numeric_cols.index("Age") if "Age" in numeric_cols else 0)
        y_axis = c2.selectbox("Y axis", numeric_cols, index=numeric_cols.index("Monthly Income") if "Monthly Income" in numeric_cols else 1)
        color_by = c3.selectbox("Color by", ["Attrition"] + categorical_cols)
        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, opacity=0.7, color_discrete_sequence=T["palette"])
        st.plotly_chart(style_fig(fig, 480), use_container_width=True)

    with tab2:
        cat_choice = st.selectbox("Category", categorical_cols,
                                   index=categorical_cols.index("Marital Status") if "Marital Status" in categorical_cols else 0)
        g = df.groupby([cat_choice, "Attrition"]).size().reset_index(name="Count")
        fig = px.bar(g, x=cat_choice, y="Count", color="Attrition", barmode="stack",
                     color_discrete_sequence=[T["accent"], T["risk_high"]])
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)
        rate = df.groupby(cat_choice)["Attrition"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        rate.columns = [cat_choice, "Attrition Rate %"]
        st.dataframe(rate.sort_values("Attrition Rate %", ascending=False), use_container_width=True, hide_index=True)

    with tab3:
        corr_cols = st.multiselect("Columns to include", numeric_cols,
                                    default=[c for c in ["Age", "Monthly Income", "Job Satisfaction",
                                                          "Years At Company", "Distance From Home",
                                                          "Work Life Balance", "Total Working Years"] if c in numeric_cols])
        if len(corr_cols) >= 2:
            corr = df[corr_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=[T["paper"], T["accent"], T["ink"]], aspect="auto")
            st.plotly_chart(style_fig(fig, 480), use_container_width=True)
        else:
            st.info("Pick at least two columns to see their correlation.")


# ============================================================================
# PAGE 3 — MODEL PERFORMANCE
# ============================================================================
elif page == "Model Performance":
    st.markdown('<div class="section-eyebrow">Under The Hood</div>', unsafe_allow_html=True)
    st.title("Model Performance")
    st.caption(f"Champion model: **{artifacts['model_name']}**, trained with scikit-learn.")

    results = artifacts.get("results")
    if results:
        rows = [{"Model": name, "ROC-AUC": round(m["roc_auc"], 3), "Accuracy": round(m["accuracy"], 3),
                  "Precision": round(m["precision"], 3), "Recall": round(m["recall"], 3), "F1": round(m["f1"], 3)}
                 for name, m in results.items()]
        res_df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
        st.subheader("Model comparison")
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        st.subheader("Multi-metric radar: all 3 candidates")
        metrics_for_radar = ["ROC-AUC", "Accuracy", "Precision", "Recall", "F1"]
        fig = go.Figure()
        for i, row in res_df.reset_index(drop=True).iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[m] for m in metrics_for_radar], theta=metrics_for_radar, fill="toself",
                name=row["Model"], line_color=T["palette"][i % len(T["palette"])],
            ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
        st.plotly_chart(style_fig(fig, 450), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC curve")
        rc = artifacts["roc_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rc["fpr"], y=rc["tpr"], mode="lines", line=dict(color=T["accent"], width=3), name=artifacts["model_name"]))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=T["slate"], dash="dash"), name="Random"))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
    with col2:
        st.subheader("Confusion matrix (test set)")
        if results:
            cm = np.array(results[artifacts["model_name"]]["confusion_matrix"])
            fig = px.imshow(cm, text_auto=True, x=["Pred: No", "Pred: Yes"], y=["Actual: No", "Actual: Yes"],
                             color_continuous_scale=[T["paper"], T["accent"]])
            st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    st.subheader("What drives the model's predictions")
    imp = pd.DataFrame(list(artifacts["feature_importances"].items()), columns=["Feature", "Importance"]).head(15)
    fig = px.bar(imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h", color_discrete_sequence=[T["accent"]])
    st.plotly_chart(style_fig(fig, 500), use_container_width=True)


# ============================================================================
# PAGE 4 — PREDICT ATTRITION
# ============================================================================
elif page == "Predict Attrition":
    st.markdown('<div class="section-eyebrow">Live Scoring</div>', unsafe_allow_html=True)
    st.title("Predict Attrition")
    st.caption("Score a single employee, explore what-if scenarios, or upload a roster for batch scoring.")

    tab1, tab2, tab3 = st.tabs(["Single Employee", "What-If Simulator", "Batch Upload"])

    with tab1:
        threshold = st.slider("Decision threshold (probability that counts as 'at risk')", 0.1, 0.9, 0.35, 0.05)

        with st.form("predict_form"):
            st.markdown("**Personal**")
            c1, c2, c3, c4 = st.columns(4)
            age = c1.number_input("Age", 18, 65, 32)
            gender = c2.selectbox("Gender", sorted(df["Gender"].unique()))
            marital = c3.selectbox("Marital Status", sorted(df["Marital Status"].unique()))
            distance = c4.number_input("Distance From Home (km)", 0, 40, 5)

            st.markdown("**Job**")
            c1, c2, c3, c4 = st.columns(4)
            dept = c1.selectbox("Department", sorted(df["Department"].unique()))
            role = c2.selectbox("Job Role", sorted(df["Job Role"].unique()))
            joblevel = c3.selectbox("Job Level", sorted(df["Job Level"].unique()))
            travel = c4.selectbox("Business Travel", sorted(df["Business Travel"].unique()))

            c1, c2, c3, c4 = st.columns(4)
            overtime = c1.selectbox("Over Time", sorted(df["Over Time"].unique()))
            edu_field = c2.selectbox("Education Field", sorted(df["Education Field"].unique()))
            education = c3.selectbox("Education", sorted(df["Education"].unique()))
            num_companies = c4.number_input("Num Companies Worked", 0, 10, 2)

            st.markdown("**Tenure**")
            c1, c2, c3, c4 = st.columns(4)
            total_years = c1.number_input("Total Working Years", 0, 40, 8)
            years_company = c2.number_input("Years At Company", 0, 40, 5)
            years_role = c3.number_input("Years In Current Role", 0, 20, 3)
            years_manager = c4.number_input("Years With Curr Manager", 0, 20, 3)
            years_promo = st.number_input("Years Since Last Promotion", 0, 20, 1)

            st.markdown("**Compensation**")
            c1, c2, c3 = st.columns(3)
            income = c1.number_input("Monthly Income ($)", 1000, 20000, 5000, step=100)
            daily_rate = c2.number_input("Daily Rate", 100, 1500, 800)
            hourly_rate = c3.number_input("Hourly Rate", 30, 100, 65)
            c1, c2 = st.columns(2)
            monthly_rate = c1.number_input("Monthly Rate", 2000, 27000, 14000, step=100)
            salary_hike = c2.number_input("Percent Salary Hike", 10, 25, 14)
            stock_level = st.selectbox("Stock Option Level", sorted(df["Stock Option Level"].unique()))

            st.markdown("**Satisfaction & Environment**")
            c1, c2, c3, c4 = st.columns(4)
            env_sat = c1.select_slider("Environment Satisfaction", [1, 2, 3, 4], value=3)
            job_sat = c2.select_slider("Job Satisfaction", [1, 2, 3, 4], value=3)
            rel_sat = c3.select_slider("Relationship Satisfaction", [1, 2, 3, 4], value=3)
            wlb = c4.select_slider("Work Life Balance", [1, 2, 3, 4], value=3)
            c1, c2, c3 = st.columns(3)
            job_inv = c1.select_slider("Job Involvement", [1, 2, 3, 4], value=3)
            perf_rating = c2.select_slider("Performance Rating", [1, 2, 3, 4], value=3)
            train_times = c3.number_input("Training Times Last Year", 0, 6, 2)

            submitted = st.form_submit_button("Predict Attrition Risk")

        if submitted:
            input_dict = {
                "Business Travel": travel, "Department": dept, "Education Field": edu_field,
                "Gender": gender, "Job Role": role, "Marital Status": marital, "Over Time": overtime,
                "Education": education, "Training Times Last Year": train_times, "Age": age,
                "Daily Rate": daily_rate, "Distance From Home": distance, "Environment Satisfaction": env_sat,
                "Hourly Rate": hourly_rate, "Job Involvement": job_inv, "Job Level": joblevel,
                "Job Satisfaction": job_sat, "Monthly Income": income, "Monthly Rate": monthly_rate,
                "Num Companies Worked": num_companies, "Percent Salary Hike": salary_hike,
                "Performance Rating": perf_rating, "Relationship Satisfaction": rel_sat,
                "Stock Option Level": stock_level, "Total Working Years": total_years,
                "Work Life Balance": wlb, "Years At Company": years_company, "Years In Current Role": years_role,
                "Years Since Last Promotion": years_promo, "Years With Curr Manager": years_manager,
            }
            st.session_state.last_input = input_dict

            pred, proba = logic.predict_single(input_dict, artifacts, threshold=threshold)
            bucket, color_key = logic.risk_bucket(proba)
            color = {"High Risk": T["risk_high"], "Medium Risk": T["risk_med"], "Low Risk": T["risk_low"]}[bucket]

            st.session_state.prediction_history.insert(0, {
                "Role": role, "Department": dept, "Probability": round(proba, 3), "Risk": bucket,
            })
            st.session_state.prediction_history = st.session_state.prediction_history[:10]

            st.write("")
            r1, r2 = st.columns([1, 1.4])
            with r1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=proba * 100, number={"suffix": "%", "font": {"size": 40}},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                           "steps": [{"range": [0, 30], "color": T["accent_soft"]},
                                     {"range": [30, 60], "color": "#F3E6C8"},
                                     {"range": [60, 100], "color": "#F3D9D2"}],
                           "threshold": {"line": {"color": T["ink"], "width": 3}, "value": threshold * 100}},
                    title={"text": "Attrition Probability"},
                ))
                st.plotly_chart(style_fig(fig, 300), use_container_width=True)
                st.markdown(f'<span class="risk-pill" style="background:{color}22;color:{color};">{bucket}</span>', unsafe_allow_html=True)

                st.write("")
                st.markdown("**Satisfaction radar vs. workforce average**")
                cats = ["Environment Satisfaction", "Job Satisfaction", "Relationship Satisfaction", "Work Life Balance", "Job Involvement"]
                emp_vals = [input_dict[c] for c in cats]
                avg_vals = [df[c].mean() for c in cats]
                fig2 = go.Figure()
                fig2.add_trace(go.Scatterpolar(r=emp_vals + [emp_vals[0]], theta=cats + [cats[0]], fill="toself",
                                                name="This employee", line_color=color))
                fig2.add_trace(go.Scatterpolar(r=avg_vals + [avg_vals[0]], theta=cats + [cats[0]], fill="toself",
                                                name="Workforce avg", line_color=T["slate"], opacity=0.5))
                fig2.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 4])), showlegend=True)
                st.plotly_chart(style_fig(fig2, 350), use_container_width=True)

            with r2:
                st.subheader("Top contributing factors")
                shap_result = logic.shap_explain(input_dict, artifacts)
                if shap_result:
                    st.caption("SHAP values \u2014 exact contribution of each feature to this prediction.")
                    top = shap_result[:8]
                    shap_df = pd.DataFrame(top, columns=["Feature", "SHAP Value", "Feature Value"])
                    fig3 = px.bar(shap_df.sort_values("SHAP Value"), x="SHAP Value", y="Feature", orientation="h",
                                  color="SHAP Value", color_continuous_scale=[T["risk_low"], T["paper"], T["risk_high"]])
                    st.plotly_chart(style_fig(fig3, 380), use_container_width=True)
                else:
                    st.caption("Heuristic driver ranking (install `shap` for exact per-feature contributions).")
                    factors = logic.top_contributing_factors(input_dict, artifacts, df, n=5)
                    if factors:
                        for feat, score, val, leaver_mean, stayer_mean in factors:
                            direction = "above" if val > stayer_mean else "below"
                            st.markdown(f"- **{feat}**: this employee is **{val}**, {direction} the typical "
                                        f"stayer's average of **{stayer_mean:.1f}** (leavers average **{leaver_mean:.1f}**).")
                    else:
                        st.info("No standout numeric drivers for this profile.")

                st.write("")
                if bucket == "High Risk":
                    st.error("Recommend a proactive retention conversation: review compensation, workload, and growth path.")
                elif bucket == "Medium Risk":
                    st.warning("Worth a check-in on job satisfaction and work-life balance in the next 1:1.")
                else:
                    st.success("Low predicted flight risk based on current profile.")

        if st.session_state.prediction_history:
            st.write("")
            st.subheader("This session's predictions")
            hist_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("What-If Simulator")
        base_input = st.session_state.last_input
        if base_input is None:
            st.info("Run a prediction in the **Single Employee** tab first \u2014 the simulator starts from that profile.")
        else:
            numeric_feats = [f for f in artifacts["feature_names"] if f not in logic.CATEGORICAL_COLS]
            sweep_feat = st.selectbox("Feature to sweep", numeric_feats,
                                       index=numeric_feats.index("Monthly Income") if "Monthly Income" in numeric_feats else 0)
            lo = float(df[sweep_feat].min())
            hi = float(df[sweep_feat].max())
            current_val = base_input.get(sweep_feat, (lo + hi) / 2)
            rng = st.slider(f"Range to sweep for {sweep_feat}", lo, hi, (lo, hi))

            values = np.linspace(rng[0], rng[1], 25)
            curve = logic.what_if_curve(base_input, artifacts, sweep_feat, values)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=values, y=[p * 100 for p in curve], mode="lines+markers",
                                      line=dict(color=T["accent"], width=3)))
            fig.add_vline(x=current_val, line_dash="dash", line_color=T["risk_high"],
                          annotation_text="Current value")
            fig.update_layout(xaxis_title=sweep_feat, yaxis_title="Attrition Probability (%)")
            st.plotly_chart(style_fig(fig, 420), use_container_width=True)
            st.caption(f"Holding every other field fixed at the last submitted profile, sweeping **{sweep_feat}** "
                       f"from {rng[0]:.0f} to {rng[1]:.0f} shows how predicted risk shifts.")

    with tab3:
        st.write("Upload a CSV with the same columns as the training data (Attrition column optional) to score many employees at once.")
        batch_file = st.file_uploader("Upload roster CSV", type=["csv"], key="batch")
        if batch_file is not None:
            batch_df = pd.read_csv(batch_file)
            probs = logic.score_bulk(batch_df, artifacts)
            batch_df["Predicted Attrition"] = np.where(probs >= 0.5, "Yes", "No")
            batch_df["Attrition Probability"] = probs.round(3)
            st.dataframe(batch_df, use_container_width=True)
            st.download_button("Download scored roster", batch_df.to_csv(index=False).encode("utf-8"),
                                "scored_roster.csv", "text/csv")


# ============================================================================
# PAGE 5 — RISK LEADERBOARD
# ============================================================================
elif page == "Risk Leaderboard":
    st.markdown('<div class="section-eyebrow">Whole-Workforce Scoring</div>', unsafe_allow_html=True)
    st.title("Risk Leaderboard")
    st.caption("Every employee in the dataset, ranked by predicted attrition probability.")

    c1, c2, c3 = st.columns(3)
    dept_f = c1.multiselect("Department", sorted(df["Department"].unique()), key="lb_dept")
    bucket_f = c2.multiselect("Risk bucket", ["High Risk", "Medium Risk", "Low Risk"], key="lb_bucket")
    top_n = c3.slider("Show top N", 10, len(df_scored), min(50, len(df_scored)))

    lb = df_scored.copy()
    if dept_f: lb = lb[lb["Department"].isin(dept_f)]
    if bucket_f: lb = lb[lb["Risk Bucket"].isin(bucket_f)]
    lb = lb.sort_values("Risk Probability", ascending=False).head(top_n)

    display_cols = ["Job Role", "Department", "Age", "Monthly Income", "Over Time", "Years At Company",
                     "Job Satisfaction", "Risk Probability", "Risk Bucket", "Attrition"]
    display_cols = [c for c in display_cols if c in lb.columns]

    def highlight_risk(row):
        color = {"High Risk": T["risk_high"], "Medium Risk": T["risk_med"], "Low Risk": T["risk_low"]}.get(row["Risk Bucket"], "")
        return [f"background-color: {color}22" if col == "Risk Bucket" else "" for col in row.index]

    styled = lb[display_cols].style.apply(highlight_risk, axis=1).format({"Risk Probability": "{:.1%}"})
    st.dataframe(styled, use_container_width=True, hide_index=True, height=460)

    st.download_button("Download this leaderboard", lb[display_cols].to_csv(index=False).encode("utf-8"),
                        "risk_leaderboard.csv", "text/csv")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Risk bucket distribution")
        counts = df_scored["Risk Bucket"].value_counts().reset_index()
        counts.columns = ["Risk Bucket", "Count"]
        color_map = {"High Risk": T["risk_high"], "Medium Risk": T["risk_med"], "Low Risk": T["risk_low"]}
        fig = px.pie(counts, names="Risk Bucket", values="Count", hole=0.5,
                     color="Risk Bucket", color_discrete_map=color_map)
        st.plotly_chart(style_fig(fig, 350), use_container_width=True)
    with c2:
        st.subheader("Avg. predicted risk by department")
        g = df_scored.groupby("Department")["Risk Probability"].mean().reset_index()
        fig = px.bar(g.sort_values("Risk Probability"), x="Risk Probability", y="Department", orientation="h",
                     color_discrete_sequence=[T["accent"]])
        st.plotly_chart(style_fig(fig, 350), use_container_width=True)


# ============================================================================
# PAGE 6 — ABOUT
# ============================================================================
elif page == "About":
    st.markdown('<div class="section-eyebrow">Project Notes</div>', unsafe_allow_html=True)
    st.title("About this app")
    st.markdown(f"""
This app is built **entirely in Python**:

- **pandas / numpy** — data loading and feature engineering
- **scikit-learn** — model training (Random Forest, Gradient Boosting, Logistic Regression compared automatically)
- **plotly** — all interactive charts
- **streamlit** — the app and dashboard framework
- **shap** *(optional)* — exact per-prediction feature attribution when installed

**Dataset:** IBM-style HR Analytics attrition dataset ({len(df):,} employees, {df.shape[1]} columns).

**Champion model:** {artifacts['model_name']}, selected by test-set ROC-AUC.

**Advanced features:**
- Live theme switcher (4 font/color palettes) in the sidebar
- Whole-workforce **Risk Leaderboard**, vectorized scoring, downloadable
- **What-If Simulator** — sweep one feature and watch predicted risk move live
- **SHAP explainability** when the `shap` package is installed, with an automatic heuristic fallback
- Session-scoped prediction history on the Predict page

**Data leakage guardrails:** columns that are direct derivatives of the target
(`CF_current Employee`, `CF_attrition label`) were excluded from training.
""")
