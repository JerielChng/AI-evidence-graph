import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from dotenv import load_dotenv
from openai import OpenAI
from io import StringIO
from utils import add_derived_columns, weekly_deficit, flag_at_risk, build_export, forecast_next_week_weight


def generate_coaching_summary(kpis: dict, at_risk_df):
    # Make a compact, model-friendly dict
    payload = {
        "summary": {
            "users": kpis.get("users"),
            "weeks": kpis.get("weeks"),
            "avg_total_loss_per_user_kg": round(kpis.get("avg_total_loss_per_user_kg", 0), 2),
            "ontrack_users_est": kpis.get("ontrack_0_5kg_per_week_users"),
        },
        "at_risk": at_risk_df.to_dict(orient="records") if at_risk_df is not None else [],
    }

    sys = (
        "You are a careful, supportive health coach. "
        "Write a concise, evidence-based summary (120–180 words) in plain English. "
        "Use metric units, avoid medical diagnoses, avoid unsafe advice, and suggest next-step coaching actions. "
        "Structure with: 1) Cohort snapshot 2) Key positives 3) Risks/flags 4) Actionable next steps."
    )
    usr = f"Cohort insights JSON:\n{payload}"

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # or any model you prefer
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": usr}],
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating summary: {e}"

st.set_page_config(page_title="Evidence Graph Dashboard", layout="wide")

# --- Load API key ---
st.cache_data.clear()
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# --- Session state defaults ---
if "ai_text" not in st.session_state:
    st.session_state.ai_text = ""
if "ai_busy" not in st.session_state:
    st.session_state.ai_busy = False

st.title("AI Evidence Graph – Weight-Loss Coaching Insights")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
        
    st.markdown("---")
    
    st.subheader("Filters")
    # sex filter
    sex_filter = st.multiselect("Sex", ["M","F"], default=["M","F"])
    # age filter
    age_bins = st.slider("Age range", 18, 75, (18, 75))
    
    st.markdown("---")
    
    st.subheader("Chart options")
    user_select_mode = st.radio("User focus", ["Single user", "Cohort average"])

# --- Load data ---
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.caption("This is a demo dashboard for the Evidence Graph project.")
else:
    st.caption("This is a demo dashboard for the Evidence Graph project. It is not yet connected to any data source, upload sample data to explore BMI, deficits, trends, and at-risk flags.")
    st.stop()

# --- Derive columns & metrics ---
df = add_derived_columns(df)
df = weekly_deficit(df)
forecast_df = forecast_next_week_weight(df, min_weeks=3)

# Apply filters
df = df[(df["sex"].isin(sex_filter)) & (df["age"].between(age_bins[0], age_bins[1]))]

# KPIs
col1, col2, col3, col4 = st.columns(4)
users = df["user_id"].nunique()
weeks = df["week_no"].max()
avg_bmi = df["bmi"].mean()
avg_deficit = (df["tdee_kcal"] - df["calorie_intake_kcal_daily"]).mean()

col1.metric("Users", users)
col2.metric("Weeks (max)", int(weeks) if pd.notna(weeks) else 0)
col3.metric("Avg BMI", f"{avg_bmi:.1f}" if not np.isnan(avg_bmi) else "-")
col4.metric("Avg Daily Deficit (kcal)", f"{avg_deficit:.0f}" if not np.isnan(avg_deficit) else "-")

st.markdown("### Trends")

if user_select_mode == "Single user":
    uid = st.selectbox("Pick a user", sorted(df["user_id"].unique()))
    g = df[df["user_id"]==uid].sort_values("week_no")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(g, x="week_no", y="weight_kg", markers=True, title=f"User {uid} – Weight over time")
        # --- add forecast point/line if available ---
        f = forecast_df[forecast_df["user_id"]==uid]
        if not f.empty and pd.notna(f["predicted_weight_kg"].iloc[0]):
            next_wk = f["next_week"].iloc[0]
            pred_w = f["predicted_weight_kg"].iloc[0]
            # add a dashed line from last point to predicted point
            last_wk = f["last_week"].iloc[0]
            last_w = f["last_weight_kg"].iloc[0]
            fig.add_scatter(x=[last_wk, next_wk], y=[last_w, pred_w],
                            mode="lines+markers", name="Forecast",
                            line=dict(dash="dash"))

        st.plotly_chart(fig, config={}, use_container_width=True)
    with c2:
        fig2 = px.line(g, x="week_no", y="bmi", markers=True, title=f"User {uid} – BMI over time")
        st.plotly_chart(fig2, config={}, use_container_width=True)
        
    # --- Show forecast table only for selected user ---    
    st.markdown("### Next-week forecast (per user)")
    f = forecast_df[forecast_df["user_id"]==uid]
    # Only show if we have a row for this user
    if not f.empty:
        cols = ["user_id","last_week","last_weight_kg","next_week","predicted_weight_kg","predicted_delta_kg","note"]
        with st.expander("View forecast details", expanded=True):
            st.dataframe(f[cols], use_container_width=True)
    else:
        st.caption("No forecast available for this user.")
else:
    # Cohort average across users per week
    agg = df.groupby("week_no").agg(weight_kg=("weight_kg","mean"), bmi=("bmi","mean")).reset_index()
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(agg, x="week_no", y="weight_kg", markers=True, title="Cohort – Avg Weight over time")
        # cohort forecast marker at next week:
        if not forecast_df.empty and forecast_df["predicted_weight_kg"].notna().any():
            next_wk = int(forecast_df["next_week"].max())
            # average of predicted next-week weights across users
            cohort_pred = float(forecast_df["predicted_weight_kg"].dropna().mean())
            last_mean = float(agg["weight_kg"].iloc[-1]) if len(agg) else None
            if len(agg):
                fig.add_scatter(x=[agg["week_no"].iloc[-1], next_wk],
                                y=[last_mean, cohort_pred],
                                mode="lines+markers",
                                name="Cohort forecast",
                                line=dict(dash="dash"))
        st.plotly_chart(fig, config={}, use_container_width=True)
    with c2:
        fig2 = px.line(agg, x="week_no", y="bmi", markers=True, title="Cohort – Avg BMI over time")
        st.plotly_chart(fig2, config={}, use_container_width=True)

df_plot = df.copy()
df_plot["weekly_deficit_kcal"] = df_plot["weekly_deficit_kcal"].clip(-3000, 3000)  # cap extreme values
st.markdown("### Calorie deficit vs weight change")
fig_facets = px.scatter(
    df_plot,
    x="weekly_deficit_kcal",
    y="weight_change_kg",
    facet_col="user_id",
    facet_col_wrap=3,             # adjust grid width
    opacity=0.8,
    labels={"weekly_deficit_kcal":"Weekly deficit (kcal)", "weight_change_kg":"Weight change (kg)"},
    title="Per-user relationship (small multiples)",
    trendline="lowess"
)
st.plotly_chart(fig_facets, config={}, use_container_width=True)

# --- At-risk table ---
st.markdown("### At-risk users")
at_risk = flag_at_risk(df)
st.dataframe(at_risk if not at_risk.empty else pd.DataFrame(columns=["user_id","reasons"]), width='stretch')

# --- Export buttons ---
st.markdown("### Export insights")
export_payload = build_export(df, at_risk)
st.download_button("Download KPI summary (JSON)", data=str(export_payload).encode("utf-8"), file_name="insights_summary.json")
st.download_button("Download at-risk users (CSV)", data=at_risk.to_csv(index=False).encode("utf-8"), file_name="at_risk.csv")

# --- AI summary ---
st.markdown("### AI coaching summary")
if client is None:
    st.info("Add your OPENAI_API_KEY in a .env file to enable AI summaries.")
else:
    # Disable button while a request is in flight
    generate_clicked = st.button(
        "Generate AI Summary",
        disabled=st.session_state.ai_busy
    )

    if generate_clicked:
        st.session_state.ai_busy = True
        # Nice, prominent progress UI
        with st.status("Generating coaching summary…", expanded=True) as status:
            try:
                with st.spinner("Calling OpenAI and composing insights…"):
                    ai_text = generate_coaching_summary(export_payload["summary"], at_risk)
                st.session_state.ai_text = ai_text
                status.update(label="Summary ready ✅", state="complete", expanded=False)
            except Exception as e:
                st.session_state.ai_text = f"⚠️ Error generating summary: {e}"
                status.update(label="Something went wrong", state="error", expanded=True)
            finally:
                st.session_state.ai_busy = False

    # Show result if we have one (persists across reruns)
    if st.session_state.ai_text:
        st.write(st.session_state.ai_text)
        st.download_button(
            "Download summary.txt",
            data=st.session_state.ai_text.encode("utf-8"),
            file_name="coaching_summary.txt",
        )
