# Evidence Graph – Weight-Loss Coaching Insights (Starter)

A minimal **Streamlit** dashboard that ingests a cohort CSV and produces BMI/deficit trends,
at‑risk user flags, and exportable summaries.

## 1) Install

```bash
# (optional) create venv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Run
```bash
streamlit run app.py
```

Your browser will open automatically (or visit http://localhost:8501).

## 3) Use it
- Upload a CSV with columns like:
  `user_id, age, sex, height_cm, weight_kg, target_weight_kg, activity_level, calorie_intake_kcal_daily, workout_mins_week, week_no`
- Or click **Use bundled sample** in the sidebar.

## 4) What it computes
- **BMI** per row and trends per user
- **Weekly calorie deficit** using a simple TDEE estimate (demo-only)
- **Expected vs actual weight change** (7700 kcal ≈ 1 kg heuristic)
- **At-risk users**: plateau ≥3 weeks or avg deficit < 500 kcal/day
- **Exports**: JSON summary + CSV of at-risk users

## Notes
- This is a demo scaffold intended for interviews. For production, replace the TDEE estimate with a medically reviewed method and add tests.
- Python 1.5 / Streamlit tested on 2025-10-09
