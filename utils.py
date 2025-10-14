
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.linear_model import LinearRegression

# --- Core calculations ---
def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # ensure types
    out["height_m"] = out["height_cm"] / 100.0
    out["bmi"] = out["weight_kg"] / (out["height_m"] ** 2)
    # weekly delta if not provided
    if "weight_change_kg" not in out.columns or out["weight_change_kg"].isna().any():
        out = out.sort_values(["user_id", "week_no"])
        out["weight_change_kg"] = out.groupby("user_id")["weight_kg"].diff().fillna(0.0)
    return out

def estimate_tdee_simple(row: pd.Series) -> float:
    """
    Very rough TDEE proxy using Mifflin-St Jeor + activity factor (for demo only).
    Assumes 'sex' in {'M','F'}, 'age', 'height_cm', 'weight_kg', 'activity_level' in {1..5}.
    """
    s = 5 if row.get("sex","M") == "M" else -161
    bmr = 10*row["weight_kg"] + 6.25*row["height_cm"] - 5*row["age"] + s
    act_map = {1:1.2, 2:1.375, 3:1.55, 4:1.725, 5:1.9}
    factor = act_map.get(int(row.get("activity_level",3)), 1.55)
    return float(bmr*factor)

def weekly_deficit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Requires calorie_intake_kcal_daily as average daily intake that week.
    Adds columns: tdee_kcal, weekly_deficit_kcal, expected_delta_kg
    """
    out = df.copy()
    out["tdee_kcal"] = out.apply(estimate_tdee_simple, axis=1)
    if "calorie_intake_kcal_daily" not in out.columns:
        out["calorie_intake_kcal_daily"] = np.nan  # optional
    out["weekly_deficit_kcal"] = (out["tdee_kcal"] - out["calorie_intake_kcal_daily"]) * 7
    # 7700 kcal ≈ 1 kg fat heuristic
    out["expected_delta_kg"] = out["weekly_deficit_kcal"] / 7700.0
    return out

def detect_plateau(user_df: pd.DataFrame, thresh_kg: float = 0.1, consec_weeks: int = 3) -> bool:
    """
    Plateau if weight change >= -thresh (i.e., not losing more than 0.1kg) for >= N consecutive weeks.
    """
    # Using negative sign because losses are negative deltas
    deltas = user_df.sort_values("week_no")["weight_change_kg"].fillna(0.0).values
    count = 0
    for d in deltas:
        if d >= -thresh_kg:
            count += 1
            if count >= consec_weeks:
                return True
        else:
            count = 0
    return False

def flag_at_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags users as at-risk if:
      - plateau True
      - or average daily calorie deficit < 500 kcal (if intake available)
    """
    out_rows = []
    for uid, g in df.groupby("user_id"):
        plateau = detect_plateau(g)
        deficit_flag = False
        if g["calorie_intake_kcal_daily"].notna().any():
            # compute average daily deficit across weeks
            tdee_d = g["tdee_kcal"].mean()
            intake_d = g["calorie_intake_kcal_daily"].mean()
            if pd.notna(tdee_d) and pd.notna(intake_d):
                daily_deficit = tdee_d - intake_d
                deficit_flag = daily_deficit < 500
        reasons = []
        if plateau:
            reasons.append("Plateau ≥3 weeks")
        if deficit_flag:
            reasons.append("Avg daily calorie deficit < 500 kcal")
        if reasons:
            out_rows.append({"user_id": uid, "reasons": "; ".join(reasons)})
    return pd.DataFrame(out_rows)

def cohort_kpis(df: pd.DataFrame) -> Dict[str, float]:
    users = df["user_id"].nunique()
    weeks = int(df["week_no"].max()) if not df["week_no"].isna().all() else 0
    total_loss = df.groupby("user_id")["weight_change_kg"].sum().clip(upper=0).abs().mean()
    # on-track to −10 kg: users with cumulative loss ≥ (target_loss_ratio * elapsed_weeks)
    # simple heuristic: if avg weekly loss >= 0.5 kg
    by_user = df.groupby("user_id").apply(lambda x: x["weight_change_kg"].sum()).values
    ontrack = int((np.array(by_user) <= -0.5 * weeks).sum()) if weeks>0 else 0
    return {
        "users": float(users),
        "weeks": float(weeks),
        "avg_total_loss_per_user_kg": float(total_loss),
        "ontrack_0_5kg_per_week_users": float(ontrack),
    }

def build_export(df: pd.DataFrame, at_risk: pd.DataFrame) -> Dict:
    kpis = cohort_kpis(df)
    summary = {
        "summary": {
            "users": int(kpis["users"]),
            "weeks": int(kpis["weeks"]),
            "avg_total_loss_per_user_kg": round(kpis["avg_total_loss_per_user_kg"], 2),
            "ontrack_users_est": int(kpis["ontrack_0_5kg_per_week_users"]),
        },
        "at_risk": at_risk.to_dict(orient="records"),
    }
    return summary

def forecast_next_week_weight(df: pd.DataFrame, min_weeks: int = 3) -> pd.DataFrame:
    """
    For each user_id, fit a linear model (weight_kg ~ week_no) ONLY if there are at least `min_weeks`
    of observations. Otherwise, return a row with a note explaining insufficient history.

    Returns columns:
      user_id, last_week, last_weight_kg, next_week, predicted_weight_kg, predicted_delta_kg, note
    """
    rows = []
    for uid, g in df.groupby("user_id"):
        g = g.sort_values("week_no")
        last_week = int(g["week_no"].max()) if len(g) else None
        next_week = (last_week + 1) if last_week is not None else None
        last_weight = float(g["weight_kg"].iloc[-1]) if len(g) else None

        # Enforce minimum history requirement
        if len(g) < min_weeks:
            rows.append({
                "user_id": uid,
                "last_week": last_week,
                "last_weight_kg": last_weight,
                "next_week": next_week,
                "predicted_weight_kg": None,
                "predicted_delta_kg": None,
                "note": f"Not enough history (need ≥{min_weeks} weeks, have {len(g)})",
            })
            continue

        # Fit simple linear model
        x = g["week_no"].values.reshape(-1, 1)
        y = g["weight_kg"].values.astype(float)
        model = LinearRegression()
        model.fit(x, y)

        pred = float(model.predict([[next_week]])[0])
        rows.append({
            "user_id": uid,
            "last_week": last_week,
            "last_weight_kg": last_weight,
            "next_week": next_week,
            "predicted_weight_kg": round(pred, 2),
            "predicted_delta_kg": round(pred - last_weight, 2),
            "note": "",
        })
    return pd.DataFrame(rows)
