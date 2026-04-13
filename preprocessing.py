import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_enrollment_to_numeric(s):
    out = s.astype(str).str.replace(",", "", regex=False).str.extract(r"(\d+\.?\d*)")[0]
    return pd.to_numeric(out, errors="coerce")

def filter_to_crispr(df):
    GE_CAT_COL = "Gene-editing category"
    GE_METHOD_COL = "Gene-editing method"
    if GE_CAT_COL in df.columns:
        cat_mask = df[GE_CAT_COL].astype(str).str.contains("crispr", case=False, na=False)
        if GE_METHOD_COL in df.columns:
            meth_mask = df[GE_METHOD_COL].astype(str).str.contains("crispr", case=False, na=False)
            return df[cat_mask | meth_mask].copy()
        return df[cat_mask].copy()
    if GE_METHOD_COL in df.columns:
        return df[df[GE_METHOD_COL].astype(str).str.contains("crispr", case=False, na=False)].copy()
    return df.copy()

def preprocess(df):
    FEATURE_COLS = {
        "Phase 1": "phase_1", "Phase 2": "phase_2", "Phase 3": "phase_3",
        "In US?": "in_us", "Start year": "start_year",
        "Enrollment (target)": "enrollment", "Disease Category": "disease_category",
        "Countries (per CMN page)": "countries", "Last-updated year": "updated_year",
        "Gene-editing method": "method",
    }
    feat_df = df.rename(columns={k: v for k, v in FEATURE_COLS.items() if k in df.columns}).copy()
    feat_df["enrollment"] = clean_enrollment_to_numeric(feat_df.get("enrollment", pd.Series(dtype=str)))
    feat_df["start_year"] = pd.to_numeric(feat_df.get("start_year", pd.Series(dtype=float)), errors="coerce")
    feat_df["updated_year"] = pd.to_numeric(feat_df.get("updated_year", pd.Series(dtype=float)), errors="coerce")
    feat_df["trial_age"] = feat_df["updated_year"] - feat_df["start_year"]
    feat_df["in_us_binary"] = feat_df["in_us"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    feat_df["num_countries"] = feat_df["countries"].fillna("").astype(str).apply(
        lambda x: len([c for c in x.split(",") if c.strip()]))
    label_encoders = {}
    for col in ["disease_category", "method"]:
        if col in feat_df.columns:
            feat_df[col] = feat_df[col].fillna("Unknown").astype(str)
            le = LabelEncoder()
            feat_df[col + "_enc"] = le.fit_transform(feat_df[col])
            label_encoders[col] = le
    # Target: advanced = reached Phase 2 or 3
    if "phase_2" in feat_df.columns and "phase_3" in feat_df.columns:
        feat_df["advanced"] = ((feat_df["phase_2"] == 1) | (feat_df["phase_3"] == 1)).astype(int)
    return feat_df, label_encoders
