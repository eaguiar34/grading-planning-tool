# integrations_connectors.py — thin stubs so the app runs
# Place this file in the SAME folder as your Streamlit app (e.g., streamlit_grading_calendar_connectors_export_v3.py)
# The app imports it via:  from integrations_connectors import to_points_df, to_path_df, bluebeam_tasks_df

import pandas as pd


def _ensure_has(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {missing}. Found: {list(df.columns)}")
    return df


def to_points_df(uploaded_file) -> pd.DataFrame:
    """
    Parse LandXML/IFC into DataFrame with columns x,y,z.
    For now this is a CSV pass‑through: the uploaded file must be a CSV with x,y,z.
    """
    df = pd.read_csv(uploaded_file)
    return _ensure_has(df, ["x", "y", "z"])  # keep exactly these names for the app


def to_path_df(uploaded_file) -> pd.DataFrame:
    """
    Parse DXF/GeoJSON/LandXML alignment into polyline (x,y).
    For now this is a CSV pass‑through: the uploaded file must be a CSV with x,y.
    """
    df = pd.read_csv(uploaded_file)
    return _ensure_has(df, ["x", "y"])  # keep exactly these names for the app


def bluebeam_tasks_df(uploaded_csv) -> pd.DataFrame:
    """
    Normalize Bluebeam Markups CSV into a tasks table.
    For now: pass‑through (echo the CSV).
    """
    return pd.read_csv(uploaded_csv)
