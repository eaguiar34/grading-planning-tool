# integrations_connectors.py (stub)
import pandas as pd


def to_points_df(uploaded_file):
# Parse LandXML/IFC into DataFrame with columns x,y,z — for now, just CSV pass‑through
return pd.read_csv(uploaded_file) # must contain x,y,z


def to_path_df(uploaded_file):
# Parse DXF/GeoJSON/LandXML alignment into x,y polyline — for now, just CSV pass‑through
return pd.read_csv(uploaded_file) # must contain x,y


def bluebeam_tasks_df(uploaded_csv):
# Normalize Bluebeam markups CSV into tasks — for now, echo input
return pd.read_csv(uploaded_csv)
