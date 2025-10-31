# Streamlit — Grading Assistant (Alignment + Profile LandXML export added)
# Balance • TIN Volumes • Cross‑Slope & Zones CSV I/O • Calendar • Connectors • DXF/LandXML (Surface+Alignment+Profile)
# -----------------------------------------------------------------------------------------------------------------
# Implements your landing rules and adds a full LandXML export that includes:
#  • Alignment (as straight line segments between path vertices)
#  • Profile (PVIs at every station using the balanced proposed profile)
#  • Surface (simple grid‑triangulated TIN from cross‑section samples)
#
# Quick start:
#   pip install streamlit pandas numpy scipy shapely plotly lxml ezdxf
#   streamlit run streamlit_grading_align_profile_export.py

import os
import io
import math
from datetime import date, datetime, time, timedelta
import numpy as np
import pandas as pd
import streamlit as st
from scipy.interpolate import LinearNDInterpolator, griddata
from scipy.spatial import Delaunay
from shapely.geometry import Point, Polygon
import plotly.express as px
import plotly.graph_objects as go

# Optional export deps
try:
    import ezdxf
except Exception:  # pragma: no cover
    ezdxf = None
try:
    from lxml import etree
except Exception:  # pragma: no cover
    etree = None

# --- Connectors (optional) ---
try:
    from integrations_connectors import to_points_df, to_path_df, bluebeam_tasks_df
    CONNECTORS_OK = True
except Exception:
    CONNECTORS_OK = False
    to_points_df = None
    to_path_df = None
    bluebeam_tasks_df = None

st.set_page_config(page_title="Grading Assistant — Align/Profile Export", layout="wide")
st.title("Grading Assistant — Balance • TIN • CSV I/O • Calendar • Connectors • LandXML (Align+Profile)")

# ============================
# Geometry & volume utilities
# ============================

def ensure_columns(df: pd.DataFrame, x_col: str, y_col: str, z_col: str) -> pd.DataFrame:
    out = df[[x_col, y_col, z_col]].copy()
    out.columns = ["x", "y", "z"]
    for c in ["x", "y", "z"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["x", "y", "z"]).reset_index(drop=True)


def make_tin_interpolator(df_xyz: pd.DataFrame):
    pts = df_xyz[["x", "y"]].to_numpy(float)
    z = df_xyz["z"].to_numpy(float)
    tri = Delaunay(pts)
    interp = LinearNDInterpolator(tri, z, fill_value=np.nan)
    def f(x, y):
        xy = np.column_stack([np.asarray(x), np.asarray(y)])
        vals = interp(xy)
        if np.isscalar(vals):
            if math.isnan(vals):
                i = np.argmin((pts[:,0]-x)**2 + (pts[:,1]-y)**2)
                return float(z[i])
            return float(vals)
        m = np.isnan(vals)
        if m.any():
            for i, miss in enumerate(m):
                if miss:
                    dx = pts[:,0]-xy[i,0]
                    dy = pts[:,1]-xy[i,1]
                    j = np.argmin(dx*dx + dy*dy)
                    vals[i] = z[j]
        return np.asarray(vals, dtype=float)
    return f, tri


def build_path_samples(df_path, ds):
    x = df_path["x"].to_numpy(float)
    y = df_path["y"].to_numpy(float)
    if len(x) < 2:
        return np.array([0.0]), np.array([x[0]]), np.array([y[0]])
    dx = np.diff(x)
    dy = np.diff(y)
    seg = np.hypot(dx, dy)
    s_cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.arange(0, float(s_cum[-1]) + ds/2.0, ds)
    xs = np.interp(s, s_cum, x)
    ys = np.interp(s, s_cum, y)
    return s, xs, ys


def slope_limited_profile(z_init, ds, smin_pct, smax_pct, anchors=None, iters=60):
    z = z_init.copy().astype(float)
    n = len(z)
    if anchors is None:
        anchors = {}
    dz_max = (smax_pct/100.0) * ds
    dz_min = (smin_pct/100.0) * ds
    for _ in range(iters):
        for idx, elev in anchors.items():
            if 0 <= idx < n:
                z[idx] = elev
        changed = False
        for i in range(n-1):
            hi = z[i] + dz_max
            lo = z[i] + dz_min
            if z[i+1] > hi:
                z[i+1] = hi
                changed = True
            if z[i+1] < lo:
                z[i+1] = lo
                changed = True
        for i in range(n-2, -1, -1):
            hi = z[i+1] - dz_min
            lo = z[i+1] - dz_max
            if z[i] > hi:
                z[i] = hi
                changed = True
            if z[i] < lo:
                z[i] = lo
                changed = True
        for idx, elev in anchors.items():
            if 0 <= idx < n:
                z[idx] = elev
        if not changed:
            break
    return z


def unit_tangent_normals(xs, ys):
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    L = np.hypot(dx, dy)
    L[L == 0] = 1.0
    tx, ty = dx/L, dy/L
    nx, ny = -ty, tx
    return tx, ty, nx, ny


def polygon_from_slice(xs, ys, i, width):
    tx, ty, nx, ny = unit_tangent_normals(xs, ys)
    half = width/2.0
    p0L = (xs[i] - nx[i]*half, ys[i] - ny[i]*half)
    p0R = (xs[i] + nx[i]*half, ys[i] + ny[i]*half)
    p1L = (xs[i+1] - nx[i+1]*half, ys[i+1] - ny[i+1]*half)
    p1R = (xs[i+1] + nx[i+1]*half, ys[i+1] + ny[i+1]*half)
    return Polygon([p0L, p1L, p1R, p0R, p0L])


def triangulate_polygon_uniform(poly: Polygon, spacing: float):
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx, maxx + spacing, spacing)
    ys = np.arange(miny, maxy + spacing, spacing)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    inside = np.array([poly.contains(Point(p)) or poly.touches(Point(p)) for p in pts])
    in_pts = pts[inside]
    if len(in_pts) < 3:
        return np.empty((0,3,2)), np.empty((0,))
    tri = Delaunay(in_pts)
    tris = in_pts[tri.simplices]
    ctr = tris.mean(axis=1)
    keep = np.array([poly.contains(Point(c)) for c in ctr])
    tris = tris[keep]
    areas = 0.5*np.abs((tris[:,1,0]-tris[:,0,0])*(tris[:,2,1]-tris[:,0,1]) - (tris[:,2,0]-tris[:,0,0])*(tris[:,1,1]-tris[:,0,1]))
    return tris, areas


def bilinear_prop_z(x, y, xs, ys, i, width, zc0, zc1, cs0_pct, cs1_pct, mode0, mode1):
    tx, ty, nx, ny = unit_tangent_normals(xs, ys)
    p0 = np.array([xs[i], ys[i]])
    p1 = np.array([xs[i+1], ys[i+1]])
    seg = p1 - p0
    seg_len = np.hypot(seg[0], seg[1])
    sparam = 0.0 if seg_len == 0 else np.clip(np.dot(np.array([x, y]) - p0, seg/seg_len)/seg_len, 0, 1)
    half = width/2.0
    nvec = np.array([nx[i], ny[i]])
    t_off = np.dot(np.array([x, y]) - p0, nvec)
    t_norm = np.clip(t_off/half, -1.0, 1.0)
    def edge_z(zc, cs_pct, mode, t):
        if mode == "crowned":
            return zc - abs(t)*half*(cs_pct/100.0)
        elif mode == "one-way left":
            return zc + (t*half)*(cs_pct/100.0)
        else:
            return zc - (t*half)*(cs_pct/100.0)
    zl0 = edge_z(zc0, cs0_pct, mode0, -1.0)
    zr0 = edge_z(zc0, cs0_pct, mode0, +1.0)
    zl1 = edge_z(zc1, cs1_pct, mode1, -1.0)
    zr1 = edge_z(zc1, cs1_pct, mode1, +1.0)
    zl = (1-sparam)*zl0 + sparam*zl1
    zr = (1-sparam)*zr0 + sparam*zr1
    return 0.5*((1-t_norm)*zl + (1+t_norm)*zr)


def slice_tin_volume(poly: Polygon, f_exist, z_prop_func, spacing):
    tris, areas = triangulate_polygon_uniform(poly, spacing)
    if len(areas) == 0:
        return 0.0, 0.0
    ctr = tris.mean(axis=1)
    zx = f_exist(ctr[:,0], ctr[:,1])
    zp = np.array([z_prop_func(x, y) for x, y in ctr])
    diff = zx - zp
    vol = diff * areas
    return float(np.clip(vol, 0, None).sum()), float(np.clip(-vol, 0, None).sum())


def corridor_slice_volumes(df_exist, xs, ys, zc, cs_pct, modes, width, spacing):
    f_exist, _ = make_tin_interpolator(df_exist)
    seg_len = np.hypot(np.diff(xs), np.diff(ys))
    rows = []
    for i in range(len(xs)-1):
        poly = polygon_from_slice(xs, ys, i, width)
        zc0, zc1 = zc[i], zc[i+1]
        cs0, cs1 = cs_pct[i], cs_pct[i+1]
        m0, m1 = modes[i], modes[i+1]
        zfun = lambda X, Y, i=i, zc0=zc0, zc1=zc1, cs0=cs0, cs1=cs1, m0=m0, m1=m1: bilinear_prop_z(X, Y, xs, ys, i, width, zc0, zc1, cs0, cs1, m0, m1)
        cut, fill = slice_tin_volume(poly, f_exist, zfun, spacing)
        rows.append({"i": i, "length": float(seg_len[i]) if i < len(seg_len) else 0.0, "cut_vol": cut, "fill_vol": fill})
    return pd.DataFrame(rows)


def compute_grades(z, ds):
    g = np.zeros_like(z)
    if len(z) > 1:
        g[:-1] = 100.0*np.diff(z)/ds
        g[-1] = g[-2] if len(z) > 2 else g[-1]
    return g


def cross_slope_series(stations, base_cs, key_df, default_mode="crowned"):
    cs = np.full_like(stations, base_cs, dtype=float)
    modes = np.array([default_mode]*len(stations), dtype=object)
    if key_df is None or len(key_df) == 0:
        return cs, modes
    kt = key_df.sort_values("station")
    s_k = kt["station"].to_numpy(float)
    cs_k = kt["cross_slope_pct"].to_numpy(float)
    mode_k = kt["mode"].astype(str).tolist()
    cs = np.interp(stations, s_k, cs_k, left=cs_k[0], right=cs_k[-1])
    idx_near = np.searchsorted(s_k, stations, side="left")
    idx_near = np.clip(idx_near, 0, len(s_k)-1)
    modes = np.array([mode_k[j] for j in idx_near], dtype=object)
    return cs, modes

# ============================
# Landing insertion (implements your rules)
# ============================

def insert_landings(z, ds, max_walk_pct=5.0, landing_len=6.0, merge_gap_frac=0.5):
    """Return (z_modified, ranges) inserting level landings per user rules.
    Rules:
      • Trigger when |running slope| > max_walk_pct (strictly greater)
      • Start landing at first violating segment
      • Merge adjacent landings if gap < merge_gap_frac * landing_len
      • Endcaps: keep landing inside [0, n-1]; if too close to end, back‑extend just
        enough to have at least half the landing length inside. Shorten if impossible.
    """
    z = np.asarray(z, dtype=float)
    n = len(z)
    if n < 3:
        return z.copy(), []
    k = max(1, int(round(landing_len/ds)))
    min_half = max(1, int(math.ceil(0.5 * k)))

    grades = compute_grades(z, ds)
    viol_idx = np.where(np.abs(grades) > max_walk_pct)[0]  # strict '>' per rule
    if len(viol_idx) == 0:
        return z.copy(), []

    # Build candidate ranges starting at the violation
    cand = []
    for i in viol_idx:
        i0 = int(i)
        i1 = min(n-1, i0 + k)
        # If shorter than half because near end, back‑extend start while keeping >=0
        length = i1 - i0
        if length < min_half:
            need = min_half - length
            shift = min(need, i0)
            i0 = i0 - shift
            length = i1 - i0
        cand.append((i0, i1))

    # Merge ranges if separation < half landing length
    gap_thresh = max(1, int(math.floor(0.5 * k)))
    cand.sort()
    merged = []
    cur_s, cur_e = cand[0]
    for s2, e2 in cand[1:]:
        if s2 - cur_e < gap_thresh:  # strictly less than half → merge
            cur_e = max(cur_e, e2)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s2, e2
    merged.append((cur_s, cur_e))

    # Apply flattening using original endpoints
    z0 = z.copy()
    z_new = z.copy()
    for a, b in merged:
        elev = 0.5 * (z0[a] + z0[b])
        z_new[a:b+1] = elev
    return z_new, merged


def balance_delta(df_exist, xs, ys, zc, cs, modes, width, spacing, swell_cut=1.0, shrink_fill=1.0, tol=1e-3):
    def net(delta):
        z = zc + delta
        dfv = corridor_slice_volumes(df_exist, xs, ys, z, cs, modes, width, spacing)
        cut = dfv["cut_vol"].sum()*swell_cut
        fill = dfv["fill_vol"].sum()/max(shrink_fill, 1e-9)
        return cut - fill, dfv
    lo, hi = -10.0, 10.0
    f_lo, _ = net(lo)
    f_hi, _ = net(hi)
    tries = 0
    while f_lo*f_hi > 0 and tries < 6:
        lo *= 2
        hi *= 2
        f_lo, _ = net(lo)
        f_hi, _ = net(hi)
        tries += 1
    for _ in range(30):
        mid = 0.5*(lo+hi)
        f_mid, df_mid = net(mid)
        if abs(f_mid) < tol:
            return mid, df_mid
        if f_lo*f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    mid = 0.5*(lo+hi)
    _, df_mid = net(mid)
    return mid, df_mid

# ============================
# Sidebar — Global Inputs + Connectors
# ============================
with st.sidebar:
    st.header("Global Inputs")
    units = st.selectbox("Units", ["US (feet)", "SI (meters)"])
    unit_len = "ft" if units.startswith("US") else "m"
    to_cy = 1/27.0 if units.startswith("US") else 1.30795061931

    ds = st.number_input(f"Station spacing ({unit_len})", value=10.0, min_value=0.5, step=0.5)
    width = st.number_input(f"Corridor width ({unit_len})", value=24.0, min_value=1.0, step=0.5)
    samp = st.number_input(f"TIN sampling spacing ({unit_len})", value=2.0, min_value=0.2, step=0.2)

    st.markdown("---")
    st.subheader("Grade limits")
    smax = st.number_input("Max grade (%)", value=8.0, step=0.1)
    smin = st.number_input("Min grade (%)", value=-8.0, step=0.1)
    warn_drain = st.number_input("Warn if |grade| < (%)", value=0.5, step=0.1)

    st.markdown("---")
    st.subheader("Sidewalk Landings")
    enable_land = st.checkbox("Enable landing insertion", value=False)
    walk_max = st.number_input("Walkway max running slope (%)", value=5.0, step=0.1)
    land_len = st.number_input(f"Landing length ({unit_len})", value=6.0, step=0.5)

    st.markdown("---")
    st.subheader("Cross-slope defaults")
    base_cs = st.number_input("Default cross slope (%)", value=2.0, step=0.1)
    default_mode = st.selectbox("Default mode", ["crowned","one-way left","one-way right"], index=0)

    st.markdown("---")
    st.subheader("Earthwork factors & Balance")
    swell_cut = st.number_input("Swell factor (CUT)", value=1.10, step=0.01)
    shrink_fill = st.number_input("Shrink factor (FILL)", value=0.90, step=0.01)
    balance_mode = st.selectbox("Balance method", ["Global","Per-zone"], index=0)

    st.markdown("---")
    st.subheader("Connectors")
    if not CONNECTORS_OK:
        st.warning("Connectors module not found. Put integrations_connectors.py next to this app (optional).")
    st.caption("CSV uploads work even without connectors.")

# ============================
# 1) Existing surface (CSV or connector)
# ============================
st.subheader("1) Existing ground surface")
colE1, colE2 = st.columns(2)
with colE1:
    exist_source = st.radio("Existing surface source", ["CSV (x,y,z)", "LandXML", "IFC"], index=0)
with colE2:
    existing_file = st.file_uploader("Upload existing surface file", type=["csv","xml","landxml","ifc"], key="exist_any")

if existing_file is not None:
    if exist_source == "CSV (x,y,z)":
        df_exist_raw = pd.read_csv(existing_file)
        st.dataframe(df_exist_raw.head(), use_container_width=True)
        cols = list(df_exist_raw.columns)
        xcol = st.selectbox("X column", cols)
        ycol = st.selectbox("Y column", cols)
        zcol = st.selectbox("Z column", cols)
        df_exist = ensure_columns(df_exist_raw, xcol, ycol, zcol)
    else:
        if not CONNECTORS_OK:
            st.error("Connectors not available. Use CSV or provide the connectors module.")
            st.stop()
        try:
            df_exist = to_points_df(existing_file)
            st.success(f"Parsed {len(df_exist)} points from {exist_source}")
        except Exception as e:
            st.exception(e)
            st.stop()
else:
    st.info("Upload an existing surface to continue.")
    st.stop()

# ============================
# 2) Path / centerline (CSV or connector)
# ============================
st.subheader("2) Corridor path / centerline")
colP1, colP2 = st.columns(2)
with colP1:
    path_source = st.radio("Path source", ["CSV (x,y)", "DXF", "GeoJSON", "LandXML alignment"], index=0)
with colP2:
    path_file = st.file_uploader("Upload path file", type=["csv","dxf","json","geojson","xml","landxml"], key="path_any")

if path_file is not None:
    if path_source == "CSV (x,y)":
        df_path_raw = pd.read_csv(path_file)
        st.dataframe(df_path_raw.head(), use_container_width=True)
        pcols = list(df_path_raw.columns)
        px = st.selectbox("Path X column", pcols)
        py = st.selectbox("Path Y column", pcols)
        df_path = df_path_raw[[px, py]].copy()
        df_path.columns = ["x", "y"]
    else:
        if not CONNECTORS_OK:
            st.error("Connectors not available. Use CSV or provide the connectors module.")
            st.stop()
        try:
            df_path = to_path_df(path_file)
            st.success(f"Parsed path with {len(df_path)} vertices from {path_source}")
        except Exception as e:
            st.exception(e)
            st.stop()
else:
    st.info("Upload a path/centerline to continue.")
    st.stop()

# Stationing and existing profile along path
s, xs, ys = build_path_samples(df_path, ds)
st.info(f"Stations: {len(s)} • Length: {s[-1]:.2f} {('ft' if units.startswith('US') else 'm')}")

f_exist, _ = make_tin_interpolator(df_exist)
z_exist_path = f_exist(xs, ys)
z_exist_path = np.array(z_exist_path, dtype=float)
mask = np.isnan(z_exist_path)
if mask.any():
    z_exist_path[mask] = griddata(df_exist[["x","y"]].to_numpy(), df_exist["z"].to_numpy(), (xs[mask], ys[mask]), method="nearest")

# ============================
# 3) Anchors (optional)
# ============================
st.subheader("3) Anchors (optional)")
c1,c2,c3 = st.columns(3)
with c1:
    use_start = st.checkbox("Anchor start elevation")
    start_elev = st.number_input("Start elev", value=float(z_exist_path[0]), step=0.1)
with c2:
    use_end = st.checkbox("Anchor end elevation")
    end_elev = st.number_input("End elev", value=float(z_exist_path[-1]), step=0.1)
with c3:
    use_ffe = st.checkbox("Anchor to FFE near station")
    ffe_val = st.number_input("FFE value", value=float(z_exist_path[len(z_exist_path)//2]), step=0.1)
    ffe_x = st.number_input("FFE x", value=float(xs[len(xs)//2]), step=1.0)
    ffe_y = st.number_input("FFE y", value=float(ys[len(ys)//2]), step=1.0)

anchors = {}
if use_start:
    anchors[0] = float(start_elev)
if use_end:
    anchors[len(s)-1] = float(end_elev)
if use_ffe:
    d2 = (xs-ffe_x)**2 + (ys-ffe_y)**2
    anchors[int(np.argmin(d2))] = float(ffe_val)

# Initial constrained profile
z0 = z_exist_path.copy()
z_profile = slope_limited_profile(z0, ds, smin, smax, anchors=anchors)

# ============================
# 4) Cross‑slope key‑stations CSV I/O
# ============================
st.subheader("4) Cross‑slope / Superelevation keys (CSV import/export)")
st.write("Columns: station, cross_slope_pct, mode (crowned|one-way left|one-way right)")
cs_csv = st.file_uploader("Import cross‑slope CSV", type=["csv"], key="cs_csv")
if cs_csv is not None:
    key_df = pd.read_csv(cs_csv)
else:
    key_df = pd.DataFrame({"station": [], "cross_slope_pct": [], "mode": []})
key_df = st.data_editor(key_df, num_rows="dynamic", use_container_width=True)
st.download_button("Download cross‑slope CSV", data=key_df.to_csv(index=False).encode("utf-8"), file_name="cross_slope_keys.csv")
cs, modes = cross_slope_series(s, base_cs, key_df, default_mode=default_mode)

# Optional sidewalk landings (with your rules)
landing_ranges = []
if enable_land:
    z_flat, landing_ranges = insert_landings(z_profile, ds, max_walk_pct=walk_max, landing_len=land_len, merge_gap_frac=0.5)
    # Re-clip to grade limits while respecting inserted flat anchors
    extra = {}
    for a, b in landing_ranges:
        elev = z_flat[a]
        for i in range(a, b+1):
            extra[i] = elev
    merge = {**anchors, **extra}
    z_profile = slope_limited_profile(z_flat, ds, smin, smax, anchors=merge)

# ============================
# 5) Haul/Balance Zones CSV I/O
# ============================
st.subheader("5) Haul/Balance Zones (CSV import/export)")
st.write("Columns: start_station, end_station, zone_name (optional)")
zone_csv = st.file_uploader("Import zones CSV", type=["csv"], key="zone_csv")
if zone_csv is not None:
    zone_df = pd.read_csv(zone_csv)
else:
    zone_df = pd.DataFrame({"start_station": [], "end_station": [], "zone_name": []})
zone_df = st.data_editor(zone_df, num_rows="dynamic", use_container_width=True)
st.download_button("Download zones CSV", data=zone_df.to_csv(index=False).encode("utf-8"), file_name="haul_balance_zones.csv")

def idx_from_sta(sta):
    return int(np.clip(np.searchsorted(s, float(sta), side='left'), 0, len(s)-1))

# Balance + volumes
zc = z_profile.copy()
if balance_mode == "Global" or len(zone_df) == 0:
    delta, vol_df = balance_delta(df_exist, xs, ys, zc, cs, modes, width, samp, swell_cut=swell_cut, shrink_fill=shrink_fill)
    zc_bal = zc + delta
    st.success(f"Suggested vertical offset Δ = {delta:.3f} {('ft' if units.startswith('US') else 'm')} (global)")
else:
    zc_bal = zc.copy()
    for _, r in zone_df.iterrows():
        try:
            i0 = idx_from_sta(r.get("start_station", s[0]))
            i1 = idx_from_sta(r.get("end_station", s[-1]))
        except Exception:
            continue
        if i1 <= i0:
            continue
        dlt, _ = balance_delta(
            df_exist,
            xs[i0:i1+1], ys[i0:i1+1],
            zc[i0:i1+1], cs[i0:i1+1], modes[i0:i1+1],
            width, samp,
            swell_cut=swell_cut, shrink_fill=shrink_fill,
        )
        zc_bal[i0:i1+1] += dlt
    vol_df = corridor_slice_volumes(df_exist, xs, ys, zc_bal, cs, modes, width, samp)

# Totals (factored)
cut_ft3 = vol_df["cut_vol"].sum()
fill_ft3 = vol_df["fill_vol"].sum()
cut_cy = cut_ft3*to_cy*swell_cut
fill_cy = (fill_ft3*to_cy)/max(shrink_fill, 1e-9)
colT1, colT2, colT3 = st.columns(3)
colT1.metric("CUT (factored, yd³)", f"{cut_cy:.1f}")
colT2.metric("FILL (factored, yd³)", f"{fill_cy:.1f}")
colT3.metric("NET (yd³)", f"{(fill_cy-cut_cy):.1f}")

# Plots — profile/grades
grades = compute_grades(zc_bal, ds)
prof = go.Figure()
prof.add_trace(go.Scatter(x=s, y=z_exist_path, mode="lines", name="Existing"))
prof.add_trace(go.Scatter(x=s, y=zc_bal, mode="lines", name="Proposed (balanced)"))
for a, b in landing_ranges:
    prof.add_vrect(x0=s[a], x1=s[b], fillcolor="LightGray", opacity=0.3, line_width=0)
prof.update_layout(title="Longitudinal Profile", xaxis_title=f"Station ({unit_len})", yaxis_title="Elevation")
st.plotly_chart(prof, use_container_width=True)

gr = go.Figure()
gr.add_trace(go.Scatter(x=s, y=grades, mode="lines", name="Grade (%)"))
gr.add_hline(y=smax, line_dash="dot")
gr.add_hline(y=smin, line_dash="dot")
gr.add_hline(y=warn_drain, line_dash="dash")
gr.add_hline(y=-warn_drain, line_dash="dash")
st.plotly_chart(gr, use_container_width=True)

# Per-slice export
vol_tbl = vol_df.copy()
vol_tbl["station_start"] = s[vol_tbl["i"].values]
vol_tbl["cut_cy_factored"] = vol_tbl["cut_vol"]*to_cy*swell_cut
vol_tbl["fill_cy_factored"] = vol_tbl["fill_vol"]*to_cy/max(shrink_fill, 1e-9)
st.download_button("Download per-slice TIN volumes", data=vol_tbl.to_csv(index=False).encode("utf-8"), file_name="tin_slice_volumes.csv")

# ============================
# 6) Bluebeam Markups (optional) — normalize and export
# ============================
st.subheader("6) Bluebeam Markups → Tasks (optional)")
if CONNECTORS_OK:
    bb_file = st.file_uploader("Upload Bluebeam Markups List CSV", type=["csv"], key="bb")
    if bb_file is not None:
        try:
            tasks_df = bluebeam_tasks_df(bb_file)
            st.dataframe(tasks_df.head(200), use_container_width=True)
            st.download_button("Download normalized tasks CSV", data=tasks_df.to_csv(index=False).encode("utf-8"), file_name="bluebeam_tasks_normalized.csv")
        except Exception as e:
            st.exception(e)
else:
    st.info("Connectors not available — place integrations_connectors.py next to this app to parse Bluebeam CSVs.")

# ============================
# 7) Schedule • Cost • Risk — Calendar aware
# ============================
st.subheader("7) Schedule • Cost • Risk (Calendar‑aware)")
corridor_len = float(s[-1]) if len(s) else 0.0
corridor_area = corridor_len * width
cal1, cal2, cal3 = st.columns([2,2,2])
with cal1:
    start_dt = st.date_input("Project start date", value=date.today())
    work_start_hour = st.number_input("Workday start hour (0–23)", value=7, min_value=0, max_value=23, step=1)
    hours_day = st.number_input("Work hours per day", value=10.0, min_value=1.0, step=0.5)
with cal2:
    wkdays = st.multiselect("Working weekdays", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], default=["Mon","Tue","Wed","Thu","Fri"]) 
with cal3:
    holiday_csv = st.file_uploader("Non‑working dates CSV (column: date in YYYY‑MM‑DD)", type=["csv"], key="calendar_csv")

wkday_map = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
working_weekdays = set(wkday_map[w] for w in wkdays)
holiday_set = set()
if holiday_csv is not None:
    try:
        df_h = pd.read_csv(holiday_csv)
        for d in pd.to_datetime(df_h[df_h.columns[0]]).dt.date:
            holiday_set.add(d)
    except Exception:
        st.warning("Could not parse holiday CSV — expecting a single 'date' column.")

surface_type = st.selectbox("Surface type", ["Asphalt", "Concrete sidewalk"], index=0)
surf_thk = st.number_input(f"Surface thickness ({'in' if units.startswith('US') else 'mm'})", value=4.0 if surface_type=="Asphalt" else 5.0, step=0.5)
if units.startswith("US"):
    density = st.number_input("Material density (ton/ft³ for Asphalt; t/m³ for Concrete)", value=float(145.0/2000.0 if surface_type=="Asphalt" else 2.40), step=0.001)
    surf_vol_ft3 = corridor_area * (surf_thk/12.0)
    surf_tons = surf_vol_ft3 * density if surface_type=="Asphalt" else None
    conc_cy = (corridor_area * (surf_thk/12.0))/27.0 if surface_type=="Concrete sidewalk" else None
    subbase_cy = (corridor_area * (6.0/12.0))/27.0
else:
    density = st.number_input("Material density (t/m³)", value=2.35 if surface_type=="Asphalt" else 2.40, step=0.01)
    surf_vol_m3 = corridor_area * (surf_thk/1000.0)
    surf_tons = surf_vol_m3 * density if surface_type=="Asphalt" else None
    conc_cy = surf_vol_m3 * 1.30795061931 if surface_type=="Concrete sidewalk" else None
    subbase_cy = (corridor_area * 0.1524) * 1.30795061931

qty = {
    "Excavate (CUT) CY": max(cut_cy, 0.0),
    "Embank (FILL) CY": max(fill_cy, 0.0),
    "Subbase CY": max(subbase_cy, 0.0),
    "Asphalt Tons": float(surf_tons or 0.0),
    "Concrete CY": float(conc_cy or 0.0),
}
st.write("Quantities (approx, factored where applicable):", qty)

st.markdown("**Productivity (units/day) & Unit Cost ($/unit)** — triangular distributions")

def tri_row(task, uom, prod_default, cost_default):
    c1,c2,c3,c4,c5 = st.columns([2,1,1,1,1])
    with c1:
        st.markdown(f"**{task}** ({uom})")
    with c2:
        p_lo = st.number_input(f"P min {task}", value=float(prod_default[0]))
    with c3:
        p_ml = st.number_input(f"P ml {task}", value=float(prod_default[1]))
    with c4:
        p_hi = st.number_input(f"P max {task}", value=float(prod_default[2]))
    with c5:
        u_cost = st.number_input(f"$/unit {task}", value=float(cost_default))
    return (p_lo, p_ml, p_hi), u_cost

prod = {}
cost = {}
prod["Excavate (CUT) CY"], cost["Excavate (CUT) CY"] = tri_row("Excavate (CUT) CY", "CY/day", (800, 1000, 1200), 6.0)
prod["Embank (FILL) CY"], cost["Embank (FILL) CY"] = tri_row("Embank (FILL) CY", "CY/day", (600, 900, 1100), 7.0)
prod["Subbase CY"], cost["Subbase CY"] = tri_row("Subbase CY", "CY/day", (300, 500, 700), 15.0)
if surface_type == "Asphalt":
    prod["Asphalt Tons"], cost["Asphalt Tons"] = tri_row("Asphalt Tons", "tons/day", (200, 300, 450), 140.0)
    chain = ["Excavate (CUT) CY","Embank (FILL) CY","Subbase CY","Asphalt Tons"]
else:
    prod["Concrete CY"], cost["Concrete CY"] = tri_row("Concrete CY", "CY/day", (40, 60, 80), 180.0)
    chain = ["Excavate (CUT) CY","Embank (FILL) CY","Subbase CY","Concrete CY"]
chain = [t for t in chain if qty.get(t, 0) > 0]

# Calendar helpers
working_weekdays = set({"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}[w] for w in wkdays)

def advance_to_next_work_start(dt):
    while (dt.weekday() not in working_weekdays) or (dt.date() in holiday_set):
        dt = datetime.combine(dt.date() + timedelta(days=1), time(hour=work_start_hour))
    if dt.hour < work_start_hour:
        dt = datetime.combine(dt.date(), time(hour=work_start_hour))
    if dt.hour + dt.minute/60.0 >= work_start_hour + hours_day:
        dt = datetime.combine(dt.date() + timedelta(days=1), time(hour=work_start_hour))
        return advance_to_next_work_start(dt)
    return dt

def add_work_hours(start_date_only: date, hours_total: float) -> datetime:
    dt = datetime.combine(start_date_only, time(hour=work_start_hour))
    dt = advance_to_next_work_start(dt)
    hrs_left = float(hours_total)
    while hrs_left > 1e-6:
        if (dt.weekday() in working_weekdays) and (dt.date() not in holiday_set):
            today_end = datetime.combine(dt.date(), time(hour=int(work_start_hour))) + timedelta(hours=hours_day)
            avail = (today_end - dt).total_seconds()/3600.0
            if avail <= 1e-9:
                dt = advance_to_next_work_start(datetime.combine(dt.date()+timedelta(days=1), time(hour=work_start_hour)))
                continue
            step = min(avail, hrs_left)
            dt = dt + timedelta(hours=step)
            hrs_left -= step
        else:
            dt = advance_to_next_work_start(datetime.combine(dt.date()+timedelta(days=1), time(hour=work_start_hour)))
    return dt

# Deterministic schedule (most-likely rates)
det_rows = []
cur_start = advance_to_next_work_start(datetime.combine(start_dt, time(hour=work_start_hour)))
for t in chain:
    q = qty.get(t, 0.0)
    p_ml = prod[t][1]
    dur_days = q / max(p_ml, 1e-9)
    finish_dt = add_work_hours(cur_start.date(), dur_days * hours_day)
    det_rows.append({"Task": t, "Start": cur_start, "Finish": finish_dt, "Duration_days": dur_days, "Qty": q})
    cur_start = finish_dt

det_df = pd.DataFrame(det_rows)
if len(det_df):
    gantt = px.timeline(det_df, x_start="Start", x_end="Finish", y="Task", color="Task")
    gantt.update_layout(title="Deterministic schedule (calendar‑aware)")
    st.plotly_chart(gantt, use_container_width=True)
    st.download_button("Download schedule CSV (date‑based)", data=det_df.to_csv(index=False).encode("utf-8"), file_name="schedule_calendar.csv")

# Monte Carlo — calendar‑aware
st.markdown("**Risk (Monte Carlo)** — calendar‑aware finish dates")
sims = st.number_input("Simulations", value=300, min_value=100, step=100)
rng = np.random.default_rng(0)
finish_dates = []
total_costs = []
for _ in range(int(sims)):
    dt_cur = advance_to_next_work_start(datetime.combine(start_dt, time(hour=work_start_hour)))
    cost_sum = 0.0
    for t in chain:
        p_lo, p_ml, p_hi = prod[t]
        u = rng.triangular(p_lo, p_ml, p_hi)
        q = qty.get(t, 0.0)
        dur_hours = (q / max(u, 1e-9)) * hours_day
        dt_cur = add_work_hours(dt_cur.date(), dur_hours)
        cost_sum += q * cost[t]
    finish_dates.append(dt_cur)
    total_costs.append(cost_sum)

fd = np.array(finish_dates, dtype='datetime64[ns]')
duration_days = (fd - np.datetime64(datetime.combine(start_dt, time(hour=work_start_hour)))).astype('timedelta64[h]').astype(float) / 24.0
st.write({
    "Finish P10/P50/P90 (date)": (pd.to_datetime(np.percentile(fd,10)).date(), pd.to_datetime(np.percentile(fd,50)).date(), pd.to_datetime(np.percentile(fd,90)).date()),
    "Duration P10/P50/P90 (days)": (float(np.percentile(duration_days,10)), float(np.percentile(duration_days,50)), float(np.percentile(duration_days,90))),
    "Cost P10/P50/P90 ($)": (float(np.percentile(total_costs,10)), float(np.percentile(total_costs,50)), float(np.percentile(total_costs,90))),
})

st.plotly_chart(px.histogram(x=duration_days, nbins=40, labels={"x":"Total duration (days)"}), use_container_width=True)
st.plotly_chart(px.histogram(x=total_costs, nbins=40, labels={"x":"Total cost ($)"}), use_container_width=True)

# ============================
# 8) Exports — DXF centerline, LandXML (Surface only), and LandXML (Alignment+Profile+Surface)
# ============================
st.subheader("8) Exports")

# CSV profile (always available)
prof_df = pd.DataFrame({"station": s, "elev_proposed": zc_bal})
st.download_button("Download Proposed Profile CSV", data=prof_df.to_csv(index=False).encode("utf-8"), file_name="ProposedProfile.csv")

# DXF centerline

def make_dxf_centerline(df_path_xy: pd.DataFrame) -> bytes:
    if ezdxf is None:
        raise RuntimeError("ezdxf not installed")
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    points = [(float(r.x), float(r.y)) for r in df_path_xy.itertuples(index=False)]
    msp.add_lwpolyline(points)
    buf = io.BytesIO()
    doc.write(stream=buf)
    return buf.getvalue()

colx1, colx2 = st.columns(2)
with colx1:
    if ezdxf is None:
        st.info("DXF export requires `ezdxf`. Install it to enable centerline export.")
    else:
        try:
            dxf_bytes = make_dxf_centerline(df_path)
            st.download_button("Download Centerline DXF", data=dxf_bytes, file_name="Centerline.dxf")
        except Exception as e:
            st.warning(f"DXF export failed: {e}")

# LandXML proposed surface (simple TIN built from cross‑section samples)

def build_proposed_surface_points(xs, ys, s, zc, cs, modes, width, offsets=None):
    if offsets is None:
        offsets = [-width/2.0, 0.0, width/2.0]
    tx, ty, nx, ny = unit_tangent_normals(xs, ys)
    pts = []
    for i in range(len(xs)):
        zc_i = zc[i]
        cs_i = cs[i]
        m_i = modes[i]
        half = width/2.0
        for off in offsets:
            x = xs[i] + nx[i]*off
            y = ys[i] + ny[i]*off
            # simple cross‑slope at station (no along‑station interpolation here)
            if m_i == "crowned":
                z = zc_i - abs(off)* (cs_i/100.0)
            elif m_i == "one-way left":
                z = zc_i + off * (cs_i/100.0)
            else:
                z = zc_i - off * (cs_i/100.0)
            pts.append((x, y, z))
    return pts, offsets


def landxml_surface_from_grid(points, n_sta, n_off):
    if etree is None:
        raise RuntimeError("lxml not installed")
    NS = "http://www.landxml.org/schema/LandXML-1.2"
    root = etree.Element("LandXML", nsmap={None: NS})
    root.set("version", "1.2")
    surfaces = etree.SubElement(root, "Surfaces")
    surf = etree.SubElement(surfaces, "Surface")
    surf.set("name", "ProposedSurface")
    defin = etree.SubElement(surf, "Definition")
    defin.set("surfType", "TIN")
    # Points
    for idx, (x, y, z) in enumerate(points, start=1):
        p = etree.SubElement(defin, "P")
        p.set("id", str(idx))
        p.text = f"{x} {y} {z}"
    # Faces (grid triangulation)
    def pid(i_sta, i_off):
        return 1 + i_sta*n_off + i_off
    for i in range(n_sta-1):
        for j in range(n_off-1):
            a = pid(i, j)
            b = pid(i+1, j)
            c = pid(i+1, j+1)
            d = pid(i, j+1)
            f1 = etree.SubElement(defin, "F"); f1.text = f"{a} {b} {c}"
            f2 = etree.SubElement(defin, "F"); f2.text = f"{a} {c} {d}"
    return etree.tostring(root, pretty_print=True, xml_declaration=True, encoding="UTF-8")

with colx2:
    if etree is None:
        st.info("LandXML export requires `lxml`. Install it to enable proposed surface export.")
    else:
        try:
            pts, offs = build_proposed_surface_points(xs, ys, s, zc_bal, cs, modes, width)
            n_sta = len(xs)
            n_off = len(offs)
            xml_bytes = landxml_surface_from_grid(pts, n_sta, n_off)
            st.download_button("Download Proposed Surface LandXML", data=xml_bytes, file_name="ProposedSurface.landxml")
        except Exception as e:
            st.warning(f"LandXML export failed: {e}")

# -------- New: LandXML Alignment + Profile + Surface (single file) --------

st.markdown("### LandXML (Alignment + Profile + Surface)")
align_name = st.text_input("Alignment name", value="CL-1")
profile_name = st.text_input("Profile name", value="Proposed FG")


def landxml_align_profile_surface(xs, ys, s, zc, cs, modes, width, units_label, align_name="CL-1", profile_name="Proposed FG"):
    if etree is None:
        raise RuntimeError("lxml not installed")
    NS = "http://www.landxml.org/schema/LandXML-1.2"
    root = etree.Element("LandXML", nsmap={None: NS})
    root.set("version", "1.2")

    # Units
    units_elem = etree.SubElement(root, "Units")
    if str(units_label).startswith("US"):
        imp = etree.SubElement(units_elem, "Imperial")
        imp.set("linearUnit", "USSurveyFoot")
        imp.set("areaUnit", "SquareFoot")
        imp.set("volumeUnit", "CubicYard")
    else:
        met = etree.SubElement(units_elem, "Metric")
        met.set("linearUnit", "Meter")
        met.set("areaUnit", "SquareMeter")
        met.set("volumeUnit", "CubicMeter")

    # Alignment (as piecewise lines)
    aligns = etree.SubElement(root, "Alignments")
    algn = etree.SubElement(aligns, "Alignment")
    algn.set("name", str(align_name))
    coord = etree.SubElement(algn, "CoordGeom")
    for i in range(len(xs)-1):
        ln = etree.SubElement(coord, "Line")
        stp = etree.SubElement(ln, "Start"); stp.text = f"{xs[i]} {ys[i]}"
        enp = etree.SubElement(ln, "End");   enp.text = f"{xs[i+1]} {ys[i+1]}"

    # Profile (PVI at every station)
    profs = etree.SubElement(root, "Profiles")
    prof = etree.SubElement(profs, "Profile")
    prof.set("name", str(profile_name))
    prof.set("type", "FG")
    prof.set("alignRef", str(align_name))
    pal = etree.SubElement(prof, "ProfAlign")
    for i in range(len(s)):
        pvi = etree.SubElement(pal, "PVI")
        pvi.set("sta", f"{s[i]}")
        pvi.set("elev", f"{zc[i]}")

    # Surface (simple grid‑triangulated TIN)
    surfs = etree.SubElement(root, "Surfaces")
    surf = etree.SubElement(surfs, "Surface")
    surf.set("name", "ProposedSurface")
    defin = etree.SubElement(surf, "Definition")
    defin.set("surfType", "TIN")
    pts, offs = build_proposed_surface_points(xs, ys, s, zc, cs, modes, width)
    n_sta = len(xs); n_off = len(offs)
    for idx, (x, y, z) in enumerate(pts, start=1):
        p = etree.SubElement(defin, "P")
        p.set("id", str(idx))
        p.text = f"{x} {y} {z}"
    def pid(i_sta, i_off):
        return 1 + i_sta*n_off + i_off
    for i in range(n_sta-1):
        for j in range(n_off-1):
            a = pid(i, j); b = pid(i+1, j); c = pid(i+1, j+1); d = pid(i, j+1)
            f1 = etree.SubElement(defin, "F"); f1.text = f"{a} {b} {c}"
            f2 = etree.SubElement(defin, "F"); f2.text = f"{a} {c} {d}"

    return etree.tostring(root, pretty_print=True, xml_declaration=True, encoding="UTF-8")

if etree is None:
    st.info("LandXML alignment/profile export requires `lxml`.")
else:
    try:
        xml_full = landxml_align_profile_surface(xs, ys, s, zc_bal, cs, modes, width, units, align_name=align_name, profile_name=profile_name)
        st.download_button("Download LandXML (Alignment + Profile + Surface)", data=xml_full, file_name="Corridor_Align_Profile_Surface.landxml")
    except Exception as e:
        st.warning(f"Full LandXML export failed: {e}")

st.caption("Exports are planning‑grade. Validate in CAD (Civil 3D, OpenRoads) before submittal. Curves/VCs can be added later.")

# ============================
# 9) Minimal internal tests (run when GA_RUN_TESTS=1)
# ============================

def _run_internal_tests():
    # compute_grades basic behavior
    g = compute_grades(np.array([0.0, 0.1, 0.3]), ds=1.0)
    assert len(g) == 3, "grades length mismatch"
    assert abs(g[0] - 10.0) < 1e-6 and abs(g[1] - 20.0) < 1e-6, "grade calc error"

    # insert_landings: threshold strictly '>'
    z = np.array([0.0, 0.05, 0.10])  # ds=1 -> grades 5% and 5%
    z2, rngs = insert_landings(z, ds=1.0, max_walk_pct=5.0, landing_len=2.0)
    assert rngs == [], "should NOT insert when exactly at limit"

    # insert_landings: start at violation & merge rule
    z = np.array([0.0, 0.2, 0.4, 0.6, 0.6, 0.8, 1.0, 1.2])  # many 20% grades
    z3, rngs3 = insert_landings(z, ds=1.0, max_walk_pct=5.0, landing_len=4.0)  # k=4, half=2
    assert len(rngs3) >= 1 and rngs3[0][0] == 0, "landing should start at first violation"

    # insert_landings: endcap back‑extend to meet half length when near end
    z = np.array([0.0, 0.0, 0.0, 0.3, 0.6])  # violation at i=3, n=5
    z4, rngs4 = insert_landings(z, ds=1.0, max_walk_pct=5.0, landing_len=4.0)  # k=4, half=2
    a, b = rngs4[0]
    assert (b - a) >= 2, "should ensure at least half landing length near end"

    # corridor volumes smoke test (tiny)
    df_exist = pd.DataFrame({"x":[0,10,0,10],"y":[0,0,10,10],"z":[0,0,0,0]})
    df_path = pd.DataFrame({"x":[0,10],"y":[0,0]})
    s_loc, xs_loc, ys_loc = build_path_samples(df_path, 1.0)
    zc = np.zeros_like(xs_loc)
    cs_arr = np.full_like(xs_loc, 2.0, dtype=float)
    modes_arr = np.array(["crowned"]*len(xs_loc), dtype=object)
    dfv = corridor_slice_volumes(df_exist, xs_loc, ys_loc, zc, cs_arr, modes_arr, width=10.0, spacing=1.0)
    assert len(dfv) == len(xs_loc)-1, "slice volumes length mismatch"

    # LandXML (surface only)
    if etree is not None:
        xs_t = np.array([0.0, 10.0, 20.0])
        ys_t = np.array([0.0, 0.0, 0.0])
        s_t = np.array([0.0, 10.0, 20.0])
        zc_t = np.array([0.0, 0.0, 0.0])
        cs_t = np.array([2.0, 2.0, 2.0])
        modes_t = np.array(["crowned", "crowned", "crowned"], dtype=object)
        pts, offs = build_proposed_surface_points(xs_t, ys_t, s_t, zc_t, cs_t, modes_t, width=10.0)
        xml = landxml_surface_from_grid(pts, len(xs_t), len(offs))
        assert isinstance(xml, (bytes, bytearray)) and b"LandXML" in xml, "LandXML surface export failed"

        # LandXML (alignment + profile + surface)
        xml2 = landxml_align_profile_surface(xs_t, ys_t, s_t, zc_t, cs_t, modes_t, width=10.0, units_label="US (feet)")
        assert b"Alignments" in xml2 and b"Profiles" in xml2 and b"Surfaces" in xml2, "Full LandXML export missing sections"

if os.environ.get("GA_RUN_TESTS") == "1":
    try:
        _run_internal_tests()
        print("[Grading Assistant] Internal tests passed.")
    except Exception as e:
        print("[Grading Assistant] Internal tests failed:", e)
