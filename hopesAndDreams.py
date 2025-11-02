import json
from datetime import datetime, timedelta, timezone
from statistics import mean, median
from typing import Dict, List, Optional
from datetime import timezone

import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
import streamlit as st
import json

import os
import streamlit as st
import google.generativeai as genai

API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in .streamlit/secrets.toml or env.")
genai.configure(api_key=API_KEY)



# =========================
# CONFIG
# =========================
PARQUET_PATH = r"C:\Users\abdul\Desktop\durhack\2025_05.parquet"
EMISSIONS_PATH = None  # e.g. r"C:\path\to\emissions.csv" if you need external join; else None
FALLBACK_TONNES_PER_KM = 0.000115

# =========================
# CITY MAPPINGS & COORDS
# =========================
CITY_TO_IATA = {
    "London":   ["LHR", "LGW", "LTN", "STN", "LCY"],
    "Paris":    ["CDG", "ORY", "BVA"],
    "Hong Kong": ["HKG"],
    "Singapore": ["SIN"],
    "Mumbai":   ["BOM"],
    "Dubai":    ["DXB", "DWC"],
    "Shanghai": ["PVG", "SHA"],
    "Zurich":   ["ZRH"],
    "Geneva":   ["GVA"],
    "Aarhus":   ["AAR"],
    "Sydney":   ["SYD"],
    "Wroclaw":  ["WRO"],
    "Budapest": ["BUD"],
    "New York": ["JFK", "EWR", "LGA"],
}

CITY_COORDS = {
    "London": (51.5074, -0.1278),
    "Paris": (48.8566, 2.3522),
    "Hong Kong": (22.3193, 114.1694),
    "Singapore": (1.3521, 103.8198),
    "Mumbai": (19.0760, 72.8777),
    "Dubai": (25.2048, 55.2708),
    "Shanghai": (31.2304, 121.4737),
    "Zurich": (47.3769, 8.5417),
    "Geneva": (46.2044, 6.1432),
    "Aarhus": (56.1629, 10.2039),
    "Sydney": (-33.8688, 151.2093),
    "Wroclaw": (51.1079, 17.0385),
    "Budapest": (47.4979, 19.0402),
    "New York": (40.71278, -74.0060),
}

# =========================
# HELPERS
# =========================
def parse_iso_utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def to_utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def add_datetime_cols_lf(lf: pl.LazyFrame) -> pl.LazyFrame:
    fmt = "%Y-%m-%d %H:%M:%S%.f"
    return lf.with_columns(
        pl.col("SCHEDULED_DEPARTURE_DATE_TIME_UTC").cast(pl.Utf8)
            .str.strptime(pl.Datetime, fmt, strict=False)
            .dt.replace_time_zone("UTC").alias("DEP_UTC"),
        pl.col("SCHEDULED_ARRIVAL_DATE_TIME_UTC").cast(pl.Utf8)
            .str.strptime(pl.Datetime, fmt, strict=False)
            .dt.replace_time_zone("UTC").alias("ARR_UTC"),
    )

def join_emissions_lf(sched_slice_lf: pl.LazyFrame) -> pl.LazyFrame:
    if not EMISSIONS_PATH:
        return sched_slice_lf
    if EMISSIONS_PATH.lower().endswith(".parquet"):
        em = pl.scan_parquet(EMISSIONS_PATH)
    else:
        em = pl.scan_csv(EMISSIONS_PATH, infer_schema_length=10000)

    s = sched_slice_lf.with_columns(
        pl.col("CARRIER").cast(pl.Utf8).alias("CARRIER_STR"),
        pl.col("FLTNO").cast(pl.Utf8).alias("FLTNO_STR"),
    )
    e = em.with_columns(
        pl.col("CARRIER_CODE").cast(pl.Utf8).alias("CARRIER_STR"),
        pl.col("FLIGHT_NUMBER").cast(pl.Utf8).alias("FLTNO_STR"),
    )
    return (
        s.join(e, on=["CARRIER_STR", "FLTNO_STR"], how="left")
         .drop(["CARRIER_STR","FLTNO_STR"])
         .with_columns(
            pl.when(pl.col("ESTIMATED_CO2_TOTAL_TONNES").is_not_null())
              .then(pl.col("ESTIMATED_CO2_TOTAL_TONNES").cast(pl.Float64, strict=False))
              .otherwise(None).alias("CO2_TONNES")
         )
    )

def flow_map(destination_city: str, attendees: Dict[str,int], per_origin_hours: Dict[str,float]) -> go.Figure:
    fig = go.Figure()
    # destination
    if destination_city in CITY_COORDS:
        dlat, dlon = CITY_COORDS[destination_city]
        fig.add_trace(go.Scattergeo(lon=[dlon], lat=[dlat], mode="markers+text",
                                    text=[f"{destination_city}<br><b>Host</b>"],
                                    textposition="top center",
                                    marker=dict(size=16, symbol="star", line=dict(width=1)),
                                    hoverinfo="text", name="Destination"))
    # origins + flows
    for origin, count in attendees.items():
        if origin not in CITY_COORDS or destination_city not in CITY_COORDS:
            continue
        olat, olon = CITY_COORDS[origin]
        dlat, dlon = CITY_COORDS[destination_city]
        hours = per_origin_hours.get(origin, None)
        if origin != destination_city:
            fig.add_trace(go.Scattergeo(
                lon=[olon, dlon], lat=[olat, dlat], mode="lines",
                line=dict(width=max(1, min(6, 0.8 * count))),
                name=f"{origin} → {destination_city}", showlegend=False,
                hovertext=f"{origin} → {destination_city}<br>Attendees: {count}"
                          + (f"<br>Total travel: {hours:.1f}h" if hours is not None else ""),
                hoverinfo="text"
            ))
        fig.add_trace(go.Scattergeo(
            lon=[olon], lat=[olat], mode="markers+text",
            text=[f"{origin}<br>{count}"], textposition="bottom center",
            marker=dict(size=8), name="Origin", showlegend=False
        ))
    fig.update_layout(
        title=f"Travel flows → {destination_city}",
        geo=dict(projection_type="natural earth", showland=True, landcolor="rgb(240,240,240)", coastlinecolor="rgb(180,180,180)"),
        margin=dict(l=0,r=0,t=50,b=0), height=520
    )
    return fig


#gemini stuff:
def ask_gemini_for_hotels_main(city, start_iso, end_iso, party_size, budget_hint=None, vibe=None):
    """Returns a python list of dicts with hotel suggestions."""
    model = genai.GenerativeModel(MODEL_NAME)

    # Ask for structured JSON so you can render cleanly:
    system = (
        "You are a travel assistant. Return ONLY valid JSON (no markdown) "
        "with an array 'hotels', each item having:\n"
        "{name, neighborhood, description, sustainability_hint, distance_km, "
        "price_tier, why_good_for_group, suggested_booking_sites}\n"
        "Keep distance_km numeric, price_tier in {budget, mid, upper, luxury}."
    )
    user = (
        f"Destination: {city}\n"
        f"Dates: {start_iso} to {end_iso}\n"
        f"Group size: {party_size}\n"
        f"Budget hint: {budget_hint or 'mid'}\n"
        f"Vibe: {vibe or 'business-friendly, walkable to meeting area, quiet rooms'}\n"
        "Prefer properties with meeting rooms, good Wi-Fi, late check-in, and transit access."
    )

    resp = model.generate_content([system, user])
    text = resp.text.strip()

    # Robust parse: if model wraps in code fences, strip them
    if text.startswith("```"):
        text = text.strip("`")
        # e.g. ```json ... ```
        text = text.split("\n", 1)[1] if "\n" in text else text

    try:
        data = json.loads(text)
        return data.get("hotels", [])
    except Exception:
        # Fallback: return a tiny safe default if parsing fails
        return []

# =========================
# DATA LOADING
# =========================
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pl.DataFrame:
    lf = pl.scan_parquet(path)
    lf = add_datetime_cols_lf(lf)
    # Keep lean set of columns; add CO2 if you later join
    cols = ["CARRIER","FLTNO","DEPAPT","ARRAPT","DEP_UTC","ARR_UTC","ELPTIM","DISTANCE","STOPS"]
    if EMISSIONS_PATH:
        lf = join_emissions_lf(lf)
        cols += ["CO2_TONNES"]
    return lf.select(cols).collect(streaming=True)

def ensure_utc_py(dt_obj):
    """
    Return a Python datetime guaranteed to be tz-aware in UTC,
    whether input is pandas Timestamp, numpy datetime64, or naive datetime.
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime

    if isinstance(dt_obj, pd.Timestamp):
        # If tz-naive → localize; if tz-aware → convert
        return (dt_obj.tz_localize("UTC") if dt_obj.tzinfo is None else dt_obj.tz_convert("UTC")).to_pydatetime()

    if isinstance(dt_obj, np.datetime64):
        return pd.to_datetime(dt_obj, utc=True).to_pydatetime()

    if isinstance(dt_obj, datetime):
        return dt_obj if dt_obj.tzinfo is not None else dt_obj.replace(tzinfo=timezone.utc)

    # Fallback: let pandas parse anything else
    ts = pd.to_datetime(dt_obj, utc=True)
    return ts.to_pydatetime()


# =========================
# CANDIDATE BUILDING & SCORING
# =========================
def slice_candidates(df: pl.DataFrame, dep_iata: List[str], arr_iata: List[str],
                     tmin: datetime, tmax: datetime, direct_only: bool) -> pl.DataFrame:
    out = df.filter(
        pl.col("DEPAPT").is_in(dep_iata) &
        pl.col("ARRAPT").is_in(arr_iata) &
        (pl.col("DEP_UTC") >= pl.lit(tmin)) &
        (pl.col("DEP_UTC") <= pl.lit(tmax))
    )
    if direct_only and "STOPS" in out.columns:
        out = out.filter(pl.col("STOPS") == 0)
    return out

def normalize_series(s: pd.Series) -> pd.Series:
    # Min-max normalize; fall back to zeros if flat
    vmin, vmax = s.min(), s.max()
    if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - vmin) / (vmax - vmin)

def score_leg_table(pdf: pd.DataFrame, w_time: float, w_dist: float, w_co2: float) -> pd.DataFrame:
    # Build columns: minutes, km, CO2_tonnes (fallback if missing)
    m = pdf["ELPTIM"].astype(float)             # minutes
    d = pdf["DISTANCE"].fillna(0).astype(float) # km
    if "CO2_TONNES" in pdf.columns:
        c = pdf["CO2_TONNES"].astype(float)
        # fallback per-row if CO2 missing
        c = c.where(c.notna(), d * FALLBACK_TONNES_PER_KM)
    else:
        c = d * FALLBACK_TONNES_PER_KM

    # Normalize each signal
    n_m = normalize_series(m)    # shorter better
    n_d = normalize_series(d)    # shorter better
    n_c = normalize_series(c)    # lower better

    # Weighted sum (all to minimize)
    score = w_time * n_m + w_dist * n_d + w_co2 * n_c
    out = pdf.copy()
    out["__score__"] = score
    out["__mins__"] = m
    out["__km__"] = d
    out["__co2__"] = c
    return out

def choose_best_roundtrip(out_df: pl.DataFrame, ret_df: pl.DataFrame,
                          w_time: float, w_dist: float, w_co2: float,
                          top_k: int = 30) -> Optional[dict]:
    if out_df.height == 0 or ret_df.height == 0:
        return None
    o = out_df.sort("ARR_UTC").head(top_k).to_pandas()
    r = ret_df.sort("DEP_UTC").head(top_k).to_pandas()

    o_sc = score_leg_table(o, w_time, w_dist, w_co2)
    r_sc = score_leg_table(r, w_time, w_dist, w_co2)

    # Cartesian over top K each (K^2 small), pick min total score
    best = None; best_score = float("inf")
    for _, ro in o_sc.iterrows():
        for _, rr in r_sc.iterrows():
            total_score = ro["__score__"] + rr["__score__"]
            if total_score < best_score:
                best_score = total_score
                best = {
                    "outbound": ro.to_dict(),
                    "return": rr.to_dict(),
                    "total_hours": (float(ro["ELPTIM"]) + float(rr["ELPTIM"])) / 60.0,
                    "total_co2": float(ro["__co2__"] + rr["__co2__"]),
                    "arr_at_host": ro["ARR_UTC"],
                    "dep_from_host": rr["DEP_UTC"]
                }
    return best

# =========================
# OPTIMIZATION PIPELINE
# =========================
def optimize_for_destination(df: pl.DataFrame,
                             destination: str,
                             attendees: Dict[str,int],
                             win_start: datetime,
                             win_end: datetime,
                             duration: timedelta,
                             w_time: float, w_dist: float, w_co2: float,
                             direct_only: bool = True) -> Optional[dict]:
    dest_airports = CITY_TO_IATA.get(destination, [])
    if not dest_airports:
        return None

    # buffers
    arrival_buffer = timedelta(hours=3)
    departure_buffer = timedelta(hours=2)

    # Step A: for each origin, build candidates, choose best round-trip by weighted score
    per_origin_plan = {}
    per_attendee_hours = []
    per_origin_hours = {}
    any_missing = False

    for origin_city, count in attendees.items():
        if origin_city == destination:
            per_origin_plan[origin_city] = None
            per_origin_hours[origin_city] = 0.0
            per_attendee_hours += [0.0] * int(count)
            continue

        origin_airports = CITY_TO_IATA.get(origin_city, [])
        if not origin_airports:
            any_missing = True
            continue

        # Outbound must ARRIVE within the availability window
        out_cand = df.filter(
            pl.col("DEPAPT").is_in(origin_airports) &
            pl.col("ARRAPT").is_in(dest_airports) &
            (pl.col("ARR_UTC") >= pl.lit(win_start)) &
            (pl.col("ARR_UTC") <= pl.lit(win_end))
        )
        if direct_only:
            out_cand = out_cand.filter(pl.col("STOPS") == 0)

        # We'll choose returns later once we know event_end; for now pre-slice a broad set
        ret_cand_broad = df.filter(
            pl.col("DEPAPT").is_in(dest_airports) &
            pl.col("ARRAPT").is_in(origin_airports) &
            (pl.col("DEP_UTC") >= pl.lit(win_start))
        )
        if direct_only:
            ret_cand_broad = ret_cand_broad.filter(pl.col("STOPS") == 0)

        # choose provisional best outbound assuming return exists later
        out_pd = out_cand.sort("ARR_UTC").head(150).to_pandas()
        if out_pd.empty:
            any_missing = True
            continue
        out_scored = score_leg_table(out_pd, w_time, w_dist, w_co2).sort_values("__score__")
        # keep top N to try with returns later
        per_origin_plan[origin_city] = {
            "out_candidates": out_scored.head(40),   # pandas df
            "ret_candidates": ret_cand_broad.sort("DEP_UTC").head(400).to_pandas()  # pandas df
        }

    # If we missed anyone entirely, destination infeasible
    if any_missing:
        return None

    # Step B: pick one outbound per origin (best score) → derive event_start
    chosen_out = {}
    for origin, packs in per_origin_plan.items():
        if packs is None:  # local
            continue
        chosen_out[origin] = dict(packs["out_candidates"].iloc[0])

    # Event start = latest arrival + buffer; event end = start + duration
    event_start = max(ensure_utc_py(pd.to_datetime(d["ARR_UTC"])) for d in chosen_out.values()) + arrival_buffer
    if event_start > win_end:
        return None
    event_end = event_start + duration

    # Step C: choose best RETURN per origin given event_end + buffer
    per_origin_detail = {}
    for origin, packs in per_origin_plan.items():
        if packs is None:  # local
            per_origin_detail[origin] = {
                "outbound": None, "return": None, "total_hours": 0.0, "total_co2": 0.0,
                "arr_at_host": event_start - arrival_buffer, "dep_from_host": event_end + departure_buffer
            }
            continue

        ret_min = event_end + departure_buffer
        ret_pdf = packs["ret_candidates"]
        # filter in pandas (fast enough for the small subset)
        ret_pdf = ret_pdf[ret_pdf["DEP_UTC"] >= pd.Timestamp(ret_min)]
        # pick best round-trip by combined score
        out_df_pl = pl.from_pandas(packs["out_candidates"])
        ret_df_pl = pl.from_pandas(ret_pdf)
        best = choose_best_roundtrip(out_df_pl, ret_df_pl, w_time, w_dist, w_co2, top_k=20)
        if best is None:
            return None
        per_origin_detail[origin] = best

    # Step D: recompute final event overlap (everyone on-site) and span
    intervals = []
    for origin, plan in per_origin_detail.items():
        if origin == destination:
            intervals.append((event_start - arrival_buffer, event_end + departure_buffer, origin))
        else:
            arr = ensure_utc_py(plan["arr_at_host"])
            dep = ensure_utc_py(plan["dep_from_host"])

            intervals.append((arr + arrival_buffer, dep - departure_buffer, origin))

    overlap_start = max(s for s, e, _ in intervals)
    overlap_end = min(e for s, e, _ in intervals)
    if overlap_start + duration > overlap_end:
        # slide to earliest feasible if possible
        event_start2 = overlap_start
        event_end2 = event_start2 + duration
        if event_end2 > win_end:
            return None
        event_start, event_end = event_start2, event_end2

    span_start = min(s for s, _, _ in intervals)
    span_end = max(e for _, e, _ in intervals)

    # Step E: aggregate stats
    per_origin_hours = {}
    total_co2_t = 0.0
    for origin, count in attendees.items():
        if origin == destination:
            per_origin_hours[origin] = 0.0
            continue
        plan = per_origin_detail[origin]
        per_origin_hours[origin] = float(plan["total_hours"])
        total_co2_t += float(plan["total_co2"]) * float(count)

    # weighted individual list
    weighted_hours = []
    for origin, n in attendees.items():
        weighted_hours += [per_origin_hours.get(origin, 0.0)] * int(n)

    avg_h = float(mean(weighted_hours)) if weighted_hours else 0.0
    med_h = float(median(weighted_hours)) if weighted_hours else 0.0
    max_h = max(weighted_hours) if weighted_hours else 0.0
    min_h = min(weighted_hours) if weighted_hours else 0.0
    fairness_sd = float(np.std(np.array(weighted_hours))) if weighted_hours else 0.0

    return {
        "destination": destination,
        "event_dates": {"start": event_start, "end": event_end},
        "event_span": {"start": span_start, "end": span_end},
        "total_co2_tonnes": total_co2_t,
        "average_travel_hours": avg_h,
        "median_travel_hours": med_h,
        "max_travel_hours": max_h,
        "min_travel_hours": min_h,
        "fairness_score": fairness_sd,
        "attendee_travel_hours": per_origin_hours,
        "plans": per_origin_detail
    }

def optimize_across_hosts(df: pl.DataFrame,
                          attendees: Dict[str,int],
                          win_start: datetime,
                          win_end: datetime,
                          duration: timedelta,
                          w_time: float, w_dist: float, w_co2: float,
                          direct_only: bool) -> List[dict]:
    candidates = sorted(set(CITY_TO_IATA.keys()) | set(attendees.keys()))
    results = []
    prog = st.progress(0.0); status = st.empty()
    for i, dest in enumerate(candidates):
        status.text(f"Evaluating {dest} ({i+1}/{len(candidates)})…")
        res = optimize_for_destination(
            df, dest, attendees, win_start, win_end, duration,
            w_time, w_dist, w_co2, direct_only=direct_only
        )
        if res is not None:
            results.append(res)
        prog.progress((i+1)/len(candidates))
    prog.empty(); status.empty()
    if not results:
        return []
    # Rank by composite (you can show both fairness + CO2 + time)
    # Normalize each: lower is better
    co2 = [r["total_co2_tonnes"] for r in results]
    avg = [r["average_travel_hours"] for r in results]
    fair = [r["fairness_score"] for r in results]
    def mm(v):
        mi, ma = min(v), max(v)
        return [(x-mi)/(ma-mi) if ma>mi else 0.0 for x in v]
    n_co2, n_avg, n_fair = mm(co2), mm(avg), mm(fair)
    # Use same weights for ranking as leg scoring on time/CO2, and include fairness equally with time
    # You can tweak: composite = w_time*n_avg + w_co2*n_co2 + 0.5*w_time*n_fair
    for idx, r in enumerate(results):
        r["composite_score"] = (w_time*n_avg[idx] + w_co2*n_co2[idx] + 0.5*w_time*n_fair[idx])
    return sorted(results, key=lambda x: x["composite_score"])

# =========================
# STREAMLIT APP
# =========================
def main():
    st.set_page_config(page_title="Meeting Optimizer", layout="wide")
    st.title("🌍 Meeting Location Optimizer")
    st.caption("Adjust weights for Time, Distance, Emissions. The app finds per-origin optimal routings and the best host city (optional).")

    with st.spinner("Loading flights…"):
        df = load_data(PARQUET_PATH)
    st.success(f"Loaded {df.height:,} flights")

    # Sidebar weights (scroller wheels)
    st.sidebar.header("Optimization Weights")
    w_time = st.sidebar.slider("Weight: Travel Time", 0.0, 1.0, 0.5, 0.05)
    w_dist = st.sidebar.slider("Weight: Distance",    0.0, 1.0, 0.25, 0.05)
    w_co2  = st.sidebar.slider("Weight: Emissions",   0.0, 1.0, 0.25, 0.05)
    total = w_time + w_dist + w_co2
    if total == 0:
        w_time, w_dist, w_co2 = 1.0, 0.0, 0.0
    else:
        w_time, w_dist, w_co2 = w_time/total, w_dist/total, w_co2/total

    direct_only = st.sidebar.checkbox("Direct flights only", value=True)
    auto_host = st.sidebar.checkbox("Automatically choose best host", value=True)

    # Scenario input
    st.subheader("📝 Scenario")
    default_scn = {
        "attendees": {"London": 4, "Paris": 10, "Zurich": 7, "Geneva": 1},
        "availability_window": {"start": "2025-08-04T12:30:00Z", "end": "2025-08-08T12:00:00Z"},
        "event_duration": {"days": 1, "hours": 2}
    }
    scenario_text = st.text_area("Paste scenario JSON", value=json.dumps(default_scn, indent=2), height=180)
    try:
        scn = json.loads(scenario_text)
    except Exception as e:
        st.error(f"Invalid JSON: {e}")
        st.stop()

    attendees = scn["attendees"]
    win_start = parse_iso_utc(scn["availability_window"]["start"])
    win_end   = parse_iso_utc(scn["availability_window"]["end"])
    duration  = timedelta(days=scn["event_duration"]["days"], hours=scn["event_duration"]["hours"])

    st.caption(f"Window: {win_start.isoformat()} → {win_end.isoformat()} | Duration: {duration}")

    if auto_host:
        if st.button("🚀 Optimize across host cities", type="primary"):
            with st.spinner("Optimizing across hosts…"):
                results = optimize_across_hosts(df, attendees, win_start, win_end, duration, w_time, w_dist, w_co2, direct_only)
            if not results:
                st.error("No feasible host found. Try widening the window or disabling 'direct only'.")
                st.stop()

            best = results[0]
            # Persist for other pages
            st.session_state["chosen_city"] = best["destination"]
            st.session_state["event_start_iso"] = best["event_dates"]["start"].isoformat()
            st.session_state["event_end_iso"]   = best["event_dates"]["end"].isoformat()
            st.session_state["party_size"]      = sum(scn["attendees"].values())  # or scenario[...] if that's your var
            st.session_state["last_optimization"] = best  # keep the whole object too
            with st.expander("Debug: session_state keys"):
                st.write({k: st.session_state.get(k) for k in [
                    "chosen_city","event_start_iso","event_end_iso","party_size","last_optimization"
                ]})

            st.success(f"Recommended host: **{best['destination']}**")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("🌱 Total CO₂", f"{best['total_co2_tonnes']:.2f} t")
            c2.metric("⏱ Avg hours", f"{best['average_travel_hours']:.2f}")
            c3.metric("📊 Fairness (sd)", f"{best['fairness_score']:.2f}")
            span_hours = (best["event_span"]["end"] - best["event_span"]["start"]).total_seconds()/3600
            c4.metric("🗓 Span (hrs)", f"{span_hours:.1f}")

            t1,t2,t3,t4 = st.tabs(["🗺 Map","📋 Details","📊 Compare","📤 Export"])
            with t1:
                st.plotly_chart(flow_map(best["destination"], attendees, best["attendee_travel_hours"]), use_container_width=True)

            with t2:
                st.markdown("#### Event timing")
                st.write(f"**Event Dates:** {to_utc_iso(best['event_dates']['start'])} → {to_utc_iso(best['event_dates']['end'])}")
                st.write(f"**Event Span:**  {to_utc_iso(best['event_span']['start'])} → {to_utc_iso(best['event_span']['end'])}")

                st.markdown("#### Attendee travel hours (round-trip)")
                st.json({k: round(v,2) for k,v in best["attendee_travel_hours"].items()})

                st.markdown("#### Per-origin itinerary choices")
                rows = []
                for origin, plan in best["plans"].items():
                    if origin == best["destination"]:
                        rows.append({"Origin": origin, "Outbound":"Local", "Return":"Local", "Hours (rt)":"0.0", "CO₂ (t)":"0.00"})
                        continue
                    o = plan["outbound"]; r = plan["return"]
                    od = f"{o['DEPAPT']}→{o['ARRAPT']} {to_utc_iso(ensure_utc_py(o['DEP_UTC']))}→{to_utc_iso(ensure_utc_py(o['ARR_UTC']))}"
                    rd = f"{r['DEPAPT']}→{r['ARRAPT']} {to_utc_iso(ensure_utc_py(r['DEP_UTC']))}→{to_utc_iso(ensure_utc_py(r['ARR_UTC']))}"

                    rows.append({
                        "Origin": origin,
                        "Outbound": od,
                        "Return": rd,
                        "Hours (rt)": f"{plan['total_hours']:.1f}",
                        "CO₂ (t)": f"{plan['total_co2']:.2f}",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            with t3:
                top = results[:10]
                comp = pd.DataFrame([{
                    "Rank": i+1,
                    "City": r["destination"],
                    "CO₂ (t)": f"{r['total_co2_tonnes']:.2f}",
                    "Avg h": f"{r['average_travel_hours']:.2f}",
                    "Fairness sd": f"{r['fairness_score']:.2f}",
                    "Score": f"{r['composite_score']:.3f}",
                } for i,r in enumerate(top)])
                st.dataframe(comp, use_container_width=True)

            with t4:
                payload = {
                    "event_location": best["destination"],
                    "event_dates": {"start": to_utc_iso(best["event_dates"]["start"]), "end": to_utc_iso(best["event_dates"]["end"])},
                    "event_span": {"start": to_utc_iso(best["event_span"]["start"]), "end": to_utc_iso(best["event_span"]["end"])},
                    "total_co2": round(best["total_co2_tonnes"], 2),
                    "average_travel_hours": round(best["average_travel_hours"], 2),
                    "median_travel_hours": round(best["median_travel_hours"], 2),
                    "max_travel_hours": round(best["max_travel_hours"], 2),
                    "min_travel_hours": round(best["min_travel_hours"], 2),
                    "attendee_travel_hours": {k: round(v,2) for k,v in best["attendee_travel_hours"].items()},
                }
                st.markdown("#### Judging JSON")
                st.code(json.dumps(payload, indent=2), language="json")
                st.download_button("Download JSON", data=json.dumps(payload, indent=2),
                                   file_name=f"meeting_{best['destination'].lower()}.json", mime="application/json")

    else:
        # Manual single-destination optimization using the same weighted leg scoring
        dest = st.selectbox("Destination", list(CITY_TO_IATA.keys()), index=0)
        if st.button(f"Run optimization for {dest}"):
            res = optimize_for_destination(
                df, dest, attendees, win_start, win_end, duration,
                w_time, w_dist, w_co2, direct_only=direct_only
            )
            if not res:
                st.error("Destination infeasible with current settings. Try widening the window or disabling 'direct only'.")
                st.stop()

            st.success(f"Optimized itineraries for host: **{dest}**")
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("🌱 Total CO₂", f"{res['total_co2_tonnes']:.2f} t")
            k2.metric("⏱ Avg hours", f"{res['average_travel_hours']:.2f}")
            k3.metric("📊 Fairness (sd)", f"{res['fairness_score']:.2f}")
            span_hours = (res["event_span"]["end"] - res["event_span"]["start"]).total_seconds()/3600
            k4.metric("🗓 Span (hrs)", f"{span_hours:.1f}")

            st.plotly_chart(flow_map(dest, attendees, res["attendee_travel_hours"]), use_container_width=True)

            st.markdown("#### Event timing")
            st.write(f"**Event Dates:** {to_utc_iso(res['event_dates']['start'])} → {to_utc_iso(res['event_dates']['end'])}")
            st.write(f"**Event Span:**  {to_utc_iso(res['event_span']['start'])} → {to_utc_iso(res['event_span']['end'])}")

            st.markdown("#### Attendee travel hours (round-trip)")
            st.json({k: round(v,2) for k,v in res["attendee_travel_hours"].items()})

            st.markdown("#### Per-origin itinerary choices")
            rows = []
            for origin, plan in res["plans"].items():
                if origin == dest:
                    rows.append({"Origin": origin, "Outbound":"Local", "Return":"Local", "Hours (rt)":"0.0", "CO₂ (t)":"0.00"})
                else:
                    o = plan["outbound"]; r = plan["return"]
                    od = f"{o['DEPAPT']}→{o['ARRAPT']} {to_utc_iso(pd.Timestamp(o['DEP_UTC']).to_pydatetime())}→{to_utc_iso(pd.Timestamp(o['ARR_UTC']).to_pydatetime())}"
                    rd = f"{r['DEPAPT']}→{r['ARRAPT']} {to_utc_iso(pd.Timestamp(r['DEP_UTC']).to_pydatetime())}→{to_utc_iso(pd.Timestamp(r['ARR_UTC']).to_pydatetime())}"
                    rows.append({"Origin": origin, "Outbound": od, "Return": rd,
                                 "Hours (rt)": f"{plan['total_hours']:.1f}", "CO₂ (t)": f"{plan['total_co2']:.2f}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

if __name__ == "__main__":
    main()
