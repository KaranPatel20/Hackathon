"""Living Flood Map - Streamlit UI.

Reads the precomputed datasets in data/processed/ (built by pipeline/build_main.py and
pipeline/build_bonus.py) and gives judges a clean, filterable map + tweet list. No local
model or Gemini calls are needed to browse a dataset; "Summarize this view" makes one
Gemini call if GEMINI_API_KEY is set, otherwise it falls back to a plain excerpt.

Run:
    .venv/Scripts/streamlit run streamlit_app/app.py
"""
import gzip
import io
import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.cascade import process  # noqa: E402
from pipeline.gemini import Gemini  # noqa: E402
from pipeline.model import LocalModel  # noqa: E402
from pipeline.summary import summarize  # noqa: E402
from pipeline.taxonomy import CATEGORIES, HAZARDS  # noqa: E402
from pipeline.world import country_name  # noqa: E402

PROCESSED = ROOT / "data/processed"
URG_LABELS = {0: "None", 1: "Info", 2: "Active impact", 3: "Life / safety"}
CAT_COLORS = {
    "INFRA": "#f4874b", "EVAC": "#e8578f", "HUMAN": "#e2504f", "PROPERTY": "#9a6bd6",
    "NEEDS": "#e0a72e", "AID": "#4caf6f", "ADVISORY": "#3b8ce0", "SUPPORT": "#2fb8ae", "OTHER": "#8a94a6",
}
DATASET_STEMS = {"main": "main_results", "bonus": "bonus_results"}
UPLOAD_BUDGET = int(os.getenv("UPLOAD_CALL_BUDGET", "20"))

st.set_page_config(page_title="Living Flood Map", layout="wide", initial_sidebar_state="expanded")


def to_records(df: pd.DataFrame) -> list[dict]:
    """Same record shape as the precomputed datasets (app/server.py:to_records), for uploaded CSVs."""
    recs = []
    for r in df.itertuples():
        recs.append({
            "id": int(r.row_id), "g": int(r.group_id), "n": int(r.group_size), "rt": bool(r.is_retweet),
            "canon": bool(r.is_canonical), "text": r.text, "rel": int(r.relevant),
            "p": round(float(r.p_relevant), 3), "conf": round(float(r.confidence), 2),
            "cat": r.category if isinstance(r.category, str) else "", "urg": int(r.urgency),
            "hz": (getattr(r, "hazard", "") or "") if isinstance(getattr(r, "hazard", ""), str) else "",
            "cc": getattr(r, "cc", None) if isinstance(getattr(r, "cc", None), str) else None,
            "by": r.decided_by,
            "places": [{"name": p["name"], "lat": p["lat"], "lon": p["lon"], "prec": p.get("precision", "point"),
                        "src": p.get("source", ""), "cc": p.get("cc")}
                       for p in (r.places if isinstance(r.places, list) else [])],
            "ts": (r.timestamp.isoformat() if hasattr(r, "timestamp") and pd.notna(r.timestamp) else None),
        })
    return recs


# ------------------------------------------------------------------ data loading
@st.cache_data(show_spinner=False)
def discover_datasets() -> dict:
    out = {}
    for ds_id, stem in DATASET_STEMS.items():
        meta_p = PROCESSED / f"{stem}.meta.json"
        plain_p = PROCESSED / f"{stem}.json"
        if meta_p.exists():
            out[ds_id] = json.loads(meta_p.read_text())["name"]
        elif plain_p.exists():
            out[ds_id] = json.loads(plain_p.read_text())["name"]
    return out


@st.cache_data(show_spinner="Loading dataset...")
def load_builtin_records(ds_id: str) -> pd.DataFrame:
    stem = DATASET_STEMS[ds_id]
    meta_p, gz_p, plain_p = PROCESSED / f"{stem}.meta.json", PROCESSED / f"{stem}.json.gz", PROCESSED / f"{stem}.json"
    if meta_p.exists() and gz_p.exists():
        with gzip.open(gz_p, "rt", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = json.loads(plain_p.read_text())
    return records_df(payload["records"])


def records_df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["cat"] = df["cat"].fillna("").replace({None: ""})
    df["hz"] = df["hz"].fillna("").replace({None: ""})
    df["cc"] = df["cc"].where(df["cc"].notna(), None)
    return df


def load_records(ds_id: str) -> pd.DataFrame:
    upload = st.session_state.get("uploads", {}).get(ds_id)
    if upload is not None:
        return records_df(upload["records"])
    return load_builtin_records(ds_id)


def top_places(ds_id: str) -> list[str]:
    df = load_records(ds_id)
    counts = {}
    for places in df.loc[(df["rel"] == 1) & df["canon"], "places"]:
        for p in places:
            counts[p["name"]] = counts.get(p["name"], 0) + 1
    return [n for n, _ in sorted(counts.items(), key=lambda kv: -kv[1])]


def top_countries(ds_id: str) -> list[str]:
    df = load_records(ds_id)
    counts = df.loc[(df["rel"] == 1) & df["canon"], "cc"].dropna()
    counts = counts[counts != ""]
    return [cc for cc, _ in counts.value_counts().items()]


# ------------------------------------------------------------------ filtering
def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    d = df
    if f["unique_only"]:
        d = d[d["canon"]]
    if f["relevance"] == "Relevant":
        d = d[d["rel"] == 1]
    elif f["relevance"] == "Not relevant":
        d = d[d["rel"] == 0]
    elif f["relevance"] == "Uncertain":
        unsure = ((d["by"] == "local") & d["p"].between(0.15, 0.85)) | (d["conf"] < 0.5)
        d = d[unsure]
    if f["hazards"]:
        d = d[d["hz"].isin(f["hazards"]) | (d["hz"] == "")]
    if f["categories"]:
        d = d[d["cat"].isin(f["categories"])]
    if f["urgencies"]:
        d = d[d["urg"].isin(f["urgencies"])]
    if f["countries"]:
        d = d[d["cc"].isin(f["countries"])]
    if f["places"]:
        wanted = set(f["places"])
        d = d[d["places"].map(lambda ps: any(p["name"] in wanted for p in ps))]
    if f["mapped_only"]:
        d = d[d["places"].map(lambda ps: any(p.get("prec") != "region" for p in ps))]
    if f["query"]:
        q = f["query"].lower()
        d = d[d["text"].str.lower().str.contains(q, regex=False)]
    return d


def aggregate_places(view: pd.DataFrame, include_region: bool) -> pd.DataFrame:
    rows = []
    for r in view.itertuples():
        for p in r.places:
            if not include_region and p.get("prec") == "region":
                continue
            rows.append({"name": p["name"], "lat": p["lat"], "lon": p["lon"],
                         "cat": r.cat or "OTHER", "urg": r.urg})
    if not rows:
        return pd.DataFrame(columns=["name", "lat", "lon", "n", "max_urg", "dom_cat"])
    pdf = pd.DataFrame(rows)
    agg = pdf.groupby("name").agg(
        lat=("lat", "first"), lon=("lon", "first"), n=("name", "size"), max_urg=("urg", "max"),
        dom_cat=("cat", lambda s: s.value_counts().idxmax()),
    ).reset_index()
    return agg


# ------------------------------------------------------------------ sidebar
uploads = st.session_state.setdefault("uploads", {})
datasets = {**discover_datasets(), **{k: v["name"] for k, v in uploads.items()}}
if not datasets:
    st.error("No datasets found in data/processed/. Run the build pipeline first.")
    st.stop()

with st.sidebar:
    st.title("Living Flood Map")
    ds_id = st.selectbox("Dataset", list(datasets), format_func=lambda k: datasets[k], key="ds_select")

    with st.expander("Upload your own CSV"):
        up_file = st.file_uploader("CSV with a tweet/text column", type=["csv"])
        budget = st.number_input(
            "Gemini call budget for this upload", min_value=0, max_value=UPLOAD_BUDGET, value=0,
            help="0 = local model only, no Gemini calls even if a key is configured. Each call reviews "
                 "up to 200 tweets.",
        )
        if up_file is not None and st.button("Process upload", type="primary"):
            with st.spinner("Cleaning, scoring and mapping the upload (can take a minute)..."):
                raw = pd.read_csv(io.BytesIO(up_file.getvalue()), encoding_errors="replace", on_bad_lines="skip")
                df, stats = process(raw, budget=int(budget), gemini=Gemini(budget=int(budget)),
                                     model=LocalModel.load())
                key = f"up_{abs(hash(up_file.getvalue())) % 10**10}"
                uploads[key] = {"name": up_file.name, "records": to_records(df), "stats": stats}
                st.session_state["ds_select"] = key
            st.rerun()

    st.divider()

    st.subheader("Disaster type")
    hazard_options = list(HAZARDS)
    hazards = st.multiselect(
        "Show only these hazards", hazard_options, default=["FLOOD"],
        help="Excludes tweets labeled as a different disaster type (earthquake, fire, storm, ...). "
             "Tweets with no hazard label (not flood-relevant) are unaffected by this filter.",
    )

    st.subheader("Relevance")
    relevance = st.selectbox("Relevance", ["Relevant", "Uncertain", "Not relevant", "All"], index=0)
    unique_only = st.checkbox("Unique tweets only (fold retweets & copies)", value=True)

    st.subheader("Filters")
    categories = st.multiselect("Impact category", list(CATEGORIES), default=[])
    urgencies = st.multiselect(
        "Urgency", [3, 2, 1, 0], default=[], format_func=lambda u: f"{u} - {URG_LABELS[u]}",
    )
    countries = st.multiselect("Country", top_countries(ds_id), default=[], format_func=country_name)
    places_sel = st.multiselect("Place", top_places(ds_id), default=[])
    mapped_only = st.checkbox("Only tweets with a mapped location", value=False)
    query = st.text_input("Search tweet text", value="")

    st.divider()
    include_region = st.checkbox("Include province/state-level places on map", value=False)
    color_by = st.radio("Color map by", ["Category", "Urgency"], horizontal=True)

filters = {
    "unique_only": unique_only, "relevance": relevance, "hazards": hazards, "categories": categories,
    "urgencies": urgencies, "countries": countries, "places": places_sel, "mapped_only": mapped_only,
    "query": query,
}

records = load_records(ds_id)
view = apply_filters(records, filters)

# ------------------------------------------------------------------ header + KPIs
st.header(datasets[ds_id])

uniq = records[records["canon"]]
rel_n = int((uniq["rel"] == 1).sum())
mapped_n = int(view["places"].map(lambda ps: any(p.get("prec") != "region" for p in ps)).sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tweets", f"{len(records):,}")
c2.metric("Relevant", f"{round(100 * rel_n / max(1, len(uniq)))}%")
c3.metric("In view", f"{len(view):,}")
c4.metric("Mapped", f"{mapped_n:,}")

tab_map, tab_tweets, tab_summary = st.tabs(["Map", "Tweets", "Situation summary"])

with tab_map:
    agg = aggregate_places(view, include_region)
    if agg.empty:
        st.info("No mapped tweets match the current filters.")
    else:
        color_col = "dom_cat" if color_by == "Category" else "max_urg"
        # fit to the densest 90% by tweet count (not by unique place) so a few far-flung
        # single-mention places don't drag the view away from the dense cluster
        w_lat = agg["lat"].repeat(agg["n"])
        w_lon = agg["lon"].repeat(agg["n"])
        lat_lo, lat_hi = w_lat.quantile(0.05), w_lat.quantile(0.95)
        lon_lo, lon_hi = w_lon.quantile(0.05), w_lon.quantile(0.95)
        lat_span, lon_span = max(lat_hi - lat_lo, 0.05), max(lon_hi - lon_lo, 0.05)
        zoom = min(10.0, max(1.0, 8 - (max(lat_span, lon_span / 1.6)) ** 0.5))
        center = {"lat": (lat_lo + lat_hi) / 2, "lon": (lon_lo + lon_hi) / 2}
        fig = px.scatter_map(
            agg, lat="lat", lon="lon", size="n", color=color_col,
            hover_name="name", hover_data={"n": True, "max_urg": True, "lat": False, "lon": False},
            color_discrete_map=CAT_COLORS if color_by == "Category" else None,
            color_continuous_scale="OrRd" if color_by == "Urgency" else None,
            size_max=32, zoom=zoom, center=center, map_style="open-street-map", height=560,
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, width="stretch")

with tab_tweets:
    sort_by = st.selectbox("Sort", ["Most urgent", "Most shared", "Newest"], index=0)
    show = view.copy()
    if sort_by == "Most urgent":
        show = show.sort_values("urg", ascending=False)
    elif sort_by == "Most shared":
        show = show.sort_values("n", ascending=False)
    elif sort_by == "Newest" and show["ts"].notna().any():
        show = show.sort_values("ts", ascending=False)

    table = show[["text", "cat", "urg", "hz", "n", "conf"]].head(500).copy()
    table["urg"] = table["urg"].map(URG_LABELS)
    table["places"] = show["places"].head(500).map(lambda ps: ", ".join(p["name"] for p in ps[:3]))
    table.columns = ["Tweet", "Category", "Urgency", "Hazard", "Shares", "Confidence", "Places"]
    st.caption(f"{len(view):,} tweets in view (showing up to 500)")
    st.dataframe(table, width="stretch", height=520)

    exp1, exp2 = st.columns(2)
    csv_bytes = show[["id", "text", "cat", "urg", "hz", "conf", "n"]].to_csv(index=False).encode()
    exp1.download_button("Download CSV", csv_bytes, file_name=f"{ds_id}_view.csv", mime="text/csv")

    features = []
    for r in show.itertuples():
        for p in r.places:
            features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
                             "properties": {"name": p["name"], "text": r.text, "category": r.cat, "urgency": r.urg}})
    geojson_bytes = json.dumps({"type": "FeatureCollection", "features": features}).encode()
    exp2.download_button("Download GeoJSON", geojson_bytes, file_name=f"{ds_id}_view.geojson", mime="application/geo+json")

with tab_summary:
    st.caption("Uses one Gemini call on ~150 representative tweets from the current filters.")
    if st.button("Summarize this view", type="primary"):
        summary_view = pd.DataFrame({
            "row_id": view["id"], "text": view["text"], "group_size": view["n"], "urgency": view["urg"],
            "category": view["cat"].replace("", "OTHER"), "confidence": view["conf"], "hazard": view["hz"],
            "cc": view["cc"], "places": view["places"],
        })
        filters_desc = f"{relevance.lower()} tweets" + (f", hazards: {', '.join(hazards)}" if hazards else "")
        with st.spinner("Summarizing..."):
            res = summarize(summary_view, filters_desc, Gemini(budget=1, allow_reserve=True))
        st.markdown(res["markdown"])
