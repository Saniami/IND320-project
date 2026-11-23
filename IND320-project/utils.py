"""Shared code for the IND320 Streamlit app (part 4).

All pages import from here, so the data loading, caching, error handling, theme and analysis
functions are only written once. The functions are mostly the same as in the Jupyter Notebooks,
but the plots are made with Plotly instead of matplotlib.
"""
import json, math, time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.fft import dct, idct
from scipy.signal import spectrogram
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.seasonal import STL



DATA_DIR = Path(__file__).parent / "data"

# =====================================================================================
# Theme
# =====================================================================================
# The same colours as in .streamlit/config.toml
PRIMARY = "#1f6f8b"
TEXT = "#1d2b36"
COLORWAY = ["#1f6f8b", "#e07a5f", "#3d9970", "#8e6bbf", "#f2b134", "#5c7c8a"]
OUTLIER = "#d62839"                       # red, used for outliers, anomalies and the selected area
FORECAST = "#8e6bbf"                      # purple, the forecast group colour

# Each group of pages has its own colour, so it is easy to see which group is active
GROUP_COLORS = {"Home": "#1f6f8b", "Explore": "#3d9970", "Anomalies": "#e07a5f", "Forecast": "#8e6bbf"}

# Plotly template so all plots have the same look as the app
pio.templates["ind320"] = go.layout.Template(layout=dict(
    colorway=COLORWAY,
    font=dict(family="Source Sans Pro, sans-serif", size=13, color=TEXT),
    paper_bgcolor="white", plot_bgcolor="white",
    xaxis=dict(gridcolor="#e6edf1", linecolor="#b8c7d0", zeroline=False),
    yaxis=dict(gridcolor="#e6edf1", linecolor="#b8c7d0", zeroline=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=60, r=20, t=60, b=40),
    hoverlabel=dict(font_size=12),
))
pio.templates.default = "plotly_white+ind320"


def page_header(title, group, intro=None):
    """Title with a coloured band for the active group, and the same colour in the sidebar."""
    color = GROUP_COLORS[group]
    st.markdown(
        f"<div style='border-left: 8px solid {color}; background: {color}14; padding: 0.6rem 1rem;"
        f" border-radius: 6px; margin-bottom: 0.8rem'>"
        f"<span style='color:{color}; font-weight:700; letter-spacing:0.08em; font-size:0.8rem'>"
        f"{group.upper()}</span><h2 style='margin:0; padding:0'>{title}</h2></div>",
        unsafe_allow_html=True)
    st.sidebar.markdown(
        f"<div style='background:{color}; color:white; padding:0.4rem 0.8rem; border-radius:6px;"
        f" font-weight:600'>Active group: {group}</div>", unsafe_allow_html=True)
    if intro:
        st.write(intro)


def show_selection_in_sidebar():
    """Short summary of the shared selections, so the user always sees what is chosen."""
    init_state()
    ss = st.session_state
    st.sidebar.markdown("**Shared selections**")
    st.sidebar.caption(f"Area: **{area_label()}**  \nWeek: **{ss['map_date']}**  \n"
                       f"Weather property: **{ss['weather_prop']}**")
    st.sidebar.caption("Change them on the Home page (map), or with the selectors on each page.")


# =====================================================================================
# Shared selections (session state)
# =====================================================================================
# Area numbers that have reservoir data for each area type (VASS 4 has no data)
AREA_NUMBERS = {"EL": [1, 2, 3, 4, 5], "NO": [0], "VASS": [1, 2, 3]}
TYPE_TEXT = {"EL": "EL: the five electricity price areas (NO 1 - NO 5).",
             "NO": "NO: the whole of Norway.",
             "VASS": "VASS: watercourse areas. The data only has areas 1-3."}


def init_state():
    """Default values the first time the app opens. Widget keys are removed when the user
    changes page, so the values are kept in our own keys."""
    ss = st.session_state
    ss.setdefault("area_type", "EL")
    ss.setdefault("area_number", 3)
    ss.setdefault("map_date", last_sunday(date.today() - timedelta(days=7)))
    ss.setdefault("weather_prop", "Temperature (°C)")


def area_selector(show_number=True):
    """Area selector at the top of a page. The choice is saved in st.session_state, so it stays the
    same on all pages and is also changed by clicking on the map."""
    init_state()
    ss = st.session_state

    def _save_type():
        ss["area_type"] = ss["_type_widget"]
        if ss["area_number"] not in AREA_NUMBERS[ss["area_type"]]:
            ss["area_number"] = AREA_NUMBERS[ss["area_type"]][0]

    def _save_number():
        ss["area_number"] = ss["_number_widget"]

    types = list(AREA_NUMBERS)
    c1, c2 = st.columns([2, 1]) if show_number else (st, None)
    c1.radio("Area type", types, index=types.index(ss["area_type"]), horizontal=True,
             key="_type_widget", on_change=_save_type)
    if show_number:
        nums = AREA_NUMBERS[ss["area_type"]]
        c2.selectbox("Area number", nums, index=nums.index(ss["area_number"]),
                     key="_number_widget", on_change=_save_number)
    st.caption(TYPE_TEXT[ss["area_type"]])
    return ss["area_type"], ss["area_number"]


def area_label():
    """Short name of the chosen area, e.g. "EL 3"."""
    return f"{st.session_state['area_type']} {st.session_state['area_number']}"


def weather_property_selector(label="Weather property", key="wp", options=None):
    """Select a weather property. The choice is shared between pages (session state)."""
    init_state()
    ss = st.session_state
    options = options or list(WEATHER_PROPS)
    if ss["weather_prop"] not in options:
        ss["weather_prop"] = options[0]

    def _save():
        ss["weather_prop"] = ss[f"_{key}_widget"]

    return st.selectbox(label, options, index=options.index(ss["weather_prop"]),
                        key=f"_{key}_widget", on_change=_save)


def shared_date(label, key, min_value=None, max_value=None, help=None):
    """Date input that reads and writes the shared map date (the week chosen on the map)."""
    init_state()
    ss = st.session_state

    def _save():
        ss["map_date"] = last_sunday(ss[f"_{key}_widget"])

    value = ss["map_date"]
    if min_value is not None:
        value = max(value, min_value)                      # keep the value inside the allowed range
    if max_value is not None:
        value = min(value, max_value)
    st.date_input(label, value=value, min_value=min_value, max_value=max_value,
                  key=f"_{key}_widget", on_change=_save, help=help)
    return ss["map_date"]


def last_sunday(d):
    """The Sunday at the end of the week of d (Magasinstatistikk labels each week with its Sunday)."""
    d = pd.Timestamp(d).date()
    return d + timedelta(days=(6 - d.weekday()))


# =====================================================================================
# Error handling
# =====================================================================================
def load_or_stop(fn, *args, what="the data", **kwargs):
    """Run a loading function. If the API or the database fails, show a useful message
    instead of a crash, and stop the page."""
    try:
        return fn(*args, **kwargs)
    except requests.exceptions.Timeout:
        st.error(f"Could not load {what}: the server took too long to answer. Please try again later.")
    except requests.exceptions.ConnectionError:
        st.error(f"Could not load {what}: no connection to the server. Check the internet connection.")
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        if code == 429:
            st.error(f"Could not load {what}: the API limit is reached (HTTP 429). Please try again later.")
        else:
            st.error(f"Could not load {what}: the server answered with HTTP {code}.")
    except (KeyError, ValueError) as e:
        st.error(f"Could not load {what}: the data had an unexpected format ({e}).")
    except Exception as e:                                   # database errors and anything else
        st.error(f"Could not load {what}: {type(e).__name__}. {str(e)[:200]}")
    st.stop()


def report_missing(df_or_s, name):
    """Tell the user if there are missing values, and fill short gaps."""
    n = int(pd.DataFrame(df_or_s).isna().sum().sum())
    if n:
        st.info(f"{name}: {n} missing values were filled by interpolation.")
        return df_or_s.interpolate(limit_direction="both")
    return df_or_s


# =====================================================================================
# Reservoir data (NVE Magasinstatistikk)
# =====================================================================================
NVE_BASE = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk"


@st.cache_data(ttl=24 * 3600, show_spinner="Downloading reservoir data from NVE ...")
def load_reservoir():
    """All reservoir statistics (the API has no date parameters). Fill is given in percent (0-100)."""
    r = requests.get(f"{NVE_BASE}/HentOffentligData", headers={"Accept": "application/json"}, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json()).rename(columns={
        "dato_Id": "Date", "omrType": "Area_Type", "omrnr": "Area_Number",
        "fyllingsgrad": "Fill_Percentage", "kapasitet_TWh": "Capacity_TWh", "fylling_TWh": "Fill_TWh",
        "fyllingsgrad_forrige_uke": "Fill_Percentage_Previous_Week",
        "endring_fyllingsgrad": "Fill_Percentage_Change"})
    df["Date"] = pd.to_datetime(df["Date"])
    for c in ["Fill_Percentage", "Fill_Percentage_Previous_Week", "Fill_Percentage_Change"]:
        if c in df:
            df[c] = df[c] * 100                              # share (0-1) -> percent
    return df.sort_values(["Area_Type", "Area_Number", "Date"]).reset_index(drop=True)


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def load_area_names():
    """Area names from NVE (e.g. "NO 1"), used in the data table."""
    r = requests.get(f"{NVE_BASE}/HentOmråder", headers={"Accept": "application/json"}, timeout=60)
    r.raise_for_status()
    data = r.json()
    rows = []
    for block in (data if isinstance(data, list) else [data]):
        for group in block.values():                         # land, elspot, vassdrag
            rows += group
    return pd.DataFrame(rows)


def fill_series(area_type, area_number, column="Fill_Percentage"):
    """Weekly series for one area, with regular weekly steps (Sunday)."""
    df = load_or_stop(load_reservoir, what="the reservoir data from NVE")
    d = df[(df.Area_Type == area_type) & (df.Area_Number == area_number)]
    if d.empty:
        st.warning(f"No reservoir data for {area_type} {area_number}.")
        st.stop()
    return d.set_index("Date")[column].asfreq("W-SUN").interpolate(limit=4)


# =====================================================================================
# Time zones and weekly values (same solution as in the notebook)
# =====================================================================================
def to_weekly(s, how="mean", source_tz=None):
    """Weekly values (Monday-Sunday, labelled with the Sunday, like Magasinstatistikk).
    source_tz: the time zone of data without time zone (e.g. "UTC"), changed to Norwegian time first."""
    s = s.copy()
    if source_tz is not None:
        s.index = s.index.tz_localize(source_tz).tz_convert("Europe/Oslo").tz_localize(None)
    weekly = s.resample("W-SUN").agg(how)
    weekly.index = weekly.index.normalize()
    return weekly


# =====================================================================================
# Weather data (open-meteo, ERA5)
# =====================================================================================
# Name in the app -> (hourly variable, daily variable, how to aggregate to a week)
WEATHER_PROPS = {
    "Temperature (°C)":   ("temperature_2m", "temperature_2m_mean", "mean"),
    "Precipitation (mm)": ("precipitation", "precipitation_sum", "sum"),
    "Rain (mm)":          ("rain", "rain_sum", "sum"),
    "Snowfall (cm)":      ("snowfall", "snowfall_sum", "sum"),
    "Wind speed (m/s)":   ("wind_speed_10m", "wind_speed_10m_max", "mean"),
    "Wind gusts (m/s)":   ("wind_gusts_10m", "wind_gusts_10m_max", "mean"),
}
HOURLY = [v[0] for v in WEATHER_PROPS.values()]
DAILY = [v[1] for v in WEATHER_PROPS.values()]
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


@st.cache_data
def load_locations():
    """The 1000 random points per area, made in the part 3 notebook."""
    return pd.read_csv(DATA_DIR / "locations.csv")


def _grid_cells(area_type, area_number, max_cells):
    """Snap the random points to the ERA5 grid (0.25°), and keep at most max_cells cells
    (cells with many points are picked more often)."""
    loc = load_locations()
    loc = loc[(loc.Area_Type == area_type) & (loc.Area_Number == area_number)]
    cells = (loc.assign(lat=(loc.lat * 4).round() / 4, lon=(loc.lon * 4).round() / 4)
                .groupby(["lat", "lon"]).size().rename("weight").reset_index())
    if len(cells) > max_cells:
        cells = cells.sample(max_cells, weights="weight", random_state=0)
    return cells


def _get_open_meteo(params):
    """One request to open-meteo, with one retry if the API is rate-limited."""
    for attempt in range(2):
        r = requests.get(ARCHIVE_URL, params=params, timeout=180)
        if r.status_code != 429:
            break
        time.sleep(10)
    r.raise_for_status()
    js = r.json()
    return js if isinstance(js, list) else [js]


def _weighted_mean(frames_by_cell, weights, columns):
    """Weighted mean over the grid cells, one row per time step."""
    frames = []
    for f, w in zip(frames_by_cell, weights):
        f = pd.DataFrame(f); f["weight"] = w
        frames.append(f)
    d = pd.concat(frames); d["time"] = pd.to_datetime(d["time"])
    return (d[columns].mul(d["weight"], axis=0).groupby(d["time"]).sum()
              .div(d.groupby("time")["weight"].sum(), axis=0))


@st.cache_data(show_spinner=False, persist="disk")
def download_weather(area_type, area_number, start, end, max_cells=20):
    """Hourly weather (UTC) for one area, as in part 3 (used for SATV and LOF)."""
    cells = _grid_cells(area_type, area_number, max_cells)
    js = _get_open_meteo({"latitude": ",".join(map(str, cells.lat)), "longitude": ",".join(map(str, cells.lon)),
                          "start_date": start, "end_date": end, "hourly": ",".join(HOURLY),
                          "models": "era5", "timezone": "UTC", "wind_speed_unit": "ms"})
    return _weighted_mean([j["hourly"] for j in js], cells.weight, HOURLY)


def get_weather(year):
    """Hourly weather for the chosen area and year, with a spinner and error handling."""
    init_state()
    ss = st.session_state
    end = min(date(year, 12, 31), date.today() - timedelta(days=6))      # ERA5 is a few days behind
    with st.spinner(f"Downloading hourly weather for {area_label()} {year} ..."):
        w = load_or_stop(download_weather, ss["area_type"], ss["area_number"], f"{year}-01-01", str(end),
                         what="the weather data from open-meteo")
    w = report_missing(w, "Weather data")
    st.caption(f"Weather data: ERA5 from open-meteo, {area_label()}, {year} ({len(w)} hours, "
               f"weighted mean over up to 20 grid cells). Times in UTC.")
    return w


@st.cache_data(show_spinner=False, persist="disk")
def download_weather_daily_year(area_type, area_number, year, max_cells=6):
    """Daily weather for one area and one year, in Norwegian time. One cached download per year,
    so moving a time slider only downloads the years that are not already in the cache."""
    end = min(date(year, 12, 31), date.today() - timedelta(days=6))
    cells = _grid_cells(area_type, area_number, max_cells)
    js = _get_open_meteo({"latitude": ",".join(map(str, cells.lat)), "longitude": ",".join(map(str, cells.lon)),
                          "start_date": f"{year}-01-01", "end_date": str(end), "daily": ",".join(DAILY),
                          "models": "era5", "timezone": "Europe/Oslo", "wind_speed_unit": "ms"})
    return _weighted_mean([j["daily"] for j in js], cells.weight, DAILY)


def weather_daily(area_type, area_number, start, end):
    """Daily weather between two dates, downloaded year by year with a progress bar."""
    years = list(range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1))
    years = [y for y in years if date(y, 1, 1) <= date.today() - timedelta(days=6)]
    if not years:
        st.warning("There is no weather data for this period yet.")
        st.stop()
    bar = st.progress(0.0, text=f"Weather for {area_type} {area_number} ...") if len(years) > 1 else None
    parts = []
    for i, y in enumerate(years):
        parts.append(load_or_stop(download_weather_daily_year, area_type, area_number, y,
                                  what="the weather data from open-meteo"))
        if bar:
            bar.progress((i + 1) / len(years), text=f"Weather for {area_type} {area_number}: {y} done")
    if bar:
        bar.empty()
    d = pd.concat(parts)
    return d[(d.index >= pd.Timestamp(start)) & (d.index <= pd.Timestamp(end))]


def weather_weekly(area_type, area_number, start, end, props=None):
    """Weekly weather (Norwegian weeks). Columns get the app names, e.g. "Temperature (°C)"."""
    props = props or list(WEATHER_PROPS)
    d = weather_daily(area_type, area_number, start, end)
    d = d.interpolate(limit_direction="both")
    return pd.concat({p: to_weekly(d[WEATHER_PROPS[p][1]], WEATHER_PROPS[p][2]) for p in props}, axis=1)


# =====================================================================================
# Currency (Norges Bank)
# =====================================================================================
NB_URL = "https://data.norges-bank.no/api/data/EXR/B.{cur}.NOK.SP"


@st.cache_data(ttl=24 * 3600, show_spinner="Downloading exchange rates from Norges Bank ...")
def download_exchange_rates(currencies=("EUR", "USD"), start="2015-01-01", end=None):
    """Daily exchange rates (NOK per unit) from Norges Bank, one column per currency."""
    import io
    end = end or date.today().isoformat()
    r = requests.get(NB_URL.format(cur="+".join(currencies)),
                     params={"format": "csv", "startPeriod": start, "endPeriod": end, "locale": "en"}, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep=None, engine="python")[["BASE_CUR", "TIME_PERIOD", "OBS_VALUE"]]
    df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    rates = df.pivot(index="TIME_PERIOD", columns="BASE_CUR", values="OBS_VALUE")
    rates.index.name = "date"; rates.columns.name = None
    return rates


def rates_weekly(rates):
    """Fill weekends and holidays with the last rate, then the weekly mean."""
    return to_weekly(rates.asfreq("D").ffill(), "mean")


# =====================================================================================
# Cross-border flows (ENTSO-E, from MongoDB)
# =====================================================================================
@st.cache_resource(show_spinner=False)
def get_mongo():
    """One MongoDB connection for the whole app. The connection string is in Streamlit secrets."""
    from pymongo import MongoClient
    client = MongoClient(st.secrets["mongo"]["uri"], serverSelectionTimeoutMS=8000)
    client.admin.command("ping")                            # fails here if the database is not reachable
    return client


@st.cache_data(ttl=3600, show_spinner="Reading cross-border flows from MongoDB ...")
def load_flows():
    """All hourly flows. The time is UTC without time zone (saved like that in part 2)."""
    docs = get_mongo()["ind320"]["crossborder_flows"].find({}, {"_id": 0})
    df = pd.DataFrame(list(docs))
    if df.empty:
        raise ValueError("the collection crossborder_flows is empty")
    df["ts"] = pd.to_datetime(df["ts"])
    df["pair"] = np.where(df["from_area"] < df["to_area"],                 # one label per connection
                          df["from_area"] + " – " + df["to_area"], df["to_area"] + " – " + df["from_area"])
    return df


def flows_weekly(area_type, area_number):
    """Weekly mean net export (MW) from the chosen price area, or all of Norway. Positive = export."""
    df = load_flows()
    prefix = f"NO_{area_number}" if area_type == "EL" else "NO"
    if not (df.from_area.str.startswith(prefix) | df.to_area.str.startswith(prefix)).any():
        prefix = "NO"                                        # only NO 1 and NO 2 were downloaded in part 2
    out_ = df[df.from_area.str.startswith(prefix) & ~df.to_area.str.startswith("NO")]
    in_ = df[df.to_area.str.startswith(prefix) & ~df.from_area.str.startswith("NO")]
    net = out_.groupby("ts")["mw"].sum().sub(in_.groupby("ts")["mw"].sum(), fill_value=0)
    return to_weekly(net, "mean", source_tz="UTC").rename("Net export (MW)")


# =====================================================================================
# Map
# =====================================================================================
@st.cache_data
def load_geojson():
    """The areas from the shapefiles in part 3, converted to GeoJSON (see the notebook)."""
    with open(DATA_DIR / "areas.geojson", encoding="utf-8") as f:
        return json.load(f)


def map_values(area_type, colour_by, week, period_weeks=4):
    """One value per area of the chosen type, for the chosen week.
    Reservoir values are taken for the week itself, weather is aggregated over the last period_weeks."""
    week = pd.Timestamp(week)
    values = {}
    if colour_by in ("Reservoir fill (%)", "Change in fill last week (pp)"):
        col = "Fill_Percentage" if colour_by == "Reservoir fill (%)" else "Fill_Percentage_Change"
        df = load_or_stop(load_reservoir, what="the reservoir data from NVE")
        d = df[(df.Area_Type == area_type) & (df.Date == week)]
        values = dict(zip(d.Area_Number, d[col]))
    else:
        var, daily_var, how = WEATHER_PROPS[colour_by]
        start = week - pd.Timedelta(weeks=period_weeks) + pd.Timedelta(days=1)
        for n in AREA_NUMBERS[area_type]:
            d = weather_daily(area_type, n, start, week)[daily_var]
            # mean for temperature and wind, mean daily sum for precipitation, rain and snow
            values[n] = float(d.mean()) if len(d) else np.nan
    return values


def make_map(area_type, values, selected, colour_by):
    """Folium map with the areas of one type: outlines, transparent colour by value,
    and a thick red outline around the selected area."""
    import folium
    import branca.colormap as cm

    gj = load_geojson()
    feats = [f for f in gj["features"] if f["properties"]["Area_Type"] == area_type]
    vals = [v for v in values.values() if v is not None and not np.isnan(v)]
    vmin, vmax = (min(vals), max(vals)) if vals else (0, 1)
    if vmin == vmax:
        vmin, vmax = vmin - 1, vmax + 1
    cmap_colors = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"] if "Temperature" in colour_by \
        else ["#f7fbff", "#9ecae1", "#4292c6", "#1f6f8b", "#08306b"]
    cmap = cm.LinearColormap(cmap_colors, vmin=vmin, vmax=vmax, caption=colour_by)

    for f in feats:                                       # add the value and a label to each area
        n = f["properties"]["Area_Number"]
        v = values.get(n)
        f["properties"]["value"] = None if v is None or np.isnan(v) else round(float(v), 1)
        f["properties"]["label"] = "no data" if f["properties"]["value"] is None else f"{v:.1f}"

    def style(feature):
        p = feature["properties"]
        is_sel = p["Area_Number"] == selected
        return {"fillColor": "#cccccc" if p["value"] is None else cmap(p["value"]),
                "fillOpacity": 0.55, "color": OUTLIER if is_sel else "#34495e",
                "weight": 4 if is_sel else 1}

    m = folium.Map(location=[65, 15], zoom_start=4, tiles="OpenStreetMap", control_scale=True)
    folium.GeoJson({"type": "FeatureCollection", "features": feats}, name="areas", style_function=style,
                   highlight_function=lambda f: {"weight": 3, "color": PRIMARY},
                   tooltip=folium.GeoJsonTooltip(fields=["name", "label"], aliases=["Area", colour_by])).add_to(m)
    cmap.add_to(m)
    return m


# =====================================================================================
# Analysis functions (from the notebooks), now with Plotly
# =====================================================================================
def spc_outliers(s, freq_cutoff=1/30, n_std=3.0, title="Temperature"):
    """SATV: remove the season with a DCT high-pass filter, then SPC limits from the median and MAD."""
    s = s.dropna()
    x = s.to_numpy(dtype=float); n = len(x)
    c = dct(x, norm="ortho"); k = int(math.ceil(2 * n * freq_cutoff / 24))
    c_low = c.copy(); c_low[k:] = 0
    seasonal = idct(c_low, norm="ortho"); satv = x - seasonal
    med = np.median(satv); sigma = 1.4826 * np.median(np.abs(satv - med))
    upper, lower = seasonal + med + n_std * sigma, seasonal + med - n_std * sigma
    is_out = (x > upper) | (x < lower)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=s.index, y=x, mode="lines", name=title, line=dict(width=1, color=PRIMARY)))
    fig.add_trace(go.Scattergl(x=s.index, y=upper, mode="lines", name=f"SPC limits (±{n_std}σ)",
                               line=dict(width=1, dash="dash", color="#555")))
    fig.add_trace(go.Scattergl(x=s.index, y=lower, mode="lines", showlegend=False,
                               line=dict(width=1, dash="dash", color="#555")))
    fig.add_trace(go.Scattergl(x=s.index[is_out], y=x[is_out], mode="markers", name="Outliers",
                               marker=dict(size=5, color=OUTLIER)))
    fig.update_layout(height=430, title=f"{title} with SPC limits", xaxis_title="Time", yaxis_title=title)

    outliers = pd.DataFrame({"value": x[is_out], "upper": upper[is_out], "lower": lower[is_out]},
                            index=s.index[is_out])
    summary = {"n_points": n, "n_outliers": int(is_out.sum()), "share_outliers_%": float(round(100 * is_out.mean(), 2)),
               "n_above": int((x > upper).sum()), "n_below": int((x < lower).sum())}
    return fig, summary, outliers


def lof_anomalies(s, contamination=0.01, n_neighbors=50, title="Temperature"):
    """Local Outlier Factor on standardised time and value."""
    s = s.dropna(); t = (s.index - s.index[0]).total_seconds().to_numpy()
    X = np.column_stack([(t - t.mean()) / t.std(), (s - s.mean()) / (s.std() or 1)])
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    is_out = lof.fit_predict(X) == -1; score = -lof.negative_outlier_factor_

    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=s.index, y=s.values, mode="lines", name=title, line=dict(width=1, color=PRIMARY)))
    fig.add_trace(go.Scattergl(x=s.index[is_out], y=s.values[is_out], mode="markers", name="LOF anomalies",
                               marker=dict(size=6, color=OUTLIER)))
    fig.update_layout(height=430, title=f"{title} with LOF anomalies ({contamination:.1%})",
                      xaxis_title="Time", yaxis_title=title)

    anomalies = pd.DataFrame({"value": s[is_out], "lof_score": score[is_out]}).sort_values("lof_score", ascending=False)
    summary = {"n_points": len(s), "n_anomalies": int(is_out.sum()), "share_anomalies_%": float(round(100 * is_out.mean(), 2)),
               "mean_value_anomalies": float(round(s[is_out].mean(), 2)), "mean_value_all": float(round(s.mean(), 2))}
    return fig, summary, anomalies


def stl_plot(s, period=52, seasonal=17, trend=105, robust=True, title="Fill percentage"):
    """STL decomposition in four panels."""
    r = STL(s.interpolate(limit_direction="both"), period=period, seasonal=seasonal, trend=trend, robust=robust).fit()
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=["Observed", "Trend", "Seasonal", "Residual"])
    fig.add_trace(go.Scatter(x=s.index, y=r.observed, line=dict(color=PRIMARY)), 1, 1)
    fig.add_trace(go.Scatter(x=s.index, y=r.trend, line=dict(color=COLORWAY[1])), 2, 1)
    fig.add_trace(go.Scatter(x=s.index, y=r.seasonal, line=dict(color=COLORWAY[2])), 3, 1)
    fig.add_trace(go.Scatter(x=s.index, y=r.resid, mode="markers", marker=dict(size=3, color=COLORWAY[3])), 4, 1)
    fig.add_hline(y=0, line_color="#888", line_width=1, row=4, col=1)
    fig.update_layout(height=720, showlegend=False, title=f"STL of {title}")
    return fig


def spectrogram_plot(s, window_length=104, window_overlap=73, title="Fill percentage"):
    """Spectrogram of a weekly series, frequency in cycles per year."""
    f, t, Sxx = spectrogram(s.interpolate(limit_direction="both").to_numpy(dtype=float), fs=52.0, window="hann",
                            nperseg=window_length, noverlap=window_overlap, detrend="constant")
    times = s.index[0] + pd.to_timedelta(t * 365.25, unit="D")
    fig = go.Figure(go.Heatmap(x=times, y=f, z=10 * np.log10(Sxx + 1e-12), colorscale="Viridis",
                               colorbar=dict(title="Power (dB)")))
    fig.update_layout(height=430, title=f"Spectrogram of {title}", xaxis_title="Date",
                      yaxis_title="Frequency (cycles per year)")
    return fig


def sliding_window_corr(x, y, lag=0, window=26):
    """Correlation between x (moved lag weeks forward) and y in a sliding window.
    lag > 0 means that x happens before y, e.g. snow now and fill later."""
    df = pd.concat({"x": x.shift(lag), "y": y}, axis=1).dropna()
    return df["x"].rolling(window, center=True).corr(df["y"]), df


# =====================================================================================
# Forecasting with River (same model as in the notebook)
# =====================================================================================
FALLBACK_DEFAULTS = {"sources": ["weather"], "history_weeks": 260, "horizon": 12,
                     "model": "snarimax", "p": 2, "d": 1, "q": 0, "sp": 0, "sd": 1, "sq": 0}


def load_forecast_defaults():
    """The best setting from the test in the notebook (saved as json), or a fallback."""
    try:
        with open(DATA_DIR / "forecast_defaults.json", encoding="utf-8") as f:
            return {**FALLBACK_DEFAULTS, **json.load(f)}
    except (FileNotFoundError, json.JSONDecodeError):
        return FALLBACK_DEFAULTS


def _season(ts):
    w = ts.isocalendar()[1]
    return {"sin_week": np.sin(2 * np.pi * w / 52.18), "cos_week": np.cos(2 * np.pi * w / 52.18)}


def _future_exog(X, future_index):
    """Exogenous values for the forecast weeks: last value for currency, mean of the same week
    in earlier years (climatology) for weather and flows."""
    out = pd.DataFrame(index=future_index, columns=X.columns, dtype=float)
    week = X.index.isocalendar().week
    for c in X.columns:
        if c in ("EUR", "USD"):
            out[c] = X[c].iloc[-1]
        else:
            clim = X[c].groupby(week).mean()
            out[c] = [clim.get(w, X[c].mean()) for w in future_index.isocalendar().week]
    return out


def forecast_fill(y, exog=None, model="snarimax", horizon=12, p=2, d=1, q=0, sp=0, sd=1, sq=0, m=52,
                  n_lags=4, lr=0.01, quantile=None):
    """Dynamic forecast: the model learns the history one week at a time and then forecasts
    `horizon` weeks, where each forecast is used for the next one.
    quantile: None for a normal forecast, or e.g. 0.05 / 0.95 for the limits of an interval
    (optim.losses.Quantile instead of squared loss)."""
    from river import compose, linear_model, optim, preprocessing, time_series

    y = y.dropna()
    future_index = pd.date_range(y.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="W-SUN")
    X = X_future = None
    if exog is not None and exog.shape[1] > 0:
        X = exog.reindex(y.index)
        X = X.loc[:, X.isna().mean() < 0.2].ffill().bfill()   # drop sources that do not cover the history
        if X.shape[1] > 0:
            X_future = _future_exog(X, future_index)
        else:
            X = None

    def feats(ts, row):
        f = _season(ts)
        if row is not None:
            f.update({k: float(v) for k, v in row.items()})
        return f

    def regressor():
        loss = optim.losses.Quantile(quantile) if quantile else optim.losses.Squared()
        return compose.Pipeline(preprocessing.StandardScaler(),
                                linear_model.LinearRegression(optimizer=optim.SGD(lr), intercept_lr=lr, loss=loss))

    if model == "snarimax":
        mdl = time_series.SNARIMAX(p=p, d=d, q=q, m=m, sp=sp, sd=sd, sq=sq, regressor=regressor())
        for ts, val in y.items():
            mdl.learn_one(val, x=feats(ts, X.loc[ts] if X is not None else None))
        pred = mdl.forecast(horizon=horizon, xs=[feats(ts, X_future.loc[ts] if X is not None else None)
                                                 for ts in future_index])
    else:
        mdl = preprocessing.TargetStandardScaler(regressor=regressor())
        vals = list(y.values)

        def lag_feats(hist, ts, row):
            f = feats(ts, row)
            f.update({f"lag_{k}": hist[-k] for k in range(1, n_lags + 1)})
            return f

        for i in range(n_lags, len(y)):
            ts = y.index[i]
            mdl.learn_one(lag_feats(vals[:i], ts, X.loc[ts] if X is not None else None), y.iloc[i])
        hist, pred = vals.copy(), []
        for ts in future_index:
            yhat = mdl.predict_one(lag_feats(hist, ts, X_future.loc[ts] if X is not None else None))
            pred.append(yhat); hist.append(yhat)

    return pd.Series(np.clip(pred, 0, 100), index=future_index)


def forecast_animation(history, forecast, actual=None, lower=None, upper=None, title=""):
    """Plotly animation: the forecast is drawn one week at a time (press Play)."""
    hist = history.iloc[-104:]
    traces_static = [go.Scatter(x=hist.index, y=hist.values, name="History", line=dict(color=PRIMARY))]
    if actual is not None and len(actual):
        traces_static.append(go.Scatter(x=actual.index, y=actual.values, name="Actual",
                                        line=dict(color="black", dash="dash")))

    def frame_traces(k):
        fc = forecast.iloc[:k]
        out = []
        if lower is not None:
            lo, up = lower.iloc[:k], upper.iloc[:k]
            out.append(go.Scatter(x=list(fc.index) + list(fc.index[::-1]), y=list(up.values) + list(lo.values[::-1]),
                                  fill="toself", fillcolor="rgba(142,107,191,0.2)", line=dict(width=0),
                                  name="90 % interval", hoverinfo="skip"))
        # the forecast starts at the last known week, so the line is connected to the history
        x = [history.index[-1]] + list(fc.index); yv = [history.iloc[-1]] + list(fc.values)
        out.append(go.Scatter(x=x, y=yv, name="Forecast", mode="lines+markers",
                              line=dict(color=FORECAST, width=3), marker=dict(size=6)))
        return out

    n = len(forecast)
    fig = go.Figure(data=traces_static + frame_traces(n),
                    frames=[go.Frame(data=traces_static + frame_traces(k), name=str(k)) for k in range(1, n + 1)])
    all_y = [hist.values, forecast.values] + ([upper.values, lower.values] if lower is not None else []) \
        + ([actual.values] if actual is not None and len(actual) else [])
    ymin, ymax = float(np.nanmin(np.concatenate(all_y))), float(np.nanmax(np.concatenate(all_y)))
    fig.update_layout(
        title=title, height=480, xaxis_title="Week", yaxis_title="Fill percentage (%)",
        xaxis=dict(range=[hist.index[0], forecast.index[-1] + pd.Timedelta(weeks=2)]),
        yaxis=dict(range=[max(0, ymin - 5), min(100, ymax + 5)]),
        updatemenus=[dict(type="buttons", x=0, y=-0.15, xanchor="left", direction="left", buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, dict(frame=dict(duration=250, redraw=True), fromcurrent=False, transition=dict(duration=0))]),
            dict(label="❚❚ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0), mode="immediate")])])],
        sliders=[dict(active=n - 1, x=0.2, y=-0.12, len=0.8, currentvalue=dict(prefix="Weeks ahead: "),
                      steps=[dict(label=str(k), method="animate",
                                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                             for k in range(1, n + 1)])],
        margin=dict(b=110))
    return fig