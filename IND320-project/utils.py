import math, time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.fft import dct, idct
from scipy.signal import spectrogram
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.seasonal import STL

# Global area selector
# Area numbers that have reservoir data for each area type (VASS 4 has no data)
AREA_NUMBERS = {"EL": [1, 2, 3, 4, 5], "NO": [0], "VASS": [1, 2, 3]}
TYPE_TEXT = {"EL": "EL: the five electricity price areas (NO 1 - NO 5).",
             "NO": "NO: the whole of Norway.",
             "VASS": "VASS: watercourse areas. The data only has areas 1-3."}

def area_selector(show_number=True):
    """Area selector at the top of the page. The choice is saved in st.session_state, so it stays the same when you go to another page.
    Use show_number=False on pages that show all areas of one type, then only the type is shown."""
    ss = st.session_state
    ss.setdefault("area_type", "EL")                # default the first time the app opens
    ss.setdefault("area_number", 3)

    # Widget keys are removed when the user changes page, so the values are copied to our own keys
    def _save_type():
        ss["area_type"] = ss["_type_widget"]
        if ss["area_number"] not in AREA_NUMBERS[ss["area_type"]]:
            ss["area_number"] = AREA_NUMBERS[ss["area_type"]][0]
    def _save_number():
        ss["area_number"] = ss["_number_widget"]

    types = list(AREA_NUMBERS)
    if show_number:
        c1, c2 = st.columns([2, 1])                          # type and number side by side
    else:
        c1, c2 = st, None                                    # only the type
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

# Reservoir data (NVE)
@st.cache_data(ttl=24 * 3600)                                # download at most once a day
def load_reservoir():
    base = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk"
    r = requests.get(f"{base}/HentOffentligData", headers={"Accept": "application/json"}, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json()).rename(columns={"dato_Id": "Date", "omrType": "Area_Type",
                                                "omrnr": "Area_Number", "fyllingsgrad": "Fill_Percentage"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Area_Type", "Area_Number", "Date"])

def fill_series(area_type, area_number):
    """Weekly fill percentage for one area, with regular weekly steps."""
    df = load_reservoir()
    d = df[(df.Area_Type == area_type) & (df.Area_Number == area_number)]
    return d.set_index("Date")["Fill_Percentage"].asfreq("W-SUN", method="ffill")

# Weather data (open-meteo, ERA5)
HOURLY = ["temperature_2m", "precipitation"] #the app only use these two, but tried with more variables but it took really long time.

@st.cache_data
def load_locations():
    """The 1000 random points per area, made in the notebook and saved as locations.csv."""
    path=Path(__file__).parent / "data" / "locations.csv"
    return pd.read_csv(path)

@st.cache_data(show_spinner=False, persist="disk")                    # hidden download: runs once per area and year
def download_weather(area_type, area_number, start, end, batch=20, max_cells=20):
    """Same as in the notebook: use the points to the ERA5 grid, download each cell once, with weighted mean."""
    loc = load_locations()
    loc = loc[(loc.Area_Type == area_type) & (loc.Area_Number == area_number)]
    cells = (loc.assign(lat=(loc.lat * 4).round() / 4, lon=(loc.lon * 4).round() / 4)
                .groupby(["lat", "lon"]).size().rename("weight").reset_index())
    if len(cells) > max_cells:              # keep the app fast and below the API limit         
        cells = cells.sample(max_cells, weights="weight", random_state=0)  # cells with many points are picked more often
    frames = []
    for i in range(0, len(cells), batch):
        part = cells.iloc[i:i + batch]
        params = {"latitude": ",".join(map(str, part.lat)), "longitude": ",".join(map(str, part.lon)),
                  "start_date": start, "end_date": end, "hourly": ",".join(HOURLY),
                  "models": "era5", "timezone": "UTC"}
        for _ in range(2):                                   # retry if rate-limited
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=180)
            if r.status_code != 429:
                break
            time.sleep(10)
        if r.status_code == 429:                             # hourly or daily limit reached at open-meteo
            raise RuntimeError("The open-meteo API limit is reached. Please try again later.")
        r.raise_for_status()
        js = r.json(); js = js if isinstance(js, list) else [js]
        for w, j in zip(part.weight, js):
            f = pd.DataFrame(j["hourly"]); f["weight"] = w
            frames.append(f)
    d = pd.concat(frames); d["time"] = pd.to_datetime(d["time"])
    out = (d[HOURLY].mul(d["weight"], axis=0).groupby(d["time"]).sum()
              .div(d.groupby("time")["weight"].sum(), axis=0))
    return out

def get_weather(year):
    """Weather for the area chosen in the global selector. The user never starts the download self"""
    area_type, area_number = st.session_state["area_type"], st.session_state["area_number"]
    with st.spinner(f"Downloading weather data for {area_label()} {year} ..."):
        try:
            w = download_weather(area_type, area_number, f"{year}-01-01", f"{year}-12-31")
        except RuntimeError as e:
            st.warning(str(e))
            st.stop()
    st.caption(f"Weather data: ERA5 from open-meteo, {area_label()}, {year} "
               f"({len(w)} hours, average over up to 20 random locations).")
    return w

# Analysis functions (from the notebook)
def spc_outliers(s, freq_cutoff=1/30, n_std=3.0, title="Precipitation"):
    x = s.to_numpy(dtype=float); n = len(x)
    c = dct(x, norm="ortho"); k = int(math.ceil(2 * n * freq_cutoff / 24))
    c_low = c.copy(); c_low[k:] = 0
    seasonal = idct(c_low, norm="ortho"); satv = x - seasonal
    med = np.median(satv); sigma = 1.4826 * np.median(np.abs(satv - med))
    upper, lower = seasonal + med + n_std * sigma, seasonal + med - n_std * sigma
    is_out = (x > upper) | (x < lower)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(s.index, x, lw=0.6, color="tab:blue", label=title)
    ax.plot(s.index, upper, color="black", lw=1, ls="--", label=f"SPC limits (±{n_std}σ)")
    ax.plot(s.index, lower, color="black", lw=1, ls="--")
    ax.scatter(s.index[is_out], x[is_out], s=8, color="tab:red", zorder=3, label="Outliers")
    ax.set_xlabel("Time"); ax.set_ylabel(title); ax.legend(loc="upper right"); fig.tight_layout()
    outliers = pd.DataFrame({"value": x[is_out], "upper": upper[is_out], "lower": lower[is_out]}, index=s.index[is_out])
    summary = {"n_points": n, "n_outliers": int(is_out.sum()), "share_outliers_%": float(round(100 * is_out.mean(), 2)),
               "n_above": int((x > upper).sum()), "n_below": int((x < lower).sum())}
    return fig, summary, outliers

def lof_anomalies(s, contamination=0.01, n_neighbors=50, title="Temperature"):
    s = s.dropna(); t = (s.index - s.index[0]).total_seconds().to_numpy()
    X = np.column_stack([(t - t.mean()) / t.std(), (s - s.mean()) / s.std()])
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    is_out = lof.fit_predict(X) == -1; score = -lof.negative_outlier_factor_
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(s.index, s, lw=0.6, color="tab:blue", label=title)
    ax.scatter(s.index[is_out], s[is_out], s=10, color="tab:red", zorder=3, label="LOF anomalies")
    ax.set_xlabel("Time"); ax.set_ylabel(title); ax.legend(loc="upper right"); fig.tight_layout()
    anomalies = pd.DataFrame({"value": s[is_out], "lof_score": score[is_out]}).sort_values("lof_score", ascending=False)
    summary = {"n_points": len(s), "n_anomalies": int(is_out.sum()), "share_anomalies_%": float(round(100 * is_out.mean(), 2)),
               "mean_value_anomalies": float(round(s[is_out].mean(), 2)), "mean_value_all": float(round(s.mean(), 2))}
    return fig, summary, anomalies

def stl_plot(s, period=52, seasonal=17, trend=105, robust=True, title="Fill percentage"):
    r = STL(s, period=period, seasonal=seasonal, trend=trend, robust=robust).fit()
    fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    for ax, (name, y) in zip(axs, [("Observed", r.observed), ("Trend", r.trend),
                                   ("Seasonal", r.seasonal), ("Residual", r.resid)]):
        if name == "Residual":
            ax.scatter(s.index, y, s=3); ax.axhline(0, color="black", lw=0.8)
        else:
            ax.plot(s.index, y)
        ax.set_ylabel(name)
    axs[0].set_title(f"STL of {title}"); axs[-1].set_xlabel("Date"); fig.tight_layout()
    return fig

def spectrogram_plot(s, window_length=104, window_overlap=73, title="Fill percentage"):
    f, t, Sxx = spectrogram(s.to_numpy(dtype=float), fs=52.0, window="hann",
                            nperseg=window_length, noverlap=window_overlap, detrend="constant")
    times = s.index[0] + pd.to_timedelta(t * 365.25, unit="D")
    fig, ax = plt.subplots(figsize=(12, 4))
    m = ax.pcolormesh(times, f, 10 * np.log10(Sxx + 1e-12), shading="auto", cmap="viridis")
    fig.colorbar(m, ax=ax, label="Power (dB)")
    ax.set_ylabel("Frequency (cycles per year)"); ax.set_xlabel("Date")
    ax.set_title(f"Spectrogram of {title}"); fig.tight_layout()
    return fig