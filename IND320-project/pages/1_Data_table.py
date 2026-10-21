import streamlit as st
import pandas as pd
import requests

st.set_page_config(layout="wide")

BASE = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk"
HEAD = {"Accept": "application/json"}

# Cached so the API is only called once per hour, not on every click
@st.cache_data(ttl=3600)
def fetch_all():
    # Download all reservoir statistics from the NVE API (it has no date parameters)
    r = requests.get(f"{BASE}/HentOffentligData", headers=HEAD, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df = df.rename(columns={"dato_Id": "Date", "omrType": "Area_Type", "omrnr": "Area_Number",
        "fyllingsgrad": "Fill_Percentage", "kapasitet_TWh": "Capacity_TWh",
        "endring_fyllingsgrad": "Fill_Percentage_Change"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Area_Type", "Area_Number", "Date"]).reset_index(drop=True)

# Area names (e.g. "NO 1" = south-east Norway), so the table says which region a row is
@st.cache_data(ttl=86400)
def fetch_areas():
    r = requests.get(f"{BASE}/HentOmråder", headers=HEAD, timeout=60)
    r.raise_for_status()
    data = r.json()
    rows = []
    for block in (data if isinstance(data, list) else [data]):
        for group in block.values():          # land, elspot, vassdrag
            rows += group
    return pd.DataFrame(rows)

df = fetch_all()
lookup = {(a.omrType, a.omrnr): a for a in fetch_areas().itertuples()}

st.title("Reservoir data table")
st.write(
    "Each row is one area. The fill level is measured weekly by NVE. "
    "The two small charts show the first month of the data series (1995) and the latest "
    "12 weeks, both on the same 0-100 % scale so the rows can be compared."
)

# Radio buttons choose the type of area
area_type = st.radio("Area type", ["EL", "NO", "VASS"], horizontal=True)
st.caption({"EL": "EL: the five electricity price areas (NO 1 - NO 5).",
            "NO": "NO: the whole of Norway.",
            "VASS": "VASS: watercourse areas. The data only has areas 1-3."}[area_type])
sub = df[df["Area_Type"] == area_type]

# One row per area number
rows = []
for nr, g in sub.groupby("Area_Number"):
    g = g.sort_values("Date")
    first_month = g[g["Date"].dt.to_period("M") == g["Date"].min().to_period("M")]
    info = lookup.get((area_type, int(nr)))
    rows.append({
        "Area": info.navn if info else str(nr),
        "Region": info.beskrivelse.split(".")[0] if info else "",
        "First month (%)": (first_month["Fill_Percentage"] * 100).tolist(),
        "Last 12 weeks (%)": (g["Fill_Percentage"].tail(12) * 100).tolist(),
        "Latest fill (%)": g["Fill_Percentage"].iloc[-1] * 100,
        "Change last week (pp)": g["Fill_Percentage_Change"].iloc[-1] * 100,
        "Capacity (TWh)": g["Capacity_TWh"].iloc[-1],
        "Latest week": str(g["Date"].max().date()),
    })

st.dataframe(
    pd.DataFrame(rows),
    column_config={
        "First month (%)": st.column_config.LineChartColumn("First month (%)", y_min=0, y_max=100),
        "Last 12 weeks (%)": st.column_config.LineChartColumn("Last 12 weeks (%)", y_min=0, y_max=100),
        "Latest fill (%)": st.column_config.ProgressColumn(
            "Latest fill (%)", min_value=0, max_value=100, format="%.0f"),
        "Change last week (pp)": st.column_config.NumberColumn(format="%+.1f"),
        "Capacity (TWh)": st.column_config.NumberColumn(format="%.1f"),
    },
    hide_index=True,
    width="stretch",
)

st.caption("pp = percentage points. Source: NVE Magasinstatistikk API.")