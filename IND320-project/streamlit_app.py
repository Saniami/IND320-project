import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="IND320 - Energy data", layout="wide")

BASE = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk"

# Cached so the API is only called once per hour, not on every click in the app
@st.cache_data(ttl=3600)
def fetch_all():
    # Download all reservoir statistics from the NVE API (it has no date parameters)
    r = requests.get(f"{BASE}/HentOffentligData", headers={"Accept": "application/json"}, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    # Give the columns English names
    df = df.rename(columns={"dato_Id": "Date", "omrType": "Area_Type", "omrnr": "Area_Number",
        "fyllingsgrad": "Fill_Percentage", "kapasitet_TWh": "Capacity_TWh", "fylling_TWh": "Fill_TWh"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Area_Type", "Area_Number", "Date"]).reset_index(drop=True)

df = fetch_all()

st.title("IND320 - Energy data")
st.write("""
This app uses two data sources:
- **NVE Magasinstatistikk API**: weekly reservoir fill levels, by area type (EL, NO, VASS).
- **ENTSO-E cross-border flows**: collected with Spark, stored in Cassandra and curated into MongoDB.

Use the sidebar to open the pages.
""")

# Latest measurement for the whole of Norway (area type NO)
st.subheader("Latest measurement, whole Norway")
no = df[df["Area_Type"] == "NO"]
if not no.empty:
    latest = no.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Fill percentage", f"{latest['Fill_Percentage']:.1%}")
    c2.metric("Fill volume", f"{latest['Fill_TWh']:.1f} TWh")
    c3.metric("Capacity", f"{latest['Capacity_TWh']:.1f} TWh")
    st.caption(f"Latest data point: {latest['Date'].date()}")