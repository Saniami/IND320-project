import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

BASE = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk"

# Cached so the API is only called once per hour
@st.cache_data(ttl=3600)
def fetch_all():
    # Download all reservoir statistics from the NVE API
    r = requests.get(f"{BASE}/HentOffentligData", headers={"Accept": "application/json"}, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df = df.rename(columns={
        "dato_Id": "Date", "omrType": "Area_Type", "omrnr": "Area_Number",
        "fyllingsgrad": "Fill_Percentage", "kapasitet_TWh": "Capacity_TWh",
        "fylling_TWh": "Fill_TWh",
        "fyllingsgrad_forrige_uke": "Fill_Percentage_Previous_Week",
        "endring_fyllingsgrad": "Fill_Percentage_Change"})
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    return df.sort_values(["Area_Type", "Area_Number", "Date"]).reset_index(drop=True)

@st.cache_data
def load_interval(start, end):
    # The API has no date parameters, so the full data set is downloaded once (cached)
    # and the interval chosen with the slider is cut out here.
    df = fetch_all()
    return df[(df["Month"] >= start) & (df["Month"] <= end)]

st.title("Reservoir plot")

all_df = fetch_all()
months = sorted(all_df["Month"].unique())

# Radio buttons: type of area
area_type = st.radio("Area type", ["EL", "NO", "VASS"], horizontal=True)

# Dropdown: one column or all columns
cols = ["Fill_Percentage", "Capacity_TWh", "Fill_TWh",
        "Fill_Percentage_Previous_Week", "Fill_Percentage_Change"]
choice = st.selectbox("Column", ["All columns"] + cols)

# Slider: interval of months (default = first month only)
start, end = st.select_slider("Interval (months)", options=months, value=(months[0], months[0]))

sub = load_interval(start, end)
sub = sub[sub["Area_Type"] == area_type]
if sub.empty:
    st.info("No data for this selection.")
    st.stop()

shown = cols if choice == "All columns" else [choice]

# One panel per column, one line per area number
fig, axes = plt.subplots(len(shown), 1, figsize=(10, 3 * len(shown)), sharex=True, squeeze=False)
for ax, col in zip(axes[:, 0], shown):
    for nr, g in sub.groupby("Area_Number"):
        ax.plot(g["Date"], g[col], marker="o", markersize=3, label=f"{area_type} {nr}")
    ax.set_ylabel(col)
axes[0, 0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
axes[-1, 0].set_xlabel("Date")
fig.suptitle(f"{choice}: {area_type} areas, {start} to {end}")
fig.tight_layout()
st.pyplot(fig)