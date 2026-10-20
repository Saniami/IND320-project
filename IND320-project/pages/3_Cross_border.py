import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient

st.title("Cross-border flows")

# Cached so MongoDB is only queried once per hour, not on every click
@st.cache_data(ttl=3600)
def load_flows():
    # The connection string comes from Streamlit secrets (never from GitHub)
    client = MongoClient(st.secrets["mongo"]["uri"])
    docs = client["ind320"]["crossborder_flows"].find({}, {"_id": 0})
    df = pd.DataFrame(list(docs))
    df["ts"] = pd.to_datetime(df["ts"])
    # One label per connection, regardless of direction (e.g. "NL – NO_2")
    df["pair"] = np.where(df["from_area"] < df["to_area"],
                          df["from_area"] + " – " + df["to_area"],
                          df["to_area"] + " – " + df["from_area"])
    return df

df = load_flows()
years = sorted(df["ts"].dt.year.unique())
month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
               7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

left, right = st.columns(2)

# Left: choose connection and year, show a pie chart of flow in and out of Norway
with left:
    st.subheader("Total flow in one year")
    pair = st.radio("Connection", sorted(df["pair"].unique()))
    year = int(st.selectbox("Year", years, index=years.index(2025) if 2025 in years else 0))

    sub = df[(df["pair"] == pair) & (df["ts"].dt.year == year)]
    totals = sub.groupby(["from_area", "to_area"])["mw"].sum()   # hourly MW summed = MWh
    if totals.sum() == 0:
        st.info("No flow in this connection and year.")
    else:
        fig, ax = plt.subplots()
        ax.pie(totals.values, labels=[f"{f} → {t}" for f, t in totals.index], autopct="%1.1f%%")
        ax.set_title(f"{pair}, {year} (MWh)")
        st.pyplot(fig)

# Right: choose months with pills, show one panel per connection with a line for each direction
with right:
    st.subheader("Flow over time")
    months = st.pills("Months", list(month_names), selection_mode="multi",
                      default=[1], format_func=lambda m: month_names[m])
    resolution = st.radio("Resolution", ["Hourly", "Daily mean"], horizontal=True)

    if not months:
        st.info("Select at least one month.")
    else:
        m = df[(df["ts"].dt.year == year) & (df["ts"].dt.month.isin(months))]
        freq = "h" if resolution == "Hourly" else "D"
        panels = sorted(m["pair"].unique())

        # One panel per connection (own y-axis, since the flows differ a lot in size)
        fig2, axes = plt.subplots(len(panels), 1, figsize=(7, 2.3 * len(panels)),
                                  sharex=True, squeeze=False)
        for ax2, p in zip(axes[:, 0], panels):
            for (f, t), g in m[m["pair"] == p].groupby(["from_area", "to_area"]):
                # Resampling leaves empty bins as NaN, so months that are not selected show as gaps
                s = g.set_index("ts")["mw"].resample(freq).mean()
                leaving = f.startswith("NO")          # flow from Norway to the neighbour
                ax2.plot(s.index, s.values,
                         color="tab:blue" if leaving else "tab:orange",
                         label="Out of Norway" if leaving else "Into Norway")
            ax2.set_title(p, fontsize=9, loc="left")
            ax2.set_ylabel("MW")
        axes[0, 0].legend(fontsize=7, loc="upper right")
        axes[-1, 0].set_xlabel("Time")
        fig2.suptitle(f"{year}, {resolution.lower()}", fontsize=10)
        fig2.autofmt_xdate()
        fig2.tight_layout()
        st.pyplot(fig2)
        
# Below the columns: document the source of the data
with st.expander("About the data"):
    st.write(
        "Source: ENTSO-E Transparency Platform, physical cross-border flows between "
        "Norwegian bidding zones and neighbouring areas, downloaded with the ENTSO-E API "
        "for the last two years. The 15-minute values were averaged to hourly MW, "
        "stored in Cassandra with Spark, and curated into MongoDB, which this page reads."
    )