"""Reservoir data overview: the LineChartColumn table and the line plot from part 1-3,
joined into one page with two tabs. Only the area type is used here, since all areas of
one type are shown together."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from utils import page_header, area_selector, load_or_stop, load_reservoir, load_area_names, COLORWAY

page_header("Reservoir data overview", "Explore",
            "Weekly reservoir statistics from NVE Magasinstatistikk, for all areas of the chosen type.")
area_type, _ = area_selector(show_number=False)

df = load_or_stop(load_reservoir, what="the reservoir data from NVE")
sub = df[df["Area_Type"] == area_type]
if sub.empty:
    st.warning("No data for this area type.")
    st.stop()

tab_table, tab_plot = st.tabs(["Table", "Plot"])

# ---------------- Table with small line charts (LineChartColumn from part 1)
with tab_table:
    # Area names are nice to have, but the table also works without them
    try:
        names = load_area_names()
        lookup = {(a.omrType, a.omrnr): a for a in names.itertuples()}
    except Exception:
        lookup = {}
        st.caption("Area names could not be loaded from NVE, so only the numbers are shown.")
    REGION_EN = {"Hele landet": "Whole country", "Sørøst-Norge": "Southeastern Norway",
                 "Sørvest-Norge": "Southwestern Norway", "Midt-Norge": "Central Norway",
                 "Nord-Norge": "Northern Norway", "Vest-Norge": "Western Norway"}

    rows = []
    for nr, g in sub.groupby("Area_Number"):
        g = g.sort_values("Date")
        info = lookup.get((area_type, int(nr)))
        region_no = info.beskrivelse.split(".")[0] if info is not None else ""
        rows.append({
            "Area": info.navn if info is not None else f"{area_type} {nr}",
            "Region": REGION_EN.get(region_no, region_no),
            "Last 52 weeks (%)": g["Fill_Percentage"].tail(52).round(1).tolist(),
            "Latest fill (%)": g["Fill_Percentage"].iloc[-1],
            "Change last week (pp)": g["Fill_Percentage_Change"].iloc[-1],
            "Capacity (TWh)": g["Capacity_TWh"].iloc[-1],
            "Latest week": str(g["Date"].max().date()),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Last 52 weeks (%)": st.column_config.LineChartColumn("Last 52 weeks (%)", y_min=0, y_max=100),
        "Latest fill (%)": st.column_config.ProgressColumn("Latest fill (%)", min_value=0, max_value=100, format="%.0f"),
        "Change last week (pp)": st.column_config.NumberColumn(format="%+.1f"),
        "Capacity (TWh)": st.column_config.NumberColumn(format="%.1f"),
    })
    st.caption("pp = percentage points. Source: NVE Magasinstatistikk API.")

# ---------------- Interactive line plot (was matplotlib in part 1-3)
with tab_plot:
    cols = {"Fill_Percentage": "Fill (%)", "Fill_TWh": "Fill (TWh)", "Capacity_TWh": "Capacity (TWh)",
            "Fill_Percentage_Change": "Change from last week (pp)"}
    c1, c2 = st.columns([1, 2])
    chosen = c1.multiselect("Columns", list(cols), default=["Fill_Percentage"], format_func=cols.get)
    first, last = sub["Date"].min().date(), sub["Date"].max().date()
    start, end = c2.slider("Period", min_value=first, max_value=last,
                           value=(max(first, last.replace(year=last.year - 3)), last), format="YYYY-MM")
    if not chosen:
        st.info("Select at least one column.")
        st.stop()

    d = sub[(sub["Date"].dt.date >= start) & (sub["Date"].dt.date <= end)]
    fig = make_subplots(rows=len(chosen), cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=[cols[c] for c in chosen])
    for i, col in enumerate(chosen, start=1):
        for j, (nr, g) in enumerate(d.groupby("Area_Number")):
            fig.add_trace(go.Scatter(x=g["Date"], y=g[col], name=f"{area_type} {nr}", legendgroup=str(nr),
                                     showlegend=(i == 1), line=dict(color=COLORWAY[j % len(COLORWAY)])), i, 1)
    fig.update_layout(height=280 * len(chosen) + 80, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")