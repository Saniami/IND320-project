"""Cross-border flows from ENTSO-E, stored in MongoDB in part 2. Now with Plotly instead of matplotlib."""
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from utils import page_header, load_or_stop, load_flows, COLORWAY

page_header("Cross-border flows", "Explore",
            "Physical power flows between Norwegian price areas and neighbouring countries (ENTSO-E).")

df = load_or_stop(load_flows, what="the cross-border flows from MongoDB")
years = sorted(df["ts"].dt.year.unique())
month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
               7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
OUT, IN = COLORWAY[0], COLORWAY[1]                      # out of Norway = blue, into Norway = orange

left, right = st.columns(2)

# ---------------- Left: total flow in and out in one year (pie)
with left:
    st.subheader("Total flow in one year")
    pair = st.radio("Connection", sorted(df["pair"].unique()))
    year = int(st.selectbox("Year", years, index=len(years) - 1))   # local selection, only for this page

    sub = df[(df["pair"] == pair) & (df["ts"].dt.year == year)]
    totals = sub.groupby(["from_area", "to_area"])["mw"].sum()     # hourly MW summed = MWh
    if totals.sum() == 0:
        st.info("No flow in this connection and year.")
    else:
        labels = [f"{f} → {t}" for f, t in totals.index]
        colors = [OUT if f.startswith("NO") else IN for f, _ in totals.index]
        fig = go.Figure(go.Pie(labels=labels, values=totals.values / 1e6, hole=0.45, marker=dict(colors=colors),
                               hovertemplate="%{label}<br>%{value:.2f} TWh<br>%{percent}<extra></extra>"))
        fig.update_layout(title=f"{pair}, {year} (TWh)", height=420)
        st.plotly_chart(fig, width="stretch")

# ---------------- Right: flow over time, one panel per connection
with right:
    st.subheader("Flow over time")
    months = st.pills("Months", list(month_names), selection_mode="multi", default=[1],
                      format_func=lambda m: month_names[m])
    resolution = st.radio("Resolution", ["Hourly", "Daily mean"], horizontal=True)
    if not months:
        st.info("Select at least one month.")
        st.stop()

    m = df[(df["ts"].dt.year == year) & (df["ts"].dt.month.isin(months))]
    if m.empty:
        st.info("No data for the chosen months.")
        st.stop()
    freq = "h" if resolution == "Hourly" else "D"
    panels = sorted(m["pair"].unique())
    fig2 = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=panels)
    shown = set()
    for i, p in enumerate(panels, start=1):
        for (f, t), g in m[m["pair"] == p].groupby(["from_area", "to_area"]):
            s = g.set_index("ts")["mw"].resample(freq).mean()       # months not chosen become gaps
            leaving = f.startswith("NO")
            name = "Out of Norway" if leaving else "Into Norway"
            fig2.add_trace(go.Scatter(x=s.index, y=s.values, name=name, legendgroup=name,
                                      showlegend=name not in shown, line=dict(color=OUT if leaving else IN, width=1)), i, 1)
            shown.add(name)
        fig2.update_yaxes(title_text="MW", row=i, col=1)
    fig2.update_layout(height=220 * len(panels) + 80, title=f"{year}, {resolution.lower()} (times in UTC)")
    st.plotly_chart(fig2, width="stretch")

with st.expander("About the data"):
    st.write("Source: ENTSO-E Transparency Platform, physical cross-border flows, downloaded with the ENTSO-E API "
             "for the last two years. The 15-minute values were averaged to hourly MW, stored in Cassandra with "
             "Spark, and curated into MongoDB, which this page reads. The times are stored in UTC.")