"""Exchange rates from Norges Bank (EUR and USD against NOK)."""
from datetime import date
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils import page_header, load_or_stop, download_exchange_rates, rates_weekly, COLORWAY

page_header("Currency", "Explore",
            "Daily exchange rates from Norges Bank: how many NOK one euro or one dollar costs. "
            "The rates can be used as exogenous variables on the Forecast page.")

rates = load_or_stop(download_exchange_rates, ("EUR", "USD"), "2015-01-01", what="the exchange rates from Norges Bank")
if rates.empty:
    st.warning("Norges Bank returned no exchange rates.")
    st.stop()

# ---------------- Key numbers
last = rates.dropna().iloc[-1]
year_ago = rates[rates.index <= rates.index[-1] - pd.DateOffset(years=1)].dropna().iloc[-1]
c1, c2, c3 = st.columns(3)
c1.metric("EUR", f"{last['EUR']:.2f} NOK", f"{last['EUR'] - year_ago['EUR']:+.2f} last 12 months", delta_color="inverse")
c2.metric("USD", f"{last['USD']:.2f} NOK", f"{last['USD'] - year_ago['USD']:+.2f} last 12 months", delta_color="inverse")
c3.metric("Latest rate", str(rates.index[-1].date()))

# ---------------- Plot
c1, c2 = st.columns([2, 1])
start, end = c1.slider("Period", min_value=rates.index[0].date(), max_value=rates.index[-1].date(),
                       value=(date(2020, 1, 1), rates.index[-1].date()), format="YYYY-MM")
resolution = c2.radio("Resolution", ["Daily (business days)", "Weekly mean"], horizontal=True)

r = rates_weekly(rates) if resolution == "Weekly mean" else rates
r = r[(r.index.date >= start) & (r.index.date <= end)]
fig = go.Figure()
for i, cur in enumerate(r.columns):
    fig.add_trace(go.Scatter(x=r.index, y=r[cur], name=cur, line=dict(color=COLORWAY[i], width=1.5)))
fig.update_layout(height=450, yaxis_title="NOK per unit", hovermode="x unified",
                  title=f"EUR and USD against NOK ({resolution.lower()})")
st.plotly_chart(fig, width="stretch")
st.caption("A higher value means a weaker krone. Norges Bank has no rates on weekends and public holidays, so for "
           "the weekly mean these days are filled with the last known rate.")