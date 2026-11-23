"""Meteorology and Magasinstatistikk: sliding window correlation between a weather property
and a reservoir variable, with selectable lag and window length."""
from datetime import date, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from utils import (page_header, area_selector, area_label, weather_property_selector, fill_series,
                   weather_weekly, sliding_window_corr, COLORWAY, PRIMARY, OUTLIER)

page_header("Weather vs. reservoir", "Explore",
            "Sliding window correlation between a weather property and the reservoir in the same area. "
            "With a lag, the weather is moved forward in time, so we can see if the weather now is related "
            "to the reservoir some weeks later (for example snow in winter and filling in spring).")
area_type, area_number = area_selector()

# ---------------- Selectors
c1, c2, c3 = st.columns(3)
with c1:
    prop = weather_property_selector()
with c2:
    targets = {"Fill_Percentage_Change": "Weekly change in fill (pp)", "Fill_Percentage": "Fill percentage (%)"}
    target = st.selectbox("Reservoir variable", list(targets), format_func=targets.get,
                          help="The weekly change reacts directly to the weather. The fill level itself is "
                               "mostly a yearly cycle, which gives high correlations for everything with a season.")
with c3:
    this_year = date.today().year
    years = st.slider("Years", 2015, this_year, (this_year - 5, this_year))   # local, only for this page

c1, c2 = st.columns(2)
lag = c1.slider("Lag (weeks, weather before reservoir)", 0, 26, 0)
window = c2.slider("Window length (weeks)", 4, 104, 26, step=2)

# ---------------- Data (weekly, Norwegian weeks)
y = fill_series(area_type, area_number, target)
start = date(years[0], 1, 1)
end = min(date(years[1], 12, 31), date.today() - timedelta(days=6))
with st.spinner("Downloading weather ..."):
    w = weather_weekly(area_type, area_number, start, end, props=[prop])[prop]
y = y[(y.index >= pd.Timestamp(start)) & (y.index <= pd.Timestamp(end))]

corr, both = sliding_window_corr(w, y, lag=lag, window=window)
if len(both) < window + 2:
    st.warning("The period is too short for this window length. Choose more years or a shorter window.")
    st.stop()

# ---------------- Plot: both series (standardised) and the sliding correlation
z = lambda s: (s - s.mean()) / s.std()
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.55, 0.45],
                    subplot_titles=[f"{prop} (moved {lag} weeks forward) and {targets[target]}, standardised",
                                    f"Correlation in a {window}-week window"])
fig.add_trace(go.Scatter(x=both.index, y=z(both["x"]), name=prop, line=dict(color=COLORWAY[1], width=1.3)), 1, 1)
fig.add_trace(go.Scatter(x=both.index, y=z(both["y"]), name=targets[target], line=dict(color=PRIMARY, width=1.3)), 1, 1)
fig.add_trace(go.Scatter(x=corr.index, y=corr.values, name="Correlation", line=dict(color=COLORWAY[3], width=2),
                         fill="tozeroy", fillcolor="rgba(142,107,191,0.15)"), 2, 1)
fig.add_hline(y=0, line_color="#888", line_width=1, row=2, col=1)
fig.update_yaxes(title_text="z-score", row=1, col=1)
fig.update_yaxes(title_text="r", range=[-1, 1], row=2, col=1)
fig.update_layout(height=680, hovermode="x unified", title=f"{area_label()}: {prop} vs. {targets[target]}",
                  margin=dict(t=110), legend=dict(y=1.12))
st.plotly_chart(fig, width="stretch")

# ---------------- Which lag gives the strongest correlation over the whole period?
lags = np.arange(0, 27)
overall = [pd.concat({"x": w.shift(k), "y": y}, axis=1).dropna().corr().iloc[0, 1] for k in lags]
best = int(lags[np.nanargmax(np.abs(overall))])
fig2 = go.Figure(go.Bar(x=lags, y=overall, marker_color=[OUTLIER if k == lag else PRIMARY for k in lags],
                        hovertemplate="lag %{x} weeks<br>r = %{y:.2f}<extra></extra>"))
fig2.update_layout(height=300, title="Correlation over the whole period for each lag (chosen lag in red)",
                   xaxis_title="Lag (weeks)", yaxis_title="r", yaxis_range=[-1, 1])
st.plotly_chart(fig2, width="stretch")

c1, c2, c3 = st.columns(3)
c1.metric("Correlation, chosen lag", f"{overall[lag]:.2f}")
c2.metric("Strongest lag", f"{best} weeks", f"r = {overall[best]:.2f}", delta_color="off")
c3.metric("Weeks in the comparison", len(both))
st.caption("The weather is aggregated to Norwegian weeks (Monday-Sunday) like Magasinstatistikk: mean for "
           "temperature and wind, sum for precipitation, rain and snowfall.")