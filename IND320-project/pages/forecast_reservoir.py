"""Forecast of the reservoir filling with River (SNARIMAX or a lag model), with selectable parameters,
exogenous variables, training period and horizon. The last training week is the week chosen on the map.
The default setting is the best one from the test in the part 4 notebook (data/forecast_defaults.json)."""
from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
from utils import (page_header, area_selector, area_label, shared_date, fill_series, load_or_stop, weather_weekly,
                   download_exchange_rates, rates_weekly, flows_weekly, forecast_fill, forecast_animation,
                   load_forecast_defaults, WEATHER_PROPS)

page_header("Reservoir forecast", "Forecast",
            "Dynamic forecast of the weekly fill percentage: the model learns the history one week at a time, "
            "and then forecasts week by week, where each forecast is used for the next week.")
area_type, area_number = area_selector()
D = load_forecast_defaults()
y_all = fill_series(area_type, area_number)

# ---------------- Time frame (synced with the map)
st.subheader("Time frame")
c1, c2, c3 = st.columns(3)
with c1:
    end = shared_date("Last week of training data (= week on the map)", key="fc_end",
                      min_value=(y_all.index[0] + pd.Timedelta(weeks=110)).date(), max_value=y_all.index[-1].date(),
                      help="This is the same week as on the map. Choose a week in the past to compare the "
                           "forecast with what really happened.")
end = min(pd.Timestamp(end), y_all.index[-1])
start_default = max(end - pd.Timedelta(weeks=D["history_weeks"]), y_all.index[0])
start = c2.date_input("First week of training data", value=start_default.date(),
                      min_value=y_all.index[0].date(), max_value=(end - pd.Timedelta(weeks=60)).date())
horizon = c3.slider("Forecast horizon (weeks)", 4, 52, int(D["horizon"]))
st.caption(f"Default from the notebook test: {D['history_weeks']} weeks of history and {D['horizon']} weeks horizon.")

# ---------------- Model and parameters
st.subheader("Model")
models = {"snarimax": "SNARIMAX (River)", "lags": "Lag model (linear regression on past weeks)"}
c1, c2 = st.columns([1, 2])
model = c1.radio("Model type", list(models), format_func=models.get, index=list(models).index(D["model"]))
lr = c1.select_slider("Learning rate (SGD)", [0.001, 0.003, 0.01, 0.03], value=0.01)
params = {}
with c2:
    if model == "snarimax":
        a, b, c = st.columns(3)
        params["p"] = a.number_input("p (AR order)", 0, 6, int(D.get("p", 2)))
        params["d"] = b.number_input("d (differencing)", 0, 2, int(D.get("d", 1)))
        params["q"] = c.number_input("q (MA order)", 0, 6, int(D.get("q", 0)))
        params["sp"] = a.number_input("sp (seasonal AR)", 0, 2, int(D.get("sp", 0)))
        params["sd"] = b.number_input("sd (seasonal differencing)", 0, 1, int(D.get("sd", 1)))
        params["sq"] = c.number_input("sq (seasonal MA)", 0, 2, int(D.get("sq", 0)))
        st.caption("The season is m = 52 weeks. sd = 1 means that the model looks at the change from the same "
                   "week last year.")
    else:
        params["n_lags"] = st.slider("Number of past weeks (lags)", 1, 12, int(D.get("n_lags", 4)))
        st.caption("The model also gets the week of the year (as sine and cosine) and the exogenous variables.")

# ---------------- Exogenous variables
st.subheader("Exogenous variables")
default_exog = []
if "weather" in D["sources"]:
    default_exog += ["Temperature (°C)", "Precipitation (mm)", "Snowfall (cm)"]
if "currency" in D["sources"]:
    default_exog += ["EUR", "USD"]
if "flows" in D["sources"]:
    default_exog += ["Net export (MW)"]
options = list(WEATHER_PROPS) + ["EUR", "USD", "Net export (MW)"]
exog_choice = st.multiselect("Variables", options, default=default_exog,
                             help="For the forecast weeks the future is unknown: weather and flows use the mean "
                                  "of the same week in the training years, currency uses the last known rate.")
show_interval = st.checkbox("Show 90 % interval (quantile loss, bonus)", value=True,
                            help="Two extra models are trained with optim.losses.Quantile(0.05) and (0.95).")

# ---------------- Collect the data
y = y_all[(y_all.index >= pd.Timestamp(start)) & (y_all.index <= end)]
if len(y) < 60:
    st.warning("The training period is too short. Choose at least 60 weeks.")
    st.stop()
if model == "snarimax" and (params["sd"] or params["sp"]) and len(y) < 110:
    st.warning("With seasonal terms the model needs at least two years of history.")
    st.stop()

parts = []
weather_cols = [c for c in exog_choice if c in WEATHER_PROPS]
w_end = min(end.date(), date.today() - timedelta(days=6))
if weather_cols:
    with st.spinner("Downloading weather ..."):
        parts.append(weather_weekly(area_type, area_number, y.index[0].date(), w_end, props=weather_cols))
if "EUR" in exog_choice or "USD" in exog_choice:
    rates = load_or_stop(download_exchange_rates, ("EUR", "USD"), "2015-01-01",
                         what="the exchange rates from Norges Bank")
    parts.append(rates_weekly(rates)[[c for c in ["EUR", "USD"] if c in exog_choice]])
if "Net export (MW)" in exog_choice:
    parts.append(load_or_stop(flows_weekly, area_type, area_number, what="the cross-border flows from MongoDB").to_frame())
exog = pd.concat(parts, axis=1, sort=True) if parts else None

if exog is not None:
    coverage = exog.reindex(y.index).notna().mean()
    dropped = list(coverage[coverage < 0.8].index)
    if dropped:
        st.info(f"Not used, because they cover less than 80 % of the training period: {', '.join(dropped)}. "
                f"(The cross-border flows only cover the last two years, and the exchange rates start in 2015.)")

# ---------------- Forecast
with st.spinner("Training the model and forecasting ..."):
    fc = forecast_fill(y, exog, model=model, horizon=horizon, lr=lr, **params)
    lower = upper = None
    if show_interval:
        lower = forecast_fill(y, exog, model=model, horizon=horizon, lr=lr, quantile=0.05, **params)
        upper = forecast_fill(y, exog, model=model, horizon=horizon, lr=lr, quantile=0.95, **params)
        lower, upper = np.minimum(lower, fc), np.maximum(upper, fc)    # the interval always contains the forecast

actual = y_all.reindex(fc.index).dropna()
fig = forecast_animation(y, fc, actual if len(actual) else None, lower, upper,
                         title=f"{area_label()}: forecast from {end.date()} ({horizon} weeks)")
st.plotly_chart(fig, width="stretch")
st.caption("Press ▶ Play to see the forecast week by week, or move the slider under the plot.")

# ---------------- Key numbers
c1, c2, c3 = st.columns(3)
c1.metric("Fill now", f"{y.iloc[-1]:.1f} %")
c2.metric(f"Forecast in {horizon} weeks", f"{fc.iloc[-1]:.1f} %", f"{fc.iloc[-1] - y.iloc[-1]:+.1f} pp",
          delta_color="off")
if len(actual):
    mae = float((fc.reindex(actual.index) - actual).abs().mean())
    c3.metric("Mean absolute error (known weeks)", f"{mae:.1f} pp")
else:
    c3.metric("Real values", "not known yet")

with st.expander("Forecast as a table"):
    table = pd.DataFrame({"Forecast (%)": fc.round(1)})
    if lower is not None:
        table["Lower 5 % (%)"], table["Upper 95 % (%)"] = lower.round(1), upper.round(1)
    if len(actual):
        table["Actual (%)"] = actual.round(1)
    table.index = table.index.date
    st.dataframe(table, width="stretch")