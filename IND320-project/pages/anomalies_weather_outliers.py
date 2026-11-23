"""Weather outliers: SATV/SPC and LOF, now with a selector for the weather property
(temperature is the default). The area comes from the shared selection."""
from datetime import date
import streamlit as st
from utils import page_header, area_selector, get_weather, spc_outliers, lof_anomalies, WEATHER_PROPS

page_header("Weather outliers", "Anomalies",
            "Unusual hours in the hourly weather for the chosen area, found with two methods.")
area_selector()
c1, c2 = st.columns(2)
this_year = date.today().year
year = c1.selectbox("Weather year", list(range(this_year, 2014, -1)), index=1)   # local to this page
weather = get_weather(year)                                                     # spinner + error handling

props = list(WEATHER_PROPS)                      # temperature is first, so it is the default
tab_spc, tab_lof = st.tabs(["Outliers (SATV / SPC)", "Anomalies (LOF)"])

with tab_spc:
    st.write("The season is removed with a DCT high-pass filter (SATV), and the SPC limits are based on the "
             "median and MAD of what is left. The limits follow the seasonal level.")
    c1, c2, c3 = st.columns(3)
    prop = c1.selectbox("Weather property", props, index=0, key="spc_prop")
    period = c2.slider("Cut-off: season is everything slower than (days)", 7, 180, 30)
    n_std = c3.slider("Number of standard deviations", 1.0, 6.0, 3.0, 0.5)
    fig, summary, outliers = spc_outliers(weather[WEATHER_PROPS[prop][0]], freq_cutoff=1 / period,
                                          n_std=n_std, title=prop)
    st.plotly_chart(fig, width="stretch")
    m1, m2, m3 = st.columns(3)
    m1.metric("Outliers", summary["n_outliers"])
    m2.metric("Share of hours", f"{summary['share_outliers_%']} %")
    m3.metric("Above / below", f"{summary['n_above']} / {summary['n_below']}")
    with st.expander("Largest outliers"):
        st.dataframe(outliers.sort_values("value", ascending=False).head(20))

with tab_lof:
    st.write("Local Outlier Factor compares each hour with the hours around it, so a value that is normal in "
             "one season can be an anomaly in another.")
    c1, c2, c3 = st.columns(3)
    prop = c1.selectbox("Weather property", props, index=0, key="lof_prop")
    share = c2.slider("Share of anomalies (%)", 0.1, 5.0, 1.0, 0.1)
    neighbours = c3.slider("Number of neighbours", 10, 200, 50, 10)
    with st.spinner("Running LOF ..."):
        fig, summary, anomalies = lof_anomalies(weather[WEATHER_PROPS[prop][0]], contamination=share / 100,
                                                n_neighbors=neighbours, title=prop)
    st.plotly_chart(fig, width="stretch")
    m1, m2, m3 = st.columns(3)
    m1.metric("Anomalies", summary["n_anomalies"])
    m2.metric("Mean of anomalies", summary["mean_value_anomalies"])
    m3.metric("Mean of all hours", summary["mean_value_all"])
    with st.expander("Strongest anomalies"):
        st.dataframe(anomalies.head(20))