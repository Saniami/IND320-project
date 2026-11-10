"""Weather page: outliers in precipitation (SPC) and anomalies in temperature (LOF).
The area comes from the global selector, so the weather always matches the reservoir data."""
import streamlit as st
from utils import area_selector, area_label, get_weather, spc_outliers, lof_anomalies

st.title("Weather: outliers and anomalies")
area_selector()                                              # global setting, shared with the other pages
year = st.selectbox("Weather year", list(range(2024, 2014, -1)))   # local to this page
weather = get_weather(year)                                  # hidden, cached download

tab_spc, tab_lof = st.tabs(["Outliers (SPC)", "Anomalies (LOF)"])

with tab_spc:
    st.write("Precipitation with SPC limits. The season is removed with a DCT high-pass filter, "
             "and the limits are based on the median and MAD of what is left.")
    c1, c2 = st.columns(2)
    period = c1.slider("Cut-off: season is everything slower than (days)", 7, 180, 30)   # cut-off = 1/period
    n_std = c2.slider("Number of standard deviations", 1.0, 6.0, 3.0, 0.5)
    fig, summary, outliers = spc_outliers(weather["precipitation"], freq_cutoff=1 / period,
                                          n_std=n_std, title="Precipitation (mm)")
    st.pyplot(fig)
    m1, m2, m3 = st.columns(3)
    m1.metric("Outliers", summary["n_outliers"])
    m2.metric("Share of hours", f"{summary['share_outliers_%']} %")
    m3.metric("Above / below", f"{summary['n_above']} / {summary['n_below']}")
    with st.expander("Largest outliers"):
        st.dataframe(outliers.sort_values("value", ascending=False).head(20))

with tab_lof:
    st.write("Temperature with anomalies from Local Outlier Factor. Each hour is compared with the hours around it.")
    c1, c2 = st.columns(2)
    share = c1.slider("Share of anomalies (%)", 0.1, 5.0, 1.0, 0.1)
    neighbours = c2.slider("Number of neighbours", 10, 200, 50, 10)
    fig, summary, anomalies = lof_anomalies(weather["temperature_2m"], contamination=share / 100,
                                            n_neighbors=neighbours, title="Temperature (°C)")
    st.pyplot(fig)
    m1, m2, m3 = st.columns(3)
    m1.metric("Anomalies", summary["n_anomalies"])
    m2.metric("Mean temp. of anomalies", f"{summary['mean_value_anomalies']} °C")
    m3.metric("Mean temp. all hours", f"{summary['mean_value_all']} °C")
    with st.expander("Strongest anomalies"):
        st.dataframe(anomalies.head(20))