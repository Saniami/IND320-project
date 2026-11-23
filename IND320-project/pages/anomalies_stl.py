"""Time series analysis of the reservoir fill percentage: STL and spectrogram (Plotly)."""
import streamlit as st
from utils import page_header, area_selector, area_label, fill_series, stl_plot, spectrogram_plot

page_header("STL and spectrogram", "Anomalies",
            "The weekly fill percentage split into trend, season and residual (STL), and the repeating "
            "patterns over time (spectrogram). Large residuals are unusual weeks.")
area_type, area_number = area_selector()
fill = fill_series(area_type, area_number)
st.caption(f"Weekly fill percentage for {area_label()}, {fill.index.min():%Y}-{fill.index.max():%Y}.")

tab_stl, tab_spec = st.tabs(["STL", "Spectrogram"])

with tab_stl:
    c1, c2, c3 = st.columns(3)
    seasonal = c1.slider("Seasonal smoother", 7, 51, 17, step=2)        # odd numbers
    trend = c2.slider("Trend smoother (weeks)", 53, 261, 105, step=2)   # odd and longer than the period
    robust = c3.checkbox("Robust", value=True)
    with st.spinner("Calculating STL ..."):
        st.plotly_chart(stl_plot(fill, seasonal=seasonal, trend=trend, robust=robust, title=area_label()),
                        width="stretch")

with tab_spec:
    c1, c2 = st.columns(2)
    window = c1.slider("Window length (weeks)", 26, 260, 104, step=2)
    overlap_pct = c2.slider("Window overlap (%)", 0, 90, 70, step=5)
    overlap = min(round(window * overlap_pct / 100), window - 1)        # must be smaller than the window
    st.plotly_chart(spectrogram_plot(fill, window_length=window, window_overlap=overlap, title=area_label()),
                    width="stretch")