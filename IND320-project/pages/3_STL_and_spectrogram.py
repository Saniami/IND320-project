"""Time series analysis of the reservoir fill percentage: STL and spectrogram.
Uses the same global area selector as the reservoir pages."""
import streamlit as st
from utils import area_selector, area_label, fill_series, stl_plot, spectrogram_plot

st.title("Reservoir filling: STL and spectrogram")
area_type, area_number = area_selector()                     # global setting
fill = fill_series(area_type, area_number)
st.caption(f"Weekly fill percentage for {area_label()}, {fill.index.min():%Y}-{fill.index.max():%Y}.")

tab_stl, tab_spec = st.tabs(["STL", "Spectrogram"])

with tab_stl:
    st.write("STL splits the fill percentage into trend, a yearly season and residual (period = 52 weeks).")
    c1, c2, c3 = st.columns(3)
    seasonal = c1.slider("Seasonal smoother", 7, 51, 17, step=2)       # step 2 from 7 keeps it odd
    trend = c2.slider("Trend smoother (weeks)", 53, 261, 105, step=2)  # odd and longer than the period
    robust = c3.checkbox("Robust", value=True)
    st.pyplot(stl_plot(fill, seasonal=seasonal, trend=trend, robust=robust, title=area_label()))

with tab_spec:
    st.write("The spectrogram shows which repeating patterns are strong, and how they change over time.")
    c1, c2 = st.columns(2)
    window = c1.slider("Window length (weeks)", 26, 260, 104, step=2)
    overlap_pct = c2.slider("Window overlap (%)", 0, 90, 70, step=5)
    overlap = min(round(window * overlap_pct / 100), window - 1)        # must be smaller than the window
    st.pyplot(spectrogram_plot(fill, window_length=window, window_overlap=overlap, title=area_label()))