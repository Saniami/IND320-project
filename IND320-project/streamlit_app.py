"""Main file of the app. It only sets up the page config and the menu.
The pages are grouped by type of task (see the sketch in the part 4 notebook):
Home (map) -> Explore, Anomalies and Forecast. Each group has its own colour (utils.GROUP_COLORS)."""
import streamlit as st
from utils import init_state, show_selection_in_sidebar


st.set_page_config(page_title="IND320 - Energy and weather", page_icon=":material/water_drop:", layout="wide")
init_state()                                  # default area, week and weather property for all pages

pages = {
    "Home": [
        st.Page("pages/home.py", title="Map and selection", icon=":material/map:", default=True),
    ],
    "Explore": [
        st.Page("pages/explore_data_overview.py", title="Reservoir data overview", icon=":material/table_chart:"),
        st.Page("pages/explore_cross_border.py", title="Cross-border flows", icon=":material/swap_horiz:"),
        st.Page("pages/explore_currency.py", title="Currency", icon=":material/euro:"),
        st.Page("pages/explore_correlation.py", title="Weather vs. reservoir", icon=":material/stacked_line_chart:"),
    ],
    "Anomalies": [
        st.Page("pages/anomalies_stl.py", title="STL and spectrogram", icon=":material/waves:"),
        st.Page("pages/anomalies_weather_outliers.py", title="Weather outliers", icon=":material/thunderstorm:"),
    ],
    "Forecast": [
        st.Page("pages/forecast_reservoir.py", title="Reservoir forecast", icon=":material/trending_up:"),
    ],
}

pg = st.navigation(pages)
show_selection_in_sidebar()
pg.run()