"""Home page: map of the areas from Magasinstatistikk. The user chooses area type, what to colour by
and the week. A click on an area selects it (thick red outline) and stores it for all other pages."""
from datetime import date, timedelta
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from utils import (page_header, init_state, load_or_stop, load_reservoir, map_values, make_map,
                   WEATHER_PROPS, AREA_NUMBERS, TYPE_TEXT, last_sunday, area_label)

page_header("Map and selection", "Home",
            "Choose the type of area and what the colours should show. Click on an area to select it. "
            "The selected area and week are used as defaults on all the other pages.")
init_state()
ss = st.session_state

df = load_or_stop(load_reservoir, what="the reservoir data from NVE")
first_week, last_week = df["Date"].min().date(), df["Date"].max().date()

# ---------------- Selectors
c1, c2, c3 = st.columns([1.2, 1.4, 1])
types = {"EL": "Price areas (EL)", "VASS": "Water areas (VASS)", "NO": "Whole country (NO)"}
area_type = c1.radio("Areas", list(types), format_func=types.get, index=list(types).index(ss["area_type"]))
if area_type != ss["area_type"]:                                       # keep the shared selection valid
    ss["area_type"], ss["area_number"] = area_type, AREA_NUMBERS[area_type][0]

colour_options = ["Reservoir fill (%)", "Change in fill last week (pp)"] + list(WEATHER_PROPS)
colour_by = c2.selectbox("Colour the areas by", colour_options)
if colour_by in WEATHER_PROPS:
    ss["weather_prop"] = colour_by                                    # shared with the weather pages
    period = c3.selectbox("Weather aggregated over", [1, 4, 13], index=1,
                          format_func=lambda w: f"last {w} week(s)")
else:
    period = 1
    c3.metric("Latest data", str(last_week))


# ---------------- Time (week)
def _save_week():
    ss["map_date"] = last_sunday(ss["_week_widget"])


week = st.slider("Week", min_value=first_week, max_value=last_week,
                 value=min(max(ss["map_date"], first_week), last_week),
                 step=timedelta(weeks=1), format="YYYY-MM-DD", key="_week_widget", on_change=_save_week,
                 help="The week is also used as the start of the forecast on the Forecast page.")
week = last_sunday(week)
if colour_by in WEATHER_PROPS and week > date.today() - timedelta(days=6):
    st.info("The weather data (ERA5) is a few days behind, so the last days of the period are missing.")

# ---------------- Map
with st.spinner("Preparing the map ..."):
    values = map_values(area_type, colour_by, week, period)
selected = ss["area_number"] if ss["area_type"] == area_type else None
m = make_map(area_type, values, selected, colour_by)

left, right = st.columns([2.2, 1])
with left:
    out = st_folium(m, height=560, use_container_width=True, returned_objects=["last_active_drawing"],
                    key=f"map_{area_type}")
with right:
    st.markdown(f"**Selected area: {area_label()}**")
    st.caption(TYPE_TEXT[area_type])
    st.write(f"Week ending **{week}**")
    table = pd.DataFrame({"Area": [f"{area_type} {n}" for n in values], colour_by: list(values.values())})
    st.dataframe(table.round(1), hide_index=True, width="stretch")
    if all(pd.isna(v) for v in values.values()):
        st.warning("There are no values for this week. Try another week.")
    st.caption("Click on an area in the map to select it. The selection is kept when you open other pages.")

# ---------------- Click on the map -> new selected area
clicked = (out or {}).get("last_active_drawing")
if clicked:
    p = clicked["properties"]
    click_id = (p["Area_Type"], p["Area_Number"])
    # Only react to a new click (the map remembers the last click when the page is redrawn)
    if click_id != ss.get("_last_click") and p["Area_Number"] in AREA_NUMBERS[p["Area_Type"]]:
        ss["_last_click"] = click_id
        ss["area_type"], ss["area_number"] = click_id
        st.rerun()