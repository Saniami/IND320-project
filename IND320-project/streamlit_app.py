import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="IND320 - Reservoir Data", layout="wide")

@st.cache_data
def load_data():
    # Build the path to the CSV relative to this file, so it works locally and on Streamlit Cloud
    df = pd.read_csv(Path(__file__).parent / "data" / "reservoirs.csv")
    df["dato_Id"] = pd.to_datetime(df["dato_Id"])

    df = df.rename(columns={
        "dato_Id": "Date",
        "omrType": "Area_Type",
        "omrnr": "Area_Number",
        "iso_aar": "ISO_Year",
        "iso_uke": "ISO_Week",
        "fyllingsgrad": "Fill_Percentage",
        "kapasitet_TWh": "Capacity_TWh",
        "fylling_TWh": "Fill_TWh",
        "neste_Publiseringsdato": "Next_Publication_Date",
        "fyllingsgrad_forrige_uke": "Fill_Percentage_Previous_Week",
        "endring_fyllingsgrad": "Fill_Percentage_Change",
    })

    # Filter to a single area (Area 4) for a clean, consistent time series
    df = df[df["Area_Number"] == df["Area_Number"].iloc[0]]
    df = df.sort_values("Date").reset_index(drop=True)

    return df

df = load_data()

st.title("IND320 - Reservoir Data App")

st.write(
    """
    This app explores Norwegian reservoir (vannmagasin) data, showing fill levels 
    and capacity over time. The data is read weekly and includes fill percentage, 
    capacity, and week-over-week change.
    
    **Note:** The dataset contains multiple reservoir areas mixed together. 
    To keep the data consistent and avoid misleading jumps between unrelated 
    time series, this app only shows measurements for **Area 4**.
    
    Use the sidebar to navigate between the pages:
    - **Data Table** — an overview table of the dataset, with a mini trend chart 
      for each column showing the first month of data.
    - **Plot** — an interactive plot where you can choose a column and a date range 
      to explore how it changes over time.
    - **Page 4** — placeholder page for future project parts.
    """
)

st.subheader("Latest Measurement (Only for Area 4)")

# Get the most recent row in the dataset
latest = df.iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Fill Percentage", f"{latest['Fill_Percentage']:.1%}")
col2.metric("Fill Volume", f"{latest['Fill_TWh']:.2f} TWh")
col3.metric("Capacity", f"{latest['Capacity_TWh']:.2f} TWh")

st.caption(f"Latest data point: {latest['Date'].date()} (Area {latest['Area_Number']})")