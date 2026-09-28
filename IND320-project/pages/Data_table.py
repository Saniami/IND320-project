import streamlit as st
import pandas as pd

# This function loads and cleans up the data.
# @st.cache_data means Streamlit remembers the result, so the file isn't re-read every time something changes in the app (makes it faster).
@st.cache_data
def load_data():
    df = pd.read_csv("data/reservoirs.csv")

    # Turn the date column into a real date type, so we can sort and filter by it
    df["dato_Id"] = pd.to_datetime(df["dato_Id"])

    # Give the columns clear English names instead of the original Norwegian ones
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

    # The dataset mixes several reservoir areas together in one file.
    # We only keep one area so that the data is consistent and we don't get misleading jumps between unrelated time series.
    df = df[df["Area_Number"] == df["Area_Number"].iloc[0]]

    # Put the rows in date order, since the original file is not sorted
    df = df.sort_values("Date").reset_index(drop=True)

    return df

# Load the cleaned data
df = load_data()

st.title("Reservoir Data Table")

# Find the very first date in the dataset
first_month_date = df["Date"].min()

# Keep only the rows from that same month and year (i.e. the first month of data)
first_month_data = df[
    (df["Date"].dt.year == first_month_date.year) &
    (df["Date"].dt.month == first_month_date.month)
]

# Get a list of only the numeric columns (the ones we can show as a trend line)
numeric_cols = df.select_dtypes(include="number").columns

# Build a small table: one row per column, with its first-month values stored as a list
table_rows = []
for col in numeric_cols:
    table_rows.append({
        "Column": col,
        "First month trend": first_month_data[col].tolist()
    })

# Turn that list of rows into a proper table (DataFrame)
summary_df = pd.DataFrame(table_rows)

# Display the table, showing "First month trend" as a small line chart in each row
st.dataframe(
    summary_df,
    column_config={
        "First month trend": st.column_config.LineChartColumn("First month trend")
    },
    hide_index=True,
)