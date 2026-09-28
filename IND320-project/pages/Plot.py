import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

@st.cache_data
def load_data():
    df = pd.read_csv(Path(__file__).parent.parent / "data" / "reservoirs.csv")
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

    df = df[df["Area_Number"] == df["Area_Number"].iloc[0]]
    df = df.sort_values("Date").reset_index(drop=True)

    return df

df = load_data()

# Dropdown menu: let the user pick one column, or all columns together
columns = df.columns.tolist()
choice = st.selectbox("Choose column", options=["All columns"] + columns)

# Build a list of year-month values (e.g. "1995-09") to use in the slider below
df["Month"] = df["Date"].dt.to_period("M").astype(str)
months = sorted(df["Month"].unique())

# Slider: let the user pick a range of months to display.
# Default value is just the first month (both start and end set to the first month).
selected_range = st.select_slider(
    "Select months",
    options=months,
    value=(months[0], months[0])
)

# Keep only the rows that fall inside the selected month range
start, end = selected_range
filtered = df[(df["Month"] >= start) & (df["Month"] <= end)]

st.subheader(f"Showing: {choice}")

# Create an empty plot to draw on
fig, ax1 = plt.subplots(figsize=(10, 5))

if choice == "All columns":
    # Percentage-level columns share a natural 0-1 scale (left axis).
    # TWh-based columns share their own energy scale (right axis).
    # Fill_Percentage_Change is excluded here since it can be negative and has 
    # a fundamentally different meaning (a rate of change, not a level) — 
    # it can still be viewed on its own via the column dropdown.
    left_cols = ["Fill_Percentage", "Fill_Percentage_Previous_Week"]
    right_cols = ["Capacity_TWh", "Fill_TWh"]

    ax2 = ax1.twinx()

    colors_left = ["tab:blue", "tab:green"]
    colors_right = ["tab:orange", "tab:purple"]

    # Fill_Percentage and Fill_Percentage_Previous_Week are nearly identical, 
    # so transparency (alpha) and thicker lines help show both overlapping lines.
    lines = []
    for col, color in zip(left_cols, colors_left):
        line, = ax1.plot(filtered["Date"], filtered[col], label=col, color=color, alpha=0.7, linewidth=2)
        lines.append(line)

    for col, color in zip(right_cols, colors_right):
        line, = ax2.plot(filtered["Date"], filtered[col], label=col, color=color, linestyle="--", alpha=0.7, linewidth=2)
        lines.append(line)

    ax1.set_ylabel("Fill Percentage (fraction)")
    ax2.set_ylabel("Energy (TWh)")

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", bbox_to_anchor=(1.1, 1))

else:
    ax1.plot(filtered["Date"], filtered[choice], color="tab:blue")
    ax1.set_ylabel(choice)

# Add axis titles, a header, and tidy up the date labels
ax1.set_xlabel("Date")
ax1.set_title(f"{choice} over time")
plt.xticks(rotation=45)
plt.tight_layout()

# Show the finished plot in the app
st.pyplot(fig)