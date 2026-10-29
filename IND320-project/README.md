# IND320 - Data to decision

Project work for the course IND320. The repository has four parts that build on each other.
This README covers part 1 and part 2.

- **Streamlit app:** https://ind320-project-697hjpoelth3xk2xbtm6tk.streamlit.app/


## Branches
| Branch | Content |
|---|---|
| `main` | Part 1 (and later parts after peer review and feedback) |
| `part2` | Part 2, merged into `main` after peer review |

## Part 1 - Reservoir data from a CSV file
**Notebook:** `notebook/Project_work_p1.ipynb`
- Reads `reservoirs.csv` (weekly reservoir statistics from NVE) with pandas.
- Renames the Norwegian columns to English names.
- Prints the shape, data types, date range and areas, and plots each column separately.
- Plots all columns together with two y-axes (percentages on the left, TWh on the right).
- Finding: the file mixes several areas and is not sorted by date, so one area (EL 4) is
  selected and sorted by date before plotting.

**Streamlit app (four pages):** home page with the latest measurement, a table with
small line charts (`LineChartColumn`), a plot page with a column selector and a month
slider, and a placeholder page. Data is read from the CSV file with `st.cache_data`.

## Part 2 - APIs and databases
**Notebook:** `notebook/Project_work_p2.ipynb`
1. **NVE Magasinstatistikk API** replaces the CSV file. The areas are of three types:
   NO (whole Norway), EL (the five electricity price areas) and VASS (watercourse areas).
   An area is identified by both type and number.
2. **ENTSO-E API:** two years of physical cross-border power flow between Norwegian
   price areas and neighbouring areas (four connections, both directions). The 15-minute
   values are averaged to hourly values.
3. **Cassandra and Spark:** the flows are written to Cassandra (local, in Docker) with
   Spark, and the relevant columns are read back with Spark.
4. **MongoDB:** the data extracted with Spark is saved in MongoDB (Atlas, in the cloud).
5. **Plots:** a pie chart of the total flow in and out of Norway for one connection and
   year, and a line plot of the first month with one line per connection and direction.

**Streamlit app:**
| File | Content |
|---|---|
| `streamlit_app.py` | Home page: data sources and the latest measurement for Norway |
| `pages/Data_table.py` | `st.radio` for area type (EL, NO, VASS); one row per area with small line charts (first month and last 12 weeks) |
| `pages/Plot.py` | Reservoir plots with area type, column and a month interval slider |
| `pages/Cross_border.py` | Two columns: pie chart (`st.radio`, year) and line plots (`st.pills` for months), with an `st.expander` about the data source. Reads from MongoDB. |

## Data flow