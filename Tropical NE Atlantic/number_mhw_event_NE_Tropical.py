import pandas as pd
import matplotlib.pyplot as plt
# ==========================
# LOAD EVENT CATALOGUE
# ==========================
df = pd.read_csv(
    "mhw_events_1981_2024.csv",
    parse_dates=["start_date", "end_date", "peak_date"]
)
# ==========================
# SETTINGS
# ==========================
YEAR_MIN = 1981
YEAR_MAX = 2024
years = list(range(YEAR_MIN, YEAR_MAX + 1))
# ==========================
# COUNT EVENTS PER YEAR
# Option 1: grid-cell average frequency
# ==========================
cell_counts = (
    df.groupby(["start_year", "lat", "lon"])
      .size()
      .reset_index(name="events")
)
events_per_year = (
    cell_counts
    .groupby("start_year")["events"]
    .mean()
    .reindex(years, fill_value=0)
)
# ==========================
# PLOT BAR CHART
# ==========================
plt.figure(figsize=(12, 5))
plt.bar(
    events_per_year.index,
    events_per_year.values,
    width=0.7,
    color="red",
    edgecolor="black",
    linewidth=0.6
)
plt.xlabel("Year")
plt.ylabel("Number of events")
plt.title(
    "Number of MHW events in Tropical North East Atlantic area",
    fontsize=14,
    fontweight="bold"
)
plt.grid(axis="y", alpha=0.3)
plt.xticks(years[::5], rotation=0)
plt.tight_layout()
plt.show()
print(events_per_year)

