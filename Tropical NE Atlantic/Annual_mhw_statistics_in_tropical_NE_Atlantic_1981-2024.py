import pandas as pd
import matplotlib.pyplot as plt
# ==========================
# LOAD CSV
# ==========================
df = pd.read_csv(
    "mhw_events_1981_2024.csv",
    parse_dates=["start_date", "end_date", "peak_date"]
)
# ==========================
# YEARS
# ==========================
YEAR_MIN = 1981
YEAR_MAX = 2024
years = list(range(YEAR_MIN, YEAR_MAX + 1))
df = df[
    (df["start_year"] >= YEAR_MIN) &
    (df["start_year"] <= YEAR_MAX)
].copy()
# ==========================
# A. MHW FREQUENCY
# Average events per grid cell per year
# ==========================
cell_event_counts = (
    df.groupby(["start_year", "lat", "lon"])
      .size()
      .reset_index(name="event_count")
)
mhw_frequency = (
    cell_event_counts
    .groupby("start_year")["event_count"]
    .mean()
    .reindex(years, fill_value=0)
)
# ==========================
# B. MAX INTENSITY
# Maximum intensity reached each year
# ==========================
mhw_max_intensity = (
    df.groupby("start_year")["max_intensity"]
      .max()
      .reindex(years, fill_value=0)
)
# ==========================
# C. MHW DURATION
# Mean duration each year
# ==========================
mhw_duration = (
    df.groupby("start_year")["duration_days"]
      .mean()
      .reindex(years, fill_value=0)
)
# ==========================
# D. TOTAL DAYS OF MHWs
# Average total MHW days per grid cell per year
# ==========================
cell_total_days = (
    df.groupby(["start_year", "lat", "lon"])["duration_days"]
      .sum()
      .reset_index(name="total_days")
)
mhw_total_days = (
    cell_total_days
    .groupby("start_year")["total_days"]
    .mean()
    .reindex(years, fill_value=0)
)
# ==========================
# E. MEAN INTENSITY
# Mean of event mean intensities per year
# ==========================
mhw_mean_intensity = (
    df.groupby("start_year")["mean_intensity"]
      .mean()
      .reindex(years, fill_value=0)
)
# ==========================
# F. CUMULATIVE INTENSITY
# Mean cumulative intensity per event per year
# ==========================
mhw_cumulative_intensity = (
    df.groupby("start_year")["cumulative_intensity"]
      .mean()
      .reindex(years, fill_value=0)
)
# ==========================
# COLOR FUNCTION
# Highlight ALL exact maximum years
# ==========================
def highlight_all_max(series):
    max_value = series.max()
    colors = [
        "red" if value == max_value else "steelblue"
        for value in series.values
    ]
    max_years = series[
        series == max_value
    ].index.tolist()
    return colors, max_years
freq_colors, freq_years = highlight_all_max(mhw_frequency)
max_int_colors, max_int_years = highlight_all_max(mhw_max_intensity)
dur_colors, dur_years = highlight_all_max(mhw_duration)
days_colors, days_years = highlight_all_max(mhw_total_days)
mean_int_colors, mean_int_years = highlight_all_max(mhw_mean_intensity)
cum_int_colors, cum_int_years = highlight_all_max(mhw_cumulative_intensity)
print("Highest frequency year(s):", freq_years)
print("Highest max intensity year(s):", max_int_years)
print("Highest duration year(s):", dur_years)
print("Highest total MHW days year(s):", days_years)
print("Highest mean intensity year(s):", mean_int_years)
print("Highest cumulative intensity year(s):", cum_int_years)
# ==========================
# 3x2 FIGURE
# ==========================
fig, axes = plt.subplots(
    3,
    2,
    figsize=(16, 12)
)
axes = axes.flatten()
plots = [
    {
        "data": mhw_frequency,
        "colors": freq_colors,
        "title": "MHWs Frequency",
        "ylabel": "Events per grid cell",
        "label": "A"
    },
    {
        "data": mhw_max_intensity,
        "colors": max_int_colors,
        "title": "MHWs Max Intensity",
        "ylabel": "°C",
        "label": "B"
    },
    {
        "data": mhw_duration,
        "colors": dur_colors,
        "title": "MHWs Duration",
        "ylabel": "Days",
        "label": "C"
    },
    {
        "data": mhw_total_days,
        "colors": days_colors,
        "title": "Total Days of MHWs",
        "ylabel": "Days per grid cell",
        "label": "D"
    },
    {
        "data": mhw_mean_intensity,
        "colors": mean_int_colors,
        "title": "MHWs Mean Intensity",
        "ylabel": "°C",
        "label": "E"
    },
    {
        "data": mhw_cumulative_intensity,
        "colors": cum_int_colors,
        "title": "MHWs Cumulative Intensity",
        "ylabel": "°C days",
        "label": "F"
    }
]
for ax, item in zip(axes, plots):
    ax.bar(
        item["data"].index,
        item["data"].values,
        color=item["colors"]
    )
    ax.set_title(item["title"])
    ax.set_ylabel(item["ylabel"])
    ax.text(
        -0.08,
        1.05,
        item["label"],
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.35)
    ax.set_xticks(years[::4])
    ax.tick_params(axis="x", rotation=45)
# Add x-label only to bottom row
axes[4].set_xlabel("Year")
axes[5].set_xlabel("Year")
fig.suptitle(
    "Annual MHW Statistics in Tropical North East Atlantic Area (1981–2024)",
    fontsize=16,
    fontweight="bold"
)
plt.tight_layout()
plt.show()