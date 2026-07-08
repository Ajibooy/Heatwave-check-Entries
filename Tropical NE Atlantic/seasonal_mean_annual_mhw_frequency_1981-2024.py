import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

# ==========================
# LOAD CSV
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
N_YEARS = YEAR_MAX - YEAR_MIN + 1

SEASONS = ["DJF", "MAM", "JJA", "SON"]

df = df[
    (df["start_year"] >= YEAR_MIN) &
    (df["start_year"] <= YEAR_MAX)
].copy()

# ==========================
# FUNCTION: MEAN ANNUAL FREQUENCY GRID
# ==========================

def make_frequency_grid(data):
    freq = (
        data.groupby(["lat", "lon"])
            .size()
            .reset_index(name="total_events")
    )

    freq["events_per_year"] = freq["total_events"] / N_YEARS

    grid = (
        freq.pivot(
            index="lat",
            columns="lon",
            values="events_per_year"
        )
        .sort_index()
    )

    return grid

# ==========================
# FORMAT AXIS LABELS
# ==========================

def format_lon(x):
    return f"{abs(int(x))}°W"

def format_lat(y):
    return f"{int(y)}°N"

# ==========================
# CREATE SEASONAL GRIDS
# ==========================

season_grids = {
    season: make_frequency_grid(df[df["season"] == season])
    for season in SEASONS
}

# ==========================
# CLASSIFIED COLOR SCALE
# Data remains events year⁻¹
# Legend shows frequency classes 0–5
# ==========================

levels = np.array([0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50])
class_labels = ["0", "1", "2", "3", "4", "5"]

cmap = plt.get_cmap("YlOrRd", len(levels) - 1)

norm = BoundaryNorm(
    boundaries=levels,
    ncolors=cmap.N,
    clip=True
)

# ==========================
# PLOT
# ==========================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(16, 9),
    sharex=True,
    sharey=True
)

axes = axes.flatten()

for ax, season in zip(axes, SEASONS):

    grid = season_grids[season]

    lon = grid.columns.values
    lat = grid.index.values
    z = grid.values

    mesh = ax.pcolormesh(
        lon,
        lat,
        z,
        cmap=cmap,
        norm=norm,
        shading="auto"
    )

    ax.set_title(season, fontsize=14)

    xticks = np.arange(-60, -14, 5)
    yticks = np.arange(0, 31, 5)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.set_xticklabels(
        [format_lon(x) for x in xticks],
        fontsize=10
    )

    ax.set_yticklabels(
        [format_lat(y) for y in yticks],
        fontsize=10
    )

# ==========================
# COLORBAR
# Use midpoint ticks for class labels
# ==========================

fig.subplots_adjust(
    right=0.86,
    top=0.88,
    wspace=0.10,
    hspace=0.22
)

cbar_ax = fig.add_axes([
    0.89,
    0.18,
    0.025,
    0.64
])

class_tick_positions = (levels[:-1] + levels[1:]) / 2

cbar = fig.colorbar(
    mesh,
    cax=cbar_ax,
    ticks=class_tick_positions
)

cbar.set_ticklabels(class_labels)

cbar.set_label(
    "Mean annual MHW frequency class",
    fontsize=12
)

cbar.ax.tick_params(labelsize=10)

# ==========================
# TITLE
# ==========================

fig.suptitle(
    "Seasonal Mean Annual MHW Frequency Classes (1981–2024)",
    fontsize=18,
    fontweight="bold"
)

plt.show()