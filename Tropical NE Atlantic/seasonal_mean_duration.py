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

SEASONS = ["DJF", "MAM", "JJA", "SON"]

df = df[
    (df["start_year"] >= YEAR_MIN) &
    (df["start_year"] <= YEAR_MAX)
].copy()

# ==========================
# FORMAT AXIS LABELS
# ==========================

def format_lon(x):
    return f"{abs(int(x))}°W"

def format_lat(y):
    return f"{int(y)}°N"

# ==========================
# GRID FUNCTION
# ==========================

def make_mean_grid(data, value_col):
    mean_values = (
        data.groupby(["lat", "lon"])[value_col]
            .mean()
            .reset_index(name="mean_value")
    )

    grid = (
        mean_values.pivot(
            index="lat",
            columns="lon",
            values="mean_value"
        )
        .sort_index()
    )

    return grid

# ==========================
# MAP PLOTTING FUNCTION
# ==========================

def plot_seasonal_maps(
    grids,
    title,
    cbar_label,
    levels,
    cmap_name="YlOrRd"
):
    cmap = plt.get_cmap(
        cmap_name,
        len(levels) - 1
    )

    norm = BoundaryNorm(
        boundaries=levels,
        ncolors=cmap.N,
        clip=True
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 9),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    for ax, season in zip(axes, SEASONS):

        grid = grids[season]

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

    cbar = fig.colorbar(
        mesh,
        cax=cbar_ax,
        ticks=levels
    )

    cbar.set_label(
        cbar_label,
        fontsize=12
    )

    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(
        title,
        fontsize=18,
        fontweight="bold"
    )

    plt.show()

# ==========================
# MEAN MHW INTENSITY MAPS
# ==========================

intensity_grids = {
    season: make_mean_grid(
        df[df["season"] == season],
        "mean_intensity"
    )
    for season in SEASONS
}

intensity_levels = np.arange(0, 1.6, 0.25)

plot_seasonal_maps(
    grids=intensity_grids,
    title="Seasonal Mean MHW Intensity (1981–2024)",
    cbar_label="Mean MHW intensity (°C)",
    levels=intensity_levels
)

# ==========================
# MEAN MHW DURATION MAPS
# ==========================

duration_grids = {
    season: make_mean_grid(
        df[df["season"] == season],
        "duration_days"
    )
    for season in SEASONS
}

duration_levels = np.arange(0, 65, 5)

plot_seasonal_maps(
    grids=duration_grids,
    title="Seasonal Mean MHW Duration (1981–2024)",
    cbar_label="Mean MHW duration (days)",
    levels=duration_levels
)