import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

# =========================================================
# SETTINGS
# =========================================================

CSV_FILE = "mhw_events_1981_2024.csv"

YEAR_MIN = 1981
YEAR_MAX = 2024

STUDY_START = pd.Timestamp(f"{YEAR_MIN}-01-01")
STUDY_END = pd.Timestamp(f"{YEAR_MAX}-12-31")

SEASONS = ["DJF", "MAM", "JJA", "SON"]

# Minimum number of completed events required at each
# grid cell within each season.
MIN_EVENTS_PER_CELL = 5

# Plotting ranges.
# Values above the final boundary are retained and shown
# using the extended top colour.
DURATION_LEVELS = np.arange(0, 35, 5)       # 0, 5, ..., 30 days
INTENSITY_LEVELS = np.arange(0, 1.76, 0.25)

# =========================================================
# LOAD AND VALIDATE DATA
# =========================================================

df = pd.read_csv(
    CSV_FILE,
    parse_dates=["start_date", "end_date", "peak_date"]
)

required_columns = {
    "lat",
    "lon",
    "start_date",
    "end_date",
    "duration_days",
    "mean_intensity"
}

missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

df = df[
    df["start_date"].notna()
    & df["end_date"].notna()
    & df["lat"].notna()
    & df["lon"].notna()
].copy()

# Keep events beginning during the study period.
df = df[
    df["start_date"].between(STUDY_START, STUDY_END)
].copy()

print("Initial rows:", len(df))

# =========================================================
# REMOVE DUPLICATE EVENT RECORDS
# =========================================================

# Prefer event_id when it exists.
if "event_id" in df.columns:
    duplicate_subset = ["event_id"]
else:
    duplicate_subset = [
        "lat",
        "lon",
        "start_date",
        "end_date"
    ]

before_duplicates = len(df)

df = (
    df.drop_duplicates(subset=duplicate_subset)
      .reset_index(drop=True)
)

print(
    "Duplicate rows removed:",
    before_duplicates - len(df)
)

# =========================================================
# VERIFY EVENT DURATION
# =========================================================

# Inclusive duration: start and end dates both count.
df["calculated_duration"] = (
    df["end_date"].dt.normalize()
    - df["start_date"].dt.normalize()
).dt.days + 1

duration_difference = (
    df["duration_days"]
    - df["calculated_duration"]
).abs()

print(
    "Rows with duration disagreement:",
    int((duration_difference > 1).sum())
)

# Use the duration calculated directly from dates.
df["duration_days"] = df["calculated_duration"]

# Basic validity checks.
df = df[
    (df["duration_days"] >= 5)
    & np.isfinite(df["duration_days"])
    & np.isfinite(df["mean_intensity"])
].copy()

# =========================================================
# REMOVE CENSORED EVENTS
# =========================================================

# An event touching the first or last study date may have
# started earlier or continued beyond the available record.
df["left_censored"] = (
    df["start_date"].dt.normalize() <= STUDY_START
)

df["right_censored"] = (
    df["end_date"].dt.normalize() >= STUDY_END
)

print(
    "Left-censored events removed:",
    int(df["left_censored"].sum())
)

print(
    "Right-censored events removed:",
    int(df["right_censored"].sum())
)

df = df[
    ~df["left_censored"]
    & ~df["right_censored"]
].copy()

# =========================================================
# ASSIGN SEASON FROM EVENT START DATE
# =========================================================

def month_to_season(month):
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


df["season"] = (
    df["start_date"]
    .dt.month
    .map(month_to_season)
)

df["start_year"] = df["start_date"].dt.year

# =========================================================
# DIAGNOSTICS
# =========================================================

print("\nCompleted events retained:", len(df))

print("\nDuration summary:")
print(
    df["duration_days"]
    .describe(
        percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
    )
)

print("\nMean-intensity summary:")
print(
    df["mean_intensity"]
    .describe(
        percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
    )
)

print("\nLongest completed events:")
diagnostic_columns = [
    column for column in [
        "event_id",
        "lat",
        "lon",
        "start_date",
        "end_date",
        "season",
        "duration_days",
        "mean_intensity"
    ]
    if column in df.columns
]

print(
    df.nlargest(10, "duration_days")[
        diagnostic_columns
    ].to_string(index=False)
)

# =========================================================
# COMPUTE SEASONAL GRID STATISTICS
# =========================================================

# One efficient groupby calculates both metrics and event count.
seasonal_stats = (
    df.groupby(
        ["season", "lat", "lon"],
        observed=True
    )
    .agg(
        mean_duration=("duration_days", "mean"),
        mean_intensity=("mean_intensity", "mean"),
        event_count=("duration_days", "size")
    )
    .reset_index()
)

# Mask cells with too few events.
seasonal_stats.loc[
    seasonal_stats["event_count"] < MIN_EVENTS_PER_CELL,
    ["mean_duration", "mean_intensity"]
] = np.nan

print(
    "\nCells retained after minimum-event masking:",
    seasonal_stats[
        seasonal_stats["mean_duration"].notna()
    ].shape[0]
)

# =========================================================
# BUILD GRIDS
# =========================================================

def build_seasonal_grids(value_column):
    grids = {}

    for season in SEASONS:
        subset = seasonal_stats[
            seasonal_stats["season"] == season
        ]

        grid = (
            subset.pivot(
                index="lat",
                columns="lon",
                values=value_column
            )
            .sort_index()
            .sort_index(axis=1)
        )

        grids[season] = grid

    return grids


duration_grids = build_seasonal_grids(
    "mean_duration"
)

intensity_grids = build_seasonal_grids(
    "mean_intensity"
)

# =========================================================
# AXIS LABEL FORMATTERS
# =========================================================

def format_lon(value):
    if value < 0:
        return f"{abs(int(value))}°W"
    if value > 0:
        return f"{int(value)}°E"
    return "0°"


def format_lat(value):
    if value > 0:
        return f"{int(value)}°N"
    if value < 0:
        return f"{abs(int(value))}°S"
    return "0°"


# =========================================================
# SEASONAL MAP FUNCTION
# =========================================================

def plot_seasonal_maps(
    grids,
    title,
    colorbar_label,
    levels,
    cmap_name="YlOrRd"
):
    if len(levels) < 2:
        raise ValueError(
            "At least two colour boundaries are required."
        )

    cmap = plt.get_cmap(
        cmap_name,
        len(levels) - 1
    )

    norm = BoundaryNorm(
        boundaries=levels,
        ncolors=cmap.N,
        clip=False
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 9),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    mesh = None

    for ax, season in zip(axes, SEASONS):
        grid = grids[season]

        if grid.empty:
            ax.set_title(f"{season} — no data")
            ax.set_axis_off()
            continue

        longitude = grid.columns.to_numpy()
        latitude = grid.index.to_numpy()
        values = grid.to_numpy(dtype=float)

        mesh = ax.pcolormesh(
            longitude,
            latitude,
            values,
            cmap=cmap,
            norm=norm,
            shading="auto"
        )

        ax.set_title(
            season,
            fontsize=14
        )

        longitude_ticks = np.arange(
            -60,
            -9,
            5
        )

        latitude_ticks = np.arange(
            0,
            31,
            5
        )

        ax.set_xticks(longitude_ticks)
        ax.set_yticks(latitude_ticks)

        ax.set_xticklabels(
            [
                format_lon(value)
                for value in longitude_ticks
            ],
            fontsize=10
        )

        ax.set_yticklabels(
            [
                format_lat(value)
                for value in latitude_ticks
            ],
            fontsize=10
        )

        ax.set_xlim(-60, -10)
        ax.set_ylim(0, 30)

    if mesh is None:
        raise RuntimeError(
            "No valid grid was available for plotting."
        )

    fig.subplots_adjust(
        right=0.86,
        top=0.88,
        wspace=0.10,
        hspace=0.22
    )

    colorbar_axis = fig.add_axes(
        [0.89, 0.18, 0.025, 0.64]
    )

    colorbar = fig.colorbar(
        mesh,
        cax=colorbar_axis,
        ticks=levels,
        extend="max"
    )

    colorbar.set_label(
        colorbar_label,
        fontsize=12
    )

    colorbar.ax.tick_params(
        labelsize=10
    )

    fig.suptitle(
        title,
        fontsize=18,
        fontweight="bold"
    )

    plt.show()
    plt.close(fig)

# =========================================================
# PLOT MEAN MHW DURATION
# =========================================================

plot_seasonal_maps(
    grids=duration_grids,
    title=(
        "Seasonal Mean MHW Duration "
        "(1981–2024)"
    ),
    colorbar_label=(
        "Mean MHW duration (days)"
    ),
    levels=DURATION_LEVELS,
    cmap_name="YlOrRd"
)

# =========================================================
# PLOT MEAN MHW INTENSITY
# =========================================================

plot_seasonal_maps(
    grids=intensity_grids,
    title=(
        "Seasonal Mean MHW Intensity "
        "(1981–2024)"
    ),
    colorbar_label=(
        "Mean MHW intensity (°C)"
    ),
    levels=INTENSITY_LEVELS,
    cmap_name="YlOrRd"
)