# ==============================================================
# TROPICAL NORTH EAST ATLANTIC
# SEASONAL MEAN SST TIME SERIES
#
# Region:
#   0-30°N, 60-10°W
#
# Period:
#   1982-2024
#
# Seasons:
#   DJF = December-January-February
#   MAM = March-April-May
#   JJA = June-July-August
#   SON = September-October-November
#
# Output:
#   Four seasonal panels
#
# IMPORTANT:
#   - Mean SST values displayed on ALL four y-axes
#   - Same y-axis range for all panels
#   - No trend line
#   - No °C/decade
#   - No R²
#   - No p-value
# ==============================================================


import os
import glob
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ==============================================================
# 1. SETTINGS
# ==============================================================

DATA_DIR = r"C:\Users\Aina Ajibola\Desktop\oisst_data"

LAT_MIN = 0.0
LAT_MAX = 30.0

LON_MIN = -60.0
LON_MAX = -10.0

# December 1981 is needed for DJF 1982
START_DATE = "1981-12-01"
END_DATE = "2024-12-31"

START_YEAR = 1982
END_YEAR = 2024


# ==============================================================
# 2. PREPROCESS OISST
# ==============================================================

def preprocess(ds):

    rename = {}

    for old, new in {
        "latitude": "lat",
        "longitude": "lon",
        "Time": "time",
        "TIME": "time"
    }.items():

        if old in ds.coords or old in ds.dims:
            rename[old] = new

    if rename:
        ds = ds.rename(rename)

    # ----------------------------------------------------------
    # Remove singleton vertical dimensions
    # ----------------------------------------------------------

    for dim in [
        "zlev",
        "depth",
        "lev",
        "level"
    ]:

        if (
            dim in ds.dims
            and ds.sizes[dim] == 1
        ):

            ds = ds.squeeze(
                dim,
                drop=True
            )

    # ----------------------------------------------------------
    # Check SST variable
    # ----------------------------------------------------------

    if "sst" not in ds.data_vars:

        raise KeyError(
            "Variable 'sst' was not found in the OISST files."
        )

    ds = ds[["sst"]]

    # ----------------------------------------------------------
    # Convert longitude from 0-360 to -180...180 if necessary
    # ----------------------------------------------------------

    if float(ds.lon.max()) > 180:

        ds = ds.assign_coords(
            lon=((ds.lon + 180.0) % 360.0) - 180.0
        )

    ds = ds.sortby("lat")
    ds = ds.sortby("lon")

    # ----------------------------------------------------------
    # Tropical North East Atlantic
    # ----------------------------------------------------------

    ds = ds.sel(
        lat=slice(
            LAT_MIN,
            LAT_MAX
        ),
        lon=slice(
            LON_MIN,
            LON_MAX
        )
    )

    return ds


# ==============================================================
# 3. SEASON ASSIGNMENT
#
# Avoids the np.select string/integer dtype problem.
# ==============================================================

def assign_season(month):

    if month in [12, 1, 2]:
        return "DJF"

    elif month in [3, 4, 5]:
        return "MAM"

    elif month in [6, 7, 8]:
        return "JJA"

    elif month in [9, 10, 11]:
        return "SON"

    return "Unknown"


# ==============================================================
# 4. FIND OISST FILES
# ==============================================================

files = sorted(
    glob.glob(
        os.path.join(
            DATA_DIR,
            "*_oisst.nc"
        )
    )
)

if not files:

    files = sorted(
        glob.glob(
            os.path.join(
                DATA_DIR,
                "*.nc"
            )
        )
    )

if not files:

    raise FileNotFoundError(
        f"No NetCDF files were found in:\n{DATA_DIR}"
    )


print("=" * 80)

print(
    "TROPICAL NORTH EAST ATLANTIC "
    "SEASONAL SST ANALYSIS"
)

print("=" * 80)

print(
    f"\nSST files found: {len(files):,}"
)

print(
    f"First file: {os.path.basename(files[0])}"
)

print(
    f"Last file:  {os.path.basename(files[-1])}"
)


# ==============================================================
# 5. OPEN OISST DATA
# ==============================================================

print(
    "\nOpening OISST dataset..."
)

ds = xr.open_mfdataset(
    files,
    combine="by_coords",
    preprocess=preprocess,
    parallel=False,
    data_vars="minimal",
    coords="minimal",
    compat="override",
    join="outer",
    engine="netcdf4",
    chunks={
        "time": 365
    }
)

ds = ds.sortby("time")

sst = ds["sst"]


# ==============================================================
# 6. NORMALIZE TIME
# ==============================================================

dates = pd.DatetimeIndex(
    sst.time.values
).normalize()

sst = sst.assign_coords(
    time=dates
)

time_index = pd.DatetimeIndex(
    sst.time.values
)

# --------------------------------------------------------------
# Remove duplicate dates
# --------------------------------------------------------------

keep = np.where(
    ~time_index.duplicated(
        keep="first"
    )
)[0]

sst = sst.isel(
    time=keep
)

sst = sst.sortby("time")


# ==============================================================
# 7. SELECT ANALYSIS PERIOD
# ==============================================================

sst = sst.sel(
    time=slice(
        START_DATE,
        END_DATE
    )
)

dates = pd.DatetimeIndex(
    sst.time.values
)


print(
    f"\nAvailable period: "
    f"{dates[0].date()} to {dates[-1].date()}"
)

print(
    f"Number of days: {len(dates):,}"
)

print(
    f"Grid: "
    f"{sst.sizes['lat']} × {sst.sizes['lon']}"
)


# ==============================================================
# 8. SST UNIT CHECK
# ==============================================================

sample = float(
    sst.isel(
        time=slice(
            0,
            min(
                10,
                sst.sizes["time"]
            )
        )
    )
    .mean(
        skipna=True
    )
    .compute()
)


if sample > 100:

    print(
        "\nConverting SST from Kelvin to °C..."
    )

    sst = sst - 273.15

else:

    print(
        "\nSST already appears to be °C."
    )


# ==============================================================
# 9. AREA-WEIGHTED REGIONAL DAILY MEAN SST
# ==============================================================

print(
    "\nCalculating area-weighted regional daily SST..."
)

# --------------------------------------------------------------
# Latitude weighting
# --------------------------------------------------------------

weights = np.cos(
    np.deg2rad(
        sst.lat
    )
)


# --------------------------------------------------------------
# Average longitude first.
#
# Then calculate cosine-latitude weighted mean across latitude.
# --------------------------------------------------------------

regional_daily = (
    sst
    .mean(
        dim="lon",
        skipna=True
    )
    .weighted(
        weights
    )
    .mean(
        dim="lat",
        skipna=True
    )
)


print(
    "Loading regional daily SST..."
)

regional_daily = regional_daily.compute()


# ==============================================================
# 10. CREATE DAILY DATAFRAME
# ==============================================================

df = pd.DataFrame(
    {
        "date":
            pd.DatetimeIndex(
                regional_daily.time.values
            ),

        "sst":
            np.asarray(
                regional_daily.values
            ).reshape(-1)
    }
)


df = df.dropna(
    subset=["sst"]
)


print(
    f"\nValid daily regional SST values: {len(df):,}"
)


# ==============================================================
# 11. MONTH
# ==============================================================

df["month"] = (
    df["date"].dt.month
)


# ==============================================================
# 12. ASSIGN SEASON
# ==============================================================

df["season"] = (
    df["month"]
    .apply(
        assign_season
    )
)


if (
    df["season"]
    ==
    "Unknown"
).any():

    raise ValueError(
        "Some dates could not be assigned to a season."
    )


# ==============================================================
# 13. DEFINE SEASON YEAR
# ==============================================================

df["season_year"] = (
    df["date"].dt.year
)


# --------------------------------------------------------------
# DJF requires special treatment.
#
# December belongs to the NEXT year.
#
# Example:
#
# December 2009
# January 2010
# February 2010
#
# = DJF 2010
# --------------------------------------------------------------

december_mask = (
    df["month"]
    ==
    12
)

df.loc[
    december_mask,
    "season_year"
] += 1


# ==============================================================
# 14. COUNT VALID DAYS IN EACH SEASON
# ==============================================================

season_counts = (
    df
    .groupby(
        [
            "season_year",
            "season"
        ]
    )
    ["sst"]
    .count()
)


# ==============================================================
# 15. KEEP COMPLETE SEASONS
#
# Seasons contain about 90-92 days.
#
# >=89 prevents substantially incomplete seasons from entering
# the calculation.
# ==============================================================

complete_seasons = (
    season_counts[
        season_counts >= 89
    ]
    .index
)


# ==============================================================
# 16. CALCULATE SEASONAL MEAN SST
# ==============================================================

seasonal_sst = (
    df
    .groupby(
        [
            "season_year",
            "season"
        ]
    )
    ["sst"]
    .mean()
)


seasonal_sst = seasonal_sst.loc[
    seasonal_sst.index.isin(
        complete_seasons
    )
]


seasonal_sst = (
    seasonal_sst
    .reset_index(
        name="mean_sst"
    )
)


# ==============================================================
# 17. SELECT 1982-2024
# ==============================================================

seasonal_sst = seasonal_sst[
    (
        seasonal_sst["season_year"]
        >=
        START_YEAR
    )
    &
    (
        seasonal_sst["season_year"]
        <=
        END_YEAR
    )
].copy()


# ==============================================================
# 18. SEASON ORDER
# ==============================================================

season_order = [
    "DJF",
    "MAM",
    "JJA",
    "SON"
]


# ==============================================================
# 19. PRINT RESULTS
# ==============================================================

seasonal_table = (
    seasonal_sst
    .pivot(
        index="season_year",
        columns="season",
        values="mean_sst"
    )
    .reindex(
        columns=season_order
    )
)


print(
    "\n"
    + "=" * 80
)

print(
    "SEASONAL MEAN SST"
)

print(
    "=" * 80
)

print(
    seasonal_table.round(3)
)


# ==============================================================
# 20. CHECK EACH SEASON
# ==============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "AVAILABLE SEASONS"
)

print(
    "=" * 80
)


for season in season_order:

    temp = seasonal_sst[
        seasonal_sst["season"]
        ==
        season
    ]

    if len(temp) > 0:

        print(
            f"{season}: "
            f"{int(temp['season_year'].min())}-"
            f"{int(temp['season_year'].max())} "
            f"({len(temp)} seasons)"
        )


# ==============================================================
# 21. CALCULATE COMMON Y-AXIS RANGE
#
# IMPORTANT:
#
# All four panels use exactly the same scale.
#
# This means a temperature difference visible between DJF and
# SON, for example, is a genuine difference rather than an
# artifact caused by different y-axis ranges.
# ==============================================================

all_sst = (
    seasonal_sst[
        "mean_sst"
    ]
    .to_numpy()
)


# Use whole-degree tick marks
tick_min = np.floor(
    np.nanmin(
        all_sst
    )
)

tick_max = np.ceil(
    np.nanmax(
        all_sst
    )
)


# Give a little visual space above and below
y_min = tick_min - 0.25
y_max = tick_max + 0.75


# Explicit y-axis ticks
y_ticks = np.arange(
    tick_min,
    tick_max + 1,
    1.0
)


print(
    f"\nCommon plot range: "
    f"{y_min:.2f} to {y_max:.2f} °C"
)


# ==============================================================
# 22. CREATE FOUR-PANEL FIGURE
#
# IMPORTANT:
#
# sharey=False is deliberate here.
#
# We manually give every plot the SAME ylim and yticks.
#
# This retains scientific comparability while ensuring that
# MAM and SON also display their y-axis temperature labels.
# ==============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(
        14,
        8
    ),
    sharex=True,
    sharey=False
)


axes = axes.flatten()


panel_labels = [
    "(a)",
    "(b)",
    "(c)",
    "(d)"
]


# ==============================================================
# 23. YEAR TICKS
# ==============================================================

year_ticks = np.arange(
    1982,
    2025,
    5
)


# ==============================================================
# 24. PLOT ALL FOUR SEASONS
# ==============================================================

for ax, season, panel in zip(
    axes,
    season_order,
    panel_labels
):

    # ----------------------------------------------------------
    # Extract season
    # ----------------------------------------------------------

    season_data = (
        seasonal_sst[
            seasonal_sst["season"]
            ==
            season
        ]
        .sort_values(
            "season_year"
        )
    )


    years = (
        season_data[
            "season_year"
        ]
        .to_numpy()
    )


    mean_sst = (
        season_data[
            "mean_sst"
        ]
        .to_numpy()
    )


    # ----------------------------------------------------------
    # Seasonal mean SST
    # ----------------------------------------------------------

    ax.plot(
        years,
        mean_sst,
        linewidth=1.8
    )


    # ----------------------------------------------------------
    # Panel title
    # ----------------------------------------------------------

    ax.set_title(
        f"{panel} {season}",
        fontsize=13,
        fontweight="bold",
        pad=10
    )


    # ----------------------------------------------------------
    # X axis
    # ----------------------------------------------------------

    ax.set_xlim(
        START_YEAR,
        END_YEAR
    )

    ax.set_xticks(
        year_ticks
    )


    # ----------------------------------------------------------
    # Y axis
    #
    # SAME range on ALL four panels
    # ----------------------------------------------------------

    ax.set_ylim(
        y_min,
        y_max
    )

    ax.set_yticks(
        y_ticks
    )


    # ----------------------------------------------------------
    # IMPORTANT FIX
    #
    # Explicitly show numerical SST tick labels on every panel.
    # This fixes MAM and SON.
    # ----------------------------------------------------------

    ax.tick_params(
        axis="y",
        labelleft=True,
        labelsize=10
    )

    ax.tick_params(
        axis="x",
        labelsize=10
    )


    # ----------------------------------------------------------
    # Y-axis label on EVERY panel
    # ----------------------------------------------------------

    ax.set_ylabel(
        "Mean SST (°C)",
        fontsize=11,
        fontweight="bold"
    )


    # ----------------------------------------------------------
    # Grid
    # ----------------------------------------------------------

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.30
    )


# ==============================================================
# 25. X-AXIS LABELS
#
# Put Year on the bottom panels.
# ==============================================================

axes[2].set_xlabel(
    "Year",
    fontsize=11,
    fontweight="bold"
)

axes[3].set_xlabel(
    "Year",
    fontsize=11,
    fontweight="bold"
)


# ==============================================================
# 26. OVERALL TITLE
# ==============================================================

fig.suptitle(
    "Seasonal Mean SST in the Tropical North East Atlantic "
    "(1982–2024)",
    fontsize=16,
    fontweight="bold",
    y=0.98
)


# ==============================================================
# 27. ADJUST PANEL SPACING
#
# This provides enough room for the MAM and SON y-axis labels
# without making the overall figure unnecessarily large.
# ==============================================================

fig.subplots_adjust(
    left=0.08,
    right=0.98,
    bottom=0.09,
    top=0.89,
    wspace=0.18,
    hspace=0.25
)


# ==============================================================
# 28. SHOW FIGURE
# ==============================================================

plt.show()


# ==============================================================
# 29. CLOSE DATASET
# ==============================================================

ds.close()


print(
    "\nSeasonal SST analysis completed successfully."
)
