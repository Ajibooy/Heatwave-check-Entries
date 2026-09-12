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
#   Four panels:
#       (a) DJF
#       (b) MAM
#       (c) JJA
#       (d) SON
#
# No trend line
# No °C/decade
# No R²
# No p-value
#
# Regional SST uses cosine(latitude) area weighting.
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

# Need December 1981 for DJF 1982
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

    # Remove singleton vertical dimensions
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

    if "sst" not in ds.data_vars:
        raise KeyError(
            "Variable 'sst' was not found."
        )

    ds = ds[["sst"]]

    # Convert longitude 0-360 -> -180...180
    if float(ds.lon.max()) > 180:

        ds = ds.assign_coords(
            lon=((ds.lon + 180.0) % 360.0) - 180.0
        )

    ds = ds.sortby("lat")
    ds = ds.sortby("lon")

    # Tropical North East Atlantic
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
# 3. SAFE SEASON FUNCTION
#
# This completely avoids the np.select() dtype error.
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

    else:
        return "Unknown"


# ==============================================================
# 4. FIND FILES
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
        f"No NetCDF files found in:\n{DATA_DIR}"
    )


print("=" * 80)

print(
    "TROPICAL NORTH EAST ATLANTIC "
    "SEASONAL SST ANALYSIS"
)

print("=" * 80)

print(
    f"\nSST files found: "
    f"{len(files):,}"
)

print(
    f"First file: "
    f"{os.path.basename(files[0])}"
)

print(
    f"Last file: "
    f"{os.path.basename(files[-1])}"
)


# ==============================================================
# 5. OPEN OISST
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

ds = ds.sortby(
    "time"
)

sst = ds[
    "sst"
]


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

# Remove duplicates
keep = np.where(
    ~time_index.duplicated(
        keep="first"
    )
)[0]

sst = sst.isel(
    time=keep
)

sst = sst.sortby(
    "time"
)


# ==============================================================
# 7. SELECT PERIOD
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
    f"{dates[0].date()} to "
    f"{dates[-1].date()}"
)

print(
    f"Number of days: "
    f"{len(dates):,}"
)

print(
    f"Grid: "
    f"{sst.sizes['lat']} × "
    f"{sst.sizes['lon']}"
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

    sst = (
        sst
        -
        273.15
    )

else:

    print(
        "\nSST already appears to be °C."
    )


# ==============================================================
# 9. AREA-WEIGHTED REGIONAL DAILY SST
# ==============================================================

print(
    "\nCalculating area-weighted regional daily SST..."
)

weights = np.cos(
    np.deg2rad(
        sst.lat
    )
)

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

regional_daily = (
    regional_daily
    .compute()
)


# ==============================================================
# 10. CREATE DATAFRAME
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


# ==============================================================
# 11. ASSIGN SEASON
#
# Safe method:
# apply the assign_season() function month-by-month.
# ==============================================================

df["month"] = (
    df["date"].dt.month
)

df["season"] = (
    df["month"]
    .apply(
        assign_season
    )
)


# ==============================================================
# 12. CHECK SEASON ASSIGNMENT
# ==============================================================

print(
    "\nSeason assignment counts:"
)

print(
    df["season"]
    .value_counts()
)

if (
    df["season"]
    ==
    "Unknown"
).any():

    raise ValueError(
        "Some dates were not assigned to a season."
    )


# ==============================================================
# 13. SEASON YEAR
# ==============================================================

df[
    "season_year"
] = (
    df["date"].dt.year
)


# --------------------------------------------------------------
# December belongs to the following DJF year.
#
# Example:
# Dec 1999 + Jan 2000 + Feb 2000 = DJF 2000
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
# 14. COUNT DAYS IN EACH SEASON
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
# 15. KEEP COMPLETE SEASONS ONLY
# ==============================================================

# Seasons normally contain 90-92 days.
# Requiring >=89 prevents incomplete seasons being used.

complete_seasons = (
    season_counts[
        season_counts >= 89
    ]
    .index
)


# ==============================================================
# 16. CALCULATE MEAN SST FOR EACH SEASON/YEAR
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
        seasonal_sst[
            "season_year"
        ]
        >=
        START_YEAR
    )
    &
    (
        seasonal_sst[
            "season_year"
        ]
        <=
        END_YEAR
    )
].copy()


# ==============================================================
# 18. PRINT SEASONAL TABLE
# ==============================================================

season_order = [
    "DJF",
    "MAM",
    "JJA",
    "SON"
]

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
    seasonal_table.round(
        3
    )
)


# ==============================================================
# 19. CHECK AVAILABLE YEARS
# ==============================================================

print(
    "\nAvailable years per season:"
)

for season in season_order:

    season_data = seasonal_sst[
        seasonal_sst[
            "season"
        ]
        ==
        season
    ]

    print(
        f"{season}: "
        f"{season_data['season_year'].min()}-"
        f"{season_data['season_year'].max()} | "
        f"{len(season_data)} seasons"
    )


# ==============================================================
# 20. COMMON Y-AXIS RANGE
# ==============================================================

all_sst = (
    seasonal_sst[
        "mean_sst"
    ]
    .to_numpy()
)


y_min = (
    np.floor(
        np.nanmin(
            all_sst
        )
        *
        2
    )
    /
    2
    -
    0.25
)


y_max = (
    np.ceil(
        np.nanmax(
            all_sst
        )
        *
        2
    )
    /
    2
    +
    0.25
)


# ==============================================================
# 21. FOUR-PANEL FIGURE
# ==============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(
        15,
        10
    ),
    sharex=True,
    sharey=True
)

axes = axes.flatten()


panel_labels = [
    "(a)",
    "(b)",
    "(c)",
    "(d)"
]


# ==============================================================
# 22. PLOT EACH SEASON
# ==============================================================

for ax, season, panel in zip(
    axes,
    season_order,
    panel_labels
):

    season_data = (
        seasonal_sst[
            seasonal_sst[
                "season"
            ]
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
    # Seasonal SST line only
    # ----------------------------------------------------------

    ax.plot(
        years,
        mean_sst,
        linewidth=1.8
    )

    ax.set_title(
        f"{panel} {season}",
        fontsize=13,
        fontweight="bold",
        pad=10
    )

    ax.set_xlim(
        START_YEAR,
        END_YEAR
    )

    ax.set_ylim(
        y_min,
        y_max
    )

    ax.grid(
        linestyle="--",
        alpha=0.3
    )


# ==============================================================
# 23. AXIS LABELS
# ==============================================================

axes[0].set_ylabel(
    "Mean SST (°C)",
    fontsize=11,
    fontweight="bold"
)

axes[2].set_ylabel(
    "Mean SST (°C)",
    fontsize=11,
    fontweight="bold"
)

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
# 24. YEAR TICKS
# ==============================================================

year_ticks = np.arange(
    1982,
    2025,
    5
)

for ax in axes:

    ax.set_xticks(
        year_ticks
    )


# ==============================================================
# 25. OVERALL TITLE
# ==============================================================

fig.suptitle(
    "Seasonal Mean SST in the Tropical North East Atlantic "
    "(1982–2024)",
    fontsize=16,
    fontweight="bold",
    y=0.98
)


plt.tight_layout(
    rect=[
        0,
        0,
        1,
        0.95
    ]
)

plt.show()


# ==============================================================
# 26. CLOSE DATASET
# ==============================================================

ds.close()

print(
    "\nSeasonal SST analysis completed successfully."
)