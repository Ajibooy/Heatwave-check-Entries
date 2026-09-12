# ==============================================================
# TROPICAL NORTH EAST ATLANTIC
# MONTHLY SST CLIMATOLOGY MAPS
#
# Region:
#   0-30°N, 60-10°W
#
# Climatology period:
#   1981-2010
#
# Output:
#   12 monthly climatology maps
#   arranged in a 3 x 4 panel figure
#
# Each panel:
#   Mean SST for that calendar month
#   averaged across the climatological period
#
# One shared colorbar:
#   Mean SST (°C)
#
# No P90
# No ±5-day window
# No 31-day smoothing
# No CSV
# No figure saved
# ==============================================================

import os
import glob
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from cartopy.mpl.ticker import (
    LongitudeFormatter,
    LatitudeFormatter
)

warnings.filterwarnings("ignore")


# ==============================================================
# 1. SETTINGS
# ==============================================================

DATA_DIR = r"C:\Users\Aina Ajibola\Desktop\oisst_data"

LAT_MIN = 0.0
LAT_MAX = 30.0

LON_MIN = -60.0
LON_MAX = -10.0

START_DATE = "1981-01-01"
END_DATE = "2010-12-31"


MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]


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

    # Convert longitude to -180...180
    if float(ds.lon.max()) > 180:

        ds = ds.assign_coords(
            lon=((ds.lon + 180.0) % 360.0) - 180.0
        )

    ds = ds.sortby("lat")
    ds = ds.sortby("lon")

    # Select Tropical North East Atlantic
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
# 3. FIND OISST FILES
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


print("=" * 85)

print(
    "TROPICAL NORTH EAST ATLANTIC "
    "MONTHLY SST CLIMATOLOGY"
)

print("=" * 85)

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
# 4. OPEN DATASET
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
# 5. NORMALIZE TIME
# ==============================================================

time_index = pd.DatetimeIndex(
    sst.time.values
).normalize()

sst = sst.assign_coords(
    time=time_index
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
# 6. SELECT CLIMATOLOGY PERIOD
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
    f"\nActual data period used: "
    f"{dates[0].date()} to "
    f"{dates[-1].date()}"
)

print(
    f"Number of daily observations: "
    f"{len(dates):,}"
)

print(
    f"Grid: "
    f"{sst.sizes['lat']} × "
    f"{sst.sizes['lon']}"
)


# ==============================================================
# 7. UNIT CHECK
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
# 8. CALCULATE MONTHLY CLIMATOLOGY
#
# groupby("time.month") gives:
# month = 1...12
#
# For each month:
# average every daily SST observation from that month
# across all years in the climatology.
# ==============================================================

print(
    "\nCalculating monthly SST climatology..."
)

monthly_climatology = (
    sst
    .groupby(
        "time.month"
    )
    .mean(
        dim="time",
        skipna=True
    )
    .compute()
)


print(
    "Monthly climatology calculation completed."
)


# ==============================================================
# 9. CONVERT TO NUMPY
# ==============================================================

monthly_values = np.asarray(
    monthly_climatology.values,
    dtype=float
)


# Shape should be:
# (12, lat, lon)

print(
    f"\nMonthly climatology array shape: "
    f"{monthly_values.shape}"
)


# ==============================================================
# 10. OCEAN MASK / VALID VALUES
# ==============================================================

valid_values = monthly_values[
    np.isfinite(
        monthly_values
    )
]


print(
    "\n"
    + "=" * 85
)

print(
    "MONTHLY CLIMATOLOGY SUMMARY"
)

print(
    "=" * 85
)


for month_number in range(
    1,
    13
):

    month_data = monthly_values[
        month_number - 1
    ]

    month_ocean = month_data[
        np.isfinite(
            month_data
        )
    ]

    print(
        f"{MONTH_NAMES[month_number - 1]:>9}: "
        f"regional mean = "
        f"{np.nanmean(month_ocean):.2f} °C | "
        f"min = "
        f"{np.nanmin(month_ocean):.2f} °C | "
        f"max = "
        f"{np.nanmax(month_ocean):.2f} °C"
    )


# ==============================================================
# 11. COMMON COLOR SCALE
#
# CRITICAL:
# All 12 maps use exactly the same color scale.
#
# This allows direct month-to-month comparison.
# ==============================================================

vmin = float(
    np.nanpercentile(
        valid_values,
        1
    )
)

vmax = float(
    np.nanpercentile(
        valid_values,
        99
    )
)


# Round outward to nearest 0.5°C

vmin = (
    np.floor(
        vmin * 2
    )
    /
    2
)

vmax = (
    np.ceil(
        vmax * 2
    )
    /
    2
)


print(
    f"\nCommon color scale: "
    f"{vmin:.1f} to {vmax:.1f} °C"
)


# ==============================================================
# 12. LEVELS
# ==============================================================

levels = np.linspace(
    vmin,
    vmax,
    20
)


# ==============================================================
# 13. COLORBAR TICKS
# ==============================================================

ticks = np.linspace(
    vmin,
    vmax,
    7
)


# ==============================================================
# 14. MAP COORDINATES
# ==============================================================

lon_values = (
    monthly_climatology.lon.values
)

lat_values = (
    monthly_climatology.lat.values
)


LON, LAT = np.meshgrid(
    lon_values,
    lat_values
)


projection = ccrs.PlateCarree()


# ==============================================================
# 15. CREATE 3 × 4 FIGURE
# ==============================================================

fig, axes = plt.subplots(
    3,
    4,
    figsize=(
        18,
        11
    ),
    subplot_kw={
        "projection":
            projection
    }
)


axes = axes.flatten()


# ==============================================================
# 16. MAP TICKS
# ==============================================================

longitude_ticks = [
    -60,
    -50,
    -40,
    -30,
    -20,
    -10
]


latitude_ticks = [
    0,
    10,
    20,
    30
]


# ==============================================================
# 17. PLOT EACH MONTH
# ==============================================================

cf = None


for month_index, ax in enumerate(
    axes
):

    month_data = monthly_values[
        month_index
    ]


    # ----------------------------------------------------------
    # SST climatology map
    # ----------------------------------------------------------

    cf = ax.contourf(
        LON,
        LAT,
        month_data,
        levels=levels,
        cmap="turbo",
        extend="both",
        transform=projection
    )


    # ----------------------------------------------------------
    # Map extent
    # ----------------------------------------------------------

    ax.set_extent(
        [
            LON_MIN,
            LON_MAX,
            LAT_MIN,
            LAT_MAX
        ],
        crs=projection
    )


    # ----------------------------------------------------------
    # Land
    # ----------------------------------------------------------

    ax.add_feature(
        cfeature.LAND.with_scale(
            "50m"
        ),
        facecolor="0.70",
        edgecolor="black",
        linewidth=0.4,
        zorder=10
    )


    # ----------------------------------------------------------
    # Coastlines
    # ----------------------------------------------------------

    ax.coastlines(
        resolution="50m",
        linewidth=0.6,
        zorder=11
    )


    # ----------------------------------------------------------
    # Borders
    # ----------------------------------------------------------

    ax.add_feature(
        cfeature.BORDERS.with_scale(
            "50m"
        ),
        linewidth=0.3,
        edgecolor="black",
        zorder=11
    )


    # ----------------------------------------------------------
    # Ticks
    # ----------------------------------------------------------

    ax.set_xticks(
        longitude_ticks,
        crs=projection
    )

    ax.set_yticks(
        latitude_ticks,
        crs=projection
    )


    ax.xaxis.set_major_formatter(
        LongitudeFormatter(
            degree_symbol="°"
        )
    )

    ax.yaxis.set_major_formatter(
        LatitudeFormatter(
            degree_symbol="°"
        )
    )


    ax.tick_params(
        labelsize=8
    )


    # ----------------------------------------------------------
    # Grid
    # ----------------------------------------------------------

    ax.gridlines(
        crs=projection,
        draw_labels=False,
        xlocs=longitude_ticks,
        ylocs=latitude_ticks,
        linewidth=0.3,
        linestyle="--",
        alpha=0.35
    )


    # ----------------------------------------------------------
    # Month title
    # ----------------------------------------------------------

    ax.set_title(
        MONTH_NAMES[
            month_index
        ],
        fontsize=11,
        fontweight="bold",
        pad=6
    )


# ==============================================================
# 18. SHARED COLORBAR
#
# One single colorbar for all 12 panels.
# ==============================================================

cbar = fig.colorbar(
    cf,
    ax=axes.tolist(),
    orientation="vertical",
    fraction=0.025,
    pad=0.025,
    ticks=ticks
)


cbar.set_label(
    "Mean SST (°C)",
    fontsize=11,
    fontweight="bold"
)


cbar.ax.yaxis.set_major_formatter(
    mticker.FormatStrFormatter(
        "%.1f"
    )
)


cbar.update_ticks()


# ==============================================================
# 19. OVERALL TITLE
# ==============================================================

fig.suptitle(
    "Monthly Mean SST Climatology in the "
    "Tropical North East Atlantic\n"
    "1981–2010",
    fontsize=16,
    fontweight="bold",
    y=0.97
)


# ==============================================================
# 20. SPACING
# ==============================================================

fig.subplots_adjust(
    left=0.05,
    right=0.90,
    bottom=0.06,
    top=0.90,
    wspace=0.16,
    hspace=0.22
)


# ==============================================================
# 21. SHOW
# ==============================================================

plt.show()


# ==============================================================
# 22. CLOSE DATASET
# ==============================================================

ds.close()


print(
    "\nMonthly SST climatology maps completed successfully."
)