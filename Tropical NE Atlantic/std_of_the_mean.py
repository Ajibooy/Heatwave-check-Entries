# ==============================================================
# TROPICAL NORTH EAST ATLANTIC
# STANDARD DEVIATION OF DAILY SST CLIMATOLOGY
#
# Region:
#   0-30°N, 60-10°W
#
# Climatology:
#   1981-2010
#
# Calculation at every grid cell:
#
#   SD = standard deviation of daily SST values
#        during the climatological period
#
# Output:
#   One spatial map only
#
# Colorbar:
#   SST Standard Deviation (°C)
#
# No mean SST map
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

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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

        ds = ds.rename(
            rename
        )


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
            and
            ds.sizes[dim] == 1
        ):

            ds = ds.squeeze(
                dim,
                drop=True
            )


    # ----------------------------------------------------------
    # Check SST
    # ----------------------------------------------------------

    if "sst" not in ds.data_vars:

        raise KeyError(
            "Variable 'sst' was not found."
        )


    ds = ds[
        ["sst"]
    ]


    # ----------------------------------------------------------
    # Convert longitude:
    # 0-360 -> -180...180
    # ----------------------------------------------------------

    if float(
        ds.lon.max()
    ) > 180:

        ds = ds.assign_coords(
            lon=(
                (
                    ds.lon
                    +
                    180.0
                )
                %
                360.0
            )
            -
            180.0
        )


    ds = ds.sortby(
        "lat"
    )

    ds = ds.sortby(
        "lon"
    )


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

        f"No NetCDF files found in:\n"
        f"{DATA_DIR}"

    )


print(
    "=" * 80
)

print(
    "TROPICAL NORTH EAST ATLANTIC "
    "SST STANDARD DEVIATION"
)

print(
    "=" * 80
)


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
# 4. OPEN OISST
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
# 5. NORMALIZE TIME
# ==============================================================

time_index = pd.DatetimeIndex(
    sst.time.values
).normalize()


sst = sst.assign_coords(
    time=time_index
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
# 7. IMPORTANT COVERAGE WARNING
# ==============================================================

requested_start = pd.Timestamp(
    START_DATE
)


actual_start = pd.Timestamp(
    dates[0]
)


if actual_start > requested_start:

    print(
        "\nWARNING:"
    )

    print(
        f"Requested climatology begins "
        f"{requested_start.date()}, "
        f"but the local archive begins "
        f"{actual_start.date()}."
    )

    print(
        "The calculation therefore uses "
        "the SST observations actually available."
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
# 9. STANDARD DEVIATION AT EVERY GRID CELL
#
# ddof=0:
# population standard deviation across all available daily
# observations in the climatological period.
# ==============================================================

print(
    "\nCalculating SST standard deviation "
    "at every grid cell..."
)


sst_std = (

    sst.std(

        dim="time",

        skipna=True,

        ddof=0

    )

    .compute()

)


print(
    "Standard deviation calculation completed."
)


# ==============================================================
# 10. CONVERT TO NUMPY
# ==============================================================

std_values = np.asarray(

    sst_std.values,

    dtype=float

)


# ==============================================================
# 11. OCEAN MASK
# ==============================================================

ocean_mask = np.isfinite(
    std_values
)


std_values = np.where(

    ocean_mask,

    std_values,

    np.nan

)


ocean_std = std_values[
    ocean_mask
]


# ==============================================================
# 12. RESULTS
# ==============================================================

print(
    "\n"
    +
    "=" * 80
)


print(
    "SST STANDARD DEVIATION RESULTS"
)


print(
    "=" * 80
)


print(
    f"Valid ocean grid cells: "
    f"{np.sum(ocean_mask):,}"
)


print(
    f"Minimum SST SD: "
    f"{np.nanmin(ocean_std):.2f} °C"
)


print(
    f"Mean SST SD: "
    f"{np.nanmean(ocean_std):.2f} °C"
)


print(
    f"Maximum SST SD: "
    f"{np.nanmax(ocean_std):.2f} °C"
)


print(
    f"Median SST SD: "
    f"{np.nanmedian(ocean_std):.2f} °C"
)


# ==============================================================
# 13. MAP COORDINATES
# ==============================================================

lon_values = (
    sst_std.lon.values
)


lat_values = (
    sst_std.lat.values
)


LON, LAT = np.meshgrid(

    lon_values,

    lat_values

)


projection = ccrs.PlateCarree()


# ==============================================================
# 14. COLOR SCALE
#
# Start at zero because standard deviation cannot be negative.
#
# 99th percentile prevents a tiny number of unusually large
# cells from compressing the useful color range.
# ==============================================================

vmin = 0.0


vmax = float(

    np.nanpercentile(

        ocean_std,

        99

    )

)


# --------------------------------------------------------------
# Round upward to nearest 0.1°C
# --------------------------------------------------------------

vmax = (

    np.ceil(
        vmax
        *
        10
    )

    /

    10

)


if vmax <= 0:

    vmax = 0.1


# ==============================================================
# 15. CONTOUR LEVELS
# ==============================================================

levels = np.linspace(

    vmin,

    vmax,

    16

)


# ==============================================================
# 16. COLORBAR TICKS
# ==============================================================

ticks = np.linspace(

    vmin,

    vmax,

    7

)


# ==============================================================
# 17. CREATE MAP
# ==============================================================

fig = plt.figure(

    figsize=(
        11,
        7
    )

)


ax = plt.axes(
    projection=projection
)


# ==============================================================
# 18. STANDARD DEVIATION MAP
# ==============================================================

cf = ax.contourf(

    LON,

    LAT,

    std_values,

    levels=levels,

    cmap="turbo",

    extend="max",

    transform=projection

)


# ==============================================================
# 19. MAP EXTENT
# ==============================================================

ax.set_extent(

    [
        LON_MIN,
        LON_MAX,
        LAT_MIN,
        LAT_MAX
    ],

    crs=projection

)


# ==============================================================
# 20. LAND
# ==============================================================

ax.add_feature(

    cfeature.LAND.with_scale(
        "50m"
    ),

    facecolor="0.70",

    edgecolor="black",

    linewidth=0.5,

    zorder=10

)


# ==============================================================
# 21. COASTLINES
# ==============================================================

ax.coastlines(

    resolution="50m",

    linewidth=0.8,

    zorder=11

)


# ==============================================================
# 22. BORDERS
# ==============================================================

ax.add_feature(

    cfeature.BORDERS.with_scale(
        "50m"
    ),

    linewidth=0.35,

    edgecolor="black",

    zorder=11

)


# ==============================================================
# 23. LONGITUDE / LATITUDE TICKS
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
    5,
    10,
    15,
    20,
    25,
    30

]


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


# ==============================================================
# 24. GRIDLINES
# ==============================================================

ax.gridlines(

    crs=projection,

    draw_labels=False,

    xlocs=longitude_ticks,

    ylocs=latitude_ticks,

    linewidth=0.4,

    linestyle="--",

    alpha=0.4

)


# ==============================================================
# 25. TITLE
# ==============================================================

ax.set_title(

    "Standard Deviation of SST in the "
    "Tropical North East Atlantic\n"
    "1981–2010 Climatology",

    fontsize=15,

    fontweight="bold",

    pad=15

)


# ==============================================================
# 26. COLORBAR
#
# Same height as map
# ==============================================================

cax = inset_axes(

    ax,

    width="3.2%",

    height="100%",

    loc="lower left",

    bbox_to_anchor=(
        1.035,
        0,
        1,
        1
    ),

    bbox_transform=ax.transAxes,

    borderpad=0

)


cbar = fig.colorbar(

    cf,

    cax=cax,

    orientation="vertical",

    ticks=ticks

)


cbar.set_label(

    "SST Standard Deviation (°C)",

    fontsize=11

)


# ==============================================================
# 27. ONE DECIMAL PLACE
# ==============================================================

cbar.ax.yaxis.set_major_formatter(

    mticker.FormatStrFormatter(
        "%.1f"
    )

)


cbar.update_ticks()


# ==============================================================
# 28. SPACING
# ==============================================================

plt.subplots_adjust(

    left=0.08,

    right=0.88,

    bottom=0.10,

    top=0.87

)


# ==============================================================
# 29. DISPLAY ONLY
# ==============================================================

plt.show()


# ==============================================================
# 30. CLOSE DATASET
# ==============================================================

ds.close()


print(
    "\nSST standard deviation map completed successfully."
)