# ==============================================================
# TROPICAL NORTH EAST ATLANTIC
# MEAN DAILY P90 CLIMATOLOGICAL THRESHOLD MAP
#
# Region:
#   0-30°N, 60-10°W
#
# Baseline:
#   1981-2010
#
# P90 method:
#   - Daily 90th percentile
#   - ±5-day climatological window
#   - 31-day circular smoothing
#
# Final map:
#   Mean of the 366 daily P90 values at every grid cell
#
# Colorbar:
#   Mean P90 (°C)
#
# No CSV
# No figure saved
# ==============================================================

import os
import glob
import warnings
import gc

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

BASE_START = "1981-01-01"
BASE_END = "2010-12-31"

PERCENTILE = 90

HALF_WINDOW = 5

SMOOTH_WINDOW = 31

LAT_BLOCK_SIZE = 10


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
            and
            ds.sizes[dim] == 1
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
# 3. LEAP-SAFE CLIMATOLOGICAL DAY INDEX
# ==============================================================

def get_clim_day(dates):

    dates = pd.DatetimeIndex(
        dates
    )

    reference_dates = pd.to_datetime(
        {
            "year": np.full(
                len(dates),
                2000
            ),

            "month": dates.month,

            "day": dates.day
        }
    )

    return (
        pd.DatetimeIndex(
            reference_dates
        )
        .dayofyear
        .to_numpy(
            dtype=np.int16
        )
    )


# ==============================================================
# 4. CIRCULAR SMOOTHING ALONG CLIMATOLOGICAL DAY
# ==============================================================

def circular_smooth_3d(
    values,
    window=31
):

    """
    values shape:
        (366, lat, lon)

    Performs circular moving average along day dimension.
    """

    half = (
        window // 2
    )

    extended = np.concatenate(
        [
            values[-half:, :, :],
            values,
            values[:half, :, :]
        ],
        axis=0
    )

    smoothed = np.full_like(
        values,
        np.nan,
        dtype=np.float32
    )

    # Use cumulative sum for speed
    finite = np.isfinite(
        extended
    )

    filled = np.where(
        finite,
        extended,
        0.0
    )

    csum = np.cumsum(
        filled,
        axis=0,
        dtype=np.float64
    )

    ccount = np.cumsum(
        finite.astype(
            np.int32
        ),
        axis=0
    )

    # pad cumulative arrays
    csum = np.concatenate(
        [
            np.zeros(
                (
                    1,
                    csum.shape[1],
                    csum.shape[2]
                ),
                dtype=np.float64
            ),
            csum
        ],
        axis=0
    )

    ccount = np.concatenate(
        [
            np.zeros(
                (
                    1,
                    ccount.shape[1],
                    ccount.shape[2]
                ),
                dtype=np.int32
            ),
            ccount
        ],
        axis=0
    )

    for d in range(
        366
    ):

        start = d
        end = d + window

        total = (
            csum[end]
            -
            csum[start]
        )

        count = (
            ccount[end]
            -
            ccount[start]
        )

        smoothed[d] = np.where(
            count > 0,
            total / count,
            np.nan
        )

    return smoothed


# ==============================================================
# 5. FIND OISST FILES
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


print("=" * 85)

print(
    "TROPICAL NORTH EAST ATLANTIC "
    "MEAN P90 CLIMATOLOGY"
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
# 6. OPEN OISST DATA
# ==============================================================

print(
    "\nOpening daily OISST..."
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
    engine="netcdf4"
)

ds = ds.sortby(
    "time"
)

sst = ds[
    "sst"
]


# ==============================================================
# 7. NORMALIZE TIME
# ==============================================================

time_index = pd.DatetimeIndex(
    sst.time.values
).normalize()

sst = sst.assign_coords(
    time=time_index
)

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
# 8. SELECT BASELINE
# ==============================================================

sst = sst.sel(
    time=slice(
        BASE_START,
        BASE_END
    )
)

dates = pd.DatetimeIndex(
    sst.time.values
)


print(
    f"\nActual baseline available: "
    f"{dates[0].date()} to "
    f"{dates[-1].date()}"
)

print(
    f"Daily observations: "
    f"{len(dates):,}"
)

print(
    f"Grid: "
    f"{sst.sizes['lat']} × "
    f"{sst.sizes['lon']}"
)


# ==============================================================
# 9. UNIT CHECK
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
# 10. CLIMATOLOGICAL DAY INDEX
# ==============================================================

clim_day = get_clim_day(
    dates
)


# ==============================================================
# 11. OUTPUT ARRAY
# ==============================================================

n_lat = sst.sizes[
    "lat"
]

n_lon = sst.sizes[
    "lon"
]


mean_p90 = np.full(
    (
        n_lat,
        n_lon
    ),
    np.nan,
    dtype=np.float32
)


# ==============================================================
# 12. PROCESS IN LATITUDE BLOCKS
# ==============================================================

total_blocks = int(
    np.ceil(
        n_lat
        /
        LAT_BLOCK_SIZE
    )
)


print(
    f"\nProcessing "
    f"{total_blocks} latitude blocks..."
)


for block_number, block_start in enumerate(
    range(
        0,
        n_lat,
        LAT_BLOCK_SIZE
    ),
    start=1
):

    block_end = min(
        block_start
        +
        LAT_BLOCK_SIZE,
        n_lat
    )


    print(
        f"\nBlock "
        f"{block_number}/{total_blocks} "
        f"| rows "
        f"{block_start + 1}-{block_end}"
    )


    # ==========================================================
    # LOAD SST BLOCK
    # ==========================================================

    sst_block = np.asarray(
        sst.isel(
            lat=slice(
                block_start,
                block_end
            )
        ).values,
        dtype=np.float32
    )


    block_lat = (
        block_end
        -
        block_start
    )


    # ==========================================================
    # DAILY P90 ARRAY
    # ==========================================================

    daily_p90 = np.full(
        (
            366,
            block_lat,
            n_lon
        ),
        np.nan,
        dtype=np.float32
    )


    # ==========================================================
    # CALCULATE P90 FOR EACH CLIMATOLOGICAL DAY
    # ==========================================================

    for day in range(
        1,
        367
    ):

        distance = np.abs(
            clim_day
            -
            day
        )

        distance = np.minimum(
            distance,
            366
            -
            distance
        )


        selected = (
            distance
            <=
            HALF_WINDOW
        )


        selected_sst = sst_block[
            selected,
            :,
            :
        ]


        daily_p90[
            day - 1,
            :,
            :
        ] = np.nanpercentile(
            selected_sst,
            PERCENTILE,
            axis=0
        )


    # ==========================================================
    # 31-DAY CIRCULAR SMOOTHING
    # ==========================================================

    print(
        "Applying 31-day smoothing..."
    )


    daily_p90_smoothed = circular_smooth_3d(
        daily_p90,
        SMOOTH_WINDOW
    )


    # ==========================================================
    # AVERAGE THE 366 DAILY P90 VALUES
    # ==========================================================

    block_mean_p90 = np.nanmean(
        daily_p90_smoothed,
        axis=0
    )


    mean_p90[
        block_start:block_end,
        :
    ] = block_mean_p90


    # ==========================================================
    # MEMORY CLEANUP
    # ==========================================================

    del sst_block
    del daily_p90
    del daily_p90_smoothed
    del block_mean_p90

    gc.collect()


    print(
        "Block completed."
    )


# ==============================================================
# 13. OCEAN MASK
# ==============================================================

ocean_mask = np.isfinite(
    mean_p90
)


ocean_p90 = mean_p90[
    ocean_mask
]


# ==============================================================
# 14. SUMMARY
# ==============================================================

print(
    "\n"
    +
    "=" * 85
)

print(
    "MEAN P90 RESULTS"
)

print(
    "=" * 85
)


print(
    f"Valid ocean grid cells: "
    f"{np.sum(ocean_mask):,}"
)

print(
    f"Minimum mean P90: "
    f"{np.nanmin(ocean_p90):.2f} °C"
)

print(
    f"Regional average P90: "
    f"{np.nanmean(ocean_p90):.2f} °C"
)

print(
    f"Maximum mean P90: "
    f"{np.nanmax(ocean_p90):.2f} °C"
)


# ==============================================================
# 15. MAP COORDINATES
# ==============================================================

lon_values = (
    sst.lon.values
)

lat_values = (
    sst.lat.values
)


LON, LAT = np.meshgrid(
    lon_values,
    lat_values
)


projection = ccrs.PlateCarree()


# ==============================================================
# 16. COLOR SCALE
# ==============================================================

vmin = float(
    np.nanpercentile(
        ocean_p90,
        1
    )
)

vmax = float(
    np.nanpercentile(
        ocean_p90,
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


levels = np.linspace(
    vmin,
    vmax,
    18
)


ticks = np.linspace(
    vmin,
    vmax,
    7
)


# ==============================================================
# 17. CREATE FIGURE
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
# 18. PLOT MEAN P90
# ==============================================================

cf = ax.contourf(
    LON,
    LAT,
    mean_p90,
    levels=levels,
    cmap="turbo",
    extend="both",
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
# 23. AXIS TICKS
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
    "Mean Daily P90 SST Threshold in the "
    "Tropical North East Atlantic\n"
    "1981–2010 Climatology",
    fontsize=15,
    fontweight="bold",
    pad=15
)


# ==============================================================
# 26. COLORBAR
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
    "Mean P90 (°C)",
    fontsize=11
)


cbar.ax.yaxis.set_major_formatter(
    mticker.FormatStrFormatter(
        "%.1f"
    )
)


cbar.update_ticks()


# ==============================================================
# 27. SPACING
# ==============================================================

plt.subplots_adjust(
    left=0.08,
    right=0.88,
    bottom=0.10,
    top=0.87
)


# ==============================================================
# 28. SHOW
# ==============================================================

plt.show()


# ==============================================================
# 29. CLOSE DATASET
# ==============================================================

ds.close()


print(
    "\nMean P90 climatology map completed successfully."
)