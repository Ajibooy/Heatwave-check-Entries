# ==============================================================
# TROPICAL NORTH EAST ATLANTIC
# MEAN MHW DURATION AND MHW DURATION TREND
#
# Region:
#   0-30°N, 60-10°W
#
# P90 baseline:
#   1981-2010
#
# MHW analysis:
#   1982-2024
#
# MHW definition:
#   SST > daily P90 for >=5 consecutive days
#
# P90:
#   90th percentile
#   +/-5-day climatological window
#   31-day circular smoothing
#
# OUTPUT:
#
# (a) Mean MHW Duration
#     Units = Days/Event
#
# (b) MHW Duration Trend
#     Units = Days/Decade
#
# IMPORTANT:
#
# Annual mean duration =
# mean duration of MHW events that START in that year.
#
# If a year has no MHW events:
# annual mean duration = NaN, NOT zero.
#
# An event crossing Dec-Jan remains ONE event and is assigned
# to its START year.
# ==============================================================


import os
import glob
import gc
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors

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


# --------------------------------------------------------------
# P90 climatology
# --------------------------------------------------------------

BASE_START = "1981-01-01"
BASE_END = "2010-12-31"


# --------------------------------------------------------------
# MHW analysis
# --------------------------------------------------------------

START_YEAR = 1982
END_YEAR = 2024

ANALYSIS_END = "2024-12-31"


# --------------------------------------------------------------
# MHW parameters
# --------------------------------------------------------------

PERCENTILE = 90

HALF_WINDOW = 5

SMOOTH_WINDOW = 31

MIN_DURATION = 5


# --------------------------------------------------------------
# Memory control
# --------------------------------------------------------------

LAT_BLOCK_SIZE = 5


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
            and
            ds.sizes[dim] == 1
        ):

            ds = ds.squeeze(
                dim,
                drop=True
            )


    # ----------------------------------------------------------
    # SST variable
    # ----------------------------------------------------------

    if "sst" not in ds.data_vars:

        raise KeyError(
            "Variable 'sst' was not found."
        )


    ds = ds[["sst"]]


    # ----------------------------------------------------------
    # Longitude:
    # 0-360 -> -180...180
    # ----------------------------------------------------------

    if float(ds.lon.max()) > 180:

        ds = ds.assign_coords(
            lon=((ds.lon + 180.0) % 360.0) - 180.0
        )


    ds = ds.sortby("lat")
    ds = ds.sortby("lon")


    # ----------------------------------------------------------
    # Select Tropical North East Atlantic
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
# 3. CLIMATOLOGICAL DAY
# ==============================================================

def get_clim_day(dates):

    dates = pd.DatetimeIndex(
        dates
    )


    reference = pd.to_datetime(
        {
            "year":
                np.full(
                    len(dates),
                    2000
                ),

            "month":
                dates.month,

            "day":
                dates.day
        }
    )


    return (
        pd.DatetimeIndex(
            reference
        )
        .dayofyear
        .to_numpy(
            dtype=np.int16
        )
    )


# ==============================================================
# 4. CIRCULAR 31-DAY SMOOTHING
# ==============================================================

def circular_smooth_3d(
    values,
    window=31
):

    half = window // 2


    extended = np.concatenate(
        [
            values[-half:, :, :],
            values,
            values[:half, :, :]
        ],
        axis=0
    )


    valid = np.isfinite(
        extended
    )


    filled = np.where(
        valid,
        extended,
        0.0
    )


    csum = np.cumsum(
        filled,
        axis=0,
        dtype=np.float64
    )


    ccount = np.cumsum(
        valid.astype(np.int32),
        axis=0
    )


    # ----------------------------------------------------------
    # Add initial zero layer
    # ----------------------------------------------------------

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


    smoothed = np.full(
        values.shape,
        np.nan,
        dtype=np.float32
    )


    for d in range(366):

        start = d

        end = (
            d
            +
            window
        )


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


        np.divide(
            total,
            count,
            out=smoothed[d],
            where=(count > 0)
        )


    return smoothed


# ==============================================================
# 5. CALCULATE ANNUAL MEAN MHW DURATION
#
# Event duration = number of consecutive SST>P90 days
#
# Example:
#
# Event 1 = 7 days
# Event 2 = 10 days
# Event 3 = 16 days
#
# annual mean duration:
#
# (7 + 10 + 16) / 3 = 11 days/event
#
#
# IMPORTANT:
#
# Events crossing Dec-Jan remain ONE continuous event.
#
# Such an event is assigned to the year in which it STARTS.
#
# Example:
#
# Dec 27, 2019 - Jan 8, 2020
#
# duration = 13 days
#
# Assigned to 2019.
# ==============================================================

def calculate_annual_mean_duration(
    above,
    dates,
    years
):

    n_time, n_lat, n_lon = (
        above.shape
    )


    n_cells = (
        n_lat
        *
        n_lon
    )


    hot = above.reshape(
        n_time,
        n_cells
    )


    # ----------------------------------------------------------
    # Event duration total for each year/grid cell
    # ----------------------------------------------------------

    duration_sum = np.zeros(
        (
            len(years),
            n_cells
        ),
        dtype=np.float32
    )


    # ----------------------------------------------------------
    # Number of events in each year/grid cell
    # ----------------------------------------------------------

    event_count = np.zeros(
        (
            len(years),
            n_cells
        ),
        dtype=np.int16
    )


    # ----------------------------------------------------------
    # Loop through grid cells
    # ----------------------------------------------------------

    for cell in range(
        n_cells
    ):

        series = hot[
            :,
            cell
        ]


        start = None


        for t in range(
            n_time
        ):

            is_hot = bool(
                series[t]
            )


            # --------------------------------------------------
            # Start exceedance run
            # --------------------------------------------------

            if (
                is_hot
                and
                start is None
            ):

                start = t


            # --------------------------------------------------
            # Finish run
            # --------------------------------------------------

            if start is not None:

                run_finished = (

                    (not is_hot)

                    or

                    (
                        t
                        ==
                        n_time - 1
                    )

                )


                if run_finished:

                    if (
                        is_hot
                        and
                        t == n_time - 1
                    ):

                        end = t

                    else:

                        end = (
                            t - 1
                        )


                    duration = (
                        end
                        -
                        start
                        +
                        1
                    )


                    # ------------------------------------------
                    # Confirm MHW
                    # ------------------------------------------

                    if (
                        duration
                        >=
                        MIN_DURATION
                    ):

                        event_year = int(
                            dates[start].year
                        )


                        if (
                            START_YEAR
                            <=
                            event_year
                            <=
                            END_YEAR
                        ):

                            year_index = (
                                event_year
                                -
                                START_YEAR
                            )


                            duration_sum[
                                year_index,
                                cell
                            ] += duration


                            event_count[
                                year_index,
                                cell
                            ] += 1


                    start = None


    # ----------------------------------------------------------
    # Calculate annual mean duration
    #
    # No-event years stay NaN.
    # ----------------------------------------------------------

    annual_mean_duration = np.full(
        duration_sum.shape,
        np.nan,
        dtype=np.float32
    )


    np.divide(
        duration_sum,
        event_count,
        out=annual_mean_duration,
        where=(event_count > 0)
    )


    return (

        annual_mean_duration.reshape(
            len(years),
            n_lat,
            n_lon
        ),

        event_count.reshape(
            len(years),
            n_lat,
            n_lon
        )

    )


# ==============================================================
# 6. LINEAR TREND PER DECADE
#
# NaN years are ignored.
#
# Units:
# Days/Decade
# ==============================================================

def calculate_trend_per_decade(
    annual_values,
    years
):

    n_years, n_lat, n_lon = (
        annual_values.shape
    )


    trend = np.full(
        (
            n_lat,
            n_lon
        ),
        np.nan,
        dtype=np.float32
    )


    x = np.asarray(
        years,
        dtype=np.float64
    )


    for i in range(
        n_lat
    ):

        for j in range(
            n_lon
        ):

            y = annual_values[
                :,
                i,
                j
            ].astype(
                np.float64
            )


            valid = np.isfinite(
                y
            )


            # Require at least three years with events
            if np.sum(valid) < 3:

                continue


            xv = x[
                valid
            ]


            yv = y[
                valid
            ]


            xv_centered = (
                xv
                -
                np.mean(xv)
            )


            yv_centered = (
                yv
                -
                np.mean(yv)
            )


            denominator = np.sum(
                xv_centered ** 2
            )


            if denominator == 0:

                continue


            slope_per_year = (

                np.sum(
                    xv_centered
                    *
                    yv_centered
                )

                /

                denominator

            )


            trend[
                i,
                j
            ] = (
                slope_per_year
                *
                10.0
            )


    return trend


# ==============================================================
# 7. FIND FILES
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
    "=" * 90
)


print(
    "TROPICAL NORTH EAST ATLANTIC "
    "MHW DURATION AND TREND"
)


print(
    "=" * 90
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
# 8. OPEN OISST
# ==============================================================

print(
    "\nOpening OISST..."
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
# 9. NORMALIZE TIME
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
# 10. UNIT CHECK
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
# 11. BASELINE
# ==============================================================

baseline = sst.sel(

    time=slice(
        BASE_START,
        BASE_END
    )

)


baseline_dates = pd.DatetimeIndex(
    baseline.time.values
)


if len(
    baseline_dates
) == 0:

    raise ValueError(
        "No baseline observations found."
    )


print(
    f"\nBaseline actually available: "
    f"{baseline_dates[0].date()} to "
    f"{baseline_dates[-1].date()}"
)


if baseline_dates[0] > pd.Timestamp(
    BASE_START
):

    print(
        "\nWARNING:"
    )

    print(
        "Your OISST archive does not contain "
        "the complete 1981 climatological year."
    )


    print(
        f"Actual baseline begins "
        f"{baseline_dates[0].date()}."
    )


baseline_clim_day = get_clim_day(
    baseline_dates
)


# ==============================================================
# 12. ANALYSIS PERIOD
#
# Use all available pre-1982 observations internally,
# allowing events extending into January 1982 to remain
# continuous.
# ==============================================================

analysis_start = pd.Timestamp(
    sst.time.values[0]
)


analysis = sst.sel(

    time=slice(
        analysis_start,
        ANALYSIS_END
    )

)


analysis_dates = pd.DatetimeIndex(
    analysis.time.values
)


analysis_clim_day = get_clim_day(
    analysis_dates
)


years = np.arange(
    START_YEAR,
    END_YEAR + 1
)


n_years = len(
    years
)


print(
    f"\nDuration analysis: "
    f"{START_YEAR}-{END_YEAR}"
)


print(
    f"Number of analysis years: "
    f"{n_years}"
)


# ==============================================================
# 13. OUTPUT ARRAYS
# ==============================================================

n_lat = sst.sizes[
    "lat"
]


n_lon = sst.sizes[
    "lon"
]


annual_mean_duration = np.full(

    (
        n_years,
        n_lat,
        n_lon
    ),

    np.nan,

    dtype=np.float32

)


annual_event_count = np.zeros(

    (
        n_years,
        n_lat,
        n_lon
    ),

    dtype=np.int16

)


# ==============================================================
# 14. BLOCK PROCESSING
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


    block_lat = (
        block_end
        -
        block_start
    )


    print(
        "\n"
        + "=" * 72
    )


    print(
        f"BLOCK "
        f"{block_number}/{total_blocks}"
    )


    print(
        f"Latitude rows: "
        f"{block_start + 1}-{block_end}"
    )


    print(
        "=" * 72
    )


    # ==========================================================
    # 14A. BASELINE SST
    # ==========================================================

    print(
        "Loading baseline SST..."
    )


    base_block = np.asarray(

        baseline.isel(

            lat=slice(
                block_start,
                block_end
            )

        ).values,

        dtype=np.float32

    )


    # ==========================================================
    # 14B. DAILY P90
    # ==========================================================

    print(
        "Calculating daily P90..."
    )


    daily_p90 = np.full(

        (
            366,
            block_lat,
            n_lon
        ),

        np.nan,

        dtype=np.float32

    )


    for day in range(
        1,
        367
    ):

        distance = np.abs(
            baseline_clim_day
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


        selected_sst = base_block[
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
    # 14C. SMOOTH P90
    # ==========================================================

    print(
        "Applying 31-day circular smoothing..."
    )


    daily_p90 = circular_smooth_3d(

        daily_p90,

        SMOOTH_WINDOW

    )


    # ==========================================================
    # 14D. ANALYSIS SST
    # ==========================================================

    print(
        "Loading analysis SST..."
    )


    analysis_block = np.asarray(

        analysis.isel(

            lat=slice(
                block_start,
                block_end
            )

        ).values,

        dtype=np.float32

    )


    # ==========================================================
    # 14E. DAILY THRESHOLD FOR ACTUAL DATES
    # ==========================================================

    threshold_block = daily_p90[
        analysis_clim_day - 1,
        :,
        :
    ]


    # ==========================================================
    # 14F. SST > P90
    # ==========================================================

    valid = (

        np.isfinite(
            analysis_block
        )

        &

        np.isfinite(
            threshold_block
        )

    )


    above = (

        valid

        &

        (
            analysis_block
            >
            threshold_block
        )

    )


    # ==========================================================
    # 14G. EVENT DURATION
    # ==========================================================

    print(
        "Detecting MHW events and durations..."
    )


    (
        block_duration,
        block_event_count

    ) = calculate_annual_mean_duration(

        above,

        analysis_dates,

        years

    )


    annual_mean_duration[
        :,
        block_start:block_end,
        :
    ] = block_duration


    annual_event_count[
        :,
        block_start:block_end,
        :
    ] = block_event_count


    # ==========================================================
    # 14H. MEMORY CLEANUP
    # ==========================================================

    del base_block
    del daily_p90
    del analysis_block
    del threshold_block
    del valid
    del above
    del block_duration
    del block_event_count


    gc.collect()


    print(
        "Block completed."
    )


# ==============================================================
# 15. OCEAN MASK
# ==============================================================

print(
    "\nCreating ocean mask..."
)


ocean_mask = np.isfinite(

    sst.sel(

        time="2000-01-01",

        method="nearest"

    ).values

)


# ==============================================================
# 16. LAND -> NAN
# ==============================================================

annual_mean_duration[
    :,
    ~ocean_mask
] = np.nan


# ==============================================================
# 17. CLIMATOLOGICAL MEAN MHW DURATION
#
# IMPORTANT:
#
# Here we calculate the EVENT-WEIGHTED mean duration across
# the complete 1982-2024 period:
#
# total duration of all events /
# total number of events
#
# This is preferable to simply averaging annual means because
# years with only one event should not receive the same weight
# as years with many events.
#
# Units:
# Days/Event
# ==============================================================

print(
    "\nCalculating mean MHW duration..."
)


# --------------------------------------------------------------
# Annual total duration =
# annual mean duration × annual event count
# --------------------------------------------------------------

annual_duration_sum = (

    annual_mean_duration

    *

    annual_event_count

)


total_duration = np.nansum(
    annual_duration_sum,
    axis=0
)


total_events = np.sum(
    annual_event_count,
    axis=0
)


mean_mhw_duration = np.full(
    (
        n_lat,
        n_lon
    ),
    np.nan,
    dtype=np.float32
)


np.divide(

    total_duration,

    total_events,

    out=mean_mhw_duration,

    where=(
        total_events
        >
        0
    )

)


mean_mhw_duration[
    ~ocean_mask
] = np.nan


# ==============================================================
# 18. ANNUAL MEAN DURATION TREND
#
# Trend is fitted to annual mean event duration.
#
# No-event years are NaN and excluded.
#
# Units:
# Days/Decade
# ==============================================================

print(
    "Calculating MHW duration trend..."
)


duration_trend = calculate_trend_per_decade(

    annual_mean_duration,

    years

)


duration_trend[
    ~ocean_mask
] = np.nan


# ==============================================================
# 19. RESULTS
# ==============================================================

print(
    "\n"
    + "=" * 90
)


print(
    "MHW DURATION RESULTS"
)


print(
    "=" * 90
)


print(
    f"\nAnalysis period: "
    f"{START_YEAR}-{END_YEAR}"
)


print(
    f"Total analysis years: "
    f"{n_years}"
)


print(
    f"\nSpatial mean of mean MHW duration: "
    f"{np.nanmean(mean_mhw_duration):.1f} days/event"
)


print(
    f"Minimum mean MHW duration: "
    f"{np.nanmin(mean_mhw_duration):.1f} days/event"
)


print(
    f"Maximum mean MHW duration: "
    f"{np.nanmax(mean_mhw_duration):.1f} days/event"
)


print(
    f"\nSpatial mean duration trend: "
    f"{np.nanmean(duration_trend):+.1f} days/decade"
)


print(
    f"Minimum duration trend: "
    f"{np.nanmin(duration_trend):+.1f} days/decade"
)


print(
    f"Maximum duration trend: "
    f"{np.nanmax(duration_trend):+.1f} days/decade"
)


# ==============================================================
# 20. MAP COORDINATES
# ==============================================================

lon_values = sst.lon.values

lat_values = sst.lat.values


LON, LAT = np.meshgrid(
    lon_values,
    lat_values
)


projection = ccrs.PlateCarree()


# ==============================================================
# 21. COLORMAP
# ==============================================================

cmap = plt.get_cmap(
    "turbo"
)


# ==============================================================
# 22. MEAN DURATION COLOR SCALE
# ==============================================================

duration_min = float(
    MIN_DURATION
)


duration_max = float(

    np.nanpercentile(

        mean_mhw_duration,

        99

    )

)


duration_max = np.ceil(
    duration_max
)


if duration_max <= duration_min:

    duration_max = (
        duration_min
        +
        1
    )


duration_levels = np.linspace(

    duration_min,

    duration_max,

    16

)


duration_ticks = np.linspace(

    duration_min,

    duration_max,

    7

)


# ==============================================================
# 23. TREND SCALE
#
# Symmetric about zero.
# ==============================================================

trend_abs = float(

    np.nanpercentile(

        np.abs(
            duration_trend
        ),

        99

    )

)


# Round upward to nearest 0.5 day/decade

trend_abs = (

    np.ceil(
        trend_abs
        *
        2
    )

    /

    2

)


if trend_abs <= 0:

    trend_abs = 0.5


trend_levels = np.linspace(

    -trend_abs,

    trend_abs,

    17

)


trend_ticks = np.linspace(

    -trend_abs,

    trend_abs,

    7

)


trend_norm = mcolors.TwoSlopeNorm(

    vmin=-trend_abs,

    vcenter=0.0,

    vmax=trend_abs

)


# ==============================================================
# 24. FIGURE
# ==============================================================

fig, axes = plt.subplots(

    1,
    2,

    figsize=(
        16,
        6
    ),

    subplot_kw={
        "projection":
            projection
    }

)


# ==============================================================
# 25. PANEL A
# ==============================================================

cf1 = axes[0].contourf(

    LON,

    LAT,

    mean_mhw_duration,

    levels=duration_levels,

    cmap=cmap,

    extend="max",

    transform=projection

)


# ==============================================================
# 26. PANEL B
# ==============================================================

cf2 = axes[1].contourf(

    LON,

    LAT,

    duration_trend,

    levels=trend_levels,

    cmap=cmap,

    norm=trend_norm,

    extend="both",

    transform=projection

)


# ==============================================================
# 27. MAP FORMATTING
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


for ax in axes:


    ax.set_extent(

        [
            LON_MIN,
            LON_MAX,
            LAT_MIN,
            LAT_MAX
        ],

        crs=projection

    )


    ax.add_feature(

        cfeature.LAND.with_scale(
            "50m"
        ),

        facecolor="0.70",

        edgecolor="black",

        linewidth=0.45,

        zorder=10

    )


    ax.coastlines(

        resolution="50m",

        linewidth=0.7,

        zorder=11

    )


    ax.add_feature(

        cfeature.BORDERS.with_scale(
            "50m"
        ),

        linewidth=0.3,

        edgecolor="black",

        zorder=11

    )


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
        labelsize=9
    )


    ax.gridlines(

        crs=projection,

        draw_labels=False,

        xlocs=longitude_ticks,

        ylocs=latitude_ticks,

        linewidth=0.3,

        linestyle="--",

        alpha=0.35

    )


# ==============================================================
# 28. TITLES
# ==============================================================

axes[0].set_title(

    "(a) Mean MHW Duration",

    fontsize=13,

    fontweight="bold",

    pad=10

)


axes[1].set_title(

    "(b) MHW Duration Trend",

    fontsize=13,

    fontweight="bold",

    pad=10

)


fig.suptitle(

    "Tropical North East Atlantic Marine Heatwave "
    "Duration and Trend (1982–2024)",

    fontsize=16,

    fontweight="bold",

    y=0.97

)


# ==============================================================
# 29. SPACING
# ==============================================================

fig.subplots_adjust(

    left=0.05,

    right=0.91,

    bottom=0.10,

    top=0.86,

    wspace=0.25

)


# ==============================================================
# 30. MEAN DURATION COLORBAR
# ==============================================================

pos1 = axes[0].get_position()


cax1 = fig.add_axes(
    [
        pos1.x1 + 0.008,
        pos1.y0,
        0.013,
        pos1.height
    ]
)


cbar1 = fig.colorbar(

    cf1,

    cax=cax1,

    orientation="vertical",

    ticks=duration_ticks

)


cbar1.set_label(

    "Days/Event",

    fontsize=10,

    fontweight="bold"

)


cbar1.ax.yaxis.set_major_formatter(

    mticker.FormatStrFormatter(
        "%.1f"
    )

)


# ==============================================================
# 31. TREND COLORBAR
# ==============================================================

pos2 = axes[1].get_position()


cax2 = fig.add_axes(
    [
        pos2.x1 + 0.008,
        pos2.y0,
        0.013,
        pos2.height
    ]
)


cbar2 = fig.colorbar(

    cf2,

    cax=cax2,

    orientation="vertical",

    ticks=trend_ticks

)


cbar2.set_label(

    "Days/Decade",

    fontsize=10,

    fontweight="bold"

)


cbar2.ax.yaxis.set_major_formatter(

    mticker.FormatStrFormatter(
        "%.1f"
    )

)


# ==============================================================
# 32. SHOW
# ==============================================================

plt.show()


# ==============================================================
# 33. CLOSE
# ==============================================================

ds.close()


print(
    "\nMHW duration and duration trend analysis "
    "completed successfully."
)