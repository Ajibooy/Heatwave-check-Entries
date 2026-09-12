# ==============================================================
# TROPICAL NORTH EAST ATLANTIC
#
# REGIONAL ANNUAL TOTAL MHW DAYS
# AND RELATIONSHIP WITH ANNUAL MEAN SST
#
# Region:
#   0-30°N, 60-10°W
#
# Climatological baseline:
#   1981-2010
#
# Annual analysis:
#   1982-2024
#
# MHW definition:
#   SST > daily P90 for >= 5 consecutive days
#
# Daily P90:
#   +/-5-day climatological window
#   31-day circular smoothing
#
# PANEL (g):
#   Regional annual total MHW days
#   Units = Days
#
# PANEL (h):
#   Annual Mean SST vs Regional Total MHW Days
#   Units = Days
#
# Black line:
#   Best-fit linear regression
#
# Regional averaging:
#   cosine(latitude) area weighting
#
# IMPORTANT:
#   A confirmed 10-day MHW contributes 10 MHW days.
#   A 4-day SST>P90 run contributes 0 MHW days.
#
#   For an event crossing Dec-Jan, the event is detected as one
#   continuous event, but its MHW days are assigned to their
#   actual calendar year.
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

from scipy.stats import linregress

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
# Climatological baseline
# --------------------------------------------------------------

BASE_START = "1981-01-01"
BASE_END = "2010-12-31"


# --------------------------------------------------------------
# Annual analysis
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
    # Check SST
    # ----------------------------------------------------------

    if "sst" not in ds.data_vars:

        raise KeyError(
            "Variable 'sst' was not found."
        )


    ds = ds[["sst"]]


    # ----------------------------------------------------------
    # Convert longitude:
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
#
# Map every calendar date to leap year 2000.
#
# This prevents the leap-year shifting problem after Feb 28.
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
        pd.DatetimeIndex(reference)
        .dayofyear
        .to_numpy(dtype=np.int16)
    )


# ==============================================================
# 4. 31-DAY CIRCULAR SMOOTHING
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
    # Add zero layer
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


        np.divide(
            total,
            count,
            out=smoothed[d],
            where=(count > 0)
        )


    return smoothed


# ==============================================================
# 5. CALCULATE ANNUAL TOTAL MHW DAYS
#
# Example:
#
# Event 1 = 6 days
# Event 2 = 12 days
# Event 3 = 9 days
#
# Total MHW Days =
#
# 6 + 12 + 9 = 27 days
#
#
# IMPORTANT:
#
# Exceedance runs shorter than 5 days contribute ZERO.
#
# For events crossing Dec-Jan:
#
# Example:
# Dec 28-Jan 6 = 10-day confirmed MHW
#
# Previous year gets:
# Dec 28-31 = 4 MHW days
#
# New year gets:
# Jan 1-6 = 6 MHW days
#
# It is still ONE MHW event for detection purposes.
# ==============================================================

def calculate_annual_mhw_days(
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


    annual_days = np.zeros(
        (
            len(years),
            n_cells
        ),
        dtype=np.int16
    )


    date_years = np.asarray(
        dates.year
    )


    # ----------------------------------------------------------
    # Process every grid cell
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
            # Determine whether run finishes
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

                        end = t - 1


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

                    if duration >= MIN_DURATION:

                        event_years = date_years[
                            start:end + 1
                        ]


                        # --------------------------------------
                        # Count MHW days by actual year
                        # --------------------------------------

                        unique_years, counts = np.unique(

                            event_years,

                            return_counts=True

                        )


                        for event_year, count in zip(

                            unique_years,

                            counts

                        ):


                            if (
                                START_YEAR
                                <=
                                event_year
                                <=
                                END_YEAR
                            ):

                                year_index = (
                                    int(event_year)
                                    -
                                    START_YEAR
                                )


                                annual_days[
                                    year_index,
                                    cell
                                ] += int(count)


                    start = None


    return annual_days.reshape(
        len(years),
        n_lat,
        n_lon
    )


# ==============================================================
# 6. FIND FILES
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


print("=" * 90)

print(
    "REGIONAL TOTAL MHW DAYS AND SST ANALYSIS"
)

print("=" * 90)


print(
    f"\nFiles found: "
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
# 7. OPEN OISST
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
# 8. NORMALIZE OISST TIME
#
# ESSENTIAL:
# OISST timestamps may be at 12:00.
# Convert them to normalized calendar dates.
# ==============================================================

time_index = pd.DatetimeIndex(
    sst.time.values
).normalize()


sst = sst.assign_coords(
    time=time_index
)


# --------------------------------------------------------------
# Remove duplicate calendar dates if present
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
# 9. SST UNIT CHECK
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
# 10. CLIMATOLOGICAL BASELINE
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
        "Your archive does not contain the complete "
        "1981 climatological year."
    )


    print(
        f"Actual baseline begins "
        f"{baseline_dates[0].date()}."
    )


baseline_clim_day = get_clim_day(
    baseline_dates
)


# ==============================================================
# 11. INTERNAL ANALYSIS PERIOD
#
# Keep available 1981 data internally so events crossing
# Dec 1981-Jan 1982 can be identified as continuous events.
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
    f"\nAnnual analysis period: "
    f"{START_YEAR}-{END_YEAR}"
)


print(
    f"Number of annual values: "
    f"{n_years}"
)


# ==============================================================
# 12. GRID DIMENSIONS
# ==============================================================

n_lat = sst.sizes[
    "lat"
]


n_lon = sst.sizes[
    "lon"
]


# --------------------------------------------------------------
# Output
# --------------------------------------------------------------

annual_mhw_days = np.zeros(

    (
        n_years,
        n_lat,
        n_lon
    ),

    dtype=np.int16

)


# ==============================================================
# 13. PROCESS LATITUDE BLOCKS
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
    # 13A. LOAD BASELINE SST
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
    # 13B. DAILY P90
    # ==============================================================

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

        # ------------------------------------------------------
        # Circular climatological-day distance
        # ------------------------------------------------------

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


        # +/- 5 days
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
    # 13C. 31-DAY CIRCULAR SMOOTHING
    # ==============================================================

    print(
        "Applying 31-day circular smoothing..."
    )


    daily_p90 = circular_smooth_3d(

        daily_p90,

        SMOOTH_WINDOW

    )


    # ==========================================================
    # 13D. LOAD ANALYSIS SST
    # ==============================================================

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
    # 13E. MATCH P90 TO ACTUAL CALENDAR DATES
    # ==============================================================

    threshold_block = daily_p90[
        analysis_clim_day - 1,
        :,
        :
    ]


    # ==========================================================
    # 13F. VALID SST/P90
    # ==============================================================

    valid = (

        np.isfinite(
            analysis_block
        )

        &

        np.isfinite(
            threshold_block
        )

    )


    # ==========================================================
    # 13G. SST > P90
    # ==============================================================

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
    # 13H. DETECT CONFIRMED MHW DAYS
    # ==============================================================

    print(
        "Detecting confirmed MHW events and total days..."
    )


    block_mhw_days = calculate_annual_mhw_days(

        above,

        analysis_dates,

        years

    )


    annual_mhw_days[
        :,
        block_start:block_end,
        :
    ] = block_mhw_days


    # ==========================================================
    # CLEAN MEMORY
    # ==============================================================

    del base_block
    del daily_p90
    del analysis_block
    del threshold_block
    del valid
    del above
    del block_mhw_days

    gc.collect()


    print(
        "Block completed."
    )


# ==============================================================
# 14. OCEAN MASK
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


# --------------------------------------------------------------
# Convert to float so land can be NaN
# --------------------------------------------------------------

annual_mhw_days_float = (
    annual_mhw_days.astype(
        np.float32
    )
)


annual_mhw_days_float[
    :,
    ~ocean_mask
] = np.nan


# ==============================================================
# 15. AREA WEIGHTS
#
# For a regular latitude-longitude grid:
#
# area is proportional to cos(latitude)
# ==============================================================

latitudes = sst.lat.values


weights_1d = np.cos(

    np.deg2rad(
        latitudes
    )

)


weights_2d = np.broadcast_to(

    weights_1d[:, None],

    (
        n_lat,
        n_lon
    )

).astype(
    np.float64
)


# --------------------------------------------------------------
# Remove land weights
# --------------------------------------------------------------

weights_2d = np.where(

    ocean_mask,

    weights_2d,

    np.nan

)


# ==============================================================
# 16. REGIONAL ANNUAL TOTAL MHW DAYS
#
# Area-weighted spatial average of annual total MHW days.
#
# IMPORTANT:
#
# A value such as 40 days means the area-weighted average
# ocean grid cell in the region experienced ~40 MHW days
# during that year.
#
# Units = Days
# ==============================================================

print(
    "\nCalculating regional annual total MHW days..."
)


regional_annual_mhw_days = np.full(

    n_years,

    np.nan,

    dtype=np.float64

)


for yi in range(
    n_years
):


    field = annual_mhw_days_float[
        yi
    ]


    valid = (

        np.isfinite(
            field
        )

        &

        np.isfinite(
            weights_2d
        )

    )


    if np.any(
        valid
    ):

        regional_annual_mhw_days[
            yi
        ] = (

            np.sum(

                field[valid]

                *

                weights_2d[valid]

            )

            /

            np.sum(
                weights_2d[valid]
            )

        )


# ==============================================================
# 17. REGIONAL ANNUAL MEAN SST
#
# 1. Calculate annual mean SST at each grid cell.
#
# 2. Area-weight across all ocean grid cells.
#
# Units = °C
# ==============================================================

print(
    "\nCalculating regional annual mean SST..."
)


regional_annual_sst = np.full(

    n_years,

    np.nan,

    dtype=np.float64

)


for yi, year in enumerate(
    years
):


    print(
        f"Annual SST: {year}"
    )


    year_sst = sst.sel(

        time=slice(
            f"{year}-01-01",
            f"{year}-12-31"
        )

    )


    # ----------------------------------------------------------
    # Annual mean field
    # ----------------------------------------------------------

    annual_sst_field = np.asarray(

        year_sst.mean(

            dim="time",

            skipna=True

        )
        .compute()
        .values,

        dtype=np.float64

    )


    # ----------------------------------------------------------
    # Regional weighted mean
    # ----------------------------------------------------------

    valid = (

        np.isfinite(
            annual_sst_field
        )

        &

        np.isfinite(
            weights_2d
        )

    )


    regional_annual_sst[
        yi
    ] = (

        np.sum(

            annual_sst_field[valid]

            *

            weights_2d[valid]

        )

        /

        np.sum(
            weights_2d[valid]
        )

    )


# ==============================================================
# 18. LINEAR REGRESSION
#
# X = Regional Annual Mean SST
#
# Y = Regional Annual Total MHW Days
# ==============================================================

valid_regression = (

    np.isfinite(
        regional_annual_sst
    )

    &

    np.isfinite(
        regional_annual_mhw_days
    )

)


x = regional_annual_sst[
    valid_regression
]


y = regional_annual_mhw_days[
    valid_regression
]


regression = linregress(
    x,
    y
)


slope = regression.slope

intercept = regression.intercept

r_value = regression.rvalue

p_value = regression.pvalue


r_squared = (
    r_value ** 2
)


# ==============================================================
# 19. BEST-FIT REGRESSION LINE
# ==============================================================

x_line = np.linspace(

    np.nanmin(x),

    np.nanmax(x),

    200

)


y_line = (

    slope
    *
    x_line

    +

    intercept

)


# ==============================================================
# 20. PRINT RESULTS
# ==============================================================

print(
    "\n"
    + "=" * 90
)


print(
    "REGIONAL ANNUAL TOTAL MHW DAYS"
)


print(
    "=" * 90
)


for year, mhw_days, temp in zip(

    years,

    regional_annual_mhw_days,

    regional_annual_sst

):


    print(

        f"{year}:  "

        f"Total MHW Days = "
        f"{mhw_days:.2f} days   "

        f"Annual Mean SST = "
        f"{temp:.2f} °C"

    )


print(
    "\n"
    + "=" * 90
)


print(
    "SST–TOTAL MHW DAYS REGRESSION"
)


print(
    "=" * 90
)


print(
    f"Slope = "
    f"{slope:.3f} days °C⁻¹"
)


print(
    f"Intercept = "
    f"{intercept:.3f}"
)


print(
    f"R = "
    f"{r_value:.3f}"
)


print(
    f"R² = "
    f"{r_squared:.3f}"
)


print(
    f"p-value = "
    f"{p_value:.6f}"
)


# ==============================================================
# 21. CREATE FIGURE
# ==============================================================

fig, axes = plt.subplots(

    1,
    2,

    figsize=(
        14,
        5.8
    )

)


# ==============================================================
# 22. PANEL G
#
# REGIONAL ANNUAL TOTAL MHW DAYS
# ==============================================================

ax = axes[0]


ax.bar(

    years,

    regional_annual_mhw_days,

    width=0.72,

    edgecolor="black",

    linewidth=0.35

)


# --------------------------------------------------------------
# Labels
# --------------------------------------------------------------

ax.set_xlabel(

    "Year",

    fontsize=11,

    fontweight="bold"

)


ax.set_ylabel(

    "Total MHW Days [Days]",

    fontsize=11,

    fontweight="bold"

)


ax.set_title(

    "(g) MHW Days",

    fontsize=12,

    fontweight="bold",

    loc="left"

)


# --------------------------------------------------------------
# X axis
# --------------------------------------------------------------

ax.set_xlim(

    START_YEAR - 1,

    END_YEAR + 1

)


xticks = np.arange(

    START_YEAR,

    END_YEAR + 1,

    2

)


ax.set_xticks(
    xticks
)


ax.set_xticklabels(

    xticks,

    rotation=90,

    fontsize=8

)


# --------------------------------------------------------------
# MHW days cannot be negative
# --------------------------------------------------------------

ax.set_ylim(
    bottom=0
)


# --------------------------------------------------------------
# Grid
# --------------------------------------------------------------

ax.grid(

    True,

    linestyle="-",

    linewidth=0.4,

    alpha=0.35

)


ax.set_axisbelow(
    True
)


# ==============================================================
# 23. PANEL H
#
# SST vs TOTAL MHW DAYS
# ==============================================================

ax = axes[1]


# --------------------------------------------------------------
# Scatter
# --------------------------------------------------------------

ax.scatter(

    regional_annual_sst,

    regional_annual_mhw_days,

    s=32,

    zorder=3

)


# --------------------------------------------------------------
# Best-fit linear regression
# --------------------------------------------------------------

ax.plot(

    x_line,

    y_line,

    color="black",

    linewidth=1.5,

    zorder=2

)


# --------------------------------------------------------------
# Axis labels
# --------------------------------------------------------------

ax.set_xlabel(

    "Annual Mean SST [°C]",

    fontsize=11,

    fontweight="bold"

)


ax.set_ylabel(

    "Total MHW Days [Days]",

    fontsize=11,

    fontweight="bold"

)


# --------------------------------------------------------------
# Panel title
# --------------------------------------------------------------

ax.set_title(

    "(h) MHW Days",

    fontsize=12,

    fontweight="bold",

    loc="left"

)


# --------------------------------------------------------------
# Grid
# --------------------------------------------------------------

ax.grid(

    True,

    linestyle="-",

    linewidth=0.4,

    alpha=0.35

)


ax.set_axisbelow(
    True
)


# ==============================================================
# 24. R-SQUARED
#
# Bottom-right like the reference figure.
# ==============================================================

ax.text(

    0.97,

    0.06,

    f"R²={r_squared:.2f}",

    transform=ax.transAxes,

    horizontalalignment="right",

    verticalalignment="bottom",

    fontsize=10,

    fontweight="bold"

)


# ==============================================================
# 25. CONSISTENT APPEARANCE
# ==============================================================

for ax in axes:


    ax.tick_params(

        direction="out",

        width=0.8

    )


    for spine in ax.spines.values():

        spine.set_linewidth(
            0.8
        )


# ==============================================================
# 26. MAIN TITLE
# ==============================================================

fig.suptitle(

    "Regional Annual Marine Heatwave Days "
    "in the Tropical North East Atlantic (1982–2024)",

    fontsize=14,

    fontweight="bold",

    y=0.99

)


# ==============================================================
# 27. LAYOUT
# ==============================================================

plt.tight_layout(

    rect=[
        0,
        0,
        1,
        0.95
    ]

)


# ==============================================================
# 28. SHOW
# ==============================================================

plt.show()


# ==============================================================
# 29. CLOSE
# ==============================================================

ds.close()


print(
    "\nRegional total MHW days–SST analysis "
    "completed successfully."
)