# ==============================================================
# TROPICAL NORTH EAST ATLANTIC
# SEASONAL MEAN SST AND LINEAR TRENDS
# 1982–2024
#
# Region:
#   0–30°N, 60–10°W
#
# Seasons:
#   DJF = December, January, February
#   MAM = March, April, May
#   JJA = June, July, August
#   SON = September, October, November
#
# Analysis:
#   - Linear SST trend for each season
#   - Trend in °C/decade
#   - 95% confidence interval
#   - p-value for each trend
#
# Figure:
#   - All four seasons in ONE plot
#   - Solid lines = seasonal mean SST
#   - Dashed lines = linear trends
#   - Season names directly beside corresponding trend lines
#   - Statistics box contains:
#         trend + 95% CI
#   - If all seasonal trends have p < 0.001,
#         p < 0.001 is stated only once
# ==============================================================
# 0. IMPORT PACKAGES
# ==============================================================

import os
import glob
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt

from scipy.stats import linregress
from scipy.stats import t as student_t


warnings.filterwarnings("ignore")


# ==============================================================
# 1. SETTINGS
# ==============================================================

DATA_DIR = r"C:\Users\Aina Ajibola\Desktop\oisst_data"


# --------------------------------------------------------------
# Study region
# --------------------------------------------------------------

LAT_MIN = 0.0
LAT_MAX = 30.0

LON_MIN = -60.0
LON_MAX = -10.0


# --------------------------------------------------------------
# Analysis years
# --------------------------------------------------------------

START_YEAR = 1982
END_YEAR = 2024


# --------------------------------------------------------------
# December 1981 is required for DJF 1982:
#
# DJF 1982 =
# December 1981 + January 1982 + February 1982
#
# December 2024 belongs to DJF 2025,
# therefore the analysis ends in November 2024.
# --------------------------------------------------------------

DATA_START_DATE = "1981-12-01"
DATA_END_DATE = "2024-11-30"


# ==============================================================
# 2. PREPROCESS EACH OISST FILE
# ==============================================================

def preprocess(ds):

    rename = {}

    coordinate_names = {

        "latitude": "lat",
        "Latitude": "lat",
        "LATITUDE": "lat",

        "longitude": "lon",
        "Longitude": "lon",
        "LONGITUDE": "lon",

        "Time": "time",
        "TIME": "time"

    }

    for old, new in coordinate_names.items():

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
            "Variable 'sst' was not found in the OISST file."
        )


    ds = ds[["sst"]]


    # ----------------------------------------------------------
    # Convert longitude from 0–360 to -180–180
    # ----------------------------------------------------------

    if float(ds.lon.max()) > 180:

        ds = ds.assign_coords(

            lon=(
                (ds.lon + 180.0) % 360.0
            ) - 180.0

        )


    # ----------------------------------------------------------
    # Sort coordinates
    # ----------------------------------------------------------

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


print("=" * 80)

print(
    "TROPICAL NORTH EAST ATLANTIC"
)

print(
    "SEASONAL SST TREND ANALYSIS"
)

print(
    "1982–2024"
)

print("=" * 80)


print(
    f"\nSST files found: {len(files):,}"
)

print(
    f"First file: {os.path.basename(files[0])}"
)

print(
    f"Last file: {os.path.basename(files[-1])}"
)


# ==============================================================
# 4. OPEN OISST DATA
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
#
# OISST timestamps may occur at noon.
# Normalize all timestamps to calendar dates.
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


sst = sst.sortby("time")


# ==============================================================
# 6. SELECT ANALYSIS PERIOD
# ==============================================================

sst = sst.sel(

    time=slice(
        DATA_START_DATE,
        DATA_END_DATE
    )

)


dates = pd.DatetimeIndex(
    sst.time.values
)


if len(dates) == 0:

    raise ValueError(

        "No SST observations were found "
        "during the requested period."

    )


print(
    f"\nData period used: "
    f"{dates[0].date()} to "
    f"{dates[-1].date()}"
)


print(
    f"Number of daily observations: "
    f"{len(dates):,}"
)


print(
    f"Regional grid: "
    f"{sst.sizes['lat']} × "
    f"{sst.sizes['lon']}"
)


# ==============================================================
# 7. SST UNIT CHECK
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
        "\nSST already appears to be in °C."
    )


# ==============================================================
# 8. COSINE-LATITUDE AREA WEIGHTS
#
# Weight = cos(latitude)
# ==============================================================

weights = xr.DataArray(

    np.cos(

        np.deg2rad(
            sst.lat.values
        )

    ),

    coords={
        "lat": sst.lat
    },

    dims=[
        "lat"
    ]

)


# ==============================================================
# 9. AREA-WEIGHTED REGIONAL DAILY SST
# ==============================================================

print(

    "\nCalculating area-weighted "
    "regional daily SST..."

)


regional_daily_sst = (

    sst

    .weighted(
        weights
    )

    .mean(

        dim=[
            "lat",
            "lon"
        ],

        skipna=True

    )

    .compute()

)


# ==============================================================
# 10. CONVERT TO DATAFRAME
# ==============================================================

df = pd.DataFrame({

    "date":
        pd.DatetimeIndex(
            regional_daily_sst.time.values
        ),

    "sst":
        np.asarray(
            regional_daily_sst.values,
            dtype=float
        )

})


df = df.dropna(
    subset=["sst"]
).copy()


# ==============================================================
# 11. ASSIGN SEASONS
# ==============================================================

month = df[
    "date"
].dt.month


conditions = [

    month.isin(
        [12, 1, 2]
    ),

    month.isin(
        [3, 4, 5]
    ),

    month.isin(
        [6, 7, 8]
    ),

    month.isin(
        [9, 10, 11]
    )

]


choices = [

    "DJF",
    "MAM",
    "JJA",
    "SON"

]


df["season"] = np.select(

    conditions,

    choices,

    default="Unknown"

)


# ==============================================================
# 12. ASSIGN SEASON YEAR
#
# December belongs to the following DJF year.
#
# Example:
#
# December 1981
# January 1982
# February 1982
#
# = DJF 1982
# ==============================================================

df["season_year"] = (
    df["date"].dt.year
)


december_mask = (

    df["date"].dt.month == 12

)


df.loc[

    december_mask,

    "season_year"

] = (

    df.loc[

        december_mask,

        "season_year"

    ]

    + 1

)


# ==============================================================
# 13. CHECK SEASON COMPLETENESS
# ==============================================================

df["month"] = (
    df["date"].dt.month
)


month_counts = (

    df

    .groupby(
        [
            "season_year",
            "season"
        ]
    )

    ["month"]

    .nunique()

    .reset_index(
        name="n_months"
    )

)


# ==============================================================
# 14. CALCULATE SEASONAL MEAN SST
# ==============================================================

seasonal = (

    df

    .groupby(
        [
            "season_year",
            "season"
        ],

        as_index=False

    )

    ["sst"]

    .mean()

)


# --------------------------------------------------------------
# Add month count
# --------------------------------------------------------------

seasonal = seasonal.merge(

    month_counts,

    on=[
        "season_year",
        "season"
    ],

    how="left"

)


# --------------------------------------------------------------
# Keep only complete three-month seasons
# --------------------------------------------------------------

seasonal = seasonal[

    seasonal["n_months"] == 3

].copy()


# --------------------------------------------------------------
# Keep only 1982–2024
# --------------------------------------------------------------

seasonal = seasonal[

    (
        seasonal["season_year"]
        >= START_YEAR
    )

    &

    (
        seasonal["season_year"]
        <= END_YEAR
    )

].copy()


# ==============================================================
# 15. SEASON ORDER
# ==============================================================

season_order = [

    "DJF",
    "MAM",
    "JJA",
    "SON"

]


# ==============================================================
# 16. CHECK DATA COVERAGE
# ==============================================================

print(
    "\n" + "=" * 80
)

print(
    "SEASONAL DATA COVERAGE"
)

print(
    "=" * 80
)


for season in season_order:

    temp = seasonal[

        seasonal["season"]
        == season

    ].copy()


    print(
        f"\n{season}:"
    )


    print(
        f"  Number of seasonal means: "
        f"{len(temp)}"
    )


    if len(temp) > 0:

        print(

            f"  Period: "
            f"{int(temp['season_year'].min())}"
            f"–"
            f"{int(temp['season_year'].max())}"

        )


# ==============================================================
# 17. FUNCTION TO CALCULATE LINEAR TREND
#
# Returns:
#
# - °C/year
# - °C/decade
# - 95% confidence interval
# - p-value
# - R²
# ==============================================================

def calculate_trend(data):

    data = data.dropna(

        subset=[
            "season_year",
            "sst"
        ]

    ).copy()


    x = data[
        "season_year"
    ].values.astype(float)


    y = data[
        "sst"
    ].values.astype(float)


    if len(x) < 3:

        raise ValueError(

            "At least 3 observations are required "
            "to calculate the trend."

        )


    # ----------------------------------------------------------
    # Linear regression
    # ----------------------------------------------------------

    regression = linregress(
        x,
        y
    )


    slope_year = (
        regression.slope
    )


    intercept = (
        regression.intercept
    )


    p_value = (
        regression.pvalue
    )


    stderr_year = (
        regression.stderr
    )


    r_squared = (
        regression.rvalue ** 2
    )


    # ----------------------------------------------------------
    # Convert °C/year to °C/decade
    # ----------------------------------------------------------

    slope_decade = (
        slope_year * 10.0
    )


    stderr_decade = (
        stderr_year * 10.0
    )


    # ----------------------------------------------------------
    # 95% confidence interval
    # ----------------------------------------------------------

    n = len(x)

    degrees_freedom = (
        n - 2
    )


    t_critical = student_t.ppf(

        0.975,

        degrees_freedom

    )


    ci_lower = (

        slope_decade

        -

        t_critical
        *
        stderr_decade

    )


    ci_upper = (

        slope_decade

        +

        t_critical
        *
        stderr_decade

    )


    return {

        "n":
            n,

        "slope_year":
            slope_year,

        "slope_decade":
            slope_decade,

        "intercept":
            intercept,

        "p_value":
            p_value,

        "stderr_decade":
            stderr_decade,

        "ci_lower":
            ci_lower,

        "ci_upper":
            ci_upper,

        "r_squared":
            r_squared

    }


# ==============================================================
# 18. CALCULATE TREND FOR EACH SEASON
# ==============================================================

trend_results = {}


print(
    "\n" + "=" * 80
)

print(
    "SEASONAL SST TREND RESULTS"
)

print(
    "=" * 80
)


for season in season_order:

    temp = seasonal[

        seasonal["season"]
        == season

    ].copy()


    temp = temp.sort_values(
        "season_year"
    )


    result = calculate_trend(
        temp
    )


    trend_results[
        season
    ] = result


    if result["p_value"] < 0.05:

        significance = (
            "STATISTICALLY SIGNIFICANT"
        )

    else:

        significance = (
            "NOT STATISTICALLY SIGNIFICANT"
        )


    print(
        f"\n{season}"
    )

    print(
        "-" * 60
    )


    print(

        f"Trend: "
        f"{result['slope_decade']:+.3f} "
        f"°C/decade"

    )


    print(

        f"95% CI: "
        f"{result['ci_lower']:+.3f} to "
        f"{result['ci_upper']:+.3f} "
        f"°C/decade"

    )


    print(

        f"p-value: "
        f"{result['p_value']:.8f}"

    )


    print(

        f"R²: "
        f"{result['r_squared']:.3f}"

    )


    print(

        f"Result: "
        f"{significance}"

    )


# ==============================================================
# 19. CREATE FIGURE
# ==============================================================

fig, ax = plt.subplots(

    figsize=(
        14,
        8
    )

)


# ==============================================================
# 20. STORE SEASON COLORS
# ==============================================================

season_colors = {}


# ==============================================================
# 21. PLOT ALL FOUR SEASONS
#
# Solid:
#   Observed seasonal mean SST
#
# Dashed:
#   Linear trend
# ==============================================================

for season in season_order:

    temp = seasonal[

        seasonal["season"]
        == season

    ].copy()


    temp = temp.sort_values(
        "season_year"
    )


    years = temp[
        "season_year"
    ].values.astype(float)


    sst_values = temp[
        "sst"
    ].values.astype(float)


    result = trend_results[
        season
    ]


    # ----------------------------------------------------------
    # Observed seasonal mean SST
    # ----------------------------------------------------------

    observed_line = ax.plot(

        years,

        sst_values,

        marker="o",

        markersize=3.5,

        linewidth=1.4,

        alpha=0.82

    )[0]


    # ----------------------------------------------------------
    # Save line color
    # ----------------------------------------------------------

    line_color = (
        observed_line.get_color()
    )


    season_colors[
        season
    ] = line_color


    # ----------------------------------------------------------
    # Fitted linear trend
    # ----------------------------------------------------------

    fitted_sst = (

        result[
            "intercept"
        ]

        +

        result[
            "slope_year"
        ]

        *
        years

    )


    ax.plot(

        years,

        fitted_sst,

        linestyle="--",

        linewidth=2.2,

        color=line_color,

        alpha=0.95

    )


# ==============================================================
# 22. DIRECT SEASON LABELS
#
# Labels are attached to the fitted trend lines rather than
# the final observed SST point.
# ==============================================================

LABEL_YEAR = (
    END_YEAR + 0.7
)


# --------------------------------------------------------------
# Small vertical offsets for visual separation
# ==============================================================

label_offsets = {

    "DJF": +0.08,

    "MAM": -0.08,

    "JJA": -0.03,

    "SON": +0.03

}


for season in season_order:

    result = trend_results[
        season
    ]


    # ----------------------------------------------------------
    # Fitted SST at final analysis year
    # ----------------------------------------------------------

    fitted_end = (

        result[
            "intercept"
        ]

        +

        result[
            "slope_year"
        ]

        *
        END_YEAR

    )


    # ----------------------------------------------------------
    # Add direct season label
    # ----------------------------------------------------------

    ax.text(

        LABEL_YEAR,

        fitted_end
        +
        label_offsets[
            season
        ],

        season,

        color=season_colors[
            season
        ],

        fontsize=11,

        fontweight="bold",

        verticalalignment="center",

        horizontalalignment="left",

        clip_on=False

    )


# ==============================================================
# 23. X-AXIS RANGE
#
# Extra space is included on the right for direct labels.
# ==============================================================

ax.set_xlim(

    START_YEAR - 1,

    END_YEAR + 3

)


# ==============================================================
# 24. Y-AXIS RANGE
# ==============================================================

all_sst = seasonal[
    "sst"
].values


y_min = (

    np.floor(
        np.nanmin(all_sst)
    )

    - 0.25

)


y_max = (

    np.ceil(
        np.nanmax(all_sst)
    )

    + 0.25

)


ax.set_ylim(

    y_min,

    y_max

)


# ==============================================================
# 25. YEAR TICKS
# ==============================================================

year_ticks = np.arange(

    START_YEAR,

    END_YEAR + 1,

    4

)


ax.set_xticks(
    year_ticks
)


# ==============================================================
# 26. AXIS LABELS
# ==============================================================

ax.set_xlabel(

    "Year",

    fontsize=12,

    fontweight="bold"

)


ax.set_ylabel(

    "Seasonal Mean SST (°C)",

    fontsize=12,

    fontweight="bold"

)


# ==============================================================
# 27. TITLE
# ==============================================================

ax.set_title(

    "Seasonal Mean SST and Linear Trends in the "
    "Tropical North East Atlantic (1982–2024)",

    fontsize=15,

    fontweight="bold",

    pad=16

)


# ==============================================================
# 28. GRID
# ==============================================================

ax.grid(

    True,

    linestyle="--",

    linewidth=0.5,

    alpha=0.30

)


# ==============================================================
# 29. CREATE CLEAN STATISTICS BOX
#
# Instead of repeating:
#
# p < 0.001
# p < 0.001
# p < 0.001
# p < 0.001
#
# the individual lines contain only:
#
# trend + 95% CI
#
# If all trends have p < 0.001, this is stated once.
# ==============================================================

statistics_lines = []


for season in season_order:

    result = trend_results[
        season
    ]


    line = (

        f"{season}: "
        f"{result['slope_decade']:+.2f} °C/decade "
        f"(95% CI: "
        f"{result['ci_lower']:+.2f} to "
        f"{result['ci_upper']:+.2f})"

    )


    statistics_lines.append(
        line
    )


statistics_text = "\n".join(
    statistics_lines
)


# ==============================================================
# 30. ADD P-VALUE INFORMATION ONCE
# ==============================================================

all_p_values = [

    trend_results[
        season
    ][
        "p_value"
    ]

    for season in season_order

]


# --------------------------------------------------------------
# Case 1:
# All seasonal trends have p < 0.001
# --------------------------------------------------------------

if all(

    p < 0.001
    for p in all_p_values

):

    statistics_text += (

        "\nAll seasonal trends: p < 0.001"

    )


# --------------------------------------------------------------
# Case 2:
# All are significant at p < 0.05,
# but not all reach p < 0.001
# --------------------------------------------------------------

elif all(

    p < 0.05
    for p in all_p_values

):

    statistics_text += (

        "\nAll seasonal trends: p < 0.05"

    )


# --------------------------------------------------------------
# Case 3:
# Mixed significance.
#
# In this situation, show the individual p-values because
# summarizing them with one statement would be misleading.
# --------------------------------------------------------------

else:

    statistics_text += (
        "\n"
    )


    for season in season_order:

        p = trend_results[
            season
        ][
            "p_value"
        ]


        if p < 0.001:

            p_text = (
                "p < 0.001"
            )

        else:

            p_text = (
                f"p = {p:.3f}"
            )


        statistics_text += (

            f"\n{season}: "
            f"{p_text}"

        )


# ==============================================================
# 31. ADD STATISTICS BOX TO FIGURE
# ==============================================================

ax.text(

    0.985,

    0.025,

    statistics_text,

    transform=ax.transAxes,

    fontsize=9.5,

    verticalalignment="bottom",

    horizontalalignment="right",

    bbox=dict(

        boxstyle="round,pad=0.55",

        facecolor="white",

        edgecolor="0.50",

        alpha=0.92

    )

)


# ==============================================================
# 32. TICK APPEARANCE
# ==============================================================

ax.tick_params(

    axis="both",

    labelsize=10

)


# ==============================================================
# 33. FIGURE LAYOUT
# ==============================================================

plt.subplots_adjust(

    left=0.08,

    right=0.94,

    bottom=0.10,

    top=0.90

)


# ==============================================================
# 34. DISPLAY
# ==============================================================

plt.show()


# ==============================================================
# 35. FINAL PRINTED SUMMARY
# ==============================================================

print(
    "\n" + "=" * 80
)

print(
    "FINAL SEASONAL SST TREND SUMMARY"
)

print(
    "=" * 80
)


for season in season_order:

    result = trend_results[
        season
    ]


    if result["p_value"] < 0.05:

        significance = (
            "SIGNIFICANT"
        )

    else:

        significance = (
            "NOT SIGNIFICANT"
        )


    print(
        f"\n{season}"
    )


    print(

        f"  Trend: "
        f"{result['slope_decade']:+.3f} "
        f"°C/decade"

    )


    print(

        f"  95% CI: "
        f"{result['ci_lower']:+.3f} to "
        f"{result['ci_upper']:+.3f} "
        f"°C/decade"

    )


    print(

        f"  p-value: "
        f"{result['p_value']:.8f}"

    )


    print(

        f"  R²: "
        f"{result['r_squared']:.3f}"

    )


    print(

        f"  Statistical result: "
        f"{significance}"

    )


# ==============================================================
# 36. SIMPLE INTERPRETATION
# ==============================================================

print(
    "\n" + "=" * 80
)

print(
    "INTERPRETATION"
)

print(
    "=" * 80
)


print(

    "\nThe trend for each season represents the "
    "change in regional seasonal mean SST per decade "
    "during 1982–2024."

)


print(

    "The 95% confidence interval represents the "
    "uncertainty around each estimated trend."

)


if all(

    p < 0.001
    for p in all_p_values

):

    print(

        "\nAll four seasonal SST trends are "
        "statistically significant (p < 0.001)."

    )


print(

    "\nThis analysis does not test whether the trends "
    "are statistically different from one season to another."

)


# ==============================================================
# 37. CLOSE DATASET
# ==============================================================

ds.close()


print(

    "\nSeasonal SST trend analysis completed successfully."

)
