import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path(
    r"C:/Users/Aina Ajibola/Desktop/oisst_data"
)

P90_FILE = Path(
    r"C:/Users/Aina Ajibola/Desktop/P90_1981-2010/threshold.nc"
)


# ============================================================
# SETTINGS
# ============================================================

CLIM_START = 1981
CLIM_END = 2010

LAT_MIN = 0.125
LAT_MAX = 29.875

# OISST uses 0–360° longitude
LON_MIN = 300.125
LON_MAX = 349.875

MONTHS = np.arange(1, 13)

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def remove_zlev(data_array):
    """
    Remove the zlev dimension only when it exists.
    """
    if "zlev" in data_array.dims:
        data_array = data_array.isel(
            zlev=0,
            drop=True
        )

    return data_array


def regional_weighted_mean(data_array):
    """
    Compute a cosine-latitude weighted regional mean.
    """

    weights = np.cos(
        np.deg2rad(data_array["lat"])
    )

    return data_array.weighted(weights).mean(
        dim=["lat", "lon"],
        skipna=True
    )


# ============================================================
# 1. LOAD DAILY P90 CLIMATOLOGY
# ============================================================

print("Loading daily P90 threshold...")

with xr.open_dataset(P90_FILE) as p90_ds:

    if "p90_threshold" not in p90_ds.data_vars:
        raise KeyError(
            "Variable 'p90_threshold' was not found. "
            f"Available variables: {list(p90_ds.data_vars)}"
        )

    p90 = remove_zlev(
        p90_ds["p90_threshold"]
    )

    p90 = p90.sel(
        lat=slice(LAT_MIN, LAT_MAX),
        lon=slice(LON_MIN, LON_MAX)
    )

    p90_regional_daily = (
        regional_weighted_mean(p90)
        .load()
    )

print(
    "Number of P90 climatology days:",
    p90_regional_daily.sizes.get("dayofyear", 0)
)

if "dayofyear" not in p90_regional_daily.dims:
    raise ValueError(
        "The P90 variable must contain a "
        "'dayofyear' dimension."
    )


# ============================================================
# 2. CONVERT DAILY P90 TO MONTHLY CLIMATOLOGY
# ============================================================

p90_dataframe = (
    p90_regional_daily
    .to_dataframe(name="p90")
    .reset_index()
)

# Use the leap year 2000 so day 366 is retained
p90_dataframe["reference_date"] = (
    pd.Timestamp("2000-01-01")
    + pd.to_timedelta(
        p90_dataframe["dayofyear"] - 1,
        unit="D"
    )
)

p90_dataframe["month"] = (
    p90_dataframe["reference_date"].dt.month
)

monthly_p90 = (
    p90_dataframe
    .groupby("month")["p90"]
    .mean()
    .reindex(MONTHS)
)


# ============================================================
# 3. COMPUTE REGIONAL MONTHLY SST CLIMATOLOGY
# ============================================================

print(
    f"Computing monthly SST climatology "
    f"for {CLIM_START}–{CLIM_END}..."
)

monthly_sst_records = []

for year in range(CLIM_START, CLIM_END + 1):

    files = sorted(
        DATA_DIR.glob(f"*{year}*.nc")
    )

    print(
        f"{year}: {len(files)} SST files found"
    )

    if not files:
        continue

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=False,
        chunks={
            "time": 31
        }
    )

    try:
        if "sst" not in ds.data_vars:
            raise KeyError(
                f"'sst' not found for {year}. "
                f"Available variables: {list(ds.data_vars)}"
            )

        sst = remove_zlev(ds["sst"])

        sst = sst.sel(
            lat=slice(LAT_MIN, LAT_MAX),
            lon=slice(LON_MIN, LON_MAX)
        )

        # First reduce the spatial dimensions.
        regional_daily_sst = (
            regional_weighted_mean(sst)
        )

        # Then calculate monthly means for this year.
        regional_monthly_sst = (
            regional_daily_sst
            .resample(time="MS")
            .mean(skipna=True)
            .compute()
        )

        temp = (
            regional_monthly_sst
            .to_dataframe(name="mean_sst")
            .reset_index()
        )

        temp["year"] = temp["time"].dt.year
        temp["month"] = temp["time"].dt.month

        monthly_sst_records.append(
            temp[
                [
                    "time",
                    "year",
                    "month",
                    "mean_sst"
                ]
            ]
        )

    finally:
        ds.close()
        del ds
        gc.collect()


if not monthly_sst_records:
    raise RuntimeError(
        "No SST data were successfully processed."
    )

monthly_sst_all = pd.concat(
    monthly_sst_records,
    ignore_index=True
)

monthly_sst_climatology = (
    monthly_sst_all
    .groupby("month")["mean_sst"]
    .mean()
    .reindex(MONTHS)
)


# ============================================================
# 4. COMBINE ABSOLUTE VALUES AND DIFFERENCE
# ============================================================

monthly_results = pd.DataFrame(
    {
        "month": MONTHS,
        "month_name": MONTH_NAMES,
        "mean_sst_climatology": (
            monthly_sst_climatology.values
        ),
        "p90_threshold": monthly_p90.values
    }
)

monthly_results["p90_minus_mean_sst"] = (
    monthly_results["p90_threshold"]
    - monthly_results["mean_sst_climatology"]
)

print("\nMonthly climatological results:")
print(
    monthly_results.to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}"
    )
)


# ============================================================
# 5. VALIDATION
# ============================================================

if monthly_results[
    [
        "mean_sst_climatology",
        "p90_threshold",
        "p90_minus_mean_sst"
    ]
].isna().any().any():

    raise ValueError(
        "One or more monthly values are missing. "
        "Check the SST and P90 input files."
    )

if (
    monthly_results["p90_minus_mean_sst"] < 0
).any():

    print(
        "\nWARNING: The monthly P90 threshold is below "
        "the monthly mean SST in one or more months."
    )


# ============================================================
# 6. PLOT TWO-PANEL FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(12, 9),
    sharex=True,
    gridspec_kw={
        "height_ratios": [1.15, 1.0]
    }
)


# ------------------------------------------------------------
# PANEL A: ABSOLUTE SST AND P90
# ------------------------------------------------------------

axes[0].plot(
    monthly_results["month"],
    monthly_results["mean_sst_climatology"],
    linewidth=2.2,
    marker="o",
    label="Mean SST climatology"
)

axes[0].plot(
    monthly_results["month"],
    monthly_results["p90_threshold"],
    linewidth=2.2,
    marker="o",
    label="P90 threshold"
)

axes[0].set_ylabel(
    "Temperature (°C)"
)

axes[0].set_title(
    "Monthly Mean SST Climatology and P90 Threshold",
    fontsize=13,
    fontweight="bold"
)

axes[0].legend(
    loc="best",
    frameon=True
)

axes[0].grid(
    alpha=0.30
)

axes[0].text(
    -0.07,
    1.04,
    "A",
    transform=axes[0].transAxes,
    fontsize=14,
    fontweight="bold"
)


# ------------------------------------------------------------
# PANEL B: P90 MINUS MEAN SST
# ------------------------------------------------------------

axes[1].plot(
    monthly_results["month"],
    monthly_results["p90_minus_mean_sst"],
    linewidth=2.2,
    marker="o"
)

axes[1].set_xlabel(
    "Month"
)

axes[1].set_ylabel(
    "P90 − mean SST (°C)"
)

axes[1].set_title(
    "Monthly Thermal Offset of the P90 Threshold Above Mean SST",
    fontsize=13,
    fontweight="bold"
)

axes[1].grid(
    alpha=0.30
)

axes[1].text(
    -0.07,
    1.04,
    "B",
    transform=axes[1].transAxes,
    fontsize=14,
    fontweight="bold"
)


# ------------------------------------------------------------
# SHARED X-AXIS
# ------------------------------------------------------------

axes[1].set_xticks(MONTHS)
axes[1].set_xticklabels(MONTH_NAMES)


fig.suptitle(
    (
        "Regional Monthly SST Climatology and Marine "
        "Heatwave Threshold (1981–2010)"
    ),
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout(
    rect=[0, 0, 1, 0.96]
)

plt.show()