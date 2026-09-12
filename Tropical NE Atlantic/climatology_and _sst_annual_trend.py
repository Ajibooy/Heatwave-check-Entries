# ==============================================================
# TROPICAL NORTH EAST ATLANTIC SST CLIMATOLOGY AND ANNUAL TREND
#
# Region:
#   0-30°N, 60-10°W
#
# Panel A:
#   Daily climatological mean SST, baseline 1981-2010
#
# Panel B:
#   Annual mean SST, 1982-2024
#   + linear trend
#
# NOTE:
# OISST begins on 1 September 1981.
# Therefore 1981 is used in the climatological baseline where
# observations exist, but is excluded from annual-mean Panel B.
#
# No CSV output
# No figures saved
# ==============================================================

import os
import glob
import traceback
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.stats import linregress

warnings.filterwarnings("ignore")


# ==============================================================
# 1. SETTINGS
# ==============================================================

DATA_DIR = r"C:\Users\Aina Ajibola\Desktop\oisst_data"

SST_VARIABLE = "sst"

# Study region
LAT_MIN = 0.0
LAT_MAX = 30.0

LON_MIN = -60.0
LON_MAX = -10.0

# Climatology baseline
CLIM_START = "1981-01-01"
CLIM_END = "2010-12-31"

# Full-year annual means
ANNUAL_START = "1982-01-01"
ANNUAL_END = "2024-12-31"


# ==============================================================
# 2. PREPROCESS EACH DAILY OISST FILE
# ==============================================================

def preprocess(ds):

    rename = {}

    if "latitude" in ds.coords:
        rename["latitude"] = "lat"

    if "longitude" in ds.coords:
        rename["longitude"] = "lon"

    if "Time" in ds.coords:
        rename["Time"] = "time"

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

    if SST_VARIABLE not in ds.data_vars:
        raise KeyError(
            f"Variable '{SST_VARIABLE}' was not found."
        )

    ds = ds[[SST_VARIABLE]]

    # ----------------------------------------------------------
    # Convert longitude from 0...360 to -180...180
    # ----------------------------------------------------------

    if float(ds.lon.max()) > 180:

        ds = ds.assign_coords(
            lon=((ds.lon + 180.0) % 360.0) - 180.0
        )

    ds = ds.sortby("lon")
    ds = ds.sortby("lat")

    # ----------------------------------------------------------
    # Subset immediately to study region
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
# 3. AREA-WEIGHTED REGIONAL MEAN
# ==============================================================

def area_weighted_mean(da):

    """
    Area-weighted mean for a regular latitude-longitude grid.

    Latitude weighting:
        cos(latitude)
    """

    weights = np.cos(
        np.deg2rad(
            da.lat
        )
    )

    weights.name = "latitude_weights"

    return da.weighted(
        weights
    ).mean(
        dim=[
            "lat",
            "lon"
        ],
        skipna=True
    )


# ==============================================================
# 4. LEAP-SAFE CLIMATOLOGICAL DAY
# ==============================================================

def climatological_day_366(dates):

    """
    Map every actual calendar date onto leap year 2000.

    This produces climatological-day numbers 1...366
    while retaining 29 February.
    """

    dates = pd.DatetimeIndex(
        dates
    )

    reference = pd.DatetimeIndex(
        pd.to_datetime(
            {
                "year": np.full(
                    len(dates),
                    2000
                ),
                "month": dates.month,
                "day": dates.day
            }
        )
    )

    return reference.dayofyear.to_numpy(
        dtype=np.int16
    )


# ==============================================================
# 5. MAIN
# ==============================================================

def main():

    ds = None

    try:

        # ======================================================
        # FIND OISST FILES
        # ======================================================

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

        print("=" * 80)
        print("TROPICAL NORTH EAST ATLANTIC SST ANALYSIS")
        print("=" * 80)

        print(
            f"Files found: "
            f"{len(files):,}"
        )

        print(
            f"First file: "
            f"{os.path.basename(files[0])}"
        )

        print(
            f"Last file:  "
            f"{os.path.basename(files[-1])}"
        )


        # ======================================================
        # OPEN DAILY OISST
        # ======================================================

        print(
            "\nOpening daily OISST files..."
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
            chunks={"time": 365}
        )

        ds = ds.sortby(
            "time"
        )

        sst = ds[
            SST_VARIABLE
        ]


        # ======================================================
        # NORMALIZE TIME
        # ======================================================

        normalized_time = (
            pd.DatetimeIndex(
                sst.time.values
            )
            .normalize()
        )

        sst = sst.assign_coords(
            time=normalized_time
        )


        # ------------------------------------------------------
        # Remove duplicate dates if present
        # ------------------------------------------------------

        time_index = pd.DatetimeIndex(
            sst.time.values
        )

        unique_indices = np.where(
            ~time_index.duplicated(
                keep="first"
            )
        )[0]

        sst = sst.isel(
            time=unique_indices
        )

        sst = sst.sortby(
            "time"
        )


        # ======================================================
        # CHECK TEMPERATURE UNITS
        # ======================================================

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
                "Converting SST from Kelvin to °C..."
            )

            sst = (
                sst
                - 273.15
            )

        else:

            print(
                "SST already appears to be in °C."
            )

        sst = sst.astype(
            np.float32
        )


        print(
            "\nDataset opened successfully."
        )

        print(
            f"Available period: "
            f"{pd.Timestamp(sst.time.values[0]).date()} "
            f"to "
            f"{pd.Timestamp(sst.time.values[-1]).date()}"
        )

        print(
            f"Regional grid: "
            f"{sst.sizes['lat']} × "
            f"{sst.sizes['lon']}"
        )


        # ======================================================
        # PANEL A DATA
        # DAILY CLIMATOLOGY 1981-2010
        # ======================================================

        print(
            "\nCalculating 1981-2010 daily climatological mean SST..."
        )

        sst_clim_period = sst.sel(
            time=slice(
                CLIM_START,
                CLIM_END
            )
        )


        # ------------------------------------------------------
        # First calculate area-weighted regional SST for each day
        # ------------------------------------------------------

        daily_regional_sst = area_weighted_mean(
            sst_clim_period
        ).compute()


        daily_dates = pd.DatetimeIndex(
            daily_regional_sst.time.values
        )


        # ------------------------------------------------------
        # Map dates to climatological day 1...366
        # ------------------------------------------------------

        clim_day_values = climatological_day_366(
            daily_dates
        )


        daily_regional_sst = (
            daily_regional_sst
            .assign_coords(
                clim_day=(
                    "time",
                    clim_day_values
                )
            )
        )


        # ------------------------------------------------------
        # Mean SST for each climatological day
        # ------------------------------------------------------

        daily_climatology = (
            daily_regional_sst
            .groupby(
                "clim_day"
            )
            .mean(
                dim="time",
                skipna=True
            )
        )


        # Ensure full 366-day calendar
        daily_climatology = (
            daily_climatology
            .reindex(
                clim_day=np.arange(
                    1,
                    367
                )
            )
        )


        daily_climatology_values = np.asarray(
            daily_climatology.values,
            dtype=float
        )


        # ------------------------------------------------------
        # Reference dates purely for Jan-Dec plotting
        # ------------------------------------------------------

        reference_dates = pd.date_range(
            "2000-01-01",
            "2000-12-31",
            freq="D"
        )


        # ======================================================
        # OVERALL CLIMATOLOGICAL MEAN SST
        # ======================================================

        climatological_mean_sst = float(
            np.nanmean(
                daily_climatology_values
            )
        )


        # ======================================================
        # PANEL B DATA
        # ANNUAL MEAN SST 1982-2024
        # ======================================================

        print(
            "Calculating annual mean SST, 1982-2024..."
        )

        sst_annual_period = sst.sel(
            time=slice(
                ANNUAL_START,
                ANNUAL_END
            )
        )


        # ------------------------------------------------------
        # Calculate annual mean SST fields
        # ------------------------------------------------------

        annual_sst_fields = (
            sst_annual_period
            .groupby(
                "time.year"
            )
            .mean(
                dim="time",
                skipna=True
            )
        )


        # ------------------------------------------------------
        # Area-weighted basin mean for every year
        # ------------------------------------------------------

        annual_mean_sst = (
            area_weighted_mean(
                annual_sst_fields
            )
            .compute()
        )


        years = np.asarray(
            annual_mean_sst.year.values,
            dtype=float
        )


        annual_values = np.asarray(
            annual_mean_sst.values,
            dtype=float
        )


        # ======================================================
        # LINEAR TREND
        # ======================================================

        valid = (
            np.isfinite(
                annual_values
            )
        )


        regression = linregress(
            years[
                valid
            ],
            annual_values[
                valid
            ]
        )


        trend_per_decade = (
            regression.slope
            * 10.0
        )


        fitted_values = (
            regression.intercept
            +
            regression.slope
            * years
        )


        annual_period_mean = float(
            np.nanmean(
                annual_values
            )
        )


        # ======================================================
        # TERMINAL RESULTS
        # ======================================================

        print(
            "\n"
            + "=" * 80
        )

        print(
            "CLIMATOLOGY RESULTS"
        )

        print(
            "=" * 80
        )

        print(
            f"Climatological baseline: "
            f"1981-2010"
        )

        print(
            f"Regional climatological mean SST: "
            f"{climatological_mean_sst:.2f} °C"
        )

        print(
            f"Coldest climatological day: "
            f"{np.nanmin(daily_climatology_values):.2f} °C"
        )

        print(
            f"Warmest climatological day: "
            f"{np.nanmax(daily_climatology_values):.2f} °C"
        )


        print(
            "\n"
            + "=" * 80
        )

        print(
            "ANNUAL SST RESULTS"
        )

        print(
            "=" * 80
        )

        print(
            f"Annual period: "
            f"1982-2024"
        )

        print(
            f"1982-2024 mean SST: "
            f"{annual_period_mean:.2f} °C"
        )

        print(
            f"1982 annual mean SST: "
            f"{annual_values[0]:.2f} °C"
        )

        print(
            f"2024 annual mean SST: "
            f"{annual_values[-1]:.2f} °C"
        )

        print(
            f"Linear trend: "
            f"{trend_per_decade:+.2f} °C/decade"
        )

        print(
            f"Trend p-value: "
            f"{regression.pvalue:.6g}"
        )

        print(
            f"R²: "
            f"{regression.rvalue ** 2:.3f}"
        )


        # ======================================================
        # CREATE TWO-PANEL FIGURE
        # ======================================================

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(
                14,
                11
            )
        )


        # ======================================================
        # PANEL A
        # DAILY CLIMATOLOGICAL MEAN SST
        # ======================================================

        ax1 = axes[
            0
        ]


        ax1.plot(
            reference_dates,
            daily_climatology_values,
            linewidth=2.0,
            label="Daily climatological mean SST"
        )


        # ------------------------------------------------------
        # Overall climatological mean
        # ------------------------------------------------------

        ax1.axhline(
            climatological_mean_sst,
            linestyle="--",
            linewidth=1.5,
            label=(
                f"1981–2010 mean = "
                f"{climatological_mean_sst:.2f} °C"
            )
        )


        ax1.set_title(
            "(a) Daily Climatological Mean SST (1981–2010)",
            fontsize=14,
            fontweight="bold",
            pad=12
        )


        ax1.set_ylabel(
            "Mean SST (°C)",
            fontsize=12,
            fontweight="bold"
        )


        ax1.set_xlabel(
            "Month",
            fontsize=12,
            fontweight="bold"
        )


        ax1.xaxis.set_major_locator(
            mdates.MonthLocator()
        )


        ax1.xaxis.set_major_formatter(
            mdates.DateFormatter(
                "%b"
            )
        )


        ax1.set_xlim(
            pd.Timestamp(
                "2000-01-01"
            ),
            pd.Timestamp(
                "2000-12-31"
            )
        )


        ax1.grid(
            alpha=0.3
        )


        ax1.legend(
            loc="best"
        )


        # ======================================================
        # PANEL B
        # ANNUAL MEAN SST 1982-2024
        # ======================================================

        ax2 = axes[
            1
        ]


        # ------------------------------------------------------
        # Annual SST line
        # NO circle markers
        # ------------------------------------------------------

        ax2.plot(
            years,
            annual_values,
            linewidth=1.8,
            label="Annual mean SST"
        )


        # ------------------------------------------------------
        # Linear trend line
        # ------------------------------------------------------

        ax2.plot(
            years,
            fitted_values,
            linestyle="--",
            linewidth=2.0,
            label=(
                f"Linear trend = "
                f"{trend_per_decade:+.2f} °C/decade"
            )
        )


        # ------------------------------------------------------
        # 1982-2024 mean reference
        # ------------------------------------------------------

        ax2.axhline(
            annual_period_mean,
            linestyle=":",
            linewidth=1.3,
            label=(
                f"1982–2024 mean = "
                f"{annual_period_mean:.2f} °C"
            )
        )


        ax2.set_title(
            "(b) Annual Mean SST (1982–2024)",
            fontsize=14,
            fontweight="bold",
            pad=12
        )


        ax2.set_xlabel(
            "Year",
            fontsize=12,
            fontweight="bold"
        )


        ax2.set_ylabel(
            "Mean SST (°C)",
            fontsize=12,
            fontweight="bold"
        )


        ax2.set_xticks(
            np.arange(
                1982,
                2025,
                2
            )
        )


        ax2.tick_params(
            axis="x",
            rotation=45
        )


        ax2.grid(
            alpha=0.3
        )


        ax2.legend(
            loc="best"
        )


        # ======================================================
        # OVERALL TITLE
        # ======================================================

        fig.suptitle(
            "Tropical North East Atlantic SST Climatology and Trend",
            fontsize=16,
            fontweight="bold",
            y=0.985
        )


        plt.tight_layout(
            rect=[
                0,
                0,
                1,
                0.965
            ]
        )


        plt.show()


        print(
            "\nAnalysis completed successfully."
        )


    except Exception:

        print(
            "\nPROCESS FAILED"
        )

        print(
            traceback.format_exc()
        )


    finally:

        if ds is not None:
            ds.close()


# ==============================================================
# RUN
# ==============================================================

if __name__ == "__main__":
    main()