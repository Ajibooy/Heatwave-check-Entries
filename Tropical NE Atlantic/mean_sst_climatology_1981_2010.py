# ==============================================================
# TROPICAL NORTH EAST ATLANTIC
# MEAN SST CLIMATOLOGY MAP
#
# Region:
#   0-30°N, 60-10°W
#
# Climatology period:
#   1981-2010
#
# Output:
#   Spatial map of mean SST climatology
#
# Colorbar:
#   Mean SST (°C)
#
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
# 2. PREPROCESS EACH OISST FILE
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
    # Check SST variable
    # ----------------------------------------------------------

    if "sst" not in ds.data_vars:

        raise KeyError(
            "Variable 'sst' was not found."
        )

    ds = ds[["sst"]]

    # ----------------------------------------------------------
    # Convert longitude 0-360 -> -180...180
    # ----------------------------------------------------------

    if float(ds.lon.max()) > 180:

        ds = ds.assign_coords(
            lon=((ds.lon + 180.0) % 360.0) - 180.0
        )

    ds = ds.sortby("lat")
    ds = ds.sortby("lon")

    # ----------------------------------------------------------
    # Select study region
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
    "TROPICAL NORTH EAST ATLANTIC "
    "MEAN SST CLIMATOLOGY"
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

# Remove duplicate dates
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
# 6. SELECT 1981-2010
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
    f"\nClimatology period: "
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
# 8. CALCULATE MEAN SST CLIMATOLOGY AT EVERY GRID CELL
# ==============================================================

print(
    "\nCalculating 1981-2010 mean SST climatology..."
)

mean_sst = (
    sst
    .mean(
        dim="time",
        skipna=True
    )
    .compute()
)


# ==============================================================
# 9. MASK LAND / INVALID CELLS
# ==============================================================

mean_sst_values = np.asarray(
    mean_sst.values,
    dtype=float
)

ocean_mask = np.isfinite(
    mean_sst_values
)

mean_sst_values = np.where(
    ocean_mask,
    mean_sst_values,
    np.nan
)


# ==============================================================
# 10. SUMMARY
# ==============================================================

ocean_values = mean_sst_values[
    ocean_mask
]


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
    f"Ocean grid cells: "
    f"{np.sum(ocean_mask):,}"
)

print(
    f"Regional minimum mean SST: "
    f"{np.nanmin(ocean_values):.2f} °C"
)

print(
    f"Regional average mean SST: "
    f"{np.nanmean(ocean_values):.2f} °C"
)

print(
    f"Regional maximum mean SST: "
    f"{np.nanmax(ocean_values):.2f} °C"
)


# ==============================================================
# 11. MAP COORDINATES
# ==============================================================

lon_values = mean_sst.lon.values

lat_values = mean_sst.lat.values


LON, LAT = np.meshgrid(
    lon_values,
    lat_values
)


projection = ccrs.PlateCarree()


# ==============================================================
# 12. COLORBAR RANGE
# ==============================================================

vmin = float(
    np.nanpercentile(
        ocean_values,
        1
    )
)

vmax = float(
    np.nanpercentile(
        ocean_values,
        99
    )
)


# Round to nearest 0.5°C
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
# 13. CREATE FIGURE
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
# 14. PLOT MEAN SST CLIMATOLOGY
# ==============================================================

cf = ax.contourf(
    LON,
    LAT,
    mean_sst_values,
    levels=levels,
    cmap="turbo",
    extend="both",
    transform=projection
)


# ==============================================================
# 15. MAP EXTENT
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
# 16. LAND
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
# 17. COASTLINES
# ==============================================================

ax.coastlines(
    resolution="50m",
    linewidth=0.8,
    zorder=11
)


# ==============================================================
# 18. BORDERS
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
# 19. AXIS TICKS
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
# 20. GRIDLINES
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
# 21. TITLE
# ==============================================================

ax.set_title(
    "Mean SST Climatology in the Tropical North East Atlantic\n"
    "1981–2010",
    fontsize=15,
    fontweight="bold",
    pad=15
)


# ==============================================================
# 22. COLORBAR
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
    "Mean SST (°C)",
    fontsize=11
)


cbar.ax.yaxis.set_major_formatter(
    mticker.FormatStrFormatter(
        "%.1f"
    )
)

cbar.update_ticks()


# ==============================================================
# 23. DISPLAY
# ==============================================================

plt.subplots_adjust(
    left=0.08,
    right=0.88,
    bottom=0.10,
    top=0.88
)

plt.show()


# ==============================================================
# 24. CLOSE
# ==============================================================

ds.close()


print(
    "\n1981-2010 mean SST climatology map completed successfully."
)