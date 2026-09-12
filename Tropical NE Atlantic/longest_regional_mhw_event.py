# ==============================================================
# LONGEST REGIONAL MARINE HEATWAVE EVENT
# TROPICAL NORTH EAST ATLANTIC
#
# Region:
#   0–30°N, 60–10°W
#
# Period:
#   1982–2024
#
# Baseline:
#   1981–2010
#
# Outputs:
#   (a) SST / climatology / P90 time series for longest MHW
#   (b) Mean SST anomaly map during that MHW
# ==============================================================

import os
import glob
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore")


# ==============================================================
# 1. PATHS
# ==============================================================

DATA_DIR = r"C:\Users\Aina Ajibola\Desktop\oisst_data"

THRESHOLD_FILE = (
    r"C:\Users\Aina Ajibola\Desktop\P90_1981-2010\threshold.nc"
)


# ==============================================================
# 2. SETTINGS
# ==============================================================

LAT_MIN = 0.0
LAT_MAX = 30.0

LON_MIN = -60.0
LON_MAX = -10.0

BASE_START = "1981-01-01"
BASE_END = "2010-12-31"

ANALYSIS_START = "1982-01-01"
ANALYSIS_END = "2024-12-31"

MIN_DURATION = 5

HALF_WINDOW = 5
SMOOTH_WINDOW = 31


# ==============================================================
# 3. HELPER FUNCTIONS
# ==============================================================

def preprocess_oisst(ds):

    rename_dict = {}

    for old, new in {
        "latitude": "lat",
        "longitude": "lon",
        "Latitude": "lat",
        "Longitude": "lon",
        "Time": "time",
        "TIME": "time"
    }.items():

        if old in ds.coords or old in ds.dims:
            rename_dict[old] = new

    if rename_dict:
        ds = ds.rename(rename_dict)

    for dim in ["zlev", "depth", "lev", "level"]:

        if dim in ds.dims and ds.sizes[dim] == 1:
            ds = ds.squeeze(dim, drop=True)

    ds = ds[["sst"]]

    if float(ds.lon.max()) > 180:

        ds = ds.assign_coords(
            lon=((ds.lon + 180) % 360) - 180
        )

    ds = ds.sortby("lat")
    ds = ds.sortby("lon")

    ds = ds.sel(
        lat=slice(LAT_MIN, LAT_MAX),
        lon=slice(LON_MIN, LON_MAX)
    )

    return ds


def get_clim_day(dates):

    dates = pd.DatetimeIndex(dates)

    ref = pd.to_datetime({
        "year": np.full(len(dates), 2000),
        "month": dates.month,
        "day": dates.day
    })

    return pd.DatetimeIndex(ref).dayofyear.to_numpy()


def circular_smooth(values, window=31):

    values = np.asarray(values, dtype=float)

    half = window // 2

    ext = np.concatenate([
        values[-half:],
        values,
        values[:half]
    ])

    output = np.full_like(values, np.nan)

    for i in range(len(values)):

        output[i] = np.nanmean(
            ext[i:i + window]
        )

    return output


def find_events(condition, min_duration=5):

    """
    Finds continuous True runs >= min_duration.
    Returns list of:
    (start_index, end_index, duration)
    """

    condition = np.asarray(condition, dtype=bool)

    events = []

    start = None

    for i, value in enumerate(condition):

        if value and start is None:
            start = i

        if start is not None:

            finished = (
                (not value)
                or
                (i == len(condition) - 1)
            )

            if finished:

                if value and i == len(condition) - 1:
                    end = i
                else:
                    end = i - 1

                duration = end - start + 1

                if duration >= min_duration:

                    events.append(
                        (start, end, duration)
                    )

                start = None

    return events


# ==============================================================
# 4. OPEN OISST
# ==============================================================

files = sorted(
    glob.glob(
        os.path.join(DATA_DIR, "*_oisst.nc")
    )
)

if not files:

    files = sorted(
        glob.glob(
            os.path.join(DATA_DIR, "*.nc")
        )
    )

print(f"Files found: {len(files):,}")


ds = xr.open_mfdataset(
    files,
    combine="by_coords",
    preprocess=preprocess_oisst,
    parallel=False,
    data_vars="minimal",
    coords="minimal",
    compat="override",
    join="outer",
    engine="netcdf4"
)


sst = ds["sst"].sortby("time")


# ==============================================================
# 5. NORMALIZE DATES
# ==============================================================

dates = pd.DatetimeIndex(
    sst.time.values
).normalize()

sst = sst.assign_coords(time=dates)

keep = np.where(
    ~dates.duplicated(keep="first")
)[0]

sst = sst.isel(time=keep)

dates = pd.DatetimeIndex(
    sst.time.values
)


# ==============================================================
# 6. CHECK SST UNITS
# ==============================================================

sample = float(
    sst.isel(time=0).mean(skipna=True).compute()
)

if sample > 100:
    sst = sst - 273.15


# ==============================================================
# 7. AREA-WEIGHTED REGIONAL SST
# ==============================================================

lat_weights = xr.DataArray(
    np.cos(np.deg2rad(sst.lat.values)),
    coords={"lat": sst.lat},
    dims=["lat"]
)


regional_sst = (
    sst
    .weighted(lat_weights)
    .mean(
        dim=["lat", "lon"],
        skipna=True
    )
    .compute()
)


# ==============================================================
# 8. BUILD REGIONAL DAILY CLIMATOLOGY
# ==============================================================

baseline = regional_sst.sel(
    time=slice(BASE_START, BASE_END)
)

baseline_dates = pd.DatetimeIndex(
    baseline.time.values
)

baseline_values = baseline.values

baseline_days = get_clim_day(
    baseline_dates
)


regional_climatology = np.full(
    366,
    np.nan
)


for day in range(1, 367):

    distance = np.abs(
        baseline_days - day
    )

    distance = np.minimum(
        distance,
        366 - distance
    )

    mask = distance <= HALF_WINDOW

    regional_climatology[day - 1] = (
        np.nanmean(
            baseline_values[mask]
        )
    )


regional_climatology = circular_smooth(
    regional_climatology,
    window=SMOOTH_WINDOW
)


# ==============================================================
# 9. OPEN P90 THRESHOLD
# ==============================================================

thr_ds = xr.open_dataset(
    THRESHOLD_FILE
)


rename_thr = {}

for old, new in {
    "latitude": "lat",
    "longitude": "lon"
}.items():

    if old in thr_ds.coords or old in thr_ds.dims:
        rename_thr[old] = new

if rename_thr:
    thr_ds = thr_ds.rename(rename_thr)


# ==============================================================
# 10. IDENTIFY P90 VARIABLE
# ==============================================================

candidates = []

for var in thr_ds.data_vars:

    name = var.lower()

    if any(
        x in name
        for x in [
            "threshold",
            "thresh",
            "p90",
            "percentile"
        ]
    ):
        candidates.append(var)


if candidates:

    threshold_var = candidates[0]

elif len(thr_ds.data_vars) == 1:

    threshold_var = list(
        thr_ds.data_vars
    )[0]

else:

    raise ValueError(
        "Could not identify threshold variable."
    )


threshold = thr_ds[
    threshold_var
].squeeze(drop=True)


# ==============================================================
# 11. STANDARDIZE P90 GRID
# ==============================================================

if float(threshold.lon.max()) > 180:

    threshold = threshold.assign_coords(
        lon=((threshold.lon + 180) % 360) - 180
    )


threshold = threshold.sortby("lat")
threshold = threshold.sortby("lon")

threshold = threshold.sel(
    lat=slice(LAT_MIN, LAT_MAX),
    lon=slice(LON_MIN, LON_MAX)
)


# ==============================================================
# 12. IDENTIFY P90 DAILY DIMENSION
# ==============================================================

day_dim = None

for dim in threshold.dims:

    if threshold.sizes[dim] in [365, 366]:

        day_dim = dim
        break


if day_dim is None:

    raise ValueError(
        "Could not identify threshold day dimension."
    )


threshold = threshold.transpose(
    day_dim,
    "lat",
    "lon"
)


# ==============================================================
# 13. AREA-WEIGHTED REGIONAL P90
# ==============================================================

regional_p90 = (
    threshold
    .weighted(lat_weights)
    .mean(
        dim=["lat", "lon"],
        skipna=True
    )
    .values
)


if len(regional_p90) != 366:

    raise ValueError(
        "Expected a 366-day P90 climatology."
    )


# ==============================================================
# 14. ANALYSIS PERIOD
# ==============================================================

regional_analysis = regional_sst.sel(
    time=slice(
        ANALYSIS_START,
        ANALYSIS_END
    )
)


analysis_dates = pd.DatetimeIndex(
    regional_analysis.time.values
)


analysis_sst = np.asarray(
    regional_analysis.values
)


analysis_clim_days = get_clim_day(
    analysis_dates
)


analysis_clim = regional_climatology[
    analysis_clim_days - 1
]


analysis_p90 = regional_p90[
    analysis_clim_days - 1
]


# ==============================================================
# 15. FIND ALL REGIONAL MHW EVENTS
# ==============================================================

above_p90 = (
    analysis_sst
    >
    analysis_p90
)


events = find_events(
    above_p90,
    min_duration=MIN_DURATION
)


if not events:

    raise RuntimeError(
        "No MHW events found."
    )


# ==============================================================
# 16. LONGEST EVENT
# ==============================================================

longest_event = max(
    events,
    key=lambda x: x[2]
)


start_idx, end_idx, duration = longest_event


event_start = analysis_dates[
    start_idx
]

event_end = analysis_dates[
    end_idx
]


print("\n" + "=" * 75)

print("LONGEST REGIONAL MHW EVENT")

print("=" * 75)

print(
    f"Start date : "
    f"{event_start.strftime('%d %B %Y')}"
)

print(
    f"End date   : "
    f"{event_end.strftime('%d %B %Y')}"
)

print(
    f"Duration   : "
    f"{duration} days"
)


# ==============================================================
# 17. EVENT INTENSITY
#
# Relative to climatological mean
# ==============================================================

event_sst = analysis_sst[
    start_idx:end_idx + 1
]

event_clim = analysis_clim[
    start_idx:end_idx + 1
]


event_intensity = (
    event_sst
    -
    event_clim
)


mean_intensity = np.nanmean(
    event_intensity
)

max_intensity = np.nanmax(
    event_intensity
)

cumulative_intensity = np.nansum(
    event_intensity
)


print(
    f"Mean intensity       : "
    f"{mean_intensity:.2f} °C"
)

print(
    f"Maximum intensity    : "
    f"{max_intensity:.2f} °C"
)

print(
    f"Cumulative intensity : "
    f"{cumulative_intensity:.2f} °C days"
)


# ==============================================================
# 18. TIME WINDOW FOR PANEL A
#
# Add 15 days before and after event
# ==============================================================

padding = 15


plot_start_idx = max(
    0,
    start_idx - padding
)

plot_end_idx = min(
    len(analysis_dates) - 1,
    end_idx + padding
)


ts_dates = analysis_dates[
    plot_start_idx:
    plot_end_idx + 1
]


ts_sst = analysis_sst[
    plot_start_idx:
    plot_end_idx + 1
]


ts_clim = analysis_clim[
    plot_start_idx:
    plot_end_idx + 1
]


ts_p90 = analysis_p90[
    plot_start_idx:
    plot_end_idx + 1
]


# ==============================================================
# 19. BUILD GRIDDED DAILY CLIMATOLOGY
#
# Needed for spatial anomaly map.
#
# We calculate only climatology days that occur
# during the longest event.
# ==============================================================

print(
    "\nCalculating event SST anomaly map..."
)


baseline_grid = sst.sel(
    time=slice(
        BASE_START,
        BASE_END
    )
)


baseline_grid_dates = pd.DatetimeIndex(
    baseline_grid.time.values
)

baseline_grid_days = get_clim_day(
    baseline_grid_dates
)


event_dates = analysis_dates[
    start_idx:
    end_idx + 1
]

event_days = get_clim_day(
    event_dates
)


unique_event_days = np.unique(
    event_days
)


# ==============================================================
# 20. DAILY CLIMATOLOGY FOR EACH EVENT DAY
# ==============================================================

clim_maps = {}


for day in unique_event_days:

    distance = np.abs(
        baseline_grid_days - day
    )

    distance = np.minimum(
        distance,
        366 - distance
    )

    selected_indices = np.where(
        distance <= HALF_WINDOW
    )[0]


    clim_map = (
        baseline_grid
        .isel(time=selected_indices)
        .mean(
            dim="time",
            skipna=True
        )
        .compute()
    )


    clim_maps[int(day)] = clim_map


# ==============================================================
# 21. EVENT SST ANOMALY
# ==============================================================

event_grid = sst.sel(
    time=slice(
        event_start,
        event_end
    )
)


daily_anomaly_maps = []


for i, date in enumerate(event_dates):

    clim_day = int(
        event_days[i]
    )


    daily_sst_map = (
        event_grid
        .sel(time=date)
    )


    anomaly = (
        daily_sst_map
        -
        clim_maps[clim_day]
    )


    daily_anomaly_maps.append(
        anomaly
    )


event_anomaly = xr.concat(
    daily_anomaly_maps,
    dim="event_time"
).mean(
    dim="event_time",
    skipna=True
)


event_anomaly = event_anomaly.compute()


# ==============================================================
# 22. FIGURE
# ==============================================================

fig = plt.figure(
    figsize=(13, 12)
)


gs = fig.add_gridspec(
    2,
    1,
    height_ratios=[1, 1.3],
    hspace=0.28
)


# ==============================================================
# PANEL A
# ==============================================================

ax1 = fig.add_subplot(
    gs[0, 0]
)


ax1.plot(
    ts_dates,
    ts_clim,
    linewidth=2,
    label="Climatology"
)


ax1.plot(
    ts_dates,
    ts_p90,
    linewidth=2,
    label="P90 threshold"
)


ax1.plot(
    ts_dates,
    ts_sst,
    linewidth=2,
    label="SST"
)


# --------------------------------------------------------------
# Shade event
# --------------------------------------------------------------

event_mask = (
    (ts_dates >= event_start)
    &
    (ts_dates <= event_end)
)


ax1.fill_between(
    ts_dates,
    ts_p90,
    ts_sst,
    where=(
        event_mask
        &
        (ts_sst > ts_p90)
    ),
    alpha=0.45,
    label="MHW event"
)


ax1.axvline(
    event_start,
    linestyle="--",
    linewidth=1
)

ax1.axvline(
    event_end,
    linestyle="--",
    linewidth=1
)


ax1.set_ylabel(
    "SST (°C)",
    fontweight="bold"
)


ax1.set_title(
    f"(a) Longest Regional MHW Event: "
    f"{duration} Days\n"
    f"{event_start.strftime('%d %b %Y')} – "
    f"{event_end.strftime('%d %b %Y')}",
    fontweight="bold"
)


ax1.legend(
    loc="best"
)


ax1.grid(
    alpha=0.2
)


ax1.xaxis.set_major_formatter(
    mdates.DateFormatter(
        "%d/%m/%Y"
    )
)


plt.setp(
    ax1.get_xticklabels(),
    rotation=30,
    ha="right"
)


# ==============================================================
# PANEL B
# ==============================================================

ax2 = fig.add_subplot(
    gs[1, 0],
    projection=ccrs.PlateCarree()
)


levels = np.arange(
    -2.0,
    2.01,
    0.2
)


cf = ax2.contourf(
    event_anomaly.lon,
    event_anomaly.lat,
    event_anomaly.values,
    levels=levels,
    cmap="RdBu_r",
    extend="both",
    transform=ccrs.PlateCarree()
)


# --------------------------------------------------------------
# Contour lines
# --------------------------------------------------------------

cs = ax2.contour(
    event_anomaly.lon,
    event_anomaly.lat,
    event_anomaly.values,
    levels=np.arange(
        -2,
        2.1,
        0.4
    ),
    linewidths=0.5,
    transform=ccrs.PlateCarree()
)


ax2.clabel(
    cs,
    inline=True,
    fontsize=7,
    fmt="%.1f"
)


# --------------------------------------------------------------
# Geography
# --------------------------------------------------------------

ax2.add_feature(
    cfeature.LAND,
    facecolor="0.75",
    zorder=3
)


ax2.add_feature(
    cfeature.COASTLINE,
    linewidth=0.7,
    zorder=4
)


ax2.add_feature(
    cfeature.BORDERS,
    linewidth=0.4,
    zorder=4
)


ax2.set_extent(
    [
        LON_MIN,
        LON_MAX,
        LAT_MIN,
        LAT_MAX
    ],
    crs=ccrs.PlateCarree()
)


gl = ax2.gridlines(
    draw_labels=True,
    linewidth=0.3,
    alpha=0.3
)

gl.top_labels = False
gl.right_labels = False


ax2.set_title(
    f"(b) Mean SST Anomaly During Longest MHW\n"
    f"{event_start.strftime('%d %b %Y')} – "
    f"{event_end.strftime('%d %b %Y')}",
    fontweight="bold"
)


# ==============================================================
# COLORBAR
# ==============================================================

cbar = fig.colorbar(
    cf,
    ax=ax2,
    orientation="vertical",
    pad=0.03,
    shrink=0.9
)


cbar.set_label(
    "SST Anomaly (°C)",
    fontweight="bold"
)


# ==============================================================
# OVERALL TITLE
# ==============================================================

fig.suptitle(
    "Longest Marine Heatwave Event in the Tropical North East Atlantic",
    fontsize=15,
    fontweight="bold",
    y=0.98
)


plt.show()


# ==============================================================
# 23. SUMMARY
# ==============================================================

print("\n" + "=" * 75)

print("EVENT SUMMARY")

print("=" * 75)

print(
    f"Longest event: "
    f"{event_start.strftime('%d %B %Y')} "
    f"to "
    f"{event_end.strftime('%d %B %Y')}"
)

print(
    f"Duration: "
    f"{duration} days"
)

print(
    f"Mean intensity: "
    f"{mean_intensity:.2f} °C"
)

print(
    f"Maximum intensity: "
    f"{max_intensity:.2f} °C"
)

print(
    f"Cumulative intensity: "
    f"{cumulative_intensity:.2f} °C days"
)

print("=" * 75)