import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
# ==========================
# PATHS
# ==========================
DATA_DIR = Path(r"C:/Users/Aina Ajibola/Desktop/oisst_data")
P90_FILE = Path(
    r"C:/Users/Aina Ajibola/Desktop/P90_1981-2010/threshold.nc"
)
EVENT_CSV = Path("mhw_events_1981_2024.csv")
# ==========================
# LOAD EVENT WITH HIGHEST MAX INTENSITY
# ==========================
events = pd.read_csv(
    EVENT_CSV,
    parse_dates=["start_date", "end_date", "peak_date"]
)
event = events.loc[
    events["max_intensity"].idxmax()
]
lat = float(event["lat"])
lon = float(event["lon"])
lon_360 = lon % 360
event_start = event["start_date"]
event_end = event["end_date"]
plot_start = event_start - pd.DateOffset(months=6)
plot_end = event_end + pd.DateOffset(months=6)
plot_end = min(
    plot_end,
    pd.Timestamp("2024-12-31 12:00:00")
)
print("\nStrongest MHW Event by Maximum Intensity")
print(event)
print("\nPlot window")
print("Start:", plot_start)
print("End:", plot_end)
# ==========================
# LOAD SST FOR PLOT WINDOW
# ==========================
files = []
for year in range(plot_start.year, plot_end.year + 1):
    files.extend(
        sorted(DATA_DIR.glob(f"*{year}*.nc"))
    )
ds_event = xr.open_mfdataset(
    files,
    combine="by_coords"
)
sst = (
    ds_event["sst"]
    .squeeze("zlev")
    .sel(
        lat=lat,
        lon=lon_360,
        method="nearest"
    )
    .sel(
        time=slice(plot_start, plot_end)
    )
    .load()
)
# ==========================
# LOAD P90 THRESHOLD
# ==========================
p90_ds = xr.open_dataset(P90_FILE)
p90 = (
    p90_ds["p90_threshold"]
    .squeeze("zlev")
    .sel(
        lat=lat,
        lon=lon_360,
        method="nearest"
    )
)
doy = sst.time.dt.dayofyear
threshold = xr.DataArray(
    p90.sel(dayofyear=doy.values).values,
    coords={"time": sst.time},
    dims=["time"]
)
# ==========================
# DAILY CLIMATOLOGY
# YEAR-BY-YEAR FOR SPEED
# ==========================
clim_series = []
for year in range(1981, 2011):
    print(f"Loading climatology year {year}...")
    year_files = sorted(DATA_DIR.glob(f"*{year}*.nc"))
    if len(year_files) == 0:
        continue
    ds_y = xr.open_mfdataset(
        year_files,
        combine="by_coords"
    )
    sst_y = (
        ds_y["sst"]
        .squeeze("zlev")
        .sel(
            lat=lat,
            lon=lon_360,
            method="nearest"
        )
        .load()
    )
    clim_series.append(sst_y)
    ds_y.close()
sst_clim = xr.concat(
    clim_series,
    dim="time"
)
climatology_doy = (
    sst_clim
    .groupby("time.dayofyear")
    .mean("time", skipna=True)
)
climatology = xr.DataArray(
    climatology_doy.sel(dayofyear=doy.values).values,
    coords={"time": sst.time},
    dims=["time"]
)
# ==========================
# MHW MASK
# Shade only selected event
# ==========================
mhw_mask = (
    (sst > threshold) &
    (sst.time >= event_start) &
    (sst.time <= event_end)
)
time = pd.to_datetime(sst.time.values)
# ==========================
# PLOT
# ==========================
plt.figure(figsize=(14, 6))
plt.plot(
    time,
    sst.values,
    color="black",
    linewidth=1.4,
    label="Daily SST"
)
plt.plot(
    time,
    climatology.values,
    color="blue",
    linewidth=2,
    label="Seasonal climatology"
)
plt.plot(
    time,
    threshold.values,
    color="green",
    linewidth=2,
    label="P90 threshold"
)
plt.fill_between(
    time,
    threshold.values,
    sst.values,
    where=mhw_mask.values,
    color="red",
    alpha=0.7,
    label="Detected MHW event"
)
plt.xlabel("Year", fontsize=12)
plt.ylabel("SST (°C)", fontsize=12)
plt.title(
    "Strongest Marine Heatwave Event by Maximum Intensity",
    fontsize=15,
    fontweight="bold"
)
plt.legend(
    loc="lower left",
    fontsize=11,
    frameon=True,
    facecolor="white",
    edgecolor="gray"
)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=True)
# ==========================
# CLOSE FILES
# ==========================
ds_event.close()
p90_ds.close()