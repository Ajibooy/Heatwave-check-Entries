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
# ==========================
# SETTINGS
# ==========================
CLIM_START = 1981
CLIM_END = 2010
LAT_SLICE = slice(0.125, 29.875)
LON_SLICE = slice(300.125, 349.875)
MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]
# ==========================
# LOAD P90 THRESHOLD
# ==========================
print("Loading P90 threshold...")
p90_ds = xr.open_dataset(P90_FILE)
p90 = (
    p90_ds["p90_threshold"]
    .squeeze("zlev")
    .sel(
        lat=LAT_SLICE,
        lon=LON_SLICE
    )
)
# Regional mean P90 for each day of year
p90_regional_doy = (
    p90.mean(dim=["lat", "lon"], skipna=True)
       .load()
)
print("P90 loaded")
# ==========================
# COMPUTE DAILY MEAN SST CLIMATOLOGY
# ==========================
print("Computing daily mean SST climatology...")
clim_series = []
for year in range(CLIM_START, CLIM_END + 1):
    print(f"Processing {year}...")
    files = sorted(DATA_DIR.glob(f"*{year}*.nc"))
    if len(files) == 0:
        print(f"No files found for {year}")
        continue
    ds = xr.open_mfdataset(
        files,
        combine="by_coords"
    )
    sst_regional = (
        ds["sst"]
        .squeeze("zlev")
        .sel(
            lat=LAT_SLICE,
            lon=LON_SLICE
        )
        .mean(dim=["lat", "lon"], skipna=True)
        .load()
    )
    clim_series.append(sst_regional)
    ds.close()
sst_clim_all = xr.concat(
    clim_series,
    dim="time"
)
mean_sst_climatology_doy = (
    sst_clim_all
    .groupby("time.dayofyear")
    .mean("time", skipna=True)
)
print("Mean SST climatology computed")
# ==========================
# P90 - MEAN SST CLIMATOLOGY
# ==========================
delta_p90 = (
    p90_regional_doy - mean_sst_climatology_doy
)
delta_df = (
    delta_p90
    .to_dataframe(name="p90_minus_mean_sst")
    .reset_index()
)
# Remove leap day for clean monthly grouping
delta_df = delta_df[delta_df["dayofyear"] <= 365].copy()
# Use dummy non-leap year to assign months
delta_df["date"] = (
    pd.Timestamp("2001-01-01") +
    pd.to_timedelta(delta_df["dayofyear"] - 1, unit="D")
)
delta_df["month"] = delta_df["date"].dt.month
monthly_delta = (
    delta_df
    .groupby("month")["p90_minus_mean_sst"]
    .mean()
    .reset_index()
)
monthly_delta["month_name"] = [
    MONTH_NAMES[m - 1] for m in monthly_delta["month"]
]
print("\nMonthly P90 - Mean SST Difference")
print(monthly_delta)
# ==========================
# PLOT REGIONAL MONTHLY LINE GRAPH
# ==========================
plt.figure(figsize=(10, 5))
plt.plot(
    monthly_delta["month"],
    monthly_delta["p90_minus_mean_sst"],
    linewidth=2
)
plt.xlabel("Month")
plt.ylabel("P90 - Mean SST Climatology (°C)")
plt.title(
    "Monthly Mean Difference Between the Marine Heatwave Threshold (P90) and Mean SST Climatology (1981–2010)",
    fontsize=13,
    fontweight="bold"
)
plt.xticks(
    monthly_delta["month"],
    monthly_delta["month_name"]
)
plt.grid(alpha=0.35)
plt.tight_layout()
plt.show()
# ==========================
# CLOSE DATASET
# ==========================
p90_ds.close()