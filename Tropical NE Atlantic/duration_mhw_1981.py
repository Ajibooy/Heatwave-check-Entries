import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import label

# Paths
sst_path = r"C:/Users/Aina Ajibola/Desktop/oisst_data/1981*_oisst.nc"
p90_path = r"C:/Users/Aina Ajibola/Desktop/P90_1981-2010/threshold.nc"

# Load datasets
sst_ds = xr.open_mfdataset(sst_path, combine="by_coords")
p90_ds = xr.open_dataset(p90_path)

# Extract variables
sst = sst_ds["sst"].squeeze(drop=True)
p90 = p90_ds["p90_threshold"].squeeze(drop=True)

# Select 1981
sst = sst.sel(time=slice("1981-01-01", "1981-12-31"))

# Match P90 to SST day-of-year
doy_index = xr.DataArray(
    sst.time.dt.dayofyear.values - 1,
    dims="time",
    coords={"time": sst.time}
)

p90_1981 = p90.isel(dayofyear=doy_index)
p90_1981 = p90_1981.assign_coords(time=sst.time)

# Exceedance
exceed = sst > p90_1981
exceed = exceed.chunk({"time": -1})

# Duration function
def total_mhw_duration(x):
    x = np.asarray(x)
    x = np.nan_to_num(x, nan=False).astype(bool)

    labeled, n_events = label(x)

    total_days = 0
    for event_id in range(1, n_events + 1):
        duration = np.sum(labeled == event_id)
        if duration >= 5:
            total_days += duration

    return total_days

# Compute duration
mhw_duration = xr.apply_ufunc(
    total_mhw_duration,
    exceed,
    input_core_dims=[["time"]],
    vectorize=True,
    dask="parallelized",
    dask_gufunc_kwargs={"allow_rechunk": True},
    output_dtypes=[float]
).compute()

mhw_duration.name = "mhw_duration_1981"

# 🔑 Summary statistics
total_mhw_days = int(mhw_duration.sum(skipna=True).values)
max_duration_at_one_cell = int(mhw_duration.max(skipna=True).values)

print("Marine Heatwave Duration Summary for 1981")
print("-----------------------------------------")
print("Total grid-cell MHW duration days:", total_mhw_days)
print("Maximum duration at one grid cell:", max_duration_at_one_cell)

# Plot
fig = plt.figure(figsize=(11, 6), constrained_layout=True)
ax = plt.axes(projection=ccrs.PlateCarree())

plot = mhw_duration.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="hot_r",
    add_colorbar=False
)

ax.coastlines()
ax.add_feature(cfeature.LAND, facecolor="lightgray")

gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False

ax.set_title("Marine Heatwave Duration (1981)", fontsize=14, pad=18)

cbar = fig.colorbar(
    plot,
    ax=ax,
    orientation="vertical",
    shrink=0.82,
    pad=0.04
)

cbar.set_label("Total MHW Duration (days)", fontsize=11)

plt.show()