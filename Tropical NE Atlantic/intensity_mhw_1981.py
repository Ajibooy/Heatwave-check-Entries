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

# Daily exceedance
exceed = sst > p90_1981

# Intensity = SST above P90 threshold
intensity = sst - p90_1981

# Keep intensity only on heatwave/exceedance days
intensity = intensity.where(exceed, 0)  # Mask out non-MHW events as 0

# Mean intensity per grid cell
mean_intensity = intensity.mean(dim="time", skipna=True)
mean_intensity.name = "mean_mhw_intensity_1981"

# Maximum intensity per grid cell
max_intensity_map = intensity.max(dim="time", skipna=True)
max_intensity_map.name = "max_mhw_intensity_1981"

# Cumulative intensity per grid cell
cumulative_intensity = intensity.sum(dim="time", skipna=True)
cumulative_intensity.name = "cumulative_mhw_intensity_1981"

# Summary statistics
global_mean_intensity = float(mean_intensity.mean(skipna=True).values)
maximum_mean_intensity = float(mean_intensity.max(skipna=True).values)
maximum_daily_intensity = float(max_intensity_map.max(skipna=True).values)
total_cumulative_intensity = float(cumulative_intensity.sum(skipna=True).values)

print("Marine Heatwave Intensity Summary for 1981")
print("------------------------------------------")
print("Mean MHW intensity over affected cells (°C):", round(global_mean_intensity, 2))
print("Maximum mean intensity at one grid cell (°C):", round(maximum_mean_intensity, 2))
print("Maximum daily intensity at one grid cell (°C):", round(maximum_daily_intensity, 2))
print("Total cumulative grid-cell intensity (°C-days):", round(total_cumulative_intensity, 2))

# Plot mean intensity
fig = plt.figure(figsize=(11, 6), constrained_layout=True)
ax = plt.axes(projection=ccrs.PlateCarree())

plot = mean_intensity.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="YlOrRd",  # Suitable colormap for intensity
    add_colorbar=False
)

ax.coastlines()
ax.add_feature(cfeature.LAND, facecolor="lightgray")

gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False

ax.set_title("Marine Heatwave Mean Intensity (1981)", fontsize=14, pad=18)

cbar = fig.colorbar(
    plot,
    ax=ax,
    orientation="vertical",
    shrink=0.82,
    pad=0.04
)

cbar.set_label("Mean Intensity (°C above P90)", fontsize=11)
plt.show()
