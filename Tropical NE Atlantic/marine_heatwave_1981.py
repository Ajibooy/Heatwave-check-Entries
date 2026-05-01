import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import label
import matplotlib.colors as mcolors

# Paths
sst_path = r"C:/Users/Aina Ajibola/Desktop/oisst_data/1981*_oisst.nc"
p90_path = r"C:/Users/Aina Ajibola/Desktop/P90_1981-2010/threshold.nc"

# Load datasets
sst_ds = xr.open_mfdataset(sst_path, combine="by_coords")
p90_ds = xr.open_dataset(p90_path)

# Extract variables and remove zlev
sst = sst_ds["sst"].squeeze(drop=True)
p90 = p90_ds["p90_threshold"].squeeze(drop=True)

# Select 1981 SST
sst = sst.sel(time=slice("1981-01-01", "1981-12-31"))

# Match P90 to SST day-of-year
doy_index = xr.DataArray(
    sst.time.dt.dayofyear.values - 1,
    dims="time",
    coords={"time": sst.time}
)

p90_1981 = p90.isel(dayofyear=doy_index)
p90_1981 = p90_1981.assign_coords(time=sst.time)

# SST exceeds P90
exceed = sst > p90_1981
exceed = exceed.chunk({"time": -1})

# Count MHW events per grid cell
def count_mhw_events(x):
    x = np.asarray(x)
    x = np.nan_to_num(x, nan=False).astype(bool)

    labeled, n_events = label(x)

    count = 0
    for event_id in range(1, n_events + 1):
        duration = np.sum(labeled == event_id)
        if duration >= 5:
            count += 1

    return int(count)

# Apply MHW frequency calculation
mhw_frequency = xr.apply_ufunc(
    count_mhw_events,
    exceed,
    input_core_dims=[["time"]],
    vectorize=True,
    dask="parallelized",
    dask_gufunc_kwargs={"allow_rechunk": True},
    output_dtypes=[float]  # Use float for internal calculation, but cast later
)

mhw_frequency = mhw_frequency.compute()

# Summary statistics
total_grid_cell_events = int(mhw_frequency.sum(skipna=True).values)
max_events_at_one_cell = int(mhw_frequency.max(skipna=True).values)
affected_grid_cells = int((mhw_frequency > 0).sum(skipna=True).values)

print("Marine Heatwave Summary for 1981")
print("---------------------------------")
print("Total grid-cell MHW events:", total_grid_cell_events)
print("Maximum events at one grid cell:", max_events_at_one_cell)
print("Number of affected grid cells:", affected_grid_cells)

# Create a custom colormap (light blue for 0, dark red for 5)
colors = ["lightblue", "yellow", "orange", "red", "darkred"]
n_bins = 5  # Discrete bins for MHW count (0, 1, 2, 3, 4, 5)
cmap = mcolors.ListedColormap(colors)
bounds = [0, 1, 2, 3, 4, 5]  # MHW event counts as discrete levels
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Plot with the custom colormap
fig = plt.figure(figsize=(11, 6), constrained_layout=True)
ax = plt.axes(projection=ccrs.PlateCarree())

plot = mhw_frequency.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap=cmap,  # Custom discrete colormap
    norm=norm,  # Apply the custom bounds
    add_colorbar=False
)

ax.coastlines()
ax.add_feature(cfeature.LAND, facecolor="lightgray")

gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False

ax.set_title(
    "Marine Heatwave Frequency (1981)",
    fontsize=14,
    pad=18
)

# Colorbar
cbar = fig.colorbar(
    plot,
    ax=ax,
    orientation="vertical",
    shrink=0.82,
    pad=0.04
)

cbar.set_label("Number of Marine Heatwave Events", fontsize=11)

# Summary text below map
summary_text = (
    f"Total grid-cell events: {total_grid_cell_events}   |   "
    f"Max events/grid cell: {max_events_at_one_cell}   |   "
    f"Affected grid cells: {affected_grid_cells}"
)

fig.text(
    0.5,
    -0.02,
    summary_text,
    ha="center",
    fontsize=11
)

plt.show()
