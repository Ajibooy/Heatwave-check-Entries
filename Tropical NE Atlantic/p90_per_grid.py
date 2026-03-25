import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load data
ds = xr.open_dataset(r"C:/Users/Aina Ajibola/Desktop/P90_1981-2010/threshold.nc")

# Average P90 per grid cell
mean_p90_per_grid = ds.p90_threshold.mean(dim="dayofyear")

# Plot map
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

mean_p90_per_grid.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="hot",
    cbar_kwargs={"label": "P90 (°C)"}
)

ax.coastlines()
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.gridlines(draw_labels=True)
ax.set_title("Average P90 per Grid Cell (1981–2010)")
plt.show()
