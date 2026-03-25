import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load final threshold file
ds = xr.open_dataset("C:/Users/Aina Ajibola/Desktop/P90_1981-2010/threshold.nc")

# STD over dayofyear
std_p90 = ds.p90_threshold.std(dim="dayofyear")

# Plot map
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

std_p90.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="viridis",
    cbar_kwargs={"label": "STD of P90 (°C)"}
)

ax.coastlines()
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.gridlines(draw_labels=True)
ax.set_title("STD of P90 over Day of Year (1981–2010)")
plt.show()