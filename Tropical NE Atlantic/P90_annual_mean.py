import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load final threshold file
ds = xr.open_dataset(r"C:/Users/Aina Ajibola/Desktop/P90_1981-2010/threshold.nc")

# Annual mean over dayofyear
annual_mean_p90 = ds.p90_threshold.mean(dim="dayofyear")

# Plot map
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

annual_mean_p90.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="hot",
    cbar_kwargs={"label": "P90 (°C)"}
)

ax.coastlines()
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.gridlines(draw_labels=True)
ax.set_title("Annual Mean P90 Climatology (1981–2010)")
plt.show()
