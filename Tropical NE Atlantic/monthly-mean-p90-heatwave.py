import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load dataset
ds = xr.open_dataset(r"C:/Users/Aina Ajibola/Desktop/P90_1981-2010/threshold.nc")

# Create proper month coordinate from dayofyear
dates = xr.cftime_range(start="2000-01-01", periods=366, calendar="gregorian")
ds = ds.assign_coords(month=("dayofyear", [d.month for d in dates]))

# Compute monthly mean P90
monthly_mean_p90 = ds.p90_threshold.groupby("month").mean(dim="dayofyear")

# Create figure
fig, axes = plt.subplots(
    3, 4,
    figsize=(16, 10),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

axes = axes.flatten()

month_names = [
    "Jan","Feb","Mar","Apr","May","Jun",
    "Jul","Aug","Sep","Oct","Nov","Dec"
]

# Consistent color scale
vmin = float(monthly_mean_p90.min().values)
vmax = float(monthly_mean_p90.max().values)

# Plot each month
for i in range(12):
    ax = axes[i]

    pcm = monthly_mean_p90.sel(month=i+1).plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="hot_r",   # 🔥 REVERSED colormap (this is your fix)
        add_colorbar=False,
        vmin=vmin,
        vmax=vmax
    )

    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor="lightgray")

    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(month_names[i])

# Shared horizontal colorbar
cbar = fig.colorbar(
    pcm,
    ax=axes,
    orientation="horizontal",
    fraction=0.04,
    pad=0.15
)

# Layout spacing
plt.subplots_adjust(
    bottom=0.22,
    top=0.92,
    left=0.05,
    right=0.95,
    wspace=0.15,
    hspace=0.25
)

plt.suptitle("Monthly Mean P90 Climatology (1981–2010)", y=0.98)

plt.show()
