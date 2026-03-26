import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

DATA_DIR = r"C:/Users/Aina Ajibola/Desktop/oisst_data"

LAT_MIN, LAT_MAX = 0.0, 30.0
LON_MIN, LON_MAX = -60.0, -10.0
VAR_NAME = "sst"

def lon_to_360(lon):
    return lon % 360.0

def is_lon_0_360(lon_values):
    lon_values = np.asarray(lon_values)
    return np.nanmax(lon_values) > 180.0

def subset_region(ds):
    if "lon" not in ds.coords or "lat" not in ds.coords:
        raise ValueError("Dataset must have 'lat' and 'lon' coordinates.")

    lon0_360 = is_lon_0_360(ds["lon"].values)

    if lon0_360:
        a = lon_to_360(LON_MIN)   # 300
        b = lon_to_360(LON_MAX)   # 350
        if a <= b:
            ds = ds.sel(lon=slice(a, b))
        else:
            ds1 = ds.sel(lon=slice(a, 360))
            ds2 = ds.sel(lon=slice(0, b))
            ds = xr.concat([ds1, ds2], dim="lon")
    else:
        ds = ds.sel(lon=slice(LON_MIN, LON_MAX))

    ds = ds.sel(lat=slice(LAT_MIN, LAT_MAX))
    return ds

# Open files
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.nc")))
if not files:
    raise FileNotFoundError(f"No .nc files found in {DATA_DIR}")

ds = xr.open_mfdataset(files, combine="by_coords", parallel=False)

if VAR_NAME not in ds.data_vars:
    raise ValueError(f"'{VAR_NAME}' not found. Available variables: {list(ds.data_vars)}")

# Climatology period
ds = ds.sel(time=slice("1981-01-01", "2010-12-31"))

# Region subset
ds = subset_region(ds)

# Select SST
sst = ds[VAR_NAME]

# Monthly climatology
monthly_mean = sst.groupby("time.month").mean(dim="time")

# Fix zlev dimension
if "zlev" in monthly_mean.dims:
    monthly_mean = monthly_mean.squeeze("zlev", drop=True)

print(monthly_mean)
print("Shape:", monthly_mean.shape)

# Plot 12 monthly maps
fig, axes = plt.subplots(
    nrows=3,
    ncols=4,
    figsize=(16, 10),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

axes = axes.flatten()
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

vmin = float(monthly_mean.min().compute().values)
vmax = float(monthly_mean.max().compute().values)

for i in range(12):
    ax = axes[i]

    pcm = monthly_mean.sel(month=i + 1).plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
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

# Adjust the layout and position of the colorbar
cbar = fig.colorbar(
    pcm,
    ax=axes,
    orientation="horizontal",
    fraction=0.05,  # Smaller fraction for colorbar size
    pad=0.1         # Adjust padding between the plots and the colorbar
)
cbar.set_label("SST (°C)")

# Adjust the layout of the subplots to prevent overlap
plt.subplots_adjust(bottom=0.2, top=0.85, left=0.05, right=0.95)

plt.suptitle("Monthly Mean SST Climatology (1981–2010)", y=0.98)
plt.tight_layout()
plt.show()