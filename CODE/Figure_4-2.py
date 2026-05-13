import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # NASEM Future of Drought Figure 4-2

    Four-panel map of D1+ drought event characteristics from the
    rasterized US Drought Monitor (0.125-deg, 2000--2026):

    - **(a)** Number of independent drought events
    - **(b)** Mean event duration (weeks)
    - **(c)** Mean weeks from onset to maximum severity
    - **(d)** Fraction of record in drought (stippled where < 20%,
      i.e., at or below USDM's baseline drought frequency)

    *Justin Mankin -- 2026*
    """)
    return


@app.cell
def _():
    import marimo as mo
    import os
    import numpy as np
    import xarray as xr
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    return ccrs, cfeature, mcolors, mo, mpl, np, os, plt, xr


@app.cell
def _(os):
    """Paths — NASEM root is one level up from CODE/."""
    PROJ = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    SUMMARY_NC = os.path.join(PROJ, 'DATA', 'USDM_drought_events_summary.nc')
    MASK_NC    = os.path.join(PROJ, DATA', 'state_masks_0p125deg.nc')
    return MASK_NC, SUMMARY_NC


@app.cell
def _(MASK_NC, SUMMARY_NC, np, xr):
    """Load drought event summary and CONUS mask."""
    ds    = xr.open_dataset(SUMMARY_NC)
    masks = xr.open_dataset(MASK_NC)
    lats  = ds['lat'].values
    lons  = ds['lon'].values
    conus_mask = masks['conus_mask'].values.astype(bool)

    n_ev     = ds['n_events'].values.copy()
    mean_dur = ds['mean_duration'].values.copy()
    frac_dr  = ds['frac_in_drought'].values.copy()
    mean_ttm = ds['mean_weeks_to_max'].values.copy()

    for arr in [n_ev, mean_dur, frac_dr, mean_ttm]:
        arr[~conus_mask] = np.nan

    # Stippling: cells below USDM's baseline 20% drought percentile
    STIP_THRESH = 0.20
    stipple_mask = (frac_dr < STIP_THRESH) & conus_mask & np.isfinite(frac_dr)
    stipple_lons, stipple_lats = np.meshgrid(lons, lats)
    step = 3
    sub_mask = np.zeros_like(stipple_mask)
    sub_mask[::step, ::step] = True
    sx = stipple_lons[stipple_mask & sub_mask]
    sy = stipple_lats[stipple_mask & sub_mask]
    return frac_dr, lats, lons, mean_dur, mean_ttm, n_ev, sx, sy


@app.cell
def _(
    ccrs,
    cfeature,
    frac_dr,
    lats,
    lons,
    mcolors,
    mean_dur,
    mean_ttm,
    mpl,
    n_ev,
    plt,
    sx,
    sy,
):
    """Build the 2x2 CONUS map figure."""
    CONUS_EXTENT = [-126, -66, 24, 50]

    def add_conus_features(ax):
        ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='#444444')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor='#333333')
        ax.coastlines(resolution='50m', linewidth=0.3)

    def plot_conus_field(ax, data, cmap, bounds, title, cbar_label, extend='max'):
        ax.set_title(title, fontsize=10, fontweight='bold', loc='left')
        if isinstance(cmap, str):
            cmap = mpl.colormaps[cmap].resampled(len(bounds) - 1)
        else:
            cmap = cmap.resampled(len(bounds) - 1)
        norm = mcolors.BoundaryNorm(bounds, cmap.N, clip=False)
        im = ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(),
                           cmap=cmap, norm=norm, zorder=1)
        add_conus_features(ax)
        cb = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.04,
                          fraction=0.046, aspect=25, extend=extend, ticks=bounds)
        cb.set_label(cbar_label, fontsize=8)
        cb.ax.tick_params(labelsize=7)

    proj = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=37.5,
                                standard_parallels=(29.5, 45.5))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), subplot_kw={'projection': proj})

    plot_conus_field(axes[0, 0], n_ev,
                     'YlOrBr', [0, 4, 8, 12, 16, 20, 24, 28, 32],
                     'A  NUMBER OF DROUGHT EVENTS', 'COUNT')

    plot_conus_field(axes[0, 1], mean_dur,
                     'YlOrRd', [0, 5, 10, 20, 30, 40, 60, 80, 120],
                     'B  MEAN EVENT DURATION', 'WEEKS')

    plot_conus_field(axes[1, 0], mean_ttm,
                     'PuBuGn', [0, 2, 4, 6, 8, 12, 16, 20, 28],
                     'C  MEAN WEEKS TO MAX SEVERITY', 'WEEKS')

    plot_conus_field(axes[1, 1], frac_dr * 100,
                     'RdPu', [0, 5, 10, 15, 20, 30, 40, 50, 65],
                     'D  FRACTION OF RECORD IN DROUGHT', '% OF WEEKS')
    axes[1, 1].scatter(sx, sy, s=4.0, c='black', alpha=1,
                       transform=ccrs.PlateCarree(), zorder=5,
                       marker='.', linewidths=0)

    fig.suptitle('USDM DROUGHT EVENT CHARACTERISTICS, 2000\u20132026 (D1+ threshold)',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.01, 1, 0.96], h_pad=3.0)
    fig
    return


if __name__ == "__main__":
    app.run()
