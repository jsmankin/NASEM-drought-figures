import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # NASEM Future of Drought Figure 4-3

    ## Aridity Index Convergence on USDM D2+ Drought

    Ensemble-mean convergence of the Aridity Index (AI = PET/P) toward
    the USDM D2+ drought state across CONUS, using annual AI
    (sum of monthly PET / sum of monthly P).

    **Four ensemble members** (equal weight):

    1. ERA5 P + ERA5 PET (reanalysis)
    2. ERA5 P + CRU PET (reanalysis P, station PET)
    3. CPC P + ERA5 PET (gauge P, reanalysis PET)
    4. CPC P + CRU PET (fully observational)

    **Convergence** = (Late anomaly - Early anomaly) / (Drought anomaly - Early anomaly).
    A value of 100% means current climate now matches the historical drought state.

    - **Panel A:** Epoch change in AI (2010--2024 minus 1980--1999)
    - **Panel B:** Convergence on D2+ drought state (%)
    - Stippling where < 3/4 members agree on sign

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
def _(np, os, xr):
    """Load pre-computed ensemble convergence results."""
    PROJ = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    CONV_NC = os.path.join(PROJ, 'DATA', 'aridity_convergence_ensemble.nc')

    ds = xr.open_dataset(CONV_NC)
    lats = ds['lat'].values
    lons = ds['lon'].values

    ens_epoch = ds['ens_epoch_diff'].values
    ens_conv  = ds['ens_convergence'].values
    epoch_agree = ds['epoch_agree'].values.astype(int)
    conv_agree  = ds['conv_agree'].values.astype(int)
    ens_valid   = ds['ens_valid'].values

    # WUS bounding box
    wus_box = ((lats[:, None] >= 25) & (lats[:, None] <= 50) &
               (lons[None, :] >= -125) & (lons[None, :] <= -95))

    # Quick summary
    wus_med = np.nanmedian(ens_conv[wus_box & ens_valid]) * 100
    print('WUS median convergence: %.1f%%' % wus_med)
    return conv_agree, ens_conv, ens_epoch, epoch_agree, lats, lons, wus_box


@app.cell
def _(
    ccrs,
    cfeature,
    conv_agree,
    ens_conv,
    ens_epoch,
    epoch_agree,
    lats,
    lons,
    mcolors,
    mpl,
    np,
    plt,
    wus_box,
):
    """Two-panel CONUS map: epoch change in AI and convergence on D2+."""
    CONUS_EXTENT = [-126, -66, 24, 50]
    proj = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=37.5,
                                standard_parallels=(29.5, 45.5))

    def add_conus(ax):
        ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=5)
        ax.add_feature(cfeature.LAKES, facecolor='white', zorder=5)
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='#444444', zorder=6)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=6)
        ax.coastlines(resolution='50m', linewidth=0.3, zorder=6)

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle('ARIDITY INDEX (PET/P) CONVERGENCE ON USDM D2+ DROUGHT, '
                 '2010\u20132024 vs. 1980\u20131999',
                 fontsize=13, fontweight='bold', y=0.98)
    gs = fig.add_gridspec(1, 2, wspace=0.25)

    # Adaptive bounds for Panel A
    epoch_wus = ens_epoch[wus_box & np.isfinite(ens_epoch)]
    p5, p95 = np.percentile(epoch_wus, [5, 95])
    if p95 > 1.5 or p5 < -1.5:
        bounds_a = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
    elif p95 > 0.75:
        bounds_a = [-1.5, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 1.5]
    else:
        bounds_a = [-1, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 1]

    # Diverging teal-to-brown colormap
    colors_neg = np.array([
        [0.00, 0.35, 0.35, 1], [0.10, 0.55, 0.55, 1],
        [0.40, 0.75, 0.75, 1], [0.75, 0.92, 0.92, 1],
    ])
    colors_pos = np.array([
        [1.00, 0.95, 0.80, 1], [0.99, 0.82, 0.50, 1],
        [0.93, 0.60, 0.25, 1], [0.70, 0.30, 0.10, 1],
    ])
    cmap_a = mcolors.ListedColormap(np.vstack([colors_neg, colors_pos]))
    norm_a = mcolors.BoundaryNorm(bounds_a, cmap_a.N)

    # Panel A: epoch change in aridity index
    ax1 = fig.add_subplot(gs[0, 0], projection=proj)
    ens_epoch_masked = np.where(np.isfinite(ens_conv), ens_epoch, np.nan)
    ax1.pcolormesh(lons, lats, ens_epoch_masked,
                   transform=ccrs.PlateCarree(), cmap=cmap_a, norm=norm_a, zorder=1)

    stip_lons, stip_lats = np.meshgrid(lons, lats)
    step = 2
    sub = np.zeros_like(epoch_agree, dtype=bool)
    sub[::step, ::step] = True
    disagree_a = (epoch_agree < 3) & np.isfinite(ens_epoch_masked) & sub
    ax1.scatter(stip_lons[disagree_a], stip_lats[disagree_a], s=6, c='black',
                alpha=1, transform=ccrs.PlateCarree(), zorder=4, marker='.', linewidths=0)

    add_conus(ax1)
    ax1.set_title('A  CHANGE IN ARIDITY INDEX', fontsize=10, fontweight='bold', loc='left')
    cb1 = plt.colorbar(mpl.cm.ScalarMappable(norm=norm_a, cmap=cmap_a), ax=ax1,
                       orientation='horizontal', pad=0.04, fraction=0.046, aspect=25,
                       extend='both', ticks=bounds_a)
    cb1.set_label('\u0394(PET / P)', fontsize=8)
    cb1.ax.tick_params(labelsize=7)

    # Panel B: convergence on D2+ drought state
    ax2 = fig.add_subplot(gs[0, 1], projection=proj)
    bounds_c = [-50, -25, 0, 15, 30, 45, 60, 75, 100]
    cneg = np.array([[0, 0.35, 0.35, 1], [0.4, 0.75, 0.75, 1]])
    cpos = np.array([
        [1, 0.97, 0.85, 1], [1, 0.90, 0.65, 1], [0.99, 0.78, 0.43, 1],
        [0.96, 0.65, 0.25, 1], [0.90, 0.50, 0.15, 1], [0.60, 0.22, 0.08, 1],
    ])
    cmap_c = mcolors.ListedColormap(np.vstack([cneg, cpos]))
    norm_c = mcolors.BoundaryNorm(bounds_c, cmap_c.N)

    ax2.pcolormesh(lons, lats, ens_conv * 100,
                   transform=ccrs.PlateCarree(), cmap=cmap_c, norm=norm_c, zorder=1)
    disagree_c = (conv_agree < 3) & np.isfinite(ens_conv) & sub
    ax2.scatter(stip_lons[disagree_c], stip_lats[disagree_c], s=6, c='black',
                alpha=1, transform=ccrs.PlateCarree(), zorder=4, marker='.', linewidths=0)

    add_conus(ax2)
    ax2.set_title('B  CONVERGENCE ON D2+ DROUGHT STATE', fontsize=10, fontweight='bold', loc='left')
    cb2 = plt.colorbar(mpl.cm.ScalarMappable(norm=norm_c, cmap=cmap_c), ax=ax2,
                       orientation='horizontal', pad=0.04, fraction=0.046, aspect=25,
                       extend='both', ticks=bounds_c)
    cb2.set_label('% OF DROUGHT AI ANOMALY GAP CLOSED', fontsize=8)
    cb2.ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    fig
    return


if __name__ == "__main__":
    app.run()
