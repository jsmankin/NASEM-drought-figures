"""
dataprep.py
===========
Generate the three pre-computed NetCDF files needed by the NASEM
drought figure notebooks (Figure_4-1, Figure_4-2, Figure_4-3).

Figure_4-3 (schematic) uses no data. The other two need:

    output/data/state_masks_0p125deg.nc          <- Step 1 (Census shapefiles)
    output/data/USDM_drought_events_summary.nc   <- Step 2 (USDM raster + masks)
    output/data/aridity_convergence_ensemble.nc   <- Step 3 (P, PET, USDM raster)

Raw data inputs (in DATA/ — all pre-clipped to CONUS):
    DATA/USDM/USDM_CONUS_0p125deg.nc     ~11 MB  (rasterized USDM, 0.125-deg)
    DATA/P/P.ERA5.{1950..2025}.nc         ~46 MB  (monthly precip, 0.5-deg)
    DATA/P/P.CPC.{1979..2026}.nc          ~1 MB   (monthly precip, clipped to CONUS)
    DATA/PET/PET.ERA5.{1950..2025}.nc     ~46 MB  (monthly PET, 0.5-deg)
    DATA/PET/PET.CRU.{1901..2024}.nc      ~73 MB  (monthly PET, 0.5-deg)

Usage:
    # Default: reads from ../DATA/, writes to ../DATA/
    python dataprep.py

    # Or point to a different data directory:
    python dataprep.py --data-dir /path/to/raw/data

    # Run only one step:
    python dataprep.py --step masks
    python dataprep.py --step events
    python dataprep.py --step convergence

Justin Mankin -- 2026
"""

import os
import sys
import glob
import argparse
import warnings
from datetime import datetime

import numpy as np
import xarray as xr

warnings.filterwarnings('ignore')

SCRIPT = os.path.basename(__file__)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR_DEFAULT = os.path.join(PROJ_DIR, 'DATA')
OUT_DATA = DATA_DIR_DEFAULT  # pre-computed outputs go alongside raw subdirs

# ---- 0.125-deg CONUS grid (matches USDM rasterizer) ----
DLAT = 0.125
DLON = 0.125
LAT_MIN = 25.0625
LAT_MAX = 52.9375
LON_MIN = -124.9375
LON_MAX = -67.0625


def build_grid():
    nlat = int(round((LAT_MAX - LAT_MIN) / DLAT)) + 1
    nlon = int(round((LON_MAX - LON_MIN) / DLON)) + 1
    lat = np.linspace(LAT_MIN, LAT_MAX, nlat)
    lon = np.linspace(LON_MIN, LON_MAX, nlon)
    assert nlat == 224, 'expected 224 lat, got %d' % nlat
    assert nlon == 464, 'expected 464 lon, got %d' % nlon
    return lat, lon


# CONUS FIPS codes (49 states + DC)
CONUS_FIPS = {
    'AL': 1, 'AZ': 4, 'AR': 5, 'CA': 6, 'CO': 8,
    'CT': 9, 'DE': 10, 'DC': 11, 'FL': 12, 'GA': 13,
    'ID': 16, 'IL': 17, 'IN': 18, 'IA': 19, 'KS': 20,
    'KY': 21, 'LA': 22, 'ME': 23, 'MD': 24, 'MA': 25,
    'MI': 26, 'MN': 27, 'MS': 28, 'MO': 29, 'MT': 30,
    'NE': 31, 'NV': 32, 'NH': 33, 'NJ': 34, 'NM': 35,
    'NY': 36, 'NC': 37, 'ND': 38, 'OH': 39, 'OK': 40,
    'OR': 41, 'PA': 42, 'RI': 44, 'SC': 45, 'SD': 46,
    'TN': 47, 'TX': 48, 'UT': 49, 'VT': 50, 'VA': 51,
    'WA': 53, 'WV': 54, 'WI': 55, 'WY': 56,
}
FIPS_TO_ABBR = {v: k for k, v in CONUS_FIPS.items()}
WUS_STATES = ['CA', 'OR', 'WA', 'ID', 'MT', 'WY', 'CO', 'NV', 'UT', 'AZ', 'NM']
WUS_FIPS = sorted(CONUS_FIPS[s] for s in WUS_STATES)

CENSUS_URL = 'https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_state_5m.zip'


# =========================================================================
# STEP 1: State masks
# =========================================================================
def step_masks(data_dir):
    """Rasterize Census state boundaries to 0.125-deg USDM grid.

    Requires: geopandas, rasterio, affine, requests
    """
    import zipfile
    from io import BytesIO
    import geopandas as gpd
    import requests
    from affine import Affine
    from rasterio import features

    out_nc = os.path.join(OUT_DATA, 'state_masks_0p125deg.nc')
    print('=== STEP 1: State masks ===', flush=True)

    # download Census shapefile
    cache_dir = os.path.join(data_dir, 'masks', 'cb_2024_us_state_5m')
    os.makedirs(cache_dir, exist_ok=True)
    shp_path = os.path.join(cache_dir, 'cb_2024_us_state_5m.shp')
    if not os.path.exists(shp_path):
        print('Downloading %s' % CENSUS_URL, flush=True)
        r = requests.get(CENSUS_URL, timeout=60, verify=False)
        r.raise_for_status()
        with zipfile.ZipFile(BytesIO(r.content)) as zf:
            zf.extractall(cache_dir)
    else:
        print('Using cached shapefile: %s' % shp_path, flush=True)

    gdf = gpd.read_file(shp_path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs('EPSG:4326')

    # filter to CONUS
    gdf['STATEFP_INT'] = gdf['STATEFP'].astype(int)
    gdf_conus = gdf[gdf['STATEFP_INT'].isin(FIPS_TO_ABBR.keys())].copy()
    print('  %d CONUS features' % len(gdf_conus), flush=True)

    # build grid + affine
    lat, lon = build_grid()
    dlon = float(lon[1] - lon[0])
    dlat = float(lat[1] - lat[0])
    transform = Affine.translation(lon[0] - dlon / 2.0, lat[0] - dlat / 2.0) * \
                Affine.scale(dlon, dlat)

    # rasterize FIPS
    shapes = list(zip(gdf_conus.geometry, gdf_conus['STATEFP_INT'].astype(int)))
    state_fips = features.rasterize(
        shapes, out_shape=(len(lat), len(lon)), fill=0,
        transform=transform, dtype='int32', all_touched=False,
    ).astype(np.int16)

    conus_mask = (state_fips != 0).astype(np.uint8)
    wus_mask = np.isin(state_fips, WUS_FIPS).astype(np.uint8)

    n_conus = int(conus_mask.sum())
    n_wus = int(wus_mask.sum())
    print('  CONUS cells: %d  |  WUS cells: %d (%.1f%% of CONUS)' % (
        n_conus, n_wus, 100 * n_wus / n_conus), flush=True)

    ds = xr.Dataset({
        'state_fips': (['lat', 'lon'], state_fips,
                       {'long_name': 'US state FIPS code'}),
        'wus_mask':   (['lat', 'lon'], wus_mask,
                       {'long_name': 'Western US mask (11 states)'}),
        'conus_mask': (['lat', 'lon'], conus_mask,
                       {'long_name': 'CONUS mask (49 states + DC)'}),
    }, coords={'lat': lat, 'lon': lon}, attrs={
        'polygon_source': CENSUS_URL,
        'wus_states': ','.join(WUS_STATES),
        'script': SCRIPT,
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    })
    encoding = {v: {'zlib': True, 'complevel': 4} for v in ds.data_vars}
    ds.to_netcdf(out_nc, encoding=encoding)
    print('  Wrote: %s' % out_nc, flush=True)
    return out_nc


# =========================================================================
# STEP 2: Drought event summary
# =========================================================================
def step_events(data_dir):
    """Extract drought events from rasterized USDM, write per-cell summary.

    Input: USDM_CONUS_0p125deg.nc + state_masks_0p125deg.nc
    """
    THRESHOLD = 1      # D1+
    MIN_GAP = 2        # bridge gaps shorter than this (weeks)

    usdm_nc = os.path.join(data_dir, 'USDM', 'USDM_CONUS_0p125deg.nc')
    mask_nc = os.path.join(OUT_DATA, 'state_masks_0p125deg.nc')
    out_nc  = os.path.join(OUT_DATA, 'USDM_drought_events_summary.nc')

    print('\n=== STEP 2: Drought event summary ===', flush=True)
    if not os.path.exists(mask_nc):
        print('ERROR: %s not found. Run --step masks first.' % mask_nc, flush=True)
        sys.exit(1)

    print('Loading USDM raster: %s' % usdm_nc, flush=True)
    usdm = xr.open_dataset(usdm_nc)
    dc = usdm['droughtclass']
    times = dc['time'].values
    lats = dc['lat'].values
    lons = dc['lon'].values
    T, nlat, nlon = dc.shape
    print('  %d weeks x %d lat x %d lon' % (T, nlat, nlon), flush=True)

    masks = xr.open_dataset(mask_nc)
    conus_mask = masks['conus_mask'].values.astype(bool)
    n_land = int(conus_mask.sum())
    print('  CONUS cells: %d' % n_land, flush=True)

    data = dc.values

    # per-cell summary arrays
    n_events_arr = np.full((nlat, nlon), np.nan)
    mean_dur_arr = np.full((nlat, nlon), np.nan)
    median_dur_arr = np.full((nlat, nlon), np.nan)
    max_dur_arr = np.full((nlat, nlon), np.nan)
    mean_max_sev_arr = np.full((nlat, nlon), np.nan)
    mean_ttm_arr = np.full((nlat, nlon), np.nan)
    frac_arr = np.full((nlat, nlon), np.nan)

    land_idx = np.argwhere(conus_mask)
    t0 = datetime.now()
    print('Processing %d cells (D%d+, gap bridge=%d wk)...' % (
        n_land, THRESHOLD, MIN_GAP), flush=True)

    for count, (i, j) in enumerate(land_idx):
        ts = data[:, i, j]
        mask = (ts >= THRESHOLD).copy()

        # bridge short gaps
        in_gap = False
        gap_start = 0
        for t in range(T):
            if mask[t]:
                if in_gap and (t - gap_start) < MIN_GAP:
                    mask[gap_start:t] = True
                in_gap = False
            else:
                if not in_gap:
                    gap_start = t
                    in_gap = True

        # label contiguous events
        events = []
        in_event = False
        for t in range(T):
            if mask[t] and not in_event:
                ev_start = t
                in_event = True
            elif not mask[t] and in_event:
                events.append((ev_start, t - 1))
                in_event = False
        if in_event:
            events.append((ev_start, T - 1))

        # extract stats per event
        valid_events = []
        for (s, e) in events:
            sev = ts[s:e+1]
            ok = np.isfinite(sev)
            if ok.sum() == 0:
                continue
            mx = int(np.nanmax(sev))
            if mx < THRESHOLD:
                continue
            valid_events.append({
                'dur': e - s + 1,
                'max_sev': mx,
                'ttm': int(np.where(sev == mx)[0][0]),
            })

        n_ev = len(valid_events)
        n_events_arr[i, j] = n_ev
        total_wk = sum(ev['dur'] for ev in valid_events) if valid_events else 0
        frac_arr[i, j] = total_wk / T

        if n_ev > 0:
            durs = np.array([ev['dur'] for ev in valid_events])
            mean_dur_arr[i, j] = durs.mean()
            median_dur_arr[i, j] = np.median(durs)
            max_dur_arr[i, j] = durs.max()
            mean_max_sev_arr[i, j] = np.mean([ev['max_sev'] for ev in valid_events])
            mean_ttm_arr[i, j] = np.mean([ev['ttm'] for ev in valid_events])

        if (count + 1) % 10000 == 0:
            elapsed = (datetime.now() - t0).total_seconds()
            print('  %d/%d  (%.0fs)' % (count + 1, n_land, elapsed), flush=True)

    elapsed = (datetime.now() - t0).total_seconds()
    print('  Done: %d cells in %.0fs' % (n_land, elapsed), flush=True)

    def make_da(arr, long_name, units=''):
        return xr.DataArray(arr, dims=['lat', 'lon'],
                            coords={'lat': lats, 'lon': lons},
                            attrs={'long_name': long_name, 'units': units})

    ds_out = xr.Dataset({
        'n_events':          make_da(n_events_arr, 'Number of drought events', 'count'),
        'mean_duration':     make_da(mean_dur_arr, 'Mean event duration', 'weeks'),
        'median_duration':   make_da(median_dur_arr, 'Median event duration', 'weeks'),
        'max_duration':      make_da(max_dur_arr, 'Max event duration', 'weeks'),
        'mean_max_severity': make_da(mean_max_sev_arr, 'Mean max severity', 'USDM class'),
        'mean_weeks_to_max': make_da(mean_ttm_arr, 'Mean weeks to max severity', 'weeks'),
        'frac_in_drought':   make_da(frac_arr, 'Fraction of record in drought', ''),
    }, attrs={
        'source': 'USDM_CONUS_0p125deg.nc',
        'threshold': 'D%d+' % THRESHOLD,
        'min_gap': '%d weeks' % MIN_GAP,
        'record': '%s to %s (%d weeks)' % (str(times[0])[:10], str(times[-1])[:10], T),
        'script': SCRIPT,
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    })
    ds_out.to_netcdf(out_nc)
    print('  Wrote: %s' % out_nc, flush=True)
    return out_nc


# =========================================================================
# STEP 3: Aridity convergence ensemble
# =========================================================================
def step_convergence(data_dir):
    """Compute 4-member AI convergence on USDM D2+ drought.

    Ensemble:
      1. ERA5 P + ERA5 PET
      2. ERA5 P + CRU PET
      3. CPC P  + ERA5 PET
      4. CPC P  + CRU PET

    Input: monthly P and PET files + USDM raster
    """
    EARLY = (1980, 1999)
    LATE  = (2010, 2024)
    MIN_GAP = 0.2            # AI units
    D2_YEAR_THRESH = 3       # months

    COMBOS = [('ERA5', 'ERA5'), ('ERA5', 'CRU'), ('CPC', 'ERA5'), ('CPC', 'CRU')]
    LABELS = ['ERA5 P + ERA5 PET', 'ERA5 P + CRU PET',
              'CPC P + ERA5 PET',  'CPC P + CRU PET']

    usdm_nc = os.path.join(data_dir, 'USDM', 'USDM_CONUS_0p125deg.nc')
    out_nc  = os.path.join(OUT_DATA, 'aridity_convergence_ensemble.nc')

    print('\n=== STEP 3: Aridity convergence ensemble ===', flush=True)

    # master grid from ERA5 PET
    era5_pet_files = sorted(glob.glob(os.path.join(data_dir, 'PET', 'PET.ERA5.*.nc')))
    if not era5_pet_files:
        print('ERROR: no PET.ERA5.*.nc files in %s/PET/' % data_dir, flush=True)
        sys.exit(1)
    sample = xr.open_dataset(era5_pet_files[0])
    master_lats = sample.lat.values
    master_lons = sample.lon.values
    nlat, nlon = len(master_lats), len(master_lons)
    sample.close()
    print('Master grid: %d lat x %d lon (ERA5 0.5-deg)' % (nlat, nlon), flush=True)

    def load_var(source, variable):
        pattern = os.path.join(data_dir, variable, '%s.%s.*.nc' % (variable, source))
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError('No files: %s' % pattern)
        ds = xr.open_mfdataset(files, combine='by_coords')
        da = ds[variable]
        if source == 'CPC':
            da = da.assign_coords(lon=((da.lon + 180) % 360) - 180)
            da = da.sortby('lon').sortby('lat')
        if source == 'CRU':
            da['time'] = da.indexes['time'].to_period('M').to_timestamp()
        da = da.sel(lat=slice(23, 51), lon=slice(-127, -65))
        da = da.interp(lat=master_lats, lon=master_lons, method='nearest')
        return da

    # USDM D2+ annual mask on master grid
    print('Building annual D2+ mask from USDM...', flush=True)
    usdm_ds = xr.open_dataset(usdm_nc)
    dc = usdm_ds['droughtclass']
    d2plus = (dc >= 2)
    d2f = d2plus.astype(float).coarsen(lat=4, lon=4, boundary='trim').mean()
    d2c = (d2f >= 0.5)
    d2_wk_count = d2c.astype(int).resample(time='MS').sum()
    d2_wk_total = d2c.astype(int).resample(time='MS').count()
    d2_monthly = ((d2_wk_count / d2_wk_total) >= 0.5)
    d2_monthly_m = d2_monthly.astype(float).interp(
        lat=master_lats, lon=master_lons, method='nearest')
    d2_monthly_m = (d2_monthly_m >= 0.5)

    d2_yrs = d2_monthly_m.time.dt.year.values
    unique_yrs = np.unique(d2_yrs)
    months_per_yr = np.array([(d2_yrs == yr).sum() for yr in unique_yrs])
    complete_yrs = unique_yrs[months_per_yr >= 12]
    d2_complete = d2_monthly_m.sel(time=d2_monthly_m.time.dt.year.isin(complete_yrs))
    d2_year_count = d2_complete.astype(int).resample(time='YS').sum(dim='time')
    d2_annual = (d2_year_count >= D2_YEAR_THRESH)
    n_drt_years = d2_annual.sum(dim='time')
    has_any_drt = (n_drt_years >= 1).values
    usdm_ds.close()
    print('  D2+ years: %d complete years, %d cells with >= 1 D2+ year' % (
        len(complete_yrs), has_any_drt.sum()), flush=True)

    wus_box = ((master_lats[:, None] >= 25) & (master_lats[:, None] <= 50) &
               (master_lons[None, :] >= -125) & (master_lons[None, :] <= -95))

    # loop over ensemble members
    all_convergence = np.full((len(COMBOS), nlat, nlon), np.nan)
    all_epoch_diff  = np.full((len(COMBOS), nlat, nlon), np.nan)
    all_valid       = np.full((len(COMBOS), nlat, nlon), False)

    for ci, (p_src, pet_src) in enumerate(COMBOS):
        print('\n  %s...' % LABELS[ci], flush=True)
        p_mon = load_var(p_src, 'P')
        pet_mon = load_var(pet_src, 'PET')

        # align to common months
        p_ym = p_mon.time.dt.year.values * 100 + p_mon.time.dt.month.values
        pet_ym = pet_mon.time.dt.year.values * 100 + pet_mon.time.dt.month.values
        common_ym = np.intersect1d(p_ym, pet_ym)
        p_sel = p_mon.isel(time=np.isin(p_ym, common_ym)).compute()
        pet_sel = pet_mon.isel(time=np.isin(pet_ym, common_ym)).compute()

        # annual totals
        p_annual = p_sel.resample(time='YS').sum(dim='time', min_count=12)
        pet_annual = pet_sel.resample(time='YS').sum(dim='time', min_count=12)
        both_ok = (p_annual.notnull().any(dim=['lat', 'lon']) &
                   pet_annual.notnull().any(dim=['lat', 'lon']))
        p_annual = p_annual.sel(time=both_ok)
        pet_annual = pet_annual.sel(time=both_ok)

        # annual AI = PET / P
        p_vals = np.maximum(p_annual.values.copy(), 1.0)
        ai_annual = xr.DataArray(pet_annual.values / p_vals,
                                 dims=p_annual.dims, coords=p_annual.coords)
        landmsk = np.isfinite(ai_annual).all(dim='time').values

        # epoch difference
        early_m = ((ai_annual.time.dt.year >= EARLY[0]) &
                   (ai_annual.time.dt.year <= EARLY[1]))
        late_m = ((ai_annual.time.dt.year >= LATE[0]) &
                  (ai_annual.time.dt.year <= LATE[1]))
        epoch_diff = (ai_annual.isel(time=late_m).mean(dim='time') -
                      ai_annual.isel(time=early_m).mean(dim='time'))
        all_epoch_diff[ci] = epoch_diff.values

        # convergence
        ai_mean = ai_annual.mean(dim='time')
        ai_anom = ai_annual - ai_mean
        anom_yr = ai_anom.time.dt.year.values
        d2_yr = d2_annual.time.dt.year.values
        olap = np.intersect1d(anom_yr, d2_yr)
        ai_drt = ai_anom.isel(time=np.isin(anom_yr, olap)).where(
            d2_annual.isel(time=np.isin(d2_yr, olap))).mean(dim='time')
        ai_early = ai_anom.isel(time=early_m).mean(dim='time')
        ai_late = ai_anom.isel(time=late_m).mean(dim='time')

        gap = (ai_drt - ai_early)
        valid = has_any_drt & landmsk & (gap.values > MIN_GAP)
        convergence = ((ai_late - ai_early) / gap).where(valid)

        all_convergence[ci] = convergence.values
        all_valid[ci] = valid

        wus_cv = convergence.values[wus_box & valid]
        wus_cv = wus_cv[np.isfinite(wus_cv)]
        print('    WUS median convergence: %.1f%%' % (np.median(wus_cv) * 100), flush=True)

    # ensemble statistics
    ens_epoch = np.nanmean(all_epoch_diff, axis=0)
    ens_conv  = np.nanmean(all_convergence, axis=0)
    epoch_signs = np.sign(all_epoch_diff)
    epoch_agree = np.maximum((epoch_signs > 0).sum(axis=0),
                             (epoch_signs < 0).sum(axis=0))
    conv_signs = np.sign(all_convergence)
    conv_agree = np.maximum((conv_signs > 0).sum(axis=0),
                            (conv_signs < 0).sum(axis=0))
    ens_valid = all_valid.sum(axis=0) >= 3

    ens_wus = ens_conv[wus_box & ens_valid]
    ens_wus = ens_wus[np.isfinite(ens_wus)]
    print('\nEnsemble WUS median convergence: %.1f%%' % (np.median(ens_wus) * 100), flush=True)

    # save
    ds_out = xr.Dataset({
        'ens_convergence':     (['lat', 'lon'], ens_conv),
        'ens_convergence_pct': (['lat', 'lon'], ens_conv * 100),
        'ens_epoch_diff':      (['lat', 'lon'], ens_epoch),
        'epoch_agree':         (['lat', 'lon'], epoch_agree.astype(float)),
        'conv_agree':          (['lat', 'lon'], conv_agree.astype(float)),
        'ens_valid':           (['lat', 'lon'], ens_valid),
        'n_d2_years':          (['lat', 'lon'], n_drt_years.values.astype(float)),
    }, coords={'lat': master_lats, 'lon': master_lons})

    for ci, label in enumerate(LABELS):
        ds_out['convergence_%d' % ci] = (['lat', 'lon'], all_convergence[ci])
        ds_out['epoch_diff_%d' % ci]  = (['lat', 'lon'], all_epoch_diff[ci])

    ds_out.attrs = {
        'members': str(LABELS),
        'early_period': '%d-%d' % EARLY,
        'late_period': '%d-%d' % LATE,
        'method': 'Annual AI = sum(PET) / sum(P). '
                  'Convergence = (late_anom - early_anom) / (drought_anom - early_anom)',
        'aridity_index': 'PET / P (annual totals)',
        'd2_year_threshold': '%d months' % D2_YEAR_THRESH,
        'min_gap': '%.1f AI units' % MIN_GAP,
        'script': SCRIPT,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    ds_out.to_netcdf(out_nc)
    print('  Wrote: %s' % out_nc, flush=True)
    return out_nc


# =========================================================================
# Main
# =========================================================================
def main():
    ap = argparse.ArgumentParser(
        description='Generate pre-computed NetCDF files for NASEM drought figures.')
    ap.add_argument('--data-dir', default=DATA_DIR_DEFAULT,
                    help='Path to raw data directory (contains P/, PET/, USDM/ subdirs). '
                         'Default: ../DATA/')
    ap.add_argument('--step', choices=['masks', 'events', 'convergence', 'all'],
                    default='all',
                    help='Which step to run (default: all)')
    args = ap.parse_args()

    if not os.path.isdir(args.data_dir):
        print('ERROR: --data-dir does not exist: %s' % args.data_dir, flush=True)
        sys.exit(1)

    os.makedirs(OUT_DATA, exist_ok=True)

    if args.step in ('all', 'masks'):
        step_masks(args.data_dir)
    if args.step in ('all', 'events'):
        step_events(args.data_dir)
    if args.step in ('all', 'convergence'):
        step_convergence(args.data_dir)

    print('\nDone.', flush=True)


if __name__ == '__main__':
    main()
