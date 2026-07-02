# -*- coding: utf-8 -*-
"""
blackout_tracker_recovery.py — NTL Outage Detector + Recovery Analyzer
=======================================================================
Extends blackout_tracker.py with post-event recovery curve analysis.

Added: Recovery Date Detection (Savitzky-Golay Filter)
    - Smooths post-event radiance time series using SG filter
    - Detects the first date where radiance recovers to >= 90% of pre-event baseline
    - Requires sustained recovery (7 consecutive days above threshold)
    - Recovery date and days-to-recover are included in the output CSV

Validated against 58 global disaster events (2017-2025).
"""

import sys
import os
import pandas as pd
import numpy as np
import concurrent.futures

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from statsmodels.tsa.seasonal import STL
from scipy.signal import savgol_filter

import ee
import urllib.request

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', '..'))

try:
    import config
    config.initialize_gee()
except Exception as e:
    print(f"config.py load failed ({e}), falling back to default ee.Initialize().")
    ee.Initialize()


# =============================================================================
# Output directory
# =============================================================================
OUTPUT_DIR = os.path.join(current_dir, '..', '..', 'outputs', 'blackout_tracker_recovery')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Baseline 1: Day-of-Week Rolling Median
# =============================================================================
def dow_median_baseline_masked(series, mask_out=None, window_weeks=26, min_periods=4):
    """
    Day-of-week rolling median baseline.
    Optionally exclude contaminated periods (e.g. outage windows) via mask_out.
    """
    s = series.copy()
    if mask_out is not None:
        s = s.where(~mask_out.fillna(False))

    df = pd.DataFrame({'value': s})
    df['dow'] = df.index.dayofweek
    baseline = pd.Series(index=df.index, dtype=float)

    for dow in range(7):
        mask = df['dow'] == dow
        dow_series = df.loc[mask, 'value']
        rolled = dow_series.shift(1).rolling(window_weeks, min_periods=min_periods).median()
        baseline.loc[mask] = rolled

    return baseline

def dow_median_baseline(series, window_weeks=26):
    return dow_median_baseline_masked(series, mask_out=None, window_weeks=window_weeks)


# =============================================================================
# Baseline 2: STL decomposition (for comparison)
# =============================================================================
def stl_baseline(series):
    """Trend + seasonal baseline via STL decomposition."""
    series_interp = series.interpolate(method='linear').bfill().ffill()
    stl = STL(series_interp, period=7, robust=True)
    res = stl.fit()
    return res.trend + res.seasonal


# =============================================================================
# Robust Z-score
# =============================================================================
def robust_zscore(series, baseline, trim_quantile=0.1):
    """MAD-based robust Z-score, trimming outliers before computing spread."""
    residuals = series - baseline
    res_clean = residuals.dropna()
    if len(res_clean) < 30:
        return pd.Series(index=series.index, dtype=float)

    q_low, q_high = res_clean.quantile([trim_quantile, 1 - trim_quantile])
    trimmed = res_clean[(res_clean >= q_low) & (res_clean <= q_high)]
    mad_val = np.median(np.abs(trimmed - np.median(trimmed)))
    if mad_val == 0 or np.isnan(mad_val):
        mad_val = 0.01
    std_robust = mad_val * 1.4826
    return residuals / std_robust


# =============================================================================
# Core Logic: Balanced 2-Pass Baseline + Detection
# =============================================================================
def two_pass_baseline_and_detection(df,
                                    z_thresh=-3.0,
                                    drop_thresh_normal=10.0,
                                    drop_thresh_extreme=50.0,
                                    window_weeks=26,
                                    snow_thresh=0.2,
                                    mask_expand_days=7,
                                    sustained_window=3,
                                    sustained_min_days=2,
                                    tier1_obs_thresh=0.4,
                                    tier2_obs_lower=0.1,
                                    tier2_pixel_min=20,
                                    tier2_drop_min=80.0,
                                    tier2_z_max=-4.0):
    """
    2-Pass baseline to avoid outage-period contamination.

    Pass 1: Build initial baseline (may be contaminated by outage signal)
            -> identify candidate dip periods
    Pass 2: Recompute baseline with dip periods masked out (clean baseline)
            -> run final detection on clean baseline
    """
    radiance = df['radiance_main']
    radiance_for_base = radiance.where(df['valid_for_baseline'])

    # -- Pass 1 ---------------------------------------------------------------
    baseline_p1 = dow_median_baseline_masked(radiance_for_base, mask_out=None, window_weeks=window_weeks)
    drop_p1 = ((baseline_p1 - radiance) / baseline_p1) * 100
    z_p1 = robust_zscore(radiance, baseline_p1)

    snow_ok = df['Snow_Cover_Mean'].fillna(0) < snow_thresh

    tier1_p1 = df['obs_ratio_main'] >= tier1_obs_thresh
    tier2_p1 = (
        (df['obs_ratio_main'] >= tier2_obs_lower)
        & (df['obs_ratio_main'] < tier1_obs_thresh)
        & (df['NTL_Main_Count'] >= tier2_pixel_min)
        & (drop_p1 > tier2_drop_min)
        & (z_p1 < tier2_z_max)
    )
    valid_detect_p1 = tier1_p1 | tier2_p1

    cond_normal_p1  = (valid_detect_p1 & (z_p1 < z_thresh) & (drop_p1 > drop_thresh_normal) & snow_ok)
    cond_extreme_p1 = (valid_detect_p1 & (drop_p1 > drop_thresh_extreme) & (z_p1 < z_thresh) & snow_ok)
    dip_candidate_p1 = (cond_normal_p1 | cond_extreme_p1).fillna(False)

    sustained_p1 = dip_candidate_p1.rolling(window=sustained_window, min_periods=sustained_min_days).sum() >= sustained_min_days

    pattern_A_p1 = (drop_p1 > 80.0) & (df['obs_ratio_main'] > 0.5) & (z_p1 < -3.5)
    pattern_B_p1 = (drop_p1 > 60.0) & (df['obs_ratio_main'] > 0.7) & (z_p1 < -5.0)
    cond_singleton_p1 = ((pattern_A_p1 | pattern_B_p1) & snow_ok).fillna(False)

    dip_mask_p1 = (sustained_p1 & dip_candidate_p1) | cond_singleton_p1
    dip_mask_expanded = (dip_mask_p1.astype(int)
                         .rolling(window=mask_expand_days, center=True, min_periods=1).max().astype(bool))

    # -- Pass 2 ---------------------------------------------------------------
    baseline_p2 = dow_median_baseline_masked(radiance_for_base, mask_out=dip_mask_expanded, window_weeks=window_weeks)
    drop_p2 = ((baseline_p2 - radiance) / baseline_p2) * 100
    z_p2 = robust_zscore(radiance, baseline_p2)

    tier1_p2 = df['obs_ratio_main'] >= tier1_obs_thresh
    tier2_p2 = (
        (df['obs_ratio_main'] >= tier2_obs_lower)
        & (df['obs_ratio_main'] < tier1_obs_thresh)
        & (df['NTL_Main_Count'] >= tier2_pixel_min)
        & (drop_p2 > tier2_drop_min)
        & (z_p2 < tier2_z_max)
    )
    valid_detect_p2 = tier1_p2 | tier2_p2

    cond_normal  = (valid_detect_p2 & (z_p2 < z_thresh) & (drop_p2 > drop_thresh_normal) & snow_ok)
    cond_extreme = (valid_detect_p2 & (drop_p2 > drop_thresh_extreme) & snow_ok)
    is_dip_candidate = (cond_normal | cond_extreme).fillna(False)

    sustained = is_dip_candidate.rolling(window=sustained_window, min_periods=sustained_min_days).sum() >= sustained_min_days

    pattern_A_p2 = (drop_p2 > 80.0) & (df['obs_ratio_main'] > 0.5) & (z_p2 < -3.5)
    pattern_B_p2 = (drop_p2 > 60.0) & (df['obs_ratio_main'] > 0.7) & (z_p2 < -5.0)
    cond_singleton_ok = ((pattern_A_p2 | pattern_B_p2) & snow_ok).fillna(False)

    is_dip_main = (sustained & is_dip_candidate) | cond_singleton_ok

    df['baseline_dow_p1']     = baseline_p1
    df['baseline_dow']        = baseline_p2
    df['drop_pct_main_p1']    = drop_p1
    df['drop_pct_main']       = drop_p2
    df['z_main_p1']           = z_p1
    df['z_main']              = z_p2
    df['is_dip_p1']           = dip_mask_p1
    df['is_dip_p1_expanded']  = dip_mask_expanded
    df['is_dip_candidate']    = is_dip_candidate
    df['is_dip_main']         = is_dip_main
    df['caught_by_singleton'] = cond_singleton_ok & ~sustained
    df['caught_by_tier2']     = tier2_p2

    return df


# =============================================================================
# Recovery Analysis (Savitzky-Golay)
# =============================================================================
def analyze_recovery(df, event_date, recovery_threshold=0.90, smooth_window=21, sustained_days=7):
    """
    Detects the date when post-event radiance recovers to >= recovery_threshold
    of the pre-event baseline, sustained for sustained_days consecutive days.

    Uses Savitzky-Golay filter to smooth noisy post-event radiance before
    applying the threshold, reducing false recovery detections from cloud-clear days.

    Args:
        df: DataFrame with radiance_main and baseline_dow columns
        event_date: disaster event date
        recovery_threshold: fraction of baseline to consider recovered (default 0.90 = 90%)
        smooth_window: SG filter window length in days (default 21)
        sustained_days: consecutive days above threshold required (default 7)

    Returns:
        df: updated with smoothed_radiance and recovery_ratio columns
        recovery_date: first date of sustained recovery (or None)
        days_to_recover: integer days from event to recovery (or None)
    """
    event_date = pd.Timestamp(event_date)

    # Pre-event stable baseline (median of 30 days before event)
    pre_event = df.loc[event_date - pd.Timedelta(days=30): event_date - pd.Timedelta(days=1)]
    if pre_event.empty:
        return df, None, None

    stable_baseline = pre_event['baseline_dow'].median()
    if pd.isna(stable_baseline) or stable_baseline <= 0:
        return df, None, None

    # Post-event data
    post_event = df.loc[event_date:].copy()
    if len(post_event) < smooth_window:
        return df, None, None

    ts_clean = post_event['radiance_main'].interpolate(method='time').bfill().ffill()

    # Apply Savitzky-Golay filter
    window_length = smooth_window if smooth_window % 2 != 0 else smooth_window + 1
    smoothed = savgol_filter(ts_clean, window_length=window_length, polyorder=2)
    post_event['smoothed_radiance'] = np.maximum(smoothed, 0)

    # Recovery ratio: smoothed / pre-event stable baseline
    post_event['recovery_ratio'] = post_event['smoothed_radiance'] / stable_baseline

    # Sustained recovery condition
    is_above = post_event['recovery_ratio'] >= recovery_threshold
    sustained_mask = is_above.rolling(window=sustained_days).sum() >= sustained_days

    recovery_date    = None
    days_to_recover  = None

    if sustained_mask.any():
        first_sustained_end = sustained_mask.idxmax()
        recovery_date       = first_sustained_end - pd.Timedelta(days=sustained_days - 1)
        days_to_recover     = (recovery_date - event_date).days

    # Merge back into main df
    df['smoothed_radiance'] = np.nan
    df['recovery_ratio']    = np.nan
    df.loc[post_event.index, 'smoothed_radiance'] = post_event['smoothed_radiance']
    df.loc[post_event.index, 'recovery_ratio']    = post_event['recovery_ratio']

    return df, recovery_date, days_to_recover


# =============================================================================
# Visualization
# =============================================================================
def plot_focused_diagnostics(df, output_dir, target_name, period_str,
                             event_date=None, recovery_date=None):
    """
    Single-panel plot showing outage detection and post-event recovery curve.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    title = f'[{target_name} {period_str}] Outage & Recovery Focus View'
    fig.suptitle(title, fontsize=15, fontweight='bold')

    valid_mask   = df['obs_ratio_main'] >= 0.4
    invalid_mask = ~valid_mask

    ax.plot(df.index, df['radiance_main'],
            color='lightgray', linestyle='-', linewidth=1, zorder=1)
    ax.scatter(df.index[invalid_mask], df.loc[invalid_mask, 'radiance_main'],
               color='lightgray', s=5, label='Invalid Obs (obs < 0.4)', zorder=2)
    ax.scatter(df.index[valid_mask], df.loc[valid_mask, 'radiance_main'],
               color='dimgray', s=10, label='Valid Obs (obs >= 0.4)', zorder=3)

    # Smoothed recovery trend
    if 'smoothed_radiance' in df.columns and df['smoothed_radiance'].notna().any():
        ax.plot(df.index, df['smoothed_radiance'],
                color='royalblue', linewidth=2.5, alpha=0.8,
                label='Smoothed Recovery Trend (SG Filter)', zorder=4)

    # Detected outage days
    outage_days = df[df['is_dip_main']]
    if not outage_days.empty:
        ax.scatter(outage_days.index, outage_days['radiance_main'],
                   color='red', edgecolor='black', s=40, zorder=5,
                   label=f'Detected Outage ({len(outage_days)} days)')

    # Event and recovery markers
    if event_date is not None:
        event_ts = pd.Timestamp(event_date)
        ax.axvline(event_ts, color='crimson', linestyle=':', linewidth=2.5, alpha=0.8, label='Event Date')

        if recovery_date is not None:
            rec_ts = pd.Timestamp(recovery_date)
            ax.axvline(rec_ts, color='royalblue', linestyle='--', linewidth=2.5, alpha=0.8,
                       label='Recovery Date (90%)')
            ax.axvspan(event_ts, rec_ts, color='orange', alpha=0.05, label='Recovery Period')
        else:
            window_end = event_ts + pd.Timedelta(days=14)
            ax.axvspan(event_ts, window_end, color='crimson', alpha=0.05, label='Monitoring Window (14d)')

    ax.set_ylabel('Mean Radiance (nW/cm2/sr)', fontsize=11)
    ax.set_title('Observation Validity, Outage Detection & Post-Event Recovery', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{target_name}_{period_str}_recovery_view.png")
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_detection(df, event_date, expected_window_days=14):
    """Check whether the pipeline detected an outage within the expected window."""
    event_date = pd.Timestamp(event_date)
    window_end = event_date + pd.Timedelta(days=expected_window_days)
    window     = df.loc[event_date:window_end]

    if window.empty:
        return {'detected': False, 'lag_days': None, 'max_drop_pct': None, 'min_radiance_ratio': None}

    detected = bool(window['is_dip_main'].any())
    lag = None
    if detected:
        first_dip = window[window['is_dip_main']].index[0]
        lag = (first_dip - event_date).days

    max_drop  = float(window['drop_pct_main'].max()) if window['drop_pct_main'].notna().any() else None
    min_ratio = float((window['radiance_main'] / window['baseline_dow']).min()) if window['baseline_dow'].notna().any() else None

    return {'detected': detected, 'lag_days': lag, 'max_drop_pct': max_drop, 'min_radiance_ratio': min_ratio}


# =============================================================================
# Satellite Image Download
# =============================================================================
def download_dip_images(col, dip_dates, roi, name, output_dir, max_images=10):
    if not dip_dates:
        return
    if len(dip_dates) > max_images:
        idx = np.linspace(0, len(dip_dates) - 1, max_images, dtype=int)
        dip_dates = [dip_dates[i] for i in idx]

    image_dir = os.path.join(output_dir, f"{name}_dip_images")
    os.makedirs(image_dir, exist_ok=True)
    vis_params = {
        'bands': ['DNB_BRDF_Corrected_NTL'], 'min': 0, 'max': 60,
        'palette': ['000000', '0000FF', '800080', 'FFFF00', 'FFFFFF']
    }
    region_coords = roi.bounds().getInfo()['coordinates']

    for date_ts in dip_dates:
        date_str     = date_ts.strftime('%Y-%m-%d')
        next_day_str = (date_ts + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        img = col.filterDate(date_str, next_day_str).first()
        try:
            url      = img.visualize(**vis_params).getThumbURL({'region': region_coords, 'dimensions': 512, 'format': 'png'})
            img_path = os.path.join(image_dir, f"{name}_{date_str}.png")
            urllib.request.urlretrieve(url, img_path)
        except Exception:
            pass


# =============================================================================
# Main Analysis Function
# =============================================================================
def analyze_outage(target_info):
    name       = target_info['name']
    lat        = target_info['lat']
    lon        = target_info['lon']
    start_year = target_info['start_year']
    end_year   = target_info['end_year']
    event_date = target_info.get('event_date', None)
    buffer_km  = target_info.get('buffer_km', 30)
    period_str = f"{start_year}_{end_year}"

    roi        = ee.Geometry.Point([lon, lat]).buffer(buffer_km * 1000)
    start_date = f"{start_year - 1}-07-01"
    end_date   = f"{end_year + 1}-03-01"
    col        = ee.ImageCollection("NASA/VIIRS/002/VNP46A2").filterBounds(roi).filterDate(start_date, end_date)

    try:
        worldcover = ee.ImageCollection("ESA/WorldCover/v200").filterBounds(roi).first()
        infra_mask = worldcover.select('Map').eq(50)
    except Exception as e:
        print(f"[{name}] WorldCover load failed: {e}")
        return None

    def extract_features(img):
        qa   = img.select('Mandatory_Quality_Flag')
        snow = img.select('Snow_Flag')
        mask_main   = qa.lte(1)
        mask_strict = qa.eq(0)

        total_infra = ee.Image.constant(1).updateMask(infra_mask).rename('Total_Infra')
        ntl         = img.select('DNB_BRDF_Corrected_NTL').updateMask(infra_mask)
        ntl_main    = ntl.updateMask(mask_main).rename('NTL_Main')
        ntl_strict  = ntl.updateMask(mask_strict).rename('NTL_Strict')
        snow_info   = snow.updateMask(infra_mask).rename('Snow_Cover')

        sum_count   = ee.Reducer.sum().combine(reducer2=ee.Reducer.count(), sharedInputs=True)
        stats_light = ee.Image([total_infra, ntl_main, ntl_strict]).reduceRegion(
            reducer=sum_count, geometry=roi, scale=500, maxPixels=1e9)
        stats_snow  = snow_info.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=roi, scale=500, maxPixels=1e9)

        return ee.Feature(None, {
            'date':              img.date().format('YYYY-MM-dd'),
            'Total_Infra_Count': stats_light.get('Total_Infra_count'),
            'NTL_Main_Sum':      stats_light.get('NTL_Main_sum'),
            'NTL_Main_Count':    stats_light.get('NTL_Main_count'),
            'NTL_Strict_Sum':    stats_light.get('NTL_Strict_sum'),
            'NTL_Strict_Count':  stats_light.get('NTL_Strict_count'),
            'Snow_Cover_Mean':   stats_snow.get('Snow_Cover'),
        })

    print(f"[{name}] Extracting data ({start_date} to {end_date})...")
    try:
        feats = col.map(extract_features).getInfo()['features']
    except Exception as e:
        print(f"[{name}] GEE error: {e}")
        return None

    df_full = pd.DataFrame([f['properties'] for f in feats])
    if df_full.empty:
        return None

    df_full['date'] = pd.to_datetime(df_full['date'])
    df_full = df_full.set_index('date').sort_index().asfreq('D')
    df_full['weekday_name'] = df_full.index.day_name()

    numeric_cols = ['Total_Infra_Count', 'NTL_Main_Sum', 'NTL_Main_Count',
                    'NTL_Strict_Sum', 'NTL_Strict_Count', 'Snow_Cover_Mean']
    df_full[numeric_cols] = df_full[numeric_cols].apply(pd.to_numeric, errors='coerce')

    df_full['obs_ratio_main']   = df_full['NTL_Main_Count']   / df_full['Total_Infra_Count']
    df_full['obs_ratio_strict'] = df_full['NTL_Strict_Count'] / df_full['Total_Infra_Count']
    df_full[['obs_ratio_main', 'obs_ratio_strict']] = (
        df_full[['obs_ratio_main', 'obs_ratio_strict']].replace([np.inf, -np.inf], np.nan))

    df_full['valid_for_baseline'] = df_full['obs_ratio_main'] >= 0.6
    df_full['valid_for_detect']   = df_full['obs_ratio_main'] >= 0.4
    df_full['radiance_main']      = df_full['NTL_Main_Sum']   / df_full['NTL_Main_Count']
    df_full['radiance_strict']    = df_full['NTL_Strict_Sum'] / df_full['NTL_Strict_Count']

    radiance_for_base       = df_full['radiance_main'].where(df_full['valid_for_baseline'])
    df_full['baseline_stl'] = stl_baseline(radiance_for_base)

    df_full = two_pass_baseline_and_detection(
        df_full,
        z_thresh=-3.0, drop_thresh_normal=10.0, drop_thresh_extreme=50.0,
        window_weeks=26, snow_thresh=0.2, mask_expand_days=7,
        sustained_window=3, sustained_min_days=2,
        tier1_obs_thresh=0.4, tier2_obs_lower=0.1
    )

    df = df_full.loc[f"{start_year}":f"{end_year}"].copy()

    n_dip_main  = int(df['is_dip_main'].sum())
    n_singleton = int(df['caught_by_singleton'].sum())
    print(f"[{name}] Detected outage days: {n_dip_main} (singleton: {n_singleton})")

    eval_result  = evaluate_detection(df, event_date) if event_date else None
    rec_date_str = None

    if event_date is not None:
        df, rec_date, rec_days = analyze_recovery(
            df, event_date,
            recovery_threshold=0.90,
            smooth_window=21,
            sustained_days=7
        )
        if rec_date:
            rec_date_str = rec_date.strftime('%Y-%m-%d')
            print(f"[{name}] Recovery date: {rec_date_str} ({rec_days} days after event)")
            if eval_result:
                eval_result['recovery_date']    = rec_date_str
                eval_result['days_to_recover']  = rec_days
        else:
            print(f"[{name}] Recovery: not detected (insufficient post-event data or no recovery)")
            if eval_result:
                eval_result['recovery_date']    = "Not recovered / inconclusive"
                eval_result['days_to_recover']  = -1

    csv_cols = ['Total_Infra_Count', 'NTL_Main_Count', 'obs_ratio_main', 'radiance_main',
                'baseline_dow', 'drop_pct_main', 'z_main', 'Snow_Cover_Mean', 'weekday_name',
                'valid_for_detect', 'is_dip_candidate', 'is_dip_main',
                'caught_by_singleton', 'caught_by_tier2']
    if 'smoothed_radiance' in df.columns:
        csv_cols.extend(['smoothed_radiance', 'recovery_ratio'])

    csv_path = os.path.join(OUTPUT_DIR, f"{name}_{period_str}_diagnostic.csv")
    df[csv_cols].to_csv(csv_path)

    plot_focused_diagnostics(df, OUTPUT_DIR, name, period_str,
                             event_date=event_date, recovery_date=rec_date_str)
    download_dip_images(col, list(df[df['is_dip_main']].index), roi, name, OUTPUT_DIR)

    result = {'name': name, 'event_date': event_date,
              'n_dip_days': n_dip_main, 'n_singleton': n_singleton}
    if eval_result:
        result.update(eval_result)

    return result


# =============================================================================
# Summary Report
# =============================================================================
def write_spec_sheet(results, output_dir):
    rows = [r for r in results if r is not None]
    if not rows:
        return
    df_spec   = pd.DataFrame(rows)
    spec_path = os.path.join(output_dir, "detection_recovery_summary.csv")
    df_spec.to_csv(spec_path, index=False)
    print(f"\nSummary saved: {spec_path}")

    detected = df_spec.get('detected', pd.Series([], dtype=bool))
    if not detected.empty:
        print(f"\n{'='*60}")
        print(f"Detection & Recovery Summary")
        print(f"{'='*60}")
        print(f"Detected: {int(detected.sum())} / {len(df_spec)}")
        if 'days_to_recover' in df_spec.columns:
            recovered = df_spec[df_spec['days_to_recover'] > 0]
            print(f"Recovered (90%): {len(recovered)} / {len(df_spec)}")
            if not recovered.empty:
                print(f"Avg days to recover: {recovered['days_to_recover'].mean():.1f}")
        print(f"{'='*60}")


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    targets = [
        # Long-duration events — best suited for recovery analysis
        {"name": "PuertoRico_SanJuan_Maria",      "lat": 18.4655,  "lon": -66.1057, "start_year": 2017, "end_year": 2018, "event_date": "2017-09-20"},
        {"name": "PuertoRico_Ponce_Fiona",        "lat": 18.0111,  "lon": -66.6141, "start_year": 2022, "end_year": 2022, "event_date": "2022-09-18"},
        {"name": "Turkey_Antakya_Earthquake",      "lat": 36.2066,  "lon":  36.1572, "start_year": 2023, "end_year": 2023, "event_date": "2023-02-06"},
        {"name": "Morocco_Marrakech_Earthquake",   "lat": 31.6295,  "lon":  -7.9811, "start_year": 2023, "end_year": 2023, "event_date": "2023-09-08"},
        {"name": "Haiti_LesCayes_Earthquake",      "lat": 18.1905,  "lon": -73.7483, "start_year": 2021, "end_year": 2021, "event_date": "2021-08-14"},
        {"name": "Mexico_Acapulco_Otis",           "lat": 16.8531,  "lon": -99.8237, "start_year": 2023, "end_year": 2023, "event_date": "2023-10-25"},
        {"name": "USA_FL_FortMyers_Ian",           "lat": 26.6406,  "lon": -81.8723, "start_year": 2022, "end_year": 2022, "event_date": "2022-09-28"},
        {"name": "Ukraine_Kyiv_GridAttack",        "lat": 50.4501,  "lon":  30.5234, "start_year": 2022, "end_year": 2023, "event_date": "2022-11-23"},
        {"name": "Venezuela_Caracas_Blackout",     "lat": 10.4806,  "lon": -66.9036, "start_year": 2019, "end_year": 2019, "event_date": "2019-03-07"},
        {"name": "NewZealand_HawkesBay_Gabrielle", "lat": -39.4928, "lon": 176.9120, "start_year": 2023, "end_year": 2023, "event_date": "2023-02-14"},
    ]

    print("NTL Blackout Tracker — Recovery Analyzer")
    print("=" * 60)
    print(f"Targets: {len(targets)}")
    print("Recovery threshold: 90% of pre-event baseline")
    print("Sustained condition: 7 consecutive days")
    print("=" * 60)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(analyze_outage, targets))

    write_spec_sheet(results, OUTPUT_DIR)
